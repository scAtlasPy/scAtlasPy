import duckdb
import numpy as np
import multiprocessing as mp
import threading  # 🔹 修改：导入 threading
from multiprocessing.shared_memory import SharedMemory
import time
import scipy.sparse as sp

def prepare_partition_index_only_p(db_path, producer_num = 10, fetch_size = 6793163):
    """
    为 read_func_use_index 版本准备分区字段和索引

    1. 添加 thread_id 列（如果不存在）
    2. 按 block 分片规则填充
    """

    conn = duckdb.connect(db_path)

    print("Step 1: 添加列（如果不存在）...")

    # 添加列（如果不存在）
    conn.execute("""
        ALTER TABLE X_CSR_data
        ADD COLUMN IF NOT EXISTS pid_only INTEGER;
    """)

    print("Step 2: 填充分区字段...")

    # 按你原有的 block 分片规则填充
    conn.execute(f"""
        UPDATE X_CSR_data
        SET
            pid_only = (id // {fetch_size}) % {producer_num}
    """)

    conn.close()

# producer 使用索引分片 ，120b/s 左右
class CSRBatchFetcherMP_SharedMemNoQueue:
    """
    多进程 Arrow 解包 + 单进程 CSR batch
    producer 直接写入共享内存，consumer 直接读取
    Queue 仅用于通知 batch 是否完成（可选）
    """

    # todo 1. 初始化相关变量 + 预计算 batch_nnz + 预定义 共享内存
    def __init__(self, db_path, batch_size=2048, producer_num=10, slot_num=10):

        self.db_path = db_path
        self.batch_size = batch_size
        self.producer_num = producer_num # 进程数量 slot_num = producer_num

        # consumer 私有 pool , 消费
        self.pool_indices = np.empty(0, dtype=np.int32)
        self.pool_data = np.empty(0, dtype=np.float32)

        # 🔹 预计算 indptr
        self.indptr_queue = self._prepare_indptr()

        # 🔹 预计算 batch nnz
        self.batch_nnz = self._prepare_batch_nnz_sql()
        self.batch_idx = 0
        self.total_batches = mp.Value('i', 0)
        self.batch_num = len(self.batch_nnz)

        max_nnz = max(self.batch_nnz)  # 🔹 动态计算最大 nnz ，所有批次最大的 nzz 数量
        self.fetch_size = max_nnz #  fetch_record_batch ，每次的读取数量
        print(f"max_nnz = {max_nnz}")

        # 🔹 共享内存池
        self.slot_num = slot_num # 共享内存池 槽位slot个数
        self.shm_indices_pool = []   # 存 indices -- gene_id
        self.shm_data_pool = []      # 存 data
        self.shm_flags = []         # 0=空 没数据, 1 = 非空 有数据
        self.shm_to_unlink = []     # 进程结束统一释放
        for i in range(slot_num):
            shm_idx = SharedMemory(create=True, size=max_nnz * np.int32().nbytes)
            # indices 专用共享内存； 创建一块 系统级共享内存 # 大小 = max_nnz × 4 bytes
            shm_val = SharedMemory(create=True, size=max_nnz * np.float32().nbytes) # data 专用共享内存
            shm_flag = mp.Value('i', 0)  # 0=empty, 1=full
            # 共享的 int， 所有进程都能看到。自带 Lock

            self.shm_indices_pool.append(shm_idx)
            self.shm_data_pool.append(shm_val)
            self.shm_flags.append(shm_flag)
            # slot 的完整结构
            # ┌─────────────── slot i ───────────────┐
            # │ indices SHM (int32[max_nnz])          │
            # │ data    SHM (float32[max_nnz])        │
            # │ flag    (0 / 1)     0=空 没数据, 1 = 非空 有数据  │
            # └──────────────────────────────────────┘

            self.shm_to_unlink.extend([shm_idx.name, shm_val.name])
            # 存进程结束标识

    # todo 2. 预计算 indptr_queue
    def _prepare_indptr(self):
        # 🔹 读取 indptr
        conn = duckdb.connect(self.db_path, read_only=True)

        fetch_record_indptr = (conn.execute("SELECT indptr FROM X_CSR_indptr")
                               .fetch_record_batch (rows_per_batch=self.batch_size))
        record_indptr_queue = mp.Queue()

        for record_indptr in fetch_record_indptr:
            record_indptr_queue.put(record_indptr)

        conn.close()
        return record_indptr_queue

    # todo 3. SQL 预计算 batch_nnz
    def _prepare_batch_nnz_sql(self):
        conn = duckdb.connect(self.db_path, read_only=True)
        query = f"""
        WITH t AS (
            SELECT indptr, ROW_NUMBER() OVER () AS rn
            FROM X_CSR_indptr
        ),
        picked AS (
            SELECT indptr FROM t WHERE rn % {self.batch_size} = 0
        )
        SELECT indptr - LAG(indptr, 1, 0) OVER (ORDER BY indptr)
        FROM picked
        """
        rows = conn.execute(query).fetchall()
        conn.close()
        return [int(r[0]) for r in rows]

    # todo 4. 多进程 ，producer 生产者，从 X_CSR_data 提取出 indices, data 写入共享内存
    def _producer(self, pid):

        conn = duckdb.connect(self.db_path, read_only=True)

        query = f"""
            SELECT indices, data
            FROM X_CSR_data
            WHERE pid_only = {pid};
        """

        result = conn.execute(query).fetch_record_batch(rows_per_batch=self.fetch_size)

        for rid, rb in enumerate(result):
            slot_id = pid  # 每个 producer 写入自己的 slot

            indices = rb.column(0).to_numpy()
            data = rb.column(1).to_numpy()

            shm_idx = self.shm_indices_pool[slot_id]
            shm_val = self.shm_data_pool[slot_id]
            flag = self.shm_flags[slot_id]

            wait_start = time.time()
            while flag.value != 0:
                time.sleep(0.0001)
            wait_end = time.time()
            if wait_end - wait_start > 0.01:
                print(f"[Producer-{pid}] wait slot_{slot_id} 的时间为 {wait_end - wait_start:.4f}s")

            np.ndarray(indices.shape, dtype=indices.dtype, buffer=shm_idx.buf)[:] = indices
            np.ndarray(data.shape, dtype=data.dtype, buffer=shm_val.buf)[:] = data
            flag.value = 1

            print(f"[Producer-{pid}] : Done 将 fetch 批次 rid = {rid} 写入 slot_{slot_id}, nnz={len(indices)}")

        print(f"[Producer-{pid}] 进程完成")
        conn.close()

    # -----------------------------
    # todo 5. 单进程 ，consumer 消费者，从 共享内存 中取出 indices, data + indptr构建 CSR
    # -----------------------------
    def _consumer(self):

        conn = duckdb.connect(self.db_path, read_only=True)
        global_indptr_offset = 0
        slot_id = 0

        t_start = time.time()

        # ============================
        # 🔥 新增：时间统计变量
        # ============================
        wait_slot_time = 0.0

        read_shm_time1 = 0.0
        read_shm_time2 = 0.0

        build_csr_time1 = 0.0
        build_csr_time2 = 0.0
        build_csr_time3 = 0.0
        build_csr_time4 = 0.0

        pool_slice_time = 0.0
        # ============================

        while self.batch_idx < self.batch_num:

            need = self.batch_nnz[self.batch_idx]

            # ============================================================
            # 情况 1：pool 足够 -> 构建 CSR
            # ============================================================
            if self.pool_data.size >= need:

                t0 = time.time()  # 🔥 build csr start
                vals = self.pool_data[:need]
                cols = self.pool_indices[:need]
                build_csr_time1 += time.time() - t0  # 🔥 累加 CSR 时间
                # 0.2111s (6.12%)

                t1 = time.time()  # 🔥 build csr start
                indptr_now = self.indptr_queue.get().column(0).to_numpy()
                build_csr_time2 += time.time() - t1  # 🔥 累加 CSR 时间
                # 0.3227s (9.36%)

                t2 = time.time()  # 🔥 build csr start
                last_val = indptr_now[-1]
                indptr_now = indptr_now - global_indptr_offset
                indptr_now = np.concatenate(([0], indptr_now))
                global_indptr_offset = last_val

                n_genes = int(cols.max()) + 1 if len(cols) else 0

                build_csr_time3 += time.time() - t2  # 🔥 累加 CSR 时间
                # 0.1887s (5.47%)

                t3 = time.time()  # 🔥 build csr start
                X = sp.csr_matrix((self.batch_size, n_genes), dtype=np.float32)
                X.data = vals
                X.indices = cols
                X.indptr = indptr_now

                build_csr_time4 += time.time() - t3  # 🔥 累加 CSR 时间
                # 1.2330s (35.75%)
                # todo 优化1 结果， 节约1s
                # 0.2916s (12.12%)

                # ============================
                # 🔥 pool slice 时间统计 # 0.0015s (0.04%)
                # ============================
                t1 = time.time()

                self.pool_indices = self.pool_indices[need:]
                self.pool_data = self.pool_data[need:]

                pool_slice_time += time.time() - t1
                # ============================

                now = time.time()
                elapsed = now - t_start

                print(
                    f"[Consumer] batch {self.batch_idx}, nnz={need}, "
                    f"elapsed={elapsed:.2f}s, "
                    f"batch/s={self.batch_idx / (elapsed + 1e-8):.2f}"
                )

                self.batch_idx += 1
                with self.total_batches.get_lock():
                    self.total_batches.value += 1

            # ============================================================
            # 情况 2：pool 不够 -> 读 slot
            # ============================================================
            else:

                flag = self.shm_flags[slot_id]

                # 🔥 等待时间统计
                t_wait_start = time.time()
                while flag.value == 0:
                    time.sleep(0.0001)
                wait_slot_time += time.time() - t_wait_start

                # 🔥 读取共享内存时间统计
                t_read_start = time.time()

                shm_idx = self.shm_indices_pool[slot_id]
                shm_val = self.shm_data_pool[slot_id]

                indices = np.ndarray(
                    shm_idx.size // np.int32().nbytes,
                    dtype=np.int32,
                    buffer=shm_idx.buf,
                )

                data = np.ndarray(
                    shm_val.size // np.float32().nbytes,
                    dtype=np.float32,
                    buffer=shm_val.buf,
                )

                read_shm_time1 += time.time() - t_read_start
                # 0.0030s (0.09%)

                t_read_start1 = time.time()
                self.pool_indices = np.concatenate([self.pool_indices, indices])
                self.pool_data = np.concatenate([self.pool_data, data])
                read_shm_time2 += time.time() - t_read_start1
                # 1.4709s (42.65%) todo 优化2 很耗时

                flag.value = 0

                slot_id = (slot_id + 1) % self.slot_num

        # ============================================================
        # 🔥 最终统计输出
        # ============================================================
        total_time = time.time() - t_start

        print("\n================ Consumer Time Analysis ================")
        print(f"Total time        : {total_time:.4f}s")
        print(f"Wait slot time    : {wait_slot_time:.4f}s ({wait_slot_time / total_time * 100:.2f}%)")
        print(f"Read SHM time1     : {read_shm_time1:.4f}s ({read_shm_time1 / total_time * 100:.2f}%)")
        print(f"Read SHM time2     : {read_shm_time2:.4f}s ({read_shm_time2 / total_time * 100:.2f}%)")
        print(f"Build CSR time1    : {build_csr_time1:.4f}s ({build_csr_time1 / total_time * 100:.2f}%)")
        print(f"Build CSR time2    : {build_csr_time2:.4f}s ({build_csr_time2 / total_time * 100:.2f}%)")
        print(f"Build CSR time3    : {build_csr_time3:.4f}s ({build_csr_time3 / total_time * 100:.2f}%)")
        print(f"Build CSR time4    : {build_csr_time4:.4f}s ({build_csr_time4 / total_time * 100:.2f}%)")
        print(f"Pool slice time   : {pool_slice_time:.4f}s ({pool_slice_time / total_time * 100:.2f}%)")
        print("========================================================\n")

        conn.close()

    # --------------------------------------------------
    # todo 运行入口函数
    # --------------------------------------------------
    def run(self):
        t0 = time.time()

        # todo 多进程 ，producer 生产者，从 X_CSR_data 提取出 indices, data 写入共享内存
        producers = []
        for i in range(self.producer_num):
            p = mp.Process(target=self._producer, args=(i,))
            p.start()
            producers.append(p)

        # todo 单进程 ，consumer 消费者，从 共享内存 中取出 indices, data ， 并与 indptr构建 CSR
        consumer_p = mp.Process(target=self._consumer)
        consumer_p.start()

        # 等待所有进程结束
        for p in producers:
            p.join()
        consumer_p.join()

        # 🔹 统一 unlink 关闭 所有共享内存
        for name in self.shm_to_unlink:
            try:
                SharedMemory(name=name).unlink()
            except FileNotFoundError:
                pass

        dt = time.time() - t0
        print("\n[Done]")
        print(f"Total batches: {self.total_batches.value}")
        print(f"Time: {dt:.2f}s")
        print(f"batch/s: {self.total_batches.value / dt:.2f}")



# 测试入口
if __name__ == "__main__":

    # 建立索引
    # prepare_partition_index_only_p(db_path=r"E:\python\scAtlas\Package\python-version-scAtlasAnalysis\test\database\test_819200.sasql")

    fetcher = CSRBatchFetcherMP_SharedMemNoQueue(
        db_path=r"E:\python\scAtlas\Package\python-version-scAtlasAnalysis\test\database\test_819200.sasql",
        batch_size=2048,
        producer_num=10,
        slot_num=10,  # 循环池大小
    )
    fetcher.run()
