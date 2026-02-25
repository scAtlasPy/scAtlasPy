import duckdb
import numpy as np
import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
import time
import scipy.sparse as sp

# ==================================================
# conn.execute("PRAGMA threads=10")
# todo 正确代码 ： 多进程 + 共享内存 ； 当前 最快 50 b/s 左右
#       多进程的 producer，每个进程都读取全部的fetch_record_batch，然后只取需要的 batch，冗余！
# ==================================================
class CSRBatchFetcherMP_SharedMemNoQueue:
    """
    多进程 Arrow 解包 + 单进程 CSR batch
    producer 直接写入共享内存，consumer 直接读取
    Queue 仅用于通知 batch 是否完成（可选）
    """

    # todo 1. 初始化相关变量 + 预计算 batch_nnz + 预定义 共享内存
    def __init__(self, db_path, batch_size=2048, producer_num=4, slot_num=10):

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
            # 这两块共享内存的名字， 存进了一个 list （self.shm_to_unlink）
            # name 是 唯一稳定的跨进程标识
            # shm.unlink()， 从系统删除
            # shm.close()， 当前进程不再使用， close 不会释放内存
            # 所有进程结束后 → 父进程统一 unlink

    # todo 2. 预计算 indptr_queue
    def _prepare_indptr(self):
        # 🔹 读取 indptr
        conn = duckdb.connect(self.db_path, read_only=True)
        conn.execute("PRAGMA threads=10")

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
        conn.execute("PRAGMA threads=10")
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
        conn.execute("PRAGMA threads=10")

        result = conn.execute(
            "SELECT indices, data FROM X_CSR_data"
        ).fetch_record_batch(rows_per_batch=self.fetch_size)

        for rid, rb in enumerate(result):
            if rid % self.slot_num == pid: # 该进程 要 该批次 ；将数据库中的数据 ——> 共享内存中

                indices = rb.column(0).to_numpy()
                data = rb.column(1).to_numpy()

                slot_id = pid # 进程 pid 将 rid 的数据 ——> slot_id 的共享内存里
                print(f"[Producer-{pid}] : Start 将 fetch 批次 rid = {rid} 写入 slot_{slot_id} ")

                shm_idx = self.shm_indices_pool[slot_id]
                shm_val = self.shm_data_pool[slot_id]
                flag = self.shm_flags[slot_id] # shm_flag = mp.Value('i', 0)  # 0=空, 1=非空 ； 自带 Lock
                # ┌─────────────── slot i ───────────────┐
                # │ indices SHM (int32[max_nnz])         │
                # │ data    SHM (float32[max_nnz])       │
                # │ flag    (0 / 1)     0=空, 1 = 非空    │
                # └──────────────────────────────────────┘

                wait_start = time.time()
                while flag.value != 0: # 只要这个 slot 还不是空的，我就不能写，就需要等
                    time.sleep(0.0001) # todo 可改进
                wait_end = time.time()
                if wait_end - wait_start > 0.01: # 如果 Producer 等待超过 10 ms，输出调试语句
                    print(f"[Producer-{pid}] 等待slot_{slot_id} 的时间为 {wait_end - wait_start:.4f}s")

                # 🔹 将数据库中读取到的 indices + data 写入共享内存 shm_idx + shm_val ，（唯一一次 内存copy）
                np.ndarray(indices.shape, dtype=indices.dtype, buffer=shm_idx.buf)[:] = indices
                np.ndarray(data.shape, dtype=data.dtype, buffer=shm_val.buf)[:] = data

                flag.value = 1 # 写入完成, 1 = 非空 有数据. Consumer 现在可以读
                print(f"[Producer-{pid}] : Done 将 fetch 批次 rid = {rid} 写入 slot_{slot_id} ,nnz={len(indices)}")

        print(f"[Producer-{pid}] 进程完成")
        conn.close()

    # -----------------------------
    # todo 5. 单进程 ，consumer 消费者，从 共享内存 中取出 indices, data + indptr构建 CSR
    # -----------------------------
    def _consumer(self):

        conn = duckdb.connect(self.db_path, read_only=True)
        conn.execute("PRAGMA threads=10")

        global_indptr_offset = 0 # 取indptr的全局偏移量
        slot_id = 0 # 从 第0个slot开始拿数据
        t_start = time.time() # 时间统计

        while self.batch_idx < self.batch_num :

            need = self.batch_nnz[self.batch_idx]  # 当前 batch 需要的 nnz 数量

            # todo data 数量够， 取 数据
            if self.pool_data.size >= need: # 私有内存池中的data 数量足够

                # 从内存pool中获取 need 个 数据
                vals = self.pool_data[:need]  # csr三元组 1 — data
                cols = self.pool_indices[:need]  # csr三元组 2 — gene_id

                # 🔹 取indptr
                indptr_now = self.indptr_queue.get().column(0).to_numpy()
                last_val = indptr_now[-1]
                indptr_now = indptr_now - global_indptr_offset
                indptr_now = np.concatenate(([0], indptr_now))  # csr三元组 3 — indptr
                global_indptr_offset = last_val

                n_genes = int(cols.max()) + 1 if len(cols) else 0

                X = sp.csr_matrix(  # 🔹 构建 csr 格式的 X
                    (vals, cols, indptr_now),
                    shape=(self.batch_size, n_genes),
                    dtype=np.float32,
                )

                # 🔹 打印时间信息
                now = time.time()
                elapsed = now - t_start
                print(
                    f"[Consumer] batch {self.batch_idx}, nnz={need}, elapsed={elapsed:.2f}s, batch/s={self.batch_idx / (elapsed + 1e-8):.2f}"
                )

                # 移动 pool（丢弃已消费的 nnz）
                self.pool_indices = self.pool_indices[need:]
                self.pool_data = self.pool_data[need:]

                print(f"[Consumer] : 已完成 batch_id = {self.batch_idx}，nnz={need}")

                self.batch_idx = self.batch_idx + 1  # batch id 顺序增加
                with self.total_batches.get_lock(): # 拿到 Value 自带的互斥锁
                    self.total_batches.value += 1 # 统计总完成 batch 数（多进程安全）


            # todo data 数量不够， 存 数据
            else: # self.pool_data.size < need 私有内存池中的data 数量不够

                flag = self.shm_flags[slot_id]  # 当前的 slot 是否为空 的 标识

                if flag.value == 0: # 当前 slot 没数据，等待
                    time.sleep(0.0001)
                    continue

                # if flag.value == 1: # 当前 slot 有数据，可读
                shm_idx = self.shm_indices_pool[slot_id] # 共享内存 gene_id
                shm_val = self.shm_data_pool[slot_id]

                # 🔹 读取数据 (零拷贝 view)
                indices = np.ndarray(shm_idx.size // np.int32().nbytes, dtype=np.int32, buffer=shm_idx.buf)
                data = np.ndarray(shm_val.size // np.float32().nbytes, dtype=np.float32, buffer=shm_val.buf)

                # 将 indices, data 拼接到 内存 pool
                self.pool_indices = np.concatenate([self.pool_indices, indices])
                self.pool_data = np.concatenate([self.pool_data, data])

                flag.value = 0  # 释放 slot（置空）， 告诉producer， 该slot的数据已经读取完毕
                print(f"[Consumer] : 已读取 slot_{slot_id}的数据")

                slot_id = (slot_id + 1) % self.slot_num  # 拿下一块数据

        print("[Consumer] : Done")
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


# ==================================================
# 测试入口
# ==================================================
if __name__ == "__main__":
    fetcher = CSRBatchFetcherMP_SharedMemNoQueue(
        db_path=r"E:\python\scAtlas\Package\python-version-scAtlasAnalysis\test\database\test_819200.sasql",
        batch_size=2048,
        producer_num=1,
        slot_num=1,  # 循环池大小
    )
    fetcher.run()