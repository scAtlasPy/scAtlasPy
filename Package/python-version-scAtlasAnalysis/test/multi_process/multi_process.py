import duckdb
import numpy as np
import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
import time
import scipy.sparse as sp

# ==================================================
# todo 错误代码 多进程 + 共享内存 ； 当前 最快 120 -190 b/s
# ==================================================
class CSRBatchFetcherMP_SharedMemNoQueue:
    """
    多进程 Arrow 解包 + 单进程 CSR batch
    producer 直接写入共享内存，consumer 直接读取
    Queue 仅用于通知 batch 是否完成（可选）
    """

    # todo 1. 初始化相关变量 + 预计算 batch_nnz + 预定义 共享内存
    def __init__(self, db_path, batch_size=2048, fetch_rows=2048*2000, n_producers=4, shm_pool_size=10):
        self.db_path = db_path
        self.batch_size = batch_size
        self.fetch_rows = fetch_rows #  fetch_record_batch ，每次的读取数量
        self.n_producers = n_producers

        # consumer 私有 pool
        self.pool_indices = np.empty(0, dtype=np.int32)
        self.pool_data = np.empty(0, dtype=np.float32)

        # 🔹 预计算 batch nnz
        self.batch_nnz = self._prepare_batch_nnz_sql()
        self.batch_idx = 0
        self.total_batches = mp.Value('i', 0)
        # OS 级共享内存，值存放在共享内存里，所有进程看到的是 同一块内存
        # 是一个 multiprocessing.Lock
        # 防止 多个进程同时写导致竞态

        # 🔹 全局 seq_id，用于多 producer 同步
        self.global_seq = mp.Value('i', 0)

        # 🔹 共享内存池
        self.shm_pool_size = shm_pool_size # 共享内存池 槽位slot个数
        self.shm_indices_pool = []   # 存 indices -- gene_id
        self.shm_data_pool = []      # 存 data
        self.shm_flags = []         # 0=空 没数据, 1 = 非空 有数据
        self.shm_to_unlink = []     # 进程结束统一释放

        # 🔹 初始化共享内存池
        max_nnz = max(self.batch_nnz)  # 🔹 动态计算最大 nnz （100批里面最大的nzz）
        self.fetch_rows = max_nnz #  fetch_record_batch ，每次的读取数量
        for i in range(shm_pool_size):
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


    # --------------------------------------------------
    # todo 2. SQL 预计算 batch_nnz
    # --------------------------------------------------
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

    # -----------------------------
    # todo 3. 多进程 ，producer 生产者，从 X_CSR_data 提取出 indices, data 写入共享内存
    # -----------------------------
    def _producer(self, pid):
        conn = duckdb.connect(self.db_path, read_only=True)
        rbr = conn.execute(
            "SELECT indices, data FROM X_CSR_data"
        ).fetch_record_batch(rows_per_batch=self.fetch_rows)

        for rid,rb in enumerate(rbr):  # todo 有10个for，如何保证1 拿到 1 ， 2拿到 2
            # 🔹 获取全局 seq_id
            with self.global_seq.get_lock():
                # 抢全局 seq_id（并发核心）
                if self.global_seq.value >= len(self.batch_nnz):
                    break # 所有 batch 都已经分配完，当前 Producer 立即退出
                seq_id = self.global_seq.value
                self.global_seq.value += 1
                # 每个 batch：拿到一个 唯一、严格递增的编号：
                # 不会重复 ； 不会跳号
            print(f"日志统计数量！！！")
            print(f"当前的 rid 是 {rid}")
            print(f"当前的 seq_id 是 { seq_id }")

            indices = rb.column(0).to_numpy()
            data = rb.column(1).to_numpy()

            # 🔹 slot 对应 seq_id，保证顺序正确
            slot = seq_id % self.shm_pool_size
            # slot 0 = 0 / 10
            # slot i 始终对应 batch ： i, i+pool_size, i+2*pool_size, ...
            shm_idx = self.shm_indices_pool[slot]
            shm_val = self.shm_data_pool[slot]
            flag = self.shm_flags[slot]
            # shm_flag = mp.Value('i', 0)  # 0=empty, 1=full
            # 共享的 int， 所有进程都能看到。自带 Lock

            # slot 的完整结构
            # ┌─────────────── slot i ───────────────┐
            # │ indices SHM (int32[max_nnz])          │
            # │ data    SHM (float32[max_nnz])        │
            # │ flag    (0 / 1)     0=空 没数据, 1 = 非空 有数据  │
            # └──────────────────────────────────────┘

            # 🔹 等待 consumer 读取完毕， Producer 在等：这个 slot 是否已经被 Consumer 用完
            wait_start = time.time()
            while flag.value != 0: # 只要这个 slot 还不是空的，我就不能写，就需要等
                time.sleep(0.0001)
            wait_end = time.time()
            if wait_end - wait_start > 0.01: # 如果 Producer 等待超过 10 ms，输出调试语句
                print(f"[Producer-{pid}] waited {wait_end - wait_start:.4f}s for slot {slot}")
                # 可能原因： slot 太少？Consumer 太慢？batch_size 太大？nnz 分布不均？
            # flag == 0：slot 是空的；Producer 可以写
            # flag == 1：Consumer 还没读完； Producer 必须等

            # 🔹 将数据库中读取到的 indices + data 写入共享内存 shm_idx + shm_val ，（唯一一次 内存copy）
            np.ndarray(indices.shape, dtype=indices.dtype, buffer=shm_idx.buf)[:] = indices
            np.ndarray(data.shape, dtype=data.dtype, buffer=shm_val.buf)[:] = data
            # [:] 把 data 里的所有元素，逐字节拷贝到共享内存 buffer 中
            # DuckDB / Arrow
            #       │
            #       ▼
            #  NumPy data (私有内存) 临时变量 data
            #       │   memcpy (唯一一次 copy)
            #       ▼
            #  SharedMemory.buf (OS 共享内存 shm_val )

            # 🔹 标记 slot 0=空 没数据, 1 = 非空 有数据
            flag.value = 1 # 写入完成，该slot有数据， Consumer 现在可以读
            print(f"[Producer-{pid}] wrote batch {seq_id} to slot {slot}, nnz={len(indices)}")

        print(f"[Producer-{pid}] done, last_written batch: {seq_id}")
        conn.close()

    # -----------------------------
    # todo 4. 单进程 ，consumer 消费者，从 共享内存 中取出 indices, data ， 并与 indptr构建 CSR
    # -----------------------------
    def _consumer(self):
        conn = duckdb.connect(self.db_path, read_only=True)

        # 🔹 读取 indptr
        fetch_record_indptr = conn.execute("SELECT indptr FROM X_CSR_indptr").fetch_record_batch(
            rows_per_batch=self.batch_size)
        record_indptr_queue = mp.Queue()
        for record_indptr in fetch_record_indptr:
            record_indptr_queue.put(record_indptr)

        global_indptr_offset = 0
        t_start = time.time()

        while self.batch_idx < len(self.batch_nnz): # len(self.batch_nnz)批次数量

            slot = self.batch_idx % self.shm_pool_size
            # 获取 slot编号； 环形缓冲
            # slot = 0 = self.batch_idx
            flag = self.shm_flags[slot] # 当前的 slot 是否为空 的 标识
            # flag = 1 有数据

            if flag.value == 1: # 当前 slot 是写好的，可进行数据读取
                shm_idx = self.shm_indices_pool[slot] # 将该slot的共享内存 标识为 SharedMemory 对象 —— shm_idx
                shm_val = self.shm_data_pool[slot]

                # 🔹 读取数据 (零拷贝 view)
                indices = np.ndarray(shm_idx.size // np.int32().nbytes, dtype=np.int32, buffer=shm_idx.buf)
                data = np.ndarray(shm_val.size // np.float32().nbytes, dtype=np.float32, buffer=shm_val.buf)
                # 将共享内存 shm_idx 中的 数据 零拷贝 给 indices
                # shm_val --> data , 给了多少数据

                # indices.ndarray ──►  shm_idx.buf (共享内存)
                #  self.pool_indices = np.empty(0, dtype=np.int32)
                #  self.pool_data = np.empty(0, dtype=np.float32)
                #  self.pool_data <--- data , 从这里拿数据
                #

                # 🔹 取出当前 batch nnz
                need = self.batch_nnz[self.batch_idx]
                vals = data[:need]    # csr三元组 1 — data
                cols = indices[:need] # csr三元组 2 — gene_id

                # 🔹 取indptr
                indptr_now = record_indptr_queue.get().column(0).to_numpy()
                last_val = indptr_now[-1]
                indptr_now = indptr_now - global_indptr_offset
                indptr_now = np.concatenate(([0], indptr_now)) # csr三元组 3 — indptr
                global_indptr_offset = last_val

                n_genes = int(cols.max()) + 1 if len(cols) else 0
                # 🔹 构建 csr 格式的 X
                X = sp.csr_matrix(
                    (vals, cols, indptr_now),
                    shape=(self.batch_size, n_genes),
                    dtype=np.float32,
                )

                # 🔹 打印信息
                now = time.time()
                elapsed = now - t_start
                print(
                    f"[Consumer] batch {self.batch_idx}, nnz={need}, elapsed={elapsed:.2f}s, batch/s={self.batch_idx / (elapsed + 1e-8):.2f}"
                )

                flag.value = 0 # 释放 slot（置空）， 告诉producer， 该slot的数据已经读取完毕
                self.batch_idx += 1 # batch id 顺序增加
                with self.total_batches.get_lock(): # 拿到 Value 自带的互斥锁，保证多进程正确性
                    self.total_batches.value += 1 # 成功消费了一个 batch
                    # 统计总完成 batch 数（多进程安全）
            else: # flag.value == 0 ，等待 producer 写入
                time.sleep(0.0001)

        print("[Consumer] done")
        conn.close()

    # --------------------------------------------------
    # todo 运行入口函数
    # --------------------------------------------------
    def run(self):
        t0 = time.time()

        # todo 3. 多进程 ，producer 生产者，从 X_CSR_data 提取出 indices, data 写入共享内存
        producers = []
        for i in range(self.n_producers):
            p = mp.Process(target=self._producer, args=(i,))
            p.start()
            producers.append(p)

        # todo 4. 单进程 ，consumer 消费者，从 共享内存 中取出 indices, data ， 并与 indptr构建 CSR
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
        db_path=r"E:\python\scAtlas\Package\python-version-scAtlasAnalysis\test\database\test_204800.sasql",
        batch_size=2048,
        fetch_rows=2048 * 2000,
        n_producers=10,
        shm_pool_size=10,  # 循环池大小
    )
    fetcher.run()
# todo
#  一些bug： 后续解决
#    情况1： n_producers=1,会卡住， [Consumer] batch 217, nnz=1525607, elapsed=5.61s, batch/s=38.69 不继续
# 当 n_producers = 1：
# producer 只能顺序写 batch
# seq_id = 0,1,2,...
# consumer 严格顺序消费 batch_idx = 0,1,2,...


# todo
#    情况2：n_producers > shm_pool_size, 会卡住
# [Consumer] batch 394, nnz=1662575, elapsed=2.53s, batch/s=155.50
# [Producer-0] done, last_written batch: 396
# [Producer-1] done, last_written batch: 397
# [Producer-5] done, last_written batch: 398
# [Producer-2] done, last_written batch: 399
# 会有2个p 在等同一个 slot ，