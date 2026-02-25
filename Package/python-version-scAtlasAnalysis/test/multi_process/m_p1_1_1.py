import duckdb
import numpy as np
import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
import time
import scipy.sparse as sp

# ==================================================
# 多进程 + 共享内存 (零 meta Queue)
# ==================================================
class CSRBatchFetcherMP_SharedMemNoQueue:
    """
    多进程 Arrow 解包 + 单进程 CSR batch
    producer 直接写入共享内存，consumer 直接读取
    Queue 仅用于通知 batch 是否完成（可选）
    """

    def __init__(self, db_path, batch_size=2048, fetch_rows=2048*2000, n_producers=4, shm_pool_size=10):
        self.db_path = db_path
        self.batch_size = batch_size
        self.fetch_rows = fetch_rows
        self.n_producers = n_producers

        # consumer 私有 pool
        self.pool_indices = np.empty(0, dtype=np.int32)
        self.pool_data = np.empty(0, dtype=np.float32)

        # 🔹 预计算 batch nnz
        self.batch_nnz = self._prepare_batch_nnz_sql()
        self.batch_idx = 0
        self.total_batches = mp.Value('i', 0)

        # 🔹 共享内存池
        self.shm_pool_size = shm_pool_size
        self.shm_indices_pool = []
        self.shm_data_pool = []
        self.shm_flags = []  # 用于标记该 slot 是否 ready
        self.shm_to_unlink = []

        for i in range(shm_pool_size):
            max_nnz = fetch_rows * 10
            shm_idx = SharedMemory(create=True, size=max_nnz * np.int32().nbytes)
            shm_val = SharedMemory(create=True, size=max_nnz * np.float32().nbytes)
            shm_flag = mp.Value('i', 0)  # 0=empty, 1=full, 2=consumed

            self.shm_indices_pool.append(shm_idx)
            self.shm_data_pool.append(shm_val)
            self.shm_flags.append(shm_flag)

            self.shm_to_unlink.extend([shm_idx.name, shm_val.name])

    # --------------------------------------------------
    # SQL 预计算 batch_nnz
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

    # --------------------------------------------------
    # scan process
    # --------------------------------------------------
    def _scan_process(self):
        conn = duckdb.connect(self.db_path, read_only=True)
        rel = conn.execute("SELECT indices, data FROM X_CSR_data").fetch_record_batch(rows_per_batch=self.fetch_rows)
        for seq_id, rb in enumerate(rel):
            # 🔹 放到共享内存池的逻辑由 producer 处理
            self.rb_queue.put((seq_id, rb))
        # 🔹 通知 producers 完成
        for _ in range(self.n_producers):
            self.rb_queue.put(None)
        conn.close()

    # ... 原来的类保持不变，只修改 _producer 和 _consumer 添加调试信息

    def _producer(self, pid):
        shm_index = 0
        conn = duckdb.connect(self.db_path, read_only=True)
        rel = conn.execute("SELECT indices, data FROM X_CSR_data").fetch_record_batch(rows_per_batch=self.fetch_rows)

        for seq_id, rb in enumerate(rel):
            indices = rb.column(0).to_numpy()
            data = rb.column(1).to_numpy()

            # 🔹 找到循环池 slot
            shm_idx = self.shm_indices_pool[shm_index % self.shm_pool_size]
            shm_val = self.shm_data_pool[shm_index % self.shm_pool_size]
            flag = self.shm_flags[shm_index % self.shm_pool_size]

            # 🔹 等待 consumer 读取完毕
            wait_start = time.time()
            while flag.value != 0:
                time.sleep(0.0001)
            wait_end = time.time()
            if wait_end - wait_start > 0.01:
                print(f"[Producer-{pid}] waited {wait_end - wait_start:.4f}s for slot {shm_index % self.shm_pool_size}")

            # 🔹 写入数据
            shm_idx_arr = np.ndarray(indices.shape, dtype=indices.dtype, buffer=shm_idx.buf)
            shm_val_arr = np.ndarray(data.shape, dtype=data.dtype, buffer=shm_val.buf)
            shm_idx_arr[:] = indices
            shm_val_arr[:] = data

            # 🔹 设置标记
            flag.value = 1  # full

            print(f"[Producer-{pid}] wrote batch {seq_id} to slot {shm_index % self.shm_pool_size}, nnz={len(indices)}")

            shm_index += 1

        print(f"[Producer-{pid}] done")
        conn.close()

    def _consumer(self):
        conn = duckdb.connect(self.db_path, read_only=True)

        # 🔹 读取 indptr record
        fetch_record_indptr = conn.execute("SELECT indptr FROM X_CSR_indptr").fetch_record_batch(
            rows_per_batch=self.batch_size)
        record_indptr_queue = mp.Queue()
        for record_indptr in fetch_record_indptr:
            record_indptr_queue.put(record_indptr)

        global_indptr_offset = 0
        finished_producers = 0
        shm_index = 0
        t_start = time.time()

        while finished_producers < self.n_producers:
            flag = self.shm_flags[shm_index % self.shm_pool_size]

            if flag.value == 1:  # producer 已写入
                shm_idx = self.shm_indices_pool[shm_index % self.shm_pool_size]
                shm_val = self.shm_data_pool[shm_index % self.shm_pool_size]

                # 🔹 读取数据 (零拷贝 view)
                indices = np.ndarray(shm_idx.size // np.int32().nbytes, dtype=np.int32, buffer=shm_idx.buf)
                data = np.ndarray(shm_val.size // np.float32().nbytes, dtype=np.float32, buffer=shm_val.buf)

                # 🔹 取出当前 batch nnz
                need = self.batch_nnz[self.batch_idx]
                cols = indices[:need]
                vals = data[:need]

                # 🔹 indptr
                indptr_now = record_indptr_queue.get().column(0).to_numpy()
                last_val = indptr_now[-1]
                indptr_now = indptr_now - global_indptr_offset
                indptr_now = np.concatenate(([0], indptr_now))
                global_indptr_offset = last_val

                n_genes = int(cols.max()) + 1 if len(cols) else 0
                X = sp.csr_matrix(
                    (vals, cols, indptr_now),
                    shape=(self.batch_size, n_genes),
                    dtype=np.float32,
                )

                # 🔹 打印信息
                now = time.time()
                elapsed = now - t_start
                print(
                    f"[Consumer] batch {self.batch_idx}, nnz={need}, elapsed={elapsed:.2f}s, batch/s={self.batch_idx / (elapsed + 1e-8):.2f}")

                # 🔹 标记 slot 可写
                flag.value = 0
                shm_index += 1
                self.batch_idx += 1
                with self.total_batches.get_lock():
                    self.total_batches.value += 1
            else:
                time.sleep(0.0001)

        print("[Consumer] done")
        conn.close()

    # --------------------------------------------------
    # run
    # --------------------------------------------------
    def run(self):
        t0 = time.time()

        producers = []
        for i in range(self.n_producers):
            p = mp.Process(target=self._producer, args=(i,))
            p.start()
            producers.append(p)

        consumer_p = mp.Process(target=self._consumer)
        consumer_p.start()

        for p in producers:
            p.join()
        consumer_p.join()

        # 🔹 统一 unlink 所有共享内存
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


if __name__ == "__main__":
    fetcher = CSRBatchFetcherMP_SharedMemNoQueue(
        db_path=r"E:\python\scAtlas\Package\python-version-scAtlasAnalysis\test\database\test_819200.sasql",
        batch_size=2048,
        fetch_rows=2048 * 2000,
        n_producers=5,
        shm_pool_size=10,  # 🔹 循环池大小
    )
    fetcher.run()
