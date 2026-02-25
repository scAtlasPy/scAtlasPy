import multiprocessing as mp
from multiprocessing import Semaphore, Process, shared_memory, Queue
import threading
import numpy as np
import scipy.sparse as sp
import duckdb
from anndata import AnnData
from datetime import datetime
import os

# ============================================================
# 1️⃣ 子进程内部多线程生产器
# ============================================================
class MultiThreadCSRProducer:
    def __init__(self, db_path, total_num_cells, batch_size,
                 n_threads, shm_pools, sem_empty, sem_full, ready_queue, drop_last=False):
        self.db_path = db_path
        self.total_num_cells = total_num_cells
        self.batch_size = batch_size
        self.n_threads = n_threads
        self.drop_last = drop_last
        self.shm_pools = shm_pools
        self.sem_empty = sem_empty
        self.sem_full = sem_full
        self.ready_queue = ready_queue

        if drop_last:
            self.num_batches = total_num_cells // batch_size
        else:
            self.num_batches = (total_num_cells + batch_size - 1) // batch_size

        self._batch_cursor = 0
        self._cursor_lock = threading.Lock()  # 🔹 线程锁必须在子进程内创建
        self.threads = []
        self._start_threads()

    def _start_threads(self):
        for tid in range(self.n_threads):
            t = threading.Thread(target=self._worker, name=f"Worker-{tid}", daemon=True)
            t.start()
            self.threads.append(t)

    def _worker(self):
        conn = duckdb.connect(self.db_path, read_only=True)  # 🔹 每个线程自己的连接
        thread_name = threading.current_thread().name
        pid = os.getpid()

        while True:
            with self._cursor_lock:
                if self._batch_cursor >= self.num_batches:
                    return
                batch_id = self._batch_cursor
                self._batch_cursor += 1

            offset = batch_id * self.batch_size
            if not self.drop_last and batch_id == self.num_batches - 1:
                cell_count = self.total_num_cells - offset
            else:
                cell_count = self.batch_size
            if cell_count == 0:
                continue

            df = conn.execute(f"""
                SELECT indices, data, cell_index
                FROM X_CSR_data
                WHERE cell_index >= {offset} AND cell_index < {offset + cell_count}
                ORDER BY id
            """).fetch_arrow_table().to_pandas()

            rows = df["cell_index"].to_numpy() - offset
            cols = df["indices"].to_numpy()
            vals = df["data"].to_numpy()
            gene_count = conn.execute("SELECT COUNT(*) FROM var").fetchone()[0]

            X = sp.coo_matrix((vals, (rows, cols)), shape=(cell_count, gene_count)).tocsr()

            self.sem_empty.acquire()
            pool_index = batch_id % len(self.shm_pools['indptr'])
            np.ndarray(X.indptr.shape, dtype=X.indptr.dtype, buffer=self.shm_pools['indptr'][pool_index].buf)[:] = X.indptr
            np.ndarray(X.indices.shape, dtype=X.indices.dtype, buffer=self.shm_pools['indices'][pool_index].buf)[:] = X.indices
            np.ndarray(X.data.shape, dtype=X.data.dtype, buffer=self.shm_pools['data'][pool_index].buf)[:] = X.data
            self.sem_full.release()
            self.ready_queue.put(batch_id)

    def join(self):
        for t in self.threads:
            t.join()


# ============================================================
# 2️⃣ 子进程入口
# ============================================================
def csr_producer_process(db_path, total_num_cells, batch_size, n_threads,
                         shm_pools, sem_empty, sem_full, ready_queue, drop_last):
    producer = MultiThreadCSRProducer(
        db_path=db_path,
        total_num_cells=total_num_cells,
        batch_size=batch_size,
        n_threads=n_threads,
        shm_pools=shm_pools,
        sem_empty=sem_empty,
        sem_full=sem_full,
        ready_queue=ready_queue,
        drop_last=drop_last
    )
    producer.join()


# ============================================================
# 3️⃣ 主进程消费函数（非顺序消费）
# ============================================================
def minibatch_scan_order_mproc(db_path, batch_size, n_processes=2, n_threads=4, pool_size=10, drop_last=False):
    shm_pools = {'indptr': [], 'indices': [], 'data': []}
    for _ in range(pool_size):
        shm_pools['indptr'].append(shared_memory.SharedMemory(create=True, size=(batch_size+1)*4))
        shm_pools['indices'].append(shared_memory.SharedMemory(create=True, size=batch_size*3000*4))
        shm_pools['data'].append(shared_memory.SharedMemory(create=True, size=batch_size*3000*8))

    sem_empty = Semaphore(pool_size)
    sem_full = Semaphore(0)
    ready_queue = Queue()

    conn = duckdb.connect(db_path, read_only=True)
    total_num_cells = conn.execute("SELECT COUNT(*) FROM obs").fetchone()[0]
    conn.close()

    processes = []
    for _ in range(n_processes):
        p = Process(target=csr_producer_process,
                    args=(db_path, total_num_cells, batch_size, n_threads,
                          shm_pools, sem_empty, sem_full, ready_queue, drop_last))
        p.start()
        processes.append(p)

    num_batches = total_num_cells // batch_size + (0 if drop_last else int(total_num_cells % batch_size > 0))
    consumed = 0
    conn = duckdb.connect(db_path, read_only=True)

    while consumed < num_batches:
        sem_full.acquire()
        batch_id = ready_queue.get()
        pool_index = batch_id % pool_size

        if batch_id == num_batches - 1 and not drop_last:
            cell_count = total_num_cells - batch_id*batch_size
        else:
            cell_count = batch_size

        gene_count = conn.execute("SELECT COUNT(*) FROM var").fetchone()[0]

        indptr = np.ndarray((cell_count+1,), dtype=np.int32, buffer=shm_pools['indptr'][pool_index].buf)
        indices = np.ndarray((indptr[-1],), dtype=np.int32, buffer=shm_pools['indices'][pool_index].buf)
        data = np.ndarray((indptr[-1],), dtype=np.float64, buffer=shm_pools['data'][pool_index].buf)
        X = sp.csr_matrix((data, indices, indptr), shape=(cell_count, gene_count))

        offset = batch_id * batch_size
        sub_obs = conn.execute(f"SELECT * FROM obs LIMIT {cell_count} OFFSET {offset}").fetchdf()
        sub_var = conn.execute("SELECT * FROM var").fetchdf()
        sub_obs.index = sub_obs["cell_id"].astype(str)
        sub_var.index = sub_var["gene_id"].astype(str)

        adata = AnnData(X=X, obs=sub_obs, var=sub_var)
        adata.obs_names = sub_obs.index
        adata.var_names = sub_var.index

        yield adata

        sem_empty.release()
        consumed += 1

    # 关闭进程和共享内存
    for p in processes:
        p.terminate()
        p.join()
    for i in range(pool_size):
        shm_pools['indptr'][i].close()
        shm_pools['indptr'][i].unlink()
        shm_pools['indices'][i].close()
        shm_pools['indices'][i].unlink()
        shm_pools['data'][i].close()
        shm_pools['data'][i].unlink()
    conn.close()


# ============================================================
# 4️⃣ 测试入口
# ============================================================
if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)  # 🔹 Windows 必须

    DB_PATH = r"E:\python\scAtlas\Package\python-version-scAtlasAnalysis\test\database\test_204800.sasql"
    BATCH_SIZE = 2048

    start_time = datetime.now()
    count = 0
    for adata in minibatch_scan_order_mproc(DB_PATH, batch_size=BATCH_SIZE,
                                            n_processes=2, n_threads=4,
                                            pool_size=4, drop_last=False):
        count += 1
    end_time = datetime.now()
    minibatch_time = (end_time - start_time).total_seconds()
    print(f"#### 总批次数量 : {count} ")
    print(f"#### 批次大小 : {BATCH_SIZE} ")
    print(f"#### minibatch_time 总时间 : {minibatch_time:.2f} 秒")
    print(f"#### 1个 minibatch 耗时 : { (minibatch_time / count) :.2f} 秒")
    print(f"#### 每秒 minibatch 数量 : { (count / minibatch_time) :.2f} 个")


