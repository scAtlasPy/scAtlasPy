import multiprocessing as mp
from multiprocessing import Semaphore, Process, shared_memory
import threading
import numpy as np
import scipy.sparse as sp
import duckdb
from anndata import AnnData
from datetime import datetime
import os

# ============================================================
# 1️⃣ 单进程内部多线程 CSR 生产器（共享内存池版）
# ============================================================
class MultiThreadCSRProducer:
    """
    每个子进程内部多线程生成 CSR batch，并写入共享内存池。
    通过 semaphore 管理共享内存池空位。
    """

    def __init__(self, db_path, total_num_cells, batch_size,
                 n_threads, shm_pools, sem_empty, sem_full, drop_last=False):
        self.db_path = db_path
        self.total_num_cells = total_num_cells
        self.batch_size = batch_size
        self.n_threads = n_threads # 该进程内部开的线程数
        self.drop_last = drop_last
        self.shm_pools = shm_pools  # 共享内存池（CSR 的 indptr / indices / data）
        self.sem_empty = sem_empty  # 控制空位
        self.sem_full = sem_full    # 控制已写入

        # 计算总批次数
        if drop_last:
            self.num_batches = total_num_cells // batch_size
        else:
            self.num_batches = (total_num_cells + batch_size - 1) // batch_size

        self._batch_cursor = 0 # 线程共享的全局计数器， 下一个应该被哪个线程处理的 batch_id
        self._cursor_lock = threading.Lock() # _batch_cursor 加锁
        self.threads = [] # 保存每个 threading.Thread 对象

        self._start_threads() # 启动内部线程

    # -----------------------------
    # 启动内部线程
    # -----------------------------
    def _start_threads(self):
        for tid in range(self.n_threads): # 同时运行的 worker 线程数
            t = threading.Thread(target=self._worker, # 线程启动后执行的函数
                                 name=f"Worker-{tid}", # 线程名字
                                 daemon=True) # 主线程一结束，这些 worker 线程会被强制终止
            t.start() # 开启线程
            self.threads.append(t)

    # -----------------------------
    # Worker 线程函数
    # -----------------------------
    def _worker(self):
        conn = duckdb.connect(self.db_path, read_only=True) # 每个线程一个 connection
        thread_name = threading.current_thread().name # 线程名称
        process_id = os.getpid() # # 进程名称

        while True:
            # -----------------------------
            # 抢批次
            # -----------------------------
            with self._cursor_lock: # self._cursor_lock 是 threading.Lock() 对象
                # with 是上下文管理器：
                # __enter__ → 获取锁
                # __exit__ → 自动释放锁
                if self._batch_cursor >= self.num_batches:
                    # _batch_cursor 全局批次游标，记录已经分配了多少 batch
                    # self.num_batches：总批次数
                    # 如果 self._batch_cursor >= self.num_batches  已经没有 batch 可以分配了：
                    #  当前线程 直接退出
                    #  避免多余循环 / 无限等待
                    print(f"[DEBUG] 线程名称 {thread_name} ；进程名称 PID={process_id} 无剩余批次，退出")  # 🔹 修改
                    return
                batch_id = self._batch_cursor
                self._batch_cursor += 1
                # print(f"[DEBUG] {thread_name} PID={process_id} 抢到批次 {batch_id}")

            # -----------------------------
            # 计算 batch 偏移量和大小
            # -----------------------------
            offset = batch_id * self.batch_size # 计算 当前 batch 在整体数据中的起始位置
            # 判断是否是最后一个 batch 并处理不整除情况
            if (not self.drop_last and batch_id == self.num_batches - 1
                    and self.total_num_cells % self.batch_size != 0):
                cell_count = self.total_num_cells % self.batch_size
            else:
                cell_count = self.batch_size

            if cell_count == 0: # 跳过空 batch
                continue

            # -----------------------------
            # 从 DuckDB 读取 CSR 数据,
            # -----------------------------
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
            # todo。 csr coo 转换·

            # -----------------------------
            # 等待共享内存池空位
            # -----------------------------
            #抢 batch 成功后再 acquire，确保不会阻塞
            self.sem_empty.acquire()  # 我要占用一个共享内存槽位
            print(f"[DEBUG] 线程编号：{thread_name} 进程编号： PID={process_id} 批次 {batch_id} 获得 sem_empty 空槽位")

            # -----------------------------
            # 分配共享内存池 slot（循环使用）
            # -----------------------------
            pool_index = batch_id % len(self.shm_pools['indptr'])
            shm_indptr = self.shm_pools['indptr'][pool_index]
            shm_indices = self.shm_pools['indices'][pool_index]
            shm_data = self.shm_pools['data'][pool_index]

            # -----------------------------
            # todo 写入共享内存
            # -----------------------------
            np.ndarray(X.indptr.shape, dtype=X.indptr.dtype, buffer=shm_indptr.buf)[:] = X.indptr
            np.ndarray(X.indices.shape, dtype=X.indices.dtype, buffer=shm_indices.buf)[:] = X.indices
            np.ndarray(X.data.shape, dtype=X.data.dtype, buffer=shm_data.buf)[:] = X.data

            print(f"[DEBUG] {thread_name} PID={process_id} 批次 {batch_id} 写入共享内存完成")

            # -----------------------------
            # 通知主进程数据已写入
            # -----------------------------
            self.sem_full.release()
            # todo 生产者-消费者信号量
            #  用 Semaphore（信号量） 来控制共享内存池（或数据缓冲区）的生产者和消费者流量
            #  sem_empty → 表示 可用空槽位，生产者使用 acquire() 前检查是否有空槽位，生产完成后 release() 给消费者用。
            #  sem_full → 表示 已生产但未消费的槽位/批次，消费者使用 acquire() 检查是否有可消费的数据，
            #             消费完成后 release() 给生产者使用
            #  生产者：先 acquire(sem_empty) → 生产数据 → release(sem_full)
            #  消费者：先 acquire(sem_full) → 消费数据 → release(sem_empty)

            # print(f"[DEBUG] {thread_name} PID={process_id} 批次 {batch_id} 释放 sem_full")

    def join(self):
        for t in self.threads:
            t.join()  # 确保所有 worker 都完成任务； 主线程不会提前退出


# ============================================================
# 2️⃣ 子进程入口函数
# ============================================================
def csr_producer_process(db_path, total_num_cells, batch_size, n_threads,
                         shm_pools, sem_empty, sem_full, drop_last):
    # 进程内部 开 多线程
    producer = MultiThreadCSRProducer(
        db_path=db_path,
        total_num_cells=total_num_cells,
        batch_size=batch_size,
        n_threads=n_threads, # 该进程内部开的线程数
        shm_pools=shm_pools, # 共享内存池（CSR 的 indptr / indices / data）
        sem_empty=sem_empty, # 表示“共享内存可写”的信号量
        sem_full=sem_full, # 表示“共享内存已写好”的信号量
        drop_last=drop_last
    )
    producer.join() # 阻塞当前子进程，直到所有生产线程完成


# ============================================================
# 3️⃣ 主进程消费函数（共享内存池版）
# ============================================================
def minibatch_scan_order_mproc(db_path, batch_size, n_processes=2,
                               n_threads=4, pool_size=10, drop_last=False):

    # -----------------------------
    # 创建共享内存池
    # -----------------------------
    shm_pools = {'indptr': [], 'indices': [], 'data': []} #分成3块，存CSR格式的3个数组
    max_indptr_bytes = (batch_size + 1) * 4   # indptr
    max_indices_bytes = batch_size * 3000 * 4 # indices  假设每个cell 有3000个非零值
    max_data_bytes = batch_size * 3000 * 8    # data

    for i in range(pool_size):  # pool_size = 10 共享内存槽位数
        shm_pools['indptr'].append(shared_memory.SharedMemory(create=True, size=max_indptr_bytes))
        shm_pools['indices'].append(shared_memory.SharedMemory(create=True, size=max_indices_bytes))
        shm_pools['data'].append(shared_memory.SharedMemory(create=True, size=max_data_bytes))

    sem_empty = Semaphore(pool_size) # 当前有多少个“空槽位”可以写；
    # 初始值 = pool_size；表示：所有 槽位 都是空的
    sem_full = Semaphore(0) # 当前有多少个“已写好数据”的槽位
    # 初始值 = 0： 主进程一开始没东西可读

    # -----------------------------
    # 总细胞数
    # -----------------------------
    conn = duckdb.connect(db_path, read_only=True)
    total_num_cells = conn.execute("SELECT COUNT(*) FROM obs").fetchone()[0]
    conn.close()

    # -----------------------------
    # 启动子进程
    # -----------------------------
    processes = []  # 子进程
    for _ in range(n_processes): # 子进程数量
        p = Process(target=csr_producer_process, # 子进程 csr_producer_process
                    args=(db_path, total_num_cells, batch_size, n_threads,
                          shm_pools, sem_empty, sem_full, drop_last)) # 这几个对象被 传给了所有子进程
        p.start() # 启动 进程
        processes.append(p)

    # -----------------------------
    # 顺序消费
    # -----------------------------
    num_batches = total_num_cells // batch_size + (0 if drop_last else int(total_num_cells % batch_size > 0))
    # 批次数量
    expected = 0 # todo 唯一决定 consumer 读哪个 batch 的变量；保证按顺序输出

    conn = duckdb.connect(db_path, read_only=True)
    time_obs_total = 0 # 时间统计分析
    time_var_total = 0 # 时间统计分析
    time_adata_total = 0 # 时间统计分析

    while expected < num_batches:
        sem_full.acquire()  # 等待生产者写好 # 当前至少有 1 个 batch 已被写入共享内存池
        # print(f"[DEBUG] 主进程 acquire sem_full 批次 {expected}")

        # producer 不允许在 slot 未释放前写新 batch
        pool_index = expected % pool_size
        # todo 顺序控制
        #  batch 0 → slot 0
        #  batch 1 → slot 1
        #  batch N → slot (N % pool_size)

        shm_indptr = shm_pools['indptr'][pool_index]  # 共享内存
        shm_indices = shm_pools['indices'][pool_index]
        shm_data = shm_pools['data'][pool_index]

        # 先判断是不是最后一个 batch ， 最后一个batch的处理
        if expected == num_batches - 1 and not drop_last: 
            cell_count = total_num_cells - expected * batch_size
        else:
            cell_count = batch_size
            
        gene_count = conn.execute("SELECT COUNT(*) FROM var").fetchone()[0] # 获取 var 表

        indptr = np.ndarray((cell_count + 1,), dtype=np.int32, buffer=shm_indptr.buf) # 共享内存读取，零拷贝
        indices = np.ndarray((indptr[-1],), dtype=np.int32, buffer=shm_indices.buf)
        data = np.ndarray((indptr[-1],), dtype=np.float64, buffer=shm_data.buf)

        X = sp.csr_matrix((data, indices, indptr), shape=(cell_count, gene_count))

        t0 = datetime.now()
        offset = expected * batch_size
        sub_obs = conn.execute(f"SELECT * FROM obs LIMIT {cell_count} OFFSET {offset}").fetchdf()
        time_obs_total += (datetime.now() - t0).total_seconds()

        t1 = datetime.now()
        sub_var = conn.execute("SELECT * FROM var").fetchdf()
        time_var_total += (datetime.now() - t1).total_seconds()

        sub_obs.index = sub_obs["cell_id"].astype(str)
        sub_var.index = sub_var["gene_id"].astype(str)

        t2 = datetime.now()
        adata = AnnData(X=X, obs=sub_obs, var=sub_var)
        adata.obs_names = sub_obs.index
        adata.var_names = sub_var.index
        time_adata_total += (datetime.now() - t2).total_seconds()

        print(f"\n--- 批次 {expected} 详细信息 ---")
        print(f"shape={X.shape}, nnz={X.nnz}")
        print(f"obs 表读取累计用时: {time_obs_total:.2f} 秒")
        print(f"var 表读取累计用时: {time_var_total:.2f} 秒")
        print(f"生成 AnnData 累计用时: {time_adata_total:.2f} 秒")

        yield 1

        sem_empty.release()  # 告诉 producer：有 1 个 slot 空了
        # print(f"[DEBUG] 主进程 release sem_empty 批次 {expected}")  # 🔹 修改/调试
        expected += 1 # 推进顺序；consumer 永远 先读 expected，再 expected += 1
        # print(f"当前的 expected = { expected }")

        # -----------------------------
        # 最后一个 batch，直接关闭进程线程和共享内存
        # -----------------------------
        if expected == num_batches :  # 最后一个 batch
            for p in processes:
                p.terminate()   # 终止子进程
                p.join() # 阻塞主进程
            for i in range(pool_size):  # 关闭所有的共享内存
                shm_pools['indptr'][i].close()
                shm_pools['indptr'][i].unlink()
                shm_pools['indices'][i].close()
                shm_pools['indices'][i].unlink()
                shm_pools['data'][i].close()
                shm_pools['data'][i].unlink()
            # print("最后一个 batch 已处理，所有 worker 已退出")  # 修改点
            break  # 结束循环

    conn.close()

    # # -----------------------------
    # # 等待子进程结束
    # # -----------------------------
    # for p in processes:
    #     p.join()
    #
    # # -----------------------------
    # # 统一释放共享内存池
    # # -----------------------------
    # for i in range(pool_size):
    #     shm_pools['indptr'][i].close()
    #     shm_pools['indptr'][i].unlink()
    #     shm_pools['indices'][i].close()
    #     shm_pools['indices'][i].unlink()
    #     shm_pools['data'][i].close()
    #     shm_pools['data'][i].unlink()
    #
    # print("所有批次处理完成，worker 已退出")


# ============================================================
# 4️⃣ 测试入口
# ============================================================
if __name__ == "__main__":
    DB_PATH = r"E:\python\scAtlas\Package\python-version-scAtlasAnalysis\test\database\test_204800.sasql"
    BATCH_SIZE = 2048

    start_time = datetime.now()
    count = 0

    for adata in minibatch_scan_order_mproc(DB_PATH, batch_size=BATCH_SIZE,
                                            n_processes=2, n_threads=8,
                                            pool_size=30, drop_last=False):
        count += 1

    end_time = datetime.now()
    minibatch_time = (end_time - start_time).total_seconds()
    print(f"#### 总批次数量 : {count} ")
    print(f"#### 批次大小 : {BATCH_SIZE} ")
    print(f"#### minibatch_time 总时间 : {minibatch_time:.2f} 秒")
    print(f"#### 1个 minibatch 耗时 : { (minibatch_time / count) :.2f} 秒")
    print(f"#### 每秒 minibatch 数量 : { (count / minibatch_time) :.2f} 个")



