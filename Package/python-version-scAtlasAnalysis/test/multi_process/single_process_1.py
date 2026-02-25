import duckdb
import numpy as np
import threading
from queue import Queue
import time
import scipy.sparse as sp
from anndata import AnnData

# todo scan多线程 + producer 单线程 + consumer 单线程
class CSRBatchFetcherMT:
    """
    多线程 Arrow 解包 + 单线程 CSR batch 切分

    整体设计思想（非常重要）：
    --------------------------------------------------
    1. DuckDB 只做一次顺序 scan（IO 线程）
    2. Arrow RecordBatch → numpy 的解包是 CPU-bound，可并行
    3. CSR batch 的“顺序边界”由 indptr / nnz 决定，必须单线程
    4. producer / consumer 使用 sentinel 计数，保证一定能正常退出
    --------------------------------------------------
    """

    def __init__(
        self,
        db_path,
        batch_size=2048,
        fetch_rows=2048 * 2000,
        n_scan_er=4,
    ):
        # -----------------------------
        # 基本配置参数
        # -----------------------------
        self.db_path = db_path
        self.batch_size = batch_size # 一个 CSR batch 中包含的 cell 数
        self.fetch_rows = fetch_rows # DuckDB 每次 scan 返回多少 nnz 行（Arrow RecordBatch 粒度）
        self.n_scan_er = n_scan_er

        # -----------------------------
        # 线程间队列
        # -----------------------------
        # scan_thread → producer
        # 存放 Arrow RecordBatch
        self.rb_queue = Queue(maxsize=8)

        # producer → consumer
        # 存放 (gene_id, data) 的 numpy tuple
        self.data_queue = Queue(maxsize=8)

        # -----------------------------
        # CSR pool 内存池
        # -----------------------------
        # pool 的含义 - 从 DuckDB 读出，已经解包成 numpy，但尚未组成完整 batch 的 nnz
        # consumer 是唯一读写 pool 的线程： 消费线程
        self.pool_indices = np.empty(0, dtype=np.int32) # ✅ 创建 一个“空的、但类型已经确定”的数组
        # 创建一个 长度为 0 的 NumPy 数组，元素类型是 int32，用来存 CSR 的 gene_id。
        self.pool_data = np.empty(0, dtype=np.float32)

        # -----------------------------
        # 从 X_CSR_indptr 中读取完整 indptr，
        # 并按照 batch_size 切分出每个 batch 对应的 nnz 数量。
        # -----------------------------
        self.batch_nnz = self._prepare_batch_nnz_sql()
        # 一次加载   流式加载   SQL处理  819200细胞
        #  0.193s   0.080    0.022

        self.batch_idx = 0 # 当前 batch index（consumer 独占）
        self.total_batches = 0 # 实际生成的 batch 数量

    # --------------------------------------------------
    # 预计算每个 batch 的 nnz（严格顺序）
    # --------------------------------------------------

    # 直接用SQL语句,
    def _prepare_batch_nnz_sql(self):
        """
        方案 B：
        在 DuckDB 中直接计算每个 batch 的 nnz

        思路：
        - 每 batch_size 个 cell 取一个 indptr
        - 用 lag 计算差分
        - Python 只接收 batch 级别结果

        特点：
        - Python 内存占用极小
        - 极快（DuckDB 向量化执行）
        - 非常适合超大规模数据
        """

        t0 = time.time()
        conn = duckdb.connect(self.db_path, read_only=True)  # ✅ 修改：主线程用自己的 conn

        query = f"""
        WITH batch_end AS (
            SELECT
                id,
                indptr,
                ROW_NUMBER() OVER (ORDER BY id) AS rn
            FROM X_CSR_indptr
        ),
        picked AS (
            SELECT
                indptr
            FROM batch_end
            WHERE rn % {self.batch_size} = 0
        )
        SELECT
            indptr - LAG(indptr, 1, 0) OVER (ORDER BY indptr) AS batch_nnz
        FROM picked
        ORDER BY indptr
        """

        rows = conn.execute(query).fetchall()

        # batch_nnz 量级很小（≈ total_cells / batch_size）
        batch_nnz = [int(r[0]) for r in rows]

        print(
            f"[Init] SQL 计算 batch_nnz 完成，"
            f"batch 数量 {len(batch_nnz)}, "
            f"耗时 {time.time() - t0:.3f}s"
        )

        conn.close()  # ✅ 执行完立即关闭 conn

        return batch_nnz

    # --------------------------------------------------
    # todo： 生产线程 类型1 ：DuckDB 顺序 scan（单线程）
    # --------------------------------------------------
    # --------------------------------------------------
    # scan 线程：顺序扫描 + 分配 seq_id
    # --------------------------------------------------
    def _scan_thread(self):
        """
        负责：
        - 顺序扫描 X_CSR_data
        - 以 Arrow RecordBatch 形式推送到 rb_queue
        设计约束：
        - 只能有一个 scan 线程
        - DuckDB scan 本身是顺序 IO
        """

        t0 = time.time()

        conn = duckdb.connect(self.db_path, read_only=True)  # ✅ 修改：scan 线程独立 conn
        # fetch_record_batch 会返回一个可迭代的 Arrow RecordBatch 流
        rel = (
            conn.execute(
                "SELECT indices, data FROM X_CSR_data"
            )
            .fetch_record_batch(rows_per_batch=self.fetch_rows)
        )
        # 正确顺序 0 1 2 3 4 5 6 7 8 9
        # [0 1 2]     [3 4 5]    [6 7 8] 真实的 coo
        #   0            1          2
        # [3 4 5]  [0 1 2]   [6 7 8] 没有 seq_id，容易非顺序 data_queue
        #  1         0          2
        # 顺序推送 RecordBatch
        for seq_id, rb in enumerate(rel):
            self.rb_queue.put((seq_id, rb))  # ★ 带 seq_id

        t_scan_t = time.time()
        print(f"！！！ scan_t函数耗时： {t_scan_t - t0} s")

        # scan 完成后，向每个 producer 发送一个 结束信号 sentinel(None)
        for _ in range(self.n_producers):
            self.rb_queue.put(None)

        conn.close()  # ✅ 修改：scan 完成立即关闭 conn


    # --------------------------------------------------
    # todo 生产线程 类型2： Producer：Arrow → numpy（并行） → data_queue(gene_id, data, cell_id)中的数据不是按顺序的；
    # --------------------------------------------------
    def _producer(self, pid):
        """
        producer 的职责：
        - 从 rb_queue 中取 Arrow RecordBatch
        - 将每一列转换为 numpy array，再推送到 data_queue
        - 这是一个 CPU-bound 阶段，可以多线程并行，data_queue中的数据不是按顺序的；
        """
        while True:

            item = self.rb_queue.get() # 从 rb_queue 中取

            if item is None: # 结束处理：收到 None，说明 scan 已结束
                self.data_queue.put(None)  # 向 consumer 转发一个 结束信号None
                break

            seq_id, rb = item  # ★ 拿到 seq_id 和 Arrow RecordBatch

            # todo Arrow RecordBatch → numpy # 实际产生读取内容加载
            gene_id = rb.column(0).to_numpy()
            data = rb.column(1).to_numpy()

            # ★ 把 seq_id 一起传给 consumer
            self.data_queue.put((seq_id, gene_id, data)) # 将 numpy 推送到 data_queue
            # data     [0 1 2]     [3 4 5]    [6 7 8]
            # seq_id     0            1          2
        print(f"[Producer-{pid}] 完成")

    # --------------------------------------------------
    # todo：消费线程 ：Consumer：单线程顺序切 batch + 构建 COO
    # --------------------------------------------------
    def _consumer(self):
        """
           ★ 顺序输出（核心）：
           - buffer: 暂存乱序到达的数据块
           - next_seq_id: consumer 当前期望的 RecordBatch 顺序
           - 只有 seq_id == next_seq_id 才能进入 pool
        """

        conn = duckdb.connect(self.db_path, read_only=True)  # ✅ 修改：consumer 独立 conn

        # todo ===== 加上 indptr 信息 =======

        fetch_record_indptr = (
            conn.execute(
                "SELECT indptr FROM X_CSR_indptr"
            )
            .fetch_record_batch(rows_per_batch=self.batch_size)
        )
        record_indptr_queue = Queue()
        for record_indptr in fetch_record_indptr:
            record_indptr_queue.put(record_indptr)

        global_indptr_offset = 0  # 偏移量初始值

        # todo ===== 加上 indptr 信息 =======

        finished = 0  # 已完成的 producer 数量
        buffer = {}  # seq_id -> (gene_id, data)
        next_seq_id = 0  # ★ 顺序游标 ，按这个顺序进行输出

        # ================= 新增：性能统计 =================
        start_time = time.time()  # ⬅️ 新增
        last_report_batch = 0  # ⬅️ 新增
        # =================================================

        while True:
            item = self.data_queue.get() # 从data_queue中，获取（seq_id, gene_id, data = item ）

            # 处理 producer 结束信号
            if item is None:
                finished += 1
                if finished == self.n_producers:
                    break
                continue

            seq_id, gene_id, data = item
            buffer[seq_id] = (gene_id, data) # buffer 是一个 字典，
            # buffer[0] = (0 1 2)
            # key = seq_id ( 流式加载的批次id) ， value = (gene_id, data)

            # ★ 只按 seq_id 顺序消费
            while next_seq_id in buffer: # 只处理 已经到达的且符合顺序的 batch
                # next_seq_id 是 Consumer 期望处理的下一个 batch 序号，初始为 0。
                indices, data = buffer.pop(next_seq_id) # 按序输出
                # (0 1 2) = buffer.pop(0)

                # -----------------------------
                # 将 gene_id, data 拼接到 内存 pool
                # -----------------------------
                self.pool_indices = np.concatenate([self.pool_indices, gene_id])
                self.pool_data = np.concatenate([self.pool_data, data])

                # -----------------------------
                # 尝试从 pool 中切 batch ， 滑动窗口
                # -----------------------------
                while self.batch_idx < len(self.batch_nnz):
                    # self.batch_idx 当前 batch index（consumer 独占）；consumer 是唯一知道当前 batch_idx 的线程；
                    # self.batch_nnz 是一个数组，是有序的 ；
                    # batch_nnz[i] = 第 i 个 batch 应该消耗多少 nnz
                    # batch_nnz[0] = 3
                    need = self.batch_nnz[self.batch_idx] #  当前 batch 需要的 nnz 数量
                    # need = 3

                    # pool 中 nnz 不够，等下一个数据块
                    if self.pool_indices.size < need:
                        break

                    # --------- 构建一个 batch 的 csr ----------

                    # 从内存pool中获取 need 个 （当前 batch 需要的 nnz 数量）数据
                    cols = self.pool_indices[:need] # gene_id
                    vals = self.pool_data[:need] # data

                    n_cells = self.batch_size
                    n_genes = int(cols.max()) + 1 # 编号是从0开始，所以要加1

                    # 引入 indptr ， 直接生成 csr 格式 ====
                    indptr_now = record_indptr_queue.get().column(0).to_numpy() # 取出 indptr（Arrow → numpy）
                    last_val = indptr_now[-1] # 1️⃣ 先保存“原始最后一个值”，用于更新 offset
                    indptr_now = indptr_now - global_indptr_offset # 2️⃣ 每个值减去偏移量
                    indptr_now = np.concatenate(([0], indptr_now)) # 3️⃣ 在开头补一个 0
                    global_indptr_offset = last_val # 4️⃣ 更新偏移量（⚠️ 用减之前的 last_val）

                    # (data, indices, indptr)
                    X = sp.csr_matrix(
                        (vals, cols, indptr_now),
                        shape=(n_cells, n_genes),
                        dtype=np.float32,
                    )

                    print(f"[Consumer] batch 编号： {self.batch_idx}, nnz={need}")
                    # print(f"================== 本批次构建完成！===================\n")
                    # ================= 新增：每 5 batch 输出一次性能 =================
                    if (self.batch_idx + 1) % 5 == 0:
                        elapsed = time.time() - start_time
                        batches_done = self.batch_idx + 1
                        speed = batches_done / elapsed if elapsed > 0 else 0.0
                        print(
                            f"[Consumer][Perf] "
                            f"batch={batches_done}, "
                            f"elapsed={elapsed:.2f}s, "
                            f"batch/s={speed:.2f}"
                        )
                    # ================================================================
                    # -----------------------------
                    # 移动 pool（丢弃已消费的 nnz）
                    # -----------------------------
                    self.pool_indices = self.pool_indices[need:]
                    self.pool_data = self.pool_data[need:]

                    self.batch_idx += 1
                    self.total_batches += 1

                next_seq_id += 1

        conn.close()  # ✅ 修改：consumer 完成后关闭 conn
        print("[Consumer] 完成")

    # --------------------------------------------------
    # 运行入口
    # --------------------------------------------------
    def run(self):
        """
        启动并管理所有线程，并在最后输出性能统计
        """
        t0 = time.time()

        # todo 1. 生产线程 类型1： _scan_thread ; 多线程;
        #  顺序扫描 X_CSR_data, 惰性加载 ; 以 Arrow RecordBatch 形式推送到 rb_queue
        scan_er = []
        for i in range(self.n_scan_er):
            t = threading.Thread(target=self._scan_thread, args=(i,))
            t.start()
            scan_er.append(t)

        # todo 2. 生产线程 类型2： _producer; 单线程；
        #   从 rb_queue 中取 Arrow RecordBatch -> numpy array -> 推送到 data_queue
        producer_t = threading.Thread(target=self._producer)
        producer_t.start()

        # todo 3. 消费线程：启动 consumer，负责 单线程顺序切 batch + 构建 CSR
        consumer_t = threading.Thread(target=self._consumer)
        consumer_t.start()

        # todo 4. 等待所有线程结束
        for t in scan_er:
            t.join()
        producer_t.join()
        consumer_t.join()

        total_time = time.time() - t0

        # 5. 统计信息
        print("\n[Done] 所有 batch 处理完成")
        print(f"总耗时: {total_time:.3f} 秒")
        print(f"总 batch 数量: {self.total_batches}")
        print(f"平均 batch/s: {self.total_batches / total_time:.2f}")


# ======================================================
# main
# ======================================================
if __name__ == "__main__":

    # for i in range(20):
    #     print(f"do {i}", flush=True)  # print 默认是有缓冲区的，缓冲区满了才会有一次真实的向stdout输出内容
    #     # flush=True的意思是 不需要等缓冲区满，每次都立即输出
    #
    # import os
    #
    # path=[r"/mnt/data/test_409600.sasql", r"E:\data\test_409600.sasql"]
    # for e in path:
    #     if os.path.exists(e):
    #         print(e)
    #         fetcher = CSRBatchFetcherMT(  # 初始化， 完成indptr数组的获取
    #             db_path=e,
    #             batch_size=2048,
    #             fetch_rows=2048 * 2000,
    #             n_producers=1,
    #         )
    #         fetcher.run()

    # print("running")
    # fetcher = CSRBatchFetcherMT( # 初始化， 完成indptr数组的获取
    #     # db_path=r"E:\python\scAtlas\Package\python-version-scAtlasAnalysis\test\database\test_819200.sasql",
    #     # db_path=r"/home/hanxu/test_ddb/experimental_script_xuhan/test_409600.sasql",
    #     db_path=r"/mnt/data/test_409600.sasql",
    #     db_path=e,
    #     batch_size=2048,
    #     fetch_rows=2048 * 2000,
    #     n_producers=1,
    # )
    # fetcher.run()
    # print(flush=True)

    fetcher = CSRBatchFetcherMT(  # 初始化， 完成indptr数组的获取
        db_path=r"E:\python\scAtlas\Package\python-version-scAtlasAnalysis\test\database\test_204800.sasql",
        batch_size=2048,
        fetch_rows=2048 * 2000,
        n_producers=1,
    )
    fetcher.run()


    # todo
    #   从 fetch_arrow_table().to_pandas() -->  fetch_record_batch
    #   构建COO格式，转CSR， 35 b/s
    #   构建COO格式，不转CSR, 55 b/s
    #   单线程和多线程 一样快;
    #   优化 1 ：
    #   不获取 X_CSR_data 表的 cell_index， 只提取（indices 和 data ）
    #   加上 X_CSR_data 表的 indptr ,
    #   构建CSR（ indptr + indices + data ）  80 b/s 左右

    # 多线程应该在fetch_record_batch + for， 不是在producer；
    # win - xh      40 b/s
    # win -yyz      80 b/s
    # linux -amax   2  b/s只关注  fetch_record_batch + for ：
    #    float 64：单线程6 b/s; 多线程 35b/s；
    # linux -02     40 b/s

    # 现在是 同一个 sql 文件，多线程无法避免 非顺序
    # 多个sql文件， 对应多线程， 可解决 多线程的顺序问题
    # 存入数据的时候，导入成 4 个 sql 文件， （ 也是多线程加载 ）
    #  0.sql   线程t0   0 1 2    12  13  14
    #  1.sql   t1   3 4 5   ....
    #  2.sql   t2   6 7 8   ...
    #  3.sql   t3  9 10 11 ...
    # 问题1： 多个数据库文件不好处理，（把4个文件在一个数据库进行操作）
    # todo  建立连接 （要同时和4个数据库文件进行连接） ？
    # 建立 4 个 fetch_record_batch + for
        #conn
        # 已验证： 报错 。 4 个线程里共享这个conn，每个线程里都自行从这个conn产生自己的rel，然后各自对自己的rel迭代
        # 待验证： 一个conn，一个共享的rel，不同的线程对这个rel进行next
        # rel = (
        #     conn.execute(
        #         "SELECT indices, data FROM X_CSR_data"
        #     )
        #     .fetch_record_batch(rows_per_batch=2048)
        # )
        #
        # r_id = 0  # 线程0
        # for r_id, rb in enumerate(rel):
        #     self.rb_queue.put((r_id, rb))  # ★ 带 seq_id
        #
        # r_id = 1  # 线程1
        # r_id = 2  # 线程2
        # r_id = 3  # 线程3

        # sasql1  #t1 一开始读数据，就自动让出CPU了
        # sasql2  #t2  获得运行的机会，开始读数据，刚开始读数据就直接让出CPU
        # sasql3  #t3
        # sasql4  #t4 CPU密集的任务，都不会主动让出CPU，都是等CPU时间片用完，抢占式的被动释放，同时释放python的GIL

        # t1 t2 t3 t4 每个线程里的 next(rel)是阻塞的函数，阻塞的意思是，当我执行next(rel)时，我向操作系统请求从磁盘读下一个数据
        # 读数据本身数不需要CPU参与的，“从磁盘到内存，有专门的控制器”（是实际瓶颈，但这个速度极快），CPU只负责发送一个任务，这个线程就会自己阻塞自己。
        # 阻塞就直接让出CPU时间片了，即使我的这个时间片还没用 wait(), 知道控制器读取数据完成，这个线程才会被唤醒，
        #唤醒之后，他就可以参与竞争下一次的时间片了