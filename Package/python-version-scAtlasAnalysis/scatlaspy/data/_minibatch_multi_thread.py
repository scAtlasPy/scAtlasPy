import duckdb
import numpy as np
import threading
import queue
import time
import scipy.sparse as sp


''' 输出缓存区 ShuffleBuffer ，存入10个batch的cell数据，随机打乱，再输出； 保证多次遍历的随机性 '''
class ShuffleBuffer:
    """
    一个用于流式随机化 (streaming shuffle) 的缓冲区。  
    1. 不需要把整个数据集加载到内存
    2. 只用固定大小 buffer 做随机化
    3. 支持 streaming 数据流
    工作方式：
        输入数据流
        batch1 -> batch2 -> batch3 -> ...

        写入 buffer
        buffer = [cells...]

        当 buffer 凑满 N 个 batch 时
            shuffle 一次

        然后按顺序输出 N 个 batch

        输出完成后
            清空 buffer
            再继续下一轮
    """

    ''' 初始化 缓冲区'''
    def __init__(self, gene_num, batch_size, buffer_batch_num ):

        self.batch_size = batch_size
        self.gene_num = gene_num
        self.buffer_batch_num = buffer_batch_num


        # buffer 最大 cell 数
        self.buffer_cells = buffer_batch_num * batch_size

        # 实际 buffer
        self.X = np.zeros((self.buffer_cells, gene_num), dtype=np.float32)

        # 写入指针
        self.write_ptr = 0

        # 当前输出 batch id
        self.output_batch_id = 0

        # buffer 是否已经 shuffle
        self.shuffled = False

    ''' 写入一个 batch 到 缓冲区'''
    def add_batch(self, X_batch):

        # 如果 buffer 已经满并进入输出阶段，不再写入
        if self.shuffled:
            return

        n = X_batch.shape[0]

        # 修改2：安全保护（防止未来出现越界）
        if self.write_ptr + n > self.buffer_cells:
            raise RuntimeError("ShuffleBuffer overflow")

        # 写入 buffer
        self.X[self.write_ptr:self.write_ptr + n] = X_batch
        self.write_ptr += n

        # 如果 buffer 已满
        if self.write_ptr == self.buffer_cells:

            # 随机打乱
            perm = np.random.permutation(self.buffer_cells)
            self.X[:] = self.X[perm]

            # 进入输出阶段
            self.output_batch_id = 0
            self.shuffled = True

    ''' 输出一个 batch '''
    def sample_batch(self):
        
        if not self.shuffled: # 如果还没凑够 buffer
            return None

        start = self.output_batch_id * self.batch_size
        end = start + self.batch_size

        batch = self.X[start:end]

        self.output_batch_id += 1

        # 如果已经输出完所有 batch
        if self.output_batch_id == self.buffer_batch_num:

            # 🔴 修改3：reset buffer 状态
            self.write_ptr = 0
            self.output_batch_id = 0
            self.shuffled = False

        return batch

    def flush_remaining(self):
        """
        ✅【新增】输出未凑满 buffer 的剩余 batch
        防止数据集 batch 数 < buffer_batch_num 时，一个 batch 都不输出
        """

        if self.write_ptr == 0:
            return []

        n_cells = self.write_ptr

        # 只打乱已经写入的 cell
        perm = np.random.permutation(n_cells)
        X_remain = self.X[:n_cells][perm]

        batches = []

        start = 0
        while start < n_cells:
            end = min(start + self.batch_size, n_cells)
            batches.append(X_remain[start:end].copy())
            start = end

        # reset
        self.write_ptr = 0
        self.output_batch_id = 0
        self.shuffled = False

        return batches



''' 多线程 输出minibatch：  Producer → Queue → Reorder → RingBuffer → Consumer（有序） '''
class MinibatchFetchMultiThreads:

    """ 初始化 """
    def __init__(self, file_path, batch_size=2048, producer_num=10 , X_type = "CSR" , pass_mode = "single-pass" , buffer_batch_num = 5  ):

        self.X_type = X_type # 输出的X表格式 "CSR" "dense"(宽表)
        self.file_path = file_path    # sasql 文件的绝对路径
        self.batch_size = batch_size
        self.producer_num = producer_num # 线程数量
        self.gene_num = self._get_gene_num() # 获取基因数量
        self.zero_scale_transform = self._get_zero_scale_transform()
        # 获取var 表的 zero_scale_transform ，就是每个基因的 ( 0 - g.mean) / g.std
        # gene_id 的索引数组

        # 🔥 修改1：新增输出队列（用于解耦 yield）
        self.out_queue = queue.Queue(maxsize=20)

        self.fetch_size = 1_0000_0000 # fetch_record_batch 流式读取的size
        self.pass_mode = pass_mode    # single-pass 单次遍历 ，multi-pass 多次遍历
        self.buffer_batch_num = buffer_batch_num  # 多次遍历 ，缓冲区的容量，n: 表示batch_size * n

        self.indptr_queue = self._prepare_indptr() # 获取 indptr 的 rb 读取数据流

        self.batch_nnz = self._prepare_batch_nnz_sql() # 获取batch_nnz (每批cell的非零值数量)
        self.batch_idx = 0  # 批次的编号
        self.batch_num = len(self.batch_nnz) # 批次数量

        self.queue = queue.Queue(maxsize=producer_num * 5) #  数据缓存队列 ：Queue（核心）
        self.global_seq = 0 # 顺序ID（关键）
        self.seq_lock = threading.Lock() # 互斥锁

        # Ring Buffer ：切分出batch的环形缓冲池
        self.pool_size = self.fetch_size * 10 # 容量
        self.pool_gene_id = np.empty(self.pool_size, dtype=np.uint16)
        self.pool_data = np.empty(self.pool_size, dtype=np.float32)

        self.read_ptr = 0    # 读指针
        self.write_ptr = 0   # 写指针
        self.used_size = 0   # 当前 Ring Buffer 里的 nnz 数据
        self.total_batches = 0 #计数器 todo 和 batch_idx一样，是否可以删掉


    """ 获取基因( 0 - g.mean) / g.std """
    def _get_zero_scale_transform(self):
        """
        获取每个基因的 zero_scale_transform
        返回：np.ndarray（index == filter_gene_id）
        """
        conn = duckdb.connect(self.file_path)

        arr = conn.execute("""
            SELECT zero_scale_transform
            FROM var
            WHERE filter_gene_id IS NOT NULL
            ORDER BY filter_gene_id
        """).fetchnumpy()["zero_scale_transform"]

        conn.close()
        return arr.astype("float32")

    """ 获取基因数量 """
    def _get_gene_num(self):
        conn = duckdb.connect(self.file_path)
        gene_num = conn.execute(
            "SELECT COUNT(*) FROM var WHERE filter_gene_id IS NOT NULL"
        ).fetchone()[0]
        print("gene_num:", gene_num)
        conn.close()
        return gene_num


    """ 获取 indptr 的 rb 读取数据流 """
    def _prepare_indptr(self):
        conn = duckdb.connect(self.file_path)
        conn.execute("PRAGMA enable_progress_bar=false")


        fetch_record_indptr = conn.execute(
            "SELECT indptr FROM X_CSRO_indptr_filtered"
        ).fetch_record_batch(rows_per_batch=self.batch_size)

        q = queue.Queue()
        for rb in fetch_record_indptr:
            q.put(rb)
        conn.close()
        return q


    """ 获取 batch_nnz : SQL 计算 """
    def _prepare_batch_nnz_sql(self):
        conn = duckdb.connect(self.file_path)
        conn.execute("PRAGMA enable_progress_bar=false")

        query = f"""
        WITH t AS (
            SELECT indptr, ROW_NUMBER() OVER (ORDER BY filter_cell_id) AS rn
            FROM X_CSRO_indptr_filtered
        ),
        picked AS (
            SELECT indptr
            FROM t
            WHERE rn % {self.batch_size} = 0
        )
        SELECT indptr - LAG(indptr, 1, 0) OVER (ORDER BY indptr)
        FROM picked
        """

        rows = conn.execute(query).fetchall()
        conn.close()
        return [int(r[0]) for r in rows]


    """ 获取 batch_nnz : 流式计算 """
    def _prepare_batch_nnz_streaming(self):

        t0 = time.time()
        conn = duckdb.connect(self.file_path)
        conn.execute("PRAGMA enable_progress_bar=false")

        batch_nnz = []  # 每个 batch 的 nnz 数
        prev = 0  # 上一个 batch 的 nnz 累积值
        cell_count = 0  # 已处理的 cell 数

        # 使用 Arrow RecordBatch 流式读取
        rel = (
            conn.execute(
                "SELECT indptr FROM X_CSRO_indptr ORDER BY atlas_cell_id"
            )
            .fetch_record_batch(rows_per_batch=100_000)  # 可调
        )

        for rb in rel:
            # Arrow → numpy（零拷贝视图）
            indptr_chunk = rb.column(0).to_numpy()

            for cur_indptr in indptr_chunk:
                cell_count += 1

                # 每 batch_size 个 cell，形成一个 batch
                if cell_count % self.batch_size == 0:
                    cur = int(cur_indptr)
                    batch_nnz.append(cur - prev)
                    prev = cur

        print(
            f"[Init] streaming 读取 indptr 完成，"
            f"batch 数量 {len(batch_nnz)}, "
            f"耗时 {time.time() - t0:.3f}s"
        )

        return batch_nnz


    """ producer: 多线程 切分data ，写入queue中  """
    def _producer(self, tid):

        conn = duckdb.connect(self.file_path)
        conn.execute("PRAGMA enable_progress_bar=false")
        # ✅ 修改1：producer 总计时开始
        t0 = time.time()
        print(f"[Producer-{tid}] start")

        query = f"SELECT filter_gene_id, data FROM X_CSRO_data_filtered WHERE tid={tid};"
        result = conn.execute(query).fetch_record_batch(rows_per_batch=self.fetch_size)

        # ✅ 修改3：记录 SQL 创建/启动耗时
        t_query = time.time()
        result = conn.execute(query).fetch_record_batch(rows_per_batch=self.fetch_size)
        print(f"[Producer-{tid}] query ready, cost={time.time() - t_query:.2f}s")

        # ✅ 修改4：统计这个 producer 读了多少个 record batch / nnz
        rb_count = 0
        nnz_count = 0

        for rb in result: # fetch_record_batch 流式读取的 结果 rb =（ gene_id ，data ）

            gene_id = rb.column(0).to_numpy().astype(np.uint16)
            data = rb.column(1).to_numpy().astype(np.float32)

            # 关键 🔥 全局顺序ID ，唯一
            with self.seq_lock:
                seq_id = self.global_seq # 全局递增序号，用于标记每条数据的顺序
                self.global_seq += 1

            self.queue.put((seq_id, gene_id, data)) # 将标记有唯一顺序id的 数据块放入 queue 中
            #  [ 0 , gene_id, data ] --- 表示第0块数据
            #  [ 1 , gene_id, data ] --- 表示第1块数据

            # print(f"[Producer-{tid}] seq={seq_id}, nnz={len(gene_id)}")
            # ✅ 修改5：每读 5 个 record batch 打印一次进度
            if rb_count % 5 == 0:
                elapsed = time.time() - t0
                print(
                    f"[Producer-{tid}] rb={rb_count}, "
                    f"nnz={nnz_count:,}, "
                    f"speed={nnz_count / (elapsed + 1e-8):,.0f} nnz/s"
                )

        conn.close()

        # ✅ 修改6：producer 完成总耗时
        print(
            f"[Producer-{tid}] done, "
            f"rb={rb_count}, "
            f"nnz={nnz_count:,}, "
            f"time={time.time() - t0:.2f}s"
        )

        self.queue.put(None)  # 哨兵，结束信号（poison pill），告诉 Consumer：这个 Producer 已经“没数据了”
        # print(f"[Producer-{tid}] 完成")


    ''' Consumer: 单线程，负责从queue中获取数据流，并切分成batch数据 '''
    def _consumer(self):

        reorder_buffer = {}  # 乱序数据 缓存区 : reorder_buffer[seq_id] = (gene_id, data) ； 大小是动态的

        expected_seq = 0          # 下一个想要的的 batch 序号
        global_indptr_offset = 0  # 用于修正 indptr 的累积偏移量

        t_start = time.time()

        # 用于宽表生成
        template = np.empty((self.batch_size, self.gene_num), dtype=np.float32)
        X_dense = np.zeros_like(template, dtype=np.float32)

        # 构建宽表的输出缓冲区
        shuffle_buffer = ShuffleBuffer(
            gene_num = self.gene_num,
            batch_size = self.batch_size,
            buffer_batch_num = self.buffer_batch_num
        )

        # 当前批次号 < 批次数量
        while self.batch_idx < self.batch_num:

            need = self.batch_nnz[self.batch_idx]  # 当前 batch 所需 nnz

            # RingBuffer 中的数据不够， 填充 RingBuffer，直到够一个 batch
            while self.used_size < need:

                item = self.queue.get()  # # 从 Queue 获取下一条数据 ，阻塞等待，自带锁
                if item is None: #  某个 Producer 已完成任务 → 哨兵数据None，不存入 buffer
                    continue

                seq_id, gene_id, data = item # 解析 queue 中的数据 item = (seq_id, gene_id, data)
                reorder_buffer[seq_id] = (gene_id, data)  # 存入 乱序数据 缓存区

                # 按序取数据，当前需要的批次编号 expected_seq， 数据在 乱序数据 缓存区 reorder_buffer 中
                while expected_seq in reorder_buffer:

                    gene_id, data = reorder_buffer.pop(expected_seq) # 取出需要的数据
                    length = len(gene_id)

                    # 写入 Ring Buffer ：切分出batch的环形缓冲池
                    end_space = self.pool_size - self.write_ptr

                    if length <= end_space: # 顺序写
                        self.pool_gene_id[self.write_ptr:self.write_ptr + length] = gene_id
                        self.pool_data[self.write_ptr:self.write_ptr + length] = data

                    else: # 跨界写
                        self.pool_gene_id[self.write_ptr:] = gene_id[:end_space]
                        self.pool_gene_id[:length - end_space] = gene_id[end_space:]

                        self.pool_data[self.write_ptr:] = data[:end_space]
                        self.pool_data[:length - end_space] = data[end_space:]

                    self.write_ptr = (self.write_ptr + length) % self.pool_size
                    self.used_size += length
                    expected_seq += 1

            # RingBuffer 中的数据 够一个 batch
            if self.used_size >= need:

                end_space = self.pool_size - self.read_ptr

                if need <= end_space: # 顺序读
                    vals = self.pool_data[self.read_ptr:self.read_ptr + need]
                    cols = self.pool_gene_id[self.read_ptr:self.read_ptr + need]

                else: # 跨界，两段读取
                    first_len = end_space
                    second_len = need - first_len

                    vals = np.empty(need, dtype=self.pool_data.dtype)
                    cols = np.empty(need, dtype=self.pool_gene_id.dtype)

                    # 尾部
                    vals[:first_len] = self.pool_data[self.read_ptr:]
                    cols[:first_len] = self.pool_gene_id[self.read_ptr:]

                    # 头部
                    vals[first_len:] = self.pool_data[:second_len]
                    cols[first_len:] = self.pool_gene_id[:second_len]

                self.read_ptr = (self.read_ptr + need) % self.pool_size
                self.used_size -= need

                # 构建 indptr
                indptr_now = np.array(self.indptr_queue.get().column(0), dtype=np.int64)
                last_val = indptr_now[-1]
                indptr_now = np.concatenate(([0], indptr_now - global_indptr_offset))
                global_indptr_offset = last_val

                if(self.X_type == "CSR"):
                    # 输出 ： 1.构建 CSR 格式
                    X = sp.csr_matrix((self.batch_size, self.gene_num), dtype=np.float32)
                    X.data = vals
                    X.indices = cols
                    X.indptr = indptr_now
                    # print(" 输出一个 CSR")
                    self.out_queue.put(X)
                    # yield X

                if (self.X_type == "dense"):
                    # 输出 ：2 .构建 宽表

                    # X_dense.fill(0)  # todo 原来
                    # for r in range(self.batch_size):
                    #     start = indptr_now[r]
                    #     end = indptr_now[r + 1]
                    #     X_dense[r, cols[start:end]] = vals[start:end]

                    X_dense[:] = self.zero_scale_transform # todo 修改： 按gene_id填充，self.zero_scale_transform
                    #  zero_scale_transform    将每个基因的 ( 0 - g.mean) / g.std 存入var表的该字段，以便将来调用
                    # X_dense =
                    # [ 填充每个基因的 zero_scale_transform
                    #  [-0.5, 0.2, -1.1, ...],
                    #  [-0.5, 0.2, -1.1, ...],
                    #  ...
                    # ]
                    rows = np.repeat( # [0,0, 1, 2,2,2] 👉 每个非零元素对应的“行号”
                        np.arange(self.batch_size), # [0,1,2,...] 👉 表示每个 cell（行）
                        np.diff(indptr_now)  # [2, 1, 3, ...]  👉 每个 cell 有多少个非零值（nnz）
                    )
                    X_dense[rows, cols] = vals
                    # 将非零值写入对应的 行列
                    # X_dense[0,1] = 10
                    # X_dense[0,3] = 20
                    # X_dense[1,0] = 30
                    # X_dense[2,2] = 40


                    if self.pass_mode == "single-pass": # 单次遍历
                        print("single-pass 输出一个 随机 batch")
                        self.out_queue.put(X_dense.copy())

                    if self.pass_mode == "multi-pass":  # 多次遍历 （加入缓存区，保证多次的随机性）

                        shuffle_buffer.add_batch(X_dense) # 写入 输出缓存区 shuffle buffer
                        X_dense_random = shuffle_buffer.sample_batch() # 从 输出缓存区 随机采样 batch ， 保证多次遍历的随机性
                        if X_dense_random is not None:
                            print("multi-pass 输出一个 随机 batch")
                            self.out_queue.put(X_dense_random.copy())
                            # 你每轮都会复用同一块 buffer
                            # 👉 不 copy 会被覆盖

                elapsed = time.time() - t_start
                print(f"[Consumer] batch {self.batch_idx}, batch/s={self.batch_idx / (elapsed + 1e-8):.2f}")

                self.batch_idx += 1
                self.total_batches += 1

        # todo 修改
        #  ✅【新增】multi-pass 模式：输出 ShuffleBuffer 里没凑满的尾部 batch
        if self.X_type == "dense" and self.pass_mode == "multi-pass":
            remain_batches = shuffle_buffer.flush_remaining()

            for X_remain in remain_batches:
                print("multi-pass 输出尾部随机 batch")
                self.out_queue.put(X_remain.copy())

        print(f"[Done] total_batches={self.total_batches}")

        # 🆕 通知 run() 结束
        self.out_queue.put(None)


    ''' 多线程 运行函数  '''
    def run(self):

        producers = []

        for i in range(self.producer_num):
            t = threading.Thread(target=self._producer, args=(i,))
            t.start()
            producers.append(t)

        consumer = threading.Thread(target=self._consumer)
        consumer.start()

        # 🔥 修改4：从 out_queue 统一 yield
        while True:
            batch = self.out_queue.get()  # 阻塞
            if batch is None:  # 👈 收到哨兵，说明所有 batch 都吐完
                break
            yield batch  # 正常 batch 继续向外 yield

        # print("  while ... 处理完毕 ✅")

        for t in producers:
            t.join()

        consumer.join()

# [Consumer] batch 405, batch/s=362.03
# [Done] total_batches=406

# [Consumer] batch 405, batch/s=410.40
# [Done] total_batches=406