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
    def __init__(self, file_path,
                 batch_size=2048,
                 producer_num=10 ,
                 X_type = "CSR" ,
                 pass_mode = "multi-pass" ,
                 buffer_batch_num = 5  ,
                 max_batches=None, # 新增：本轮最多输出多少个 batch
                 ):

        self.X_type = X_type # 输出的X表格式 "CSR" "dense"(宽表)
        self.file_path = file_path    # sasql 文件的绝对路径
        self.batch_size = batch_size
        self.producer_num = producer_num # 线程数量
        self.gene_num = self._get_gene_num() # 获取基因数量
        self.zero_scale_transform = self._get_zero_scale_transform()
        # 获取var 表的 zero_scale_transform ，就是每个基因的 ( 0 - g.mean) / g.std
        # gene_id 的索引数组

        # ✅ 新增：本轮最多输出多少个 batch
        self.max_batches = max_batches

        # ✅ 新增：提前停止信号
        self.stop_event = threading.Event()

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

        # print(
        #     f"[Init] streaming 读取 indptr 完成，"
        #     f"batch 数量 {len(batch_nnz)}, "
        #     f"耗时 {time.time() - t0:.3f}s"
        # )

        return batch_nnz


    """ producer: 多线程 切分data ，写入queue中  """
    def _producer(self, tid):

        conn = duckdb.connect(self.file_path)
        conn.execute("PRAGMA enable_progress_bar=false")

        t0 = time.time()
        # print(f"[Producer-{tid}] start")

        query = f"""
            SELECT filter_gene_id, data
            FROM X_CSRO_data_filtered
            WHERE tid={tid};
        """

        t_query = time.time()
        result = conn.execute(query).fetch_record_batch(
            rows_per_batch=self.fetch_size
        )
        # print(f"[Producer-{tid}] query ready, cost={time.time() - t_query:.2f}s")

        rb_count = 0
        nnz_count = 0

        try:
            for rb in result:

                # ✅ 新增：提前停止
                if self.stop_event.is_set():
                    break

                gene_id = rb.column(0).to_numpy().astype(np.uint16)
                data = rb.column(1).to_numpy().astype(np.float32)

                with self.seq_lock:
                    seq_id = self.global_seq
                    self.global_seq += 1

                # ✅ 修改：防止 out/queue 满时死锁
                while not self.stop_event.is_set():
                    try:
                        self.queue.put((seq_id, gene_id, data), timeout=0.5)
                        break
                    except queue.Full:
                        continue

                rb_count += 1
                nnz_count += len(gene_id)

                if rb_count % 5 == 0:
                    elapsed = time.time() - t0
                    # print(
                    #     f"[Producer-{tid}] rb={rb_count}, "
                    #     f"nnz={nnz_count:,}, "
                    #     f"speed={nnz_count / (elapsed + 1e-8):,.0f} nnz/s"
                    # )

        finally:
            conn.close()

            # print(
            #     f"[Producer-{tid}] done, "
            #     f"rb={rb_count}, "
            #     f"nnz={nnz_count:,}, "
            #     f"time={time.time() - t0:.2f}s"
            # )

            # ✅ 通知 consumer：这个 producer 完成
            try:
                self.queue.put(None, timeout=0.5)
            except queue.Full:
                pass


    ''' Consumer: 单线程，负责从queue中获取数据流，并切分成batch数据 '''
    def _consumer(self):

        reorder_buffer = {}  # 乱序数据 缓存区 : reorder_buffer[seq_id] = (gene_id, data) ； 大小是动态的

        expected_seq = 0  # 下一个想要的的 batch 序号
        global_indptr_offset = 0  # 用于修正 indptr 的累积偏移量

        t_start = time.time()

        # ✅【新增】速度统计用, 瞬时速度
        # -----------------------------------------------------
        # last_log_time:
        #   上一次打印日志的时间，用来计算“瞬时速度”
        #
        # last_batch_idx:
        #   上一次打印时已经处理的 batch 数
        #
        # last_output_batches:
        #   上一次打印时已经输出的 batch 数
        #
        # log_every:
        #   每多少个 batch 打印一次日志
        #   不建议每个 batch 都打印，否则控制台 IO 会拖慢速度
        # =====================================================
        last_log_time = t_start
        last_batch_idx = 0
        last_output_batches = 0
        log_every = 20
        # =====================================================
        # ✅【新增1】支持 max_batches / stop_event
        # -----------------------------------------------------
        # max_batches:
        #   本次 fetcher 最多向外输出多少个 batch
        #
        # stop_event:
        #   通知 producer 可以提前停止，避免 consumer 已经够了，
        #   producer 还在继续读数据库。
        #
        # 注意：
        #   为了兼容旧代码，这里用 getattr。
        #   但更推荐你在 __init__ 里显式加：
        #
        #   self.max_batches = max_batches
        #   self.stop_event = threading.Event()
        # =====================================================
        max_batches = getattr(self, "max_batches", None)
        stop_event = getattr(self, "stop_event", None)

        def _set_stop_event():
            """✅【新增】安全设置停止信号"""
            if stop_event is not None:
                stop_event.set()

        def _output_limit_reached():
            """
            ✅【新增】是否已经输出够 max_batches。

            self.total_batches 在这个版本里表示：
                已经真正 put 到 out_queue、外部可以 yield 的 batch 数。
            """
            return max_batches is not None and self.total_batches >= max_batches

        # ✅【新增】multi-pass dense 特殊计数：
        # prepared_batches 表示已经读入并加入 ShuffleBuffer 的 batch 数。
        #
        # 为什么需要它？
        #   multi-pass 下 batch 先进入 ShuffleBuffer，
        #   不一定马上输出。
        #
        #   如果 max_batches < buffer_batch_num，
        #   只看 self.total_batches 会一直是 0，
        #   consumer 就会继续读很多无用 batch。
        #
        # 所以：
        #   multi-pass dense 下，prepared_batches 达到 max_batches 后，
        #   就停止继续读取，然后 flush_remaining() 输出尾部。
        prepared_batches = 0

        def _read_limit_reached():
            """
            ✅【新增】是否应该停止继续从 RingBuffer / Queue 读新 batch。

            - dense + multi-pass:
                看 prepared_batches，因为 batch 会先进入 ShuffleBuffer。
            - 其他模式:
                看 self.total_batches，因为读取后会立即输出。
            """
            if max_batches is None:
                return False

            if self.X_type == "dense" and self.pass_mode == "multi-pass":
                return prepared_batches >= max_batches

            return self.total_batches >= max_batches

        def _safe_put_output(X_batch, msg: str | None = None):
            """
            ✅【新增】统一输出 batch，并计数。

            返回：
                True  -> 成功输出
                False -> 已达到 max_batches，没有输出
            """
            if _output_limit_reached():
                _set_stop_event()
                return False

            if msg is not None:
                print(msg)

            self.out_queue.put(X_batch)
            self.total_batches += 1

            if _output_limit_reached():
                print(f"[Consumer] reach max_batches={max_batches}, stop")
                _set_stop_event()

            return True

        # 用于宽表生成
        template = np.empty((self.batch_size, self.gene_num), dtype=np.float32)
        X_dense = np.zeros_like(template, dtype=np.float32)

        # 构建宽表的输出缓冲区
        shuffle_buffer = ShuffleBuffer(
            gene_num=self.gene_num,
            batch_size=self.batch_size,
            buffer_batch_num=self.buffer_batch_num
        )

        # 当前批次号 < 批次数量
        while self.batch_idx < self.batch_num:

            # =====================================================
            # ✅【新增2】如果已经准备/输出够 max_batches，提前结束本轮
            # =====================================================
            if _read_limit_reached():
                print(
                    f"[Consumer] read limit reached, "
                    f"batch_idx={self.batch_idx}, "
                    f"prepared_batches={prepared_batches}, "
                    f"output_batches={self.total_batches}"
                )
                _set_stop_event()
                break

            need = self.batch_nnz[self.batch_idx]  # 当前 batch 所需 nnz

            # RingBuffer 中的数据不够， 填充 RingBuffer，直到够一个 batch
            while self.used_size < need:

                # =================================================
                # ✅【新增3】如果已经不需要继续读，跳出，避免死等 queue
                # =================================================
                if _read_limit_reached():
                    _set_stop_event()
                    break

                # ✅【修改1】加 timeout，避免 producer 提前停止后 consumer 永久阻塞
                try:
                    item = self.queue.get(timeout=0.5)  # 从 Queue 获取下一条数据，阻塞等待，自带锁
                except queue.Empty:
                    if stop_event is not None and stop_event.is_set():
                        break
                    continue

                if item is None:  # 某个 Producer 已完成任务 → 哨兵数据None，不存入 buffer
                    continue

                seq_id, gene_id, data = item  # 解析 queue 中的数据 item = (seq_id, gene_id, data)
                reorder_buffer[seq_id] = (gene_id, data)  # 存入 乱序数据 缓存区

                # 按序取数据，当前需要的批次编号 expected_seq， 数据在 乱序数据 缓存区 reorder_buffer 中
                while expected_seq in reorder_buffer:

                    gene_id, data = reorder_buffer.pop(expected_seq)  # 取出需要的数据
                    length = len(gene_id)

                    # 写入 Ring Buffer ：切分出batch的环形缓冲池
                    end_space = self.pool_size - self.write_ptr

                    if length <= end_space:  # 顺序写
                        self.pool_gene_id[self.write_ptr:self.write_ptr + length] = gene_id
                        self.pool_data[self.write_ptr:self.write_ptr + length] = data

                    else:  # 跨界写
                        self.pool_gene_id[self.write_ptr:] = gene_id[:end_space]
                        self.pool_gene_id[:length - end_space] = gene_id[end_space:]

                        self.pool_data[self.write_ptr:] = data[:end_space]
                        self.pool_data[:length - end_space] = data[end_space:]

                    self.write_ptr = (self.write_ptr + length) % self.pool_size
                    self.used_size += length
                    expected_seq += 1

            # =====================================================
            # ✅【新增4】如果因为 max_batches / stop_event 跳出，
            # 且 RingBuffer 还不够当前 batch，就结束主循环
            # =====================================================
            if self.used_size < need:
                break

            # RingBuffer 中的数据 够一个 batch
            if self.used_size >= need:

                end_space = self.pool_size - self.read_ptr

                if need <= end_space:  # 顺序读
                    vals = self.pool_data[self.read_ptr:self.read_ptr + need]
                    cols = self.pool_gene_id[self.read_ptr:self.read_ptr + need]

                else:  # 跨界，两段读取
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

                if self.X_type == "CSR":
                    # 输出 ： 1.构建 CSR 格式
                    X = sp.csr_matrix((self.batch_size, self.gene_num), dtype=np.float32)

                    # ✅【修改2】建议 copy，避免 RingBuffer 后续覆盖导致外部 batch 数据被污染
                    X.data = vals.copy()
                    X.indices = cols.copy()
                    X.indptr = indptr_now

                    # print(" 输出一个 CSR")
                    _safe_put_output(X)
                    # yield X

                if self.X_type == "dense":
                    # 输出 ：2 .构建 宽表

                    # X_dense.fill(0)  # todo 原来
                    # for r in range(self.batch_size):
                    #     start = indptr_now[r]
                    #     end = indptr_now[r + 1]
                    #     X_dense[r, cols[start:end]] = vals[start:end]

                    X_dense[:] = self.zero_scale_transform  # todo 修改： 按gene_id填充，self.zero_scale_transform
                    #  zero_scale_transform    将每个基因的 ( 0 - g.mean) / g.std 存入var表的该字段，以便将来调用
                    # X_dense =
                    # [ 填充每个基因的 zero_scale_transform
                    #  [-0.5, 0.2, -1.1, ...],
                    #  [-0.5, 0.2, -1.1, ...],
                    #  ...
                    # ]
                    rows = np.repeat(  # [0,0, 1, 2,2,2] 👉 每个非零元素对应的“行号”
                        np.arange(self.batch_size),  # [0,1,2,...] 👉 表示每个 cell（行）
                        np.diff(indptr_now)  # [2, 1, 3, ...]  👉 每个 cell 有多少个非零值（nnz）
                    )
                    X_dense[rows, cols] = vals
                    # 将非零值写入对应的 行列
                    # X_dense[0,1] = 10
                    # X_dense[0,3] = 20
                    # X_dense[1,0] = 30
                    # X_dense[2,2] = 40

                    if self.pass_mode == "single-pass":  # 单次遍历
                        _safe_put_output(
                            X_dense.copy(),
                            msg="single-pass 输出一个 随机 batch"
                        )

                    if self.pass_mode == "multi-pass":  # 多次遍历 （加入缓存区，保证多次的随机性）

                        # =================================================
                        # ✅【修改3】multi-pass 下，先统计 prepared_batches
                        # -------------------------------------------------
                        # 这个 batch 已经进入 ShuffleBuffer，
                        # 即使暂时没输出，后面 flush_remaining 也会输出。
                        # =================================================
                        shuffle_buffer.add_batch(X_dense)  # 写入 输出缓存区 shuffle buffer
                        prepared_batches += 1

                        # =================================================
                        # ✅【修改4】一旦 ShuffleBuffer 满了，不再像原来那样
                        # 每读一个新 batch 才吐一个旧 batch；
                        # 而是立刻把当前 shuffle 后的 buffer 全部吐出去。
                        #
                        # 好处：
                        #   max_batches 比 buffer_batch_num 小时，不会额外读很多 batch。
                        # =================================================
                        while True:
                            X_dense_random = shuffle_buffer.sample_batch()  # 从 输出缓存区 随机采样 batch ， 保证多次遍历的随机性

                            if X_dense_random is None:
                                break

                            ok = _safe_put_output(
                                X_dense_random.copy(),
                                msg="multi-pass 输出一个 随机 batch"
                            )
                            # 你每轮都会复用同一块 buffer
                            # 👉 不 copy 会被覆盖

                            if not ok or _output_limit_reached():
                                break

                        # ✅【新增5】如果已经准备够 max_batches，
                        # 后面不再继续读新 batch，交给尾部 flush 输出剩余。
                        if _read_limit_reached():
                            _set_stop_event()

                # elapsed = time.time() - t_start
                # print(
                #     f"[Consumer] batch {self.batch_idx}, "
                #     f"batch/s={self.batch_idx / (elapsed + 1e-8):.2f}, "
                #     f"output_batches={self.total_batches}"
                # )
                # =====================================================
                # ✅【修改】更合理的中文速度统计
                # -----------------------------------------------------
                # processed_batches:
                #   已经从 RingBuffer 构建出的原始 batch 数
                #
                # output_batches:
                #   已经真正 put 到 out_queue、外部可以 yield 的 batch 数
                #
                # 瞬时处理速度：
                #   距离上次打印期间，consumer 构建 batch 的速度
                #
                # 瞬时输出速度：
                #   距离上次打印期间，真正输出给外部的速度
                #
                # 平均处理速度：
                #   从 consumer 开始到现在的平均构建速度
                #
                # 平均输出速度：
                #   从 consumer 开始到现在的平均输出速度
                # =====================================================
                processed_batches = self.batch_idx + 1

                if processed_batches % log_every == 0 or processed_batches == 1:
                    now = time.time()
                    elapsed = now - t_start
                    dt = now - last_log_time

                    output_batches = self.total_batches

                    avg_processed_bps = processed_batches / (elapsed + 1e-8)
                    avg_output_bps = output_batches / (elapsed + 1e-8)

                    inst_processed_bps = (
                            (processed_batches - last_batch_idx) / (dt + 1e-8)
                    )

                    inst_output_bps = (
                            (output_batches - last_output_batches) / (dt + 1e-8)
                    )

                    print(
                        f"[Consumer] 处理 {processed_batches}/{self.batch_num} | "
                        f"输出 {output_batches} | "
                        f"瞬时处理 {inst_processed_bps:.2f} batch/s | "
                        f"瞬时输出 {inst_output_bps:.2f} batch/s | "
                        f"平均处理 {avg_processed_bps:.2f} batch/s | "
                        f"平均输出 {avg_output_bps:.2f} batch/s"
                    )

                    last_log_time = now
                    last_batch_idx = processed_batches
                    last_output_batches = output_batches

                self.batch_idx += 1

                # =====================================================
                # ✅【修改5】这里不再 self.total_batches += 1
                # -----------------------------------------------------
                # 原来 total_batches 表示“处理过多少个原始 batch”。
                # 现在为了支持 max_batches，total_batches 改成：
                #
                #   实际输出给外部 yield 的 batch 数
                #
                # 所以 total_batches 统一在 _safe_put_output() 里增加。
                # =====================================================

        # todo 修改
        # ✅【新增】multi-pass 模式：输出 ShuffleBuffer 里没凑满的尾部 batch
        if self.X_type == "dense" and self.pass_mode == "multi-pass":
            remain_batches = shuffle_buffer.flush_remaining()

            for X_remain in remain_batches:

                # ✅【新增6】防止尾部输出超过 max_batches
                if _output_limit_reached():
                    _set_stop_event()
                    break

                _safe_put_output(
                    X_remain.copy(),
                    msg="multi-pass 输出尾部随机 batch"
                )

        print(
            f"[Done] processed_batches={self.batch_idx}, "
            f"output_batches={self.total_batches}"
        )

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