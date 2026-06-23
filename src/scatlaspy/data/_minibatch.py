import duckdb
import numpy as np
import threading
import queue
import time
from os import PathLike, fspath
import scipy.sparse as sp
import logging

logger = logging.getLogger("Atlas")
logger.addHandler(logging.NullHandler())

''' 输出缓存区 ShuffleBuffer ，存入5个batch的cell数据，随机打乱，再输出； 保证多次遍历的随机性 '''
class ShuffleBuffer:

    """dense minibatch 随机缓冲区。

    该类在 ``multi-pass`` 小批量读取模式中缓存若干 dense minibatch，达到指定容量后
    对细胞顺序进行随机打乱，再按 batch 输出。它用于减少多轮训练时的顺序偏差，
    主要服务于 PCA、K-means 等流式模型。

    Parameters
    ----------
    gene_num
        dense minibatch 中的基因数量，也就是输出矩阵的列数。
    batch_size
        每个 minibatch 的细胞数量。
    buffer_batch_num
        缓冲区中最多缓存的 minibatch 数量；总容量为
        ``batch_size * buffer_batch_num`` 个细胞。

    Notes
    -----
    这是内部工具类。普通用户通常通过 ``atlas.get_minibatch_dense(...)`` 间接使用。

    Examples
    --------
    在内部测试中缓存并抽样 dense minibatch::

        buffer = ShuffleBuffer(gene_num=2000, batch_size=128, buffer_batch_num=2)
        buffer.add_batch(
            np.zeros((128, 2000), dtype=np.float32),
            np.arange(128, dtype=np.int64),
        )
        buffer.add_batch(
            np.ones((128, 2000), dtype=np.float32),
            np.arange(128, 256, dtype=np.int64),
        )
        X_batch, filter_cell_ids = buffer.sample_batch()

    处理最后不足一个完整 buffer 的剩余数据::

        buffer = ShuffleBuffer(gene_num=100, batch_size=32, buffer_batch_num=4)
        buffer.add_batch(
            np.zeros((20, 100), dtype=np.float32),
            np.arange(20, dtype=np.int64),
        )
        remaining_batches = buffer.flush_remaining()"""

    def __init__(self, gene_num: int, batch_size: int, buffer_batch_num: int):

        """初始化对象。

        该内部函数属于minibatch 流式读取模块，用于支撑同一模块中的公共 API。

        从过滤后的 HyS 稀疏表恢复 CSR 或 dense minibatch，服务于 PCA、KMeans 和大规模训练。

        它通常不会作为用户入口直接调用；直接调用时需要保证输入对象、数据库连接和相关临时表已经由上游步骤准备好。

        Parameters
        ----------
        gene_num
            dense minibatch 中的基因数量。

        batch_size
            每批读取、写入或处理的细胞数量；较大值通常更快但占用更多内存。

        buffer_batch_num
            shuffle buffer 中缓存的 minibatch 数量。

        Notes
        -----
        这是内部 helper；除非需要扩展 scAtlasPy 内部流程，一般不建议在用户代码中直接调用。
        """
        self.batch_size = batch_size
        self.gene_num = gene_num
        self.buffer_batch_num = buffer_batch_num

        # buffer 最大 cell 数
        self.buffer_cells = buffer_batch_num * batch_size

        # 实际 buffer：表达矩阵
        self.X = np.zeros((self.buffer_cells, gene_num), dtype=np.float32)

        # 同步保存每一行对应的 filter_cell_id，保证 shuffle 后还能写回原细胞
        self.filter_cell_ids = np.empty(self.buffer_cells, dtype=np.int64)

        # 写入指针
        self.write_ptr = 0

        # 当前输出 batch id
        self.output_batch_id = 0

        # buffer 是否已经 shuffle
        self.shuffled = False


    # 写入一个 batch 到 缓冲区
    def add_batch(
            self,
            X_batch: np.ndarray,
            filter_cell_ids: np.ndarray,
    ):
        """向 shuffle buffer 写入一个 dense minibatch。

        该方法把当前 dense 表达矩阵追加到缓冲区。当缓冲区累计达到
        ``batch_size * buffer_batch_num`` 个细胞后，会随机打乱缓冲区中的细胞顺序，
        并切换到可通过 ``sample_batch`` 读取的输出状态。

        Parameters
        ----------
        X_batch
            当前 dense minibatch。行表示细胞，列表示基因；列数需要与初始化时的
            ``gene_num`` 一致，行数通常为 ``batch_size``，最后一个 batch 可以更小。
        filter_cell_ids
            当前 ``X_batch`` 每一行对应的 ``filter_cell_id``，长度必须等于
            ``X_batch.shape[0]``。

        Returns
        -------
        None
            该方法只更新缓冲区状态，不直接返回训练数据。

        Examples
        --------
        写入两个 batch 并触发 shuffle::

            buffer = ShuffleBuffer(gene_num=50, batch_size=16, buffer_batch_num=2)
            buffer.add_batch(
                np.random.rand(16, 50).astype(np.float32),
                np.arange(16, dtype=np.int64),
            )
            buffer.add_batch(
                np.random.rand(16, 50).astype(np.float32),
                np.arange(16, 32, dtype=np.int64),
            )
            X_batch, filter_cell_ids = buffer.sample_batch()

        在缓冲区尚未填满时，``sample_batch`` 会返回 ``None``::

            buffer = ShuffleBuffer(gene_num=50, batch_size=16, buffer_batch_num=2)
            buffer.add_batch(
                np.random.rand(16, 50).astype(np.float32),
                np.arange(16, dtype=np.int64),
            )
            assert buffer.sample_batch() is None"""

        # 如果 buffer 已经满并进入输出阶段，不再写入
        if self.shuffled:
            return

        n = X_batch.shape[0]

        if len(filter_cell_ids) != n:
            raise RuntimeError(
                "ShuffleBuffer.add_batch: filter_cell_ids 长度必须等于 X_batch 行数。"
            )

        # 安全保护（防止越界）
        if self.write_ptr + n > self.buffer_cells:
            raise RuntimeError("ShuffleBuffer overflow")

        # 写入 buffer，X 和 filter_cell_ids 必须保持同一行对应关系
        self.X[self.write_ptr:self.write_ptr + n] = X_batch
        self.filter_cell_ids[self.write_ptr:self.write_ptr + n] = filter_cell_ids
        self.write_ptr += n

        # 如果 buffer 已满
        if self.write_ptr == self.buffer_cells:
            # 随机打乱：X 和 filter_cell_ids 使用同一个 perm
            perm = np.random.permutation(self.buffer_cells)
            self.X[:] = self.X[perm]
            self.filter_cell_ids[:] = self.filter_cell_ids[perm]

            # 进入输出阶段
            self.output_batch_id = 0
            self.shuffled = True


    # 输出一个 batch
    def sample_batch(self):

        """执行 ``sample_batch`` 的核心功能。

        从过滤后的 HyS 稀疏表恢复 CSR 或 dense minibatch，服务于 PCA、KMeans 和大规模训练。

        函数会直接读取或写入 Atlas 数据库中的相关表，并尽量通过 SQL、分块读取或流式计算减少内存占用。

        整体用法和 Scanpy 中相近的 ``sap.sample_batch`` 风格 API 类似，但结果保存在 Atlas
        数据库表中，便于后续步骤复用。

        Returns
        -------
        result
            函数返回结果。具体类型取决于参数设置和内部执行路径。

        Examples
        --------
        调用该函数：::

            sap.sample_batch(...)
        """
        if not self.shuffled:  # 如果还没凑够 buffer
            return None

        start = self.output_batch_id * self.batch_size
        end = start + self.batch_size

        X_batch = self.X[start:end]
        filter_cell_ids = self.filter_cell_ids[start:end]

        self.output_batch_id += 1

        # 如果已经输出完所有 batch
        if self.output_batch_id == self.buffer_batch_num:
            # reset buffer 状态
            self.write_ptr = 0
            self.output_batch_id = 0
            self.shuffled = False

        return X_batch, filter_cell_ids


    # 输出未凑满 buffer 的剩余 batch， 防止数据集 batch 数 < buffer_batch_num 时，一个 batch 都不输出
    def flush_remaining(self):

        """执行 ``flush_remaining`` 的核心功能。

        从过滤后的 HyS 稀疏表恢复 CSR 或 dense minibatch，服务于 PCA、KMeans 和大规模训练。

        函数会直接读取或写入 Atlas 数据库中的相关表，并尽量通过 SQL、分块读取或流式计算减少内存占用。

        整体用法和 Scanpy 中相近的 ``sap.flush_remaining`` 风格 API 类似，但结果保存在 Atlas
        数据库表中，便于后续步骤复用。

        Returns
        -------
        result
            函数返回结果。具体类型取决于参数设置和内部执行路径。

        Examples
        --------
        调用该函数：::

            sap.flush_remaining(...)
        """
        if self.write_ptr == 0:
            return []

        n_cells = self.write_ptr

        # 只打乱已经写入的 cell；X 和 filter_cell_ids 使用同一个 perm
        perm = np.random.permutation(n_cells)
        X_remain = self.X[:n_cells][perm]
        filter_cell_ids_remain = self.filter_cell_ids[:n_cells][perm]

        batches = []

        start = 0
        while start < n_cells:
            end = min(start + self.batch_size, n_cells)
            batches.append((
                X_remain[start:end].copy(),
                filter_cell_ids_remain[start:end].copy(),
            ))
            start = end

        # reset
        self.write_ptr = 0
        self.output_batch_id = 0
        self.shuffled = False

        return batches


''' 多线程 输出minibatch：  Producer → Queue → Reorder → RingBuffer → Consumer（有序） '''
class MultiThreadedMinibatchFetcher:

    """多线程 minibatch 读取器。

    该类从 Atlas 过滤后的 HyS 表中按 batch 恢复 CSR 或 dense 表达矩阵，
    使用 producer/queue 结构预取数据，并保证消费者按 batch 顺序获得结果。
    它是 ``atlas.get_minibatch_csr`` 和 ``atlas.get_minibatch_dense`` 的底层实现。

    Parameters
    ----------
    file_path
        Atlas ``.sasql`` 数据库文件路径。
    batch_size
        每个 minibatch 的细胞数量。
    x_type
        输出矩阵类型。常用值为 ``"CSR"`` 或 ``"dense"``。
    pass_mode
        遍历模式。``"single-pass"`` 顺序遍历一次，``"multi-pass"`` 可结合
        ``ShuffleBuffer`` 做随机化多批训练。
    buffer_batch_num
        ``multi-pass`` 模式下 shuffle buffer 缓存的 batch 数量。
    max_batches
        最多输出的 minibatch 数量；为 ``None`` 时不限制。
    return_cell_ids
        是否在输出中携带每行对应的 ``filter_cell_id``。默认关闭以兼容旧版矩阵输出。

    Notes
    -----
    这是内部流式读取器。普通用户通常通过 Atlas 对象方法读取 minibatch。

    Examples
    --------
    通过 Atlas 对象按 dense batch 读取数据::

        atlas.build_read_index(use_hvg=True)
        for X_batch in atlas.get_minibatch_dense(batch_size=2048):
            print(X_batch.shape)
            break

    直接创建读取器进行底层调试::

        fetcher = MultiThreadedMinibatchFetcher(
            atlas.file_path,
            batch_size=1024,
            x_type="CSR",
            pass_mode="single-pass",
            max_batches=10,
        )
        for batch in fetcher.run():
            X_batch = batch["X"] if isinstance(batch, dict) else batch
            print(X_batch.shape)"""

    def __init__(self, file_path: PathLike[str] | str,
                 batch_size: int=2048,
                 x_type: str= "CSR",
                 pass_mode: str="multi-pass",
                 buffer_batch_num: int=5,
                 max_batches: int | None=None,  # 最多输出多少个 batch
                 return_cell_ids: bool=False,
                 ):

        """初始化对象。

        该内部函数属于minibatch 流式读取模块，用于支撑同一模块中的公共 API。

        从过滤后的 HyS 稀疏表恢复 CSR 或 dense minibatch，服务于 PCA、KMeans 和大规模训练。

        它通常不会作为用户入口直接调用；直接调用时需要保证输入对象、数据库连接和相关临时表已经由上游步骤准备好。

        Parameters
        ----------
        file_path
            输入文件路径或 Atlas ``.sasql`` 数据库文件路径。

        batch_size
            每批读取、写入或处理的细胞数量；较大值通常更快但占用更多内存。

        x_type
            输出矩阵类型，通常为 ``"CSR"`` 或 ``"dense"``。

        pass_mode
            minibatch 遍历模式，通常为 ``"single-pass"`` 或 ``"multi-pass"``。

        buffer_batch_num
            shuffle buffer 中缓存的 minibatch 数量。

        max_batches
            最多输出的 minibatch 数量；为 ``None`` 时不限制。

        return_cell_ids
            是否在输出 batch 时同时返回 ``filter_cell_ids``。默认 ``False``，保持旧版只返回矩阵的行为；
            为 ``True`` 时返回 ``{"X": X_batch, "filter_cell_ids": filter_cell_ids}``。

        Notes
        -----
        这是内部 helper；除非需要扩展 scAtlasPy 内部流程，一般不建议在用户代码中直接调用。
        """

        self.X_type = x_type  # 输出的X表格式 "CSR" "dense"(宽表)
        self.file_path = fspath(file_path)  # sasql 文件的绝对路径
        self.batch_size = batch_size
        self.producer_num = 10  # 线程数量
        self.gene_num = self._get_gene_num()  # 获取基因数量
        self.index_data = self._get_index_data()  # 从数据库读取 build_read_index(use_data=...) 保存的信息 “data_log1p”或者“data_scale”等等
        self.zero_scale_transform = self._get_zero_scale_transform()
        # 获取 var 表的 zero_scale_transform ，就是每个基因的 ( 0 - g.mean) / g.std

        # 本轮输出多少个 batch
        self.max_batches = max_batches
        self.return_cell_ids = return_cell_ids
        self.stop_event = threading.Event()  # 提前停止信号

        self.out_queue = queue.Queue(maxsize=20)  # 输出队列

        self.fetch_size = 1_0000_0000  # fetch_record_batch 流式读取的size
        self.pass_mode = pass_mode  # single-pass 单次遍历 ，multi-pass 多次遍历
        self.buffer_batch_num = buffer_batch_num  # 多次遍历 ，缓冲区的容量，n: 表示batch_size * n

        self.indptr_queue = self._prepare_indptr()  # 获取 indptr 的 rb 读取数据流

        self.batch_cell_counts, self.batch_nnz = self._prepare_batch_info_sql()  # 获取batch_nnz (每批cell的非零值数量)
        self.batch_idx = 0  # 批次的编号
        self.batch_num = len(self.batch_nnz)  # 批次数量

        self.queue = queue.Queue(maxsize=self.producer_num * 5)  # 数据缓存队列 ：Queue（核心）

        # Ring Buffer ：切分出batch的环形缓冲池
        self.pool_size = self.fetch_size * 10  # 容量
        self.pool_gene_id = np.empty(self.pool_size, dtype=np.uint16)
        self.pool_data = np.empty(self.pool_size, dtype=np.float32)

        self.read_ptr = 0  # 读指针
        self.write_ptr = 0  # 写指针
        self.used_size = 0  # 当前 Ring Buffer 里的 nnz 数据
        self.total_batches = 0  # 计数器

        # 输出速度统计
        self.output_start_time = None
        self.output_last_time = None
        self.output_cells = 0
        self.speed_log_every = 5  # 每输出多少个 batch 打印一次速度；想每个batch都打印就改成1


    def _get_zero_scale_transform(self):
        """获取 dense 矩阵中稀疏 0 位点的填充值。

        如果当前 build_read_index 使用的是 data_scale，
        说明数据已经 scale，稀疏矩阵中原来的 0 值需要填成
        var.zero_scale_transform。

        如果当前使用的是 data、data_normalize、data_log1p 等未 scale 数据，
        稀疏矩阵中原来的 0 位点应该填成 0.0。
        """

        logger.info(
            f"[Minibatch] index_data = {self.index_data!r}; "
        )

        # 只有 data_scale 才需要读取 zero_scale_transform
        if self.index_data != "data_scale":
            logger.info(
                f"[Minibatch] read index use_data={self.index_data!r}; "
                "use 0.0 as dense fill value."
            )
            return np.zeros(self.gene_num, dtype=np.float32)

        conn = duckdb.connect(self.file_path)

        try:
            # 新增：即使是 data_scale，也先检查字段是否存在
            var_columns = conn.execute("PRAGMA table_info('var')").fetchdf()["name"].tolist()

            if "zero_scale_transform" not in var_columns:
                logger.info(
                    "[Minibatch] read index use_data='data_scale', "
                    "but var.zero_scale_transform not found; "
                    "use 0.0 as dense fill value."
                )
                return np.zeros(self.gene_num, dtype=np.float32)

            arr = conn.execute("""
                   SELECT zero_scale_transform
                   FROM var
                   WHERE filter_gene_id IS NOT NULL
                   ORDER BY filter_gene_id
               """).fetchnumpy()["zero_scale_transform"]

            return arr.astype("float32")

        finally:
            conn.close()


    def _get_index_data(self) -> str | None:
        """从数据库读取当前 read index 使用的表达值字段。"""

        conn = duckdb.connect(self.file_path)

        try:
            tables = [r[0] for r in conn.execute("SHOW TABLES").fetchall()]

            if "atlas_read_index_meta" not in tables:
                return None

            row = conn.execute("""
                SELECT value
                FROM atlas_read_index_meta
                WHERE key = 'use_data'
            """).fetchone()

            if row is None:
                return None

            return row[0]

        finally:
            conn.close()


    def _get_gene_num(self):
        """获取数据库或对象中的内部信息。

        该内部函数属于minibatch 流式读取模块，用于支撑同一模块中的公共 API。

        从过滤后的 HyS 稀疏表恢复 CSR 或 dense minibatch，服务于 PCA、KMeans 和大规模训练。

        它通常不会作为用户入口直接调用；直接调用时需要保证输入对象、数据库连接和相关临时表已经由上游步骤准备好。

        当前实现中会访问或生成的关键表包括：``var``。

        Returns
        -------
        result
            函数返回结果。具体类型取决于参数设置和内部执行路径。

        Notes
        -----
        这是内部 helper；除非需要扩展 scAtlasPy 内部流程，一般不建议在用户代码中直接调用。
        """

        conn = duckdb.connect(self.file_path)
        gene_num = conn.execute(
            "SELECT COUNT(*) FROM var WHERE filter_gene_id IS NOT NULL"
        ).fetchone()[0]
        ("gene_num:", gene_num)
        conn.close()
        return gene_num


    def _prepare_indptr(self):
        """准备按 ``filter_cell_id`` 顺序读取的 indptr 数据。

        返回的 record batch 第 0 列为 ``filter_cell_id``，第 1 列为累积 ``indptr``。
        consumer 会同时使用二者，保证 ``X_batch[i, :]`` 能对应到 ``filter_cell_ids[i]``。

        它通常不会作为用户入口直接调用；直接调用时需要保证输入对象、数据库连接和相关临时表已经由上游步骤准备好。

        当前实现中会访问或生成的关键表包括：``X_HyS_indptr_filtered``。

        Returns
        -------
        result
            函数返回结果。具体类型取决于参数设置和内部执行路径。

        Notes
        -----
        这是内部 helper；除非需要扩展 scAtlasPy 内部流程，一般不建议在用户代码中直接调用。
        """

        conn = duckdb.connect(self.file_path)
        conn.execute("PRAGMA enable_progress_bar=false")

        fetch_record_indptr = conn.execute(
            """
            SELECT
                filter_cell_id,
                indptr
            FROM X_HyS_indptr_filtered
            ORDER BY filter_cell_id
            """

        ).fetch_record_batch(rows_per_batch=self.batch_size)

        q = queue.Queue()
        for rb in fetch_record_indptr:
            q.put(rb)
        conn.close()
        return q


    def _prepare_batch_info_sql(self):
        """执行 ``_prepare_batch_info_sql`` 的核心功能。

        该内部函数属于minibatch 流式读取模块，用于支撑同一模块中的公共 API。

        从过滤后的 HyS 稀疏表恢复 CSR 或 dense minibatch，服务于 PCA、KMeans 和大规模训练。

        它通常不会作为用户入口直接调用；直接调用时需要保证输入对象、数据库连接和相关临时表已经由上游步骤准备好。

        当前实现中会访问或生成的关键表包括：``X_HyS_indptr_filtered``。

        Returns
        -------
        result
            函数返回结果。具体类型取决于参数设置和内部执行路径。

        Notes
        -----
        这是内部 helper；除非需要扩展 scAtlasPy 内部流程，一般不建议在用户代码中直接调用。
        """

        conn = duckdb.connect(self.file_path)
        conn.execute("PRAGMA enable_progress_bar=false")

        query = f"""
        WITH t AS (
            SELECT
                filter_cell_id,
                indptr,
                ROW_NUMBER() OVER (ORDER BY filter_cell_id) - 1 AS rn
            FROM X_HyS_indptr_filtered
        ),
        b AS (
            SELECT
                rn // {self.batch_size} AS batch_id,
                COUNT(*) AS n_cells,
                MAX(indptr) AS end_indptr
            FROM t
            GROUP BY batch_id
        )
        SELECT
            batch_id,
            CAST(n_cells AS INTEGER) AS n_cells,
            CAST(
                end_indptr - LAG(end_indptr, 1, 0) OVER (ORDER BY batch_id)
                AS BIGINT
            ) AS batch_nnz
        FROM b
        ORDER BY batch_id
        """

        rows = conn.execute(query).fetchall()
        conn.close()

        batch_cell_counts = [int(r[1]) for r in rows]
        batch_nnz = [int(r[2]) for r in rows]

        return batch_cell_counts, batch_nnz


    def _producer(self, tid: int):
        """执行 ``_producer`` 的核心功能。

        该内部函数属于minibatch 流式读取模块，用于支撑同一模块中的公共 API。

        从过滤后的 HyS 稀疏表恢复 CSR 或 dense minibatch，服务于 PCA、KMeans 和大规模训练。

        它通常不会作为用户入口直接调用；直接调用时需要保证输入对象、数据库连接和相关临时表已经由上游步骤准备好。

        当前实现中会访问或生成的关键表包括：``X_HyS_data_filtered``。

        Parameters
        ----------
        tid
            producer 线程或数据分片编号。

        Notes
        -----
        这是内部 helper；除非需要扩展 scAtlasPy 内部流程，一般不建议在用户代码中直接调用。
        """

        conn = duckdb.connect(self.file_path)
        conn.execute("PRAGMA enable_progress_bar=false")

        query = f"""
            SELECT
                rowid,
                filter_gene_id,
                data
            FROM X_HyS_data_filtered
            WHERE tid = {tid}
            -- ORDER BY rowid  -- 新增
        """

        result = conn.execute(query).fetch_record_batch(
            rows_per_batch=self.fetch_size
        )

        rb_count = 0
        nnz_count = 0

        try:
            for rb in result:

                # 新增：提前停止
                if self.stop_event.is_set():
                    break

                # 读取 rowid / gene_id / data
                rowids = rb.column(0).to_numpy().astype(np.int64)
                gene_id = rb.column(1).to_numpy().astype(np.uint16)
                data = rb.column(2).to_numpy().astype(np.float32)

                if len(rowids) == 0:
                    continue

                # 用真实 rowid 计算 seq_id
                seq_start = int(rowids[0] // self.fetch_size)  # 当前 rb 第一行 rowid 属于哪个 block
                seq_end = int(rowids[-1] // self.fetch_size)  # 当前 rb 最后一行 rowid 属于哪个 block

                # 安全检查: 一个 rb 理论上只能属于同一个 seq block
                # 如果跨 block，说明 rows_per_batch / tid 分片不匹配
                if seq_start != seq_end:
                    raise RuntimeError(
                        f"[Producer-{tid}] 一个 record batch 跨越多个 seq block: "
                        f"{seq_start} -> {seq_end}, "
                        f"rowid_start={rowids[0]}, rowid_end={rowids[-1]}, "
                        f"fetch_size={self.fetch_size}"
                    )

                seq_id = seq_start

                while not self.stop_event.is_set():
                    try:
                        self.queue.put((seq_id, gene_id, data), timeout=0.5)
                        break
                    except queue.Full:
                        continue

                rb_count += 1
                nnz_count += len(gene_id)

        finally:
            conn.close()

            # 通知 consumer：这个 producer 完成
            try:
                self.queue.put(None, timeout=0.5)
            except queue.Full:
                pass


    def _consumer(self):
        """执行 ``_consumer`` 的核心功能。

        该内部函数属于minibatch 流式读取模块，用于支撑同一模块中的公共 API。

        从过滤后的 HyS 稀疏表恢复 CSR 或 dense minibatch，服务于 PCA、KMeans 和大规模训练。

        它通常不会作为用户入口直接调用；直接调用时需要保证输入对象、数据库连接和相关临时表已经由上游步骤准备好。

        Notes
        -----
        这是内部 helper；除非需要扩展 scAtlasPy 内部流程，一般不建议在用户代码中直接调用。
        """

        reorder_buffer = {}  # 乱序数据 缓存区 : reorder_buffer[seq_id] = (gene_id, data) ； 大小是动态的

        expected_seq = 0  # 下一个想要的的 batch 序号
        global_indptr_offset = 0  # 用于修正 indptr 的累积偏移量

        prepared_batches = 0  # 已经读出来、构建成 dense、并放进 ShuffleBuffer 的原始 batch 数。

        # 构建宽表的输出缓冲区
        shuffle_buffer = ShuffleBuffer(
            gene_num=self.gene_num,
            batch_size=self.batch_size,
            buffer_batch_num=self.buffer_batch_num
        )

        # 当前批次号 < 批次数量
        while self.batch_idx < self.batch_num:

            # 如果已经准备/输出够 max_batches，提前结束本轮
            if self._read_limit_reached(prepared_batches):
                logger.debug(
                    f"[Consumer] read limit reached, "
                    f"batch_idx={self.batch_idx}, "
                    f"prepared_batches={prepared_batches}, "
                    f"output_batches={self.total_batches}"
                )
                self.stop_event.set()
                break

            need = self.batch_nnz[self.batch_idx]  # 当前 batch 所需 nnz
            current_batch_cells = self.batch_cell_counts[
                self.batch_idx]  # 当前 batch 的真实 cell 数,最后一个 batch 可能小于 self.batch_size

            # RingBuffer 中的数据不够， 填充 RingBuffer，直到够一个 batch
            while self.used_size < need:

                # 如果已经不需要继续读，跳出，避免死等 queue
                if self._read_limit_reached(prepared_batches):
                    self.stop_event.set()
                    break

                # 加 timeout，避免 producer 提前停止后 consumer 永久阻塞
                try:
                    item = self.queue.get(timeout=0.5)  # 从 Queue 获取下一条数据，阻塞等待，自带锁
                except queue.Empty:
                    if self.stop_event.is_set():
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

            # 如果因为 max_batches / stop_event 跳出，且 RingBuffer 还不够当前 batch，就结束主循环
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

                # 构建当前 batch 的 filter_cell_ids 和 indptr
                indptr_rb = self.indptr_queue.get()

                # 当前 X 每一行对应的 filter_cell_id
                filter_cell_ids = np.array(indptr_rb.column(0), dtype=np.int64)

                # 原始累积 indptr 在第 1 列
                indptr_raw = np.array(indptr_rb.column(1), dtype=np.int64)
                last_val = indptr_raw[-1]
                indptr_now = np.concatenate(([0], indptr_raw - global_indptr_offset))
                global_indptr_offset = last_val

                # 检查 indptr 行数是否等于当前 batch cell 数
                if len(indptr_now) != current_batch_cells + 1:
                    raise RuntimeError(
                        f"[Consumer] indptr 长度不匹配: "
                        f"len(indptr_now)={len(indptr_now)}, "
                        f"current_batch_cells={current_batch_cells}, "
                        f"batch_idx={self.batch_idx}"
                    )

                if len(filter_cell_ids) != current_batch_cells:
                    raise RuntimeError(
                        f"[Consumer] filter_cell_ids 长度不匹配: "
                        f"len(filter_cell_ids)={len(filter_cell_ids)}, "
                        f"current_batch_cells={current_batch_cells}, "
                        f"batch_idx={self.batch_idx}"
                    )

                if self.X_type == "CSR":
                    # 输出类型1 ：CSR 格式
                    X = sp.csr_matrix((current_batch_cells, self.gene_num), dtype=np.float32)

                    X.data = vals.copy()
                    X.indices = cols.copy()
                    X.indptr = indptr_now

                    self._put_output(
                        X,
                        filter_cell_ids.copy(),
                    )

                if self.X_type == "dense":
                    # 输出类型2 ：dense 格式

                    X_dense = np.empty((current_batch_cells, self.gene_num), dtype=np.float32)
                    X_dense[:] = self.zero_scale_transform  # 按 gene_id 填充，self.zero_scale_transform
                    #  zero_scale_transform    将每个基因的 ( 0 - g.mean) / g.std 存入var表的该字段，以便将来调用
                    # 如果没有运行过 scale()，说明当前数据仍然是 normalize/log1p 等未中心化数据，则 self.zero_scale_transform 为全0填充
                    # X_dense =
                    # [ 填充每个基因的 zero_scale_transform
                    #  [-0.5, 0.2, -1.1, ...],
                    #  [-0.5, 0.2, -1.1, ...],
                    #  ...
                    # ]
                    rows = np.repeat(  # [0,0, 1, 2,2,2] 每个非零元素对应的“行号”
                        np.arange(current_batch_cells),  # [0,1,2,...]  表示每个 cell（行）
                        np.diff(indptr_now)  # [2, 1, 3, ...]  每个 cell 有多少个非零值（nnz）
                    )
                    X_dense[rows, cols] = vals
                    # 将非零值写入对应的 行列
                    # X_dense[0,1] = 10
                    # X_dense[0,3] = 20

                    if self.pass_mode == "single-pass":  # 单次遍历
                        self._put_output(
                            X_dense.copy(),
                            filter_cell_ids.copy(),
                        )

                    if self.pass_mode == "multi-pass":  # 多次遍历 （加入缓存区，保证多次的随机性）

                        # multi-pass 下，先统计 prepared_batches
                        # 这个 batch 已经进入 ShuffleBuffer，
                        # 即使暂时没输出，后面 flush_remaining 也会输出。
                        shuffle_buffer.add_batch(
                            X_dense,
                            filter_cell_ids,
                        )  # 写入输出缓存区 shuffle buffer
                        prepared_batches += 1

                        # 一旦 ShuffleBuffer 满了，立刻把当前 shuffle 后的 buffer 全部吐出去。
                        while True:
                            batch_random = shuffle_buffer.sample_batch()  # 从输出缓存区随机采样 batch，保证多次遍历随机性

                            if batch_random is None:
                                break

                            X_dense_random, filter_cell_ids_random = batch_random

                            ok = self._put_output(
                                X_dense_random.copy(),  # 你每轮都会复用同一块 buffer， 不 copy 会被覆盖
                                filter_cell_ids_random.copy(),
                            )

                            if not ok or self._output_limit_reached():
                                break

                        # 如果已经准备够 max_batches，后面不再继续读新 batch，交给尾部 flush 输出剩余。
                        if self._read_limit_reached(prepared_batches):
                            self.stop_event.set()

                self.batch_idx += 1

        #  multi-pass 模式：输出 ShuffleBuffer 里没凑满的尾部 batch
        if self.X_type == "dense" and self.pass_mode == "multi-pass":

            remain_batches = shuffle_buffer.flush_remaining()

            for X_remain, filter_cell_ids_remain in remain_batches:

                # 防止尾部输出超过 max_batches
                if self._output_limit_reached():
                    self.stop_event.set()
                    break

                self._put_output(
                    X_remain.copy(),
                    filter_cell_ids_remain.copy(),
                )

        logger.info(
            f"[Done] processed_batches={self.batch_idx}, "
            f"output_batches={self.total_batches}"
        )

        # 通知 run() 结束
        self.out_queue.put(None)


    def run(self):
        """执行 ``run`` 的核心功能。

        从过滤后的 HyS 稀疏表恢复 CSR 或 dense minibatch，服务于 PCA、KMeans 和大规模训练。

        函数会直接读取或写入 Atlas 数据库中的相关表，并尽量通过 SQL、分块读取或流式计算减少内存占用。

        整体用法和 Scanpy 中相近的 ``sap.run`` 风格 API 类似，但结果保存在 Atlas 数据库表中，便于后续步骤复用。

        Yields
        -------
        batch
            当 ``return_cell_ids=False`` 时，逐批生成 CSR 或 dense 矩阵；
            当 ``return_cell_ids=True`` 时，逐批生成
            ``{"X": X_batch, "filter_cell_ids": filter_cell_ids}``。

        Examples
        --------
        调用该函数：::

            sap.run(...)
        """

        # producers 多线程
        producers = []
        for i in range(self.producer_num):
            t = threading.Thread(target=self._producer, args=(i,))
            t.start()
            producers.append(t)

        # consumer 单线程
        consumer = threading.Thread(target=self._consumer)
        consumer.start()

        # 从 out_queue 统一 yield
        while True:
            batch = self.out_queue.get()  # 阻塞
            if batch is None:  # 收到哨兵，说明所有 batch 都吐完
                break
            yield batch  # 正常 batch 继续向外 yield

        for t in producers:
            t.join()

        consumer.join()


    # 辅助函数 1：是否已经达到输出上限
    def _output_limit_reached(self):

        """执行 ``_output_limit_reached`` 的核心功能。

        该内部函数属于minibatch 流式读取模块，用于支撑同一模块中的公共 API。

        从过滤后的 HyS 稀疏表恢复 CSR 或 dense minibatch，服务于 PCA、KMeans 和大规模训练。

        它通常不会作为用户入口直接调用；直接调用时需要保证输入对象、数据库连接和相关临时表已经由上游步骤准备好。

        Returns
        -------
        result
            函数返回结果。具体类型取决于参数设置和内部执行路径。

        Notes
        -----
        这是内部 helper；除非需要扩展 scAtlasPy 内部流程，一般不建议在用户代码中直接调用。
        """
        return (
                self.max_batches is not None
                and self.total_batches >= self.max_batches
        )


    # 辅助函数 2：是否应该停止继续读取新 batch
    def _read_limit_reached(self, prepared_batches: int):

        """执行 ``_read_limit_reached`` 的核心功能。

        该内部函数属于minibatch 流式读取模块，用于支撑同一模块中的公共 API。

        从过滤后的 HyS 稀疏表恢复 CSR 或 dense minibatch，服务于 PCA、KMeans 和大规模训练。

        它通常不会作为用户入口直接调用；直接调用时需要保证输入对象、数据库连接和相关临时表已经由上游步骤准备好。

        Parameters
        ----------
        prepared_batches
            已经准备并放入 shuffle buffer 的 batch 数量。

        Returns
        -------
        result
            函数返回结果。具体类型取决于参数设置和内部执行路径。

        Notes
        -----
        这是内部 helper；除非需要扩展 scAtlasPy 内部流程，一般不建议在用户代码中直接调用。
        """
        if self.max_batches is None:
            return False

        # dense + multi-pass 下，batch 先进入 ShuffleBuffer，
        # 不一定马上输出，所以看 prepared_batches
        if self.X_type == "dense" and self.pass_mode == "multi-pass":
            return prepared_batches >= self.max_batches

        # 其他情况，读取后基本就会输出，所以看 total_batches
        return self.total_batches >= self.max_batches


    # 辅助函数 3：统一输出 batch
    def _put_output(
            self,
            X_batch,
            filter_cell_ids: np.ndarray,
    ):

        """执行 ``_put_output`` 的核心功能。

        该内部函数属于minibatch 流式读取模块，用于支撑同一模块中的公共 API。

        从过滤后的 HyS 稀疏表恢复 CSR 或 dense minibatch，服务于 PCA、KMeans 和大规模训练。

        它通常不会作为用户入口直接调用；直接调用时需要保证输入对象、数据库连接和相关临时表已经由上游步骤准备好。

        Parameters
        ----------
        X_batch
            当前 batch 的表达矩阵或 embedding 矩阵。

        filter_cell_ids
            当前 batch 中每一行对应的 ``filter_cell_id``。

        Returns
        -------
        result
            函数返回结果。具体类型取决于参数设置和内部执行路径。

        Notes
        -----
        这是内部 helper；除非需要扩展 scAtlasPy 内部流程，一般不建议在用户代码中直接调用。
        """
        if self._output_limit_reached():
            self.stop_event.set()
            return False

        if len(filter_cell_ids) != X_batch.shape[0]:
            raise RuntimeError(
                f"filter_cell_ids 长度必须等于 X_batch 行数: "
                f"len(filter_cell_ids)={len(filter_cell_ids)}, "
                f"X_batch.shape[0]={X_batch.shape[0]}"
            )

        if self.return_cell_ids:
            batch = {
                "X": X_batch,
                "filter_cell_ids": np.asarray(filter_cell_ids, dtype=np.int64),
            }
        else:
            batch = X_batch

        self.out_queue.put(batch)
        self.total_batches += 1

        # 当前速度 + 平均速度
        now = time.perf_counter()

        if self.output_start_time is None:
            self.output_start_time = now
            self.output_last_time = now

        n_cells = X_batch.shape[0]
        self.output_cells += n_cells

        interval = now - self.output_last_time
        elapsed = now - self.output_start_time

        current_batch_speed = 1.0 / interval if interval > 0 else 0.0
        current_cell_speed = n_cells / interval if interval > 0 else 0.0

        avg_batch_speed = self.total_batches / elapsed if elapsed > 0 else 0.0
        avg_cell_speed = self.output_cells / elapsed if elapsed > 0 else 0.0

        if self.total_batches % self.speed_log_every == 0:
            logger.info(
                f"[Speed] output_batches={self.total_batches}, "
                f"[ current={current_batch_speed:.2f} batch/s, "
                f"{current_cell_speed:.0f} cells/s, ]"
                f"[ avg={avg_batch_speed:.2f} batch/s, "
                f"{avg_cell_speed:.0f} cells/s ]"
            )

        self.output_last_time = now

        if self._output_limit_reached():
            logger.info(f"[Consumer] reach max_batches={self.max_batches}, stop")
            self.stop_event.set()

        return True
