import duckdb
from os import PathLike, fspath
from ..io import progress
from datetime import datetime
import logging

logger = logging.getLogger("Atlas")
logger.addHandler(logging.NullHandler())


class FilterIndexBuilder:

    """过滤索引构建器。

    该类根据 ``obs`` 和 ``var`` 中的过滤标记，重建连续的细胞索引和基因索引，
    并生成后续小批量读取所需的过滤后 HyS 表。它是 ``Atlas.build_read_index``
    的底层实现，普通用户通常不需要直接实例化。

    Parameters
    ----------
    file_path
        Atlas ``.sasql`` 数据库文件路径。
    cell_condition
        ``obs`` 中用于筛选细胞的布尔列名或条件；为 ``None`` 时保留全部细胞。
    gene_condition
        ``var`` 中用于筛选基因的布尔列名或条件；为 ``None`` 时保留全部基因。
    use_hvg
        是否在基因过滤条件之外继续限制为高变基因。
    use_data
    从 ``X_HyS_data`` 表中读取的表达值列名，例如 ``"data_count"``、
    ``"data_normalize"``、``"data_log1p"`` 或 ``"data_scale"``。

    Notes
    -----
    推荐使用 ``atlas.build_read_index(...)`` 调用该流程。直接使用本类时，需要确保
    Atlas 数据库已经包含 ``obs``、``var`` 和 ``X_HyS_data`` 等基础表。

    Examples
    --------
    通过 Atlas 对象构建过滤索引::

        sap.pp.filter_cells(atlas, min_genes=200)
        sap.pp.filter_genes(atlas, min_cells=3)
        atlas.build_read_index(
            cell_condition="filter_cells",
            gene_condition="filter_genes",
            use_hvg=True,
            use_data="data_log1p",
        )

    直接使用底层构建器，适合调试或扩展内部流程::

        builder = FilterIndexBuilder(
            atlas.file_path,
            cell_condition="filter_cells",
            gene_condition="filter_genes",
            use_hvg=False,
            use_data="data_count",
        )
        builder.run()"""
    def __init__(
        self,
        file_path: PathLike[str] | str,
        *, # file_path 可以按位置传， * 后面的参数必须写参数名
        cell_condition: str | None = None,
        gene_condition: str | None = None,
        use_hvg: bool = True,
        use_data: str = "data_log1p",
    ):
        """初始化过滤索引构建器。

        该构造函数保存 Atlas 数据库路径、过滤条件、HVG 设置和表达值列名，
        并建立一个新的 DuckDB 连接。实际的索引重建流程由 ``run()`` 执行。

        Parameters
        ----------
        file_path
            Atlas ``.sasql`` 数据库文件路径。

        cell_condition
            ``obs`` 表中用于筛选细胞的布尔列名。为 ``None`` 时保留全部细胞。

        gene_condition
            ``var`` 表中用于筛选基因的布尔列名。为 ``None`` 时保留全部基因。

        use_hvg
            是否在基因过滤条件之外继续叠加 ``highly_variable_genes=TRUE``。

        use_data
            从 ``X_HyS_data`` 表中读取的表达值列名。

        Notes
        -----
        该类通常由 ``Atlas.build_read_index(...)`` 调用，普通用户一般不需要直接实例化。
        """

        self.file_path = fspath(file_path)       # sasql 文件绝对路径
        self.producer_num = 10           # minibatch 流式读取的线程数
        self.fetch_size = 500_0000    # minibatch 流式读取的size
        self.chunk_size = 1000_0000    # 每次处理的数据量

        self.cell_condition = cell_condition # cell 过滤条件 filter_cells ：表示只选 filter_cells = True 的cell
        self.gene_condition = gene_condition # gene 过滤条件 filter_genes ：表示只选 filter_genes = True 的gene
        self.use_hvg = use_hvg               # 是否 使用hvg基因
        self.use_data = use_data       # 选择什么数据进行处理

        self.conn = duckdb.connect(file_path)
        self.conn.execute("PRAGMA preserve_insertion_order=true")
        #  false 不强制保留插入时的输入顺序，允许 DuckDB 为了性能重新组织执行和写入顺序。
        #  true  尽量保留 INSERT / COPY / SELECT 写入时的输入顺序


    def _has_column(self, table_name: str, column_name: str) -> bool:

        return self.conn.execute(
            """
            SELECT 1
            FROM information_schema.columns
            WHERE table_name = ?
              AND column_name = ?
            LIMIT 1
            """,
            [table_name, column_name],
        ).fetchone() is not None


    def _require_filter_column(
        self,
        *,
        table_name: str,
        column_name: str,
        function_name: str,
    ) -> None:

        if self._has_column(table_name, column_name):
            return

        raise ValueError(
            f"{table_name} 表中缺少过滤字段 {column_name}，"
            f"请先运行 sap.pp.{function_name}(...) 生成该字段，"
        )


    # 外部入口
    def run(self):

        """执行完整的读取索引构建流程。

        该方法依次完成：

        1. 重建 ``obs.filter_cell_id``
        2. 重建 ``var.filter_gene_id``
        3. 构建过滤后的表达矩阵表 ``X_HyS_data_filtered``
        4. 构建过滤后的 indptr 表 ``X_HyS_indptr_filtered``

        执行完成后会关闭当前 DuckDB 连接。

        Returns
        -------
        None
            结果直接写入 Atlas ``.sasql`` 数据库。
        """

        start = datetime.now()

        self._rebuild_obs_filter_id()   # 重排 obs： 过滤细胞 + 生成 filter_cell_id
        self._rebuild_var_filter_id()   # 重排 var： 过滤基因 + 选择HVG基因 + 生成 filter_gene_id

        self._rebuild_x_hys_data_filtered()

        self._rebuild_x_hys_indptr_filtered()

        self.conn.close()

        logger.info(f"build_read_index Done, 耗时: {(datetime.now() - start).total_seconds():.2f} 秒")


    # 1.重排 obs： 过滤细胞 + 生成 filter_cell_id
    def _rebuild_obs_filter_id(self):

        """重建 ``obs.filter_cell_id``。

        该方法先删除旧的 ``filter_cell_id`` 列，再重新创建该列。
        随后根据 ``cell_condition`` 选择需要保留的细胞，并按 ``atlas_cell_id``
        顺序生成从 0 开始的连续 ``filter_cell_id``。

        未通过过滤条件的细胞，其 ``filter_cell_id`` 保持为 ``NULL``。

        Returns
        -------
        None
            结果直接写入 ``obs`` 表。

        Raises
        ------
        ValueError
            当 ``cell_condition`` 指向的过滤列不存在时，返回中文报错。
        """

        # 如果使用过滤列，先确认 obs 中已经有对应字段，避免 DuckDB 返回不易理解的英文错误
        if self.cell_condition is not None and self.cell_condition.isidentifier():
            self._require_filter_column(
                table_name="obs",
                column_name=self.cell_condition,
                function_name="filter_cells",
            )

        # 删除旧列
        self.conn.execute(""" ALTER TABLE obs DROP COLUMN IF EXISTS filter_cell_id """)

        # 新增列
        self.conn.execute(""" ALTER TABLE obs ADD COLUMN filter_cell_id INTEGER """)

        # cell_condition=None 表示不过滤细胞；否则按指定布尔字段筛选细胞
        if self.cell_condition is None:
            where_sql = "TRUE"
            logger.info("  -> 不使用 cell 过滤，保留全部 cells")
        else:
            where_sql = f"{self.cell_condition}=TRUE"
            logger.info(f"  -> 使用 cell 条件: {where_sql} 过滤")

        # 只对满足条件的 cell 重新编号
        self.conn.execute(f"""
        UPDATE obs
        SET filter_cell_id = sub.new_id
        FROM (
            SELECT
                atlas_cell_id,
                ROW_NUMBER() OVER (ORDER BY atlas_cell_id) - 1 AS new_id
            FROM obs
            WHERE {where_sql}
        ) AS sub
        WHERE obs.atlas_cell_id = sub.atlas_cell_id
        """)


    # 2.重排 var： 过滤基因 + 选择HVG基因 + 生成 filter_gene_id
    def _rebuild_var_filter_id(self):

        """重建 ``var.filter_gene_id``。

        该方法先删除旧的 ``filter_gene_id`` 列，再重新创建该列。
        随后根据 ``gene_condition`` 和 ``use_hvg`` 选择需要保留的基因，
        并按 ``atlas_gene_id`` 顺序生成从 0 开始的连续 ``filter_gene_id``。

        当 ``use_hvg=True`` 时，基因需要同时满足：

        - ``gene_condition=TRUE``
        - ``highly_variable_genes=TRUE``

        未通过过滤条件的基因，其 ``filter_gene_id`` 保持为 ``NULL``。

        Returns
        -------
        None
            结果直接写入 ``var`` 表。

        Raises
        ------
        ValueError
            当 ``gene_condition`` 指向的过滤列不存在时，返回中文报错。
        """

        # 如果使用过滤列，先确认 var 中已经有对应字段，避免 DuckDB 返回不易理解的英文错误
        if self.gene_condition is not None and self.gene_condition.isidentifier():
            self._require_filter_column(
                table_name="var",
                column_name=self.gene_condition,
                function_name="filter_genes",
            )

        # 删除旧列 + 新增列
        self.conn.execute(""" ALTER TABLE var DROP COLUMN IF EXISTS filter_gene_id """)

        self.conn.execute(""" ALTER TABLE var ADD COLUMN filter_gene_id USMALLINT """)

        # gene_condition=None 表示不过滤基因；否则按指定布尔字段筛选基因
        conditions = []

        if self.gene_condition is not None:
            conditions.append(f"({self.gene_condition})=TRUE")
            logger.info(f"  -> 使用 gene 条件: {self.gene_condition}=TRUE")
        else:
            logger.info("  -> 不使用 gene 过滤条件")

        # 如果启用 HVG，则叠加 highly_variable_genes
        if self.use_hvg:
            conditions.append("highly_variable_genes=TRUE")
            logger.info("  -> 使用 HVG gene 子集")
        else:
            logger.info("  -> 不使用 HVG 过滤，保留全部 genes")

        condition = " AND ".join(conditions) if conditions else "TRUE"

        # 重排 gene_id
        self.conn.execute(f"""
        UPDATE var
        SET filter_gene_id = sub.new_id
        FROM (
            SELECT
                atlas_gene_id,
                ROW_NUMBER() OVER (ORDER BY atlas_gene_id) - 1 AS new_id
            FROM var
            WHERE {condition}
        ) AS sub
        WHERE var.atlas_gene_id = sub.atlas_gene_id
        """)


    # 3.重建新表：X_HyS_data_filtered
    def _rebuild_x_hys_data_filtered(self):

        """构建过滤后的表达矩阵表 ``X_HyS_data_filtered``。

        该方法从原始 ``X_HyS_data`` 表中读取表达记录，并通过
        ``obs.filter_cell_id`` 和 ``var.filter_gene_id`` 只保留通过过滤的细胞和基因。

        输出表包含：

        - ``filter_cell_id``：过滤后的连续细胞索引
        - ``filter_gene_id``：过滤后的连续基因索引
        - ``data``：由 ``use_data`` 指定的表达值列
        - ``tid``：用于后续 minibatch 流式读取的分片编号

        实现上会先创建 ``_obs_keep`` 和 ``_var_keep`` 临时映射表，
        然后按 ``X_HyS_data.rowid`` 分块扫描和插入，避免一次性处理过大的表达矩阵。

        Returns
        -------
        None
            结果直接写入 ``X_HyS_data_filtered`` 表。
        """

        conn = self.conn

        conn.execute("PRAGMA preserve_insertion_order = true")
        conn.execute("PRAGMA threads=10")

        # 提前建 轻量映射表
        conn.execute("""
        CREATE OR REPLACE TEMP TABLE _obs_keep AS
        SELECT
            atlas_cell_id,
            CAST(filter_cell_id AS INTEGER) AS filter_cell_id
        FROM obs
        WHERE filter_cell_id IS NOT NULL
        """)

        conn.execute("""
        CREATE OR REPLACE TEMP TABLE _var_keep AS
        SELECT
            atlas_gene_id,
            CAST(filter_gene_id AS USMALLINT) AS filter_gene_id
        FROM var
        WHERE filter_gene_id IS NOT NULL
        """)

        # ANALYZE 是给 DuckDB 优化器看的统计信息，不改变数据、不建索引，只是可能让后面的 JOIN 更快更稳
        try:
            conn.execute("ANALYZE _obs_keep")
            conn.execute("ANALYZE _var_keep")
        except Exception:
            pass

        # 创建目标表
        conn.execute("DROP TABLE IF EXISTS X_HyS_data_filtered")

        conn.execute("""
        CREATE TABLE X_HyS_data_filtered (
            filter_cell_id INTEGER,
            filter_gene_id USMALLINT,
            data REAL,
            tid TINYINT
        )
        """)

        # 获取 rowid 范围
        min_id, max_id = conn.execute("""
            SELECT MIN(rowid), MAX(rowid)
            FROM X_HyS_data
        """).fetchone()

        if min_id is None:
            logger.debug(" X_HyS_data 是空表，跳过")
            return

        total_rows = max_id - min_id + 1
        current = min_id

        logger.debug(f"rowid 范围: {min_id:,} ~ {max_id:,}")
        logger.debug(f"总扫描行数: {total_rows:,}")
        logger.debug(f"chunk_size = {self.chunk_size:,}")

        pbar = progress(
            total=total_rows,
            unit="rows",
            desc="build_read_index",
            ncols=130
        )

        # 分块顺序插入
        while current <= max_id:
            end = min(current + self.chunk_size, max_id + 1)

            conn.execute(f"""
            INSERT INTO X_HyS_data_filtered
            SELECT
                obs.filter_cell_id,
                var.filter_gene_id,
                CAST(X.{self.use_data} AS REAL) AS data,
                CAST(0 AS TINYINT) AS tid
            FROM X_HyS_data AS X
            JOIN _obs_keep AS obs
              ON X.atlas_cell_id = obs.atlas_cell_id
            JOIN _var_keep AS var
              ON X.atlas_gene_id = var.atlas_gene_id
            WHERE X.rowid >= {current}
              AND X.rowid < {end}
            ORDER BY X.rowid
            """)

            pbar.update(end - current)
            current = end

        pbar.close()

        # 计算 tid
        conn.execute(f"""
            UPDATE X_HyS_data_filtered
            SET tid = CAST((rowid // {self.fetch_size}) % {self.producer_num} AS TINYINT)
        """)

        final_nnz = conn.execute("""
            SELECT COUNT(*)
            FROM X_HyS_data_filtered
        """).fetchone()[0]

        logger.info(f" X_HyS_data_filtered 构建完成！nnz = {final_nnz:,}")

        # 清理临时表
        conn.execute("DROP TABLE IF EXISTS _obs_keep")
        conn.execute("DROP TABLE IF EXISTS _var_keep")


    # 4.重建新表：X_HyS_indptr_filtered
    def _rebuild_x_hys_indptr_filtered(self):
        """构建过滤后的 CSR-like indptr 表 ``X_HyS_indptr_filtered``。

        该方法统计 ``X_HyS_data_filtered`` 中每个 ``filter_cell_id`` 的非零元素数量，
        并从 ``obs`` 表出发补齐所有保留下来的细胞。即使某些细胞没有任何非零表达值，
        也会在 indptr 表中保留对应记录。

        最终生成的 ``indptr`` 表示每个细胞的累计结束位置，即 end pointer：

        - 第 0 个细胞的起始位置默认为 0
        - 第 i 个细胞的结束位置为 ``indptr[i]``
        - 第 i 个细胞的起始位置需要由前一个细胞的 ``indptr[i-1]`` 得到

        Returns
        -------
        None
            结果直接写入 ``X_HyS_indptr_filtered`` 表。
        """

        conn = self.conn

        # 1. 先统计 X 表中每个 cell 的非零元素数量
        conn.execute("""
        CREATE OR REPLACE TEMP TABLE cell_nnz_raw AS
        SELECT
            CAST(filter_cell_id AS INTEGER) AS filter_cell_id,
            COUNT(*) AS cnt
        FROM X_HyS_data_filtered
        GROUP BY filter_cell_id
        """)

        # 2. 从 obs 出发，补齐所有保留下来的 cell
        #    如果某个 cell 在 X 里不存在，说明它没有非零值，cnt 补 0
        conn.execute("""
        CREATE OR REPLACE TEMP TABLE cell_nnz AS
        SELECT
            CAST(obs.filter_cell_id AS INTEGER) AS filter_cell_id,
            CAST(COALESCE(cell_nnz_raw.cnt, 0) AS BIGINT) AS cnt
        FROM obs
        LEFT JOIN cell_nnz_raw
          ON obs.filter_cell_id = cell_nnz_raw.filter_cell_id
        WHERE obs.filter_cell_id IS NOT NULL
        ORDER BY obs.filter_cell_id
        """)

        # 3. 生成 prefix sum ；这里 indptr 存的是每个 cell 的结束位置 end_ptr
        conn.execute("""
        CREATE OR REPLACE TABLE X_HyS_indptr_filtered AS
        SELECT
            CAST(filter_cell_id AS INTEGER) AS filter_cell_id,
            CAST(SUM(cnt) OVER (ORDER BY filter_cell_id) AS BIGINT) AS indptr
        FROM cell_nnz
        ORDER BY filter_cell_id
        """)

        conn.execute("DROP TABLE IF EXISTS cell_nnz_raw")
        conn.execute("DROP TABLE IF EXISTS cell_nnz")

