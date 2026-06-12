import duckdb
from os import PathLike, fspath
from tqdm import tqdm
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
        用于构建过滤表达矩阵的表达值列或数据层名称。

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
            use_data="data",
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
        """初始化对象。

        该内部函数属于过滤索引模块，用于支撑同一模块中的公共 API。

        根据 ``obs``、``var`` 中的过滤标记重建连续 cell/gene 索引，并生成过滤后的 HyS 表。

        它通常不会作为用户入口直接调用；直接调用时需要保证输入对象、数据库连接和相关临时表已经由上游步骤准备好。

        Parameters
        ----------
        file_path
            输入文件路径或 Atlas ``.sasql`` 数据库文件路径。

        cell_condition
            ``obs`` 中用于筛选细胞的布尔列名或条件。

        gene_condition
            ``var`` 中用于筛选基因的布尔列名或条件。

        use_hvg
            是否只处理高变基因。

        use_data
            从 ``X_HyS_data`` 中读取的表达字段。

        Notes
        -----
        这是内部 helper；除非需要扩展 scAtlasPy 内部流程，一般不建议在用户代码中直接调用。
        """
        self.file_path = fspath(file_path)       # sasql 文件绝对路径
        self.producer_num = 10           # minibatch 流式读取的线程数
        self.fetch_size = 1_0000_0000    # minibatch 流式读取的size
        self.chunk_size = 2_0000_0000    # 每次处理的数据量

        self.cell_condition = cell_condition # cell 过滤条件 filter_cells 表示只选 filter_cells = True 的cell
        self.gene_condition = gene_condition # gene 过滤条件 filter_genes 表示只选 filter_genes = True 的gene
        self.use_hvg = use_hvg               # 是否 使用hvg基因
        self.use_data = use_data       # 选择什么数据进行处理

        self.conn = duckdb.connect(file_path)
        self.conn.execute("PRAGMA preserve_insertion_order=true")
        #  false 不强制保留插入时的输入顺序，允许 DuckDB 为了性能重新组织执行和写入顺序。
        #  true  尽量保留 INSERT / COPY / SELECT 写入时的输入顺序


    # 外部入口
    def run(self):

        """执行 ``run`` 的核心功能。

        根据 ``obs``、``var`` 中的过滤标记重建连续 cell/gene 索引，并生成过滤后的 HyS 表。

        函数会直接读取或写入 Atlas 数据库中的相关表，并尽量通过 SQL、分块读取或流式计算减少内存占用。

        整体用法和 Scanpy 中相近的 ``sap.run`` 风格 API 类似，但结果保存在 Atlas 数据库表中，便于后续步骤复用。

        当前实现中会访问或生成的关键表包括：``obs``、``var``。

        Examples
        --------
        调用该函数：::

            sap.run(...)
        """

        start = datetime.now()

        self._rebuild_obs_filter_id()   # 重排 obs： 过滤细胞 + 生成 filter_cell_id
        self._rebuild_var_filter_id()   # 重排 var： 过滤基因 + 选择HVG基因 + 生成 filter_gene_id

        self._rebuild_x_hys_data_filtered()

        self._rebuild_x_hys_indptr_filtered()

        self.conn.close()

        print(f"build_read_index Done, 耗时: {(datetime.now() - start).total_seconds():.2f} 秒")


    # 1.重排 obs： 过滤细胞 + 生成 filter_cell_id
    def _rebuild_obs_filter_id(self):

        """执行 ``_rebuild_obs_filter_id`` 的核心功能。

        该内部函数属于过滤索引模块，用于支撑同一模块中的公共 API。

        根据 ``obs``、``var`` 中的过滤标记重建连续 cell/gene 索引，并生成过滤后的 HyS 表。

        它通常不会作为用户入口直接调用；直接调用时需要保证输入对象、数据库连接和相关临时表已经由上游步骤准备好。

        当前实现中会访问或生成的关键表包括：``obs``。

        Notes
        -----
        这是内部 helper；除非需要扩展 scAtlasPy 内部流程，一般不建议在用户代码中直接调用。
        """

        # 删除旧列
        self.conn.execute(""" ALTER TABLE obs DROP COLUMN IF EXISTS filter_cell_id """)

        # 新增列
        self.conn.execute(""" ALTER TABLE obs ADD COLUMN filter_cell_id INTEGER """)

        # 如果 self.cell_condition is None，则不过滤 cell
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

        """执行 ``_rebuild_var_filter_id`` 的核心功能。

        该内部函数属于过滤索引模块，用于支撑同一模块中的公共 API。

        根据 ``obs``、``var`` 中的过滤标记重建连续 cell/gene 索引，并生成过滤后的 HyS 表。

        它通常不会作为用户入口直接调用；直接调用时需要保证输入对象、数据库连接和相关临时表已经由上游步骤准备好。

        当前实现中会访问或生成的关键表包括：``var``。

        Notes
        -----
        这是内部 helper；除非需要扩展 scAtlasPy 内部流程，一般不建议在用户代码中直接调用。
        """

        # 删除旧列 + 新增列
        self.conn.execute(""" ALTER TABLE var DROP COLUMN IF EXISTS filter_gene_id """)

        self.conn.execute(""" ALTER TABLE var ADD COLUMN filter_gene_id USMALLINT """)

        # 如果 self.gene_condition is None，则不过滤 gene
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

        """执行 ``_rebuild_x_hys_data_filtered`` 的核心功能。

        该内部函数属于过滤索引模块，用于支撑同一模块中的公共 API。

        根据 ``obs``、``var`` 中的过滤标记重建连续 cell/gene 索引，并生成过滤后的 HyS 表。

        它通常不会作为用户入口直接调用；直接调用时需要保证输入对象、数据库连接和相关临时表已经由上游步骤准备好。

        当前实现中会访问或生成的关键表包括：``X_HyS_data``、``X_HyS_data_filtered``、``obs``、``var``。

        Notes
        -----
        这是内部 helper；除非需要扩展 scAtlasPy 内部流程，一般不建议在用户代码中直接调用。
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

        pbar = tqdm(
            total=total_rows,
            unit="rows",
            desc="Processing",
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
        """执行 ``_rebuild_x_hys_indptr_filtered`` 的核心功能。

        该内部函数属于过滤索引模块，用于支撑同一模块中的公共 API。

        根据 ``obs``、``var`` 中的过滤标记重建连续 cell/gene 索引，并生成过滤后的 HyS 表。

        它通常不会作为用户入口直接调用；直接调用时需要保证输入对象、数据库连接和相关临时表已经由上游步骤准备好。

        当前实现中会访问或生成的关键表包括：``X_HyS_data_filtered``、``X_HyS_indptr_filtered``、``obs``。

        Notes
        -----
        这是内部 helper；除非需要扩展 scAtlasPy 内部流程，一般不建议在用户代码中直接调用。
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

