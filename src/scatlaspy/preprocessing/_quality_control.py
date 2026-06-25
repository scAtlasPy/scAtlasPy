import os
from datetime import datetime
from _duckdb import DuckDBPyConnection
from ..data import Atlas
from typing import Any
from typing import Optional
import logging
import math
import gc
from ..io import progress

logger = logging.getLogger('Atlas') # 获取日志记录器
logger.addHandler(logging.NullHandler())


def filter_cells(
        atlas: Atlas,
        min_counts: Optional[int] = None,
        min_genes: Optional[int] = None,
        max_counts: Optional[int] = None,
        max_genes: Optional[int] = None,
        add_data: str = "filter_cells",
        chunk_cells: int = 500_000,   # 分块大小
):
    """根据表达量和检测基因数过滤细胞。

    该函数从表达矩阵分块统计每个细胞的总表达量和检测基因数，再根据阈值在 ``obs`` 表中写入细胞过滤标记。结果保存在 Atlas 数据库中。

    Parameters
    ----------
    atlas
        Atlas 对象。通常需要已经连接到 DuckDB 数据库，并包含该函数读取或写入所需的 ``obs``、``var``、表达矩阵或结果表。

    min_counts
        细胞总表达量下限。只有 ``sum_expr >= min_counts`` 的细胞才会通过该条件。
        为 ``None`` 时不使用该下限条件。

    min_genes
        细胞检测到的非零表达基因数下限。只有
        ``nonzero_genes >= min_genes`` 的细胞才会通过该条件。
        为 ``None`` 时不使用该下限条件。

    max_counts
        细胞总表达量上限。只有 ``sum_expr <= max_counts`` 的细胞才会通过该条件。
        为 ``None`` 时不使用该上限条件。

    max_genes
        细胞检测到的非零表达基因数上限。只有
        ``nonzero_genes <= max_genes`` 的细胞才会通过该条件。
        为 ``None`` 时不使用该上限条件。

    add_data
        写入 ``obs`` 表的布尔过滤列名。默认值为 ``"filter_cells"``。
        如果该列不存在，函数会自动新增该列；
        如果该列已经存在，函数会先将其全部重置为 ``FALSE``，再把通过过滤的细胞
        更新为 ``TRUE``。

    chunk_cells
        按 ``atlas_cell_id`` 范围分块处理时每个 chunk 覆盖的细胞 ID 数量。
        较大的值通常运行更快，但会增加单个 chunk 聚合时的内存占用；
        较小的值更稳，但会增加 SQL 循环次数。

    Returns
    -------
    None
        结果直接写入 Atlas 数据库。
        数据库中的 obs表，会新增 add_data: str = "filter_cells"字段，符合过滤条件则为 true，否则为 false

    Examples
    --------
    保留至少 200 个检测基因且总表达量不低于 500 的细胞::

        sap.pp.filter_cells(atlas, min_genes=200, min_counts=500)

    同时设置上下限，并写入自定义过滤列::

        sap.pp.filter_cells(
            atlas,
            min_counts=500,
            max_counts=50000,
            min_genes=200,
            max_genes=6000,
            add_data="filter_cells",
        )
    """

    start_time = datetime.now()
    conn = atlas.connection

    # 0. DuckDB 参数
    try:
        th = os.cpu_count() or 1
        conn.execute(f"PRAGMA threads={th}")

    except Exception:
        pass


    # 1. 添加 obs 过滤字段
    conn.execute(f"""
        ALTER TABLE obs
        ADD COLUMN IF NOT EXISTS {add_data} BOOLEAN DEFAULT FALSE
    """)

    total_cells = conn.execute("SELECT COUNT(*) FROM obs").fetchone()[0]

    conn.execute(f"""
        UPDATE obs
        SET {add_data} = FALSE
    """)

    # 2. 构建过滤条件
    conds = []
    if min_counts is not None:
        conds.append(f"sum_expr >= {min_counts}")
    if max_counts is not None:
        conds.append(f"sum_expr <= {max_counts}")
    if min_genes is not None:
        conds.append(f"nonzero_genes >= {min_genes}")
    if max_genes is not None:
        conds.append(f"nonzero_genes <= {max_genes}")

    condition = " AND ".join(conds) if conds else "TRUE"

    # 3. 获取 cell_id 范围
    min_cell, max_cell = conn.execute("""
        SELECT
            MIN(atlas_cell_id),
            MAX(atlas_cell_id)
        FROM obs
    """).fetchone()

    if min_cell is None or max_cell is None:
        logger.info("obs 为空，跳过。")
        return

    n_chunks = (max_cell - min_cell + chunk_cells) // chunk_cells

    keep_total = 0

    # 4. 分块聚合 + 分块写回
    pbar = progress(
        range(n_chunks),
        total=n_chunks,
        desc="filter_cells",
        unit="chunk",
    )

    for i in pbar:

        c_start = min_cell + i * chunk_cells
        c_end = min(c_start + chunk_cells - 1, max_cell)

        # 只创建当前 chunk 的临时表，不再创建全量 keep_cells
        conn.execute("DROP TABLE IF EXISTS keep_cells_chunk")

        conn.execute(f"""
            CREATE TEMP TABLE keep_cells_chunk AS
            SELECT atlas_cell_id
            FROM (
                SELECT
                    atlas_cell_id,
                    SUM(data_count) AS sum_expr,
                    COUNT(*) AS nonzero_genes
                FROM X_HyS_data
                WHERE atlas_cell_id BETWEEN {c_start} AND {c_end}
                GROUP BY atlas_cell_id
            )
            WHERE {condition}
        """)

        # 当前 chunk 保留数量
        keep_now = conn.execute("""
            SELECT COUNT(*) FROM keep_cells_chunk
        """).fetchone()[0]

        keep_total += keep_now

        # 只更新当前 chunk 内 TRUE 的 cells
        conn.execute(f"""
            UPDATE obs
            SET {add_data} = TRUE
            WHERE atlas_cell_id IN (
                SELECT atlas_cell_id FROM keep_cells_chunk
            )
        """)

        # 每个 chunk 后立即清理小临时表
        conn.execute("DROP TABLE IF EXISTS keep_cells_chunk")

    # 5. 统计结果
    removed = total_cells - keep_total

    logger.info(f"filter_cells Done, 耗时: {(datetime.now() - start_time).total_seconds():.2f} 秒")
    logger.info(f"保留细胞 = {keep_total} / {total_cells} , ({keep_total / total_cells * 100:.2f}%)")

    # 内存清理
    _cleanup_qc_after_step(
        conn,
        temp_tables=["keep_cells_chunk"],
        checkpoint=False,
        collect=True,
    )


def filter_genes(
        atlas: Atlas,
        min_counts: Optional[int] = None,
        min_cells: Optional[int] = None,
        max_counts: Optional[int] = None,
        max_cells: Optional[int] = None,
        add_data: str = "filter_genes"
) -> None:
    """根据表达量和检出细胞数过滤基因。

    该函数从表达矩阵统计每个基因的总表达量和被检测到的细胞数，再根据阈值在
    ``var`` 表中写入基因过滤标记。结果保存在 Atlas 数据库中。

    不会直接删除基因，而是在 ``var`` 表中新增或更新一个布尔列，用于标记哪些基因通过过滤条件。

    Parameters
    ----------
    atlas
        Atlas 对象。通常需要已经连接到 DuckDB 数据库，并包含该函数读取或写入所需的
        ``var`` 表和 ``X_HyS_data`` 表。

        ``var`` 表需要包含 ``atlas_gene_id`` 字段；
        ``X_HyS_data`` 表需要包含 ``atlas_gene_id`` 和 ``data_count`` 字段。

    min_counts
        基因总表达量下限。只有 ``sum_expr >= min_counts`` 的基因才会通过该条件。
        为 ``None`` 时不使用该下限条件。

    min_cells
        检测到该基因的细胞数下限。只有
        ``nonzero_expr >= min_cells`` 的基因才会通过该条件。
        为 ``None`` 时不使用该下限条件。

    max_counts
        基因总表达量上限。只有 ``sum_expr <= max_counts`` 的基因才会通过该条件。
        为 ``None`` 时不使用该上限条件。

    max_cells
        检测到该基因的细胞数上限。只有
        ``nonzero_expr <= max_cells`` 的基因才会通过该条件。
        为 ``None`` 时不使用该上限条件。

    add_data
        写入 ``var`` 表的布尔过滤列名。默认值为 ``"filter_genes"``。

        如果该列不存在，函数会自动新增该列；
        如果该列已经存在，函数会根据当前过滤条件重新写入 ``TRUE`` 或 ``FALSE``。

    Returns
    -------
    None
        结果直接写入 Atlas 数据库。
        数据库中的 ``var`` 表会新增或更新 ``add_data`` 指定的字段，
        符合过滤条件的基因标记为 ``TRUE``，否则标记为 ``FALSE``。

    Notes
    -----
    该函数只写入过滤标记，不会删除 ``var`` 或 ``X_HyS_data`` 中的原始数据。

    Examples
    --------
    保留至少在 3 个细胞中被检测到的基因::

        sap.pp.filter_genes(
            atlas,
            min_cells=3,
        )

    保留总表达量不低于 10，且至少在 5 个细胞中被检测到的基因::

        sap.pp.filter_genes(
            atlas,
            min_counts=10,
            min_cells=5,
        )

    查看过滤结果统计::

        atlas.query(
            "SELECT filter_genes, COUNT(*) AS n_genes "
            "FROM var GROUP BY filter_genes"
        )

    基于过滤后的基因构建读取索引::

        atlas.build_read_index(
            cell_condition="filter_cells",
            gene_condition="filter_genes",
            use_hvg=True,
            use_data="data_log1p",
        )

    """

    start_time = datetime.now()

    conn = atlas.connection

    # DuckDB 多线程
    try:
        th = os.cpu_count() or 1
        conn.execute(f"PRAGMA threads={th}")
    except Exception:
        pass

    #  统计基因数量
    n_genes = conn.execute("""
        SELECT COUNT(*) FROM var
    """).fetchone()[0]

    # 添加过滤字段
    conn.execute(f"""
        ALTER TABLE var
        ADD COLUMN IF NOT EXISTS {add_data} BOOLEAN DEFAULT FALSE
    """)

    # 构建 SQL 条件
    conds = []

    if min_counts is not None:
        conds.append(f"COALESCE(s.sum_expr, 0) >= {min_counts}")

    if max_counts is not None:
        conds.append(f"COALESCE(s.sum_expr, 0) <= {max_counts}")

    if min_cells is not None:
        conds.append(f"COALESCE(s.nonzero_expr, 0) >= {min_cells}")

    if max_cells is not None:
        conds.append(f"COALESCE(s.nonzero_expr, 0) <= {max_cells}")

    condition = " AND ".join(conds) if conds else "TRUE"

    # 聚合 X_HyS_data

    conn.execute("DROP TABLE IF EXISTS gene_filter_stats_tmp")

    # 只生成 gene 级小临时表；结果规模 ≈ 基因数，不是 nnz 数
    conn.execute("""
        CREATE TEMP TABLE gene_filter_stats_tmp AS
        SELECT
            atlas_gene_id,
            SUM(data_count) AS sum_expr,
            COUNT(*) AS nonzero_expr
        FROM X_HyS_data
        GROUP BY atlas_gene_id
    """)

    # 纯 SQL 写回 var

    conn.execute(f"""
        UPDATE var
        SET {add_data} =
            CASE
                WHEN {condition}
                THEN TRUE
                ELSE FALSE
            END
        FROM gene_filter_stats_tmp AS s
        WHERE var.atlas_gene_id = s.atlas_gene_id
    """)

    # 处理完全零表达基因；CSR 中完全没出现的 gene，默认不通过过滤
    conn.execute(f"""
        UPDATE var
        SET {add_data} = FALSE
        WHERE atlas_gene_id NOT IN (
            SELECT atlas_gene_id FROM gene_filter_stats_tmp
        )
    """)

    # 统计结果
    keep_count = conn.execute(f"""
        SELECT COUNT(*) FROM var
        WHERE {add_data} = TRUE
    """).fetchone()[0]

    conn.execute("DROP TABLE IF EXISTS gene_filter_stats_tmp")

    logger.info(f"filter_genes Done, 耗时: {(datetime.now() - start_time).total_seconds():.2f} 秒")
    logger.info(f"保留基因 = {keep_count} / {n_genes} , ({keep_count / n_genes * 100:.2f}%)")

    # 内存清理
    _cleanup_qc_after_step(
        conn,
        temp_tables=["gene_filter_stats_tmp"],
        checkpoint=False,
        collect=True,
    )


def calculate_cell_total_counts(
        atlas: Atlas,
        add_data: str = "cell_total_counts",
        chunk_cells: int = 1_000_000,
) -> None:
    """计算每个细胞的总 UMI counts。

    该函数用于在 Atlas 数据库中计算每个细胞的总表达量，即每个细胞在
    ``X_HyS_data.data_count`` 字段上的 counts 总和，并将结果写入 ``obs`` 表。

    计算结果通常用于单细胞数据的质量控制，例如检查每个细胞的测序深度、
    过滤低质量细胞、辅助归一化检查，以及绘制 QC 分布图。

    函数采用按 ``atlas_cell_id`` 范围分块的方式处理表达矩阵。每个 chunk
    只聚合当前细胞范围内的表达记录，并将结果写回 ``obs`` 表，避免一次性
    对全量 ``X_HyS_data`` 做聚合导致内存压力过大。

    Parameters
    ----------
    atlas
        Atlas 对象。要求对象已经连接到 DuckDB 数据库，并且数据库中至少包含
        ``obs`` 表和 ``X_HyS_data`` 表。

        ``obs`` 表需要包含 ``atlas_cell_id`` 字段；
        ``X_HyS_data`` 表需要包含 ``atlas_cell_id`` 和 ``data_count`` 字段。

    add_data
        写入 ``obs`` 表的结果列名。默认值为 ``"cell_total_counts"``。

        如果该列不存在，函数会自动新增该列；
        如果该列已经存在，函数会先将该列全部重置为 ``0``，再把每个细胞的
        总 counts 写回该列。

    chunk_cells
        按 ``atlas_cell_id`` 范围分块处理时每个 chunk 覆盖的细胞 ID 数量。
        默认值为 ``1_000_000``。

        较大的值通常可以减少 SQL 循环次数、提高运行速度，但会增加单个 chunk
        聚合时的内存占用；较小的值更稳，但运行时间可能更长。

    Returns
    -------
    None
        结果直接写入 ``obs`` 表中的 ``add_data`` 列（每个细胞的总 UMI 计数），不返回对象。

    Notes
    -----
    该函数不会修改表达矩阵本身，只会在 ``obs`` 表中新增或更新一个细胞级 QC 指标列。

    对于在 ``X_HyS_data`` 中没有任何表达记录的细胞，其 ``add_data`` 值会保持为
    ``0``。因此，该函数可以安全处理完全没有非零表达记录的细胞。

    Examples
    --------
    使用默认列名写入每个细胞的总 UMI counts::

        sap.pp.calculate_cell_total_counts(atlas)

    调整分块大小以降低内存压力::

        sap.pp.calculate_cell_total_counts(
            atlas,
            add_data="cell_total_counts",
            chunk_cells=200_000,
        )

    查看结果统计::

        atlas.query(
            "SELECT "
            "MIN(cell_total_counts) AS min_counts, "
            "MAX(cell_total_counts) AS max_counts, "
            "AVG(cell_total_counts) AS mean_counts "
            "FROM obs"
        )

    绘制或过滤前，可以先检查 ``obs`` 表中新生成的字段::
        atlas.head("obs")
    """

    start_time = datetime.now()

    conn = atlas.connection

    # DuckDB 性能参数
    try:
        th = os.cpu_count() or 1
        conn.execute(f"PRAGMA threads={th}")
    except Exception:
        pass

    # 确保 obs 有目标列
    conn.execute(f"""
        ALTER TABLE obs
        ADD COLUMN IF NOT EXISTS {add_data} DOUBLE
    """)

    conn.execute(f"""
        UPDATE obs
        SET {add_data} = 0
    """)

    # 获取 cell_id 范围
    min_cell, max_cell = conn.execute("""
        SELECT
            MIN(atlas_cell_id),
            MAX(atlas_cell_id)
        FROM obs
    """).fetchone()

    if min_cell is None or max_cell is None:
        logger.info("obs 为空，跳过。")
        return

    n_chunks = math.ceil((max_cell - min_cell + 1) / chunk_cells)

    total_updated_cells = 0

    # 分块聚合 + 分块写回
    pbar = progress(
        range(n_chunks),
        total=n_chunks,
        desc="calculate_cell_total_counts",
        unit="chunk",
    )

    for i in pbar:

        c_start = min_cell + i * chunk_cells
        c_end = min(c_start + chunk_cells - 1, max_cell)

        # 只创建当前 chunk 的小临时表
        conn.execute("DROP TABLE IF EXISTS cell_total_counts_chunk")

        conn.execute(f"""
            CREATE TEMP TABLE cell_total_counts_chunk AS
            SELECT
                atlas_cell_id,
                SUM(data_count) AS total_counts
            FROM X_HyS_data
            WHERE atlas_cell_id BETWEEN {c_start} AND {c_end}
            GROUP BY atlas_cell_id
        """)

        n_now = conn.execute("""
            SELECT COUNT(*) FROM cell_total_counts_chunk
        """).fetchone()[0]

        total_updated_cells += n_now

        # 只写回当前 chunk 有表达记录的 cell
        conn.execute(f"""
            UPDATE obs
            SET {add_data} = t.total_counts
            FROM cell_total_counts_chunk t
            WHERE obs.atlas_cell_id = t.atlas_cell_id
        """)

        # 每个 chunk 后立即清理
        conn.execute("DROP TABLE IF EXISTS cell_total_counts_chunk")

    logger.info(f"calculate_cell_total_counts Done, 耗时: {(datetime.now() - start_time).total_seconds():.2f} 秒")

    # 内存清理
    _cleanup_qc_after_step(
        conn,
        temp_tables=["cell_total_counts_chunk"],
        checkpoint=False,
        collect=True,
    )


def calculate_gene_total_counts(
                    atlas: 'Atlas',
                    add_gene_total_counts: str = "gene_total_counts",
                    add_gene_mean_counts: str = "gene_mean_counts",
                    ) -> None:
    """计算每个基因的总 counts 和平均 counts。

    该函数用于在 Atlas 数据库中计算基因级 QC 统计指标，并将结果写入
    ``var`` 表。函数会从 ``X_HyS_data`` 表中按 ``atlas_gene_id`` 聚合
    ``data_count``，得到每个基因在所有细胞中的总表达量；同时根据 ``obs`` 表
    中的细胞总数计算每个基因的平均表达量。

    计算结果通常用于基因质量控制、基因过滤、高表达基因检查、数据概览和
    后续可视化分析。

    Parameters
    ----------
    atlas
        Atlas 对象。通常需要已经连接到 DuckDB 数据库，并且数据库中至少包含
        ``obs`` 表、``var`` 表和 ``X_HyS_data`` 表。

        ``obs`` 表用于统计细胞总数；
        ``var`` 表需要包含 ``atlas_gene_id`` 字段；
        ``X_HyS_data`` 表需要包含 ``atlas_gene_id`` 和 ``data_count`` 字段。

    add_gene_total_counts
        写入 ``var`` 表的基因总表达量列名。默认值为
        ``"gene_total_counts"``。

        如果该列不存在，函数会自动新增该列；
        如果该列已经存在，函数会重新写入当前计算得到的基因总 counts。

    add_gene_mean_counts
        写入 ``var`` 表的基因平均表达量列名。默认值为
        ``"gene_mean_counts"``。

        该值计算方式为：

        ``gene_mean_counts = gene_total_counts / obs 表中的细胞总数``

        如果该列不存在，函数会自动新增该列；
        如果该列已经存在，函数会重新写入当前计算得到的基因平均 counts。

    Returns
    -------
    None
        结果直接写入 Atlas 数据库中的 ``var`` 表，不返回对象。

        数据库中的 ``var`` 表会新增或更新两个字段：

        1. ``add_gene_total_counts``：每个基因的总 counts；
        2. ``add_gene_mean_counts``：每个基因的平均 counts。

    Notes
    -----
    该函数不会修改表达矩阵本身，只会在 ``var`` 表中新增或更新基因级统计列。

    Examples
    --------
    使用默认列名计算基因总 counts 和平均 counts::

        sap.pp.calculate_gene_total_counts(atlas)
        atlas.head("var")

    查看基因统计结果::

        atlas.query(
            "SELECT atlas_gene_id, gene_total_counts, gene_mean_counts "
            "FROM var "
            "ORDER BY gene_total_counts DESC "
            "LIMIT 10"
        )

    检查基因总 counts 的整体范围::

        atlas.query(
            "SELECT "
            "MIN(gene_total_counts) AS min_counts, "
            "MAX(gene_total_counts) AS max_counts, "
            "AVG(gene_total_counts) AS mean_counts "
            "FROM var"
        )
    """

    start_time = datetime.now()

    conn = atlas.connection

    # DuckDB 并行设置
    try:
        th = os.cpu_count() or 1
        conn.execute(f"PRAGMA threads={th}")

    except:
        pass

    # 确保 var 表有目标列
    cols = [r[1] for r in conn.execute("PRAGMA table_info(var)").fetchall()]

    if add_gene_total_counts not in cols:
        conn.execute(f"ALTER TABLE var ADD COLUMN {add_gene_total_counts} DOUBLE DEFAULT 0")
    if add_gene_mean_counts not in cols:
        conn.execute(f"ALTER TABLE var ADD COLUMN {add_gene_mean_counts} DOUBLE DEFAULT 0")

    # 细胞总数
    total_cells = conn.execute("SELECT COUNT(*) FROM obs").fetchone()[0]

    conn.execute("DROP TABLE IF EXISTS gene_stats_tmp")
    conn.execute("""
        CREATE TEMP TABLE gene_stats_tmp AS
        SELECT
            atlas_gene_id,
            SUM(data_count) AS total_counts
        FROM X_HyS_data
        GROUP BY atlas_gene_id
    """)

    conn.execute(f"""
        UPDATE var
        SET
            {add_gene_total_counts} = s.total_counts,
            {add_gene_mean_counts} = s.total_counts / {total_cells}
        FROM gene_stats_tmp AS s
        WHERE var.atlas_gene_id = s.atlas_gene_id
    """)

    #  零表达基因补零
    conn.execute(f"""
        UPDATE var
        SET
            {add_gene_total_counts} = 0,
            {add_gene_mean_counts} = 0
        WHERE atlas_gene_id NOT IN (SELECT atlas_gene_id FROM gene_stats_tmp)
    """)

    # 内存清理
    conn.execute("DROP TABLE IF EXISTS gene_stats_tmp")

    _cleanup_qc_after_step(
        conn,
        temp_tables=["gene_stats_tmp"],
        checkpoint=False,
        collect=True,
    )

    logger.info(f"calculate_gene_total_counts Done, 耗时: {(datetime.now() - start_time).total_seconds():.2f} 秒")


def calculate_qc_metrics(
    atlas: Atlas,
    qc_vars: dict[str, Any] | None=None,
    chunk_cells: int=100_000
):
    """计算常用单细胞 QC 指标。

    该函数用于在 Atlas 数据库中计算细胞级和基因级质量控制指标，直接把计算结果写入
    Atlas 数据库中的 ``obs`` 表和 ``var`` 表。

    函数主要计算两类指标：

    1. cell-wise QC 指标，写入 ``obs`` 表；
    2. gene-wise QC 指标，写入 ``var`` 表。

    对于每个细胞，函数会统计该细胞的总 counts、检测到的非零基因数，
    以及指定 QC 基因集合的 counts 和比例。例如默认会计算线粒体基因
    和核糖体基因相关指标。

    对于每个基因，函数会统计该基因在所有细胞中的总 counts，以及有多少
    个细胞检测到该基因的非零表达。

    函数采用按 ``atlas_cell_id`` 范围分块的方式计算 cell-wise QC 指标，
    避免一次性聚合所有细胞造成较大的内存压力。gene-wise QC 指标的结果规模
    约等于基因数，因此在循环外一次性计算。

    Parameters
    ----------
    atlas
        Atlas 对象。要求对象已经连接到 DuckDB 数据库，并且数据库中至少包含
        ``obs`` 表、``var`` 表和 ``X_HyS_data`` 表。

        ``obs`` 表需要包含 ``atlas_cell_id`` 字段；
        ``var`` 表需要包含 ``atlas_gene_id`` 和 ``atlas_gene_name`` 字段；
        ``X_HyS_data`` 表需要包含 ``atlas_cell_id``、``atlas_gene_id`` 和
        ``data_count`` 字段。

    qc_vars
        QC 基因集合定义。为 ``None`` 时使用默认设置::

            {
                "mt": "MT-",
                "ribo": "^(RPS|RPL)"
            }

        字典的 key 会作为 QC 指标名称，例如 ``"mt"`` 会生成
        ``var.mt``、``obs.total_counts_mt`` 和 ``obs.pct_counts_mt``；
        ``"ribo"`` 会生成 ``var.ribo``、``obs.total_counts_ribo`` 和
        ``obs.pct_counts_ribo``。

        字典的 value 是基因名匹配模式：

        - 如果字符串以 ``"^"`` 开头，则使用正则表达式匹配 ``atlas_gene_name``；
        - 如果字符串不以 ``"^"`` 开头，则按基因名前缀匹配。

        例如 ``"MT-"`` 表示匹配以 ``MT-`` 开头的基因；
        ``"^(RPS|RPL)"`` 表示匹配以 ``RPS`` 或 ``RPL`` 开头的基因。

    chunk_cells
        按 ``atlas_cell_id`` 范围分块处理时每个 chunk 覆盖的细胞 ID 数量。
        默认值为 ``100_000``。

        较大的值通常可以减少 SQL 循环次数、提高运行速度，但会增加单个 chunk
        聚合时的内存占用；较小的值更稳，但运行时间可能更长。

    Returns
    -------
    None
        结果直接写入 Atlas 数据库，不返回对象。

        ``obs`` 表会新增或更新以下字段：

        - ``cell_total_counts``：每个细胞的总 counts；
        - ``n_genes_by_counts``：每个细胞检测到的非零表达基因数；
        - ``total_counts_{qc_key}``：该 QC 基因集合在每个细胞中的 counts 总和；
        - ``pct_counts_{qc_key}``：该 QC 基因集合 counts 占细胞总 counts 的比例。

        ``var`` 表会新增或更新以下字段：

        - ``{qc_key}``：该基因是否属于指定 QC 基因集合；
        - ``gene_total_counts``：该基因在所有细胞中的 counts 总和；
        - ``n_cells_by_counts``：检测到该基因非零表达的细胞数量。

    Notes
    -----
    该函数不会删除细胞或基因，也不会修改表达矩阵本身，只会在 ``obs`` 和
    ``var`` 表中新增或更新 QC 指标列。

    Examples
    --------
    使用默认 QC 基因集合计算指标::

        sap.pp.calculate_qc_metrics(atlas)

    默认会计算线粒体和核糖体相关指标::

        qc_vars = {
            "mt": "MT-",
            "ribo": "^(RPS|RPL)",
        }
        sap.pp.calculate_qc_metrics(atlas, qc_vars=qc_vars)

    自定义 QC 基因集合，例如计算血红蛋白基因比例::

        sap.pp.calculate_qc_metrics(
            atlas,
            qc_vars={
                "mt": "MT-",
                "ribo": "^(RPS|RPL)",
                "hb": "^(HBA|HBB)",
            },
        )

    查看细胞级 QC 指标::

        atlas.query(
            "SELECT cell_total_counts, n_genes_by_counts, "
            "pct_counts_mt, pct_counts_ribo "
            "FROM obs LIMIT 5"
        )

    查看基因级 QC 指标::

        atlas.query(
            "SELECT atlas_gene_name, mt, ribo, gene_total_counts, n_cells_by_counts "
            "FROM var LIMIT 5"
        )

    """

    start_time = datetime.now()
    conn = atlas.connection

    if qc_vars is None:
        qc_vars = {
            "mt": "MT-",
            "ribo": "^(RPS|RPL)"
        }

    try:
        th = os.cpu_count() or 8
        conn.execute(f"PRAGMA threads={th}")
    except Exception:
        pass

    #  var 打 qc 标记
    for qc_key, pattern in qc_vars.items():
        conn.execute(f"""
            ALTER TABLE var
            ADD COLUMN IF NOT EXISTS {qc_key} BOOLEAN
        """)

        if pattern.startswith("^"):
            conn.execute(f"""
                UPDATE var
                SET {qc_key} = regexp_matches(atlas_gene_name, '{pattern}', 'i')
            """)
        else:
            conn.execute(f"""
                UPDATE var
                SET {qc_key} =
                    UPPER(atlas_gene_name) LIKE '{pattern.upper()}%'
            """)

    # 初始化 obs 列
    conn.execute("""
        ALTER TABLE obs
        ADD COLUMN IF NOT EXISTS cell_total_counts REAL
    """)
    conn.execute("""
        ALTER TABLE obs
        ADD COLUMN IF NOT EXISTS n_genes_by_counts INTEGER
    """)

    for qc_key in qc_vars.keys():
        conn.execute(f"""
            ALTER TABLE obs
            ADD COLUMN IF NOT EXISTS total_counts_{qc_key} REAL
        """)
        conn.execute(f"""
            ALTER TABLE obs
            ADD COLUMN IF NOT EXISTS pct_counts_{qc_key} REAL
        """)

    # 先初始化 obs，避免完全空 cell 保留旧值
    conn.execute("""
        UPDATE obs
        SET
            cell_total_counts = 0,
            n_genes_by_counts = 0
    """)

    for qc_key in qc_vars.keys():
        conn.execute(f"""
            UPDATE obs
            SET
                total_counts_{qc_key} = 0,
                pct_counts_{qc_key} = 0
        """)

    # 计算 cell_id 范围
    min_cell, max_cell = conn.execute("""
        SELECT MIN(atlas_cell_id), MAX(atlas_cell_id)
        FROM obs
    """).fetchone()

    if min_cell is None or max_cell is None:
        logger.info("obs 为空，跳过。")
        return

    n_chunks = math.ceil((max_cell - min_cell + 1) / chunk_cells)

    # 分块处理
    pbar = progress(
        range(n_chunks),
        total=n_chunks,
        desc="calculate_qc_metrics",
        unit="chunk",
    )

    for i in pbar:

        c_start = min_cell + i * chunk_cells
        c_end = min(c_start + chunk_cells - 1, max_cell)

        qc_sum_expr = []
        for qc_key in qc_vars.keys():
            qc_sum_expr.append(
                f"SUM(CASE WHEN v.{qc_key} THEN x.data_count ELSE 0 END)"
                f" AS total_counts_{qc_key}"
            )
        qc_sum_sql = ",\n".join(qc_sum_expr)

        conn.execute("DROP TABLE IF EXISTS _cell_chunk")

        conn.execute(f"""
            CREATE TEMP TABLE _cell_chunk AS
            SELECT
                x.atlas_cell_id,
                SUM(x.data_count) AS cell_total_counts,
                COUNT(*)    AS n_genes_by_counts,
                {qc_sum_sql}
            FROM X_HyS_data x
            JOIN var v
              ON x.atlas_gene_id = v.atlas_gene_id
            WHERE x.atlas_cell_id BETWEEN {c_start} AND {c_end}
            GROUP BY x.atlas_cell_id
        """)

        set_expr = [
            "cell_total_counts = c.cell_total_counts",
            "n_genes_by_counts = c.n_genes_by_counts"
        ]

        for qc_key in qc_vars.keys():
            set_expr.append(
                f"total_counts_{qc_key} = c.total_counts_{qc_key}"
            )
            set_expr.append(
                f"""
                pct_counts_{qc_key} =
                CASE WHEN c.cell_total_counts > 0
                THEN 100.0 * c.total_counts_{qc_key} / c.cell_total_counts
                ELSE 0 END
                """
            )

        conn.execute(f"""
            UPDATE obs
            SET {",".join(set_expr)}
            FROM _cell_chunk c
            WHERE obs.atlas_cell_id = c.atlas_cell_id
        """)

        conn.execute("DROP TABLE IF EXISTS _cell_chunk")

    # gene-wise QC：循环外一次性计算

    conn.execute("""
        ALTER TABLE var
        ADD COLUMN IF NOT EXISTS gene_total_counts REAL
    """)
    conn.execute("""
        ALTER TABLE var
        ADD COLUMN IF NOT EXISTS n_cells_by_counts INTEGER
    """)

    # gene 级临时表很小，不需要分块
    conn.execute("DROP TABLE IF EXISTS _gene_qc")
    conn.execute("""
        CREATE TEMP TABLE _gene_qc AS
        SELECT
            atlas_gene_id,
            SUM(data_count) AS gene_total_counts,
            COUNT(*)  AS n_cells_by_counts
        FROM X_HyS_data
        GROUP BY atlas_gene_id
    """)

    # 先把 var 置 0，处理完全零表达基因
    conn.execute("""
        UPDATE var
        SET
            gene_total_counts = 0,
            n_cells_by_counts = 0
    """)

    conn.execute("""
        UPDATE var
        SET
            gene_total_counts = g.gene_total_counts,
            n_cells_by_counts = g.n_cells_by_counts
        FROM _gene_qc g
        WHERE var.atlas_gene_id = g.atlas_gene_id
    """)

    conn.execute("DROP TABLE IF EXISTS _gene_qc")

    # 内存清理
    _cleanup_qc_after_step(
        conn,
        temp_tables=["_cell_chunk", "_gene_qc"],
        checkpoint=False,
        collect=True,
    )

    logger.info(f"calculate_qc_metrics Done, 耗时: {(datetime.now() - start_time).total_seconds():.2f} 秒")


def _cleanup_qc_after_step(
        conn: DuckDBPyConnection,
        temp_tables: list[str]=None,
        checkpoint: bool = False,
        collect: bool = True,
):
    """清理当前步骤产生的临时资源。

    该内部函数属于质量控制模块，用于支撑同一模块中的公共 API。

    在数据库层面计算细胞/基因 QC 指标，并写回过滤标记。

    它通常不会作为用户入口直接调用；直接调用时需要保证输入对象、数据库连接和相关临时表已经由上游步骤准备好。

    Parameters
    ----------
    conn
        DuckDB 数据库连接。

    temp_tables
        需要清理的临时表名称列表。

    checkpoint
        清理后是否执行 DuckDB ``CHECKPOINT``。

    collect
        清理后是否触发 Python 垃圾回收。

    Notes
    -----
    这是内部 helper；除非需要扩展 scAtlasPy 内部流程，一般不建议在用户代码中直接调用。
    """

    if temp_tables is None:
        temp_tables = []

    for t in temp_tables:
        try:
            conn.execute(f"DROP TABLE IF EXISTS {t}")
        except Exception:
            pass

    if checkpoint:
        try:
            conn.execute("CHECKPOINT")
        except Exception:
            pass

    if collect:
        try:
            gc.collect()
        except Exception:
            pass