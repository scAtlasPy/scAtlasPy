import os
from datetime import datetime
from _duckdb import DuckDBPyConnection
from ..data import Atlas
from typing import Any
from typing import Optional
import logging
import math
import gc
from tqdm import tqdm


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

# 获取日志记录器
logger = logging.getLogger('Atlas')
logger.addHandler(logging.NullHandler())

''' 过滤细胞 '''
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

    该函数从表达矩阵分块统计每个细胞的总表达量和检测基因数，再根据阈值在 ``obs`` 中写入细胞过滤标记。它类似 Scanpy 的 ``sc.pp.filter_cells``，但结果保存在 Atlas 数据库中。

    Parameters
    ----------
    atlas
        Atlas 对象。通常需要已经连接到 DuckDB 数据库，并包含该函数读取或写入所需的 ``obs``、``var``、表达矩阵或结果表。
    min_counts
        总表达量下限。为 ``None`` 时不使用该阈值。
    min_genes
        检测到的基因数下限。为 ``None`` 时不使用该阈值。
    max_counts
        总表达量上限。为 ``None`` 时不使用该阈值。
    max_genes
        检测到的基因数上限。为 ``None`` 时不使用该阈值。
    add_data
        写入数据库的新表达矩阵表名或结果列名。
    chunk_cells
        按细胞分块处理时每个 chunk 的细胞数量。

    Returns
    -------
    None
        结果直接写入 Atlas 数据库或当前图形窗口。

    Examples
    --------
    保留至少 200 个检测基因且总表达量不低于 500 的细胞::

        sap.pp.filter_cells(atlas, min_genes=200, min_counts=500)
        atlas.build_read_index(cell_condition="filter_cells")

    同时设置上下限，并写入自定义过滤列::

        sap.pp.filter_cells(
            atlas,
            min_counts=500,
            max_counts=50000,
            min_genes=200,
            max_genes=6000,
            add_data="filter_cells_qc",
        )

    和基因过滤一起重建读取索引::

        sap.pp.filter_genes(atlas, min_cells=3)
        atlas.build_read_index(
            cell_condition="filter_cells_qc",
            gene_condition="filter_genes",
        )"""

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
    pbar = tqdm(
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
                    SUM(data) AS sum_expr,
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

    print(f"filter_cells Done, 耗时: {(datetime.now() - start_time).total_seconds():.2f} 秒")
    print(f"保留细胞 = {keep_total} / {total_cells} , ({keep_total / total_cells * 100:.2f}%)")

    # 内存清理
    _cleanup_qc_after_step(
        conn,
        temp_tables=["keep_cells_chunk"],
        checkpoint=False,
        collect=True,
    )


# 运行结果
# obs 表  新增字段
#     字段	                    含义
#  filter_cells	       该 cell 是否符合过滤条件 true


''' 过滤基因 '''
def filter_genes(
        atlas: Atlas,
        min_counts: Optional[int] = None,
        min_cells: Optional[int] = None,
        max_counts: Optional[int] = None,
        max_cells: Optional[int] = None,
        add_data: str = "filter_genes"
) -> None:
    """根据表达量和检出细胞数过滤基因。

    该函数统计每个基因的总表达量和被多少细胞检测到，并根据阈值在 ``var`` 中写入基因过滤标记。它类似 Scanpy 的 ``sc.pp.filter_genes``，适合在 PCA 和高变基因筛选前移除极低表达基因。

    Parameters
    ----------
    atlas
        Atlas 对象。通常需要已经连接到 DuckDB 数据库，并包含该函数读取或写入所需的 ``obs``、``var``、表达矩阵或结果表。
    min_counts
        总表达量下限。为 ``None`` 时不使用该阈值。
    min_cells
        检测到该基因的细胞数下限。为 ``None`` 时不使用该阈值。
    max_counts
        总表达量上限。为 ``None`` 时不使用该阈值。
    max_cells
        检测到该基因的细胞数上限。为 ``None`` 时不使用该阈值。
    add_data
        写入数据库的新表达矩阵表名或结果列名。

    Returns
    -------
    None
        结果直接写入 Atlas 数据库或当前图形窗口。

    Examples
    --------
    保留至少在 3 个细胞中被检测到的基因::

        sap.pp.filter_genes(atlas, min_cells=3)

    设置表达量和细胞数范围::

        sap.pp.filter_genes(
            atlas,
            min_counts=10,
            min_cells=5,
            max_cells=90000,
            add_data="filter_genes_qc",
        )

    过滤后重建读取索引::

        atlas.build_read_index(gene_condition="filter_genes_qc")"""

    start_time = datetime.now()

    conn = atlas.connection

    # 0. DuckDB 多线程
    try:
        th = os.cpu_count() or 1
        conn.execute(f"PRAGMA threads={th}")
    except Exception:
        pass

    # 1. 统计基因数量
    n_genes = conn.execute("""
        SELECT COUNT(*) FROM var
    """).fetchone()[0]

    # 2. 添加过滤字段
    conn.execute(f"""
        ALTER TABLE var
        ADD COLUMN IF NOT EXISTS {add_data} BOOLEAN DEFAULT FALSE
    """)

    # 3. 构建 SQL 条件
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

    # 4. 聚合 X_HyS_data

    conn.execute("DROP TABLE IF EXISTS gene_filter_stats_tmp")

    # 只生成 gene 级小临时表；结果规模 ≈ 基因数，不是 nnz 数
    conn.execute("""
        CREATE TEMP TABLE gene_filter_stats_tmp AS
        SELECT
            atlas_gene_id,
            SUM(data) AS sum_expr,
            COUNT(*) AS nonzero_expr
        FROM X_HyS_data
        GROUP BY atlas_gene_id
    """)

    # 5. 纯 SQL 写回 var

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

    # 6. 处理完全零表达基因；CSR 中完全没出现的 gene，默认不通过过滤
    conn.execute(f"""
        UPDATE var
        SET {add_data} = FALSE
        WHERE atlas_gene_id NOT IN (
            SELECT atlas_gene_id FROM gene_filter_stats_tmp
        )
    """)

    # 7. 统计结果
    keep_count = conn.execute(f"""
        SELECT COUNT(*) FROM var
        WHERE {add_data} = TRUE
    """).fetchone()[0]

    conn.execute("DROP TABLE IF EXISTS gene_filter_stats_tmp")

    print(f"filter_genes Done, 耗时: {(datetime.now() - start_time).total_seconds():.2f} 秒")
    print(f"保留基因 = {keep_count} / {n_genes} , ({keep_count / n_genes * 100:.2f}%)")

    # 函数结束后兜底清理
    _cleanup_qc_after_step(
        conn,
        temp_tables=["gene_filter_stats_tmp"],
        checkpoint=False,
        collect=True,
    )

# 运行结果
# var 表  新增字段
#     字段	                    含义
#  filter_genes	       该 gene是否符合过滤条件 true


''' 计算每个细胞的总 UMI（Unique Molecular Identifier）计数 '''
def calculate_cell_total_counts(
        atlas: Atlas,
        add_data: str = "cell_total_counts",
        chunk_cells: int = 1_000_000,
) -> None:
    """计算每个细胞的总表达量。

    该函数按细胞分块聚合表达矩阵，将每个细胞的总 counts 写入 ``obs``。该结果可用于 QC、过滤、归一化检查和绘图。

    Parameters
    ----------
    atlas
        Atlas 对象。通常需要已经连接到 DuckDB 数据库，并包含该函数读取或写入所需的 ``obs``、``var``、表达矩阵或结果表。
    add_data
        写入数据库的新表达矩阵表名或结果列名。
    chunk_cells
        按细胞分块处理时每个 chunk 的细胞数量。

    Returns
    -------
    None
        结果直接写入 Atlas 数据库或当前图形窗口。

    Examples
    --------
    写入默认列::

        sap.pp.calculate_cell_total_counts(atlas)
        atlas.head("obs")

    写入自定义列并调大分块::

        sap.pp.calculate_cell_total_counts(
            atlas,
            add_data="total_counts_raw",
            chunk_cells=1000000,
        )"""

    start_time = datetime.now()

    conn = atlas.connection

    # DuckDB 性能参数
    try:
        th = os.cpu_count() or 1
        conn.execute(f"PRAGMA threads={th}")
    except Exception:
        pass

    # Step 0：确保 obs 有目标列
    conn.execute(f"""
        ALTER TABLE obs
        ADD COLUMN IF NOT EXISTS {add_data} DOUBLE
    """)

    conn.execute(f"""
        UPDATE obs
        SET {add_data} = 0
    """)

    # Step 1：获取 cell_id 范围
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

    # Step 2：分块聚合 + 分块写回

    pbar = tqdm(
        range(n_chunks),
        total=n_chunks,
        desc="cell_total_counts",
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
                SUM(data) AS total_counts
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

    print(f"calculate_cell_total_counts Done, 耗时: {(datetime.now() - start_time).total_seconds():.2f} 秒")

    # 内存清理
    _cleanup_qc_after_step(
        conn,
        temp_tables=["cell_total_counts_chunk"],
        checkpoint=False,
        collect=True,
    )

# 运行结果
# obs 表  新增字段
#     字段	                    含义
#  cell_total_counts	     每个细胞的总 UMI 计数


'''  计算每个基因的表达值 '''
def calculate_gene_total_counts(
                    atlas: 'Atlas',
                    add_gene_total_counts: str = "gene_total_counts",
                    add_gene_mean_counts: str = "gene_mean_counts",
                    ) -> None:
    """计算每个基因的总表达量和平均表达量。

    该函数按基因聚合表达矩阵，并把总表达量和平均表达量写入 ``var``。这些指标可用于基因过滤、高表达基因检查和数据质量评估。

    Parameters
    ----------
    atlas
        Atlas 对象。通常需要已经连接到 DuckDB 数据库，并包含该函数读取或写入所需的 ``obs``、``var``、表达矩阵或结果表。
    add_gene_total_counts
        写入 ``var`` 的第一个结果列名。
    add_gene_mean_counts
        写入 ``var`` 的第二个结果列名。

    Returns
    -------
    None
        结果直接写入 Atlas 数据库或当前图形窗口。

    Examples
    --------
    计算默认基因统计列::

        sap.pp.calculate_gene_total_counts(atlas)

    写入自定义列名::

        sap.pp.calculate_gene_total_counts(
            atlas,
            add_gene_total_counts="gene_total_counts_raw",
            add_gene_mean_counts="gene_mean_counts_raw",
        )"""

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

    # 细胞总数（用于 mean）
    total_cells = conn.execute("SELECT COUNT(*) FROM obs").fetchone()[0]

    # Step 1: CSR 聚合（一次扫描）
    conn.execute("DROP TABLE IF EXISTS gene_stats_tmp")
    conn.execute("""
        CREATE TEMP TABLE gene_stats_tmp AS
        SELECT
            atlas_gene_id,
            SUM(data) AS total_counts
        FROM X_HyS_data
        GROUP BY atlas_gene_id
    """)

    # Step 2: 写回 var（atlas_gene_id == atlas_gene_id）

    conn.execute(f"""
        UPDATE var
        SET
            {add_gene_total_counts} = s.total_counts,
            {add_gene_mean_counts} = s.total_counts / {total_cells}
        FROM gene_stats_tmp AS s
        WHERE var.atlas_gene_id = s.atlas_gene_id
    """)

    # Step 3: 零表达基因补零（CSR 中不存在）

    conn.execute(f"""
        UPDATE var
        SET
            {add_gene_total_counts} = 0,
            {add_gene_mean_counts} = 0
        WHERE atlas_gene_id NOT IN (SELECT atlas_gene_id FROM gene_stats_tmp)
    """)

    # Step 4: 清理临时表
    conn.execute("DROP TABLE IF EXISTS gene_stats_tmp")

    _cleanup_qc_after_step(
        conn,
        temp_tables=["gene_stats_tmp"],
        checkpoint=False,
        collect=True,
    )

    print(f"calculate_gene_total_counts Done, 耗时: {(datetime.now() - start_time).total_seconds():.2f} 秒")

# 运行结果
# var 表  新增字段
#     字段	                    含义
#  gene_total_counts	     每个基因的总表达值（SUM）
#  gene_mean_counts	         每个基因的平均表达值（SUM / 总细胞数）


''' qc 控制指标 '''
def calculate_qc_metrics(
    atlas: Atlas,
    qc_vars: dict[str, Any] | None=None,
    chunk_cells: int=100_000
):

    """计算常用单细胞 QC 指标。

    该函数从 Atlas 表中计算每个细胞的总 counts、检测基因数，以及可选 QC 基因集合的 counts 和比例。它类似 Scanpy 的 ``sc.pp.calculate_qc_metrics``，但以分块 SQL/矩阵操作写回数据库。

    Parameters
    ----------
    atlas
        Atlas 对象。通常需要已经连接到 DuckDB 数据库，并包含该函数读取或写入所需的 ``obs``、``var``、表达矩阵或结果表。
    qc_vars
        QC 基因集合定义。可以用字典把 QC 名称映射到基因筛选条件，用于计算线粒体、核糖体等指标。
    chunk_cells
        按细胞分块处理时每个 chunk 的细胞数量。

    Returns
    -------
    None
        结果直接写入 Atlas 数据库或当前图形窗口。

    Examples
    --------
    计算基础 QC 指标::

        sap.pp.calculate_qc_metrics(atlas)

    计算线粒体和核糖体基因比例::

        sap.pp.calculate_qc_metrics(
            atlas,
            qc_vars={
                "mt": "atlas_gene_name LIKE 'MT-%'",
                "ribo": "atlas_gene_name LIKE 'RPL%' OR atlas_gene_name LIKE 'RPS%'",
            },
        )

    结合 QC 结果过滤细胞::

        sap.pp.calculate_qc_metrics(atlas, qc_vars={"mt": "atlas_gene_name LIKE 'MT-%'"})
        sap.pp.filter_cells(atlas, min_genes=200, max_counts=50000)"""

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

    # cell-wise QC：分块处理
    pbar = tqdm(
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
                f"SUM(CASE WHEN v.{qc_key} THEN x.data ELSE 0 END)"
                f" AS total_counts_{qc_key}"
            )
        qc_sum_sql = ",\n".join(qc_sum_expr)

        conn.execute("DROP TABLE IF EXISTS _cell_chunk")

        conn.execute(f"""
            CREATE TEMP TABLE _cell_chunk AS
            SELECT
                x.atlas_cell_id,
                SUM(x.data) AS cell_total_counts,
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
            SUM(data) AS gene_total_counts,
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

    print(f"calculate_gene_total_counts Done, 耗时: {(datetime.now() - start_time).total_seconds():.2f} 秒")

# 运行结果
# var 表  新增字段
#     字段	                    含义
#  mt	                 是否是线粒体基因（MT- 前缀）
#  ribo                  是否是核糖体基因（^RP[SL]）
#  gene_total_counts	 该 gene 在所有 cells 的 counts 之和
#  n_cells_by_counts	 非零 cell 数 ：有多少 cell 表达该 gene（非零）


# obs 表  新增字段
#     字段	                   含义
#  cell_total_counts	    每个 cell 的总 counts
#  n_genes_by_counts       	每个 cell 的 非零基因数
#  total_counts_mt	        线粒体基因 counts 之和
#  pct_counts_mt           	线粒体基因的比例 (%)
#  total_counts_ribo	    核糖体基因ribo counts
#  pct_counts_ribo	        核糖体基因ribo 比例 (%)
