import os
from datetime import datetime
from ..data import Atlas
from typing import Optional
import logging
import math

# 获取日志记录器
logger = logging.getLogger('Atlas')


'''==== 过滤细胞 ：  使用 mmap 扫描 CSR 过滤细胞（安全支持 1 亿细胞） ======'''
# 833206 * 17745  sap.pp.filter_cells(atlas, min_genes=200) 过滤细胞 = 0 (0.00%) 总耗时 0.54 秒
def filter_cells_fast(atlas: 'Atlas',
                       min_counts: Optional[int] = None,
                       min_genes: Optional[int] = None,
                       max_counts: Optional[int] = None,
                       max_genes: Optional[int] = None,
                       add_key="filter_cells"):
    """
    根据参数过滤细胞的离群点，并将过滤条件的布尔向量写入obs表
    该细胞所有基因的表达值之和小于min_counts或大于max_counts
    Args:
        atlas: Atlas对象
            min_counts: 最小总表达值 该细胞所有基因的表达值之和 sum_expr <  min_counts
            min_genes: 最小非零基因数 该细胞的非零表达的基因个数 nonzero_expr < min_genes
            max_counts: 最大总表达值 该细胞所有基因的表达值之和 sum_expr > max_counts
            max_genes: 最大非零基因数 该细胞的非零表达的基因个数 nonzero_expr > max_genes
        add_key: 写入obs表的字段名
    """

    print("开始过滤细胞...")
    start = datetime.now()

    conn = atlas.connection
    th = os.cpu_count()
    conn.execute(f"PRAGMA threads={th}")
    # conn.execute(f"PRAGMA temp_directory='.tmp_duckdb'")
    print(f"DuckDB threads = {th}")

    # 预先添加列
    conn.execute(f"""
        ALTER TABLE obs 
        ADD COLUMN IF NOT EXISTS {add_key} BOOLEAN DEFAULT FALSE
    """)

    # 统计细胞数
    total_cells = conn.execute("SELECT COUNT(*) FROM obs").fetchone()[0]
    print(f"总细胞数 = {total_cells:,}")

    # -------------------------------
    # 构建 SQL 过滤条件
    # -------------------------------
    conds = []
    if min_counts is not None: conds.append(f"sum_expr >= {min_counts}")
    if max_counts is not None: conds.append(f"sum_expr <= {max_counts}")
    if min_genes  is not None: conds.append(f"nonzero_genes >= {min_genes}")
    if max_genes  is not None: conds.append(f"nonzero_genes <= {max_genes}")
    condition = " AND ".join(conds) if conds else "TRUE"

    # -------------------------------
    # Step1：先把需要保留的 atlas_cell_id 算好
    # -------------------------------
    print("统计基因数量与表达量（流式聚合）...")


    conn.execute(f"""
        CREATE TEMP TABLE keep_cells AS
        SELECT atlas_cell_id
        FROM (
            SELECT 
                atlas_cell_id,
                SUM(data) AS sum_expr,
            COUNT(*) AS nonzero_genes
            FROM X_CSRO_data
            GROUP BY atlas_cell_id
        ) WHERE {condition}
    """)

    # -------------------------------
    # Step2：只更新 TRUE（避免笨重 UPDATE JOIN）
    # -------------------------------
    print("更新 obs（仅 TRUE，加速 3~10x）...")

    conn.execute(f"UPDATE obs SET {add_key}=FALSE")   # 全部设为 FALSE
    conn.execute(f"""
        UPDATE obs SET {add_key}=TRUE
        WHERE atlas_cell_id 
        IN (SELECT atlas_cell_id FROM keep_cells)
    """)

    # -------------------------------
    # Step3: 统计结果
    # -------------------------------
    keep_cells = conn.execute(f"SELECT COUNT(*) FROM keep_cells").fetchone()[0]
    removed = total_cells - keep_cells

    # 删除临时表
    conn.execute("DROP TABLE IF EXISTS keep_cells")

    print(f"保留细胞 = {keep_cells:,}")
    print(f"过滤细胞 = {removed:,} ({removed/total_cells*100:.2f}%)")
    print("总耗时 {:.2f} 秒".format((datetime.now() - start).total_seconds()))

# 运行结果
# obs 表  新增字段
#     字段	                    含义
#  filter_cells	       该 cell 是否符合过滤条件 true

def filter_cells(
        atlas,
        min_counts: Optional[int] = None,
        min_genes: Optional[int] = None,
        max_counts: Optional[int] = None,
        max_genes: Optional[int] = None,
        add_key: str = "filter_cells",
        cell_chunk_size: int = 500_000,   # ✅ 修改1：新增分块大小
):
    """
    根据参数过滤细胞，并将过滤条件的布尔向量写入 obs 表。

    ✅ 大数据安全版：
    - 不创建全量 keep_cells 大临时表
    - 按 atlas_cell_id 分块聚合
    - 每次只产生一个小的 keep_cells_chunk
    - 支持小内存 + 大数据
    """

    print("开始过滤细胞...（CHUNKED / BIG-DATA SAFE）")
    start = datetime.now()

    conn = atlas.connection

    # -------------------------------------------------
    # 0. DuckDB 参数
    # -------------------------------------------------
    try:
        th = os.cpu_count() or 1
        conn.execute(f"PRAGMA threads={th}")
        print(f"DuckDB threads = {th}")
    except Exception:
        pass

    # -------------------------------------------------
    # 1. 添加 obs 过滤字段
    # -------------------------------------------------
    conn.execute(f"""
        ALTER TABLE obs 
        ADD COLUMN IF NOT EXISTS {add_key} BOOLEAN DEFAULT FALSE
    """)

    total_cells = conn.execute("SELECT COUNT(*) FROM obs").fetchone()[0]
    print(f"总细胞数 = {total_cells:,}")

    # ✅ 修改2：一次性初始化为 FALSE
    # 这一步仍然会写 obs 一遍，但不会产生大临时表
    print("初始化 obs 过滤字段为 FALSE ...")
    conn.execute(f"""
        UPDATE obs
        SET {add_key} = FALSE
    """)

    # -------------------------------------------------
    # 2. 构建过滤条件
    # -------------------------------------------------
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

    # -------------------------------------------------
    # 3. 获取 cell_id 范围
    # -------------------------------------------------
    min_cell, max_cell = conn.execute("""
        SELECT 
            MIN(atlas_cell_id),
            MAX(atlas_cell_id)
        FROM obs
    """).fetchone()

    if min_cell is None or max_cell is None:
        print("obs 为空，跳过。")
        return

    n_chunks = (max_cell - min_cell + cell_chunk_size) // cell_chunk_size

    print(f"cell_id 范围: {min_cell:,} ~ {max_cell:,}")
    print(f"cell_chunk_size = {cell_chunk_size:,}")
    print(f"预计分块数 = {n_chunks:,}")

    keep_total = 0

    # -------------------------------------------------
    # 4. 分块聚合 + 分块写回
    # -------------------------------------------------
    print("开始分块统计并写回 obs ...")

    for i in range(n_chunks):

        c_start = min_cell + i * cell_chunk_size
        c_end = min(c_start + cell_chunk_size - 1, max_cell)

        print(f"[Chunk {i + 1}/{n_chunks}] cells {c_start:,} ~ {c_end:,}")

        # ✅ 修改3：只创建当前 chunk 的临时表
        # 不再创建全量 keep_cells
        conn.execute("DROP TABLE IF EXISTS keep_cells_chunk")

        conn.execute(f"""
            CREATE TEMP TABLE keep_cells_chunk AS
            SELECT atlas_cell_id
            FROM (
                SELECT 
                    atlas_cell_id,
                    SUM(data) AS sum_expr,
                    COUNT(*) AS nonzero_genes
                FROM X_CSRO_data
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

        # ✅ 修改4：只更新当前 chunk 内 TRUE 的 cells
        conn.execute(f"""
            UPDATE obs
            SET {add_key} = TRUE
            WHERE atlas_cell_id IN (
                SELECT atlas_cell_id FROM keep_cells_chunk
            )
        """)

        # ✅ 修改5：每个 chunk 后立即清理小临时表
        conn.execute("DROP TABLE IF EXISTS keep_cells_chunk")

    # -------------------------------------------------
    # 5. 统计结果
    # -------------------------------------------------
    removed = total_cells - keep_total

    print(f"保留细胞 = {keep_total:,}")
    print(f"过滤细胞 = {removed:,} ({removed / total_cells * 100:.2f}%)")
    print("总耗时 {:.2f} 秒".format((datetime.now() - start).total_seconds()))


'''==== 过滤基因 ：行-批 + NumPy 聚合 + 批写回 var ，当前最快 =========='''
# 833206 * 17745  sap.pp.filter_genes(atlas, min_cells=3)  保留基因 17745  耗时: 3.35 秒
def filter_genes(
        atlas,
        min_counts: Optional[int] = None,
        min_cells: Optional[int] = None,
        max_counts: Optional[int] = None,
        max_cells: Optional[int] = None,
        add_key: str = "filter_genes"
) -> None:
    """
    基因过滤：大数据安全 + 更快版本

    ✅ 修改点：
    1. 不再 fetchall 到 Python
    2. 不再 Python for 循环判断
    3. 不再 executemany 写 tmp_stats
    4. 不再用 atlas_gene_name 文本 join
    5. 直接用 atlas_gene_id 纯 SQL 写回 var
    """

    print("==== 开始基因过滤（SQL FAST）====")
    start_time = datetime.now()

    conn = atlas.connection

    # -------------------------------------------------
    # 0. DuckDB 多线程
    # -------------------------------------------------
    try:
        th = os.cpu_count() or 1
        conn.execute(f"PRAGMA threads={th}")
        print(f"DuckDB 多线程: {th}")
    except Exception:
        pass

    # -------------------------------------------------
    # 1. 统计基因数量
    # -------------------------------------------------
    n_genes = conn.execute("""
        SELECT COUNT(*) FROM var
    """).fetchone()[0]

    print(f"检测到基因数量: {n_genes:,}")

    # -------------------------------------------------
    # 2. 添加过滤字段
    # -------------------------------------------------
    conn.execute(f"""
        ALTER TABLE var 
        ADD COLUMN IF NOT EXISTS {add_key} BOOLEAN DEFAULT FALSE
    """)

    # -------------------------------------------------
    # 3. 构建 SQL 条件
    # -------------------------------------------------
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

    # -------------------------------------------------
    # 4. 聚合 X_CSRO_data
    # -------------------------------------------------
    print("开始聚合 X_CSRO_data ...")

    conn.execute("DROP TABLE IF EXISTS gene_filter_stats_tmp")

    # ✅ 修改1：只生成 gene 级小临时表
    # 结果规模 ≈ 基因数，不是 nnz 数
    conn.execute("""
        CREATE TEMP TABLE gene_filter_stats_tmp AS
        SELECT 
            atlas_gene_id,
            SUM(data) AS sum_expr,
            COUNT(*) AS nonzero_expr
        FROM X_CSRO_data
        GROUP BY atlas_gene_id
    """)

    # -------------------------------------------------
    # 5. 纯 SQL 写回 var
    # -------------------------------------------------
    print("写回 var.filter_genes ...")

    # ✅ 修改2：不用 atlas_gene_name，直接用 atlas_gene_id
    # ✅ 修改3：LEFT JOIN 语义，完全零表达基因也能被置为 FALSE
    conn.execute(f"""
        UPDATE var
        SET {add_key} =
            CASE
                WHEN {condition}
                THEN TRUE
                ELSE FALSE
            END
        FROM gene_filter_stats_tmp AS s
        WHERE var.atlas_gene_id = s.atlas_gene_id
    """)

    # -------------------------------------------------
    # 6. 处理完全零表达基因
    # -------------------------------------------------
    # ✅ 修改4：CSR 中完全没出现的 gene，默认不通过过滤
    conn.execute(f"""
        UPDATE var
        SET {add_key} = FALSE
        WHERE atlas_gene_id NOT IN (
            SELECT atlas_gene_id FROM gene_filter_stats_tmp
        )
    """)

    # -------------------------------------------------
    # 7. 统计结果
    # -------------------------------------------------
    keep_count = conn.execute(f"""
        SELECT COUNT(*) FROM var
        WHERE {add_key} = TRUE
    """).fetchone()[0]

    conn.execute("DROP TABLE IF EXISTS gene_filter_stats_tmp")

    print(f"过滤完成: 保留基因 {keep_count:,} / 总 {n_genes:,}")
    print("总耗时: {:.2f} 秒".format((datetime.now() - start_time).total_seconds()))

# 运行结果
# var 表  新增字段
#     字段	                    含义
#  filter_genes	       该 gene是否符合过滤条件 true

'''====== 计算每个细胞的总 UMI（Unique Molecular Identifier）计数 ========== '''
# 833206 * 17745  sap.pp.calculate_cell_total_counts(atlas)   耗时 1.00 秒
def calculate_cell_total_counts(
        atlas,
        add_key: str = "cell_total_counts",
        cell_chunk_size: int = 1_000_000,   # ✅ 修改1：新增分块大小
) -> None:
    """
    使用 DuckDB 原生 CSR 表计算每个细胞的总 UMI 计数。

    ✅ 大数据安全版：
    - 不创建全量 cell_total_counts_tmp
    - 按 atlas_cell_id 分块聚合
    - 每个 chunk 只生成小临时表
    - 写回 obs 后立即清理
    - 支持小内存 + 大数据
    """

    print("==== DuckDB CSR 原生计算每个细胞的总 UMI（CHUNKED）====")
    start = datetime.now()

    conn = atlas.connection

    # --------------------------------
    # DuckDB 性能参数
    # --------------------------------
    try:
        th = os.cpu_count() or 1
        conn.execute(f"PRAGMA threads={th}")
        print(f"DuckDB threads = {th}")
    except Exception:
        pass

    # --------------------------------
    # Step 0：确保 obs 有目标列
    # --------------------------------
    conn.execute(f"""
        ALTER TABLE obs
        ADD COLUMN IF NOT EXISTS {add_key} DOUBLE
    """)

    # ✅ 修改2：先初始化为 0，处理完全空 cell
    conn.execute(f"""
        UPDATE obs
        SET {add_key} = 0
    """)

    # --------------------------------
    # Step 1：获取 cell_id 范围
    # --------------------------------
    min_cell, max_cell = conn.execute("""
        SELECT 
            MIN(atlas_cell_id),
            MAX(atlas_cell_id)
        FROM obs
    """).fetchone()

    if min_cell is None or max_cell is None:
        print("obs 为空，跳过。")
        return

    n_chunks = math.ceil((max_cell - min_cell + 1) / cell_chunk_size)

    print(f"cell_id 范围: {min_cell:,} ~ {max_cell:,}")
    print(f"cell_chunk_size = {cell_chunk_size:,}")
    print(f"预计分块数 = {n_chunks:,}")

    total_updated_cells = 0

    # --------------------------------
    # Step 2：分块聚合 + 分块写回
    # --------------------------------
    print("开始分块统计每个细胞 total UMI ...")

    for i in range(n_chunks):

        c_start = min_cell + i * cell_chunk_size
        c_end = min(c_start + cell_chunk_size - 1, max_cell)

        print(f"[Chunk {i + 1}/{n_chunks}] cells {c_start:,} ~ {c_end:,}")

        # ✅ 修改3：只创建当前 chunk 的小临时表
        conn.execute("DROP TABLE IF EXISTS cell_total_counts_chunk")

        conn.execute(f"""
            CREATE TEMP TABLE cell_total_counts_chunk AS
            SELECT
                atlas_cell_id,
                SUM(data) AS total_counts
            FROM X_CSRO_data
            WHERE atlas_cell_id BETWEEN {c_start} AND {c_end}
            GROUP BY atlas_cell_id
        """)

        n_now = conn.execute("""
            SELECT COUNT(*) FROM cell_total_counts_chunk
        """).fetchone()[0]

        total_updated_cells += n_now

        # ✅ 修改4：只写回当前 chunk 有表达记录的 cell
        conn.execute(f"""
            UPDATE obs
            SET {add_key} = t.total_counts
            FROM cell_total_counts_chunk t
            WHERE obs.atlas_cell_id = t.atlas_cell_id
        """)

        # ✅ 修改5：每个 chunk 后立即清理
        conn.execute("DROP TABLE IF EXISTS cell_total_counts_chunk")

    print(f"已计算细胞数 = {total_updated_cells:,}")
    print("完成，总耗时 {:.2f} 秒".format(
        (datetime.now() - start).total_seconds()
    ))

# 运行结果
# obs 表  新增字段
#     字段	                    含义
#  cell_total_counts	     每个细胞的总 UMI 计数

''' ====== 计算每个基因的表达值 ========== '''
# 833206 * 17745   sap.pp.calculate_gene_total_counts(atlas)    耗时  0.86  秒
def calculate_gene_total_counts(
                    atlas: 'Atlas',
                    add_key1: str = "gene_total_counts",
                    add_key2: str = "gene_mean_counts",
                    ) -> None:
    """
    使用 X_CSRO_data 计算：
        - 每个基因的总表达值（SUM）
        - 每个基因的平均表达值（SUM / 总细胞数）
    结果写入 var 表

    ✔ 低内存
    ✔ 单次扫描 CSR
    ✔ 适配超大数据
    """
    logger.info("开始使用 CSR 计算每个基因的总表达值和平均表达值")
    start = datetime.now()

    conn = atlas.connection

    # -------------------------------------------------
    # DuckDB 并行设置
    # -------------------------------------------------
    try:
        th = os.cpu_count() or 1
        conn.execute(f"PRAGMA threads={th}")
        logger.info(f"DuckDB threads = {th}")
    except:
        pass

    # -------------------------------------------------
    # 确保 var 表有目标列
    # -------------------------------------------------
    cols = [r[1] for r in conn.execute("PRAGMA table_info(var)").fetchall()]

    if add_key1 not in cols:
        conn.execute(f"ALTER TABLE var ADD COLUMN {add_key1} DOUBLE DEFAULT 0")
    if add_key2 not in cols:
        conn.execute(f"ALTER TABLE var ADD COLUMN {add_key2} DOUBLE DEFAULT 0")

    # -------------------------------------------------
    # 细胞总数（用于 mean）
    # -------------------------------------------------
    total_cells = conn.execute("SELECT COUNT(*) FROM obs").fetchone()[0]
    logger.info(f"总细胞数 = {total_cells:,}")

    # -------------------------------------------------
    # Step 1: CSR 聚合（一次扫描）
    # -------------------------------------------------
    logger.info("聚合 X_CSRO_data（按 atlas_gene_id）...")

    conn.execute("DROP TABLE IF EXISTS gene_stats_tmp")
    conn.execute("""
        CREATE TEMP TABLE gene_stats_tmp AS
        SELECT
            atlas_gene_id,
            SUM(data) AS total_counts
        FROM X_CSRO_data
        GROUP BY atlas_gene_id
    """)

    # -------------------------------------------------
    # Step 2: 写回 var（atlas_gene_id == atlas_gene_id）
    # -------------------------------------------------
    logger.info("更新 var 表...")

    conn.execute(f"""
        UPDATE var
        SET
            {add_key1} = s.total_counts,
            {add_key2} = s.total_counts / {total_cells}
        FROM gene_stats_tmp AS s
        WHERE var.atlas_gene_id = s.atlas_gene_id
    """)

    # -------------------------------------------------
    # Step 3: 零表达基因补零（CSR 中不存在）
    # -------------------------------------------------
    logger.info("处理完全零表达的基因...")

    conn.execute(f"""
        UPDATE var
        SET
            {add_key1} = 0,
            {add_key2} = 0
        WHERE atlas_gene_id NOT IN (SELECT atlas_gene_id FROM gene_stats_tmp)
    """)

    # -------------------------------------------------
    # Step 4: 清理临时表
    # -------------------------------------------------
    conn.execute("DROP TABLE gene_stats_tmp")

    logger.info("基因表达统计完成")
    logger.info("耗时 {:.2f} 秒".format((datetime.now() - start).total_seconds()))

# 运行结果
# var 表  新增字段
#     字段	                    含义
#  gene_total_counts	     每个基因的总表达值（SUM）
#  gene_mean_counts	         每个基因的平均表达值（SUM / 总细胞数）



''' ====== 质量控制指标  +  线粒体基因 + 核糖体基因比例计算  ========== '''
# 833206 * 17745  sap.pp.calculate_qc_metrics(atlas)  24.10 秒
def calculate_qc_metrics_fast(atlas, qc_vars: dict | None = None):

    print("==== calculate_qc_metrics (SINGLE PASS) ====")
    start = datetime.now()

    conn = atlas.connection

    if qc_vars is None:
        qc_vars = {
            "mt": "MT-",
            "ribo": "^(RPS|RPL)"
        }

    # ============================================
    # 1️⃣ 标记 var
    # ============================================
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
                SET {qc_key} = UPPER(atlas_gene_name) LIKE '{pattern.upper()}%'
            """)

    # ============================================
    # 2️⃣ SINGLE PASS 聚合
    # ============================================

    # 动态生成 qc SUM
    qc_sum_expr = []
    for qc_key in qc_vars.keys():
        qc_sum_expr.append(
            f"SUM(CASE WHEN v.{qc_key} THEN x.data ELSE 0 END) AS total_counts_{qc_key}"
        )

    qc_sum_sql = ",\n".join(qc_sum_expr)

    # -------- cell + qc --------
    conn.execute(f"""
        CREATE OR REPLACE TEMP TABLE _cell_qc AS
        SELECT
            x.atlas_cell_id,

            SUM(x.data) AS cell_total_counts,
            COUNT(*)    AS n_genes_by_counts,

            {qc_sum_sql}

        FROM X_CSRO_data x
        JOIN var v
          ON x.atlas_gene_id = v.atlas_gene_id
        GROUP BY x.atlas_cell_id
    """)

    # -------- gene --------
    conn.execute("""
        CREATE OR REPLACE TEMP TABLE _gene_qc AS
        SELECT
            atlas_gene_id,
            SUM(data) AS gene_total_counts,
            COUNT(*)  AS n_cells_by_counts
        FROM X_CSRO_data
        GROUP BY atlas_gene_id
    """)

    # ============================================
    # 3️⃣ 写入 obs
    # ============================================
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

    # UPDATE obs
    set_expr = [
        "cell_total_counts = c.cell_total_counts",
        "n_genes_by_counts = c.n_genes_by_counts"
    ]

    for qc_key in qc_vars.keys():
        set_expr.append(f"total_counts_{qc_key} = c.total_counts_{qc_key}")
        set_expr.append(f"""
            pct_counts_{qc_key} =
            CASE WHEN c.cell_total_counts > 0
            THEN 100.0 * c.total_counts_{qc_key} / c.cell_total_counts
            ELSE 0 END
        """)

    conn.execute(f"""
        UPDATE obs
        SET {",".join(set_expr)}
        FROM _cell_qc c
        WHERE obs.atlas_cell_id = c.atlas_cell_id
    """)

    # ============================================
    # 4️⃣ 写入 var
    # ============================================
    conn.execute("""
        ALTER TABLE var
        ADD COLUMN IF NOT EXISTS gene_total_counts REAL
    """)
    conn.execute("""
        ALTER TABLE var
        ADD COLUMN IF NOT EXISTS n_cells_by_counts INTEGER
    """)

    conn.execute("""
        UPDATE var
        SET
            gene_total_counts = g.gene_total_counts,
            n_cells_by_counts = g.n_cells_by_counts
        FROM _gene_qc g
        WHERE var.atlas_gene_id = g.atlas_gene_id
    """)

    # ============================================
    # 5️⃣ 清理
    # ============================================
    conn.execute("DROP TABLE _cell_qc")
    conn.execute("DROP TABLE _gene_qc")

    print("✅ SINGLE PASS QC 完成")
    print("耗时:", (datetime.now() - start).total_seconds())

# 把 gene-wise 统计从 chunk 循环里拿出来，避免每个 chunk 都重复扫一遍 X_CSRO_data
def calculate_qc_metrics(
    atlas,
    qc_vars=None,
    cell_chunk_size=100_000
):
    """
    真·大数据安全 QC

    ✅ 最小化修改版：
    - cell-wise: 仍然按 cell 分块，避免生成巨大 _cell_qc
    - gene-wise: 改为循环外一次性 GROUP BY atlas_gene_id
      因为 gene 数量很小，不需要分块
    """

    print("==== calculate_qc_metrics (CHUNK CELL + ONE-PASS GENE) ====")
    start = datetime.now()
    conn = atlas.connection

    if qc_vars is None:
        qc_vars = {
            "mt": "MT-",
            "ribo": "^(RPS|RPL)"
        }

    # =================================================
    # 0️⃣ DuckDB 参数
    # =================================================
    try:
        th = os.cpu_count() or 8
        conn.execute(f"PRAGMA threads={th}")
        conn.execute("PRAGMA memory_limit='8GB'")
        print(f"DuckDB threads = {th}")
    except Exception:
        pass

    # =================================================
    # 1️⃣ var 打 qc 标记
    # =================================================
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

    # =================================================
    # 2️⃣ 初始化 obs 列
    # =================================================
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

    # ✅ 修改1：先初始化 obs，避免完全空 cell 保留旧值
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

    # =================================================
    # 3️⃣ 计算 cell_id 范围
    # =================================================
    min_cell, max_cell = conn.execute("""
        SELECT MIN(atlas_cell_id), MAX(atlas_cell_id)
        FROM obs
    """).fetchone()

    if min_cell is None or max_cell is None:
        print("obs 为空，跳过。")
        return

    n_chunks = math.ceil((max_cell - min_cell + 1) / cell_chunk_size)

    print(f"Cells: {max_cell - min_cell + 1:,}")
    print(f"Chunk size: {cell_chunk_size:,}")
    print(f"Chunks: {n_chunks:,}")

    # =================================================
    # 4️⃣ cell-wise QC：分块处理
    # =================================================
    for i in range(n_chunks):

        c_start = min_cell + i * cell_chunk_size
        c_end = min(c_start + cell_chunk_size - 1, max_cell)

        print(f"[Chunk {i + 1}/{n_chunks}] cells {c_start:,} ~ {c_end:,}")

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
            FROM X_CSRO_data x
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

    # =================================================
    # 5️⃣ gene-wise QC：循环外一次性计算
    # ✅ 修改2：从 chunk 循环里拿出来
    # =================================================
    print("计算 gene-wise QC ...")

    conn.execute("""
        ALTER TABLE var
        ADD COLUMN IF NOT EXISTS gene_total_counts REAL
    """)
    conn.execute("""
        ALTER TABLE var
        ADD COLUMN IF NOT EXISTS n_cells_by_counts INTEGER
    """)

    # ✅ 修改3：gene 级临时表很小，不需要分块
    conn.execute("DROP TABLE IF EXISTS _gene_qc")
    conn.execute("""
        CREATE TEMP TABLE _gene_qc AS
        SELECT
            atlas_gene_id,
            SUM(data) AS gene_total_counts,
            COUNT(*)  AS n_cells_by_counts
        FROM X_CSRO_data
        GROUP BY atlas_gene_id
    """)

    # ✅ 修改4：先把 var 置 0，处理完全零表达基因
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

    print("✅ QC 完成")
    print("耗时:", (datetime.now() - start).total_seconds())

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