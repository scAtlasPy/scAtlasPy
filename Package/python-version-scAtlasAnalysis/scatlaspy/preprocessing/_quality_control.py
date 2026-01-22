import os
from datetime import datetime
from ..data import Atlas
from typing import Optional
import logging
# 获取日志记录器
logger = logging.getLogger('Atlas')


# ========== todo 过滤细胞 ：  使用 mmap 扫描 CSR 过滤细胞（安全支持 1 亿细胞） ==========
def filter_cells(atlas: 'Atlas',
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
    conn.execute(f"PRAGMA memory_limit='55GB'")
    conn.execute(f"PRAGMA temp_directory='.tmp_duckdb'")
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
    # Step1：先把需要保留的 cell_index 算好
    # -------------------------------
    print("统计基因数量与表达量（流式聚合）...")

    conn.execute("DROP TABLE IF EXISTS keep_cells")
    conn.execute(f"""
        CREATE TEMP TABLE keep_cells AS
        SELECT cell_index
        FROM (
            SELECT 
                cell_index,
                SUM(data) AS sum_expr,
            COUNT(*) AS nonzero_genes
            FROM X_CSR_data
            GROUP BY cell_index
        ) WHERE {condition}
    """)

    # -------------------------------
    # Step2：只更新 TRUE（避免笨重 UPDATE JOIN）
    # -------------------------------
    print("更新 obs（仅 TRUE，加速 3~10x）...")

    conn.execute(f"UPDATE obs SET {add_key}=FALSE")   # 全部设为 FALSE
    conn.execute(f"""
        UPDATE obs SET {add_key}=TRUE
        WHERE id IN (SELECT cell_index FROM keep_cells)
    """)

    # -------------------------------
    # Step3: 统计结果
    # -------------------------------
    keep_cells = conn.execute(f"SELECT COUNT(*) FROM keep_cells").fetchone()[0]
    removed = total_cells - keep_cells

    print(f"保留细胞 = {keep_cells:,}")
    print(f"过滤细胞 = {removed:,} ({removed/total_cells*100:.2f}%)")
    print("总耗时 {:.2f} 秒".format((datetime.now() - start).total_seconds()))

# ========== todo 过滤基因 ：行-批 + NumPy 聚合 + 批写回 var ，当前最快 ==========
def filter_genes(atlas: 'Atlas',
                     min_counts: Optional[int] = None,
                     min_cells: Optional[int] = None,
                     max_counts: Optional[int] = None,
                     max_cells: Optional[int] = None,
                     add_key: str = "filter_genes") -> None:
    """
    使用 CSR 数据（X_CSR_indptr + X_CSR_data）计算每个基因的：
        - sum_expr
        - nonzero_expr
    并写入 var 表。
    """

    print("==== 开始基因过滤 ====")
    start_time = datetime.now()

    conn = atlas.connection

    # 允许 DuckDB 多线程
    try:
        th = os.cpu_count() or 1
        conn.execute(f"PRAGMA threads={th}")
        print(f"DuckDB 多线程: {th}")
    except:
        pass

    # =======================================================
    # 👇 修改点：CSR indices 对应的是 var.id
    # =======================================================
    gene_map = dict(conn.execute("SELECT id, gene_id FROM var").fetchall())
    n_genes = len(gene_map)
    print(f"检测到基因数量: {n_genes}")

    # 添加过滤字段
    conn.execute(f"""
        ALTER TABLE var 
        ADD COLUMN IF NOT EXISTS {add_key} BOOLEAN DEFAULT FALSE;
    """)

    print("开始聚合 X_CSR_data ...")

    # -------- 核心：CSR 聚合统计 --------
    rows = conn.execute("""
        SELECT 
            indices AS gene_index,
            SUM(data) AS sum_expr,
            COUNT(*) AS nonzero_expr
        FROM X_CSR_data
        GROUP BY indices
        ORDER BY indices
    """).fetchall()

    print(f"聚合完成，共统计到 {len(rows)} 个非零基因")

    # -------- 写入临时表 --------
    conn.execute("CREATE TEMP TABLE tmp_stats (gene_id TEXT, flag BOOLEAN)")

    keep_count = 0
    insert_rows = []

    # 收集已经出现的基因
    appeared_gene_ids = set()

    for gene_index, sum_expr, nonzero_expr in rows:

        # id → gene_id
        gene_id = gene_map.get(gene_index)
        if gene_id is None:
            continue

        appeared_gene_ids.add(gene_id)

        ok = True
        if min_counts is not None and sum_expr < min_counts:
            ok = False
        if max_counts is not None and sum_expr > max_counts:
            ok = False
        if min_cells is not None and nonzero_expr < min_cells:
            ok = False
        if max_cells is not None and nonzero_expr > max_cells:
            ok = False

        insert_rows.append((gene_id, ok))
        if ok:
            keep_count += 1

    # =======================================================
    # 👇 完全零表达的基因（不在 CSR 中出现）
    # =======================================================
    all_gene_ids = set(gene_map.values())
    zero_genes = all_gene_ids - appeared_gene_ids
    for g in zero_genes:
        insert_rows.append((g, False))

    # 写入临时表
    conn.executemany("INSERT INTO tmp_stats VALUES (?,?)", insert_rows)

    # 更新 var 表
    conn.execute(f"""
        UPDATE var
        SET {add_key} = tmp.flag
        FROM tmp_stats AS tmp
        WHERE var.gene_id = tmp.gene_id
    """)

    print(f"过滤完成: 保留基因 {keep_count} / 总 {n_genes}")
    print("总耗时: {:.2f} 秒".format((datetime.now() - start_time).total_seconds()))

#========== todo  计算每个细胞的总 UMI（Unique Molecular Identifier）计数 ==========
def calculate_cell_total_counts(atlas: 'Atlas', add_key: str = "cell_total_counts") -> None:
    """
    使用 DuckDB 原生 CSR 表（X_CSR_data）计算每个细胞的总 UMI 计数
    - 无 Python 循环
    - 无 AnnData
    - 低内存
    - 支持超大规模数据
    """

    print("==== DuckDB CSR 原生计算每个细胞的总 UMI ====")
    start = datetime.now()

    conn = atlas.connection

    # --------------------------------
    # DuckDB 性能参数
    # --------------------------------
    th = os.cpu_count()
    conn.execute(f"PRAGMA threads={th}")
    conn.execute("PRAGMA memory_limit='55GB'")
    conn.execute("PRAGMA temp_directory='.tmp_duckdb'")
    print(f"DuckDB threads = {th}")

    # --------------------------------
    # Step 0：确保 obs 有目标列
    # --------------------------------
    conn.execute(f"""
        ALTER TABLE obs
        ADD COLUMN IF NOT EXISTS {add_key} BIGINT
    """)

    # --------------------------------
    # Step 1：在 CSR 表上做流式聚合
    # --------------------------------
    print("统计每个细胞的 total UMI（SUM(data)）...")

    conn.execute("DROP TABLE IF EXISTS cell_total_counts_tmp")
    conn.execute(f"""
        CREATE TEMP TABLE cell_total_counts_tmp AS
        SELECT
            cell_index,
            SUM(data) AS total_counts
        FROM X_CSR_data
        GROUP BY cell_index
    """)

    # --------------------------------
    # Step 2：一次性更新 obs
    # --------------------------------
    print("更新 obs 表 ...")

    conn.execute(f"""
        UPDATE obs
        SET {add_key} = t.total_counts
        FROM cell_total_counts_tmp t
        WHERE obs.id = t.cell_index
    """)

    # --------------------------------
    # Step 3：统计信息
    # --------------------------------
    n_cells = conn.execute(
        "SELECT COUNT(*) FROM cell_total_counts_tmp"
    ).fetchone()[0]

    conn.execute("DROP TABLE IF EXISTS cell_total_counts_tmp")

    print(f"已计算细胞数 = {n_cells:,}")
    print("完成，总耗时 {:.2f} 秒".format(
        (datetime.now() - start).total_seconds()
    ))


# ========== todo  计算每个基因的表达值 ==========
def calculate_gene_total_counts(
                    atlas: 'Atlas',
                    add_key1: str = "gene_total_counts",
                    add_key2: str = "gene_mean_counts",
                    ) -> None:
    """
    使用 X_CSR_data 计算：
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
    logger.info("聚合 X_CSR_data（按 gene_index）...")

    conn.execute("DROP TABLE IF EXISTS gene_stats_tmp")
    conn.execute("""
        CREATE TEMP TABLE gene_stats_tmp AS
        SELECT
            indices AS gene_index,
            SUM(data) AS total_counts
        FROM X_CSR_data
        GROUP BY indices
    """)

    # -------------------------------------------------
    # Step 2: 写回 var（id == gene_index）
    # -------------------------------------------------
    logger.info("更新 var 表...")

    conn.execute(f"""
        UPDATE var
        SET
            {add_key1} = s.total_counts,
            {add_key2} = s.total_counts / {total_cells}
        FROM gene_stats_tmp AS s
        WHERE var.id = s.gene_index
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
        WHERE id NOT IN (SELECT gene_index FROM gene_stats_tmp)
    """)

    # -------------------------------------------------
    # Step 4: 清理临时表
    # -------------------------------------------------
    conn.execute("DROP TABLE gene_stats_tmp")

    logger.info("基因表达统计完成")
    logger.info("耗时 {:.2f} 秒".format((datetime.now() - start).total_seconds()))


# ========== todo 质量控制指标  +  线粒体基因比例计算 ==========
def calculate_qc_metrics(atlas: Atlas,
                        qc_prefix: str = "MT-",   # 线粒体基因名前缀，如 MT-CO1
                        qc_key: str = "mt"        # Scanpy 中 qc_vars=['mt']
                    ) -> None:
    """
    使用 DuckDB + CSR 稀疏存储，实现 Scanpy 的 calculate_qc_metrics
    X_CSR_data:
        - cell_index : 细胞 ID（obs.id）
        - indices    : 基因索引（var.id）
        - data       : 表达值（非零）
    var:
        - id         : gene_index（与 indices 对齐）
        - gene_id    : 真实基因名（如 MT-CO1）
    obs:
        - id         : cell_id
    """

    print("==== calculate_qc_metrics (CSR + DuckDB) ====")
    start = datetime.now()

    # DuckDB 连接
    conn = atlas.connection

    # =================================================
    # 0️⃣ 并行执行设置
    # =================================================
    # DuckDB 支持多线程 pipeline aggregation
    # 对 GROUP BY / JOIN 非常关键
    try:
        conn.execute(f"PRAGMA threads={os.cpu_count()}")
    except:
        pass

    # =================================================
    # 1️⃣ 在 var 表中生成 qc 标记列（如 mt）
    # =================================================
    # 等价 Scanpy：
    # adata.var['mt'] = adata.var_names.str.startswith("MT-")

    print(f"-> 标记 qc gene: {qc_key} (prefix='{qc_prefix}')")

    # 若不存在 qc_key（如 mt），先添加一列 BOOLEAN
    conn.execute(f"""
        ALTER TABLE var
        ADD COLUMN IF NOT EXISTS {qc_key} BOOLEAN
    """)

    # 根据 gene_id 前缀判断是否为 qc gene
    # SQL LIKE 'MT-%' 等价 Python startswith("MT-")
    conn.execute(f"""
        UPDATE var
        SET {qc_key} =
            CASE
                WHEN gene_id LIKE '{qc_prefix}%'
                THEN TRUE
                ELSE FALSE
            END
    """)

    # =================================================
    # 2️⃣ Cell-wise QC（每个细胞）
    # =================================================
    print("-> 计算 cell-wise QC")

    # -------------------------------------------------
    # 2.1 total_counts / n_genes_by_counts
    # -------------------------------------------------
    # Scanpy 对应：
    # adata.obs['total_counts'] = X.sum(axis=1)
    # adata.obs['n_genes_by_counts'] = (X > 0).sum(axis=1)

    # CSR 中：
    # - SUM(data)        → total_counts  每个 cell 的总 counts
    # - COUNT(*)         → 非零基因数
    conn.execute("""
        CREATE OR REPLACE TEMP TABLE _cell_basic AS
        SELECT
            cell_index AS id,
            SUM(data)  AS total_counts,
            COUNT(*)   AS n_genes_by_counts
        FROM X_CSR_data
        WHERE data IS NOT NULL
        GROUP BY cell_index
    """)

    # 给 obs 表准备字段
    conn.execute("""
        ALTER TABLE obs
        ADD COLUMN IF NOT EXISTS total_counts REAL
    """)
    conn.execute("""
        ALTER TABLE obs
        ADD COLUMN IF NOT EXISTS n_genes_by_counts INTEGER
    """)

    # 把聚合结果写回 obs
    conn.execute("""
        UPDATE obs
        SET
            total_counts = c.total_counts,  -- 每个 cell 的总 counts
            n_genes_by_counts = c.n_genes_by_counts  -- 每个 cell 中 非零基因数
        FROM _cell_basic c
        WHERE obs.id = c.id
    """)

    # -------------------------------------------------
    # 2.2 total_counts_mt / pct_counts_mt
    # -------------------------------------------------
    # Scanpy 对应：
    # adata.obs['total_counts_mt']
    # adata.obs['pct_counts_mt']

    # 做法：
    # X_CSR_data → JOIN var → 只保留 mt 基因 → cell-wise SUM
    conn.execute(f"""
        CREATE OR REPLACE TEMP TABLE _cell_qc AS
        SELECT
            x.cell_index AS id,
            SUM(x.data)  AS total_counts_qc
        FROM X_CSR_data x
        JOIN var v
          ON x.indices = v.id
        WHERE v.{qc_key} = TRUE
        GROUP BY x.cell_index
    """)

    # obs 中增加字段
    conn.execute("""
        ALTER TABLE obs
        ADD COLUMN IF NOT EXISTS total_counts_mt REAL
    """)
    conn.execute("""
        ALTER TABLE obs
        ADD COLUMN IF NOT EXISTS pct_counts_mt REAL
    """)

    # 写回 obs
    # pct = total_counts_mt / total_counts * 100 , 线粒体基因的百分比
    conn.execute("""
        UPDATE obs
        SET
            total_counts_mt = COALESCE(q.total_counts_qc, 0),  -- 线粒体基因 counts 之和
            pct_counts_mt =
                CASE
                    WHEN obs.total_counts > 0
                    THEN 100.0 * COALESCE(q.total_counts_qc, 0) / obs.total_counts
                    ELSE 0
                END
        FROM _cell_qc q
        WHERE obs.id = q.id
    """)

    # =================================================
    # 3️⃣ Gene-wise QC（每个基因）
    # =================================================
    print("-> 计算 gene-wise QC")

    # Scanpy 对应：
    # adata.var['total_counts'] = X.sum(axis=0)
    # adata.var['n_cells_by_counts'] = (X > 0).sum(axis=0)

    conn.execute("""
        CREATE OR REPLACE TEMP TABLE _gene_qc AS
        SELECT
            indices AS id,
            SUM(data) AS total_counts,
            COUNT(DISTINCT cell_index) AS n_cells_by_counts
        FROM X_CSR_data
        WHERE data IS NOT NULL
        GROUP BY indices
    """)

    # var 表中增加字段
    conn.execute("""
        ALTER TABLE var
        ADD COLUMN IF NOT EXISTS total_counts REAL
    """)
    conn.execute("""
        ALTER TABLE var
        ADD COLUMN IF NOT EXISTS n_cells_by_counts INTEGER
    """)

    # 写回 var
    conn.execute("""
        UPDATE var
        SET
            total_counts = g.total_counts,  -- 该 gene 在所有 cells 的 counts 之和
            n_cells_by_counts = g.n_cells_by_counts  -- 有多少 cell 表达该 gene（非零）
        FROM _gene_qc g
        WHERE var.id = g.id
    """)

    # =================================================
    # 结束
    # =================================================
    print("calculate_qc_metrics 完成")
    print("耗时: {:.2f} 秒".format((datetime.now() - start).total_seconds()))

# 运行结果
# var 表  新增字段
#    字段	                    含义
# mt	              gene_id 是否以 MT- 开头
# total_counts	       该 gene 在所有 cells 的 counts 之和
# n_cells_by_counts	     非零 cell 数 ：有多少 cell 表达该 gene（非零）

# obs 表  新增字段
#    字段	                   含义
# total_counts	            每个 cell 的总 counts
# n_genes_by_counts       	每个 cell 的 非零基因数
# total_counts_mt	        线粒体基因 counts 之和
# pct_counts_mt           	线粒体基因的百分比
