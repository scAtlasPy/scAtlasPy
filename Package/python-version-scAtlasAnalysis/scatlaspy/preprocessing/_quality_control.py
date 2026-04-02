import os
from datetime import datetime
from ..data import Atlas
from typing import Optional
import logging
# 获取日志记录器
logger = logging.getLogger('Atlas')


# ========== todo 过滤细胞 ：  使用 mmap 扫描 CSR 过滤细胞（安全支持 1 亿细胞） ==========
# 833206 * 17745  sap.pp.filter_cells(atlas, min_genes=200) 过滤细胞 = 0 (0.00%) 总耗时 0.54 秒
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
            FROM X_CSR_data
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

# ========== todo 过滤基因 ：行-批 + NumPy 聚合 + 批写回 var ，当前最快 ==========
# 833206 * 17745  sap.pp.filter_genes(atlas, min_cells=3)  保留基因 17745  耗时: 3.35 秒
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

    gene_map = dict(conn.execute("SELECT atlas_gene_id, atlas_gene_name FROM var").fetchall())
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
            atlas_gene_id,
            SUM(data) AS sum_expr,
            COUNT(*) AS nonzero_expr
        FROM X_CSR_data
        GROUP BY atlas_gene_id
        ORDER BY atlas_gene_id
    """).fetchall()

    print(f"聚合完成，共统计到 {len(rows)} 个非零基因")

    # -------- 写入临时表 --------
    conn.execute("CREATE TEMP TABLE tmp_stats (atlas_gene_name TEXT, flag BOOLEAN)")

    keep_count = 0
    insert_rows = []

    # 收集已经出现的基因
    appeared_gene_names = set()

    for atlas_gene_id, sum_expr, nonzero_expr in rows:

        # atlas_gene_id → atlas_gene_name
        atlas_gene_name = gene_map.get(atlas_gene_id)
        if atlas_gene_name is None:
            continue

        appeared_gene_names.add(atlas_gene_name)

        ok = True
        if min_counts is not None and sum_expr < min_counts:
            ok = False
        if max_counts is not None and sum_expr > max_counts:
            ok = False
        if min_cells is not None and nonzero_expr < min_cells:
            ok = False
        if max_cells is not None and nonzero_expr > max_cells:
            ok = False

        insert_rows.append((atlas_gene_name, ok))
        if ok:
            keep_count += 1

    # 完全零表达的基因（不在 CSR 中出现）
    all_gene_names = set(gene_map.values())
    zero_genes = all_gene_names - appeared_gene_names
    for g in zero_genes:
        insert_rows.append((g, False))

    # 写入临时表
    conn.executemany("INSERT INTO tmp_stats VALUES (?,?)", insert_rows)

    # 先全部设为 FALSE（保证覆盖）
    conn.execute(f"UPDATE var SET {add_key}=FALSE")

    # 更新 var 表
    conn.execute(f"""
        UPDATE var
        SET {add_key} = tmp.flag
        FROM tmp_stats AS tmp
        WHERE var.atlas_gene_name = tmp.atlas_gene_name
    """)

    # 删除临时表
    conn.execute("DROP TABLE IF EXISTS tmp_stats")

    print(f"过滤完成: 保留基因 {keep_count} / 总 {n_genes}")
    print("总耗时: {:.2f} 秒".format((datetime.now() - start_time).total_seconds()))

# 运行结果
# var 表  新增字段
#     字段	                    含义
#  filter_genes	       该 gene是否符合过滤条件 true


#========== todo  计算每个细胞的总 UMI（Unique Molecular Identifier）计数 ==========
# 833206 * 17745 耗时 1.58 秒
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
            atlas_cell_id,
            SUM(data) AS total_counts
        FROM X_CSR_data
        GROUP BY atlas_cell_id
    """)

    # --------------------------------
    # Step 2：一次性更新 obs
    # --------------------------------
    print("更新 obs 表 ...")

    conn.execute(f"""
        UPDATE obs
        SET {add_key} = t.total_counts
        FROM cell_total_counts_tmp t
        WHERE obs.atlas_cell_id = t.atlas_cell_id
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

# 运行结果
# obs 表  新增字段
#     字段	                    含义
#  cell_total_counts	     每个细胞的总 UMI 计数

# ========== todo  计算每个基因的表达值 ==========
# 833206 * 17745 耗时 0.77 秒
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
    logger.info("聚合 X_CSR_data（按 atlas_gene_id）...")

    conn.execute("DROP TABLE IF EXISTS gene_stats_tmp")
    conn.execute("""
        CREATE TEMP TABLE gene_stats_tmp AS
        SELECT
            atlas_gene_id,
            SUM(data) AS total_counts
        FROM X_CSR_data
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


# ========== todo 质量控制指标  +  线粒体基因比例计算 ==========
#  833206 * 17745 23.85 秒
def calculate_qc_metrics(atlas: Atlas,
                        qc_prefix: str = "MT-",   # 线粒体基因名前缀，如 MT-CO1
                        qc_key: str = "mt"        # Scanpy 中 qc_vars=['mt']
                    ) -> None:
    # todo 补充 过滤细胞的额外条件：mitochondrial gene和ribosomal gene比例过高的细胞会被过滤
    #  “ mt” 和 ^RP[SL] 补充
    """
    使用 DuckDB + CSR 稀疏存储，实现 Scanpy 的 calculate_qc_metrics
    X_CSR_data:
        - atlas_cell_id : 细胞 ID（obs.id）
        - atlas_gene_id : 基因索引（var.id）
        - data       : 表达值（非零）
    var:
        - atlas_gene_id    :
        - atlas_gene_name    : 真实基因名（如 MT-CO1）
    obs:
        - atlas_cell_id
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

    # 根据 atlas_gene_name  前缀判断是否为 qc gene
    # SQL LIKE 'MT-%' 等价 Python startswith("MT-")
    conn.execute(f"""
        UPDATE var
        SET {qc_key} =
            CASE
                WHEN atlas_gene_name LIKE '{qc_prefix}%'
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
            atlas_cell_id,
            SUM(data)  AS cell_total_counts,
            COUNT(*)   AS n_genes_by_counts
        FROM X_CSR_data
        WHERE data IS NOT NULL
        GROUP BY atlas_cell_id
    """)

    # 给 obs 表准备字段
    conn.execute("""
        ALTER TABLE obs
        ADD COLUMN IF NOT EXISTS cell_total_counts REAL
    """)
    conn.execute("""
        ALTER TABLE obs
        ADD COLUMN IF NOT EXISTS n_genes_by_counts INTEGER
    """)

    # 把聚合结果写回 obs
    conn.execute("""
        UPDATE obs
        SET
            cell_total_counts = c.cell_total_counts, -- 每个 cell 的总 counts
            n_genes_by_counts = c.n_genes_by_counts  -- 每个 cell 中 非零基因数
        FROM _cell_basic c
        WHERE obs.atlas_cell_id = c.atlas_cell_id
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
            x.atlas_cell_id AS atlas_cell_id,
            SUM(x.data)  AS total_counts_qc
        FROM X_CSR_data x
        JOIN var v
          ON x.atlas_gene_id = v.atlas_gene_id
        WHERE v.{qc_key} = TRUE
        GROUP BY x.atlas_cell_id
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
    # pct = total_counts_mt / cell_total_counts * 100 , 线粒体基因的百分比
    conn.execute("""
        UPDATE obs
        SET
            total_counts_mt = COALESCE(q.total_counts_qc, 0),  -- 线粒体基因 counts 之和
            pct_counts_mt =
                CASE
                    WHEN obs.cell_total_counts > 0
                    THEN 100.0 * COALESCE(q.total_counts_qc, 0) / obs.cell_total_counts
                    ELSE 0
                END
        FROM _cell_qc q
        WHERE obs.atlas_cell_id = q.atlas_cell_id
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
            atlas_gene_id,
            SUM(data) AS gene_total_counts,
            COUNT ( DISTINCT atlas_cell_id ) AS n_cells_by_counts
        FROM X_CSR_data
        WHERE data IS NOT NULL
        GROUP BY atlas_gene_id
    """)

    # var 表中增加字段
    conn.execute("""
        ALTER TABLE var
        ADD COLUMN IF NOT EXISTS gene_total_counts REAL
    """)
    conn.execute("""
        ALTER TABLE var
        ADD COLUMN IF NOT EXISTS n_cells_by_counts INTEGER
    """)

    # 写回 var
    conn.execute("""
        UPDATE var
        SET
            gene_total_counts = g.gene_total_counts, -- 该 gene 在所有 cells 的 counts 之和
            n_cells_by_counts = g.n_cells_by_counts  -- 有多少 cell 表达该 gene（非零）
        FROM _gene_qc g
        WHERE var.atlas_gene_id = g.atlas_gene_id
    """)

    # 删除临时表
    conn.execute("DROP TABLE IF EXISTS _cell_basic")
    conn.execute("DROP TABLE IF EXISTS _cell_qc")
    conn.execute("DROP TABLE IF EXISTS _gene_qc")

    # =================================================
    # 结束
    # =================================================
    print("calculate_qc_metrics 完成")
    print("耗时: {:.2f} 秒".format((datetime.now() - start).total_seconds()))

# 运行结果
# var 表  新增字段
#     字段	                    含义
#  mt	                 atlas_gene_id 是否以 MT- 开头
#  gene_total_counts	 该 gene 在所有 cells 的 counts 之和
#  n_cells_by_counts	 非零 cell 数 ：有多少 cell 表达该 gene（非零）

# obs 表  新增字段
#     字段	                   含义
#  cell_total_counts	    每个 cell 的总 counts
#  n_genes_by_counts       	每个 cell 的 非零基因数
#  total_counts_mt	        线粒体基因 counts 之和
#  pct_counts_mt           	线粒体基因的百分比

# todo 补充核糖体基因
def calculate_qc_metrics1(atlas: Atlas,
                         qc_vars: dict = {
                             "mt": "MT-",
                             "ribo": "^RP[SL]"
                         }
                    ) -> None:
    """
    CSR + DuckDB 实现 Scanpy calculate_qc_metrics（支持多个 qc_vars）

    qc_vars 示例：
        {
            "mt": "MT-",
            "ribo": "^RP[SL]"
        }
    """

    print("==== calculate_qc_metrics (CSR + DuckDB) ====")
    start = datetime.now()

    conn = atlas.connection

    # =================================================
    # 0️⃣ 默认 qc_vars（对齐 scanpy）
    # =================================================
    if qc_vars is None:
        qc_vars = {
            "mt": "MT-",
            "ribo": "^RP[SL]"
        }

    # =================================================
    # 1️⃣ 并行设置
    # =================================================
    try:
        conn.execute(f"PRAGMA threads={os.cpu_count()}")
    except:
        pass

    # =================================================
    # 2️⃣ 在 var 中打 qc 标记（支持多个）
    # =================================================
    for qc_key, pattern in qc_vars.items():

        print(f"-> 标记 qc gene: {qc_key} ({pattern})")

        conn.execute(f"""
            ALTER TABLE var
            ADD COLUMN IF NOT EXISTS {qc_key} BOOLEAN
        """)

        # 区分 prefix / regex
        if pattern.startswith("^"):
            # regex
            conn.execute(f"""
                UPDATE var
                SET {qc_key} =
                    CASE
                        WHEN atlas_gene_name ~ '{pattern}'
                        THEN TRUE
                        ELSE FALSE
                    END
            """)
        else:
            # prefix（兼容旧逻辑）
            conn.execute(f"""
                UPDATE var
                SET {qc_key} =
                    CASE
                        WHEN atlas_gene_name LIKE '{pattern}%'
                        THEN TRUE
                        ELSE FALSE
                    END
            """)

    # =================================================
    # 3️⃣ Cell-wise QC
    # =================================================
    print("-> 计算 cell-wise QC")

    conn.execute("""
        CREATE OR REPLACE TEMP TABLE _cell_basic AS
        SELECT
            atlas_cell_id,
            SUM(data)  AS cell_total_counts,
            COUNT(*)   AS n_genes_by_counts
        FROM X_CSR_data
        WHERE data IS NOT NULL
        GROUP BY atlas_cell_id
    """)

    conn.execute("""
        ALTER TABLE obs
        ADD COLUMN IF NOT EXISTS cell_total_counts REAL
    """)
    conn.execute("""
        ALTER TABLE obs
        ADD COLUMN IF NOT EXISTS n_genes_by_counts INTEGER
    """)

    conn.execute("""
        UPDATE obs
        SET
            cell_total_counts = c.cell_total_counts,
            n_genes_by_counts = c.n_genes_by_counts
        FROM _cell_basic c
        WHERE obs.atlas_cell_id = c.atlas_cell_id
    """)

    # =================================================
    # 4️⃣ 每个 qc_var 单独计算（最小修改方式）
    # =================================================
    for qc_key in qc_vars.keys():

        print(f"-> 计算 {qc_key} QC")

        conn.execute(f"""
            CREATE OR REPLACE TEMP TABLE _cell_{qc_key} AS
            SELECT
                x.atlas_cell_id,
                SUM(x.data) AS total_counts_qc
            FROM X_CSR_data x
            JOIN var v
              ON x.atlas_gene_id = v.atlas_gene_id
            WHERE v.{qc_key} = TRUE
            GROUP BY x.atlas_cell_id
        """)

        # 新字段
        conn.execute(f"""
            ALTER TABLE obs
            ADD COLUMN IF NOT EXISTS total_counts_{qc_key} REAL
        """)
        conn.execute(f"""
            ALTER TABLE obs
            ADD COLUMN IF NOT EXISTS pct_counts_{qc_key} REAL
        """)

        conn.execute(f"""
            UPDATE obs
            SET
                total_counts_{qc_key} = COALESCE(q.total_counts_qc, 0),
                pct_counts_{qc_key} =
                    CASE
                        WHEN obs.cell_total_counts > 0
                        THEN 100.0 * COALESCE(q.total_counts_qc, 0) / obs.cell_total_counts
                        ELSE 0
                    END
            FROM _cell_{qc_key} q
            WHERE obs.atlas_cell_id = q.atlas_cell_id
        """)

    # =================================================
    # 5️⃣ Gene-wise QC（不变）
    # =================================================
    print("-> 计算 gene-wise QC")

    conn.execute("""
        CREATE OR REPLACE TEMP TABLE _gene_qc AS
        SELECT
            atlas_gene_id,
            SUM(data) AS gene_total_counts,
            COUNT(DISTINCT atlas_cell_id) AS n_cells_by_counts
        FROM X_CSR_data
        WHERE data IS NOT NULL
        GROUP BY atlas_gene_id
    """)

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

    # =================================================
    # 6️⃣ 清理
    # =================================================
    conn.execute("DROP TABLE IF EXISTS _cell_basic")
    conn.execute("DROP TABLE IF EXISTS _gene_qc")

    for qc_key in qc_vars.keys():
        conn.execute(f"DROP TABLE IF EXISTS _cell_{qc_key}")

    print("calculate_qc_metrics 完成")
    print("耗时: {:.2f} 秒".format((datetime.now() - start).total_seconds()))


#     HVG基因过滤
#