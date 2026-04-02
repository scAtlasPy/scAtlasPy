from ..data import Atlas
from typing import Literal
from typing import Optional
import logging
from datetime import datetime
from numbers import Number
import os
import math
# 获取日志记录器
logger = logging.getLogger('Atlas')

'''=== 归一化 normalize 法 1 ： 分块， 在 X_CSRO_data 表上直接处理 ， 不替换原数据  =========='''
#  833206 * 17745  sap.pp.normalize_total(atlas)  49.41 秒
def normalize_total(
        atlas,
        target_sum: float = 10000,
        chunk_size: int = 50_000_000,
        add_field: str = "data_normalize",
        select_data: str = "data"
) -> None:

    print("==== normalize_total_streaming ====")
    start = datetime.now()

    conn = atlas.connection

    # -------------------------------------------------
    # 0. 设置线程（可选）
    # -------------------------------------------------
    try:
        conn.execute(f"PRAGMA threads={os.cpu_count()}")
    except:
        pass

    # -------------------------------------------------
    # 1. 字段检查
    # -------------------------------------------------
    col_exists = conn.execute(f"""
        SELECT COUNT(*)
        FROM information_schema.columns
        WHERE table_name = 'X_CSRO_data'
          AND column_name = '{select_data}'
    """).fetchone()[0]

    if col_exists == 0:
        raise ValueError(f"X_CSRO_data 中不存在字段: {select_data}")

    # -------------------------------------------------
    # 2. 计算每个 cell 的总表达（只做一次）
    # -------------------------------------------------
    print("Step 1: compute cell sums")

    conn.execute(f"""
        CREATE OR REPLACE TEMP TABLE _cell_sum AS
        SELECT
            atlas_cell_id,
            SUM({select_data}) AS total
        FROM X_CSRO_data
        GROUP BY atlas_cell_id
        HAVING total > 0
    """)

    # -------------------------------------------------
    # 🔥 关键修改：先删除源表中的旧列（保证 x.* 不冲突）
    # -------------------------------------------------
    print("Step 1.5: drop old column in source table")

    conn.execute(f"""
        ALTER TABLE X_CSRO_data
        DROP COLUMN IF EXISTS {add_field}
    """)

    # -------------------------------------------------
    # 3. 创建目标表（结构复制 + 新列）
    # -------------------------------------------------
    print("Step 2: create target table")

    conn.execute("""
        CREATE OR REPLACE TABLE X_CSR_data_norm AS
        SELECT * FROM X_CSRO_data WHERE 1=0
    """)

    # 现在源表已经没有该列 → 可以安全添加
    conn.execute(f"""
        ALTER TABLE X_CSR_data_norm
        ADD COLUMN {add_field} DOUBLE
    """)

    # -------------------------------------------------
    # 4. 获取 id 范围
    # -------------------------------------------------
    min_id, max_id = conn.execute("""
        SELECT MIN(id), MAX(id)
        FROM X_CSRO_data
    """).fetchone()

    if min_id is None:
        print("X_CSRO_data 为空，跳过")
        return

    n_chunks = math.ceil((max_id - min_id + 1) / chunk_size)
    print(f"Step 3: {n_chunks} chunks")

    # -------------------------------------------------
    # 5. 分块 INSERT（保持 x.*）
    # -------------------------------------------------
    for i in range(n_chunks):
        start_id = min_id + i * chunk_size
        end_id = start_id + chunk_size - 1

        print(f"  -> chunk {i+1}/{n_chunks}: id [{start_id}, {end_id}]")

        conn.execute(f"""
            INSERT INTO X_CSR_data_norm
            SELECT
                x.*,
                x.{select_data} * {float(target_sum)} / s.total AS {add_field}
            FROM X_CSRO_data x
            JOIN _cell_sum s
              ON x.atlas_cell_id = s.atlas_cell_id
            WHERE x.id BETWEEN {start_id} AND {end_id}
        """)

    # -------------------------------------------------
    # 6. 替换原表
    # -------------------------------------------------
    print("Step 4: replace table")

    conn.execute("DROP TABLE X_CSRO_data")
    conn.execute("ALTER TABLE X_CSR_data_norm RENAME TO X_CSRO_data")

    # 清理临时表（建议）
    conn.execute("DROP TABLE IF EXISTS _cell_sum")

    print("normalize_total_streaming 完成")
    print("耗时: {:.2f} 秒".format((datetime.now() - start).total_seconds()))

# 运行结果
#   X_CSRO_data 表
#     新增字段	             含义
#  data_normalize	     归一化后的data值


'''=== 归一化 normalize 法 2： 参照scanpy，在 obs表上记录 scale_factor ， 等到使用的时候在计算 ==========='''
# 833206 * 17745  sap.pp.normalize_total_scale_factor(atlas) 0.86 秒
def normalize_total_scale_factor(
                    atlas: Atlas,
                    target_sum: float = 10000,
                    add_key: str = "scale_factor",
                    select_data: str = "data" ) -> None:
    """
    高性能 normalize_total（Scanpy 等价）：
    - 不修改 X_CSRO_data
    - 只计算每个 cell 的 scale_factor
    - 支持指定使用 X_CSRO_data 中的任意字段作为表达值

    Args:
        atlas: Atlas 对象
        target_sum: 归一化目标和；
        add_key: 写入 obs 的 scale_factor 字段名
        select_data: X_CSRO_data 中用于计算 total 的字段名
                     （如 'data', 'data_normalized', 'X_log1p'）
    """

    print("==== normalize_total (scale_factor only) ====")
    start = datetime.now()

    conn = atlas.connection
    try:
        conn.execute(f"PRAGMA threads={os.cpu_count()}")
    except:
        pass

    # 0. 基本安全检查
    col_exists = conn.execute(f"""
        SELECT COUNT(*)
        FROM information_schema.columns
        WHERE table_name = 'X_CSRO_data'
          AND column_name = '{select_data}'
    """).fetchone()[0]

    if col_exists == 0:
        raise ValueError(f"X_CSRO_data 中不存在字段: {select_data}")

    # 1. 计算每个 cell 的 total
    conn.execute(f"""
        CREATE OR REPLACE TEMP TABLE _cell_sum AS
        SELECT
            atlas_cell_id,
            SUM({select_data}) AS total
        FROM X_CSRO_data
        GROUP BY atlas_cell_id
        ORDER BY atlas_cell_id
    """)

    # 3. 在 obs 中写 scale_factor
    conn.execute(f"""
        ALTER TABLE obs
        ADD COLUMN IF NOT EXISTS {add_key} REAL
    """)

    conn.execute(f"""
        UPDATE obs
        SET {add_key} =
            CASE
                WHEN s.total > 0
                THEN {float(target_sum)} / s.total
                ELSE 0
            END
        FROM _cell_sum AS s
        WHERE obs.atlas_cell_id = s.atlas_cell_id
    """)

    print(f"normalize_total 完成，target_sum={target_sum}")
    print("耗时: {:.2f} 秒".format((datetime.now() - start).total_seconds()))

# 运行结果
#   obs 表
#     新增字段	             含义
#   scale_factor	     scale_factor，等到使用的时候在计算，data * scale_factor ，即可


'''===  log1p_fast  法 1 ： 不分块 ， 大数据不安全  ========== '''
# 833206 * 17745    2.04秒 只适用于小数据
def log1p_fast(
            atlas: 'Atlas',
            base: Optional[Number] = None,
            add_field: str = "data_log1p",
            select_data: str = "data" ) -> None:
    """
    对表达值进行 log(1+x) 转换
    - 支持选择 X_CSRO_data 中任意字段进行计算
    """

    logger.info("开始执行 log(1+x) 转换...")
    print("==== log1p ====")
    start = datetime.now()

    try:
        atlas.connection.execute(f"PRAGMA threads={os.cpu_count()}")
    except:
        pass

    conn = atlas.connection

    # 0. 字段存在性检查
    col_exists = conn.execute(f"""
        SELECT COUNT(*)
        FROM information_schema.columns
        WHERE table_name = 'X_CSRO_data'
          AND column_name = '{select_data}'
    """).fetchone()[0]

    if col_exists == 0:
        raise ValueError(f"X_CSRO_data 中不存在字段: {select_data}")

    # 1. 构造 log1p 表达式
    if base is None:
        log_expr = f"ln(1.0 + {select_data})"
    else:
        log_expr = f"log({float(base)}, 1.0 + {select_data})"

    if add_field is None:
        raise ValueError("必须指定 add_field")

    print(f"创建新字段 X_CSRO_data.{add_field} ...")

    # 2. 确保输出字段存在
    conn.execute(f"""
        ALTER TABLE X_CSRO_data
        ADD COLUMN IF NOT EXISTS {add_field} REAL
    """)

    # 3. 执行 log1p
    conn.execute(f"""
        UPDATE X_CSRO_data
        SET {add_field} = {log_expr}
        WHERE {select_data} IS NOT NULL
    """)

    # 4. 结束
    print("log1p 完成")
    print("耗时: {:.2f} 秒".format((datetime.now() - start).total_seconds()))

# 运行结果
#   X_CSRO_data 表
#     新增字段	             含义
#   data_log1p	     对表达值进行 log(1+x) 转换


'''===  log1p  法 2 ： 分块  大数据安全 ========== '''
# 833206 * 17745  2.65 秒
def log1p(
                atlas: 'Atlas',
                base: Optional[Number] = None,
                add_field: str = "data_log1p",
                select_data: str = "data",
                chunk_size: int = 100_000_000) -> None:
    """
    1e8 级 CSR 安全的 log1p 实现
    - 不一次性 UPDATE 全表
    - 按 id 分块
    - 支持指定 X_CSRO_data 中的任意字段作为输入
    """

    logger.info("开始执行 log1p (chunked)...")
    print("==== log1p (chunked) ====")
    start = datetime.now()

    conn = atlas.connection

    try:
        conn.execute(f"PRAGMA threads={os.cpu_count()}")
    except:
        pass

    # -------------------------------------------------
    # 0. 字段存在性检查（重要）
    # -------------------------------------------------
    col_exists = conn.execute(f"""
        SELECT COUNT(*)
        FROM information_schema.columns
        WHERE table_name = 'X_CSRO_data'
          AND column_name = '{select_data}'
    """).fetchone()[0]

    if col_exists == 0:
        raise ValueError(f"X_CSRO_data 中不存在字段: {select_data}")

    # -------------------------------------------------
    # 1. 构造 log 表达式
    # -------------------------------------------------
    if base is None:
        log_expr = f"ln(1.0 + {select_data})"
    else:
        log_expr = f"log({float(base)}, 1.0 + {select_data})"

    # -------------------------------------------------
    # 2. 确保输出字段存在
    # -------------------------------------------------
    conn.execute(f"""
        ALTER TABLE X_CSRO_data
        ADD COLUMN IF NOT EXISTS {add_field} REAL
    """)

    # -------------------------------------------------
    # 3. 获取 id 范围
    # -------------------------------------------------
    min_id, max_id = conn.execute("""
        SELECT MIN(id), MAX(id)
        FROM X_CSRO_data
    """).fetchone()

    if min_id is None:
        print("X_CSRO_data 为空，跳过")
        return

    n_chunks = math.ceil((max_id - min_id + 1) / chunk_size)
    print(f"共 {n_chunks} 个 chunk")

    # -------------------------------------------------
    # 4. 分块 UPDATE
    # -------------------------------------------------
    for i in range(n_chunks):
        start_id = min_id + i * chunk_size
        end_id = start_id + chunk_size - 1

        print(f"  -> chunk {i+1}/{n_chunks}: id [{start_id}, {end_id}]")

        conn.execute(f"""
            UPDATE X_CSRO_data
            SET {add_field} = {log_expr}
            WHERE id BETWEEN {start_id} AND {end_id}
              AND {select_data} IS NOT NULL
        """)

    # -------------------------------------------------
    # 5. 结束
    # -------------------------------------------------
    print("log1p (chunked) 完成")
    print("耗时: {:.2f} 秒".format((datetime.now() - start).total_seconds()))

# 运行结果
#   X_CSRO_data 表
#     新增字段	             含义
#   data_log1p	     对表达值进行 log(1+x) 转换


'''===== exp1 是  log1p的逆运算 ========== '''
# 833206 * 17745  2.46 秒 运行完别的函数再运行这个， 16.65 秒，
def exp1(
        atlas: 'Atlas',
        base: Optional[Number] = None,
        add_field: str = "data_exp1",
        select_data: str = "data_log1p",
        chunk_size: int = 100_000_000 ) -> None:
    """
    log1p_chunked 的逆运算：exp(x) - 1
    - ln(1+x)  → exp(x) - 1
    - log_b(1+x) → b^x - 1
    - 按 X_CSRO_data.id 分块，支持 1e8 CSR
    """

    logger.info("开始执行 exp1 (chunked)...")
    print("==== exp1 (chunked) ====")
    start = datetime.now()

    conn = atlas.connection

    try:
        conn.execute(f"PRAGMA threads={os.cpu_count()}")
    except:
        pass

    # -------------------------------------------------
    # 0. 字段存在性检查
    # -------------------------------------------------
    col_exists = conn.execute(f"""
        SELECT COUNT(*)
        FROM information_schema.columns
        WHERE table_name = 'X_CSRO_data'
          AND column_name = '{select_data}'
    """).fetchone()[0]

    if col_exists == 0:
        raise ValueError(f"X_CSRO_data 中不存在字段: {select_data}")

    # -------------------------------------------------
    # 1. 构造 exp 表达式
    # -------------------------------------------------
    if base is None:
        exp_expr = f"exp({select_data}) - 1.0"
    else:
        exp_expr = f"pow({float(base)}, {select_data}) - 1.0"

    # -------------------------------------------------
    # 2. 确保输出字段存在
    # -------------------------------------------------
    conn.execute(f"""
        ALTER TABLE X_CSRO_data
        ADD COLUMN IF NOT EXISTS {add_field} REAL
    """)

    # -------------------------------------------------
    # 3. 获取 id 范围
    # -------------------------------------------------
    min_id, max_id = conn.execute("""
        SELECT MIN(id), MAX(id)
        FROM X_CSRO_data
    """).fetchone()

    if min_id is None:
        print("X_CSRO_data 为空，跳过")
        return

    n_chunks = math.ceil((max_id - min_id + 1) / chunk_size)
    print(f"共 {n_chunks} 个 chunk")

    # -------------------------------------------------
    # 4. 分块 UPDATE
    # -------------------------------------------------
    for i in range(n_chunks):
        start_id = min_id + i * chunk_size
        end_id = start_id + chunk_size - 1

        print(f"  -> chunk {i+1}/{n_chunks}: id [{start_id}, {end_id}]")

        conn.execute(f"""
            UPDATE X_CSRO_data
            SET {add_field} = {exp_expr}
            WHERE id BETWEEN {start_id} AND {end_id}
              AND {select_data} IS NOT NULL
        """)

    # -------------------------------------------------
    # 5. 结束
    # -------------------------------------------------
    print("exp1 (chunked) 完成")
    print("耗时: {:.2f} 秒".format((datetime.now() - start).total_seconds()))

# 运行结果
#   X_CSRO_data 表
#     新增字段	             含义
#   data_exp1	     对表达值进行 log(1+x) 转换 的 还原


'''===== normalize_and_log1p： normalize 法 3 + log1p 法 2 ===='''
# 833206 * 17745     9.34 秒 再次运行  3.43 秒
def normalize_and_log1p(
            atlas: Atlas,
            target_sum: Optional[float] = 10000,
            scale_key: str = "scale_factor",
            add_field: str = "data_log1p",
            select_data: str = "data",
            base: Optional[Number] = None,
            chunk_size: int = 100_000_000 ) -> None:
    """
    Scanpy 等价的 normalize_total + log1p
    - normalize_total：只计算 scale_factor（obs）
    - log1p：log(1 + select_data * scale_factor)
    - 按 X_CSRO_data.id 分块（✔ 正确的 1e8 CSR 方式）
    """

    print("==== normalize_and_log1p (Scanpy-equivalent) ====")
    start = datetime.now()

    conn = atlas.connection
    try:
        conn.execute(f"PRAGMA threads={os.cpu_count()}")
    except:
        pass

    # =================================================
    # 0. 字段存在性检查（防止 silent bug）
    # =================================================
    col_exists = conn.execute(f"""
        SELECT COUNT(*)
        FROM information_schema.columns
        WHERE table_name = 'X_CSRO_data'
          AND column_name = '{select_data}'
    """).fetchone()[0]

    if col_exists == 0:
        raise ValueError(f"X_CSRO_data 中不存在字段: {select_data}")

    # =================================================
    # 1. 调用上面的函数 normalize_total → 计算 scale_factor
    # =================================================
    normalize_total_scale_factor(
        atlas=atlas,
        target_sum=target_sum,
        add_key=scale_key
    )

    # =================================================
    # 2. 构造 log 表达式（🔥 改这里）
    # =================================================
    if base is None:
        log_expr = f"ln(1.0 + x.{select_data} * o.{scale_key})"
    else:
        log_expr = f"log({float(base)}, 1.0 + x.{select_data} * o.{scale_key})"

    # =================================================
    # 3. 准备输出字段
    # =================================================
    conn.execute(f"""
        ALTER TABLE X_CSRO_data
        ADD COLUMN IF NOT EXISTS {add_field} REAL
    """)

    # =================================================
    # 4. 获取 X_CSRO_data.id 范围
    # =================================================
    min_id, max_id = conn.execute("""
        SELECT MIN(id), MAX(id)
        FROM X_CSRO_data
    """).fetchone()

    if min_id is None:
        print("X_CSRO_data 为空，跳过")
        return

    n_chunks = math.ceil((max_id - min_id + 1) / chunk_size)
    print(f"log1p 分 {n_chunks} 个 id chunk")

    # =================================================
    # 5. 分块 UPDATE（核心）
    # =================================================
    for i in range(n_chunks):
        start_id = min_id + i * chunk_size
        end_id   = start_id + chunk_size - 1

        print(f"  -> chunk {i+1}/{n_chunks}: id [{start_id}, {end_id}]")

        conn.execute(f"""
            UPDATE X_CSRO_data AS x
            SET {add_field} = {log_expr}
            FROM obs AS o
            WHERE x.atlas_cell_id = o.atlas_cell_id
              AND x.id BETWEEN {start_id} AND {end_id}
        """)

    # =================================================
    # 6. 结束
    # =================================================
    print("normalize_and_log1p 完成")
    print("总耗时: {:.2f} 秒".format((datetime.now() - start).total_seconds()))

# 运行结果
#   X_CSRO_data 表
#     新增字段	             含义
#   data_log1p	     对表达值进行 normalize_total +  log(1+x) 转换


'''===== highly_variable_genes：  识别高变基因 - 在 X_CSRO 表上进行操作 '''
# 833206 * 17745  1.51 秒
def highly_variable_genes(
                        atlas: Atlas,
                        flavor: Literal["var", "cv"] = "var",
                        n_top_genes: int = 2000,
                        add_key: str = "highly_variable_genes",
                        select_data: str = "data"
                    ) -> None:
    """
    类似 sc.pp.highly_variable_genes（简化版）
    - 不修改 X_CSRO_data
    - 在 var 表中新建布尔字段 add_key
    - flavor: flavor 这个参数只能取 "var" 或 "cv" 这两个字符串之一
        - "var": 按方差排序
        - "cv" : 按变异系数（std / mean）排序
    """

    print("==== highly_variable_genes (CSR + DuckDB) ====")
    start = datetime.now()

    conn = atlas.connection
    try:
        conn.execute(f"PRAGMA threads={os.cpu_count()}")
    except:
        pass

    # 0. 检查字段存在
    col_exists = conn.execute(f"""
        SELECT COUNT(*)
        FROM information_schema.columns
        WHERE table_name = 'X_CSRO_data'
          AND column_name = '{select_data}'
    """).fetchone()[0]

    if col_exists == 0:
        raise ValueError(f"X_CSRO_data 中不存在字段: {select_data}")

    # 1. 计算每个 gene 的 mean / var / std
    print("Step 1: 计算 gene-level 统计量")

    conn.execute(f"""
        CREATE OR REPLACE TEMP TABLE _gene_stats AS
        SELECT
            atlas_gene_id,
            COUNT(*)                      AS n,
            AVG({select_data})            AS mean,
            VAR_POP({select_data})        AS var,
            STDDEV_POP({select_data})     AS std
        FROM X_CSRO_data
        WHERE {select_data} IS NOT NULL
        GROUP BY atlas_gene_id
    """)

    # -------------------------------------------------
    # 2. 计算排序指标
    # -------------------------------------------------
    if flavor == "var":
        score_expr = "var"
    elif flavor == "cv":
        # CV = std / mean（避免除 0）
        score_expr = "CASE WHEN mean > 0 THEN std / mean ELSE 0 END"
    else:
        raise ValueError(f"不支持的 flavor: {flavor}")

    conn.execute(f"""
        CREATE OR REPLACE TEMP TABLE _gene_score AS
        SELECT
            atlas_gene_id,
            {score_expr} AS score
        FROM _gene_stats
    """)

    # 3. 选 top genes
    if n_top_genes is not None:
        print(f"Step 2: 选取 top {n_top_genes} genes")

        conn.execute(f"""
            CREATE OR REPLACE TEMP TABLE _hvg AS
            SELECT atlas_gene_id
            FROM _gene_score
            ORDER BY score DESC
            LIMIT {int(n_top_genes)}
        """)
    else:
        # 全部保留
        print("Step 2: n_top_genes=None，全部标记为 True")

        conn.execute("""
            CREATE OR REPLACE TEMP TABLE _hvg AS
            SELECT atlas_gene_id
            FROM _gene_score
        """)

    # 4. 在 var 表中写入布尔结果
    print(f"Step 3: 写入 var.{add_key}")

    conn.execute(f"""
        ALTER TABLE var
        ADD COLUMN IF NOT EXISTS {add_key} BOOLEAN
    """)

    # 默认 False
    conn.execute(f"""
        UPDATE var
        SET {add_key} = FALSE
    """)

    # Top genes → True
    conn.execute(f"""
        UPDATE var
        SET {add_key} = TRUE
        FROM _hvg
        WHERE var.atlas_gene_id = _hvg.atlas_gene_id
    """)

    # -------------------------------------------------
    # 5. 结束
    # -------------------------------------------------
    print("highly_variable_genes 完成")
    print("耗时: {:.2f} 秒".format((datetime.now() - start).total_seconds()))

# 运行结果
#   var 表
#     新增字段	                     含义
#   highly_variable_genes	     对 n_top_genes 标记为true


'''=====  scale ：  进行 z-score转换 - 分块，大数据安全 '''
# 833206 * 17745   21.48 秒 大数据安全
def scale(
            atlas,
            select_data: str = "data",
            add_field: str = "data_scale",
            add_field_to_var: str = "zero_scale_transform",
            max_value: float = 10.0,
            use_hvg: bool = False,
            hvg_key: str = "highly_variable_genes"):
    """
    Gene-wise z-score scale（chunk 内 in-place UPDATE 版）

    - X_CSRO_data：chunk 内直接 UPDATE（无临时表、无 merge）
    - var 表：写入 zero -> z-score 的变换值 (0 - mean) / std
    """

    print("\n==== scale_gene_chunked_inplace (OLAP optimized) ====")
    start_all = datetime.now()
    conn = atlas.connection

    # -------------------------------------------------
    # 0. 并行
    # -------------------------------------------------
    try:
        n_threads = os.cpu_count()
        conn.execute(f"PRAGMA threads={n_threads}")
        print(f"-> DuckDB threads = {n_threads}")
    except Exception:
        pass

    # 自适应计算  gene_chunk_size 的大小
    total_nnz = conn.execute("""
    SELECT COUNT(*)
    FROM X_CSRO_data
    """).fetchone()[0]

    gene_num = conn.execute("""
    SELECT COUNT(*)
    FROM var
    """).fetchone()[0]

    nnz_per_gene = total_nnz / gene_num

    target_rows = 100_000_000

    gene_chunk_size = int(target_rows / nnz_per_gene)

    gene_chunk_size = max(16, min(1024, gene_chunk_size))

    # cells	nnz_per_gene	gene_chunk_size
    # 1M	100k	1024
    # 5M	500k	200
    # 10M	1M	100
    # 50M	5M	20
    # 100M	10M	16

    # -------------------------------------------------
    # 1. 输入字段检查
    # -------------------------------------------------
    if conn.execute(f"""
        SELECT COUNT(*)
        FROM information_schema.columns
        WHERE table_name='X_CSRO_data'
          AND column_name='{select_data}'
    """).fetchone()[0] == 0:
        raise ValueError(f"X_CSRO_data 中不存在字段: {select_data}")

    # -------------------------------------------------
    # 2. 输出字段准备
    # -------------------------------------------------
    conn.execute(f"""
        ALTER TABLE X_CSRO_data
        ADD COLUMN IF NOT EXISTS {add_field} REAL
    """)

    conn.execute(f"""
        ALTER TABLE var
        ADD COLUMN IF NOT EXISTS {add_field_to_var} REAL
    """)

    # -------------------------------------------------
    # 3. gene 列表
    # -------------------------------------------------
    if use_hvg:
        print("-> 使用 HVG gene 子集")
        gene_ids = conn.execute(f"""
            SELECT atlas_gene_id FROM var
            WHERE {hvg_key} = TRUE
            ORDER BY atlas_gene_id
        """).fetchall()
        gene_ids = [g[0] for g in gene_ids]
    else:
        gene_ids = conn.execute("""
            SELECT DISTINCT atlas_gene_id
            FROM X_CSRO_data
            ORDER BY atlas_gene_id
        """).fetchall()
        gene_ids = [g[0] for g in gene_ids]

    n_genes = len(gene_ids)
    if n_genes == 0:
        print("无 gene，退出")
        return

    n_chunks = math.ceil(n_genes / gene_chunk_size)

    print(f"-> Total genes: {n_genes}")
    print(f"-> Gene chunk size: {gene_chunk_size}")
    print(f"-> Total chunks: {n_chunks}")

    # -------------------------------------------------
    # 4. 主循环（gene chunk）
    # -------------------------------------------------
    for i in range(n_chunks):
        chunk_start = i * gene_chunk_size
        chunk_genes = gene_ids[chunk_start: chunk_start + gene_chunk_size]
        gene_list_sql = ",".join(map(str, chunk_genes))

        print(f"\n[Chunk {i+1}/{n_chunks}] genes={len(chunk_genes)}")

        # ---------------------------------------------
        # 4.1 gene-wise mean / std
        # ---------------------------------------------
        conn.execute(f"""
            CREATE OR REPLACE TEMP TABLE _gene_stat AS
            SELECT
                atlas_gene_id,
                AVG({select_data})        AS mean,
                STDDEV_POP({select_data}) AS std
            FROM X_CSRO_data
            WHERE atlas_gene_id IN ({gene_list_sql})
              AND {select_data} IS NOT NULL
            GROUP BY atlas_gene_id
        """)

        # ---------------------------------------------
        # 4.2 chunk 内直接 UPDATE X_CSRO_data
        # ---------------------------------------------
        conn.execute(f"""
            UPDATE X_CSRO_data x
            SET {add_field} =
                CASE
                    WHEN g.std > 0 THEN
                        LEAST(
                            {float(max_value)},
                            GREATEST(
                                -{float(max_value)},
                                (x.{select_data} - g.mean) / g.std
                            )
                        )
                    ELSE 0
                END
            FROM _gene_stat g
            WHERE x.atlas_gene_id = g.atlas_gene_id
              AND x.atlas_gene_id IN ({gene_list_sql})
              AND x.{select_data} IS NOT NULL
        """)

        # ---------------------------------------------
        # 4.3 写入 var：zero → z-score
        # ---------------------------------------------
        conn.execute(f"""
            UPDATE var v
            SET {add_field_to_var} =
                CASE
                    WHEN g.std > 0 THEN
                        LEAST(
                            {float(max_value)},
                            GREATEST(
                                -{float(max_value)},
                                (0 - g.mean) / g.std
                            )
                        )
                    ELSE 0
                END
            FROM _gene_stat g
            WHERE v.atlas_gene_id = g.atlas_gene_id
        """)

    print("\n==== scale_gene_chunked_inplace 完成 ====")
    print("耗时: {:.2f} 秒".format((datetime.now() - start_all).total_seconds()))

# 运行结果
#   X_CSRO_data 表
#     新增字段	                     含义
#   data_scale	             对 data 进行 z-score 标准化， z = (x - mean_g) / std_g
#   var 表  新增字段
#   zero_scale_transform    将每个基因的 ( 0 - g.mean) / g.std 存入var表的该字段，以便将来调用


'''=====  scale_fast ： 进行 z-score转换 - 在 X_CSRO 表上进行操作 '''
# 833206 * 17745    3.67 秒 大数据不安全
def scale_fast(
        atlas,
        select_data: str = "data",
        add_field: str = "data_scale",
        add_field_to_var: str = "zero_scale_transform",
        max_value: float = 10.0,
        use_hvg: bool = False,
        hvg_key: str = "highly_variable_genes"):
    """
    工业级 Gene-wise z-score scale（超大数据版）

    特点：
    - 完全不需要 chunk
    - 使用线性化公式 z = a*x + b
    - X_CSRO_data：直接写入 scale
    - var 表：写入 zero -> z-score 的变换值 (0 - mean) / std
    - 支持 HVG 子集
    """
    import os
    from datetime import datetime

    print("\n==== scale_ultra (industrial OLAP optimized) ====")
    start_all = datetime.now()
    conn = atlas.connection

    # -------------------------------------------------
    # 0. 并行
    # -------------------------------------------------
    try:
        n_threads = os.cpu_count()
        conn.execute(f"PRAGMA threads={n_threads}")
        print(f"-> DuckDB threads = {n_threads}")
    except Exception:
        pass

    # -------------------------------------------------
    # 1. 输入字段检查
    # -------------------------------------------------
    col_exists = conn.execute(f"""
        SELECT COUNT(*)
        FROM information_schema.columns
        WHERE table_name='X_CSRO_data'
          AND column_name='{select_data}'
    """).fetchone()[0]

    if col_exists == 0:
        raise ValueError(f"X_CSRO_data 中不存在字段: {select_data}")

    # -------------------------------------------------
    # 2. 输出字段准备
    # -------------------------------------------------
    conn.execute(f"""
        ALTER TABLE X_CSRO_data
        ADD COLUMN IF NOT EXISTS {add_field} REAL
    """)
    conn.execute(f"""
        ALTER TABLE var
        ADD COLUMN IF NOT EXISTS {add_field_to_var} REAL
    """)

    # -------------------------------------------------
    # 3. gene 列表
    # -------------------------------------------------
    if use_hvg:
        print("-> 使用 HVG gene 子集")
        gene_ids = conn.execute(f"""
            SELECT atlas_gene_id
            FROM var
            WHERE {hvg_key} = TRUE
            ORDER BY atlas_gene_id
        """).fetchall()
    else:
        gene_ids = conn.execute("""
            SELECT atlas_gene_id
            FROM var
            ORDER BY atlas_gene_id
        """).fetchall()

    gene_ids = [g[0] for g in gene_ids]
    n_genes = len(gene_ids)
    if n_genes == 0:
        print("无 gene，退出")
        return

    print(f"-> Total genes: {n_genes}")

    gene_list_sql = ",".join(map(str, gene_ids))

    # -------------------------------------------------
    # 4. 计算 gene-wise统计 + 线性化系数 a,b
    # -------------------------------------------------
    print("-> 计算 gene-wise mean/std -> a,b")
    conn.execute(f"""
        CREATE OR REPLACE TEMP TABLE _gene_stat AS
        SELECT
            atlas_gene_id,
            AVG({select_data}) AS mean,
            STDDEV_POP({select_data}) AS std,
            CASE WHEN STDDEV_POP({select_data}) > 0
                 THEN 1.0 / STDDEV_POP({select_data})
                 ELSE 0
            END AS a,
            CASE WHEN STDDEV_POP({select_data}) > 0
                 THEN -AVG({select_data}) / STDDEV_POP({select_data})
                 ELSE 0
            END AS b
        FROM X_CSRO_data
        WHERE atlas_gene_id IN ({gene_list_sql})
          AND {select_data} IS NOT NULL
        GROUP BY atlas_gene_id
    """)

    # -------------------------------------------------
    # 5. 直接 UPDATE X_CSRO_data
    # -------------------------------------------------
    print("-> 直接应用线性化 z-score + clip")
    conn.execute(f"""
        UPDATE X_CSRO_data x
        SET {add_field} = 
            LEAST(
                {float(max_value)},
                GREATEST(
                    -{float(max_value)},
                    g.a * x.{select_data} + g.b
                )
            )
        FROM _gene_stat g
        WHERE x.atlas_gene_id = g.atlas_gene_id
          AND x.{select_data} IS NOT NULL
    """)

    # -------------------------------------------------
    # 6. 更新 var 表 zero_scale_transform
    # -------------------------------------------------
    print("-> 更新 var zero_scale_transform")
    conn.execute(f"""
        UPDATE var v
        SET {add_field_to_var} = g.b
        FROM _gene_stat g
        WHERE v.atlas_gene_id = g.atlas_gene_id
    """)

    print("\n==== scale_ultra 完成 ====")
    print("耗时: {:.2f} 秒".format((datetime.now() - start_all).total_seconds()))

# 运行结果
#   X_CSRO_data 表
#     新增字段	                     含义
#   data_scale	             对 data 进行 z-score 标准化， z = (x - mean_g) / std_g
#   var 表  新增字段
#   zero_scale_transform    将每个基因的 ( 0 - g.mean) / g.std 存入var表的该字段，以便将来调用


'''===== sqrt  法 1 ： 不分块，大数据 不安全========== '''
# 833206 * 17745   1.62 秒
def sqrt_fast(
    atlas: "Atlas",
    add_field: str = "data_sqrt",
    select_data: str = "data") -> None:
    """
    对表达值进行 sqrt 转换

    - 直接执行 sqrt(x)
    - 不对 0 做任何处理
    - 仅跳过 NULL
    """

    logger.info("开始执行 sqrt(x) 转换...")
    print("==== sqrt ====")
    start = datetime.now()

    try:
        atlas.connection.execute(f"PRAGMA threads={os.cpu_count()}")
    except Exception:
        pass

    conn = atlas.connection

    # -------------------------------------------------
    # 0. 字段存在性检查（防止 silent bug）
    # -------------------------------------------------
    col_exists = conn.execute(
        f"""
        SELECT COUNT(*)
        FROM information_schema.columns
        WHERE table_name = 'X_CSRO_data'
          AND column_name = '{select_data}'
        """
    ).fetchone()[0]

    if col_exists == 0:
        raise ValueError(f"X_CSRO_data 中不存在字段: {select_data}")

    if add_field is None:
        raise ValueError("必须指定 add_field")

    print(f"创建 / 使用字段 X_CSRO_data.{add_field} ...")

    # -------------------------------------------------
    # 1. 构造 sqrt 表达式（不处理 0）
    # -------------------------------------------------
    sqrt_expr = f"sqrt({select_data})"

    # -------------------------------------------------
    # 2. 确保输出字段存在
    # -------------------------------------------------
    conn.execute(
        f"""
        ALTER TABLE X_CSRO_data
        ADD COLUMN IF NOT EXISTS {add_field} REAL
        """
    )

    # -------------------------------------------------
    # 3. 执行 sqrt
    # -------------------------------------------------
    conn.execute(
        f"""
        UPDATE X_CSRO_data
        SET {add_field} = {sqrt_expr}
        WHERE {select_data} IS NOT NULL
        """
    )

    # -------------------------------------------------
    # 4. 结束
    # -------------------------------------------------
    elapsed = (datetime.now() - start).total_seconds()
    print("sqrt 转换完成")
    print(f"耗时: {elapsed:.2f} 秒")

# 运行结果
#   X_CSRO_data 表
#     新增字段	                     含义
#   data_sqrt	             对 data 进行 sqrt


'''===== sqrt  法 2 ： 分块，大数据 安全========== '''
#   833206 * 17745   2.97 秒 二次 1.97 秒   其他  61.69 秒
def sqrt(
    atlas: "Atlas",
    add_field: str = "data_sqrt",
    select_data: str = "data",
    chunk_size: int = 100_000_000) -> None:
    """
    1e8 级 CSR 安全的 sqrt 实现
    - 不一次性 UPDATE 全表
    - 按 id 分块
    - 支持指定 X_CSRO_data 中的任意字段作为输入
    - 不对 0 做任何处理（sqrt(0)=0）
    """

    logger.info("开始执行 sqrt (chunked)...")
    print("==== sqrt (chunked) ====")
    start = datetime.now()

    conn = atlas.connection

    try:
        conn.execute(f"PRAGMA threads={os.cpu_count()}")
    except Exception:
        pass

    # -------------------------------------------------
    # 0. 字段存在性检查（重要）
    # -------------------------------------------------
    col_exists = conn.execute(
        f"""
        SELECT COUNT(*)
        FROM information_schema.columns
        WHERE table_name = 'X_CSRO_data'
          AND column_name = '{select_data}'
        """
    ).fetchone()[0]

    if col_exists == 0:
        raise ValueError(f"X_CSRO_data 中不存在字段: {select_data}")

    if add_field is None:
        raise ValueError("必须指定 add_field")

    # -------------------------------------------------
    # 1. 构造 sqrt 表达式（不处理 0）
    # -------------------------------------------------
    sqrt_expr = f"sqrt({select_data})"

    # -------------------------------------------------
    # 2. 确保输出字段存在
    # -------------------------------------------------
    conn.execute(
        f"""
        ALTER TABLE X_CSRO_data
        ADD COLUMN IF NOT EXISTS {add_field} REAL
        """
    )

    # -------------------------------------------------
    # 3. 获取 id 范围
    # -------------------------------------------------
    min_id, max_id = conn.execute(
        """
        SELECT MIN(id), MAX(id)
        FROM X_CSRO_data
        """
    ).fetchone()

    if min_id is None:
        print("X_CSRO_data 为空，跳过")
        return

    n_chunks = math.ceil((max_id - min_id + 1) / chunk_size)
    print(f"共 {n_chunks} 个 chunk")

    # -------------------------------------------------
    # 4. 分块 UPDATE
    # -------------------------------------------------
    for i in range(n_chunks):
        start_id = min_id + i * chunk_size
        end_id = start_id + chunk_size - 1

        print(f"  -> chunk {i + 1}/{n_chunks}: id [{start_id}, {end_id}]")

        conn.execute(
            f"""
            UPDATE X_CSRO_data
            SET {add_field} = {sqrt_expr}
            WHERE id BETWEEN {start_id} AND {end_id}
              AND {select_data} IS NOT NULL
            """
        )

    # -------------------------------------------------
    # 5. 结束
    # -------------------------------------------------
    elapsed = (datetime.now() - start).total_seconds()
    print("sqrt (chunked) 完成")
    print(f"耗时: {elapsed:.2f} 秒")

# 运行结果
#   X_CSRO_data 表
#     新增字段	                     含义
#   data_sqrt	             对 data 进行 sqrt