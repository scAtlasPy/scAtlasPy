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

#========== todo 归一化 normalize 法 1 ： 不分块， 在 X_CSR_data 表上直接处理 ， 不替换原数据 ==========
def normalize_total_new(atlas: Atlas,
                    target_sum: float = 10000,
                    add_key: Optional[str] = None,
                    add_field: str = "data_normalize") -> None:
    """
    类似scanpy中的normalize_total的行为，对数据按行归一化，使得细胞的所有基因表达值之和等于给定值target_sum。

    Args:
        atlas: Atlas对象，包含单细胞数据
        target_sum: 归一化后的总计数目标值。默认 10,000；如果为 None，则使用中位数
        add_key: 在obs中动态增加add_key指定的字段，在该字段存放target_sum值
        add_field: 指定新字段的名称,将结果存入

    Returns:
        None
    """
    print("==== normalize_total (CSR + DuckDB) ====")
    start = datetime.now()

    conn = atlas.connection

    # -----------------------------
    # 0. DuckDB 并行 & 内存设置
    # -----------------------------
    try:
        conn.execute(f"PRAGMA threads={os.cpu_count()}")
    except:
        pass

    # -----------------------------
    # 1. 计算每个 cell 的 total counts
    # -----------------------------
    print("计算每个 cell 的 total counts ...")

    conn.execute("""
           CREATE OR REPLACE TEMP TABLE _cell_sum AS
           SELECT
               cell_index,
               SUM(data) AS total
           FROM X_CSR_data
           GROUP BY cell_index
       """)

    # -----------------------------
    # 2. 如果 target_sum = None → 使用中位数
    # -----------------------------
    if target_sum is None:
        target_sum = conn.execute("""
               SELECT median(total) FROM _cell_sum
           """).fetchone()[0]

        print(f"自动使用中位数作为 target_sum = {target_sum}")

    # -----------------------------
    # 3. 是否在 obs 中记录 target_sum
    # -----------------------------
    if add_key is not None:
        conn.execute(f"""
               ALTER TABLE obs
               ADD COLUMN IF NOT EXISTS {add_key} DOUBLE
           """)
        conn.execute(f"""
               UPDATE obs
               SET {add_key} = {float(target_sum)}
           """)
    # -----------------------------
    # 4. 执行归一化
    # -----------------------------
    if add_field is None:
        raise ValueError("当 inplace=False 时，必须指定 add_field")

    print(f"创建新字段 X_CSR_data.{add_field} ...")

    new_table = f"X_CSR_data_{add_field}"

    conn.execute(f"""
        CREATE TABLE {new_table} AS
        SELECT
            x.id,
            x.cell_index,
            x.indices,
            x.data,
            x.data * {float(target_sum)} / s.total AS {add_field}
        FROM X_CSR_data x
        JOIN _cell_sum s
        ON x.cell_index = s.cell_index
        WHERE s.total > 0
    """)

    # 可选：替换当前 X
    conn.execute("DROP TABLE X_CSR_data")
    conn.execute(f"ALTER TABLE {new_table} RENAME TO X_CSR_data")

    # -----------------------------
    # 5. 结束
    # -----------------------------
    print("normalize_total 完成")
    print("耗时: {:.2f} 秒".format((datetime.now() - start).total_seconds()))

#========== todo 归一化 normalize 法 2 ： 分块， 在 X_CSR_data 表上直接处理 ， 不替换原数据  ==========
def normalize_total_new_chunked(atlas: Atlas,
                    target_sum: float = 10000,
                    add_key: Optional[str] = None,
                    chunk_size: int = 100_000_000,
                    add_field: str = "data_normalize") -> None:
    """
    类似scanpy中的normalize_total的行为，对数据按行归一化，使得细胞的所有基因表达值之和等于给定值target_sum。

    Args:
        atlas: Atlas对象，包含单细胞数据
        target_sum: 归一化后的总计数目标值。默认 10,000；如果为 None，则使用中位数
        add_key: 在obs中动态增加add_key指定的字段，在该字段存放target_sum值
        inplace: 是否原地修改表达值。如果为True，则在 X_CSR_data 表中 的 data 字段修改；如果为False，则创建新字段存值
        add_field: 当inplace为False时，指定新字段的名称

    Returns:
        None
    """

    print("==== normalize_total_chunked (CSR + DuckDB) ====")
    start = datetime.now()

    conn = atlas.connection

    try:
        conn.execute(f"PRAGMA threads={os.cpu_count()}")
    except:
        pass

    # -------------------------------------------------
    # 1. 计算每个 cell 的 total counts
    # -------------------------------------------------
    print("Step 1: compute cell sums")

    conn.execute("""
        CREATE OR REPLACE TEMP TABLE _cell_sum AS
        SELECT
            cell_index,
            SUM(data) AS total
        FROM X_CSR_data
        GROUP BY cell_index
    """)

    if target_sum is None:
        target_sum = conn.execute(
            "SELECT median(total) FROM _cell_sum"
        ).fetchone()[0]
        print(f"Auto target_sum = {target_sum}")

    # -------------------------------------------------
    # 2. 记录 target_sum 到 obs
    # -------------------------------------------------
    if add_key is not None:
        conn.execute(f"""
            ALTER TABLE obs
            ADD COLUMN IF NOT EXISTS {add_key} DOUBLE
        """)
        conn.execute(f"""
            UPDATE obs
            SET {add_key} = {float(target_sum)}
        """)

    # -------------------------------------------------
    # 3. 获取 cell_index 范围
    # -------------------------------------------------
    min_cell, max_cell = conn.execute("""
        SELECT MIN(cell_index), MAX(cell_index)
        FROM _cell_sum
    """).fetchone()

    n_chunks = math.ceil((max_cell - min_cell + 1) / chunk_size)
    print(f"Step 2: {n_chunks} chunks")

    part_tables = []

    # -------------------------------------------------
    # 4. 分块 CTAS（保留 id）
    # -------------------------------------------------
    for i in range(n_chunks):
        start_cell = min_cell + i * chunk_size
        end_cell = start_cell + chunk_size - 1

        part_table = f"_X_norm_part_{i}"
        part_tables.append(part_table)

        print(f"  -> chunk {i+1}/{n_chunks}: cell_index [{start_cell}, {end_cell}]")

        conn.execute(f"""
            CREATE TABLE {part_table} AS
            SELECT
                x.id,
                x.cell_index,
                x.indices,
                x.data,
                x.data * {float(target_sum)} / s.total AS {add_field}
            FROM X_CSR_data x
            JOIN _cell_sum s
              ON x.cell_index = s.cell_index
            WHERE x.cell_index BETWEEN {start_cell} AND {end_cell}
              AND s.total > 0
        """)

    # -------------------------------------------------
    # 5. 合并所有 chunk
    # -------------------------------------------------
    print("Step 3: union all chunks")

    union_sql = "\nUNION ALL\n".join(
        [f"SELECT * FROM {t}" for t in part_tables]
    )

    conn.execute(f"""
        CREATE TABLE X_CSR_data_norm AS
        {union_sql}
    """)

    # -------------------------------------------------
    # 6. 替换原 X_CSR_data
    # -------------------------------------------------
    print("Step 4: replace X_CSR_data")

    conn.execute("DROP TABLE X_CSR_data")
    conn.execute("ALTER TABLE X_CSR_data_norm RENAME TO X_CSR_data")

    for t in part_tables:
        conn.execute(f"DROP TABLE {t}")

    print("normalize_total_chunked 完成")
    print("耗时: {:.2f} 秒".format((datetime.now() - start).total_seconds()))


#========== todo 归一化 normalize 法 3 ： 参照scanpy，在 obs表上记录 scale_factor ， 等到使用的时候在计算=========
def normalize_total_scale_factor(
                    atlas: Atlas,
                    target_sum: Optional[float] = 10000,
                    add_key: str = "scale_factor",
                    select_data: str = "data" ) -> None:
    """
    高性能 normalize_total（Scanpy 等价）：
    - 不修改 X_CSR_data
    - 只计算每个 cell 的 scale_factor
    - 支持指定使用 X_CSR_data 中的任意字段作为表达值

    Args:
        atlas: Atlas 对象
        target_sum: 归一化目标和；None 表示使用中位数
        add_key: 写入 obs 的 scale_factor 字段名
        select_data: X_CSR_data 中用于计算 total 的字段名
                     （如 'data', 'data_normalized', 'X_log1p'）
    """

    print("==== normalize_total (scale_factor only) ====")
    start = datetime.now()

    conn = atlas.connection
    try:
        conn.execute(f"PRAGMA threads={os.cpu_count()}")
    except:
        pass

    # -------------------------------------------------
    # 0. 基本安全检查（强烈建议保留）
    # -------------------------------------------------
    col_exists = conn.execute(f"""
        SELECT COUNT(*)
        FROM information_schema.columns
        WHERE table_name = 'X_CSR_data'
          AND column_name = '{select_data}'
    """).fetchone()[0]

    if col_exists == 0:
        raise ValueError(f"X_CSR_data 中不存在字段: {select_data}")

    # -------------------------------------------------
    # 1. 计算每个 cell 的 total
    # -------------------------------------------------
    conn.execute(f"""
        CREATE OR REPLACE TEMP TABLE _cell_sum AS
        SELECT
            cell_index,
            SUM({select_data}) AS total
        FROM X_CSR_data
        GROUP BY cell_index
    """)

    # -------------------------------------------------
    # 2. target_sum = median(total)
    # -------------------------------------------------
    if target_sum is None:
        target_sum = conn.execute(
            "SELECT median(total) FROM _cell_sum"
        ).fetchone()[0]

    # -------------------------------------------------
    # 3. 在 obs 中写 scale_factor
    # -------------------------------------------------
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
        WHERE obs.id = s.cell_index
    """)

    print(f"normalize_total 完成，target_sum={target_sum}")
    print("耗时: {:.2f} 秒".format((datetime.now() - start).total_seconds()))


#========== todo log1p  法 1 ： 不分块   ==========

def log1p(
            atlas: 'Atlas',
            base: Optional[Number] = None,
            add_field: str = "log1p_factor",
            select_data: str = "data" ) -> None:
    """
    对表达值进行 log(1+x) 转换
    - 支持选择 X_CSR_data 中任意字段进行计算
    """

    logger.info("开始执行 log(1+x) 转换...")
    print("==== log1p ====")
    start = datetime.now()

    try:
        atlas.connection.execute(f"PRAGMA threads={os.cpu_count()}")
    except:
        pass

    conn = atlas.connection

    # -------------------------------------------------
    # 0. 字段存在性检查（防止 silent bug）
    # -------------------------------------------------
    col_exists = conn.execute(f"""
        SELECT COUNT(*)
        FROM information_schema.columns
        WHERE table_name = 'X_CSR_data'
          AND column_name = '{select_data}'
    """).fetchone()[0]

    if col_exists == 0:
        raise ValueError(f"X_CSR_data 中不存在字段: {select_data}")

    # -------------------------------------------------
    # 1. 构造 log1p 表达式
    # -------------------------------------------------
    if base is None:
        log_expr = f"ln(1.0 + {select_data})"
    else:
        log_expr = f"log({float(base)}, 1.0 + {select_data})"

    if add_field is None:
        raise ValueError("必须指定 add_field")

    print(f"创建新字段 X_CSR_data.{add_field} ...")

    # -------------------------------------------------
    # 2. 确保输出字段存在
    # -------------------------------------------------
    conn.execute(f"""
        ALTER TABLE X_CSR_data
        ADD COLUMN IF NOT EXISTS {add_field} REAL
    """)

    # -------------------------------------------------
    # 3. 执行 log1p
    # -------------------------------------------------
    conn.execute(f"""
        UPDATE X_CSR_data
        SET {add_field} = {log_expr}
        WHERE {select_data} IS NOT NULL
    """)

    # -------------------------------------------------
    # 4. 结束
    # -------------------------------------------------
    print("log1p 完成")
    print("耗时: {:.2f} 秒".format((datetime.now() - start).total_seconds()))

#========== todo log1p  法 2 ： 分块  ==========
def log1p_chunked(
                atlas: 'Atlas',
                base: Optional[Number] = None,
                add_field: str = "log1p_factor",
                select_data: str = "data",
                chunk_size: int = 100_000_000) -> None:
    """
    1e8 级 CSR 安全的 log1p 实现
    - 不一次性 UPDATE 全表
    - 按 id 分块
    - 支持指定 X_CSR_data 中的任意字段作为输入
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
        WHERE table_name = 'X_CSR_data'
          AND column_name = '{select_data}'
    """).fetchone()[0]

    if col_exists == 0:
        raise ValueError(f"X_CSR_data 中不存在字段: {select_data}")

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
        ALTER TABLE X_CSR_data
        ADD COLUMN IF NOT EXISTS {add_field} REAL
    """)

    # -------------------------------------------------
    # 3. 获取 id 范围
    # -------------------------------------------------
    min_id, max_id = conn.execute("""
        SELECT MIN(id), MAX(id)
        FROM X_CSR_data
    """).fetchone()

    if min_id is None:
        print("X_CSR_data 为空，跳过")
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
            UPDATE X_CSR_data
            SET {add_field} = {log_expr}
            WHERE id BETWEEN {start_id} AND {end_id}
              AND {select_data} IS NOT NULL
        """)

    # -------------------------------------------------
    # 5. 结束
    # -------------------------------------------------
    print("log1p (chunked) 完成")
    print("耗时: {:.2f} 秒".format((datetime.now() - start).total_seconds()))

#========== todo exp1 是  log1p的逆运算 ==========
def exp1_chunked(
        atlas: 'Atlas',
        base: Optional[Number] = None,
        add_field: str = "exp1_factor",
        select_data: str = "log1p_factor",
        chunk_size: int = 100_000_000 ) -> None:
    """
    log1p_chunked 的逆运算：exp(x) - 1
    - ln(1+x)  → exp(x) - 1
    - log_b(1+x) → b^x - 1
    - 按 X_CSR_data.id 分块，支持 1e8 CSR
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
        WHERE table_name = 'X_CSR_data'
          AND column_name = '{select_data}'
    """).fetchone()[0]

    if col_exists == 0:
        raise ValueError(f"X_CSR_data 中不存在字段: {select_data}")

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
        ALTER TABLE X_CSR_data
        ADD COLUMN IF NOT EXISTS {add_field} REAL
    """)

    # -------------------------------------------------
    # 3. 获取 id 范围
    # -------------------------------------------------
    min_id, max_id = conn.execute("""
        SELECT MIN(id), MAX(id)
        FROM X_CSR_data
    """).fetchone()

    if min_id is None:
        print("X_CSR_data 为空，跳过")
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
            UPDATE X_CSR_data
            SET {add_field} = {exp_expr}
            WHERE id BETWEEN {start_id} AND {end_id}
              AND {select_data} IS NOT NULL
        """)

    # -------------------------------------------------
    # 5. 结束
    # -------------------------------------------------
    print("exp1 (chunked) 完成")
    print("耗时: {:.2f} 秒".format((datetime.now() - start).total_seconds()))



#========== todo normalize_and_log1p： normalize 法 3 + log1p 法 2 ==========
def normalize_and_log1p(
            atlas: Atlas,
            target_sum: Optional[float] = 10000,
            scale_key: str = "scale_factor",
            add_field: str = "X_log1p",
            select_data: str = "data",
            base: Optional[Number] = None,
            chunk_size: int = 100_000_000 ) -> None:
    """
    Scanpy 等价的 normalize_total + log1p
    - normalize_total：只计算 scale_factor（obs）
    - log1p：log(1 + select_data * scale_factor)
    - 按 X_CSR_data.id 分块（✔ 正确的 1e8 CSR 方式）
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
        WHERE table_name = 'X_CSR_data'
          AND column_name = '{select_data}'
    """).fetchone()[0]

    if col_exists == 0:
        raise ValueError(f"X_CSR_data 中不存在字段: {select_data}")

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
        ALTER TABLE X_CSR_data
        ADD COLUMN IF NOT EXISTS {add_field} REAL
    """)

    # =================================================
    # 4. 获取 X_CSR_data.id 范围
    # =================================================
    min_id, max_id = conn.execute("""
        SELECT MIN(id), MAX(id)
        FROM X_CSR_data
    """).fetchone()

    if min_id is None:
        print("X_CSR_data 为空，跳过")
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
            UPDATE X_CSR_data AS x
            SET {add_field} = {log_expr}
            FROM obs AS o
            WHERE x.cell_index = o.id
              AND x.id BETWEEN {start_id} AND {end_id}
        """)

    # =================================================
    # 6. 结束
    # =================================================
    print("normalize_and_log1p 完成")
    print("总耗时: {:.2f} 秒".format((datetime.now() - start).total_seconds()))


#========== todo  highly_variable_genes：  识别高变基因 - 在 X_CSR 表上进行操作
def highly_variable_genes(
                        atlas: Atlas,
                        flavor: Literal["var", "cv"] = "var",
                        n_top_genes: int | None = None,
                        add_key: str = "highly_variable_genes",
                        select_data: str = "data"
                    ) -> None:
    """
    类似 sc.pp.highly_variable_genes（简化版）
    - 不修改 X_CSR_data
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

    # -------------------------------------------------
    # 0. 检查字段存在
    # -------------------------------------------------
    col_exists = conn.execute(f"""
        SELECT COUNT(*)
        FROM information_schema.columns
        WHERE table_name = 'X_CSR_data'
          AND column_name = '{select_data}'
    """).fetchone()[0]

    if col_exists == 0:
        raise ValueError(f"X_CSR_data 中不存在字段: {select_data}")

    # -------------------------------------------------
    # 1. 计算每个 gene 的 mean / var / std
    # -------------------------------------------------
    print("Step 1: 计算 gene-level 统计量")

    conn.execute(f"""
        CREATE OR REPLACE TEMP TABLE _gene_stats AS
        SELECT
            indices AS gene_index,
            COUNT(*)                      AS n,
            AVG({select_data})            AS mean,
            VAR_POP({select_data})        AS var,
            STDDEV_POP({select_data})     AS std
        FROM X_CSR_data
        WHERE {select_data} IS NOT NULL
        GROUP BY indices
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
            gene_index,
            {score_expr} AS score
        FROM _gene_stats
    """)

    # -------------------------------------------------
    # 3. 选 top genes
    # -------------------------------------------------
    if n_top_genes is not None:
        print(f"Step 2: 选取 top {n_top_genes} genes")

        conn.execute(f"""
            CREATE OR REPLACE TEMP TABLE _hvg AS
            SELECT gene_index
            FROM _gene_score
            ORDER BY score DESC
            LIMIT {int(n_top_genes)}
        """)
    else:
        # 全部保留
        print("Step 2: n_top_genes=None，全部标记为 True")

        conn.execute("""
            CREATE OR REPLACE TEMP TABLE _hvg AS
            SELECT gene_index
            FROM _gene_score
        """)

    # -------------------------------------------------
    # 4. 在 var 表中写入布尔结果
    # -------------------------------------------------
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
        WHERE var.id = _hvg.gene_index
    """)

    # -------------------------------------------------
    # 5. 结束
    # -------------------------------------------------
    print("highly_variable_genes 完成")
    print("耗时: {:.2f} 秒".format((datetime.now() - start).total_seconds()))


#========== todo scale ：  进行 z-score转换 - 在 X_CSR 表上进行操作
def scale(
            atlas: Atlas,
            max_value: float = 10.0,
            add_field: str = "X_scale",
            select_data: str = "data",
            use_hvg: bool = False,
            hvg_key: str = "highly_variable" ) -> None:
    """
    Gene-wise z-score 标准化（sc.pp.scale 等价）
    数学定义：
        z = (x - mean_g) / std_g
        z = clip(z, -max_value, max_value)
    特性：
    - 在 X_CSR_data（稀疏 CSR）上直接计算
    - 按 gene 维度计算 mean / std
    - 支持 Scanpy 风格的 max_value 截断
    - 可选：仅对高可变基因（HVG）执行 scale
    参数说明：
    ----------
    atlas : Atlas ； 数据库与元信息管理对象
    max_value : float ； z-score 的截断阈值（Scanpy 默认 10）
    add_field : str ； scale 后结果写入 X_CSR_data 的字段名
    select_data : str ； 用于 scale 的输入字段（如 data / log1p / normalized）
    use_hvg : bool ； 是否只对高可变基因执行 scale
    hvg_key : str  ； var 表中标记高可变基因的布尔字段名

    """
    print("==== scale (gene-wise z-score) ====")
    start = datetime.now()
    conn = atlas.connection

    # -------------------------------------------------
    # 0. DuckDB 并行设置
    # -------------------------------------------------
    try:
        conn.execute(f"PRAGMA threads={os.cpu_count()}")
    except:
        pass

    # -------------------------------------------------
    # 1. 检查输入字段
    # -------------------------------------------------
    col_exists = conn.execute(f"""
        SELECT COUNT(*)
        FROM information_schema.columns
        WHERE table_name = 'X_CSR_data'
          AND column_name = '{select_data}'
    """).fetchone()[0]

    if col_exists == 0:
        raise ValueError(f"X_CSR_data 中不存在字段: {select_data}")

    # -------------------------------------------------
    # 2. 是否仅对 HVG 进行 scale
    # -------------------------------------------------
    gene_filter_clause = ""
    if use_hvg:
        print("-> 仅对高可变基因 (HVG) 执行 scale")

        hvg_exists = conn.execute(f"""
            SELECT COUNT(*)
            FROM information_schema.columns
            WHERE table_name = 'var'
              AND column_name = '{hvg_key}'
        """).fetchone()[0]

        if hvg_exists == 0:
            raise ValueError(f"var 表中不存在 HVG 字段: {hvg_key}")

        gene_filter_clause = f"""
            WHERE indices IN (
                SELECT id FROM var WHERE {hvg_key} = TRUE
            )
        """

    # -------------------------------------------------
    # 3. 计算 gene-wise mean / std
    # -------------------------------------------------
    print("-> 计算 gene-wise mean / std")

    conn.execute(f"""
        CREATE OR REPLACE TEMP TABLE _gene_stat AS
        SELECT
            indices,
            AVG(val)        AS mean,
            STDDEV_POP(val) AS std
        FROM (
            SELECT
                indices,
                {select_data} AS val
            FROM X_CSR_data
            WHERE {select_data} IS NOT NULL
        )
        {gene_filter_clause}
        GROUP BY indices
    """)
    # 创建一个 临时表 _gene_stat
    # 列名	       含义
    # indices	gene_index
    # mean	    该基因在所有细胞中的平均表达
    # std	    该基因的总体标准差

    # -------------------------------------------------
    # 4. 准备输出字段
    # -------------------------------------------------
    conn.execute(f"""
        ALTER TABLE X_CSR_data
        ADD COLUMN IF NOT EXISTS {add_field} REAL
    """)

    # -------------------------------------------------
    # 5. 执行 z-score + clip
    # 对每个非零表达值做 gene-wise z-score 标准化，并进行截断（clip）后写入新字段
    # -------------------------------------------------
    print("-> 执行 z-score + clip")

    conn.execute(f"""
        UPDATE X_CSR_data AS x
        SET {add_field} =
            CASE
                WHEN g.std > 0 THEN  -- 某些基因：只在 1 个细胞中表达 或 所有值相同；std = 0；要排除； 
                    LEAST(
                        {float(max_value)},  -- 防止极端值 含义 z > max_value → max_value 
                        GREATEST(
                            -{float(max_value)},  -- 防止极端值 含义 z < -max_value → -max_value
                            (x.{select_data} - g.mean) / g.std
                        )
                    )
                ELSE 0
            END
        FROM _gene_stat AS g
        WHERE x.indices = g.indices
          AND x.{select_data} IS NOT NULL
    """)

    # CASE
    #     WHEN 条件1 THEN 结果1
    #     WHEN 条件2 THEN 结果2
    #     ELSE 默认结果
    # END
    # CASE WHEN 就是 SQL 里的 if / else

    # -------------------------------------------------
    # 6. 结束
    # -------------------------------------------------
    print("scale 完成")
    print("耗时: {:.2f} 秒".format((datetime.now() - start).total_seconds()))


def scale_gene_chunked(
            atlas,
            select_data: str = "data",
            add_field: str = "X_scale",
            add_field_to_var: str = "zero_scale_transform", # 增加该字段，将每个基因的 ( 0 - g.mean) / g.std 存入var表的该字段，以便将来调用
            max_value: float = 10.0,
            gene_chunk_size: int = 512, # gene_chunk_size 的设置与cell数量 相关
            use_hvg: bool = False,
            hvg_key: str = "highly_variable"):
    """
    Gene-wise z-score scale（DuckDB OLAP 优化版，1e8 cell 安全）

    - X_CSR_data：写入 scale 后的非零值
    - var 表：写入 zero -> z-score 的变换值 (0 - mean) / std
    """

    print("\n==== scale_gene_chunked_vA (OLAP optimized) ====")
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
    if conn.execute(f"""
        SELECT COUNT(*)
        FROM information_schema.columns
        WHERE table_name='X_CSR_data'
          AND column_name='{select_data}'
    """).fetchone()[0] == 0:
        raise ValueError(f"X_CSR_data 中不存在字段: {select_data}")

    # -------------------------------------------------
    # 2. 输出字段准备
    # -------------------------------------------------
    conn.execute(f"""
        ALTER TABLE X_CSR_data
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
            SELECT id FROM var
            WHERE {hvg_key} = TRUE
            ORDER BY id
        """).fetchall()
        gene_ids = [g[0] for g in gene_ids]
    else:
        gene_ids = conn.execute("""
            SELECT DISTINCT indices
            FROM X_CSR_data
            ORDER BY indices
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
    # 4. 临时表（存 scale 后的非零值）
    # -------------------------------------------------
    conn.execute("DROP TABLE IF EXISTS _X_scale_tmp") # 清理之前的缓存

    conn.execute("""
        CREATE TEMP TABLE _X_scale_tmp (
            id BIGINT,
            indices INTEGER,
            val REAL
        )
    """)

    # -------------------------------------------------
    # 5. 主循环（gene chunk）
    # -------------------------------------------------
    for i in range(n_chunks):
        chunk_start = i * gene_chunk_size
        chunk_genes = gene_ids[chunk_start: chunk_start + gene_chunk_size]
        gene_list_sql = ",".join(map(str, chunk_genes))

        print(f"\n[Chunk {i+1}/{n_chunks}] genes={len(chunk_genes)}")

        # ---------------------------------------------
        # 5.1 gene-wise mean / std
        # ---------------------------------------------
        conn.execute(f"""
            CREATE OR REPLACE TEMP TABLE _gene_stat AS
            SELECT
                indices,
                AVG({select_data})        AS mean,
                STDDEV_POP({select_data}) AS std
            FROM X_CSR_data
            WHERE indices IN ({gene_list_sql})
              AND {select_data} IS NOT NULL
            GROUP BY indices
        """)

        # ---------------------------------------------
        # 5.2 非零值 scale → 临时表
        # ---------------------------------------------
        conn.execute(f"""
            INSERT INTO _X_scale_tmp
            SELECT
                x.id,
                x.indices,
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
            FROM X_CSR_data x
            JOIN _gene_stat g
              ON x.indices = g.indices
            WHERE x.indices IN ({gene_list_sql})
              AND x.{select_data} IS NOT NULL
        """)

        # ---------------------------------------------
        # 5.3 写入 var：zero → z-score
        # (0 - mean) / std
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
            WHERE v.id = g.indices
        """)

    # -------------------------------------------------
    # 6. merge 回 X_CSR_data
    # -------------------------------------------------
    print("\n-> Merging scaled values back to X_CSR_data ...")

    conn.execute(f"""
        UPDATE X_CSR_data x
        SET {add_field} = t.val
        FROM _X_scale_tmp t
        WHERE x.id = t.id
    """)

    # 在函数末尾 清理缓存
    conn.execute("DROP TABLE IF EXISTS _X_scale_tmp")

    print("\n==== scale_gene_chunked 完成 ====")
    print("耗时: {:.2f} 秒".format((datetime.now() - start_all).total_seconds()))


#========== todo scale ：  对 scale_gene_chunked 的优化 1
#                           直接在 chunk 内 UPDATE，彻底删掉临时表 + merge
def scale_gene_chunked_1(
            atlas,
            select_data: str = "data",
            add_field: str = "X_scale",
            add_field_to_var: str = "zero_scale_transform",
            max_value: float = 10.0,
            gene_chunk_size: int = 512,
            use_hvg: bool = False,
            hvg_key: str = "highly_variable"):
    """
    Gene-wise z-score scale（chunk 内 in-place UPDATE 版）

    - X_CSR_data：chunk 内直接 UPDATE（无临时表、无 merge）
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

    # -------------------------------------------------
    # 1. 输入字段检查
    # -------------------------------------------------
    if conn.execute(f"""
        SELECT COUNT(*)
        FROM information_schema.columns
        WHERE table_name='X_CSR_data'
          AND column_name='{select_data}'
    """).fetchone()[0] == 0:
        raise ValueError(f"X_CSR_data 中不存在字段: {select_data}")

    # -------------------------------------------------
    # 2. 输出字段准备
    # -------------------------------------------------
    conn.execute(f"""
        ALTER TABLE X_CSR_data
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
            SELECT id FROM var
            WHERE {hvg_key} = TRUE
            ORDER BY id
        """).fetchall()
        gene_ids = [g[0] for g in gene_ids]
    else:
        gene_ids = conn.execute("""
            SELECT DISTINCT indices
            FROM X_CSR_data
            ORDER BY indices
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
                indices,
                AVG({select_data})        AS mean,
                STDDEV_POP({select_data}) AS std
            FROM X_CSR_data
            WHERE indices IN ({gene_list_sql})
              AND {select_data} IS NOT NULL
            GROUP BY indices
        """)

        # ---------------------------------------------
        # 4.2 chunk 内直接 UPDATE X_CSR_data
        # ---------------------------------------------
        conn.execute(f"""
            UPDATE X_CSR_data x
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
            WHERE x.indices = g.indices
              AND x.indices IN ({gene_list_sql})
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
            WHERE v.id = g.indices
        """)

    print("\n==== scale_gene_chunked_inplace 完成 ====")
    print("耗时: {:.2f} 秒".format((datetime.now() - start_all).total_seconds()))


#========== todo sqrt  法 1 ： 在 X_CSR_data 表上直接处理 ==========
def sqrt(
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
        WHERE table_name = 'X_CSR_data'
          AND column_name = '{select_data}'
        """
    ).fetchone()[0]

    if col_exists == 0:
        raise ValueError(f"X_CSR_data 中不存在字段: {select_data}")

    if add_field is None:
        raise ValueError("必须指定 add_field")

    print(f"创建 / 使用字段 X_CSR_data.{add_field} ...")

    # -------------------------------------------------
    # 1. 构造 sqrt 表达式（不处理 0）
    # -------------------------------------------------
    sqrt_expr = f"sqrt({select_data})"

    # -------------------------------------------------
    # 2. 确保输出字段存在
    # -------------------------------------------------
    conn.execute(
        f"""
        ALTER TABLE X_CSR_data
        ADD COLUMN IF NOT EXISTS {add_field} REAL
        """
    )

    # -------------------------------------------------
    # 3. 执行 sqrt
    # -------------------------------------------------
    conn.execute(
        f"""
        UPDATE X_CSR_data
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


#========== todo sqrt  法 2 ： 法 1 +  分块 ==========
def sqrt_chunked(
    atlas: "Atlas",
    add_field: str = "data_sqrt_chunked",
    select_data: str = "data",
    chunk_size: int = 100_000_000) -> None:
    """
    1e8 级 CSR 安全的 sqrt 实现
    - 不一次性 UPDATE 全表
    - 按 id 分块
    - 支持指定 X_CSR_data 中的任意字段作为输入
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
        WHERE table_name = 'X_CSR_data'
          AND column_name = '{select_data}'
        """
    ).fetchone()[0]

    if col_exists == 0:
        raise ValueError(f"X_CSR_data 中不存在字段: {select_data}")

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
        ALTER TABLE X_CSR_data
        ADD COLUMN IF NOT EXISTS {add_field} REAL
        """
    )

    # -------------------------------------------------
    # 3. 获取 id 范围
    # -------------------------------------------------
    min_id, max_id = conn.execute(
        """
        SELECT MIN(id), MAX(id)
        FROM X_CSR_data
        """
    ).fetchone()

    if min_id is None:
        print("X_CSR_data 为空，跳过")
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
            UPDATE X_CSR_data
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