from ..data import Atlas
from typing import Literal
from typing import Optional
import logging
from numbers import Number
import math
import os
from datetime import datetime
import numpy as np
import pandas as pd

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
        ADD COLUMN {add_field} REAL
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


# todo 修改
# ✅ 不再有全量 _cell_sum
# ✅ 每次只产生 cell_chunk_size 级别的小临时表
# ✅ 仍然生成 data_normalize 字段
# ⚠️ 仍然需要一个完整 X_CSR_data_norm 新表，这是写回大表不可避免的代价
def normalize_total_chunked(
        atlas,
        target_sum: float = 10000,
        cell_chunk_size: int = 500_000,      # ✅ 修改1：从 id chunk 改成 cell chunk
        add_field: str = "data_normalize",
        select_data: str = "data"
) -> None:

    print("==== normalize_total_streaming_cell_chunk ====")
    start = datetime.now()

    conn = atlas.connection

    # -------------------------------------------------
    # 0. 设置线程
    # -------------------------------------------------
    try:
        conn.execute(f"PRAGMA threads={os.cpu_count() or 1}")
    except Exception:
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
    # 2. 删除源表旧 normalize 字段
    # -------------------------------------------------
    print("Step 1: drop old normalize column")

    conn.execute(f"""
        ALTER TABLE X_CSRO_data
        DROP COLUMN IF EXISTS {add_field}
    """)

    # -------------------------------------------------
    # 3. 创建目标表
    # -------------------------------------------------
    print("Step 2: create target table")

    conn.execute("DROP TABLE IF EXISTS X_CSR_data_norm")

    conn.execute("""
        CREATE TABLE X_CSR_data_norm AS
        SELECT * FROM X_CSRO_data WHERE 1=0
    """)

    conn.execute(f"""
        ALTER TABLE X_CSR_data_norm
        ADD COLUMN {add_field} REAL
    """)

    # -------------------------------------------------
    # 4. 获取 cell_id 范围
    # -------------------------------------------------
    min_cell, max_cell = conn.execute("""
        SELECT MIN(atlas_cell_id), MAX(atlas_cell_id)
        FROM X_CSRO_data
    """).fetchone()

    if min_cell is None:
        print("X_CSRO_data 为空，跳过")
        return

    n_chunks = math.ceil((max_cell - min_cell + 1) / cell_chunk_size)

    print(f"cell_id range = {min_cell:,} ~ {max_cell:,}")
    print(f"cell_chunk_size = {cell_chunk_size:,}")
    print(f"chunks = {n_chunks:,}")

    # -------------------------------------------------
    # 5. cell 分块：小 _cell_sum_chunk + 写入目标表
    # -------------------------------------------------
    for i in range(n_chunks):

        c_start = min_cell + i * cell_chunk_size
        c_end = min(c_start + cell_chunk_size - 1, max_cell)

        print(f"  -> chunk {i + 1}/{n_chunks}: cell [{c_start:,}, {c_end:,}]")

        # ✅ 修改2：只计算当前 cell chunk 的 sum
        # 不再创建全量 _cell_sum
        conn.execute("DROP TABLE IF EXISTS _cell_sum_chunk")

        conn.execute(f"""
            CREATE TEMP TABLE _cell_sum_chunk AS
            SELECT
                atlas_cell_id,
                SUM({select_data}) AS total
            FROM X_CSRO_data
            WHERE atlas_cell_id BETWEEN {c_start} AND {c_end}
            GROUP BY atlas_cell_id
            HAVING total > 0
        """)

        # ✅ 修改3：只写入当前 cell chunk 的 X 数据
        conn.execute(f"""
            INSERT INTO X_CSR_data_norm
            SELECT
                x.*,
                x.{select_data} * {float(target_sum)} / s.total AS {add_field}
            FROM X_CSRO_data x
            JOIN _cell_sum_chunk s
              ON x.atlas_cell_id = s.atlas_cell_id
            WHERE x.atlas_cell_id BETWEEN {c_start} AND {c_end}
        """)

        # ✅ 修改4：每个 chunk 后立即清理
        conn.execute("DROP TABLE IF EXISTS _cell_sum_chunk")

    # -------------------------------------------------
    # 6. 替换原表
    # -------------------------------------------------
    print("Step 4: replace table")

    conn.execute("DROP TABLE X_CSRO_data")
    conn.execute("ALTER TABLE X_CSR_data_norm RENAME TO X_CSRO_data")

    print("normalize_total_streaming_cell_chunk 完成")
    print("耗时: {:.2f} 秒".format((datetime.now() - start).total_seconds()))



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


# todo normalize_total_scale_factor_chunked
def normalize_total_scale_factor_chunked(
        atlas,
        target_sum: float = 10000,
        add_key: str = "scale_factor",
        select_data: str = "data",
        cell_chunk_size: int = 500_000,   # ✅ 修改1：新增 cell 分块
) -> None:
    """
    高性能 normalize_total（scale_factor only）

    ✅ 大数据安全版：
    - 不修改 X_CSRO_data
    - 不创建全量 _cell_sum
    - 按 cell_id 分块计算 total
    - 每个 chunk 只生成小 _cell_sum_chunk
    - 写入 obs.scale_factor
    """

    print("==== normalize_total (scale_factor only, CHUNKED) ====")
    start = datetime.now()

    conn = atlas.connection

    try:
        conn.execute(f"PRAGMA threads={os.cpu_count() or 1}")
    except Exception:
        pass

    # -------------------------------------------------
    # 0. 基本安全检查
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
    # 1. obs 添加 scale_factor 字段
    # -------------------------------------------------
    conn.execute(f"""
        ALTER TABLE obs
        ADD COLUMN IF NOT EXISTS {add_key} REAL
    """)

    # ✅ 修改2：先初始化，避免空 cell 或未命中 cell 保留旧值
    conn.execute(f"""
        UPDATE obs
        SET {add_key} = 0
    """)

    # -------------------------------------------------
    # 2. 获取 cell_id 范围
    # -------------------------------------------------
    min_cell, max_cell = conn.execute("""
        SELECT MIN(atlas_cell_id), MAX(atlas_cell_id)
        FROM obs
    """).fetchone()

    if min_cell is None or max_cell is None:
        print("obs 为空，跳过")
        return

    n_chunks = math.ceil((max_cell - min_cell + 1) / cell_chunk_size)

    print(f"cell_id range = {min_cell:,} ~ {max_cell:,}")
    print(f"cell_chunk_size = {cell_chunk_size:,}")
    print(f"chunks = {n_chunks:,}")

    # -------------------------------------------------
    # 3. 分块计算 total + 写回 obs
    # -------------------------------------------------
    for i in range(n_chunks):

        c_start = min_cell + i * cell_chunk_size
        c_end = min(c_start + cell_chunk_size - 1, max_cell)

        print(f"[Chunk {i + 1}/{n_chunks}] cells {c_start:,} ~ {c_end:,}")

        # ✅ 修改3：只计算当前 chunk 的 cell sum
        conn.execute("DROP TABLE IF EXISTS _cell_sum_chunk")

        conn.execute(f"""
            CREATE TEMP TABLE _cell_sum_chunk AS
            SELECT
                atlas_cell_id,
                SUM({select_data}) AS total
            FROM X_CSRO_data
            WHERE atlas_cell_id BETWEEN {c_start} AND {c_end}
            GROUP BY atlas_cell_id
        """)

        # ✅ 修改4：只更新当前 chunk 对应 obs
        conn.execute(f"""
            UPDATE obs
            SET {add_key} =
                CASE
                    WHEN s.total > 0
                    THEN {float(target_sum)} / s.total
                    ELSE 0
                END
            FROM _cell_sum_chunk AS s
            WHERE obs.atlas_cell_id = s.atlas_cell_id
        """)

        # ✅ 修改5：每个 chunk 后立即清理
        conn.execute("DROP TABLE IF EXISTS _cell_sum_chunk")

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
            select_data: str = "data_normalize" ) -> None:
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
                select_data: str = "data_normalize",
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
        DROP COLUMN IF EXISTS {add_field}
    """)

    conn.execute(f"""
        ALTER TABLE X_CSRO_data
        ADD COLUMN  {add_field} REAL
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


'''===== expm1 是  log1p的逆运算 ========== '''
# 833206 * 17745  2.46 秒 运行完别的函数再运行这个， 16.65 秒，
def expm1(
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

    logger.info("开始执行 expm1 (chunked)...")
    print("==== expm1 (chunked) ====")
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
        DROP COLUMN IF EXISTS {add_field}
    """)

    conn.execute(f"""
        ALTER TABLE X_CSRO_data
        ADD COLUMN {add_field} REAL
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
    print("expm1 (chunked) 完成")
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
    normalize_total_scale_factor_chunked(
        atlas=atlas,
        target_sum=target_sum,
        add_key=scale_key,
        select_data=select_data,
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
        DROP COLUMN IF EXISTS {add_field}
    """)

    conn.execute(f"""
        ALTER TABLE X_CSRO_data
        ADD COLUMN {add_field} REAL
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
# 类似 sc.pp.highly_variable_genes（全细胞含0统计版，最小化修改）
# 第二次运行覆盖第一次,旧的 TRUE 不会残留
def highly_variable_genes(
                        atlas: Atlas,
                        flavor: Literal["var", "cv"] = "var",
                        n_top_genes: int = 2000,
                        add_key: str = "highly_variable_genes",
                        select_data: str = "data_log1p"
                    ) -> None:
    """
    类似 sc.pp.highly_variable_genes（全细胞含0统计版，最小化修改）

    - 不修改 X_CSRO_data
    - 在 var 表中新建布尔字段 add_key
    - 使用“全细胞（含0）”定义的 mean / var / std
    - 不会真的补 0，仍然保持稀疏、大数据安全

    flavor:
        - "var": 按方差排序
        - "cv" : 按变异系数（std / mean）排序
    """

    print("==== highly_variable_genes (CSR + DuckDB, all-cells stats) ====")
    start = datetime.now()

    conn = atlas.connection
    try:
        conn.execute(f"PRAGMA threads={os.cpu_count()}")
    except:
        pass

    # -------------------------------------------------
    # 0️⃣ 检查字段存在
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
    # ✅【修改1】确保 var 表有可复用统计列
    # -------------------------------------------------
    conn.execute("""
        ALTER TABLE var
        ADD COLUMN IF NOT EXISTS hvg_mean REAL
    """)
    conn.execute("""
        ALTER TABLE var
        ADD COLUMN IF NOT EXISTS hvg_var REAL
    """)
    conn.execute("""
        ALTER TABLE var
        ADD COLUMN IF NOT EXISTS hvg_std REAL
    """)
    conn.execute("""
        ALTER TABLE var
        ADD COLUMN IF NOT EXISTS hvg_score REAL
    """)
    conn.execute("""
        ALTER TABLE var
        ADD COLUMN IF NOT EXISTS hvg_nnz BIGINT
    """)

    # -------------------------------------------------
    # ✅【修改2】取总细胞数 N（全细胞统计的关键）
    # -------------------------------------------------
    n_cells = conn.execute("""
        SELECT COUNT(*) FROM obs
    """).fetchone()[0]

    if n_cells == 0:
        raise ValueError("obs 为空，无法计算 highly_variable_genes")

    # -------------------------------------------------
    # 1️⃣ 计算每个 gene 的全细胞 mean / var / std
    #    不补 0，直接用 sum / sumsq / N_cells 推导
    # -------------------------------------------------
    print("Step 1: 计算 gene-level 统计量（全细胞含0）")

    conn.execute(f"""
        CREATE OR REPLACE TEMP TABLE _gene_stats AS
        WITH gene_sum AS (
            SELECT
                atlas_gene_id,
                COUNT(*) AS nnz,
                SUM({select_data}) AS sum_x,
                SUM(({select_data}) * ({select_data})) AS sum_x2
            FROM X_CSRO_data
            WHERE {select_data} IS NOT NULL
            GROUP BY atlas_gene_id
        )
        SELECT
            v.atlas_gene_id,
            COALESCE(g.nnz, 0) AS nnz,
            COALESCE(g.sum_x, 0.0) / {n_cells} AS mean,
            GREATEST(
                COALESCE(g.sum_x2, 0.0) / {n_cells}
                - POWER(COALESCE(g.sum_x, 0.0) / {n_cells}, 2),
                0.0
            ) AS var,
            SQRT(
                GREATEST(
                    COALESCE(g.sum_x2, 0.0) / {n_cells}
                    - POWER(COALESCE(g.sum_x, 0.0) / {n_cells}, 2),
                    0.0
                )
            ) AS std
        FROM var v
        LEFT JOIN gene_sum g
          ON v.atlas_gene_id = g.atlas_gene_id
    """)

    # -------------------------------------------------
    # 2️⃣ 计算排序指标
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

    # -------------------------------------------------
    # ✅【修改3】把统计量和 score 写回 var，后续画图直接复用
    # -------------------------------------------------
    print("Step 1.5: 写入 var.hvg_mean / hvg_var / hvg_std / hvg_score")

    conn.execute("""
        UPDATE var
        SET
            hvg_mean = NULL,
            hvg_var = NULL,
            hvg_std = NULL,
            hvg_score = NULL,
            hvg_nnz = NULL
    """)

    conn.execute("""
        UPDATE var v
        SET
            hvg_mean = s.mean,
            hvg_var  = s.var,
            hvg_std  = s.std,
            hvg_nnz  = s.nnz
        FROM _gene_stats s
        WHERE v.atlas_gene_id = s.atlas_gene_id
    """)

    conn.execute("""
        UPDATE var v
        SET
            hvg_score = gs.score
        FROM _gene_score gs
        WHERE v.atlas_gene_id = gs.atlas_gene_id
    """)

    # -------------------------------------------------
    # 3️⃣ 选 top genes
    # -------------------------------------------------
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
        print("Step 2: n_top_genes=None，全部标记为 True")

        conn.execute("""
            CREATE OR REPLACE TEMP TABLE _hvg AS
            SELECT atlas_gene_id
            FROM _gene_score
        """)

    # -------------------------------------------------
    # 4️⃣ 在 var 表中写入布尔结果
    # -------------------------------------------------
    print(f"Step 3: 写入 var.{add_key}")

    conn.execute(f"""
        ALTER TABLE var
        ADD COLUMN IF NOT EXISTS {add_key} BOOLEAN
    """)

    conn.execute(f"""
        UPDATE var
        SET {add_key} = FALSE
    """)

    conn.execute(f"""
        UPDATE var
        SET {add_key} = TRUE
        FROM _hvg
        WHERE var.atlas_gene_id = _hvg.atlas_gene_id
    """)

    # -------------------------------------------------
    # 5️⃣ 清理临时表
    # -------------------------------------------------
    conn.execute("DROP TABLE IF EXISTS _gene_stats")
    conn.execute("DROP TABLE IF EXISTS _gene_score")
    conn.execute("DROP TABLE IF EXISTS _hvg")

    # -------------------------------------------------
    # 6️⃣ 结束
    # -------------------------------------------------
    print("highly_variable_genes 完成（全细胞含0统计版）")
    print("耗时: {:.2f} 秒".format((datetime.now() - start).total_seconds()))

# 运行结果
#   var 表
#     新增字段	                     含义
#   highly_variable_genes	     对 n_top_genes 标记为true

# highly_variable_genes_like_seurat_v3 # 第二次运行覆盖第一次,旧的 TRUE 不会残留
def highly_variable_genes_like_seurat_v3(
        atlas,
        n_top_genes: int = 2000,
        add_key: str = "highly_variable_genes",
        select_data: str = "data",          # ✅ seurat_v3 用原始 counts
        n_bins: int = 20,
        min_mean: float | None = None,
        max_mean: float | None = None,
        inplace: bool = True
):
    """
    数据库版 flavor="seurat_v3"（第二版：bin 内标准化 variances_norm）

    思路
    ----
    1. SQL 在 X_CSRO_data 上按 gene 聚合：
       - nnz
       - sum_x
       - sum_x2
    2. 用全细胞（含0）公式在 Python 小表上算：
       - means
       - variances
    3. 对 genes 按 mean 分箱（qcut）
    4. 在 bin 内做标准化：
       variances_norm = (variance - mean_bin) / std_bin
    5. 按 variances_norm 排序，取 top n_top_genes
    6. 把结果写回 var：
       - add_key
       - highly_variable_rank
       - means
       - variances
       - variances_norm

    参数
    ----
    atlas : Atlas
    n_top_genes : int
        选前多少个高变基因
    add_key : str
        写回 var 的布尔列名，默认 "highly_variable_genes"
    select_data : str
        X_CSRO_data 中用于 HVG 的数据列，seurat_v3 推荐原始 counts，所以默认 "data"
    n_bins : int
        按 mean 分箱数量
    min_mean / max_mean : float | None
        可选 mean 过滤范围
    inplace : bool
        True -> 结果写回 var
        False -> 返回 DataFrame
    """

    print("==== highly_variable_genes_like_seurat_v3 (v2: bin-standardized) ====")
    start = datetime.now()
    conn = atlas.connection

    try:
        conn.execute(f"PRAGMA threads={os.cpu_count()}")
    except Exception:
        pass

    # -------------------------------------------------
    # 0️⃣ 检查输入列
    # -------------------------------------------------
    col_exists = conn.execute(f"""
        SELECT COUNT(*)
        FROM information_schema.columns
        WHERE table_name = 'X_CSRO_data'
          AND column_name = '{select_data}'
    """).fetchone()[0]

    if col_exists == 0:
        raise ValueError(f"X_CSRO_data 中不存在字段: {select_data}")

    n_cells = conn.execute("SELECT COUNT(*) FROM obs").fetchone()[0]
    if n_cells == 0:
        raise ValueError("obs 为空，无法计算 HVG")

    # -------------------------------------------------
    # 1️⃣ SQL：按 gene 聚合基础统计（只扫稀疏表一次）
    # -------------------------------------------------
    print("Step 1: SQL 聚合 gene-level 基础统计量")

    conn.execute(f"""
        CREATE OR REPLACE TEMP TABLE _gene_sum AS
        SELECT
            atlas_gene_id,
            COUNT(*) AS nnz,
            SUM({select_data}) AS sum_x,
            SUM(({select_data}) * ({select_data})) AS sum_x2
        FROM X_CSRO_data
        WHERE {select_data} IS NOT NULL
        GROUP BY atlas_gene_id
    """)

    gene_df = conn.execute("""
        SELECT
            v.atlas_gene_id,
            v.atlas_gene_name,
            COALESCE(g.nnz, 0) AS nnz,
            COALESCE(g.sum_x, 0.0) AS sum_x,
            COALESCE(g.sum_x2, 0.0) AS sum_x2
        FROM var v
        LEFT JOIN _gene_sum g
          ON v.atlas_gene_id = g.atlas_gene_id
        ORDER BY v.atlas_gene_id
    """).fetchdf()

    # -------------------------------------------------
    # 2️⃣ Python：全细胞（含0）统计
    # -------------------------------------------------
    print("Step 2: Python 计算全细胞 means / variances")

    means = gene_df["sum_x"].to_numpy(dtype=np.float64) / float(n_cells)
    variances = gene_df["sum_x2"].to_numpy(dtype=np.float64) / float(n_cells) - means ** 2
    variances = np.maximum(variances, 0.0)

    gene_df["means"] = means
    gene_df["variances"] = variances

    # -------------------------------------------------
    # 3️⃣ 可选 mean 过滤
    # -------------------------------------------------
    valid = np.ones(len(gene_df), dtype=bool)

    if min_mean is not None:
        valid &= gene_df["means"].to_numpy() >= float(min_mean)
    if max_mean is not None:
        valid &= gene_df["means"].to_numpy() <= float(max_mean)

    # 去掉完全0均值的基因
    valid &= gene_df["means"].to_numpy() > 0
    valid &= gene_df["variances"].to_numpy() >= 0

    gene_df["variances_norm"] = np.nan
    gene_df["highly_variable_rank"] = np.nan
    gene_df[add_key] = False

    work = gene_df.loc[valid, ["atlas_gene_id", "means", "variances"]].copy()

    if len(work) == 0:
        raise ValueError("没有可用于 seurat_v3 HVG 的基因（检查 select_data / min_mean / max_mean）")

    # -------------------------------------------------
    # 4️⃣ mean 分箱
    # -------------------------------------------------
    print("Step 3: mean 分箱")

    work["_mean_log1p"] = np.log1p(work["means"].to_numpy())

    try:
        work["_mean_bin"] = pd.qcut(
            work["_mean_log1p"],
            q=n_bins,
            labels=False,
            duplicates="drop"
        )
    except ValueError:
        work["_mean_bin"] = 0

    # -------------------------------------------------
    # 5️⃣ ✅【核心修改】bin 内标准化 variances_norm
    #     第一版是 ratio：
    #         var / median(var_bin)
    #     第二版改成 z-score 风格：
    #         (var - mean_bin) / std_bin
    # -------------------------------------------------
    print("Step 4: bin 内标准化 variances_norm")

    bin_mean = work.groupby("_mean_bin")["variances"].transform("mean").to_numpy()
    bin_std = work.groupby("_mean_bin")["variances"].transform("std").to_numpy()

    # 防止 std 为 0 或 NaN
    bin_std = np.where(np.isnan(bin_std), 0.0, bin_std)
    bin_std = np.maximum(bin_std, 1e-12)

    work["variances_norm"] = (work["variances"].to_numpy() - bin_mean) / bin_std

    # -------------------------------------------------
    # 6️⃣ 排序 + 选 top genes
    # -------------------------------------------------
    print(f"Step 5: 选取 top {n_top_genes} genes")

    work = work.sort_values(
        by=["variances_norm", "atlas_gene_id"],
        ascending=[False, True]
    ).reset_index(drop=True)

    work["highly_variable_rank"] = np.arange(len(work), dtype=np.float64)

    top_n = min(int(n_top_genes), len(work))
    top_ids = set(work.loc[:top_n - 1, "atlas_gene_id"].tolist())

    rank_map = dict(zip(work["atlas_gene_id"], work["highly_variable_rank"]))
    varnorm_map = dict(zip(work["atlas_gene_id"], work["variances_norm"]))

    gene_df["variances_norm"] = gene_df["atlas_gene_id"].map(varnorm_map)
    gene_df["highly_variable_rank"] = gene_df["atlas_gene_id"].map(rank_map)
    gene_df[add_key] = gene_df["atlas_gene_id"].isin(top_ids)

    # -------------------------------------------------
    # 7️⃣ 写回 var
    # -------------------------------------------------
    if inplace:
        print("Step 6: 写回 var")

        conn.execute(f"""
            ALTER TABLE var
            ADD COLUMN IF NOT EXISTS {add_key} BOOLEAN
        """)
        conn.execute("""
            ALTER TABLE var
            ADD COLUMN IF NOT EXISTS highly_variable_rank REAL
        """)
        conn.execute("""
            ALTER TABLE var
            ADD COLUMN IF NOT EXISTS means REAL
        """)
        conn.execute("""
            ALTER TABLE var
            ADD COLUMN IF NOT EXISTS variances REAL
        """)
        conn.execute("""
            ALTER TABLE var
            ADD COLUMN IF NOT EXISTS variances_norm REAL
        """)

        write_df = gene_df[[
            "atlas_gene_id",
            add_key,
            "highly_variable_rank",
            "means",
            "variances",
            "variances_norm"
        ]].copy()

        conn.register("_hvg_py", write_df)

        conn.execute(f"""
            UPDATE var AS v
            SET
                {add_key} = p.{add_key},
                highly_variable_rank = p.highly_variable_rank,
                means = p.means,
                variances = p.variances,
                variances_norm = p.variances_norm
            FROM _hvg_py AS p
            WHERE v.atlas_gene_id = p.atlas_gene_id
        """)

        conn.unregister("_hvg_py")

    # 清理
    conn.execute("DROP TABLE IF EXISTS _gene_sum")

    print("highly_variable_genes_like_seurat_v3 完成（v2）")
    print("耗时: {:.2f} 秒".format((datetime.now() - start).total_seconds()))

    if not inplace:
        return gene_df

# 数据库版 seurat_v3-like HVG
# X_CSRO_data 稀疏表达矩阵
#         ↓
# 每个 gene 统计 mean / variance
#         ↓
# 按 mean 分箱
# 低表达组： mean < 1
# 中表达组： 1 ~ 10
# 高表达组： > 10
#         ↓
# 👉 不同组之间不比较 👉 只在“同一表达水平”的 gene 之间比较
# 在每个表达水平 bin 内比较 variance
# variances_norm = (var - mean_bin) / std_bin
# ✅ 工程化简化版 Seurat v3
#         ↓
# 选 variances_norm 最大的前 n_top_genes 个基因
#         ↓
# 写回 var.highly_variable_genes


# todo 补充
def highly_variable_genes_seurat(
        atlas,
        n_top_genes: int = 2000,
        add_key: str = "highly_variable_genes",
        select_data: str = "data_log1p",
        n_bins: int = 20,
        min_mean: float = 0.0125,
        max_mean: float = 3.0,
        min_disp: float = 0.5,
        max_disp: float = float("inf"),
        use_filtered: bool = True,
        obs_filter_col: str = "filter_cells",
        var_filter_col: str = "filter_genes",
        inplace: bool = True,
):
    """
    数据库版 Scanpy flavor='seurat' HVG。

    尽量对齐 scanpy.pp.highly_variable_genes(flavor='seurat') 的核心逻辑：

    1. 输入默认使用 log-normalized data，例如 data_log1p
    2. 内部先 expm1 还原：
           x = exp(data_log1p) - 1
    3. 对每个 gene 计算全细胞含 0 的 mean / variance
       注意：variance 使用 sample variance，近似 Scanpy correction=1
    4. 计算 dispersion:
           dispersion = variance / mean
    5. Seurat flavor:
           means = log1p(mean)
           dispersions = log(dispersion)
    6. 按 means 等宽分箱，默认 n_bins=20
    7. 每个 bin 内：
           dispersions_norm = (dispersions - bin_mean) / bin_std
       如果某个 bin 只有一个 gene，则 dispersions_norm 设为 1
    8. 如果 n_top_genes 不为 None：
           按 dispersions_norm 选 top n_top_genes
       否则按 min_mean / max_mean / min_disp / max_disp cutoffs 选
    9. 写回 var:
           add_key
           means
           dispersions
           dispersions_norm
           highly_variable_rank

    Parameters
    ----------
    atlas
        scAtlasPy Atlas 对象，要求 atlas.connection 已连接。

    n_top_genes
        选取前多少个 HVG。
        如果不为 None，则 min_mean / max_mean / min_disp / max_disp 会被忽略，
        这和 Scanpy 的行为一致。

    add_key
        写回 var 的 HVG 布尔列名。

    select_data
        X_CSRO_data 中用于计算 HVG 的字段。
        对 flavor='seurat'，推荐使用 data_log1p。

    n_bins
        mean 分箱数量，Scanpy 默认 20。

    min_mean / max_mean / min_disp / max_disp
        当 n_top_genes=None 时使用的筛选阈值。
        当 n_top_genes 不为 None 时忽略。

    use_filtered
        如果 True，并且 obs.filter_cells / var.filter_genes 存在，
        则只在过滤后的 cells / genes 上计算 HVG。

    inplace
        True：写回 var。
        False：返回 gene_df，不写回。
    """

    import os
    import numpy as np
    import pandas as pd
    from datetime import datetime

    print("==== highly_variable_genes_seurat (Scanpy-like) ====")
    start = datetime.now()

    conn = atlas.connection

    if conn is None:
        raise ValueError("atlas.connection 为空，请先连接数据库")

    try:
        conn.execute(f"PRAGMA threads={os.cpu_count()}")
    except Exception:
        pass

    # -------------------------------------------------
    # 0. DuckDB 字段安全引用
    # -------------------------------------------------
    def _q(name: str) -> str:
        return '"' + name.replace('"', '""') + '"'

    # -------------------------------------------------
    # 1. 检查基础表
    # -------------------------------------------------
    for table_name in ["obs", "var", "X_CSRO_data"]:
        exists = conn.execute("""
            SELECT COUNT(*)
            FROM information_schema.tables
            WHERE table_name = ?
        """, [table_name]).fetchone()[0]

        if exists == 0:
            raise ValueError(f"数据库中不存在表: {table_name}")

    # -------------------------------------------------
    # 2. 检查 select_data 字段
    # -------------------------------------------------
    col_exists = conn.execute("""
        SELECT COUNT(*)
        FROM information_schema.columns
        WHERE table_name = 'X_CSRO_data'
          AND column_name = ?
    """, [select_data]).fetchone()[0]

    if col_exists == 0:
        raise ValueError(f"X_CSRO_data 中不存在字段: {select_data}")

    # -------------------------------------------------
    # 3. 检查 filter 字段是否存在
    # -------------------------------------------------
    obs_cols = [
        r[0]
        for r in conn.execute("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'obs'
        """).fetchall()
    ]

    var_cols = [
        r[0]
        for r in conn.execute("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'var'
        """).fetchall()
    ]

    has_obs_filter = obs_filter_col in obs_cols
    has_var_filter = var_filter_col in var_cols

    if use_filtered:
        print("[INFO] use_filtered=True")

        if has_obs_filter:
            print(f"[INFO] 使用 obs.{obs_filter_col}=TRUE 的 cells")
        else:
            print(f"[WARN] obs 中不存在 {obs_filter_col}，将使用全部 cells")

        if has_var_filter:
            print(f"[INFO] 使用 var.{var_filter_col}=TRUE 的 genes")
        else:
            print(f"[WARN] var 中不存在 {var_filter_col}，将使用全部 genes")
    else:
        print("[INFO] use_filtered=False，使用全部 cells / genes")

    # -------------------------------------------------
    # 4. 构建临时 keep 表
    # -------------------------------------------------
    conn.execute("DROP TABLE IF EXISTS _hvg_obs_keep")
    conn.execute("DROP TABLE IF EXISTS _hvg_var_keep")

    if use_filtered and has_obs_filter:
        conn.execute(f"""
            CREATE TEMP TABLE _hvg_obs_keep AS
            SELECT atlas_cell_id
            FROM obs
            WHERE {_q(obs_filter_col)} = TRUE
        """)
    else:
        conn.execute("""
            CREATE TEMP TABLE _hvg_obs_keep AS
            SELECT atlas_cell_id
            FROM obs
        """)

    if use_filtered and has_var_filter:
        conn.execute(f"""
            CREATE TEMP TABLE _hvg_var_keep AS
            SELECT atlas_gene_id, atlas_gene_name
            FROM var
            WHERE {_q(var_filter_col)} = TRUE
        """)
    else:
        conn.execute("""
            CREATE TEMP TABLE _hvg_var_keep AS
            SELECT atlas_gene_id, atlas_gene_name
            FROM var
        """)

    n_cells = conn.execute("""
        SELECT COUNT(*)
        FROM _hvg_obs_keep
    """).fetchone()[0]

    n_genes = conn.execute("""
        SELECT COUNT(*)
        FROM _hvg_var_keep
    """).fetchone()[0]

    if n_cells == 0:
        raise ValueError("用于 HVG 的 cell 数量为 0")

    if n_genes == 0:
        raise ValueError("用于 HVG 的 gene 数量为 0")

    print(f"[INFO] HVG cells = {n_cells:,}")
    print(f"[INFO] HVG genes = {n_genes:,}")
    print(f"[INFO] select_data = {select_data}")

    # -------------------------------------------------
    # 5. SQL 聚合 gene-level sum / sumsq
    #
    # Scanpy flavor='seurat' 输入是 log-normalized data，
    # 内部先 expm1(x)。
    #
    # 所以这里:
    #     x_raw = EXP(select_data) - 1
    #
    # 然后全细胞含 0 统计：
    #     sum_x
    #     sum_x2
    # -------------------------------------------------
    print("Step 1: SQL 聚合 gene-level sum / sumsq（expm1 后）")

    conn.execute(f"""
        CREATE OR REPLACE TEMP TABLE _gene_sum AS
        SELECT
            x.atlas_gene_id,
            COUNT(*) AS nnz,
            SUM(EXP(x.{_q(select_data)}) - 1.0) AS sum_x,
            SUM(POWER(EXP(x.{_q(select_data)}) - 1.0, 2)) AS sum_x2
        FROM X_CSRO_data AS x
        JOIN _hvg_obs_keep AS o
          ON x.atlas_cell_id = o.atlas_cell_id
        JOIN _hvg_var_keep AS v
          ON x.atlas_gene_id = v.atlas_gene_id
        WHERE x.{_q(select_data)} IS NOT NULL
        GROUP BY x.atlas_gene_id
    """)

    gene_df = conn.execute("""
        SELECT
            v.atlas_gene_id,
            v.atlas_gene_name,
            COALESCE(g.nnz, 0) AS nnz,
            COALESCE(g.sum_x, 0.0) AS sum_x,
            COALESCE(g.sum_x2, 0.0) AS sum_x2
        FROM _hvg_var_keep AS v
        LEFT JOIN _gene_sum AS g
          ON v.atlas_gene_id = g.atlas_gene_id
        ORDER BY v.atlas_gene_id
    """).fetchdf()

    # -------------------------------------------------
    # 6. Python 小表计算 mean / variance / dispersion
    # -------------------------------------------------
    print("Step 2: Python 计算 means / dispersions")

    sum_x = gene_df["sum_x"].to_numpy(dtype=np.float64)
    sum_x2 = gene_df["sum_x2"].to_numpy(dtype=np.float64)

    # mean：全细胞含 0
    mean_raw = sum_x / float(n_cells)

    # variance：sample variance，尽量对齐 Scanpy correction=1
    if n_cells > 1:
        var_raw = (sum_x2 - (sum_x ** 2) / float(n_cells)) / float(n_cells - 1)
    else:
        var_raw = np.zeros_like(mean_raw)

    var_raw = np.maximum(var_raw, 0.0)

    # Scanpy:
    # mean[mean == 0] = 1e-12
    mean_safe = mean_raw.copy()
    mean_safe[mean_safe == 0] = 1e-12

    dispersion = var_raw / mean_safe

    # Scanpy seurat:
    # dispersion[dispersion == 0] = nan
    # dispersion = log(dispersion)
    # mean = log1p(mean)
    dispersion = dispersion.astype(np.float64)
    dispersion[dispersion == 0] = np.nan

    dispersions = np.log(dispersion)
    means = np.log1p(mean_raw)

    gene_df["means"] = means
    gene_df["dispersions"] = dispersions

    # -------------------------------------------------
    # 7. mean 分箱：Scanpy seurat 使用 pd.cut(means, bins=n_bins)
    # -------------------------------------------------
    print("Step 3: mean 分箱")

    work = gene_df[[
        "atlas_gene_id",
        "atlas_gene_name",
        "means",
        "dispersions"
    ]].copy()

    # 注意：
    # pd.cut 和 Scanpy 一致，是等宽分箱；
    # 不是 qcut。
    try:
        work["mean_bin"] = pd.cut(
            work["means"],
            bins=n_bins
        )
    except ValueError:
        # 极端情况下 means 全一样
        work["mean_bin"] = pd.Series(["single_bin"] * len(work), index=work.index)

    # -------------------------------------------------
    # 8. bin 内计算 avg/dev
    #
    # Scanpy seurat:
    #     avg = mean(dispersions)
    #     dev = std(dispersions)
    #
    # 单 gene bin：
    #     dev = avg
    #     avg = 0
    # 这样 normalized dispersion = dispersion / dispersion = 1
    # -------------------------------------------------
    print("Step 4: bin 内标准化 dispersions_norm")

    disp_stats = work.groupby("mean_bin", observed=True)["dispersions"].agg(
        avg="mean",
        dev="std",
        count="count",
    )

    # 单 gene bin：模拟 Scanpy _postprocess_dispersions_seurat
    one_gene_bins = disp_stats["dev"].isna()

    if one_gene_bins.any():
        disp_stats.loc[one_gene_bins, "dev"] = disp_stats.loc[one_gene_bins, "avg"]
        disp_stats.loc[one_gene_bins, "avg"] = 0.0

    # 防止 dev 为 0
    disp_stats["dev"] = disp_stats["dev"].replace(0, np.nan)

    # 映射回每个 gene
    avg_map = disp_stats["avg"]
    dev_map = disp_stats["dev"]

    work["_disp_avg"] = work["mean_bin"].map(avg_map).astype(float)
    work["_disp_dev"] = work["mean_bin"].map(dev_map).astype(float)

    work["dispersions_norm"] = (
        (work["dispersions"] - work["_disp_avg"])
        / work["_disp_dev"]
    )

    # -------------------------------------------------
    # 9. 选择 HVG
    #
    # Scanpy 行为：
    # - 如果 n_top_genes 不为 None，则 cutoffs 被忽略
    # - 选 normalized dispersion 最高的 n_top_genes
    # - ties 可能导致数量略多；这里为了工程可控，严格选 top N
    # -------------------------------------------------
    print("Step 5: 选择 highly variable genes")

    work["highly_variable_rank"] = np.nan
    work[add_key] = False

    if n_top_genes is not None:
        valid_score = work["dispersions_norm"].replace([np.inf, -np.inf], np.nan)

        rank_df = work.loc[valid_score.notna()].copy()

        rank_df = rank_df.sort_values(
            by=["dispersions_norm", "atlas_gene_id"],
            ascending=[False, True],
        ).reset_index(drop=True)

        top_n = min(int(n_top_genes), len(rank_df))

        rank_df["highly_variable_rank"] = np.arange(
            len(rank_df),
            dtype=np.float64,
        )

        top_ids = set(rank_df.loc[:top_n - 1, "atlas_gene_id"].tolist())

        rank_map = dict(zip(
            rank_df["atlas_gene_id"],
            rank_df["highly_variable_rank"],
        ))

        work["highly_variable_rank"] = work["atlas_gene_id"].map(rank_map)
        work[add_key] = work["atlas_gene_id"].isin(top_ids)

        print(f"[INFO] n_top_genes={n_top_genes}，cutoffs 已忽略")

    else:
        # Scanpy cutoff 模式：nan_to_num 后判断范围
        score = work["dispersions_norm"].replace([np.inf, -np.inf], np.nan)
        score_for_cutoff = score.fillna(0.0)

        hv_mask = (
            (work["means"] > float(min_mean))
            & (work["means"] < float(max_mean))
            & (score_for_cutoff > float(min_disp))
            & (score_for_cutoff < float(max_disp))
        )

        work[add_key] = hv_mask.to_numpy()

        rank_df = work.loc[score.notna()].copy()
        rank_df = rank_df.sort_values(
            by=["dispersions_norm", "atlas_gene_id"],
            ascending=[False, True],
        ).reset_index(drop=True)

        rank_df["highly_variable_rank"] = np.arange(
            len(rank_df),
            dtype=np.float64,
        )

        rank_map = dict(zip(
            rank_df["atlas_gene_id"],
            rank_df["highly_variable_rank"],
        ))

        work["highly_variable_rank"] = work["atlas_gene_id"].map(rank_map)

        print(
            f"[INFO] cutoff 模式: "
            f"min_mean={min_mean}, max_mean={max_mean}, "
            f"min_disp={min_disp}, max_disp={max_disp}"
        )

    # -------------------------------------------------
    # 10. 合并回 gene_df
    # -------------------------------------------------
    gene_df = gene_df.drop(
        columns=[
            c for c in [
                "means",
                "dispersions",
                "dispersions_norm",
                "highly_variable_rank",
                add_key,
            ]
            if c in gene_df.columns
        ],
        errors="ignore",
    )

    gene_df = gene_df.merge(
        work[[
            "atlas_gene_id",
            "means",
            "dispersions",
            "dispersions_norm",
            "highly_variable_rank",
            add_key,
        ]],
        on="atlas_gene_id",
        how="left",
    )

    gene_df[add_key] = gene_df[add_key].fillna(False).astype(bool)

    hvg_count = int(gene_df[add_key].sum())
    print(f"[INFO] selected HVGs = {hvg_count:,}")

    # -------------------------------------------------
    # 11. 写回 var
    # -------------------------------------------------
    if inplace:
        print("Step 6: 写回 var")

        conn.execute(f"""
            ALTER TABLE var
            ADD COLUMN IF NOT EXISTS {_q(add_key)} BOOLEAN
        """)

        conn.execute("""
            ALTER TABLE var
            ADD COLUMN IF NOT EXISTS highly_variable_rank REAL
        """)

        conn.execute("""
            ALTER TABLE var
            ADD COLUMN IF NOT EXISTS means REAL
        """)

        conn.execute("""
            ALTER TABLE var
            ADD COLUMN IF NOT EXISTS dispersions REAL
        """)

        conn.execute("""
            ALTER TABLE var
            ADD COLUMN IF NOT EXISTS dispersions_norm REAL
        """)

        write_df = gene_df[[
            "atlas_gene_id",
            add_key,
            "highly_variable_rank",
            "means",
            "dispersions",
            "dispersions_norm",
        ]].copy()

        conn.register("_hvg_seurat_py", write_df)

        conn.execute(f"""
            UPDATE var AS v
            SET
                {_q(add_key)} = p.{_q(add_key)},
                highly_variable_rank = p.highly_variable_rank,
                means = p.means,
                dispersions = p.dispersions,
                dispersions_norm = p.dispersions_norm
            FROM _hvg_seurat_py AS p
            WHERE v.atlas_gene_id = p.atlas_gene_id
        """)

        conn.unregister("_hvg_seurat_py")

    # -------------------------------------------------
    # 12. 清理临时表
    # -------------------------------------------------
    conn.execute("DROP TABLE IF EXISTS _gene_sum")
    conn.execute("DROP TABLE IF EXISTS _hvg_obs_keep")
    conn.execute("DROP TABLE IF EXISTS _hvg_var_keep")

    print("highly_variable_genes_seurat 完成")
    print("耗时: {:.2f} 秒".format((datetime.now() - start).total_seconds()))

    if not inplace:
        return gene_df


'''=====  scale ：  进行 z-score转换 - 大数据安全 '''
# 保留原结构 + 保留原 id + 一次性算 gene_stat + 按 id 分块回写 + 直接 update
#  2840130 x 24552  耗时:  1187.70 秒   833206 * 17745 51.21 秒 大数据安全
def scale(
        atlas,
        select_data: str = "data_log1p",
        add_field: str = "data_scale",
        add_field_to_var: str = "zero_scale_transform",
        max_value: float = 10.0,
        use_hvg: bool = True,
        hvg_key: str = "highly_variable_genes",
        id_chunk_size: int = 20_000_000,
        ):
    """
    Gene-wise z-score scale（超大数据安全版：保留原结构 + 按 id 分块回写）

    ✅【修改版思路】
    1. 保留原有 scale_id_chunk 的整体框架
    2. 仍然一次性计算 _gene_stat
    3. 仍然按 id 范围分块
    4. ❌不再 CREATE TEMP TABLE _scale_chunk
    5. ✅改为当前 id chunk 直接 UPDATE X_CSRO_data
    6. 保留原顺序，不做 CTAS，不做 ORDER BY
    """

    print("\n==== scale_ultra_safe_update_by_id_chunk_direct ====")
    start_all = datetime.now()
    conn = atlas.connection

    # -------------------------------------------------
    # 0. 并行
    # -------------------------------------------------
    try:
        # ✅【保留】线程不要太大，避免 spill / temp 文件过多
        n_threads = 4
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
        WHERE table_name = 'X_CSRO_data'
          AND column_name = '{select_data}'
    """).fetchone()[0] == 0:
        raise ValueError(f"X_CSRO_data 中不存在字段: {select_data}")

    if conn.execute("""
        SELECT COUNT(*)
        FROM information_schema.columns
        WHERE table_name = 'X_CSRO_data'
          AND column_name = 'id'
    """).fetchone()[0] == 0:
        raise ValueError("X_CSRO_data 中不存在 id 字段，无法按 id 分块回写")

    if conn.execute("""
        SELECT COUNT(*)
        FROM information_schema.columns
        WHERE table_name = 'X_CSRO_data'
          AND column_name = 'atlas_gene_id'
    """).fetchone()[0] == 0:
        raise ValueError("X_CSRO_data 中不存在 atlas_gene_id 字段")

    # -------------------------------------------------
    # 2. 输出字段准备
    # -------------------------------------------------
    # ✅【修改】尽量最小改动：保留你的写法
    # 如果你后面想再优化，这里可以改成只 ADD，不 DROP/ADD
    conn.execute(f""" ALTER TABLE X_CSRO_data DROP COLUMN IF EXISTS {add_field} """)
    conn.execute(f""" ALTER TABLE X_CSRO_data ADD COLUMN IF NOT EXISTS {add_field} REAL """)

    conn.execute(f""" ALTER TABLE var DROP COLUMN IF EXISTS {add_field_to_var} """)
    conn.execute(f""" ALTER TABLE var ADD COLUMN IF NOT EXISTS {add_field_to_var} REAL """)

    # -------------------------------------------------
    # 3. 准备目标 gene 集合
    # -------------------------------------------------
    print("-> 准备 target genes ...")
    conn.execute("DROP TABLE IF EXISTS _target_genes")

    if use_hvg:
        print("-> 使用 HVG gene 子集")
        conn.execute(f"""
            CREATE TEMP TABLE _target_genes AS
            SELECT atlas_gene_id
            FROM var
            WHERE {hvg_key} = TRUE
        """)
    else:
        conn.execute("""
            CREATE TEMP TABLE _target_genes AS
            SELECT atlas_gene_id
            FROM var
        """)

    n_genes = conn.execute("""
        SELECT COUNT(*) FROM _target_genes
    """).fetchone()[0]

    if n_genes == 0:
        print("无 gene，退出")
        conn.execute("DROP TABLE IF EXISTS _target_genes")
        return

    print(f"-> Total target genes: {n_genes}")

    # -------------------------------------------------
    # 4. 一次性计算所有目标 gene 的 mean/std
    # -------------------------------------------------
    print("-> 计算 _gene_stat（一次，全局）...")
    t0 = datetime.now()

    conn.execute("DROP TABLE IF EXISTS _gene_stat")
    conn.execute(f"""
        CREATE TEMP TABLE _gene_stat AS
        SELECT
            x.atlas_gene_id,
            AVG(x.{select_data})        AS mean,
            STDDEV_POP(x.{select_data}) AS std
        FROM X_CSRO_data x
        JOIN _target_genes t
          ON x.atlas_gene_id = t.atlas_gene_id
        WHERE x.{select_data} IS NOT NULL
        GROUP BY x.atlas_gene_id
    """)

    print("   _gene_stat 完成，耗时: {:.2f} 秒".format(
        (datetime.now() - t0).total_seconds()
    ))

    # -------------------------------------------------
    # 5. 更新 var：记录 0 值 的缩放因子 -> z-score
    # -------------------------------------------------
    print("-> 更新 var ...")
    t0 = datetime.now()

    conn.execute("BEGIN")
    try:
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

        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise

    print("   var 更新完成，耗时: {:.2f} 秒".format(
        (datetime.now() - t0).total_seconds()
    ))

    # -------------------------------------------------
    # 6. 获取 id 范围
    # -------------------------------------------------
    print("-> 获取 id 范围 ...")
    min_id, max_id, total_rows = conn.execute("""
        SELECT MIN(id), MAX(id), COUNT(*)
        FROM X_CSRO_data
    """).fetchone()

    print(f"-> X_CSRO_data rows: {total_rows}")
    print(f"-> id range: {min_id} ~ {max_id}")
    print(f"-> id_chunk_size: {id_chunk_size}")

    n_chunks = math.ceil((max_id - min_id + 1) / id_chunk_size)
    print(f"-> Total id chunks: {n_chunks}")

    # -------------------------------------------------
    # 7. 按 id 分块直接 UPDATE
    # -------------------------------------------------
    print("-> 开始按 id 分块直接回写 ...")

    done_chunks = 0
    update_start_all = datetime.now()

    for chunk_idx in range(n_chunks):
        chunk_start = min_id + chunk_idx * id_chunk_size
        chunk_end = min(chunk_start + id_chunk_size - 1, max_id)

        t0 = datetime.now()
        print(
            f"\n[Chunk {chunk_idx + 1}/{n_chunks}] "
            f"id: {chunk_start} ~ {chunk_end}"
        )

        conn.execute("BEGIN")
        try:
            # =================================================
            # ✅【核心修改 1】先把当前 chunk 全部置 NULL
            # 这样：
            # - 非目标 gene 会保持 NULL
            # - select_data 为 NULL 的行也保持 NULL
            # - 避免旧值残留
            # =================================================
            conn.execute(f"""
                UPDATE X_CSRO_data
                SET {add_field} = NULL
                WHERE id BETWEEN {chunk_start} AND {chunk_end}
            """)

            # =================================================
            # ✅【核心修改 2】不再 CREATE TEMP TABLE _scale_chunk
            # ✅【核心修改 3】改为当前 chunk 直接 UPDATE
            #
            # 只更新：
            # - 在 _gene_stat 中能匹配到的目标 gene
            # - select_data 非 NULL 的行
            #
            # 保留原顺序：
            # - 不 CTAS
            # - 不 ORDER BY
            # - 不重建表
            # =================================================
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
                  AND x.id BETWEEN {chunk_start} AND {chunk_end}
                  AND x.{select_data} IS NOT NULL
            """)

            conn.execute("COMMIT")

        except Exception:
            conn.execute("ROLLBACK")
            raise

        chunk_seconds = (datetime.now() - t0).total_seconds()
        done_chunks += 1

        avg_chunk_seconds = (
            (datetime.now() - update_start_all).total_seconds() / done_chunks
        )
        remain_chunks = n_chunks - done_chunks
        eta_seconds = avg_chunk_seconds * remain_chunks

        print(
            "   本块完成，耗时: {:.2f} 秒 | 进度: {}/{} | 预计剩余: {:.2f} 分钟".format(
                chunk_seconds, done_chunks, n_chunks, eta_seconds / 60
            )
        )

    # -------------------------------------------------
    # 8. 清理临时表
    # -------------------------------------------------
    print("\n-> 清理临时表 ...")
    conn.execute("DROP TABLE IF EXISTS _target_genes")
    conn.execute("DROP TABLE IF EXISTS _gene_stat")

    print("\n==== scale_ultra_safe_update_by_id_chunk_direct 完成 ====")
    print("总耗时: {:.2f} 秒".format(
        (datetime.now() - start_all).total_seconds()
    ))



# todo 含0值统计，不存储，和scanpy的计算相同
# 保留原结构 + 保留原 id + 一次性算 gene_stat + 按 id 分块回写 + 直接 update 56.51 秒
def scale_zero(
        atlas,
        select_data: str = "data_log1p",
        add_field: str = "data_scale",
        add_field_to_var: str = "zero_scale_transform",
        max_value: float = 10.0,
        use_hvg: bool = True,
        hvg_key: str = "highly_variable_genes",
        id_chunk_size: int = 20_000_000,
        ):
    """
    Gene-wise z-score scale（超大数据安全版：保留原结构 + 按 id 分块回写）

    ✅ 含 0 版本：
    - mean/std 按全细胞统计，包括隐式 0
    - 不真的补 0
    - 不增加 X_CSRO_data 行数
    - 只改变 gene_stat 的计算公式
    """

    print("\n==== scale_ultra_safe_update_by_id_chunk_direct_zero_aware ====")
    start_all = datetime.now()
    conn = atlas.connection

    # -------------------------------------------------
    # 0. 并行
    # -------------------------------------------------
    try:
        n_threads = 4
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
        WHERE table_name = 'X_CSRO_data'
          AND column_name = '{select_data}'
    """).fetchone()[0] == 0:
        raise ValueError(f"X_CSRO_data 中不存在字段: {select_data}")

    if conn.execute("""
        SELECT COUNT(*)
        FROM information_schema.columns
        WHERE table_name = 'X_CSRO_data'
          AND column_name = 'id'
    """).fetchone()[0] == 0:
        raise ValueError("X_CSRO_data 中不存在 id 字段，无法按 id 分块回写")

    if conn.execute("""
        SELECT COUNT(*)
        FROM information_schema.columns
        WHERE table_name = 'X_CSRO_data'
          AND column_name = 'atlas_gene_id'
    """).fetchone()[0] == 0:
        raise ValueError("X_CSRO_data 中不存在 atlas_gene_id 字段")

    # -------------------------------------------------
    # 2. 输出字段准备
    # -------------------------------------------------
    conn.execute(f""" ALTER TABLE X_CSRO_data DROP COLUMN IF EXISTS {add_field} """)
    conn.execute(f""" ALTER TABLE X_CSRO_data ADD COLUMN IF NOT EXISTS {add_field} REAL """)

    conn.execute(f""" ALTER TABLE var DROP COLUMN IF EXISTS {add_field_to_var} """)
    conn.execute(f""" ALTER TABLE var ADD COLUMN IF NOT EXISTS {add_field_to_var} REAL """)

    # -------------------------------------------------
    # 3. 准备目标 gene 集合
    # -------------------------------------------------
    print("-> 准备 target genes ...")
    conn.execute("DROP TABLE IF EXISTS _target_genes")

    if use_hvg:
        print("-> 使用 HVG gene 子集")
        conn.execute(f"""
            CREATE TEMP TABLE _target_genes AS
            SELECT atlas_gene_id
            FROM var
            WHERE {hvg_key} = TRUE
        """)
    else:
        conn.execute("""
            CREATE TEMP TABLE _target_genes AS
            SELECT atlas_gene_id
            FROM var
        """)

    n_genes = conn.execute("""
        SELECT COUNT(*) FROM _target_genes
    """).fetchone()[0]

    if n_genes == 0:
        print("无 gene，退出")
        conn.execute("DROP TABLE IF EXISTS _target_genes")
        return

    print(f"-> Total target genes: {n_genes}")

    # -------------------------------------------------
    # ✅ 修改1：获取全细胞数量
    # -------------------------------------------------
    n_cells = conn.execute("""
        SELECT COUNT(*) FROM obs
    """).fetchone()[0]

    if n_cells == 0:
        conn.execute("DROP TABLE IF EXISTS _target_genes")
        raise ValueError("obs 为空，无法计算 scale")

    print(f"-> Total cells for zero-aware scaling: {n_cells:,}")

    # -------------------------------------------------
    # 4. 一次性计算所有目标 gene 的 mean/std
    # -------------------------------------------------
    print("-> 计算 _gene_stat（一次，全局，含 0 统计）...")
    t0 = datetime.now()

    conn.execute("DROP TABLE IF EXISTS _gene_stat")

    # =================================================
    # ✅ 修改2：把 AVG / STDDEV_POP 改成“含 0”公式
    #
    # 原来：
    #   AVG(x.select_data)
    #   STDDEV_POP(x.select_data)
    #
    # 现在：
    #   mean = SUM(x) / n_cells
    #   std  = sqrt(SUM(x^2) / n_cells - mean^2)
    #
    # 注意：
    #   这里仍然只扫描 CSR 中的非零行
    #   不会补 0
    #   不会增加 X_CSRO_data 存储量
    # =================================================
    conn.execute(f"""
        CREATE TEMP TABLE _gene_stat AS
        SELECT
            x.atlas_gene_id,

            SUM(x.{select_data}) / {n_cells} AS mean,

            SQRT(
                GREATEST(
                    SUM(x.{select_data} * x.{select_data}) / {n_cells}
                    - POWER(SUM(x.{select_data}) / {n_cells}, 2),
                    0.0
                )
            ) AS std

        FROM X_CSRO_data x
        JOIN _target_genes t
          ON x.atlas_gene_id = t.atlas_gene_id
        WHERE x.{select_data} IS NOT NULL
        GROUP BY x.atlas_gene_id
    """)

    print("   _gene_stat 完成，耗时: {:.2f} 秒".format(
        (datetime.now() - t0).total_seconds()
    ))

    # -------------------------------------------------
    # 5. 更新 var：记录 0 值 的缩放因子 -> z-score
    # -------------------------------------------------
    print("-> 更新 var ...")
    t0 = datetime.now()

    conn.execute("BEGIN")
    try:
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

        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise

    print("   var 更新完成，耗时: {:.2f} 秒".format(
        (datetime.now() - t0).total_seconds()
    ))

    # -------------------------------------------------
    # 6. 获取 id 范围
    # -------------------------------------------------
    print("-> 获取 id 范围 ...")
    min_id, max_id, total_rows = conn.execute("""
        SELECT MIN(id), MAX(id), COUNT(*)
        FROM X_CSRO_data
    """).fetchone()

    print(f"-> X_CSRO_data rows: {total_rows}")
    print(f"-> id range: {min_id} ~ {max_id}")
    print(f"-> id_chunk_size: {id_chunk_size}")

    n_chunks = math.ceil((max_id - min_id + 1) / id_chunk_size)
    print(f"-> Total id chunks: {n_chunks}")

    # -------------------------------------------------
    # 7. 按 id 分块直接 UPDATE
    # -------------------------------------------------
    print("-> 开始按 id 分块直接回写 ...")

    done_chunks = 0
    update_start_all = datetime.now()

    for chunk_idx in range(n_chunks):
        chunk_start = min_id + chunk_idx * id_chunk_size
        chunk_end = min(chunk_start + id_chunk_size - 1, max_id)

        t0 = datetime.now()
        print(
            f"\n[Chunk {chunk_idx + 1}/{n_chunks}] "
            f"id: {chunk_start} ~ {chunk_end}"
        )

        conn.execute("BEGIN")
        try:
            # todo ✅ 修改：删除 chunk 内的 SET NULL
            # 当前 chunk 先置 NULL，避免旧值残留
            # conn.execute(f"""
            #     UPDATE X_CSRO_data
            #     SET {add_field} = NULL
            #     WHERE id BETWEEN {chunk_start} AND {chunk_end}
            # """)
            #
            # 原因：
            # 前面已经 DROP COLUMN + ADD COLUMN，
            # 新增的 {add_field} 默认就是 NULL。
            #
            # 所以这里不需要每个 chunk 再 UPDATE 成 NULL，
            # 否则会导致每个 chunk 多写一遍大表。
            # =================================================

            # 对显式存储的非零值写入 z-score
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
                  AND x.id BETWEEN {chunk_start} AND {chunk_end}
                  AND x.{select_data} IS NOT NULL
            """)

            conn.execute("COMMIT")

        except Exception:
            conn.execute("ROLLBACK")
            raise

        chunk_seconds = (datetime.now() - t0).total_seconds()
        done_chunks += 1

        avg_chunk_seconds = (
            (datetime.now() - update_start_all).total_seconds() / done_chunks
        )
        remain_chunks = n_chunks - done_chunks
        eta_seconds = avg_chunk_seconds * remain_chunks

        print(
            "   本块完成，耗时: {:.2f} 秒 | 进度: {}/{} | 预计剩余: {:.2f} 分钟".format(
                chunk_seconds, done_chunks, n_chunks, eta_seconds / 60
            )
        )

    # -------------------------------------------------
    # 8. 清理临时表
    # -------------------------------------------------
    print("\n-> 清理临时表 ...")
    conn.execute("DROP TABLE IF EXISTS _target_genes")
    conn.execute("DROP TABLE IF EXISTS _gene_stat")

    print("\n==== scale_ultra_safe_update_by_id_chunk_direct_zero_aware 完成 ====")
    print("总耗时: {:.2f} 秒".format(
        (datetime.now() - start_all).total_seconds()
    ))

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
        select_data: str = "data_log1p",
        add_field: str = "data_scale",
        add_field_to_var: str = "zero_scale_transform",
        max_value: float = 10.0,
        use_hvg: bool = True,
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


# scale_fast_zero
def scale_fast_zero(
        atlas,
        select_data: str = "data_log1p",
        add_field: str = "data_scale",
        add_field_to_var: str = "zero_scale_transform",
        max_value: float = 10.0,
        use_hvg: bool = True,
        hvg_key: str = "highly_variable_genes"):
    """
    工业级 Gene-wise z-score scale（含 0 统计版）

    特点：
    - mean/std 按全细胞含 0 统计
    - 不真的补 0
    - X_CSRO_data 只写显式非零值的 scale
    - var.zero_scale_transform 存隐式 0 的 scale 值
    """

    print("\n==== scale_ultra (zero-aware industrial OLAP optimized) ====")
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
    # ✅ 修改1：获取总细胞数，用于含 0 统计
    # -------------------------------------------------
    n_cells = conn.execute("""
        SELECT COUNT(*) FROM obs
    """).fetchone()[0]

    if n_cells == 0:
        raise ValueError("obs 为空，无法计算 scale")

    print(f"-> Total cells for zero-aware scaling: {n_cells:,}")

    # -------------------------------------------------
    # 4. 计算 gene-wise 统计 + 线性化系数 a,b
    # ✅ 修改2：AVG/STDDEV_POP 改成含 0 公式
    # -------------------------------------------------
    print("-> 计算 gene-wise mean/std -> a,b（含 0 统计）")

    conn.execute(f"""
        CREATE OR REPLACE TEMP TABLE _gene_stat AS
        WITH gene_sum AS (
            SELECT
                atlas_gene_id,
                SUM({select_data}) AS sum_x,
                SUM({select_data} * {select_data}) AS sum_x2
            FROM X_CSRO_data
            WHERE atlas_gene_id IN ({gene_list_sql})
              AND {select_data} IS NOT NULL
            GROUP BY atlas_gene_id
        ),
        gene_stat AS (
            SELECT
                atlas_gene_id,
                sum_x / {n_cells} AS mean,
                SQRT(
                    GREATEST(
                        sum_x2 / {n_cells}
                        - POWER(sum_x / {n_cells}, 2),
                        0.0
                    )
                ) AS std
            FROM gene_sum
        )
        SELECT
            atlas_gene_id,
            mean,
            std,
            CASE WHEN std > 0
                 THEN 1.0 / std
                 ELSE 0
            END AS a,
            CASE WHEN std > 0
                 THEN -mean / std
                 ELSE 0
            END AS b
        FROM gene_stat
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

    conn.execute("DROP TABLE IF EXISTS _gene_stat")

    print("\n==== scale_ultra zero-aware 完成 ====")
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
    # 先删除旧列，避免旧值残留
    conn.execute(f"""
        ALTER TABLE X_CSRO_data
        DROP COLUMN IF EXISTS {add_field}
    """)

    conn.execute(f"""
        ALTER TABLE X_CSRO_data
        ADD COLUMN {add_field} REAL
    """)

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
    # ✅ 修改1：先删旧列
    conn.execute(f"""
        ALTER TABLE X_CSRO_data
        DROP COLUMN IF EXISTS {add_field}
    """)

    # ✅ 修改2：重新添加 REAL 字段
    conn.execute(f"""
        ALTER TABLE X_CSRO_data
        ADD COLUMN {add_field} REAL
    """)

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