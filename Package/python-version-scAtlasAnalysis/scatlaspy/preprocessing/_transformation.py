from _duckdb import DuckDBPyConnection
from ..data import Atlas
from typing import Any
from typing import Literal
from typing import Optional
import logging
from numbers import Number
import math
import os
import numpy as np
import pandas as pd
from datetime import datetime
import gc

# 获取日志记录器
logger = logging.getLogger('Atlas')

def _cleanup_transform_after_step(
        conn: DuckDBPyConnection,
        temp_tables: list[str]=None,
        unregister_tables: list[str]=None,
        checkpoint: bool = False,
        collect: bool = True,
):
    """清理当前步骤产生的临时资源。

    该内部函数属于表达矩阵转换模块，用于支撑同一模块中的公共 API。

    对 ``X_HyS_data`` 执行 normalize、log1p、sqrt、scale 和 HVG 筛选。

    它通常不会作为用户入口直接调用；直接调用时需要保证输入对象、数据库连接和相关临时表已经由上游步骤准备好。

    Parameters
    ----------
    conn
        DuckDB 数据库连接。

    temp_tables
        需要清理的临时表名称列表。

    unregister_tables
        需要从 DuckDB 连接中 unregister 的临时 relation 名称列表。

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

    if unregister_tables is None:
        unregister_tables = []

    # 1. 取消注册 pandas / Arrow 临时对象
    for t in unregister_tables:
        try:
            conn.unregister(t)
        except Exception:
            pass

    # 2. 删除 DuckDB 临时表
    for t in temp_tables:
        try:
            conn.execute(f"DROP TABLE IF EXISTS {t}")
        except Exception:
            pass

    # 3. 大表 UPDATE / DROP / RENAME 后建议 checkpoint
    if checkpoint:
        try:
            conn.execute("CHECKPOINT")
        except Exception:
            pass

    # 4. Python 层垃圾回收
    if collect:
        try:
            gc.collect()
        except Exception:
            pass

'''normalize 法 1 ： 小内存 快速版 '''
def normalize_total_fast(
        atlas: Atlas,
        target_sum: float = 10000,
        chunk_ids: int = 50_000_000,
        add_field: str = "data_normalize",
        select_data: str = "data"
) -> None:

    """快速执行 total-count 归一化。

    该函数在 DuckDB 中统计每个细胞的总表达量，计算 ``target_sum / total`` 缩放因子，并把归一化后的表达值写入
    ``X_HyS_data`` 的新字段。

    功能上类似 ``scanpy.pp.normalize_total``，但不会覆盖原始表达字段，便于在同一数据库中同时保留
    raw、normalized 和 log-scale 表达。

    Parameters
    ----------
    atlas
        Atlas 对象。通常要求已经连接数据库，并包含该函数所需的 ``obs``、``var``、``X_HyS_data`` 或
        embedding 结果表。

    target_sum
        归一化后每个细胞的目标总表达量。

    chunk_ids
        分块处理大小，用于控制内存峰值和单次 SQL 更新规模。

    add_field
        写入 ``X_HyS_data`` 的新表达字段名。

    select_data
        从 ``X_HyS_data`` 中读取的表达字段。

    Notes
    -----
    该函数会把结果写回 Atlas 数据库；如果后续需要只使用过滤后的细胞或基因，通常还需要重建过滤索引。

    Examples
    --------
    调用该函数：::

        sap.pp.normalize_total_fast(...)
    """
    print("==== normalize_total_streaming ====")
    start = datetime.now()

    conn = atlas.connection

    # 0. 设置线程（可选）
    try:
        conn.execute(f"PRAGMA threads={os.cpu_count()}")
    except:
        pass

    # 1. 字段检查
    col_exists = conn.execute(f"""
        SELECT COUNT(*)
        FROM information_schema.columns
        WHERE table_name = 'X_HyS_data'
          AND column_name = '{select_data}'
    """).fetchone()[0]

    if col_exists == 0:
        raise ValueError(f"X_HyS_data 中不存在字段: {select_data}")

    # 2. 计算每个 cell 的总表达（只做一次）
    print("Step 1: compute cell sums")

    conn.execute(f"""
        CREATE OR REPLACE TEMP TABLE _cell_sum AS
        SELECT
            atlas_cell_id,
            SUM({select_data}) AS total
        FROM X_HyS_data
        GROUP BY atlas_cell_id
        HAVING total > 0
    """)

    # 先删除源表中的旧列（保证 x.* 不冲突）
    print("Step 1.5: drop old column in source table")

    conn.execute(f"""
        ALTER TABLE X_HyS_data
        DROP COLUMN IF EXISTS {add_field}
    """)

    # 3. 创建目标表（结构复制 + 新列）
    print("Step 2: create target table")

    conn.execute("""
        CREATE OR REPLACE TABLE X_HyS_data_norm AS
        SELECT * FROM X_HyS_data WHERE 1=0
    """)

    # 现在源表已经没有该列 → 可以安全添加
    conn.execute(f"""
        ALTER TABLE X_HyS_data_norm
        ADD COLUMN {add_field} REAL
    """)

    # 4. 获取 id 范围
    min_id, max_id = conn.execute("""
        SELECT MIN(id), MAX(id)
        FROM X_HyS_data
    """).fetchone()

    if min_id is None:
        print("X_HyS_data 为空，跳过")
        return

    n_chunks = math.ceil((max_id - min_id + 1) / chunk_ids)
    print(f"Step 3: {n_chunks} chunks")

    # 5. 分块 INSERT（保持 x.*）
    for i in range(n_chunks):
        start_id = min_id + i * chunk_ids
        end_id = start_id + chunk_ids - 1

        print(f"  -> chunk {i+1}/{n_chunks}: id [{start_id}, {end_id}]")

        conn.execute(f"""
            INSERT INTO X_HyS_data_norm
            SELECT
                x.*,
                x.{select_data} * {float(target_sum)} / s.total AS {add_field}
            FROM X_HyS_data x
            JOIN _cell_sum s
              ON x.atlas_cell_id = s.atlas_cell_id
            WHERE x.id BETWEEN {start_id} AND {end_id}
            ORDER BY x.id
        """)

    # 6. 替换原表
    print("Step 4: replace table")

    conn.execute("DROP TABLE X_HyS_data")
    conn.execute("ALTER TABLE X_HyS_data_norm RENAME TO X_HyS_data")
    # 内存清理
    _cleanup_transform_after_step(
        conn,
        temp_tables=["_cell_sum"],
        checkpoint=True,
        collect=True,
    )

    print("normalize_total_streaming 完成")
    print("耗时: {:.2f} 秒".format((datetime.now() - start).total_seconds()))


''' normalize 法 2 ： 大数据 安全版 '''
def normalize_total(
        atlas: Atlas,
        target_sum: float = 10000,
        chunk_cells: int = 500_000,  
        add_field: str = "data_normalize",
        select_data: str = "data"
) -> None:

    """执行 total-count 归一化。

    该函数按细胞分块计算总表达量，并将每个非零表达值按 ``target_sum`` 重新缩放后写入新字段。

    与 fast 版本相比，该实现更强调分块稳健性，适合在内存或临时表压力较大时使用。

    Parameters
    ----------
    atlas
        Atlas 对象。通常要求已经连接数据库，并包含该函数所需的 ``obs``、``var``、``X_HyS_data`` 或
        embedding 结果表。

    target_sum
        归一化后每个细胞的目标总表达量。

    chunk_cells
        按细胞分块处理时每个 chunk 的细胞数量。

    add_field
        写入 ``X_HyS_data`` 的新表达字段名。

    select_data
        从 ``X_HyS_data`` 中读取的表达字段。

    Notes
    -----
    该函数会把结果写回 Atlas 数据库；如果后续需要只使用过滤后的细胞或基因，通常还需要重建过滤索引。

    Examples
    --------
    调用该函数：::

        sap.pp.normalize_total(...)
    """
    print("==== normalize_total_streaming_cell_chunk ====")
    start = datetime.now()

    conn = atlas.connection

    # 0. 设置线程
    try:
        conn.execute(f"PRAGMA threads={os.cpu_count() or 1}")
    except Exception:
        pass

    # 1. 字段检查
    col_exists = conn.execute(f"""
        SELECT COUNT(*)
        FROM information_schema.columns
        WHERE table_name = 'X_HyS_data'
          AND column_name = '{select_data}'
    """).fetchone()[0]

    if col_exists == 0:
        raise ValueError(f"X_HyS_data 中不存在字段: {select_data}")

    # 2. 删除源表旧 normalize 字段
    print("Step 1: drop old normalize column")

    conn.execute(f"""
        ALTER TABLE X_HyS_data
        DROP COLUMN IF EXISTS {add_field}
    """)

    # 3. 创建目标表
    print("Step 2: create target table")

    conn.execute("DROP TABLE IF EXISTS X_HyS_data_norm")

    conn.execute("""
        CREATE TABLE X_HyS_data_norm AS
        SELECT * FROM X_HyS_data WHERE 1=0
    """)

    conn.execute(f"""
        ALTER TABLE X_HyS_data_norm
        ADD COLUMN {add_field} REAL
    """)

    # 4. 获取 cell_id 范围
    min_cell, max_cell = conn.execute("""
        SELECT MIN(atlas_cell_id), MAX(atlas_cell_id)
        FROM X_HyS_data
    """).fetchone()

    if min_cell is None:
        print("X_HyS_data 为空，跳过")
        return

    n_chunks = math.ceil((max_cell - min_cell + 1) / chunk_cells)

    print(f"cell_id range = {min_cell:,} ~ {max_cell:,}")
    print(f"chunk_cells = {chunk_cells:,}")
    print(f"chunks = {n_chunks:,}")

    # 5. cell 分块：小 _cell_sum_chunk + 写入目标表
    for i in range(n_chunks):

        c_start = min_cell + i * chunk_cells
        c_end = min(c_start + chunk_cells - 1, max_cell)

        print(f"  -> chunk {i + 1}/{n_chunks}: cell [{c_start:,}, {c_end:,}]")

        # 只计算当前 cell chunk 的 sum
        conn.execute("DROP TABLE IF EXISTS _cell_sum_chunk")

        conn.execute(f"""
            CREATE TEMP TABLE _cell_sum_chunk AS
            SELECT
                atlas_cell_id,
                SUM({select_data}) AS total
            FROM X_HyS_data
            WHERE atlas_cell_id BETWEEN {c_start} AND {c_end}
            GROUP BY atlas_cell_id
            HAVING total > 0
        """)

        # 只写入当前 cell chunk 的 X 数据
        conn.execute(f"""
            INSERT INTO X_HyS_data_norm
            SELECT
                x.*,
                x.{select_data} * {float(target_sum)} / s.total AS {add_field}
            FROM X_HyS_data x
            JOIN _cell_sum_chunk s
              ON x.atlas_cell_id = s.atlas_cell_id
            WHERE x.atlas_cell_id BETWEEN {c_start} AND {c_end}
            ORDER BY x.id
        """)

        # 每个 chunk 后立即清理
        conn.execute("DROP TABLE IF EXISTS _cell_sum_chunk")

    # 6. 替换原表
    print("Step 4: replace table")

    conn.execute("DROP TABLE X_HyS_data")
    conn.execute("ALTER TABLE X_HyS_data_norm RENAME TO X_HyS_data")

    # 内存清理
    _cleanup_transform_after_step(
        conn,
        temp_tables=["_cell_sum_chunk"],
        checkpoint=True,
        collect=True,
    )

    print("normalize_total_streaming_cell_chunk 完成")
    print("耗时: {:.2f} 秒".format((datetime.now() - start).total_seconds()))

# atlas.connection.execute("CHECKPOINT")
# atlas.connection.close()
# atlas.connection = atlas.connect("r+")
# gc.collect()

# 运行结果
#   X_HyS_data 表
#     新增字段	             含义
#  data_normalize	     归一化后的data值


''' normalize 法 3： 小内存 快速版，在 obs表上记录 scale_factor ， 等到使用的时候再计算 '''
def normalize_total_scale_factor_fast(
                    atlas: Atlas,
                    target_sum: float = 10000,
                    add_key: str = "scale_factor",
                    select_data: str = "data" ) -> None:
    """快速计算 total-count 归一化缩放因子。

    该函数只计算每个细胞的 total-count scale factor，并写入 ``obs``，不直接修改 ``X_HyS_data`` 表达值。

    当多个表达字段需要复用同一套归一化系数时，可以先计算 scale factor，再在后续步骤中引用。

    Parameters
    ----------
    atlas
        Atlas 对象。通常要求已经连接数据库，并包含该函数所需的 ``obs``、``var``、``X_HyS_data`` 或
        embedding 结果表。

    target_sum
        归一化后每个细胞的目标总表达量。

    add_key
        写回 ``obs`` 或 ``var`` 的结果列名。

    select_data
        从 ``X_HyS_data`` 中读取的表达字段。

    Notes
    -----
    该函数会把结果写回 Atlas 数据库；如果后续需要只使用过滤后的细胞或基因，通常还需要重建过滤索引。

    Examples
    --------
    调用该函数：::

        sap.pp.normalize_total_scale_factor_fast(...)
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
        WHERE table_name = 'X_HyS_data'
          AND column_name = '{select_data}'
    """).fetchone()[0]

    if col_exists == 0:
        raise ValueError(f"X_HyS_data 中不存在字段: {select_data}")

    # 1. 计算每个 cell 的 total
    conn.execute(f"""
        CREATE OR REPLACE TEMP TABLE _cell_sum AS
        SELECT
            atlas_cell_id,
            SUM({select_data}) AS total
        FROM X_HyS_data
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

    # 内存清理
    _cleanup_transform_after_step(
        conn,
        temp_tables=["_cell_sum"],
        checkpoint=False,
        collect=True,
    )

    print(f"normalize_total 完成，target_sum={target_sum}")
    print("耗时: {:.2f} 秒".format((datetime.now() - start).total_seconds()))


''' normalize 法 4： 大数据 安全版，在 obs表上记录 scale_factor ， 等到使用的时候再计算 '''
def normalize_total_scale_factor(
        atlas: Atlas,
        target_sum: float = 10000,
        add_key: str = "scale_factor",
        select_data: str = "data",
        chunk_cells: int = 500_000,
) -> None:
    """计算 total-count 归一化缩放因子。

    该函数分块统计每个细胞的总表达量，并把归一化缩放因子写入 ``obs`` 指定列。

    结果可用于后续自定义 SQL 转换或检查不同细胞测序深度的差异。

    Parameters
    ----------
    atlas
        Atlas 对象。通常要求已经连接数据库，并包含该函数所需的 ``obs``、``var``、``X_HyS_data`` 或
        embedding 结果表。

    target_sum
        归一化后每个细胞的目标总表达量。

    add_key
        写回 ``obs`` 或 ``var`` 的结果列名。

    select_data
        从 ``X_HyS_data`` 中读取的表达字段。

    chunk_cells
        按细胞分块处理时每个 chunk 的细胞数量。

    Notes
    -----
    该函数会把结果写回 Atlas 数据库；如果后续需要只使用过滤后的细胞或基因，通常还需要重建过滤索引。

    Examples
    --------
    调用该函数：::

        sap.pp.normalize_total_scale_factor(...)
    """

    print("==== normalize_total (scale_factor only, CHUNKED) ====")
    start = datetime.now()

    conn = atlas.connection

    try:
        conn.execute(f"PRAGMA threads={os.cpu_count() or 1}")
    except Exception:
        pass

    # 0. 基本安全检查
    col_exists = conn.execute(f"""
        SELECT COUNT(*)
        FROM information_schema.columns
        WHERE table_name = 'X_HyS_data'
          AND column_name = '{select_data}'
    """).fetchone()[0]

    if col_exists == 0:
        raise ValueError(f"X_HyS_data 中不存在字段: {select_data}")

    # 1. obs 添加 scale_factor 字段
    conn.execute(f"""
        ALTER TABLE obs
        ADD COLUMN IF NOT EXISTS {add_key} REAL
    """)

    # 先初始化，避免空 cell 或未命中 cell 保留旧值
    conn.execute(f"""
        UPDATE obs
        SET {add_key} = 0
    """)

    # 2. 获取 cell_id 范围
    min_cell, max_cell = conn.execute("""
        SELECT MIN(atlas_cell_id), MAX(atlas_cell_id)
        FROM obs
    """).fetchone()

    if min_cell is None or max_cell is None:
        print("obs 为空，跳过")
        return

    n_chunks = math.ceil((max_cell - min_cell + 1) / chunk_cells)

    print(f"cell_id range = {min_cell:,} ~ {max_cell:,}")
    print(f"chunk_cells = {chunk_cells:,}")
    print(f"chunks = {n_chunks:,}")

    # 3. 分块计算 total + 写回 obs
    for i in range(n_chunks):

        c_start = min_cell + i * chunk_cells
        c_end = min(c_start + chunk_cells - 1, max_cell)

        print(f"[Chunk {i + 1}/{n_chunks}] cells {c_start:,} ~ {c_end:,}")

        # 只计算当前 chunk 的 cell sum
        conn.execute("DROP TABLE IF EXISTS _cell_sum_chunk")

        conn.execute(f"""
            CREATE TEMP TABLE _cell_sum_chunk AS
            SELECT
                atlas_cell_id,
                SUM({select_data}) AS total
            FROM X_HyS_data
            WHERE atlas_cell_id BETWEEN {c_start} AND {c_end}
            GROUP BY atlas_cell_id
        """)

        # 只更新当前 chunk 对应 obs
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

        # 每个 chunk 后立即清理
        conn.execute("DROP TABLE IF EXISTS _cell_sum_chunk")

    # 内存清理
    _cleanup_transform_after_step(
        conn,
        temp_tables=["_cell_sum_chunk"],
        checkpoint=False,
        collect=True,
    )

    print(f"normalize_total 完成，target_sum={target_sum}")
    print("耗时: {:.2f} 秒".format((datetime.now() - start).total_seconds()))

# 运行结果
#   obs 表
#     新增字段	             含义
#   scale_factor	     scale_factor，等到使用的时候在计算，data * scale_factor ，即可


''' log1p 法 1 ： 小内存 快速版  '''
def log1p_fast(
            atlas: 'Atlas',
            base: Optional[Number] = None,
            add_field: str = "data_log1p",
            select_data: str = "data_normalize" ) -> None:
    """快速执行 log1p 转换。

    该函数对 ``X_HyS_data`` 中指定表达字段执行 ``log(1 + x)``，并将结果写入新的表达字段。

    常用于 total-count 归一化之后，为 PCA、HVG、UMAP feature plot 和 marker 可视化准备 log-scale
    表达。

    Parameters
    ----------
    atlas
        Atlas 对象。通常要求已经连接数据库，并包含该函数所需的 ``obs``、``var``、``X_HyS_data`` 或
        embedding 结果表。

    base
        对数或指数转换使用的底数；为 ``None`` 时使用自然底。

    add_field
        写入 ``X_HyS_data`` 的新表达字段名。

    select_data
        从 ``X_HyS_data`` 中读取的表达字段。

    Notes
    -----
    该函数会把结果写回 Atlas 数据库；如果后续需要只使用过滤后的细胞或基因，通常还需要重建过滤索引。

    Examples
    --------
    调用该函数：::

        sap.pp.log1p_fast(...)
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
        WHERE table_name = 'X_HyS_data'
          AND column_name = '{select_data}'
    """).fetchone()[0]

    if col_exists == 0:
        raise ValueError(f"X_HyS_data 中不存在字段: {select_data}")

    # 1. 构造 log1p 表达式
    if base is None:
        log_expr = f"ln(1.0 + {select_data})"
    else:
        log_expr = f"log({float(base)}, 1.0 + {select_data})"

    if add_field is None:
        raise ValueError("必须指定 add_field")

    print(f"创建新字段 X_HyS_data.{add_field} ...")

    # 2. 确保输出字段存在
    conn.execute(f"""
        ALTER TABLE X_HyS_data
        ADD COLUMN IF NOT EXISTS {add_field} REAL
    """)

    # 3. 执行 log1p
    conn.execute(f"""
        UPDATE X_HyS_data
        SET {add_field} = {log_expr}
        WHERE {select_data} IS NOT NULL
    """)

    # 内存清理
    _cleanup_transform_after_step(
        conn,
        temp_tables=[],
        checkpoint=True,
        collect=True,
    )

    # 4. 结束
    print("log1p 完成")
    print("耗时: {:.2f} 秒".format((datetime.now() - start).total_seconds()))


''' log1p 法 2 ： 大数据 安全版 '''
def log1p(
                atlas: 'Atlas',
                base: Optional[Number] = None,
                add_field: str = "data_log1p",
                select_data: str = "data_normalize",
                chunk_ids: int = 100_000_000) -> None:
    """执行 log1p 转换。

    该函数以 chunk 方式对表达字段执行 log1p 转换，适合在大表更新时控制单次 SQL 写入规模。

    它不会删除原始表达字段，而是把结果写入 ``add_field``，方便比较不同转换结果。

    Parameters
    ----------
    atlas
        Atlas 对象。通常要求已经连接数据库，并包含该函数所需的 ``obs``、``var``、``X_HyS_data`` 或
        embedding 结果表。

    base
        对数或指数转换使用的底数；为 ``None`` 时使用自然底。

    add_field
        写入 ``X_HyS_data`` 的新表达字段名。

    select_data
        从 ``X_HyS_data`` 中读取的表达字段。

    chunk_ids
        分块处理大小，用于控制内存峰值和单次 SQL 更新规模。

    Notes
    -----
    该函数会把结果写回 Atlas 数据库；如果后续需要只使用过滤后的细胞或基因，通常还需要重建过滤索引。

    Examples
    --------
    调用该函数：::

        sap.pp.log1p(...)
    """

    logger.info("开始执行 log1p (chunked)...")
    print("==== log1p (chunked) ====")
    start = datetime.now()

    conn = atlas.connection

    conn.execute(f"PRAGMA threads = 10 ")

    # 0. 字段存在性检查（重要）
    col_exists = conn.execute(f"""
        SELECT COUNT(*)
        FROM information_schema.columns
        WHERE table_name = 'X_HyS_data'
          AND column_name = '{select_data}'
    """).fetchone()[0]

    if col_exists == 0:
        raise ValueError(f"X_HyS_data 中不存在字段: {select_data}")

    # 1. 构造 log 表达式
    if base is None:
        log_expr = f"ln(1.0 + {select_data})"
    else:
        log_expr = f"log({float(base)}, 1.0 + {select_data})"

    # 2. 确保输出字段存在
    conn.execute(f"""
        ALTER TABLE X_HyS_data
        DROP COLUMN IF EXISTS {add_field}
    """)

    conn.execute(f"""
        ALTER TABLE X_HyS_data
        ADD COLUMN  {add_field} REAL
    """)

    # 3. 获取 id 范围
    min_id, max_id = conn.execute("""
        SELECT MIN(id), MAX(id)
        FROM X_HyS_data
    """).fetchone()

    if min_id is None:
        print("X_HyS_data 为空，跳过")
        return

    n_chunks = math.ceil((max_id - min_id + 1) / chunk_ids)
    print(f"共 {n_chunks} 个 chunk")

    # 4. 分块 UPDATE
    for i in range(n_chunks):
        start_id = min_id + i * chunk_ids
        end_id = start_id + chunk_ids - 1

        print(f"  -> chunk {i+1}/{n_chunks}: id [{start_id}, {end_id}]")

        conn.execute(f"""
            UPDATE X_HyS_data
            SET {add_field} = {log_expr}
            WHERE id BETWEEN {start_id} AND {end_id}
              AND {select_data} IS NOT NULL
        """)

    # 内存清理
    _cleanup_transform_after_step(
        conn,
        temp_tables=[],
        checkpoint=True,
        collect=True,
    )

    # 5. 结束
    print("log1p (chunked) 完成")
    print("耗时: {:.2f} 秒".format((datetime.now() - start).total_seconds()))

# 运行结果
#   X_HyS_data 表
#     新增字段	             含义
#   data_log1p	     对表达值进行 log(1+x) 转换


''' expm1：log1p的逆运算 '''
def expm1(
        atlas: 'Atlas',
        base: Optional[Number] = None,
        add_field: str = "data_exp1",
        select_data: str = "data_log1p",
        chunk_ids: int = 50_000_000 ) -> None:
    """执行 expm1 逆转换。

    该函数对 log-scale 表达值执行 ``exp(x) - 1``，将表达恢复到近似 count/linear scale，并写入新字段。

    常用于需要从 log1p 表达回到线性表达空间进行比较或导出的场景。

    Parameters
    ----------
    atlas
        Atlas 对象。通常要求已经连接数据库，并包含该函数所需的 ``obs``、``var``、``X_HyS_data`` 或
        embedding 结果表。

    base
        对数或指数转换使用的底数；为 ``None`` 时使用自然底。

    add_field
        写入 ``X_HyS_data`` 的新表达字段名。

    select_data
        从 ``X_HyS_data`` 中读取的表达字段。

    chunk_ids
        分块处理大小，用于控制内存峰值和单次 SQL 更新规模。

    Notes
    -----
    该函数会把结果写回 Atlas 数据库；如果后续需要只使用过滤后的细胞或基因，通常还需要重建过滤索引。

    Examples
    --------
    调用该函数：::

        sap.pp.expm1(...)
    """

    logger.info("开始执行 expm1 (chunked)...")
    print("==== expm1 (chunked) ====")
    start = datetime.now()

    conn = atlas.connection

    try:
        conn.execute(f"PRAGMA threads={os.cpu_count()}")
    except:
        pass

    # 0. 字段存在性检查
    col_exists = conn.execute(f"""
        SELECT COUNT(*)
        FROM information_schema.columns
        WHERE table_name = 'X_HyS_data'
          AND column_name = '{select_data}'
    """).fetchone()[0]

    if col_exists == 0:
        raise ValueError(f"X_HyS_data 中不存在字段: {select_data}")

    # 1. 构造 exp 表达式
    if base is None:
        exp_expr = f"exp({select_data}) - 1.0"
    else:
        exp_expr = f"pow({float(base)}, {select_data}) - 1.0"

    # 2. 确保输出字段存在
    conn.execute(f"""
        ALTER TABLE X_HyS_data
        DROP COLUMN IF EXISTS {add_field}
    """)

    conn.execute(f"""
        ALTER TABLE X_HyS_data
        ADD COLUMN {add_field} REAL
    """)

    # 3. 获取 id 范围
    min_id, max_id = conn.execute("""
        SELECT MIN(id), MAX(id)
        FROM X_HyS_data
    """).fetchone()

    if min_id is None:
        print("X_HyS_data 为空，跳过")
        return

    n_chunks = math.ceil((max_id - min_id + 1) / chunk_ids)
    print(f"共 {n_chunks} 个 chunk")

    # 4. 分块 UPDATE
    for i in range(n_chunks):
        start_id = min_id + i * chunk_ids
        end_id = start_id + chunk_ids - 1

        print(f"  -> chunk {i+1}/{n_chunks}: id [{start_id}, {end_id}]")

        conn.execute(f"""
            UPDATE X_HyS_data
            SET {add_field} = {exp_expr}
            WHERE id BETWEEN {start_id} AND {end_id}
              AND {select_data} IS NOT NULL
        """)

    # 内存清理
    _cleanup_transform_after_step(
        conn,
        temp_tables=[],
        checkpoint=True,
        collect=True,
    )

    # 5. 结束
    print("expm1 (chunked) 完成")
    print("耗时: {:.2f} 秒".format((datetime.now() - start).total_seconds()))

# 运行结果
#   X_HyS_data 表
#     新增字段	             含义
#   data_exp1	     对表达值进行 log(1+x) 转换 的 还原


''' normalize_and_log1p： normalize 法 4 + log1p 法 2 ===='''
def normalize_and_log1p(
            atlas: Atlas,
            target_sum: Optional[float] = 10000,
            scale_key: str = "scale_factor",
            add_field: str = "data_log1p",
            select_data: str = "data",
            base: Optional[Number] = None,
            chunk_ids: int = 50_000_000 ) -> None:
    """依次执行归一化和 log1p 转换。

    该函数把 total-count 归一化和 log1p 转换合并在一个高层入口中，先根据 ``target_sum`` 缩放表达值，再写入
    log-scale 表达字段。

    它对应 Scanpy 工作流中常见的 ``normalize_total`` + ``log1p`` 组合，适合为
    PCA/HVG/可视化准备表达矩阵。

    Parameters
    ----------
    atlas
        Atlas 对象。通常要求已经连接数据库，并包含该函数所需的 ``obs``、``var``、``X_HyS_data`` 或
        embedding 结果表。

    target_sum
        归一化后每个细胞的目标总表达量。

    scale_key
        保存或读取归一化缩放因子的 ``obs`` 列名。

    add_field
        写入 ``X_HyS_data`` 的新表达字段名。

    select_data
        从 ``X_HyS_data`` 中读取的表达字段。

    base
        对数或指数转换使用的底数；为 ``None`` 时使用自然底。

    chunk_ids
        分块处理大小，用于控制内存峰值和单次 SQL 更新规模。

    Notes
    -----
    该函数会把结果写回 Atlas 数据库；如果后续需要只使用过滤后的细胞或基因，通常还需要重建过滤索引。

    Examples
    --------
    调用该函数：::

        sap.pp.normalize_and_log1p(...)
    """

    print("==== normalize_and_log1p (Scanpy-equivalent) ====")
    start = datetime.now()

    conn = atlas.connection
    try:
        conn.execute(f"PRAGMA threads={os.cpu_count()}")
    except:
        pass

    # 0. 字段存在性检查（防止 silent bug）
    col_exists = conn.execute(f"""
        SELECT COUNT(*)
        FROM information_schema.columns
        WHERE table_name = 'X_HyS_data'
          AND column_name = '{select_data}'
    """).fetchone()[0]

    if col_exists == 0:
        raise ValueError(f"X_HyS_data 中不存在字段: {select_data}")

    # 1. 调用上面的函数 normalize_total → 计算 scale_factor
    normalize_total_scale_factor(
        atlas=atlas,
        target_sum=target_sum,
        add_key=scale_key,
        select_data=select_data,
    )

    # 2. 构造 log 表达式
    if base is None:
        log_expr = f"ln(1.0 + x.{select_data} * o.{scale_key})"
    else:
        log_expr = f"log({float(base)}, 1.0 + x.{select_data} * o.{scale_key})"

    # 3. 准备输出字段
    conn.execute(f"""
        ALTER TABLE X_HyS_data
        DROP COLUMN IF EXISTS {add_field}
    """)

    conn.execute(f"""
        ALTER TABLE X_HyS_data
        ADD COLUMN {add_field} REAL
    """)

    # 4. 获取 X_HyS_data.id 范围
    min_id, max_id = conn.execute("""
        SELECT MIN(id), MAX(id)
        FROM X_HyS_data
    """).fetchone()

    if min_id is None:
        print("X_HyS_data 为空，跳过")
        return

    n_chunks = math.ceil((max_id - min_id + 1) / chunk_ids)
    print(f"log1p 分 {n_chunks} 个 id chunk")

    # 5. 分块 UPDATE
    for i in range(n_chunks):
        start_id = min_id + i * chunk_ids
        end_id   = start_id + chunk_ids - 1

        print(f"  -> chunk {i+1}/{n_chunks}: id [{start_id}, {end_id}]")

        conn.execute(f"""
            UPDATE X_HyS_data AS x
            SET {add_field} = {log_expr}
            FROM obs AS o
            WHERE x.atlas_cell_id = o.atlas_cell_id
              AND x.id BETWEEN {start_id} AND {end_id}
        """)

    # 内存清理
    _cleanup_transform_after_step(
        conn,
        temp_tables=["_cell_sum_chunk"],
        checkpoint=True,
        collect=True,
    )

    # 6. 结束
    print("normalize_and_log1p 完成")
    print("总耗时: {:.2f} 秒".format((datetime.now() - start).total_seconds()))

# 运行结果
#   X_HyS_data 表
#     新增字段	             含义
#   data_log1p	     对表达值进行 normalize_total +  log(1+x) 转换


''' HVG '''
def highly_variable_genes(
        atlas: Atlas,
        flavor: Literal["seurat", "cv", "var"] = "seurat",
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
    """识别高变基因。

    通过 ``flavor`` 参数选择不同的高变基因筛选方法。

    Parameters
    ----------
    atlas
        Atlas 对象。

    flavor
        高变基因筛选方法。

        - ``"seurat"``：使用 Seurat / Scanpy-like 方法。
        - ``"cv"``：按变异系数 ``std / mean`` 排序。
        - ``"var"``：按方差排序。

    n_top_genes
        需要选择的高变基因数量。

    add_key
        写入 ``var`` 表的布尔标记字段名。

    select_data
        从 ``X_HyS_data`` 中读取的表达字段。

    n_bins
        Seurat 方法中用于 mean 分箱的数量，仅 ``flavor="seurat"`` 时使用。

    min_mean
        Seurat cutoff 模式下的均值下限，仅 ``flavor="seurat"`` 且 ``n_top_genes=None`` 时使用。

    max_mean
        Seurat cutoff 模式下的均值上限，仅 ``flavor="seurat"`` 且 ``n_top_genes=None`` 时使用。

    min_disp
        Seurat cutoff 模式下的离散度下限，仅 ``flavor="seurat"`` 且 ``n_top_genes=None`` 时使用。

    max_disp
        Seurat cutoff 模式下的离散度上限，仅 ``flavor="seurat"`` 且 ``n_top_genes=None`` 时使用。

    use_filtered
        是否只使用过滤后的细胞和基因，仅 ``flavor="seurat"`` 时使用。

    obs_filter_col
        ``obs`` 中表示细胞过滤状态的列名，仅 ``flavor="seurat"`` 时使用。

    var_filter_col
        ``var`` 中表示基因过滤状态的列名，仅 ``flavor="seurat"`` 时使用。

    inplace
        是否将结果写回 Atlas 数据库，仅 ``flavor="seurat"`` 时支持返回结果。

    Returns
    -------
    result
        当 ``flavor="seurat"`` 且 ``inplace=False`` 时，返回 gene-level 统计结果；
        其他情况下返回 ``None``。
    """

    if flavor in ["cv", "var"]:
        return _highly_variable_genes_basic(
            atlas=atlas,
            flavor=flavor,
            n_top_genes=n_top_genes,
            add_key=add_key,
            select_data=select_data,
        )

    elif flavor == "seurat":
        return _highly_variable_genes_seurat(
            atlas=atlas,
            n_top_genes=n_top_genes,
            add_key=add_key,
            select_data=select_data,
            n_bins=n_bins,
            min_mean=min_mean,
            max_mean=max_mean,
            min_disp=min_disp,
            max_disp=max_disp,
            use_filtered=use_filtered,
            obs_filter_col=obs_filter_col,
            var_filter_col=var_filter_col,
            inplace=inplace,
        )

    else:
        raise ValueError(
            f"不支持的 flavor: {flavor}. "
            "可选值为: 'seurat', 'cv', 'var'"
        )

def _highly_variable_genes_basic(
                        atlas: Atlas,
                        flavor: Literal["var", "cv"] = "cv",
                        n_top_genes: int = 2000,
                        add_key: str = "highly_variable_genes",
                        select_data: str = "data_log1p"
                    ) -> None:
    """识别高变基因。

    该函数在数据库中按基因计算均值、方差、标准差、非零数量和变异性得分，并按 ``n_top_genes`` 选择高变基因。

    结果写入 ``var`` 表中的 ``add_key`` 以及相关统计字段，可供 ``build_read_index``、PCA 和 scale
    步骤使用。

    Parameters
    ----------
    atlas
        Atlas 对象。通常要求已经连接数据库，并包含该函数所需的 ``obs``、``var``、``X_HyS_data`` 或
        embedding 结果表。

    flavor
        高变基因筛选方法名称。

    n_top_genes
        需要选择的高变基因数量。

    add_key
        写回 ``obs`` 或 ``var`` 的结果列名。

    select_data
        从 ``X_HyS_data`` 中读取的表达字段。

    Notes
    -----
    该函数会把结果写回 Atlas 数据库；如果后续需要只使用过滤后的细胞或基因，通常还需要重建过滤索引。

    Examples
    --------
    调用该函数：::

        sap.pp.highly_variable_genes(...)
    """

    print("==== highly_variable_genes (CSR + DuckDB, all-cells stats) ====")
    start = datetime.now()

    conn = atlas.connection
    try:
        conn.execute(f"PRAGMA threads={os.cpu_count()}")
    except:
        pass

    # 检查字段存在
    col_exists = conn.execute(f"""
        SELECT COUNT(*)
        FROM information_schema.columns
        WHERE table_name = 'X_HyS_data'
          AND column_name = '{select_data}'
    """).fetchone()[0]

    if col_exists == 0:
        raise ValueError(f"X_HyS_data 中不存在字段: {select_data}")

    # 确保 var 表有可复用统计列
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

    # 取总细胞数 N（全细胞统计的关键）
    n_cells = conn.execute("""
        SELECT COUNT(*) FROM obs
    """).fetchone()[0]

    if n_cells == 0:
        raise ValueError("obs 为空，无法计算 highly_variable_genes")

    # 计算每个 gene 的全细胞 mean / var / std ; 不补 0，直接用 sum / sumsq / N_cells 推导
    print("Step 1: 计算 gene-level 统计量（全细胞含0）")

    conn.execute(f"""
        CREATE OR REPLACE TEMP TABLE _gene_stats AS
        WITH gene_sum AS (
            SELECT
                atlas_gene_id,
                COUNT(*) AS nnz,
                SUM({select_data}) AS sum_x,
                SUM(({select_data}) * ({select_data})) AS sum_x2
            FROM X_HyS_data
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

    # 计算排序指标
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

    # 把统计量和 score 写回 var，后续画图直接复用
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

    # 选 top genes
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

    # 在 var 表中写入布尔结果
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

    # 内存清理
    _cleanup_transform_after_step(
        conn,
        temp_tables=["_gene_stats", "_gene_score", "_hvg"],
        checkpoint=False,
        collect=True,
    )

    # 结束
    print("highly_variable_genes 完成（全细胞含0统计版）")
    print("耗时: {:.2f} 秒".format((datetime.now() - start).total_seconds()))

# 运行结果
#   var 表
#     新增字段	                     含义
#   highly_variable_genes	     对 n_top_genes 标记为true

''' HVG - seurat '''
def _highly_variable_genes_seurat(
        atlas: Atlas,
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
    """使用 Seurat 风格方法识别高变基因。

    该函数实现类似 Scanpy ``flavor="seurat"`` 的流程：计算基因均值和离散度，按均值分箱，在每个 bin 内标准化离散度，再选择
    top HVGs。

    函数支持只在过滤后的细胞/基因上计算，并把 ``means``、``dispersions``、``dispersions_norm``、rank
    和布尔选择结果写回 ``var``。

    Parameters
    ----------
    atlas
        Atlas 对象。通常要求已经连接数据库，并包含该函数所需的 ``obs``、``var``、``X_HyS_data`` 或
        embedding 结果表。

    n_top_genes
        需要选择的高变基因数量。

    add_key
        写回 ``obs`` 或 ``var`` 的结果列名。

    select_data
        从 ``X_HyS_data`` 中读取的表达字段。

    n_bins
        Seurat 风格 HVG 分箱数量。

    min_mean
        参与 HVG 选择的基因均值下限。

    max_mean
        参与 HVG 选择的基因均值上限。

    min_disp
        参与 HVG 选择的离散度下限。

    max_disp
        参与 HVG 选择的离散度上限。

    use_filtered
        是否只使用通过过滤的细胞或基因。

    obs_filter_col
        ``obs`` 中表示细胞过滤状态的列名。

    var_filter_col
        ``var`` 中表示基因过滤状态的列名。

    inplace
        是否将结果写回 Atlas 数据库。

    Returns
    -------
    result
        函数返回结果。具体类型取决于参数设置和内部执行路径。

    Notes
    -----
    该函数会把结果写回 Atlas 数据库；如果后续需要只使用过滤后的细胞或基因，通常还需要重建过滤索引。

    Examples
    --------
    调用该函数：::

        sap.pp._highly_variable_genes_seurat(...)
    """


    print("==== _highly_variable_genes_seurat (Scanpy-like) ====")
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
        """为 SQL 标识符添加安全引用。

        该内部函数属于表达矩阵转换模块，用于支撑同一模块中的公共 API。

        对 ``X_HyS_data`` 执行 normalize、log1p、sqrt、scale 和 HVG 筛选。

        它通常不会作为用户入口直接调用；直接调用时需要保证输入对象、数据库连接和相关临时表已经由上游步骤准备好。

        Parameters
        ----------
        name
            对象名称、列名或 SQL 标识符，具体含义由调用位置决定。

        Returns
-------
        quoted_name
            加双引号后的 SQL 标识符。

        Notes
        -----
        这是内部 helper；除非需要扩展 scAtlasPy 内部流程，一般不建议在用户代码中直接调用。
        """
        return '"' + name.replace('"', '""') + '"'

    # -------------------------------------------------
    # 1. 检查基础表
    # -------------------------------------------------
    for table_name in ["obs", "var", "X_HyS_data"]:
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
        WHERE table_name = 'X_HyS_data'
          AND column_name = ?
    """, [select_data]).fetchone()[0]

    if col_exists == 0:
        raise ValueError(f"X_HyS_data 中不存在字段: {select_data}")

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
            atlas_gene_id,
            COUNT(*) AS nnz,
            SUM(x_raw) AS sum_x,
            SUM(x_raw * x_raw) AS sum_x2
        FROM (
            SELECT
                x.atlas_gene_id,
                EXP(x.{_q(select_data)}) - 1.0 AS x_raw
            FROM X_HyS_data AS x
            JOIN _hvg_obs_keep AS o
              ON x.atlas_cell_id = o.atlas_cell_id
            JOIN _hvg_var_keep AS v
              ON x.atlas_gene_id = v.atlas_gene_id
            WHERE x.{_q(select_data)} IS NOT NULL
        ) AS t
        GROUP BY atlas_gene_id
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

        # ✅ 修改 A：先清空全量 var 的旧结果，避免 use_filtered=True 时旧 TRUE 残留
        conn.execute(f"""
            UPDATE var
            SET
                {_q(add_key)} = FALSE,
                highly_variable_rank = NULL,
                means = NULL,
                dispersions = NULL,
                dispersions_norm = NULL
        """)

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

    # 12. 统一清理 SQL 临时表 / pandas 注册表
    _cleanup_transform_after_step(
        conn,
        temp_tables=["_gene_sum", "_hvg_obs_keep", "_hvg_var_keep"],
        unregister_tables=["_hvg_seurat_py"],
        checkpoint=False,
        collect=False,
    )

    # 如果 inplace=True，不需要返回 gene_df，就删除 Python 大对象
    if inplace:
        try:
            del gene_df
        except Exception:
            pass
        try:
            del work
        except Exception:
            pass
        try:
            del rank_df
        except Exception:
            pass
        try:
            del write_df
        except Exception:
            pass
        try:
            del disp_stats
        except Exception:
            pass

    gc.collect()

    print("_highly_variable_genes_seurat 完成")
    print("耗时: {:.2f} 秒".format((datetime.now() - start).total_seconds()))

    if not inplace:
        return gene_df


''' scale  法 1 ： 大数据，安全版 '''
def scale(
        atlas: Atlas,
        select_data: str = "data_log1p",
        add_field: str = "data_scale",
        add_field_to_var: str = "zero_scale_transform",
        max_value: float = 10.0,
        use_hvg: bool = True,
        hvg_key: str = "highly_variable_genes",
        chunk_ids: int = 20_000_000,
        ):
    """对表达矩阵进行基因级标准化缩放。

    该函数按基因计算均值和标准差，将每个非零表达值转换为 z-score，并把零值在标准化后的对应值保存到 ``var``。

    功能上类似 ``scanpy.pp.scale``；当 ``use_hvg=True`` 时只缩放高变基因，``max_value``
    可用于截断过大的 z-score。

    Parameters
    ----------
    atlas
        Atlas 对象。通常要求已经连接数据库，并包含该函数所需的 ``obs``、``var``、``X_HyS_data`` 或
        embedding 结果表。

    select_data
        从 ``X_HyS_data`` 中读取的表达字段。

    add_field
        写入 ``X_HyS_data`` 的新表达字段名。

    add_field_to_var
        写入 ``var`` 表的辅助统计列名。

    max_value
        标准化后允许的最大绝对值；用于截断极端 z-score。

    use_hvg
        是否只处理高变基因。

    hvg_key
        ``var`` 中表示高变基因的布尔列名。

    chunk_ids
        按 ``X_HyS_data.id`` 分块更新时每个 chunk 的行数。

    Notes
    -----
    该函数会把结果写回 Atlas 数据库；如果后续需要只使用过滤后的细胞或基因，通常还需要重建过滤索引。

    Examples
    --------
    调用该函数：::

        sap.pp.scale(...)
    """

    print("\n==== scale_ultra_safe_update_by_id_chunk_direct_zero_aware ====")
    start_all = datetime.now()
    conn = atlas.connection

    # 0. 并行
    try:
        n_threads = 4
        conn.execute(f"PRAGMA threads={n_threads}")
        print(f"-> DuckDB threads = {n_threads}")
    except Exception:
        pass

    # 1. 输入字段检查
    if conn.execute(f"""
        SELECT COUNT(*)
        FROM information_schema.columns
        WHERE table_name = 'X_HyS_data'
          AND column_name = '{select_data}'
    """).fetchone()[0] == 0:
        raise ValueError(f"X_HyS_data 中不存在字段: {select_data}")

    if conn.execute("""
        SELECT COUNT(*)
        FROM information_schema.columns
        WHERE table_name = 'X_HyS_data'
          AND column_name = 'id'
    """).fetchone()[0] == 0:
        raise ValueError("X_HyS_data 中不存在 id 字段，无法按 id 分块回写")

    if conn.execute("""
        SELECT COUNT(*)
        FROM information_schema.columns
        WHERE table_name = 'X_HyS_data'
          AND column_name = 'atlas_gene_id'
    """).fetchone()[0] == 0:
        raise ValueError("X_HyS_data 中不存在 atlas_gene_id 字段")

    # 2. 输出字段准备
    conn.execute(f""" ALTER TABLE X_HyS_data DROP COLUMN IF EXISTS {add_field} """)
    conn.execute(f""" ALTER TABLE X_HyS_data ADD COLUMN IF NOT EXISTS {add_field} REAL """)

    conn.execute(f""" ALTER TABLE var DROP COLUMN IF EXISTS {add_field_to_var} """)
    conn.execute(f""" ALTER TABLE var ADD COLUMN IF NOT EXISTS {add_field_to_var} REAL """)

    # 3. 准备目标 gene 集合
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

    # 获取全细胞数量
    n_cells = conn.execute("""
        SELECT COUNT(*) FROM obs
    """).fetchone()[0]

    if n_cells == 0:
        conn.execute("DROP TABLE IF EXISTS _target_genes")
        raise ValueError("obs 为空，无法计算 scale")

    print(f"-> Total cells for zero-aware scaling: {n_cells:,}")

    # 4. 一次性计算所有目标 gene 的 mean/std
    print("-> 计算 _gene_stat（一次，全局，含 0 统计）...")
    t0 = datetime.now()

    conn.execute("DROP TABLE IF EXISTS _gene_stat")

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

        FROM X_HyS_data x
        JOIN _target_genes t
          ON x.atlas_gene_id = t.atlas_gene_id
        WHERE x.{select_data} IS NOT NULL
        GROUP BY x.atlas_gene_id
    """)

    print("   _gene_stat 完成，耗时: {:.2f} 秒".format(
        (datetime.now() - t0).total_seconds()
    ))

    # 5. 更新 var：记录 0 值 的缩放因子 -> z-score
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

    # 6. 获取 id 范围
    print("-> 获取 id 范围 ...")
    min_id, max_id, total_rows = conn.execute("""
        SELECT MIN(id), MAX(id), COUNT(*)
        FROM X_HyS_data
    """).fetchone()

    print(f"-> X_HyS_data rows: {total_rows}")
    print(f"-> id range: {min_id} ~ {max_id}")
    print(f"-> chunk_ids: {chunk_ids}")

    n_chunks = math.ceil((max_id - min_id + 1) / chunk_ids)
    print(f"-> Total id chunks: {n_chunks}")

    # 7. 按 id 分块直接 UPDATE
    print("-> 开始按 id 分块直接回写 ...")

    done_chunks = 0
    update_start_all = datetime.now()

    for chunk_idx in range(n_chunks):
        chunk_start = min_id + chunk_idx * chunk_ids
        chunk_end = min(chunk_start + chunk_ids - 1, max_id)

        t0 = datetime.now()
        print(
            f"\n[Chunk {chunk_idx + 1}/{n_chunks}] "
            f"id: {chunk_start} ~ {chunk_end}"
        )

        conn.execute("BEGIN")
        try:

            # 对显式存储的非零值写入 z-score
            conn.execute(f"""
                UPDATE X_HyS_data x
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

    # 8. 清理临时表
    print("\n-> 清理临时表 ...")

    # 清理内存
    _cleanup_transform_after_step(
        conn,
        temp_tables=["_target_genes", "_gene_stat"],
        checkpoint=True,
        collect=True,
    )

    print("\n==== scale_ultra_safe_update_by_id_chunk_direct_zero_aware 完成 ====")
    print("总耗时: {:.2f} 秒".format(
        (datetime.now() - start_all).total_seconds()
    ))


''' scale_fast 法 2 ： 小内存 快速版   '''
def scale_fast(
        atlas: Atlas,
        select_data: str = "data_log1p",
        add_field: str = "data_scale",
        add_field_to_var: str = "zero_scale_transform",
        max_value: float = 10.0,
        use_hvg: bool = True,
        hvg_key: str = "highly_variable_genes"):
    """快速对表达矩阵进行基因级标准化缩放。

    该函数使用 DuckDB 聚合和批量 UPDATE 计算标准化表达，比逐 chunk 版本更直接。

    它适合已经准备好 HVG 标记和表达字段的大数据预处理流程。

    Parameters
    ----------
    atlas
        Atlas 对象。通常要求已经连接数据库，并包含该函数所需的 ``obs``、``var``、``X_HyS_data`` 或
        embedding 结果表。

    select_data
        从 ``X_HyS_data`` 中读取的表达字段。

    add_field
        写入 ``X_HyS_data`` 的新表达字段名。

    add_field_to_var
        写入 ``var`` 表的辅助统计列名。

    max_value
        标准化后允许的最大绝对值；用于截断极端 z-score。

    use_hvg
        是否只处理高变基因。

    hvg_key
        ``var`` 中表示高变基因的布尔列名。

    Notes
    -----
    该函数会把结果写回 Atlas 数据库；如果后续需要只使用过滤后的细胞或基因，通常还需要重建过滤索引。

    Examples
    --------
    调用该函数：::

        sap.pp.scale_fast(...)
    """

    print("\n==== scale_ultra (zero-aware industrial OLAP optimized) ====")
    start_all = datetime.now()
    conn = atlas.connection

    # 0. 并行
    try:
        n_threads = os.cpu_count()
        conn.execute(f"PRAGMA threads={n_threads}")
        print(f"-> DuckDB threads = {n_threads}")
    except Exception:
        pass

    # 1. 输入字段检查
    col_exists = conn.execute(f"""
        SELECT COUNT(*)
        FROM information_schema.columns
        WHERE table_name='X_HyS_data'
          AND column_name='{select_data}'
    """).fetchone()[0]

    if col_exists == 0:
        raise ValueError(f"X_HyS_data 中不存在字段: {select_data}")

    # 2. 输出字段准备
    conn.execute(f"""
        ALTER TABLE X_HyS_data
        ADD COLUMN IF NOT EXISTS {add_field} REAL
    """)
    conn.execute(f"""
        ALTER TABLE var
        ADD COLUMN IF NOT EXISTS {add_field_to_var} REAL
    """)

    # 3. gene 列表
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

    # 获取总细胞数，用于含 0 统计
    n_cells = conn.execute("""
        SELECT COUNT(*) FROM obs
    """).fetchone()[0]

    if n_cells == 0:
        raise ValueError("obs 为空，无法计算 scale")

    print(f"-> Total cells for zero-aware scaling: {n_cells:,}")

    # 4. 计算 gene-wise 统计 + 线性化系数 a,b；AVG/STDDEV_POP 改成含 0 公式
    print("-> 计算 gene-wise mean/std -> a,b（含 0 统计）")

    conn.execute(f"""
        CREATE OR REPLACE TEMP TABLE _gene_stat AS
        WITH gene_sum AS (
            SELECT
                atlas_gene_id,
                SUM({select_data}) AS sum_x,
                SUM({select_data} * {select_data}) AS sum_x2
            FROM X_HyS_data
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

    # 5. 直接 UPDATE X_HyS_data
    print("-> 直接应用线性化 z-score + clip")
    conn.execute(f"""
        UPDATE X_HyS_data x
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

    # 6. 更新 var 表 zero_scale_transform
    print("-> 更新 var zero_scale_transform")
    conn.execute(f"""
        UPDATE var v
        SET {add_field_to_var} = g.b
        FROM _gene_stat g
        WHERE v.atlas_gene_id = g.atlas_gene_id
    """)

    # 内存清理
    _cleanup_transform_after_step(
        conn,
        temp_tables=["_gene_stat"],
        checkpoint=True,
        collect=True,
    )

    print("\n==== scale_ultra zero-aware 完成 ====")
    print("耗时: {:.2f} 秒".format((datetime.now() - start_all).total_seconds()))



# 运行结果
#   X_HyS_data 表
#     新增字段	                     含义
#   data_scale	             对 data 进行 z-score 标准化， z = (x - mean_g) / std_g
#   var 表  新增字段
#   zero_scale_transform    将每个基因的 ( 0 - g.mean) / g.std 存入var表的该字段，以便将来调用


''' sqrt 法 1 ： 小内存，快速版 '''
def sqrt_fast(
    atlas: "Atlas",
    add_field: str = "data_sqrt",
    select_data: str = "data") -> None:
    """快速执行平方根转换。

    该函数对指定表达字段执行平方根转换，并把结果写入 ``X_HyS_data`` 的新字段。

    平方根转换可作为 count 数据的轻量方差稳定化方法，用于探索性分析或特定模型输入。

    Parameters
    ----------
    atlas
        Atlas 对象。通常要求已经连接数据库，并包含该函数所需的 ``obs``、``var``、``X_HyS_data`` 或
        embedding 结果表。

    add_field
        写入 ``X_HyS_data`` 的新表达字段名。

    select_data
        从 ``X_HyS_data`` 中读取的表达字段。

    Notes
    -----
    该函数会把结果写回 Atlas 数据库；如果后续需要只使用过滤后的细胞或基因，通常还需要重建过滤索引。

    Examples
    --------
    调用该函数：::

        sap.pp.sqrt_fast(...)
    """

    logger.info("开始执行 sqrt(x) 转换...")
    print("==== sqrt ====")
    start = datetime.now()

    try:
        atlas.connection.execute(f"PRAGMA threads={os.cpu_count()}")
    except Exception:
        pass

    conn = atlas.connection

    # 0. 字段存在性检查（防止 silent bug）
    col_exists = conn.execute(
        f"""
        SELECT COUNT(*)
        FROM information_schema.columns
        WHERE table_name = 'X_HyS_data'
          AND column_name = '{select_data}'
        """
    ).fetchone()[0]

    if col_exists == 0:
        raise ValueError(f"X_HyS_data 中不存在字段: {select_data}")

    if add_field is None:
        raise ValueError("必须指定 add_field")

    print(f"创建 / 使用字段 X_HyS_data.{add_field} ...")

    # 1. 构造 sqrt 表达式（不处理 0）
    sqrt_expr = f"sqrt({select_data})"

    # 2. 确保输出字段存在

    # 先删除旧列，避免旧值残留
    conn.execute(f"""
        ALTER TABLE X_HyS_data
        DROP COLUMN IF EXISTS {add_field}
    """)

    conn.execute(f"""
        ALTER TABLE X_HyS_data
        ADD COLUMN {add_field} REAL
    """)

    # 3. 执行 sqrt
    conn.execute(
        f"""
        UPDATE X_HyS_data
        SET {add_field} = {sqrt_expr}
        WHERE {select_data} IS NOT NULL
        """
    )

    # 内存清理
    _cleanup_transform_after_step(
        conn,
        temp_tables=[],
        checkpoint=True,
        collect=True,
    )

    # 4. 结束
    elapsed = (datetime.now() - start).total_seconds()
    print("sqrt 转换完成")
    print(f"耗时: {elapsed:.2f} 秒")


''' sqrt  法 3 ： 大数据，安全版 '''
def sqrt(
    atlas: "Atlas",
    add_field: str = "data_sqrt",
    select_data: str = "data",
    chunk_ids: int = 100_000_000) -> None:
    """执行平方根转换。

    该函数以 chunk 方式对表达字段执行平方根转换，适合在较大 ``X_HyS_data`` 表上降低单次更新压力。

    转换结果写入新字段，不会覆盖原始表达值。

    Parameters
    ----------
    atlas
        Atlas 对象。通常要求已经连接数据库，并包含该函数所需的 ``obs``、``var``、``X_HyS_data`` 或
        embedding 结果表。

    add_field
        写入 ``X_HyS_data`` 的新表达字段名。

    select_data
        从 ``X_HyS_data`` 中读取的表达字段。

    chunk_ids
        分块处理大小，用于控制内存峰值和单次 SQL 更新规模。

    Notes
    -----
    该函数会把结果写回 Atlas 数据库；如果后续需要只使用过滤后的细胞或基因，通常还需要重建过滤索引。

    Examples
    --------
    调用该函数：::

        sap.pp.sqrt(...)
    """

    logger.info("开始执行 sqrt (chunked)...")
    print("==== sqrt (chunked) ====")
    start = datetime.now()

    conn = atlas.connection

    try:
        conn.execute(f"PRAGMA threads={os.cpu_count()}")
    except Exception:
        pass

    # 0. 字段存在性检查
    col_exists = conn.execute(
        f"""
        SELECT COUNT(*)
        FROM information_schema.columns
        WHERE table_name = 'X_HyS_data'
          AND column_name = '{select_data}'
        """
    ).fetchone()[0]

    if col_exists == 0:
        raise ValueError(f"X_HyS_data 中不存在字段: {select_data}")

    if add_field is None:
        raise ValueError("必须指定 add_field")

    # 1. 构造 sqrt 表达式（不处理 0）
    sqrt_expr = f"sqrt({select_data})"

    # 2. 确保输出字段存在
    # 先删旧列
    conn.execute(f"""
        ALTER TABLE X_HyS_data
        DROP COLUMN IF EXISTS {add_field}
    """)

    # 重新添加 REAL 字段
    conn.execute(f"""
        ALTER TABLE X_HyS_data
        ADD COLUMN {add_field} REAL
    """)

    # 3. 获取 id 范围
    min_id, max_id = conn.execute(
        """
        SELECT MIN(id), MAX(id)
        FROM X_HyS_data
        """
    ).fetchone()

    if min_id is None:
        print("X_HyS_data 为空，跳过")
        return

    n_chunks = math.ceil((max_id - min_id + 1) / chunk_ids)
    print(f"共 {n_chunks} 个 chunk")

    # 4. 分块 UPDATE
    for i in range(n_chunks):
        start_id = min_id + i * chunk_ids
        end_id = start_id + chunk_ids - 1

        print(f"  -> chunk {i + 1}/{n_chunks}: id [{start_id}, {end_id}]")

        conn.execute(
            f"""
            UPDATE X_HyS_data
            SET {add_field} = {sqrt_expr}
            WHERE id BETWEEN {start_id} AND {end_id}
              AND {select_data} IS NOT NULL
            """
        )

    # 内存清理
    _cleanup_transform_after_step(
        conn,
        temp_tables=[],
        checkpoint=True,
        collect=True,
    )

    # 5. 结束
    elapsed = (datetime.now() - start).total_seconds()
    print("sqrt (chunked) 完成")
    print(f"耗时: {elapsed:.2f} 秒")

# 运行结果
#   X_HyS_data 表
#     新增字段	                     含义
#   data_sqrt	             对 data 进行 sqrt
