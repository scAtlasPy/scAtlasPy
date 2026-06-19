from _duckdb import DuckDBPyConnection
from ..data import Atlas
from tqdm import tqdm
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


''' normalize '''
def normalize_total(
        atlas: Atlas,
        target_sum: float = 10000,
        chunk_cells: int = 500_000,
        add_data: str = "data_normalize",
        use_data: str = "data"
) -> None:

    """按细胞总表达量进行归一化。

    该函数把每个细胞的表达总量缩放到 ``target_sum``，并把归一化结果写入新的表达矩阵表。它类似 Scanpy 的 ``sc.pp.normalize_total``。

    Parameters
    ----------
    atlas
        Atlas 对象。通常需要已经连接到 DuckDB 数据库，并包含该函数读取或写入所需的 ``obs``、``var``、表达矩阵或结果表。
    target_sum
        归一化后每个细胞的目标总表达量。常用值为 ``10000``。
    chunk_cells
        按细胞分块处理时每个 chunk 的细胞数量。
    add_data
        写入数据库的新表达矩阵表名或结果列名。
    use_data
        读取的表达矩阵或结果表名称。常用值包括 ``"data"``、``"data_normalize"``、``"data_log1p"`` 和
        ``"data_scale"``。

    Returns
    -------
    None
        结果直接写入 Atlas 数据库或当前图形窗口。

    Examples
    --------
    归一化原始 counts 到每细胞 1 万::

        sap.pp.normalize_total(atlas, target_sum=10000)

    对自定义矩阵归一化，并保存为新表::

        sap.pp.normalize_total(
            atlas,
            use_data="data",
            add_data="data_cpm",
            target_sum=1000000,
        )"""

    start_time = datetime.now()

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
          AND column_name = '{use_data}'
    """).fetchone()[0]

    if col_exists == 0:
        raise ValueError(f"X_HyS_data 中不存在字段: {use_data}")

    # 2. 删除源表旧 normalize 字段
    conn.execute(f"""
        ALTER TABLE X_HyS_data
        DROP COLUMN IF EXISTS {add_data}
    """)

    # 3. 创建目标表
    conn.execute("DROP TABLE IF EXISTS X_HyS_data_norm")

    conn.execute("""
        CREATE TABLE X_HyS_data_norm AS
        SELECT * FROM X_HyS_data WHERE 1=0
    """)

    conn.execute(f"""
        ALTER TABLE X_HyS_data_norm
        ADD COLUMN {add_data} REAL
    """)

    # 4. 获取 cell_id 范围
    min_cell, max_cell = conn.execute("""
        SELECT MIN(atlas_cell_id), MAX(atlas_cell_id)
        FROM X_HyS_data
    """).fetchone()

    if min_cell is None:
        logger.info("X_HyS_data 为空，跳过")
        return

    n_chunks = math.ceil((max_cell - min_cell + 1) / chunk_cells)

    # 5. cell 分块：小 _cell_sum_chunk + 写入目标表
    pbar = tqdm(
        range(n_chunks),
        total=n_chunks,
        desc="normalize_total",
        unit="chunk",
    )

    for i in pbar:

        c_start = min_cell + i * chunk_cells
        c_end = min(c_start + chunk_cells - 1, max_cell)

        # 只计算当前 cell chunk 的 sum
        conn.execute("DROP TABLE IF EXISTS _cell_sum_chunk")

        conn.execute(f"""
            CREATE TEMP TABLE _cell_sum_chunk AS
            SELECT
                atlas_cell_id,
                SUM({use_data}) AS total
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
                x.{use_data} * {float(target_sum)} / s.total AS {add_data}
            FROM X_HyS_data x
            JOIN _cell_sum_chunk s
              ON x.atlas_cell_id = s.atlas_cell_id
            WHERE x.atlas_cell_id BETWEEN {c_start} AND {c_end}
            ORDER BY x.id
        """)

        # 每个 chunk 后立即清理
        conn.execute("DROP TABLE IF EXISTS _cell_sum_chunk")

    # 6. 替换原表
    conn.execute("DROP TABLE X_HyS_data")
    conn.execute("ALTER TABLE X_HyS_data_norm RENAME TO X_HyS_data")

    # 内存清理
    _cleanup_transform_after_step(
        conn,
        temp_tables=["_cell_sum_chunk"],
        checkpoint=True,
        collect=True,
    )

    logger.info(f"normalize_total Done, 耗时: {(datetime.now() - start_time).total_seconds():.2f} 秒")

# atlas.connection.execute("CHECKPOINT")
# atlas.connection.close()
# atlas.connection = atlas.connect("r+")
# gc.collect()

# 运行结果
#   X_HyS_data 表
#     新增字段	             含义
#  data_normalize	     归一化后的data值


''' normalize 法 2：在 obs表上记录 scale_factor ， 等到使用的时候再计算 '''
def normalize_total_scale_factor(
        atlas: Atlas,
        target_sum: float = 10000,
        add_obs_col: str = "scale_factor",
        use_data: str = "data",
        chunk_cells: int = 500_000,
) -> None:
    """计算每个细胞的归一化 scale factor。

    该函数根据每个细胞总表达量和 ``target_sum`` 计算 scale factor，并写入 ``obs``。后续可用 ``normalize_and_log1p`` 在一次分块处理中完成缩放和对数变换。

    Parameters
    ----------
    atlas
        Atlas 对象。通常需要已经连接到 DuckDB 数据库，并包含该函数读取或写入所需的 ``obs``、``var``、表达矩阵或结果表。
    target_sum
        归一化后每个细胞的目标总表达量。常用值为 ``10000``。
    add_obs_col
        写入 ``obs`` 的结果列名。
    use_data
        读取的表达矩阵或结果表名称。常用值包括 ``"data"``、``"data_normalize"``、``"data_log1p"`` 和
        ``"data_scale"``。
    chunk_cells
        按细胞分块处理时每个 chunk 的细胞数量。

    Returns
    -------
    None
        结果直接写入 Atlas 数据库或当前图形窗口。

    Examples
    --------
    计算默认 scale factor::

        sap.pp.normalize_total_scale_factor(atlas, target_sum=10000)

    保存到自定义 obs 列::

        sap.pp.normalize_total_scale_factor(
            atlas,
            add_obs_col="scale_factor_1e4",
            target_sum=10000,
        )"""

    start_time = datetime.now()

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
          AND column_name = '{use_data}'
    """).fetchone()[0]

    if col_exists == 0:
        raise ValueError(f"X_HyS_data 中不存在字段: {use_data}")

    # 1. obs 添加 scale_factor 字段
    conn.execute(f"""
        ALTER TABLE obs
        ADD COLUMN IF NOT EXISTS {add_obs_col} REAL
    """)

    # 先初始化，避免空 cell 或未命中 cell 保留旧值
    conn.execute(f"""
        UPDATE obs
        SET {add_obs_col} = 0
    """)

    # 2. 获取 cell_id 范围
    min_cell, max_cell = conn.execute("""
        SELECT MIN(atlas_cell_id), MAX(atlas_cell_id)
        FROM obs
    """).fetchone()

    if min_cell is None or max_cell is None:
        logger.info("obs 为空，跳过")
        return

    n_chunks = math.ceil((max_cell - min_cell + 1) / chunk_cells)

    # 3. 分块计算 total + 写回 obs
    pbar = tqdm(
        range(n_chunks),
        total=n_chunks,
        desc="normalize_total_scale_factor",
        unit="chunk",
    )

    for i in pbar:

        c_start = min_cell + i * chunk_cells
        c_end = min(c_start + chunk_cells - 1, max_cell)

        # 只计算当前 chunk 的 cell sum
        conn.execute("DROP TABLE IF EXISTS _cell_sum_chunk")

        conn.execute(f"""
            CREATE TEMP TABLE _cell_sum_chunk AS
            SELECT
                atlas_cell_id,
                SUM({use_data}) AS total
            FROM X_HyS_data
            WHERE atlas_cell_id BETWEEN {c_start} AND {c_end}
            GROUP BY atlas_cell_id
        """)

        # 只更新当前 chunk 对应 obs
        conn.execute(f"""
            UPDATE obs
            SET {add_obs_col} =
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

    logger.info(f"normalize_total_scale_factor Done, 耗时: {(datetime.now() - start_time).total_seconds():.2f} 秒")

# 运行结果
#   obs 表
#     新增字段	             含义
#   scale_factor	     scale_factor，等到使用的时候在计算，data * scale_factor ，即可


''' log1p '''
def log1p(
                atlas: 'Atlas',
                base: Optional[Number] = None,
                add_data: str = "data_log1p",
                use_data: str = "data_normalize",
                chunk_ids: int = 100_000_000) -> None:
    """对表达矩阵执行 log1p 变换。

    该函数读取指定表达矩阵表，对表达值执行 ``log(1 + x)`` 或指定底数的 log1p 变换，并写入新的表达矩阵表。它类似 Scanpy 的 ``sc.pp.log1p``。

    Parameters
    ----------
    atlas
        Atlas 对象。通常需要已经连接到 DuckDB 数据库，并包含该函数读取或写入所需的 ``obs``、``var``、表达矩阵或结果表。
    base
        对数变换的底数。为 ``None`` 时使用自然对数。
    add_data
        写入数据库的新表达矩阵表名或结果列名。
    use_data
        读取的表达矩阵或结果表名称。常用值包括 ``"data"``、``"data_normalize"``、``"data_log1p"`` 和
        ``"data_scale"``。
    chunk_ids
        按表达记录 ID 分块处理时每个 chunk 的记录数量。

    Returns
    -------
    None
        结果直接写入 Atlas 数据库或当前图形窗口。

    Examples
    --------
    对归一化矩阵进行自然对数变换::

        sap.pp.normalize_total(atlas)
        sap.pp.log1p(atlas)

    使用 2 为底的对数并写入自定义表::

        sap.pp.log1p(
            atlas,
            use_data="data_normalize",
            add_data="data_log1p_base2",
            base=2,
        )"""

    start_time = datetime.now()

    conn = atlas.connection

    conn.execute(f"PRAGMA threads = 10 ")

    # 0. 字段存在性检查（重要）
    col_exists = conn.execute(f"""
        SELECT COUNT(*)
        FROM information_schema.columns
        WHERE table_name = 'X_HyS_data'
          AND column_name = '{use_data}'
    """).fetchone()[0]

    if col_exists == 0:
        raise ValueError(f"X_HyS_data 中不存在字段: {use_data}")

    # 1. 构造 log 表达式
    if base is None:
        log_expr = f"ln(1.0 + {use_data})"
    else:
        log_expr = f"log({float(base)}, 1.0 + {use_data})"

    # 2. 确保输出字段存在
    conn.execute(f"""
        ALTER TABLE X_HyS_data
        DROP COLUMN IF EXISTS {add_data}
    """)

    conn.execute(f"""
        ALTER TABLE X_HyS_data
        ADD COLUMN  {add_data} REAL
    """)

    # 3. 获取 id 范围
    min_id, max_id = conn.execute("""
        SELECT MIN(id), MAX(id)
        FROM X_HyS_data
    """).fetchone()

    if min_id is None:
        logger.info("X_HyS_data 为空，跳过")
        return

    n_chunks = math.ceil((max_id - min_id + 1) / chunk_ids)

    # 4. 分块 UPDATE
    pbar = tqdm(
        range(n_chunks),
        total=n_chunks,
        desc="log1p",
        unit="chunk",
    )

    for i in pbar:
        start_id = min_id + i * chunk_ids
        end_id = start_id + chunk_ids - 1

        conn.execute(f"""
            UPDATE X_HyS_data
            SET {add_data} = {log_expr}
            WHERE id BETWEEN {start_id} AND {end_id}
              AND {use_data} IS NOT NULL
        """)

    # 内存清理
    _cleanup_transform_after_step(
        conn,
        temp_tables=[],
        checkpoint=True,
        collect=True,
    )

    logger.info(f"log1p Done, 耗时: {(datetime.now() - start_time).total_seconds():.2f} 秒")

# 运行结果
#   X_HyS_data 表
#     新增字段	             含义
#   data_log1p	     对表达值进行 log(1+x) 转换


''' expm1：log1p的逆运算 '''
def expm1(
        atlas: 'Atlas',
        base: Optional[Number] = None,
        add_data: str = "data_exp1",
        use_data: str = "data_log1p",
        chunk_ids: int = 50_000_000 ) -> None:
    """对 log1p 表达矩阵执行反变换。

    该函数读取 log1p 变换后的表达值，执行 ``exp(x) - 1`` 或指定底数的反变换，并写入新的表达矩阵表。

    Parameters
    ----------
    atlas
        Atlas 对象。通常需要已经连接到 DuckDB 数据库，并包含该函数读取或写入所需的 ``obs``、``var``、表达矩阵或结果表。
    base
        对数变换的底数。为 ``None`` 时使用自然对数。
    add_data
        写入数据库的新表达矩阵表名或结果列名。
    use_data
        读取的表达矩阵或结果表名称。常用值包括 ``"data"``、``"data_normalize"``、``"data_log1p"`` 和
        ``"data_scale"``。
    chunk_ids
        按表达记录 ID 分块处理时每个 chunk 的记录数量。

    Returns
    -------
    None
        结果直接写入 Atlas 数据库或当前图形窗口。

    Examples
    --------
    将默认 log1p 矩阵还原到线性空间::

        sap.pp.expm1(atlas, use_data="data_log1p", add_data="data_exp1")

    还原以 2 为底的 log1p 矩阵::

        sap.pp.expm1(
            atlas,
            use_data="data_log1p_base2",
            add_data="data_exp1_base2",
            base=2,
        )"""

    start_time = datetime.now()

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
          AND column_name = '{use_data}'
    """).fetchone()[0]

    if col_exists == 0:
        raise ValueError(f"X_HyS_data 中不存在字段: {use_data}")

    # 1. 构造 exp 表达式
    if base is None:
        exp_expr = f"exp({use_data}) - 1.0"
    else:
        exp_expr = f"pow({float(base)}, {use_data}) - 1.0"

    # 2. 确保输出字段存在
    conn.execute(f"""
        ALTER TABLE X_HyS_data
        DROP COLUMN IF EXISTS {add_data}
    """)

    conn.execute(f"""
        ALTER TABLE X_HyS_data
        ADD COLUMN {add_data} REAL
    """)

    # 3. 获取 id 范围
    min_id, max_id = conn.execute("""
        SELECT MIN(id), MAX(id)
        FROM X_HyS_data
    """).fetchone()

    if min_id is None:
        logger.info("X_HyS_data 为空，跳过")
        return

    n_chunks = math.ceil((max_id - min_id + 1) / chunk_ids)

    # 4. 分块 UPDATE
    pbar = tqdm(
        range(n_chunks),
        total=n_chunks,
        desc="expm1",
        unit="chunk",
    )

    for i in pbar:

        start_id = min_id + i * chunk_ids
        end_id = start_id + chunk_ids - 1

        conn.execute(f"""
            UPDATE X_HyS_data
            SET {add_data} = {exp_expr}
            WHERE id BETWEEN {start_id} AND {end_id}
              AND {use_data} IS NOT NULL
        """)

    # 内存清理
    _cleanup_transform_after_step(
        conn,
        temp_tables=[],
        checkpoint=True,
        collect=True,
    )

    logger.info(f"expm1 Done, 耗时: {(datetime.now() - start_time).total_seconds():.2f} 秒")

# 运行结果
#   X_HyS_data 表
#     新增字段	             含义
#   data_exp1	     对表达值进行 log(1+x) 转换 的 还原


''' normalize_and_log1p ===='''
def normalize_and_log1p(
            atlas: Atlas,
            target_sum: Optional[float] = 10000,
            use_obs_col: str = "scale_factor",
            add_data: str = "data_log1p",
            use_data: str = "data",
            base: Optional[Number] = None,
            chunk_ids: int = 50_000_000 ) -> None:
    """在一次流程中完成总量归一化和 log1p 变换。

    该函数使用 ``obs`` 中预先计算的 scale factor，对表达矩阵分块执行归一化和 log1p 变换，并写入新的表达矩阵表。适合大规模数据，避免中间归一化矩阵占用额外空间。

    Parameters
    ----------
    atlas
        Atlas 对象。通常需要已经连接到 DuckDB 数据库，并包含该函数读取或写入所需的 ``obs``、``var``、表达矩阵或结果表。
    target_sum
        归一化后每个细胞的目标总表达量。常用值为 ``10000``。
    use_obs_col
        读取或写入 ``obs`` 的列名。
    add_data
        写入数据库的新表达矩阵表名或结果列名。
    use_data
        读取的表达矩阵或结果表名称。常用值包括 ``"data"``、``"data_normalize"``、``"data_log1p"`` 和
        ``"data_scale"``。
    base
        对数变换的底数。为 ``None`` 时使用自然对数。
    chunk_ids
        按表达记录 ID 分块处理时每个 chunk 的记录数量。

    Returns
    -------
    None
        结果直接写入 Atlas 数据库或当前图形窗口。

    Examples
    --------
    先计算 scale factor，再写入 log1p 矩阵::

        sap.pp.normalize_total_scale_factor(atlas, target_sum=10000)
        sap.pp.normalize_and_log1p(atlas)

    使用自定义 scale factor 列和输出表::

        sap.pp.normalize_total_scale_factor(atlas, add_obs_col="scale_factor_1e4")
        sap.pp.normalize_and_log1p(
            atlas,
            use_obs_col="scale_factor_1e4",
            add_data="data_log1p_1e4",
        )"""

    start_time = datetime.now()

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
          AND column_name = '{use_data}'
    """).fetchone()[0]

    if col_exists == 0:
        raise ValueError(f"X_HyS_data 中不存在字段: {use_data}")

    # 1. 调用上面的函数 normalize_total → 计算 scale_factor
    normalize_total_scale_factor(
        atlas=atlas,
        target_sum=target_sum,
        add_obs_col=use_obs_col,
        use_data=use_data,
    )

    # 2. 构造 log 表达式
    if base is None:
        log_expr = f"ln(1.0 + x.{use_data} * o.{use_obs_col})"
    else:
        log_expr = f"log({float(base)}, 1.0 + x.{use_data} * o.{use_obs_col})"

    # 3. 准备输出字段
    conn.execute(f"""
        ALTER TABLE X_HyS_data
        DROP COLUMN IF EXISTS {add_data}
    """)

    conn.execute(f"""
        ALTER TABLE X_HyS_data
        ADD COLUMN {add_data} REAL
    """)

    # 4. 获取 X_HyS_data.id 范围
    min_id, max_id = conn.execute("""
        SELECT MIN(id), MAX(id)
        FROM X_HyS_data
    """).fetchone()

    if min_id is None:
        logger.info("X_HyS_data 为空，跳过")
        return

    n_chunks = math.ceil((max_id - min_id + 1) / chunk_ids)

    # 5. 分块 UPDATE
    # cell-wise QC：分块处理
    pbar = tqdm(
        range(n_chunks),
        total=n_chunks,
        desc="normalize_and_log1p",
        unit="chunk",
    )

    for i in pbar:

        start_id = min_id + i * chunk_ids
        end_id   = start_id + chunk_ids - 1

        conn.execute(f"""
            UPDATE X_HyS_data AS x
            SET {add_data} = {log_expr}
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

    logger.info(f"normalize_and_log1p Done, 耗时: {(datetime.now() - start_time).total_seconds():.2f} 秒")

# 运行结果
#   X_HyS_data 表
#     新增字段	             含义
#   data_log1p	     对表达值进行 normalize_total +  log(1+x) 转换


''' HVG '''
def highly_variable_genes(
        atlas: Atlas,
        flavor: Literal["seurat", "cv", "var"] = "seurat",
        n_top_genes: int = 2000,
        add_var_col: str = "highly_variable_genes",
        use_data: str = "data_log1p",
        n_bins: int = 20,
        min_mean: float = 0.0125,
        max_mean: float = 3.0,
        min_disp: float = 0.5,
        max_disp: float = float("inf"),
        use_filtered: bool = True,
        obs_filter_col: str = "filter_cells",
        var_filter_col: str = "filter_genes",
        inplace: bool = True,
) -> None:
    """识别高变基因并写入 var 表。

    该函数基于指定表达矩阵计算基因平均表达、方差、标准差和高变评分，并在 ``var`` 中写入高变基因标记。它类似 Scanpy 的 ``sc.pp.highly_variable_genes``，但结果保存在 Atlas 数据库中。

    Parameters
    ----------
    atlas
        Atlas 对象。通常需要已经连接到 DuckDB 数据库，并包含该函数读取或写入所需的 ``obs``、``var``、表达矩阵或结果表。
    flavor
        算法风格或统计方法。不同函数中可用于选择 Seurat 风格、CV 风格或方差风格等。
    n_top_genes
        需要标记为高变基因的基因数量。
    add_var_col
        写入 ``var`` 的结果列名。
    use_data
        读取的表达矩阵或结果表名称。常用值包括 ``"data"``、``"data_normalize"``、``"data_log1p"`` 和
        ``"data_scale"``。
    n_bins
        按平均表达量分箱时使用的箱数。
    min_mean
        高变基因筛选的平均表达量下限。
    max_mean
        高变基因筛选的平均表达量上限。
    min_disp
        高变基因筛选的离散度下限。
    max_disp
        高变基因筛选的离散度上限。
    use_filtered
        是否只在已经通过过滤标记的细胞或基因上计算。
    obs_filter_col
        ``obs`` 中用于筛选细胞的布尔列名。
    var_filter_col
        ``var`` 中用于筛选基因的布尔列名。
    inplace
        是否把结果写回 Atlas 数据库。

    Returns
    -------
    None
        结果直接写入 Atlas 数据库或当前图形窗口。

    Examples
    --------
    使用默认 Seurat 风格选择 2000 个高变基因::

        sap.pp.highly_variable_genes(atlas, n_top_genes=2000)

    在过滤后的细胞和基因上选择 3000 个高变基因::

        sap.pp.filter_cells(atlas, min_genes=200)
        sap.pp.filter_genes(atlas, min_cells=3)
        sap.pp.highly_variable_genes(
            atlas,
            n_top_genes=3000,
            use_filtered=True,
            obs_filter_col="filter_cells",
            var_filter_col="filter_genes",
        )

    为后续 PCA 构建只包含高变基因的读取索引::

        atlas.build_read_index(use_hvg=True, gene_condition="filter_genes")"""

    start_time = datetime.now()

    if flavor in ["cv", "var"]:
         _highly_variable_genes_basic(
            atlas=atlas,
            flavor=flavor,
            n_top_genes=n_top_genes,
            add_var_col=add_var_col,
            use_data=use_data,
        )

    elif flavor == "seurat":
        _highly_variable_genes_seurat(
            atlas=atlas,
            n_top_genes=n_top_genes,
            add_var_col=add_var_col,
            use_data=use_data,
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

    logger.info(f"highly_variable_genes Done, 耗时: {(datetime.now() - start_time).total_seconds():.2f} 秒")
    return None


def _highly_variable_genes_basic(
                        atlas: Atlas,
                        flavor: Literal["var", "cv"] = "cv",
                        n_top_genes: int = 2000,
                        add_var_col: str = "highly_variable_genes",
                        use_data: str = "data_log1p"
                    ) -> None:
    """识别高变基因。

    该函数在数据库中按基因计算均值、方差、标准差、非零数量和变异性得分，并按 ``n_top_genes`` 选择高变基因。

    结果写入 ``var`` 表中的 ``add_var_col`` 以及相关统计字段，可供 ``build_read_index``、PCA 和 scale
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

    add_var_col
        写回 ``obs`` 或 ``var`` 的结果列名。

    use_data
        从 ``X_HyS_data`` 中读取的表达字段。

    Notes
    -----
    该函数会把结果写回 Atlas 数据库；如果后续需要只使用过滤后的细胞或基因，通常还需要重建过滤索引。

    """

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
          AND column_name = '{use_data}'
    """).fetchone()[0]

    if col_exists == 0:
        raise ValueError(f"X_HyS_data 中不存在字段: {use_data}")

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

    conn.execute(f"""
        CREATE OR REPLACE TEMP TABLE _gene_stats AS
        WITH gene_sum AS (
            SELECT
                atlas_gene_id,
                COUNT(*) AS nnz,
                SUM({use_data}) AS sum_x,
                SUM(({use_data}) * ({use_data})) AS sum_x2
            FROM X_HyS_data
            WHERE {use_data} IS NOT NULL
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

        conn.execute(f"""
            CREATE OR REPLACE TEMP TABLE _hvg AS
            SELECT atlas_gene_id
            FROM _gene_score
            ORDER BY score DESC
            LIMIT {int(n_top_genes)}
        """)
    else:

        conn.execute("""
            CREATE OR REPLACE TEMP TABLE _hvg AS
            SELECT atlas_gene_id
            FROM _gene_score
        """)

    # 在 var 表中写入布尔结果

    conn.execute(f"""
        ALTER TABLE var
        ADD COLUMN IF NOT EXISTS {add_var_col} BOOLEAN
    """)

    conn.execute(f"""
        UPDATE var
        SET {add_var_col} = FALSE
    """)

    conn.execute(f"""
        UPDATE var
        SET {add_var_col} = TRUE
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


# 运行结果
#   var 表
#     新增字段	                     含义
#   highly_variable_genes	     对 n_top_genes 标记为true


def _highly_variable_genes_seurat(
        atlas: Atlas,
        n_top_genes: int = 2000,
        add_var_col: str = "highly_variable_genes",
        use_data: str = "data_log1p",
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

    add_var_col
        写回 ``obs`` 或 ``var`` 的结果列名。

    use_data
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

    """

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
    # 2. 检查 use_data 字段
    # -------------------------------------------------
    col_exists = conn.execute("""
        SELECT COUNT(*)
        FROM information_schema.columns
        WHERE table_name = 'X_HyS_data'
          AND column_name = ?
    """, [use_data]).fetchone()[0]

    if col_exists == 0:
        raise ValueError(f"X_HyS_data 中不存在字段: {use_data}")

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
        logger.info("[INFO] use_filtered=True")

        if has_obs_filter:
            logger.info(f"[INFO] 使用 obs.{obs_filter_col}=TRUE 的 cells")
        else:
            logger.info(f"[WARN] obs 中不存在 {obs_filter_col}，将使用全部 cells")

        if has_var_filter:
            logger.info(f"[INFO] 使用 var.{var_filter_col}=TRUE 的 genes")
        else:
            logger.info(f"[WARN] var 中不存在 {var_filter_col}，将使用全部 genes")
    else:
        logger.info("[INFO] use_filtered=False，使用全部 cells / genes")

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

    # -------------------------------------------------
    # 5. SQL 聚合 gene-level sum / sumsq
    #
    # Scanpy flavor='seurat' 输入是 log-normalized data，
    # 内部先 expm1(x)。
    #
    # 所以这里:
    #     x_raw = EXP(use_data) - 1
    #
    # 然后全细胞含 0 统计：
    #     sum_x
    #     sum_x2
    # -------------------------------------------------

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
                EXP(x.{_q(use_data)}) - 1.0 AS x_raw
            FROM X_HyS_data AS x
            JOIN _hvg_obs_keep AS o
              ON x.atlas_cell_id = o.atlas_cell_id
            JOIN _hvg_var_keep AS v
              ON x.atlas_gene_id = v.atlas_gene_id
            WHERE x.{_q(use_data)} IS NOT NULL
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

    work["highly_variable_rank"] = np.nan
    work[add_var_col] = False

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
        work[add_var_col] = work["atlas_gene_id"].isin(top_ids)

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

        work[add_var_col] = hv_mask.to_numpy()

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
                add_var_col,
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
            add_var_col,
        ]],
        on="atlas_gene_id",
        how="left",
    )

    gene_df[add_var_col] = gene_df[add_var_col].fillna(False).astype(bool)

    hvg_count = int(gene_df[add_var_col].sum())

    # -------------------------------------------------
    # 11. 写回 var
    # -------------------------------------------------
    if inplace:

        conn.execute(f"""
            ALTER TABLE var
            ADD COLUMN IF NOT EXISTS {_q(add_var_col)} BOOLEAN
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
            add_var_col,
            "highly_variable_rank",
            "means",
            "dispersions",
            "dispersions_norm",
        ]].copy()

        # 先清空全量 var 的旧结果，避免 use_filtered=True 时旧 TRUE 残留
        conn.execute(f"""
            UPDATE var
            SET
                {_q(add_var_col)} = FALSE,
                highly_variable_rank = NULL,
                means = NULL,
                dispersions = NULL,
                dispersions_norm = NULL
        """)

        conn.register("_hvg_seurat_py", write_df)

        conn.execute(f"""
            UPDATE var AS v
            SET
                {_q(add_var_col)} = p.{_q(add_var_col)},
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

    if not inplace:
        return gene_df


''' scale '''
def scale(
        atlas: Atlas,
        use_data: str = "data_log1p",
        add_data: str = "data_scale",
        add_var_col: str = "zero_scale_transform",
        max_value: float = 10.0,
        use_hvg: bool = True,
        hvg_key: str = "highly_variable_genes",
        chunk_ids: int = 20_000_000,
        ):
    """对表达矩阵按基因中心化和标准化。

    该函数基于指定表达矩阵计算每个基因的缩放值，并把结果写入新的表达矩阵表。它类似 Scanpy 的 ``sc.pp.scale``，常用于 PCA 前处理。

    Parameters
    ----------
    atlas
        Atlas 对象。通常需要已经连接到 DuckDB 数据库，并包含该函数读取或写入所需的 ``obs``、``var``、表达矩阵或结果表。
    use_data
        读取的表达矩阵或结果表名称。常用值包括 ``"data"``、``"data_normalize"``、``"data_log1p"`` 和
        ``"data_scale"``。
    add_data
        写入数据库的新表达矩阵表名或结果列名。
    add_var_col
        写入 ``var`` 的结果列名。
    max_value
        缩放后表达值的截断上限。为 ``None`` 时不截断。
    use_hvg
        是否优先使用高变基因列构建读取索引或是否只使用高变基因。
    hvg_key
        ``var`` 中标记高变基因的列名。
    chunk_ids
        按表达记录 ID 分块处理时每个 chunk 的记录数量。

    Returns
    -------
    None
        结果直接写入 Atlas 数据库或当前图形窗口。

    Examples
    --------
    对 log1p 矩阵进行标准化::

        sap.pp.scale(atlas, use_data="data_log1p", add_data="data_scale")

    只缩放高变基因，并截断极端值::

        sap.pp.highly_variable_genes(atlas, n_top_genes=3000)
        sap.pp.scale(
            atlas,
            use_hvg=True,
            hvg_key="highly_variable_genes",
            max_value=10,
        )"""

    start_time = datetime.now()
    conn = atlas.connection

    # 0. 并行
    try:
        n_threads = 4
        conn.execute(f"PRAGMA threads={n_threads}")
    except Exception:
        pass

    # 1. 输入字段检查
    if conn.execute(f"""
        SELECT COUNT(*)
        FROM information_schema.columns
        WHERE table_name = 'X_HyS_data'
          AND column_name = '{use_data}'
    """).fetchone()[0] == 0:
        raise ValueError(f"X_HyS_data 中不存在字段: {use_data}")

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
    conn.execute(f""" ALTER TABLE X_HyS_data DROP COLUMN IF EXISTS {add_data} """)
    conn.execute(f""" ALTER TABLE X_HyS_data ADD COLUMN IF NOT EXISTS {add_data} REAL """)

    conn.execute(f""" ALTER TABLE var DROP COLUMN IF EXISTS {add_var_col} """)
    conn.execute(f""" ALTER TABLE var ADD COLUMN IF NOT EXISTS {add_var_col} REAL """)

    # 3. 准备目标 gene 集合
    conn.execute("DROP TABLE IF EXISTS _target_genes")

    if use_hvg:
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
        conn.execute("DROP TABLE IF EXISTS _target_genes")
        return

    # 获取全细胞数量
    n_cells = conn.execute("""
        SELECT COUNT(*) FROM obs
    """).fetchone()[0]

    if n_cells == 0:
        conn.execute("DROP TABLE IF EXISTS _target_genes")
        raise ValueError("obs 为空，无法计算 scale")

    # 4. 一次性计算所有目标 gene 的 mean/std
    t0 = datetime.now()

    conn.execute("DROP TABLE IF EXISTS _gene_stat")

    conn.execute(f"""
        CREATE TEMP TABLE _gene_stat AS
        SELECT
            x.atlas_gene_id,

            SUM(x.{use_data}) / {n_cells} AS mean,

            SQRT(
                GREATEST(
                    SUM(x.{use_data} * x.{use_data}) / {n_cells}
                    - POWER(SUM(x.{use_data}) / {n_cells}, 2),
                    0.0
                )
            ) AS std

        FROM X_HyS_data x
        JOIN _target_genes t
          ON x.atlas_gene_id = t.atlas_gene_id
        WHERE x.{use_data} IS NOT NULL
        GROUP BY x.atlas_gene_id
    """)

    # 5. 更新 var：记录 0 值 的缩放因子 -> z-score
    t0 = datetime.now()

    conn.execute("BEGIN")
    try:
        conn.execute(f"""
            UPDATE var v
            SET {add_var_col} =
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

    # 6. 获取 id 范围
    min_id, max_id, total_rows = conn.execute("""
        SELECT MIN(id), MAX(id), COUNT(*)
        FROM X_HyS_data
    """).fetchone()

    n_chunks = math.ceil((max_id - min_id + 1) / chunk_ids)

    # 7. 按 id 分块直接 UPDATE

    done_chunks = 0
    update_start_all = datetime.now()

    pbar = tqdm(
        range(n_chunks),
        total=n_chunks,
        desc="scale",
        unit="chunk",
    )

    for chunk_idx in pbar:

        chunk_start = min_id + chunk_idx * chunk_ids
        chunk_end = min(chunk_start + chunk_ids - 1, max_id)

        t0 = datetime.now()

        conn.execute("BEGIN")
        try:

            # 对显式存储的非零值写入 z-score
            conn.execute(f"""
                UPDATE X_HyS_data x
                SET {add_data} =
                    CASE
                        WHEN g.std > 0 THEN
                            LEAST(
                                {float(max_value)},
                                GREATEST(
                                    -{float(max_value)},
                                    (x.{use_data} - g.mean) / g.std
                                )
                            )
                        ELSE 0
                    END
                FROM _gene_stat g
                WHERE x.atlas_gene_id = g.atlas_gene_id
                  AND x.id BETWEEN {chunk_start} AND {chunk_end}
                  AND x.{use_data} IS NOT NULL
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

    # 8. 清理内存
    _cleanup_transform_after_step(
        conn,
        temp_tables=["_target_genes", "_gene_stat"],
        checkpoint=True,
        collect=True,
    )

    logger.info(f"scale Done, 耗时: {(datetime.now() - start_time).total_seconds():.2f} 秒")

# 运行结果
#   X_HyS_data 表
#     新增字段	                     含义
#   data_scale	             对 data 进行 z-score 标准化， z = (x - mean_g) / std_g
#   var 表  新增字段
#   zero_scale_transform    将每个基因的 ( 0 - g.mean) / g.std 存入var表的该字段，以便将来调用


''' sqrt '''
def sqrt(
    atlas: "Atlas",
    add_data: str = "data_sqrt",
    use_data: str = "data",
    chunk_ids: int = 100_000_000) -> None:
    """对表达矩阵执行平方根变换。

    该函数读取指定表达矩阵表，对表达值执行平方根变换，并写入新的表达矩阵表。平方根变换可作为某些 count 数据降噪或可视化前处理的选择。

    Parameters
    ----------
    atlas
        Atlas 对象。通常需要已经连接到 DuckDB 数据库，并包含该函数读取或写入所需的 ``obs``、``var``、表达矩阵或结果表。
    add_data
        写入数据库的新表达矩阵表名或结果列名。
    use_data
        读取的表达矩阵或结果表名称。常用值包括 ``"data"``、``"data_normalize"``、``"data_log1p"`` 和
        ``"data_scale"``。
    chunk_ids
        按表达记录 ID 分块处理时每个 chunk 的记录数量。

    Returns
    -------
    None
        结果直接写入 Atlas 数据库或当前图形窗口。

    Examples
    --------
    对原始 counts 执行平方根变换::

        sap.pp.sqrt(atlas, use_data="data", add_data="data_sqrt")

    调整分块大小以适配大数据::

        sap.pp.sqrt(atlas, chunk_ids=50000000)"""

    start_time = datetime.now()

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
          AND column_name = '{use_data}'
        """
    ).fetchone()[0]

    if col_exists == 0:
        raise ValueError(f"X_HyS_data 中不存在字段: {use_data}")

    if add_data is None:
        raise ValueError("必须指定 add_data")

    # 1. 构造 sqrt 表达式（不处理 0）
    sqrt_expr = f"sqrt({use_data})"

    # 2. 确保输出字段存在
    # 先删旧列
    conn.execute(f"""
        ALTER TABLE X_HyS_data
        DROP COLUMN IF EXISTS {add_data}
    """)

    # 重新添加 REAL 字段
    conn.execute(f"""
        ALTER TABLE X_HyS_data
        ADD COLUMN {add_data} REAL
    """)

    # 3. 获取 id 范围
    min_id, max_id = conn.execute(
        """
        SELECT MIN(id), MAX(id)
        FROM X_HyS_data
        """
    ).fetchone()

    if min_id is None:
        logger.info("X_HyS_data 为空，跳过")
        return

    n_chunks = math.ceil((max_id - min_id + 1) / chunk_ids)

    # 4. 分块 UPDATE
    # cell-wise QC：分块处理
    pbar = tqdm(
        range(n_chunks),
        total=n_chunks,
        desc="sqrt",
        unit="chunk",
    )

    for i in pbar:

        start_id = min_id + i * chunk_ids
        end_id = start_id + chunk_ids - 1

        conn.execute(
            f"""
            UPDATE X_HyS_data
            SET {add_data} = {sqrt_expr}
            WHERE id BETWEEN {start_id} AND {end_id}
              AND {use_data} IS NOT NULL
            """
        )

    # 内存清理
    _cleanup_transform_after_step(
        conn,
        temp_tables=[],
        checkpoint=True,
        collect=True,
    )

    logger.info(f"sqrt Done, 耗时: {(datetime.now() - start_time).total_seconds():.2f} 秒")

# 运行结果
#   X_HyS_data 表
#     新增字段	                     含义
#   data_sqrt	             对 data 进行 sqrt
