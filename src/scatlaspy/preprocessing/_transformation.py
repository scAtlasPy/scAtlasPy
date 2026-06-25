from _duckdb import DuckDBPyConnection
from ..data import Atlas
from ..io import progress
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

    该内部函数用于在表达矩阵转换步骤结束后统一释放临时资源。多个预处理
    函数会在运行过程中创建 DuckDB 临时表、注册 pandas/Arrow 临时 relation，
    或执行大表 ``UPDATE``、``DROP``、``RENAME``。该函数集中处理这些清理动作，
    使每个转换步骤结束后数据库连接和 Python 进程的临时占用尽量回到稳定状态。

    函数只负责清理资源，不修改 ``X_HyS_data``、``obs`` 或 ``var`` 中的正式
    结果字段。通常由 ``normalize_total``、``log1p``、``scale``、``sqrt`` 和
    高变基因计算等内部流程在收尾阶段调用。

    Parameters
    ----------
    conn
        DuckDB 数据库连接。要求连接仍然有效，并且能执行 ``DROP TABLE``、
        ``CHECKPOINT`` 或 ``unregister`` 等清理操作。
    temp_tables
        需要删除的 DuckDB 临时表名称列表。默认值为 ``None``，会被视为
        空列表。

        函数会对列表中的每个名称执行 ``DROP TABLE IF EXISTS``。如果某个表
        已经不存在，或删除失败，清理异常会被忽略。
    unregister_tables
        需要从 DuckDB 连接中取消注册的 pandas/Arrow relation 名称列表。
        默认值为 ``None``，会被视为空列表。

        该参数常用于清理通过 ``conn.register(...)`` 注册到 DuckDB 的临时
        DataFrame。
    checkpoint
        清理后是否执行 DuckDB ``CHECKPOINT``。默认值为 ``False``。

        对于涉及大表重建、删除列、重命名表或批量更新的步骤，设置为
        ``True`` 可以尽早落盘并释放部分 DuckDB 内部空间。
    collect
        清理后是否触发 Python 垃圾回收 ``gc.collect()``。默认值为 ``True``。

    Returns
    -------
    None
        该函数只执行资源清理，不返回对象。

    Notes
    -----
    清理阶段的异常会被捕获并忽略，目的是避免清理临时对象失败时覆盖前面
    预处理步骤已经完成的主要结果。如果需要排查临时表或 DuckDB 连接状态，
    可以在调用该函数前手动检查数据库中的临时对象。

    这是内部 helper；除非需要扩展 scAtlasPy 内部流程，一般不建议在用户代码中
    直接调用。
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


def normalize_total(
        atlas: Atlas,
        target_sum: float = 10000,
        chunk_cells: int = 500_000,
        add_data: str = "data_normalize",
        use_data: str = "data_count"
) -> None:

    """按细胞总表达量进行归一化。

    该函数用于在 Atlas 数据库中对表达矩阵进行按细胞总量归一化。
    对每个细胞，函数先计算该细胞在 ``use_data`` 字段上的表达总和，
    再把该细胞所有非零表达值缩放到 ``target_sum`` 对应的尺度，
    并将结果写入 ``X_HyS_data`` 表中的 ``add_data`` 字段。

    该流程类似 Scanpy 的 ``sc.pp.normalize_total``，常用于把不同测序深度
    的细胞调整到可比较的表达尺度。例如默认参数会把每个细胞的总表达量归一化到 10,000。

    函数采用按 ``atlas_cell_id`` 范围分块的方式处理数据。每个 chunk 只计算
    当前细胞范围内的 total counts，并将当前 chunk 的表达记录写入临时目标表。
    全部 chunk 完成后，会用归一化后的表替换原 ``X_HyS_data`` 表，从而避免
    对超大表达矩阵一次性聚合造成过高内存压力。

    Parameters
    ----------
    atlas
        Atlas 对象。要求对象已经连接到 DuckDB 数据库，并且数据库中至少包含
        ``obs`` 表和 ``X_HyS_data`` 表。

        ``X_HyS_data`` 表需要包含 ``atlas_cell_id``、``id`` 以及由 ``use_data``
        指定的表达值字段。
    target_sum
        归一化后每个细胞的目标总表达量。默认值为 ``10000``。

        对每个细胞，输出值近似为：

        ``x_normalized = x / cell_total * target_sum``

        其中 ``cell_total`` 是该细胞在 ``use_data`` 字段上的表达总和。
    chunk_cells
        按 ``atlas_cell_id`` 范围分块处理时每个 chunk 覆盖的细胞 ID 数量。
        默认值为 ``500_000``。

        较大的值通常可以减少 SQL 循环次数、提高运行速度，但会增加单个 chunk
        聚合和写入时的内存占用；较小的值更稳，但运行时间可能更长。
    add_data
        写入 ``X_HyS_data`` 表的归一化结果字段名。默认值为
        ``"data_normalize"``。

        如果原表中已经存在同名字段，函数会先删除旧字段；随后通过临时表重建
        ``X_HyS_data``，并把归一化结果写入该字段。
    use_data
        从 ``X_HyS_data`` 表中读取的表达值字段名。默认值为 ``"data_count"``。
        常用值包括 ``"data_count"``、``"data_normalize"``、``"data_log1p"``
        和 ``"data_scale"``。

    Returns
    -------
    None
        结果直接写入 ``X_HyS_data`` 表中的 ``add_data`` 字段，不返回对象。

    Notes
    -----
    该函数只对 ``X_HyS_data`` 中显式存储的非零表达记录进行归一化写入。
    对于总表达量为 0 的细胞，不会产生新的表达记录。

    运行完成后，函数会清理临时表并执行 checkpoint，以降低后续步骤读取到
    中间状态或占用过多 DuckDB 临时空间的风险。

    Examples
    --------
    归一化原始 counts 到每细胞 1 万::

        sap.pp.normalize_total(atlas, target_sum=10000)

    """

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
    pbar = progress(
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


def normalize_total_scale_factor(
        atlas: Atlas,
        target_sum: float = 10000,
        add_obs_col: str = "scale_factor",
        use_data: str = "data_count",
        chunk_cells: int = 500_000,
) -> None:
    """计算每个细胞的归一化 scale factor。

    该函数用于在 ``obs`` 表中预先计算每个细胞的归一化缩放因子。
    对每个细胞，函数会统计该细胞在 ``X_HyS_data`` 表中 ``use_data`` 字段的
    表达总和，然后计算：

    ``scale_factor = target_sum / cell_total``

    结果写入 ``obs`` 表中的 ``add_obs_col`` 字段。

    该函数本身不会修改表达矩阵，也不会新增 ``X_HyS_data`` 中的表达字段。
    它主要用于配合 ``normalize_and_log1p``，让后续步骤可以在一次分块
    ``UPDATE`` 中完成归一化和 log1p，避免先写出一份完整的中间归一化矩阵。

    函数采用按 ``atlas_cell_id`` 范围分块的方式处理数据。每个 chunk 只计算
    当前细胞范围内的表达总和，并将 scale factor 写回 ``obs`` 表，适合较大
    数据集。

    Parameters
    ----------
    atlas
        Atlas 对象。要求对象已经连接到 DuckDB 数据库，并且数据库中至少包含
        ``obs`` 表和 ``X_HyS_data`` 表。

        ``obs`` 表需要包含 ``atlas_cell_id`` 字段；``X_HyS_data`` 表需要包含
        ``atlas_cell_id`` 和由 ``use_data`` 指定的表达字段。
    target_sum
        归一化后每个细胞的目标总表达量。默认值为 ``10000``。
        该值越大，后续归一化表达值的整体尺度越大。
    add_obs_col
        写入 ``obs`` 表的 scale factor 字段名。默认值为 ``"scale_factor"``。

        如果该列不存在，函数会自动新增；如果已经存在，函数会先把该列全部
        重置为 ``0``，再写入当前计算结果。
    use_data
        从 ``X_HyS_data`` 表中读取的表达值字段名。默认值为 ``"data_count"``。
        常用值包括 ``"data_count"``、``"data_normalize"``、``"data_log1p"``
        和 ``"data_scale"``。
    chunk_cells
        按 ``atlas_cell_id`` 范围分块处理时每个 chunk 覆盖的细胞 ID 数量。
        默认值为 ``500_000``。

    Returns
    -------
    None
        结果直接写入 ``obs`` 表中的 ``add_obs_col`` 字段，不返回对象。

    Notes
    -----
    对于在当前 ``use_data`` 字段上总表达量为 0 的细胞，scale factor 会写为
    ``0``，避免后续归一化时发生除零。

    该函数只写入细胞级元数据，不改变 ``X_HyS_data`` 中的表达值。

    Examples
    --------
    计算默认 scale factor::

        sap.pp.normalize_total_scale_factor(atlas, target_sum=10000)
    """

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
    pbar = progress(
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


def log1p(
                atlas: 'Atlas',
                base: Optional[Number] = None,
                add_data: str = "data_log1p",
                use_data: str = "data_normalize",
                chunk_ids: int = 100_000_000) -> None:
    """对表达矩阵执行 log1p 变换。

    该函数用于在 ``X_HyS_data`` 表中对指定表达字段执行 log1p 变换，并将
    结果写入新的表达字段。默认情况下，函数读取 ``data_normalize`` 字段，
    计算自然对数 ``ln(1 + x)``，并写入 ``data_log1p`` 字段。

    该流程类似 Scanpy 的 ``sc.pp.log1p``，通常在总量归一化之后使用，
    用于压缩表达值动态范围、降低高表达基因对后续 PCA 或聚类的影响。

    函数按 ``X_HyS_data.id`` 范围分块执行 ``UPDATE``。每个 chunk 只更新
    当前 ID 范围内且 ``use_data`` 不为 ``NULL`` 的表达记录，适合较大的
    稀疏表达矩阵。

    Parameters
    ----------
    atlas
        Atlas 对象。要求对象已经连接到 DuckDB 数据库，并且数据库中包含
        ``X_HyS_data`` 表。
    base
        对数变换的底数。默认值为 ``None``，表示使用自然对数：

        ``ln(1 + x)``

        如果传入数值，例如 ``base=2``，则计算：

        ``log_base(1 + x)``
    add_data
        写入 ``X_HyS_data`` 表的 log1p 结果字段名。默认值为
        ``"data_log1p"``。

        如果该字段已经存在，函数会先删除旧字段，再重新创建并写入。
    use_data
        从 ``X_HyS_data`` 表中读取的表达值字段名。默认值为
        ``"data_normalize"``。常用值包括 ``"data_count"``、
        ``"data_normalize"``、``"data_log1p"`` 和 ``"data_scale"``。
    chunk_ids
        按 ``X_HyS_data.id`` 范围分块处理时每个 chunk 覆盖的记录 ID 数量。
        默认值为 ``100_000_000``。

    Returns
    -------
    None
        结果直接写入 ``X_HyS_data`` 表中的 ``add_data`` 字段，不返回对象。

    Notes
    -----
    该函数不会改变 ``use_data`` 原字段，只会新增或重建 ``add_data`` 字段。
    对于 ``use_data`` 为 ``NULL`` 的记录，``add_data`` 保持为 ``NULL``。

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
    pbar = progress(
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


def expm1(
        atlas: 'Atlas',
        base: Optional[Number] = None,
        add_data: str = "data_exp1",
        use_data: str = "data_log1p",
        chunk_ids: int = 50_000_000 ) -> None:
    """对 log1p 表达矩阵执行反变换。

    该函数用于在 ``X_HyS_data`` 表中对 log1p 后的表达字段执行反变换，
    并将结果写入新的表达字段。默认情况下，函数读取 ``data_log1p``，
    计算自然指数反变换 ``exp(x) - 1``，并写入 ``data_exp1``。

    当 ``log1p`` 使用了非自然对数底数时，可以通过 ``base`` 指定同一个底数，
    使反变换与原始 log1p 变换匹配。

    函数按 ``X_HyS_data.id`` 范围分块执行 ``UPDATE``，只处理 ``use_data``
    不为 ``NULL`` 的表达记录。

    Parameters
    ----------
    atlas
        Atlas 对象。要求对象已经连接到 DuckDB 数据库，并且数据库中包含
        ``X_HyS_data`` 表。
    base
        原 log1p 变换使用的对数底数。默认值为 ``None``，表示使用自然指数：

        ``exp(x) - 1``

        如果传入数值，例如 ``base=2``，则计算：

        ``base ** x - 1``
    add_data
        写入 ``X_HyS_data`` 表的反变换结果字段名。默认值为
        ``"data_exp1"``。
    use_data
        从 ``X_HyS_data`` 表中读取的 log1p 表达字段名。默认值为
        ``"data_log1p"``。
    chunk_ids
        按 ``X_HyS_data.id`` 范围分块处理时每个 chunk 覆盖的记录 ID 数量。
        默认值为 ``50_000_000``。

    Returns
    -------
    None
        结果直接写入 ``X_HyS_data`` 表中的 ``add_data`` 字段，不返回对象。

    Notes
    -----
    该函数通常用于调试、对照或需要把 log1p 表达值恢复到线性空间的场景。
    如果输入字段并不是 log1p 尺度，反变换结果没有生物学含义。

    Examples
    --------
    将默认 log1p 矩阵还原到线性空间::

        sap.pp.expm1(atlas, use_data="data_log1p", add_data="data_exp1")

    还原以 2 为底的 log1p 矩阵::

        sap.pp.expm1(
            atlas,
            use_data="data_log1p",
            add_data="data_exp1",
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
    pbar = progress(
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


def normalize_and_log1p(
            atlas: Atlas,
            target_sum: Optional[float] = 10000,
            use_obs_col: str = "scale_factor",
            add_data: str = "data_log1p",
            use_data: str = "data_count",
            base: Optional[Number] = None,
            chunk_ids: int = 50_000_000 ) -> None:
    """在一次流程中完成总量归一化和 log1p 变换。

    该函数用于在 Atlas 数据库中把总量归一化和 log1p 变换合并为一个流程。
    它会先调用 ``normalize_total_scale_factor``，在 ``obs`` 表中计算每个细胞的scale factor；
    随后按 ``X_HyS_data.id`` 范围分块更新表达矩阵，直接计算：

    ``log(1 + x * scale_factor)``

    并将结果写入 ``X_HyS_data`` 表中的 ``add_data`` 字段。

    与先运行 ``normalize_total`` 再运行 ``log1p`` 相比，
    该函数不需要先写出一份完整的中间归一化字段，因此更适合大规模数据。
    它常用于从``data_count`` 直接生成 ``data_log1p``。

    Parameters
    ----------
    atlas
        Atlas 对象。要求对象已经连接到 DuckDB 数据库，并且数据库中至少包含
        ``obs`` 表和 ``X_HyS_data`` 表。
    target_sum
        归一化后每个细胞的目标总表达量。默认值为 ``10000``。

        该值会传给 ``normalize_total_scale_factor``，用于计算每个细胞的
        ``scale_factor``。
    use_obs_col
        ``obs`` 表中保存 scale factor 的字段名。默认值为 ``"scale_factor"``。

        函数会先在该字段中写入每个细胞的 scale factor，然后在表达矩阵分块
        更新时读取该字段。
    add_data
        写入 ``X_HyS_data`` 表的归一化并 log1p 后的表达字段名。默认值为
        ``"data_log1p"``。

        如果该字段已经存在，函数会先删除旧字段，再重新创建。
    use_data
        从 ``X_HyS_data`` 表中读取的原始表达字段名。默认值为
        ``"data_count"``。
    base
        对数变换的底数。默认值为 ``None``，表示使用自然对数 e 。
        如果传入数值，例如 ``base=2``，则计算对应底数的 log1p。
    chunk_ids
        按 ``X_HyS_data.id`` 范围分块处理时每个 chunk 覆盖的记录 ID 数量。
        默认值为 ``50_000_000``。

    Returns
    -------
    None
        结果直接写入 ``X_HyS_data`` 表中的 ``add_data`` 字段，并在 ``obs`` 表
        中写入 ``use_obs_col`` 对应的 scale factor，不返回对象。

    Notes
    -----
    该函数会覆盖 ``obs`` 表中 ``use_obs_col`` 的旧值，并重建
    ``X_HyS_data`` 表中的 ``add_data`` 字段。

    Examples
    --------
    先计算 scale factor，再写入 log1p 矩阵::

        sap.pp.normalize_total_scale_factor(atlas, target_sum=10000)
        sap.pp.normalize_and_log1p(atlas)
    """

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
    pbar = progress(
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

    该函数用于在 Atlas 数据库中根据表达矩阵识别高变基因，并将结果写入``var`` 表。
    高变基因通常用于后续 PCA、邻居图、聚类和 UMAP 等流程，
    可以减少噪声基因和低信息量基因对降维结果的影响。

    函数支持三种计算风格：

    - ``"seurat"``：类似 Scanpy/Seurat 的分箱标准化离散度方法；
    - ``"cv"``：按变异系数 ``std / mean`` 排序；
    - ``"var"``：按方差排序。

    默认 ``flavor="seurat"``。计算完成后，函数会在 ``var`` 表中写入
``add_var_col`` 指定的布尔列，标记被选中的高变基因。不同 flavor 还会
    写入对应的统计字段，例如均值、方差、离散度、标准化离散度或排名。

    Parameters
    ----------
    atlas
        Atlas 对象。要求对象已经连接到 DuckDB 数据库，并且数据库中至少包含
        ``obs``、``var`` 和 ``X_HyS_data`` 表。
    flavor
        高变基因计算方法。可选值为 ``"seurat"``、``"cv"`` 和 ``"var"``。

        ``"seurat"`` 会调用 Seurat 风格的均值分箱和标准化离散度流程；
        ``"cv"`` 和 ``"var"`` 会调用基础统计流程。
    n_top_genes
        需要标记为高变基因的数量。默认值为 ``2000``。

        当 ``flavor="seurat"`` 且 ``n_top_genes`` 不为 ``None`` 时，会优先选择
        标准化离散度最高的前 ``n_top_genes`` 个基因。
    add_var_col
        写入 ``var`` 表的高变基因布尔标记列名。默认值为
        ``"highly_variable_genes"``。
    use_data
        从 ``X_HyS_data`` 表中读取的表达字段名。默认值为 ``"data_log1p"``。
        高变基因通常建议基于 log1p 后的表达值计算。
    n_bins
        ``flavor="seurat"`` 时按平均表达量分箱的数量。默认值为 ``20``。
    min_mean
        ``flavor="seurat"`` 且使用 cutoff 模式时的平均表达量下限。
    max_mean
        ``flavor="seurat"`` 且使用 cutoff 模式时的平均表达量上限。
    min_disp
        ``flavor="seurat"`` 且使用 cutoff 模式时的标准化离散度下限。
    max_disp
        ``flavor="seurat"`` 且使用 cutoff 模式时的标准化离散度上限。
    use_filtered
        是否只在过滤后的细胞和基因上计算。默认值为 ``True``。

        当为 ``True`` 时，函数会优先使用 ``obs_filter_col`` 和 ``var_filter_col``
        指定的布尔列；如果对应列不存在，会回退到全部细胞或全部基因。
    obs_filter_col
        ``obs`` 表中用于筛选细胞的布尔列名。默认值为 ``"filter_cells"``。
    var_filter_col
        ``var`` 表中用于筛选基因的布尔列名。默认值为 ``"filter_genes"``。
    inplace
        是否把结果写回 ``var`` 表。默认值为 ``True``。

    Returns
    -------
    None
        结果直接写入 ``var`` 表中的高变基因标记列和相关统计列，不返回对象。

    Notes
    -----
    该函数不会自动重建 minibatch 读取索引。如果后续希望 PCA 或 KMeans 只使用
    新标记的高变基因，需要在运行后调用：

    ``atlas.build_read_index(use_hvg=True, gene_condition=...)``

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
    """

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
    """使用基础统计量识别高变基因。

    该内部函数用于支撑 ``highly_variable_genes(flavor="cv")`` 和
    ``highly_variable_genes(flavor="var")``。函数会在 Atlas 数据库中按基因
    聚合 ``X_HyS_data`` 表里的表达值，计算每个基因在全体细胞上的均值、
    方差、标准差、非零表达记录数量和排序得分，然后按 ``n_top_genes`` 选择
    高变基因。

    与只统计非零表达值不同，该函数会把没有显式存储在稀疏表中的 0 值也纳入
    全细胞统计。具体做法是用 ``obs`` 表中的细胞总数作为分母，并用
    ``SUM(x)`` 和 ``SUM(x * x)`` 推导每个基因的全细胞均值和方差。这样得到的
    统计量更接近完整 dense 表达矩阵上的结果。

    计算完成后，函数会把统计结果写回 ``var`` 表，供后续可视化、PCA、
    ``scale`` 或 ``build_read_index`` 等步骤复用。

    Parameters
    ----------
    atlas
        Atlas 对象。要求对象已经连接到 DuckDB 数据库，并且数据库中至少包含
        ``obs``、``var`` 和 ``X_HyS_data`` 表。

        ``obs`` 表用于统计细胞总数；
        ``var`` 表需要包含 ``atlas_gene_id`` 字段；
        ``X_HyS_data`` 表需要包含 ``atlas_gene_id`` 以及由 ``use_data`` 指定的
        表达字段。
    flavor
        基础高变基因筛选方法。可选值为 ``"cv"`` 和 ``"var"``。

        ``"cv"`` 使用变异系数 ``std / mean`` 作为排序得分；
        ``"var"`` 使用方差作为排序得分。
    n_top_genes
        需要标记为高变基因的数量。默认值为 ``2000``。

        当为 ``None`` 时，不再截取前 N 个基因，而是把参与计算的所有基因写入
        临时高变基因集合。
    add_var_col
        写入 ``var`` 表的高变基因布尔标记列名。默认值为
        ``"highly_variable_genes"``。
    use_data
        从 ``X_HyS_data`` 表中读取的表达字段名。默认值为 ``"data_log1p"``。

    Returns
    -------
    None
        结果直接写入 ``var`` 表，不返回对象。

    Notes
    -----
    该函数会在 ``var`` 表中新增或更新以下字段：

    - ``hvg_mean``：每个基因在全体细胞上的均值；
    - ``hvg_var``：每个基因在全体细胞上的方差；
    - ``hvg_std``：每个基因在全体细胞上的标准差；
    - ``hvg_score``：按 ``flavor`` 计算得到的排序得分；
    - ``hvg_nnz``：每个基因在 ``X_HyS_data`` 中显式存储的非零记录数量；
    - ``add_var_col``：是否被选为高变基因的布尔标记。

    该基础方法不读取 ``filter_cells`` 或 ``filter_genes``。如果需要只在过滤后
    的细胞和基因上使用 Seurat 风格流程，可以通过
    ``highly_variable_genes(flavor="seurat", use_filtered=True)`` 调用。

    Examples
    --------
    使用变异系数选择 2000 个高变基因::

        _highly_variable_genes_basic(atlas, flavor="cv", n_top_genes=2000)

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

    该内部函数用于支撑 ``highly_variable_genes(flavor="seurat")``。函数实现
    类似 Scanpy/Seurat 的高变基因流程：先按基因计算均值和离散度，再按平均
    表达量分箱，在每个 bin 内对离散度做标准化，最后根据标准化离散度选择
    高变基因。

    默认情况下，函数假设 ``use_data`` 是 log1p 后的表达字段。因此在计算
    Seurat 风格的均值和离散度前，会先对显式表达值执行
    ``EXP(use_data) - 1.0``，近似还原到原始表达尺度，再把稀疏矩阵中未显式
    存储的 0 值纳入全细胞统计。

    函数支持只在过滤后的细胞和基因上计算。当 ``use_filtered=True`` 时，
    会优先读取 ``obs_filter_col`` 和 ``var_filter_col`` 指定的布尔列；
    如果某个过滤列不存在，则对应维度自动回退到使用全部细胞或全部基因。

    Parameters
    ----------
    atlas
        Atlas 对象。要求对象已经连接到 DuckDB 数据库，并且数据库中至少包含
        ``obs``、``var`` 和 ``X_HyS_data`` 表。

        ``obs`` 表需要包含 ``atlas_cell_id`` 字段；
        ``var`` 表需要包含 ``atlas_gene_id`` 和 ``atlas_gene_name`` 字段；
        ``X_HyS_data`` 表需要包含 ``atlas_cell_id``、``atlas_gene_id`` 以及
        由 ``use_data`` 指定的表达字段。
    n_top_genes
        需要标记为高变基因的数量。默认值为 ``2000``。

        当不为 ``None`` 时，函数会忽略 ``min_mean``、``max_mean``、
        ``min_disp`` 和 ``max_disp``，直接选择标准化离散度最高的前
        ``n_top_genes`` 个基因。

        当为 ``None`` 时，函数进入 cutoff 模式，根据均值和标准化离散度的
        阈值范围选择高变基因。
    add_var_col
        写入 ``var`` 表的高变基因布尔标记列名。默认值为
        ``"highly_variable_genes"``。
    use_data
        从 ``X_HyS_data`` 表中读取的表达字段名。默认值为 ``"data_log1p"``。

        该 Seurat 风格实现通常应基于 log1p 后的数据运行，因为内部会用
        ``EXP(use_data) - 1.0`` 还原到原始尺度。
    n_bins
        按平均表达量分箱的数量。默认值为 ``20``。

        分箱用于在相似平均表达水平的基因之间比较离散度，降低平均表达量对
        离散度排序的影响。
    min_mean
        cutoff 模式下参与高变基因选择的平均表达量下限。
    max_mean
        cutoff 模式下参与高变基因选择的平均表达量上限。
    min_disp
        cutoff 模式下参与高变基因选择的标准化离散度下限。
    max_disp
        cutoff 模式下参与高变基因选择的标准化离散度上限。
    use_filtered
        是否只在过滤后的细胞和基因上计算。默认值为 ``True``。

        当过滤列存在时，只保留对应列为 ``TRUE`` 的细胞或基因；
        当过滤列不存在时，会记录日志并使用该维度的全部对象。
    obs_filter_col
        ``obs`` 表中用于筛选细胞的布尔列名。默认值为 ``"filter_cells"``。
    var_filter_col
        ``var`` 表中用于筛选基因的布尔列名。默认值为 ``"filter_genes"``。
    inplace
        是否将结果写回 ``var`` 表。默认值为 ``True``。

        为 ``True`` 时，函数会更新数据库并返回 ``None``；
        为 ``False`` 时，函数不会写回 ``var`` 表，而是返回包含统计结果的
        ``pandas.DataFrame``。

    Returns
    -------
    None or pandas.DataFrame
        当 ``inplace=True`` 时，结果直接写入 ``var`` 表，不返回对象。

        当 ``inplace=False`` 时，返回一个以基因为行的 DataFrame，包含
        ``atlas_gene_id``、``atlas_gene_name``、``means``、``dispersions``、
        ``dispersions_norm``、``highly_variable_rank`` 和 ``add_var_col`` 等字段。

    Notes
    -----
    ``inplace=True`` 时，函数会在 ``var`` 表中新增或更新以下字段：

    - ``add_var_col``：是否被选为高变基因的布尔标记；
    - ``highly_variable_rank``：按标准化离散度排序得到的排名；
    - ``means``：Seurat 风格均值，基于 ``log1p(mean_raw)``；
    - ``dispersions``：离散度，基于 ``log(variance / mean)``；
    - ``dispersions_norm``：按均值分箱后标准化的离散度。

    写回前会先清空 ``var`` 表中的旧结果，避免在 ``use_filtered=True`` 时旧的
    ``TRUE`` 标记残留。

    Examples
    --------
    在过滤后的细胞和基因上选择 2000 个高变基因::

        _highly_variable_genes_seurat(
            atlas,
            n_top_genes=2000,
            use_filtered=True,
        )
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

        该内部 helper 用于在拼接 DuckDB SQL 时引用动态列名或表名。函数会先
        转义名称中已有的双引号，再在外层补上双引号，避免字段名中包含特殊
        字符时造成 SQL 解析错误。

        Parameters
        ----------
        name
            需要引用的列名、表名或其他 SQL 标识符。

        Returns
        -------
        quoted_name
            加双引号后的 SQL 标识符。

        Notes
        -----
        该函数只负责 SQL 标识符引用，不负责检查字段是否存在，也不应当用于
        引用 SQL 字符串字面量。
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

    该函数用于在 Atlas 数据库中对表达矩阵按基因进行中心化和标准化，
    并将 z-score 结果写入 ``X_HyS_data`` 表中的 ``add_data`` 字段。
    它类似 Scanpy 的 ``sc.pp.scale``，常用于 PCA、KMeans 和其他需要
    标准化输入的下游分析。

    对每个目标基因，函数先基于 ``use_data`` 计算全体细胞上的均值和标准差，
    然后对显式存储的非零表达记录计算：

    ``z = (x - mean_gene) / std_gene``

    如果 ``max_value`` 不为 ``None``，结果会被截断到
    ``[-max_value, max_value]`` 范围内。

    由于稀疏矩阵中未显式存储的 0 值在 scale 后通常不再等于 0，函数还会在
    ``var`` 表中写入 ``add_var_col`` 字段，用于记录每个基因原始 0 值对应的
    scale 后填充值，即 ``(0 - mean_gene) / std_gene``。minibatch dense 读取
    ``data_scale`` 时会使用该字段填充稀疏 0 位点。

    Parameters
    ----------
    atlas
        Atlas 对象。要求对象已经连接到 DuckDB 数据库，并且数据库中至少包含
        ``obs``、``var`` 和 ``X_HyS_data`` 表。
    use_data
        从 ``X_HyS_data`` 表中读取的表达字段名。默认值为 ``"data_log1p"``。
    add_data
        写入 ``X_HyS_data`` 表的 scale 结果字段名。默认值为 ``"data_scale"``。
    add_var_col
        写入 ``var`` 表的零值 scale 填充值字段名。默认值为
        ``"zero_scale_transform"``。
    max_value
        scale 后表达值的截断上限。默认值为 ``10.0``。

        当前实现会使用 ``[-max_value, max_value]`` 作为截断范围。
    use_hvg
        是否只对高变基因集合计算并写入 scale 结果。默认值为 ``True``。

        当为 ``True`` 时，函数只使用 ``var`` 表中 ``hvg_key=TRUE`` 的基因；
        当为 ``False`` 时，使用全部基因。
    hvg_key
        ``var`` 表中标记高变基因的布尔列名。默认值为
        ``"highly_variable_genes"``。
    chunk_ids
        按 ``X_HyS_data.id`` 范围分块写回 scale 结果时每个 chunk 覆盖的记录
        ID 数量。默认值为 ``20_000_000``。

    Returns
    -------
    None
        不返回对象。
        结果直接写入 ``X_HyS_data`` 表中的 ``add_data`` 字段；
        默认：data_scale，表示 对 data 进行 z-score 标准化， z = (x - mean_g) / std_g;
        以及 ``var`` 表中的 ``add_var_col`` 字段。
        默认：zero_scale_transform ，表示 将每个基因的 ( 0 - g.mean) / g.std 存入var表的该字段，以便将来调用。
    Notes
    -----
    如果 ``use_hvg=True``，只有高变基因会获得 ``data_scale`` 结果；非目标基因
    不会参与后续基于 ``data_scale`` 的 HVG 读取索引。

    运行完成后，如果下游 minibatch、PCA 或 KMeans 需要读取 scale 后数据，
    应运行 ``atlas.build_read_index(use_hvg=True, use_data=add_data, ...)``。

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

    pbar = progress(
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


def sqrt(
    atlas: "Atlas",
    add_data: str = "data_sqrt",
    use_data: str = "data_count",
    chunk_ids: int = 100_000_000) -> None:
    """对表达矩阵执行平方根变换。

    该函数用于在 ``X_HyS_data`` 表中对指定表达字段执行平方根变换，并将
    结果写入新的表达字段。默认情况下，函数读取 ``data_count``，计算
    ``sqrt(x)``，并写入 ``data_sqrt``。

    平方根变换可作为 count 数据的一种简单方差稳定或可视化前处理方式。
    与 log1p 相比，sqrt 对低表达 count 的压缩更温和。

    函数按 ``X_HyS_data.id`` 范围分块执行 ``UPDATE``，只处理 ``use_data``
    不为 ``NULL`` 的表达记录。

    Parameters
    ----------
    atlas
        Atlas 对象。要求对象已经连接到 DuckDB 数据库，并且数据库中包含
        ``X_HyS_data`` 表。
    add_data
        写入 ``X_HyS_data`` 表的平方根变换结果字段名。默认值为
        ``"data_sqrt"``。
    use_data
        从 ``X_HyS_data`` 表中读取的表达字段名。默认值为 ``"data_count"``。
        常用值包括 ``"data_count"``、``"data_normalize"``、``"data_log1p"``
        和 ``"data_scale"``。
    chunk_ids
        按 ``X_HyS_data.id`` 范围分块处理时每个 chunk 覆盖的记录 ID 数量。
        默认值为 ``100_000_000``。

    Returns
    -------
    None
        结果直接写入 ``X_HyS_data`` 表中的 ``add_data`` 字段，不返回对象。

    Notes
    -----
    该函数不会修改 ``use_data`` 原字段，只会新增或重建 ``add_data`` 字段。
    如果 ``use_data`` 中存在负值，DuckDB 的 ``sqrt`` 会产生无效结果或错误；
    因此该函数通常应作用于 count 或非负归一化表达字段。

    Examples
    --------
    对原始 counts 执行平方根变换::

        sap.pp.sqrt(atlas, use_data="data_count", add_data="data_sqrt")

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
    pbar = progress(
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
