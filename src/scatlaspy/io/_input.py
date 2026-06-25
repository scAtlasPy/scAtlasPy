from __future__ import annotations
from _duckdb import DuckDBPyConnection
import os
import logging
import h5py
import pyarrow as pa
import time
from datetime import datetime
import gc
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from os import PathLike
from scipy import sparse
from . import progress
from typing import TYPE_CHECKING, Any, Literal
if TYPE_CHECKING:   # TYPE_CHECKING = 给 IDE / 类型检查器看的导入;  正常运行时 = 不执行这个导入，避免循环导入
    from ..data import Atlas

XScale = Literal["count", "log"]

# 获取日志记录器
logger = logging.getLogger("Atlas")
logger.addHandler(logging.NullHandler())


# 统一的 h5ad 导入接口
def load_h5ad(
    h5ad_path: PathLike[str] | str | list[PathLike[str] | str],
    atlas: Atlas,
    *,
    load_type: Literal["order", "random"] = "random",
    cells_per_block: int | None = None,
) -> Any:
    """将 h5ad 文件导入 Atlas 数据库。

    该函数是 h5ad 导入 Atlas 的统一入口。它可以读取单个 ``.h5ad`` 文件，也可以
    读取多个 ``.h5ad`` 文件组成的列表，并把细胞元数据、基因元数据和表达矩阵
    写入 Atlas 的 DuckDB 数据库。

    导入时会根据 ``load_type`` 和 ``h5ad_path`` 的类型自动分派到对应实现：

    - 单文件 + ``"order"``：按原始细胞顺序导入；
    - 单文件 + ``"random"``：按 shuffle-window 方式随机导入；
    - 多文件 + ``"order"``：按文件列表顺序和文件内细胞顺序导入；
    - 多文件 + ``"random"``：把多个文件切成 block 后全局随机导入。

    表达矩阵会统一保存为 count 尺度，写入 ``X_HyS_data.data_count`` 字段。
    如果输入 ``X`` 被检测为 log 尺度，会在写入前转换回 count。

    Parameters
    ----------
    h5ad_path
        输入 ``.h5ad`` 文件路径，或多个 ``.h5ad`` 文件路径组成的列表。
    atlas
        Atlas 对象。函数会通过 ``atlas.connect("r+")`` 获取 DuckDB 连接，并把
        数据写入该 Atlas 数据库。
    load_type
        导入方式，只支持 ``"order"`` 和 ``"random"``。
        当 ``h5ad_path`` 是单个路径时，分别执行单文件顺序或随机导入；
        当 ``h5ad_path`` 是列表时，分别执行多文件顺序或随机导入。
    cells_per_block
        读取和写入表达矩阵时每个连续 cell block 包含的细胞数量。
        为 ``None`` 时会根据细胞总数自动估算一个默认值。

    Returns
    -------
    Any
        返回所调用底层导入函数的结果。当前主要用于执行导入副作用，通常不依赖
        返回值。

    Notes
    -----
    ``random`` 导入会重排细胞顺序，因此单文件随机导入默认不会导入 ``obsm``，
    以避免 embedding 与重排后的 ``obs`` 错位；
    ``order`` 导入会保留原始顺序，因此可以安全导入 ``obsm`` 和 ``varm``。

    Examples
    --------
    顺序导入单个 h5ad 文件::

        atlas = sap.Atlas(r"F:\\data\\pbmc")
        atlas.load_h5ad(r"F:\\data\\pbmc.h5ad", load_type="order")

    随机分块导入多个文件::

        atlas.load_h5ad(
            [r"F:\\data\\batch1.h5ad", r"F:\\data\\batch2.h5ad"],
            load_type="random",
            cells_per_block=1000,
        )"""

    start_time = datetime.now()

    # =====================================================
    # 1. 参数检查
    # =====================================================
    valid_load_types = {
        "order",
        "random",
    }

    if load_type not in valid_load_types:
        raise ValueError(
            "load_type 只能是 "
            f"{sorted(valid_load_types)}，当前为: {load_type}"
        )

    if cells_per_block is not None and not isinstance(cells_per_block, int):
        raise TypeError(
            f"cells_per_block 必须是 int 或 None，当前类型为: {type(cells_per_block)}"
        )

    if cells_per_block is not None and cells_per_block <= 0:
        raise ValueError("cells_per_block 必须 > 0")


    # =====================================================
    # 2. 多文件导入：根据 load_type 自动选择顺序或随机逻辑
    # =====================================================
    if isinstance(h5ad_path, (list, tuple)):
        if load_type == "order":
            logger.info("[INFO] load_type = order，多文件顺序导入")
            return _load_h5ad_list_order(
                h5ad_paths=h5ad_path,
                atlas=atlas,
                cells_per_block=cells_per_block,
            )

        logger.info("[INFO] load_type = random，多文件随机导入")
        return _load_h5ad_list_random(
            h5ad_paths=h5ad_path,
            atlas=atlas,
            cells_per_block=cells_per_block,
        )

    # =====================================================
    # 3. 单文件导入：先轻量读取 n_cells，再进入对应逻辑
    # =====================================================

    if not isinstance(h5ad_path, (str, PathLike)):
        raise TypeError(
            f"h5ad_path 必须是 str，当前类型为: {type(h5ad_path)}"
        )

    h5ad_path = os.fspath(h5ad_path)

    # 轻量读取 n_cells，用于统一 cells_per_block 默认值
    adata_backed = sc.read_h5ad(h5ad_path, backed="r")
    n_cells = adata_backed.n_obs
    cells_per_block = _normalize_cells_per_block(cells_per_block, n_cells)

    # =====================================================
    # 4. order：顺序导入
    # =====================================================
    if load_type == "order":

        logger.info("[INFO] load_type = order")

        return _load_h5ad_order(
            h5ad_path=h5ad_path,
            atlas=atlas,
            cells_per_block=cells_per_block,
        )

    # =====================================================
    # 5. random：普通随机导入
    # =====================================================
    if load_type == "random":

        logger.info("[INFO] load_type = random")

        return _load_h5ad_random(
            h5ad_path=h5ad_path,
            atlas=atlas,
            cells_per_block=cells_per_block,
        )

    logger.info(f"load_h5ad Done, 耗时: {(datetime.now() - start_time).total_seconds():.2f} 秒")
    return None


# 将 AnnData 对象写入 Atlas 数据库
def load_anndata(adata:AnnData, atlas:Atlas):

    """将 AnnData 对象写入 Atlas 数据库。

    该函数直接接收内存中的 AnnData 对象，并把其中的 ``obs``、``var``、
    ``X``、``obsm`` 和 ``varm`` 写入 Atlas 数据库。适合已经用 Scanpy 或其他
    工具完成读取、筛选或预处理后，再转入 Atlas 管理的场景。

    与 ``load_h5ad`` 的 backed 分块导入不同，该函数要求 AnnData 已经在内存中，
    因此更适合中小型数据或已经抽样后的数据。

    Parameters
    ----------
    adata
        AnnData 对象。函数会把其中的 ``obs``、``var``、表达矩阵和可支持的结果写入 Atlas 数据库。
    atlas
        Atlas 对象。要求对象已经连接或可连接到 DuckDB 数据库。

    Returns
    -------
    None
        结果直接写入 Atlas 数据库，不返回对象。

    Notes
    -----
    该路径会重建 ``obs``、``var``、``X_HyS_indptr`` 和 ``X_HyS_data`` 表，并把
    ``obsm``、``varm`` 中的二维数组写成 ``obsm_*``、``varm_*`` 表。

    Examples
    --------
    从 Scanpy 读取并导入::

        adata = sc.read_h5ad(r"F:\\data\\pbmc.h5ad")
        atlas = sap.Atlas(r"F:\\data\\pbmc")
        atlas.load_anndata(adata)
    """

    try:
        logger.info("准备数据表...")

        if hasattr(adata, 'obs'):
            _add_obs(adata, atlas)  # 细胞表数据（对应obs）,
        else:
            logger.info("Skipping obs layer")

        if hasattr(adata, 'var'):
            _add_var(adata, atlas)  # 基因表数据（对应var）
        else:
            logger.info("Skipping var layer")

        if hasattr(adata, 'X'):
            start_time = time.time()
            _add_x_hys_chunked(adata, atlas, chunk_size=500) # 分块导入X表数据
            end_time = time.time()
            logger.info(" X表的导入用时为： " + str(end_time - start_time))
        else:
            logger.info("Skipping X layer")

        if hasattr(adata, 'obsm'):
            _add_obsm(adata,atlas)
        else:
            logger.info("Skipping obsm layer")

        if hasattr(adata, 'varm'):
            _add_varm(adata,atlas)
        else:
            logger.info("Skipping varm layer")

        # 显示表结构
        logger.debug("数据库表结构:")
        tables = atlas.connection.execute("SHOW TABLES")
        if tables:
            logger.debug(f"数据库中的表: {tables}")

        logger.info("AnnData数据成功加载到数据库")

    except Exception as e:
        logger.error(f"加载数据失败: {str(e)}")
        logger.exception("加载数据异常详情:")
        raise


# 顺序读取，小文件读取，支持多种数据格式的导入
def load_multi_format(file_path: PathLike[str] | str, atlas: Atlas):

    """根据文件格式导入数据到 Atlas。

    该函数是小型或通用格式数据的导入入口。它会先根据 ``file_path`` 的后缀
    调用 ``_read_smart`` 读取为内存 AnnData，然后调用 ``load_anndata`` 写入
    Atlas 数据库。

    与 ``load_h5ad`` 的 backed 分块导入不同，该函数会先把数据完整读入内存，
    因此更适合小文件、临时转换或非 h5ad 格式数据。

    Parameters
    ----------
    file_path
        输入文件路径。函数会根据文件格式选择合适的读取方式。
    atlas
        Atlas 对象。要求对象已经连接或可连接到 DuckDB 数据库。

    Returns
    -------
    None
        结果直接写入 Atlas 数据库，不返回对象。

    Notes
    -----
    支持的读取格式由 ``_read_smart`` 决定，包括 h5ad、loom、Matrix Market、
    csv、txt/tsv、Excel、10x h5 和 UMI-tools 等常见格式。

    Examples
    --------
    自动识别并导入文件::

        atlas = sap.Atlas(r"F:\\data\\pbmc")
        atlas.load_multi_format(r"F:\\data\\pbmc.h5ad")
    """

    start_time = datetime.now()
    adata = _read_smart(file_path)
    load_anndata(adata, atlas)
    logger.info(f"load_multi_format Done, 耗时: {(datetime.now() - start_time).total_seconds():.2f} 秒")


# 检查 Atlas 数据库中的基因名是否重复
def rename_duplicated_genes(atlas: Atlas, gene_name_column: str = "atlas_gene_name"):
    """检查 Atlas 数据库中的基因名是否重复。

    该函数读取 ``var`` 表中的基因名称列，判断是否存在重复基因名，并为重复项
    添加 ``_1``、``_2`` 等后缀。重复基因名可能影响按名称绘图、差异基因展示
    和 AnnData 导出，因此导入后可以运行该函数进行清洗。

    对每个重复基因名，第一次出现的名称保持不变，后续重复项按
    ``原名_1``、``原名_2`` 的形式重命名。

    Parameters
    ----------
    atlas
        Atlas 对象。要求对象已经连接到 DuckDB 数据库，并且数据库中存在``var`` 表。
    gene_name_column
        保存基因名称的 ``var`` 列名。通常为 ``"atlas_gene_name"``。

    Returns
    -------
    bool
        成功检查或更新时返回 ``True``；如果 ``var`` 表不存在，返回 ``False``。

    Notes
    -----
    只有检测到重复基因名时才会更新 ``var`` 表；没有重复时保持原表不变。

    Examples
    --------
    检查默认基因名称列::

        atlas.rename_duplicated_genes()
    """
    logger.info(f" 开始在数据库 var表 中清洗基因名 ")

    # 检查表是否存在
    tables = atlas.connection.execute("SHOW TABLES").df()
    if 'var' not in tables['name'].values:
        logger.error("var表 不存在，请先导入数据")
        return False

    logger.info("开始添加后缀模式...")

    # 1. 构建带后缀的临时 var 表（var_with_suffix）
    atlas.connection.execute(f"""
        CREATE OR REPLACE TEMPORARY TABLE var_with_suffix AS
        WITH ranked_genes AS (
            SELECT *,
                   ROW_NUMBER() OVER (
                       PARTITION BY {gene_name_column}
                       ORDER BY atlas_gene_id
                   ) AS rn
            FROM var
        )
        SELECT
            atlas_gene_id,
            CASE
                WHEN rn = 1 THEN {gene_name_column} -- 第一个出现的基因名保持不变
                ELSE {gene_name_column} || '_' || (rn - 1)::VARCHAR -- 后续重复基因添加后缀：gene_1, gene_2, ...
            END AS {gene_name_column}
        FROM ranked_genes
        ORDER BY atlas_gene_id
    """)

    # 创建带后缀的基因名临时表 var_with_suffix
    # atlas_gene_id | atlas_gene_name
    # ---|---------
    # 1  | TP53     -- 第一个TP53保持原名
    # 2  | EGFR     -- EGFR只有一个，保持原名
    # 3  | TP53_1   -- 第二个TP53添加_1后缀
    # 4  | BRAF     -- BRAF只有一个，保持原名
    # 5  | TP53_2   -- 第三个TP53添加_2后缀

    # 2. 记录 atlas_gene_name 实际发生变化的映射关系（仅用于日志）
    gene_mapping = atlas.connection.execute(f"""
        SELECT
            v.{gene_name_column}  AS original_gene_name,
            vs.{gene_name_column} AS new_gene_name
        FROM var v
        JOIN var_with_suffix vs
            ON v.atlas_gene_id = vs.atlas_gene_id
        WHERE v.{gene_name_column} != vs.{gene_name_column}
    """).df()
    # 获取基因名映射关系。gene_mapping数据框
    #    original_gene_name  new_gene_name
    # 0              TP53       TP53_1
    # 1              TP53       TP53_2

    # 3.  根据是否存在重复基因输出不同日志
    if len(gene_mapping) > 0:
        logger.info(
            f"发现 {len(gene_mapping)} 个重复基因，已成功添加后缀"
        )
    else:
        logger.info("未发现重复基因，var 表保持不变")

    # 4. 更新var表
    if(len(gene_mapping)>0):
        atlas.connection.execute("DROP TABLE var")
        atlas.connection.execute("ALTER TABLE var_with_suffix RENAME TO var")
        atlas.connection.execute("ALTER TABLE var ADD PRIMARY KEY (atlas_gene_id)")
        logger.info("var表已更新")

    logger.info("rename_duplicated_genes Done")
    return True


def _load_h5ad_list_random(
    h5ad_paths: PathLike[str] | str | list[PathLike[str] | str],
    atlas: Atlas,
    cells_per_block: int | None = None,
    *,
    shuffle_blocks: bool = True,
    shuffle_cells: bool = True,
):
    """随机导入多个 h5ad 文件到 Atlas 数据库。

    该内部函数用于多文件 ``load_type="random"`` 场景。它会先把每个 h5ad 文件
    按 ``cells_per_block`` 切成连续 block，再把所有文件的 block 合并到全局
    block pool 中随机打乱。

    每次读取 ``blocks_per_pool`` 个 block 后，会合并为 cell pool，对 pool 内
    细胞整体随机一次，然后写入 ``obs``、``var``、``X_HyS_indptr`` 和
    ``X_HyS_data``。表达矩阵会统一转换并保存为 count 尺度。

    该策略适合多个文件大小不一致的场景，既避免 round-robin 后期只剩大文件，也减少 h5ad cell-level 随机 IO。

    Parameters
    ----------
    h5ad_paths
        一个或多个 h5ad 文件路径。

    atlas
        Atlas 对象。函数会连接并写入对应 DuckDB 数据库。

    cells_per_block
        每批读取、写入或处理的细胞数量；较大值通常更快但占用更多内存。
    shuffle_blocks
        是否打乱全局 block 顺序。多文件随机导入开启，多文件顺序导入关闭。
    shuffle_cells
        是否打乱每个 cell pool 内部的细胞顺序。多文件随机导入开启，多文件顺序导入关闭。

    Returns
    -------
    None
        结果直接写入 Atlas 数据库，不返回对象。

    Notes
    -----
    该函数会根据 Atlas 的 ``db_memory_limit`` 和样本估算的单细胞内存占用动态
    估算 ``blocks_per_pool``，以控制 shuffle window 的内存峰值。

    Examples
    --------
    内部随机导入多个文件::

        _load_h5ad_list_random([path1, path2], atlas, cells_per_block=1000)
    """


    # 支持单路径 / 多路径
    if isinstance(h5ad_paths, (str, PathLike)):
        h5ad_paths = [h5ad_paths]

    h5ad_paths = [os.fspath(path) for path in h5ad_paths]

    if len(h5ad_paths) == 0:
        raise ValueError("h5ad_paths 不能为空")

    # ===== 统计全局 n_cells =====
    total_n_cells = 0
    file_cell_counts = []

    for path in h5ad_paths:
        ad = sc.read_h5ad(path, backed="r")
        n = ad.n_obs
        total_n_cells += n
        file_cell_counts.append(n)
        ad.file.close()

    # ===== 全局 cells_per_block 计算  =====
    cells_per_block = _normalize_cells_per_block(cells_per_block, total_n_cells)

    commit_every = 5  # 每多少次 pool flush 提交一次
    gc_every = 5    # 每多少次 pool flush 做一次 gc

    file_num = len(h5ad_paths)

    rng = np.random.default_rng()

    # 连接数据库
    conn = atlas.connect("r+")
    atlas.connection = conn

    # 全局游标
    global_cell_id = 0
    global_indptr_id = 0
    global_indptr_offset = 0
    global_data_id = 0

    var_written = False

    logger.info(f"[INFO] 文件数量: {file_num:,}")
    logger.info("[INFO] 表达矩阵统一写入为 count")

    file_states = []

    ref_var_names = None
    ref_n_genes = None

    # 全局 block 索引池 ; 每个元素记录一个 block 来自哪个文件、起止位置是多少
    all_block_refs = []
    max_estimated_bytes_per_cell = 0.0

    try:
        for file_idx, h5ad_path in enumerate(h5ad_paths):

            adata_backed = sc.read_h5ad(h5ad_path, backed="r")

            n_cells = adata_backed.n_obs
            n_genes = adata_backed.n_vars

            logger.info(f"[INFO] 当前文件维度 : {n_cells:,} × {n_genes:,}")

            # 每个文件单独检测 X scale，并估算内存占用
            x_info = _inspect_x_from_backed(
                adata_backed,
                sample_n=5000,
            )
            source_x_scale = x_info["x_scale"]
            max_estimated_bytes_per_cell = max(
                max_estimated_bytes_per_cell,
                float(x_info["estimated_bytes_per_cell"]),
            )

            logger.info(f"[INFO] 当前文件 X 判断为: {source_x_scale}")

            if source_x_scale == "count":
                logger.info("[INFO] 当前文件 X 已是 count，直接写入。")
            else:
                logger.info("[INFO] 当前文件 X 将在读取 block 后转换为 count")

            # ---------------- 检查 gene 数量和顺序 ----------------
            cur_var_names = adata_backed.var.index.astype(str).to_numpy()

            if file_idx == 0:
                ref_var_names = cur_var_names
                ref_n_genes = n_genes
            else:
                if n_genes != ref_n_genes:
                    raise ValueError(
                        f"第 {file_idx + 1} 个文件 gene 数量不一致："
                        f"{n_genes} != {ref_n_genes}"
                    )

                if not np.array_equal(cur_var_names, ref_var_names):
                    raise ValueError(
                        f"第 {file_idx + 1} 个文件 gene 顺序与第一个文件不一致，"
                        f"不能直接合并导入。"
                    )

            # ---------------- 每个文件内部按 cells_per_block 切 block ----------------
            block_starts = np.arange(0, n_cells, cells_per_block, dtype=np.int64)

            # 把所有文件的 block 放进 all_block_refs；随机模式会在后面统一全局 shuffle。
            for block_start in block_starts:
                block_start = int(block_start)
                block_end = min(block_start + cells_per_block, n_cells)

                all_block_refs.append(
                    {
                        "file_idx": file_idx,
                        "block_start": block_start,
                        "block_end": block_end,
                    }
                )

            file_states.append(
                {
                    "file_idx": file_idx,
                    "h5ad_path": h5ad_path,
                    "adata_backed": adata_backed,
                    "n_cells": n_cells,
                    "n_genes": n_genes,
                    "source_x_scale": source_x_scale,
                }
            )

        # 全局 block 索引池：随机模式打乱；顺序模式保留文件列表和文件内部顺序。
        total_blocks = len(all_block_refs)

        if total_blocks == 0:
            raise ValueError("所有 h5ad 文件的 cell 数量为 0，无法导入")

        estimated_window_cells, blocks_per_pool = _estimate_window_cells_and_blocks_per_pool(
            memory_limit=_get_atlas_memory_limit(atlas),
            cells_per_block=cells_per_block,
            estimated_bytes_per_cell=max_estimated_bytes_per_cell,
        )

        if shuffle_blocks:
            rng.shuffle(all_block_refs)

        # 动态建表：只用第一个文件建表
        first_backed = file_states[0]["adata_backed"]

        _create_obs_table_from_adata(conn, first_backed[:1])
        _create_var_table_from_adata(conn, first_backed[:1])
        _create_hys_tables(conn)

        # 多个 block 读到的 batch 合并后写入数据库；随机模式会先整体打乱 cell 顺序。
        def _flush_cell_pool(cell_pool: list[AnnData], flush_i: int):

            """合并并写入当前多文件导入的 cell pool。

            该嵌套 helper 用于 ``_load_h5ad_list_random`` 和
            ``_load_h5ad_list_order`` 的共享主体。函数会把当前 pool 中多个
            AnnData block 的 ``X``、``obs`` 和 ``var`` 合并起来；随机模式下会在
            pool 内再次打乱细胞顺序；随后把结果写入 ``obs``、``var``、
            ``X_HyS_indptr`` 和 ``X_HyS_data``。

            Parameters
            ----------
            cell_pool
                当前 flush 中收集到的 AnnData block 列表。
            flush_i
                当前 flush 的编号，用于日志输出。

            Returns
            -------
            tuple[int, int]
                当前 pool 写入的细胞数量和非零表达记录数量。

            Notes
            -----
            ``var`` 表只会在第一次 flush 时写入；后续 flush 只追加细胞和表达矩阵。
            """
            nonlocal global_cell_id
            nonlocal global_indptr_id
            nonlocal global_indptr_offset
            nonlocal global_data_id
            nonlocal var_written

            if len(cell_pool) == 0:
                return 0, 0

            t0 = time.time()

            # 1. 合并 cell_pool 中的多个 AnnData batch
            X_list = []
            obs_list = []

            total_pool_cells = 0
            total_pool_nnz = 0

            for ad in cell_pool:
                X = ad.X

                if sparse.issparse(X):
                    X = X.tocsr()
                else:
                    X = sparse.csr_matrix(X)

                X_list.append(X)
                obs_list.append(ad.obs.copy())

                total_pool_cells += ad.n_obs
                total_pool_nnz += X.nnz

            X_pool = sparse.vstack(X_list, format="csr")
            obs_pool = pd.concat(obs_list, axis=0)

            # var 所有文件一致，直接使用第一个 batch 的 var
            var_pool = cell_pool[0].var.copy()

            pool_adata = AnnData(
                X=X_pool,
                obs=obs_pool,
                var=var_pool,
            )


            # 2. cell_pool 内部整体随机；顺序模式关闭该步骤以保留原始细胞顺序
            if shuffle_cells and pool_adata.n_obs > 1:
                pool_perm = rng.permutation(pool_adata.n_obs)
                pool_adata = pool_adata[pool_perm].copy()

            t_shuffle = time.time() - t0
            t1 = time.time()

            # 3. 写 obs
            global_cell_id = _append_obs_rows(
                pool_adata,
                conn,
                start_cell_id=global_cell_id,
            )

            # 4. 写 var，只写一次
            if not var_written:
                _append_var(pool_adata, conn)
                var_written = True

            # 5. 写 X_CSRO，使用 Arrow 加速版
            (
                global_indptr_id,
                global_indptr_offset,
                global_data_id,
            ) = _append_x_hys(
                pool_adata,
                conn,
                base_cell_id=global_cell_id - pool_adata.n_obs,
                global_indptr_id=global_indptr_id,
                global_indptr_offset=global_indptr_offset,
                global_data_id=global_data_id,
            )

            t_write = time.time() - t1

            # 6. 清理
            for obj_name in [
                "X_list",
                "obs_list",
                "X_pool",
                "obs_pool",
                "var_pool",
                "pool_adata",
                "pool_perm",
            ]:
                try:
                    del locals()[obj_name]
                except Exception:
                    pass

            gc.collect()

            return total_pool_cells, total_pool_nnz

        # 主循环：每次读取 blocks_per_pool 个 block；随机模式下 block list 已被打乱
        processed_blocks = 0
        flush_counter = 0

        pbar = progress(total=total_blocks, desc="load_h5ad")

        # 事务：多个 flush 共用事务
        conn.execute("BEGIN TRANSACTION")

        try:
            block_cursor = 0

            while block_cursor < total_blocks:

                # 每次从全局 block list 中取 blocks_per_pool 个 block
                block_group = all_block_refs[block_cursor:block_cursor + blocks_per_pool]
                block_cursor += len(block_group)

                # 当前 flush 的 cell_pool
                cell_pool = []

                # 读取这一组 block
                for block_ref in block_group:

                    state = file_states[block_ref["file_idx"]]

                    block_start = block_ref["block_start"]
                    block_end = block_ref["block_end"]

                    # 从 h5ad 连续读取一个 batch block
                    t_read0 = time.time()

                    adata = state["adata_backed"][block_start:block_end].to_memory()

                    t_read = time.time() - t_read0

                    # 导入时统一将表达矩阵转换为 count 后写入
                    adata = _convert_x_to_count_inplace(
                        adata,
                        source_x_scale=state["source_x_scale"],
                    )

                    cell_pool.append(adata)

                    processed_blocks += 1
                    pbar.update(1)

                    # 不要每个 block 都打印，避免刷屏
                    if processed_blocks == 1 or processed_blocks % 50 == 0:
                        if sparse.issparse(adata.X):
                            block_nnz = adata.X.nnz
                        else:
                            block_nnz = np.count_nonzero(adata.X)

                # 这一组 block 读完后，按当前模式写入
                if len(cell_pool) > 0:

                    flush_counter += 1
                    _flush_cell_pool(cell_pool, flush_counter)

                    # 清空 pool
                    try:
                        for ad in cell_pool:
                            del ad
                        cell_pool.clear()
                    except Exception:
                        pass

                    gc.collect()

                    # 每 commit_every 次 flush 提交一次
                    if flush_counter % commit_every == 0:
                        conn.execute("COMMIT")
                        conn.execute("BEGIN TRANSACTION")

                    # 每 gc_every 次 flush 做一次 gc
                    if flush_counter % gc_every == 0:
                        gc.collect()

            pbar.close()

            # 最后提交
            conn.execute("COMMIT")

        except Exception:
            pbar.close()
            conn.execute("ROLLBACK")
            raise

        # 主键
        conn.execute("ALTER TABLE obs ADD PRIMARY KEY (atlas_cell_id)")
        conn.execute("ALTER TABLE var ADD PRIMARY KEY (atlas_gene_id)")

        # varm 是 gene 维度，多个文件 var 一致时，可以只导入第一个文件的 varm。
        _add_varm_from_h5ad(h5ad_paths[0], atlas)

        # 导入完成后整理 DuckDB 文件状态
        try:
            conn.execute("CHECKPOINT")
        except Exception:
            pass
        # 释放全局 block 索引池等大对象
        try:
            del all_block_refs
        except Exception:
            pass
        try:
            del first_backed
        except Exception:
            pass
        gc.collect()

        return {
            "files": file_num,
            "cells": global_cell_id,
            "genes": ref_n_genes,
            "nnz": global_data_id,
            "blocks": total_blocks,
            "flush": flush_counter,
        }


    finally:
        # 无论成功/失败，都关闭所有 backed 文件
        for s in file_states:
            try:
                s["adata_backed"].file.close()
            except Exception:
                pass
        # ：异常或成功退出时，尽量清空大对象引用
        try:
            file_states.clear()
        except Exception:
            pass
        try:
            all_block_refs.clear()
        except Exception:
            pass
        try:
            for ad in cell_pool:
                del ad
            cell_pool.clear()
        except Exception:
            pass
        try:
            gc.collect()
        except Exception:
            pass


def _load_h5ad_list_order(
    h5ad_paths: PathLike[str] | str | list[PathLike[str] | str],
    atlas: Atlas,
    cells_per_block: int | None = None,
):
    """按文件列表顺序导入多个 h5ad 文件到 Atlas 数据库。

    该函数复用多文件导入主体，但关闭全局 block 打乱和 cell pool 内部打乱，
    因此写入顺序与 ``h5ad_paths`` 的顺序以及每个文件内部的细胞顺序一致。
    表达矩阵会统一以 count 尺度写入。

    Parameters
    ----------
    h5ad_paths
        一个或多个 h5ad 文件路径。
    atlas
        Atlas 对象。函数会连接并写入对应 DuckDB 数据库。
    cells_per_block
        每个连续 cell block 的细胞数量。为 ``None`` 时根据总细胞数自动估算。

    Returns
    -------
    None
        结果直接写入 Atlas 数据库，不返回对象。

    Notes
    -----
    该函数通过调用 ``_load_h5ad_list_random`` 并设置
    ``shuffle_blocks=False``、``shuffle_cells=False`` 实现最小化重复逻辑。
    """
    return _load_h5ad_list_random(
        h5ad_paths=h5ad_paths,
        atlas=atlas,
        cells_per_block=cells_per_block,
        shuffle_blocks=False,
        shuffle_cells=False,
    )


def _load_h5ad_random(
    h5ad_path: PathLike[str] | str,
    atlas: Atlas,
    cells_per_block: int | None = None,
):
    """以 shuffle-window 方式随机导入单个 h5ad 文件。

    该内部函数用于单文件 ``load_type="random"`` 场景。它以 backed 模式打开
    h5ad 文件，将文件切成连续 block，随机打乱 block 顺序，并把多个 block
    合并成一个 shuffle window。

    每个 window 内的细胞会整体随机打乱，再批量写入 Atlas 数据库，从而在控制内存占用的同时获得随机导入顺序。
    表达矩阵会统一以 count 尺度写入。

    由于细胞顺序被重排，函数默认不导入 ``obsm``；``varm`` 与基因对齐，可以正常导入。

    Parameters
    ----------
    h5ad_path
        h5ad 文件路径。

    atlas
        Atlas 对象。函数会连接并写入对应 DuckDB 数据库。

    cells_per_block
        每批读取、写入或处理的细胞数量；较大值通常更快但占用更多内存。

    Returns
    -------
    None
        结果直接写入 Atlas 数据库，不返回对象。

    Notes
    -----
    随机导入会重排细胞顺序，因此该函数默认不导入 ``obsm``；``varm`` 与基因
    维度对齐，不受细胞重排影响，会正常导入。

    Examples
    --------
    内部随机导入单个文件::

        _load_h5ad_random(path, atlas, cells_per_block=1000)
    """

    h5ad_path = os.fspath(h5ad_path)

    t_start= time.time()

    h5ad_path = os.fspath(h5ad_path)

    commit_every = 5
    gc_every = 10

    # 连接数据库
    conn = atlas.connect("r+")
    atlas.connection = conn

    # 全局游标
    global_cell_id = 0
    global_indptr_id = 0
    global_indptr_offset = 0
    global_data_id = 0

    var_written = False

    # backed 打开
    adata_backed = sc.read_h5ad(h5ad_path, backed="r")
    n_cells = adata_backed.n_obs

    # 预读取 sample_n 个细胞，同时判断 X scale 和估算导入内存
    x_info = _inspect_x_from_backed(
        adata_backed,
        sample_n=5000,
    )

    source_x_scale = x_info["x_scale"]

    estimated_window_cells, blocks_per_pool = _estimate_window_cells_and_blocks_per_pool(
        memory_limit=_get_atlas_memory_limit(atlas),
        cells_per_block=cells_per_block,
        estimated_bytes_per_cell=x_info["estimated_bytes_per_cell"],
    )

    logger.info(f"[INFO] 文件中 X 判断为: {source_x_scale}")
    logger.info("[INFO] 表达矩阵统一写入为 count")

    if source_x_scale == "count":
        logger.info("[INFO] X 数据已是 count，直接写入。")
    else:
        logger.info("[INFO] X 数据将在写入前转换为 count")

    # 5 个 block 合并后再统一随机
    block_starts = np.arange(0, n_cells, cells_per_block, dtype=np.int64)
    np.random.shuffle(block_starts)

    # 动态建表
    _create_obs_table_from_adata(conn, adata_backed[:1])
    _create_var_table_from_adata(conn, adata_backed[:1])
    _create_hys_tables(conn)

    logger.info(f"[INFO] 数据集维度: {adata_backed.n_obs:,} × {adata_backed.n_vars:,}")

    # 窗口缓存: 每次攒够 blocks_per_pool 个 batch 再统一随机写入
    window_adatas = []
    window_batch_count = 0
    total_batch_counter = 0
    window_counter = 0

    conn.execute("BEGIN TRANSACTION")

    try:
        for block_i, block_start in enumerate(
            progress(
                block_starts,
                desc="load_h5ad",
            )
        ):
            block_end = min(int(block_start) + cells_per_block, n_cells)

            # 连续读取一个 batch block
            t0 = time.time()
            adata = adata_backed[int(block_start):block_end].to_memory()
            t_read = time.time() - t0

            # 当前 block 的 nnz
            if sparse.issparse(adata.X):
                block_nnz = adata.X.nnz
            else:
                block_nnz = np.count_nonzero(adata.X)

            # 不再单 batch 内部随机后立刻写入,而是先放入 window
            window_adatas.append(adata)
            window_batch_count += 1

            if (block_i + 1) % 20 == 0 or block_i == 0:
                logger.info(
                    f"\n[read block {block_i}] "
                    f"cells={adata.n_obs:,}, "
                    f"nnz={block_nnz:,}, "
                    f"read={t_read:.2f}s, "
                    f"window_batches={window_batch_count}/{blocks_per_pool}"
                )

            # window 满了，统一随机 + 统一写入
            if window_batch_count >= blocks_per_pool:
                t1 = time.time()

                (
                    global_cell_id,
                    global_indptr_id,
                    global_indptr_offset,
                    global_data_id,
                    var_written,
                    window_cells,
                    window_nnz,
                ) = _write_shuffle_window_to_duckdb(
                    window_adatas=window_adatas,
                    conn=conn,
                    global_cell_id=global_cell_id,
                    global_indptr_id=global_indptr_id,
                    global_indptr_offset=global_indptr_offset,
                    global_data_id=global_data_id,
                    var_written=var_written,
                    source_x_scale=source_x_scale,
                )

                t_write = time.time() - t1

                total_batch_counter += window_batch_count
                window_counter += 1

                # 清空 window
                for x in window_adatas:
                    del x
                window_adatas.clear()
                window_batch_count = 0

                # 每 commit_every 个 batch 提交一次,等价于 blocks_per_pool=5 时，每 2 个 window commit
                if window_counter % commit_every == 0:
                    conn.execute("COMMIT")
                    conn.execute("BEGIN TRANSACTION")
                    logger.info(
                        f"[COMMIT] processed_windows={window_counter:,}, "
                        f"processed_batches={total_batch_counter:,}"
                    )

                # 每 gc_every 个 batch gc 一次
                if window_counter % gc_every == 0:
                    gc.collect()
                    logger.info(
                        f"[GC] processed_windows={window_counter:,}, "
                        f"processed_batches={total_batch_counter:,}"
                    )

        # 处理最后不足 5 个 batch 的剩余 window
        if window_batch_count > 0:
            t1 = time.time()

            (
                global_cell_id,
                global_indptr_id,
                global_indptr_offset,
                global_data_id,
                var_written,
                window_cells,
                window_nnz,
            ) = _write_shuffle_window_to_duckdb(
                window_adatas=window_adatas,
                conn=conn,
                global_cell_id=global_cell_id,
                global_indptr_id=global_indptr_id,
                global_indptr_offset=global_indptr_offset,
                global_data_id=global_data_id,
                var_written=var_written,
                source_x_scale=source_x_scale,
            )

            t_write = time.time() - t1

            total_batch_counter += window_batch_count
            window_counter += 1

            for x in window_adatas:
                del x
            window_adatas.clear()
            window_batch_count = 0
            gc.collect()

        # 最后提交
        conn.execute("COMMIT")

    except Exception:
        conn.execute("ROLLBACK")
        # 异常时也清理 window_adatas
        try:
            for x in window_adatas:
                del x
            window_adatas.clear()
        except Exception:
            pass
        # 异常时也关闭 h5ad backed 文件
        try:
            adata_backed.file.close()
        except Exception:
            pass
        gc.collect()
        raise

    # 主键
    conn.execute("ALTER TABLE obs ADD PRIMARY KEY (atlas_cell_id)")
    conn.execute("ALTER TABLE var ADD PRIMARY KEY (atlas_gene_id)")

    # 随机导入时不建议导入 obsm
    # 因为 obs / X 已经随机重排，obsm 直接按原 h5ad 顺序导入会错位; varm 是 gene 维度，一般没问题
    # _add_obsm_from_h5ad(h5ad_path, atlas)
    _add_varm_from_h5ad(h5ad_path, atlas)

    # 导入完成后整理 DuckDB 文件状态
    try:
        conn.execute("CHECKPOINT")
    except Exception:
        pass

    try:
        adata_backed.file.close()
    except Exception:
        pass

    # 导入结束后主动释放 Python 对象
    try:
        del window_adatas
    except Exception:
        pass

    try:
        del adata_backed
    except Exception:
        pass

    gc.collect()

    t_end = time.time()


def _load_h5ad_order(
    h5ad_path: PathLike[str] | str,
    atlas: Atlas,
    cells_per_block: int | None = None,
):

    """按原始细胞顺序导入单个 h5ad 文件。

    该内部函数用于单文件 ``load_type="order"`` 场景。它以 backed 模式顺序读取
    h5ad，把数据按 mega-batch 载入内存，再拆成较小 batch 写入 Atlas。
    表达矩阵会统一转换并保存为 count 尺度。

    与随机导入不同，它不会打乱细胞顺序，因此可以安全导入 ``obsm`` 和 ``varm``，适合需要保留原始 AnnData 行顺序的场景。

    Parameters
    ----------
    h5ad_path
        h5ad 文件路径。

    atlas
        Atlas 对象。函数会连接并写入对应 DuckDB 数据库。

    cells_per_block
        每批读取、写入或处理的细胞数量；较大值通常更快但占用更多内存。

    Returns
    -------
    None
        结果直接写入 Atlas 数据库，不返回对象。

    Notes
    -----
    ``cells_per_block`` 会参与内存窗口估算。顺序导入中估算出的
    ``window_cells`` 会作为 ``mega_batch_size`` 使用。

    Examples
    --------
    内部顺序导入单个文件::

        _load_h5ad_order(path, atlas, cells_per_block=1000)
    """
    commit_every = 5
    gc_every = 5

    conn = atlas.connect("r+")
    atlas.connection = conn

    global_cell_id = 0
    global_indptr_id = 0
    global_indptr_offset = 0
    global_data_id = 0

    var_written = False

    adata_backed = sc.read_h5ad(h5ad_path, backed="r")
    n_cells = adata_backed.n_obs

    # 简单判断 h5ad.X 底层格式
    x_format = _print_h5ad_x_format(h5ad_path)

    # 预读取 1000 个细胞，判断文件里的 X 是 count 还是 log
    x_info = _inspect_x_from_backed(
        adata_backed,
        sample_n=5000,
    )

    source_x_scale = x_info["x_scale"]

    estimated_window_cells, blocks_per_pool = _estimate_window_cells_and_blocks_per_pool(
        memory_limit=_get_atlas_memory_limit(atlas),
        cells_per_block=cells_per_block,
        estimated_bytes_per_cell=x_info["estimated_bytes_per_cell"],
    )

    # order 模式下，window_cells 就是 mega-batch 大小
    mega_batch_size = estimated_window_cells

    logger.info(f"[INFO] 文件中 X 判断为: {source_x_scale}")
    logger.info("[INFO] 表达矩阵统一写入为 count")

    if source_x_scale == "count":
        logger.info("[INFO] X 数据已是 count，直接写入。")
    else:
        logger.info("[INFO] X 数据将在 mega-batch 读入后转换为 count")

    _create_obs_table_from_adata(conn, adata_backed[:1])
    _create_var_table_from_adata(conn, adata_backed[:1])
    _create_hys_tables(conn)

    logger.info(f"[INFO] 数据集维度: {adata_backed.n_obs:,} × {adata_backed.n_vars:,}")

    mini_batch_counter = 0

    # 事务放在大循环外
    conn.execute("BEGIN TRANSACTION")

    try:
        for mega_i, mega_start in enumerate(
            progress(
                range(0, n_cells, mega_batch_size),
                desc="load_h5ad",
            )
        ):
            mega_end = min(mega_start + mega_batch_size, n_cells)

            t0 = time.time()

            # 真正触发磁盘读取
            mega = adata_backed[mega_start:mega_end].to_memory()

            t_read = time.time() - t0

            # mega-batch 读入后，统一转换为 count
            mega = _convert_x_to_count_inplace(
                mega,
                source_x_scale=source_x_scale,
            )

            # 统计当前 mega 的 nnz
            if sparse.issparse(mega.X):
                mega_nnz = mega.X.nnz
            else:
                mega_nnz = np.count_nonzero(mega.X)

            # 按 cells_per_block 分批导入
            for start in range(0, mega.n_obs, cells_per_block):
                end = min(start + cells_per_block, mega.n_obs)
                adata = mega[start:end]

                t1 = time.time()

                # ---------------- batch 导入 obs ----------------
                global_cell_id = _append_obs_rows(
                    adata,
                    conn,
                    start_cell_id=global_cell_id,
                )

                # ---------------- 导入 var（一次） ----------------
                if not var_written:
                    _append_var(adata, conn)
                    var_written = True

                # ---------------- batch 导入 X（CSRO） ----------------
                (
                    global_indptr_id,
                    global_indptr_offset,
                    global_data_id,
                ) = _append_x_hys(
                    adata,
                    conn,
                    base_cell_id=global_cell_id - adata.n_obs,
                    global_indptr_id=global_indptr_id,
                    global_indptr_offset=global_indptr_offset,
                    global_data_id=global_data_id,
                )

                t_write = time.time() - t1

                mini_batch_counter += 1

                # 每 commit_every 个 mini-batch 提交一次
                if mini_batch_counter % commit_every == 0:
                    conn.execute("COMMIT")
                    conn.execute("BEGIN TRANSACTION")

                del adata

            del mega

            if (mega_i + 1) % gc_every == 0:
                gc.collect()

        # 最后提交
        conn.execute("COMMIT")

    except Exception:
        conn.execute("ROLLBACK")
        raise

    # 主键：必须在数据写完之后
    conn.execute("ALTER TABLE obs ADD PRIMARY KEY (atlas_cell_id)")
    conn.execute("ALTER TABLE var ADD PRIMARY KEY (atlas_gene_id)")

    # 顺序导入没有打乱 cell，所以 obsm 可以正常导入
    _add_obsm_from_h5ad(
        h5ad_path,
        atlas,
        cells_per_block=cells_per_block,
    )
    _add_varm_from_h5ad(h5ad_path, atlas)

    try:
        adata_backed.file.close()
    except Exception:
        pass


# 将一个 shuffle window 合并并写入 DuckDB
def _write_shuffle_window_to_duckdb(
    window_adatas: list[AnnData],
    conn: DuckDBPyConnection,
    global_cell_id: int,
    global_indptr_id: int,
    global_indptr_offset: int,
    global_data_id: int,
    var_written: bool,
    source_x_scale: XScale,
):
    """将一个 shuffle window 写入 Atlas 数据库。

    该内部函数用于随机导入流程。它接收一个 window 中收集到的多个 AnnData
    block，先合并为一个 AnnData，再在 window 内随机打乱细胞顺序，然后依次
    写入 ``obs``、``var``、``X_HyS_indptr`` 和 ``X_HyS_data``。

    写入表达矩阵前，函数会根据 ``source_x_scale`` 把 ``adata.X`` 统一转换为
    count 尺度，最终写入 ``X_HyS_data.data_count`` 字段。

    Parameters
    ----------
    window_adatas
        一个 shuffle window 中收集到的 AnnData block 列表。
    conn
        DuckDB 数据库连接。
    global_cell_id
        下一个待写入的全局 ``atlas_cell_id``。
    global_indptr_id
        下一个待写入的 indptr 行 ID。
    global_indptr_offset
        当前已经累计写入的非零值数量，用于重定位 indptr。
    global_data_id
        下一个待写入的 ``X_HyS_data.id``。
    var_written
        是否已经写入 ``var`` 表。
    source_x_scale
        输入表达矩阵当前的尺度，通常为 ``"count"`` 或 ``"log"``。

    Returns
    -------
    tuple
        返回更新后的全局游标和 window 统计信息，顺序为：

        ``global_cell_id``、``global_indptr_id``、``global_indptr_offset``、
        ``global_data_id``、``var_written``、``window_cells``、``window_nnz``。

    Notes
    -----
    ``var`` 表只在第一个 window 写入一次；后续 window 只追加 ``obs`` 和表达
    矩阵相关表。
    """

    # 1. 合并 window 内的多个 batch
    adata_window = sc.concat(
        window_adatas,
        axis=0,
        join="outer",
        merge="first",
        index_unique=None,
    )

    # 2. window 内所有 cell 统一随机
    if adata_window.n_obs > 1:
        perm = np.random.permutation(adata_window.n_obs)
        adata_window = adata_window[perm].copy()

    # window 统计
    window_cells = adata_window.n_obs

    if sparse.issparse(adata_window.X):
        window_nnz = adata_window.X.nnz
    else:
        window_nnz = np.count_nonzero(adata_window.X)

    # 3. 写入 obs
    global_cell_id = _append_obs_rows(
        adata_window,
        conn,
        start_cell_id=global_cell_id,
    )

    # 4. 写入 var，只写一次
    if not var_written:
        _append_var(adata_window, conn)
        var_written = True

    # 导入时统一将表达矩阵转换为 count 后写入
    adata_window = _convert_x_to_count_inplace(
        adata_window,
        source_x_scale=source_x_scale,
    )

    # 5. 写入 X_HyS_data / X_HyS_indptr
    (
        global_indptr_id,
        global_indptr_offset,
        global_data_id,
    ) = _append_x_hys(
        adata_window,
        conn,
        base_cell_id=global_cell_id - adata_window.n_obs,
        global_indptr_id=global_indptr_id,
        global_indptr_offset=global_indptr_offset,
        global_data_id=global_data_id,
    )

    del adata_window

    gc.collect()

    return (
        global_cell_id,
        global_indptr_id,
        global_indptr_offset,
        global_data_id,
        var_written,
        window_cells,
        window_nnz,
    )


def _get_atlas_memory_limit(atlas: Atlas) -> str | int | None:
    """获取 Atlas 对象中保存的内存限制参数。

    该函数用于从 Atlas 对象中读取内存限制值，例如 ``"32GB"``、``"8G"``、
    ``16`` 或 ``None``。

    之所以单独写成 helper，是为了让导入模块尽量兼容不同版本的 Atlas 类：
    1. 旧版本可能使用 ``__memory_limit``；
    2. 新版本可能使用 ``__db_memory_limit``；
    3. 有些版本可能提供公开属性 ``memory_limit``；
    4. 当前推荐版本使用公开属性 ``db_memory_limit``。

    Parameters
    ----------
    atlas
        Atlas 对象。通常由 ``load_h5ad(..., atlas=atlas)`` 或
        ``atlas.load_h5ad(...)`` 传入。

    Returns
    -------
    memory_limit
        Atlas 中保存的内存限制参数。

        可能返回：
        - ``str``，例如 ``"32GB"``、``"8G"``、``"1024MB"``；
        - ``int``，例如 ``32``，表示 32GB；
        - ``None``，表示没有设置可用的内存限制。

    Notes
    -----
    该函数只负责“读取”内存限制，不负责把内存限制应用到 DuckDB。
    真正执行 DuckDB ``SET memory_limit`` 的逻辑应该在 Atlas 类的
    ``_apply_memory_limit()`` 中完成。

    这里返回的值主要用于后续估算 Python 导入窗口大小，例如：

    ``memory_limit -> memory_limit_bytes -> window_cells -> blocks_per_pool``。
    """

    # 1. 优先尝试读取私有属性。
    #    这样可以兼容旧版本 Atlas 类中可能存在的字段。
    for attr_name in ("_Atlas__memory_limit", "_Atlas__db_memory_limit"):
        try:
            value = getattr(atlas, attr_name)
        except Exception:
            value = None

        if value not in (None, "", "None"):
            return value

    # 2. 再尝试读取公开属性。
    #    当前推荐使用 atlas.db_memory_limit。
    for attr_name in ("memory_limit", "db_memory_limit"):
        try:
            value = getattr(atlas, attr_name)
        except RecursionError:
            # 防止旧 property 写错导致递归报错。
            value = None
        except Exception:
            value = None

        if value not in (None, "", "None"):
            return value

    # 3. 如果都没有读到，则认为没有设置内存限制。
    return None


def _parse_memory_limit_to_bytes(memory_limit: str | int | None) -> int | None:
    """将内存限制参数转换为字节数。

    该函数用于把 Atlas 中保存的 ``db_memory_limit`` 转成字节数，
    方便后续计算导入窗口大小。

    支持的输入形式包括：

    - ``None``：表示没有内存限制，返回 ``None``；
    - ``int``：按 GB 解释，例如 ``32`` 表示 ``32GB``；
    - 无单位数字字符串：按 GB 解释，例如 ``"32"`` 表示 ``32GB``；
    - 带单位字符串：例如 ``"32GB"``、``"32G"``、``"1024MB"``、``"512M"``。

    Parameters
    ----------
    memory_limit
        Atlas 中保存的内存限制值。

        常见形式包括：
        - ``"32GB"``
        - ``"16G"``
        - ``"8000MB"``
        - ``32``
        - ``None``

    Returns
    -------
    memory_limit_bytes
        转换后的字节数。

        如果 ``memory_limit`` 为 ``None``、空字符串或 ``"None"``，
        则返回 ``None``。

    Notes
    -----
    注意：这里的 ``int`` 不按“字节数”解释，而是按“GB”解释。
    这是为了和 Atlas 类中 ``db_memory_limit`` 的设计保持一致：

    ``db_memory_limit=32`` 等价于 ``db_memory_limit="32GB"``。
    """

    if memory_limit is None:
        return None

    # int 按 GB 解释，例如 32 -> 32GB。
    if isinstance(memory_limit, int):
        if memory_limit <= 0:
            raise ValueError("memory_limit 必须 > 0")
        return int(memory_limit * 1024 ** 3)

    if not isinstance(memory_limit, str):
        raise TypeError(
            f"memory_limit 必须是 str、int 或 None，当前类型为: {type(memory_limit)}"
        )

    value_text = memory_limit.strip().upper().replace(" ", "")

    if value_text in {"", "NONE"}:
        return None

    units = {
        "KB": 1024,
        "K": 1024,
        "MB": 1024 ** 2,
        "M": 1024 ** 2,
        "GB": 1024 ** 3,
        "G": 1024 ** 3,
    }

    # 处理带单位的字符串，例如 "32GB"、"16G"、"1024MB"。
    for unit, factor in units.items():
        if value_text.endswith(unit):
            value = float(value_text[: -len(unit)])
            if value <= 0:
                raise ValueError("memory_limit 必须 > 0")
            return int(value * factor)

    # 没有单位时，也按 GB 解释。
    value = float(value_text)
    if value <= 0:
        raise ValueError("memory_limit 必须 > 0")
    return int(value * 1024 ** 3)


def _inspect_x_from_backed(
    adata_backed: AnnData,
    sample_n: int = 5000,
    overhead_factor: float = 4.0,
) -> dict[str, Any]:
    """预读取部分细胞，同时判断 X 的尺度并估算每个细胞的内存占用。

    该函数会从 backed 模式打开的 AnnData 中读取前 ``sample_n`` 个细胞，
    然后基于这部分样本完成两个任务：

    1. 判断 ``adata.X`` 当前是 count scale 还是 log scale；
    2. 估算每个 cell 在导入过程中大约会占用多少内存。

    这样做的好处是：
    - 不需要额外读取两次 sample；
    - 判断 X scale 和估算导入窗口可以共用同一份 sample；
    - 后续可以根据 ``estimated_bytes_per_cell`` 和 ``db_memory_limit``
      自动计算 ``window_cells`` 和 ``blocks_per_pool``。

    Parameters
    ----------
    adata_backed
        以 backed 模式打开的 AnnData 对象。

        通常来自：

        ``adata_backed = sc.read_h5ad(h5ad_path, backed="r")``

    sample_n
        预读取的细胞数量。

        实际读取数量为：

        ``min(sample_n, adata_backed.n_obs)``

        默认读取前 5000 个细胞。

    overhead_factor
        内存放大系数。

        因为导入过程中不只保存原始 ``X``，还会产生一些临时对象，例如：

        - ``window_adatas``
        - ``sc.concat(...)`` 或 ``sparse.vstack(...)``
        - ``adata_window.copy()``
        - ``obs_df``
        - Arrow table
        - DuckDB register 临时对象

        所以不能只按 ``X`` 本身的内存估算。这里默认乘以 ``4.0``，
        用于更保守地估算导入时每个 cell 的实际内存占用。

    Returns
    -------
    info
        一个字典，包含以下字段：

        - ``x_scale``：判断得到的 X 尺度，通常为 ``"count"`` 或 ``"log"``；
        - ``sample_cells``：实际抽样的细胞数量；
        - ``nonzero_n``：sample 中非零表达值数量；
        - ``max``：sample 中非零表达值最大值；
        - ``q95``：sample 中非零表达值的 95 分位数；
        - ``frac_le_10``：sample 中非零表达值小于等于 10 的比例；
        - ``x_bytes``：sample 中 X 按目标导入结构估算的基础字节数；
        - ``x_bytes_per_cell``：单个 cell 的基础 X 内存估算；
        - ``estimated_bytes_per_cell``：乘以 overhead_factor 后的单个 cell 内存估算。

    Notes
    -----
    稀疏矩阵的内存估算按 Atlas 写入结构来估计：

    - CSR 非零值数组 ``X.data`` 按 ``float32`` 估算；
    - ``indices`` 按 ``uint16`` 估算，对应 ``atlas_gene_id`` / ``USMALLINT``；
    - ``indptr`` 按 ``int64`` 估算。

    这个估算不追求精确到 Python 对象级别，而是用于得到一个足够稳的
    ``window_cells`` 和 ``blocks_per_pool``。
    """

    n = min(sample_n, adata_backed.n_obs)

    if n <= 0:
        raise ValueError("[ERROR] h5ad 中没有细胞，无法检测 X。")

    # 预读取前 n 个细胞。
    adata_sample = adata_backed[:n].to_memory()
    X = adata_sample.X

    if sparse.issparse(X):
        # 稀疏矩阵统一转成 CSR，方便统计 data / indices / indptr。
        X = X.tocsr()

        # 非零表达值，用于判断 count / log。
        values = np.asarray(X.data, dtype=np.float32)

        # 按 Atlas 最终写入结构估算 X 的基础内存：
        # data    -> float32
        # indices -> uint16
        # indptr  -> int64
        x_bytes = (
            X.data.size * np.dtype(np.float32).itemsize
            + X.indices.size * np.dtype(np.uint16).itemsize
            + X.indptr.size * np.dtype(np.int64).itemsize
        )
    else:
        # dense 矩阵按 float32 估算。
        X_arr = np.asarray(X)

        # 只取非零值用于判断 count / log。
        values = X_arr[X_arr != 0].astype(np.float32, copy=False)

        x_bytes = X_arr.size * np.dtype(np.float32).itemsize

    if values.size == 0:
        del adata_sample
        gc.collect()
        raise ValueError(
            "[ERROR] 预读取的细胞中没有非零表达值，无法判断 X 是 count 还是 log。"
        )

    vmax = float(np.max(values))
    q95 = float(np.percentile(values, 95))
    frac_le_10 = float(np.mean(values <= 10))

    # 经验判断：
    # count 数据中通常会出现较大的整数计数；
    # log 数据通常绝大多数非零值都不会太大。
    if vmax > 50 or q95 > 10:
        x_scale: XScale = "count"
    else:
        x_scale: XScale = "log"

    # 估算单个 cell 的基础 X 内存。
    x_bytes_per_cell = float(x_bytes / n)

    # 加上导入过程中各种临时对象的经验放大系数。
    estimated_bytes_per_cell = max(1.0, x_bytes_per_cell * overhead_factor)

    logger.info("[INFO] X scale / memory 预检测结果:")
    logger.info(f"  - sample_cells = {n:,}")
    logger.info(f"  - nonzero_n    = {values.size:,}")
    logger.info(f"  - max          = {vmax:.4f}")
    logger.info(f"  - q95          = {q95:.4f}")
    logger.info(f"  - frac <= 10   = {frac_le_10:.4f}")
    logger.info(f"  - x_scale      = {x_scale}")
    logger.info(f"  - x_bytes_per_cell = {x_bytes_per_cell:.2f}")
    logger.info(f"  - estimated_bytes_per_cell = {estimated_bytes_per_cell:.2f}")

    del adata_sample
    gc.collect()

    return {
        "x_scale": x_scale,
        "sample_cells": n,
        "nonzero_n": int(values.size),
        "max": vmax,
        "q95": q95,
        "frac_le_10": frac_le_10,
        "x_bytes": int(x_bytes),
        "x_bytes_per_cell": x_bytes_per_cell,
        "estimated_bytes_per_cell": estimated_bytes_per_cell,
    }


def _estimate_window_cells_and_blocks_per_pool(
    *,
    memory_limit: str | int | None,
    cells_per_block: int,
    estimated_bytes_per_cell: float,
    memory_fraction: float = 0.25,      #
    default_blocks_per_pool: int = 20,  #
    max_blocks_per_pool: int = 100,     # min = 5
) -> tuple[int, int]:
    """根据内存限制估算导入窗口大小和 blocks_per_pool。

    该函数根据 Atlas 的 ``db_memory_limit``、用户输入的 ``cells_per_block``，
    以及 sample 估算得到的 ``estimated_bytes_per_cell``，自动计算：

    1. ``window_cells``：一次导入窗口中最多包含多少个细胞；
    2. ``blocks_per_pool``：一个窗口中包含多少个 block。

    它们之间的关系是：

    ``window_cells = cells_per_block * blocks_per_pool``

    Parameters
    ----------
    memory_limit
        Atlas 中保存的内存限制。

        常见形式包括：
        - ``"32GB"``
        - ``"16G"``
        - ``32``
        - ``None``

    cells_per_block
        用户输入的每个连续读取 block 中包含的 cell 数量。

        例如：

        ``cells_per_block = 500``

    estimated_bytes_per_cell
        由 ``_inspect_x_from_backed()`` 估算得到的单个 cell 内存占用。

        该值已经乘过 ``overhead_factor``，因此比单纯的 X 内存更保守。

    memory_fraction
        用于估算导入窗口的内存比例。

        例如 ``memory_limit="32GB"`` 且 ``memory_fraction=0.25`` 时，
        表示最多使用约 8GB 来估算导入窗口。

        这里不建议使用 1.0，因为 Python、NumPy、pandas、AnnData、
        Arrow 和 DuckDB 都会额外占用内存。

    default_blocks_per_pool
        当 ``memory_limit`` 为 ``None`` 时使用的默认窗口 block 数量。

        默认值为 20，等价于旧版本的默认行为：

        ``window_cells = cells_per_block * 20``

    max_blocks_per_pool
        ``blocks_per_pool`` 的最大上限。

        该参数用于避免内存估算过于乐观，导致一次窗口包含过多 block，
        从而造成 ``sc.concat``、``sparse.vstack`` 或 Arrow 写入时内存峰值过高。

    Returns
    -------
    window_cells
        一个导入窗口中实际使用的 cell 数量。

        该值会被重新对齐为：

        ``cells_per_block * blocks_per_pool``

    blocks_per_pool
        一个导入窗口中包含的 block 数量。

        计算方式近似为：

        ``blocks_per_pool = window_cells // cells_per_block``

    Notes
    -----
    该函数只负责估算导入窗口大小，不直接读取 h5ad，也不直接写入数据库。

    在不同导入模式中的含义：

    - ``random`` 模式：表示一个 shuffle window 中包含多少个 block；
    - 多文件 ``random`` / ``order`` 模式：表示一次从全局 block pool 中读取多少个 block；
    - ``order`` 模式：``window_cells`` 直接作为 ``mega_batch_size`` 使用。
    """

    if cells_per_block <= 0:
        raise ValueError("cells_per_block 必须 > 0")

    if estimated_bytes_per_cell <= 0:
        raise ValueError("estimated_bytes_per_cell 必须 > 0")

    memory_limit_bytes = _parse_memory_limit_to_bytes(memory_limit)

    # 没有设置内存限制时，保持旧版本默认行为。
    if memory_limit_bytes is None:
        blocks_per_pool = default_blocks_per_pool
        window_cells = cells_per_block * blocks_per_pool
        return window_cells, blocks_per_pool

    if not 0 < memory_fraction <= 1:
        raise ValueError("memory_fraction 必须在 (0, 1] 范围内")

    # 只使用 memory_limit 的一部分估算导入窗口，避免内存峰值过高。
    usable_memory_bytes = memory_limit_bytes * memory_fraction

    # 根据单个 cell 的估算内存，反推一个窗口中最多能容纳多少个 cell。
    window_cells = int(usable_memory_bytes // estimated_bytes_per_cell)

    # 至少保证一个 block。
    window_cells = max(cells_per_block, window_cells)

    # 根据 window_cells 和 cells_per_block 计算 blocks_per_pool。
    blocks_per_pool = window_cells // cells_per_block
    blocks_per_pool = max(1, blocks_per_pool)

    # 设置上限，避免 window 过大。
    blocks_per_pool = min(blocks_per_pool, max_blocks_per_pool)

    # 重新对齐 window_cells，保证它一定等于 cells_per_block * blocks_per_pool。
    window_cells = cells_per_block * blocks_per_pool

    logger.info("[INFO] 自动估算 h5ad 导入窗口参数:")
    logger.info(f"  - memory_limit = {memory_limit}")
    logger.info(f"  - memory_limit_bytes = {memory_limit_bytes:,}")
    logger.info(f"  - memory_fraction = {memory_fraction}")
    logger.info(f"  - cells_per_block = {cells_per_block:,}")
    logger.info(f"  - estimated_bytes_per_cell = {estimated_bytes_per_cell:.2f}")
    logger.info(f"  - window_cells = {window_cells:,}")
    logger.info(f"  - blocks_per_pool = {blocks_per_pool:,}")

    return window_cells, blocks_per_pool


def _detect_x_scale_from_backed(
    adata_backed: AnnData,
    sample_n: int = 5000,
) -> XScale:
    """检测输入表达矩阵的存储尺度。

    该内部函数是 ``_inspect_x_from_backed`` 的轻量包装，只返回 X 的尺度判断
    结果。它会从 backed AnnData 中预读取部分细胞，根据非零表达值的最大值、
    分位数和低值比例判断输入更像 count 数据还是 log 数据。

    Parameters
    ----------
    adata_backed
        以 backed 模式打开的 AnnData 对象。
    sample_n
        用于预检测的细胞数量。实际读取数量为
        ``min(sample_n, adata_backed.n_obs)``。

    Returns
    -------
    x_scale
        检测得到的表达矩阵尺度，通常为 ``"count"`` 或 ``"log"``。

    Notes
    -----
    该函数不修改 AnnData，也不写入数据库；真正的转换由
    ``_convert_x_to_count_inplace`` 完成。
    """

    x_info = _inspect_x_from_backed(
        adata_backed,
        sample_n=sample_n,
    )

    return x_info["x_scale"]


def _normalize_cells_per_block(cells_per_block, n_cells):
    """规范化导入时的 cell block 大小。

    该内部函数用于 ``load_h5ad`` 及其底层导入流程。用户没有显式指定
    ``cells_per_block`` 时，函数会根据总细胞数估算默认值，并限制在
    ``512`` 到 ``4096`` 之间。

    Parameters
    ----------
    cells_per_block
        用户传入的 block 大小，或 ``None``。
    n_cells
        当前导入数据集的总细胞数。

    Returns
    -------
    int
        规范化后的正整数 block 大小。

    Notes
    -----
    该函数只负责生成 block 大小，不负责检查内存窗口；内存窗口由
    ``_estimate_window_cells_and_blocks_per_pool`` 进一步估算。
    """

    if cells_per_block is None:
        cells_per_block = int(0.001 * n_cells)
        cells_per_block = max(512, min(cells_per_block, 4096))

    cells_per_block = int(cells_per_block)
    logger.info(f"cells_per_block = {cells_per_block}")
    return cells_per_block

# 根据输入 X 尺度原地转换 adata.X；导入时统一写入 count，log 转换底数为 e
def _convert_x_to_count_inplace(
    adata: AnnData,
    source_x_scale: XScale,
):
    """将表达矩阵转换为 count 尺度。

    该内部函数用于导入前统一表达矩阵尺度。Atlas 导入流程默认把表达矩阵存储为
    count，因此当输入 ``adata.X`` 被判断为 log 尺度时，函数会执行
    ``expm1`` 转换并把负值截断为 0 , 默认以 e 为转换底数 。

    对稀疏矩阵，函数只转换非零值数组 ``X.data``，不破坏稀疏结构；对 dense
    矩阵，函数会把整个矩阵转换为 ``float32`` 并原地处理。

    Parameters
    ----------
    adata
        需要转换的 AnnData 对象。函数会更新其中的 ``adata.X``。
    source_x_scale
        输入表达矩阵当前的尺度。支持 ``"count"`` 和 ``"log"``。

    Returns
    -------
    AnnData
        转换后的同一个 AnnData 对象。

    Notes
    -----
    当 ``source_x_scale="count"`` 时，函数直接返回原对象，不做复制。
    """

    if source_x_scale == "count":
        return adata

    X = adata.X

    # 1. sparse matrix：只改非零值 X.data，不破坏稀疏结构
    if sparse.issparse(X):
        if not sparse.isspmatrix_csr(X):
            X = X.tocsr()

        X.data = X.data.astype(np.float32, copy=False)

        if source_x_scale == "log":
            np.expm1(X.data, out=X.data)
            X.data[X.data < 0] = 0

        else:
            raise ValueError(f"不支持的 X 尺度: {source_x_scale}")

        adata.X = X
        return adata

    # 2. dense matrix：直接对整个矩阵转换
    X = np.asarray(X, dtype=np.float32)

    if source_x_scale == "log":
        np.expm1(X, out=X)
        np.maximum(X, 0, out=X)

    else:
        raise ValueError(f"不支持的 X 尺度: {source_x_scale}")

    adata.X = X
    return adata


# 简单判断 h5ad.X 的底层稀疏格式
def _print_h5ad_x_format(h5ad_path: PathLike[str] | str):
    """检测并打印 h5ad 文件中 ``X`` 的底层存储格式。

    该内部函数直接读取 h5ad 的 HDF5 结构，判断 ``X`` 是 dense dataset 还是
    sparse matrix group，并根据 ``encoding-type`` 返回 ``csr``、``csc``、
    ``coo`` 或其他格式名称。它只用于导入前日志和诊断，不会读取完整表达矩阵。

    Parameters
    ----------
    h5ad_path
        h5ad 文件路径。

    Returns
    -------
    str or None
        检测到的 ``X`` 存储格式。文件中没有 ``X`` 时返回 ``None``。

    Notes
    -----
    该函数不判断表达值是 count 还是 log；表达尺度判断由
    ``_inspect_x_from_backed`` 完成。
    """

    h5ad_path = os.fspath(h5ad_path)

    with h5py.File(h5ad_path, "r") as f:
        if "X" not in f:
            logger.info("[INFO] h5ad.X format = None（文件中没有 X）")
            return None

        X = f["X"]

        # dense matrix
        if isinstance(X, h5py.Dataset):
            logger.info("[INFO] h5ad.X format = dense")
            return "dense"

        # sparse matrix group
        if isinstance(X, h5py.Group):
            encoding_type = X.attrs.get("encoding-type", "unknown")

            if isinstance(encoding_type, bytes):
                encoding_type = encoding_type.decode("utf-8")

            if encoding_type == "csr_matrix":
                logger.info("[INFO] h5ad.X format = CSR")
                return "csr"

            elif encoding_type == "csc_matrix":
                logger.info("[INFO] h5ad.X format = CSC")
                return "csc"

            elif encoding_type == "coo_matrix":
                logger.info("[INFO] h5ad.X format = COO")
                return "coo"

            else:
                logger.info(f"[INFO] h5ad.X format = unknown ({encoding_type})")
                return encoding_type

        logger.info("[INFO] h5ad.X format = unknown")
        return "unknown"


# 推断数据类型
def _infer_duckdb_type_from_series(series: pd.Series) -> str:

    """根据 pandas Series 推断 DuckDB 字段类型。

    该内部函数用于从 AnnData 的 ``obs`` 或 ``var`` 元数据创建 DuckDB 表结构。
    它会把 pandas 整数、浮点和布尔类型分别映射到 DuckDB 的 ``BIGINT``、
    ``DOUBLE`` 和 ``BOOLEAN``；其他类型统一按 ``VARCHAR`` 处理。

    Parameters
    ----------
    series
        需要推断 DuckDB 类型的 pandas Series。

    Returns
    -------
    str
        DuckDB 字段类型名称。

    Notes
    -----
    字符串、分类变量、object 类型和无法精确映射的类型都会写成 ``VARCHAR``。
    """
    if pd.api.types.is_integer_dtype(series):
        return "BIGINT"
    if pd.api.types.is_float_dtype(series):
        return "DOUBLE"
    if pd.api.types.is_bool_dtype(series):
        return "BOOLEAN"
    return "VARCHAR"


# 建立 obs表
def _create_obs_table_from_adata(conn: DuckDBPyConnection, adata: AnnData):
    """根据 AnnData 的 obs 元数据创建 Atlas ``obs`` 表。

    该内部函数读取 ``adata.obs`` 的列名和 pandas dtype，推断对应的 DuckDB 字段类型，并创建包含
    ``atlas_cell_id`` 与 ``atlas_cell_name`` 的标准 ``obs`` 表。

    与 Scanpy 中把细胞注释保存在 ``adata.obs`` 的约定类似，Atlas 会把细胞级元数据持久化到数据库表中，
    供后续 QC、过滤、聚类、差异分析和绘图函数复用。

    Parameters
    ----------
    conn
        DuckDB 连接对象。

    adata
        输入 AnnData 对象。

        要求包含 ``obs`` 和 ``obs_names``；来源数据中若已经存在 Atlas 系统字段，会在建表时跳过。

    Notes
    -----
    该函数只创建表结构，不负责写入具体细胞元数据。写入过程由上游导入函数继续完成。

    Returns
    -------
    None
        表结构直接创建到 DuckDB 数据库中，不返回对象。
    """

    # 系统保留字段：由 scAtlasPy 统一创建
    reserved_cols = {"atlas_cell_id", "atlas_cell_name"}

    # 强制使用要求的类型
    cols = [
        "atlas_cell_id   INTEGER",
        "atlas_cell_name VARCHAR",
    ]

    for col in adata.obs.columns:
        # 跳过来源 h5ad 中已有的旧系统字段
        if col in reserved_cols:
            continue

        duck_type = _infer_duckdb_type_from_series(adata.obs[col])
        cols.append(f'"{col}" {duck_type}')

    ddl = f"""
    CREATE OR REPLACE TABLE obs (
        {", ".join(cols)}
    )
    """

    conn.execute(ddl)


# 建立 var表
def _create_var_table_from_adata(conn: DuckDBPyConnection, adata: AnnData):
    """根据 AnnData 的 var 元数据创建 Atlas ``var`` 表。

    该内部函数读取 ``adata.var`` 的列名和 pandas dtype，推断对应的 DuckDB 字段类型，并创建包含
    ``atlas_gene_id`` 与 ``atlas_gene_name`` 的标准 ``var`` 表。

    与 Scanpy 中把基因注释保存在 ``adata.var`` 的约定类似，Atlas 会把基因级元数据持久化到数据库表中，
    供 HVG、PCA loadings、marker gene 和 feature plot 等步骤使用。

    Parameters
    ----------
    conn
        DuckDB 连接对象。

    adata
        输入 AnnData 对象。

        要求包含 ``var`` 和 ``var_names``；来源数据中若已经存在 Atlas 系统字段，会在建表时跳过。

    Notes
    -----
    该函数只创建表结构，不负责写入具体基因元数据。写入过程由上游导入函数继续完成。

    Returns
    -------
    None
        表结构直接创建到 DuckDB 数据库中，不返回对象。
    """

    # 系统保留字段：由 scAtlasPy 统一创建
    reserved_cols = {"atlas_gene_id", "atlas_gene_name"}

    # 强制使用你要求的类型
    cols = [
        "atlas_gene_id USMALLINT",
        "atlas_gene_name VARCHAR",
    ]

    for col in adata.var.columns:
        # 跳过来源 h5ad 中已有的旧系统字段
        if col in reserved_cols:
            continue

        duck_type = _infer_duckdb_type_from_series(adata.var[col])
        cols.append(f'"{col}" {duck_type}')

    ddl = f"""
    CREATE OR REPLACE TABLE var (
        {", ".join(cols)}
    )
    """

    conn.execute(ddl)


# 建立 HyS 存储结构
def _create_hys_tables(conn: DuckDBPyConnection):

    """创建 Atlas 表达矩阵的 HyS 存储表。

    该内部函数创建 Atlas 表达矩阵使用的两张核心表：

    - ``X_HyS_indptr``：保存 CSR ``indptr`` 的累计非零值位置；
    - ``X_HyS_data``：保存非零表达记录，包括 ``id``、``atlas_cell_id``、
      ``atlas_gene_id`` 和 ``data_count``。

    函数会使用 ``CREATE OR REPLACE TABLE``，因此会覆盖同名旧表。

    Parameters
    ----------
    conn
        DuckDB 数据库连接。

    Returns
    -------
    None
        表结构直接创建到 DuckDB 数据库中，不返回对象。

    Notes
    -----
    ``X_HyS_indptr`` 不存第一个 0，只存每个细胞结束位置的累计 indptr；
    ``X_HyS_data.data_count`` 是导入后的 count 尺度表达值。
    """
    conn.execute(
        """ -- 不存第一个0值
        CREATE OR REPLACE TABLE X_HyS_indptr (
            atlas_cell_id  INTEGER,  --   int32
            indptr BIGINT,           --   int64
        )
        """
    )
    conn.execute(
        """
        CREATE OR REPLACE TABLE X_HyS_data (
            id BIGINT,                --   int64
            atlas_cell_id  INTEGER,   --   int32
            atlas_gene_id  USMALLINT,  --  无符号 int16 0 ~ 65535 之间
            data_count REAL            --  原始 count，float 32 单精度浮点数（4字节）
        )
        """
    )


# 导入 obs 表
def _append_obs_rows(adata: AnnData, conn: DuckDBPyConnection, start_cell_id: int) -> int:

    """追加写入一个 AnnData block 的 ``obs`` 行。

    该内部函数用于 backed h5ad 分块导入。函数会复制 ``adata.obs``，移除来源中
    可能已有的 Atlas 系统字段，然后重新生成当前数据库的 ``atlas_cell_id`` 和
    ``atlas_cell_name``，最后追加写入 ``obs`` 表。

    Parameters
    ----------
    adata
        当前待写入的 AnnData block。
    conn
        DuckDB 数据库连接。
    start_cell_id
        当前 obs block 写入时使用的起始 ``atlas_cell_id``。

    Returns
    -------
    int
        下一个 block 应使用的起始 ``atlas_cell_id``。

    Notes
    -----
    ``atlas_cell_name`` 来自当前 block 的 ``adata.obs.index``。
    """
    n = adata.n_obs

    obs_df = adata.obs.copy()

    # 删除来源 h5ad 中已有的旧系统字段
    for c in ["atlas_cell_id", "atlas_cell_name"]:
        if c in obs_df.columns:
            obs_df = obs_df.drop(columns=[c])

    # 重新生成当前数据库自己的 atlas_cell_id / atlas_cell_name
    obs_df["atlas_cell_id"] = np.arange(
        start_cell_id,
        start_cell_id + n,
        dtype=np.int32,
    )

    obs_df["atlas_cell_name"] = adata.obs.index.astype(str)

    # 固定列顺序：系统字段永远在最前面
    obs_df = obs_df[
        ["atlas_cell_id", "atlas_cell_name"]
        + [
            c for c in obs_df.columns
            if c not in ("atlas_cell_id", "atlas_cell_name")
        ]
    ]

    conn.register("obs_df", obs_df)
    conn.execute("INSERT INTO obs SELECT * FROM obs_df")
    conn.unregister("obs_df")

    return start_cell_id + n


# 导入 var 表
def _append_var(adata: AnnData, conn: DuckDBPyConnection):

    """写入 AnnData 的 ``var`` 基因元数据。

    该内部函数用于 h5ad 分块导入流程。函数会复制 ``adata.var``，移除来源中
    可能已有的 Atlas 系统字段，重新生成 ``atlas_gene_id`` 和
    ``atlas_gene_name``，并插入 ``var`` 表。

    Parameters
    ----------
    adata
        用于提供基因元数据的 AnnData 对象。
    conn
        DuckDB 数据库连接。

    Returns
    -------
    None
        基因元数据直接写入 ``var`` 表，不返回对象。

    Notes
    -----
    多 block 导入时 ``var`` 只需要写入一次。
    """
    var_df = adata.var.copy()

    # 删除来源 h5ad 中已有的旧系统字段
    for c in ["atlas_gene_id", "atlas_gene_name"]:
        if c in var_df.columns:
            var_df = var_df.drop(columns=[c])

    # 重新生成当前数据库自己的 atlas_gene_id / atlas_gene_name
    var_df["atlas_gene_id"] = np.arange(
        adata.n_vars,
        dtype=np.uint16,
    )

    var_df["atlas_gene_name"] = adata.var.index.astype(str)

    # 固定列顺序：系统字段永远在最前面
    var_df = var_df[
        ["atlas_gene_id", "atlas_gene_name"]
        + [
            c for c in var_df.columns
            if c not in ("atlas_gene_id", "atlas_gene_name")
        ]
    ]

    conn.register("var_df", var_df)
    conn.execute("INSERT INTO var SELECT * FROM var_df")
    conn.unregister("var_df")


# 导入 X_HyS 表
def _append_x_hys(
    adata: AnnData,
    conn: DuckDBPyConnection,
    *,
    base_cell_id: int,
    global_indptr_id: int,
    global_indptr_offset: int,
    global_data_id: int,
):

    """追加写入一个 AnnData block 的 HyS 稀疏表达矩阵。

    该内部函数把 ``adata.X`` 转换为 CSR 矩阵，并追加写入 Atlas 的 HyS 表：
    ``X_HyS_indptr`` 保存每个细胞的累计 indptr，``X_HyS_data`` 保存非零表达值
    及其对应的 cell/gene ID。表达值会写入 ``data_count`` 字段。

    Parameters
    ----------
    adata
        当前待写入的 AnnData block。要求 ``adata.X`` 已经是 count 尺度。
    conn
        DuckDB 数据库连接。
    base_cell_id
        当前 AnnData block 第一行对应的 Atlas 细胞 ID。
    global_indptr_id
        下一个待写入的 indptr 行 ID。
    global_indptr_offset
        当前已经累计写入的非零值数量，用于重定位 indptr。
    global_data_id
        下一个待写入的 ``X_HyS_data.id``。

    Returns
    -------
    tuple[int, int, int]
        更新后的 ``global_indptr_id``、``global_indptr_offset`` 和
        ``global_data_id``。

    Notes
    -----
    ``atlas_gene_id`` 使用 ``uint16`` 写入，对应数据库中的 ``USMALLINT``。
    """
    X = adata.X

    if not sparse.issparse(X):
        X = sparse.csr_matrix(X)
    elif not sparse.isspmatrix_csr(X):
        X = X.tocsr()


    indptr = X.indptr.astype(np.int64, copy=False)
    indices = X.indices.astype(np.uint16, copy=False)
    data_count = X.data.astype(np.float32, copy=False)

    # ================= indptr =================
    row_nnz = np.diff(indptr)

    # 不存 indptr[0]，只存 indptr[1:]
    adj_indptr = indptr[1:] + np.int64(global_indptr_offset)

    indptr_table = pa.table({
        "atlas_cell_id": pa.array(
            np.arange(
                global_indptr_id,
                global_indptr_id + len(adj_indptr),
                dtype=np.int32,
            ),
            type=pa.int32(),
        ),
        "indptr": pa.array(
            adj_indptr,
            type=pa.int64(),
        ),
    })

    conn.register("_indptr_arrow", indptr_table)
    conn.execute("""
        INSERT INTO X_HyS_indptr (
            atlas_cell_id,
            indptr
        )
        SELECT
            atlas_cell_id,
            indptr
        FROM _indptr_arrow
    """)
    conn.unregister("_indptr_arrow")

    global_indptr_id += len(adj_indptr)

    # ================= data_count =================
    nnz = len(data_count)

    if nnz > 0:

        cell_index = np.repeat(
            np.arange(
                base_cell_id,
                base_cell_id + adata.n_obs,
                dtype=np.int32,
            ),
            row_nnz,
        )

        data_table = pa.table({
            "id": pa.array(
                np.arange(
                    global_data_id,
                    global_data_id + nnz,
                    dtype=np.int64,
                ),
                type=pa.int64(),
            ),
            "atlas_cell_id": pa.array(
                cell_index,
                type=pa.int32(),
            ),
            "atlas_gene_id": pa.array(
                indices,
                type=pa.uint16(),
            ),
            "data_count": pa.array(
                data_count,
                type=pa.float32(),
            ),
        })

        conn.register("_data_arrow", data_table)
        conn.execute("""
            INSERT INTO X_HyS_data (
                id,
                atlas_cell_id,
                atlas_gene_id,
                data_count
            )
            SELECT
                id,
                atlas_cell_id,
                atlas_gene_id,
                data_count
            FROM _data_arrow
        """)
        conn.unregister("_data_arrow")

        global_data_id += nnz
        global_indptr_offset += nnz

    return global_indptr_id, global_indptr_offset, global_data_id


# 导入 obsm
def _add_obsm_from_h5ad(h5ad_path: PathLike[str] | str, atlas: Atlas, cells_per_block: int = 500):

    """从 h5ad 文件导入 ``obsm`` 到 Atlas 数据库。

    该内部函数直接读取 h5ad 文件中的 ``obsm`` group，并为每个 ``obsm`` key
    创建一张 ``obsm_{key}`` 表。每张表包含 ``atlas_cell_id`` 和若干
    ``dim_0``、``dim_1`` 等维度列。

    该函数适用于保留原始细胞顺序的导入模式；如果细胞顺序已经随机重排，
    不应直接按 h5ad 原始顺序导入 ``obsm``。

    Parameters
    ----------
    h5ad_path
        h5ad 文件路径。
    atlas
        Atlas 对象。要求对象已经连接到 DuckDB 数据库。
    cells_per_block
        分批读取 ``obsm`` 数组时每批包含的细胞数。

    Returns
    -------
    None
        ``obsm`` 结果直接写入数据库，不返回对象。

    Notes
    -----
    如果 h5ad 中不存在 ``obsm``，函数会记录日志并跳过。
    """
    h5ad_path = os.fspath(h5ad_path)

    logger.info("导入 obsm")
    conn = atlas.connection

    with h5py.File(h5ad_path, "r") as f:

        if "obsm" not in f:
            logger.info("  - h5ad 中不存在 obsm，跳过")
            return

        obsm_grp = f["obsm"]

        for key in obsm_grp.keys():
            dset = obsm_grp[key]
            n_cells, k = dset.shape

            logger.info(f"  - obsm[{key}] shape={dset.shape}")

            cols = ", ".join([f"dim_{i} DOUBLE" for i in range(k)])
            conn.execute(f"""
                CREATE OR REPLACE TABLE obsm_{key} (
                    atlas_cell_id BIGINT,
                    {cols}
                )
            """)

            for start in range(0, n_cells, cells_per_block):

                end = min(start + cells_per_block, n_cells)
                block = dset[start:end]
                df = pd.DataFrame(
                    block,
                    columns=[f"dim_{i}" for i in range(k)]
                )
                df["atlas_cell_id"] = np.arange(start, end, dtype=np.int32)
                df = df[["atlas_cell_id"] + [c for c in df.columns if c != "atlas_cell_id"]]

                conn.register("obsm_df", df)
                conn.execute(f"INSERT INTO obsm_{key} SELECT * FROM obsm_df")
                conn.unregister("obsm_df")

    logger.info("obsm 导入完成")


# 导入 varm
def _add_varm_from_h5ad(h5ad_path: PathLike[str] | str, atlas: Atlas):

    """从 h5ad 文件导入 ``varm`` 到 Atlas 数据库。

    该内部函数直接读取 h5ad 文件中的 ``varm`` group，并为每个 ``varm`` key
    创建一张 ``varm_{key}`` 表。每张表包含 ``atlas_gene_id`` 和若干
    ``dim_0``、``dim_1`` 等维度列。

    ``varm`` 与基因维度对齐，不依赖细胞顺序，因此顺序导入和随机导入都可以
    安全导入。

    Parameters
    ----------
    h5ad_path
        h5ad 文件路径。
    atlas
        Atlas 对象。要求对象已经连接到 DuckDB 数据库。

    Returns
    -------
    None
        ``varm`` 结果直接写入数据库，不返回对象。

    Notes
    -----
    如果 h5ad 中不存在 ``varm``，函数会记录日志并跳过。
    """
    h5ad_path = os.fspath(h5ad_path)

    logger.info("导入 varm")

    conn = atlas.connection

    with h5py.File(h5ad_path, "r") as f:

        if "varm" not in f:
            logger.info("  - h5ad 中不存在 varm，跳过")
            return

        varm_grp = f["varm"]

        for key in varm_grp.keys():
            dset = varm_grp[key]
            n_genes, k = dset.shape

            logger.info(f"  - varm[{key}] shape={dset.shape}")

            df = pd.DataFrame(
                dset[:],
                columns=[f"dim_{i}" for i in range(k)]
            )
            df["atlas_gene_id"] = np.arange(n_genes, dtype=np.uint16)
            df = df[["atlas_gene_id"] + [c for c in df.columns if c != "atlas_gene_id"]]

            conn.register("varm_df", df)
            conn.execute(f"""
                CREATE OR REPLACE TABLE varm_{key} AS
                SELECT * FROM varm_df
            """)
            conn.unregister("varm_df")

    logger.info("varm 导入完成")


# 多种数据格式的导入
def _read_smart(file_path: PathLike[str] | str):
    """根据文件后缀自动选择 Scanpy 读取函数。

    该函数检查输入文件路径的扩展名，并调用对应的 ``scanpy.read_*`` 函数读取为 AnnData。

    它适合小型数据或临时转换场景，能够统一处理 h5ad、loom、Matrix Market、csv、文本表格、Excel、
    10x h5 和 UMI-tools 等常见输入格式。

    Parameters
    ----------
    file_path
        输入单细胞数据文件路径。

    Returns
    -------
    adata
        读取完成的 AnnData 对象。

    Notes
    -----
    该函数只负责读取文件，不会直接写入 Atlas 数据库。若要导入数据库，可继续调用 ``load_anndata`` 或
    ``load_multi_format``。

    Examples
    --------
    读取 h5ad 文件：::

        adata = _read_smart("example.h5ad")
    """

    # 获取文件后缀名（小写形式）
    file_path = os.fspath(file_path)
    file_ext = os.path.splitext(file_path)[1].lower()

    # 根据文件后缀选择对应的读取方法
    if file_ext == '.h5ad':
        # h5ad格式
        return sc.read_h5ad(file_path)
    elif file_ext == '.loom':
        # loom格式
        return sc.read_loom(file_path)
    elif file_ext in ['.mtx', '.mtx.gz']:
        # mtx格式 (Matrix Market格式)
        return sc.read_mtx(file_path)
    elif file_ext in ['.csv', '.csv.gz']:
        # csv格式
        return sc.read_csv(file_path)
    elif file_ext in ['.txt', '.tsv', '.tab']:
        # 文本格式，默认制表符分隔
        return sc.read_text(file_path)
    elif file_ext in ['.xlsx', '.xls']:
        # Excel格式
        return sc.read_excel(file_path)
    elif file_ext == '.h5':
        # 10x Genomics h5格式
        return sc.read_10x_h5(file_path)
    elif 'umi_tools' in file_path.lower():
        # UMI-tools格式
        return sc.read_umi_tools(file_path)
    else:
        # 如果不认识的后缀，尝试使用通用的read函数
        return sc.read(file_path)


# 以下函数只在 load_anndata 中使用
# 导入 obs
def _add_obs(adata:AnnData, atlas:Atlas):

    """从内存 AnnData 创建并写入 ``obs`` 表。

    该内部函数用于 ``load_anndata``。它复制 ``adata.obs``，把 AnnData 的细胞
    index 写入 ``atlas_cell_name``，并按当前行顺序生成 ``atlas_cell_id``，
    最后创建或替换 Atlas 数据库中的 ``obs`` 表。

    Parameters
    ----------
    adata
        输入 AnnData 对象。
    atlas
        Atlas 对象。要求对象已经连接到 DuckDB 数据库。

    Returns
    -------
    None
        ``obs`` 表直接写入数据库，不返回对象。

    Notes
    -----
    写入后会为 ``obs.atlas_cell_id`` 添加主键。
    """
    logger.info("导入obs数据")

    obs_df = adata.obs.copy()
    obs_df['atlas_cell_name'] = adata.obs.index

    obs_df['atlas_cell_id'] = range(len(obs_df))  # 添加 id 列

    obs_df = obs_df[['atlas_cell_id', 'atlas_cell_name'] + [col for col in obs_df.columns if col not in ['atlas_cell_id', 'atlas_cell_name']]]  # 直接指定列的顺序
    logger.info(f"obs表数据准备完成，行数: {len(obs_df)}")

    atlas.connection.register('obs_df', obs_df)
    atlas.connection.execute("CREATE OR REPLACE TABLE obs AS SELECT * FROM obs_df")
    atlas.connection.execute("ALTER TABLE obs ADD PRIMARY KEY (atlas_cell_id)")  # 设置ID字段为主码，保证唯一性
    atlas.connection.unregister('obs_df')
    logger.info("导入obs数据成功")


# 导入 var
def _add_var(adata:AnnData, atlas:Atlas):

    """从内存 AnnData 创建并写入 ``var`` 表。

    该内部函数用于 ``load_anndata``。它把 AnnData 的基因 index 写入
    ``atlas_gene_name``，按当前基因顺序生成 ``atlas_gene_id``，并创建或替换
    Atlas 数据库中的 ``var`` 表。

    Parameters
    ----------
    adata
        输入 AnnData 对象。
    atlas
        Atlas 对象。要求对象已经连接到 DuckDB 数据库。

    Returns
    -------
    None
        ``var`` 表直接写入数据库，不返回对象。

    Notes
    -----
    写入后会为 ``var.atlas_gene_id`` 添加主键。
    """
    logger.info("导入var数据")
    var_df = adata.var.reset_index().rename(columns={'index': 'atlas_gene_name'})
    var_df['atlas_gene_id'] = range(len(var_df))
    var_df = var_df[['atlas_gene_id', 'atlas_gene_name'] + [col for col in var_df.columns if col not in ['atlas_gene_id', 'atlas_gene_name']]]  # 直接指定列的顺序
    logger.info(f"var表数据准备完成，行数: {len(var_df)}")

    atlas.connection.register('var_df', var_df)
    atlas.connection.execute("CREATE OR REPLACE TABLE var AS SELECT * FROM var_df")
    atlas.connection.execute("ALTER TABLE var ADD PRIMARY KEY (atlas_gene_id)")  # 设置ID字段为主码，保证唯一性
    atlas.connection.unregister('var_df')
    logger.info("导入var数据成功")


# 导入 obsm
def _add_obsm(adata: AnnData, atlas: Atlas):

    """从内存 AnnData 导入 ``obsm`` 矩阵。

    该内部函数用于 ``load_anndata``。它遍历 ``adata.obsm`` 中的二维矩阵，
    为每个 key 创建或替换一张 ``obsm_{key}`` 表，字段包括 ``atlas_cell_id``
    和 ``dim_0``、``dim_1`` 等维度列。

    Parameters
    ----------
    adata
        输入 AnnData 对象。
    atlas
        Atlas 对象。要求对象已经连接到 DuckDB 数据库。

    Returns
    -------
    None
        ``obsm`` 结果直接写入数据库，不返回对象。

    Notes
    -----
    如果 ``adata.obsm`` 为空，函数会记录日志并跳过。
    """
    logger.info("导入 obsm ")

    conn = atlas.connection
    n_cells = adata.n_obs

    if not adata.obsm:
        logger.info("  - 无 obsm，跳过")
        return

    for key, mat in adata.obsm.items():
        logger.info(f"  - obsm[{key}] shape = {mat.shape}")

        # 强制 numpy（避免 pandas 稀疏坑）
        mat = np.asarray(mat)

        df = pd.DataFrame(
            mat,
            columns=[f"dim_{i}" for i in range(mat.shape[1])]
        )

        df.insert(0, "atlas_cell_id", np.arange(n_cells, dtype=np.int32))

        conn.register("obsm_df", df)
        conn.execute(f"""
            CREATE OR REPLACE TABLE obsm_{key} AS
            SELECT * FROM obsm_df
        """)
        conn.unregister("obsm_df")

    logger.info("obsm 导入完成（统一 schema）")


# 导入 varm
def _add_varm(adata: AnnData, atlas: Atlas):

    """从内存 AnnData 导入 ``varm`` 矩阵。

    该内部函数用于 ``load_anndata``。它遍历 ``adata.varm`` 中的二维矩阵，
    为每个 key 创建或替换一张 ``varm_{key}`` 表，字段包括 ``atlas_gene_id``
    和 ``dim_0``、``dim_1`` 等维度列。

    Parameters
    ----------
    adata
        输入 AnnData 对象。
    atlas
        Atlas 对象。要求对象已经连接到 DuckDB 数据库。

    Returns
    -------
    None
        ``varm`` 结果直接写入数据库，不返回对象。

    Notes
    -----
    如果 ``adata.varm`` 为空，函数会记录日志并跳过。
    """
    logger.info("导入 varm（统一 schema）")

    conn = atlas.connection
    n_genes = adata.n_vars

    if not adata.varm:
        logger.info("  - 无 varm，跳过")
        return

    for key, mat in adata.varm.items():
        logger.info(f"  - varm[{key}] shape = {mat.shape}")

        mat = np.asarray(mat)

        df = pd.DataFrame(
            mat,
            columns=[f"dim_{i}" for i in range(mat.shape[1])]
        )

        df.insert(0, "atlas_gene_id", np.arange(n_genes, dtype=np.uint16))

        conn.register("varm_df", df)
        conn.execute(f"""
            CREATE OR REPLACE TABLE varm_{key} AS
            SELECT * FROM varm_df
        """)
        conn.unregister("varm_df")

    logger.info("varm 导入完成（统一 schema）")


# 导入 x_hys
def _add_x_hys_chunked(adata: AnnData, atlas: Atlas, chunk_size: int = 500):

    """从内存 AnnData 分块写入 HyS 表达矩阵。

    该内部函数用于 ``load_anndata``。它把 ``adata.X`` 按细胞分块转换为 CSR，
    并写入 ``X_HyS_indptr`` 和 ``X_HyS_data``。非零表达值写入
    ``X_HyS_data.data_count`` 字段。

    Parameters
    ----------
    adata
        输入 AnnData 对象。函数会读取 ``adata.X``。
    atlas
        Atlas 对象。函数会连接并写入对应 DuckDB 数据库。
    chunk_size
        每个 chunk 包含的细胞数量，用于控制内存峰值和单次写入规模。

    Returns
    -------
    bool
        成功写入时返回 ``True``。异常时回滚事务并重新抛出错误。

    Notes
    -----
    该函数会重建 ``X_HyS_indptr`` 和 ``X_HyS_data`` 表。
    """
    logger.info("开始导入 X_HyS ")

    conn = atlas.connect("r+")
    atlas.connection = conn

    n_cells = adata.n_obs

    # ===================== 建表 =====================
    conn.execute(
        """ -- 不存第一个0值
        CREATE OR REPLACE TABLE X_HyS_indptr(
            atlas_cell_id  INTEGER,
            indptr BIGINT
        )
        """
    )
    conn.execute(
        """
        CREATE OR REPLACE TABLE X_HyS_data (
            id BIGINT,
            atlas_cell_id INTEGER,
            atlas_gene_id USMALLINT,  --  无符号 int16 0 ~ 65535 之间
            data_count REAL           --  原始 count，float 32 单精度浮点数（4字节）
        )
        """
    )

    conn.execute("BEGIN TRANSACTION")

    try:
        total_chunks = (n_cells + chunk_size - 1) // chunk_size

        global_data_counter = np.int64(0)
        global_indptr_offset = np.int64(0)

        for chunk_idx in progress(range(total_chunks), desc="load"):
            start = chunk_idx * chunk_size
            end = min(start + chunk_size, n_cells)
            size = end - start

            # === 取数据 ===
            X_chunk = adata.X[start:end]

            if sparse.issparse(X_chunk):
                csr = X_chunk.tocsr()
            else:
                csr = sparse.csr_matrix(X_chunk)

            data_count = csr.data
            indices = csr.indices.astype(np.uint16)
            indptr = csr.indptr.astype(np.int64)

            # ================= indptr 表 =================
            adj_indptr = indptr[1:] + global_indptr_offset

            indptr_df = pd.DataFrame({
                "atlas_cell_id": np.arange(start, end, dtype=np.int32),
                "indptr": adj_indptr
            })

            conn.execute("INSERT INTO X_HyS_indptr SELECT * FROM indptr_df")

            global_indptr_offset = adj_indptr[-1]

            # ================= data_count 表达值 =================
            if len(data_count) > 0:
                nnz = len(data_count)
                data_ids = np.arange(
                    global_data_counter,
                    global_data_counter + nnz,
                    dtype=np.int64
                )

                # 直接在 chunk 内构造 cell_index（CSR → COO）
                row_lengths = np.diff(indptr)
                cell_index = np.repeat(
                    np.arange(start, end, dtype=np.int32),
                    row_lengths
                )

                data_df = pd.DataFrame({
                    "id": data_ids,
                    "atlas_cell_id": cell_index,
                    "atlas_gene_id": indices,
                    "data_count": data_count
                })

                conn.execute("INSERT INTO X_HyS_data SELECT * FROM data_df")

                global_data_counter += nnz

            # === 清理 ===
            del X_chunk, csr, indptr_df
            if len(data_count) > 0:
                del data_df
            gc.collect()

        conn.execute("COMMIT")

        logger.info(
            f"导入完成：cells={n_cells:,}, nnz={global_data_counter:,}"
        )

        return True

    except Exception as e:
        conn.execute("ROLLBACK")
        logger.error(f"CSR 导入失败: {e}")
        raise
