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
from tqdm import tqdm
from typing import TYPE_CHECKING, Any, Literal
if TYPE_CHECKING:   # TYPE_CHECKING = 给 IDE / 类型检查器看的导入;  正常运行时 = 不执行这个导入，避免循环导入
    from ..data import Atlas

StoreType = Literal["count", "log"]

# 获取日志记录器
logger = logging.getLogger("Atlas")
logger.addHandler(logging.NullHandler())


# 统一的 h5ad 导入接口
def load_h5ad(
    h5ad_path: PathLike[str] | str | list[PathLike[str] | str],
    atlas: Atlas,
    *,
    load_type: Literal["order", "random", "list_random"] = "random",
    store_type: StoreType = "count",
    cells_per_block: int = 500,
    blocks_per_pool: int = 10,
) -> Any:
    """统一 h5ad 导入接口。

    根据 ``load_type`` 选择不同的大 h5ad 导入策略。

    Parameters
    ----------
    h5ad_path : PathLike[str] | str | list[PathLike[str] | str]
        h5ad 文件路径。

        - 当 ``load_type="order"``、``"random"``时：
          传入单个 h5ad 文件路径，类型为 ``str``。

        - 当 ``load_type="list_random"`` 时：
          可以传入多个 h5ad 文件路径，类型为 ``list[str]``；
          也可以只传入单个路径 ``str``，函数内部会自动转成单元素列表。

    atlas : Atlas
        Atlas 对象。

        通常是通过 ``sap.Atlas(...)`` 创建的数据库对象。
        函数会将 h5ad 中的 ``obs``、``var`` 和表达矩阵写入该 Atlas 对应的 DuckDB 数据库。

    load_type : {"order", "random", "list_random"}, default: "random"
        导入方式。

        - ``"order"``：
          顺序读取单个 h5ad，保留原始 cell 顺序。
          适合需要保留 AnnData 行顺序、并安全导入 ``obsm`` / ``varm`` 的场景。

        - ``"random"``：
          单个 h5ad 的 shuffle-window 随机导入。
          会随机重排 cell，因此默认不导入 ``obsm``，只导入 ``varm``。

        - ``"list_random"``：
          多个 h5ad 文件的全局 block 随机导入。
          适合多个文件大小不一致、希望整体随机混合导入的场景。

    cells_per_block : int, default: 500
        每个连续读取 block 中包含的细胞数量。

        函数会按 ``cells_per_block`` 将 h5ad 切成连续 block。
        该值越大，连续读取效率通常越高，但单个 block 占用内存也越大。

    blocks_per_pool : int, default: 5
        每次合并多少个 block 形成一个 pool / window 后写入数据库。

        实际每次写入或 flush 的细胞数量大约为：

        ``cells_per_block × blocks_per_pool``

        该值越大，随机混合程度通常越高，但单次内存占用也越大。

    store_type : {"count", "log"}, default: "count"
        目标表达矩阵尺度。

        - ``"count"``：
          将表达矩阵以 count 尺度写入数据库。
          如果输入 h5ad 的 ``X`` 被检测为 log 尺度，会在导入时执行 ``expm1`` 转换。

        - ``"log"``：
          将表达矩阵以 log1p 尺度写入数据库。
          如果输入 h5ad 的 ``X`` 被检测为 count 尺度，会在导入时执行 ``log1p`` 转换。

    Returns
    -------
    Any
        返回底层导入函数的返回结果。

        不同 ``load_type`` 对应的底层函数返回值可能不同；
        有些函数返回导入统计信息，有些函数只执行导入流程而不显式返回结果。
    """
    start_time = datetime.now()

    # =====================================================
    # 1. 参数检查
    # =====================================================
    valid_load_types = {
        "order",
        "random",
        "list_random",
    }

    if load_type not in valid_load_types:
        raise ValueError(
            "load_type 只能是 "
            f"{sorted(valid_load_types)}，当前为: {load_type}"
        )

    if store_type not in {"count", "log"}:
        raise ValueError(
            f"store_type 只能是 'count' 或 'log'，当前为: {store_type}"
        )

    if not isinstance(cells_per_block, int):
        raise TypeError(
            f"cells_per_block 必须是 int，当前类型为: {type(cells_per_block)}"
        )

    if cells_per_block <= 0:
        raise ValueError("cells_per_block 必须 > 0")

    if not isinstance(blocks_per_pool, int):
        raise TypeError(
            f"blocks_per_pool 必须是 int，当前类型为: {type(blocks_per_pool)}"
        )

    if blocks_per_pool <= 0:
        raise ValueError("blocks_per_pool 必须 > 0")

    # =====================================================
    # 2. list_random：多文件随机导入
    # =====================================================
    if load_type == "list_random":

        logger.info("[INFO] load_type = list_random")

        return _load_h5ad_list_random(
            h5ad_paths=h5ad_path,
            atlas=atlas,
            cells_per_block=cells_per_block,
            blocks_per_pool=blocks_per_pool,
            store_type=store_type,
        )

    # =====================================================
    # 3. 其他模式必须是单个 h5ad 路径
    # =====================================================
    if isinstance(h5ad_path, (list, tuple)):
        raise ValueError(
            f"load_type='{load_type}' 只支持单个 h5ad_path；"
            "如果要导入多个 h5ad，请使用 load_type='list_random'"
        )

    if not isinstance(h5ad_path, (str, PathLike)):
        raise TypeError(
            f"h5ad_path 必须是 str，当前类型为: {type(h5ad_path)}"
        )

    h5ad_path = os.fspath(h5ad_path)

    # =====================================================
    # 4. order：顺序导入
    # =====================================================
    if load_type == "order":

        logger.info("[INFO] load_type = order")

        return _load_h5ad_order(
            h5ad_path=h5ad_path,
            atlas=atlas,
            cells_per_block=cells_per_block,
            blocks_per_pool=blocks_per_pool,
            store_type=store_type,
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
            blocks_per_pool=blocks_per_pool,
            store_type=store_type,
        )

    print(f"load_h5ad Done, 耗时: {(datetime.now() - start_time).total_seconds():.2f} 秒")
    return None


''' 方法1 ： 随机读取 , 多个大文件, 只支持 h5ad格式 '''
def _load_h5ad_list_random(
    h5ad_paths: PathLike[str] | str | list[PathLike[str] | str],
    atlas: Atlas,
    store_type: StoreType = "count",  # 目标存储类型，"count" 或 "log"
    cells_per_block: int = 500,
    blocks_per_pool: int = 10,    # 每次从全局随机 block 池读取多少个 block 后 flush
):
    """随机导入多个 h5ad 文件到 Atlas 数据库。

    该函数先把每个 h5ad 文件按 ``cells_per_block`` 切成连续 block，再把所有文件的 block 合并到全局 block pool
    中随机打乱。

    每次读取 ``blocks_per_pool`` 个 block 后，会合并为 cell pool，对 pool 内细胞整体随机一次，然后写入
    ``obs``、``var``、``X_HyS_indptr`` 和 ``X_HyS_data``。

    该策略适合多个文件大小不一致的场景，既避免 round-robin 后期只剩大文件，也减少 h5ad cell-level 随机 IO。

    Parameters
    ----------
    h5ad_paths
        一个或多个 h5ad 文件路径。

    atlas
        Atlas 对象。通常要求已经连接数据库，并包含该函数所需的 ``obs``、``var``、``X_HyS_data`` 或
        embedding 结果表。

    cells_per_block
        每批读取、写入或处理的细胞数量；较大值通常更快但占用更多内存。

    blocks_per_pool
        多文件随机导入时每次合并并 flush 的 block 数量。

    store_type
        目标表达矩阵尺度，通常为 ``"count"`` 或 ``"log"``。

    Returns
    -------
    result
        函数返回结果。具体类型取决于参数设置和内部执行路径。

    Notes
    -----
    导入导出过程可能涉及较大的磁盘 IO 和内存占用，可根据数据规模调整 batch、chunk 或 worker 参数。

    Examples
    --------
    调用该函数：::

        sap.io._load_h5ad_fast_random(...)
    """


    # 支持单路径 / 多路径
    if isinstance(h5ad_paths, (str, PathLike)):
        h5ad_paths = [h5ad_paths]

    h5ad_paths = [os.fspath(path) for path in h5ad_paths]

    if len(h5ad_paths) == 0:
        raise ValueError("h5ad_paths 不能为空")

    if blocks_per_pool <= 0:
        raise ValueError("blocks_per_pool 必须 > 0")

    commit_every = 5  # 每多少次 pool flush 提交一次
    gc_every = 5    # 每多少次 pool flush 做一次 gc

    # 检查目标存储类型
    if store_type not in {"count", "log"}:
        raise ValueError(
            f"store_type 只能是 'count' 或 'log'，当前为: {store_type}"
        )

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

    print(f"[INFO] 文件数量: {file_num:,}")
    print(f"[INFO] store_type: {store_type}")

    file_states = []

    ref_var_names = None
    ref_n_genes = None

    # 全局 block 索引池 ; 每个元素记录一个 block 来自哪个文件、起止位置是多少
    all_block_refs = []

    try:
        for file_idx, h5ad_path in enumerate(h5ad_paths):

            adata_backed = sc.read_h5ad(h5ad_path, backed="r")

            n_cells = adata_backed.n_obs
            n_genes = adata_backed.n_vars

            print(f"[INFO] 当前文件维度 : {n_cells:,} × {n_genes:,}")

            # 每个文件单独检测 X 是 count 还是 log
            source_store_type = _detect_x_store_type_from_backed(
                adata_backed,
                sample_n=5000,
            )

            logger.info(f"[INFO] 当前文件 X 判断为: {source_store_type}")

            if source_store_type == store_type:
                logger.info("[INFO] 当前文件 X 不需要转换，直接写入。")
            else:
                logger.info(f"[INFO] 当前文件 X 将在读取 block 后转换: {source_store_type} -> {store_type}")

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

            # 把所有文件的 block 放进 all_block_refs，最后统一全局 shuffle。
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
                    "source_store_type": source_store_type,
                }
            )

        # 全局 block 索引池统一随机打乱
        total_blocks = len(all_block_refs)

        if total_blocks == 0:
            raise ValueError("所有 h5ad 文件的 cell 数量为 0，无法导入")

        rng.shuffle(all_block_refs)

        # 动态建表：只用第一个文件建表
        first_backed = file_states[0]["adata_backed"]

        _create_obs_table_from_adata(conn, first_backed[:1])
        _create_var_table_from_adata(conn, first_backed[:1])
        _create_hys_tables(conn)

        # 多个 block 读到的 batch 合并后，整体随机，然后写入数据库
        def _flush_cell_pool(cell_pool: list[AnnData], flush_i: int):

            """执行 ``_flush_cell_pool`` 的核心功能。

            该内部函数属于数据导入模块，用于支撑同一模块中的公共 API。

            把 h5ad、AnnData 或 Scanpy 支持的数据格式写入 Atlas 的
            ``obs``、``var``、``X_HyS_*``、``obsm`` 和 ``varm`` 表。

            它通常不会作为用户入口直接调用；直接调用时需要保证输入对象、数据库连接和相关临时表已经由上游步骤准备好。

            当前实现中会访问或生成的关键表包括：``obs``、``var``。

            Parameters
            ----------
            cell_pool
                当前 flush 中收集到的 AnnData block 列表。

            flush_i
                当前 flush 的编号，用于日志输出。

            Returns
            -------
            result
                函数返回结果。具体类型取决于参数设置和内部执行路径。

            Notes
            -----
            这是内部 helper；除非需要扩展 scAtlasPy 内部流程，一般不建议在用户代码中直接调用。
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


            # 2. cell_pool 内部整体随机
            if pool_adata.n_obs > 1:
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

        # 主循环：全局 block list 随机后，每次读取 blocks_per_pool 个 block
        processed_blocks = 0
        flush_counter = 0

        pbar = tqdm(total=total_blocks, desc="读取进度")

        # 事务：多个 flush 共用事务
        conn.execute("BEGIN TRANSACTION")

        try:
            block_cursor = 0

            while block_cursor < total_blocks:

                # 每次从全局随机 block list 中取 blocks_per_pool 个 block
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

                    # 根据该文件的 source_store_type 转换当前 block
                    adata = _convert_x_store_type_inplace(
                        adata,
                        source_store_type=state["source_store_type"],
                        target_store_type=store_type,
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

                # 这一组 block 读完后，整体 shuffle + 写入
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


''' 方法2 ： 随机读取 , 单个大文件, 只支持 h5ad格式 '''
def _load_h5ad_random(
    h5ad_path: PathLike[str] | str,
    atlas: Atlas,
    store_type: StoreType = "count",  # 目标存储类型，"count" 或 "log"
    cells_per_block: int = 500,
    blocks_per_pool: int = 10,   # 固定窗口级随机，默认 5 个 batch
):
    """以 shuffle-window 方式随机导入单个 h5ad 文件。

    该函数把 backed h5ad 文件切成连续 block，随机打乱 block 顺序，并把多个 block 合并成一个 shuffle
    window。

    每个 window 内的细胞会整体随机打乱，再批量写入 Atlas 数据库，从而在控制内存占用的同时获得随机导入顺序。

    由于细胞顺序被重排，函数默认不导入 ``obsm``；``varm`` 与基因对齐，可以正常导入。

    Parameters
    ----------
    h5ad_path
        h5ad 文件路径。

    atlas
        Atlas 对象。通常要求已经连接数据库，并包含该函数所需的 ``obs``、``var``、``X_HyS_data`` 或
        embedding 结果表。

    cells_per_block
        每批读取、写入或处理的细胞数量；较大值通常更快但占用更多内存。

    blocks_per_pool
        单文件随机导入时每个 shuffle window 包含的 block 数量。

    store_type
        目标表达矩阵尺度，通常为 ``"count"`` 或 ``"log"``。

    Notes
    -----
    导入导出过程可能涉及较大的磁盘 IO 和内存占用，可根据数据规模调整 batch、chunk 或 worker 参数。

    Examples
    --------
    调用该函数：::

        sap.io._load_h5ad_random(...)
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

    # 检查目标存储类型
    if store_type not in {"count", "log"}:
        raise ValueError(
            f"store_type 只能是 'count' 或 'log'，当前为: {store_type}"
        )

    # 预读取 1000 个细胞，判断文件里的 X 是 count 还是 log
    source_store_type = _detect_x_store_type_from_backed(
        adata_backed,
        sample_n=5000,
    )

    logger.info(f"[INFO] 文件中 X 判断为: {source_store_type}")
    logger.info(f"[INFO] 目标存储类型 store_type = {store_type}")

    if source_store_type == store_type:
        logger.info("[INFO] X 数据不需要转换，直接写入。")
    else:
        logger.info(f"[INFO] X 数据将在写入前转换: {source_store_type} -> {store_type}")

    # 5 个 block 合并后再统一随机
    block_starts = np.arange(0, n_cells, cells_per_block, dtype=np.int64)
    np.random.shuffle(block_starts)

    # 动态建表
    _create_obs_table_from_adata(conn, adata_backed[:1])
    _create_var_table_from_adata(conn, adata_backed[:1])
    _create_hys_tables(conn)

    print(f"[INFO] 数据集维度: {adata_backed.n_obs:,} × {adata_backed.n_vars:,}")
    print(f"[INFO] store_type = {store_type}")

    # 窗口缓存: 每次攒够 blocks_per_pool 个 batch 再统一随机写入
    window_adatas = []
    window_batch_count = 0
    total_batch_counter = 0
    window_counter = 0

    conn.execute("BEGIN TRANSACTION")

    try:
        for block_i, block_start in enumerate(
            tqdm(
                block_starts,
                desc="读取进度",
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
                    source_store_type=source_store_type,
                    target_store_type=store_type,
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
                if total_batch_counter % commit_every == 0:
                    conn.execute("COMMIT")
                    conn.execute("BEGIN TRANSACTION")
                    logger.info(f"[COMMIT] processed_batches={total_batch_counter:,}")

                # 每 gc_every 个 batch gc 一次
                if total_batch_counter % gc_every == 0:
                    gc.collect()
                    logger.info(f"[GC] processed_batches={total_batch_counter:,}")

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
                source_store_type=source_store_type,
                target_store_type=store_type,
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


''' 方法3 ： 顺序读取 , 单个大文件, 只支持 h5ad格式 '''
def _load_h5ad_order(
    h5ad_path: PathLike[str] | str,
    atlas: Atlas,
    store_type: StoreType = "count",  # 目标存储类型，"count" 或 "log"
    cells_per_block: int = 500,
    blocks_per_pool: int = 10,
):

    """按原始细胞顺序导入单个 h5ad 文件。

    该函数以 backed 模式顺序读取 h5ad，把数据按 mega-batch 载入内存，再拆成较小 batch 写入 Atlas。

    与随机导入不同，它不会打乱细胞顺序，因此可以安全导入 ``obsm`` 和 ``varm``，适合需要保留原始 AnnData 行顺序的场景。

    Parameters
    ----------
    h5ad_path
        h5ad 文件路径。

    atlas
        Atlas 对象。通常要求已经连接数据库，并包含该函数所需的 ``obs``、``var``、``X_HyS_data`` 或
        embedding 结果表。

    cells_per_block
        每批读取、写入或处理的细胞数量；较大值通常更快但占用更多内存。

    blocks_per_pool
        顺序导入时用于计算 mega-batch 大小的倍数。

    store_type
        目标表达矩阵尺度，通常为 ``"count"`` 或 ``"log"``。

    Notes
    -----
    导入导出过程可能涉及较大的磁盘 IO 和内存占用，可根据数据规模调整 batch、chunk 或 worker 参数。

    Examples
    --------
    调用该函数：::

        sap.io._load_h5ad_order(...)
    """
    commit_every = 5
    gc_every = 5

    # 检查目标存储类型
    if store_type not in {"count", "log"}:
        raise ValueError(
            f"store_type 只能是 'count' 或 'log'，当前为: {store_type}"
        )

    # mega_batch_size 改成可控
    mega_batch_size = cells_per_block * blocks_per_pool

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
    source_store_type = _detect_x_store_type_from_backed(
        adata_backed,
        sample_n=5000,
    )

    logger.info(f"[INFO] 文件中 X 判断为: {source_store_type}")
    logger.info(f"[INFO] 目标存储类型 store_type = {store_type}")

    if source_store_type == store_type:
        logger.info("[INFO] X 数据不需要转换，直接写入。")
    else:
        logger.info(f"[INFO] X 数据将在 mega-batch 读入后转换: {source_store_type} -> {store_type}")

    _create_obs_table_from_adata(conn, adata_backed[:1])
    _create_var_table_from_adata(conn, adata_backed[:1])
    _create_hys_tables(conn)

    print(f"[INFO] 数据集维度: {adata_backed.n_obs:,} × {adata_backed.n_vars:,}")
    print(f"[INFO] store_type = {store_type}")

    mini_batch_counter = 0

    # 事务放在大循环外
    conn.execute("BEGIN TRANSACTION")

    try:
        for mega_i, mega_start in enumerate(
            tqdm(
                range(0, n_cells, mega_batch_size),
                desc="读取进度",
            )
        ):
            mega_end = min(mega_start + mega_batch_size, n_cells)

            t0 = time.time()

            # 真正触发磁盘读取
            mega = adata_backed[mega_start:mega_end].to_memory()

            t_read = time.time() - t0

            # mega-batch 读入后，统一转换 X 的存储尺度
            mega = _convert_x_store_type_inplace(
                mega,
                source_store_type=source_store_type,
                target_store_type=store_type,
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
    _add_obsm_from_h5ad(h5ad_path, atlas)
    _add_varm_from_h5ad(h5ad_path, atlas)

    try:
        adata_backed.file.close()
    except Exception:
        pass


''' 方法4： 顺序读取，小文件读取，支持多种数据格式的导入 '''
def load_multi_format(file_path: PathLike[str] | str, atlas: Atlas):

    """导入小型单细胞数据文件。

    该函数先调用 ``_read_smart`` 根据文件后缀自动读取数据，再把得到的 AnnData 对象写入 Atlas。

    它适合可以一次性载入内存的小数据集；大 h5ad 文件建议使用 ``_load_h5ad_random``、``_load_h5ad_fast_random`` 或
    ``_load_h5ad_order``。

    Parameters
    ----------
    file_path
        输入文件路径或 Atlas ``.sasql`` 数据库文件路径。

    atlas
        Atlas 对象。通常要求已经连接数据库，并包含该函数所需的 ``obs``、``var``、``X_HyS_data`` 或
        embedding 结果表。

    Notes
    -----
    导入导出过程可能涉及较大的磁盘 IO 和内存占用，可根据数据规模调整 batch、chunk 或 worker 参数。

    Examples
    --------
    调用该函数：::

        sap.io.load_multi_format(...)
    """
    start_time = datetime.now()
    adata = _read_smart(file_path)
    load_anndata(adata, atlas)
    print(f"load_multi_format Done, 耗时: {(datetime.now() - start_time).total_seconds():.2f} 秒")


# 将计算结果写入数据库表
def _write_shuffle_window_to_duckdb(
    window_adatas: list[AnnData],
    conn: DuckDBPyConnection,
    global_cell_id: int,
    global_indptr_id: int,
    global_indptr_offset: int,
    global_data_id: int,
    var_written: bool,
    source_store_type: StoreType,
    target_store_type: StoreType,
):
    """将计算结果写入数据库表。

    该内部函数属于数据导入模块，用于支撑同一模块中的公共 API。

    把 h5ad、AnnData 或 Scanpy 支持的数据格式写入 Atlas 的
    ``obs``、``var``、``X_HyS_*``、``obsm`` 和 ``varm`` 表。

    它通常不会作为用户入口直接调用；直接调用时需要保证输入对象、数据库连接和相关临时表已经由上游步骤准备好。

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

    source_store_type
        输入表达矩阵当前的尺度。

    target_store_type
        希望写入 Atlas 的表达矩阵尺度。

    Returns
    -------
    result
        函数返回结果。具体类型取决于参数设置和内部执行路径。

    Notes
    -----
    这是内部 helper；除非需要扩展 scAtlasPy 内部流程，一般不建议在用户代码中直接调用。
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

    # 根据 store_type 转换 X 的存储尺度
    # count -> log: np.log1p
    # log   -> count: np.expm1
    adata_window = _convert_x_store_type_inplace(
        adata_window,
        source_store_type=source_store_type,
        target_store_type=target_store_type,
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


# 预读取 sample_n 个细胞，自动判断 X 是 count scale 还是 log scale。
def _detect_x_store_type_from_backed(
    adata_backed: AnnData,
    sample_n: int = 5000,
) -> StoreType:
    """检测输入表达矩阵的存储尺度。

    该内部函数属于数据导入模块，用于支撑同一模块中的公共 API。

    把 h5ad、AnnData 或 Scanpy 支持的数据格式写入 Atlas 的
    ``obs``、``var``、``X_HyS_*``、``obsm`` 和 ``varm`` 表。

    它通常不会作为用户入口直接调用；直接调用时需要保证输入对象、数据库连接和相关临时表已经由上游步骤准备好。

    Parameters
    ----------
    adata_backed
        以 backed 模式打开的 AnnData 对象。

    sample_n
        抽样细胞数量；为 ``None`` 时通常使用全部可用细胞。

    Returns
    -------
    store_type
        检测得到的表达矩阵尺度，通常为 ``"count"`` 或 ``"log"``。

    Notes
    -----
    这是内部 helper；除非需要扩展 scAtlasPy 内部流程，一般不建议在用户代码中直接调用。
    """

    n = min(sample_n, adata_backed.n_obs)

    # 预读取前 n 个细胞
    adata_sample = adata_backed[:n].to_memory()
    X = adata_sample.X

    if sparse.issparse(X):
        values = np.asarray(X.data, dtype=np.float32)
    else:
        X_arr = np.asarray(X)
        values = X_arr[X_arr != 0].astype(np.float32, copy=False)

    if values.size == 0:
        del adata_sample
        gc.collect()
        raise ValueError(
            "[ERROR] 预读取的细胞中没有非零值，无法判断 X 是 count 还是 log。"
        )

    vmax = float(np.max(values))
    q95 = float(np.percentile(values, 95))
    frac_le_10 = float(np.mean(values <= 10))

    logger.info("[INFO] X scale 预检测结果:")
    logger.info(f"  - sample_cells = {n:,}")
    logger.info(f"  - nonzero_n    = {values.size:,}")
    logger.info(f"  - max          = {vmax:.4f}")
    logger.info(f"  - q95          = {q95:.4f}")
    logger.info(f"  - frac <= 10   = {frac_le_10:.4f}")

    del adata_sample
    gc.collect()

    # 经验判断：
    # log 数据通常绝大多数非零值 <= 10
    # count 数据只要 max 或高分位明显超过 log 范围，就判断为 count
    if vmax > 50 or q95 > 10:
        return "count"
    else:
        return "log"


# 根据 source_store_type 和 target_store_type 原地转换 adata.X。 log转换的 底数为 e
def _convert_x_store_type_inplace(
    adata: AnnData,
    source_store_type: StoreType,
    target_store_type: StoreType,
):
    """转换表达矩阵的存储尺度。

    该内部函数属于数据导入模块，用于支撑同一模块中的公共 API。

    把 h5ad、AnnData 或 Scanpy 支持的数据格式写入 Atlas 的
    ``obs``、``var``、``X_HyS_*``、``obsm`` 和 ``varm`` 表。

    它通常不会作为用户入口直接调用；直接调用时需要保证输入对象、数据库连接和相关临时表已经由上游步骤准备好。

    Parameters
    ----------
    adata
        AnnData 对象。函数会读取其中的 ``obs``、``var``、``X``、``obsm`` 或 ``varm``。

    source_store_type
        输入表达矩阵当前的尺度。

    target_store_type
        希望写入 Atlas 的表达矩阵尺度。

    Returns
    -------
    result
        函数返回结果。具体类型取决于参数设置和内部执行路径。

    Notes
    -----
    这是内部 helper；除非需要扩展 scAtlasPy 内部流程，一般不建议在用户代码中直接调用。
    """

    if source_store_type == target_store_type:
        return adata

    X = adata.X

    # 1. sparse matrix：只改非零值 X.data，不破坏稀疏结构
    if sparse.issparse(X):
        if not sparse.isspmatrix_csr(X):
            X = X.tocsr()

        X.data = X.data.astype(np.float32, copy=False)

        if source_store_type == "count" and target_store_type == "log":
            # count -> log
            np.log1p(X.data, out=X.data)

        elif source_store_type == "log" and target_store_type == "count":
            # log -> count
            np.expm1(X.data, out=X.data)

            # 避免极小数值误差导致负数
            X.data[X.data < 0] = 0

        else:
            raise ValueError(
                f"不支持的转换: {source_store_type} -> {target_store_type}"
            )

        adata.X = X
        return adata

    # 2. dense matrix：直接对整个矩阵转换
    X = np.asarray(X, dtype=np.float32)

    if source_store_type == "count" and target_store_type == "log":
        np.log1p(X, out=X)

    elif source_store_type == "log" and target_store_type == "count":
        np.expm1(X, out=X)
        np.maximum(X, 0, out=X)

    else:
        raise ValueError(
            f"不支持的转换: {source_store_type} -> {target_store_type}"
        )

    adata.X = X
    return adata


# 简单判断 h5ad.X 的底层稀疏格式
def _print_h5ad_x_format(h5ad_path: PathLike[str] | str):
    """执行 ``_print_h5ad_x_format`` 的核心功能。

    该内部函数属于数据导入模块，用于支撑同一模块中的公共 API。

    把 h5ad、AnnData 或 Scanpy 支持的数据格式写入 Atlas 的
    ``obs``、``var``、``X_HyS_*``、``obsm`` 和 ``varm`` 表。

    它通常不会作为用户入口直接调用；直接调用时需要保证输入对象、数据库连接和相关临时表已经由上游步骤准备好。

    Parameters
    ----------
    h5ad_path
        h5ad 文件路径。

    Returns
    -------
    result
        函数返回结果。具体类型取决于参数设置和内部执行路径。

    Notes
    -----
    这是内部 helper；除非需要扩展 scAtlasPy 内部流程，一般不建议在用户代码中直接调用。
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

    """执行 ``_infer_duckdb_type_from_series`` 的核心功能。

    该内部函数属于数据导入模块，用于支撑同一模块中的公共 API。

    把 h5ad、AnnData 或 Scanpy 支持的数据格式写入 Atlas 的
    ``obs``、``var``、``X_HyS_*``、``obsm`` 和 ``varm`` 表。

    它通常不会作为用户入口直接调用；直接调用时需要保证输入对象、数据库连接和相关临时表已经由上游步骤准备好。

    Parameters
    ----------
    series
        需要推断 DuckDB 类型的 pandas Series。

    Returns
    -------
    result
        函数返回结果。具体类型取决于参数设置和内部执行路径。

    Notes
    -----
    这是内部 helper；除非需要扩展 scAtlasPy 内部流程，一般不建议在用户代码中直接调用。
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

    """创建 Atlas 工作流所需的数据库表。

    该内部函数属于数据导入模块，用于支撑同一模块中的公共 API。

    把 h5ad、AnnData 或 Scanpy 支持的数据格式写入 Atlas 的
    ``obs``、``var``、``X_HyS_*``、``obsm`` 和 ``varm`` 表。

    它通常不会作为用户入口直接调用；直接调用时需要保证输入对象、数据库连接和相关临时表已经由上游步骤准备好。

    当前实现中会访问或生成的关键表包括：``X_HyS_data``、``X_HyS_indptr``。

    Parameters
    ----------
    conn
        DuckDB 数据库连接。

    Notes
    -----
    这是内部 helper；除非需要扩展 scAtlasPy 内部流程，一般不建议在用户代码中直接调用。
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
            data REAL                  --  float 32 单精度浮点数（4字节）
        )
        """
    )


# 导入 obs 表
def _append_obs_rows(adata: AnnData, conn: DuckDBPyConnection, start_cell_id: int) -> int:

    """将数据写入 Atlas 数据库。

    该内部函数属于数据导入模块，用于支撑同一模块中的公共 API。

    把 h5ad、AnnData 或 Scanpy 支持的数据格式写入 Atlas 的
    ``obs``、``var``、``X_HyS_*``、``obsm`` 和 ``varm`` 表。

    它通常不会作为用户入口直接调用；直接调用时需要保证输入对象、数据库连接和相关临时表已经由上游步骤准备好。

    当前实现中会访问或生成的关键表包括：``obs``。

    Parameters
    ----------
    adata
        AnnData 对象。函数会读取其中的 ``obs``、``var``、``X``、``obsm`` 或 ``varm``。

    conn
        DuckDB 数据库连接。

    start_cell_id
        当前 obs block 写入时使用的起始 ``atlas_cell_id``。

    Returns
    -------
    result
        函数返回结果。具体类型取决于参数设置和内部执行路径。

    Notes
    -----
    这是内部 helper；除非需要扩展 scAtlasPy 内部流程，一般不建议在用户代码中直接调用。
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

    """将数据写入 Atlas 数据库。

    该内部函数属于数据导入模块，用于支撑同一模块中的公共 API。

    把 h5ad、AnnData 或 Scanpy 支持的数据格式写入 Atlas 的
    ``obs``、``var``、``X_HyS_*``、``obsm`` 和 ``varm`` 表。

    它通常不会作为用户入口直接调用；直接调用时需要保证输入对象、数据库连接和相关临时表已经由上游步骤准备好。

    当前实现中会访问或生成的关键表包括：``var``。

    Parameters
    ----------
    adata
        AnnData 对象。函数会读取其中的 ``obs``、``var``、``X``、``obsm`` 或 ``varm``。

    conn
        DuckDB 数据库连接。

    Notes
    -----
    这是内部 helper；除非需要扩展 scAtlasPy 内部流程，一般不建议在用户代码中直接调用。
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

    """将数据写入 Atlas 数据库。

    该内部函数属于数据导入模块，用于支撑同一模块中的公共 API。

    把 h5ad、AnnData 或 Scanpy 支持的数据格式写入 Atlas 的
    ``obs``、``var``、``X_HyS_*``、``obsm`` 和 ``varm`` 表。

    它通常不会作为用户入口直接调用；直接调用时需要保证输入对象、数据库连接和相关临时表已经由上游步骤准备好。

    当前实现中会访问或生成的关键表包括：``X_HyS_data``、``X_HyS_indptr``。

    Parameters
    ----------
    adata
        AnnData 对象。函数会读取其中的 ``obs``、``var``、``X``、``obsm`` 或 ``varm``。

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
    result
        函数返回结果。具体类型取决于参数设置和内部执行路径。

    Notes
    -----
    这是内部 helper；除非需要扩展 scAtlasPy 内部流程，一般不建议在用户代码中直接调用。
    """
    X = adata.X

    if not sparse.issparse(X):
        X = sparse.csr_matrix(X)
    elif not sparse.isspmatrix_csr(X):
        X = X.tocsr()


    indptr = X.indptr.astype(np.int64, copy=False)
    indices = X.indices.astype(np.uint16, copy=False)
    data = X.data.astype(np.float32, copy=False)

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

    # ================= data =================
    nnz = len(data)

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
            "data": pa.array(
                data,
                type=pa.float32(),
            ),
        })

        conn.register("_data_arrow", data_table)
        conn.execute("""
            INSERT INTO X_HyS_data (
                id,
                atlas_cell_id,
                atlas_gene_id,
                data
            )
            SELECT
                id,
                atlas_cell_id,
                atlas_gene_id,
                data
            FROM _data_arrow
        """)
        conn.unregister("_data_arrow")

        global_data_id += nnz
        global_indptr_offset += nnz

    return global_indptr_id, global_indptr_offset, global_data_id


# 导入 obsm
def _add_obsm_from_h5ad(h5ad_path: PathLike[str] | str, atlas: Atlas, cells_per_block: int = 500):

    """将数据写入 Atlas 数据库。

    该内部函数属于数据导入模块，用于支撑同一模块中的公共 API。

    把 h5ad、AnnData 或 Scanpy 支持的数据格式写入 Atlas 的
    ``obs``、``var``、``X_HyS_*``、``obsm`` 和 ``varm`` 表。

    它通常不会作为用户入口直接调用；直接调用时需要保证输入对象、数据库连接和相关临时表已经由上游步骤准备好。

    当前实现中会访问或生成的关键表包括：``obsm_df``、``obsm_grp``。

    Parameters
    ----------
    h5ad_path
        h5ad 文件路径。

    atlas
        Atlas 对象。通常要求已经连接数据库，并包含该函数所需的 ``obs``、``var``、``X_HyS_data`` 或
        embedding 结果表。

    cells_per_block
        每批读取、写入或处理的细胞数量；较大值通常更快但占用更多内存。

    Notes
    -----
    这是内部 helper；除非需要扩展 scAtlasPy 内部流程，一般不建议在用户代码中直接调用。
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

    """将数据写入 Atlas 数据库。

    该内部函数属于数据导入模块，用于支撑同一模块中的公共 API。

    把 h5ad、AnnData 或 Scanpy 支持的数据格式写入 Atlas 的
    ``obs``、``var``、``X_HyS_*``、``obsm`` 和 ``varm`` 表。

    它通常不会作为用户入口直接调用；直接调用时需要保证输入对象、数据库连接和相关临时表已经由上游步骤准备好。

    当前实现中会访问或生成的关键表包括：``varm_df``、``varm_grp``。

    Parameters
    ----------
    h5ad_path
        h5ad 文件路径。

    atlas
        Atlas 对象。通常要求已经连接数据库，并包含该函数所需的 ``obs``、``var``、``X_HyS_data`` 或
        embedding 结果表。

    Notes
    -----
    这是内部 helper；除非需要扩展 scAtlasPy 内部流程，一般不建议在用户代码中直接调用。
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


''' 方法5： 顺序读取，anndata数据导入 '''
def load_anndata(adata:AnnData, atlas:Atlas):

    """将 AnnData 对象导入 Atlas 数据库。

    该函数按 AnnData 的 ``obs``、``var``、``X``、``obsm`` 和 ``varm`` 分层写入 Atlas 数据库。

    表达矩阵会写成 HyS 稀疏存储结构，元数据和 embedding 会写成对应的 DuckDB 表。

    Parameters
    ----------
    adata
        AnnData 对象。函数会读取其中的 ``obs``、``var``、``X``、``obsm`` 或 ``varm``。

    atlas
        Atlas 对象。通常要求已经连接数据库，并包含该函数所需的 ``obs``、``var``、``X_HyS_data`` 或
        embedding 结果表。

    Notes
    -----
    导入导出过程可能涉及较大的磁盘 IO 和内存占用，可根据数据规模调整 batch、chunk 或 worker 参数。

    Examples
    --------
    调用该函数：::

        sap.io.load_anndata(...)
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


# 以下函数只在 load_anndata 中使用
# 导入 obs
def _add_obs(adata:AnnData, atlas:Atlas):

    """将数据写入 Atlas 数据库。

    该内部函数属于数据导入模块，用于支撑同一模块中的公共 API。

    把 h5ad、AnnData 或 Scanpy 支持的数据格式写入 Atlas 的
    ``obs``、``var``、``X_HyS_*``、``obsm`` 和 ``varm`` 表。

    它通常不会作为用户入口直接调用；直接调用时需要保证输入对象、数据库连接和相关临时表已经由上游步骤准备好。

    当前实现中会访问或生成的关键表包括：``obs``。

    Parameters
    ----------
    adata
        AnnData 对象。函数会读取其中的 ``obs``、``var``、``X``、``obsm`` 或 ``varm``。

    atlas
        Atlas 对象。通常要求已经连接数据库，并包含该函数所需的 ``obs``、``var``、``X_HyS_data`` 或
        embedding 结果表。

    Notes
    -----
    这是内部 helper；除非需要扩展 scAtlasPy 内部流程，一般不建议在用户代码中直接调用。
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

    """将数据写入 Atlas 数据库。

    该内部函数属于数据导入模块，用于支撑同一模块中的公共 API。

    把 h5ad、AnnData 或 Scanpy 支持的数据格式写入 Atlas 的
    ``obs``、``var``、``X_HyS_*``、``obsm`` 和 ``varm`` 表。

    它通常不会作为用户入口直接调用；直接调用时需要保证输入对象、数据库连接和相关临时表已经由上游步骤准备好。

    当前实现中会访问或生成的关键表包括：``var``。

    Parameters
    ----------
    adata
        AnnData 对象。函数会读取其中的 ``obs``、``var``、``X``、``obsm`` 或 ``varm``。

    atlas
        Atlas 对象。通常要求已经连接数据库，并包含该函数所需的 ``obs``、``var``、``X_HyS_data`` 或
        embedding 结果表。

    Notes
    -----
    这是内部 helper；除非需要扩展 scAtlasPy 内部流程，一般不建议在用户代码中直接调用。
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

    """将数据写入 Atlas 数据库。

    该内部函数属于数据导入模块，用于支撑同一模块中的公共 API。

    把 h5ad、AnnData 或 Scanpy 支持的数据格式写入 Atlas 的
    ``obs``、``var``、``X_HyS_*``、``obsm`` 和 ``varm`` 表。

    它通常不会作为用户入口直接调用；直接调用时需要保证输入对象、数据库连接和相关临时表已经由上游步骤准备好。

    当前实现中会访问或生成的关键表包括：``obsm_df``。

    Parameters
    ----------
    adata
        AnnData 对象。函数会读取其中的 ``obs``、``var``、``X``、``obsm`` 或 ``varm``。

    atlas
        Atlas 对象。通常要求已经连接数据库，并包含该函数所需的 ``obs``、``var``、``X_HyS_data`` 或
        embedding 结果表。

    Notes
    -----
    这是内部 helper；除非需要扩展 scAtlasPy 内部流程，一般不建议在用户代码中直接调用。
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

    """将数据写入 Atlas 数据库。

    该内部函数属于数据导入模块，用于支撑同一模块中的公共 API。

    把 h5ad、AnnData 或 Scanpy 支持的数据格式写入 Atlas 的
    ``obs``、``var``、``X_HyS_*``、``obsm`` 和 ``varm`` 表。

    它通常不会作为用户入口直接调用；直接调用时需要保证输入对象、数据库连接和相关临时表已经由上游步骤准备好。

    当前实现中会访问或生成的关键表包括：``varm_df``。

    Parameters
    ----------
    adata
        AnnData 对象。函数会读取其中的 ``obs``、``var``、``X``、``obsm`` 或 ``varm``。

    atlas
        Atlas 对象。通常要求已经连接数据库，并包含该函数所需的 ``obs``、``var``、``X_HyS_data`` 或
        embedding 结果表。

    Notes
    -----
    这是内部 helper；除非需要扩展 scAtlasPy 内部流程，一般不建议在用户代码中直接调用。
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

    """将数据写入 Atlas 数据库。

    该内部函数属于数据导入模块，用于支撑同一模块中的公共 API。

    把 h5ad、AnnData 或 Scanpy 支持的数据格式写入 Atlas 的
    ``obs``、``var``、``X_HyS_*``、``obsm`` 和 ``varm`` 表。

    它通常不会作为用户入口直接调用；直接调用时需要保证输入对象、数据库连接和相关临时表已经由上游步骤准备好。

    当前实现中会访问或生成的关键表包括：``X_HyS_data``、``X_HyS_indptr``。

    Parameters
    ----------
    adata
        AnnData 对象。函数会读取其中的 ``obs``、``var``、``X``、``obsm`` 或 ``varm``。

    atlas
        Atlas 对象。通常要求已经连接数据库，并包含该函数所需的 ``obs``、``var``、``X_HyS_data`` 或
        embedding 结果表。

    chunk_size
        分块处理大小，用于控制内存峰值和单次 SQL 更新规模。

    Returns
    -------
    result
        函数返回结果。具体类型取决于参数设置和内部执行路径。

    Notes
    -----
    这是内部 helper；除非需要扩展 scAtlasPy 内部流程，一般不建议在用户代码中直接调用。
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
            data REAL                 --  float 32 单精度浮点数（4字节）
        )
        """
    )

    conn.execute("BEGIN TRANSACTION")

    try:
        total_chunks = (n_cells + chunk_size - 1) // chunk_size

        global_data_counter = np.int64(0)
        global_indptr_offset = np.int64(0)

        for chunk_idx in tqdm(range(total_chunks), desc="读取进度"):
            start = chunk_idx * chunk_size
            end = min(start + chunk_size, n_cells)
            size = end - start

            # === 取数据 ===
            X_chunk = adata.X[start:end]

            if sparse.issparse(X_chunk):
                csr = X_chunk.tocsr()
            else:
                csr = sparse.csr_matrix(X_chunk)

            data = csr.data
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

            # ================= data 表 =================
            if len(data) > 0:
                nnz = len(data)
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
                    "data": data
                })

                conn.execute("INSERT INTO X_HyS_data SELECT * FROM data_df")

                global_data_counter += nnz

            # === 清理 ===
            del X_chunk, csr, indptr_df
            if len(data) > 0:
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


''' 基因名清洗 ：先导入，再清洗，var表 '''
def gene_names_duplicated(atlas: Atlas, gene_name_column: str = "atlas_gene_name"):
    """在数据库中清洗并去重基因名。

    该函数直接在 ``var`` 表中检查重复基因名，并为第二次及以后出现的重复项添加 ``_1``、``_2`` 等后缀。

    该步骤适合在导入后执行，以保证基因名在后续绘图、差异表达和 AnnData 导出中更容易唯一定位。

    Parameters
    ----------
    atlas
        Atlas 对象。通常要求已经连接数据库，并包含该函数所需的 ``obs``、``var``、``X_HyS_data`` 或
        embedding 结果表。

    gene_name_column
        ``var`` 表中保存基因名的列名。

    Returns
    -------
    result
        函数返回结果。具体类型取决于参数设置和内部执行路径。

    Notes
    -----
    导入导出过程可能涉及较大的磁盘 IO 和内存占用，可根据数据规模调整 batch、chunk 或 worker 参数。

    Examples
    --------
    调用该函数：::

        sap.io.gene_names_duplicated(...)
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

    print("gene_names_duplicated Done")
    return True

# 示例
# var 表
# atlas_gene_id | atlas_gene_name
# ---|--------
# 1  | TP53
# 2  | EGFR
# 3  | TP53    -- 重复基因
# 4  | BRAF
# 5  | TP53    -- 重复基因

# 添加后缀模式：为重复基因添加 _1, _2, _3 等后缀
# 更新后的var表：
# atlas_gene_id | atlas_gene_name
# ---|---------
# 1  | TP53
# 2  | EGFR
# 3  | TP53_1
# 4  | BRAF
# 5  | TP53_2
