from ..data import Atlas
import concurrent.futures as futures
import os
import logging
import h5py
import pyarrow as pa
import pyarrow.parquet as pq
import time
import gc
import zlib
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from scipy import sparse
from tqdm import tqdm
# 获取日志记录器
logger = logging.getLogger('Atlas')


''' 方法1 ： 随机读取 , 多个大文件, 只支持 h5ad格式 '''
def load_h5ad_list_random(
    h5ad_paths: list[str],
    atlas,
    batch_size: int = 4096,
    pool_block_num: int = 5,    # 每次从全局随机 block 池读取多少个 block 后 flush
    store_type: str = "count",  # 目标存储类型，"count" 或 "log"
):
    """随机导入多个 h5ad 文件到 Atlas 数据库。

    该函数先把每个 h5ad 文件按 ``batch_size`` 切成连续 block，再把所有文件的 block 合并到全局 block pool
    中随机打乱。

    每次读取 ``pool_block_num`` 个 block 后，会合并为 cell pool，对 pool 内细胞整体随机一次，然后写入
    ``obs``、``var``、``X_HyS_indptr`` 和 ``X_HyS_data``。

    该策略适合多个文件大小不一致的场景，既避免 round-robin 后期只剩大文件，也减少 h5ad cell-level 随机 IO。

    Parameters
    ----------
    h5ad_paths
        一个或多个 h5ad 文件路径。

    atlas
        Atlas 对象。通常要求已经连接数据库，并包含该函数所需的 ``obs``、``var``、``X_HyS_data`` 或
        embedding 结果表。

    batch_size
        每批读取、写入或处理的细胞数量；较大值通常更快但占用更多内存。

    pool_block_num
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

        sap.io.load_h5ad_list_random(...)
    """


    # 支持单路径 / 多路径
    if isinstance(h5ad_paths, str):
        h5ad_paths = [h5ad_paths]

    if len(h5ad_paths) == 0:
        raise ValueError("h5ad_paths 不能为空")

    if pool_block_num <= 0:
        raise ValueError("pool_block_num 必须 > 0")

    commit_every = 5  # 每多少次 pool flush 提交一次
    gc_every = 5    # 每多少次 pool flush 做一次 gc

    # 检查目标存储类型
    if store_type not in {"count", "log"}:
        raise ValueError(
            f"store_type 只能是 'count' 或 'log'，当前为: {store_type}"
        )

    file_num = len(h5ad_paths)

    rng = np.random.default_rng()

    #  1️⃣ 连接数据库
    conn = atlas.connect("r+")
    atlas.connection = conn

    # 全局游标
    global_cell_id = 0
    global_indptr_id = 0
    global_indptr_offset = 0
    global_data_id = 0

    var_written = False

    print("\n==== load_h5ad_list_random ====")
    print(f"[INFO] 文件数量: {file_num:,}")
    print(f"[INFO] batch_size: {batch_size:,}")
    print(f"[INFO] pool_block_num: {pool_block_num:,}")
    print(f"[INFO] store_type: {store_type}")  # ✅ 新增
    print("[INFO] 策略：全局 block 索引池随机打乱，每次读取 pool_block_num 个 block 后 cell_pool 整体随机写入")
    print("[INFO] 注意：不再做单个 block 内部 cell 随机，只做 cell_pool 整体随机")

    # =====================================================
    # ✅ 修改 3：打开所有 h5ad，并构建全局 block 索引池
    # =====================================================
    file_states = []

    ref_var_names = None
    ref_n_genes = None

    # 全局 block 索引池 ; 每个元素记录一个 block 来自哪个文件、起止位置是多少
    all_block_refs = []

    try:
        for file_idx, h5ad_path in enumerate(h5ad_paths):

            print("\n" + "=" * 80)
            print(f"[FILE {file_idx + 1}/{file_num}] {h5ad_path}")

            adata_backed = sc.read_h5ad(h5ad_path, backed="r")

            n_cells = adata_backed.n_obs
            n_genes = adata_backed.n_vars

            print(f"[INFO] 当前文件维度: {n_cells:,} × {n_genes:,}")

            # =====================================================
            # ✅ 新增：每个文件单独检测 X 是 count 还是 log
            # =====================================================
            source_store_type = _detect_X_store_type_from_backed(
                adata_backed,
                sample_n=1000,
            )

            print(f"[INFO] 当前文件 X 判断为: {source_store_type}")

            if source_store_type == store_type:
                print("[INFO] 当前文件 X 不需要转换，直接写入。")
            else:
                print(f"[INFO] 当前文件 X 将在读取 block 后转换: {source_store_type} -> {store_type}")

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

            # ---------------- 每个文件内部按 batch_size 切 block ----------------
            block_starts = np.arange(0, n_cells, batch_size, dtype=np.int64)

            # 把所有文件的 block 放进 all_block_refs，最后统一全局 shuffle。
            for block_start in block_starts:
                block_start = int(block_start)
                block_end = min(block_start + batch_size, n_cells)

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

            print(f"[INFO] batch block 数量: {len(block_starts):,}")

        # 全局 block 索引池统一随机打乱
        total_blocks = len(all_block_refs)

        if total_blocks == 0:
            raise ValueError("所有 h5ad 文件的 cell 数量为 0，无法导入")

        rng.shuffle(all_block_refs)

        print("\n" + "=" * 80)
        print(f"[INFO] 全局 block 总数: {total_blocks:,}")
        print(f"[INFO] 每次 flush 读取 block 数: {pool_block_num:,}")
        print(f"[INFO] 预计 flush 次数: {(total_blocks + pool_block_num - 1) // pool_block_num:,}")

        # 动态建表：只用第一个文件建表
        first_backed = file_states[0]["adata_backed"]

        _create_obs_table_from_adata(conn, first_backed[:1])
        _create_var_table_from_adata(conn, first_backed[:1])
        _create_HyS_tables(conn)

        # 多个 block 读到的 batch 合并后，整体随机，然后写入数据库
        def _flush_cell_pool(cell_pool, flush_i: int):

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
            ) = _append_X_HyS(
                pool_adata,
                conn,
                base_cell_id=global_cell_id - pool_adata.n_obs,
                global_indptr_id=global_indptr_id,
                global_indptr_offset=global_indptr_offset,
                global_data_id=global_data_id,
            )

            t_write = time.time() - t1

            print(
                f"\n[flush {flush_i}] "
                f"pool_blocks={len(cell_pool):,}, "
                f"pool_cells={total_pool_cells:,}, "
                f"pool_nnz={total_pool_nnz:,}, "
                f"shuffle={t_shuffle:.2f}s, "
                f"write={t_write:.2f}s, "
                f"total_cells={global_cell_id:,}, "
                f"total_nnz={global_data_id:,}"
            )

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

        # 主循环：全局 block list 随机后，每次读取 pool_block_num 个 block
        processed_blocks = 0
        flush_counter = 0

        pbar = tqdm(total=total_blocks, desc="Global block-shuffle import")

        # 事务：多个 flush 共用事务
        conn.execute("BEGIN TRANSACTION")

        try:
            block_cursor = 0

            while block_cursor < total_blocks:

                # 每次从全局随机 block list 中取 pool_block_num 个 block
                block_group = all_block_refs[block_cursor:block_cursor + pool_block_num]
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
                    adata = _convert_X_store_type_inplace(
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

                        print(
                            f"\n[read block {processed_blocks}/{total_blocks}] "
                            f"file={state['file_idx'] + 1}, "
                            f"range=[{block_start:,}, {block_end:,}), "
                            f"cells={adata.n_obs:,}, "
                            f"nnz={block_nnz:,}, "
                            f"read={t_read:.2f}s, "
                            f"pool_blocks={len(cell_pool):,}"
                        )

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


        print("\n✔ 多文件全部成功导入 DuckDB（global block shuffle + cell_pool 随机）")
        print(f"  - files: {file_num:,}")
        print(f"  - cells: {global_cell_id:,}")
        print(f"  - genes: {ref_n_genes:,}")
        print(f"  - nnz:   {global_data_id:,}")
        print(f"  - blocks:{total_blocks:,}")
        print(f"  - flush: {flush_counter:,}")

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
def load_h5ad_random(
    h5ad_path: str,
    atlas,
    batch_size: int = 4096,
    shuffle_window_batches: int = 5,   # 固定窗口级随机，默认 5 个 batch
    store_type: str = "count",  # 目标存储类型，"count" 或 "log"
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

    batch_size
        每批读取、写入或处理的细胞数量；较大值通常更快但占用更多内存。

    shuffle_window_batches
        单文件随机导入时每个 shuffle window 包含的 block 数量。

    store_type
        目标表达矩阵尺度，通常为 ``"count"`` 或 ``"log"``。

    Notes
    -----
    导入导出过程可能涉及较大的磁盘 IO 和内存占用，可根据数据规模调整 batch、chunk 或 worker 参数。

    Examples
    --------
    调用该函数：::

        sap.io.load_h5ad_random(...)
    """

    t_start= time.time()

    # =====================================================
    # ✅ 修改 2：内部自动派生 commit_every / gc_every
    # =====================================================
    commit_every = 5
    gc_every = 10
    # commit_every = shuffle_window_batches * 2
    # gc_every = shuffle_window_batches * 4

    # 1️⃣ 连接数据库
    conn = atlas.connect("r+")
    atlas.connection = conn

    # 全局游标
    global_cell_id = 0
    global_indptr_id = 0
    global_indptr_offset = 0
    global_data_id = 0

    var_written = False

    # 2️⃣ backed 打开
    adata_backed = sc.read_h5ad(h5ad_path, backed="r")
    n_cells = adata_backed.n_obs

    # 检查目标存储类型
    if store_type not in {"count", "log"}:
        raise ValueError(
            f"store_type 只能是 'count' 或 'log'，当前为: {store_type}"
        )

    # 预读取 1000 个细胞，判断文件里的 X 是 count 还是 log
    source_store_type = _detect_X_store_type_from_backed(
        adata_backed,
        sample_n=1000,
    )

    print(f"[INFO] 文件中 X 判断为: {source_store_type}")
    print(f"[INFO] 目标存储类型 store_type = {store_type}")

    if source_store_type == store_type:
        print("[INFO] X 数据不需要转换，直接写入。")
    else:
        print(f"[INFO] X 数据将在写入前转换: {source_store_type} -> {store_type}")

    # 5 个 block 合并后再统一随机
    block_starts = np.arange(0, n_cells, batch_size, dtype=np.int64)
    np.random.shuffle(block_starts)

    # 3️⃣ 动态建表
    _create_obs_table_from_adata(conn, adata_backed[:1])
    _create_var_table_from_adata(conn, adata_backed[:1])
    _create_HyS_tables(conn)

    print(f"[INFO] 数据集维度: {adata_backed.n_obs:,} × {adata_backed.n_vars:,}")
    print("[INFO] 使用 scanpy backed + shuffle-window 随机读取")
    print(f"[INFO] batch_size = {batch_size:,}")
    print(f"[INFO] shuffle_window_batches = {shuffle_window_batches:,}")
    print(f"[INFO] batch block 数量 = {len(block_starts):,}")
    print(f"[INFO] commit_every = {commit_every:,} batches")
    print(f"[INFO] gc_every = {gc_every:,} batches")

    # 窗口缓存: 每次攒够 shuffle_window_batches 个 batch 再统一随机写入
    window_adatas = []
    window_batch_count = 0
    total_batch_counter = 0
    window_counter = 0

    conn.execute("BEGIN TRANSACTION")

    try:
        for block_i, block_start in enumerate(
            tqdm(
                block_starts,
                desc="Shuffle-window 随机读取",
            )
        ):
            block_end = min(int(block_start) + batch_size, n_cells)

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
                print(
                    f"\n[read block {block_i}] "
                    f"cells={adata.n_obs:,}, "
                    f"nnz={block_nnz:,}, "
                    f"read={t_read:.2f}s, "
                    f"window_batches={window_batch_count}/{shuffle_window_batches}"
                )

            # window 满了，统一随机 + 统一写入
            if window_batch_count >= shuffle_window_batches:
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

                print(
                    f"\n[write window {window_counter}] "
                    f"batches={window_batch_count}, "
                    f"cells={window_cells:,}, "
                    f"nnz={window_nnz:,}, "
                    f"write={t_write:.2f}s, "
                    f"total_cells={global_cell_id:,}, "
                    f"total_nnz={global_data_id:,}"
                )

                # 清空 window
                for x in window_adatas:
                    del x
                window_adatas.clear()
                window_batch_count = 0

                # 每 commit_every 个 batch 提交一次,等价于 shuffle_window_batches=5 时，每 2 个 window commit
                if total_batch_counter % commit_every == 0:
                    conn.execute("COMMIT")
                    conn.execute("BEGIN TRANSACTION")
                    print(f"[COMMIT] processed_batches={total_batch_counter:,}")

                # 每 gc_every 个 batch gc 一次
                if total_batch_counter % gc_every == 0:
                    gc.collect()
                    print(f"[GC] processed_batches={total_batch_counter:,}")

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
                source_store_type=source_store_type,  # ✅ 修改
                target_store_type=store_type,  # ✅ 修改
            )

            t_write = time.time() - t1

            total_batch_counter += window_batch_count
            window_counter += 1

            print(
                f"\n[write final window {window_counter}] "
                f"batches={window_batch_count}, "
                f"cells={window_cells:,}, "
                f"nnz={window_nnz:,}, "
                f"write={t_write:.2f}s, "
                f"total_cells={global_cell_id:,}, "
                f"total_nnz={global_data_id:,}"
            )

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
    print( f"total time: {t_end-t_start:.2f} seconds ")

    print("✔ 全部数据成功导入 DuckDB（shuffle-window 随机导入）")
    print(f"  - cells: {global_cell_id:,}")
    print(f"  - nnz:   {global_data_id:,}")


def load_h5ad_fast(
    h5ad_path: str,
    atlas,
    batch_size: int = 4096,
    shuffle_window_batches: int = 5,
    store_type: str = "count",
    staging_dir: str | None = None,
    parquet_compression: str = "NONE",
    parquet_workers: int = 2,
    h5ad_read_workers: int = 2,
    keep_staging: bool = False,
):
    """使用 Parquet staging 加速导入单个 h5ad 文件。

    该函数采用与 ``load_h5ad_random`` 相同的 shuffle-window 策略，但将 ``X_HyS_indptr`` 和
    ``X_HyS_data`` 先写入 Parquet 分片。

    所有 window 处理完成后，再由 DuckDB 使用 ``read_parquet`` 一次性物化表达矩阵表，以减少大量逐 window
    INSERT 的开销。

    当 h5ad 的 ``X`` 是受支持的 gzip-compressed CSR 布局时，还可以用并行 raw chunk reader 加速读取。

    Parameters
    ----------
    h5ad_path
        h5ad 文件路径。

    atlas
        Atlas 对象。通常要求已经连接数据库，并包含该函数所需的 ``obs``、``var``、``X_HyS_data`` 或
        embedding 结果表。

    batch_size
        每批读取、写入或处理的细胞数量；较大值通常更快但占用更多内存。

    shuffle_window_batches
        单文件随机导入时每个 shuffle window 包含的 block 数量。

    store_type
        目标表达矩阵尺度，通常为 ``"count"`` 或 ``"log"``。

    staging_dir
        Parquet staging 临时目录。

    parquet_compression
        Parquet 文件压缩方式。

    parquet_workers
        并行写 Parquet 分片的 worker 数量。

    h5ad_read_workers
        并行读取 h5ad raw gzip chunk 的 worker 数量。

    keep_staging
        导入完成后是否保留 Parquet staging 临时文件。

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

        sap.io.load_h5ad_fast(...)
    """

    t_start=time.time()

    if store_type not in {"count", "log"}:
        raise ValueError(f"store_type 只能是 'count' 或 'log'，当前为: {store_type}")
    if parquet_workers <= 0:
        raise ValueError("parquet_workers 必须 > 0")
    if h5ad_read_workers <= 0:
        raise ValueError("h5ad_read_workers 必须 > 0")

    conn = atlas.connect("r+")
    atlas.connection = conn
    conn.execute("PRAGMA threads=10")
    conn.execute("PRAGMA preserve_insertion_order=false")

    # X_HyS is staged outside DuckDB first, then loaded with DuckDB's
    # parallel Parquet scanner. Keeping indptr/data in separate folders
    # mirrors the final table layout and keeps each read_parquet call simple.
    if staging_dir is None:
        staging_dir = f"{atlas.file_path}.hys_parquet_staging"
    staging_dir = os.path.abspath(staging_dir)
    indptr_dir = os.path.join(staging_dir, "X_HyS_indptr")
    data_dir = os.path.join(staging_dir, "X_HyS_data")

    if os.path.exists(staging_dir):
        import shutil

        shutil.rmtree(staging_dir)
    os.makedirs(indptr_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    conn.execute(f"PRAGMA temp_directory='{staging_dir}'")

    global_cell_id = 0
    global_indptr_id = 0
    global_indptr_offset = 0
    global_data_id = 0
    var_written = False

    adata_backed = sc.read_h5ad(h5ad_path, backed="r")
    n_cells = adata_backed.n_obs

    # Optional fast path for gzip-compressed CSR h5ad files. If the file layout
    # is not exactly supported, this returns None and the scanpy backed path is
    # used without changing caller behavior.
    x_reader = _ParallelH5adCSRReader.open_if_supported(
        h5ad_path,
        n_vars=adata_backed.n_vars,
        workers=h5ad_read_workers,
    )

    source_store_type = _detect_X_store_type_from_backed(adata_backed, sample_n=1000)
    source_var = adata_backed.var.copy()
    print(f"[INFO] 文件中 X 判断为: {source_store_type}")
    print(f"[INFO] 目标存储类型 store_type = {store_type}")
    if source_store_type == store_type:
        print("[INFO] X 数据不需要转换，直接写入。")
    else:
        print(f"[INFO] X 数据将在写入前转换: {source_store_type} -> {store_type}")

    # Randomize at block granularity: each block is read as a contiguous cell
    # range, then several blocks are merged and shuffled within a window.
    block_starts = np.arange(0, n_cells, batch_size, dtype=np.int64)
    np.random.shuffle(block_starts)

    _create_obs_table_from_adata(conn, adata_backed[:1])
    _create_var_table_from_adata(conn, adata_backed[:1])

    print(f"[INFO] 数据集维度: {adata_backed.n_obs:,} × {adata_backed.n_vars:,}")
    print("[INFO] 使用 scanpy backed + shuffle-window + Parquet staging")
    print(f"[INFO] batch_size = {batch_size:,}")
    print(f"[INFO] shuffle_window_batches = {shuffle_window_batches:,}")
    print(f"[INFO] batch block 数量 = {len(block_starts):,}")
    print(f"[INFO] staging_dir = {staging_dir}")
    print(f"[INFO] parquet_compression = {parquet_compression}")
    print(f"[INFO] parquet_workers = {parquet_workers:,}")
    print(f"[INFO] h5ad_read_workers = {h5ad_read_workers:,}")
    if x_reader is not None:
        print("[INFO] h5ad X 使用并行 raw gzip chunk reader")
    elif h5ad_read_workers > 1:
        print("[INFO] h5ad X 不满足并行 raw reader 条件，回退到 scanpy backed 读取")

    window_adatas = []
    window_batch_count = 0
    window_counter = 0
    parquet_part_counter = 0
    total_batch_counter = 0
    indptr_writer = None
    data_writer = None
    parquet_shards = (
        [
            _XHySParquetShardWriter(
                indptr_dir=indptr_dir,
                data_dir=data_dir,
                part_id=part_id,
                parquet_compression=parquet_compression,
            )
            for part_id in range(parquet_workers)
        ]
        if parquet_workers > 1
        else None
    )
    parquet_futures = []

    def read_h5ad_block(block_start: int, block_end: int):
        """执行 ``read_h5ad_block`` 的核心功能。

        把 h5ad、AnnData 或 Scanpy 支持的数据格式写入 Atlas 的
        ``obs``、``var``、``X_HyS_*``、``obsm`` 和 ``varm`` 表。

        函数会直接读取或写入 Atlas 数据库中的相关表，并尽量通过 SQL、分块读取或流式计算减少内存占用。

        整体用法和 Scanpy 中相近的 ``sap.io.read_h5ad_block`` 风格 API 类似，但结果保存在 Atlas
        数据库表中，便于后续步骤复用。

        当前实现中会访问或生成的关键表包括：``obs``、``var``。

        Parameters
        ----------
        block_start
            ``block_start`` 参数。用于控制该函数对应步骤的输入、输出或运行方式。

        block_end
            ``block_end`` 参数。用于控制该函数对应步骤的输入、输出或运行方式。

        Returns
-------
        result
            导出的对象、读取后的 AnnData/DataFrame，或文件写入结果。

        Notes
        -----
        导入导出过程可能涉及较大的磁盘 IO 和内存占用，可根据数据规模调整 batch、chunk 或 worker 参数。

        Examples
        --------
        调用该函数：::

            sap.io.read_h5ad_block(...)
        """
        if x_reader is None:
            return adata_backed[block_start:block_end].to_memory()
        X = x_reader.read_rows(block_start, block_end)
        return AnnData(
            X=X,
            obs=adata_backed.obs.iloc[block_start:block_end].copy(),
            var=source_var.copy(deep=False),
        )

    def close_parquet_writers() -> None:
        """执行 ``close_parquet_writers`` 的核心功能。

        把 h5ad、AnnData 或 Scanpy 支持的数据格式写入 Atlas 的
        ``obs``、``var``、``X_HyS_*``、``obsm`` 和 ``varm`` 表。

        函数会直接读取或写入 Atlas 数据库中的相关表，并尽量通过 SQL、分块读取或流式计算减少内存占用。

        整体用法和 Scanpy 中相近的 ``sap.io.close_parquet_writers`` 风格 API 类似，但结果保存在 Atlas
        数据库表中，便于后续步骤复用。

        Notes
        -----
        导入导出过程可能涉及较大的磁盘 IO 和内存占用，可根据数据规模调整 batch、chunk 或 worker 参数。

        Examples
        --------
        调用该函数：::

            sap.io.close_parquet_writers(...)
        """
        nonlocal indptr_writer, data_writer
        if indptr_writer is not None:
            indptr_writer.close()
            indptr_writer = None
        if data_writer is not None:
            data_writer.close()
            data_writer = None

    def append_parquet_tables(indptr_table, data_table) -> None:
        """执行 ``append_parquet_tables`` 的核心功能。

        把 h5ad、AnnData 或 Scanpy 支持的数据格式写入 Atlas 的
        ``obs``、``var``、``X_HyS_*``、``obsm`` 和 ``varm`` 表。

        函数会直接读取或写入 Atlas 数据库中的相关表，并尽量通过 SQL、分块读取或流式计算减少内存占用。

        整体用法和 Scanpy 中相近的 ``sap.io.append_parquet_tables`` 风格 API 类似，但结果保存在 Atlas
        数据库表中，便于后续步骤复用。

        Parameters
        ----------
        indptr_table
            表示 ``X_HyS_indptr`` 的 Arrow Table。

        data_table
            表示 ``X_HyS_data`` 的 Arrow Table。

        Notes
        -----
        导入导出过程可能涉及较大的磁盘 IO 和内存占用，可根据数据规模调整 batch、chunk 或 worker 参数。

        Examples
        --------
        调用该函数：::

            sap.io.append_parquet_tables(...)
        """
        nonlocal indptr_writer, data_writer, parquet_part_counter
        # In parallel mode each worker owns one Parquet shard. Windows are
        # distributed round-robin, so every shard appends row groups in order
        # while different shards write concurrently.
        if parquet_shards is not None:
            shard = parquet_shards[parquet_part_counter % len(parquet_shards)]
            parquet_part_counter += 1
            parquet_futures.append(shard.submit(indptr_table, data_table))
            return

        # Single-writer mode appends all windows as row groups in one Parquet
        # shard, avoiding a large in-memory staging buffer.
        if indptr_writer is None:
            indptr_path = os.path.join(indptr_dir, f"part_{parquet_part_counter:06d}.parquet")
            data_path = os.path.join(data_dir, f"part_{parquet_part_counter:06d}.parquet")
            indptr_writer = pq.ParquetWriter(
                indptr_path,
                indptr_table.schema,
                compression=parquet_compression,
            )
            data_writer = pq.ParquetWriter(
                data_path,
                data_table.schema,
                compression=parquet_compression,
            )
            parquet_part_counter += 1

        indptr_writer.write_table(indptr_table)
        data_writer.write_table(data_table)

    conn.execute("BEGIN TRANSACTION")

    try:
        for block_i, block_start in enumerate(
            tqdm(block_starts, desc="Shuffle-window Parquet staging")
        ):
            block_end = min(int(block_start) + batch_size, n_cells)

            adata = read_h5ad_block(int(block_start), block_end)

            block_nnz = adata.X.nnz if sparse.issparse(adata.X) else np.count_nonzero(adata.X)
            window_adatas.append(adata)
            window_batch_count += 1

            if (block_i + 1) % 20 == 0 or block_i == 0:
                print(
                    f"\n[read block {block_i}] "
                    f"cells={adata.n_obs:,}, nnz={block_nnz:,}, "
                    f"window_batches={window_batch_count}/{shuffle_window_batches}"
                )

            if window_batch_count >= shuffle_window_batches:
                # Build one shuffled business window, append obs/var to DuckDB,
                # and hand off X_HyS Arrow tables to the Parquet staging layer.
                (
                    global_cell_id,
                    global_indptr_id,
                    global_indptr_offset,
                    global_data_id,
                    var_written,
                    window_cells,
                    window_nnz,
                    indptr_table,
                    data_table,
                ) = _build_shuffle_window_for_parquet_staging(
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
                append_parquet_tables(indptr_table, data_table)
                window_counter += 1
                total_batch_counter += window_batch_count

                print(
                    f"\n[stage window {window_counter}] "
                    f"batches={window_batch_count}, cells={window_cells:,}, "
                    f"nnz={window_nnz:,}, "
                    f"total_cells={global_cell_id:,}, total_nnz={global_data_id:,}"
                )

                for x in window_adatas:
                    del x
                window_adatas.clear()
                window_batch_count = 0

                if total_batch_counter % 10 == 0:
                    gc.collect()

        if window_batch_count > 0:
            # Flush the tail window when the number of blocks is not an exact
            # multiple of shuffle_window_batches.
            (
                global_cell_id,
                global_indptr_id,
                global_indptr_offset,
                global_data_id,
                var_written,
                window_cells,
                window_nnz,
                indptr_table,
                data_table,
            ) = _build_shuffle_window_for_parquet_staging(
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
            append_parquet_tables(indptr_table, data_table)
            window_counter += 1

            print(
                f"\n[stage final window {window_counter}] "
                f"batches={window_batch_count}, cells={window_cells:,}, "
                f"nnz={window_nnz:,}, "
                f"total_cells={global_cell_id:,}, total_nnz={global_data_id:,}"
            )

            for x in window_adatas:
                del x
            window_adatas.clear()
            window_batch_count = 0
            gc.collect()

        conn.execute("COMMIT")
        close_parquet_writers()
        if parquet_shards is not None:
            for parquet_future in futures.as_completed(parquet_futures):
                parquet_future.result()
            for shard in parquet_shards:
                shard.close()

        # Materialize staged Parquet parts into DuckDB tables. This lets DuckDB
        # use its native Parquet scanner instead of many incremental INSERTs.
        conn.execute(f"""
            CREATE OR REPLACE TABLE X_HyS_indptr AS
            SELECT * FROM read_parquet('{indptr_dir}/*.parquet')
        """)
        conn.execute(f"""
            CREATE OR REPLACE TABLE X_HyS_data AS
            SELECT * FROM read_parquet('{data_dir}/*.parquet')
        """)

    except Exception:
        close_parquet_writers()
        if parquet_shards is not None:
            for shard in parquet_shards:
                shard.close(cancel_futures=True)
        conn.execute("ROLLBACK")
        try:
            for x in window_adatas:
                del x
            window_adatas.clear()
        except Exception:
            pass
        try:
            adata_backed.file.close()
        except Exception:
            pass
        if x_reader is not None:
            x_reader.close()
        gc.collect()
        raise

    conn.execute("ALTER TABLE obs ADD PRIMARY KEY (atlas_cell_id)")
    conn.execute("ALTER TABLE var ADD PRIMARY KEY (atlas_gene_id)")
    _add_varm_from_h5ad(h5ad_path, atlas)

    try:
        conn.execute("CHECKPOINT")
    except Exception:
        pass

    try:
        adata_backed.file.close()
    except Exception:
        pass
    if x_reader is not None:
        x_reader.close()

    if not keep_staging:
        import shutil

        shutil.rmtree(staging_dir, ignore_errors=True)

    gc.collect()

    t_end = time.time()
    print( f"total time: {t_end-t_start:.2f} seconds ")

    print("✔ 全部数据成功导入 DuckDB（shuffle-window + Parquet staging）")
    print(f"  - cells: {global_cell_id:,}")
    print(f"  - nnz:   {global_data_id:,}")


class _XHySParquetShardWriter:
    """X_HyS Parquet 分片写入器。

    该类属于数据导入模块，用于封装该模块中的参数、数据库连接和中间状态。

    把 h5ad、AnnData 或 Scanpy 支持的数据格式写入 Atlas 的
    ``obs``、``var``、``X_HyS_*``、``obsm`` 和 ``varm`` 表。

    对象方法通常按照固定流程依次调用，用户一般通过公共入口函数或 ``run`` 方法使用。

    Parameters
    ----------
    indptr_dir
        ``X_HyS_indptr`` Parquet 分片目录。

    data_dir
        ``X_HyS_data`` Parquet 分片目录。

    part_id
        Parquet 分片编号。

    parquet_compression
        Parquet 文件压缩方式。

    Notes
    -----
    这是内部 helper；除非需要扩展 scAtlasPy 内部流程，一般不建议在用户代码中直接调用。
    """

    def __init__(
        self,
        *,
        indptr_dir: str,
        data_dir: str,
        part_id: int,
        parquet_compression: str,
    ):
        """初始化对象。

        该内部函数属于数据导入模块，用于支撑同一模块中的公共 API。

        把 h5ad、AnnData 或 Scanpy 支持的数据格式写入 Atlas 的
        ``obs``、``var``、``X_HyS_*``、``obsm`` 和 ``varm`` 表。

        它通常不会作为用户入口直接调用；直接调用时需要保证输入对象、数据库连接和相关临时表已经由上游步骤准备好。

        Parameters
        ----------
        indptr_dir
            ``X_HyS_indptr`` Parquet 分片目录。

        data_dir
            ``X_HyS_data`` Parquet 分片目录。

        part_id
            Parquet 分片编号。

        parquet_compression
            Parquet 文件压缩方式。

        Notes
        -----
        这是内部 helper；除非需要扩展 scAtlasPy 内部流程，一般不建议在用户代码中直接调用。
        """
        self._indptr_path = os.path.join(indptr_dir, f"part_{part_id:06d}.parquet")
        self._data_path = os.path.join(data_dir, f"part_{part_id:06d}.parquet")
        self._parquet_compression = parquet_compression
        self._indptr_writer = None
        self._data_writer = None
        self._pool = futures.ThreadPoolExecutor(max_workers=1)

    def submit(self, indptr_table, data_table):
        """执行 ``submit`` 的核心功能。

        把 h5ad、AnnData 或 Scanpy 支持的数据格式写入 Atlas 的
        ``obs``、``var``、``X_HyS_*``、``obsm`` 和 ``varm`` 表。

        函数会直接读取或写入 Atlas 数据库中的相关表，并尽量通过 SQL、分块读取或流式计算减少内存占用。

        整体用法和 Scanpy 中相近的 ``sap.io.submit`` 风格 API 类似，但结果保存在 Atlas 数据库表中，便于后续步骤复用。

        Parameters
        ----------
        indptr_table
            表示 ``X_HyS_indptr`` 的 Arrow Table。

        data_table
            表示 ``X_HyS_data`` 的 Arrow Table。

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

            sap.io.submit(...)
        """
        return self._pool.submit(self._write, indptr_table, data_table)

    def _write(self, indptr_table, data_table):
        """将计算结果写入数据库表。

        该内部函数属于数据导入模块，用于支撑同一模块中的公共 API。

        把 h5ad、AnnData 或 Scanpy 支持的数据格式写入 Atlas 的
        ``obs``、``var``、``X_HyS_*``、``obsm`` 和 ``varm`` 表。

        它通常不会作为用户入口直接调用；直接调用时需要保证输入对象、数据库连接和相关临时表已经由上游步骤准备好。

        Parameters
        ----------
        indptr_table
            表示 ``X_HyS_indptr`` 的 Arrow Table。

        data_table
            表示 ``X_HyS_data`` 的 Arrow Table。

        Returns
-------
        result
            函数返回结果。具体类型取决于参数设置和内部执行路径。

        Notes
        -----
        这是内部 helper；除非需要扩展 scAtlasPy 内部流程，一般不建议在用户代码中直接调用。
        """
        if self._indptr_writer is None:
            self._indptr_writer = pq.ParquetWriter(
                self._indptr_path,
                indptr_table.schema,
                compression=self._parquet_compression,
            )
            self._data_writer = pq.ParquetWriter(
                self._data_path,
                data_table.schema,
                compression=self._parquet_compression,
            )

        self._indptr_writer.write_table(indptr_table)
        self._data_writer.write_table(data_table)
        return data_table.num_rows

    def close(self, *, cancel_futures: bool = False) -> None:
        """执行 ``close`` 的核心功能。

        把 h5ad、AnnData 或 Scanpy 支持的数据格式写入 Atlas 的
        ``obs``、``var``、``X_HyS_*``、``obsm`` 和 ``varm`` 表。

        函数会直接读取或写入 Atlas 数据库中的相关表，并尽量通过 SQL、分块读取或流式计算减少内存占用。

        整体用法和 Scanpy 中相近的 ``sap.io.close`` 风格 API 类似，但结果保存在 Atlas 数据库表中，便于后续步骤复用。

        Parameters
        ----------
        cancel_futures
            关闭 writer 时是否取消尚未执行的异步写入任务。

        Notes
        -----
        导入导出过程可能涉及较大的磁盘 IO 和内存占用，可根据数据规模调整 batch、chunk 或 worker 参数。

        Examples
        --------
        调用该函数：::

            sap.io.close(...)
        """
        self._pool.shutdown(wait=True, cancel_futures=cancel_futures)
        if self._indptr_writer is not None:
            self._indptr_writer.close()
            self._indptr_writer = None
        if self._data_writer is not None:
            self._data_writer.close()
            self._data_writer = None


class _ParallelH5adCSRReader:
    """并行读取 h5ad CSR 矩阵的内部读取器。

    该类属于数据导入模块，用于封装该模块中的参数、数据库连接和中间状态。

    把 h5ad、AnnData 或 Scanpy 支持的数据格式写入 Atlas 的
    ``obs``、``var``、``X_HyS_*``、``obsm`` 和 ``varm`` 表。

    对象方法通常按照固定流程依次调用，用户一般通过公共入口函数或 ``run`` 方法使用。

    Parameters
    ----------
    h5ad_path
        h5ad 文件路径。

    n_vars
        表达矩阵中的基因数量。

    workers
        并行 worker 数量。

    Notes
    -----
    这是内部 helper；除非需要扩展 scAtlasPy 内部流程，一般不建议在用户代码中直接调用。
    """

    def __init__(self, h5ad_path: str, n_vars: int, workers: int):
        """初始化对象。

        该内部函数属于数据导入模块，用于支撑同一模块中的公共 API。

        把 h5ad、AnnData 或 Scanpy 支持的数据格式写入 Atlas 的
        ``obs``、``var``、``X_HyS_*``、``obsm`` 和 ``varm`` 表。

        它通常不会作为用户入口直接调用；直接调用时需要保证输入对象、数据库连接和相关临时表已经由上游步骤准备好。

        Parameters
        ----------
        h5ad_path
            h5ad 文件路径。

        n_vars
            表达矩阵中的基因数量。

        workers
            并行 worker 数量。

        Notes
        -----
        这是内部 helper；除非需要扩展 scAtlasPy 内部流程，一般不建议在用户代码中直接调用。
        """
        self._file = h5py.File(h5ad_path, "r")
        self._x = self._file["X"]
        self._data = self._x["data"]
        self._indices = self._x["indices"]
        self._indptr = self._x["indptr"]
        self._n_vars = n_vars
        self._pool = futures.ThreadPoolExecutor(max_workers=workers)

    @classmethod
    def open_if_supported(cls, h5ad_path: str, n_vars: int, workers: int):
        """执行 ``open_if_supported`` 的核心功能。

        把 h5ad、AnnData 或 Scanpy 支持的数据格式写入 Atlas 的
        ``obs``、``var``、``X_HyS_*``、``obsm`` 和 ``varm`` 表。

        函数会直接读取或写入 Atlas 数据库中的相关表，并尽量通过 SQL、分块读取或流式计算减少内存占用。

        整体用法和 Scanpy 中相近的 ``sap.io.open_if_supported`` 风格 API 类似，但结果保存在 Atlas
        数据库表中，便于后续步骤复用。

        Parameters
        ----------
        h5ad_path
            h5ad 文件路径。

        n_vars
            表达矩阵中的基因数量。

        workers
            并行 worker 数量。

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

            sap.io.open_if_supported(...)
        """
        if workers <= 1:
            return None
        try:
            with h5py.File(h5ad_path, "r") as f:
                # This fast path is intentionally narrow: AnnData stores sparse
                # X as a CSR group with data/indices/indptr datasets. Other
                # encodings keep using scanpy's backed reader.
                if "X" not in f or not isinstance(f["X"], h5py.Group):
                    return None
                x = f["X"]
                if x.attrs.get("encoding-type") not in {b"csr_matrix", "csr_matrix"}:
                    return None
                for name in ("data", "indices", "indptr"):
                    if name not in x:
                        return None
                data = x["data"]
                indices = x["indices"]
                # read_direct_chunk returns raw compressed bytes only for
                # chunked datasets. We currently optimize the common Tahoe case:
                # 1-D gzip chunks for both CSR data and indices.
                if not cls._dataset_supported(data) or not cls._dataset_supported(indices):
                    return None
        except Exception:
            return None
        return cls(h5ad_path, n_vars=n_vars, workers=workers)

    @staticmethod
    def _dataset_supported(ds) -> bool:
        """执行 ``_dataset_supported`` 的核心功能。

        该内部函数属于数据导入模块，用于支撑同一模块中的公共 API。

        把 h5ad、AnnData 或 Scanpy 支持的数据格式写入 Atlas 的
        ``obs``、``var``、``X_HyS_*``、``obsm`` 和 ``varm`` 表。

        它通常不会作为用户入口直接调用；直接调用时需要保证输入对象、数据库连接和相关临时表已经由上游步骤准备好。

        Parameters
        ----------
        ds
            HDF5 dataset 对象。

        Returns
-------
        result
            函数返回结果。具体类型取决于参数设置和内部执行路径。

        Notes
        -----
        这是内部 helper；除非需要扩展 scAtlasPy 内部流程，一般不建议在用户代码中直接调用。
        """
        return (
            ds.ndim == 1
            and ds.chunks is not None
            and len(ds.chunks) == 1
            and ds.compression == "gzip"
        )

    def close(self) -> None:
        """执行 ``close`` 的核心功能。

        把 h5ad、AnnData 或 Scanpy 支持的数据格式写入 Atlas 的
        ``obs``、``var``、``X_HyS_*``、``obsm`` 和 ``varm`` 表。

        函数会直接读取或写入 Atlas 数据库中的相关表，并尽量通过 SQL、分块读取或流式计算减少内存占用。

        整体用法和 Scanpy 中相近的 ``sap.io.close`` 风格 API 类似，但结果保存在 Atlas 数据库表中，便于后续步骤复用。

        Notes
        -----
        导入导出过程可能涉及较大的磁盘 IO 和内存占用，可根据数据规模调整 batch、chunk 或 worker 参数。

        Examples
        --------
        调用该函数：::

            sap.io.close(...)
        """
        self._pool.shutdown(wait=True)
        self._file.close()

    def read_rows(self, row_start: int, row_stop: int):
        """按细胞行范围读取 h5ad CSR 矩阵片段。

        该方法根据 ``indptr`` 将请求的细胞行区间转换为 ``data`` 和 ``indices`` 中连续的非零值范围，
        并并行读取 gzip 压缩 chunk，最后构造独立的 ``scipy.sparse.csr_matrix``。

        它用于大规模 h5ad 导入过程中的按块读取，避免一次性把完整表达矩阵载入内存。

        Parameters
        ----------
        row_start
            读取细胞行区间的起始位置，包含该行。

        row_stop
            读取细胞行区间的结束位置，不包含该行。

        Returns
        -------
        X
            指定细胞行范围对应的 CSR 稀疏矩阵。

            行数为 ``row_stop - row_start``，列数为 h5ad 中的基因数量。

        Notes
        -----
        返回矩阵的 ``indptr`` 会重新以 0 为起点，因此可以作为独立 CSR block 继续写入 Atlas 数据库。
        """
        # Convert the requested cell row range into a contiguous nnz range in
        # the CSR data/indices arrays. The returned indptr is rebased to zero
        # so scipy can construct a standalone CSR matrix for this block.
        indptr_abs = self._indptr[row_start : row_stop + 1].astype(np.int64, copy=False)
        nnz_start = int(indptr_abs[0])
        nnz_stop = int(indptr_abs[-1])
        indptr = indptr_abs - nnz_start

        data = self._read_1d_range(self._data, nnz_start, nnz_stop)
        indices = self._read_1d_range(self._indices, nnz_start, nnz_stop)
        return sparse.csr_matrix(
            (data, indices, indptr),
            shape=(row_stop - row_start, self._n_vars),
        )

    def _read_1d_range(self, ds, start: int, stop: int):
        """执行 ``_read_1d_range`` 的核心功能。

        该内部函数属于数据导入模块，用于支撑同一模块中的公共 API。

        把 h5ad、AnnData 或 Scanpy 支持的数据格式写入 Atlas 的
        ``obs``、``var``、``X_HyS_*``、``obsm`` 和 ``varm`` 表。

        它通常不会作为用户入口直接调用；直接调用时需要保证输入对象、数据库连接和相关临时表已经由上游步骤准备好。

        Parameters
        ----------
        ds
            HDF5 dataset 对象。

        start
            读取或处理范围的起始位置。

        stop
            读取或处理范围的结束位置。

        Returns
-------
        result
            函数返回结果。具体类型取决于参数设置和内部执行路径。

        Notes
        -----
        这是内部 helper；除非需要扩展 scAtlasPy 内部流程，一般不建议在用户代码中直接调用。
        """
        if stop <= start:
            return np.empty(0, dtype=ds.dtype)

        chunk_size = int(ds.chunks[0])
        first_chunk = start // chunk_size
        last_chunk = (stop - 1) // chunk_size
        jobs = []

        for chunk_id in range(first_chunk, last_chunk + 1):
            chunk_start = chunk_id * chunk_size
            chunk_stop = min(chunk_start + chunk_size, ds.shape[0])
            raw_offset = (chunk_start,)

            # HDF5 gzip chunks are independent compressed byte ranges. We read
            # each full compressed chunk, then slice the decompressed array to
            # the requested nnz interval; only the first/last chunks are partial.
            filter_mask, raw_chunk = ds.id.read_direct_chunk(raw_offset)
            if filter_mask != 0:
                raise RuntimeError(
                    f"Unsupported HDF5 filter mask {filter_mask} for {ds.name}"
                )
            jobs.append(
                self._pool.submit(
                    _decompress_h5ad_1d_chunk_slice,
                    raw_chunk,
                    ds.dtype,
                    chunk_size,
                    chunk_stop - chunk_start,
                    max(start, chunk_start) - chunk_start,
                    min(stop, chunk_stop) - chunk_start,
                )
            )

        if len(jobs) == 1:
            return jobs[0].result()
        return np.concatenate([job.result() for job in jobs])


def _decompress_h5ad_1d_chunk_slice(
    raw_chunk: bytes,
    dtype,
    chunk_size: int,
    valid_items: int,
    slice_start: int,
    slice_stop: int,
):
    """解压并截取 h5ad 中的一维 gzip chunk。

    该内部函数处理 ``data`` 或 ``indices`` 这类一维 HDF5 dataset 的原始压缩 chunk。

    函数先使用 ``zlib.decompress`` 解压完整 chunk，再根据当前请求范围截取其中需要的片段，并返回拷贝后的
    NumPy 数组。它通常由线程池并行调用，用于加速大规模 h5ad CSR 数据读取。

    Parameters
    ----------
    raw_chunk
        从 HDF5 文件中读取到的原始压缩 chunk 字节。

    dtype
        解压后数组的数据类型。

    chunk_size
        HDF5 dataset 的标准 chunk 长度。

    valid_items
        当前 chunk 中实际有效的元素数量。

        最后一个 chunk 可能小于 ``chunk_size``，因此需要用该参数截断填充区域。

    slice_start
        当前 chunk 内需要返回片段的起始位置。

    slice_stop
        当前 chunk 内需要返回片段的结束位置。

    Returns
    -------
    values
        解压并切片后的 NumPy 一维数组。

    Notes
    -----
    该函数是 h5ad 快速导入流程的内部 helper；用户通常不需要直接调用。
    """
    # zlib.decompress releases the GIL, so ThreadPoolExecutor can parallelize
    # gzip chunk decompression without copying through h5py's filter pipeline.
    array = np.frombuffer(zlib.decompress(raw_chunk), dtype=dtype)
    if valid_items != chunk_size:
        array = array[:valid_items]
    return array[slice_start:slice_stop].copy()


def _build_shuffle_window_for_parquet_staging(
    window_adatas,
    conn,
    global_cell_id: int,
    global_indptr_id: int,
    global_indptr_offset: int,
    global_data_id: int,
    var_written: bool,
    source_store_type: str,
    target_store_type: str,
):
    """构建内部中间数据结构。

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
        构建得到的内部对象，通常是 DataFrame、Arrow Table 或更新后的游标元组。

    Notes
    -----
    这是内部 helper；除非需要扩展 scAtlasPy 内部流程，一般不建议在用户代码中直接调用。
    """
    adata_window = sc.concat(
        window_adatas,
        axis=0,
        join="outer",
        merge="first",
        index_unique=None,
    )

    if adata_window.n_obs > 1:
        perm = np.random.permutation(adata_window.n_obs)
        adata_window = adata_window[perm].copy()

    window_cells = adata_window.n_obs
    window_nnz = adata_window.X.nnz if sparse.issparse(adata_window.X) else np.count_nonzero(adata_window.X)

    global_cell_id = _append_obs_rows(
        adata_window,
        conn,
        start_cell_id=global_cell_id,
    )

    if not var_written:
        _append_var(adata_window, conn)
        var_written = True

    adata_window = _convert_X_store_type_inplace(
        adata_window,
        source_store_type=source_store_type,
        target_store_type=target_store_type,
    )

    (
        indptr_table,
        data_table,
        global_indptr_id,
        global_indptr_offset,
        global_data_id,
    ) = _build_X_HyS_arrow_tables(
        adata_window,
        base_cell_id=global_cell_id - adata_window.n_obs,
        global_indptr_id=global_indptr_id,
        global_indptr_offset=global_indptr_offset,
        global_data_id=global_data_id,
    )

    del adata_window

    return (
        global_cell_id,
        global_indptr_id,
        global_indptr_offset,
        global_data_id,
        var_written,
        window_cells,
        window_nnz,
        indptr_table,
        data_table,
    )


def _build_X_HyS_arrow_tables(
    adata,
    *,
    base_cell_id: int,
    global_indptr_id: int,
    global_indptr_offset: int,
    global_data_id: int,
):
    """构建内部中间数据结构。

    该内部函数属于数据导入模块，用于支撑同一模块中的公共 API。

    把 h5ad、AnnData 或 Scanpy 支持的数据格式写入 Atlas 的
    ``obs``、``var``、``X_HyS_*``、``obsm`` 和 ``varm`` 表。

    它通常不会作为用户入口直接调用；直接调用时需要保证输入对象、数据库连接和相关临时表已经由上游步骤准备好。

    Parameters
    ----------
    adata
        AnnData 对象。函数会读取其中的 ``obs``、``var``、``X``、``obsm`` 或 ``varm``。

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
        构建得到的内部对象，通常是 DataFrame、Arrow Table 或更新后的游标元组。

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
    row_nnz = np.diff(indptr)
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
        "indptr": pa.array(adj_indptr, type=pa.int64()),
    })

    nnz = len(data)
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
            np.arange(global_data_id, global_data_id + nnz, dtype=np.int64),
            type=pa.int64(),
        ),
        "atlas_cell_id": pa.array(cell_index, type=pa.int32()),
        "atlas_gene_id": pa.array(indices, type=pa.uint16()),
        "data": pa.array(data, type=pa.float32()),
    })

    return (
        indptr_table,
        data_table,
        global_indptr_id + len(adj_indptr),
        global_indptr_offset + nnz,
        global_data_id + nnz,
    )

def _write_shuffle_window_to_duckdb(
    window_adatas,
    conn,
    global_cell_id,
    global_indptr_id,
    global_indptr_offset,
    global_data_id,
    var_written,
    source_store_type: str,  
    target_store_type: str, 
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
    adata_window = _convert_X_store_type_inplace(
        adata_window,
        source_store_type=source_store_type,
        target_store_type=target_store_type,
    )

    # 5. 写入 X_HyS_data / X_HyS_indptr
    (
        global_indptr_id,
        global_indptr_offset,
        global_data_id,
    ) = _append_X_HyS(
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
def _detect_X_store_type_from_backed(
    adata_backed,
    sample_n: int = 1000,
) -> str:
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

    print("[INFO] X scale 预检测结果:")
    print(f"  - sample_cells = {n:,}")
    print(f"  - nonzero_n    = {values.size:,}")
    print(f"  - max          = {vmax:.4f}")
    print(f"  - q95          = {q95:.4f}")
    print(f"  - frac <= 10   = {frac_le_10:.4f}")

    del adata_sample
    gc.collect()

    # 经验判断：
    # log 数据通常绝大多数非零值 <= 10
    # count 数据只要 max 或高分位明显超过 log 范围，就判断为 count
    if vmax > 50 or q95 > 10:
        return "count"
    else:
        return "log"


# 根据 source_store_type 和 target_store_type 原地转换 adata.X。
def _convert_X_store_type_inplace(
    adata,
    source_store_type: str,
    target_store_type: str,
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


''' 方法3 ： 顺序读取 , 单个大文件, 只支持 h5ad格式 '''
def load_h5ad_order(
    h5ad_path: str,
    atlas,
    batch_size: int = 4096,
    mega_batch_factor: int = 5, 
    store_type: str = "count",    # 目标存储类型，"count" 或 "log"
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

    batch_size
        每批读取、写入或处理的细胞数量；较大值通常更快但占用更多内存。

    mega_batch_factor
        顺序导入时用于计算 mega-batch 大小的倍数。

    store_type
        目标表达矩阵尺度，通常为 ``"count"`` 或 ``"log"``。

    Notes
    -----
    导入导出过程可能涉及较大的磁盘 IO 和内存占用，可根据数据规模调整 batch、chunk 或 worker 参数。

    Examples
    --------
    调用该函数：::

        sap.io.load_h5ad_order(...)
    """
    commit_every = 5      
    gc_every = 5  

    # 检查目标存储类型
    if store_type not in {"count", "log"}:
        raise ValueError(
            f"store_type 只能是 'count' 或 'log'，当前为: {store_type}"
        )

    # mega_batch_size 改成可控
    mega_batch_size = batch_size * mega_batch_factor

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
    x_format = _print_h5ad_X_format(h5ad_path)

    # 预读取 1000 个细胞，判断文件里的 X 是 count 还是 log
    source_store_type = _detect_X_store_type_from_backed(
        adata_backed,
        sample_n=1000,
    )

    print(f"[INFO] 文件中 X 判断为: {source_store_type}")
    print(f"[INFO] 目标存储类型 store_type = {store_type}")

    if source_store_type == store_type:
        print("[INFO] X 数据不需要转换，直接写入。")
    else:
        print(f"[INFO] X 数据将在 mega-batch 读入后转换: {source_store_type} -> {store_type}")

    _create_obs_table_from_adata(conn, adata_backed[:1])
    _create_var_table_from_adata(conn, adata_backed[:1])
    _create_HyS_tables(conn)

    print(f"[INFO] 数据集维度: {adata_backed.n_obs:,} × {adata_backed.n_vars:,}")
    print("[INFO] 使用 scanpy backed + mega-batch + 全局游标模式")
    print(f"[INFO] batch_size = {batch_size:,}")
    print(f"[INFO] mega_batch_size = {mega_batch_size:,}")
    print(f"[INFO] mega_batch_factor = {mega_batch_factor:,}")
    print(f"[INFO] store_type = {store_type}")

    mini_batch_counter = 0

    # 事务放在大循环外
    conn.execute("BEGIN TRANSACTION")

    try:
        for mega_i, mega_start in enumerate(
            tqdm(
                range(0, n_cells, mega_batch_size),
                desc="Mega-batch（磁盘顺序读取）",
            )
        ):
            mega_end = min(mega_start + mega_batch_size, n_cells)

            t0 = time.time()

            # 真正触发磁盘读取
            mega = adata_backed[mega_start:mega_end].to_memory()

            t_read = time.time() - t0

            # mega-batch 读入后，统一转换 X 的存储尺度
            mega = _convert_X_store_type_inplace(
                mega,
                source_store_type=source_store_type,
                target_store_type=store_type,
            )

            # 统计当前 mega 的 nnz
            if sparse.issparse(mega.X):
                mega_nnz = mega.X.nnz
            else:
                mega_nnz = np.count_nonzero(mega.X)

            # 按 batch_size 分批导入
            for start in range(0, mega.n_obs, batch_size):
                end = min(start + batch_size, mega.n_obs)
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
                ) = _append_X_HyS(
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

            print(
                f"\n[mega {mega_i}] "
                f"cells={mega.n_obs:,}, "
                f"nnz={mega_nnz:,}, "
                f"read={t_read:.2f}s, "
                f"total_cells={global_cell_id:,}, "
                f"total_nnz={global_data_id:,}"
            )

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

    print("✔ 全部数据成功导入 DuckDB（顺序导入，含 obsm / varm）")
    print(f"  - cells: {global_cell_id:,}")
    print(f"  - nnz:   {global_data_id:,}")
    print(f"  - store_type: {store_type}")

# 简单判断 h5ad.X 的底层稀疏格式
def _print_h5ad_X_format(h5ad_path: str):
    """执行 ``_print_h5ad_X_format`` 的核心功能。

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

    with h5py.File(h5ad_path, "r") as f:
        if "X" not in f:
            print("[INFO] h5ad.X format = None（文件中没有 X）")
            return None

        X = f["X"]

        # dense matrix
        if isinstance(X, h5py.Dataset):
            print("[INFO] h5ad.X format = dense")
            return "dense"

        # sparse matrix group
        if isinstance(X, h5py.Group):
            encoding_type = X.attrs.get("encoding-type", "unknown")

            if isinstance(encoding_type, bytes):
                encoding_type = encoding_type.decode("utf-8")

            if encoding_type == "csr_matrix":
                print("[INFO] h5ad.X format = CSR")
                return "csr"

            elif encoding_type == "csc_matrix":
                print("[INFO] h5ad.X format = CSC")
                return "csc"

            elif encoding_type == "coo_matrix":
                print("[INFO] h5ad.X format = COO")
                return "coo"

            else:
                print(f"[INFO] h5ad.X format = unknown ({encoding_type})")
                return encoding_type

        print("[INFO] h5ad.X format = unknown")
        return "unknown"

# 推断数据类型
def _infer_duckdb_type_from_series(s: pd.Series) -> str:

    """执行 ``_infer_duckdb_type_from_series`` 的核心功能。

    该内部函数属于数据导入模块，用于支撑同一模块中的公共 API。

    把 h5ad、AnnData 或 Scanpy 支持的数据格式写入 Atlas 的
    ``obs``、``var``、``X_HyS_*``、``obsm`` 和 ``varm`` 表。

    它通常不会作为用户入口直接调用；直接调用时需要保证输入对象、数据库连接和相关临时表已经由上游步骤准备好。

    Parameters
    ----------
    s
        需要推断 DuckDB 类型的 pandas Series。

    Returns
    -------
    result
        函数返回结果。具体类型取决于参数设置和内部执行路径。

    Notes
    -----
    这是内部 helper；除非需要扩展 scAtlasPy 内部流程，一般不建议在用户代码中直接调用。
    """
    if pd.api.types.is_integer_dtype(s):
        return "BIGINT"
    if pd.api.types.is_float_dtype(s):
        return "DOUBLE"
    if pd.api.types.is_bool_dtype(s):
        return "BOOLEAN"
    return "VARCHAR"

# 建立obs表
def _create_obs_table_from_adata(conn, adata):
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

# 建立var表
def _create_var_table_from_adata(conn, adata):
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
def _create_HyS_tables(conn):

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
def _append_obs_rows(adata, conn, start_cell_id: int) -> int:

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
def _append_var(adata, conn):

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
def _append_X_HyS(
    adata,
    conn,
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
def _add_obsm_from_h5ad(h5ad_path: str, atlas, batch_size=4096):

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

    batch_size
        每批读取、写入或处理的细胞数量；较大值通常更快但占用更多内存。

    Notes
    -----
    这是内部 helper；除非需要扩展 scAtlasPy 内部流程，一般不建议在用户代码中直接调用。
    """
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

            for start in range(0, n_cells, batch_size):

                end = min(start + batch_size, n_cells)
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
def _add_varm_from_h5ad(h5ad_path: str, atlas):

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


''' 方法4： 顺序读取，小文件读取，支持多种数据格式的导入 '''
def load_small_data( file_path , atlas:Atlas):

    """导入小型单细胞数据文件。

    该函数先调用 ``read_smart`` 根据文件后缀自动读取数据，再把得到的 AnnData 对象写入 Atlas。

    它适合可以一次性载入内存的小数据集；大 h5ad 文件建议使用 ``load_h5ad_random``、``load_h5ad_fast`` 或
    ``load_h5ad_order``。

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

        sap.io.load_small_data(...)
    """
    print("小文件读取 , 开始导入数据...")
    adata = read_smart(file_path)
    load_AnnData(adata,atlas)
    print("✔ 全部数据成功导入 DuckDB ")


# 多种数据格式的导入
def read_smart(file_path, **kwargs):
    """根据文件后缀自动选择 Scanpy 读取函数。

    该函数检查输入文件路径的扩展名，并调用对应的 ``scanpy.read_*`` 函数读取为 AnnData。

    它适合小型数据或临时转换场景，能够统一处理 h5ad、loom、Matrix Market、csv、文本表格、Excel、
    10x h5 和 UMI-tools 等常见输入格式。

    Parameters
    ----------
    file_path
        输入单细胞数据文件路径。

    **kwargs
        传递给具体 ``scanpy.read_*`` 函数的额外参数。

        不同格式支持的参数不同，例如是否指定分隔符、缓存行为或基因变量名处理方式。

    Returns
    -------
    adata
        读取完成的 AnnData 对象。

    Notes
    -----
    该函数只负责读取文件，不会直接写入 Atlas 数据库。若要导入数据库，可继续调用 ``load_AnnData`` 或
    ``load_small_data``。

    Examples
    --------
    读取 h5ad 文件：::

        adata = read_smart("example.h5ad")
    """

    # 获取文件后缀名（小写形式）
    file_ext = os.path.splitext(file_path)[1].lower()

    # 根据文件后缀选择对应的读取方法
    if file_ext == '.h5ad':
        # h5ad格式
        return sc.read_h5ad(file_path, **kwargs)
    elif file_ext == '.loom':
        # loom格式
        return sc.read_loom(file_path, **kwargs)
    elif file_ext in ['.mtx', '.mtx.gz']:
        # mtx格式 (Matrix Market格式)
        return sc.read_mtx(file_path, **kwargs)
    elif file_ext in ['.csv', '.csv.gz']:
        # csv格式
        return sc.read_csv(file_path, **kwargs)
    elif file_ext in ['.txt', '.tsv', '.tab']:
        # 文本格式，默认制表符分隔
        return sc.read_text(file_path, **kwargs)
    elif file_ext in ['.xlsx', '.xls']:
        # Excel格式
        return sc.read_excel(file_path, **kwargs)
    elif file_ext == '.h5':
        # 10x Genomics h5格式
        return sc.read_10x_h5(file_path, **kwargs)
    elif 'umi_tools' in file_path.lower():
        # UMI-tools格式
        return sc.read_umi_tools(file_path, **kwargs)
    else:
        # 如果不认识的后缀，尝试使用通用的read函数
        return sc.read(file_path, **kwargs)


''' 方法5： 顺序读取，anndata数据导入 '''
def load_AnnData(adata:AnnData, atlas:Atlas):

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

        sap.io.load_AnnData(...)
    """
    try:
        logger.info("准备数据表...")

        if hasattr(adata, 'obs'):
            _add_obs(adata, atlas)  # 细胞表数据（对应obs）,
        else:
            print("Skipping obs layer")

        if hasattr(adata, 'var'):
            _add_var(adata, atlas)  # 基因表数据（对应var）
        else:
            print("Skipping var layer")

        if hasattr(adata, 'X'):
            start_time = time.time()
            _add_X_HyS_chunked(adata,atlas,chunk_size=4096) # 分块导入X表数据
            end_time = time.time()
            logger.info(" X表的导入用时为： " + str(end_time - start_time))
        else:
            print("Skipping X layer")

        if hasattr(adata, 'obsm'):
            _add_obsm(adata,atlas)
        else:
            print("Skipping obsm layer")

        if hasattr(adata, 'varm'):
            _add_varm(adata,atlas)
        else:
            print("Skipping varm layer")

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

# 导入 X_CSRO
def _add_X_HyS_chunked( adata: AnnData, atlas: Atlas, chunk_size: int = 4096):

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

        for chunk_idx in tqdm(range(total_chunks), desc="导入 X_HyS chunks"):
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
        print(f"✔ 导入完成：cells={n_cells:,}, nnz={global_data_counter:,}")

        return True

    except Exception as e:
        conn.execute("ROLLBACK")
        logger.error(f"CSR 导入失败: {e}")
        raise


''' 基因名清洗 ：先导入，再清洗，var表 '''
def clean_genes(atlas: Atlas, gene_name_column: str = "atlas_gene_name"):
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

        sap.io.clean_genes(...)
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

    logger.info("清洗基因名 已完成!")
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
