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
    """
    多个 h5ad 文件读取，global-block 级随机导入 DuckDB。

    核心随机逻辑：
    ------------------------------------------------------------
    1. 每个文件内部：
       按 batch_size * read_batch_factor 切成连续 block。
       注意：不再对单个文件内部 block 单独 shuffle。

    2. 全局 block 索引池：
       把所有文件的 block 都放入 all_block_refs。
       例如：
           A: A1,A2,A3
           B: B1,B2,B3,B4
           C: C1,C2,C3,C4,C5

       合并为：
           [A1,A2,A3,B1,B2,B3,B4,C1,C2,C3,C4,C5]

       然后整体随机打乱。

    3. 每次读取 pool_block_num 个 block：
       默认 pool_block_num=5。
       例如随机后：
           [C3,A1,B4,C1,B2, A3,C5,B1,C2,A2, B3,C4]

       则每次读取：
           第1组：C3,A1,B4,C1,B2
           第2组：A3,C5,B1,C2,A2
           第3组：B3,C4

    4. cell_pool：
       每组 block 读取后，直接放入 cell_pool。
       不再做 block 内部 cell 随机。
       只在 _flush_cell_pool() 里对整个 cell_pool 的所有 cell 整体随机一次。

    5. 写入数据库：
       将随机后的 cell_pool 写入 obs / X_HyS_indptr / X_HyS_data。

    适合：
    ------------------------------------------------------------
    - n 个 h5ad 文件
    - 每个文件大小不一致
    - 希望避免 round-robin 在后期只剩大文件的问题
    - 希望保持连续 block 读取，避免 h5ad cell 级随机 IO

    注意：
    ------------------------------------------------------------
    - 不导入 obsm：因为 cell 已经随机重排，原始 obsm 顺序会错位。
    - varm 是 gene 维度，可以只从第一个文件导入。
    - 所有文件必须 gene 数量和 gene 顺序一致。
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
    """
    大文件读取，cell 按 shuffle-window 随机导入，只支持 h5ad 格式。

    核心逻辑：
    1. 按 batch_size 切成 block
    2. block_starts 全局随机
    3. 每次读取 shuffle_window_batches 个 batch 到内存
    4. 合并成一个 adata_window
    5. 对 window 内所有 cell 做统一随机打乱
    6. 再整体写入 DuckDB

    参数简化：
    - 删除 read_batch_factor
    - commit_every = shuffle_window_batches * 2
    - gc_every     = shuffle_window_batches * 4
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
    """
    大文件 shuffle-window 随机导入 DuckDB，X 使用 Parquet staging 加速。

    保持 X_HyS_data 的业务列不变:
        id, atlas_cell_id, atlas_gene_id, data

    优化点:
    1. obs/var 仍直接写 DuckDB。
    2. X_HyS_indptr / X_HyS_data 先按 window 写入临时 Parquet parts。
    3. 最后用 DuckDB read_parquet 一次性建表，减少逐 window INSERT 开销。
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
        if x_reader is None:
            return adata_backed[block_start:block_end].to_memory()
        X = x_reader.read_rows(block_start, block_end)
        return AnnData(
            X=X,
            obs=adata_backed.obs.iloc[block_start:block_end].copy(),
            var=source_var.copy(deep=False),
        )

    def close_parquet_writers() -> None:
        nonlocal indptr_writer, data_writer
        if indptr_writer is not None:
            indptr_writer.close()
            indptr_writer = None
        if data_writer is not None:
            data_writer.close()
            data_writer = None

    def append_parquet_tables(indptr_table, data_table) -> None:
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
    """Own one pair of Parquet files and append windows as row groups."""

    def __init__(
        self,
        *,
        indptr_dir: str,
        data_dir: str,
        part_id: int,
        parquet_compression: str,
    ):
        self._indptr_path = os.path.join(indptr_dir, f"part_{part_id:06d}.parquet")
        self._data_path = os.path.join(data_dir, f"part_{part_id:06d}.parquet")
        self._parquet_compression = parquet_compression
        self._indptr_writer = None
        self._data_writer = None
        self._pool = futures.ThreadPoolExecutor(max_workers=1)

    def submit(self, indptr_table, data_table):
        return self._pool.submit(self._write, indptr_table, data_table)

    def _write(self, indptr_table, data_table):
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
        self._pool.shutdown(wait=True, cancel_futures=cancel_futures)
        if self._indptr_writer is not None:
            self._indptr_writer.close()
            self._indptr_writer = None
        if self._data_writer is not None:
            self._data_writer.close()
            self._data_writer = None


class _ParallelH5adCSRReader:
    """Read gzip-compressed CSR X blocks by decompressing HDF5 chunks in parallel."""

    def __init__(self, h5ad_path: str, n_vars: int, workers: int):
        self._file = h5py.File(h5ad_path, "r")
        self._x = self._file["X"]
        self._data = self._x["data"]
        self._indices = self._x["indices"]
        self._indptr = self._x["indptr"]
        self._n_vars = n_vars
        self._pool = futures.ThreadPoolExecutor(max_workers=workers)

    @classmethod
    def open_if_supported(cls, h5ad_path: str, n_vars: int, workers: int):
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
        return (
            ds.ndim == 1
            and ds.chunks is not None
            and len(ds.chunks) == 1
            and ds.compression == "gzip"
        )

    def close(self) -> None:
        self._pool.shutdown(wait=True)
        self._file.close()

    def read_rows(self, row_start: int, row_stop: int):
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
    """
    将多个 batch 的 AnnData 合并成一个 window，
    对 window 内所有 cell 统一随机打乱，
    然后整体写入 DuckDB。
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
    """
    预读取 sample_n 个细胞，自动判断 X 是 count scale 还是 log scale。

    判断逻辑：
    - log scale：非零值通常大量集中在 0~10
    - count scale：非零值可能出现几十、几百、几千甚至更大
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
    """
    根据 source_store_type 和 target_store_type 原地转换 adata.X。

    source_store_type:
        - "count"
        - "log"

    target_store_type:
        - "count"
        - "log"
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
    """
    简单判断 h5ad.X 的底层稀疏格式：
    - csr_matrix -> 输出 CSR
    - csc_matrix -> 输出 CSC
    - coo_matrix -> 输出 COO
    - dense      -> 输出 dense
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

    if pd.api.types.is_integer_dtype(s):
        return "BIGINT"
    if pd.api.types.is_float_dtype(s):
        return "DOUBLE"
    if pd.api.types.is_bool_dtype(s):
        return "BOOLEAN"
    return "VARCHAR"

# 建立obs表
def _create_obs_table_from_adata(conn, adata):

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

    print("小文件读取 , 开始导入数据...")
    adata = read_smart(file_path)
    load_AnnData(adata,atlas)
    print("✔ 全部数据成功导入 DuckDB ")


# 多种数据格式的导入
def read_smart(file_path, **kwargs):

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
    """
    在数据库层面清洗基因名，直接操作数据库表
    参数:
    ----------
    atlas : Atlas
        Atlas数据库对象
        gene_name_column : str  基因名，默认为'atlas_gene_name'
        在数据库中添加后缀模式：为重复基因添加 _1, _2, _3 等后缀
        仅处理 var 表
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