import time
import gc
from tqdm import tqdm
from ..data import Atlas
import os
from anndata import AnnData
from scipy import sparse
import logging
import h5py
import numpy as np
import pandas as pd
import scanpy as sc
import pyarrow as pa   # ✅ 新增：Arrow Table 加速 DuckDB INSERT
# 获取日志记录器
logger = logging.getLogger('Atlas')

''' 大文件读取, 多文件 + subbatch级伪随机 + round-robin file shuffle + cell_pool随机 导入，只支持 h5ad格式 '''
def load_big_h5ad_list_to_duckdb_random_batch_pool(
    h5ad_paths: list[str],
    atlas,
    batch_size: int = 4096,
    read_batch_factor: int = 1,
    pool_rounds: int = 1,          # ✅ 新增：累计多少轮 file-shuffle 后 flush，一般 1 就够
    check_var: bool = True,
    random_seed: int | None = None,
    commit_every: int = 5,         # ✅ 每多少次 pool flush 提交一次
    gc_every: int = 5,             # ✅ 每多少次 pool flush 做一次 gc
):
    """
    多个 h5ad 文件读取，subbatch 级伪随机导入 DuckDB。

    核心随机逻辑：
    ------------------------------------------------------------
    1. 每个文件内部：
       按 batch_size * read_batch_factor 切成连续 block。
       每个文件自己的 block 顺序先打乱。

    2. 文件之间：
       每一轮取当前还没读完的 active_files。
       对 active_files 打乱顺序。
       然后每个 active file 读取一个 batch block。

    3. cell_pool：
       一轮中读取到的多个 batch 放入 cell_pool。
       累计 pool_rounds 轮后，把 cell_pool 中所有 cell 整体随机打乱。

    4. 写入数据库：
       将打乱后的 cell_pool 写入 obs / X_CSRO_indptr / X_CSRO_data。

    适合：
    ------------------------------------------------------------
    - n 个 h5ad 文件
    - 每个文件很大
    - 需要比单文件 random_batch 更强的随机性
    - 但又不想使用 cell 级 h5ad 随机读取

    注意：
    ------------------------------------------------------------
    - 不导入 obsm：因为 cell 已经随机重排，原始 obsm 顺序会错位。
    - varm 是 gene 维度，可以只从第一个文件导入。
    - 所有文件必须 gene 数量和 gene 顺序一致。
    """

    import time
    import gc
    import numpy as np
    import pandas as pd
    import scanpy as sc
    from anndata import AnnData
    from scipy import sparse
    from tqdm import tqdm

    # =====================================================
    # ✅ 修改 1：支持单路径 / 多路径
    # =====================================================
    if isinstance(h5ad_paths, str):
        h5ad_paths = [h5ad_paths]

    if len(h5ad_paths) == 0:
        raise ValueError("h5ad_paths 不能为空")

    if pool_rounds <= 0:
        raise ValueError("pool_rounds 必须 > 0")

    if read_batch_factor <= 0:
        raise ValueError("read_batch_factor 必须 > 0")

    file_num = len(h5ad_paths)

    # 实际读取 block 大小
    read_batch_size = batch_size * read_batch_factor

    # =====================================================
    # ✅ 修改 2：独立随机生成器
    # 不污染全局 np.random
    # =====================================================
    rng = np.random.default_rng(random_seed)

    #  1️⃣ 连接数据库
    conn = atlas.connect("r+")
    atlas.connection = conn

    # 全局游标
    global_cell_id = 0
    global_indptr_id = 0
    global_indptr_offset = 0
    global_data_id = 0

    var_written = False

    print("\n==== load_big_h5ad_list_to_duckdb_random_batch_pool ====")
    print(f"[INFO] 文件数量: {file_num:,}")
    print(f"[INFO] batch_size: {batch_size:,}")
    print(f"[INFO] read_batch_factor: {read_batch_factor:,}")
    print(f"[INFO] read_batch_size: {read_batch_size:,}")
    print(f"[INFO] pool_rounds: {pool_rounds:,}")
    print(f"[INFO] commit_every: {commit_every:,}")
    print(f"[INFO] gc_every: {gc_every:,}")
    print("[INFO] 每轮打乱 active files，每个文件读一次，然后 cell_pool 整体随机写入")

    # =====================================================
    # ✅ 修改 3：打开所有 h5ad，并为每个文件生成随机 block 顺序
    # =====================================================
    file_states = []

    ref_var_names = None
    ref_n_genes = None

    try:
        for file_idx, h5ad_path in enumerate(h5ad_paths):

            print("\n" + "=" * 80)
            print(f"[FILE {file_idx + 1}/{file_num}] {h5ad_path}")

            adata_backed = sc.read_h5ad(h5ad_path, backed="r")

            n_cells = adata_backed.n_obs
            n_genes = adata_backed.n_vars

            print(f"[INFO] 当前文件维度: {n_cells:,} × {n_genes:,}")

            # ---------------- 检查 gene 数量和顺序 ----------------
            cur_var_names = adata_backed.var.index.astype(str).to_numpy()

            if file_idx == 0:
                ref_var_names = cur_var_names
                ref_n_genes = n_genes
            else:
                if check_var:
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

            # ---------------- 每个文件内部按 read_batch_size 切 block ----------------
            block_starts = np.arange(0, n_cells, read_batch_size, dtype=np.int64)

            # ✅ 每个文件自己的 block 顺序先打乱
            rng.shuffle(block_starts)

            file_states.append(
                {
                    "file_idx": file_idx,
                    "h5ad_path": h5ad_path,
                    "adata_backed": adata_backed,
                    "n_cells": n_cells,
                    "n_genes": n_genes,
                    "block_starts": block_starts,
                    "cursor": 0,
                    "done": False,
                }
            )

            print(f"[INFO] batch block 数量: {len(block_starts):,}")

        # =====================================================
        # 2️⃣ 动态建表：只用第一个文件建表
        # =====================================================
        first_backed = file_states[0]["adata_backed"]

        _create_obs_table_from_adata(conn, first_backed[:1])
        _create_var_table_from_adata(conn, first_backed[:1])
        _create_csro_tables(conn)

        # =====================================================
        # ✅ 修改 4：flush cell_pool
        # 多个文件读到的 batch 合并后，整体随机，然后写入数据库
        # =====================================================
        def _flush_cell_pool(cell_pool, flush_i: int):

            nonlocal global_cell_id
            nonlocal global_indptr_id
            nonlocal global_indptr_offset
            nonlocal global_data_id
            nonlocal var_written

            if len(cell_pool) == 0:
                return 0, 0

            t0 = time.time()

            # -------------------------------------------------
            # 1. 合并 cell_pool 中的多个 AnnData batch
            # -------------------------------------------------
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

            # -------------------------------------------------
            # 2. cell_pool 内部整体随机
            #    这是跨文件混合的关键
            # -------------------------------------------------
            if pool_adata.n_obs > 1:
                pool_perm = rng.permutation(pool_adata.n_obs)
                pool_adata = pool_adata[pool_perm].copy()

            t_shuffle = time.time() - t0
            t1 = time.time()

            # -------------------------------------------------
            # 3. 写 obs
            # -------------------------------------------------
            global_cell_id = _append_obs_rows(
                pool_adata,
                conn,
                start_cell_id=global_cell_id,
            )

            # -------------------------------------------------
            # 4. 写 var，只写一次
            # -------------------------------------------------
            if not var_written:
                _append_var(pool_adata, conn)
                var_written = True

            # -------------------------------------------------
            # 5. 写 X_CSRO，使用 Arrow 加速版
            # -------------------------------------------------
            (
                global_indptr_id,
                global_indptr_offset,
                global_data_id,
            ) = _append_X_CSRO_chunk_fast(
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
                f"pool_batches={len(cell_pool):,}, "
                f"pool_cells={total_pool_cells:,}, "
                f"pool_nnz={total_pool_nnz:,}, "
                f"shuffle={t_shuffle:.2f}s, "
                f"write={t_write:.2f}s, "
                f"total_cells={global_cell_id:,}, "
                f"total_nnz={global_data_id:,}"
            )

            # -------------------------------------------------
            # 6. 清理
            # -------------------------------------------------
            del X_list, obs_list, X_pool, obs_pool, var_pool, pool_adata
            gc.collect()

            return total_pool_cells, total_pool_nnz

        # =====================================================
        # 3️⃣ 主循环：每轮打乱 active_files，每个文件读一次
        # =====================================================
        total_blocks = sum(len(s["block_starts"]) for s in file_states)
        processed_blocks = 0

        cell_pool = []

        round_counter = 0
        flush_counter = 0

        pbar = tqdm(total=total_blocks, desc="Multi-file round-shuffle import")

        # =====================================================
        # ✅ 事务：多个 flush 共用事务
        # =====================================================
        conn.execute("BEGIN TRANSACTION")

        try:
            while processed_blocks < total_blocks:

                # 当前还没有读完的文件
                active_indices = [
                    i for i, s in enumerate(file_states)
                    if not s["done"]
                ]

                if len(active_indices) == 0:
                    break

                # =====================================================
                # ✅ 修改 5：每一轮打乱 active files
                # 而不是每次随机选一个文件
                # =====================================================
                active_order = np.array(active_indices, dtype=np.int32)
                rng.shuffle(active_order)

                round_counter += 1

                # -------------------------------------------------
                # 本轮：每个 active file 读一次
                # -------------------------------------------------
                for state_idx in active_order:

                    state = file_states[int(state_idx)]

                    if state["done"]:
                        continue

                    cursor = state["cursor"]
                    block_starts = state["block_starts"]

                    if cursor >= len(block_starts):
                        state["done"] = True
                        continue

                    block_start = int(block_starts[cursor])
                    block_end = min(block_start + read_batch_size, state["n_cells"])

                    state["cursor"] += 1
                    if state["cursor"] >= len(block_starts):
                        state["done"] = True

                    # -------------------------------------------------
                    # 从 h5ad 连续读取一个 batch block
                    # -------------------------------------------------
                    t_read0 = time.time()

                    adata = state["adata_backed"][block_start:block_end].to_memory()

                    t_read = time.time() - t_read0

                    # -------------------------------------------------
                    # 当前 batch 内部先随机一次
                    # 后面进入 cell_pool 后还会整体随机一次
                    # -------------------------------------------------
                    if adata.n_obs > 1:
                        local_perm = rng.permutation(adata.n_obs)
                        adata_random = adata[local_perm].copy()
                        del adata
                        adata = adata_random
                        del adata_random

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
                            f"cells={adata.n_obs:,}, "
                            f"nnz={block_nnz:,}, "
                            f"read={t_read:.2f}s, "
                            f"pool_batches={len(cell_pool):,}"
                        )

                # =====================================================
                # ✅ 修改 6：每 pool_rounds 轮 flush 一次
                # 默认 pool_rounds=1，也就是每轮读完所有 active file 后 flush
                # =====================================================
                if round_counter % pool_rounds == 0 and len(cell_pool) > 0:

                    flush_counter += 1
                    _flush_cell_pool(cell_pool, flush_counter)

                    # 清空 pool
                    for ad in cell_pool:
                        del ad
                    cell_pool = []
                    gc.collect()

                    # 每 commit_every 次 flush 提交一次
                    if flush_counter % commit_every == 0:
                        conn.execute("COMMIT")
                        conn.execute("BEGIN TRANSACTION")

                    # 每 gc_every 次 flush 做一次 gc
                    if flush_counter % gc_every == 0:
                        gc.collect()

            pbar.close()

            # =====================================================
            # 4️⃣ 处理最后剩余 cell_pool
            # =====================================================
            if len(cell_pool) > 0:
                flush_counter += 1
                _flush_cell_pool(cell_pool, flush_counter)

                for ad in cell_pool:
                    del ad
                cell_pool = []
                gc.collect()

            # 最后提交
            conn.execute("COMMIT")

        except Exception:
            pbar.close()
            conn.execute("ROLLBACK")
            raise

        # =====================================================
        # 5️⃣ 主键
        # =====================================================
        conn.execute("ALTER TABLE obs ADD PRIMARY KEY (atlas_cell_id)")
        conn.execute("ALTER TABLE var ADD PRIMARY KEY (atlas_gene_id)")

        # =====================================================
        # 6️⃣ obsm / varm
        # =====================================================
        # ⚠️ obsm 是 cell 维度。
        # 当前 obs / X 已经随机重排，不能直接按原 h5ad 顺序导入 obsm。
        # 所以这里不要调用 _add_obsm_from_h5ad。
        #
        # varm 是 gene 维度，多个文件 var 一致时，可以只导入第一个文件的 varm。
        # =====================================================
        _add_varm_from_h5ad(h5ad_paths[0], atlas)

        print("\n✔ 多文件全部成功导入 DuckDB（round file shuffle + cell_pool 随机）")
        print(f"  - files: {file_num:,}")
        print(f"  - cells: {global_cell_id:,}")
        print(f"  - genes: {ref_n_genes:,}")
        print(f"  - nnz:   {global_data_id:,}")
        print(f"  - flush: {flush_counter:,}")

        return {
            "files": file_num,
            "cells": global_cell_id,
            "genes": ref_n_genes,
            "nnz": global_data_id,
            "flush": flush_counter,
        }

    finally:
        # =====================================================
        # 7️⃣ 无论成功/失败，都关闭所有 backed 文件
        # =====================================================
        for s in file_states:
            try:
                s["adata_backed"].file.close()
            except Exception:
                pass

# 代码的整体流程逻辑
# 多个 h5ad 文件
# ↓
# 每个文件内部切成连续 block
# ↓
# 每个文件自己的 block 顺序随机
# ↓
# 每一轮随机选择 active files 顺序
# ↓
# 每个 active file 读一个连续 block
# ↓
# block 内部先随机
# ↓
# 放入 cell_pool
# ↓
# 累计 pool_rounds 轮后，cell_pool 整体随机
# ↓
# 写入 obs / var / X_CSRO_indptr / X_CSRO_data
# ↓
# 每 commit_every 次 flush 提交事务
# ↓
# 最后只导入 varm，不导入 obsm


''' 大文件读取, subbatch级 伪随机 导入， 只支持 h5ad格式'''
# batch 随机 + batch 内随机	[7,6],[3,2],[9,8]...
def load_big_h5ad_to_duckdb_random_batch(
    h5ad_path: str,
    atlas,
    batch_size: int = 4096,
    read_batch_factor=1,
    commit_every: int = 5,   # ✅ 修改 1：新增，每多少个 batch 提交一次
    gc_every: int = 10,       # ✅ 修改 2：新增，每多少个 batch 做一次 gc
):
    """
    大文件读取，cell 按 batch 随机导入，只支持 h5ad 格式。

    修改逻辑：
    - 按 batch_size 切成连续 block
    - 随机 block 顺序
    - 每个 block 从 h5ad 连续读取
    - 读取到内存后，再在 block 内部随机打乱 cell

    ✅ 本版新增优化：
    1. 使用 _append_X_CSRO_chunk_fast()，Arrow 写入
    2. 外层增加 BEGIN / COMMIT，避免每个 INSERT 自动提交
    3. commit_every 控制分批提交
    4. gc_every 控制 GC 频率，不再每个 batch 都强制 gc.collect()
    5. 增加 read / write / nnz 简单统计
    """

    batch_size = batch_size * read_batch_factor

    #  1️⃣ 连接数据库
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

    # =====================================================
    # ✅ 保持原逻辑：batch block 随机
    # 每个 block_start 对应连续读取区间：
    # [block_start : block_start + batch_size]
    # =====================================================
    block_starts = np.arange(0, n_cells, batch_size, dtype=np.int64)
    np.random.shuffle(block_starts)

    # 3️⃣ 动态建表
    _create_obs_table_from_adata(conn, adata_backed[:1])
    _create_var_table_from_adata(conn, adata_backed[:1])
    _create_csro_tables(conn)

    print(f"[INFO] 数据集维度: {adata_backed.n_obs:,} × {adata_backed.n_vars:,}")
    print("[INFO] 使用 scanpy backed + batch-block 随机读取 + batch 内部随机")
    print(f"[INFO] batch_size = {batch_size:,}")
    print(f"[INFO] batch block 数量 = {len(block_starts):,}")
    print(f"[INFO] commit_every = {commit_every:,}")
    print(f"[INFO] gc_every = {gc_every:,}")

    batch_counter = 0

    # =====================================================
    # ✅ 修改 3：事务放在大循环外
    # 原来：每个 INSERT 可能自动提交
    # 现在：多个 batch 共用事务
    # =====================================================
    conn.execute("BEGIN TRANSACTION")

    try:
        for block_i, block_start in enumerate(
            tqdm(
                block_starts,
                desc="Batch-block 随机读取",
            )
        ):
            block_end = min(int(block_start) + batch_size, n_cells)

            # -------------------------------------------------
            # 连续读取一个 batch block
            # -------------------------------------------------
            t0 = time.time()
            adata = adata_backed[int(block_start):block_end].to_memory()
            t_read = time.time() - t0

            # 当前 block 的 nnz
            if sparse.issparse(adata.X):
                block_nnz = adata.X.nnz
            else:
                block_nnz = np.count_nonzero(adata.X)

            # -------------------------------------------------
            # batch 内部再随机
            # 例如 [6,7] -> [7,6]
            # -------------------------------------------------
            if adata.n_obs > 1:
                local_perm = np.random.permutation(adata.n_obs)

                # ✅ 修改 4：避免旧 adata 和新 adata 同时长时间占用
                adata_random = adata[local_perm].copy()
                del adata
                adata = adata_random
                del adata_random

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
            ) = _append_X_CSRO_chunk_fast(   # ✅ 修改 5：使用 Arrow 加速版
                adata,
                conn,
                base_cell_id=global_cell_id - adata.n_obs,
                global_indptr_id=global_indptr_id,
                global_indptr_offset=global_indptr_offset,
                global_data_id=global_data_id,
            )

            t_write = time.time() - t1

            batch_counter += 1

            # =====================================================
            # ✅ 修改 6：每 commit_every 个 batch 提交一次
            # =====================================================
            if batch_counter % commit_every == 0:
                conn.execute("COMMIT")
                conn.execute("BEGIN TRANSACTION")

            # 简单日志：不要太频繁刷屏
            if (block_i + 1) % 20 == 0 or block_i == 0:
                print(
                    f"\n[block {block_i}] "
                    f"cells={adata.n_obs:,}, "
                    f"nnz={block_nnz:,}, "
                    f"read={t_read:.2f}s, "
                    f"write={t_write:.2f}s, "
                    f"total_cells={global_cell_id:,}, "
                    f"total_nnz={global_data_id:,}"
                )

            del adata

            # =====================================================
            # ✅ 修改 7：不要每个 batch 都 gc.collect()
            # =====================================================
            if (block_i + 1) % gc_every == 0:
                gc.collect()

        # 最后提交
        conn.execute("COMMIT")

    except Exception:
        conn.execute("ROLLBACK")
        raise

    # 5️⃣ 主键
    conn.execute("ALTER TABLE obs ADD PRIMARY KEY (atlas_cell_id)")
    conn.execute("ALTER TABLE var ADD PRIMARY KEY (atlas_gene_id)")

    # =====================================================
    # ⚠️ 随机导入时不建议导入 obsm
    # 因为 obs / X 已经随机重排，obsm 直接按原 h5ad 顺序导入会错位
    # varm 是 gene 维度，一般没问题
    # =====================================================
    # _add_obsm_from_h5ad(h5ad_path, atlas)
    _add_varm_from_h5ad(h5ad_path, atlas)

    try:
        adata_backed.file.close()
    except Exception:
        pass

    print("✔ 全部数据成功导入 DuckDB（batch 随机 + batch 内随机）")
    print(f"  - cells: {global_cell_id:,}")
    print(f"  - nnz:   {global_data_id:,}")


# todo 使用这个代码
def load_big_h5ad_to_duckdb_random_batch_window(
    h5ad_path: str,
    atlas,
    batch_size: int = 4096,
    shuffle_window_batches: int = 5,   # ✅ 修改 1：固定窗口级随机，默认 5 个 batch
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

    # =====================================================
    # ✅ 保持你的原逻辑：batch block 顺序随机
    # 每个 block 自身仍然是 h5ad 连续读取区间
    # 但是 5 个 block 合并后再统一随机
    # =====================================================
    block_starts = np.arange(0, n_cells, batch_size, dtype=np.int64)
    np.random.shuffle(block_starts)

    # 3️⃣ 动态建表
    _create_obs_table_from_adata(conn, adata_backed[:1])
    _create_var_table_from_adata(conn, adata_backed[:1])
    _create_csro_tables(conn)

    print(f"[INFO] 数据集维度: {adata_backed.n_obs:,} × {adata_backed.n_vars:,}")
    print("[INFO] 使用 scanpy backed + shuffle-window 随机读取")
    print(f"[INFO] batch_size = {batch_size:,}")
    print(f"[INFO] shuffle_window_batches = {shuffle_window_batches:,}")
    print(f"[INFO] batch block 数量 = {len(block_starts):,}")
    print(f"[INFO] commit_every = {commit_every:,} batches")
    print(f"[INFO] gc_every = {gc_every:,} batches")

    # =====================================================
    # ✅ 修改 3：窗口缓存
    # 每次攒够 shuffle_window_batches 个 batch 再统一随机写入
    # =====================================================
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

            # -------------------------------------------------
            # 连续读取一个 batch block
            # -------------------------------------------------
            t0 = time.time()
            adata = adata_backed[int(block_start):block_end].to_memory()
            t_read = time.time() - t0

            # 当前 block 的 nnz
            if sparse.issparse(adata.X):
                block_nnz = adata.X.nnz
            else:
                block_nnz = np.count_nonzero(adata.X)

            # -------------------------------------------------
            # ✅ 修改 4：不再单 batch 内部随机后立刻写入
            # 而是先放入 window
            # -------------------------------------------------
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

            # =================================================
            # ✅ 修改 5：window 满了，统一随机 + 统一写入
            # =================================================
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

                # =================================================
                # ✅ 修改 6：每 commit_every 个 batch 提交一次
                # 等价于 shuffle_window_batches=5 时，每 2 个 window commit
                # =================================================
                if total_batch_counter % commit_every == 0:
                    conn.execute("COMMIT")
                    conn.execute("BEGIN TRANSACTION")
                    print(f"[COMMIT] processed_batches={total_batch_counter:,}")

                # =================================================
                # ✅ 修改 7：每 gc_every 个 batch gc 一次
                # 等价于 shuffle_window_batches=5 时，每 4 个 window gc
                # =================================================
                if total_batch_counter % gc_every == 0:
                    gc.collect()
                    print(f"[GC] processed_batches={total_batch_counter:,}")

        # =====================================================
        # ✅ 修改 8：处理最后不足 5 个 batch 的剩余 window
        # =====================================================
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
        raise

    # 5️⃣ 主键
    conn.execute("ALTER TABLE obs ADD PRIMARY KEY (atlas_cell_id)")
    conn.execute("ALTER TABLE var ADD PRIMARY KEY (atlas_gene_id)")

    # =====================================================
    # ⚠️ 随机导入时不建议导入 obsm
    # 因为 obs / X 已经随机重排，obsm 直接按原 h5ad 顺序导入会错位
    # varm 是 gene 维度，一般没问题
    # =====================================================
    # _add_obsm_from_h5ad(h5ad_path, atlas)
    _add_varm_from_h5ad(h5ad_path, atlas)

    try:
        adata_backed.file.close()
    except Exception:
        pass

    print("✔ 全部数据成功导入 DuckDB（shuffle-window 随机导入）")
    print(f"  - cells: {global_cell_id:,}")
    print(f"  - nnz:   {global_data_id:,}")

def _write_shuffle_window_to_duckdb(
    window_adatas,
    conn,
    global_cell_id,
    global_indptr_id,
    global_indptr_offset,
    global_data_id,
    var_written,
):
    """
    将多个 batch 的 AnnData 合并成一个 window，
    对 window 内所有 cell 统一随机打乱，
    然后整体写入 DuckDB。
    """

    # =====================================================
    # 1. 合并 window 内的多个 batch
    # =====================================================
    adata_window = sc.concat(
        window_adatas,
        axis=0,
        join="outer",
        merge="first",
        index_unique=None,
    )

    # =====================================================
    # 2. window 内所有 cell 统一随机
    # =====================================================
    if adata_window.n_obs > 1:
        perm = np.random.permutation(adata_window.n_obs)
        adata_window = adata_window[perm].copy()

    # window 统计
    window_cells = adata_window.n_obs

    if sparse.issparse(adata_window.X):
        window_nnz = adata_window.X.nnz
    else:
        window_nnz = np.count_nonzero(adata_window.X)

    # =====================================================
    # 3. 写入 obs
    # =====================================================
    global_cell_id = _append_obs_rows(
        adata_window,
        conn,
        start_cell_id=global_cell_id,
    )

    # =====================================================
    # 4. 写入 var，只写一次
    # =====================================================
    if not var_written:
        _append_var(adata_window, conn)
        var_written = True

    # =====================================================
    # 5. 写入 X_CSRO_data / X_CSRO_indptr
    # =====================================================
    (
        global_indptr_id,
        global_indptr_offset,
        global_data_id,
    ) = _append_X_CSRO_chunk_fast(
        adata_window,
        conn,
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
    )



''' 大文件读取, cell 级 纯随机 导入， 只支持 h5ad格式'''
# 819200  耗时 8 分钟左右     2,840,130 × 24,552   2:03:48
# def load_big_h5ad_to_duckdb_random(
#     h5ad_path: str,
#     atlas,
#     batch_size: int = 4096,
# ):
#
#     mega_batch_size = batch_size * 4
#
#     #  1️⃣ 连接数据库
#     conn = atlas.connect("r+")
#
#     # 全局游标
#     global_cell_id = 0
#     global_indptr_id = 0
#     global_indptr_offset = 0
#     global_data_id = 0
#
#     obs_written = False
#     var_written = False
#
#     # 2️⃣ backed 打开
#     adata_backed = sc.read_h5ad(h5ad_path, backed="r")
#     n_cells = adata_backed.n_obs
#
#     # 生成全局随机索引
#     global_perm = np.arange(n_cells, dtype=np.int32)
#     np.random.shuffle(global_perm)
#
#     # 3️⃣ 动态建表
#     _create_obs_table_from_adata(conn, adata_backed[:1])
#     _create_var_table_from_adata(conn, adata_backed[:1])
#     _create_csro_tables(conn)
#
#     print(f"[INFO] 数据集维度: {adata_backed.n_obs:,} × {adata_backed.n_vars:,}")
#     print("[INFO] 使用 scanpy backed + mega-batch + 全局游标模式 + 全局随机顺序")
#
#     # 4️⃣ mega-batch / mini-batch 导入
#     for mega_start in tqdm(
#         range(0, n_cells, mega_batch_size),
#         desc="Mega-batch（随机顺序读取）",
#     ):
#         mega_end = min(mega_start + mega_batch_size, n_cells)
#
#         # 使用随机索引读取
#         mega_idx = global_perm[mega_start:mega_end] # 全局随机索引
#         mega = adata_backed[mega_idx].to_memory()
#
#         # 按batch_size分批导入
#         for start in range(0, mega.n_obs, batch_size):
#             end = min(start + batch_size, mega.n_obs)
#             adata = mega[start:end]
#
#             # ---------------- batch 导入 obs ----------------
#             global_cell_id = _append_obs_rows(
#                 adata,
#                 conn,
#                 start_cell_id=global_cell_id,
#             )
#
#             # ---------------- 导入 var（一次） ----------------
#             if not var_written:
#                 _append_var(adata, conn)
#                 var_written = True
#
#             # ---------------- batch 导入 X（CSRO） ----------------
#             (
#                 global_indptr_id,
#                 global_indptr_offset,
#                 global_data_id,
#             ) = _append_X_CSRO_chunk_fast(
#                 adata,
#                 conn,
#                 base_cell_id=global_cell_id - adata.n_obs,
#                 global_indptr_id=global_indptr_id,
#                 global_indptr_offset=global_indptr_offset,
#                 global_data_id=global_data_id,
#             )
#
#         del mega
#         gc.collect()
#
#     # 5️⃣ 主键
#     conn.execute("ALTER TABLE obs ADD PRIMARY KEY (atlas_cell_id)")
#     conn.execute("ALTER TABLE var ADD PRIMARY KEY (atlas_gene_id)")
#
#     # 6️⃣ obsm / varm
#     _add_obsm_from_h5ad(h5ad_path, atlas)
#     _add_varm_from_h5ad(h5ad_path, atlas)
#
#     print("✔ 全部数据成功导入 DuckDB（含 obsm / varm）")


def load_big_h5ad_to_duckdb_random(
    h5ad_path: str,
    atlas,
    batch_size: int = 4096,
    mega_batch_factor: int = 2,  # ✅ 修改 1：原来固定 batch_size * 4，现在参数化
    commit_every: int = 10,      # ✅ 修改 2：新增，每多少个 mini-batch 提交一次
    gc_every: int = 20,          # ✅ 修改 3：新增，每多少个 mega-batch 做一次 gc
):
    """
    大文件读取，cell 级纯随机导入，只支持 h5ad 格式。

    原逻辑：
    - 生成全局随机 cell 索引 global_perm
    - 每次随机读取 mega-batch
    - 再按 batch_size 分 mini-batch 写入 DuckDB

    ✅ 本版新增优化：
    1. mega_batch_size 参数化
    2. 使用 _append_X_CSRO_chunk_fast()，Arrow 写入
    3. 外层增加 BEGIN / COMMIT，减少自动提交
    4. commit_every 控制分批提交
    5. gc_every 控制 GC 频率
    6. 增加 read / write / nnz 简单统计
    """

    # =====================================================
    # ✅ 修改 4：mega_batch_size 参数化
    # 原来：
    # mega_batch_size = batch_size * 4
    # =====================================================
    mega_batch_size = batch_size * mega_batch_factor

    #  1️⃣ 连接数据库
    conn = atlas.connect("r+")
    atlas.connection = conn   # ✅ 修改 5：补上 atlas.connection，后面 obsm/varm 会用

    # 全局游标
    global_cell_id = 0
    global_indptr_id = 0
    global_indptr_offset = 0
    global_data_id = 0

    var_written = False

    # 2️⃣ backed 打开
    adata_backed = sc.read_h5ad(h5ad_path, backed="r")
    n_cells = adata_backed.n_obs

    # 生成全局随机索引
    global_perm = np.arange(n_cells, dtype=np.int64)
    np.random.shuffle(global_perm)

    # 3️⃣ 动态建表
    _create_obs_table_from_adata(conn, adata_backed[:1])
    _create_var_table_from_adata(conn, adata_backed[:1])
    _create_csro_tables(conn)

    print(f"[INFO] 数据集维度: {adata_backed.n_obs:,} × {adata_backed.n_vars:,}")
    print("[INFO] 使用 scanpy backed + mega-batch + 全局游标模式 + 全局随机顺序")
    print(f"[INFO] batch_size = {batch_size:,}")
    print(f"[INFO] mega_batch_size = {mega_batch_size:,}")
    print(f"[INFO] mega_batch_factor = {mega_batch_factor:,}")
    print(f"[INFO] commit_every = {commit_every:,}")
    print(f"[INFO] gc_every = {gc_every:,}")

    mini_batch_counter = 0

    # =====================================================
    # ✅ 修改 6：事务放在大循环外
    # =====================================================
    conn.execute("BEGIN TRANSACTION")

    try:
        # 4️⃣ mega-batch / mini-batch 导入
        for mega_i, mega_start in enumerate(
            tqdm(
                range(0, n_cells, mega_batch_size),
                desc="Mega-batch（随机顺序读取）",
            )
        ):
            mega_end = min(mega_start + mega_batch_size, n_cells)

            # 使用随机索引读取
            mega_idx = global_perm[mega_start:mega_end]

            t0 = time.time()

            # 真正触发磁盘读取
            mega = adata_backed[mega_idx].to_memory()

            t_read = time.time() - t0

            # 当前 mega 的 nnz
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
                ) = _append_X_CSRO_chunk_fast(   # ✅ 修改 7：使用 Arrow 加速版
                    adata,
                    conn,
                    base_cell_id=global_cell_id - adata.n_obs,
                    global_indptr_id=global_indptr_id,
                    global_indptr_offset=global_indptr_offset,
                    global_data_id=global_data_id,
                )

                t_write = time.time() - t1

                mini_batch_counter += 1

                # =====================================================
                # ✅ 修改 8：每 commit_every 个 mini-batch 提交一次
                # =====================================================
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

            # =====================================================
            # ✅ 修改 9：不要每个 mega 都 gc.collect()
            # =====================================================
            if (mega_i + 1) % gc_every == 0:
                gc.collect()

        # 最后提交
        conn.execute("COMMIT")

    except Exception:
        conn.execute("ROLLBACK")
        raise

    # 5️⃣ 主键
    conn.execute("ALTER TABLE obs ADD PRIMARY KEY (atlas_cell_id)")
    conn.execute("ALTER TABLE var ADD PRIMARY KEY (atlas_gene_id)")

    # =====================================================
    # ⚠️ 注意：
    # cell 级随机导入后，obs / X 已经不是原 h5ad 顺序。
    # 所以 obsm 如果直接按原顺序导入，会和 obs 错位。
    # 因此这里建议不要导入 obsm。
    # varm 是 gene 维度，一般没问题。
    # =====================================================
    # _add_obsm_from_h5ad(h5ad_path, atlas)
    _add_varm_from_h5ad(h5ad_path, atlas)

    try:
        adata_backed.file.close()
    except Exception:
        pass

    print("✔ 全部数据成功导入 DuckDB（cell 级随机，不含 obsm）")
    print(f"  - cells: {global_cell_id:,}")
    print(f"  - nnz:   {global_data_id:,}")


'''  大文件读取，cell 顺序 导入, 只支持 h5ad 格式'''
# 819200   耗时 10:54      2,840,130 × 24,552     30:08
# def load_big_h5ad_to_duckdb(
#     h5ad_path: str,
#     atlas,
#     batch_size: int = 4096,
# ):
#     """
#     从 h5ad 文件流式导入 DuckDB（全局游标版）
#     关键工程原则：
#       - scanpy backed：避免一次性加载 X
#       - mega-batch   ：顺序磁盘 IO（HDF5 友好） ; mega-batch = “一次从磁盘顺序读多大”
#       - mini-batch   ： 每批导入的细胞数量 ; mini-batch = “一次在内存里处理多大”
#       - CSR-only     ：不存稠密矩阵
#       - 全局游标     ：
#     """
#
#     mega_batch_size = batch_size * 4  # mega-batch：控制磁盘 IO 行为（性能参数，通常 = mini-batch × 常数）
#
#     #  1️⃣ 连接数据库
#     conn = atlas.connect("r+")
#
#     # 全局游标（★ 本版本的核心 ★）
#     global_cell_id = 0          # obs.id 起始游标
#     global_indptr_id = 0        # X_CSRO_indptr.id 起始游标
#     global_indptr_offset = 0    # 全局 nnz 偏移（CSR indptr 语义）
#     global_data_id = 0          # X_CSRO_data.id 起始游标
#
#     obs_written = False
#     var_written = False
#
#     # 2️⃣ backed 打开（只用于 schema + X）
#     adata_backed = sc.read_h5ad(h5ad_path, backed="r") # backed="r" 磁盘后端，只读（lazy loading）
#
#     # 不把 X / obs / var 真正读进内存 ； 只有在切片或 .to_memory() 时才触发磁盘 IO
#     n_cells = adata_backed.n_obs
#
#     # 3️⃣ 动态建表(动态 schema）
#     _create_obs_table_from_adata(conn, adata_backed[:1])
#     _create_var_table_from_adata(conn, adata_backed[:1])
#      # 从 h5ad 文件中，只抽取 1 个 cell，
#      # 用它来读取 obs / var 的字段结构（列名 + dtype），
#      # 然后在 DuckDB 里创建 obs / var 表结构，但不导入任何数据。
#
#     _create_csro_tables(conn)
#     print(f"[INFO] 数据集维度: {adata_backed.n_obs:,} × {adata_backed.n_vars:,}")
#     print("[INFO] 使用 scanpy backed + mega-batch + 全局游标模式")
#
#     # 4️⃣ mega-batch / mini-batch 导入
#     for mega_start in tqdm(
#         range(0, n_cells, mega_batch_size),
#         desc="Mega-batch（磁盘顺序读取）",
#     ):
#         mega_end = min(mega_start + mega_batch_size, n_cells)
#
#         # ★ 真正触发磁盘读取的地方
#         mega = adata_backed[mega_start:mega_end].to_memory()
#
#         # 按batch_size分批导入
#         for start in range(0, mega.n_obs, batch_size):
#             end = min(start + batch_size, mega.n_obs)
#             adata = mega[start:end]
#
#             # ---------------- batch 导入 obs ----------------
#             global_cell_id = _append_obs_rows(
#                 adata,
#                 conn,
#                 start_cell_id=global_cell_id,
#             )
#
#             # ---------------- 导入 var（一次） ----------------
#             if not var_written:
#                 _append_var(adata, conn)
#                 var_written = True
#
#             # ---------------- batch 导入 X（CSRO） ----------------
#             (
#                 global_indptr_id,
#                 global_indptr_offset,
#                 global_data_id,
#             ) = _append_X_CSRO_chunk_fast(
#                 adata,
#                 conn,
#                 base_cell_id=global_cell_id - adata.n_obs,
#                 global_indptr_id=global_indptr_id,
#                 global_indptr_offset=global_indptr_offset,
#                 global_data_id=global_data_id,
#             )
#
#         # mega-batch 结束，释放内存
#         del mega
#         gc.collect()
#
#     # 5️⃣ 主键（非常重要：必须在数据写完之后）
#     conn.execute("ALTER TABLE obs ADD PRIMARY KEY (atlas_cell_id)")
#     conn.execute("ALTER TABLE var ADD PRIMARY KEY (atlas_gene_id)")
#
#     # 6️⃣ 导入 obsm / varm
#     _add_obsm_from_h5ad(h5ad_path, atlas)
#     _add_varm_from_h5ad(h5ad_path, atlas)
#
#     print("✔ 全部数据成功导入 DuckDB（含 obsm / varm）")


# todo 优化， 多次提交加GC优化
def load_big_h5ad_to_duckdb(
    h5ad_path: str,
    atlas,
    batch_size: int = 4096,
    mega_batch_factor: int = 1 ,   # ✅ 修改 1：不要固定 * 4，改成可控参数
    commit_every: int = 5,       # ✅ 修改 2：每多少个 mini-batch 提交一次
    gc_every: int = 20,           # ✅ 修改 3：每多少个 mega-batch 做一次 gc
):
    """
    从 h5ad 文件流式导入 DuckDB（事务优化版）

    ✅ 修改点：
    1. mega_batch_size 可控，默认 batch_size * 2，更稳
    2. 多个 mini-batch 共用一个事务，减少自动提交
    3. 不再每个 mega-batch 都 gc.collect()
    4. 增加简单耗时统计，方便判断慢在哪里
    """

    # =====================================================
    # ✅ 修改 1：mega_batch_size 改成可控
    # 原来：
    # mega_batch_size = batch_size * 4
    # =====================================================
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

    # ✅ 新增：简单判断 h5ad.X 格式
    x_format = _print_h5ad_X_format(h5ad_path)

    _create_obs_table_from_adata(conn, adata_backed[:1])
    _create_var_table_from_adata(conn, adata_backed[:1])
    _create_csro_tables(conn)

    print(f"[INFO] 数据集维度: {adata_backed.n_obs:,} × {adata_backed.n_vars:,}")
    print("[INFO] 使用 scanpy backed + mega-batch + 全局游标模式")
    print(f"[INFO] batch_size = {batch_size:,}")
    print(f"[INFO] mega_batch_size = {mega_batch_size:,}")
    print(f"[INFO] commit_every = {commit_every:,}")

    mini_batch_counter = 0

    # =====================================================
    # ✅ 修改 2：事务放在大循环外
    # =====================================================
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
                ) = _append_X_CSRO_chunk_fast(
                    adata,
                    conn,
                    base_cell_id=global_cell_id - adata.n_obs,
                    global_indptr_id=global_indptr_id,
                    global_indptr_offset=global_indptr_offset,
                    global_data_id=global_data_id,
                )

                t_write = time.time() - t1

                mini_batch_counter += 1

                # =====================================================
                # ✅ 修改 3：每 commit_every 个 mini-batch 提交一次
                # =====================================================
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

            # =====================================================
            # ✅ 修改 4：不要每个 mega 都 gc.collect()
            # =====================================================
            if (mega_i + 1) % gc_every == 0:
                gc.collect()

        # 最后提交
        conn.execute("COMMIT")

    except Exception:
        conn.execute("ROLLBACK")
        raise

    # 5️⃣ 主键：必须在数据写完之后
    conn.execute("ALTER TABLE obs ADD PRIMARY KEY (atlas_cell_id)")
    conn.execute("ALTER TABLE var ADD PRIMARY KEY (atlas_gene_id)")

    # 6️⃣ 导入 obsm / varm
    _add_obsm_from_h5ad(h5ad_path, atlas)
    _add_varm_from_h5ad(h5ad_path, atlas)

    try:
        adata_backed.file.close()
    except Exception:
        pass

    print("✔ 全部数据成功导入 DuckDB（含 obsm / varm）")
    print(f"  - cells: {global_cell_id:,}")
    print(f"  - nnz:   {global_data_id:,}")


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

''' 推断数据类型'''
def _infer_duckdb_type_from_series(s: pd.Series) -> str:
    """根据 pandas dtype 推断 DuckDB 类型"""
    if pd.api.types.is_integer_dtype(s):
        return "BIGINT"
    if pd.api.types.is_float_dtype(s):
        return "DOUBLE"
    if pd.api.types.is_bool_dtype(s):
        return "BOOLEAN"
    return "VARCHAR"


''' 建立obs表'''
# def _create_obs_table_from_adata(conn, adata):
#
#     cols = ["atlas_cell_id INTEGER", "atlas_cell_name VARCHAR"]
#
#     for col in adata.obs.columns:
#         duck_type = _infer_duckdb_type_from_series(adata.obs[col])
#         cols.append(f'"{col}" {duck_type}')
#
#     ddl = f"""
#     CREATE OR REPLACE TABLE obs (
#         {", ".join(cols)}
#     )
#     """
#     conn.execute(ddl)

def _create_obs_table_from_adata(conn, adata):
    """
    建立 obs 表。

    固定系统字段：
        atlas_cell_id   INTEGER
        atlas_cell_name VARCHAR

    如果 adata.obs 中已经存在 atlas_cell_id / atlas_cell_name，
    建表时跳过，避免重复列。
    """

    # ✅ 系统保留字段：由 scAtlasPy 统一创建
    reserved_cols = {"atlas_cell_id", "atlas_cell_name"}

    # ✅ 强制使用你要求的类型
    cols = [
        "atlas_cell_id INTEGER",
        "atlas_cell_name VARCHAR",
    ]

    for col in adata.obs.columns:
        # ✅ 跳过来源 h5ad 中已有的旧系统字段
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

''' 建立var表'''
# def _create_var_table_from_adata(conn, adata):
#
#     cols = ["atlas_gene_id USMALLINT", "atlas_gene_name VARCHAR"]
#
#     for col in adata.var.columns:
#         duck_type = _infer_duckdb_type_from_series(adata.var[col])
#         cols.append(f'"{col}" {duck_type}')
#
#     ddl = f"""
#     CREATE OR REPLACE TABLE var (
#         {", ".join(cols)}
#     )
#     """
#     conn.execute(ddl)

def _create_var_table_from_adata(conn, adata):
    """
    建立 var 表。

    固定系统字段：
        atlas_gene_id   USMALLINT
        atlas_gene_name VARCHAR

    如果 adata.var 中已经存在 atlas_gene_id / atlas_gene_name，
    建表时跳过，避免重复列。
    """

    # ✅ 系统保留字段：由 scAtlasPy 统一创建
    reserved_cols = {"atlas_gene_id", "atlas_gene_name"}

    # ✅ 强制使用你要求的类型
    cols = [
        "atlas_gene_id USMALLINT",
        "atlas_gene_name VARCHAR",
    ]

    for col in adata.var.columns:
        # ✅ 跳过来源 h5ad 中已有的旧系统字段
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

''' 建立CSRO存储结构 '''
def _create_csro_tables(conn):

    conn.execute(
        """ -- 不存第一个0值
        CREATE OR REPLACE TABLE X_CSRO_indptr ( 
            atlas_cell_id   INTEGER,  --  cell id , 改成 INTEGER int32  −21 4748 3648 到 21 4748 3647
            -- atlas_cell_name VARCHAR, 用不上，暂时注释掉
            indptr BIGINT
        )
        """
    )
    conn.execute(
        """
        CREATE OR REPLACE TABLE X_CSRO_data (
            id BIGINT,
            atlas_cell_id INTEGER,    --  cell id , 改成 INTEGER，int32  −21 4748 3648 到 21 4748 3647
            atlas_gene_id USMALLINT,  --  gene id , indices 无符号 int16 0 ~ 65535 之间
            data REAL                 --  float 32 单精度浮点数（4字节）
        )
        """
    )


''' 导入 obs 表 '''
# def _append_obs_rows(adata, conn, start_cell_id: int) -> int:
#
#     n = adata.n_obs
#
#     obs_df = adata.obs.copy()
#     obs_df["atlas_cell_name"] = adata.obs.index.astype(str)
#     obs_df["atlas_cell_id"] = np.arange(start_cell_id, start_cell_id + n, dtype=np.int64)
#
#     obs_df = obs_df[
#         ["atlas_cell_id", "atlas_cell_name"] + [c for c in obs_df.columns if c not in ("atlas_cell_id", "atlas_cell_name")]
#     ]
#
#     conn.register("obs_df", obs_df)
#     conn.execute("INSERT INTO obs SELECT * FROM obs_df")
#     conn.unregister("obs_df")
#
#     return start_cell_id + n

def _append_obs_rows(adata, conn, start_cell_id: int) -> int:
    """
    导入 obs 表。

    固定重新生成：
        atlas_cell_id   INTEGER
        atlas_cell_name VARCHAR

    如果来源 adata.obs 中已经有 atlas_cell_id / atlas_cell_name，
    先删除旧列，再重新生成。
    """

    n = adata.n_obs

    obs_df = adata.obs.copy()

    # ✅ 删除来源 h5ad 中已有的旧系统字段
    for c in ["atlas_cell_id", "atlas_cell_name"]:
        if c in obs_df.columns:
            obs_df = obs_df.drop(columns=[c])

    # ✅ 重新生成当前数据库自己的 atlas_cell_id / atlas_cell_name
    # 注意：DuckDB INTEGER 对应 int32
    obs_df["atlas_cell_id"] = np.arange(
        start_cell_id,
        start_cell_id + n,
        dtype=np.int32,
    )

    obs_df["atlas_cell_name"] = adata.obs.index.astype(str)

    # ✅ 固定列顺序：系统字段永远在最前面
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

''' 导入 var 表 '''
# def _append_var(adata, conn):
#
#     var_df = adata.var.copy()
#     var_df["atlas_gene_name"] = adata.var.index.astype(str)
#     var_df["atlas_gene_id"] = np.arange(adata.n_vars, dtype=np.uint16)
#
#     var_df = var_df[
#         ["atlas_gene_id", "atlas_gene_name"] + [c for c in var_df.columns if c not in ("atlas_gene_id", "atlas_gene_name")]
#     ]
#
#     conn.register("var_df", var_df)
#     conn.execute("INSERT INTO var SELECT * FROM var_df")
#     conn.unregister("var_df")

def _append_var(adata, conn):
    """
    导入 var 表。

    固定重新生成：
        atlas_gene_id   USMALLINT
        atlas_gene_name VARCHAR

    如果来源 adata.var 中已经有 atlas_gene_id / atlas_gene_name，
    先删除旧列，再重新生成。
    """

    var_df = adata.var.copy()

    # ✅ 删除来源 h5ad 中已有的旧系统字段
    for c in ["atlas_gene_id", "atlas_gene_name"]:
        if c in var_df.columns:
            var_df = var_df.drop(columns=[c])

    # ✅ 重新生成当前数据库自己的 atlas_gene_id / atlas_gene_name
    # 注意：DuckDB USMALLINT 对应 uint16
    var_df["atlas_gene_id"] = np.arange(
        adata.n_vars,
        dtype=np.uint16,
    )

    var_df["atlas_gene_name"] = adata.var.index.astype(str)

    # ✅ 固定列顺序：系统字段永远在最前面
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

''' 导入 X_CSRO 表 '''
def _append_X_CSRO_chunk(
    adata,
    conn,
    *,
    base_cell_id: int,
    global_indptr_id: int,
    global_indptr_offset: int,
    global_data_id: int,
):
    """
      indptr 不存第一个0值
    """

    X = adata.X

    # 确保 CSR
    if not sparse.issparse(X):
        X = sparse.csr_matrix(X)
    else:
        X = X.tocsr()

    indptr = X.indptr.astype(np.int64)
    indices = X.indices.astype(np.uint16)
    data = X.data.astype(np.float32)

    # ================= indptr ================
    row_nnz = np.diff(indptr)
    adj_indptr = indptr[1:] + global_indptr_offset # 不存 indptr[0]

    indptr_df = pd.DataFrame(
        {
            "atlas_cell_id": np.arange(
                global_indptr_id,
                global_indptr_id + len(adj_indptr),
                dtype=np.int32,
            ),
            "atlas_cell_name": adata.obs.index.astype(str),
            "indptr": adj_indptr,
        }
    )

    conn.register("indptr_df", indptr_df)
    conn.execute("INSERT INTO X_CSRO_indptr SELECT * FROM indptr_df")
    conn.unregister("indptr_df")

    global_indptr_id += len(adj_indptr)

    # ================= data =================
    nnz = len(data)
    if nnz > 0:
        cell_index = np.repeat(
            np.arange(base_cell_id, base_cell_id + adata.n_obs, dtype=np.int32),
            row_nnz,
        )

        data_df = pd.DataFrame(
            {
                "id": np.arange(global_data_id, global_data_id + nnz, dtype=np.int64),
                "atlas_cell_id": cell_index,
                "atlas_gene_id": indices,
                "data": data,
            }
        )

        conn.register("data_df", data_df)
        conn.execute("INSERT INTO X_CSRO_data SELECT * FROM data_df")
        conn.unregister("data_df")

        global_data_id += nnz
        global_indptr_offset += nnz

    return global_indptr_id, global_indptr_offset, global_data_id


# todo 修改
''' 导入 X_CSRO 表：Arrow 加速版 '''
def _append_X_CSRO_chunk_fast(
    adata,
    conn,
    *,
    base_cell_id: int,
    global_indptr_id: int,
    global_indptr_offset: int,
    global_data_id: int,
):
    """
    Arrow 加速版 X_CSRO 导入函数

    对应表结构：
    X_CSRO_indptr(
        atlas_cell_id INTEGER,
        indptr BIGINT
    )

    X_CSRO_data(
        id BIGINT,
        atlas_cell_id INTEGER,
        atlas_gene_id USMALLINT,
        data REAL
    )

    ✅ 最小化修改点：
    1. indptr_df: pandas.DataFrame -> pyarrow.Table
    2. data_df: pandas.DataFrame -> pyarrow.Table
    3. 去掉 X_CSRO_indptr.atlas_cell_name
    4. CSR 已经是 csr_matrix 时，不重复 .tocsr()
    5. astype(copy=False)，减少不必要复制
    """

    X = adata.X

    # =====================================================
    # ✅ 修改 1：避免不必要的 CSR 转换
    # 原来：
    # if not sparse.issparse(X):
    #     X = sparse.csr_matrix(X)
    # else:
    #     X = X.tocsr()
    # =====================================================
    if not sparse.issparse(X):
        X = sparse.csr_matrix(X)
    elif not sparse.isspmatrix_csr(X):
        X = X.tocsr()

    # =====================================================
    # ✅ 修改 2：astype(copy=False)，能不复制就不复制
    # =====================================================
    indptr = X.indptr.astype(np.int64, copy=False)

    # 保持你原来的 uint16 / USMALLINT 设计
    # ⚠️ gene 数必须 <= 65535
    indices = X.indices.astype(np.uint16, copy=False)

    data = X.data.astype(np.float32, copy=False)

    # ================= indptr =================
    row_nnz = np.diff(indptr)

    # 不存 indptr[0]，只存 indptr[1:]
    adj_indptr = indptr[1:] + np.int64(global_indptr_offset)

    # =====================================================
    # ✅ 修改 3：indptr_df -> pyarrow.Table
    # ✅ 修改 4：去掉 atlas_cell_name 字符串列
    #
    # 原来：
    # indptr_df = pd.DataFrame({
    #     "atlas_cell_id": ...,
    #     "atlas_cell_name": adata.obs.index.astype(str),
    #     "indptr": adj_indptr,
    # })
    #
    # 现在：
    # 只写 atlas_cell_id + indptr
    # =====================================================
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
        INSERT INTO X_CSRO_indptr (
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
        # =====================================================
        # 这一行仍然保留：
        # CSR -> CSRO_data 时，每个非零值都需要对应一个 cell_id
        # 这是 data 表构造里的主要内存成本之一
        # =====================================================
        cell_index = np.repeat(
            np.arange(
                base_cell_id,
                base_cell_id + adata.n_obs,
                dtype=np.int32,
            ),
            row_nnz,
        )

        # =====================================================
        # ✅ 修改 5：data_df -> pyarrow.Table
        #
        # 原来：
        # data_df = pd.DataFrame({
        #     "id": ...,
        #     "atlas_cell_id": cell_index,
        #     "atlas_gene_id": indices,
        #     "data": data,
        # })
        # =====================================================
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
            INSERT INTO X_CSRO_data (
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

''' 导入 obsm '''
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
                df["atlas_cell_id"] = np.arange(start, end, dtype=np.int64)
                df = df[["atlas_cell_id"] + [c for c in df.columns if c != "atlas_cell_id"]]

                conn.register("obsm_df", df)
                conn.execute(f"INSERT INTO obsm_{key} SELECT * FROM obsm_df")
                conn.unregister("obsm_df")

    logger.info("obsm 导入完成")


''' 导入 varm '''
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


''' 小文件读取 ： 支持多种数据格式的导入 '''
def load_small_to_duckdb( file_path , atlas:Atlas):

    print("小文件读取 , 开始导入数据...")
    adata = read_smart(file_path)
    load_AnnData(adata,atlas)
    print("✔ 全部数据成功导入 DuckDB ")


''' 多种数据格式的导入 '''
def read_smart(file_path, **kwargs):

    # 获取文件后缀名（小写形式）
    file_ext = os.path.splitext(file_path)[1].lower()

    # 根据文件后缀选择对应的读取方法
    if file_ext == '.h5ad':
        # h5ad格式 - scanpy原生格式
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


''' 导入小数据 '''
def load_AnnData(adata:AnnData, atlas:Atlas):

    try:
        # 1. 准备数据表
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
            _add_X_CSRO_chunked(adata,atlas,chunk_size=4096) # 分块导入X表数据
            end_time = time.time()
            logger.info("######## X表的导入用时为： " + str(end_time - start_time))
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


''' 导入 obs 表 '''
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


''' 导入 var  '''
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


''' 导入 obsm '''
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


''' 导入 varm '''
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

        # ★ atlas_gene_id = var.atlas_gene_id 的语义
        df.insert(0, "atlas_gene_id", np.arange(n_genes, dtype=np.uint16))

        conn.register("varm_df", df)
        conn.execute(f"""
            CREATE OR REPLACE TABLE varm_{key} AS
            SELECT * FROM varm_df
        """)
        conn.unregister("varm_df")

    logger.info("varm 导入完成（统一 schema）")


''' 导入 X_CSRO '''
def _add_X_CSRO_chunked( adata: AnnData, atlas: Atlas, chunk_size: int = 4096):

    logger.info("开始导入 CSRO ")

    conn = atlas.connect("r+")
    atlas.connection = conn

    cell_names = adata.obs.index
    n_cells = adata.n_obs

    # ===================== 建表 =====================
    conn.execute(
        """ -- 不存第一个0值
        CREATE OR REPLACE TABLE X_CSRO_indptr ( 
            atlas_cell_id   INTEGER,  --  cell id , 改成 INTEGER int32  −21 4748 3648 到 21 4748 3647
            atlas_cell_name VARCHAR,
            indptr BIGINT
        )
        """
    )
    conn.execute(
        """
        CREATE OR REPLACE TABLE X_CSRO_data (
            id BIGINT,
            atlas_cell_id INTEGER,    --  cell id , 改成 INTEGER，int32  −21 4748 3648 到 21 4748 3647
            atlas_gene_id USMALLINT,  --  gene id , indices 无符号 int16 0 ~ 65535 之间
            data REAL                 --  float 32 单精度浮点数（4字节）
        )
        """
    )

    conn.execute("BEGIN TRANSACTION")

    try:
        total_chunks = (n_cells + chunk_size - 1) // chunk_size

        global_data_counter = np.int64(0)
        global_indptr_offset = np.int64(0)

        for chunk_idx in tqdm(range(total_chunks), desc="导入 CSR chunks"):
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
                "atlas_cell_name": cell_names[start:end],
                "indptr": adj_indptr
            })

            conn.execute("INSERT INTO X_CSRO_indptr SELECT * FROM indptr_df")

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

                conn.execute("INSERT INTO X_CSRO_data SELECT * FROM data_df")

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



# todo
# 10 个 h5ad 文件
#     ↓
# 第 1 个文件：建表 + 写 var + 写 obs + 写 X
# 第 2 个文件：不建表，只追加 obs + X
# 第 3 个文件：继续追加 obs + X
# ...
# 第 10 个文件：继续追加 obs + X
#     ↓
# 最后统一加主键

# for h5ad_path in h5ad_paths:
#     不重置 global_cell_id
#     不重置 global_indptr_offset
#     不重置 global_data_id
#     只在第一个文件建表
#     只在第一个文件写 var
#     后续文件只 INSERT 追加

# todo 多个 h5ad 文件顺序追加导入 DuckDB。
def load_big_h5ad_list_to_duckdb_random(
    h5ad_paths: list[str],
    atlas,
    batch_size: int = 4096,
    check_var: bool = True,
    random_seed: int | None = None,
):
    """
    多个 h5ad 文件追加导入 DuckDB，并且每个文件内部 cell 随机导入。

    特点：
    - 文件之间：按 h5ad_paths 顺序追加
    - 文件内部：随机 cell 顺序导入
    - var：只导入第一个文件
    - obs / X_CSRO_indptr / X_CSRO_data：跨文件连续追加
    - atlas_cell_id：按照随机后的导入顺序重新编号
    - X_CSRO_data.id：按照随机后的导入顺序连续编号

    适合：
    - 1亿细胞拆成多个 h5ad 文件
    - 希望导入后直接顺序 minibatch 就具有随机性

    注意：
    - 不导入 obsm / varm
    - 所有文件必须 gene 数量和 gene 顺序一致
    """

    import gc
    import numpy as np
    import scanpy as sc
    from tqdm import tqdm

    if isinstance(h5ad_paths, str):
        h5ad_paths = [h5ad_paths]

    if len(h5ad_paths) == 0:
        raise ValueError("h5ad_paths 不能为空")

    if random_seed is not None:
        np.random.seed(random_seed)

    mega_batch_size = batch_size * 8

    conn = atlas.connect("r+")
    atlas.connection = conn

    # =====================================================
    # 全局游标：跨文件连续累加
    # =====================================================
    global_cell_id = 0
    global_indptr_id = 0
    global_indptr_offset = 0
    global_data_id = 0

    var_written = False
    ref_var_names = None
    ref_n_genes = None

    print("\n==== load_big_h5ad_list_to_duckdb_random ====")
    print(f"[INFO] 文件数量: {len(h5ad_paths)}")
    print(f"[INFO] batch_size: {batch_size:,}")
    print(f"[INFO] mega_batch_size: {mega_batch_size:,}")
    print("[INFO] 文件之间顺序追加，文件内部随机导入")

    for file_idx, h5ad_path in enumerate(h5ad_paths):

        print("\n" + "=" * 80)
        print(f"[FILE {file_idx + 1}/{len(h5ad_paths)}] {h5ad_path}")
        print("=" * 80)

        # =====================================================
        # 1. backed 打开当前 h5ad
        # =====================================================
        adata_backed = sc.read_h5ad(h5ad_path, backed="r")

        n_cells = adata_backed.n_obs
        n_genes = adata_backed.n_vars

        print(f"[INFO] 当前文件维度: {n_cells:,} × {n_genes:,}")

        # =====================================================
        # 2. 第一个文件：建表 + 记录 gene 顺序
        # =====================================================
        if file_idx == 0:
            print("[INIT] 第一个文件：创建 obs / var / CSRO 表")

            _create_obs_table_from_adata(conn, adata_backed[:1])
            _create_var_table_from_adata(conn, adata_backed[:1])
            _create_csro_tables(conn)

            ref_var_names = adata_backed.var.index.astype(str).to_numpy()
            ref_n_genes = n_genes

        # =====================================================
        # 3. 后续文件：检查 gene 数量和顺序
        # =====================================================
        else:
            if check_var:
                if n_genes != ref_n_genes:
                    raise ValueError(
                        f"第 {file_idx + 1} 个文件 gene 数量不一致："
                        f"{n_genes} != {ref_n_genes}"
                    )

                cur_var_names = adata_backed.var.index.astype(str).to_numpy()

                if not np.array_equal(cur_var_names, ref_var_names):
                    raise ValueError(
                        f"第 {file_idx + 1} 个文件 gene 顺序与第一个文件不一致，"
                        f"不能直接追加导入。"
                    )

        # =====================================================
        # 4. 当前文件内部生成随机索引
        # =====================================================
        global_perm = np.arange(n_cells, dtype=np.int64)
        np.random.shuffle(global_perm)

        print("[INFO] 当前文件已生成随机 cell 顺序")

        # =====================================================
        # 5. 当前文件 mega-batch / mini-batch 随机导入
        # =====================================================
        for mega_start in tqdm(
            range(0, n_cells, mega_batch_size),
            desc=f"File {file_idx + 1} Random Mega-batch",
        ):
            mega_end = min(mega_start + mega_batch_size, n_cells)

            # -------------------------------------------------
            # 随机索引读取
            # -------------------------------------------------
            mega_idx = global_perm[mega_start:mega_end]

            # 真正触发磁盘读取
            mega = adata_backed[mega_idx].to_memory()

            for start in range(0, mega.n_obs, batch_size):
                end = min(start + batch_size, mega.n_obs)
                adata = mega[start:end]

                # ---------------- obs 追加 ----------------
                base_cell_id = global_cell_id

                global_cell_id = _append_obs_rows(
                    adata,
                    conn,
                    start_cell_id=global_cell_id,
                )

                # ---------------- var 只写一次 ----------------
                if not var_written:
                    _append_var(adata, conn)
                    var_written = True

                # ---------------- X_CSRO 追加 ----------------
                (
                    global_indptr_id,
                    global_indptr_offset,
                    global_data_id,
                ) = _append_X_CSRO_chunk(
                    adata,
                    conn,
                    base_cell_id=base_cell_id,
                    global_indptr_id=global_indptr_id,
                    global_indptr_offset=global_indptr_offset,
                    global_data_id=global_data_id,
                )

            del mega
            gc.collect()

        # =====================================================
        # 6. 关闭 backed 文件
        # =====================================================
        try:
            adata_backed.file.close()
        except Exception:
            pass

        print(
            f"[DONE] 文件 {file_idx + 1}/{len(h5ad_paths)} 导入完成 | "
            f"累计 cells={global_cell_id:,}, "
            f"累计 nnz={global_data_id:,}, "
            f"indptr_offset={global_indptr_offset:,}"
        )

    # =====================================================
    # 7. 所有文件导入后统一添加主键
    # =====================================================
    print("\n[INFO] 添加主键...")

    conn.execute("ALTER TABLE obs ADD PRIMARY KEY (atlas_cell_id)")
    conn.execute("ALTER TABLE var ADD PRIMARY KEY (atlas_gene_id)")

    print("\n✔ 多文件 h5ad 随机导入 DuckDB 完成")
    print(f"  - files : {len(h5ad_paths):,}")
    print(f"  - cells : {global_cell_id:,}")
    print(f"  - genes : {ref_n_genes:,}")
    print(f"  - nnz   : {global_data_id:,}")

    return {
        "files": len(h5ad_paths),
        "cells": global_cell_id,
        "genes": ref_n_genes,
        "nnz": global_data_id,
    }


''' 基因名清洗 ：先导入，再清洗，var表 '''
def clean_genes_in_database(atlas: Atlas, gene_name_column: str = "atlas_gene_name"):
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
    logger.info(f"开始在数据库 var表 中清洗基因名 ")

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