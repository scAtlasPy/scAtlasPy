import numpy as np
import scanpy as sc
from scipy import sparse
from tqdm import tqdm

def export_duckdb_to_h5ad(
    atlas,
    out_h5ad_path: str,
    *,  # * 后面的参数只能用关键字参数
    batch_size: int = 100_000,
):
    """
    从 DuckDB 中导出数据，重建为 AnnData 并保存为 .h5ad

    支持：
      - 超大规模数据
      - CSR-only X
      - obs / var / obsm / varm
    """

    conn = atlas.connection

    # =========================================================
    # 1️⃣ 读取 obs / var
    # =========================================================
    print("[EXPORT] 读取 obs / var")

    obs = conn.execute("""
        SELECT * FROM obs ORDER BY id
    """).df()

    var = conn.execute("""
        SELECT * FROM var ORDER BY id
    """).df()

    # ★ NOTE: AnnData 要求 index 是 cell_id / gene_id
    obs = obs.set_index("cell_id")
    var = var.set_index("gene_id")

    n_cells = int(obs.shape[0])
    n_genes = int(var.shape[0])

    print(f"[EXPORT] cells={n_cells:,}, genes={n_genes:,}")

    # =========================================================
    # 2️⃣ 重建 CSR indptr
    # =========================================================
    print("[EXPORT] 重建 CSR indptr")

    indptr_df = conn.execute("""
        SELECT indptr
        FROM X_CSR_indptr
        ORDER BY id
    """).df()

    # CSR 规范：len(indptr) = n_cells + 1
    indptr = np.empty(n_cells + 1, dtype=np.int64)
    indptr[0] = 0
    indptr[1:] = indptr_df["indptr"].to_numpy(dtype=np.int64)

    nnz = int(indptr[-1])
    print(f"[EXPORT] nnz={nnz:,}")

    # =========================================================
    # 3️⃣ 分批读取 CSR data（indices + data）
    # =========================================================
    print("[EXPORT] 读取 CSR data")

    indices = np.empty(nnz, dtype=np.int32)
    data = np.empty(nnz, dtype=np.float32)

    offset = 0

    # ★ FIX: 用 Python int 的 range，避免 numpy.int64
    for start in tqdm(
        range(0, nnz, int(batch_size)),
        desc="CSR data"
    ):
        end = min(start + batch_size, nnz)

        # ★ FIX: 参数显式转成 Python int
        start_i = int(start)
        end_i = int(end)

        chunk = conn.execute(
            """
            SELECT indices, data
            FROM X_CSR_data
            WHERE id >= ? AND id < ?
            ORDER BY id
            """,
            [start_i, end_i],
        ).fetchall()

        # ★ NOTE: chunk 的顺序 == id 顺序
        for i, (idx, val) in enumerate(chunk):
            indices[offset + i] = idx
            data[offset + i] = val

        offset += len(chunk)

    assert offset == nnz, f"CSR data 读取数量不一致: {offset} != {nnz}"

    # =========================================================
    # 4️⃣ 构建 CSR matrix
    # =========================================================
    print("[EXPORT] 构建 CSR matrix")

    X = sparse.csr_matrix(
        (data, indices, indptr),
        shape=(n_cells, n_genes),
    )

    # =========================================================
    # 5️⃣ 创建 AnnData
    # =========================================================
    print("[EXPORT] 创建 AnnData")

    adata = sc.AnnData(
        X=X,
        obs=obs,
        var=var,
    )

    # =========================================================
    # 6️⃣ 导入 obsm
    # =========================================================
    print("[EXPORT] 导入 obsm")

    obsm_tables = conn.execute("""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_name LIKE 'obsm_%'
    """).fetchall()

    for (table_name,) in obsm_tables:
        key = table_name.replace("obsm_", "")

        df = conn.execute(f"""
            SELECT *
            FROM {table_name}
            ORDER BY cell_index
        """).df()

        df = df.drop(columns=["cell_index"])
        adata.obsm[key] = df.to_numpy()

        print(f"  - obsm[{key}] shape={adata.obsm[key].shape}")

    # =========================================================
    # 7️⃣ 导入 varm
    # =========================================================
    print("[EXPORT] 导入 varm")

    varm_tables = conn.execute("""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_name LIKE 'varm_%'
    """).fetchall()

    for (table_name,) in varm_tables:
        key = table_name.replace("varm_", "")

        df = conn.execute(f"""
            SELECT *
            FROM {table_name}
            ORDER BY gene_index
        """).df()

        df = df.drop(columns=["gene_index"])
        adata.varm[key] = df.to_numpy()

        print(f"  - varm[{key}] shape={adata.varm[key].shape}")

    # =========================================================
    # 8️⃣ 写出 h5ad
    # =========================================================
    print(f"[EXPORT] 写出 h5ad → {out_h5ad_path}")
    adata.write_h5ad(out_h5ad_path)

    print("✅ 导出完成")




import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm


# def export_duckdb_to_h5ad_streaming(
#     atlas,
#     out_h5ad_path: str,
#     *,
#     batch_size: int = 1_000_000,  # nnz batch
# ):
#     """
#     从 DuckDB 流式导出 h5ad（不经过 AnnData）
#
#     特点：
#       - CSR-only
#       - nnz streaming
#       - 内存 O(batch_size)
#       - scanpy / anndata 完全兼容
#     """
#
#     conn = atlas.connection
#
#     # =========================================================
#     # 1️⃣ 读取 obs / var（必须完整）
#     # =========================================================
#     print("[EXPORT] 读取 obs / var")
#
#     obs = conn.execute("SELECT * FROM obs ORDER BY id").df()
#     var = conn.execute("SELECT * FROM var ORDER BY id").df()
#
#     obs = obs.set_index("cell_id")
#     var = var.set_index("gene_id")
#
#     n_cells = obs.shape[0]
#     n_genes = var.shape[0]
#
#     print(f"[EXPORT] cells={n_cells:,}, genes={n_genes:,}")
#
#     # =========================================================
#     # 2️⃣ 读取 indptr
#     # =========================================================
#     print("[EXPORT] 读取 CSR indptr")
#
#     indptr_df = conn.execute("""
#         SELECT indptr
#         FROM X_CSR_indptr
#         ORDER BY id
#     """).df()
#
#     indptr = np.empty(n_cells + 1, dtype=np.int64)
#     indptr[0] = 0
#     indptr[1:] = indptr_df["indptr"].to_numpy(dtype=np.int64)
#
#     nnz = int(indptr[-1])
#     print(f"[EXPORT] nnz={nnz:,}")
#
#     # =========================================================
#     # 3️⃣ 创建 h5ad 文件
#     # =========================================================
#     print(f"[EXPORT] 创建 h5ad → {out_h5ad_path}")
#
#     with h5py.File(out_h5ad_path, "w") as f:
#
#         # ---------- 基本属性 ----------
#         f.attrs["encoding-type"] = "anndata"
#         f.attrs["encoding-version"] = "0.1.0"
#
#         # =====================================================
#         # 4️⃣ 写 X (CSR, streaming)
#         # =====================================================
#         print("[EXPORT] 写 X (CSR streaming)")
#
#         gX = f.create_group("X")
#
#         d_data = gX.create_dataset(
#             "data",
#             shape=(nnz,),
#             maxshape=(nnz,),
#             dtype="float32",
#             chunks=(min(batch_size, nnz),),
#         )
#
#         d_indices = gX.create_dataset(
#             "indices",
#             shape=(nnz,),
#             maxshape=(nnz,),
#             dtype="int32",
#             chunks=(min(batch_size, nnz),),
#         )
#
#         gX.create_dataset("indptr", data=indptr, dtype="int64")
#         gX.create_dataset("shape", data=np.array([n_cells, n_genes], dtype="int64"))
#
#         offset = 0
#
#         for start in tqdm(
#             range(0, nnz, batch_size),
#             desc="CSR data"
#         ):
#             end = min(start + batch_size, nnz)
#
#             rows = conn.execute(
#                 """
#                 SELECT indices, data
#                 FROM X_CSR_data
#                 WHERE id >= ? AND id < ?
#                 ORDER BY id
#                 """,
#                 [int(start), int(end)],
#             ).fetchall()
#
#             n = len(rows)
#             if n == 0:
#                 continue
#
#             d_indices[offset:offset+n] = [r[0] for r in rows]
#             d_data[offset:offset+n] = [r[1] for r in rows]
#
#             offset += n
#
#         assert offset == nnz, f"nnz mismatch: {offset} != {nnz}"
#
#         # =====================================================
#         # 5️⃣ 写 obs / var
#         # =====================================================
#         print("[EXPORT] 写 obs / var")
#
#         _write_dataframe(f, "obs", obs)
#         _write_dataframe(f, "var", var)
#
#         # =====================================================
#         # 6️⃣ 写 obsm
#         # =====================================================
#         print("[EXPORT] 写 obsm")
#
#         g_obsm = f.create_group("obsm")
#
#         for (table_name,) in conn.execute("""
#             SELECT table_name
#             FROM information_schema.tables
#             WHERE table_name LIKE 'obsm_%'
#         """).fetchall():
#
#             key = table_name.replace("obsm_", "")
#
#             df = conn.execute(f"""
#                 SELECT *
#                 FROM {table_name}
#                 ORDER BY cell_index
#             """).df()
#
#             df = df.drop(columns=["cell_index"])
#             g_obsm.create_dataset(key, data=df.to_numpy())
#
#             print(f"  - obsm[{key}] {df.shape}")
#
#         # =====================================================
#         # 7️⃣ 写 varm
#         # =====================================================
#         print("[EXPORT] 写 varm")
#
#         g_varm = f.create_group("varm")
#
#         for (table_name,) in conn.execute("""
#             SELECT table_name
#             FROM information_schema.tables
#             WHERE table_name LIKE 'varm_%'
#         """).fetchall():
#
#             key = table_name.replace("varm_", "")
#
#             df = conn.execute(f"""
#                 SELECT *
#                 FROM {table_name}
#                 ORDER BY gene_index
#             """).df()
#
#             df = df.drop(columns=["gene_index"])
#             g_varm.create_dataset(key, data=df.to_numpy())
#
#             print(f"  - varm[{key}] {df.shape}")
#
#     print("✅ 导出完成")

def export_duckdb_to_h5ad_streaming(
    atlas,
    out_h5ad_path: str,
    *,
    batch_size: int = 1_000_000,  # nnz batch
):
    """
    从 DuckDB 流式导出 h5ad（不经过 AnnData）

    特点：
      - CSR-only
      - nnz streaming
      - 内存 O(batch_size)
      - scanpy / anndata 完全兼容
    """

    conn = atlas.connection

    # =========================================================
    # 1️⃣ 读取 obs / var
    # =========================================================
    print("[EXPORT] 读取 obs / var")

    obs = conn.execute("SELECT * FROM obs ORDER BY id").df()
    var = conn.execute("SELECT * FROM var ORDER BY id").df()

    obs = obs.set_index("cell_id")
    var = var.set_index("gene_id")

    n_cells = obs.shape[0]
    n_genes = var.shape[0]

    print(f"[EXPORT] cells={n_cells:,}, genes={n_genes:,}")

    # =========================================================
    # 2️⃣ 读取 CSR indptr
    # =========================================================
    print("[EXPORT] 读取 CSR indptr")

    indptr_df = conn.execute("""
        SELECT indptr
        FROM X_CSR_indptr
        ORDER BY id
    """).df()

    indptr = np.empty(n_cells + 1, dtype=np.int64)
    indptr[0] = 0
    indptr[1:] = indptr_df["indptr"].to_numpy(dtype=np.int64)

    nnz = int(indptr[-1])
    print(f"[EXPORT] nnz={nnz:,}")

    # =========================================================
    # 3️⃣ 创建 h5ad 文件
    # =========================================================
    print(f"[EXPORT] 创建 h5ad → {out_h5ad_path}")

    with h5py.File(out_h5ad_path, "w") as f:

        # ---------- 根节点属性 ----------
        f.attrs["encoding-type"] = "anndata"
        f.attrs["encoding-version"] = "0.1.0"

        # =====================================================
        # 4️⃣ 写 X (CSR, streaming)
        # =====================================================
        print("[EXPORT] 写 X (CSR streaming)")

        gX = f.create_group("X")

        # 🔴 FIX 1：AnnData CSR 必须写在 attrs，而不是 dataset
        gX.attrs["encoding-type"] = "csr_matrix"
        gX.attrs["encoding-version"] = "0.1.0"
        gX.attrs["shape"] = (n_cells, n_genes)

        # ---------- datasets ----------
        d_data = gX.create_dataset(
            "data",
            shape=(nnz,),
            dtype="float32",
            chunks=(min(batch_size, nnz),),
        )

        d_indices = gX.create_dataset(
            "indices",
            shape=(nnz,),
            dtype="int32",
            chunks=(min(batch_size, nnz),),
        )

        gX.create_dataset("indptr", data=indptr, dtype="int64")

        # 🔴 FIX 2：删除错误的 shape dataset（原来这里是 bug）
        # gX.create_dataset("shape", ...)  ← ❌ 不要！

        offset = 0

        for start in tqdm(
            range(0, nnz, batch_size),
            desc="CSR data"
        ):
            end = min(start + batch_size, nnz)

            rows = conn.execute(
                """
                SELECT indices, data
                FROM X_CSR_data
                WHERE id >= ? AND id < ?
                ORDER BY id
                """,
                [int(start), int(end)],
            ).fetchall()

            if not rows:
                continue

            idx, val = zip(*rows)

            d_indices[offset:offset + len(idx)] = idx
            d_data[offset:offset + len(val)] = val

            offset += len(idx)

        assert offset == nnz, f"nnz mismatch: {offset} != {nnz}"

        # =====================================================
        # 5️⃣ 写 obs / var
        # =====================================================
        print("[EXPORT] 写 obs / var")

        _write_dataframe(f, "obs", obs)
        _write_dataframe(f, "var", var)

        # =====================================================
        # 6️⃣ 写 obsm
        # =====================================================
        print("[EXPORT] 写 obsm")

        g_obsm = f.create_group("obsm")

        for (table_name,) in conn.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_name LIKE 'obsm_%'
        """).fetchall():

            key = table_name.replace("obsm_", "")
            df = conn.execute(f"""
                SELECT *
                FROM {table_name}
                ORDER BY cell_index
            """).df()

            df = df.drop(columns=["cell_index"])
            g_obsm.create_dataset(key, data=df.to_numpy())

            print(f"  - obsm[{key}] {df.shape}")

        # =====================================================
        # 7️⃣ 写 varm
        # =====================================================
        print("[EXPORT] 写 varm")

        g_varm = f.create_group("varm")

        for (table_name,) in conn.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_name LIKE 'varm_%'
        """).fetchall():

            key = table_name.replace("varm_", "")
            df = conn.execute(f"""
                SELECT *
                FROM {table_name}
                ORDER BY gene_index
            """).df()

            df = df.drop(columns=["gene_index"])
            g_varm.create_dataset(key, data=df.to_numpy())

            print(f"  - varm[{key}] {df.shape}")

    print("✅ 导出完成")


def _write_dataframe(f, key, df):
    """
    写 AnnData-compatible DataFrame 到 h5ad
    """
    g = f.create_group(key)

    # ---- AnnData dataframe metadata ----
    g.attrs["encoding-type"] = "dataframe"
    g.attrs["encoding-version"] = "0.2.0"

    # ======================================================
    # 1️⃣ index
    # ======================================================
    index_name = df.index.name or "_index"
    index_data = np.array(df.index.astype(str).tolist(), dtype=object)

    g.create_dataset(
        index_name,
        data=index_data,
        dtype=h5py.string_dtype(encoding="utf-8"),
    )

    # ======================================================
    # 2️⃣ columns
    # ======================================================
    colnames = []

    for col in df.columns:
        colnames.append(col)
        series = df[col]

        # pandas categorical → string
        if pd.api.types.is_categorical_dtype(series):
            series = series.astype(str)

        arr = series.to_numpy()

        # ===== 字符串列 =====
        if arr.dtype.kind in {"U", "O"}:
            data = np.array(series.astype(str).tolist(), dtype=object)

            g.create_dataset(
                col,
                data=data,
                dtype=h5py.string_dtype(encoding="utf-8"),
            )

        # ===== 数值列 =====
        else:
            g.create_dataset(col, data=arr)

    # ======================================================
    # 3️⃣ AnnData spec attrs（关键）
    # ======================================================
    g.attrs["column-order"] = np.array(colnames, dtype="S")
    g.attrs["_index"] = index_name


# def _write_dataframe(f, key, df: pd.DataFrame):
#     g = f.create_group(key)
#
#     g.attrs["encoding-type"] = "dataframe"
#     g.attrs["encoding-version"] = "0.2.0"
#
#     # ---------- index ----------
#     idx = df.index.astype(str).tolist()  # ★ 关键：list[str]
#     g.create_dataset(
#         "_index",
#         data=np.array(idx, dtype=object),
#         dtype=h5py.string_dtype(encoding="utf-8"),
#     )
#
#     # ---------- columns ----------
#     for col in df.columns:
#         series = df[col]
#
#         # pandas category → string
#         if pd.api.types.is_categorical_dtype(series):
#             series = series.astype(str)
#
#         arr = series.to_numpy()
#
#         # ===== 字符串列 =====
#         if arr.dtype == object or arr.dtype.kind == "U":
#             data = np.array(series.astype(str).tolist(), dtype=object)
#
#             g.create_dataset(
#                 col,
#                 data=data,
#                 dtype=h5py.string_dtype(encoding="utf-8"),
#             )
#
#         # ===== 数值列 =====
#         else:
#             g.create_dataset(col, data=arr)