import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm

''' 数据导出: 把数据直接变成文件'''
# 819200 耗时 11:09
def export_duckdb_to_h5ad(
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

    # 1️⃣ 读取 obs / var
    print("[EXPORT] 读取 obs / var")

    obs = conn.execute("SELECT * FROM obs ORDER BY atlas_cell_id").df()
    var = conn.execute("SELECT * FROM var ORDER BY atlas_gene_id").df()

    obs = obs.set_index("atlas_cell_id")
    var = var.set_index("atlas_gene_id")

    n_cells = obs.shape[0]
    n_genes = var.shape[0]

    print(f"[EXPORT] cells={n_cells:,}, genes={n_genes:,}")

    # 2️⃣ 读取 CSR indptr
    print("[EXPORT] 读取 CSR indptr")

    indptr_df = conn.execute("""
        SELECT indptr
        FROM X_CSRO_indptr
        ORDER BY atlas_cell_id
    """).df()

    indptr = np.empty(n_cells + 1, dtype=np.int64)
    indptr[0] = 0
    indptr[1:] = indptr_df["indptr"].to_numpy(dtype=np.int64)

    nnz = int(indptr[-1])
    print(f"[EXPORT] nnz={nnz:,}")

    # 3️⃣ 创建 h5ad 文件
    print(f"[EXPORT] 创建 h5ad → {out_h5ad_path}")

    with h5py.File(out_h5ad_path, "w") as f:

        # ---------- 根节点属性 ----------
        f.attrs["encoding-type"] = "anndata"
        f.attrs["encoding-version"] = "0.1.0"

        # 4️⃣ 写 X (CSR, streaming)
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
            dtype="uint16",
            chunks=(min(batch_size, nnz),),
        )

        gX.create_dataset("indptr", data=indptr, dtype="int64")

        offset = 0

        for start in tqdm(
            range(0, nnz, batch_size),
            desc="CSR data"
        ):
            end = min(start + batch_size, nnz)

            rows = conn.execute(
                """
                SELECT atlas_gene_id, data
                FROM X_CSRO_data
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

        # 5️⃣ 写 obs / var
        print("[EXPORT] 写 obs / var")

        _write_dataframe(f, "obs", obs)
        _write_dataframe(f, "var", var)

        # 6️⃣ 写 obsm
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
                ORDER BY atlas_cell_id
            """).df()

            df = df.drop(columns=["atlas_cell_id"])
            g_obsm.create_dataset(key, data=df.to_numpy())

            print(f"  - obsm[{key}] {df.shape}")

        # 7️⃣ 写 varm
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
                ORDER BY atlas_gene_id
            """).df()

            df = df.drop(columns=["atlas_gene_id"])
            g_varm.create_dataset(key, data=df.to_numpy())

            print(f"  - varm[{key}] {df.shape}")

    print("✅ 导出完成")


''' 写 AnnData-compatible DataFrame 到 h5ad '''
def _write_dataframe(f, key, df):
    """
    写 AnnData-compatible DataFrame 到 h5ad
    """
    g = f.create_group(key)

    # ---- AnnData dataframe metadata ----
    g.attrs["encoding-type"] = "dataframe"
    g.attrs["encoding-version"] = "0.2.0"

    # 1️⃣ index
    index_name = df.index.name or "_index"
    index_data = np.array(df.index.astype(str).tolist(), dtype=object)

    g.create_dataset(
        index_name,
        data=index_data,
        dtype=h5py.string_dtype(encoding="utf-8"),
    )

    # 2️⃣ columns
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

    # 3️⃣ AnnData spec attrs（关键）
    g.attrs["column-order"] = np.array(colnames, dtype="S")
    g.attrs["_index"] = index_name
