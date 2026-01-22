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