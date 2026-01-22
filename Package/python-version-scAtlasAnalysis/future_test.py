import scatlaspy as sap
import numpy as np
import scanpy as sc
from scipy import sparse


# 检验导出
def validate_h5ad_vs_h5ad(
    ref_h5ad: str,
    out_h5ad: str,
    n_checks: int = 10,
    rtol: float = 1e-5,
):
    """
    随机抽查若干 cell，对比 原始 h5ad 与 导出 h5ad 的 X (CSR) 是否一致

    参数
    ----
    ref_h5ad : 原始 h5ad 路径
    out_h5ad : 导出 h5ad 路径
    n_checks : 抽查 cell 数量
    rtol     : 浮点比较容忍度
    """

    print(f"[CHECK] 打开原始 h5ad（backed）: {ref_h5ad}")
    adata_ref = sc.read_h5ad(ref_h5ad, backed="r")

    print(f"[CHECK] 打开导出 h5ad（backed）: {out_h5ad}")
    adata_out = sc.read_h5ad(out_h5ad, backed="r")

    assert adata_ref.n_obs == adata_out.n_obs, "cell 数不一致"
    assert adata_ref.n_vars == adata_out.n_vars, "gene 数不一致"

    n_cells = adata_ref.n_obs

    rng = np.random.default_rng(0)
    cell_ids = rng.choice(n_cells, size=n_checks, replace=False)

    print(f"[CHECK] 随机抽查 {n_checks} 个 cell\n")

    for cell_index in cell_ids:
        print(f"--- Cell {cell_index} ---")

        # ======================================================
        # 1️⃣ 原始 h5ad 的 X
        # ======================================================
        x_ref = adata_ref[cell_index].X
        if not sparse.issparse(x_ref):
            x_ref = sparse.csr_matrix(x_ref)
        else:
            x_ref = x_ref.tocsr()

        ref_indices = x_ref.indices
        ref_data = x_ref.data

        # ======================================================
        # 2️⃣ 导出 h5ad 的 X
        # ======================================================
        x_out = adata_out[cell_index].X
        if not sparse.issparse(x_out):
            x_out = sparse.csr_matrix(x_out)
        else:
            x_out = x_out.tocsr()

        out_indices = x_out.indices
        out_data = x_out.data

        # ======================================================
        # 3️⃣ 对比
        # ======================================================
        assert len(ref_data) == len(out_data), (
            f"nnz mismatch: ref={len(ref_data)}, out={len(out_data)}"
        )

        assert np.array_equal(ref_indices, out_indices), (
            "gene indices mismatch"
        )

        if not np.allclose(ref_data, out_data, rtol=rtol):
            diff = np.abs(ref_data - out_data).max()
            raise AssertionError(f"data mismatch, max diff = {diff}")

        print(f"✔ nnz={len(ref_data)} OK")

    print("\n✅ 原始 h5ad 与 导出 h5ad 随机抽查全部通过")

# 检验导入
def validate_random_cells(
    h5ad_path: str,
    conn,
    n_checks: int = 10,
    rtol: float = 1e-5,
):
    """
    随机抽取若干 cell，对比 h5ad 与 DuckDB 中的 CSR 是否一致

    参数
    ----
    n_checks : 抽查 cell 数量
    rtol     : 浮点比较容忍度
    """

    print(f"[CHECK] 打开 h5ad（backed）: {h5ad_path}")
    adata = sc.read_h5ad(h5ad_path, backed="r")
    n_cells = adata.n_obs

    rng = np.random.default_rng(0)
    cell_ids = rng.choice(n_cells, size=n_checks, replace=False)

    print(f"[CHECK] 随机抽查 {n_checks} 个 cell\n")

    for cell_index in cell_ids:
        print(f"--- Cell {cell_index} ---")

        # ======================================================
        # 1️⃣ h5ad 中的 X（单行）
        # ======================================================
        x = adata[cell_index].X
        if not sparse.issparse(x):
            x = sparse.csr_matrix(x)
        else:
            x = x.tocsr()

        h5ad_indices = x.indices
        h5ad_data = x.data

        # ======================================================
        # 2️⃣ DuckDB 中的 X_CSR_data
        # ======================================================
        duck = conn.execute(
            """
            SELECT indices, data
            FROM X_CSR_data
            WHERE cell_index = ?
            ORDER BY indices
            """,
            [int(cell_index)],
        ).fetchall()

        if len(duck) == 0:
            if len(h5ad_data) == 0:
                print("✔ both empty")
                continue
            else:
                raise AssertionError("DuckDB empty but h5ad not empty")

        duck_indices = np.array([r[0] for r in duck], dtype=np.int64)
        duck_data = np.array([r[1] for r in duck], dtype=np.float32)

        # ======================================================
        # 3️⃣ 对比
        # ======================================================
        assert len(h5ad_data) == len(duck_data), (
            f"nnz mismatch: h5ad={len(h5ad_data)}, duckdb={len(duck_data)}"
        )

        assert np.array_equal(h5ad_indices, duck_indices), (
            "gene indices mismatch"
        )

        if not np.allclose(h5ad_data, duck_data, rtol=rtol):
            diff = np.abs(h5ad_data - duck_data).max()
            raise AssertionError(f"data mismatch, max diff = {diff}")

        print(f"✔ nnz={len(h5ad_data)} OK")

    print("\n✅ 随机抽查全部通过")

# 1. 导入
atlas = sap.Atlas("test_HBCA_in1",path=r"E:\python\scAtlas\Package\python-version-scAtlasAnalysis\test\database")
file_path = r"E:\python\data\HBCA__subsample_20__hvg_2000.h5ad" # 数据集维度: 186,961 × 2,000
# file_path = r"E:\python\data\FullMouseBrain_raw.h5ad" # 82万细胞
# file_path = r"E:\python\data\Immune_ALL_human.h5ad" # 82万细胞
# file_path = r"E:\python\data\1M_neurons_filtered_gene_bc_matrices_h5.h5" # 100万细胞

# sap.io.load_small_to_duckdb(file_path , atlas)    # 小数据导入 ， 支持多种格式
sap.io.load_big_h5ad_to_duckdb(file_path , atlas) # 大数据导入 ， 只支持h5ad

# 校验导入
validate_random_cells(
        h5ad_path=file_path,
        conn=atlas.connection,
        n_checks=20,
    )

# 2. filter_cells & filter_genes
sap.pp.filter_cells(atlas,min_counts = 200)
sap.pp.filter_genes(atlas,min_counts = 20000)

# 校验 结果
conn = atlas.connection
print("=== obs schema ===")
for r in conn.execute("PRAGMA table_info('obs')").fetchall():
    print(r)

print("\n=== obs head(3) ===")
for r in conn.execute("SELECT * FROM obs ORDER BY id LIMIT 3").fetchall():
    print(r)

print("\n=== var schema ===")
for r in conn.execute("PRAGMA table_info('var')").fetchall():
    print(r)

print("\n=== var head(3) ===")
for r in conn.execute("SELECT * FROM var ORDER BY id LIMIT 3").fetchall():
    print(r)

# 3. QC
sap.pp.calculate_qc_metrics(atlas)

# 4. calculate_cell_total_counts
sap.pp.calculate_cell_total_counts(atlas)
sap.pp.calculate_gene_total_counts(atlas)


# 5. 归一化
sap.pp.normalize_total_new(atlas)            # 法1 ：全部
# sap.pp.normalize_total_new_chunked(atlas)  # 法2 ：分块
# sap.pp.normalize_total_scale_factor(atlas) # 法3 ：分块， 在 obs表上记录 scale_factor ， 等到使用的时候在计算

print("=== X_CSR_data schema ===")
for row in conn.execute("PRAGMA table_info('X_CSR_data')").fetchall():
    print(row)

print("\n=== X_CSR_data head(3) ===")
for row in conn.execute("""
    SELECT * FROM X_CSR_data ORDER BY id LIMIT 3
""").fetchall():
    print(row)

sap.pp.log1p(atlas) # 法1：不分块
sap.pp.log1p_chunked(atlas) # 法2：分块

sap.pp.exp1_chunked(atlas) # log1p的逆运算

sap.pp.normalize_and_log1p(atlas) # normalize 法3  + log1p 法 2
sap.pp.log1p_chunked(atlas,add_field = "normalize_log1p ",select_data = "data_normalize") # normalize 法3  + log1p 法 2


# 6. 特征选择
sap.pp.highly_variable_genes(atlas) #

# 7. 标准化
sap.pp.scale(atlas)                 # 法1：不分块
sap.pp.scale_gene_chunked(atlas,gene_chunk_size = 512 ,add_field = "X_scale_2" )    # 法2：分块 ；gene_chunk_size 的设置与cell数量 相关
sap.pp.scale_gene_chunked_1(atlas,gene_chunk_size = 512 ,add_field = "X_scale_3")  # 法3： 对法2 的优化；直接在 chunk 内 UPDATE，彻底删掉临时表 + merge

# 8. sqrt
sap.pp.sqrt(atlas)           # 法1：不分块
sap.pp.sqrt_chunked(atlas)   # 法2：分块

# 9.  降维 pca
# 10. 聚类 leiden

# 11. 导出
file_path = r"E:\python\scAtlas\Package\python-version-scAtlasAnalysis\test\data\out\test_HBCA_outs.h5ad" # 导出文件
# file_path = r"E:\python\data\HBCA__subsample_20__hvg_2000.h5ad" # 原始文件
sap.io.export_duckdb_to_h5ad(atlas,file_path)

# 校验导出
validate_h5ad_vs_h5ad(
    ref_h5ad=r"E:\python\data\HBCA__subsample_20__hvg_2000.h5ad", # 原始文件
    out_h5ad=r"E:\python\scAtlas\Package\python-version-scAtlasAnalysis\test\data\out\test_HBCA_outs.h5ad", # 导出文件
    n_checks=20,
)






