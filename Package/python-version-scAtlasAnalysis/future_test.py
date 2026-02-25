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
atlas = sap.Atlas("test_819200",path=r"E:\python\scAtlas\Package\python-version-scAtlasAnalysis\test\database")
# file_path = r"E:\python\data\HBCA__subsample_20__hvg_2000.h5ad" # 数据集维度: 186,961 × 2,000
file_path = r"E:\python\data\FullMouseBrain_raw.h5ad" # 82万细胞
# file_path = r"E:\python\data\Immune_ALL_human.h5ad" # 82万细胞
# file_path = r"E:\python\data\1M_neurons_filtered_gene_bc_matrices_h5.h5" # 100万细胞

adata = sc.read_h5ad(file_path)
adata100 = adata[:100]
adata200 = adata[:200]
adata1000 = adata[:1000]
sap.io.load_AnnData(adata100, atlas)

sap.io.load_small_to_duckdb(file_path , atlas)    # 小数据导入 ， 支持多种格式
# sap.io.load_big_h5ad_to_duckdb(file_path , atlas) # 大数据导入 ， 只支持h5ad

# 查看数据规模
gene_num = atlas.connection.execute("SELECT COUNT(*) FROM var").fetchone()[0]
cell_num = atlas.connection.execute("SELECT COUNT(*) FROM obs").fetchone()[0]
print(f"[数据规模]: {cell_num} x {gene_num}" )

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
# sap.io.export_duckdb_to_h5ad(atlas,file_path)
sap.io.export_duckdb_to_h5ad_streaming(atlas,file_path) # 小内存


# 校验导出
validate_h5ad_vs_h5ad(
    ref_h5ad=r"E:\python\data\HBCA__subsample_20__hvg_2000.h5ad", # 原始文件
    out_h5ad=r"E:\python\scAtlas\Package\python-version-scAtlasAnalysis\test\data\out\test_HBCA_outs.h5ad", # 导出文件
    n_checks=20,
)

# 校验导出

import scanpy as sc
import numpy as np
import hashlib
import random
import pandas as pd


# =========================================================
#  内部工具
# =========================================================

def _hash_array(arr) -> str:
    arr = np.asarray(arr)
    h = hashlib.sha256()
    h.update(arr.tobytes())
    return h.hexdigest()


def _normalize_series(values):
    """
    🔧 关键修复：
    统一 DuckDB ↔ AnnData 的列语义表示

    - NULL / None / NaN → np.nan
    - category / object → string
    - 消除 "nan" 字符串假象
    """
    s = pd.Series(values)

    # 统一缺失值
    s = s.where(pd.notnull(s), np.nan)

    # object / category → string（AnnData 语义）
    if s.dtype.name in {"object", "category"}:
        s = s.astype(str)
        s = s.replace("nan", np.nan)

    return s.to_numpy()


def _get_csr_components(X):
    """
    🔧 FIX: 统一处理 backed / non-backed CSR 访问

    backed:
        X is _CSRDataset → 使用私有属性
    non-backed:
        X is scipy.sparse.csr_matrix
    """
    if hasattr(X, "_data"):
        return X._data, X._indices, X._indptr
    else:
        return X.data, X.indices, X.indptr

def _normalize_missing(s: pd.Series) -> pd.Series:
    """
    将各种 None / 'None' / 'nan' 统一为 np.nan
    """
    return (
        s.replace(
            {
                None: np.nan,
                "None": np.nan,
                "nan": np.nan,
                "NaN": np.nan,
                "": np.nan,
            }
        )
    )


# =========================================================
#  主入口
# =========================================================

def validate_h5ad(
    atlas,
    h5ad_path: str,
    *,
    check_obs: bool = True,
    check_var: bool = True,
    check_obsm: bool = False,
    check_varm: bool = False,
    check_csr_indices: bool = True,
    check_csr_data_sample: int = 100_000,  # 0 = skip
):
    """
    校验 DuckDB → h5ad 的全量一致性（backed-safe）

    语义一致性校验（而非 byte-level）
    """

    print(f"[VALIDATE] 打开 h5ad (backed) → {h5ad_path}")
    adata = sc.read_h5ad(h5ad_path, backed="r")
    conn = atlas.connection

    # 🔧 backed-safe CSR
    X = adata.X
    X_data, X_indices, X_indptr = _get_csr_components(X)

    # =====================================================
    # 1️⃣ 基础 meta
    # =====================================================
    print("[VALIDATE] 基础 meta")

    n_obs_db = conn.execute("SELECT COUNT(*) FROM obs").fetchone()[0]
    n_var_db = conn.execute("SELECT COUNT(*) FROM var").fetchone()[0]

    assert adata.n_obs == n_obs_db, "n_obs mismatch"
    assert adata.n_vars == n_var_db, "n_vars mismatch"

    nnz_db = conn.execute(
        "SELECT MAX(indptr) FROM X_CSR_indptr"
    ).fetchone()[0]

    assert X_data.shape[0] == nnz_db, "nnz mismatch"

    print("  ✓ shape / nnz OK")

    # =====================================================
    # 2️⃣ obs / var index
    # =====================================================
    if check_obs:
        print("[VALIDATE] obs index")

        obs_db = [
            x[0] for x in conn.execute(
                "SELECT cell_id FROM obs ORDER BY id"
            ).fetchall()
        ]

        assert list(adata.obs_names) == obs_db, "obs index mismatch"
        print("  ✓ obs index OK")

    if check_var:
        print("[VALIDATE] var index")

        var_db = [
            x[0] for x in conn.execute(
                "SELECT gene_id FROM var ORDER BY id"
            ).fetchall()
        ]

        assert list(adata.var_names) == var_db, "var index mismatch"
        print("  ✓ var index OK")

    # =====================================================
    # 3️⃣ obs / var columns（最终工程版）
    # =====================================================
    if check_obs:
        print("[VALIDATE] obs columns")

        for col in adata.obs.columns:
            db_vals = [
                x[0] for x in conn.execute(
                    f"SELECT {col} FROM obs ORDER BY id"
                ).fetchall()
            ]

            db_s = pd.Series(db_vals)
            h5_s = pd.Series(adata.obs[col].values)

            # ---- 统一缺失值语义（最终修复）----
            db_s = _normalize_missing(db_s)
            h5_s = _normalize_missing(h5_s)

            # object / category → string（但保留 NaN）
            if db_s.dtype.name in {"object", "category"}:
                db_s = db_s.astype("string")
                h5_s = h5_s.astype("string")

            # ---- 逐元素语义比较 ----
            mismatch = ~(
                    (db_s.isna() & h5_s.isna()) |
                    (db_s == h5_s)
            )

            if mismatch.any():
                i = mismatch.idxmax()
                raise AssertionError(
                    f"obs.{col} mismatch at row {i}: "
                    f"db={db_s.iloc[i]!r}, h5={h5_s.iloc[i]!r}"
                )

        print("  ✓ obs columns OK")

    if check_var:
        print("[VALIDATE] var columns")

        for col in adata.var.columns:
            db_vals = [
                x[0] for x in conn.execute(
                    f"SELECT {col} FROM var ORDER BY id"
                ).fetchall()
            ]

            db_s = pd.Series(db_vals)
            h5_s = pd.Series(adata.var[col].values)

            db_s = db_s.where(pd.notnull(db_s), np.nan)
            h5_s = h5_s.where(pd.notnull(h5_s), np.nan)

            if db_s.dtype.name in {"object", "category"}:
                db_s = db_s.astype(str).replace("nan", np.nan)
                h5_s = h5_s.astype(str).replace("nan", np.nan)

            mismatch = ~(
                    (db_s.isna() & h5_s.isna()) |
                    (db_s == h5_s)
            )

            if mismatch.any():
                i = mismatch.idxmax()
                raise AssertionError(
                    f"var.{col} mismatch at row {i}: "
                    f"db={db_s.iloc[i]!r}, h5={h5_s.iloc[i]!r}"
                )

        print("  ✓ var columns OK")

    # =====================================================
    # 4️⃣ CSR indptr
    # =====================================================
    print("[VALIDATE] CSR indptr")

    indptr_db = np.array(
        [0] + [
            x[0] for x in conn.execute(
                "SELECT indptr FROM X_CSR_indptr ORDER BY id"
            ).fetchall()
        ],
        dtype=np.int64,
    )

    assert np.array_equal(indptr_db, X_indptr), "CSR indptr mismatch"
    print("  ✓ indptr OK")

    # =====================================================
    # 5️⃣ CSR indices
    # =====================================================
    if check_csr_indices:
        print("[VALIDATE] CSR indices")

        indices_db = np.array(
            [x[0] for x in conn.execute(
                "SELECT indices FROM X_CSR_data ORDER BY id"
            ).fetchall()],
            dtype=np.int32,
        )

        assert _hash_array(indices_db) == _hash_array(X_indices), \
            "CSR indices mismatch"

        print("  ✓ indices OK")

    # =====================================================
    # 6️⃣ CSR data（抽样）
    # =====================================================
    if check_csr_data_sample and check_csr_data_sample > 0:
        print(f"[VALIDATE] CSR data 抽样 ({check_csr_data_sample:,})")

        nnz = X_data.shape[0]
        samples = random.sample(
            range(nnz),
            min(check_csr_data_sample, nnz),
        )

        for i in samples:
            val_db = conn.execute(
                "SELECT data FROM X_CSR_data WHERE id = ?",
                [int(i)],
            ).fetchone()[0]

            if not np.isclose(val_db, X_data[i]):
                raise AssertionError(f"X.data mismatch at {i}")

        print("  ✓ data sample OK")

    # =====================================================
    # 7️⃣ obsm / varm（shape only）
    # =====================================================
    if check_obsm:
        print("[VALIDATE] obsm shapes")
        for key in adata.obsm.keys():
            assert adata.obsm[key].shape[0] == adata.n_obs
        print("  ✓ obsm OK")

    if check_varm:
        print("[VALIDATE] varm shapes")
        for key in adata.varm.keys():
            assert adata.varm[key].shape[0] == adata.n_vars
        print("  ✓ varm OK")

    print("🎉 validate_h5ad：DuckDB ↔ h5ad 语义一致性校验通过")


# todo ======== minibatch 测试 ============
from datetime import datetime
import scatlaspy as sap
import logging
logging.getLogger("Atlas").setLevel(logging.WARNING)
atlas=sap.Atlas("test_204800",path=r"E:\python\scAtlas\Package\python-version-scAtlasAnalysis\test\database")

count = 0
batch_size = 2048
start_time = datetime.now()
for adata_minibatch in atlas.minibatch_scan_order_mthread(batch_size=batch_size,drop_last=False):
    # print(f"第{count}批: 生成的AnnData对象信息: ")
    # print(f"  - 细胞数: {adata_minibatch.n_obs}")
    # print(f"  - 基因数: {adata_minibatch.n_vars}")
    count = count + 1
    # print(f"  - 矩阵形状: {adata_minibatch.X.shape}")
    # print(f"  - var_names: {adata_minibatch.var_names}")
    # print(f"  - obs_names: {adata_minibatch.obs_names}")
end_time = datetime.now()
minibatch_time = (end_time - start_time).total_seconds()
print(f"#### 总批次数量 : {count} ")
print(f"#### 批次大小 : {batch_size} ")
print(f"#### minibatch_time 总时间 : {minibatch_time:.2f} 秒")
print(f"#### 1个 minibatch 耗时 : { (minibatch_time / count) :.2f} 秒")
print(f"#### 每秒 minibatch 数量 : { (count / minibatch_time) :.2f} 个")


# todo ======== minibatch 测试  随机有放回 ============
from datetime import datetime
import scatlaspy as sap
import logging

logging.getLogger("Atlas").setLevel(logging.WARNING)

atlas = sap.Atlas(
    "test_204800",
    path=r"E:\python\scAtlas\Package\python-version-scAtlasAnalysis\test\database"
# r"E:\python\scAtlas\Package\python-version-scAtlasAnalysis\test\database\test_204800.sasql"
)

max_batches = 100          # 👈 控制这里
count = 0
batch_size = 2048

start_time = datetime.now()

for adata_minibatch in atlas.minibatch_scan_random_replace_coo_join_mp(
    batch_size=batch_size,
    drop_last=False
):
    if count >= max_batches:   # 👈 核心 break
        break

    print(f"第{count}批: 生成的AnnData对象信息:")
    print(f"  - 细胞数: {adata_minibatch.n_obs}")
    print(f"  - 基因数: {adata_minibatch.n_vars}")

    count += 1

end_time = datetime.now()

minibatch_time = (end_time - start_time).total_seconds()

print(f"#### 总批次数量 : {count}")
print(f"#### 批次大小 : {batch_size}")
print(f"#### minibatch_time 总时间 : {minibatch_time:.2f} 秒")
print(f"#### 1个 minibatch 耗时 : {(minibatch_time / count):.2f} 秒")
print(f"#### 每秒 minibatch 数量 : {(count / minibatch_time):.2f} 个")








