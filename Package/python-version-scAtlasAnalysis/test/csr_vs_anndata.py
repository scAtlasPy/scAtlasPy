import scanpy as sc
import anndata as ad
from scipy import sparse
import numpy as np
from scipy.sparse import csr_matrix
import pandas as pd
from datetime import datetime
import scatlaspy as sap
import logging # 管理各种类型的日志
from scatlaspy.io.input import read_smart

verbose=True # 是否展示中间过程信息

# 创建一个atlas对象，该对象管理与duckdb的各种交互
atlas=sap.Atlas("test_204800",path=r"E:\python\scAtlasAnalysis-demo\test\database")

# # 方案1：手动执行完整优化（推荐用于生产环境）
# print("=== 执行完整数据库优化 ===")
# atlas.comprehensive_database_optimization()

# 大数据集 80万
adata=sc.read_h5ad("E:\python\data\FullMouseBrain_raw.h5ad") # E:\python\data\FullMouseBrain_raw.h5ad 大数据集地址
adata=adata[:204800] # 降采样

# sap.io.load_AnnData(adata,atlas) # 载入数据

# 方法2：使用MAX函数获取最大id对应的行
query_indptr_max = """
SELECT * FROM X_CSR_indptr 
WHERE id = (SELECT MAX(id) FROM X_CSR_indptr)
"""
last_indptr_max = atlas.connection.execute(query_indptr_max).df()
print("X_CSR_indptr 表最后一个值(使用MAX):")
print(last_indptr_max)

# 方法2：使用MAX函数获取最大id对应的行
query_data_max = """
SELECT * FROM X_CSR_data 
WHERE id = (SELECT MAX(id) FROM X_CSR_data)
"""
last_data_max = atlas.connection.execute(query_data_max).df()
print("X_CSR_data 表最后一个值(使用MAX):")
print(last_data_max)




# 测试minibatch读取

mini_start = datetime.now()

nzz = 0

for adata_minibatch in atlas.query_minibatch( batch_size=4096):
    print(f"生成的AnnData对象信息:")
    print(f"  - 细胞数: {adata_minibatch.n_obs}")
    print(f"  - 基因数: {adata_minibatch.n_vars}")
    print(f"  - 矩阵形状: {adata_minibatch.X.shape}")
    print(f"  - X 矩阵格式 : {type(adata_minibatch.X)}")

    # 检查非零元素占比，确认稀疏性
    nnz_ratio = adata_minibatch.X.nnz / (adata_minibatch.n_obs * adata_minibatch.n_vars)
    print(f"非零元素占比: {nnz_ratio:.2%}")  # 单细胞数据通常在 5% 以下

    nzz += adata_minibatch.X.nnz

    # print(f"  - var_names: {adata_minibatch.var_names}")
    # print(f"  - obs_names: {adata_minibatch.obs_names}")

# 1 minibatch 计算
nnz_ratio1 = nzz / (819200 * 17745)
print(f"minibatch 计算 非零元素个数：{nzz}  ；   占比: {nnz_ratio1:.2%}")  # 单细胞数据通常在 5% 以下

mini_end = datetime.now()

mini_time = (mini_end - mini_start).total_seconds()
print(f"***** minibatch 读取 并计算 耗时： {mini_time:.2f}秒")



anndata_start = datetime.now()
# 2 anndata 直接计算
nnz_ratio2 = adata.X.nnz / (adata.n_obs * adata.n_vars)
print(f"anndata 直接计算 非零元素个数：{adata.X.nnz}  ；   占比: {nnz_ratio2:.2%}")  # 单细胞数据通常在 5% 以下
anndata_end = datetime.now()

anndata_time = (anndata_end -anndata_start).total_seconds()
print(f"***** anndata 直接计算 耗时： {anndata_time:.2f}秒")


sap.pp.filter_cells_minibatch_yield(atlas,2000,10,20000, 17000,
                                    batch_size=4096, add_key="filter_cells_c5")

sap.pp.filter_cells_sql(atlas,2000,10,20000, 17000,
                                    batch_size=4096, add_key="filter_cells_c5")


sap.pp.filter_cells_sql(atlas,min_counts=2000,
                                    batch_size=4096, add_key="filter_cells_c5")

# sc.pp.filter_cells(adata, min_counts=2000,min_genes=10,max_counts=20000, max_genes=17000)

adata1 = adata

# 先获取原始细胞数
original_cells = adata.n_obs
print(f"原始细胞数: {original_cells}")

# # 方法1：分步过滤（推荐）
# # 第一步：按最小基因数过滤
# sc.pp.filter_cells(adata, min_genes=10)
# print(f"过滤 min_genes=10 后细胞数: {adata.n_obs}")

# 第二步：按最小counts数过滤
sc.pp.filter_cells(adata, min_counts=2000)
print(f"过滤 min_counts=2000 后细胞数: {adata.n_obs}")
#
# # 第三步：按最大基因数过滤
# sc.pp.filter_cells(adata, max_genes=17000)
# print(f"过滤 max_genes=17000 后细胞数: {adata.n_obs}")
#
# # 第四步：按最大counts数过滤
# sc.pp.filter_cells(adata, max_counts=20000)
# print(f"过滤 max_counts=20000 后细胞数: {adata.n_obs}")

# 打印总结信息
final_cells = adata.n_obs
filtered_cells = original_cells - final_cells
print(f"\n===== 过滤总结 =====")
print(f"原始细胞数: {original_cells}")
print(f"过滤细胞数: {filtered_cells}")
print(f"保留细胞数: {final_cells}")
print(f"过滤比例: {filtered_cells/original_cells:.2%}")

# # 方法1：从密集矩阵转换
# dense_matrix = np.array([
#     [0, 0, 3],
#     [0, 1, 0],
#     [4, 0, 0],
#     [0, 0, 5],
#     [0, 6, 0]
# ])
# csr = csr_matrix(dense_matrix)
# print(csr)
# # 输出：
# #   (0, 2)    3
# #   (1, 1)    1
# #   (2, 0)    4
# #   (3, 2)    5
# #   (4, 1)    6
#
# # 手动构建 CSR 结构
# data = np.array([3, 1, 4, 5, 6])
# indices = np.array([2, 1, 0, 2, 1])
# indptr = np.array([0, 1, 2, 3, 4, 5])  # 5行矩阵
#
# csr_manual = csr_matrix((data, indices, indptr), shape=(5, 3))
# print(csr_manual.toarray())  # 转换回密集矩阵验证
#
# # 获取非零元素数量
# print(csr.nnz)  # 输出: 5
#
# print(csr.data)
# print(csr.indices)
# print(csr.indptr)
#
# CSR_df = csr.data
# print(CSR_df)






# 假设你已经有一个 AnnData 对象
# adata = sc.datasets.pbmc3k()  # 示例数据
# adata = sc.datasets.pbmc68k_reduced() # 提取数据
#
# # 方法1：直接获取 X 矩阵（如果已经是稀疏格式）
# if sparse.issparse(adata.X):
#     csr_matrix = adata.X.tocsr()  # 转换为 CSR 格式
# else:
#     csr_matrix = sparse.csr_matrix(adata.X)
#
# print(f"CSR 矩阵形状: {csr_matrix.shape}")
# print(f"非零元素数量: {csr_matrix.nnz}")
# print(csr_matrix)


# def adata_to_csr_chunked(adata, chunk_size=1000):
#     """分块处理大型 AnnData 对象"""
#     n_cells = adata.shape[0]
#     csr_matrices = []
#
#     for start in range(0, n_cells, chunk_size):
#         end = min(start + chunk_size, n_cells)
#         chunk = adata[start:end].X.tocsr()
#         csr_matrices.append(chunk)
#
#     # 合并所有块
#     final_csr = sparse.vstack(csr_matrices, format='csr')
#     return final_csr
#
#
# # 使用分块处理
# large_csr = adata_to_csr_chunked(adata)