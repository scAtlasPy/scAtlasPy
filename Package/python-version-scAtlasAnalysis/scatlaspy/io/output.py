from ..data import Atlas

import pandas as pd
import anndata
import scipy.sparse as sparse

################################
# save data into a file

def save_h5ad(file:str, atlas:Atlas):
    # save data as h5ad format file.
    atlas.save_h5ad()
    pass

def save_loom(file:str, atlas:Atlas):
    pass

def save_csv(file:str, atlas:Atlas):
    pass

#######################################################
# read data from duckdb to memory
def get_adata(atlas:Atlas):
    # TODO
    return

def get_df(df:pd.DataFrame, atlas:Atlas):
    # TODO
    # return dict(X=X, obs=obs, var=var, ......)
    return dict(X=X,obs=obs,var=var)

def create_anndata_from_tables(con, output_path="output_data.h5ad"):
    """
    从三张表创建AnnData对象并保存为h5ad文件

    参数:
    con: duckdb数据库连接
    output_path: 输出h5ad文件路径
    """

    # 1. 读取cells表 (obs数据)
    print("读取cells表...")
    cells_df = con.execute("SELECT * FROM cells").fetchdf()
    cells_df = cells_df.set_index('cell_id')  # 设置cell_id为索引
    print(f"读取到 {len(cells_df)} 个细胞")

    # 2. 读取genes表 (var数据)
    print("读取genes表...")
    genes_df = con.execute("SELECT * FROM genes").fetchdf()
    genes_df = genes_df.set_index('gene_id')  # 设置gene_id为索引
    print(f"读取到 {len(genes_df)} 个基因")

    # 3. 读取expression表 (X矩阵)
    print("读取expression表...")
    expression_df = con.execute("SELECT * FROM expression").fetchdf()
    print(f"读取到 {len(expression_df)} 个表达值记录")

    # 4. 创建稀疏矩阵
    print("创建稀疏矩阵...")

    # 创建细胞和基因的映射字典
    cell_to_idx = {cell_id: idx for idx, cell_id in enumerate(cells_df.index)}
    gene_to_idx = {gene_id: idx for idx, gene_id in enumerate(genes_df.index)}

    # 准备稀疏矩阵的数据
    row_indices = [cell_to_idx[cell_id] for cell_id in expression_df['cell_id']]
    col_indices = [gene_to_idx[gene_id] for gene_id in expression_df['gene_id']]
    data_values = expression_df['expression'].values

    # 创建CSR格式的稀疏矩阵
    X_sparse = sparse.csr_matrix(
        (data_values, (row_indices, col_indices)),
        shape=(len(cells_df), len(genes_df))
    )

    # 5. 创建AnnData对象
    print("创建AnnData对象...")
    adata = anndata.AnnData(
        X=X_sparse,  # 表达矩阵
        obs=cells_df,  # 细胞元数据
        var=genes_df  # 基因元数据
    )

    # 6. 保存为h5ad文件
    print(f"保存到 {output_path}...")
    adata.write_h5ad(output_path)

    print("完成！")
    print(f"生成的AnnData对象信息:")
    print(f"  - 细胞数: {adata.n_obs}")
    print(f"  - 基因数: {adata.n_vars}")
    print(f"  - 矩阵形状: {adata.X.shape}")
    print(f"  - 矩阵密度: {adata.X.nnz / (adata.n_obs * adata.n_vars):.4f}")

    return adata

