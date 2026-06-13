import duckdb
import anndata
import numpy as np
import pandas as pd
from datetime import datetime
import scipy.sparse as sparse


# 获取当前时间并格式化为年月日_时分秒格式
current_datetime = datetime.now().strftime('%Y%m%d_%H%M%S')
print(f"当前时间：{current_datetime}")


# 1.加载ANNDATA数据
print("正在加载数据...")
adata = anndata.read_h5ad('E:\python\scsql\data\FullMouseBrain_raw.h5ad')

# 降采样
adata=adata[:200]

# 2.创建DuckDB连接

con = duckdb.connect('single_cell_test_'+current_datetime+'.db')

# 2. 准备数据表
print("准备数据表...")
# 细胞表数据（对应obs）
cells_df = adata.obs.reset_index().rename(columns={'index': 'cell_id'})
print("cells表数据准备完成，行数:", len(cells_df))

# 基因表数据（对应var）
genes_df = adata.var.reset_index().rename(columns={'index': 'gene_id'})
print("genes表数据准备完成，行数:", len(genes_df))

# 表达矩阵数据（对应X）
print("expression表数据准备中...")

X=adata.X
if not isinstance(X,np.ndarray): # 转换成密集矩阵
    X=X.toarray()

# 创建一个宽格式的 DataFrame
expression_df_wide = pd.DataFrame(X, index=adata.obs_names, columns=adata.var_names) # columns 参数用于设置 DataFrame 的列名

# 将宽格式的 DataFrame 转换为长格式
expression_df_long = expression_df_wide.stack().reset_index()
expression_df_long.columns = ['cell_id', 'gene_id', 'expression']

# 打印长格式的 DataFrame
print(expression_df_long)
print("expression表数据准备完成，行数:", len(expression_df_long))

# 创建数据表
print("正在创建数据表...")
# 3.创建cells表，对应 观察值数据表  obs
create_cells_table_sql = '''
CREATE TABLE cells (
    cell_id VARCHAR PRIMARY KEY,
    batch VARCHAR,
    celltype VARCHAR
)
'''
con.execute(create_cells_table_sql)

# 创建genes表，对应 特征和高可变基因数据表 var
create_genes_table_sql='''
CREATE TABLE genes(
    gene_id VARCHAR PRIMARY KEY,
    name    VARCHAR
)
'''
con.execute(create_genes_table_sql)

# 创建expression表 , 对应 X 矩阵中的值
create_expression_table_sql='''
CREATE TABLE expression(
    cell_id    VARCHAR,
    gene_id    VARCHAR,
    expression FLOAT,
    PRIMARY KEY (cell_id, gene_id),
    FOREIGN KEY (cell_id) REFERENCES cells (cell_id),
    FOREIGN KEY (gene_id) REFERENCES genes (gene_id)
)
'''
con.execute(create_expression_table_sql)

con.sql('show tables').show()

# 插入数据
print("正在写入数据...")

# cells_df.to_sql('cells', con, index=False, if_exists='append')
con.sql("INSERT INTO cells SELECT * FROM cells_df")  # 或者使用 INSERT INTO 插入数据

# genes_df.to_sql('genes', con, index=False, if_exists='append')
con.sql("INSERT INTO genes SELECT * FROM genes_df")  # 或者使用 INSERT INTO 插入数据

# expression_df.to_sql('expression', con, index=False, if_exists='append')
con.sql("INSERT INTO expression SELECT * FROM expression_df_long")  # 或者使用 INSERT INTO 插入数据

# index=True,将DataFrame的索引作为数据库表中的一列写入。默认写入时，索引列会被命名为index（可通过index_label自定义列名）。
# 例如，若DataFrame索引为cell_id，写入后表中会包含名为cell_id的列，值与索引一致。
# if_exists='append'append：将新数据追加到现有表末尾，不删除原有数据。


# 查询表内容以验证数据是否插入成功
print("验证数据是否正确插入...")

result = con.sql("SELECT * FROM cells_df").fetchall()  # 使用 fetchall() 获取所有结果
print(f"结果长度(行数): {len(result)}")
# 以 f 开头的字符串（如 f"..."）允许在字符串中直接嵌入表达式（用 {} 包裹）。
# 运行时，{} 内的表达式会被计算并替换为实际值

# 删除没有表达数据的细胞
def filter_cell_counts():
    # 1.新建一张表
    print("删除没有表达数据的细胞...")
    create_filtered_cells_table_sql = '''
        CREATE TABLE filtered_cells_data AS
            SELECT c.*
            FROM cells c
            WHERE cell_id IN (SELECT DISTINCT cell_id FROM expression)
          '''
    con.execute(create_filtered_cells_table_sql)

    # 查询 filtered_cells_data表以验证结果
    result = con.sql("SELECT * FROM filtered_cells_data").fetchall()
    print(f"删除空细胞后的 filtered_cells_data 的行数 ...: {len(result)}")

    # # 2.在原表上删除
    # print("删除没有表达数据的细胞...")
    # con.sql("""
    #         DELETE
    #         FROM cells
    #         WHERE cell_id NOT IN (SELECT DISTINCT cell_id FROM expression)
    #         """)  # DISTINCT 是一个关键字，用于去除查询结果中的重复值。
    #
    # # 查询 cells 表以验证结果
    # result = con.sql("SELECT * FROM cells").fetchall()
    # print(f"删除空细胞后的cells 的行数 ...: {len(result)}")

# 删除没有表达数据的细胞
# filter_cell_counts()

# 删除没有表达数据的基因
def filter_gene_counts():
    # 1.新建一张表
    print("删除没有表达数据的细胞...")
    create_filtered_genes_table_sql = '''
        CREATE TABLE filtered_genes_data AS
            SELECT g.*
            FROM genes g
            WHERE gene_id IN (
                SELECT DISTINCT gene_id 
                FROM expression)
          '''
    con.execute(create_filtered_genes_table_sql)

    # 查询 filtered_genes_data  表以验证结果
    result = con.sql("SELECT * FROM filtered_genes_data").fetchall()
    print(f"删除空基因后的 filtered_genes_data 的行数 ...: {len(result)}")

# 删除没有表达数据的基因
# filter_gene_counts()


# 将数据库文件导出为anndata格式
def sql_to_anndata():
    # 1. 读取三张表的数据（忽略警告，或改用 SQLAlchemy）
    cells_df = pd.read_sql("SELECT * FROM cells", con)
    genes_df = pd.read_sql("SELECT * FROM genes", con)
    expression_df = pd.read_sql("SELECT * FROM expression", con)

    # 2. 构建稀疏表达矩阵 (X)
    # 将 cell_id 和 gene_id 转换为分类变量，并提取类别顺序
    cell_categories = cells_df["cell_id"].values
    gene_categories = genes_df["gene_id"].values

    # 正确构造分类变量
    expression_df["cell_id"] = pd.Categorical(
        expression_df["cell_id"], categories=cell_categories
    )
    expression_df["gene_id"] = pd.Categorical(
        expression_df["gene_id"], categories=gene_categories
    )

    # 提取分类编码（索引）
    row_indices = expression_df["cell_id"].cat.codes
    col_indices = expression_df["gene_id"].cat.codes
    values = expression_df["expression"]

    # 创建稀疏矩阵 (COO 格式)
    sparse_matrix = sparse.coo_matrix(
        (values, (row_indices, col_indices)),
        shape=(len(cell_categories), len(gene_categories))
    )

    # 3. 构建 AnnData 对象
    adata = anndata.AnnData(
        X=sparse_matrix.tocsr(),  # 转换为 CSR 格式
        obs=cells_df.set_index("cell_id"),  # 细胞元数据
        var=genes_df.set_index("gene_id"),  # 基因元数据
    )

    # 4. 保存为 h5ad 文件
    adata.write("output.h5ad")
    print("AnnData 文件已保存为 output.h5ad")

    adata = anndata.read_h5ad('output.h5ad')
    print("output.h5ad 文件的信息如下，，，"  )

    # 打印基础结构信息
    print("=== 基础结构信息 ===")
    print(f"表达矩阵维度: {adata.n_obs} 细胞 × {adata.n_vars} 基因")
    print(f"细胞元数据(obs)列数: {adata.obs.shape[1]}")
    print(f"基因元数据(var)列数: {adata.var.shape[1]}")

    # 打印元数据列名
    print("\n=== 细胞元数据(obs)列名 ===")
    print(adata.obs.columns.tolist())

    print("\n=== 基因元数据(var)列名 ===")
    print(adata.var.columns.tolist())

# 将数据库文件导出为anndata格式
# sql_to_anndata()


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

create_anndata_from_tables(con, "output_data.h5ad")

# 5.关闭数据库连接
con.close()

