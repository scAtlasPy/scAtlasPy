from datetime import datetime
import scatlaspy as sap
import scanpy as sc
import pandas as pd
import logging # 管理各种类型的日志
from scatlaspy.io.input import read_smart

verbose=True # 是否展示中间过程信息

# 创建一个atlas对象，该对象管理与duckdb的各种交互
# test_102400
atlas=sap.Atlas("test_102400_noclean",path=r"E:\python\scAtlasAnalysis-demo\test\database")
atlas=sap.Atlas("big_file_test",path=r"E:\python\scAtlasAnalysis-demo\test\database")

# 大数据集 80万
adata=sc.read_h5ad("E:\python\data\FullMouseBrain_raw.h5ad") # E:\python\data\FullMouseBrain_raw.h5ad 大数据集地址
adata=adata[:102400] # 降采样

"E:\python\data\HBCA__subsample_20__hvg_2000.h5ad"

sap.io.inspect_h5ad_structure("E:\python\data\HBCA__subsample_20__hvg_2000.h5ad")
sap.io.load_AnnData_from_h5ad("E:\python\data\HBCA__subsample_20__hvg_2000.h5ad", atlas, batch_size=4096)

# 小数据集 700
# adata = sc.datasets.pbmc68k_reduced() # 提取数据


# file_path = r"E:\python\scAtlasAnalysis-demo\test\single-cell-data\HBCA__subsample_20__hvg_2000.loom"
# 根据文件的后缀名，实现不同类型的读取，调用scanpy的接口，都读取成anndata格式。
# adata=sap.io.read_smart(file_path)

print(adata.obs)
print(adata.var)
print(f"初始的AnnData对象信息:")
print(f"  - 细胞数: {adata.n_obs}")
print(f"  - 基因数: {adata.n_vars}")
print(f"  - 矩阵形状: {adata.X.shape}")
print(f"  - var_names: {adata.var_names}")
print(f"  - obs_names: {adata.obs_names}")
print(f"  - var_index: {adata.var.index}")
print(f"  - obs_index: {adata.obs.index}")


# adata = sap.io.clean_gene_names(adata) # 导入前清洗 adata数据清洗

sap.io.load_AnnData(adata,atlas) # 载入数据
sap.io.clean_genes_in_database(atlas) # 导入后清洗 在数据库中清洗数据


# 方案1：手动执行完整优化（推荐用于生产环境）
print("=== 执行完整数据库优化 ===")
atlas.comprehensive_database_optimization()

# 测试细胞过滤速度
# sap.pp.filter_cells(atlas,20,add_key="filter_cells_c1")
# sap.pp.filter_cells_minibatch(atlas,20,add_key="filter_cells_c2")
# sap.pp.filter_cells_minibatch_yield(atlas,2000,add_key="filter_cells_c2")
sap.io.load_AnnData(adata,atlas) # 载入数据
sap.io.clean_genes_in_database(atlas) # 导入后清洗 在数据库中清洗数据
sap.pp.filter_cells_minibatch_yield(atlas,2000,10,20000, 17000,
                                    batch_size=4096, add_key="filter_cells_c5")


sap.io.build_CSR_cell_index_simple(atlas) # 在CSR_data 中添加 cell_index列

sap.pp.normalize_total(atlas, target_sum = 10000, inplace=False,add_field = "n1")

sap.pp.highly_variable_genes(atlas, flavor="var", n_top_genes=2000)


import scatlaspy as sap
import scanpy as sc
import pandas as pd
atlas=sap.Atlas("test_102400_noclean",path=r"E:\python\scAtlasAnalysis-demo\test\database")


# todo 大数据导入 load_AnnData_chunk
sap.io.load_AnnData_chunk("E:\python\data\HBCA__subsample_20__hvg_2000.h5ad",atlas,batch_size=4096) # 载入数据


# todo  calculate_qc_metrics 线粒体基因过滤
sap.pp.calculate_qc_metrics(
    atlas,
    qc_prefix="MT-",   # 线粒体基因前缀
    qc_key="mt"        # 等价 adata.var['mt']
)

# 查看表结构
print("=== 表结构 ===")
structure = atlas.connection.execute("PRAGMA table_info(var);").fetchall()
for col in structure:
    print(f"字段名: {col[1]:<20} 类型: {col[2]:<15} 主键: {col[5]} 非空: {col[3]}")

print("\n=== 前3行数据 ===")
# 获取列名
cursor = atlas.connection.execute("SELECT * FROM var LIMIT 3;")
column_names = [description[0] for description in cursor.description]
rows = cursor.fetchall()

# 打印列名
print(" | ".join(column_names))
print("-" * 50)

# 打印数据
for row in rows:
    print(" | ".join(str(item) for item in row))

atlas.connection.execute("""SELECT
                          * 
                          FROM obs
                          LIMIT 3; """).fetchall()

# todo log1p and exp1
# log1p
sap.pp.log1p_chunked(
    atlas,
    base=None,
    select_data="data",
    add_field="X_log1p"
)

# exp1（还原）
sap.pp.exp1_chunked(
    atlas,
    base=None,
    select_data="X_log1p",
    add_field="X_recovered"
)

atlas.connection.execute("""SELECT
                          data,
                          X_recovered,
                          abs(data - X_recovered) AS diff
                          FROM X_CSR_data
                          LIMIT 10; """).fetchall()

atlas.connection.execute("""SELECT
                          data,
                          X_log1p,
                          X_recovered,
                          FROM X_CSR_data
                          LIMIT 3; """).fetchall()


# todo scale
# sap.pp.scale(atlas)
sap.pp.scale_gene_chunked(atlas)
atlas.connection.execute("""SELECT
                          data,
                          X_scale,
                          FROM X_CSR_data
                          LIMIT 3; """).fetchall()
atlas.connection.execute("""SELECT
                          id,
                          gene_id,
                          zero_scale_transform,
                          FROM var
                          LIMIT 3; """).fetchall()




# todo 过滤基因测试
original_genes = adata.n_vars
print(f"原始基因数: {original_genes}")

# 第二步：按最小counts数过滤
sc.pp.filter_genes(adata, min_counts=20000)
print(f"过滤 min_counts=20000 后基因数: {adata.n_vars}")





# 测试minibatch读取
for adata_minibatch in atlas.query_minibatch():
    print(f"生成的AnnData对象信息:")
    print(f"  - 细胞数: {adata_minibatch.n_obs}")
    print(f"  - 基因数: {adata_minibatch.n_vars}")
    print(f"  - 矩阵形状: {adata_minibatch.X.shape}")
    print(f"  - var_names: {adata_minibatch.var_names}")
    print(f"  - obs_names: {adata_minibatch.obs_names}")








# '''测试视图的创建'''
atlas1 = atlas[1]
atlas2 = atlas[2]
atlas3 = atlas1[0]  # 基于视图，再建视图

atlas4 = atlas[0:3]
atlas5 = atlas[4:10]
atlas6 = atlas4[0]

atlas8 = atlas['AAATTCGATGCACA-1']
atlas4['AAATTCGATGCACA-1']

c=['AAATTCGATGCACA-1','A']
b=['AAATTCGATGCACA-1','AAATTCGATGCACA-1']
b=['AAATTCGATGCACA-1','AAATTCGATGCACA-1','C']

# 2.
# atlas.load_AnnData(adata) #fig,ax=plt.subplots()
# # ax.scatter(X[:,0],X[:,1])

# atlas.query("SELECT * FROM X LIMIT 5")
#
# atlas.query_raw("SELECT * FROM X LIMIT 5")

# # 以minibatch的方式按顺序遍历该数据库中的细胞，返回其所有基因表达值和对应的obs，var
# scaner=atlas.scan(mode="order",chunk_size=2048)
# for i, adata in scaner:
#     print(i,adata.shape)

# # 把所有数据全部读取出来，返回anndata对象
# adata=sap.get_adata()
# print(adata)

#
# # 创建Atlas实例
# atlas = Atlas("my_database", "/path/to/database")
#
# 1. 像列表一样使用 - 通过索引访问
print(atlas[0])  # 获取第一个表的数据
# print(atlas[-1])  # 获取最后一个表的数据

# 2. 像列表一样使用 - 通过切片访问
print(atlas[1:3])  # 获取第2到第3个表的数据
print(atlas[:2])   # 获取前2个表的数据

# 3. 像字典一样使用 - 通过表名访问
print(atlas['X'])        # 获取名为'X'的表数据
print(atlas['obs'])  # 获取名为'samples'的表数据

# 4. 设置表数据（创建或更新）

new_data = pd.DataFrame({
    'id': [1, 2, 3],
    'value': ['A', 'B', 'C']
})
atlas['new_table'] = new_data  # 创建新表
print(atlas['new_table'])

# 5. 删除表
del atlas['new_table']     # 通过表名删除
del atlas[0]              # 通过索引删除

# 6. 其他字典式操作
print(len(atlas))           # 表数量
print('X' in atlas)         # 检查表是否存在
print(list(atlas.keys()))   # 所有表名
print(list(atlas.values())) # 所有表数据
print(list(atlas.items()))  # (表名, 表数据)对

# 7. 迭代表名
for table_name in atlas:
    print(f"表名: {table_name}")
    table_data = atlas[table_name]
    print(f"数据形状: {table_data.shape}")