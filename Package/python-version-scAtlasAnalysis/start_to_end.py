import scatlaspy as sap
import scanpy as sc
import pandas as pd

# 1. 建立数据库
atlas = sap.Atlas("test_819200",path=r"E:\python\scAtlas\Package\python-version-scAtlasAnalysis\test\database")
file_path = r"E:\python\data\FullMouseBrain_raw.h5ad" # 833206 * 17745


# 2. 导入
sap.io.load_small_to_duckdb(file_path , atlas)    # 小数据导入 ， 支持多种格式
sap.io.load_big_h5ad_to_duckdb(file_path , atlas) # 大数据导入 ，顺序， 只支持h5ad
sap.io.load_big_h5ad_to_duckdb_random(file_path , atlas) # 大数据导入 ，随机， 只支持h5ad

# 3. 过滤 filter_cells & filter_genes
sap.pp.filter_cells(atlas,min_genes = 200)
sap.pp.filter_genes(atlas,min_cells = 3)

# 4. QC
sap.pp.calculate_qc_metrics(atlas)

# 5. calculate_cell_total_counts & calculate_gene_total_counts
sap.pp.calculate_cell_total_counts(atlas)
sap.pp.calculate_gene_total_counts(atlas)

# 6. 归一化 normalize_total
sap.pp.normalize_total(atlas)  # 法1：直接计算
sap.pp.normalize_total_scale_factor(atlas) # 法2：初步计算， 在 obs表上记录 scale_factor ， 等到使用的时候在计算

# 7.log1p
sap.pp.log1p(atlas)      # 法1：分块,内存安全，适合大数据，稍慢
sap.pp.log1p_fast(atlas) # 法2：不分块，内存不安全，适合小数据，较快

sap.pp.exp1(atlas) # log1p的逆运算

sap.pp.normalize_and_log1p(atlas) # normalize 法2  + log1p法1

# 8. 特征选择：识别高变基因
sap.pp.highly_variable_genes(atlas) # 识别高变基因

# 9. scale：进行 z-score转换
sap.pp.scale(atlas)       # 法1：分块,内存安全，适合大数据，稍慢
sap.pp.scale_fast(atlas)  # 法2：不分块，内存不安全，适合小数据，较快

# 10.sqrt
sap.pp.sqrt(atlas)        # 法1：分块,内存安全，适合大数据，稍慢
sap.pp.sqrt_fast(atlas)   # 法2：不分块，内存不安全，适合小数据，较快

# 11.过滤 + 建新表 + 建tid分块索引
atlas.filter_build_index()

# 12. minibatch 格式读取
atlas.minibatch_CSR()   # CSR 格式读取
atlas.minibatch_dense(pass_mode = "single-pass") # 宽表 格式读取 , 单次遍历
atlas.minibatch_dense(pass_mode = "multi-pass") # 宽表 格式读取 , 多次遍历

# 13. 导出
file_path = r"E:\python\scAtlas\Package\python-version-scAtlasAnalysis\test\data\out\test_out.h5ad" # 导出文件路径名称
sap.io.export_duckdb_to_h5ad(atlas,file_path) # 导出文件

# 14. 降维 pca     todo 待补充
# 15. 聚类 leiden  todo 待补充

