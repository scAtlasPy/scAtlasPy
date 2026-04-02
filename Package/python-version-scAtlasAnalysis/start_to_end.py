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

# 3. QC
sap.pp.calculate_qc_metrics(atlas)

# 4. calculate_cell_total_counts & calculate_gene_total_counts
sap.pp.calculate_cell_total_counts(atlas)
sap.pp.calculate_gene_total_counts(atlas)

# 5. 归一化
# sap.pp.normalize_total(atlas)            # 法1 ：全部
sap.pp.normalize_total_chunked(atlas)  # 法2 ：分块
sap.pp.normalize_total_streaming(atlas)  # 法2 ：分块流式
# sap.pp.normalize_total_scale_factor(atlas) # 法3 ：分块， 在 obs表上记录 scale_factor ， 等到使用的时候在计算







