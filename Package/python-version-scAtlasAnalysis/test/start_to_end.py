import scatlaspy as sap
import scanpy as sc
import pandas as pd
atlas=sap.Atlas("test_big_file",path=r"E:\python\scAtlasAnalysis-demo\test\database")

# todo 大数据导入 load_AnnData_chunk
sap.io.load_AnnData_chunk("E:\python\data\HBCA__subsample_20__hvg_2000.h5ad",atlas,batch_size=4096) # 载入数据

# 数据对比
atlas=sap.Atlas("test_big_file_1",path=r"E:\python\scAtlasAnalysis-demo\test\database")
adata=sc.read_h5ad("E:\python\data\HBCA__subsample_20__hvg_2000.h5ad")
sap.io.load_AnnData(adata,atlas) # 载入数据


# 用以下的函数
# minibatch_scan_order_cursor_csr_df_arrow_onlylie












