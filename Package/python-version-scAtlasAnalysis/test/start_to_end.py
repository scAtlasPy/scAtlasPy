# import os
#
# os.environ["QT_API"] = "pyqt6"
# os.environ["MPLBACKEND"] = "QtAgg"
#
# import matplotlib
# matplotlib.use("QtAgg", force=True)
import scatlaspy as sap

atlas = sap.Atlas("test_volcano",path=r"F:\data\database")
file_path = r"F:\data\database\tahoe_10000_minimal_volcano\tahoe_10000_processed_clean.h5ad"

# 1. 建立数据库
atlas = sap.Atlas("test_1M",path=r"F:\data\database")
file_path = r"F:\data\92b37feb-aa2c-40d7-bd90-0a9b5ddb3b27.h5ad" #

atlas = sap.Atlas("test_10W",path=r"F:\data\database")
file_path = r"F:\data\ALL_Tissue_Global_Clustering_Scanpy_sample10W.h5ad"

atlas = sap.Atlas("test_20W",path=r"F:\data\database")
file_path = r"F:\data\ALL_Tissue_Global_Clustering_Scanpy_sample20W.h5ad"

atlas = sap.Atlas("test_83W",path=r"F:\data\database")
file_path = r"F:\data\FullMouseBrain_raw.h5ad"

atlas = sap.Atlas("test_HBCA",path=r"F:\data\database")
file_path = r"F:\data\HBCA__subsample_20__hvg_2000.h5ad"

atlas = sap.Atlas("test_280W_p4",path=r"F:\data\database")
file_path = r"F:\data\adata_JAX_dataset_1.h5ad"

atlas = sap.Atlas("test_500W",path=r"F:\data\database")
file_path = r"F:\data\100M\tahoe100M_2025-02-25_h5ad_plate14_filt_Vevo_Tahoe100M_WServicesFrom_ParseGigalab.h5ad"

# 多文件导入
atlas = sap.Atlas("test_100M-123",path=r"F:\data\database")
h5ad_paths = [ # 18,251,480 x 62,710
    r"F:\data\100M\tahoe100M_2025-02-25_h5ad_plate1_filt_Vevo_Tahoe100M_WServicesFrom_ParseGigalab.h5ad",
    r"F:\data\100M\tahoe100M_2025-02-25_h5ad_plate2_filt_Vevo_Tahoe100M_WServicesFrom_ParseGigalab.h5ad",
    r"F:\data\100M\tahoe100M_2025-02-25_h5ad_plate3_filt_Vevo_Tahoe100M_WServicesFrom_ParseGigalab.h5ad",
]

# 2. 大文件读取
sap.io.load_h5ad_order(file_path , atlas)   # 顺序导入
sap.io.load_h5ad_random(file_path , atlas)  # 随机导入
sap.io.load_h5ad_fast( file_path , atlas )  # 随机导入：parquet文件中转， 快速版
sap.io.load_h5ad_list_random(h5ad_paths,atlas) # 多文件随机导入

# 导入完数据，直接画 最高表达基因占比图（highest expressed genes
# 用来检查 有没有少数基因“垄断”表达（技术偏差）

#  日常 / 大数据 / 正式流程
sap.pl.highest_expr_genes(
    atlas,
    n_top=20,
    use_all_cells=False,
    show_outliers=False,
    approx_quantile=True,
    sample_cells=1_0_000,
)

#  小数据 / 和 Scanpy 对齐 / 做展示图
sap.pl.highest_expr_genes(
    atlas,
    n_top=20,
    use_all_cells=True,
    show_outliers=True,
    max_outliers=5000, # 每个基因最多绘制多少个离群点
)

# 3. 过滤 filter_cells & filter_genes
sap.pp.filter_cells(atlas, min_genes=200)
sap.pp.filter_genes(atlas, min_cells=3)

# 4. QC
sap.pp.calculate_qc_metrics(atlas)

# 可视化QC指标
sap.pl.violin_qc_metrics(atlas)
sap.pl.violin_qc_metrics(atlas,sample_n=100000) # 数据特别大，想限制绘图点数

# QC 散点图（scatter plot）
sap.pl.scatter_qc_metrics(atlas)
sap.pl.scatter_qc_metrics(atlas,sample_n=100000)

# 5. calculate_cell_total_counts & calculate_gene_total_counts
sap.pp.calculate_cell_total_counts(atlas)
sap.pp.calculate_gene_total_counts(atlas)

# 6. 归一化 normalize_total
sap.pp.normalize_total(atlas,target_sum=1e6) # 法1：直接计算
sap.pp.normalize_total_scale_factor(atlas)   # 法2：初步计算， 在 obs表上记录 scale_factor ， 等到使用的时候在计算

# 7.log1p
sap.pp.log1p(atlas)
sap.pp.log1p_fast(atlas)
sap.pp.expm1(atlas) # log1p的逆运算

sap.pp.normalize_and_log1p(atlas,target_sum=1e6) # 推荐用这个 normalize 法2  + log1p法1

# 8. 特征选择：识别高变基因
# 计算
sap.pp.highly_variable_genes(atlas) # 识别高变基因
sap.pp.highly_variable_genes_seurat(atlas) #  seurat
# 可视化
sap.pl.highly_variable_genes_plot(atlas)
sap.pl.highly_variable_genes_plot_seurat(atlas)

# 9. scale：进行 z-score转换
sap.pp.scale(atlas)

# 10.sqrt
sap.pp.sqrt(atlas)        # 法1：分块,内存安全，适合大数据，稍慢
sap.pp.sqrt_fast(atlas)   # 法2：不分块，内存不安全，适合小数据，较快

# 11.过滤 + 建新表 + 建tid分块索引
atlas.filter_build_index()

# 12. minibatch 格式读取
atlas.minibatch_CSR( X_type = "CSR")   # CSR 格式读取
atlas.minibatch_dense(pass_mode = "single-pass") # 宽表 格式读取 , 单次遍历
atlas.minibatch_dense(pass_mode = "multi-pass" , max_batches = 500) # 宽表 格式读取 , 多次遍历

# 13. 导出
file_path = r"F:\data\out\test_10W_out.h5ad" # 导出文件路径名称
sap.io.export_atlas_to_h5ad(atlas,file_path) # 导出文件

# 导出为 df
obs_df = sap.io.export_obs_to_pandas(atlas)
# atlas_cell_ids = sap.io.get_filtered_cell_ids(obs_df)
atlas_cell_ids = obs_df["atlas_cell_id"].tolist()
adata = sap.io.export_atlas_to_anndata(atlas,atlas_cell_ids)
adata.write_h5ad(r"E:\python\scAtlas\Package\python-version-scAtlasAnalysis\test\data\out\test_819200-f.h5ad") # adata 导出为 h5ad


# 14. PCA
# 计算层：训练 PCA + 写入 obsm_X_pca / varm_PCs / uns_pca_stats
sap.tl.pca(
    atlas,
    n_components=30,
    fit_batches=100, # 训练轮数
)

# 可视化层：只读 PCA 结果
sap.pl.pca(atlas, color="cell_type")
sap.pl.pca_variance_ratio(atlas, n_pcs=30)
sap.pl.pca_variance_ratio_cumsum(atlas, n_pcs=30)


# 15. KMeans：计算 + 聚类大小 QC
# 计算层：基于 PCA 结果聚类，写入 obs.kmeans / obs_cluster / kmeans_centers
sap.tl.kmeans(
    atlas,
    n_components=30,
    n_clusters=30,
    fit_batches=100, # 训练轮数
    buffer_batch_num=5,
    obs_col="kmeans"
)

# 可视化层：看每个 cluster 有多少细胞
sap.pl.kmeans_cluster_size(atlas, obs_col="kmeans")


# 16. UMAP：计算 + 可视化
# 计算层：基于 obsm_X_pca 计算 UMAP，写入 obsm_X_umap
sap.tl.umap(
    atlas,
    fit_sample_n=500000,
    transform_batch_size=100000,
    n_neighbors=45,
    min_dist=0.2,
    random_state=42,
    n_jobs=1,
    eval_sample_n=10000,
)

# 可视化层：
sap.pl.umap(
    atlas,
    color="kmeans", # cell_type
    sample_n=None,    # 全量画图
    plot_batch_size=200000  # 分批加载画图
)
# 多个 marker gene 上色
sap.pl.umap(
    atlas,
    color=["CST3", "NKG7", "PPBP"],
    use_expr_field="data_log1p"
)
# 混合显示 cluster + marker gene
sap.pl.umap(
    atlas,
    color=["kmeans", "CST3", "NKG7", "PPBP"],
    use_expr_field="data_log1p"
)

# 17. Marker gene 计算
result_dict = sap.tl.rank_genes_groups(atlas)
# 可视化：
sap.pl.rank_genes_groups(atlas)          # 差异基因排名图
sap.pl.rank_genes_groups_volcano(atlas)  # 火山图
sap.pl.rank_genes_groups_violin(atlas)   # 提琴图


# 18. 自动注释：解释 cluster
summary_df, score_df = sap.tl.annotate_clusters(
    atlas,
    rank_result=result_dict,
    groupby="kmeans",
    reference_name="builtin_pbmc",
    write_to_obs=True,
    top_n=50
)
print("========= summary_df =========")
print(summary_df)
print("========= score_df =========")
print(score_df.head(30))


# 19. 自动注释结果验证
# 用 marker 验证 kmeans cluster
sap.pl.violin(
    atlas,
    genes=["IL7R", "CD14", "MS4A1"],
    groupby="kmeans"
)

# 用 marker 验证自动注释结果
sap.pl.violin(
    atlas,
    genes=["IL7R", "NKG7", "PPBP"],
    groupby="cell_type_auto"
)

# UMAP 上看自动注释
sap.pl.umap(
    atlas,
    color="cell_type_auto",
    legend_loc="on_data",
    frameon=False
)

# 只看 high / medium confidence
sap.pl.umap(
    atlas,
    color="cell_type_auto",
    where="cell_type_auto_confidence IN ('high', 'medium')",
    legend_loc="on_data",
    frameon=False
)

# 20. Marker gene 总览图
marker_genes = [
    "IL7R", "CD79A", "MS4A1", "CD8A", "CD8B",
    "LYZ", "CD14", "LGALS3", "S100A8",
    "GNLY", "NKG7", "KLRB1",
    "FCGR3A", "MS4A7", "FCER1A", "CST3", "PPBP"
]

celltype_order = [
    "CD4 T cells",
    "B cells",
    "CD14+ Monocytes",
    "NK cells",
    "CD8 T cells",
    "FCGR3A+ Monocytes",
    "Dendritic Cells",
    "Megakaryocytes"
]

# 热图
sap.pl.dotplot(
    atlas,
    genes=marker_genes,
    groupby="cell_type_auto",
    where="cell_type_auto_confidence IN ('high', 'medium')",
    order=celltype_order,
    standard_scale="var"
)

# 堆叠提琴图
expr_df, median_df = sap.pl.stacked_violin(
    atlas,
    genes=marker_genes,
    groupby="cell_type_auto",
    where="cell_type_auto_confidence IN ('high', 'medium')",
    order=celltype_order,
    color_vmin=0,
    color_vmax=5,
    font_size=14
)
