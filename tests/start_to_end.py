# import os
#
# os.environ["QT_API"] = "pyqt6"
# os.environ["MPLBACKEND"] = "QtAgg"
#
# import matplotlib
# matplotlib.use("QtAgg", force=True)
import scatlaspy as sap

# todo 要不要推荐用户限制内存
    # 日志控制   -- OK
    # 输出控制   -- OK 排查哪些需要print， 哪些不需要
    # 注释控制
    # 命名控制    -- ok
    # 枚举类型排查 -- ok

# 1. 建立数据库
atlas = sap.Atlas(r"F:\data\database\pbmc3k")
import scanpy as sc
adata = sc.datasets.pbmc3k()
atlas.load_anndata(adata)

import scatlaspy as sap
atlas = sap.Atlas(r"F:\data\database\test_10W_1")
file_path = r"F:\data\ALL_Tissue_Global_Clustering_Scanpy_sample10W.h5ad"
atlas.load_h5ad(file_path,load_type="random")

# file_path = r"F:\data\ALL_Tissue_Global_Clustering_Scanpy_sample20W.h5ad"
# file_path = r"F:\data\FullMouseBrain_raw.h5ad"
# file_path = r"F:\data\HBCA__subsample_20__hvg_2000.h5ad"
# file_path = r"F:\data\adata_JAX_dataset_1.h5ad"
# file_path = r"F:\data\100M\tahoe100M_2025-02-25_h5ad_plate14_filt_Vevo_Tahoe100M_WServicesFrom_ParseGigalab.h5ad"

#  多文件导入
# atlas = sap.Atlas(r"F:\data\database\test_100M-123")
h5ad_paths = [ # 18,251,480 x 62,710
    r"F:\data\100M\tahoe100M_2025-02-25_h5ad_plate1_filt_Vevo_Tahoe100M_WServicesFrom_ParseGigalab.h5ad",
    r"F:\data\100M\tahoe100M_2025-02-25_h5ad_plate2_filt_Vevo_Tahoe100M_WServicesFrom_ParseGigalab.h5ad",
    r"F:\data\100M\tahoe100M_2025-02-25_h5ad_plate3_filt_Vevo_Tahoe100M_WServicesFrom_ParseGigalab.h5ad",
]

# 2. 大文件读取
atlas.load_h5ad(file_path,load_type = "order")   # 顺序导入
atlas.load_h5ad(file_path,load_type = "random")   # 随机导入
atlas.load_h5ad(h5ad_paths,load_type = "random")   # 多文件随机导入

# 导入完数据，直接画 最高表达基因占比图（highest expressed genes
# 用来检查 有没有少数基因“垄断”表达（技术偏差）

#  日常 / 大数据 / 正式流程
sap.pl.highest_expr_genes(
    atlas,
    n_top=20,
    use_all_cells=False,
    show_outliers=False,
    approx_quantile=True,
    sample_cells=10_000,
)

#  小数据 / 和 Scanpy 对齐 / 做展示图
sap.pl.highest_expr_genes(atlas)

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
sap.pl.scatter_qc_metrics(atlas,sample_n=100000)  # 数据特别大，想限制绘图点数

# 5. calculate_cell_total_counts & calculate_gene_total_counts
sap.pp.calculate_cell_total_counts(atlas)
sap.pp.calculate_gene_total_counts(atlas)

# 6. 归一化 normalize_total
sap.pp.normalize_total(atlas,target_sum=1e6) # 法1：直接计算
sap.pp.normalize_total_scale_factor(atlas)   # 法2：初步计算， 在 obs表上记录 scale_factor ， 等到使用的时候在计算

# 7.log1p
sap.pp.log1p(atlas)
sap.pp.expm1(atlas) # log1p的逆运算

sap.pp.normalize_and_log1p(atlas,target_sum=1e6) # 推荐用这个 normalize 法2  + log1p法1

# 8. 特征选择：识别高变基因
# 计算
sap.pp.highly_variable_genes(atlas,flavor = "cv") # 识别高变基因
sap.pp.highly_variable_genes(atlas,flavor = "seurat") #  seurat
# 可视化
sap.pl.highly_variable_genes(atlas,flavor="cv")
sap.pl.highly_variable_genes(atlas,flavor="seurat")

# 9. scale：进行 z-score转换
sap.pp.scale(atlas)

# 10.sqrt
sap.pp.sqrt(atlas)

# 11.过滤 + 建新表 + 建tid分块索引
atlas.build_read_index()

# 12. minibatch 格式读取
atlas.get_minibatch_csr(x_type="CSR")   # CSR 格式读取
atlas.get_minibatch_dense(pass_mode ="single-pass") # 宽表 格式读取 , 单次遍历
atlas.get_minibatch_dense(pass_mode ="multi-pass", max_batches = 500) # 宽表 格式读取 , 多次遍历

# 13. 导出
file_path = r"F:\data\out\test_10W_out.h5ad" # 导出文件路径名称
atlas.write_h5ad(file_path) # 导出文件

# 导出为 df
obs_df = atlas.get_obs_df()
atlas_cell_ids = obs_df["atlas_cell_id"].tolist()
adata = atlas.get_anndata(atlas_cell_ids)
adata.write_h5ad(r"F:\data\out\test_10W_out_1.h5ad") # adata 导出为 h5ad


# 14. PCA
# 计算层：训练 PCA + 写入 obsm_X_pca / varm_PCs / uns_pca_stats
sap.tl.pca(
    atlas,
    n_components=50, # 30
    fit_batches=100, # 训练轮数 1000
)

# 可视化层：只读 PCA 结果
sap.pl.pca(atlas, color="CST3") # embryo_id  keep organ louvain CST3
sap.pl.pca_loadings(
    atlas,
    components=(1, 2),
    include_lowest=True,
)
sap.pl.pca_variance_ratio(atlas, n_pcs=20)
sap.pl.pca_variance_ratio_cumsum(atlas, n_pcs=20)


# 15. KMeans：计算 + 聚类大小 QC
# 计算层：基于 PCA 结果聚类，写入 obs.kmeans / obs_cluster / kmeans_centers
sap.tl.kmeans(
    atlas,
    n_components=30,
    n_clusters=8,
    fit_batches=100, # 训练轮数
    buffer_batch_num=5,
    use_obs_col="kmeans"
)

# 可视化层：看每个 cluster 有多少细胞
sap.pl.kmeans_cluster_size(atlas, use_obs_col="kmeans")


# 16. UMAP：计算 + 可视化
# 计算层：基于 obsm_X_pca 计算 UMAP，写入 obsm_X_umap
# todo 这个函数的计算和画图有冲突
sap.tl.umap(
    atlas,
    fit_sample_n=500000,
    transform_batch_size=100000,
    n_neighbors=40,
    min_dist=0.5,
    n_jobs=1,
    eval_sample_n=10000,
)
sap.tl.umap(
    atlas,
    n_neighbors=15,
    min_dist=0.5,
)

# 可视化层：
sap.pl.umap(
    atlas,
    color="kmeans",
    sample_n=None,
    point_size=20,
    alpha=0.85,
    figsize=(10, 8),
)


sap.pl.umap(
    atlas,
    color="kmeans", # cell_type
    sample_n=None,    # 全量画图
)



# 多个 marker gene 上色
sap.pl.umap(
    atlas,
    color=["CST3", "NKG7", "PPBP"],
    use_data="data_log1p"
)
# 混合显示 cluster + marker gene
sap.pl.umap(
    atlas,
    color=["kmeans", "CST3", "NKG7", "PPBP"],
    use_data="data_log1p"
)

# 17. Marker gene 计算
sap.tl.rank_genes_groups(atlas)
# 可视化：
sap.pl.rank_genes_groups(atlas)          # 差异基因排名图
sap.pl.rank_genes_groups_volcano(atlas)  # 火山图
sap.pl.rank_genes_groups_violin(atlas)   # 提琴图


# 18. 自动注释：解释 cluster
summary_df, score_df = sap.tl.annotate_clusters(
    atlas,
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
