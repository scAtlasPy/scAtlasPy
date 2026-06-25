import scanpy as sc
import scatlaspy as sap

adata = sc.datasets.pbmc3k()
adata.write_h5ad(r"E:\python\scAtlas\Package\python-version-scAtlasAnalysis\test\pbmc3k\pbmc3k.h5ad")

# 1. 建立数据库
import scatlaspy as sap
sap.set_progress(True)
sap.set_verbosity("info")
atlas = sap.Atlas(r"F:\data\database\pbmc3k")
file_path = r"E:\python\scAtlas\Package\python-version-scAtlasAnalysis\test\pbmc3k\pbmc3k.h5ad"
# file_path = r"F:\data\adata_JAX_dataset_1.h5ad"
atlas.load_h5ad(file_path,load_type="random")

#  小数据 / 和 Scanpy 对齐 / 做展示图
sap.pl.highest_expr_genes(atlas)

# 3. 过滤 filter_cells & filter_genes
sap.pp.filter_cells(atlas, min_genes=200)
sap.pp.filter_genes(atlas, min_cells=3)

# 4. QC
sap.pp.calculate_qc_metrics(atlas)
# 可视化QC指标
sap.pl.violin_qc_metrics(atlas)
# QC 散点图（scatter plot）
sap.pl.scatter_qc_metrics(atlas)

# 5. calculate_cell_total_counts & calculate_gene_total_counts
sap.pp.calculate_cell_total_counts(atlas)
sap.pp.calculate_gene_total_counts(atlas)

# 6. 归一化 normalize_total
sap.pp.normalize_and_log1p(atlas,target_sum=1e4)

# 8. 特征选择：识别高变基因
# 计算
sap.pp.highly_variable_genes(atlas,flavor = "seurat") #  seurat
# 可视化
sap.pl.highly_variable_genes(atlas,flavor="seurat")

# 9. scale：进行 z-score转换
sap.pp.scale(atlas)

# 11.过滤 + 建新表 + 建tid分块索引
atlas.build_read_index(use_data = "data_scale")

# 14. PCA
# 计算层：训练 PCA + 写入 obsm_X_pca / varm_PCs / uns_pca_stats
sap.tl.pca(
    atlas,
    n_components = 50,
    fit_batches = 50,
    batch_size  = 2048,
)
# 可视化层：只读 PCA 结果
sap.pl.pca(atlas, color="CST3")
sap.pl.pca_variance_ratio(atlas, n_pcs=20)
sap.pl.pca_loadings(
    atlas,
    components=(1, 2),
    include_lowest=True,
)
sap.pl.pca_variance_ratio_cumsum(atlas, n_pcs=20)


# 15. KMeans：计算 + 聚类大小 QC
# 计算层：基于 PCA 结果聚类，写入 obs.kmeans / obs_cluster / kmeans_centers
sap.tl.kmeans(
    atlas,
    n_components=40,
    n_clusters = 8,
    batch_size = 2048,
    fit_batches= 1000, # 训练轮数
    add_obs_col="kmeans"
)

# 可视化层：看每个 cluster 有多少细胞
sap.pl.kmeans_cluster_size(atlas, use_obs_col="kmeans")

# 16. UMAP：计算 + 可视化
# 计算层：基于 obsm_X_pca 计算 UMAP，写入 obsm_X_umap
sap.tl.umap(
    atlas,
    n_pcs=30,
    n_neighbors=50,
    min_dist=0.3,
    spread=1,
)
# spread 大 → 整体更铺开
# spread 小 → 整体更压缩
# n_neighbors 小 → 更强调局部，簇容易分开
# n_neighbors 大 → 更强调整体，簇之间距离可能更自然
# min_dist 大 → 点更分散
# min_dist 小 → 点更紧凑

sap.pl.umap(
    atlas,
    color="kmeans",
    sample_n=None, # 全量画图
    point_size=20,
    figsize=(10, 8),
) # uamp1
# 多个 marker gene 上色
sap.pl.umap(
    atlas,
    color=["CST3", "NKG7", "PPBP"],
    use_data="data_log1p"
) # uamp2
sap.pl.umap(
    atlas,
    color=["CST3", "NKG7", "PPBP"],
    use_data="data_count"
) # uamp3
sap.pl.umap(
    atlas,
    color=["CST3", "NKG7", "PPBP"],
    use_data="data_scale"
) # uamp4
# 混合显示 cluster + marker gene
sap.pl.umap(
    atlas,
    color=["kmeans", "CD14", "NKG7"],
    use_data="data_log1p"
) # uamp5

# 17. Marker gene 计算
sap.tl.rank_genes_groups(atlas , use_data="data_scale")
# 可视化：
sap.pl.rank_genes_groups(atlas)          # 差异基因排名图
sap.pl.rank_genes_groups_volcano(atlas)  # 火山图
sap.pl.rank_genes_groups_violin(atlas)   # 提琴图

sap.pl.violin(
    atlas,
    genes=["CST3", "NKG7", "PPBP"],
    groupby="kmeans"
) # violin CST3


# 18. 注释cluster  需要根据实际的图像进行调整
mapping = {
    "0": "CD4 T", #
    "1": "FCGR3A+ Monocytes", #
    "2": "NK",#
    "3": "Dendritic", #
    "4": "B",  #
    "5": "CD14+ Monocytes",#
    "6": "Megakaryocytes",#
    "7": "CD8 T",#
}
summary = sap.tl.manual_annotate_clusters(
    atlas,
    cluster_to_cell_type=mapping,
    groupby="kmeans",
    obs_col="cell_type_manual",
    table_name="manual_cluster_annotation",
)

print("========= summary =========")
print(summary.to_string(index=False))

sap.pl.umap(
    atlas,
    color="cell_type_manual",
    legend_loc="on_data",
    point_size=20,
    figsize=(12, 10),
) # umap cell_type_manual


# 19. Marker gene 总览图
marker_genes = [
    *["IL7R", "CD79A", "MS4A1", "CD8A", "CD8B", "LYZ", "CD14"],
    *["LGALS3", "S100A8", "GNLY", "NKG7", "KLRB1"],
    *["FCGR3A", "MS4A7", "FCER1A", "CST3", "PPBP"],
]

cell_type_order = [
    "CD4 T",
    "B",
    "CD14+ Monocytes",
    "NK",
    "CD8 T",
    "FCGR3A+ Monocytes",
    "Dendritic",
    "Megakaryocytes",
]

# 热图
sap.pl.dotplot(
    atlas,
    genes=marker_genes,
    groupby="cell_type_manual",
    use_data="data_log1p",
    sample_cells_per_group=None,
    order=cell_type_order,
)

# 堆叠提琴图
sap.pl.stacked_violin(
    atlas,
    genes=marker_genes,
    groupby="cell_type_manual",
    order=cell_type_order,
    color_vmin=0,
    color_vmax=5,
    font_size=14
)

print(atlas)