import scatlaspy as sap

# 1. 建立数据库
# atlas = sap.Atlas("test_819200",path=r"E:\python\scAtlas\Package\python-version-scAtlasAnalysis\test\database")
# file_path = r"E:\python\data\FullMouseBrain_raw.h5ad" # 833206 * 17745

# atlas = sap.Atlas("test_jax",path=r"E:\python\scAtlas\Package\python-version-scAtlasAnalysis\test\database")
file_path = r"E:\python\scAtlas\Package\python-version-scAtlasAnalysis\test\data\adata_JAX_dataset_1.h5ad"
# 2840130 x 24552

atlas = sap.Atlas("test_pbmc3k",path=r"E:\python\scAtlas\Package\python-version-scAtlasAnalysis\test\database")
# import scanpy as sc
# adata = sc.datasets.pbmc3k()
# sap.io.load_AnnData(adata,atlas)
# 2700 x 32738

# 查看数据规模
gene_num = atlas.connection.execute("SELECT COUNT(*) FROM var").fetchone()[0]
cell_num = atlas.connection.execute("SELECT COUNT(*) FROM obs").fetchone()[0]
print(f"[数据规模]: {cell_num} x {gene_num}" )

# 2. 导入
sap.io.load_small_to_duckdb(file_path , atlas)    # 小数据导入 ， 支持多种格式
sap.io.load_big_h5ad_to_duckdb(file_path , atlas) # 大数据导入 ，顺序， 只支持h5ad
sap.io.load_big_h5ad_to_duckdb_random(file_path , atlas) # 大数据导入 ，随机， 只支持h5ad

# # 导入完数据，直接画图:
# # “最高表达基因占比图（highest expressed genes）”
# # 👉 用来检查 有没有少数基因“垄断”表达（技术偏差）
# 画图：
# 1. 日常 / 大数据 / 正式流程
sap.pl.plot_highest_expr_genes_sql(
    atlas,
    n_top=20,
    use_all_cells=False,
    show_outliers=False,
)
# 2. 小数据 / 和 Scanpy 对齐 / 做展示图
sap.pl.plot_highest_expr_genes_sql(
    atlas,
    n_top=20,
    use_all_cells=True,
    show_outliers=True,
    max_outliers=5000,
)

# 3. 过滤 filter_cells & filter_genes
sap.pp.filter_cells(atlas, min_genes=200)
sap.pp.filter_genes(atlas, min_cells=3)

# 4. QC
sap.pp.calculate_qc_metrics(atlas)

# 可视化QC指标（这里放！） 画 QC 小提琴图（这里最合适）
sap.pl.violin_qc_metrics(atlas)
sap.pl.violin_qc_metrics( atlas,sample_n = 50000 ) # 数据特别大，想限制绘图点数

# QC 散点图（scatter plot
sap.pl.scatter_qc_metrics(atlas)   # 👈 这个就放这里
sap.pl.scatter_qc_metrics(atlas,sample_n=20000)

# 5. calculate_cell_total_counts & calculate_gene_total_counts
sap.pp.calculate_cell_total_counts(atlas)
sap.pp.calculate_gene_total_counts(atlas)

# 6. 归一化 normalize_total
sap.pp.normalize_total(atlas)  # 法1：直接计算
sap.pp.normalize_total_scale_factor(atlas) # 法2：初步计算， 在 obs表上记录 scale_factor ， 等到使用的时候在计算

# 7.log1p
sap.pp.log1p(atlas)      # 法1：分块,内存安全，适合大数据，稍慢
sap.pp.log1p_fast(atlas) # 法2：不分块，内存不安全，适合小数据，较快

sap.pp.expm1(atlas) # log1p的逆运算

sap.pp.normalize_and_log1p(atlas) # normalize 法2  + log1p法1

# 8. 特征选择：识别高变基因
sap.pp.highly_variable_genes(atlas) # 识别高变基因
sap.pp.highly_variable_genes_seurat_v3(atlas)  # 识别高变基因 seurat_v3
sap.pl.highly_variable_genes_plot(atlas) # 高变基因 可视化
sap.pl.highly_variable_genes_plot_seurat_v3(atlas)

# 9. scale：进行 z-score转换
sap.pp.scale(atlas)       # 法1：分块,内存安全，适合大数据，稍慢
sap.pp.scale_fast(atlas)  # 法2：不分块，内存不安全，适合小数据，较快
sap.pp.scale_id_chunk(atlas) # 法3 ：不产生大量临时数据

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

# =========================================================
# 14. PCA：计算 + 可视化
# =========================================================

# 计算层：训练 PCA + 写入 obsm_X_pca / varm_PCs / uns_pca_stats
sap.tl.pca(atlas, n_components=50)

# 可视化层：只读 PCA 结果
sap.pl.pca_variance_ratio(atlas, n_pcs=50)
sap.pl.pca_variance_ratio_cumsum(atlas, n_pcs=50)
sap.pl.pca(atlas, color="CST3")

# =========================================================
# 15. KMeans：计算 + 聚类大小 QC
# =========================================================

# 计算层：基于 PCA 结果聚类，写入 obs.kmeans / obs_cluster / kmeans_centers
sap.tl.kmeans(
    atlas,
    n_clusters=10,
    batch_size=2048,
    obs_col="kmeans"
)

# 可视化层：看每个 cluster 有多少细胞
sap.pl.kmeans_cluster_size(atlas, obs_col="kmeans")


# =========================================================
# 16. UMAP：计算 + 可视化
# =========================================================

# 计算层：基于 obsm_X_pca 计算 UMAP，写入 obsm_X_umap
sap.tl.umap(
    atlas,
    fit_sample_n=50000,
    transform_batch_size=50000,
    n_neighbors=15,
    min_dist=0.5
)

# 可视化层：按 cluster 上色
sap.pl.umap(atlas, color="kmeans")

# 可视化层：多个 marker gene 上色
sap.pl.umap(
    atlas,
    color=["CST3", "NKG7", "PPBP"],
    use_expr_field="data_log1p"
)

# 可视化层：混合显示 cluster + marker gene
sap.pl.umap(
    atlas,
    color=["kmeans", "CST3", "NKG7", "PPBP"],
    use_expr_field="data_log1p"
)

# =========================================================
# 17. Marker gene 排名：发现特征
# =========================================================

result_dict = sap.pl.plot_rank_genes_groups(
    atlas,
    groupby="kmeans",
    use_expr_field="data_log1p",
    method="t-test",
    mask_var=None
)

sap.pl.plot_rank_genes_groups_violin(
    atlas,
    group=0,
    groupby="kmeans",
    rank_result=result_dict,
    genes=["IL32", "LTB", "IL7R", "CD2"],
    use_expr_field="data_log1p"
)


# =========================================================
# 18. 自动注释：解释 cluster
# =========================================================

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


# =========================================================
# 19. 自动注释结果验证
# =========================================================

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


# =========================================================
# 20. Marker gene 总览图
# =========================================================

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

sap.pl.dotplot(
    atlas,
    genes=marker_genes,
    groupby="cell_type_auto",
    where="cell_type_auto_confidence IN ('high', 'medium')",
    order=celltype_order,
    standard_scale="var"
)

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
