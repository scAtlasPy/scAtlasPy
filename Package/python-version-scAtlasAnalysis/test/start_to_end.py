import matplotlib
matplotlib.use("Agg")
# Matplotlib 正在用 tkagg 图形后端。
# 但你前面代码里有多线程、UMAP、可能还有旧图窗口对象。Tkinter 要求 GUI 操作必须在主线程，结果对象析构时不在主线程，就打印了这个异常。
# 在导入 scanpy / matplotlib / scatlaspy 之前，强制换成非交互后端：
import scatlaspy as sap

# 1. 建立数据库
atlas = sap.Atlas("test_1M",path=r"F:\data\database")
file_path = r"F:\data\92b37feb-aa2c-40d7-bd90-0a9b5ddb3b27.h5ad" #

atlas = sap.Atlas("test_10W",path=r"F:\data\database")
file_path = r"F:\data\ALL_Tissue_Global_Clustering_Scanpy_sample10W.h5ad"
# 数据集维度: 100,000 × 3,000    nnz:   22,841,082

atlas = sap.Atlas("test_20W",path=r"F:\data\database")
file_path = r"F:\data\ALL_Tissue_Global_Clustering_Scanpy_sample20W.h5ad"

atlas = sap.Atlas("test_83W",path=r"F:\data\database")
file_path = r"F:\data\FullMouseBrain_raw.h5ad"

atlas = sap.Atlas("test_HBCA",path=r"F:\data\database")
file_path = r"F:\data\HBCA__subsample_20__hvg_2000.h5ad"

atlas = sap.Atlas("test_280W",path=r"F:\data\database")
file_path = r"F:\data\adata_JAX_dataset_1.h5ad"

# 2. 大文件读取，subbatch 随机读取 +  每5个batch 合并 成一个大的batch ，再随机打乱，再导入, 只支持 h5ad 格式
sap.io.load_big_h5ad_to_duckdb_random_batch_window(file_path , atlas ,
                                                   batch_size=1024 ,shuffle_window_batches = 5,
                                                   store_type="count",
                                                   )
sap.io.load_big_h5ad_to_duckdb(file_path , atlas) # 顺序导入

# 多文件导入
atlas = sap.Atlas("test_100M-123",path=r"F:\data\database")
h5ad_paths = [ # 18,251,480 x 62,710
    r"F:\data\100M\tahoe100M_2025-02-25_h5ad_plate1_filt_Vevo_Tahoe100M_WServicesFrom_ParseGigalab.h5ad",
    r"F:\data\100M\tahoe100M_2025-02-25_h5ad_plate2_filt_Vevo_Tahoe100M_WServicesFrom_ParseGigalab.h5ad",
    r"F:\data\100M\tahoe100M_2025-02-25_h5ad_plate3_filt_Vevo_Tahoe100M_WServicesFrom_ParseGigalab.h5ad",
]
result = sap.io.load_big_h5ad_list_to_duckdb_random_batch_pool(h5ad_paths,atlas,store_type="count")

# 导入完数据，直接画图:
# 最高表达基因占比图（highest expressed genes
# 用来检查 有没有少数基因“垄断”表达（技术偏差）

#  日常 / 大数据 / 正式流程
sap.pl.highest_expr_genes_sql(
    atlas,
    n_top=20,
    use_all_cells=False,
    show_outliers=False,
    approx_quantile=True,
    sample_cells=1_0_000,
)
#  小数据 / 和 Scanpy 对齐 / 做展示图
sap.pl.highest_expr_genes_sql(
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

# 可视化QC指标（这里放！） 画 QC 小提琴图
sap.pl.violin_qc_metrics(atlas)
sap.pl.violin_qc_metrics(atlas,sample_n=100000) # 数据特别大，想限制绘图点数

# QC 散点图（scatter plot
sap.pl.scatter_qc_metrics(atlas)   # 👈 这个就放这里
sap.pl.scatter_qc_metrics(atlas,sample_n=100000)

# 5. calculate_cell_total_counts & calculate_gene_total_counts
sap.pp.calculate_cell_total_counts(atlas)
sap.pp.calculate_gene_total_counts(atlas)

# 6. 归一化 normalize_total
sap.pp.normalize_total(atlas,target_sum=1e6)  # 590
sap.pp.normalize_total_scale_factor(atlas) # 法2：初步计算， 在 obs表上记录 scale_factor ， 等到使用的时候在计算

# 7.log1p
sap.pp.log1p(atlas)  # 2559.50 秒
sap.pp.log1p_fast(atlas)

sap.pp.expm1(atlas) # log1p的逆运算

sap.pp.normalize_and_log1p(atlas,target_sum=1e6) # todo 推荐用这个 normalize 法2  + log1p法1

# 8. 特征选择：识别高变基因
sap.pp.highly_variable_genes(atlas) # 识别高变基因
sap.pp.highly_variable_genes_seurat(atlas) #  seurat

sap.pl.highly_variable_genes_plot(atlas) # 高变基因 可视化
sap.pl.highly_variable_genes_plot_seurat(atlas)


# 9. scale：进行 z-score转换
sap.pp.scale(atlas)       #  正确的含0值的， 时间几乎和  scale 一样

# 10.sqrt
sap.pp.sqrt(atlas)        # 法1：分块,内存安全，适合大数据，稍慢
sap.pp.sqrt_fast(atlas)   # 法2：不分块，内存不安全，适合小数据，较快

# 11.过滤 + 建新表 + 建tid分块索引
atlas.filter_build_index()

# 12. minibatch 格式读取
atlas.minibatch_CSR( X_type = "CSR")   # CSR 格式读取
atlas.minibatch_dense(pass_mode = "single-pass") # 宽表 格式读取 , 单次遍历
atlas.minibatch_dense(pass_mode = "multi-pass" , max_batches = 500) # 宽表 格式读取 , 多次遍历

# [Speed] output_batches=95, [ current=2428.36 batch/s, 4973288 cells/s, ][ avg=1808.80 batch/s, 3704416 cells/s ]

# 13. 导出
file_path = r"F:\data\out\test_10W_out.h5ad" # 导出文件路径名称
sap.io.export_duckdb_to_h5ad(atlas,file_path) # 导出文件

# 导出为 df
obs_df = sap.io.export_obs_to_pandas(atlas)
atlas_cell_ids = sap.io.get_filtered_cell_ids(obs_df)
adata = sap.io.export_cells_to_anndata(atlas,atlas_cell_ids)
#  adata 导出为 h5ad
adata.write_h5ad(r"E:\python\scAtlas\Package\python-version-scAtlasAnalysis\test\data\out\test_819200-f.h5ad")


# 14. PCA
# 计算层：训练 PCA + 写入 obsm_X_pca / varm_PCs / uns_pca_stats
sap.tl.pca(
    atlas,
    n_components=30,
    fit_batches=1000, # 训练轮数
) # minibatch pca

sap.tl.pca_simple(atlas, n_components=30) # scanpy风格的 全量的pca

# 可视化层：只读 PCA 结果
sap.pl.pca(atlas, color="cell_type")
sap.pl.pca_variance_ratio(atlas, n_pcs=30)  # ok
sap.pl.pca_variance_ratio_cumsum(atlas, n_pcs=30) # ok

# =========================================================
# 15. KMeans：计算 + 聚类大小 QC
# =========================================================

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
# todo 这个图要改成什么？ barplot 可以显示成
sap.pl.kmeans_cluster_size(atlas, obs_col="kmeans")


# =========================================================
# 16. UMAP：计算 + 可视化
# =========================================================

# 计算层：基于 obsm_X_pca 计算 UMAP，写入 obsm_X_umap
# 方法 1 ： 传统 umap 抽样转换显示
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

sap.pl.umap(
    atlas,
    color="kmeans", # cell_type
    sample_n=None,    # ✅ 全量画图
    point_size=1.0,
    figsize=(28, 12),  # ✅ 修改：更像 Scanpy 横向布局，给右侧 legend 留空间
    alpha=0.8,
    plot_batch_size=200000   # ✅ 分批加载画图
)

# todo
#  1. t-test 的代码拆解，要画火山图，问一下许晗
#  2. 排查所有新写的函数；
#  3. 首次运行会出现很慢， 内存的缓存占满了可能；看看要怎么解决； -- 暂时不管了
#  4. 这个图要改成什么？ barplot 可以显示成
#     sap.pl.kmeans_cluster_size(atlas, obs_col="kmeans")
#  5. 画图自动布局

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
