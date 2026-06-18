# 聚类与细胞类型注释

本教程覆盖单细胞分析流程的后半部分：PCA 降维、KMeans 聚类、UMAP 可视化、marker gene 排名和自动细胞类型注释。

本教程使用 **pbmc3k**（3k PBMC）数据集作为示例。开始前需要完成 {doc}`quality-control-preprocessing` 中的所有步骤，特别是 `build_read_index()`。

## 1. PCA

```python
sap.tl.pca(atlas, n_components=50, fit_batches=1000)
```

这一步使用流式增量 PCA，基于 `build_read_index()` 定义的过滤矩阵进行计算。结果会写入：

- `obsm_X_pca`：每个细胞的 PC 坐标
- `varm_PCs`：每个基因的 PC loading
- `uns_pca_stats`：每个 PC 的 variance 和 variance_ratio

### PCA 结果检查

```python
sap.pl.pca_variance_ratio(atlas, n_pcs=50)
sap.pl.pca_variance_ratio_cumsum(atlas, n_pcs=50)
```

![PCA 方差解释率](../_static/pbmc3k/pca_variance_ratio.png)

各主成分的方差解释率。第一个 PC 通常解释最多变异，曲线快速下降后趋于平缓是正常形态。

![PCA 累计方差解释率](../_static/pbmc3k/pca_variance_ratio_cumsum.png)

前 50 个 PC 累计解释 15–30% 的变异属于正常范围。

```python
sap.pl.pca(atlas, color="kmeans", sample_n=50000)
```

![PCA 散点图（按 KMeans 上色）](../_static/pbmc3k/pca.png)

观察不同 cluster 在 PCA 空间是否初步分开。如果所有 cluster 在 PC1/PC2 上完全混合，后续 UMAP 也不会有好的分离。

```python
# 查看 PC loading，了解哪些基因驱动了各主成分
sap.pl.pca_loadings(atlas, n_genes=20)
```

![PCA Loadings](../_static/pbmc3k/pca_loadings.png)

## 2. KMeans 聚类

```python
sap.tl.kmeans(atlas, n_clusters=10, fit_batches=1000)
```

这一步基于 PCA 结果，使用 MiniBatchKMeans 对细胞进行聚类。聚类结果写入 `obs.kmeans`。

```python
sap.pl.kmeans_cluster_size(atlas, use_obs_col="kmeans")
```

![KMeans 聚类大小分布](../_static/pbmc3k/kmeans_cluster_size.png)

展示每个 cluster 的细胞数量。如果某个 cluster 特别大或特别小，可能需要调整 `n_clusters`。

```{note}
当前 `sap.tl.kmeans()` 是 MiniBatchKMeans，不等同于 Scanpy 的 Leiden/Louvain 图聚类。它适合作为初步探索细胞群的工具。
```

## 3. UMAP

```python
sap.tl.umap(atlas, fit_sample_n=50000)
```

这一步基于 `obsm_X_pca` 计算 UMAP 二维坐标。参数说明：

| 参数 | 含义 | 怎么改 |
|---|---|---|
| `fit_sample_n=50000` | UMAP 拟合时抽样细胞数 | 小数据可设为 `None`（使用全部细胞），大数据可调大 |
| `n_neighbors` | 邻居数量（默认 15） | 越小越关注局部结构，越大越关注全局结构 |
| `min_dist` | 点之间的最小距离（默认 0.1） | 越小聚集越紧，越大分布越散 |

结果写入 `obsm_X_umap`，同时会写入 `uns_umap_params`（参数记录）和 `uns_umap_eval`（质量评估）。`uns_umap_eval` 表中包含两个指标：

| 指标 | 含义 | 越接近 1 越好 |
|---|---|---|
| `trustworthiness` | 降维保真度：高维空间中近邻的点在低维中是否仍然近邻 | 是 |
| `knn_overlap` | KNN 重叠率：高维和低维空间中 K 近邻集合的重叠比例 | 是 |

### UMAP 可视化

```python
sap.pl.umap(atlas, color="kmeans", sample_n=50000)
```

![UMAP（KMeans）](../_static/pbmc3k/umap_kmeans.png)

按 KMeans cluster 上色，观察聚类结果在 UMAP 空间是否连续且分离。

还可以按单个基因或多个基因的表达上色：

```python
# 按单个基因表达上色
sap.pl.umap(atlas, color="CST3", use_data="data_log1p", sample_n=50000)
```

![UMAP（CST3 基因表达）](../_static/pbmc3k/umap_gene_CST3.png)

```python
# 按多个基因并排
sap.pl.umap(
    atlas,
    color=["CST3", "NKG7", "PPBP", "MS4A1"],
    use_data="data_log1p",
    sample_n=50000,
    ncols=2,
)
```

![UMAP（多基因并排）](../_static/pbmc3k/umap_multi_gene.png)

按样本/批次上色可以检查批次效应：

```python
sap.pl.umap(atlas, color="sample_id", sample_n=50000)
```

![UMAP（按样本上色）](../_static/pbmc3k/umap_sample.png)

用 `where` 条件筛选特定 cluster 放大查看：

```python
sap.pl.umap(atlas, color="kmeans", sample_n=50000, where="kmeans IN (0, 1, 2)")
```

![UMAP（where 筛选）](../_static/pbmc3k/umap_where_filter.png)

## 4. 检查聚类结果

```python
# 统计每个 cluster 中有多少细胞
print(atlas.query("""
    SELECT kmeans, COUNT(*) AS n_cells
    FROM obs
    GROUP BY kmeans
    ORDER BY kmeans
"""))
```

若某些 cluster 过小或过大，通常需要回头检查过滤阈值、HVG 数量或 `n_clusters`。

常用可调参数回顾：

| 参数 | 含义 | 怎么改 |
|---|---|---|
| `n_components=50` | PCA 保留主成分数量 | 数据很小可用 30，复杂数据可用 50 或更多 |
| `n_clusters=10` | 预期 cluster 数 | 根据样本复杂度调大或调小 |
| `fit_batches=1000` | PCA/KMeans 训练的 batch 数 | 数据大时可调大，保证模型看到足够多数据 |

## 5. Marker gene 排名

```python
rank_result = sap.tl.rank_genes_groups(
    atlas,
    groupby="kmeans",
    n_genes=10,
)
```

`rank_result` 保存每个 cluster 的 marker gene 排名结果。你可以根据 marker gene 排名结果手动指定每个 cluster 的细胞类型，也可以用 violin、dotplot 和 stacked violin 检查 marker 表达（详见 {doc}`../advanced/marker-gene-ranking`）。

### Marker 可视化示例

**排名图**：

```python
sap.pl.rank_genes_groups(atlas, use_table="rank_genes_groups", n_genes=10)
```

![排名图](../_static/pbmc3k/rank_genes_groups.png)

**Dotplot**（圆点大小表示表达比例，颜色表示平均表达量）：

```python
marker_genes = ["IL7R", "CD79A", "MS4A1", "CD8A", "LYZ", "NKG7", "PPBP"]
sap.pl.dotplot(atlas, genes=marker_genes, groupby="kmeans", use_data="data_log1p")
```

![Dotplot](../_static/pbmc3k/dotplot.png)

**堆叠小提琴图**：

```python
sap.pl.stacked_violin(atlas, genes=marker_genes, groupby="kmeans", use_data="data_log1p")
```

![堆叠小提琴图](../_static/pbmc3k/stacked_violin.png)

**Rank genes 提琴图**（蓝色是目标 cluster，橙色是参考组）：

```python
sap.pl.rank_genes_groups_violin(
    atlas, group=0, groupby="kmeans", use_table="rank_genes_groups", n_genes=8, use_expr_field="data_log1p",
)
```

![Rank genes 提琴图](../_static/pbmc3k/rank_genes_groups_violin.png)

**火山图**：

```python
sap.pl.rank_genes_groups_volcano(atlas, group=0, use_table="rank_genes_groups")
```

![火山图](../_static/pbmc3k/rank_genes_groups_volcano.png)

**单基因小提琴图**（验证某个 marker 在各 cluster 的表达分布）：

```python
sap.pl.violin(atlas, genes=["CST3"], groupby="kmeans", use_data="data_log1p")
```

![单基因小提琴图](../_static/pbmc3k/violin_CST3.png)

## 6. 手动细胞类型注释

```python
cluster_to_cell_type = {
    "0": "CD4 T cells",
    "1": "CD14+ Monocytes",
    "2": "B cells",
    "3": "CD8 T cells",
    "4": "NK cells",
    "5": "FCGR3A+ Monocytes",
    "6": "Dendritic Cells",
    "7": "Megakaryocytes",
}

summary_df = sap.tl.manual_annotate_clusters(
    atlas,
    cluster_to_cell_type,
    groupby="kmeans",
    obs_col="cell_type_manual",
)
```

`manual_annotate_clusters()` 会把你提供的 cluster 到细胞类型的映射写回数据库。结果写入：

- `obs.cell_type_manual`：手动注释结果
- `manual_cluster_annotation`：cluster 到细胞类型的映射表

PBMC 示例中常见的 marker gene 包括：

| 细胞类型 | Marker Genes |
|---|---|
| CD4 T cells | IL7R |
| CD14+ Monocytes | CD14, LYZ |
| B cells | MS4A1 |
| CD8 T cells | CD8A |
| NK cells | GNLY, NKG7 |
| FCGR3A+ Monocytes | FCGR3A, MS4A7 |
| Dendritic Cells | FCER1A, CST3 |
| Megakaryocytes | PPBP |

```{note}
当前内置 reference 更适合 PBMC/blood 示例。其他组织需要替换 marker reference 或人工校验注释结果。你可以通过 `reference_name` 参数指定自定义 reference，或直接检查 `score_df` 中的打分结果来手动判断。
```

### 注释结果可视化

```python
sap.pl.umap(atlas, color="cell_type_manual", sample_n=50000)
```

![UMAP（细胞类型注释）](../_static/pbmc3k/umap_cell_type_manual.png)

按手动注释的细胞类型上色，验证注释结果在 UMAP 上的分布是否合理：相同类型应聚在一起形成连续区域，不同类型之间的边界应与 UMAP 的自然分隔一致。

## 7. 导出结果文件（可选）

```python
# 导出完整 h5ad 结果文件（默认导出 X_HyS_data.data 到 X）
atlas.write_h5ad("out.h5ad")

# 关闭数据库连接
atlas.close()
```

如果需要导出特定表达字段或细胞子集：

```python
obs_df = atlas.get_obs_df()
atlas_cell_ids = obs_df[obs_df["filter_cells"].notna()]["atlas_cell_id"].tolist()
adata = atlas.get_anndata(
    atlas_cell_ids,
    use_data="data_log1p",
    include_obsm=True,
    include_varm=True,
)
```

导出的 h5ad 可以继续用 Scanpy 打开，也可以交给下游工具使用。更多导出选项参考 {doc}`../../how-to/export-to-other-platforms`。

## 下一步

完成聚类和注释后，可以阅读 {doc}`../advanced/plot-parameter-guide` 学习如何用 QC 图、PCA 图、UMAP 图和 marker gene 图判断分析结果是否合理。
