# 聚类与细胞类型注释

本教程覆盖单细胞分析流程的后半部分：PCA 降维、KMeans 聚类、UMAP 可视化、marker gene 排名和自动细胞类型注释。

开始前需要完成 {doc}`quality-control-preprocessing` 中的所有步骤，特别是 `build_read_index()`。

## 1. PCA

```python
sap.tl.pca(atlas, n_components=50, fit_batches=1000)
```

这一步使用流式增量 PCA，基于 `build_read_index()` 定义的过滤矩阵进行计算。结果会写入：

- `obsm_X_pca`：每个细胞的 PC 坐标
- `varm_PCs`：每个基因的 PC loading
- `uns_pca_stats`：每个 PC 的 variance 和 variance_ratio

## 2. KMeans 聚类

```python
sap.tl.kmeans(atlas, n_clusters=10, fit_batches=1000)
```

这一步基于 PCA 结果，使用 MiniBatchKMeans 对细胞进行聚类。聚类结果写入 `obs.kmeans`。

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

`rank_result` 保存每个 cluster 的 marker gene 排名结果。你可以把它继续传给自动注释函数，也可以用 violin、dotplot 和 stacked violin 检查 marker 表达（详见 {doc}`../advanced/marker-gene-ranking`）。

## 6. 自动细胞类型注释

```python
summary_df, score_df = sap.tl.annotate_clusters(
    atlas,
    rank_result=rank_result,
    groupby="kmeans",
    reference_name="builtin_pbmc",
)
```

`annotate_clusters()` 会根据 marker gene 排名结果，自动为每个 cluster 分配细胞类型。结果写入：

- `obs.cell_type_auto`：注释结果
- `obs.cell_type_auto_confidence`：置信度（high/medium/low）
- `cluster_annotation_summary`：每个 cluster 的最终注释总表
- `cluster_annotation_scores`：每个 cluster 对每种细胞类型的打分明细

`reference_name="builtin_pbmc"` 使用内置的 PBMC marker reference，包含以下细胞类型：

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

## 7. 导出结果文件(可选)

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

完成聚类和注释后，可以阅读 {doc}`../advanced/qc-plots` 学习如何用 QC 图、PCA 图、UMAP 图和 marker gene 图判断分析结果是否合理。
