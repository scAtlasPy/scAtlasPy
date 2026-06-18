# 从其他平台迁移

本页以 Scanpy 为例说明迁移方法，核心差别是：Scanpy 主要操作内存中的 `AnnData` 对象；scAtlasPy 主要操作保存在磁盘上的 Atlas 数据库。

```{note}
Scanpy 用 `sc.read_h5ad()` 一步把数据读入内存 AnnData；scAtlasPy 需要先创建 Atlas 数据库再导入，
数据保存在 `.sasql` 文件中。具体导入方式的选择见 {doc}`../tutorials/basic/preparing-data`。
```

scAtlasPy 适用于任意规模数据的导入、质控、标准化、降维和聚类。
数据保存在 `.sasql` 数据库中，分析中间结果持久化，随时可以继续或导出。

## 常见步骤对照

| scAtlasPy 写法 | Scanpy 写法 |
|---|---|
| `sap.pp.calculate_qc_metrics(atlas, qc_vars=...)` | `sc.pp.calculate_qc_metrics(adata, qc_vars=...)` |
| `sap.pp.filter_cells(atlas, min_genes=200)` | `sc.pp.filter_cells(adata, min_genes=200)` |
| `sap.pp.filter_genes(atlas, min_cells=3)` | `sc.pp.filter_genes(adata, min_cells=3)` |
| `sap.pp.normalize_and_log1p(atlas)` | `sc.pp.normalize_total(adata); sc.pp.log1p(adata)` |
| `sap.pp.highly_variable_genes(atlas)` | `sc.pp.highly_variable_genes(adata)` |
| `sap.pp.scale(atlas)` | `sc.pp.scale(adata)` |
| `atlas.build_read_index(); sap.tl.pca(atlas)` | `sc.tl.pca(adata)` |
| `sap.tl.umap(atlas)` | `sc.tl.umap(adata)` |
| `sap.pl.umap(atlas, color=...)` | `sc.pl.umap(adata, color=...)` |
| `atlas.write_h5ad("out.h5ad")` | `adata.write_h5ad("out.h5ad")` |



## 完整迁移示例

以下分别展示用 Scanpy 和 scAtlasPy 完成同一个 PBMC 分析流程。假设输入文件为 `pbmc3k.h5ad`。

### Scanpy 写法

```python
import scanpy as sc

# 1. 读入数据
adata = sc.read_h5ad("pbmc3k.h5ad")

# 2. 质控
adata.var["mt"] = adata.var_names.str.startswith("MT-")
adata.var["ribo"] = adata.var_names.str.startswith(("RPS", "RPL"))
sc.pp.calculate_qc_metrics(adata, qc_vars=["mt", "ribo"], inplace=True)

# 3. 过滤
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)

# 4. 标准化、log1p、高变基因、scale
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=2000)
adata = adata[:, adata.var["highly_variable"]].copy()
sc.pp.scale(adata)

# 5. 降维与聚类
sc.tl.pca(adata, n_comps=50)
sc.pp.neighbors(adata)          # scAtlasPy 的 UMAP 基于 PCA，不需要此步
sc.tl.umap(adata)

# 6. 聚类（Leiden）
sc.tl.leiden(adata, resolution=0.5)

# 7. Marker gene 排名
sc.tl.rank_genes_groups(adata, groupby="leiden", method="t-test")

# 8. 可视化
sc.pl.umap(adata, color="leiden")
sc.pl.rank_genes_groups(adata, n_genes=10)

# 9. 保存
adata.write_h5ad("scanpy_result.h5ad")
```

### scAtlasPy 写法

```python
import scatlaspy as sap

# 1. 创建数据库 + 导入数据
atlas = sap.Atlas("pbmc_scatlas", "./data")
atlas.execute_sql("SET memory_limit = '8GB'")
atlas.load_h5ad("pbmc3k.h5ad", load_type="order", store_type="count")

# 2. 质控
sap.pp.calculate_qc_metrics(
    atlas,
    qc_vars={"mt": "MT-", "ribo": "^(RPS|RPL)"},
)

# 3. 过滤（标记而非删除，可随时改阈值重跑）
sap.pp.filter_cells(atlas, min_genes=200, min_counts=500)
sap.pp.filter_genes(atlas, min_cells=3)

# 4. 标准化、log1p、高变基因、scale
# 结果作为新字段写入 X_HyS_data 表，不覆盖原始 data
sap.pp.normalize_and_log1p(atlas, target_sum=1e4)
sap.pp.highly_variable_genes(atlas, n_top_genes=2000)
sap.pp.scale(atlas)

# 5. 构建过滤矩阵索引（Scanpy 无此步骤）
# 显式指定后续降维使用哪些细胞、基因和表达字段
atlas.build_read_index(
    cell_condition="filter_cells",
    gene_condition="filter_genes",
    use_hvg=True,
    use_data="data_scale",
)

# 6. 降维与 KMeans 聚类
sap.tl.pca(atlas, n_components=50, fit_batches=1000)
sap.tl.kmeans(atlas, n_clusters=10, fit_batches=1000)
sap.tl.umap(atlas, fit_sample_n=50000)

# 7. Marker gene 排名 + 手动注释
rank_result = sap.tl.rank_genes_groups(
    atlas, groupby="kmeans", n_genes=10
)
cluster_to_cell_type = {
    "0": "CD4 T cells",
    "1": "CD14+ Monocytes",
    "2": "B cells",
}
sap.tl.manual_annotate_clusters(
    atlas,
    cluster_to_cell_type,
    groupby="kmeans",
    obs_col="cell_type_manual",
)

# 8. 可视化
sap.pl.umap(atlas, color="kmeans", sample_n=50000)
sap.pl.umap(atlas, color="cell_type_manual", sample_n=50000)

# 9. 关闭连接
atlas.close()
```

### 关键差异速记

| 差异点 | scAtlasPy | Scanpy |
|---|---|---|
| 数据存储 | 磁盘 `.sasql` 数据库 | 内存 AnnData |
| 过滤 | 写入标记列，可重新调整 | 直接删除细胞/基因 |
| 表达字段 | 各步结果作为新字段共存于 `X_HyS_data` | `adata.X` 被反复覆盖 |
| PCA 前置 | 需显式运行 `build_read_index()` | 直接用 `adata.X` |
| 聚类 | `sap.tl.kmeans`（MiniBatchKMeans） | `sc.tl.leiden`（图聚类） |

## 需要特别注意的差异

`filter_cells()` 和 `filter_genes()` 是"标记过滤结果"，不是立即删除数据。这样做的好处是你可以重新调整阈值，不必重新导入原始数据。

`normalize_and_log1p()` 会写入 `X_HyS_data.data_log1p`。画 marker gene 表达图时，通常使用这个字段。

`scale()` 会写入 `X_HyS_data.data_scale`。PCA 默认更适合使用这个字段。

`build_read_index()` 是 PCA 的前置步骤，它指定后续降维使用哪些细胞、哪些基因和哪个表达字段。默认使用 `filter_cells` 通过、`filter_genes` 通过、属于高变基因的 `data_scale` 表达值。这一步在 Scanpy 中没有直接等价物——Scanpy 默认使用 `adata.X` 的全部数据做 PCA。

`sap.tl.umap()` 当前使用 PCA 结果计算 UMAP 坐标，不完全等同于 Scanpy 中基于 neighbors graph 的完整流程。常规可视化探索可以使用；如果你的分析强依赖 Scanpy graph 和 Leiden/Louvain，可以导出回 Scanpy 继续处理。
