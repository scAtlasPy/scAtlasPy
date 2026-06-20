# 重连数据库继续使用

本页说明如何重新连接已有的 `.sasql` 数据库，继续分析或导出结果。

## 连接已有数据库

`sap.Atlas(name, path)` 如果发现 `{path}/{name}.sasql` 已存在，会直接连接已有数据库，而不是重新创建。

```python
import scatlaspy as sap

atlas = sap.Atlas("pbmc_demo", "./data")

# 确认数据库存在
print(atlas.exists())
```

## 查看数据库状态

```python
# 一行查看数据库概要：文件路径、表列表、细胞数、基因数
print(atlas.describe())
```

输出示例：

```text
file_name    : ./data/pbmc_demo.sasql
tables      : 6
table names : obs, var, X_HyS_indptr, X_HyS_data, obsm_X_pca, obsm_X_umap
n_cells     : 3,000
n_genes     : 32,738
```

从 `table names` 就能直接判断当前分析进度：
- 有 `obsm_X_pca` → PCA 已完成
- 有 `obsm_X_umap` → UMAP 已完成
- obs 中有 `kmeans` 列 → KMeans 已完成

如需查看某个表的具体列名：

```python
print(atlas.query("PRAGMA table_info(obs)"))
```

## 只读模式 vs 读写模式

```python
# 只读检查（不会修改数据库）
atlas.connect("r")

# 继续写入分析结果（默认）
atlas.connect("r+")
```

## 从哪里继续分析

| 当前状态 | 下一步 |
|---|---|
| 只导入了数据，无 QC | 从 {doc}`../basic/basic_exploration` 开始 |
| 已完成 QC 和标准化，无 PCA | 运行 `build_read_index()` → `sap.tl.pca()` |
| 已完成 PCA，无聚类 | 运行 `sap.tl.kmeans()` → `sap.tl.umap()` |
| 已完成全部分析 | 直接画图 {doc}`plot-parameter-guide` 或导出 {doc}`../../how-to/export-to-other-platforms` |

## 在已有数据库基础上补充分析

```python
atlas = sap.Atlas("pbmc_demo", "./data")
atlas.execute_sql("SET memory_limit = '8GB'")

# 如果还没有 build_read_index
atlas.build_read_index(
    cell_condition="filter_cells",
    gene_condition="filter_genes",
    use_hvg=True,
    use_data="data_log1p",
)

# 补充 PCA
sap.tl.pca(atlas, n_components=50, fit_batches=1000)

# 补充 UMAP
sap.tl.umap(atlas, fit_sample_n=50000)

# 画图
sap.pl.umap(atlas, color="kmeans", sample_n=50000)
```

## 完成后关闭

```python
atlas.close()
```

## 下一步

- 继续画图分析：{doc}`plot-parameter-guide`
- 导出结果到其他平台：{doc}`../../how-to/export-to-other-platforms`
- 用 SQL 查询结果：{doc}`sql-query-cases`
