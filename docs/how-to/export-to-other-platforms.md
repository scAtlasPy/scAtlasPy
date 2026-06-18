# 导出到其他平台

分析完成后，你可能希望把结果导出给 Scanpy、cellxgene 或其他下游工具使用。本页也涵盖从 Scanpy 导入数据的常用方法，覆盖完整的来回转换流程。

## 数据在 Scanpy 和 scAtlasPy 之间来回转换

scAtlasPy 和 Scanpy 的数据可以灵活互转，以下按方向分别说明。

### Scanpy → scAtlasPy（导入）

适合已经在 Scanpy 中完成读取或预处理，想要导入 scAtlasPy 继续分析的情况。

**方式 1：从内存中的 AnnData 导入**

```python
import scanpy as sc
import scatlaspy as sap

# 在 Scanpy 中读取数据
adata = sc.read_h5ad("input.h5ad")

# 如需预处理
# adata = adata[adata.obs["condition"] == "treated"].copy()

# 创建 scAtlasPy 数据库并导入
atlas = sap.Atlas("my_project", "./data")
atlas.load_anndata(adata)

# 此时数据已写入 my_project.sasql，describe() 确认
print(atlas.describe())
```

**方式 2：直接从 h5ad 文件导入**

适合数据在磁盘上、无需在 Scanpy 中预处理的情况：

```python
import scatlaspy as sap

atlas = sap.Atlas("my_project", "./data")
atlas.execute_sql("SET memory_limit = '80GB'")

# 单个文件
atlas.load_h5ad("large.h5ad", load_type="order", store_type="count")

# 或多个文件合并
atlas.load_h5ad(
    ["sample1.h5ad", "sample2.h5ad", "sample3.h5ad"],
    load_type="list_random",
    store_type="count",
)
```

## scAtlasPy → Scanpy（导出）

### 导出为 h5ad 文件

把整个 Atlas 数据库导出为一个 h5ad 文件：

```python
import scatlaspy as sap

atlas = sap.Atlas("run01", "./data")

# 导出为 h5ad 文件
atlas.write_h5ad("out.h5ad")

atlas.close()
```

`write_h5ad()` 当前主路径默认把 `X_HyS_data.data` 写入 h5ad 的 `X`。如需指定其他表达字段（如 `data_log1p`），请使用下面的 AnnData 导出方式。

### 导出为 AnnData 对象

AnnData 导出方式更灵活，可以指定细胞子集、表达字段和是否附带降维结果。

**导出通过过滤的细胞**

```python
import scatlaspy as sap

atlas = sap.Atlas("run01", "./data")

# 获取 obs 表
obs_df = atlas.get_obs_df()

# 筛选通过过滤的细胞
atlas_cell_ids = obs_df[obs_df["filter_cells"].notna()]["atlas_cell_id"].tolist()

# 导出为 AnnData
adata = atlas.get_anndata(
    atlas_cell_ids,
    use_data="data_log1p",
    include_obsm=True,   # 同时导出 obsm_X_pca、obsm_X_umap 等降维结果
    include_varm=True,   # 同时导出 varm_PCs 等基因级降维结果
)

atlas.close()
```

**导出后继续在 Scanpy 中分析**

```python
# 回到 Scanpy 继续分析
sc.tl.leiden(adata, resolution=0.5)
sc.tl.rank_genes_groups(adata, groupby="leiden")
sc.pl.umap(adata, color="leiden")
adata.write_h5ad("filtered_subset.h5ad")
```

### 导出细胞信息为表格

```python
# 导出 obs 表为 pandas DataFrame
obs_df = atlas.get_obs_df()
print(obs_df.head())

# 只导出指定列
obs_df = atlas.get_obs_df(columns=["kmeans", "cell_type_manual", "cell_total_counts"])
```

### `use_data` 可选值

| `use_data` | 含义 |
|---|---|
| `"data"` | 原始表达值 |
| `"data_normalize"` | 总量标准化后的值 |
| `"data_log1p"` | 标准化 + log1p，适合画图和 marker gene 分析 |
| `"data_scale"` | z-score 标准化，适合 PCA 或模型输入 |
| `"data_sqrt"` | sqrt 变换，替代 log 变换的选项 |

## 常见转换路径总结

| 你的场景 | 路径 |
|---|---|
| 已有 h5ad，想用 scAtlasPy 处理 | `atlas.load_h5ad(path)` 直接导入 |
| 已有内存中的 AnnData，想导入 scAtlasPy | `atlas.load_anndata(adata)` |
| scAtlasPy 处理完后想继续在 Scanpy 分析 | `atlas.get_anndata(ids)` → Scanpy |
| scAtlasPy 处理完后想导出 h5ad 分享 | `atlas.write_h5ad(path)` |
| 多批次数据合并分析 | `atlas.load_h5ad([paths], load_type="list_random")` → 统一处理 → 导出 |

## 导出后在其他平台的典型用法

```python
import scanpy as sc

# 读取 scAtlasPy 导出的 h5ad
adata = sc.read_h5ad("filtered_log1p.h5ad")

# 在 Scanpy 中继续分析
sc.tl.leiden(adata, resolution=0.5)
sc.tl.rank_genes_groups(adata, groupby="leiden")
sc.pl.umap(adata, color="leiden")

# 也可以导入 cellxgene
# cellxgene launch filtered_log1p.h5ad
```

## 需要注意

- scAtlasPy 的 `filter_cells()` / `filter_genes()` 是标记过滤结果，不删除数据。导出时会自动只导出通过过滤的细胞。
- 表达字段（data / data_log1p / data_scale）在 scAtlasPy 中共存于 X_HyS_data 表，导出时通过 `use_data` 参数选择。
- `get_anndata()` 导出子集时需注意内存：如果导出细胞数 × 基因数很大（如 >50 万细胞 × >2000 基因），确保内存充足。
