# 数据模型

scAtlasPy 把单细胞数据保存在 DuckDB 数据库中，文件后缀为 `.sasql`。这一页解释数据库表和 AnnData 结构之间的对应关系，适合算法开发者和贡献者阅读。

## 和 AnnData 的对应关系

| AnnData 中的位置 | scAtlasPy 中的表 |
|---|---|
| `adata.obs` | `obs` |
| `adata.var` | `var` |
| `adata.X` | `X_HyS_indptr` + `X_HyS_data` |
| `adata.obsm["X_pca"]` | `obsm_X_pca` |
| `adata.obsm["X_umap"]` | `obsm_X_umap` |
| `adata.varm["PCs"]` | `varm_PCs` |
| `adata.uns["pca"]` | `uns_pca_stats` |

## 细胞和基因标识

`obs` 表保存细胞信息，通常包含：

- `atlas_cell_id`：平台内部使用的细胞 ID。
- `atlas_cell_name`：原始细胞名。
- 用户自己的 metadata 列，例如样本、批次、分组、细胞类型。

`var` 表保存基因信息，通常包含：

- `atlas_gene_id`：平台内部使用的基因 ID。
- `atlas_gene_name`：原始基因名。
- 用户自己的 gene metadata 列。

## 表达矩阵

`X_HyS_data` 保存稀疏表达矩阵中的非零表达项。预处理函数会在这个表中增加新的表达字段：

- `data`：导入时保存的表达值。
- `data_normalize`：标准化后的表达值。
- `data_log1p`：标准化并 log 转换后的表达值。
- `data_scale`：scale 后的表达值。
- `data_sqrt`：平方根转换后的表达值。

开发自定义算法时，通常需要先确定使用哪个表达字段。例如 marker gene 可视化常用 `data_log1p`，PCA 或模型训练常用 `data_scale`。

## 为什么需要 `zero_scale_transform`

稀疏矩阵不保存 0 表达值。但基因做 scale 后，原来的 0 表达值会变成：

```text
(0 - gene_mean) / gene_std
```

这个值通常不是 0。scAtlasPy 把它保存在 `var.zero_scale_transform`，这样在重建 dense batch 时，可以正确填充那些原本没有存储的 0 表达位置。

