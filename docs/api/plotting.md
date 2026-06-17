# 画图：`sap.pl`

`sap.pl` 用于从 Atlas 数据库中读取结果并生成常用单细胞图，包括 QC 图、PCA 图、UMAP 图和 marker gene 图。

```{eval-rst}
.. currentmodule:: scatlaspy

.. autosummary::
   :toctree: generated/
   :nosignatures:

   pl.highest_expr_genes
   pl.violin_qc_metrics
   pl.scatter_qc_metrics
   pl.highly_variable_genes
   pl.pca
   pl.pca_variance_ratio
   pl.pca_variance_ratio_cumsum
   pl.kmeans_cluster_size
   pl.umap
   pl.rank_genes_groups
   pl.rank_genes_groups_volcano
   pl.rank_genes_groups_violin
   pl.violin
   pl.dotplot
   pl.stacked_violin
```

## 常见上色方式

- 按 `obs` 列上色，例如 `kmeans`、`sample`、`batch`。
- 按 gene name 上色，例如 `CST3`、`NKG7`、`MS4A1`。
- 混合查看 cluster 和基因，例如 `["kmeans", "CST3", "NKG7"]`。
