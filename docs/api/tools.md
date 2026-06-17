# 分析工具：`sap.tl`

`sap.tl` 包含 PCA、KMeans、UMAP、差异表达和自动注释等分析函数。普通分析流程中，PCA 前通常需要先运行 `atlas.build_read_index()`。

```{eval-rst}
.. currentmodule:: scatlaspy

.. autosummary::
   :toctree: generated/
   :nosignatures:

   tl.pca
   tl.kmeans
   tl.umap
   tl.rank_genes_groups
   tl.annotate_clusters
```

