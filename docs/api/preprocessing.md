# 预处理：`sap.pp`

`sap.pp` 包含质控、过滤、标准化、高变基因和 scale 等函数。它们通常把结果写入数据库字段，而不是直接返回一个新对象。

```{eval-rst}
.. currentmodule:: scatlaspy

.. autosummary::
   :toctree: generated/
   :nosignatures:

   pp.calculate_qc_metrics
   pp.filter_cells
   pp.filter_genes
   pp.calculate_cell_total_counts
   pp.calculate_gene_total_counts
   pp.normalize_total
   pp.normalize_total_scale_factor
   pp.log1p
   pp.expm1
   pp.normalize_and_log1p
   pp.highly_variable_genes
   pp.scale
   pp.sqrt
```

## 常见写入结果

- 质控：`obs.cell_total_counts`、`obs.n_genes_by_counts`、`var.gene_total_counts`、`var.n_cells_by_counts`。
- 过滤：`obs.filter_cells`、`var.filter_genes`。
- 标准化和转换：`obs.scale_factor`、`X_HyS_data.data_log1p`、`X_HyS_data.data_scale`。
- 高变基因：`var.highly_variable_genes` 和 `var.hvg_*`。

