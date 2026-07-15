# Plotting: `sap.pl`

## Overview

`sap.pl` provides plotting functions for inspecting data and analysis results
stored in an Atlas.

These functions generate common single-cell visualizations for:

- quality-control assessment;
- highly variable gene selection;
- PCA diagnostics;
- clustering and UMAP inspection;
- marker-gene analysis;
- comparison of gene expression across cell groups.

```{note}
Plotting functions visualize results already stored in the Atlas. They do not
run the corresponding preprocessing or analysis step.

For example, PCA and UMAP results must be calculated before calling their
plotting functions.
```

For large atlases, use sampling or select a focused cell population when
appropriate so that figures remain readable and practical to render.

## Quality Control

```{eval-rst}
.. currentmodule:: scatlaspy

.. autosummary::
   :toctree: generated/
   :nosignatures:

   pl.highest_expr_genes
   pl.violin_qc_metrics
   pl.scatter_qc_metrics
```

These functions visualize highly expressed genes and cell-level
quality-control metrics.

```python
sap.pl.highest_expr_genes(atlas, n_top=20)
sap.pl.scatter_qc_metrics(atlas, x="total_counts", y="n_genes_by_counts")
```

## Highly Variable Genes

```{eval-rst}
.. currentmodule:: scatlaspy

.. autosummary::
   :toctree: generated/
   :nosignatures:

   pl.highly_variable_genes
```

Use this function to inspect the relationship between gene expression,
dispersion, and highly variable gene selection.

```python
sap.pl.highly_variable_genes(atlas)
```

## PCA

```{eval-rst}
.. currentmodule:: scatlaspy

.. autosummary::
   :toctree: generated/
   :nosignatures:

   pl.pca
   pl.pca_loadings
   pl.pca_variance_ratio
   pl.pca_variance_ratio_cumsum
```

These functions visualize PCA coordinates, component loadings, and explained
variance.

```python
sap.pl.pca(atlas, color="sample")
sap.pl.pca_loadings(atlas, components=[1, 2])
sap.pl.pca_variance_ratio(atlas)
```

If `point_size=None`, `sap.pl.pca()` estimates a default point size from the
number of plotted cells. Pass `point_size` explicitly when you want a fixed
marker size across figures.

## Clustering and UMAP

```{eval-rst}
.. currentmodule:: scatlaspy

.. autosummary::
   :toctree: generated/
   :nosignatures:

   pl.kmeans_cluster_size
   pl.umap
```

Use these functions to inspect cluster sizes and visualize stored UMAP
coordinates.

```python
sap.pl.kmeans_cluster_size(atlas)
```

UMAP plots can commonly be colored by:

- cell-level metadata, such as `kmeans`, `sample`, `batch`, or annotation
  columns;
- gene expression, such as `CST3`, `NKG7`, or `MS4A1`;
- several metadata columns or genes in one call.

For example:

```python
sap.pl.umap(
    atlas,
    color=[
        "kmeans",
        "sample",
        "NKG7",
    ],
)
```

If `point_size=None`, `sap.pl.umap()` estimates a default point size from the
number of plotted cells. Pass `point_size` explicitly when you want a fixed
marker size across figures.

```{note}
UMAP is primarily a visualization of local neighborhood structure. Apparent
separation in a UMAP plot should not be treated as sufficient evidence that a
clustering or integration result is correct.
```

## Marker-gene Results

```{eval-rst}
.. currentmodule:: scatlaspy

.. autosummary::
   :toctree: generated/
   :nosignatures:

   pl.rank_genes_groups
   pl.rank_genes_groups_volcano
   pl.rank_genes_groups_violin
```

These functions visualize marker-gene rankings, differential-expression
statistics, and expression distributions for ranked genes.

The corresponding marker analysis must be completed before these functions are
used.

```python
sap.pl.rank_genes_groups(atlas, group="0", n_genes=10)
sap.pl.rank_genes_groups_volcano(atlas, group="0")
```

## Gene-expression Comparison

```{eval-rst}
.. currentmodule:: scatlaspy

.. autosummary::
   :toctree: generated/
   :nosignatures:

   pl.violin
   pl.dotplot
   pl.stacked_violin
```

Use these functions to compare selected genes across clusters, annotations,
conditions, or other cell groups.

```python
sap.pl.violin(atlas, keys=["NKG7", "CST3"], groupby="kmeans")
sap.pl.dotplot(atlas, var_names=["NKG7", "CST3", "MS4A1"], groupby="kmeans")
```

For routine marker-expression visualization, a log-normalized expression field
such as `data_log1p` is generally more interpretable than scaled values.
Scaled expression is centered and standardized and therefore does not preserve
the original expression magnitude.

## Large-atlas Visualization

Plotting every cell in a very large Atlas may be slow and can produce
overplotted figures. Where supported, use sampling parameters to limit the
number of displayed cells:

```python
sap.pl.umap(
    atlas,
    color="cell_type_manual",
    sample_n=100000,
)
```

With `sample_n=None`, UMAP metadata plots use the streaming plotting path and
avoid materializing the full plot table at once.

```{warning}
A random sample may underrepresent rare cell populations. Use a focused subset
or group-aware sampling when rare populations are central to the analysis.
```

Consult the individual function entries for supported sampling, expression
field, figure size, and output-file parameters.

## Related Documentation

- {doc}`../tutorials/advanced/visualize-analysis-results` provides a guided
  workflow for inspecting stored analysis results.
- {doc}`preprocessing` documents the preprocessing functions that generate
  quality-control and highly variable gene results.
- {doc}`tools` documents the analysis functions that generate PCA, clustering,
  UMAP, and marker-gene results.
