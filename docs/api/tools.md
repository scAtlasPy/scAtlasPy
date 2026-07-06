# Analysis Tools: `sap.tl`

## Overview

`sap.tl` provides downstream analysis functions for dimensionality reduction,
clustering, UMAP, marker-gene ranking, and manual cell-type annotation.

Most functions operate on data stored in the Atlas and persist their outputs as
embeddings, cluster labels, marker statistics, or annotation columns.

For methods that read expression values in batches, first define the cells,
genes, and expression representation with `atlas.build_read_index()`.

## Dimensionality Reduction

```{eval-rst}
.. currentmodule:: scatlaspy

.. autosummary::
   :toctree: generated/
   :nosignatures:

   tl.pca
```

`pca()` calculates a low-dimensional representation of the active read-index
view.

Common stored outputs include:

- cell-level PCA coordinates in `obsm_X_pca`;
- gene loadings in `varm_PCs`;
- explained-variance statistics used by PCA plotting functions.

Example:

```python
atlas.build_read_index(
    cell_condition="filter_cells",
    gene_condition="filter_genes",
    use_hvg=True,
    use_data="data_log1p",
)

sap.tl.pca(
    atlas,
    n_components=50,
)
```

```{important}
The active read index determines which cells, genes, feature order, and
expression field are supplied to PCA.
```

## Clustering

```{eval-rst}
.. currentmodule:: scatlaspy

.. autosummary::
   :toctree: generated/
   :nosignatures:

   tl.kmeans
```

`kmeans()` clusters cells using the stored PCA representation and writes the
resulting labels to `obs`.

The default output is commonly stored in:

```text
obs.kmeans
```

Example:

```python
sap.tl.kmeans(
    atlas,
    n_clusters=10,
)
```

## UMAP

```{eval-rst}
.. currentmodule:: scatlaspy

.. autosummary::
   :toctree: generated/
   :nosignatures:

   tl.umap
```

`umap()` calculates a two-dimensional visualization from the available
low-dimensional representation.

The resulting cell coordinates are commonly stored in:

```text
obsm_X_umap
```

Example:

```python
sap.tl.umap(atlas)
```

The stored coordinates can be visualized with:

```python
sap.pl.umap(
    atlas,
    color="kmeans",
)
```

## Marker-gene Ranking

```{eval-rst}
.. currentmodule:: scatlaspy

.. autosummary::
   :toctree: generated/
   :nosignatures:

   tl.rank_genes_groups
```

`rank_genes_groups()` identifies genes associated with groups defined by an
`obs` column, such as clusters or cell-type annotations.

Example:

```python
sap.tl.rank_genes_groups(
    atlas,
    groupby="kmeans",
    n_genes=20,
)
```

The ranked statistics are stored in the Atlas for later querying and
visualization.

```{note}
Marker analysis should generally use an interpretable expression
representation, such as log-normalized expression, rather than scaled values.
Consult the function reference for the source expression field used by the
analysis.
```

## Manual Annotation

```{eval-rst}
.. currentmodule:: scatlaspy

.. autosummary::
   :toctree: generated/
   :nosignatures:

   tl.manual_annotate_clusters
```

`manual_annotate_clusters()` maps cluster identifiers to user-defined cell-type
labels and writes the annotations to an `obs` column.

Example:

```python
cluster_annotations = {
    "0": "T cell",
    "1": "B cell",
    "2": "Myeloid cell",
}

sap.tl.manual_annotate_clusters(
    atlas,
    cluster_to_cell_type=cluster_annotations,
    groupby="kmeans",
    obs_col="cell_type_manual",
)
```

The exact parameter names and accepted annotation formats are documented in the
function reference above.

## Typical Analysis Order

A common downstream workflow is:

```python
atlas.build_read_index(
    cell_condition="filter_cells",
    gene_condition="filter_genes",
    use_hvg=True,
    use_data="data_log1p",
)

sap.tl.pca(
    atlas,
    n_components=50,
)

sap.tl.kmeans(
    atlas,
    n_clusters=10,
)

sap.tl.umap(atlas)

sap.tl.rank_genes_groups(
    atlas,
    groupby="kmeans",
    n_genes=20,
)
```

After inspecting cluster structure and marker genes, clusters can be assigned
manual annotations:

```python
sap.tl.manual_annotate_clusters(
    atlas,
    cluster_to_cell_type=cluster_annotations,
    groupby="kmeans",
    obs_col="cell_type_manual",
)
```

## Common Stored Outputs

| Analysis | Common output |
|---|---|
| PCA | `obsm_X_pca`, `varm_PCs`, and explained-variance statistics |
| KMeans | Cluster labels in an `obs` column, commonly `kmeans` |
| UMAP | Two-dimensional coordinates in `obsm_X_umap` |
| Marker ranking | Ranked gene statistics stored in Atlas result tables |
| Manual annotation | User-defined labels in an `obs` annotation column |

```{note}
Output names, replacement behavior, prerequisites, and Python return values are
documented in the individual function entries.
```

## Related Documentation

- {doc}`preprocessing` documents the functions used to prepare expression data
  for downstream analysis.
- {doc}`atlas` documents `Atlas.build_read_index()` and the active expression
  view.
- {doc}`plotting` documents visualization of PCA, clusters, UMAP, markers, and
  annotations.
- {doc}`../tutorials/basic/index` provides a guided analysis workflow.
