# scAtlasPy

**A scalable Python platform for atlas-scale single-cell omics analysis beyond in-memory limits**

scAtlasPy is a Python platform for analyzing cell atlases that are too large to
fit in memory. It extends familiar single-cell analysis workflows to
atlas-scale datasets, supporting data preprocessing, dimensionality reduction,
clustering, visualization, statistical analysis, and machine learning within a
unified environment. It also provides an extensible computational foundation
for developing, integrating, and applying new methods to large cell atlases.

## Quick Start

Install scAtlasPy in the Python environment used for your analysis:

```bash
pip install scatlaspy
```

See the {doc}`installation` page for complete installation options, including
PyTorch setup for the UMAP and graph-clustering steps used below.

The example below creates a small atlas database and runs a compact
preprocessing, clustering, and visualization workflow:

```python
import scatlaspy as sap

# Create a persistent atlas database on disk.
atlas = sap.Atlas("pbmc.sasql")

# Import data from an h5ad file into the atlas.
atlas.load_h5ad("pbmc.h5ad", load_type="random")

# Quality control and preprocessing
sap.pp.calculate_qc_metrics(atlas)
sap.pp.filter_cells(atlas, min_genes=200)
sap.pp.filter_genes(atlas, min_cells=3)
sap.pp.normalize_and_log1p(atlas)
sap.pp.highly_variable_genes(atlas, n_top_genes=2000)
# Use mode="center_only" here if PCA should preserve gene-level variance differences.
sap.pp.scale(atlas)

# Prepare the PCA analysis view: filtered cells, filtered HVGs, and scaled data.
atlas.build_read_index(use_hvg=True, use_data="data_scale")

# Dimensionality reduction, clustering, and visualization
sap.tl.pca(atlas)
sap.tl.umap(atlas)
sap.tl.graph_clustering(atlas)
sap.pl.umap(atlas, color="scatlas_cluster")
```

See the {doc}`tutorials/index` for complete workflows and explanations of each
analysis step.

## What scAtlasPy Enables

scAtlasPy supports both established single-cell analysis workflows and the
development of computational methods for datasets that exceed available memory.

::::{div} feature-list

:::{div} feature-item
**Exploratory single-cell analysis workflows**

Run preprocessing, dimensionality reduction, clustering, visualization,
statistical analysis, and downstream exploration through a unified Python
interface.
:::

:::{div} feature-item
**Computation beyond memory limits**

Access selected cells, genes, data blocks, and minibatches without loading the
complete expression matrix into memory.
:::

:::{div} feature-item
**Atlas-scale method development**

Develop, integrate, and apply statistical and machine-learning methods using
sequential, randomized, or query-defined access to large expression matrices.
:::

::::

## Explore the Documentation

::::{grid} 1 2 3 3
:gutter: 2

:::{grid-item-card} Installation
:link: installation
:link-type: doc

Install scAtlasPy and verify that the package is ready to use in your Python
environment.
:::

:::{grid-item-card} Tutorials
:link: tutorials/index
:link-type: doc

Follow complete workflows for data import, preprocessing, dimensionality
reduction, clustering, marker analysis, visualization, and atlas-scale
computation.
:::

:::{grid-item-card} Cross-platform Workflows
:link: cross-platform-workflows/index
:link-type: doc

Move workflows between scAtlasPy and other single-cell platforms, including
external methods, result write-back, and third-party integrations.
:::

:::{grid-item-card} API Reference
:link: api/index
:link-type: doc

Look up public classes and functions in `sap.io`, `sap.pp`, `sap.tl`, and
`sap.pl`.
:::

:::{grid-item-card} FAQ
:link: faq/index
:link-type: doc

Find concise explanations for common observations, parameter choices, and
workflow differences from in-memory single-cell tools.
:::

:::{grid-item-card} Architecture and Development
:link: architecture-and-development/index
:link-type: doc

Understand the atlas data model, streaming architecture, performance behavior,
documentation workflow, and current implementation limits.
:::

::::

## Built for Atlas-scale Computation

scAtlasPy is built around a persistent `.sasql` atlas that keeps expression
matrices, metadata, embeddings, and analysis results together on disk. Instead
of materializing the full atlas in memory, workflows operate through filtered
analysis views, metadata queries, and ordered or randomized data streams.

This design gives built-in analyses and custom methods the same controlled
access pattern: read only the cells, genes, or minibatches needed for the current
step, then write results back to the atlas for later exploration, visualization,
or export.

See {doc}`architecture-and-development/data-model` for the atlas representation
and {doc}`architecture-and-development/index` for architecture and development
notes.

```{toctree}
:hidden:
:maxdepth: 2

installation
tutorials/index
cross-platform-workflows/index
api/index
architecture-and-development/index
community-and-contributions
release-notes
citation
```

```{toctree}
:hidden:
:maxdepth: 1

faq/index
```
