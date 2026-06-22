# scAtlasPy (要检查所有占位的东西，比如占位的链接是否指向最终的公开地址！)

**A scalable Python platform for atlas-scale single-cell omics analysis beyond in-memory limits**

scAtlasPy is a Python platform for analyzing cell atlases that are too large to
fit in memory. It extends familiar single-cell analysis workflows to
atlas-scale datasets, supporting data preprocessing, dimensionality reduction,
clustering, visualization, statistical analysis, and machine learning within a
unified environment. It also provides an extensible computational foundation
for developing, integrating, and applying new methods to large cell atlases.

To support these workflows, scAtlasPy organizes expression matrices, cell and
gene metadata, embeddings, and analysis results in a persistent `.sasql` atlas.
Analysis routines can retrieve selected cells, genes, data blocks, or
minibatches through query-based and high-throughput streaming interfaces, without
requiring the complete expression matrix to reside in memory. The same interfaces
are available to both built-in workflows and newly developed computational methods.

## Quick Start

Install scAtlasPy in the Python environment used for your analysis:

```bash
pip install scatlaspy
```

The example below creates a small atlas database and runs a compact
preprocessing, clustering, and visualization workflow:

```python
import scatlaspy as sap

atlas = sap.Atlas("pbmc.sasql", memory_limit=None)
atlas.load_h5ad("pbmc.h5ad", load_type="random")

# Quality control and preprocessing
sap.pp.calculate_qc_metrics(atlas)
sap.pp.filter_cells(atlas, min_genes=200)
sap.pp.filter_genes(atlas, min_cells=3)
sap.pp.normalize_total(atlas)
sap.pp.log1p(atlas)
sap.pp.highly_variable_genes(atlas, n_top_genes=2000)

# Prepare an analysis view using filtered cells and highly variable genes
atlas.build_read_index(
    cell_condition="filter_cells",
    gene_condition="filter_genes",
    use_hvg=True,
)

# Dimensionality reduction, clustering, and visualization
sap.tl.pca(atlas, n_components=50)
sap.tl.kmeans(atlas, n_clusters=8)
sap.tl.umap(atlas)
sap.pl.umap(atlas, color="kmeans")
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

:::{grid-item-card} Release Notes
:link: release-notes/index
:link-type: doc

Review new features, fixes, compatibility changes, and known limitations across
scAtlasPy releases.
:::

::::

## Built for Atlas-scale Computation

Each scAtlasPy atlas is maintained as a persistent `.sasql` file containing
expression data, cell and gene metadata, embeddings, and analysis results.
Rather than requiring the complete atlas to be materialized in memory,
computations can retrieve the cells, genes, or data blocks needed for a
particular analysis.

The atlas can be accessed through metadata queries, filtered analysis views, and
ordered or randomized expression-data streams. These interfaces support both
standard single-cell workflows and custom methods through the same Python
environment, allowing analysis code to operate directly on atlas-scale data
under practical memory constraints.

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
release-notes/index
citation
```

```{toctree}
:hidden:
:maxdepth: 1

faq/index
```
