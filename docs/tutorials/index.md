# Tutorials

These tutorials introduce scAtlasPy through executable analysis workflows. They
show how an atlas is created, processed, queried, visualized, and extended with
custom computational methods.

The tutorials are organized into two learning paths. **Basic Tutorials** follow
a standard single-cell analysis workflow from data preparation to cell-type
annotation. **Advanced Tutorials** focus on atlas-scale access patterns,
streaming computation, SQL queries, and custom method
development.

If you are new to scAtlasPy, start with the Basic path and read the pages in
order. If you already have an atlas database or want to develop a specialized
analysis method, you can move directly to the Advanced path.

::::{grid} 1 2 2 2
:gutter: 2

:::{grid-item-card} Basic Tutorials
:link: basic/index
:link-type: doc

Build a complete analysis workflow, including data import, quality control,
preprocessing, dimensionality reduction, clustering, visualization,
marker-gene analysis, and cell-type annotation.
:::

:::{grid-item-card} Advanced Tutorials
:link: advanced/index
:link-type: doc

Work with atlas-scale datasets through incremental data access, streaming
statistics, SQL queries, custom algorithms, and model training workflows.
:::

::::

## Choosing a Tutorial

Start with the Basic path if you want to see how scAtlasPy fits into a
complete single-cell analysis. It is the best entry point for learning the
main workflow: create an Atlas, import data, run preprocessing, build an
analysis view, compute embeddings and clusters, inspect markers, and assign
cell-type labels.

Move to the Advanced path when the basic workflow is already familiar, or when
you need a focused capability from scAtlasPy's atlas-scale data layer. These
tutorials show how to reopen an existing `.sasql` Atlas, query stored tables
with SQL, stream expression data in bounded-memory batches, and use Atlas data
access inside custom statistical or machine-learning methods.

```{tip}
Tutorials are designed as guided learning workflows. For instructions on moving
data or workflows between scAtlasPy and other single-cell platforms, see the
{doc}`../cross-platform-workflows/index`.
```

```{toctree}
:hidden:
:maxdepth: 2

basic/index
advanced/index
```
