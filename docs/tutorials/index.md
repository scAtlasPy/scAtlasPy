# Tutorials

These tutorials introduce scAtlasPy through executable analysis workflows. They
show how an atlas is created, processed, queried, visualized, and extended with
custom computational methods.

The tutorials are organized into two learning paths. **Basic Tutorials** follow
a standard single-cell analysis workflow from data preparation to cell-type
annotation. **Advanced Tutorials** focus on atlas-scale access patterns,
streaming computation, visualization, SQL queries, and custom method
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
preprocessing, dimensionality reduction, clustering, marker-gene analysis, and
cell-type annotation.
:::

:::{grid-item-card} Advanced Tutorials
:link: advanced/index
:link-type: doc

Work with atlas-scale datasets through incremental data access, streaming
statistics, plotting utilities, SQL queries, custom algorithms, and model
training workflows.
:::

::::

## Choosing a Tutorial

Use the Basic path when you want to learn the main scAtlasPy workflow on a
small or moderate dataset. These pages are written as a guided sequence and are
best read from top to bottom.

Use the Advanced path when you need a specific capability: reconnecting an
existing `.sasql` atlas, visualizing results, querying the database, streaming
minibatches, or integrating a custom method that cannot load the full expression
matrix into memory.

```{tip}
Tutorials are designed as guided learning workflows. For instructions on a
specific task—such as reconnecting an existing atlas, querying results with
SQL, exporting data, or migrating an existing workflow—see the
{doc}`../how-to/index`.
```

```{toctree}
:hidden:
:maxdepth: 2

basic/index
advanced/index
```
