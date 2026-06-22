# Atlas Object

## Overview

`Atlas` is the central object in scAtlasPy. It represents a persistent `.sasql`
database containing expression data, cell and gene metadata, analysis views,
embeddings, cluster labels, marker statistics, annotations, and other results
created during an analysis.

Most workflows begin by creating or opening an `Atlas`, importing data, and
passing the object to functions in `sap.pp`, `sap.tl`, and `sap.pl`.

```python
import scatlaspy as sap

# Create or open ./data/run01.sasql
atlas = sap.Atlas(
    "run01",
    "./data",
)
```

Many scAtlasPy operations modify the current Atlas and persist their outputs in
the database. Closing the Python session does not remove previously stored data
or analysis results.

```{eval-rst}
.. currentmodule:: scatlaspy

.. autosummary::
   :toctree: generated/
   :nosignatures:

   Atlas
```

## Lifecycle and Status

Use these methods to manage the database connection and inspect the overall
state of an Atlas.

```{eval-rst}
.. currentmodule:: scatlaspy

.. autosummary::
   :toctree: generated/
   :nosignatures:

   Atlas.connect
   Atlas.close
   Atlas.exists
   Atlas.describe
```

Close the database connection when the Atlas is no longer needed:

```python
atlas.close()
```

An existing `.sasql` Atlas can be reopened in a later Python session and used
to continue the analysis.

## SQL and Inspection

These methods provide access to stored tables, columns, metadata, and compact
analysis results.

```{eval-rst}
.. currentmodule:: scatlaspy

.. autosummary::
   :toctree: generated/
   :nosignatures:

   Atlas.query
   Atlas.execute_sql
   Atlas.head
   Atlas.table_names
   Atlas.table_info
   Atlas.has_table
   Atlas.has_column
```

Use `query()` when a SQL statement should return a pandas DataFrame:

```python
cell_counts = atlas.query("""
    SELECT
        cell_type_manual,
        COUNT(*) AS n_cells
    FROM obs
    GROUP BY cell_type_manual
    ORDER BY n_cells DESC
""")
```

Use `execute_sql()` for statements that modify database state or do not need to
return a DataFrame.

```{note}
Prefer documented Atlas methods and public result interfaces when they are
available. Internal tables and undocumented storage details should not be
treated as stable public APIs.
```

## Workflow State

These interfaces summarize the preprocessing and analysis state of the Atlas,
including the currently configured expression-data view.

```{eval-rst}
.. currentmodule:: scatlaspy

.. autosummary::
   :toctree: generated/
   :nosignatures:

   Atlas.workflow_state
   Atlas.read_index_info
```

They can be used to check whether required preprocessing steps or a read index
are available before running downstream analyses.

## Data Movement

These methods import data, retrieve selected Atlas contents, and export data to
other formats or in-memory objects.

```{eval-rst}
.. currentmodule:: scatlaspy

.. autosummary::
   :toctree: generated/
   :nosignatures:

   Atlas.load_h5ad
   Atlas.load_anndata
   Atlas.load_multi_format
   Atlas.write_h5ad
   Atlas.get_obs_df
   Atlas.get_anndata
   Atlas.gene_names_duplicated
```

For example, retrieve selected metadata columns as a pandas DataFrame:

```python
obs = atlas.get_obs_df(
    columns=[
        "atlas_cell_id",
        "sample",
        "cell_type_manual",
    ]
)
```

Or create an in-memory AnnData object for a selected cell population:

```python
selected_ids = obs.loc[
    obs["cell_type_manual"].eq("T cell"),
    "atlas_cell_id",
].tolist()

adata = atlas.get_anndata(
    selected_ids,
    use_data="data_log1p",
)
```

```{warning}
`get_anndata()` materializes the requested expression data in memory. Select a
cell population and expression representation that fit within the available
memory.
```

The same methods are also organized by task in the {doc}`io` API section.

## Analysis Views and Minibatches

These methods define how expression data are selected and provide bounded-memory
access to the resulting matrix.

```{eval-rst}
.. currentmodule:: scatlaspy

.. autosummary::
   :toctree: generated/
   :nosignatures:

   Atlas.build_read_index
   Atlas.get_minibatch_dense
```

A read index specifies:

- which cells are included;
- which genes are included;
- whether highly variable genes are used;
- which stored expression representation is read;
- the feature order supplied to downstream methods.

For example:

```python
atlas.build_read_index(
    cell_condition="filter_cells",
    gene_condition="filter_genes",
    use_hvg=True,
    use_data="data_log1p",
)
```

```{important}
A read index defines an active data view; it does not create another complete
copy of the expression matrix.

Rebuilding the read index changes the cells, genes, or expression
representation used by subsequent minibatch and analysis operations.
```

`get_minibatch_dense()` provides dense batches from the active read-index view
for streaming statistics, model training, and inference.

Worked examples are available in the
{doc}`../tutorials/advanced/index`.

## Related API Sections

- {doc}`io` organizes import and export interfaces by task.
- {doc}`preprocessing` documents functions in `sap.pp`.
- {doc}`tools` documents analysis functions in `sap.tl`.
- {doc}`plotting` documents visualization functions in `sap.pl`.