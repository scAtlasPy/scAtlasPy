# Import Data from Different Sources

This tutorial explains how to import data into scAtlasPy from several common
single-cell data sources. It is intended as a companion to the complete basic
analysis tutorial, which follows a single `.h5ad` dataset from import through
cell-type annotation.

The examples below are **alternative import workflows**. Choose the section that
matches your data source; you do not need to run every example.

By the end of this tutorial, you will be able to:

- choose an import method for your data;
- import one or several `.h5ad` files;
- import an existing `AnnData` object;
- import data from other supported formats;
- verify that the resulting Atlas contains the expected data and metadata.

## Before Importing

Before choosing an import method, inspect the source data and confirm:

- which expression representation you want to store in the Atlas;
- whether cell and gene identifiers are unique;
- whether required sample, donor, batch, condition, or technology annotations
  are present;
- whether multiple input files use compatible genes, metadata, and expression
  representations after import.

scAtlasPy stores imported expression data, cell and gene metadata, embeddings,
and supported analysis results in a persistent `.sasql` Atlas database.

```{important}
scAtlasPy automatically detects whether the source matrix contains count-scale
or log-transformed values and always stores expression data on the count scale
in the Atlas. Normalization and log transformation should be performed after
import using the `sap.pp` preprocessing functions.
```

## Choose an Import Method

Use the source object or file format to select an import path.

| Your data | Recommended method |
|---|---|
| One `.h5ad` file | `atlas.load_h5ad()` |
| Several `.h5ad` files forming one Atlas | `atlas.load_h5ad(paths, load_type="random")` |
| An `AnnData` object already loaded in Python | `atlas.load_anndata()` |
| A smaller file supported by `load_multi_format()` | `atlas.load_multi_format()` |
| A format requiring reader-specific arguments | Read it with Scanpy, then use `atlas.load_anndata()` |

For large `.h5ad` files, prefer `load_h5ad()` because it can import the
expression matrix in blocks without requiring the complete dataset to remain in
memory.

## Create an Atlas

Each import workflow writes to a `.sasql` database. Create an `Atlas` by
providing the target database path:

```python
from pathlib import Path

import scatlaspy as sap

atlas_path = Path("./data/my_atlas.sasql")

atlas = sap.Atlas(
    atlas_path,
    db_memory_limit="8GB",
)
```

If the path does not end with `.sasql`, scAtlasPy adds the suffix automatically.

`db_memory_limit` controls the memory available to DuckDB queries and
intermediate database operations. It is not a global limit on memory allocated
by Python, NumPy, pandas, plotting libraries, or external machine-learning
frameworks.

```{note}
Use a separate `.sasql` file for each analysis project or coherent dataset.
Avoid importing unrelated datasets into the same Atlas.
```

## Import a Single h5ad File

Use `load_h5ad()` when the source data are stored in one `.h5ad` file.

### Preserve the Source Cell Order

Use `load_type="order"` when you want cells to be imported in their original
order:

```python
atlas = sap.Atlas(
    "./data/pbmc_ordered.sasql",
    db_memory_limit="8GB",
)

atlas.load_h5ad(
    "./data/pbmc3k.h5ad",
    load_type="order",
)
```

This is the most straightforward option when comparing the imported Atlas with
the source `AnnData` object or when the original cell order is meaningful to
your workflow.

### Import in Randomized Blocks

Use `load_type="random"` when a single `.h5ad` file should be imported in
randomized blocks:

```python
atlas = sap.Atlas(
    "./data/large_dataset.sasql",
    db_memory_limit="8GB",
)

atlas.load_h5ad(
    "./data/large_dataset.h5ad",
    load_type="random",
    cells_per_block=2000,
)
```

`cells_per_block` controls the number of cells processed in each block. Larger
blocks may reduce import overhead but require more working memory.

When `cells_per_block` is not specified, scAtlasPy selects a block size using
the source data and Atlas configuration.

```{note}
The expression values and cell metadata remain associated during randomized
import. Randomization changes the import order, not the correspondence between
the matrix and `obs`.
```

Use randomized import when later sequential reads should not reproduce the
original ordering of the source file. Use ordered import when preserving source
order is more useful for inspection or comparison.

## Import Multiple h5ad Files

Use `load_type="random"` to combine several `.h5ad` files into one Atlas:

```python
atlas = sap.Atlas(
    "./data/multi_sample_atlas.sasql",
    db_memory_limit="8GB",
)

atlas.load_h5ad(
    [
        "./data/sample_1.h5ad",
        "./data/sample_2.h5ad",
        "./data/sample_3.h5ad",
    ],
    load_type="random",
)
```

Before importing multiple files, confirm that:

- all files use the same expression representation;
- genes can be aligned consistently across files;
- gene identifiers and gene order satisfy the requirements of
  `load_h5ad()`;
- cell identifiers are unique across files;
- important source labels are already present in `obs`;
- shared metadata columns use compatible meanings and data types.

For example, add a sample label before import when each file represents a
different sample:

```python
import scanpy as sc

adata = sc.read_h5ad("./data/sample_1.h5ad")
adata.obs["sample"] = "sample_1"
adata.write_h5ad("./data/sample_1_annotated.h5ad")
```

Repeat this preparation for each input file, then import the annotated files
together.

```{important}
Do not rely on the file name as the only record of sample identity. Store
sample, donor, batch, condition, technology, or other relevant labels in `obs`
before combining files.
```

If the input files contain different gene sets or incompatible metadata, align
and validate them before import rather than assuming that they will be
harmonized automatically.

## Import an Existing AnnData Object

Use `load_anndata()` when the data are already available as an in-memory
`AnnData` object.

This is useful when:

- the dataset is small enough to fit comfortably in memory;
- custom preparation has already been performed with Scanpy or AnnData;
- the input format requires a specialized reader;
- metadata or feature identifiers must be modified before import.

```python
import scanpy as sc
import scatlaspy as sap

adata = sc.read_h5ad("./data/pbmc3k.h5ad")
adata.obs["sample"] = "pbmc3k"

atlas = sap.Atlas(
    "./data/pbmc_from_anndata.sasql",
    db_memory_limit="8GB",
)

atlas.load_anndata(adata)
```

`load_anndata()` writes the in-memory object's expression data, `obs`, `var`,
and supported annotation matrices into the Atlas.

Because the complete `AnnData` object is already materialized in memory, this
path is generally not preferred for datasets that exceed available memory.

## Import Other File Formats

Use `load_multi_format()` for supported formats that can be read by the built-in
Scanpy reader selection without additional format-specific arguments.

```{note}
`load_multi_format()` currently imports non-h5ad formats by reading them with
Scanpy into an in-memory `AnnData` object and then writing that object into an
Atlas database. For these additional formats, we plan to progressively add
streaming, block-wise, randomized import paths to scAtlasPy.
```

The calling pattern is the same across file types: create an Atlas database, then
pass the input file path to `load_multi_format()`.

```python
import scatlaspy as sap

atlas = sap.Atlas(
    "./data/from_loom.sasql",
    db_memory_limit="8GB",
)

atlas.load_multi_format("./data/input.loom")
```

For other simple file-based formats, change only the input path and the target
Atlas path:

```python
atlas = sap.Atlas(
    "./data/from_csv.sasql",
    db_memory_limit="8GB",
)
atlas.load_multi_format("./data/expression.csv")

atlas = sap.Atlas(
    "./data/from_text.sasql",
    db_memory_limit="8GB",
)
atlas.load_multi_format("./data/expression.tsv")

atlas = sap.Atlas(
    "./data/from_10x_h5.sasql",
    db_memory_limit="8GB",
)
atlas.load_multi_format("./data/filtered_feature_bc_matrix.h5")
```

Depending on the available Scanpy readers, this convenience path may be used for
formats such as:

- `.loom`;
- delimited expression matrices such as `.csv`, `.txt`, or `.tsv`;
- spreadsheet-based matrices such as `.xlsx`;
- selected matrix or HDF5-based single-cell formats such as 10x `.h5`.

```{important}
A file extension alone may not fully describe the data layout. Formats such as
Matrix Market or 10x data may consist of several files or require arguments
that specify feature names, matrix orientation, or genome information.
```

When a format requires reader-specific options, load it with Scanpy and then use
`load_anndata()`. In this case, `load_anndata()` is a fallback after custom
reading, not the primary multi-format import interface.

For example, import a 10x Matrix Market directory with:

```python
import scanpy as sc
import scatlaspy as sap

adata = sc.read_10x_mtx(
    "./data/filtered_feature_bc_matrix",
    var_names="gene_symbols",
)

adata.var_names_make_unique()

atlas = sap.Atlas(
    "./data/from_10x_mtx.sasql",
    db_memory_limit="8GB",
)

atlas.load_anndata(adata)
```

This approach gives you control over the Scanpy reader arguments before the
data are written into the Atlas.

## Validate the Imported Atlas

After completing any import workflow, inspect the Atlas before beginning
quality control or preprocessing:

```python
atlas.describe()
atlas.head("obs", n=5)
atlas.head("var", n=5)
```

Confirm that:

- the number of cells and genes matches the source data;
- expected `obs` columns are present (sample, donor, batch, condition, etc.);
- expected `var` columns are present (gene names, IDs, etc.);
- the imported expression representation matches the intended analysis.

Normalize duplicated gene names early:

```python
atlas.rename_duplicated_genes()
```

This call keeps the first occurrence unchanged and adds suffixes to later
duplicates in the Atlas `var` table. Gene names should be unique before
workflows that select, plot, rank, or annotate genes by name.

When importing multiple files, also confirm that the expected number of cells
was imported from each source group. For example, query a sample label stored
in `obs`:

```python
atlas.query("""
    SELECT sample, COUNT(*) AS n_cells
    FROM obs
    GROUP BY sample
    ORDER BY sample
""")
```

## Close and Reopen the Atlas

Close the database connection when the current session is complete:

```python
atlas.close()
```

The imported data remain stored in the `.sasql` file.

Reconnect in a later Python session by constructing an `Atlas` with the same
path:

```python
import scatlaspy as sap

atlas = sap.Atlas(
    "./data/my_atlas.sasql",
    db_memory_limit="8GB",
)
```

You do not need to import the source data again.

## Next Steps

After importing and validating the data:

- continue with the complete basic analysis tutorial;
- calculate quality-control metrics and filter cells and genes;
- normalize the expression matrix and select highly variable genes;
- build an analysis view for dimensionality reduction and clustering.

See {doc}`basic_exploration` for a complete preprocessing, clustering, and
annotation workflow after import.

For detailed function arguments, consult the API reference for:

- `Atlas.load_h5ad()`;
- `Atlas.load_anndata()`;
- `Atlas.load_multi_format()`.
