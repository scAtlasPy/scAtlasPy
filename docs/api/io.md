# Input and Output: `sap.io`

## Overview

`sap.io` provides the public interfaces for importing data into an Atlas and
retrieving or exporting stored data.

Most functions are also available as convenience methods on `Atlas`. The
instance-method form is recommended in user workflows because it makes the
target Atlas explicit:

```python
atlas.load_h5ad("input.h5ad")
adata = atlas.get_anndata(cell_ids)
```

The equivalent module-level functions remain available through `sap.io`.

## Import Data

```{eval-rst}
.. currentmodule:: scatlaspy

.. autosummary::
   :toctree: generated/
   :nosignatures:

   io.load_h5ad
   io.load_anndata
   io.load_multi_format
```

Choose an import function according to the source data:

| Input | Recommended interface |
|---|---|
| One h5ad file | `atlas.load_h5ad(path)` |
| Multiple h5ad files | `atlas.load_h5ad(paths, load_type="random")` |
| An in-memory `AnnData` object | `atlas.load_anndata(adata)` |
| Another supported format | `atlas.load_multi_format(path)` |

### h5ad Load Modes

`load_h5ad()` supports different strategies for reading cells into the Atlas:

| `load_type` | Description |
|---|---|
| `"order"` | Read a single h5ad file in its existing cell order (or multiple files in list order) |
| `"random"` | Read a single h5ad file through randomized windows (or multiple files with global randomization) |

Use the same `load_type` for single or multiple files — the code automatically detects whether ``h5ad_path`` is a list.

Consult {meth}`scatlaspy.io.load_h5ad` for the complete parameters, defaults,
and supported combinations.

## Check Gene Names

```{eval-rst}
.. currentmodule:: scatlaspy

.. autosummary::
   :toctree: generated/
   :nosignatures:

   io.rename_duplicated_genes
```

Use `rename_duplicated_genes()` to make duplicated gene names unique when
validating imported data or preparing an export.

## Retrieve and Export Data

```{eval-rst}
.. currentmodule:: scatlaspy

.. autosummary::
   :toctree: generated/
   :nosignatures:

   io.get_obs_df
   io.get_anndata
   io.write_h5ad
```

| Goal | Recommended interface |
|---|---|
| Retrieve cell metadata as a pandas DataFrame | `atlas.get_obs_df()` |
| Create an in-memory AnnData object | `atlas.get_anndata(...)` |
| Write Atlas data to an h5ad file | `atlas.write_h5ad(path)` |

### Retrieve Cell Metadata

Use `get_obs_df()` to retrieve all or selected columns from `obs`:

```python
obs = atlas.get_obs_df(
    columns=[
        "atlas_cell_id",
        "sample",
        "cell_type_manual",
    ]
)
```

### Create an AnnData Object

Use `get_anndata()` to materialize selected Atlas data as an in-memory AnnData
object:

```python
adata = atlas.get_anndata(
    cell_ids,
    use_data="data_log1p",
    include_obsm=True,
    include_varm=False,
)
```

`use_data` selects the expression representation placed in `adata.X`.
`include_obsm` and `include_varm` control whether corresponding
multidimensional results are included.

```{warning}
`get_anndata()` materializes the requested expression matrix in memory. Select
a cell population and feature set that fit within the available memory.
```

### Write an h5ad File

Use `write_h5ad()` for direct file export:

```python
atlas.write_h5ad("output.h5ad")
```

When a specific expression representation is required, create an AnnData object
with `get_anndata(use_data=...)` and write it using AnnData:

```python
adata = atlas.get_anndata(
    cell_ids,
    use_data="data_log1p",
)

adata.write_h5ad("selected_log1p.h5ad")
```

## Calling Styles

The following forms are equivalent:

| Atlas method | Module-level function |
|---|---|
| `atlas.load_h5ad(path)` | `sap.io.load_h5ad(path, atlas)` |
| `atlas.load_anndata(adata)` | `sap.io.load_anndata(adata, atlas)` |
| `atlas.load_multi_format(path)` | `sap.io.load_multi_format(path, atlas)` |
| `atlas.get_obs_df()` | `sap.io.get_obs_df(atlas)` |
| `atlas.get_anndata(cell_ids)` | `sap.io.get_anndata(atlas, cell_ids)` |
| `atlas.write_h5ad(path)` | `sap.io.write_h5ad(atlas, path)` |
| `atlas.rename_duplicated_genes()` | `sap.io.rename_duplicated_genes(atlas)` |

The Atlas-method form is generally preferred in tutorials and interactive
analysis. The module-level form can be useful for functional APIs and
third-party integrations.

## Related Documentation

- {doc}`../tutorials/basic/import-data-from-multiple-formats` explains how to
  import common source formats.
- {doc}`../cross-platform-workflows/use-external-methods-on-atlas-subsets`
  explains how to retrieve a focused population for an in-memory method.
- {doc}`atlas` provides the complete `Atlas` method index.
