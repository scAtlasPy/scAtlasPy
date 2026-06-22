# Use External Methods on Selected Atlas Data

Use this workflow when you want to select a manageable cell population from an
Atlas and analyze it with an in-memory method from Scanpy or another
single-cell package.

The workflow has three steps:

1. retrieve the relevant columns from `obs`;
2. select cells with pandas;
3. create an AnnData object with `atlas.get_anndata()`.

## 1. Retrieve Cell Metadata

Read only the columns required to define the population:

```python
obs_df = atlas.get_obs_df(
    columns=[
        "atlas_cell_id",
        "filter_cells",
        "cell_type_manual",
        "condition",
    ]
)
```

Always include `atlas_cell_id`, which connects the selected cells and any
external results back to the original Atlas.

## 2. Select a Cell Population

Use standard pandas operations to select the cells of interest.

For example, select quality-controlled T cells from treated samples:

```python
selected_obs = obs_df[
    obs_df["filter_cells"].fillna(False)
    & obs_df["cell_type_manual"].eq("T cell")
    & obs_df["condition"].eq("treated")
].copy()

print(f"Selected cells: {len(selected_obs):,}")
```

Collect their persistent identifiers:

```python
atlas_cell_ids = selected_obs["atlas_cell_id"].tolist()

if not atlas_cell_ids:
    raise ValueError("No cells matched the selected population.")
```

```{important}
Use `atlas_cell_id` rather than the DataFrame row index. Filtering, sorting, or
external tools may change row order.
```

## 3. Create an AnnData Object

Retrieve the selected cells as an in-memory AnnData object:

```python
adata = atlas.get_anndata(
    atlas_cell_ids,
    use_data="data_log1p",
    include_obsm=True,
)
```

Inspect the result before continuing:

```python
print(adata)
print(adata.shape)
```

`get_anndata()` materializes the selected population in memory. The complete
Atlas remains stored in the `.sasql` database.

The current `get_anndata()` API subsets cells by `atlas_cell_ids` and uses the
expression field passed through `use_data`. It exports the complete Atlas
feature table from `var`; it does not automatically subset genes from the
current read index. If an external method needs highly variable genes or a
custom panel, subset the AnnData object by `adata.var["atlas_gene_id"]` or by
gene names before running the method, and account for the full exported feature
set when estimating memory.

## Example: Focused Analysis with Scanpy

The resulting AnnData object can be used directly by in-memory tools.

For example, perform focused subclustering with Scanpy:

```python
import scanpy as sc

sc.pp.scale(adata, max_value=10)
sc.tl.pca(adata)
sc.pp.neighbors(adata)
sc.tl.leiden(
    adata,
    key_added="external_leiden",
)
sc.tl.umap(adata)

sc.pl.umap(
    adata,
    color="external_leiden",
)
```

Changes made to `adata` are not automatically written back to the Atlas.

See {doc}`return-external-results-to-atlas` to associate external labels,
scores, embeddings, or other outputs with the original cells.

```{note}
Choose an expression representation compatible with the external method. For
example, do not repeat normalization or `log1p` transformation when
`use_data="data_log1p"` is used.
```
