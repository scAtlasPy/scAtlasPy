# Return External Results to the Atlas

External methods may produce labels, scores, or other results for a selected
cell population. These results can be returned to the original Atlas using the
persistent `atlas_cell_id`.

## 1. Prepare the Result Table

For example, collect a clustering result from an AnnData object:

```python
cell_results = (
    adata.obs[
        [
            "atlas_cell_id",
            "external_leiden",
        ]
    ]
    .copy()
    .reset_index(drop=True)
)
```

Confirm that the identifiers are present and unique:

```python
if cell_results["atlas_cell_id"].isna().any():
    raise ValueError("Missing atlas_cell_id values.")

if cell_results["atlas_cell_id"].duplicated().any():
    raise ValueError("Duplicated atlas_cell_id values.")
```

## 2. Write the Result to obs

Register the DataFrame temporarily:

```python
conn = atlas.connection
conn.register("external_results", cell_results)
```

Check that every returned identifier exists in the Atlas before writing:

```python
unknown_cells = atlas.query("""
    SELECT
        r.atlas_cell_id
    FROM external_results AS r
    LEFT JOIN obs AS o
      ON r.atlas_cell_id = o.atlas_cell_id
    WHERE o.atlas_cell_id IS NULL
    LIMIT 20
""")

if not unknown_cells.empty:
    raise ValueError(
        "The external result contains atlas_cell_id values "
        "that are not present in obs."
    )
```

Create a column and write the labels by matching `atlas_cell_id`:

```python
atlas.execute_sql("""
    ALTER TABLE obs
    ADD COLUMN IF NOT EXISTS external_leiden VARCHAR
""")
```

When replacing an earlier result stored under the same column name, clear the
old values first:

```python
atlas.execute_sql("""
    UPDATE obs
    SET external_leiden = NULL
""")
```

Then write the returned labels by identifier:

```python
atlas.execute_sql("""
    UPDATE obs
    SET external_leiden = r.external_leiden
    FROM external_results AS r
    WHERE obs.atlas_cell_id = r.atlas_cell_id
""")

conn.unregister("external_results")
```

Cells outside the analyzed population remain `NULL` after a full replacement.
For an intentional incremental update, skip the clearing step and document that
unmatched cells keep their previous values.

## 3. Inspect the Returned Result

```python
atlas.query("""
    SELECT
        external_leiden,
        COUNT(*) AS n_cells
    FROM obs
    WHERE external_leiden IS NOT NULL
    GROUP BY external_leiden
    ORDER BY n_cells DESC
""")
```

The returned labels can now be used in Atlas queries and visualizations:

```python
sap.pl.umap(
    atlas,
    color="external_leiden",
)
```

```{important}
Always align external results using `atlas_cell_id`, not row order. External
tools may reorder or remove cells during analysis.
```

Cell-level scalar results are usually stored in `obs`, while gene-level scalar
results can be stored in `var` using `atlas_gene_id`. Embeddings, probability
matrices, and other multidimensional outputs should use dedicated result
tables or supported result-storage interfaces.
