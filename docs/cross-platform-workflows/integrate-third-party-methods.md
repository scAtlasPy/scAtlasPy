# Integrate Third-party Methods with scAtlasPy

Third-party packages can retain their own algorithms and user-facing APIs while
using scAtlasPy to access atlas-scale data.

Depending on the method, an integration may use:

| Requirement | scAtlasPy interface |
|---|---|
| Metadata, annotations, or summaries | `atlas.get_obs_df()` or `atlas.query()` |
| Focused in-memory analysis | `atlas.get_anndata()` |
| Complete deterministic traversal | Single-pass minibatches |
| Iterative model training | Multi-pass minibatches |
| Returning method outputs | Persistent cell and gene identifiers |

## Define the Method Input

Before accessing expression data, define the cells, genes, and expression
representation used by the method:

```python
atlas.build_read_index(
    cell_condition="filter_cells",
    gene_condition="filter_genes",
    use_hvg=True,
    use_data="data_log1p",
)
```

The selected feature identities, feature order, and expression representation
should be treated as part of the method input and recorded with any fitted
model.

## Use Atlas Streams and Minibatches

Single-pass access is suitable for statistics, transformations, and inference
that require one traversal of the selected data.

Multi-pass access is suitable for iterative methods such as neural-network
training, matrix factorization, and minibatch clustering.

For dense matrix batches, use:

```python
for X_batch in atlas.get_minibatch_dense(
    pass_mode="single-pass",
    batch_size=4096,
):
    ...
```

Use `pass_mode="multi-pass"` for iterative training. To return cell identifiers
or `obs` columns together with each batch, pass `get_obs_col`:

```python
for batch in atlas.get_minibatch_dense(
    pass_mode="multi-pass",
    batch_size=4096,
    get_obs_col="kmeans",
):
    X_batch = batch["X"]
    labels = batch["kmeans"]
    filter_cell_ids = batch["filter_cell_ids"]
```

Worked examples are available in the
{doc}`../tutorials/advanced/index`, including:

- {doc}`../tutorials/advanced/stream-mean-and-variance`;
- {doc}`../tutorials/advanced/train-pytorch-model-with-minibatches`;
- {doc}`../tutorials/advanced/implement-minibatch-kmeans`;
- {doc}`../tutorials/advanced/apply-model-to-full-atlas`.

## Preserve Identifiers

Cell-level outputs should retain `atlas_cell_id`, and gene-level outputs should
retain `atlas_gene_id`.

Do not rely on row or column positions when returning results, because a method
may reorder or filter its input.

See {doc}`return-external-results-to-atlas` for a write-back example.

```{important}
Third-party integrations should use documented public interfaces rather than
depending on internal Atlas tables or implementation details.
```

We welcome third-party packages that add native support for scAtlasPy. When an
existing public interface does not provide information required by a method,
developers are encouraged to propose the required interface rather than access
the internal storage schema directly.
