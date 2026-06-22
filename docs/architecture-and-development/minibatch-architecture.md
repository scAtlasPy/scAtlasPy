# Minibatch Architecture

scAtlasPy converts a selected Atlas view into bounded-memory minibatches for
streaming statistics, model fitting, inference, and custom method development.

This page is intended for contributors and method developers who need to
understand the behavior and current implementation of
`atlas.get_minibatch_dense()`. Most users can follow the
{doc}`../tutorials/advanced/index` without relying on these internal details.

## Public Minibatch Interface

The supported public interface is:

```python
for X_batch in atlas.get_minibatch_dense(
    pass_mode="single-pass",
    batch_size=2048,
):
    ...
```

Each yielded value is a dense NumPy array with shape:

```text
n_cells_in_batch × n_selected_genes
```

The arrays use the cell and gene selection defined by the current read index.
The final batch may contain fewer cells than `batch_size`.

Before requesting minibatches, construct the read index:

```python
atlas.build_read_index(
    cell_condition="filter_cells",
    gene_condition="filter_genes",
    use_hvg=True,
    use_data="data_log1p",
)
```

The read index defines:

- the cells included in the stream;
- the genes included in the stream;
- the expression representation reconstructed into each batch;
- the cell and gene order used by the current analysis view.

```{important}
Rebuilding the read index can change the selected cells, selected genes,
expression field, and feature order used by subsequent minibatch operations.
These choices should be treated as part of a custom method's input
specification.
```

## Persistent Identifiers and Read-index Order

An Atlas distinguishes persistent identities from positions in the current
analysis view.

| Identifier | Purpose |
|---|---|
| `atlas_cell_id` | Persistent identity of a cell in the Atlas |
| `atlas_gene_id` | Persistent identity of a gene in the Atlas |
| `filter_cell_id` | Cell position in the current read-index view |
| `filter_gene_id` | Gene position in the current read-index view |

Persistent identifiers remain suitable for joins and result write-back.
Read-index positions may change whenever `build_read_index()` is called again.

The feature order supplied to a model can be inspected with:

```python
feature_order = atlas.query("""
    SELECT
        filter_gene_id,
        atlas_gene_id,
        atlas_gene_name
    FROM var
    WHERE filter_gene_id IS NOT NULL
    ORDER BY filter_gene_id
""")
```

A reusable model should record these feature identities and their order.

## Current Read-index Storage

The current implementation materializes an internal filtered view used for
minibatch reconstruction.

| Internal artifact | Current role |
|---|---|
| `obs.filter_cell_id` | Cell order in the active view |
| `var.filter_gene_id` | Gene order and output-column positions |
| `X_HyS_data_filtered` | Sparse expression records for the selected view |
| `X_HyS_indptr_filtered` | Row pointers used to divide records into cells |
| `atlas_read_index_meta` | Metadata such as the selected expression field |

These artifacts explain the current implementation but are not intended as
stable third-party integration interfaces.

```{note}
External packages should use `build_read_index()`,
`get_minibatch_dense()`, and other documented public methods rather than
depending directly on filtered internal tables.
```

## Dense Reconstruction Pipeline

The current dense minibatch pipeline can be summarized as:

```text
persistent Atlas data
        ↓
active read-index view
        ↓
parallel producers fetch sparse records
        ↓
consumer restores record and cell boundaries
        ↓
dense batch reconstruction
        ↓
yield X_batch
```

Producer threads retrieve partitions of sparse expression records from DuckDB
and place them in an internal queue. Because partitions may complete out of
order, sequence information is used to restore the required record order.

For each output batch, the consumer:

1. identifies the cells belonging to the next batch;
2. reads the corresponding row-pointer information;
3. reconstructs row and column positions in read-index space;
4. allocates a dense `float32` matrix;
5. initializes implicit sparse entries;
6. inserts the stored expression values;
7. yields the completed batch.

This pipeline allows scAtlasPy to traverse the selected view without
materializing the complete cell-by-gene matrix in Python.

```{important}
Producer counts, queue sizes, fetch sizes, and internal worker classes are
implementation details and may change without altering the public minibatch
interface.
```

## Reconstructing Implicit Zeros

Sparse expression storage omits entries that are implicitly zero.

For count-scale, normalized, log-transformed, and square-root-transformed
representations, omitted values are reconstructed as:

```text
0.0
```

Scaled expression requires different handling. After gene-wise centering and
standardization, an original zero becomes:

```text
(0 - gene_mean) / gene_std
```

scAtlasPy stores this gene-specific transformed-zero value in:

```text
var.zero_scale_transform
```

When `use_data="data_scale"` is selected, each dense column is initialized from
`zero_scale_transform` before explicitly stored values are inserted.

```{warning}
Correct reconstruction of scaled minibatches requires
`var.zero_scale_transform`.

The current implementation may fall back to `0.0` when this value is
unavailable, but that fallback does not exactly represent standardized implicit
zeros. Scaled minibatches should therefore be created from a properly completed
`sap.pp.scale()` workflow.
```

## Single-pass Mode

Use `pass_mode="single-pass"` when an algorithm should visit each selected cell
once.

```python
for X_batch in atlas.get_minibatch_dense(
    pass_mode="single-pass",
    batch_size=4096,
):
    update_statistics(X_batch)
```

Single-pass mode is appropriate for:

- streaming means, variances, and covariance calculations;
- deterministic transformations;
- model inference;
- bounded-memory export;
- diagnostics covering the complete active view.

Its main properties are:

- cells are yielded in ascending `filter_cell_id` order;
- each selected cell is visited once;
- the final batch may be smaller than `batch_size`;
- setting `max_batches` truncates the traversal.

```{warning}
A single-pass stream limited by `max_batches` no longer represents the complete
selected population. Use this option for testing or intentional partial
processing, not for full-dataset statistics.
```

### Aligning Metadata in Single-pass Mode

Because single-pass batches follow `filter_cell_id` order, cell metadata can be
retrieved in the same order:

```python
cell_order = atlas.query("""
    SELECT
        filter_cell_id,
        atlas_cell_id,
        cell_type_manual
    FROM obs
    WHERE filter_cell_id IS NOT NULL
    ORDER BY filter_cell_id
""")
```

A caller can advance through this table using the number of rows in each
`X_batch`.

Persistent results should still be associated with cells through
`atlas_cell_id`, not through `filter_cell_id`.

## Multi-pass Mode

Use `pass_mode="multi-pass"` for iterative methods that require repeated and
randomized access to the active view.

```python
for X_batch in atlas.get_minibatch_dense(
    pass_mode="multi-pass",
    batch_size=2048,
    buffer_batch_num=5,
    max_batches=1000,
):
    train_step(X_batch)
```

Typical applications include:

- neural-network training;
- stochastic optimization;
- minibatch clustering;
- iterative matrix factorization;
- representation learning.

The iterator repeatedly traverses the active read-index view until the
`max_batches` budget is exhausted.

```{warning}
When `pass_mode="multi-pass"` and `max_batches=None`, repeated passes may
continue indefinitely. Training workflows should provide an explicit stopping
condition.
```

## Shuffle Buffer

In multi-pass mode, reconstructed dense batches pass through a shuffle buffer
before they are returned.

```text
batches in read-index order
        ↓
store buffer_batch_num batches
        ↓
randomly permute cells in the buffer
        ↓
yield shuffled batches
```

`buffer_batch_num` controls the amount of local mixing.

A larger buffer generally improves randomization but increases memory
consumption.

Approximate buffer memory is:

```text
batch_size × buffer_batch_num × n_selected_genes × 4 bytes
```

The factor of four assumes `float32` values.

For `batch_size=2048` and 2,000 selected genes:

| `buffer_batch_num` | Approximate buffer memory |
|---:|---:|
| 1 | 16 MiB |
| 5 | 78 MiB |
| 10 | 156 MiB |
| 20 | 312 MiB |

These estimates include only the dense arrays retained by the shuffle buffer.
Additional memory is used by:

- DuckDB queries;
- producer and consumer queues;
- the batch currently being reconstructed;
- model parameters and optimizer state;
- temporary arrays created by the downstream method.

At the end of a pass, a partially filled buffer is flushed so that remaining
cells are not dropped.

## Understanding `max_batches`

`max_batches` limits the total number of yielded batches.

In single-pass mode, it is most useful for development and debugging:

```python
for X_batch in atlas.get_minibatch_dense(
    pass_mode="single-pass",
    max_batches=20,
):
    ...
```

In multi-pass mode, it commonly defines the training-step budget.

A rough estimate of the number of cell samples processed is:

```text
max_batches × batch_size
```

If the active view contains `N` cells and approximately `E` passes are desired,
a starting estimate is:

```text
max_batches ≈ ceil(E × N / batch_size)
```

This is not an exact epoch count because:

- the final batch of a pass may be smaller than `batch_size`;
- cells are shuffled locally through the buffer;
- batches may span repeated passes;
- `max_batches` may stop partway through a pass.

Treat `max_batches` as a computational budget rather than a strict epoch
definition.

## Labels and Identifier Alignment

The current dense minibatch interface returns only expression arrays:

```python
X_batch
```

It does not return the corresponding `atlas_cell_id` values with each batch.

For deterministic single-pass workflows, labels can be aligned through
`filter_cell_id` order.

For randomized multi-pass workflows, a separately ordered label vector cannot
be assumed to follow the same shuffle order.

```{important}
Do not pair randomized multi-pass expression batches with labels retrieved
independently from `obs` unless the integration explicitly applies the same
permutation.

Supervised randomized training requires an identifier-aware or label-aware
batch interface.
```

The same consideration applies when a method produces cell-level predictions
that must later be written back to the Atlas.

## Choosing Minibatch Parameters

The main public controls are:

| Parameter | Effect |
|---|---|
| `pass_mode` | Selects one deterministic traversal or repeated shuffled passes |
| `batch_size` | Controls the number of cells in each returned batch |
| `buffer_batch_num` | Controls local randomization and shuffle-buffer memory in multi-pass mode |
| `max_batches` | Limits the total number of returned batches |

Larger batches may improve throughput but increase:

- dense matrix memory;
- downstream temporary memory;
- model-device memory;
- latency before the first batch is returned.

Parameter selection should consider both data-access throughput and the memory
requirements of the downstream algorithm.

Use:

```python
sap.set_verbosity("info")
```

to display runtime and throughput information when available.

## Public and Internal Interfaces

The supported public minibatch interface is:

```text
Atlas.get_minibatch_dense()
```

The codebase may contain additional experimental or incomplete readers. For
example, `Atlas.get_minibatch_csr()` is not currently a supported replacement
for the dense iterator and should not be used in documented workflows.

Third-party integrations should not depend directly on internal classes such
as:

```text
MultiThreadedMinibatchFetcher
ShuffleBuffer
```

Their implementation and constructor behavior may change between releases.

## Current Limitations

The current minibatch interface has the following limitations:

- batches are returned as dense matrices;
- randomized batches do not include cell identifiers or labels;
- supervised multi-pass training therefore requires an additional alignment
  mechanism;
- scaled reconstruction depends on `var.zero_scale_transform`;
- minibatch iteration uses the current read-index view and does not preserve an
  earlier view after the index is rebuilt;
- internal concurrency and buffering settings are not part of the stable public
  API.

See {doc}`known-limitations` for other version-specific restrictions.

## Related Documentation

- {doc}`data-model` explains the Atlas tables and identifiers underlying the
  read index.
- {doc}`performance` discusses throughput, memory use, and parameter tuning.
- {doc}`../api/atlas` documents `build_read_index()` and
  `get_minibatch_dense()`.
- {doc}`../tutorials/advanced/stream-mean-and-variance` demonstrates a
  single-pass computation.
- {doc}`../tutorials/advanced/stream-covariance-matrix` extends single-pass
  statistics to a covariance matrix.
- {doc}`../tutorials/advanced/train-pytorch-model-with-minibatches`
  demonstrates iterative model training.
- {doc}`../tutorials/advanced/implement-minibatch-kmeans` demonstrates a
  multi-pass clustering method.
- {doc}`../tutorials/advanced/apply-model-to-full-atlas` demonstrates
  batch-wise inference.