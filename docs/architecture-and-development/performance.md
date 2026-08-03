# Performance and Resource Tuning

scAtlasPy is designed to analyze large single-cell atlases without loading the
complete cell-by-gene matrix into Python memory.

Runtime and memory use still depend on:

- the number of selected cells and genes;
- expression-matrix sparsity;
- the selected expression representation;
- storage performance;
- DuckDB memory settings;
- minibatch configuration;
- the downstream algorithm.

This page summarizes the main controls for improving throughput and avoiding
unnecessary memory use. Exact performance should always be evaluated with the
current scAtlasPy version, dataset, and computing environment.

```{important}
The values shown on this page are starting points rather than fixed
recommendations. Increase resource settings gradually while monitoring runtime
and peak memory.
```

## Tune the Analysis View First

Many scAtlasPy workflows read expression data through an active read index:

```python
atlas.build_read_index(
    cell_condition="filter_cells",
    gene_condition="filter_genes",
    use_hvg=True,
    use_data="data_log1p",
)
```

The read index determines which data are supplied to minibatch and streaming
operations.

| Setting | Performance effect |
|---|---|
| `cell_condition` | Controls how many cells are traversed |
| `gene_condition` | Controls which genes are included |
| `use_hvg=True` | Restricts the view to highly variable genes |
| `use_data` | Selects the expression representation reconstructed into each batch |

Reducing the active cell population or gene set is often more effective than
increasing memory limits or worker counts.

For example, a focused downstream analysis should use only the relevant cell
population rather than repeatedly processing the complete Atlas.

## Understand the Main Memory Consumers

Memory used during a scAtlasPy workflow can come from several independent
components:

- DuckDB query execution and intermediate results;
- dense minibatches reconstructed in Python;
- producer and consumer queues;
- pandas or AnnData objects;
- temporary NumPy or SciPy arrays;
- model parameters and optimizer state;
- plotting and visualization objects.

The approximate size of one dense `float32` minibatch is:

```text
batch_size × n_selected_genes × 4 bytes
```

For example, a batch containing 4,096 cells and 2,000 genes requires
approximately:

```text
4096 × 2000 × 4 bytes ≈ 31 MiB
```

This estimate covers only the main expression array. The complete operation may
use substantially more memory.

## DuckDB Memory Limit

The DuckDB memory limit can be configured when creating or opening an Atlas:

```python
atlas = sap.Atlas(
    "atlas.sasql",
    db_memory_limit="32GB",
)
```

An integer may also be interpreted as a number of gigabytes:

```python
atlas = sap.Atlas(
    "atlas.sasql",
    db_memory_limit=32,
)
```

`db_memory_limit` affects memory used by DuckDB connections and database-side
operations. It does not directly limit memory allocated by:

- Python;
- NumPy or SciPy;
- pandas;
- AnnData;
- scikit-learn;
- PyTorch;
- other external libraries.

Leave sufficient physical memory for these components and for the operating
system.

```{warning}
Setting the DuckDB limit close to the machine's total physical memory can cause
memory pressure when database operations and Python-side computations run at
the same time.
```

By default, scAtlasPy requests `db_memory_limit="10G"`. The effective DuckDB
limit is capped at 60% of detected system physical memory, rounded down to an
integer number of GB. Passing `db_memory_limit=None` explicitly uses that 60%
system-memory cap directly.

## Import Performance

Large imports may involve:

- backed reading from h5ad;
- sparse-matrix conversion;
- metadata processing;
- Python-side blocks;
- Arrow or pandas objects;
- DuckDB writes;
- index construction.

The main practical control is usually the number of cells processed in each
import block.

| Situation | Suggested adjustment |
|---|---|
| Peak memory is too high | Reduce cells per block or import batch size |
| Import is stable but storage is underused | Increase the block size gradually |
| Input contains several large files | Use incremental multi-file import |
| An AnnData object is already very large | Prefer direct file import when possible |
| Import is limited by disk throughput | Use fast local storage where available |

Smaller blocks generally reduce peak memory but may increase overhead. Larger
blocks can improve throughput until memory, storage bandwidth, or database
write performance becomes the limiting factor.

Avoid creating unnecessary complete AnnData copies before import.

## Dense Minibatch Performance

Dense batches are retrieved with:

```python
for X_batch in atlas.get_minibatch_dense(
    pass_mode="single-pass",
    batch_size=4096,
):
    ...
```

The main controls are:

| Parameter | Effect |
|---|---|
| `batch_size` | Number of cells returned in each batch |
| `pass_mode` | Selects a single traversal or repeated shuffled passes |
| `buffer_batch_num` | Number of batches retained for local shuffling in multi-pass mode |
| `max_batches` | Maximum number of batches returned |

### `batch_size`

Larger batches may:

- improve database and model throughput;
- reduce per-batch Python overhead;
- make better use of vectorized operations or accelerators.

They also increase:

- dense-array memory;
- temporary memory used by downstream methods;
- latency before a completed batch is returned;
- device memory when batches are transferred to a GPU.

Increase `batch_size` gradually and monitor both throughput and memory.

When memory use is too high, reduce either:

- `batch_size`;
- the number of selected genes;
- the selected cell population.

### `pass_mode`

Use:

```text
single-pass
```

for statistics, inference, deterministic transformations, and other workflows
that should visit each selected cell once.

Use:

```text
multi-pass
```

for iterative model fitting, stochastic optimization, and minibatch clustering.

```{warning}
A multi-pass iterator without a finite stopping condition may continue creating
new passes. Use `max_batches` or another explicit training limit.
```

### `buffer_batch_num`

In multi-pass mode, `buffer_batch_num` controls how many dense batches are
retained and shuffled together.

Approximate shuffle-buffer memory is:

```text
batch_size × buffer_batch_num × n_selected_genes × 4 bytes
```

A larger buffer improves local mixing but increases memory use. It does not
guarantee a complete random permutation of the Atlas.

Start with a modest buffer and increase it only when the method benefits from
additional randomization.

See {doc}`minibatch-architecture` for details of single-pass, multi-pass, and
shuffle-buffer behavior.

### `max_batches`

In single-pass mode, `max_batches` is mainly useful for:

- testing code;
- checking shapes and data types;
- estimating throughput;
- debugging a custom method.

A truncated single-pass stream does not represent the complete selected
population.

In multi-pass mode, `max_batches` commonly defines the optimization budget.

A rough starting estimate for approximately `E` passes over `N` cells is:

```text
max_batches ≈ ceil(E × N / batch_size)
```

This should be interpreted as an approximate training budget rather than an
exact epoch count.

## Analysis-specific Controls

High-level analysis functions may expose additional parameters that affect
runtime and resource use.

| Workflow | Parameters to inspect | Typical trade-off |
|---|---|---|
| PCA | `batch_size`, `oversample`, `n_iter` | Larger randomized subspaces and more subspace iterations improve accuracy but increase scan time |
| Distilled Louvain | `fit_sample_n`, `torch_epochs`, `transform_batch_size` | Larger teacher fitting budgets and longer student training can improve label fidelity but increase runtime |
| KMeans | `batch_size`, `fit_batches` | Optional fast partitioning backend; more updates may improve stability but increase computation |
| UMAP | `fit_sample_n`, `transform_batch_size` | A smaller teacher fitting budget is faster; smaller transform batches reduce memory |
| Large embedding plots | sampling and filtering parameters | Fewer displayed cells improve rendering speed and readability |
| External in-memory analysis | selected cells, genes, and expression field | Smaller focused subsets reduce materialization and downstream memory |

Consult the corresponding API page for the exact meaning and default value of
each parameter.

```{note}
Increasing algorithm-specific accuracy settings cannot compensate for an
incorrectly defined read index. Confirm the selected cells, genes, and
expression field before tuning PCA or clustering parameters.
```

## Storage and System Considerations

Database and minibatch throughput may depend strongly on the storage device.

When possible:

- place the active `.sasql` Atlas on fast local storage;
- avoid heavily contended network file systems for performance-sensitive runs;
- ensure sufficient free disk space for imports and intermediate tables;
- avoid running several large imports or database-heavy analyses against the
  same device simultaneously.

More CPU threads do not always improve performance. A workflow may instead be
limited by:

- disk throughput;
- database execution;
- Python processing;
- dense matrix construction;
- memory bandwidth;
- model computation.

Tune one part of the workflow at a time and measure the result.

## Avoid Unnecessary Data Movement

Converting complete Atlas contents into AnnData, pandas, or other in-memory
objects can recreate the memory limitations that scAtlasPy is designed to
avoid.

Prefer:

- selecting only required metadata columns with `get_obs_df(columns=...)`;
- retrieving only the relevant cell population with `get_anndata(...)`;
- using SQL aggregation instead of returning complete tables;
- streaming expression data through minibatches;
- writing large predictions incrementally rather than accumulating all batches
  in a Python list.

For external methods, select a focused cell population and gene set before
creating an in-memory object.

## Troubleshooting Memory Pressure

When a workflow approaches or exceeds available memory:

1. Reduce the active cell population.
2. Reduce the selected genes or enable `use_hvg=True`.
3. Reduce `batch_size`.
4. Reduce `buffer_batch_num` in multi-pass mode.
5. Reduce import block size.
6. Reduce UMAP `fit_sample_n` or `transform_batch_size`.
7. Lower `db_memory_limit` to leave more memory for Python.
8. Avoid holding complete AnnData objects or accumulated batch outputs.
9. Close other large Atlas connections or analysis processes.

Check whether the memory is being used by DuckDB, Python, an external model, or
several components simultaneously before changing settings.

## Troubleshooting Low Throughput

When a workflow is stable but slower than expected:

1. Confirm that the Atlas is stored on sufficiently fast storage.
2. Reduce the active analysis view to the data actually required.
3. Increase `batch_size` gradually.
4. Increase import block size gradually.
5. Avoid rebuilding the same read index repeatedly.
6. Avoid unnecessary conversions between Atlas, AnnData, and DataFrame
   representations.
7. Check whether model computation, rather than data retrieval, is the
   bottleneck.
8. Enable informational logging to inspect available throughput messages.

```python
sap.set_verbosity("info")
```

Use repeatable measurements rather than judging performance from a single
batch, which may include initialization overhead.

## Troubleshooting Unstable Results

When iterative results vary more than expected:

1. Confirm the read-index definition.
2. Verify the selected expression representation.
3. Record random seeds used by scikit-learn, PyTorch, or other libraries.
4. Increase method-specific accuracy settings, such as PCA `n_iter` or distilled Louvain `fit_sample_n`, where appropriate.
5. Check whether the shuffle buffer provides sufficient mixing.
6. Confirm that labels remain aligned with expression batches.
7. Do not interpret a truncated single-pass run as a full-population result.

Numerical differences between minibatch and complete in-memory algorithms may
be expected. Their importance depends on the method and scientific objective.

## Measuring Performance

Measure separate workflow stages when possible:

- data import;
- preprocessing;
- read-index construction;
- minibatch retrieval;
- model fitting;
- full-Atlas transformation or inference;
- plotting;
- export.

Separating these stages helps determine whether a bottleneck comes from
storage, database operations, matrix reconstruction, or the downstream
algorithm.

Useful measurements include:

- elapsed time;
- peak resident memory;
- cells processed per second;
- batches processed per second;
- database size;
- temporary disk usage;
- time to the first batch;
- model fitting and transformation time.

## Reporting Benchmarks

When reporting performance results or regressions, include:

- scAtlasPy version or commit;
- operating system and Python version;
- number of cells and genes in the complete Atlas;
- number of cells and genes in the active read index;
- sparse density, when available;
- CPU, memory, and storage-device information;
- `db_memory_limit`;
- import mode and block settings;
- `cell_condition`, `gene_condition`, `use_hvg`, and `use_data`;
- `batch_size`, `buffer_batch_num`, `pass_mode`, and `max_batches`;
- algorithm-specific parameters;
- which workflow stages were included in the reported time.

These details make results easier to reproduce and help distinguish software
regressions from differences in data, hardware, storage, or configuration.

## Related Documentation

- {doc}`data-model` explains how Atlas data and expression representations are
  stored.
- {doc}`minibatch-architecture` explains minibatch reconstruction and shuffle
  behavior.
- {doc}`known-limitations` records current version-specific restrictions.
- {doc}`../api/atlas` documents read-index and minibatch interfaces.
- {doc}`../tutorials/advanced/index` provides worked examples of streaming
  statistics, model training, clustering, and inference.
