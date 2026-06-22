# Preprocessing: `sap.pp`

## Overview

`sap.pp` provides functions for quality control, cell and gene filtering,
library-size normalization, expression transformation, highly variable gene
selection, and scaling.

Preprocessing functions usually update the current Atlas by writing metadata
columns or expression fields to the `.sasql` database. They do not create a new
in-memory `AnnData` object.

```{note}
Most preprocessing results are stored alongside the original data rather than
replacing it. Consult the individual function entry for the fields written,
required inputs, and behavior when an output already exists.
```

## Quality Control

```{eval-rst}
.. currentmodule:: scatlaspy

.. autosummary::
   :toctree: generated/
   :nosignatures:

   pp.calculate_qc_metrics
   pp.calculate_cell_total_counts
   pp.calculate_gene_total_counts
```

These functions calculate cell- and gene-level summary statistics used for
quality assessment and filtering.

Common stored outputs include:

- `obs.cell_total_counts`;
- `obs.n_genes_by_counts`;
- `var.gene_total_counts`;
- `var.n_cells_by_counts`;
- additional metrics for configured quality-control gene sets.

Use `calculate_qc_metrics()` for the standard QC workflow. The separate
cell- and gene-total functions are useful when only those summaries are needed.

## Cell and Gene Filtering

```{eval-rst}
.. currentmodule:: scatlaspy

.. autosummary::
   :toctree: generated/
   :nosignatures:

   pp.filter_cells
   pp.filter_genes
```

Filtering functions record whether cells and genes satisfy the requested
criteria:

- cell filtering results are stored in `obs.filter_cells`;
- gene filtering results are stored in `var.filter_genes`.

```{important}
Filtering marks cells and genes rather than immediately deleting their original
data. The resulting columns can be used when constructing analysis views with
`atlas.build_read_index()`.
```

## Normalization

```{eval-rst}
.. currentmodule:: scatlaspy

.. autosummary::
   :toctree: generated/
   :nosignatures:

   pp.normalize_total_scale_factor
   pp.normalize_total
```

These functions calculate library-size scale factors and generate normalized
expression values.

Common outputs include:

- `obs.scale_factor`;
- a normalized expression field stored in the Atlas.

Use the individual API entries to check the default target total, source
expression field, output field, and replacement behavior.

## Expression Transformation

```{eval-rst}
.. currentmodule:: scatlaspy

.. autosummary::
   :toctree: generated/
   :nosignatures:

   pp.log1p
   pp.expm1
   pp.sqrt
   pp.normalize_and_log1p
```

These functions transform stored expression values.

- `log1p()` applies the natural logarithm after adding one.
- `expm1()` applies the inverse transformation.
- `sqrt()` applies a square-root transformation.
- `normalize_and_log1p()` performs library-size normalization followed by
  `log1p` transformation in one workflow.

A common output of `normalize_and_log1p()` is:

```text
X_HyS_data.data_log1p
```

```{warning}
Avoid applying normalization or `log1p` more than once. Check the source
expression field before running a transformation.
```

## Highly Variable Genes

```{eval-rst}
.. currentmodule:: scatlaspy

.. autosummary::
   :toctree: generated/
   :nosignatures:

   pp.highly_variable_genes
```

`highly_variable_genes()` identifies genes with high variability relative to
their expression level.

Common stored outputs include:

- `var.highly_variable_genes`;
- related `var.hvg_*` statistics.

The selected genes can be used in a later analysis view:

```python
atlas.build_read_index(
    cell_condition="filter_cells",
    gene_condition="filter_genes",
    use_hvg=True,
    use_data="data_log1p",
)
```

## Scaling

```{eval-rst}
.. currentmodule:: scatlaspy

.. autosummary::
   :toctree: generated/
   :nosignatures:

   pp.scale
```

`scale()` centers and standardizes expression values for analyses that require
features on comparable scales.

A common output field is:

```text
X_HyS_data.data_scale
```

```{note}
Scaled values no longer represent the original expression magnitude. For
marker-gene visualization and biological interpretation, log-normalized values
are usually easier to interpret.
```

## Typical Order

A common preprocessing sequence is:

```python
sap.pp.calculate_qc_metrics(atlas)

sap.pp.filter_cells(
    atlas,
    min_genes=200,
)

sap.pp.filter_genes(
    atlas,
    min_cells=3,
)

sap.pp.normalize_and_log1p(
    atlas,
    target_sum=1e4,
)

sap.pp.highly_variable_genes(
    atlas,
    n_top_genes=2000,
)

sap.pp.scale(atlas)
```

Not every workflow requires every step. The appropriate expression
representation and whether scaling is needed depend on the downstream method.

## Related Documentation

- {doc}`../tutorials/basic/index` provides the standard preprocessing workflow.
- {doc}`atlas` documents `Atlas.build_read_index()` and the active analysis
  view.
- {doc}`plotting` documents QC and highly variable gene visualizations.
- {doc}`tools` documents downstream dimensionality reduction, clustering, and
  marker analysis.