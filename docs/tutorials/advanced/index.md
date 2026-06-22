# Advanced Tutorials

The advanced tutorials cover focused workflows beyond the basic single-cell
analysis pipeline. They are intended for users who already have an Atlas and
want to inspect or resume an analysis, query stored data, perform incremental
computation, or use scAtlasPy as the data foundation for their own
computational methods.

These tutorials are organized by topic rather than as a single linear
workflow. Within the streaming computation and method-development sections,
the pages are arranged from simpler examples to more complete implementations.

## Atlas Management and Inspection

Use these tutorials to reopen an existing Atlas, inspect stored results, and
customize the visualization of analysis outputs.

- {doc}`resume-existing-atlas`  
  Reconnect an existing `.sasql` Atlas, inspect its contents, and continue an
  analysis in a new Python session.

- {doc}`visualize-analysis-results`  
  Customize common plotting parameters and inspect quality-control,
  dimensionality-reduction, clustering, and marker-analysis results.

## Query Atlas Data

- {doc}`query-atlas-with-sql`  
  Use SQL to summarize metadata, inspect analysis results, retrieve selected
  records, and support custom downstream analyses.

## Streaming Statistics

These tutorials introduce incremental statistical computation on expression
data that are too large to materialize as a complete cell-by-gene matrix.

- {doc}`stream-mean-and-variance`  
  Compute means and variances in a single pass using Welford's online
  algorithm.

- {doc}`stream-covariance-matrix`  
  Extend block-wise computation to calculate a covariance matrix from streamed
  expression data.

Start with the mean-and-variance tutorial before continuing to streamed
covariance.

## Custom Method Development

These tutorials demonstrate how scAtlasPy can provide minibatches, repeated
data passes, and full-atlas inference for user-defined statistical and
machine-learning methods.

- {doc}`train-logistic-regression-with-minibatches`  
  Train a logistic-regression model using expression data retrieved in
  minibatches.

- {doc}`train-pytorch-model-with-minibatches`  
  Train a neural network with PyTorch using randomized, multi-pass expression
  streams.

- {doc}`implement-minibatch-kmeans`  
  Implement an iterative minibatch clustering method using repeated access to
  atlas-scale expression data.

- {doc}`apply-model-to-full-atlas`  
  Apply a trained model to the complete Atlas in batches and collect or store
  the resulting predictions.

The logistic-regression tutorial provides the simplest introduction to
minibatch model training. The PyTorch and minibatch KMeans tutorials then show
how the same data-access interfaces can support different computational
frameworks and algorithmic structures. The full-Atlas prediction tutorial
completes the workflow by applying a trained method across the Atlas without
loading the complete expression matrix into memory.

```{toctree}
:hidden:
:maxdepth: 1

resume-existing-atlas
visualize-analysis-results
query-atlas-with-sql
stream-mean-and-variance
stream-covariance-matrix
train-logistic-regression-with-minibatches
train-pytorch-model-with-minibatches
implement-minibatch-kmeans
apply-model-to-full-atlas
```
