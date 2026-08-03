<p align="center">
  <img src="docs/_static/img/scAtlas_full_paths.svg" alt="scAtlasPy logo" width="320">
</p>

# scAtlasPy

**A scalable Python platform for atlas-scale single-cell omics analysis beyond in-memory limits.**

scAtlasPy is a Python platform for analyzing cell atlases that are too large to
fit in memory. It extends familiar single-cell analysis workflows to
atlas-scale datasets, supporting preprocessing, dimensionality reduction,
clustering, visualization, marker analysis, and machine learning within a
unified on-disk environment.

At the center of scAtlasPy is a persistent `.sasql` atlas database. Expression
matrices, metadata, embeddings, and analysis results remain on disk, while
analysis functions retrieve only the cells, genes, or minibatches needed for the
current operation.

## Installation

Install the released package from PyPI:

```bash
pip install scatlaspy
```

Distilled UMAP and distilled Louvain clustering use PyTorch. Install `torch` in
the same environment if you plan to run these tools:

```bash
pip install torch
```

For CUDA, MPS, or other accelerator-specific PyTorch builds, follow the
installation command recommended for your hardware by the PyTorch project.

## Quick Start

```python
import scatlaspy as sap

atlas = sap.Atlas("pbmc.sasql")
atlas.load_h5ad("pbmc.h5ad", load_type="random")

sap.pp.calculate_qc_metrics(atlas)
sap.pp.filter_cells(atlas, min_genes=200)
sap.pp.filter_genes(atlas, min_cells=3)
sap.pp.normalize_and_log1p(atlas)
sap.pp.highly_variable_genes(atlas, n_top_genes=2000)

# Use mode="center_only" if PCA should preserve gene-level variance differences.
sap.pp.scale(atlas)

atlas.build_read_index(
    cell_condition="filter_cells",
    gene_condition="filter_genes",
    use_hvg=True,
    use_data="data_scale",
)

sap.tl.pca(atlas)
sap.tl.umap(atlas)
sap.tl.graph_clustering(atlas)
sap.pl.umap(atlas, color="scatlas_cluster")

atlas.close()
```

## What scAtlasPy Enables

- **Full-resolution atlas workflows:** run QC, filtering, normalization, HVG,
  PCA, clustering, UMAP, marker ranking, and visualization without loading the
  full matrix into memory.
- **High-throughput data retrieval:** stream sparse or dense minibatches from
  disk-resident atlases for downstream algorithms and machine-learning models.
- **Persistent analysis state:** store metadata, transformed expression values,
  embeddings, loadings, clusters, marker statistics, and plots inputs in one
  atlas database.
- **Interoperability:** import from AnnData-compatible `.h5ad` files and export
  selected results back to the broader single-cell ecosystem.
- **Extensibility:** build new atlas-scale methods on top of stable metadata,
  SQL, sparse retrieval, dense minibatch, and result-writing interfaces.

## Documentation

- Documentation: https://scatlaspy.readthedocs.io
- API reference: https://scatlaspy.readthedocs.io/en/latest/api/
- Tutorials: https://scatlaspy.readthedocs.io/en/latest/tutorials/

## Citation

If you use scAtlasPy in academic work, please cite the project repository for
now. A formal citation will be added when a paper or archived release is
available.

## License

scAtlasPy is released under the BSD 3-Clause License.
