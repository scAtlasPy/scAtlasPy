<p align="center">
  <img src="docs/_static/img/scAtlas_full_paths.svg" alt="scAtlasPy logo" width="320">
</p>

# scAtlasPy

scAtlasPy is an on-disk single-cell atlas analysis platform designed to make full-scale human cell atlas analysis feasible on memory-limited local hardware. As single-cell atlases grow from millions to hundreds of millions of cells, conventional in-memory workflows become increasingly constrained by RAM capacity. scAtlasPy addresses this bottleneck with compressed disk-resident data management, high-throughput random data streaming, and an extensible analysis layer for atlas-scale preprocessing, visualization, statistics, and machine learning workflows.

scAtlasPy keeps expression matrices, metadata, embeddings, and analysis results in a persistent `.sasql` atlas database and retrieves only the data needed for each operation. This design supports comprehensive analysis pipelines under strict memory constraints while remaining compatible with familiar Python single-cell analysis workflows.

scAtlasPy provides a familiar Python workflow while keeping the atlas on disk:

```python
import scatlaspy as sap

atlas = sap.Atlas("pbmc.sasql")
atlas.load_h5ad("pbmc.h5ad", load_type="order")

sap.pp.calculate_qc_metrics(atlas)
sap.pp.filter_cells(atlas, min_genes=200)
sap.pp.filter_genes(atlas, min_cells=3)
sap.pp.normalize_total(atlas)
sap.pp.log1p(atlas)
sap.pp.highly_variable_genes(atlas, n_top_genes=2000)
# Default mode="center_and_scale" normalizes gene scales before PCA.
# Use mode="center_only" to preserve gene-level variance differences.
sap.pp.scale(atlas)

atlas.build_read_index(
    cell_condition="filter_cells",
    gene_condition="filter_genes",
    use_hvg=True,
    use_data="data_scale",
)

sap.tl.pca(atlas)
sap.tl.graph_clustering(atlas)
sap.pl.pca(atlas, color="scatlas_cluster")
```

## Highlights

- Persistent `.sasql` atlas files built on an embedded analytical database. Run local, serverless analyses with compact on-disk storage.
- Chunked conversion from diverse single-cell source formats into `.sasql`. Data ingestion does not require full in-memory loading.
- Full-spectrum atlas analysis under memory constraints. This includes QC, filtering, normalization, feature selection, PCA, clustering, UMAP, marker ranking, annotation, and visualization.
- Flexible data retrieval for interactive analysis and atlas-scale algorithm development. Use SQL queries, metadata access, AnnData export, or minibatch streaming.
- Extensible architecture for emerging atlas-oriented computational methods. It is designed to support machine learning workflows that need high-throughput random access to large cell collections.
- A community-facing foundation for an open cell atlas analysis ecosystem across storage, retrieval, algorithms, and applications.

## Installation

The recommended way to install the released version of scAtlasPy is through PyPI:

```bash
pip install scatlaspy
```

To install the latest development version directly from GitHub:

```bash
git clone https://github.com/GaoLabXDU/scatlaspy.git
cd scatlaspy
pip install -e .
```

You can verify the installation with:

```bash
python -c "import scatlaspy as sap; print(sap.Atlas)"
```

## Documentation

- Documentation: https://scatlaspy.readthedocs.io
- API reference: https://scatlaspy.readthedocs.io/en/latest/api/
- Tutorials and examples: https://scatlaspy.readthedocs.io/en/latest/tutorials/

## Quick Start

Create or open an Atlas database:

```python
import scatlaspy as sap

sap.set_verbosity("info")
atlas = sap.Atlas("data/pbmc.sasql", db_memory_limit="8GB")
```

Import a single `.h5ad` file:

```python
atlas.load_h5ad(
    "data/pbmc.h5ad",
    load_type="order",
    cells_per_block=1000,  # number of cells processed per import block
)
```

Import multiple `.h5ad` files with randomized block loading:

```python
atlas.load_h5ad(
    ["data/batch1.h5ad", "data/batch2.h5ad"],
    load_type="random",
    cells_per_block=1000,  # number of cells in each source block
)
```

Run a typical preprocessing and analysis workflow:

```python
sap.pp.calculate_qc_metrics(atlas)
sap.pp.filter_cells(atlas, min_genes=200, max_genes=6000)
sap.pp.filter_genes(atlas, min_cells=3)

sap.pp.normalize_total(atlas, target_sum=1e4)
sap.pp.log1p(atlas)
sap.pp.highly_variable_genes(atlas, n_top_genes=2000)
sap.pp.scale(atlas)

atlas.build_read_index(
    cell_condition="filter_cells",
    gene_condition="filter_genes",
    use_hvg=True,
    use_data="data_scale",
)

sap.tl.pca(atlas)
sap.tl.graph_clustering(atlas)
sap.tl.umap(atlas)
sap.tl.rank_genes_groups(atlas, groupby="scatlas_cluster")
```

Visualize results:

```python
sap.pl.violin_qc_metrics(atlas)
sap.pl.pca(atlas, color="scatlas_cluster")
sap.pl.umap(atlas, color="scatlas_cluster")
sap.pl.rank_genes_groups(atlas)
sap.pl.dotplot(atlas, genes=["MS4A1", "CD3D", "LYZ"], groupby="scatlas_cluster")
```

Close the database connection when finished:

```python
atlas.close()
```

## Data Model

An Atlas database is a DuckDB file with the `.sasql` extension. scAtlasPy stores core single-cell components as database tables:

| Component | Stored as | Typical content |
| --- | --- | --- |
| `obs` | table | Cell metadata, QC metrics, filtering flags, cluster labels |
| `var` | table | Gene metadata, QC metrics, filtering flags, HVG flags |
| expression matrix | sparse tables | Count, normalized, log-transformed, scaled, or transformed matrices |
| `obsm_*` | tables | Cell embeddings such as PCA and UMAP coordinates |
| `varm_*` | tables | Gene loadings such as PCA loadings |
| `uns_*` | tables | Analysis summaries and parameters |

## API Overview

### Core

- `sap.set_verbosity(...)`: configure scAtlasPy logging output.
- `sap.Atlas(file_name, db_memory_limit="3G")`: create or open a `.sasql` database.
- `atlas.load_h5ad(...)`: import one or more `.h5ad` files.
- `atlas.load_anndata(adata)`: import an in-memory `AnnData` object.
- `atlas.load_multi_format(...)`: import supported source formats into an Atlas database.
- `atlas.get_anndata(...)`: materialize selected cells as `AnnData`.
- `atlas.write_h5ad(...)`: export the Atlas database to `.h5ad`.
- `atlas.query(...)` / `atlas.execute_sql(...)`: run SQL against the database.
- `atlas.build_read_index(...)`: build a filtered cell/gene read index for downstream analysis.
- `atlas.close()`: close the DuckDB connection.

### Preprocessing: `sap.pp`

- QC and filtering: `calculate_qc_metrics`, `calculate_cell_total_counts`, `calculate_gene_total_counts`, `filter_cells`, `filter_genes`
- Transformations: `normalize_total`, `normalize_total_scale_factor`, `log1p`, `sqrt`, `scale`
- Feature selection: `highly_variable_genes`
- Convenience workflow: `normalize_and_log1p`

### Tools: `sap.tl`

- Dimensionality reduction: `pca`, `umap`
- Clustering: `graph_clustering`, `kmeans`
- Marker analysis: `rank_genes_groups`
- Annotation: `manual_annotate_clusters`

### Plotting: `sap.pl`

- QC plots: `highest_expr_genes`, `violin_qc_metrics`, `scatter_qc_metrics`
- Embeddings and model diagnostics: `pca`, `pca_loadings`, `pca_variance_ratio`, `pca_variance_ratio_cumsum`, `umap`
- Cluster and marker plots: `cluster_size`, `rank_genes_groups`, `rank_genes_groups_volcano`, `rank_genes_groups_violin`
- Expression summaries: `dotplot`, `violin`, `stacked_violin`, `highly_variable_genes`

## Working With Large Datasets

scAtlasPy is designed around chunked reads and writes. For large `.h5ad` files:

- Increase `cells_per_block` for faster import when memory allows.
- Decrease `cells_per_block` or `batch_cells` to reduce peak memory.
- Use `db_memory_limit` when creating `Atlas` to limit DuckDB query memory.
- Build a read index after filtering before running PCA, UMAP, or minibatch workflows.
- Prefer database-side SQL summaries when you only need metadata or aggregated results.

Example:

```python
atlas = sap.Atlas("large_dataset.sasql", db_memory_limit="16GB")
atlas.load_h5ad("large_dataset.h5ad", cells_per_block=2000)

sap.pp.filter_cells(atlas, min_genes=200)
sap.pp.filter_genes(atlas, min_cells=3)
sap.pp.normalize_total(atlas)
sap.pp.log1p(atlas)
sap.pp.highly_variable_genes(atlas, n_top_genes=3000)
sap.pp.scale(atlas)

atlas.build_read_index(use_hvg=True, use_data="data_scale")
sap.tl.pca(atlas)
```

## Development

Clone the repository and install in editable mode:

```bash
git clone https://github.com/GaoLabXDU/scatlaspy.git
cd scatlaspy
pip install -e ".[test]"
```

Run tests:

```bash
pytest
```

Run a quick import check:

```bash
python -c "import scatlaspy as sap; print(sap.Atlas)"
```

## Citation

If you use scAtlasPy in academic work, please cite the project repository for now. A formal citation will be added when a paper or archived release is available.

## License

scAtlasPy is released under the BSD 3-Clause License.
