from __future__ import annotations

import numpy as np

import scatlaspy as sap

from .helpers import assert_dense_equal, dense_counts


def test_filter_and_qc_metrics_match_reference_counts(atlas_from_adata, counts_adata):
    atlas = atlas_from_adata(counts_adata)
    X = dense_counts(counts_adata)

    sap.pp.calculate_qc_metrics(atlas, qc_vars={"mt": "MT-", "ribo": "RPS"}, chunk_cells=3)
    sap.pp.filter_cells(atlas, min_counts=7, min_genes=3, chunk_cells=3)
    sap.pp.filter_genes(atlas, min_cells=3, min_counts=6)

    obs = atlas.get_obs_df()
    var = atlas.get_var_df()

    total_counts = X.sum(axis=1)
    n_genes = (X > 0).sum(axis=1)
    gene_counts = X.sum(axis=0)
    gene_cells = (X > 0).sum(axis=0)

    np.testing.assert_allclose(obs["cell_total_counts"], total_counts)
    np.testing.assert_array_equal(obs["n_genes_by_counts"], n_genes)
    np.testing.assert_allclose(obs["total_counts_mt"], X[:, 0])
    np.testing.assert_allclose(obs["pct_counts_mt"], X[:, 0] / total_counts * 100, rtol=1e-6)
    np.testing.assert_array_equal(obs["filter_cells"], (total_counts >= 7) & (n_genes >= 3))

    np.testing.assert_allclose(var["gene_total_counts"], gene_counts)
    np.testing.assert_array_equal(var["n_cells_by_counts"], gene_cells)
    np.testing.assert_array_equal(var["filter_genes"], (gene_cells >= 3) & (gene_counts >= 6))


def test_transformations_write_derived_tables_without_mutating_counts(atlas_from_adata, counts_adata):
    atlas = atlas_from_adata(counts_adata)
    before = atlas.query("SELECT * FROM X_HyS_data ORDER BY id")

    sap.pp.normalize_total(atlas, target_sum=10_000, chunk_cells=3)
    sap.pp.log1p(atlas)
    sap.pp.sqrt(atlas)
    sap.pp.normalize_and_log1p(atlas, target_sum=10_000, add_data="data_log1p_direct")

    tables = set(atlas.query("SHOW TABLES")["name"])
    assert {
        "X_HyS_data_data_normalize",
        "X_HyS_data_data_log1p",
        "X_HyS_data_data_sqrt",
        "X_HyS_data_data_log1p_direct",
    } <= tables
    assert "data_normalize" not in atlas.query("PRAGMA table_info('X_HyS_data')")["name"].tolist()

    after = atlas.query("SELECT * FROM X_HyS_data ORDER BY id")
    assert before.equals(after)

    X = dense_counts(counts_adata)
    expected_log1p = np.log1p(X / X.sum(axis=1, keepdims=True) * 10_000)
    expected_sqrt = np.sqrt(X)

    exported_log1p = atlas.get_anndata(list(range(X.shape[0])), use_data="data_log1p", include_obsm=False, include_varm=False)
    exported_sqrt = atlas.get_anndata(list(range(X.shape[0])), use_data="data_sqrt", include_obsm=False, include_varm=False)
    exported_direct = atlas.get_anndata(list(range(X.shape[0])), use_data="data_log1p_direct", include_obsm=False, include_varm=False)

    assert_dense_equal(exported_log1p.X.toarray(), expected_log1p)
    assert_dense_equal(exported_sqrt.X.toarray(), expected_sqrt)
    assert_dense_equal(exported_direct.X.toarray(), expected_log1p)


def test_hvg_and_scale_create_metadata_and_scaled_expression(atlas_from_adata, workflow_adata):
    atlas = atlas_from_adata(workflow_adata)

    sap.pp.normalize_and_log1p(atlas)
    sap.pp.highly_variable_genes(atlas, flavor="var", n_top_genes=5, use_filtered=False)
    sap.pp.scale(atlas, use_hvg=True)

    var = atlas.get_var_df()
    assert var["highly_variable_genes"].sum() == 5
    assert "zero_scale_transform" in var.columns

    tables = set(atlas.query("SHOW TABLES")["name"])
    assert "X_HyS_data_data_scale" in tables

    atlas.build_read_index(cell_condition=None, gene_condition=None, use_hvg=True, use_data="data_scale")
    batch = next(atlas.get_minibatch_dense(batch_size=7, max_batches=1))
    assert batch.shape == (7, 5)
    assert batch.dtype == np.float32
