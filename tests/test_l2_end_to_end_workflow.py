from __future__ import annotations

import numpy as np
import pytest

sap = pytest.importorskip("scatlaspy")
ad = pytest.importorskip("anndata")

from .data_generators import make_random_counts_anndata


def _workflow_adata(seed: int = 6101):
    adata, params = make_random_counts_anndata(
        seed=seed,
        n_cells=240,
        n_genes=48,
        density=0.12,
        dtype="float32",
        max_count=15,
    )
    X = adata.X.tolil()
    for i in range(adata.n_obs):
        X[i, i % adata.n_vars] = X[i, i % adata.n_vars] + 2
    for j in range(adata.n_vars):
        X[j % adata.n_obs, j] = X[j % adata.n_obs, j] + 2
    adata.X = X.tocsr().astype(np.float32)
    return adata, params


@pytest.mark.l2
@pytest.mark.reference
def test_file_import_full_analysis_workflow_and_h5ad_export(
    tmp_path, write_generation_params
):
    adata, params = _workflow_adata()
    write_generation_params("l2_full_workflow", params)
    source_h5ad = tmp_path / "workflow_source.h5ad"
    exported_h5ad = tmp_path / "workflow_exported.h5ad"
    adata.write_h5ad(source_h5ad)

    atlas = sap.Atlas(tmp_path / "workflow.sasql")
    atlas.load_h5ad(
        source_h5ad,
        load_type="order",
        cells_per_block=53,
        blocks_per_pool=2,
    )

    sap.pp.calculate_qc_metrics(atlas, chunk_cells=80)
    sap.pp.filter_cells(atlas, min_counts=2, min_genes=1, chunk_cells=80)
    sap.pp.filter_genes(atlas, min_counts=2, min_cells=1)
    sap.pp.normalize_total(atlas, target_sum=1e4, chunk_cells=80)
    sap.pp.log1p(atlas, use_data="data_normalize", add_data="data_log1p")
    sap.pp.highly_variable_genes(
        atlas,
        flavor="var",
        n_top_genes=20,
        use_data="data_log1p",
        use_filtered=False,
    )
    sap.pp.scale(atlas, use_data="data_log1p", use_hvg=True, max_value=10.0)
    atlas.build_read_index(
        cell_condition="filter_cells",
        gene_condition="filter_genes",
        use_hvg=True,
        use_data="data_scale",
    )

    sap.tl.pca(atlas, n_components=5, fit_batches=2, buffer_batch_num=2)
    sap.tl.kmeans(
        atlas,
        n_components=5,
        n_clusters=3,
        batch_size=64,
        fit_batches=2,
        buffer_batch_num=2,
        use_obs_col="kmeans_e2e",
    )

    n_obs = atlas.query("SELECT COUNT(*) AS n FROM obs")["n"].iloc[0]
    n_hvg = atlas.query(
        "SELECT COUNT(*) AS n FROM var WHERE highly_variable_genes = TRUE"
    )["n"].iloc[0]
    pca_rows = atlas.query("SELECT COUNT(*) AS n FROM obsm_X_pca")["n"].iloc[0]
    pcs_rows = atlas.query("SELECT COUNT(*) AS n FROM varm_PCs")["n"].iloc[0]
    kmeans_rows = atlas.query(
        "SELECT COUNT(*) AS n FROM obs WHERE kmeans_e2e IS NOT NULL"
    )["n"].iloc[0]

    assert n_obs == adata.n_obs
    assert n_hvg == 20
    assert pca_rows == adata.n_obs
    assert pcs_rows == 20
    assert kmeans_rows == adata.n_obs

    atlas.write_h5ad(exported_h5ad, batch_cells=128)
    exported = ad.read_h5ad(exported_h5ad, backed="r")
    try:
        assert exported.n_obs == adata.n_obs
        assert exported.n_vars == adata.n_vars
        assert "X_pca" in exported.obsm
        assert "PCs" in exported.varm
    finally:
        exported.file.close()

    atlas.close()


@pytest.mark.l2
@pytest.mark.reference
def test_file_import_workflow_can_train_pca_directly_on_log1p(tmp_path):
    adata, _ = _workflow_adata(seed=6102)
    source_h5ad = tmp_path / "workflow_log1p_source.h5ad"
    adata.write_h5ad(source_h5ad)

    atlas = sap.Atlas(tmp_path / "workflow_log1p.sasql")
    atlas.load_h5ad(
        source_h5ad,
        load_type="order",
        cells_per_block=47,
        blocks_per_pool=2,
    )

    sap.pp.filter_cells(atlas, min_counts=2, min_genes=1, chunk_cells=80)
    sap.pp.filter_genes(atlas, min_counts=2, min_cells=1)
    sap.pp.normalize_total(atlas, target_sum=1e4, chunk_cells=80)
    sap.pp.log1p(atlas, use_data="data_normalize", add_data="data_log1p")

    atlas.build_read_index(
        cell_condition="filter_cells",
        gene_condition="filter_genes",
        use_hvg=False,
        use_data="data_log1p",
    )

    sap.tl.pca(atlas, n_components=5, fit_batches=2, buffer_batch_num=2)
    sap.tl.kmeans(
        atlas,
        n_components=5,
        n_clusters=3,
        batch_size=64,
        fit_batches=2,
        buffer_batch_num=2,
        use_obs_col="kmeans_log1p",
    )

    pca_rows = atlas.query("SELECT COUNT(*) AS n FROM obsm_X_pca")["n"].iloc[0]
    pcs_rows = atlas.query("SELECT COUNT(*) AS n FROM varm_PCs")["n"].iloc[0]
    indexed_genes = atlas.query(
        "SELECT COUNT(*) AS n FROM var WHERE filter_gene_id IS NOT NULL"
    )["n"].iloc[0]
    kmeans_rows = atlas.query(
        "SELECT COUNT(*) AS n FROM obs WHERE kmeans_log1p IS NOT NULL"
    )["n"].iloc[0]

    assert pca_rows == adata.n_obs
    assert pcs_rows == indexed_genes
    assert kmeans_rows == adata.n_obs
    atlas.close()
