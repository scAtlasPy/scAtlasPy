from __future__ import annotations

import numpy as np
import pytest
from sklearn.decomposition import IncrementalPCA

sap = pytest.importorskip("scatlaspy")

from .data_generators import make_random_counts_anndata

sc = pytest.importorskip("scanpy")


def _ids(atlas):
    return atlas.query("SELECT atlas_cell_id FROM obs ORDER BY atlas_cell_id")[
        "atlas_cell_id"
    ].astype(int).tolist()


def _positive_reference_adata(seed=2024, n_cells=1200, n_genes=80):
    adata, params = make_random_counts_anndata(
        seed=seed, n_cells=n_cells, n_genes=n_genes, density=0.08, dtype="float32"
    )
    X = adata.X.tolil()
    # Ensure every cell and every gene has at least one count.
    for i in range(n_cells):
        X[i, i % n_genes] = X[i, i % n_genes] + 1
    for j in range(n_genes):
        X[j % n_cells, j] = X[j % n_cells, j] + 1
    adata.X = X.tocsr().astype(np.float32)
    return adata, params


@pytest.mark.l2
@pytest.mark.reference
def test_filter_cells_and_genes_match_scipy_counts(tmp_path, write_generation_params):
    adata, params = _positive_reference_adata()
    write_generation_params("l2_filter_reference", params)
    atlas = sap.Atlas(tmp_path / "filter_reference.sasql")
    atlas.load_anndata(adata)

    min_counts = 12
    min_genes = 3
    min_cells = 20
    min_gene_counts = 30
    sap.pp.filter_cells(atlas, min_counts=min_counts, min_genes=min_genes)
    sap.pp.filter_genes(atlas, min_counts=min_gene_counts, min_cells=min_cells)

    X = adata.X.tocsr()
    expected_cells = np.asarray(
        (X.sum(axis=1).A1 >= min_counts) & (X.getnnz(axis=1) >= min_genes)
    )
    expected_genes = np.asarray(
        (X.sum(axis=0).A1 >= min_gene_counts) & (X.getnnz(axis=0) >= min_cells)
    )

    obs = atlas.query("SELECT atlas_cell_id, filter_cells FROM obs ORDER BY atlas_cell_id")
    var = atlas.query("SELECT atlas_gene_id, filter_genes FROM var ORDER BY atlas_gene_id")
    np.testing.assert_array_equal(obs["filter_cells"].to_numpy(dtype=bool), expected_cells)
    np.testing.assert_array_equal(var["filter_genes"].to_numpy(dtype=bool), expected_genes)
    atlas.close()


@pytest.mark.l2
@pytest.mark.reference
def test_normalize_total_and_log1p_match_scanpy(tmp_path):
    adata, _ = _positive_reference_adata(seed=2025)
    h5ad_path = tmp_path / "norm_log_source.h5ad"
    adata.write_h5ad(h5ad_path)

    atlas = sap.Atlas(tmp_path / "norm_log.sasql")
    atlas.load_h5ad(
        h5ad_path,
        load_type="order",
        cells_per_block=250,
        blocks_per_pool=2,
    )

    target_sum = 1e4
    sap.pp.normalize_total(atlas, target_sum=target_sum, chunk_cells=300)
    sap.pp.log1p(atlas, use_data="data_normalize", add_data="data_log1p")

    ref = adata.copy()
    sc.pp.normalize_total(ref, target_sum=target_sum)
    sc.pp.log1p(ref)

    out = atlas.get_anndata(
        _ids(atlas), use_data="data_log1p", include_obsm=False, include_varm=False
    )
    np.testing.assert_allclose(
        out.X.toarray(), ref.X.toarray(), rtol=2e-5, atol=2e-5
    )
    atlas.close()


@pytest.mark.l2
@pytest.mark.reference
def test_scale_nonzero_records_match_manual_gene_statistics(tmp_path):
    adata, _ = _positive_reference_adata(seed=2026, n_cells=1000, n_genes=50)
    atlas = sap.Atlas(tmp_path / "scale_reference.sasql")
    atlas.load_anndata(adata)
    sap.pp.normalize_total(atlas, target_sum=1e4, chunk_cells=250)
    sap.pp.log1p(atlas)
    sap.pp.scale(atlas, use_data="data_log1p", use_hvg=False, max_value=10.0)

    ref = adata.copy()
    sc.pp.normalize_total(ref, target_sum=1e4)
    sc.pp.log1p(ref)
    dense = ref.X.toarray().astype(np.float64)
    means = dense.mean(axis=0)
    stds = np.sqrt(np.maximum((dense * dense).mean(axis=0) - means * means, 0.0))

    stored = atlas.query(
        """
        SELECT atlas_cell_id, atlas_gene_id, data_log1p, data_scale
        FROM X_HyS_data
        WHERE data_scale IS NOT NULL
        ORDER BY id
        """
    )
    expected = []
    for _, row in stored.iterrows():
        gene = int(row["atlas_gene_id"])
        if stds[gene] > 0:
            value = (float(row["data_log1p"]) - means[gene]) / stds[gene]
            value = min(10.0, max(-10.0, value))
        else:
            value = 0.0
        expected.append(value)

    np.testing.assert_allclose(
        stored["data_scale"].to_numpy(dtype=np.float64),
        np.asarray(expected),
        rtol=2e-4,
        atol=2e-4,
    )
    atlas.close()


@pytest.mark.l2
@pytest.mark.reference
def test_hvg_selects_requested_number_and_pca_outputs_are_well_formed(tmp_path):
    adata, _ = _positive_reference_adata(seed=2027, n_cells=1000, n_genes=60)
    atlas = sap.Atlas(tmp_path / "hvg_pca.sasql")
    atlas.load_anndata(adata)
    sap.pp.normalize_total(atlas, target_sum=1e4)
    sap.pp.log1p(atlas)
    sap.pp.highly_variable_genes(
        atlas,
        flavor="var",
        n_top_genes=20,
        use_filtered=False,
        use_data="data_log1p",
    )
    hvg_count = atlas.query(
        "SELECT COUNT(*) AS n FROM var WHERE highly_variable_genes = TRUE"
    )["n"].iloc[0]
    assert hvg_count == 20

    sap.pp.scale(atlas, use_data="data_log1p", use_hvg=True)
    atlas.build_read_index(use_hvg=True, use_data="data_scale")
    sap.tl.pca(atlas, n_components=5, fit_batches=3, buffer_batch_num=2)

    pca_rows = atlas.query("SELECT COUNT(*) AS n FROM obsm_X_pca")["n"].iloc[0]
    pcs_rows = atlas.query("SELECT COUNT(*) AS n FROM varm_PCs")["n"].iloc[0]
    assert pca_rows == adata.n_obs
    assert pcs_rows == hvg_count

    # A lightweight mature-implementation check on the same dense subset.
    selected_genes = atlas.query(
        "SELECT atlas_gene_id FROM var WHERE highly_variable_genes = TRUE ORDER BY filter_gene_id"
    )["atlas_gene_id"].astype(int).tolist()
    ref_dense = adata.X[:, selected_genes].toarray().astype(np.float32)
    ipca = IncrementalPCA(n_components=5)
    ipca.fit(ref_dense)
    assert ipca.components_.shape == (5, hvg_count)
    atlas.close()
