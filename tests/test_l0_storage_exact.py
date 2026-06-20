from __future__ import annotations

import shutil

import numpy as np
import pytest

sap = pytest.importorskip("scatlaspy")

from .data_generators import make_l0_edge_anndata, make_random_counts_anndata


def _all_cell_ids(atlas) -> list[int]:
    return atlas.query("SELECT atlas_cell_id FROM obs ORDER BY atlas_cell_id")[
        "atlas_cell_id"
    ].astype(int).tolist()


def _assert_dense_equal(actual, expected, *, atol=1e-5):
    np.testing.assert_allclose(
        np.asarray(actual, dtype=np.float32),
        np.asarray(expected, dtype=np.float32),
        rtol=1e-5,
        atol=atol,
        equal_nan=True,
    )


@pytest.mark.l0
def test_round_trip_preserves_shape_metadata_values_and_reopen(
    atlas_path, write_generation_params
):
    adata, params = make_l0_edge_anndata(dtype="float32")
    adata.X.data[np.isnan(adata.X.data)] = -123.0
    write_generation_params("l0_edge", params)

    atlas = sap.Atlas(atlas_path)
    atlas.load_anndata(adata)
    ids = _all_cell_ids(atlas)
    out = atlas.get_anndata(ids, use_data="data", include_obsm=False, include_varm=False)

    assert out.shape == adata.shape
    assert out.obs["atlas_cell_name"].tolist() == list(adata.obs_names)
    assert out.var["atlas_gene_name"].tolist() == list(adata.var_names)
    _assert_dense_equal(out.X.toarray(), adata.X.toarray())

    atlas.close()
    reopened = sap.Atlas(atlas_path)
    reopened_ids = _all_cell_ids(reopened)
    reopened_out = reopened.get_anndata(
        reopened_ids, use_data="data", include_obsm=False, include_varm=False
    )
    _assert_dense_equal(reopened_out.X.toarray(), adata.X.toarray())
    reopened.close()


@pytest.mark.l0
@pytest.mark.xfail(reason="NaN values are currently not preserved by the storage round trip")
def test_nan_values_are_preserved_as_known_future_contract(atlas_path):
    adata, _ = make_l0_edge_anndata(dtype="float32")
    atlas = sap.Atlas(atlas_path)
    atlas.load_anndata(adata)
    out = atlas.get_anndata(
        _all_cell_ids(atlas), use_data="data", include_obsm=False, include_varm=False
    )
    _assert_dense_equal(out.X.toarray(), adata.X.toarray())
    atlas.close()


@pytest.mark.l0
def test_non_contiguous_cell_selection_preserves_requested_order(atlas_path):
    adata, _ = make_random_counts_anndata(
        seed=11, n_cells=12, n_genes=8, density=0.35, dtype="float32"
    )
    atlas = sap.Atlas(atlas_path)
    atlas.load_anndata(adata)

    requested = [7, 1, 10, 3]
    out = atlas.get_anndata(
        requested, use_data="data", include_obsm=False, include_varm=False
    )

    assert out.obs["atlas_cell_id"].astype(int).tolist() == requested
    _assert_dense_equal(out.X.toarray(), adata.X[requested].toarray())
    atlas.close()


@pytest.mark.l0
def test_contiguous_slice_matches_source_matrix(atlas_path):
    adata, _ = make_random_counts_anndata(
        seed=12, n_cells=15, n_genes=9, density=0.3, dtype="float32"
    )
    atlas = sap.Atlas(atlas_path)
    atlas.load_anndata(adata)

    start, end = 4, 11
    requested = list(range(start, end))
    out = atlas.get_anndata(
        requested, use_data="data", include_obsm=False, include_varm=False
    )

    _assert_dense_equal(out.X.toarray(), adata.X[start:end].toarray())
    atlas.close()


@pytest.mark.l0
def test_cell_and_gene_filtering_reindexes_data_and_metadata(atlas_path):
    adata, _ = make_random_counts_anndata(
        seed=13, n_cells=9, n_genes=7, density=0.45, dtype="float32"
    )
    atlas = sap.Atlas(atlas_path)
    atlas.load_anndata(adata)
    conn = atlas.connection

    conn.execute("ALTER TABLE obs ADD COLUMN keep_cell BOOLEAN DEFAULT FALSE")
    conn.execute("UPDATE obs SET keep_cell = atlas_cell_id IN (1, 3, 4, 8)")
    conn.execute("ALTER TABLE var ADD COLUMN keep_gene BOOLEAN DEFAULT FALSE")
    conn.execute("UPDATE var SET keep_gene = atlas_gene_id IN (0, 2, 5)")

    atlas.build_read_index(
        cell_condition="keep_cell",
        gene_condition="keep_gene",
        use_hvg=False,
        use_data="data",
    )

    filtered = conn.execute(
        """
        SELECT filter_cell_id, filter_gene_id, data
        FROM X_HyS_data_filtered
        ORDER BY filter_cell_id, filter_gene_id
        """
    ).fetchall()
    dense = np.zeros((4, 3), dtype=np.float32)
    for i, j, value in filtered:
        dense[int(i), int(j)] = value

    expected = adata.X[[1, 3, 4, 8]][:, [0, 2, 5]].toarray()
    _assert_dense_equal(dense, expected)

    obs_map = conn.execute(
        "SELECT atlas_cell_id FROM obs WHERE filter_cell_id IS NOT NULL ORDER BY filter_cell_id"
    ).fetchall()
    var_map = conn.execute(
        "SELECT atlas_gene_id FROM var WHERE filter_gene_id IS NOT NULL ORDER BY filter_gene_id"
    ).fetchall()
    assert [r[0] for r in obs_map] == [1, 3, 4, 8]
    assert [r[0] for r in var_map] == [0, 2, 5]
    atlas.close()


@pytest.mark.l0
def test_database_copy_is_portable(tmp_path):
    adata, _ = make_random_counts_anndata(
        seed=14, n_cells=8, n_genes=5, density=0.4, dtype="float32"
    )
    source = tmp_path / "source.sasql"
    copied = tmp_path / "copied.sasql"

    atlas = sap.Atlas(source)
    atlas.load_anndata(adata)
    ids = _all_cell_ids(atlas)
    atlas.close()

    shutil.copy2(source, copied)
    reopened = sap.Atlas(copied)
    out = reopened.get_anndata(
        ids, use_data="data", include_obsm=False, include_varm=False
    )
    _assert_dense_equal(out.X.toarray(), adata.X.toarray())
    reopened.close()


@pytest.mark.l0
@pytest.mark.parametrize("shape", [(0, 5), (5, 0)])
def test_zero_dimension_datasets_do_not_crash_import(atlas_path, shape):
    from anndata import AnnData
    from scipy import sparse

    adata = AnnData(X=sparse.csr_matrix(shape, dtype=np.float32))
    atlas = sap.Atlas(atlas_path)
    atlas.load_anndata(adata)
    atlas.close()
