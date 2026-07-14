from __future__ import annotations

from collections import Counter

import anndata as ad
import numpy as np

import scatlaspy as sap

from .helpers import assert_dense_equal, dense_counts


def _row_counter(matrix: np.ndarray) -> Counter:
    """Represent rows as a multiset so random import can be checked after shuffling."""

    return Counter(tuple(row.tolist()) for row in matrix.astype(np.float32))


def test_ordered_load_h5ad_and_write_h5ad_round_trip(tmp_path, counts_adata):
    count_h5ad = counts_adata.copy()
    count_h5ad.X = count_h5ad.X * 20
    input_path = tmp_path / "input.h5ad"
    output_path = tmp_path / "exported_log1p.h5ad"
    count_h5ad.write_h5ad(input_path)

    atlas = sap.Atlas(tmp_path / "ordered.sasql", db_memory_limit="1GB")
    try:
        atlas.load_h5ad(input_path, load_type="order", cells_per_block=3)
        exported_counts = atlas.get_anndata(
            list(range(counts_adata.n_obs)),
            use_data="data_count",
            include_obsm=True,
            include_varm=True,
        )
        assert_dense_equal(exported_counts.X.toarray(), dense_counts(count_h5ad))
        assert "X_fixture" in exported_counts.obsm
        assert "fixture_loadings" in exported_counts.varm

        sap.pp.normalize_and_log1p(atlas)
        atlas.write_h5ad(output_path, use_data="data_log1p", batch_cells=5)
    finally:
        atlas.close()

    exported = ad.read_h5ad(output_path)
    counts = dense_counts(count_h5ad)
    expected_log1p = np.log1p(counts / counts.sum(axis=1, keepdims=True) * 10_000)
    assert_dense_equal(exported.X.toarray(), expected_log1p)


def test_random_load_h5ad_preserves_expression_rows_after_shuffle(tmp_path, counts_adata):
    count_h5ad = counts_adata.copy()
    count_h5ad.X = count_h5ad.X * 20
    input_path = tmp_path / "input_random.h5ad"
    count_h5ad.write_h5ad(input_path)

    atlas = sap.Atlas(tmp_path / "random.sasql", db_memory_limit="1GB")
    try:
        atlas.load_h5ad(input_path, load_type="random", cells_per_block=2, import_window_memory_factor=1.0)
        cell_ids = atlas.get_obs_df()["atlas_cell_id"].astype(int).tolist()
        exported = atlas.get_anndata(cell_ids, use_data="data_count", include_obsm=False, include_varm=False)
    finally:
        atlas.close()

    assert _row_counter(exported.X.toarray()) == _row_counter(dense_counts(count_h5ad))
