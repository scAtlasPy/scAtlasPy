from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from .helpers import assert_dense_equal, dense_counts, expected_hys_rows


def test_load_anndata_writes_current_hys_schema_and_metadata(atlas_from_adata, counts_adata):
    atlas = atlas_from_adata(counts_adata)

    tables = set(atlas.query("SHOW TABLES")["name"])
    assert {"obs", "var", "X_HyS_data", "X_HyS_indptr", "obsm_X_fixture", "varm_fixture_loadings"} <= tables

    schema = atlas.query("PRAGMA table_info('X_HyS_data')")
    assert schema["name"].tolist() == ["id", "atlas_cell_id", "atlas_gene_id", "data_count"]

    actual = atlas.query("SELECT * FROM X_HyS_data ORDER BY id").reset_index(drop=True)
    expected = expected_hys_rows(counts_adata)
    pd.testing.assert_series_equal(actual["id"], expected["id"], check_names=False)
    pd.testing.assert_series_equal(
        actual["atlas_cell_id"].astype("int64"),
        expected["atlas_cell_id"],
        check_names=False,
    )
    pd.testing.assert_series_equal(
        actual["atlas_gene_id"].astype("int64"),
        expected["atlas_gene_id"],
        check_names=False,
    )
    np.testing.assert_allclose(actual["data_count"], expected["data_count"])

    indptr = atlas.query("SELECT atlas_cell_id, indptr FROM X_HyS_indptr ORDER BY atlas_cell_id")
    np.testing.assert_array_equal(indptr["indptr"].to_numpy(), counts_adata.X.tocsr().indptr[1:])


def test_dataframe_accessors_and_head_are_quiet(atlas_from_adata, counts_adata, capsys):
    atlas = atlas_from_adata(counts_adata)

    obs = atlas.get_obs_df()
    var = atlas.get_var_df()
    head = atlas.head("obs", n=3)

    assert list(obs["sample"]) == list(counts_adata.obs["sample"])
    assert "atlas_gene_name" in var.columns
    assert head.shape[0] == 3
    assert capsys.readouterr().out == ""


def test_get_anndata_round_trips_counts_and_multidimensional_tables(atlas_from_adata, counts_adata):
    atlas = atlas_from_adata(counts_adata)
    exported = atlas.get_anndata([2, 0, 5], use_data="data_count", include_obsm=True, include_varm=True)

    expected = dense_counts(counts_adata)[[2, 0, 5], :]
    assert_dense_equal(exported.X.toarray(), expected)
    assert exported.obs["sample"].tolist() == counts_adata.obs["sample"].iloc[[2, 0, 5]].tolist()
    assert "X_fixture" in exported.obsm
    assert "fixture_loadings" in exported.varm
    np.testing.assert_allclose(exported.obsm["X_fixture"], counts_adata.obsm["X_fixture"][[2, 0, 5]])
    np.testing.assert_allclose(exported.varm["fixture_loadings"], counts_adata.varm["fixture_loadings"])


def test_get_anndata_rejects_empty_or_duplicate_cell_ids(atlas_from_adata, counts_adata):
    atlas = atlas_from_adata(counts_adata)

    with pytest.raises(ValueError, match="cannot be empty"):
        atlas.get_anndata([])

    with pytest.raises(ValueError, match="Duplicate"):
        atlas.get_anndata([0, 0])
