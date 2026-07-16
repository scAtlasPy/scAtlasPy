from __future__ import annotations

import numpy as np
import pytest

import scatlaspy as sap


def _read_dense_counts(atlas: sap.Atlas, shape: tuple[int, int]) -> np.ndarray:
    """Read ``X_HyS_data.data_count`` into a dense matrix for exact checks."""

    rows = atlas.connection.execute(
        """
        SELECT atlas_cell_id, atlas_gene_id, data_count
        FROM X_HyS_data
        ORDER BY atlas_cell_id, atlas_gene_id
        """
    ).fetchall()

    X = np.zeros(shape, dtype=np.float32)
    for cell_id, gene_id, value in rows:
        X[int(cell_id), int(gene_id)] = float(value)

    return X


@pytest.mark.storage
def test_atlas_context_manager_closes_connection(tmp_path) -> None:
    """``with Atlas(...)`` should keep explicit close semantics but close on exit."""

    atlas_path = tmp_path / "context_manager.sasql"

    with sap.Atlas(atlas_path, db_memory_limit="1GB") as atlas:
        assert atlas.connection is not None

    assert atlas.connection is None


@pytest.mark.storage
def test_load_h5ad_name_columns_and_log_scale_count_import(
    tmp_path,
    tiny_log_h5ad: tuple[str, np.ndarray],
) -> None:
    """Optional name columns should be honored while log-scale X is stored as counts."""

    h5ad_path, counts = tiny_log_h5ad
    atlas_path = tmp_path / "load_h5ad_names.sasql"

    with sap.Atlas(atlas_path, db_memory_limit="1GB") as atlas:
        atlas.load_h5ad(
            h5ad_path,
            load_type="order",
            cells_per_block=3,
            cell_name_col="barcode",
            gene_name_col="symbol",
        )

        obs = atlas.get_obs_df(columns=["atlas_cell_name"])
        var = atlas.get_var_df(columns=["atlas_gene_name"])
        X = _read_dense_counts(atlas, counts.shape)

    assert obs["atlas_cell_name"].tolist() == [f"barcode_{i}" for i in range(counts.shape[0])]
    assert var["atlas_gene_name"].tolist() == [f"symbol_{j}" for j in range(counts.shape[1])]
    np.testing.assert_allclose(X, counts, rtol=1e-5, atol=1e-5)


@pytest.mark.storage
def test_obsm_and_varm_accessors_preserve_requested_order(
    tmp_path,
    tiny_adata,
) -> None:
    """Accessors should use ID as the index and preserve explicit ID order."""

    atlas_path = tmp_path / "accessors.sasql"

    with sap.Atlas(atlas_path, db_memory_limit="1GB") as atlas:
        atlas.load_anndata(tiny_adata)

        obsm = atlas.get_obsm_df(
            "obsm_X_test",
            atlas_cell_id=[3, 1],
            columns=["dim_1"],
        )
        varm = atlas.get_varm_df(
            "varm_PCs",
            atlas_gene_id=[4, 0, 2],
            columns=["dim_0"],
        )

        with pytest.raises(ValueError, match="Duplicate"):
            atlas.get_obsm_df("obsm_X_test", atlas_cell_id=[1, 1])

    assert obsm.index.tolist() == [3, 1]
    assert obsm.columns.tolist() == ["dim_1"]
    np.testing.assert_allclose(obsm["dim_1"].to_numpy(), [7.0, 3.0])

    assert varm.index.tolist() == [4, 0, 2]
    assert varm.columns.tolist() == ["dim_0"]
    np.testing.assert_allclose(varm["dim_0"].to_numpy(), [12.0, 0.0, 6.0])
