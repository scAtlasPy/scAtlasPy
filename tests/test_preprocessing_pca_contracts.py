from __future__ import annotations

import pytest

import scatlaspy as sap


def _transform_meta(atlas: sap.Atlas) -> dict[str, tuple[str, bool]]:
    """Return transform metadata keyed by derived expression data name."""

    rows = atlas.connection.execute(
        """
        SELECT data_name, transform, centered
        FROM atlas_expression_transform_meta
        ORDER BY data_name
        """
    ).fetchall()

    return {data_name: (transform, bool(centered)) for data_name, transform, centered in rows}


@pytest.mark.preprocessing
def test_scale_modes_record_centering_metadata(tmp_path, tiny_adata) -> None:
    """``scale(mode=...)`` should record whether the derived data are centered."""

    atlas_path = tmp_path / "scale_modes.sasql"

    with sap.Atlas(atlas_path, db_memory_limit="1GB") as atlas:
        atlas.load_anndata(tiny_adata)

        sap.pp.scale(
            atlas,
            use_data="data_count",
            add_data="data_center_only",
            add_var_col="zero_center_only",
            use_hvg=False,
            mode="center_only",
        )
        sap.pp.scale(
            atlas,
            use_data="data_count",
            add_data="data_scale_only",
            add_var_col="zero_scale_only",
            use_hvg=False,
            mode="scale_only",
        )

        meta = _transform_meta(atlas)

    assert meta["data_center_only"] == ("scale:center_only", True)
    assert meta["data_scale_only"] == ("scale:scale_only", False)


@pytest.mark.workflow
def test_pca_rejects_uncentered_or_scale_only_read_index(tmp_path, tiny_adata) -> None:
    """PCA must fail before fitting if the read index is not centered."""

    atlas_path = tmp_path / "pca_requires_centering.sasql"

    with sap.Atlas(atlas_path, db_memory_limit="1GB") as atlas:
        atlas.load_anndata(tiny_adata)

        atlas.build_read_index(
            cell_condition=None,
            gene_condition=None,
            use_hvg=False,
            use_data="data_count",
        )
        with pytest.raises(ValueError, match="centered expression data"):
            sap.tl.pca(atlas, n_components=2, oversample=1, batch_size=3, n_iter=0)

        sap.pp.scale(
            atlas,
            use_data="data_count",
            add_data="data_scale_only",
            add_var_col="zero_scale_only",
            use_hvg=False,
            mode="scale_only",
        )
        atlas.build_read_index(
            cell_condition=None,
            gene_condition=None,
            use_hvg=False,
            use_data="data_scale_only",
        )
        with pytest.raises(ValueError, match="does not center genes"):
            sap.tl.pca(atlas, n_components=2, oversample=1, batch_size=3, n_iter=0)


@pytest.mark.workflow
def test_pca_runs_on_centered_scale_and_writes_standard_tables(tmp_path, tiny_adata) -> None:
    """Centered scale data should pass PCA and produce obsm, varm, and uns tables."""

    atlas_path = tmp_path / "pca_centered.sasql"

    with sap.Atlas(atlas_path, db_memory_limit="1GB") as atlas:
        atlas.load_anndata(tiny_adata)
        sap.pp.scale(
            atlas,
            use_data="data_count",
            add_data="data_scale",
            use_hvg=False,
            mode="center_only",
        )
        atlas.build_read_index(
            cell_condition=None,
            gene_condition=None,
            use_hvg=False,
            use_data="data_scale",
        )

        sap.tl.pca(
            atlas,
            n_components=2,
            oversample=1,
            batch_size=3,
            random_state=0,
            n_iter=0,
        )

        pca = atlas.get_obsm_df("obsm_X_pca")
        pcs = atlas.get_varm_df("varm_PCs")
        stats = atlas.get_uns_df("uns_pca_stats")

    assert pca.shape == (tiny_adata.n_obs, 2)
    assert pca.columns.tolist() == ["pc0", "pc1"]
    assert pcs.shape == (tiny_adata.n_vars, 2)
    assert pcs.columns.tolist() == ["pc0", "pc1"]
    assert stats.shape[0] == 2
