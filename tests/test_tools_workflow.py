from __future__ import annotations

import numpy as np

import scatlaspy as sap


def test_pca_kmeans_rank_and_manual_annotation_smoke(atlas_from_adata, workflow_adata):
    atlas = atlas_from_adata(workflow_adata)

    sap.pp.normalize_and_log1p(atlas)
    sap.pp.highly_variable_genes(atlas, flavor="var", n_top_genes=8, use_filtered=False)
    sap.pp.scale(atlas, use_hvg=True)
    atlas.build_read_index(cell_condition=None, gene_condition=None, use_hvg=True, use_data="data_scale")

    sap.tl.pca(atlas, n_components=3, fit_batches=2, batch_size=12)
    tables = set(atlas.query("SHOW TABLES")["name"])
    assert {"obsm_X_pca", "varm_PCs", "uns_pca_stats"} <= tables

    sap.tl.kmeans(atlas, n_components=3, n_clusters=3, batch_size=12, fit_batches=2)
    obs = atlas.get_obs_df()
    assert obs["kmeans"].notna().all()

    cluster_ids = sorted(obs["kmeans"].astype(int).unique().tolist())
    annotation = {cluster_id: f"type_{cluster_id}" for cluster_id in cluster_ids}
    annotated = sap.tl.manual_annotate_clusters(atlas, annotation, groupby="kmeans", obs_col="manual_type")
    assert set(annotated["cell_type"]) == set(annotation.values())
    assert atlas.get_obs_df()["manual_type"].notna().all()

    rank = sap.tl.rank_genes_groups(
        atlas,
        groupby="known_group",
        use_data="data_log1p",
        n_genes=3,
        input_is_log=True,
        return_df=True,
    )
    assert not rank.empty
    assert {"rank_genes_groups", "obs_cluster", "kmeans_centers"} <= set(atlas.query("SHOW TABLES")["name"])


def test_public_preprocessing_surface_matches_count_only_transform_design():
    assert hasattr(sap.pp, "sqrt")
    assert not hasattr(sap.pp, "expm1")
    assert hasattr(sap.pp, "normalize_and_log1p")
    assert np.float32(1.0).dtype == np.dtype("float32")

