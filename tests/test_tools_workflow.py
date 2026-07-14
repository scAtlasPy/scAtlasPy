from __future__ import annotations

import numpy as np
from scipy.linalg import subspace_angles
from sklearn.decomposition import PCA

import scatlaspy as sap
from scatlaspy.tools._pca import StreamingRandomizedPCA


class _ArrayAtlas:
    """Minimal atlas-like object that yields dense batches for PCA tests."""

    def __init__(self, X: np.ndarray, batch_size: int):
        self.X = X
        self.batch_size = batch_size

    def get_minibatch_dense(self, batch_size=2048, pass_mode="single-pass", **kwargs):
        assert pass_mode == "single-pass"
        for start in range(0, self.X.shape[0], self.batch_size):
            yield self.X[start:start + self.batch_size]


def test_pca_kmeans_rank_and_manual_annotation_smoke(atlas_from_adata, workflow_adata):
    atlas = atlas_from_adata(workflow_adata)

    sap.pp.normalize_and_log1p(atlas)
    sap.pp.highly_variable_genes(atlas, flavor="var", n_top_genes=8, use_filtered=False)
    sap.pp.scale(atlas, use_hvg=True)
    atlas.build_read_index(cell_condition=None, gene_condition=None, use_hvg=True, use_data="data_scale")

    sap.tl.pca(atlas, n_components=3, batch_size=12)
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


def test_randomized_streaming_pca_backend_writes_standard_tables(atlas_from_adata, workflow_adata):
    atlas = atlas_from_adata(workflow_adata)

    sap.pp.normalize_and_log1p(atlas)
    sap.pp.highly_variable_genes(atlas, flavor="var", n_top_genes=8, use_filtered=False)
    sap.pp.scale(atlas, use_hvg=True)
    atlas.build_read_index(cell_condition=None, gene_condition=None, use_hvg=True, use_data="data_scale")

    sap.tl.pca(
        atlas,
        n_components=3,
        batch_size=12,
        oversample=2,
        random_state=0,
    )

    tables = set(atlas.query("SHOW TABLES")["name"])
    assert {"obsm_X_pca", "varm_PCs", "uns_pca_stats"} <= tables

    pca = atlas.query("SELECT * FROM obsm_X_pca ORDER BY atlas_cell_id")
    pcs = atlas.query("SELECT * FROM varm_PCs ORDER BY atlas_gene_id")
    stats = atlas.query("SELECT * FROM uns_pca_stats ORDER BY pc_index")

    assert pca.shape[0] == workflow_adata.n_obs
    assert [c for c in pca.columns if c.startswith("pc")] == ["pc0", "pc1", "pc2"]
    assert pcs.shape[0] == 8
    assert [c for c in pcs.columns if c.startswith("pc")] == ["pc0", "pc1", "pc2"]
    assert stats.shape[0] == 3
    assert np.isfinite(pca[["pc0", "pc1", "pc2"]].to_numpy()).all()
    assert np.isfinite(pcs[["pc0", "pc1", "pc2"]].to_numpy()).all()
    assert np.isfinite(stats[["variance", "variance_ratio"]].to_numpy()).all()


def test_randomized_streaming_pca_matches_reference_subspace_on_low_rank_matrix():
    rng = np.random.default_rng(0)
    left = rng.normal(size=(240, 5))
    right = rng.normal(size=(5, 40))
    X = (left @ right + 0.01 * rng.normal(size=(240, 40))).astype(np.float32)
    X -= X.mean(axis=0, keepdims=True)

    reference = PCA(n_components=5, svd_solver="randomized", random_state=0)
    reference.fit(X)

    model = StreamingRandomizedPCA(
        n_components=5,
        oversample=5,
        batch_size=37,
        random_state=0,
    )
    model.fit(_ArrayAtlas(X, batch_size=37))

    angles = subspace_angles(reference.components_.T, model.components_.T)
    assert np.degrees(angles).mean() < 1.0
    assert np.allclose(
        reference.explained_variance_ratio_,
        model.explained_variance_ratio_,
        rtol=0.05,
        atol=0.01,
    )


def test_public_preprocessing_surface_matches_count_only_transform_design():
    assert hasattr(sap.pp, "sqrt")
    assert not hasattr(sap.pp, "expm1")
    assert hasattr(sap.pp, "normalize_and_log1p")
    assert np.float32(1.0).dtype == np.dtype("float32")
