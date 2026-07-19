from __future__ import annotations

import numpy as np
from scipy.linalg import subspace_angles
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

import scatlaspy as sap
from scatlaspy.tools._pca import StreamingRandomizedPCA, _check_finite_array, _matmul_ignore_blas_flags


class _ArrayAtlas:
    """Minimal atlas-like object that yields dense batches for PCA tests."""

    def __init__(self, X: np.ndarray, batch_size: int):
        self.X = X
        self.batch_size = batch_size

    def get_minibatch_dense(self, batch_size=2048, pass_mode="single-pass", **kwargs):
        assert pass_mode == "single-pass"
        for start in range(0, self.X.shape[0], self.batch_size):
            yield self.X[start:start + self.batch_size]


def _knn_indices(X: np.ndarray, k: int) -> np.ndarray:
    """Return k nearest-neighbor indices, excluding each point itself."""

    neighbors = NearestNeighbors(n_neighbors=k + 1)
    neighbors.fit(X)
    return neighbors.kneighbors(X, return_distance=False)[:, 1:]


def _mean_knn_jaccard(reference: np.ndarray, candidate: np.ndarray, k: int) -> float:
    """Return the mean Jaccard overlap between two kNN graphs."""

    ref_knn = _knn_indices(reference, k)
    cand_knn = _knn_indices(candidate, k)

    overlaps = []
    for ref_row, cand_row in zip(ref_knn, cand_knn):
        ref_set = set(ref_row.tolist())
        cand_set = set(cand_row.tolist())
        overlaps.append(len(ref_set & cand_set) / len(ref_set | cand_set))

    return float(np.mean(overlaps))


def _mean_knn_label_purity(embedding: np.ndarray, labels: np.ndarray, k: int) -> float:
    """Return the mean fraction of k nearest neighbors sharing each cell label."""

    knn = _knn_indices(embedding, k)
    purity = labels[knn] == labels[:, None]
    return float(np.mean(purity))


def test_randomized_pca_matmul_ignores_blas_flags_for_finite_output():
    left = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    right = np.array([[0.5], [1.5]], dtype=np.float32)

    out = _matmul_ignore_blas_flags(left, right)

    assert np.allclose(out, left @ right)
    assert np.isfinite(out).all()


def test_randomized_pca_finite_check_rejects_nonfinite_matmul_output():
    left = np.array([[np.finfo(np.float32).max, np.finfo(np.float32).max]], dtype=np.float32)
    right = np.array([[np.finfo(np.float32).max], [np.finfo(np.float32).max]], dtype=np.float32)

    with np.testing.assert_raises_regex(
        ValueError,
        "Non-finite values detected in overflow_case during unit test",
    ):
        out = _matmul_ignore_blas_flags(left, right)
        _check_finite_array(out, name="overflow_case", context="unit test")


def test_pca_kmeans_rank_and_manual_annotation_smoke(atlas_from_adata, workflow_adata):
    atlas = atlas_from_adata(workflow_adata)

    sap.pp.normalize_and_log1p(atlas)
    sap.pp.highly_variable_genes(atlas, flavor="var", n_top_genes=8, use_filtered=False)
    sap.pp.scale(atlas, use_hvg=True)
    atlas.build_read_index(cell_condition=None, gene_condition=None, use_hvg=True, use_data="data_scale")

    sap.tl.pca(atlas, n_components=3, batch_size=12)
    tables = set(atlas.query("SHOW TABLES")["name"])
    assert {"obsm_X_pca", "varm_PCs", "uns_pca_stats"} <= tables

    sap.tl.kmeans(atlas, use_rep="X_pca", n_components=3, n_clusters=3, batch_size=12, fit_batches=2)
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
    for _, sub in rank.groupby("group"):
        assert (sub["logfoldchanges"] > 0).sum() <= 3
        assert (sub["logfoldchanges"] < 0).sum() <= 3
        assert len(sub) <= 6
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
        fit_batch_size=37,
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


def test_randomized_streaming_pca_preserves_knn_and_label_purity():
    rng = np.random.default_rng(1)

    n_groups = 3
    cells_per_group = 60
    n_latent = 5
    n_features = 36
    labels = np.repeat(np.arange(n_groups), cells_per_group)

    centers = np.zeros((n_groups, n_latent), dtype=np.float32)
    centers[0, 0] = 5.0
    centers[1, 1] = 5.0
    centers[2, 2] = 5.0

    latent = centers[labels] + rng.normal(scale=0.35, size=(labels.size, n_latent))
    loadings = rng.normal(size=(n_latent, n_features))
    X = (latent @ loadings + 0.05 * rng.normal(size=(labels.size, n_features))).astype(np.float32)
    X -= X.mean(axis=0, keepdims=True)

    reference = PCA(n_components=5, svd_solver="randomized", random_state=0)
    reference_embedding = reference.fit_transform(X)

    model = StreamingRandomizedPCA(
        n_components=5,
        oversample=8,
        fit_batch_size=23,
        random_state=0,
        n_iter=2,
    )
    model.fit(_ArrayAtlas(X, batch_size=23))
    streaming_embedding = X @ model.components_.T

    knn_jaccard = _mean_knn_jaccard(reference_embedding, streaming_embedding, k=10)
    reference_purity = _mean_knn_label_purity(reference_embedding, labels, k=10)
    streaming_purity = _mean_knn_label_purity(streaming_embedding, labels, k=10)

    assert knn_jaccard >= 0.85
    assert reference_purity >= 0.95
    assert streaming_purity >= 0.95
    assert streaming_purity >= reference_purity - 0.02


def test_public_preprocessing_surface_matches_count_only_transform_design():
    assert hasattr(sap.pp, "sqrt")
    assert not hasattr(sap.pp, "expm1")
    assert hasattr(sap.pp, "normalize_and_log1p")
    assert np.float32(1.0).dtype == np.dtype("float32")
