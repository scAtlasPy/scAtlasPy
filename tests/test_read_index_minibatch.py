from __future__ import annotations

import numpy as np

from .helpers import assert_dense_equal, dense_counts


def test_build_read_index_filters_cells_genes_and_preserves_counts(atlas_from_adata, counts_adata):
    atlas = atlas_from_adata(counts_adata)
    atlas.connection.execute("ALTER TABLE obs ADD COLUMN keep_cell BOOLEAN")
    atlas.connection.execute("UPDATE obs SET keep_cell = atlas_cell_id IN (0, 2, 4, 6)")
    atlas.connection.execute("ALTER TABLE var ADD COLUMN keep_gene BOOLEAN")
    atlas.connection.execute("UPDATE var SET keep_gene = atlas_gene_id IN (0, 2, 4)")

    atlas.build_read_index(cell_condition="keep_cell", gene_condition="keep_gene", use_hvg=False, use_data="data_count")

    obs = atlas.get_obs_df()
    var = atlas.get_var_df()
    assert obs["filter_cell_id"].dropna().astype(int).tolist() == [0, 1, 2, 3]
    assert var["filter_gene_id"].dropna().astype(int).tolist() == [0, 1, 2]

    batches = list(atlas.get_minibatch_dense(pass_mode="single-pass", batch_size=2))
    actual = np.vstack(batches)
    expected = dense_counts(counts_adata)[[0, 2, 4, 6]][:, [0, 2, 4]]
    assert_dense_equal(actual, expected)


def test_single_pass_minibatches_return_dense_arrays_and_obs_labels(atlas_from_adata, counts_adata):
    atlas = atlas_from_adata(counts_adata)
    atlas.build_read_index(cell_condition=None, gene_condition=None, use_hvg=False, use_data="data_count")

    chunks = list(atlas.get_minibatch_dense(pass_mode="single-pass", batch_size=3, get_obs_col="known_group"))
    X = np.vstack([chunk["X"] for chunk in chunks])
    labels = np.concatenate([chunk["known_group"] for chunk in chunks])
    filter_ids = np.concatenate([chunk["filter_cell_ids"] for chunk in chunks])

    assert_dense_equal(X, dense_counts(counts_adata))
    np.testing.assert_array_equal(filter_ids, np.arange(counts_adata.n_obs))
    assert labels.tolist() == counts_adata.obs["known_group"].tolist()


def test_multi_pass_minibatches_respect_max_batches(atlas_from_adata, counts_adata):
    atlas = atlas_from_adata(counts_adata)
    atlas.build_read_index(cell_condition=None, gene_condition=None, use_hvg=False, use_data="data_count")

    batches = list(
        atlas.get_minibatch_dense(
            pass_mode="multi-pass",
            batch_size=2,
            max_batches=4,
            buffer_batch_num=2,
        )
    )

    assert len(batches) == 4
    assert all(batch.shape == (2, counts_adata.n_vars) for batch in batches)
    assert all(batch.dtype == np.float32 for batch in batches)
