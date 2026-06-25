from __future__ import annotations

import numpy as np
import pytest

sap = pytest.importorskip("scatlaspy")

from .data_generators import make_random_counts_anndata, write_h5ad_shards


def _all_cell_ids(atlas) -> list[int]:
    return atlas.query("SELECT atlas_cell_id FROM obs ORDER BY atlas_cell_id")[
        "atlas_cell_id"
    ].astype(int).tolist()


def _dense_by_cell_name(adata):
    dense = adata.X.toarray().astype(np.float32)
    return {name: dense[i] for i, name in enumerate(adata.obs_names)}


def _assert_export_matches_by_cell_name(atlas, expected):
    out = atlas.get_anndata(
        _all_cell_ids(atlas), use_data="data_count", include_obsm=False, include_varm=False
    )
    expected_by_name = _dense_by_cell_name(expected)
    actual_names = out.obs["atlas_cell_name"].tolist()
    assert sorted(actual_names) == sorted(expected_by_name)
    assert out.var["atlas_gene_name"].tolist() == list(expected.var_names)

    actual = out.X.toarray().astype(np.float32)
    for row_id, cell_name in enumerate(actual_names):
        np.testing.assert_allclose(
            actual[row_id], expected_by_name[cell_name], rtol=1e-5, atol=1e-6
        )


@pytest.mark.l1
@pytest.mark.parametrize("load_type", ["order", "random"])
def test_h5ad_file_import_round_trip_matches_source(tmp_path, load_type):
    adata, _ = make_random_counts_anndata(
        seed=2101, n_cells=140, n_genes=18, density=0.16, dtype="float32"
    )
    h5ad_path = tmp_path / f"source_{load_type}.h5ad"
    adata.write_h5ad(h5ad_path)

    atlas = sap.Atlas(tmp_path / f"{load_type}.sasql")
    atlas.load_h5ad(
        h5ad_path,
        load_type=load_type,
        cells_per_block=17,
        blocks_per_pool=3,
    )

    _assert_export_matches_by_cell_name(atlas, adata)
    atlas.close()


@pytest.mark.l1
def test_h5ad_multi_file_random_import_combines_shards_and_supports_downstream(tmp_path):
    h5ad_paths, _, shape = write_h5ad_shards(
        tmp_path,
        base_name="multi_file_random",
        seed=2201,
        n_cells=157,
        n_genes=19,
        density=0.14,
        n_shards=3,
        dtype="float32",
    )

    atlas = sap.Atlas(tmp_path / "multi_file_random.sasql")
    atlas.load_h5ad(
        [str(path) for path in h5ad_paths],
        load_type="random",
        cells_per_block=13,
        blocks_per_pool=2,
    )

    obs = atlas.query("SELECT atlas_cell_name FROM obs")
    var = atlas.query("SELECT atlas_gene_name FROM var")
    assert obs.shape[0] == shape[0]
    assert var.shape[0] == shape[1]
    assert obs["atlas_cell_name"].is_unique

    atlas.connection.execute(
        "ALTER TABLE var ADD COLUMN IF NOT EXISTS zero_scale_transform REAL DEFAULT 0.0"
    )
    atlas.build_read_index(use_hvg=False, use_data="data_count")
    batches = list(
        atlas.get_minibatch_dense(
            pass_mode="single-pass", batch_size=31, buffer_batch_num=2
        )
    )
    assert sum(batch.shape[0] for batch in batches) == shape[0]
    assert [batch.shape[1] for batch in batches] == [shape[1]] * len(batches)
    assert batches[-1].shape[0] == shape[0] % 31
    atlas.close()
