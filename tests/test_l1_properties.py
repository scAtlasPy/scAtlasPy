from __future__ import annotations

import numpy as np
import pytest

hypothesis = pytest.importorskip("hypothesis")
HealthCheck = hypothesis.HealthCheck
given = hypothesis.given
settings = hypothesis.settings
st = pytest.importorskip("hypothesis.strategies")

sap = pytest.importorskip("scatlaspy")

from .data_generators import make_random_counts_anndata


def _ids(atlas):
    return atlas.query("SELECT atlas_cell_id FROM obs ORDER BY atlas_cell_id")[
        "atlas_cell_id"
    ].astype(int).tolist()


@pytest.mark.l1
@settings(
    max_examples=5,
    deadline=None,
    suppress_health_check=[
        HealthCheck.too_slow,
        HealthCheck.function_scoped_fixture,
    ],
)
@given(
    n_cells=st.integers(min_value=100, max_value=180),
    n_genes=st.integers(min_value=8, max_value=25),
    density=st.floats(min_value=0.03, max_value=0.25),
    seed=st.integers(min_value=1, max_value=10_000),
)
def test_random_round_trip_properties(tmp_path, n_cells, n_genes, density, seed):
    adata, _ = make_random_counts_anndata(
        seed=seed,
        n_cells=n_cells,
        n_genes=n_genes,
        density=density,
        dtype="float32",
    )
    atlas = sap.Atlas(tmp_path / f"prop_{seed}.sasql")
    atlas.load_anndata(adata)

    out = atlas.get_anndata(
        _ids(atlas), use_data="data", include_obsm=False, include_varm=False
    )

    assert out.shape == adata.shape
    assert out.obs.shape[0] == n_cells
    assert out.var.shape[0] == n_genes
    np.testing.assert_allclose(out.X.toarray(), adata.X.toarray(), rtol=1e-5, atol=1e-6)
    atlas.close()


@pytest.mark.l1
@pytest.mark.parametrize("batch_size", [1, 7, 20, 99])
def test_minibatch_boundaries_cover_all_rows_once(tmp_path, batch_size):
    adata, _ = make_random_counts_anndata(
        seed=1234, n_cells=37, n_genes=11, density=0.35, dtype="float32"
    )
    atlas = sap.Atlas(tmp_path / f"batch_{batch_size}.sasql")
    atlas.load_anndata(adata)
    atlas.connection.execute(
        "ALTER TABLE var ADD COLUMN IF NOT EXISTS zero_scale_transform REAL DEFAULT 0.0"
    )
    atlas.build_read_index(use_hvg=False, use_data="data")

    batches = list(
        atlas.get_minibatch_dense(
            pass_mode="single-pass", batch_size=batch_size, buffer_batch_num=2
        )
    )
    assert [b.shape[1] for b in batches] == [adata.n_vars] * len(batches)
    assert sum(b.shape[0] for b in batches) == adata.n_obs
    np.testing.assert_allclose(
        np.vstack(batches),
        adata.X.toarray().astype(np.float32),
        rtol=1e-5,
        atol=1e-6,
    )

    expected_last = adata.n_obs % batch_size or min(batch_size, adata.n_obs)
    assert batches[-1].shape[0] == expected_last
    atlas.close()


@pytest.mark.l1
def test_randomized_filtered_index_has_deterministic_shape(tmp_path):
    adata, _ = make_random_counts_anndata(
        seed=5678, n_cells=200, n_genes=30, density=0.12, dtype="float32"
    )
    atlas = sap.Atlas(tmp_path / "filtered_shape.sasql")
    atlas.load_anndata(adata)
    conn = atlas.connection
    conn.execute("ALTER TABLE obs ADD COLUMN keep_cell BOOLEAN DEFAULT FALSE")
    conn.execute("UPDATE obs SET keep_cell = atlas_cell_id % 3 != 0")
    conn.execute("ALTER TABLE var ADD COLUMN keep_gene BOOLEAN DEFAULT FALSE")
    conn.execute("UPDATE var SET keep_gene = atlas_gene_id % 4 != 0")

    atlas.build_read_index(
        cell_condition="keep_cell",
        gene_condition="keep_gene",
        use_hvg=False,
        use_data="data",
    )

    n_cells = conn.execute(
        "SELECT COUNT(*) FROM obs WHERE filter_cell_id IS NOT NULL"
    ).fetchone()[0]
    n_genes = conn.execute(
        "SELECT COUNT(*) FROM var WHERE filter_gene_id IS NOT NULL"
    ).fetchone()[0]
    assert n_cells == sum(i % 3 != 0 for i in range(adata.n_obs))
    assert n_genes == sum(i % 4 != 0 for i in range(adata.n_vars))
    atlas.close()
