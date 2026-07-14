"""Small deterministic fixtures shared by the scAtlasPy tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy import sparse


def make_counts_adata() -> AnnData:
    """Create a hand-checkable count matrix with obs, var, obsm, and varm data."""

    X = np.array(
        [
            [1, 0, 3, 0, 2, 0],
            [0, 4, 0, 1, 0, 2],
            [5, 0, 0, 0, 1, 0],
            [0, 2, 2, 0, 0, 3],
            [3, 0, 1, 4, 0, 0],
            [0, 0, 0, 2, 5, 1],
            [2, 1, 0, 0, 0, 4],
            [0, 3, 5, 1, 0, 0],
        ],
        dtype=np.float32,
    )
    obs = pd.DataFrame(
        {
            "sample": ["s1", "s1", "s2", "s2", "s3", "s3", "s4", "s4"],
            "known_group": ["A", "A", "B", "B", "C", "C", "A", "B"],
        },
        index=[f"cell_{i}" for i in range(X.shape[0])],
    )
    var = pd.DataFrame(
        {"gene_symbol": ["MT-ND1", "RPS3", "GeneA", "GeneB", "GeneC", "GeneD"]},
        index=["MT-ND1", "RPS3", "GeneA", "GeneB", "GeneC", "GeneD"],
    )

    adata = AnnData(X=sparse.csr_matrix(X), obs=obs, var=var)
    adata.obsm["X_fixture"] = np.arange(X.shape[0] * 2, dtype=np.float32).reshape(X.shape[0], 2)
    adata.varm["fixture_loadings"] = np.arange(X.shape[1] * 3, dtype=np.float32).reshape(X.shape[1], 3)
    return adata


def make_workflow_adata(n_cells: int = 48, n_genes: int = 12, seed: int = 0) -> AnnData:
    """Create a small non-degenerate count matrix for PCA, clustering, and ranking."""

    rng = np.random.default_rng(seed)
    groups = np.resize(np.array(["A", "B", "C"], dtype=object), n_cells)
    X = rng.poisson(1.2, size=(n_cells, n_genes)).astype(np.float32)

    # Add a stable group signal so ranking and clustering smoke tests have structure.
    for group_id, group_name in enumerate(["A", "B", "C"]):
        rows = groups == group_name
        X[rows, group_id * 2 : group_id * 2 + 2] += 3

    X[X.sum(axis=1) == 0, 0] = 1
    X[:, X.sum(axis=0) == 0] = 1

    obs = pd.DataFrame(
        {
            "sample": np.resize(np.array(["s1", "s2", "s3", "s4"], dtype=object), n_cells),
            "known_group": groups,
        },
        index=[f"cell_{i}" for i in range(n_cells)],
    )
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)])
    return AnnData(X=sparse.csr_matrix(X), obs=obs, var=var)


def dense_counts(adata: AnnData) -> np.ndarray:
    """Return the count matrix as a float32 dense array."""

    return adata.X.toarray().astype(np.float32)


def expected_hys_rows(adata: AnnData) -> pd.DataFrame:
    """Return the expected row-wise HyS data table for a CSR AnnData matrix."""

    X = adata.X.tocsr()
    rows = np.repeat(np.arange(X.shape[0], dtype=np.int64), np.diff(X.indptr))
    return pd.DataFrame(
        {
            "id": np.arange(X.nnz, dtype=np.int64),
            "atlas_cell_id": rows,
            "atlas_gene_id": X.indices.astype(np.int64),
            "data_count": X.data.astype(np.float32),
        }
    )


def assert_dense_equal(actual: np.ndarray, expected: np.ndarray, *, atol: float = 1e-6) -> None:
    """Assert numeric equality for dense matrices with a readable failure mode."""

    np.testing.assert_allclose(np.asarray(actual, dtype=np.float32), expected.astype(np.float32), atol=atol)

