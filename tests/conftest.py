from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy import sparse

import scatlaspy as sap

from .helpers import make_counts_adata, make_workflow_adata


@pytest.fixture(autouse=True)
def quiet_progress() -> None:
    """Keep tests readable by disabling progress bars and verbose logs."""

    sap.set_progress(False)
    sap.set_verbosity("WARNING")


@pytest.fixture
def counts_adata() -> AnnData:
    """Return the shared hand-checkable count AnnData used by legacy tests."""

    return make_counts_adata()


@pytest.fixture
def workflow_adata() -> AnnData:
    """Return the shared non-degenerate AnnData used by workflow smoke tests."""

    return make_workflow_adata()


@pytest.fixture
def atlas_from_adata(tmp_path):
    """Create Atlas databases from AnnData and close them after each test."""

    atlases: list[sap.Atlas] = []

    def factory(
        adata: AnnData,
        name: str = "test.sasql",
        db_memory_limit: str = "1GB",
    ) -> sap.Atlas:
        atlas = sap.Atlas(tmp_path / name, db_memory_limit=db_memory_limit)
        atlas.load_anndata(adata)
        atlases.append(atlas)
        return atlas

    yield factory

    for atlas in atlases:
        atlas.close()


@pytest.fixture
def tiny_counts() -> np.ndarray:
    """Return a small count matrix with zeros and nonzero values."""

    return np.array(
        [
            [1, 0, 3, 0, 5],
            [0, 2, 0, 4, 1],
            [5, 1, 0, 2, 0],
            [0, 3, 2, 0, 4],
            [2, 0, 1, 3, 0],
            [4, 2, 0, 1, 3],
        ],
        dtype=np.float32,
    )


@pytest.fixture
def tiny_adata(tiny_counts: np.ndarray) -> AnnData:
    """Create a compact AnnData object used by storage and workflow tests."""

    n_cells, n_genes = tiny_counts.shape

    obs = pd.DataFrame(
        {
            "barcode": [f"barcode_{i}" for i in range(n_cells)],
            "sample": ["s1", "s1", "s2", "s2", "s3", "s3"],
        },
        index=[f"cell_index_{i}" for i in range(n_cells)],
    )
    var = pd.DataFrame(
        {
            "symbol": [f"symbol_{j}" for j in range(n_genes)],
            "feature_type": ["Gene Expression"] * n_genes,
        },
        index=[f"gene_index_{j}" for j in range(n_genes)],
    )

    adata = AnnData(X=sparse.csr_matrix(tiny_counts), obs=obs, var=var)
    adata.obsm["X_test"] = np.arange(n_cells * 2, dtype=np.float32).reshape(n_cells, 2)
    adata.varm["PCs"] = np.arange(n_genes * 3, dtype=np.float32).reshape(n_genes, 3)

    return adata


@pytest.fixture
def tiny_log_h5ad(tmp_path, tiny_adata: AnnData) -> tuple[str, np.ndarray]:
    """Write a small log1p-scale h5ad file and return its count matrix."""

    counts = np.asarray(tiny_adata.X.toarray(), dtype=np.float32)
    adata = tiny_adata.copy()
    adata.X = sparse.csr_matrix(np.log1p(counts))

    path = tmp_path / "tiny_log_scale.h5ad"
    adata.write_h5ad(path)

    return str(path), counts
