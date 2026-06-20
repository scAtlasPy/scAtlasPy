from __future__ import annotations

from dataclasses import asdict, dataclass
import gc
from pathlib import Path
from typing import Literal

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


@dataclass(frozen=True)
class MatrixParams:
    seed: int
    n_cells: int
    n_genes: int
    density: float
    dtype: str
    mode: str
    duplicate_coordinates: bool = False
    shuffled_coordinates: bool = False
    include_extreme_values: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def _obs(n_cells: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "batch": [f"batch_{i % 3}" for i in range(n_cells)],
            "cell_group": [f"group_{i % 2}" for i in range(n_cells)],
        },
        index=[f"cell_{i}" for i in range(n_cells)],
    )


def _var(n_genes: int) -> pd.DataFrame:
    names = []
    for i in range(n_genes):
        if i % 7 == 0:
            names.append(f"MT-GENE{i}")
        elif i % 7 == 1:
            names.append(f"RPL{i}")
        else:
            names.append(f"gene_{i}")
    return pd.DataFrame({"feature_type": "Gene Expression"}, index=names)


def make_anndata_from_csr(X: sparse.csr_matrix) -> ad.AnnData:
    return ad.AnnData(X=X, obs=_obs(X.shape[0]), var=_var(X.shape[1]))


def make_l0_edge_anndata(dtype: str = "float32") -> tuple[ad.AnnData, dict]:
    rows = np.array(
        [1, 2, 2, 3, 3, 3, 6, 7, 7, 8, 8, 8, 8],
        dtype=np.int64,
    )
    cols = np.array(
        [0, 1, 1, 0, 2, 5, 4, 0, 5, 2, 3, 3, 5],
        dtype=np.int64,
    )
    data = np.array(
        [1, 2, 3, -4, np.nan, np.inf, 1e20, 0.25, -0.5, 6, 7, 8, 9],
        dtype=dtype,
    )

    order = np.array([8, 2, 12, 0, 10, 4, 1, 6, 7, 11, 5, 3, 9])
    coo = sparse.coo_matrix((data[order], (rows[order], cols[order])), shape=(10, 6))
    csr = coo.tocsr()
    csr.sum_duplicates()

    params = MatrixParams(
        seed=0,
        n_cells=10,
        n_genes=6,
        density=float(csr.nnz / (10 * 6)),
        dtype=dtype,
        mode="l0_edge",
        duplicate_coordinates=True,
        shuffled_coordinates=True,
        include_extreme_values=True,
    ).to_dict()
    return make_anndata_from_csr(csr), params


def make_random_counts_anndata(
    *,
    seed: int,
    n_cells: int,
    n_genes: int,
    density: float,
    dtype: Literal["float32", "float64", "int32", "int64"] = "float32",
    max_count: int = 20,
) -> tuple[ad.AnnData, dict]:
    rng = np.random.default_rng(seed)
    size = n_cells * n_genes
    nnz = max(1, int(size * density)) if size else 0

    if nnz == 0:
        X = sparse.csr_matrix((n_cells, n_genes), dtype=dtype)
    else:
        rows = rng.integers(0, n_cells, size=nnz, endpoint=False)
        cols = rng.integers(0, n_genes, size=nnz, endpoint=False)
        values = rng.integers(1, max_count + 1, size=nnz).astype(dtype)
        order = rng.permutation(nnz)
        X = sparse.coo_matrix(
            (values[order], (rows[order], cols[order])),
            shape=(n_cells, n_genes),
        ).tocsr()
        X.sum_duplicates()

    params = MatrixParams(
        seed=seed,
        n_cells=n_cells,
        n_genes=n_genes,
        density=density,
        dtype=dtype,
        mode="random_counts",
        duplicate_coordinates=True,
        shuffled_coordinates=True,
        include_extreme_values=False,
    ).to_dict()
    return make_anndata_from_csr(X), params


def write_h5ad_shards(
    directory: Path,
    *,
    base_name: str,
    seed: int,
    n_cells: int,
    n_genes: int,
    density: float,
    n_shards: int,
    dtype: Literal["float32", "float64", "int32", "int64"] = "float32",
    max_count: int = 20,
) -> tuple[list[Path], dict, tuple[int, int]]:
    """Write deterministic h5ad shards without holding all shards at once."""
    directory.mkdir(parents=True, exist_ok=True)
    n_shards = max(1, int(n_shards))
    base = n_cells // n_shards
    remainder = n_cells % n_shards

    paths = []
    shard_sizes = []
    cell_offset = 0
    for shard_id in range(n_shards):
        shard_cells = base + int(shard_id < remainder)
        shard_sizes.append(shard_cells)
        adata, _ = make_random_counts_anndata(
            seed=seed + shard_id,
            n_cells=shard_cells,
            n_genes=n_genes,
            density=density,
            dtype=dtype,
            max_count=max_count,
        )
        adata.obs_names = [f"cell_{cell_offset + i}" for i in range(shard_cells)]
        adata.obs["source_shard"] = f"shard_{shard_id}"

        path = directory / f"{base_name}.shard_{shard_id}.h5ad"
        adata.write_h5ad(path)
        paths.append(path)
        cell_offset += shard_cells

        del adata
        gc.collect()

    params = {
        "seed": seed,
        "n_cells": n_cells,
        "n_genes": n_genes,
        "density": density,
        "dtype": dtype,
        "mode": "h5ad_shards",
        "n_shards": n_shards,
        "shard_sizes": shard_sizes,
        "h5ad_paths": [str(path) for path in paths],
        "duplicate_coordinates": True,
        "shuffled_coordinates": True,
        "include_extreme_values": False,
    }
    return paths, params, (n_cells, n_genes)


def finite_csr(X) -> sparse.csr_matrix:
    csr = X.tocsr().astype(np.float32)
    data = csr.data.copy()
    data[~np.isfinite(data)] = 0.0
    csr.data = data
    csr.eliminate_zeros()
    return csr


def csr_to_dense_float32(X) -> np.ndarray:
    return X.toarray().astype(np.float32, copy=False)
