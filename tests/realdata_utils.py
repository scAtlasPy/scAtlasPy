from __future__ import annotations

import re
from pathlib import Path

import pytest


def cell_count_from_name(path: Path) -> int:
    match = re.search(r"subsample_(\d+)_cells", path.name)
    if match:
        return int(match.group(1))
    return path.stat().st_size


def real_h5ad_paths(data_dir: Path, limit: int | None = None) -> list[Path]:
    paths = sorted(data_dir.glob("*.h5ad"), key=cell_count_from_name)
    if not paths:
        paths = sorted(data_dir.glob("**/*.h5ad"), key=cell_count_from_name)
    if limit is not None:
        return paths[:limit]
    return paths


def real_h5ad_paths_by_size(
    data_dir: Path,
    *,
    min_cells: int | None = None,
    max_cells: int | None = None,
    limit: int | None = None,
) -> list[Path]:
    selected = []
    for path in real_h5ad_paths(data_dir, limit=None):
        n_cells = cell_count_from_name(path)
        if min_cells is not None and n_cells < min_cells:
            continue
        if max_cells is not None and n_cells > max_cells:
            continue
        selected.append(path)
    if limit is not None:
        return selected[:limit]
    return selected


def h5ad_shape(path: Path) -> tuple[int, int]:
    ad = pytest.importorskip("anndata")
    backed = ad.read_h5ad(path, backed="r")
    try:
        return backed.n_obs, backed.n_vars
    finally:
        backed.file.close()
