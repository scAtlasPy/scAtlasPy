"""Pytest fixtures for small, isolated Atlas databases."""

from __future__ import annotations

from collections.abc import Callable, Iterator

import pytest

import scatlaspy as sap

from .helpers import make_counts_adata, make_workflow_adata


@pytest.fixture(autouse=True)
def quiet_progress() -> None:
    """Keep tests readable by disabling progress bars and verbose logs."""

    sap.set_progress(False)
    sap.set_verbosity("WARNING")


@pytest.fixture
def counts_adata():
    """Small hand-checkable AnnData fixture."""

    return make_counts_adata()


@pytest.fixture
def workflow_adata():
    """Small AnnData fixture suitable for the full analysis workflow."""

    return make_workflow_adata()


@pytest.fixture
def atlas_from_adata(tmp_path) -> Iterator[Callable]:
    """Create Atlas databases under tmp_path and close them after each test."""

    atlases = []

    def factory(adata, name: str = "test.sasql", db_memory_limit: str = "1GB"):
        atlas = sap.Atlas(tmp_path / name, db_memory_limit=db_memory_limit)
        atlas.load_anndata(adata)
        atlases.append(atlas)
        return atlas

    yield factory

    for atlas in atlases:
        atlas.close()

