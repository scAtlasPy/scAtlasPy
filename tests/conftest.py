from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest


def pytest_collection_modifyitems(config, items):
    run_l3 = (
        os.environ.get("SCATLASPY_RUN_L3") == "1"
        or config.getoption("--scatlaspy-run-l3")
    )
    run_l4 = (
        os.environ.get("SCATLASPY_RUN_L4") == "1"
        or config.getoption("--scatlaspy-run-l4")
    )
    run_realdata = (
        os.environ.get("SCATLASPY_RUN_REALDATA") == "1"
        or config.getoption("--scatlaspy-run-realdata")
    )

    skip_l3 = pytest.mark.skip(
        reason="set SCATLASPY_RUN_L3=1 to run L3 streaming/memory tests"
    )
    skip_l4 = pytest.mark.skip(
        reason="set SCATLASPY_RUN_L4=1 to run L4 release stress tests"
    )
    skip_realdata = pytest.mark.skip(
        reason="set SCATLASPY_RUN_REALDATA=1 to run real-data tests"
    )

    for item in items:
        if "l3" in item.keywords and not run_l3:
            item.add_marker(skip_l3)
        if "l4" in item.keywords and not run_l4:
            item.add_marker(skip_l4)
        if "realdata" in item.keywords and not run_realdata:
            item.add_marker(skip_realdata)


@pytest.fixture
def rng():
    return np.random.default_rng(20240229)


@pytest.fixture
def atlas_path(tmp_path: Path) -> Path:
    return tmp_path / "atlas.sasql"


@pytest.fixture
def write_generation_params(tmp_path: Path):
    def _write(name: str, params: dict) -> Path:
        path = tmp_path / f"{name}.params.json"
        path.write_text(json.dumps(params, indent=2, sort_keys=True), encoding="utf-8")
        return path

    return _write


@pytest.fixture
def report_dir() -> Path:
    path = Path.cwd() / "tmp" / "pytest_reports"
    path.mkdir(parents=True, exist_ok=True)
    return path


def pytest_addoption(parser):
    parser.addoption(
        "--scatlaspy-l3-cells",
        action="store",
        default=os.environ.get("SCATLASPY_L3_CELLS", "100000"),
        help="Number of synthetic cells for L3 tests.",
    )
    parser.addoption(
        "--scatlaspy-l4-cells",
        action="store",
        default=os.environ.get("SCATLASPY_L4_CELLS", "1000000"),
        help="Number of synthetic cells for L4 tests.",
    )
    parser.addoption(
        "--scatlaspy-memory-mb",
        action="store",
        default=os.environ.get("SCATLASPY_MEMORY_LIMIT_MB", "2048"),
        help="RSS ceiling in MB for opt-in L3/L4 streaming tests.",
    )
    parser.addoption(
        "--scatlaspy-run-l3",
        action="store_true",
        default=False,
        help="Run opt-in L3 streaming/memory tests.",
    )
    parser.addoption(
        "--scatlaspy-run-l4",
        action="store_true",
        default=False,
        help="Run opt-in L4 release stress tests.",
    )
    parser.addoption(
        "--scatlaspy-run-realdata",
        action="store_true",
        default=False,
        help="Run opt-in tests against user-provided real datasets.",
    )
    parser.addoption(
        "--scatlaspy-realdata-dir",
        action="store",
        default=os.environ.get("SCATLASPY_REALDATA_DIR", "data"),
        help="Directory containing user-provided real h5ad datasets.",
    )
    parser.addoption(
        "--scatlaspy-realdata-limit",
        action="store",
        default=os.environ.get("SCATLASPY_REALDATA_LIMIT", "2"),
        help="Maximum number of real h5ad files to use in one test.",
    )
    parser.addoption(
        "--scatlaspy-realdata-max-cells",
        action="store",
        default=os.environ.get("SCATLASPY_REALDATA_MAX_CELLS", "0"),
        help="Optional maximum cell count for real-data tests; 0 means no limit.",
    )
    parser.addoption(
        "--scatlaspy-realdata-min-cells",
        action="store",
        default=os.environ.get("SCATLASPY_REALDATA_MIN_CELLS", "0"),
        help="Optional minimum cell count for real-data tests; 0 means no limit.",
    )


@pytest.fixture
def l3_cells(request) -> int:
    return int(request.config.getoption("--scatlaspy-l3-cells"))


@pytest.fixture
def l4_cells(request) -> int:
    return int(request.config.getoption("--scatlaspy-l4-cells"))


@pytest.fixture
def memory_limit_mb(request) -> int:
    return int(request.config.getoption("--scatlaspy-memory-mb"))


@pytest.fixture
def realdata_dir(request) -> Path:
    return Path(request.config.getoption("--scatlaspy-realdata-dir"))


@pytest.fixture
def realdata_limit(request) -> int:
    return int(request.config.getoption("--scatlaspy-realdata-limit"))


@pytest.fixture
def realdata_max_cells(request) -> int | None:
    value = int(request.config.getoption("--scatlaspy-realdata-max-cells"))
    return value or None


@pytest.fixture
def realdata_min_cells(request) -> int | None:
    value = int(request.config.getoption("--scatlaspy-realdata-min-cells"))
    return value or None
