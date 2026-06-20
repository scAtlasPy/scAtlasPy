from __future__ import annotations

import json
import os
import shutil
import threading
import time
from pathlib import Path

import pytest

from .realdata_utils import cell_count_from_name, h5ad_shape, real_h5ad_paths_by_size


class _RssMonitor:
    def __init__(self, process, interval_seconds: float) -> None:
        self.process = process
        self.interval_seconds = interval_seconds
        self.start_rss = process.memory_info().rss
        self.peak_rss = self.start_rss
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._poll, daemon=True)

    def _poll(self) -> None:
        while not self._stop.is_set():
            self.peak_rss = max(self.peak_rss, self.process.memory_info().rss)
            self._stop.wait(self.interval_seconds)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.peak_rss = max(self.peak_rss, self.process.memory_info().rss)
        self._stop.set()
        self._thread.join(timeout=max(1.0, self.interval_seconds * 2))
        self.peak_rss = max(self.peak_rss, self.process.memory_info().rss)

    @property
    def peak_delta_mb(self) -> float:
        return (self.peak_rss - self.start_rss) / (1024 * 1024)


@pytest.mark.import_benchmark
@pytest.mark.realdata
@pytest.mark.slow
def test_realdata_h5ad_import_time_and_memory(
    tmp_path,
    realdata_dir,
    realdata_limit,
    realdata_min_cells,
    realdata_max_cells,
    report_dir,
):
    sap = pytest.importorskip("scatlaspy")
    psutil = pytest.importorskip("psutil")

    limit = realdata_limit if realdata_limit > 0 else None
    paths = real_h5ad_paths_by_size(
        realdata_dir,
        min_cells=realdata_min_cells,
        max_cells=realdata_max_cells,
        limit=limit,
    )
    if not paths:
        pytest.skip(f"no .h5ad files found under {realdata_dir}")

    import_mode = os.environ.get("SCATLASPY_IMPORT_BENCHMARK_IMPORT", "order")
    cells_per_block = int(os.environ.get("SCATLASPY_IMPORT_BENCHMARK_CELLS_PER_BLOCK", "1000"))
    blocks_per_pool = int(os.environ.get("SCATLASPY_IMPORT_BENCHMARK_BLOCKS_PER_POOL", "20"))
    sample_interval = float(os.environ.get("SCATLASPY_IMPORT_BENCHMARK_SAMPLE_SECONDS", "0.5"))
    keep_sasql = os.environ.get("SCATLASPY_IMPORT_BENCHMARK_KEEP_SASQL") == "1"
    keep_cells = {
        int(value)
        for value in os.environ.get("SCATLASPY_IMPORT_BENCHMARK_KEEP_CELLS", "").split(",")
        if value.strip()
    }

    run_id = os.environ.get("SCATLASPY_IMPORT_BENCHMARK_RUN_ID")
    if not run_id:
        run_id = time.strftime("%Y%m%d_%H%M%S")
    report_stem = (
        "realdata_import_benchmark"
        f"_{import_mode}_cpb{cells_per_block}_bpp{blocks_per_pool}_{run_id}"
    )
    report_json = report_dir / f"{report_stem}.json"
    report_jsonl = report_dir / f"{report_stem}.jsonl"
    latest_json = report_dir / (
        "realdata_import_benchmark"
        f"_latest_{import_mode}_cpb{cells_per_block}_bpp{blocks_per_pool}.json"
    )
    latest_jsonl = report_dir / (
        "realdata_import_benchmark"
        f"_latest_{import_mode}_cpb{cells_per_block}_bpp{blocks_per_pool}.jsonl"
    )
    report_jsonl.write_text("", encoding="utf-8")

    reports = []
    process = psutil.Process()

    for path in paths:
        expected_cells, expected_genes = h5ad_shape(path)
        cell_count_label = cell_count_from_name(path)
        atlas_path = tmp_path / f"import_benchmark_{cell_count_label}.sasql"
        atlas = sap.Atlas(atlas_path)
        started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")

        t0 = time.perf_counter()
        status = "passed"
        error = None
        with _RssMonitor(process, sample_interval) as monitor:
            try:
                atlas.load_h5ad(
                    str(path),
                    load_type=import_mode,
                    cells_per_block=cells_per_block,
                    blocks_per_pool=blocks_per_pool,
                )
                obs_n = atlas.query("SELECT COUNT(*) AS n FROM obs")["n"].iloc[0]
                var_n = atlas.query("SELECT COUNT(*) AS n FROM var")["n"].iloc[0]
                assert obs_n == expected_cells
                assert var_n == expected_genes
            except BaseException as exc:
                status = "failed"
                error = repr(exc)
                raise
            finally:
                import_seconds = time.perf_counter() - t0
                finished_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
                sasql_bytes = atlas_path.stat().st_size if atlas_path.exists() else 0
                record = {
                    "status": status,
                    "error": error,
                    "input_h5ad_path": str(path),
                    "input_h5ad_bytes": path.stat().st_size,
                    "cells": expected_cells,
                    "genes": expected_genes,
                    "import_load_type": import_mode,
                    "import_cells_per_block": cells_per_block,
                    "import_blocks_per_pool": blocks_per_pool,
                    "keep_sasql": keep_sasql or cell_count_label in keep_cells,
                    "import_seconds": import_seconds,
                    "peak_rss_delta_mb": monitor.peak_delta_mb,
                    "peak_rss_mb": monitor.peak_rss / (1024 * 1024),
                    "sasql_path": str(atlas_path),
                    "sasql_bytes": sasql_bytes,
                    "started_at": started_at,
                    "finished_at": finished_at,
                }
                reports.append(record)
                with report_jsonl.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(record, sort_keys=True) + "\n")
                report_json.write_text(
                    json.dumps(reports, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
                latest_json.write_text(
                    json.dumps(reports, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
                latest_jsonl.write_text(
                    report_jsonl.read_text(encoding="utf-8"),
                    encoding="utf-8",
                )
                print(
                    "IMPORT_BENCHMARK "
                    f"cells={expected_cells} genes={expected_genes} "
                    f"seconds={import_seconds:.2f} "
                    f"peak_rss_delta_mb={monitor.peak_delta_mb:.2f} "
                    f"sasql_gb={sasql_bytes / (1024 ** 3):.2f} "
                    f"status={status}",
                    flush=True,
                )
                atlas.close()
                if not (keep_sasql or cell_count_label in keep_cells):
                    if atlas_path.exists():
                        atlas_path.unlink()
                    wal_path = Path(str(atlas_path) + ".wal")
                    if wal_path.exists():
                        wal_path.unlink()
                    tmp_dir = atlas_path.parent / f"{atlas_path.name}.tmp"
                    if tmp_dir.exists():
                        shutil.rmtree(tmp_dir)
