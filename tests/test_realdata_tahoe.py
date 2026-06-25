from __future__ import annotations

import json
import os
import time

import pytest

from .realdata_utils import h5ad_shape, real_h5ad_paths_by_size


@pytest.mark.realdata
@pytest.mark.slow
def test_real_tahoe_h5ad_import_streaming_and_downstream(
    tmp_path,
    realdata_dir,
    realdata_limit,
    realdata_max_cells,
    memory_limit_mb,
    report_dir,
):
    paths = real_h5ad_paths_by_size(
        realdata_dir,
        max_cells=realdata_max_cells,
        limit=realdata_limit,
    )
    if not paths:
        pytest.skip(f"no .h5ad files found under {realdata_dir}")

    sap = pytest.importorskip("scatlaspy")
    psutil = pytest.importorskip("psutil")
    MiniBatchKMeans = pytest.importorskip(
        "sklearn.cluster"
    ).MiniBatchKMeans

    import_mode = os.environ.get("SCATLASPY_REALDATA_IMPORT", "order")
    if import_mode not in {"order", "random"}:
        raise ValueError("SCATLASPY_REALDATA_IMPORT 只能是 order 或 random")

    multi_file_import = os.environ.get("SCATLASPY_REALDATA_MULTI_FILE", "0") == "1"
    cells_per_block = int(os.environ.get("SCATLASPY_REALDATA_CELLS_PER_BLOCK", "1000"))
    blocks_per_pool = int(os.environ.get("SCATLASPY_REALDATA_BLOCKS_PER_POOL", "4"))
    batch_size = int(os.environ.get("SCATLASPY_REALDATA_BATCH_SIZE", "4096"))
    max_train_batches = int(os.environ.get("SCATLASPY_REALDATA_TRAIN_BATCHES", "4"))

    if multi_file_import:
        selected_paths = paths
        expected_cells = sum(h5ad_shape(path)[0] for path in selected_paths)
        expected_genes = h5ad_shape(selected_paths[0])[1]
        import_input = [str(path) for path in selected_paths]
    else:
        selected_paths = paths[:1]
        expected_cells, expected_genes = h5ad_shape(selected_paths[0])
        import_input = str(selected_paths[0])

    process = psutil.Process()
    start_rss = process.memory_info().rss
    peak_rss = start_rss
    t0 = time.perf_counter()

    atlas = sap.Atlas(tmp_path / "real_tahoe.sasql")
    atlas.load_h5ad(
        import_input,
        load_type=import_mode,
        cells_per_block=cells_per_block,
        blocks_per_pool=blocks_per_pool,
    )
    import_seconds = time.perf_counter() - t0
    peak_rss = max(peak_rss, process.memory_info().rss)

    obs_n = atlas.query("SELECT COUNT(*) AS n FROM obs")["n"].iloc[0]
    var_n = atlas.query("SELECT COUNT(*) AS n FROM var")["n"].iloc[0]
    assert obs_n == expected_cells
    assert var_n == expected_genes

    atlas.connection.execute(
        "ALTER TABLE var ADD COLUMN IF NOT EXISTS zero_scale_transform REAL DEFAULT 0.0"
    )
    atlas.build_read_index(use_hvg=False, use_data="data_count")

    rows = 0
    n_batches = 0
    for batch in atlas.get_minibatch_dense(
        pass_mode="single-pass", batch_size=batch_size, buffer_batch_num=2
    ):
        rows += batch.shape[0]
        n_batches += 1
        assert batch.shape[1] == expected_genes
        peak_rss = max(peak_rss, process.memory_info().rss)

    assert rows == expected_cells
    assert n_batches >= 1

    model = MiniBatchKMeans(n_clusters=4, random_state=0, batch_size=batch_size)
    train_batches = 0
    for batch in atlas.get_minibatch_dense(
        pass_mode="multi-pass",
        batch_size=batch_size,
        max_batches=max_train_batches,
        buffer_batch_num=2,
    ):
        model.partial_fit(batch)
        train_batches += 1

    assert train_batches == max_train_batches
    assert model.cluster_centers_.shape == (4, expected_genes)

    peak_delta_mb = (peak_rss - start_rss) / (1024 * 1024)
    assert peak_delta_mb < memory_limit_mb

    report = {
        "input_h5ad_paths": [str(path) for path in selected_paths],
        "import_load_type": import_mode,
        "import_cells_per_block": cells_per_block,
        "import_blocks_per_pool": blocks_per_pool,
        "batch_size": batch_size,
        "expected_cells": expected_cells,
        "expected_genes": expected_genes,
        "stream_batches": n_batches,
        "train_batches": train_batches,
        "peak_rss_delta_mb": peak_delta_mb,
        "import_seconds": import_seconds,
    }
    (report_dir / "real_tahoe_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    atlas.close()
