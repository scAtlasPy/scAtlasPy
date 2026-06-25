from __future__ import annotations

import gc
import json
import os
import time

import numpy as np
import pytest
from sklearn.cluster import MiniBatchKMeans

sap = pytest.importorskip("scatlaspy")

from .data_generators import write_h5ad_shards

psutil = pytest.importorskip("psutil")


def _prepare_streaming_atlas(
    tmp_path,
    path,
    *,
    n_cells,
    n_genes,
    density,
    seed,
    base_name,
    load_type="random",
):
    cells_per_block = int(os.environ.get("SCATLASPY_L3_IMPORT_CELLS_PER_BLOCK", "1000"))
    blocks_per_pool = int(os.environ.get("SCATLASPY_L3_IMPORT_BLOCKS_PER_POOL", "4"))
    n_shards = int(os.environ.get("SCATLASPY_L3_SHARDS", "1"))
    is_multi_file = n_shards > 1

    paths, params, shape = write_h5ad_shards(
        tmp_path,
        base_name=base_name,
        seed=seed,
        n_cells=n_cells,
        n_genes=n_genes,
        density=density,
        n_shards=n_shards,
        dtype="float32",
        max_count=10,
    )
    atlas = sap.Atlas(path)
    import_input = [str(p) for p in paths] if is_multi_file else str(paths[0])
    import_t0 = time.perf_counter()
    atlas.load_h5ad(
        import_input,
        load_type=load_type,
        cells_per_block=cells_per_block,
        blocks_per_pool=blocks_per_pool,
    )
    params.update(
        {
            "import_load_type": load_type,
            "import_cells_per_block": cells_per_block,
            "import_blocks_per_pool": blocks_per_pool,
            "import_seconds": time.perf_counter() - import_t0,
        }
    )
    atlas.connection.execute(
        "ALTER TABLE var ADD COLUMN IF NOT EXISTS zero_scale_transform REAL DEFAULT 0.0"
    )
    atlas.build_read_index(use_hvg=False, use_data="data_count")
    gc.collect()
    return atlas, params, shape, paths


@pytest.mark.l3
@pytest.mark.slow
def test_l3_single_pass_streaming_completeness_and_memory(
    tmp_path, l3_cells, memory_limit_mb, write_generation_params, report_dir
):
    n_genes = int(os.environ.get("SCATLASPY_L3_GENES", "64"))
    density = float(os.environ.get("SCATLASPY_L3_DENSITY", "0.001"))
    batch_size = int(os.environ.get("SCATLASPY_L3_BATCH_SIZE", "4096"))
    load_type = os.environ.get("SCATLASPY_L3_IMPORT", "random")

    atlas, params, shape, input_paths = _prepare_streaming_atlas(
        tmp_path,
        tmp_path / "l3_stream.sasql",
        n_cells=l3_cells,
        n_genes=n_genes,
        density=density,
        seed=3001,
        base_name="l3_stream",
        load_type=load_type,
    )
    params_path = write_generation_params("l3_stream", params)

    process = psutil.Process()
    start_rss = process.memory_info().rss
    peak_rss = start_rss
    rows = 0
    batch_shapes = []
    t0 = time.perf_counter()

    for batch in atlas.get_minibatch_dense(
        pass_mode="single-pass", batch_size=batch_size, buffer_batch_num=2
    ):
        rows += batch.shape[0]
        batch_shapes.append(batch.shape)
        peak_rss = max(peak_rss, process.memory_info().rss)

    elapsed = time.perf_counter() - t0
    assert rows == shape[0]
    assert all(cols == shape[1] for _, cols in batch_shapes)
    expected_last = shape[0] % batch_size or min(batch_size, shape[0])
    assert batch_shapes[-1][0] == expected_last

    delta_mb = (peak_rss - start_rss) / (1024 * 1024)
    assert delta_mb < memory_limit_mb

    report = {
        "params_path": str(params_path),
        "rows": rows,
        "n_batches": len(batch_shapes),
        "batch_size": batch_size,
        "input_h5ad_paths": [str(path) for path in input_paths],
        "import_load_type": load_type,
        "import_cells_per_block": params["import_cells_per_block"],
        "import_blocks_per_pool": params["import_blocks_per_pool"],
        "import_seconds": params["import_seconds"],
        "peak_rss_delta_mb": delta_mb,
        "elapsed_seconds": elapsed,
    }
    (report_dir / "l3_stream_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    atlas.close()


@pytest.mark.l3
@pytest.mark.slow
@pytest.mark.ml
def test_l3_multi_pass_stops_and_supports_minibatch_ml(tmp_path, l3_cells, report_dir):
    n_cells = min(l3_cells, int(os.environ.get("SCATLASPY_L3_ML_CELLS", "120000")))
    n_genes = int(os.environ.get("SCATLASPY_L3_ML_GENES", "32"))
    density = float(os.environ.get("SCATLASPY_L3_ML_DENSITY", "0.002"))
    batch_size = int(os.environ.get("SCATLASPY_L3_ML_BATCH_SIZE", "2048"))
    max_batches = int(os.environ.get("SCATLASPY_L3_ML_MAX_BATCHES", "6"))
    load_type = os.environ.get("SCATLASPY_L3_ML_IMPORT", "random")

    atlas, params, shape, input_paths = _prepare_streaming_atlas(
        tmp_path,
        tmp_path / "l3_ml.sasql",
        n_cells=n_cells,
        n_genes=n_genes,
        density=density,
        seed=3002,
        base_name="l3_ml",
        load_type=load_type,
    )

    model = MiniBatchKMeans(n_clusters=4, random_state=0, batch_size=batch_size)
    seen = 0
    n_batches = 0
    t0 = time.perf_counter()
    for batch in atlas.get_minibatch_dense(
        pass_mode="multi-pass",
        batch_size=batch_size,
        max_batches=max_batches,
        buffer_batch_num=2,
    ):
        model.partial_fit(batch)
        seen += batch.shape[0]
        n_batches += 1

    assert n_batches == max_batches
    assert seen > 0
    assert model.cluster_centers_.shape == (4, shape[1])
    elapsed = time.perf_counter() - t0
    assert elapsed < 120
    report = {
        "rows_seen": seen,
        "n_batches": n_batches,
        "max_batches": max_batches,
        "batch_size": batch_size,
        "elapsed_seconds": elapsed,
        "input_h5ad_paths": [str(path) for path in input_paths],
        "import_load_type": load_type,
        "import_cells_per_block": params["import_cells_per_block"],
        "import_blocks_per_pool": params["import_blocks_per_pool"],
        "import_seconds": params["import_seconds"],
        "model": "MiniBatchKMeans",
        "cluster_centers_shape": list(model.cluster_centers_.shape),
    }
    (report_dir / "l3_ml_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    atlas.close()
