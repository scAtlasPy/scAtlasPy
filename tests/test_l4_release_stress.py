from __future__ import annotations

import json
import os
import time

import pytest

from .realdata_utils import cell_count_from_name, h5ad_shape, real_h5ad_paths_by_size


@pytest.mark.l4
@pytest.mark.stress
@pytest.mark.ml
@pytest.mark.realdata
def test_l4_real_tahoe_release_scale_streaming_and_training(
    tmp_path,
    realdata_dir,
    realdata_limit,
    realdata_min_cells,
    realdata_max_cells,
    memory_limit_mb,
    report_dir,
):
    sap = pytest.importorskip("scatlaspy")
    psutil = pytest.importorskip("psutil")
    IncrementalPCA = pytest.importorskip(
        "sklearn.decomposition"
    ).IncrementalPCA

    paths = real_h5ad_paths_by_size(
        realdata_dir,
        min_cells=realdata_min_cells or 1_000_000,
        max_cells=realdata_max_cells,
        limit=realdata_limit,
    )
    if not paths:
        pytest.skip(f"no >=1,000,000-cell .h5ad files found under {realdata_dir}")

    import_mode = os.environ.get("SCATLASPY_L4_REALDATA_IMPORT", "order")
    cells_per_block = int(os.environ.get("SCATLASPY_L4_IMPORT_CELLS_PER_BLOCK", "2000"))
    blocks_per_pool = int(os.environ.get("SCATLASPY_L4_IMPORT_BLOCKS_PER_POOL", "4"))
    batch_size = int(os.environ.get("SCATLASPY_L4_BATCH_SIZE", "512"))
    gene_limit = int(os.environ.get("SCATLASPY_L4_GENE_LIMIT", "512"))
    max_train_batches = int(os.environ.get("SCATLASPY_L4_TRAIN_BATCHES", "8"))

    reports = []
    process = psutil.Process()

    for path in paths:
        expected_cells, expected_genes = h5ad_shape(path)
        n_index_genes = min(gene_limit, expected_genes)
        atlas = sap.Atlas(tmp_path / f"l4_real_{cell_count_from_name(path)}.sasql")

        start_rss = process.memory_info().rss
        peak_rss = start_rss
        t0 = time.perf_counter()

        atlas.load_h5ad(
            str(path),
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

        conn = atlas.connection
        conn.execute("ALTER TABLE var ADD COLUMN IF NOT EXISTS l4_gene_subset BOOLEAN")
        conn.execute("UPDATE var SET l4_gene_subset = atlas_gene_id < ?", [n_index_genes])
        conn.execute(
            "ALTER TABLE var ADD COLUMN IF NOT EXISTS zero_scale_transform REAL DEFAULT 0.0"
        )
        atlas.build_read_index(
            gene_condition="l4_gene_subset",
            use_hvg=False,
            use_data="data_count",
        )

        rows = 0
        n_batches = 0
        t_stream = time.perf_counter()
        for batch in atlas.get_minibatch_dense(
            pass_mode="single-pass",
            batch_size=batch_size,
            buffer_batch_num=2,
        ):
            rows += batch.shape[0]
            n_batches += 1
            assert batch.shape[1] == n_index_genes
            peak_rss = max(peak_rss, process.memory_info().rss)
        stream_seconds = time.perf_counter() - t_stream

        assert rows == expected_cells
        assert n_batches >= 1

        ipca = IncrementalPCA(n_components=min(8, n_index_genes))
        train_batches = 0
        t_train = time.perf_counter()
        for batch in atlas.get_minibatch_dense(
            pass_mode="multi-pass",
            batch_size=batch_size,
            max_batches=max_train_batches,
            buffer_batch_num=2,
        ):
            ipca.partial_fit(batch)
            train_batches += 1
            peak_rss = max(peak_rss, process.memory_info().rss)
        train_seconds = time.perf_counter() - t_train

        assert train_batches == max_train_batches
        assert ipca.components_.shape[1] == n_index_genes

        peak_delta_mb = (peak_rss - start_rss) / (1024 * 1024)
        assert peak_delta_mb < memory_limit_mb

        reports.append(
            {
                "input_h5ad_path": str(path),
                "cells": expected_cells,
                "genes": expected_genes,
                "indexed_genes": n_index_genes,
                "import_load_type": import_mode,
                "import_cells_per_block": cells_per_block,
                "import_blocks_per_pool": blocks_per_pool,
                "batch_size": batch_size,
                "stream_batches": n_batches,
                "train_batches": train_batches,
                "peak_rss_delta_mb": peak_delta_mb,
                "import_seconds": import_seconds,
                "stream_seconds": stream_seconds,
                "train_seconds": train_seconds,
            }
        )
        atlas.close()

    (report_dir / "l4_real_tahoe_release_report.json").write_text(
        json.dumps(reports, indent=2, sort_keys=True),
        encoding="utf-8",
    )
