from __future__ import annotations
from _duckdb import DuckDBPyConnection
import os
import logging
import h5py
import pyarrow as pa
import time
from datetime import datetime
import gc
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from os import PathLike
from scipy import sparse
from . import progress
from typing import TYPE_CHECKING, Any, Literal
if TYPE_CHECKING:   # TYPE_CHECKING = imports for IDEs / type checkers; at runtime, this import is not executed to avoid circular imports
    from ..data import Atlas

XScale = Literal["count", "log"]

# Get the logger
logger = logging.getLogger("Atlas")
logger.addHandler(logging.NullHandler())


# Unified h5ad import interface
def load_h5ad(
    h5ad_path: PathLike[str] | str | list[PathLike[str] | str],
    atlas: Atlas,
    *,
    load_type: Literal["order", "random"] = "random",
    cells_per_block: int | None = None,
    commit_every: int = 1,
) -> Any:
    """Import h5ad files into an Atlas database.

    This function is the unified entry point for importing h5ad data into Atlas. It can read a single ``.h5ad`` file or
    a list of multiple ``.h5ad`` files, and writes cell metadata, gene metadata, and the expression matrix
    into the Atlas DuckDB database.

    During import, it automatically dispatches to the corresponding implementation based on ``load_type`` and the type of ``h5ad_path``:

    - single file + ``"order"``：import in the original cell order；
    - single file + ``"random"``：import randomly using a shuffle-window strategy；
    - multiple files + ``"order"``：import according to the file list order and the original cell order within each file；
    - multiple files + ``"random"``：split multiple files into blocks and import them with global randomization。

    The expression matrix is stored uniformly on the count scale and written to the ``X_HyS_data.data_count`` field.
    If the input ``X`` is detected to be on the log scale, it is converted back to counts before writing.

    Parameters
    ----------
    h5ad_path
        Input ``.h5ad`` file path, or a list of multiple ``.h5ad`` file paths.
    atlas
        Atlas object. The function obtains a DuckDB connection through ``atlas.connect("r+")`` and
        writes data into this Atlas database.
    load_type
        Import mode. Only ``"order"`` and ``"random"`` are supported.
        When ``h5ad_path`` is a single path, single-file ordered or random import is performed respectively;
        when ``h5ad_path`` is a list, multi-file ordered or random import is performed respectively.
    cells_per_block
        Number of cells contained in each contiguous cell block when reading and writing the expression matrix.
        If ``None``, a default value is automatically estimated based on the total number of cells.
    commit_every
        Commit the active DuckDB transaction once every N import windows or mini-batches.

    Returns
    -------
    Any
        Returns the result of the called underlying import function. Currently, this is mainly used to execute import side effects, and the return value is usually not relied upon.
        return value.

    Notes
    -----
    ``random`` import reorders cells, so single-file random import does not import ``obsm`` by default,
    to avoid misalignment between embeddings and the reordered ``obs``;
    ``order`` import preserves the original order, so ``obsm`` and ``varm`` can be safely imported.

    Examples
    --------
    Import a single h5ad file in order::

        atlas = sap.Atlas(r"F:\\data\\pbmc")
        atlas.load_h5ad(r"F:\\data\\pbmc.h5ad", load_type="order")

    Import multiple files with random blocks::

        atlas.load_h5ad(
            [r"F:\\data\\batch1.h5ad", r"F:\\data\\batch2.h5ad"],
            load_type="random",
            cells_per_block=1000,
        )"""

    start_time = datetime.now()

    # =====================================================
    # 1. Parameter checks
    # =====================================================
    valid_load_types = {
        "order",
        "random",
    }

    if load_type not in valid_load_types:
        raise ValueError(
            "load_type must be "
            f"{sorted(valid_load_types)}, current value: {load_type}"
        )

    if cells_per_block is not None and not isinstance(cells_per_block, int):
        raise TypeError(
            f"cells_per_block must be int or None, current type: {type(cells_per_block)}"
        )

    if cells_per_block is not None and cells_per_block <= 0:
        raise ValueError("cells_per_block must be > 0")

    if not isinstance(commit_every, int):
        raise TypeError(
            f"commit_every must be int, current type: {type(commit_every)}"
        )

    if commit_every <= 0:
        raise ValueError("commit_every must be > 0")


    # =====================================================
    # 2. Multi-file import: automatically select ordered or random logic based on load_type
    # =====================================================
    if isinstance(h5ad_path, (list, tuple)):
        if load_type == "order":
            logger.info("[INFO] load_type = order, multi-file ordered import")
            return _load_h5ad_list_order(
                h5ad_paths=h5ad_path,
                atlas=atlas,
                cells_per_block=cells_per_block,
                commit_every=commit_every,
            )

        logger.info("[INFO] load_type = random, multi-file random import")
        return _load_h5ad_list_random(
            h5ad_paths=h5ad_path,
            atlas=atlas,
            cells_per_block=cells_per_block,
            commit_every=commit_every,
        )

    # =====================================================
    # 3. Single-file import: first read n_cells lightly, then enter the corresponding logic
    # =====================================================

    if not isinstance(h5ad_path, (str, PathLike)):
        raise TypeError(
            f"h5ad_path must be str, current type: {type(h5ad_path)}"
        )

    h5ad_path = os.fspath(h5ad_path)

    # Lightly read n_cells to unify the default value of cells_per_block
    adata_backed = sc.read_h5ad(h5ad_path, backed="r")
    n_cells = adata_backed.n_obs
    cells_per_block = _normalize_cells_per_block(cells_per_block, n_cells)

    # =====================================================
    # 4. order: ordered import
    # =====================================================
    if load_type == "order":

        logger.info("[INFO] load_type = order")

        return _load_h5ad_order(
            h5ad_path=h5ad_path,
            atlas=atlas,
            cells_per_block=cells_per_block,
            commit_every=commit_every,
        )

    # =====================================================
    # 5. random: regular random import
    # =====================================================
    if load_type == "random":

        logger.info("[INFO] load_type = random")

        return _load_h5ad_random(
            h5ad_path=h5ad_path,
            atlas=atlas,
            cells_per_block=cells_per_block,
            commit_every=commit_every,
        )

    logger.info(f"load_h5ad Done, elapsed time: {(datetime.now() - start_time).total_seconds():.2f} seconds")
    return None


# Write an AnnData object into an Atlas database
def load_anndata(adata:AnnData, atlas:Atlas):

    """Write an AnnData object into an Atlas database.

    This function directly accepts an in-memory AnnData object and writes its ``obs``, ``var``,
    ``X``, ``obsm``, and ``varm`` into the Atlas database. It is suitable for scenarios where Scanpy or other
    tools have already been used for reading, filtering, or preprocessing before transferring the data to Atlas for management.

    Unlike the backed chunked import of ``load_h5ad``, this function requires AnnData to already be in memory,
    so it is more suitable for small to medium-sized datasets or already sampled datasets.

    Parameters
    ----------
    adata
        AnnData object. The function writes its ``obs``, ``var``, expression matrix, and supported results into the Atlas database.
    atlas
        Atlas object. The object must already be connected or be connectable to a DuckDB database.

    Returns
    -------
    None
        The result is written directly into the Atlas database and no object is returned.

    Notes
    -----
    This path rebuilds the ``obs``, ``var``, ``X_HyS_indptr``, and ``X_HyS_data`` tables, and writes
    two-dimensional arrays in ``obsm`` and ``varm`` as ``obsm_*`` and ``varm_*`` tables.

    Examples
    --------
    Read with Scanpy and import::

        adata = sc.read_h5ad(r"F:\\data\\pbmc.h5ad")
        atlas = sap.Atlas(r"F:\\data\\pbmc")
        atlas.load_anndata(adata)
    """

    try:
        logger.info("Preparing data tables...")

        if hasattr(adata, 'obs'):
            _add_obs(adata, atlas)  # cell table data (corresponding to obs),
        else:
            logger.info("Skipping obs layer")

        if hasattr(adata, 'var'):
            _add_var(adata, atlas)  # gene table data (corresponding to var)
        else:
            logger.info("Skipping var layer")

        if hasattr(adata, 'X'):
            start_time = time.time()
            _add_x_hys_chunked(adata, atlas, chunk_size=500) # Import X table data in chunks
            end_time = time.time()
            logger.info(" Time used to import the X table: " + str(end_time - start_time))
        else:
            logger.info("Skipping X layer")

        if hasattr(adata, 'obsm'):
            _add_obsm(adata,atlas)
        else:
            logger.info("Skipping obsm layer")

        if hasattr(adata, 'varm'):
            _add_varm(adata,atlas)
        else:
            logger.info("Skipping varm layer")

        # Display table structure
        logger.debug("Database table structure:")
        tables = atlas.connection.execute("SHOW TABLES")
        if tables:
            logger.debug(f"Tables in the database: {tables}")

        logger.info("AnnData data has been successfully loaded into the database")

    except Exception as e:
        logger.error(f"Failed to load data: {str(e)}")
        logger.exception("Details of the data loading exception:")
        raise


# Ordered reading, small-file reading, and import support for multiple data formats
def load_multi_format(file_path: PathLike[str] | str, atlas: Atlas):

    """Import data into Atlas according to the file format.

    This function is the import entry point for small or general-format data. It first uses the suffix of ``file_path``
    to call ``_read_smart`` to read the data into an in-memory AnnData object, and then calls ``load_anndata`` to write it into
    the Atlas database.

    Unlike the backed chunked import of ``load_h5ad``, this function first loads the full data into memory,
    so it is more suitable for small files, temporary conversion, or non-h5ad format data.

    Parameters
    ----------
    file_path
        Input file path. The function selects an appropriate reading method according to the file format.
    atlas
        Atlas object. The object must already be connected or be connectable to a DuckDB database.

    Returns
    -------
    None
        The result is written directly into the Atlas database and no object is returned.

    Notes
    -----
    Supported reading formats are determined by ``_read_smart``, including h5ad, loom, Matrix Market,
    csv, txt/tsv, Excel, 10x h5, UMI-tools, and other common formats.

    Examples
    --------
    Automatically detect and import a file::

        atlas = sap.Atlas(r"F:\\data\\pbmc")
        atlas.load_multi_format(r"F:\\data\\pbmc.h5ad")
    """

    start_time = datetime.now()
    adata = _read_smart(file_path)
    load_anndata(adata, atlas)
    logger.info(f"load_multi_format Done, elapsed time: {(datetime.now() - start_time).total_seconds():.2f} seconds")


# Check whether gene names in the Atlas database are duplicated
def rename_duplicated_genes(atlas: Atlas, gene_name_column: str = "atlas_gene_name"):
    """Check whether gene names in the Atlas database are duplicated.

    This function reads the gene name column in the ``var`` table, checks whether duplicate gene names exist, and for duplicated entries
    appends suffixes such as ``_1`` and ``_2``. Duplicate gene names may affect plotting by name, differential gene display,
    and AnnData export, so this function can be run after import for cleanup.

    For each duplicated gene name, the first occurrence remains unchanged, and subsequent duplicated entries are renamed
    in the form ``original_name_1`` and ``original_name_2``.

    Parameters
    ----------
    atlas
        Atlas object. The object must already be connected to a DuckDB database, and the database must contain the ``var`` table.
    gene_name_column
        The ``var`` column name that stores gene names. Usually ``"atlas_gene_name"``.

    Returns
    -------
    bool
        Returns ``True`` when checking or updating succeeds; returns ``False`` if the ``var`` table does not exist.

    Notes
    -----
    The ``var`` table is updated only when duplicated gene names are detected. If no duplicates are found, the original table remains unchanged.

    Examples
    --------
    Check the default gene name column::

        atlas.rename_duplicated_genes()
    """
    logger.info(f" Start cleaning gene names in the database var table ")

    # Check whether the table exists
    tables = atlas.connection.execute("SHOW TABLES").df()
    if 'var' not in tables['name'].values:
        logger.error("The var table does not exist. Please import data first")
        return False

    logger.info("Starting suffix-adding mode...")

    # 1. Build the temporary var table with suffixes (var_with_suffix)
    atlas.connection.execute(f"""
        CREATE OR REPLACE TEMPORARY TABLE var_with_suffix AS
        WITH ranked_genes AS (
            SELECT *,
                   ROW_NUMBER() OVER (
                       PARTITION BY {gene_name_column}
                       ORDER BY atlas_gene_id
                   ) AS rn
            FROM var
        )
        SELECT
            atlas_gene_id,
            CASE
                WHEN rn = 1 THEN {gene_name_column} -- Keep the first occurrence of the gene name unchanged
                ELSE {gene_name_column} || '_' || (rn - 1)::VARCHAR -- Add suffixes to subsequent duplicated genes: gene_1, gene_2, ...
            END AS {gene_name_column}
        FROM ranked_genes
        ORDER BY atlas_gene_id
    """)

    # Create a temporary gene-name table with suffixes: var_with_suffix
    # atlas_gene_id | atlas_gene_name
    # ---|---------
    # 1  | TP53     -- The first TP53 keeps the original name
    # 2  | EGFR     -- There is only one EGFR, so keep the original name
    # 3  | TP53_1   -- The second TP53 is appended with the _1 suffix
    # 4  | BRAF     -- There is only one BRAF, so keep the original name
    # 5  | TP53_2   -- The third TP53 is appended with the _2 suffix

    # 2. Record the mapping relationships where atlas_gene_name actually changes (for logging only)
    gene_mapping = atlas.connection.execute(f"""
        SELECT
            v.{gene_name_column}  AS original_gene_name,
            vs.{gene_name_column} AS new_gene_name
        FROM var v
        JOIN var_with_suffix vs
            ON v.atlas_gene_id = vs.atlas_gene_id
        WHERE v.{gene_name_column} != vs.{gene_name_column}
    """).df()
    # Get the gene name mapping relationship. The gene_mapping DataFrame
    #    original_gene_name  new_gene_name
    # 0              TP53       TP53_1
    # 1              TP53       TP53_2

    # 3.  Output different logs depending on whether duplicated genes exist
    if len(gene_mapping) > 0:
        logger.info(
            f"Found {len(gene_mapping)} duplicated genes; suffixes have been added successfully"
        )
    else:
        logger.info("No duplicated genes found; the var table remains unchanged")

    # 4. Update the var table
    if(len(gene_mapping)>0):
        atlas.connection.execute("DROP TABLE var")
        atlas.connection.execute("ALTER TABLE var_with_suffix RENAME TO var")
        atlas.connection.execute("ALTER TABLE var ADD PRIMARY KEY (atlas_gene_id)")
        logger.info("The var table has been updated")

    logger.info("rename_duplicated_genes Done")
    return True


def _load_h5ad_list_random(
    h5ad_paths: PathLike[str] | str | list[PathLike[str] | str],
    atlas: Atlas,
    cells_per_block: int | None = None,
    *,
    commit_every: int = 1,
    shuffle_blocks: bool = True,
    shuffle_cells: bool = True,
):
    """Randomly import multiple h5ad files into an Atlas database.

    This internal function is used for multi-file ``load_type="random"`` scenarios. It first splits each h5ad file
    into contiguous blocks according to ``cells_per_block``, then merges blocks from all files into a global
    block pool and shuffles them randomly.

    After reading ``blocks_per_pool`` blocks each time, they are merged into a cell pool, and cells inside the pool
    are shuffled once as a whole, then written into ``obs``, ``var``, ``X_HyS_indptr``, and
    ``X_HyS_data``. The expression matrix is uniformly converted and stored on the count scale.

    This strategy is suitable when multiple files have different sizes. It avoids having only large files left in the later stage of round-robin import and reduces h5ad cell-level random I/O.

    Parameters
    ----------
    h5ad_paths
        One or more h5ad file paths.

    atlas
        Atlas object. The function connects to and writes into the corresponding DuckDB database.

    cells_per_block
        Number of cells to read, write, or process in each batch; larger values are usually faster but consume more memory.
    commit_every
        Commit the active DuckDB transaction once every N cell-pool flushes.
    shuffle_blocks
        Whether to shuffle the global block order. Enabled for multi-file random import and disabled for multi-file ordered import.
    shuffle_cells
        Whether to shuffle the cell order inside each cell pool. Enabled for multi-file random import and disabled for multi-file ordered import.

    Returns
    -------
    None
        The result is written directly into the Atlas database and no object is returned.

    Notes
    -----
    This function dynamically estimates ``blocks_per_pool`` according to Atlas ``db_memory_limit`` and the sample-estimated per-cell memory usage,
    to control the peak memory usage of the shuffle window.

    Examples
    --------
    Internal random import of multiple files::

        _load_h5ad_list_random([path1, path2], atlas, cells_per_block=1000)
    """


    # Support single path / multiple paths
    if isinstance(h5ad_paths, (str, PathLike)):
        h5ad_paths = [h5ad_paths]

    h5ad_paths = [os.fspath(path) for path in h5ad_paths]

    if len(h5ad_paths) == 0:
        raise ValueError("h5ad_paths cannot be empty")

    # ===== Count global n_cells =====
    total_n_cells = 0
    file_cell_counts = []

    for path in h5ad_paths:
        ad = sc.read_h5ad(path, backed="r")
        n = ad.n_obs
        total_n_cells += n
        file_cell_counts.append(n)
        ad.file.close()

    # ===== Compute global cells_per_block  =====
    cells_per_block = _normalize_cells_per_block(cells_per_block, total_n_cells)

    file_num = len(h5ad_paths)

    rng = np.random.default_rng()

    # Connect to the database
    conn = atlas.connect("r+")
    atlas.connection = conn

    # Global cursor
    global_cell_id = 0
    global_indptr_id = 0
    global_indptr_offset = 0
    global_data_id = 0

    var_written = False

    logger.info(f"[INFO] number of files: {file_num:,}")
    logger.info("[INFO] The expression matrix is uniformly written as count")

    file_states = []

    ref_var_names = None
    ref_n_genes = None

    # Global block index pool; each element records which file a block comes from and its start/end positions
    all_block_refs = []
    max_estimated_bytes_per_cell = 0.0

    try:
        for file_idx, h5ad_path in enumerate(h5ad_paths):

            adata_backed = sc.read_h5ad(h5ad_path, backed="r")

            n_cells = adata_backed.n_obs
            n_genes = adata_backed.n_vars

            logger.info(f"[INFO] current file dimensions: {n_cells:,} × {n_genes:,}")

            # Detect X scale and estimate memory usage separately for each file
            x_info = _inspect_x_from_backed(
                adata_backed,
                sample_n=5000,
            )
            source_x_scale = x_info["x_scale"]
            max_estimated_bytes_per_cell = max(
                max_estimated_bytes_per_cell,
                float(x_info["estimated_bytes_per_cell"]),
            )

            logger.info(f"[INFO] current file X detected as: {source_x_scale}")

            if source_x_scale == "count":
                logger.info("[INFO] Current file X is already count; writing directly.")
            else:
                logger.info("[INFO] Current file X will be converted to count after reading each block")

            # ---------------- Check gene number and order ----------------
            cur_var_names = adata_backed.var.index.astype(str).to_numpy()

            if file_idx == 0:
                ref_var_names = cur_var_names
                ref_n_genes = n_genes
            else:
                if n_genes != ref_n_genes:
                    raise ValueError(
                        f"File {file_idx + 1} has inconsistent gene count:"
                        f"{n_genes} != {ref_n_genes}"
                    )

                if not np.array_equal(cur_var_names, ref_var_names):
                    raise ValueError(
                        f"File {file_idx + 1} has gene order inconsistent with the first file,"
                        f"cannot be directly merged for import."
                    )

            # ---------------- Split each file into blocks according to cells_per_block ----------------
            block_starts = np.arange(0, n_cells, cells_per_block, dtype=np.int64)

            # Put blocks from all files into all_block_refs; random mode will shuffle them globally later.
            for block_start in block_starts:
                block_start = int(block_start)
                block_end = min(block_start + cells_per_block, n_cells)

                all_block_refs.append(
                    {
                        "file_idx": file_idx,
                        "block_start": block_start,
                        "block_end": block_end,
                    }
                )

            file_states.append(
                {
                    "file_idx": file_idx,
                    "h5ad_path": h5ad_path,
                    "adata_backed": adata_backed,
                    "n_cells": n_cells,
                    "n_genes": n_genes,
                    "source_x_scale": source_x_scale,
                }
            )

        # Global block index pool: shuffled in random mode; file list order and within-file order are preserved in ordered mode.
        total_blocks = len(all_block_refs)

        if total_blocks == 0:
            raise ValueError("All h5ad files have 0 cells and cannot be imported")

        estimated_window_cells, blocks_per_pool = _estimate_window_cells_and_blocks_per_pool(
            memory_limit=_get_atlas_memory_limit(atlas),
            cells_per_block=cells_per_block,
            estimated_bytes_per_cell=max_estimated_bytes_per_cell,
        )

        if shuffle_blocks:
            rng.shuffle(all_block_refs)

        # Dynamically create tables: use only the first file to create tables
        first_backed = file_states[0]["adata_backed"]

        _create_obs_table_from_adata(conn, first_backed[:1])
        _create_var_table_from_adata(conn, first_backed[:1])
        _create_hys_tables(conn)

        # Batches read from multiple blocks are merged and then written to the database; in random mode, the cell order is shuffled as a whole first.
        def _flush_cell_pool(cell_pool: list[AnnData], flush_i: int):

            """Merge and write the current cell pool for multi-file import.

            This nested helper is used by the shared body of ``_load_h5ad_list_random`` and
            ``_load_h5ad_list_order``. The function merges multiple
            AnnData blocks in the current pool, including ``X``, ``obs``, and ``var``; in random mode, it
            shuffles the cell order inside the pool again; then writes the result into ``obs``, ``var``,
            ``X_HyS_indptr`` and ``X_HyS_data``.

            Parameters
            ----------
            cell_pool
                List of AnnData blocks collected in the current flush.
            flush_i
                Current flush index, used for log output.

            Returns
            -------
            tuple[int, int]
                Number of cells and nonzero expression records written by the current pool.

            Notes
            -----
            The ``var`` table is written only during the first flush; subsequent flushes only append cells and the expression matrix.
            """
            nonlocal global_cell_id
            nonlocal global_indptr_id
            nonlocal global_indptr_offset
            nonlocal global_data_id
            nonlocal var_written

            if len(cell_pool) == 0:
                return 0, 0

            t0 = time.time()

            # 1. Merge multiple AnnData batches in cell_pool
            X_list = []
            obs_list = []

            total_pool_cells = 0
            total_pool_nnz = 0

            for ad in cell_pool:
                X = ad.X

                if sparse.issparse(X):
                    X = X.tocsr()
                else:
                    X = sparse.csr_matrix(X)

                X_list.append(X)
                obs_list.append(ad.obs.copy())

                total_pool_cells += ad.n_obs
                total_pool_nnz += X.nnz

            X_pool = sparse.vstack(X_list, format="csr")
            obs_pool = pd.concat(obs_list, axis=0)

            # The var information is consistent across all files, so use the var from the first batch directly
            var_pool = cell_pool[0].var.copy()

            pool_adata = AnnData(
                X=X_pool,
                obs=obs_pool,
                var=var_pool,
            )


            # 2. Shuffle the entire cell_pool internally; ordered mode disables this step to preserve the original cell order
            if shuffle_cells and pool_adata.n_obs > 1:
                pool_perm = rng.permutation(pool_adata.n_obs)
                pool_adata = pool_adata[pool_perm].copy()

            t_shuffle = time.time() - t0
            t1 = time.time()

            # 3. Write obs
            global_cell_id = _append_obs_rows(
                pool_adata,
                conn,
                start_cell_id=global_cell_id,
            )

            # 4. Write var, only once
            if not var_written:
                _append_var(pool_adata, conn)
                var_written = True

            # 5. Write X_CSRO using the Arrow-accelerated version
            (
                global_indptr_id,
                global_indptr_offset,
                global_data_id,
            ) = _append_x_hys(
                pool_adata,
                conn,
                base_cell_id=global_cell_id - pool_adata.n_obs,
                global_indptr_id=global_indptr_id,
                global_indptr_offset=global_indptr_offset,
                global_data_id=global_data_id,
            )

            t_write = time.time() - t1

            # 6. Clean up
            for obj_name in [
                "X_list",
                "obs_list",
                "X_pool",
                "obs_pool",
                "var_pool",
                "pool_adata",
                "pool_perm",
            ]:
                try:
                    del locals()[obj_name]
                except Exception:
                    pass

            gc.collect()

            return total_pool_cells, total_pool_nnz

        # Main loop: read blocks_per_pool blocks each time; in random mode, the block list has already been shuffled
        processed_blocks = 0
        flush_counter = 0

        pbar = progress(total=total_blocks, desc="load_h5ad")

        # Transaction: multiple flushes share one transaction
        conn.execute("BEGIN TRANSACTION")

        try:
            block_cursor = 0

            while block_cursor < total_blocks:

                # Take blocks_per_pool blocks from the global block list each time
                block_group = all_block_refs[block_cursor:block_cursor + blocks_per_pool]
                block_cursor += len(block_group)

                # cell_pool for the current flush
                cell_pool = []

                # Read this group of blocks
                for block_ref in block_group:

                    state = file_states[block_ref["file_idx"]]

                    block_start = block_ref["block_start"]
                    block_end = block_ref["block_end"]

                    # Continuously read one batch block from h5ad
                    t_read0 = time.time()

                    adata = state["adata_backed"][block_start:block_end].to_memory()

                    t_read = time.time() - t_read0

                    # During import, uniformly convert the expression matrix to count before writing
                    adata = _convert_x_to_count_inplace(
                        adata,
                        source_x_scale=state["source_x_scale"],
                    )

                    cell_pool.append(adata)

                    processed_blocks += 1
                    pbar.update(1)

                    # Do not print every block to avoid flooding the log
                    if processed_blocks == 1 or processed_blocks % 50 == 0:
                        if sparse.issparse(adata.X):
                            block_nnz = adata.X.nnz
                        else:
                            block_nnz = np.count_nonzero(adata.X)

                # After this group of blocks has been read, write it according to the current mode
                if len(cell_pool) > 0:

                    flush_counter += 1
                    _flush_cell_pool(cell_pool, flush_counter)

                    # Clear the pool
                    try:
                        for ad in cell_pool:
                            del ad
                        cell_pool.clear()
                    except Exception:
                        pass

                    gc.collect()

                    # Commit once every commit_every flushes
                    if flush_counter % commit_every == 0:
                        conn.execute("COMMIT")
                        conn.execute("BEGIN TRANSACTION")

            pbar.close()

            # Final commit
            conn.execute("COMMIT")

        except Exception:
            pbar.close()
            conn.execute("ROLLBACK")
            raise

        # Primary keys
        conn.execute("ALTER TABLE obs ADD PRIMARY KEY (atlas_cell_id)")
        conn.execute("ALTER TABLE var ADD PRIMARY KEY (atlas_gene_id)")

        # varm is gene-dimensional. When var is consistent across multiple files, only the varm from the first file needs to be imported.
        _add_varm_from_h5ad(h5ad_paths[0], atlas)

        # Clean up the DuckDB file state after import is complete
        try:
            conn.execute("CHECKPOINT")
        except Exception:
            pass
        # Release large objects such as the global block index pool
        try:
            del all_block_refs
        except Exception:
            pass
        try:
            del first_backed
        except Exception:
            pass
        gc.collect()

        return {
            "files": file_num,
            "cells": global_cell_id,
            "genes": ref_n_genes,
            "nnz": global_data_id,
            "blocks": total_blocks,
            "flush": flush_counter,
        }


    finally:
        # Close all backed files regardless of success or failure
        for s in file_states:
            try:
                s["adata_backed"].file.close()
            except Exception:
                pass
        # ：On exception or successful exit, try to clear references to large objects
        try:
            file_states.clear()
        except Exception:
            pass
        try:
            all_block_refs.clear()
        except Exception:
            pass
        try:
            for ad in cell_pool:
                del ad
            cell_pool.clear()
        except Exception:
            pass
        try:
            gc.collect()
        except Exception:
            pass


def _load_h5ad_list_order(
    h5ad_paths: PathLike[str] | str | list[PathLike[str] | str],
    atlas: Atlas,
    cells_per_block: int | None = None,
    commit_every: int = 1,
):
    """Import multiple h5ad files into an Atlas database in file-list order.

    This function reuses the multi-file import body, but disables global block shuffling and cell-pool internal shuffling,
    so the write order is consistent with the order of ``h5ad_paths`` and the cell order within each file.
    The expression matrix is uniformly written on the count scale.

    Parameters
    ----------
    h5ad_paths
        One or more h5ad file paths.
    atlas
        Atlas object. The function connects to and writes into the corresponding DuckDB database.
    cells_per_block
        Number of cells in each contiguous cell block. If ``None``, it is automatically estimated based on the total number of cells.
    commit_every
        Commit the active DuckDB transaction once every N cell-pool flushes.

    Returns
    -------
    None
        The result is written directly into the Atlas database and no object is returned.

    Notes
    -----
    This function minimizes duplicated logic by calling ``_load_h5ad_list_random`` and setting
    ``shuffle_blocks=False`` and ``shuffle_cells=False``.
    """
    return _load_h5ad_list_random(
        h5ad_paths=h5ad_paths,
        atlas=atlas,
        cells_per_block=cells_per_block,
        commit_every=commit_every,
        shuffle_blocks=False,
        shuffle_cells=False,
    )


def _load_h5ad_random(
    h5ad_path: PathLike[str] | str,
    atlas: Atlas,
    cells_per_block: int | None = None,
    commit_every: int = 1,
):
    """Randomly import a single h5ad file using a shuffle-window strategy.

    This internal function is used for single-file ``load_type="random"`` scenarios. It opens the
    h5ad file in backed mode, splits the file into contiguous blocks, randomly shuffles block order, and merges multiple blocks
    into one shuffle window.

    Cells within each window are shuffled as a whole and then written into the Atlas database in batches,
    giving a randomized import order while controlling memory usage.
    The expression matrix is uniformly written on the count scale.

    Because the cell order is reordered, this function does not import ``obsm`` by default;
    ``varm`` is aligned to genes and can be imported normally.

    Parameters
    ----------
    h5ad_path
        Path to the h5ad file.

    atlas
        Atlas object. The function connects to and writes into the corresponding DuckDB database.

    cells_per_block
        Number of cells to read, write, or process in each batch; larger values are usually faster but consume more memory.
    commit_every
        Commit the active DuckDB transaction once every N shuffle windows.

    Returns
    -------
    None
        The result is written directly into the Atlas database and no object is returned.

    Notes
    -----
    Random import reorders cells, so this function does not import ``obsm`` by default; ``varm`` is aligned to the gene
    dimension and is not affected by cell reordering, so it is imported normally.

    Examples
    --------
    Internal random import of a single file::

        _load_h5ad_random(path, atlas, cells_per_block=1000)
    """

    h5ad_path = os.fspath(h5ad_path)

    t_start= time.time()

    h5ad_path = os.fspath(h5ad_path)

    # Connect to the database
    conn = atlas.connect("r+")
    atlas.connection = conn

    # Global cursor
    global_cell_id = 0
    global_indptr_id = 0
    global_indptr_offset = 0
    global_data_id = 0

    var_written = False

    # Open in backed mode
    adata_backed = sc.read_h5ad(h5ad_path, backed="r")
    n_cells = adata_backed.n_obs

    # Pre-read sample_n cells while detecting X scale and estimating import memory
    x_info = _inspect_x_from_backed(
        adata_backed,
        sample_n=5000,
    )

    source_x_scale = x_info["x_scale"]

    estimated_window_cells, blocks_per_pool = _estimate_window_cells_and_blocks_per_pool(
        memory_limit=_get_atlas_memory_limit(atlas),
        cells_per_block=cells_per_block,
        estimated_bytes_per_cell=x_info["estimated_bytes_per_cell"],
    )

    logger.info(f"[INFO] X in the file detected as: {source_x_scale}")
    logger.info("[INFO] The expression matrix is uniformly written as count")

    if source_x_scale == "count":
        logger.info("[INFO] X data is already count; writing directly.")
    else:
        logger.info("[INFO] X data will be converted to count before writing")

    # Merge 5 blocks and then shuffle them uniformly
    block_starts = np.arange(0, n_cells, cells_per_block, dtype=np.int64)
    np.random.shuffle(block_starts)

    # Create tables dynamically
    _create_obs_table_from_adata(conn, adata_backed[:1])
    _create_var_table_from_adata(conn, adata_backed[:1])
    _create_hys_tables(conn)

    logger.info(f"[INFO] dataset dimensions: {adata_backed.n_obs:,} × {adata_backed.n_vars:,}")

    # Window cache: collect blocks_per_pool batches each time before unified random writing
    window_adatas = []
    window_batch_count = 0
    total_batch_counter = 0
    window_counter = 0

    conn.execute("BEGIN TRANSACTION")

    try:
        for block_i, block_start in enumerate(
            progress(
                block_starts,
                desc="load_h5ad",
            )
        ):
            block_end = min(int(block_start) + cells_per_block, n_cells)

            # Continuously read one batch block
            t0 = time.time()
            adata = adata_backed[int(block_start):block_end].to_memory()
            t_read = time.time() - t0

            # nnz of the current block
            if sparse.issparse(adata.X):
                block_nnz = adata.X.nnz
            else:
                block_nnz = np.count_nonzero(adata.X)

            # Do not shuffle and write immediately inside a single batch; put it into the window first
            window_adatas.append(adata)
            window_batch_count += 1

            if (block_i + 1) % 20 == 0 or block_i == 0:
                logger.info(
                    f"\n[read block {block_i}] "
                    f"cells={adata.n_obs:,}, "
                    f"nnz={block_nnz:,}, "
                    f"read={t_read:.2f}s, "
                    f"window_batches={window_batch_count}/{blocks_per_pool}"
                )

            # When the window is full, shuffle and write it uniformly
            if window_batch_count >= blocks_per_pool:
                t1 = time.time()

                (
                    global_cell_id,
                    global_indptr_id,
                    global_indptr_offset,
                    global_data_id,
                    var_written,
                    window_cells,
                    window_nnz,
                ) = _write_shuffle_window_to_duckdb(
                    window_adatas=window_adatas,
                    conn=conn,
                    global_cell_id=global_cell_id,
                    global_indptr_id=global_indptr_id,
                    global_indptr_offset=global_indptr_offset,
                    global_data_id=global_data_id,
                    var_written=var_written,
                    source_x_scale=source_x_scale,
                )

                t_write = time.time() - t1

                total_batch_counter += window_batch_count
                window_counter += 1

                # Clear the window
                for x in window_adatas:
                    del x
                window_adatas.clear()
                window_batch_count = 0

                # Commit every commit_every batches; equivalent to committing every 2 windows when blocks_per_pool=5
                if window_counter % commit_every == 0:
                    conn.execute("COMMIT")
                    conn.execute("BEGIN TRANSACTION")
                    logger.info(
                        f"[COMMIT] processed_windows={window_counter:,}, "
                        f"processed_batches={total_batch_counter:,}"
                    )

        # Process the remaining window with fewer than 5 batches
        if window_batch_count > 0:
            t1 = time.time()

            (
                global_cell_id,
                global_indptr_id,
                global_indptr_offset,
                global_data_id,
                var_written,
                window_cells,
                window_nnz,
            ) = _write_shuffle_window_to_duckdb(
                window_adatas=window_adatas,
                conn=conn,
                global_cell_id=global_cell_id,
                global_indptr_id=global_indptr_id,
                global_indptr_offset=global_indptr_offset,
                global_data_id=global_data_id,
                var_written=var_written,
                source_x_scale=source_x_scale,
            )

            t_write = time.time() - t1

            total_batch_counter += window_batch_count
            window_counter += 1

            for x in window_adatas:
                del x
            window_adatas.clear()
            window_batch_count = 0
            gc.collect()

        # Final commit
        conn.execute("COMMIT")

    except Exception:
        conn.execute("ROLLBACK")
        # Also clean up window_adatas on exception
        try:
            for x in window_adatas:
                del x
            window_adatas.clear()
        except Exception:
            pass
        # Also close the h5ad backed file on exception
        try:
            adata_backed.file.close()
        except Exception:
            pass
        gc.collect()
        raise

    # Primary keys
    conn.execute("ALTER TABLE obs ADD PRIMARY KEY (atlas_cell_id)")
    conn.execute("ALTER TABLE var ADD PRIMARY KEY (atlas_gene_id)")

    # It is not recommended to import obsm during random import
    # Because obs / X have already been randomly reordered, directly importing obsm in the original h5ad order will cause misalignment; varm is gene-dimensional and is usually fine
    # _add_obsm_from_h5ad(h5ad_path, atlas)
    _add_varm_from_h5ad(h5ad_path, atlas)

    # Clean up the DuckDB file state after import is complete
    try:
        conn.execute("CHECKPOINT")
    except Exception:
        pass

    try:
        adata_backed.file.close()
    except Exception:
        pass

    # Actively release Python objects after import ends
    try:
        del window_adatas
    except Exception:
        pass

    try:
        del adata_backed
    except Exception:
        pass

    gc.collect()

    t_end = time.time()


def _load_h5ad_order(
    h5ad_path: PathLike[str] | str,
    atlas: Atlas,
    cells_per_block: int | None = None,
    commit_every: int = 1,
):

    """Import a single h5ad file in the original cell order.

    This internal function is used for single-file ``load_type="order"`` scenarios. It sequentially reads the
    h5ad file in backed mode, loads data into memory by mega-batches, and then splits them into smaller batches to write into Atlas.
    The expression matrix is uniformly converted and stored on the count scale.

    Unlike random import, it does not shuffle cell order, so ``obsm`` and ``varm`` can be safely imported.
    It is suitable for scenarios that need to preserve the original AnnData row order.

    Parameters
    ----------
    h5ad_path
        Path to the h5ad file.

    atlas
        Atlas object. The function connects to and writes into the corresponding DuckDB database.

    cells_per_block
        Number of cells to read, write, or process in each batch; larger values are usually faster but consume more memory.
    commit_every
        Commit the active DuckDB transaction once every N mini-batches.

    Returns
    -------
    None
        The result is written directly into the Atlas database and no object is returned.

    Notes
    -----
    ``cells_per_block`` participates in memory window estimation. In ordered import, the estimated
    ``window_cells`` is used as ``mega_batch_size``.

    Examples
    --------
    Internal ordered import of a single file::

        _load_h5ad_order(path, atlas, cells_per_block=1000)
    """
    conn = atlas.connect("r+")
    atlas.connection = conn

    global_cell_id = 0
    global_indptr_id = 0
    global_indptr_offset = 0
    global_data_id = 0

    var_written = False

    adata_backed = sc.read_h5ad(h5ad_path, backed="r")
    n_cells = adata_backed.n_obs

    # Simply detect the underlying format of h5ad.X
    x_format = _print_h5ad_x_format(h5ad_path)

    # Pre-read 1000 cells to determine whether X in the file is count or log
    x_info = _inspect_x_from_backed(
        adata_backed,
        sample_n=5000,
    )

    source_x_scale = x_info["x_scale"]

    estimated_window_cells, blocks_per_pool = _estimate_window_cells_and_blocks_per_pool(
        memory_limit=_get_atlas_memory_limit(atlas),
        cells_per_block=cells_per_block,
        estimated_bytes_per_cell=x_info["estimated_bytes_per_cell"],
    )

    # In order mode, window_cells is the mega-batch size
    mega_batch_size = estimated_window_cells

    logger.info(f"[INFO] X in the file detected as: {source_x_scale}")
    logger.info("[INFO] The expression matrix is uniformly written as count")

    if source_x_scale == "count":
        logger.info("[INFO] X data is already count; writing directly.")
    else:
        logger.info("[INFO] X data will be converted to count after the mega-batch is read")

    _create_obs_table_from_adata(conn, adata_backed[:1])
    _create_var_table_from_adata(conn, adata_backed[:1])
    _create_hys_tables(conn)

    logger.info(f"[INFO] dataset dimensions: {adata_backed.n_obs:,} × {adata_backed.n_vars:,}")

    mini_batch_counter = 0

    # Place the transaction outside the main loop
    conn.execute("BEGIN TRANSACTION")

    try:
        for mega_i, mega_start in enumerate(
            progress(
                range(0, n_cells, mega_batch_size),
                desc="load_h5ad",
            )
        ):
            mega_end = min(mega_start + mega_batch_size, n_cells)

            t0 = time.time()

            # Actually trigger disk reading
            mega = adata_backed[mega_start:mega_end].to_memory()

            t_read = time.time() - t0

            # After the mega-batch is read, convert it uniformly to count
            mega = _convert_x_to_count_inplace(
                mega,
                source_x_scale=source_x_scale,
            )

            # Count nnz in the current mega-batch
            if sparse.issparse(mega.X):
                mega_nnz = mega.X.nnz
            else:
                mega_nnz = np.count_nonzero(mega.X)

            # Import in batches according to cells_per_block
            for start in range(0, mega.n_obs, cells_per_block):
                end = min(start + cells_per_block, mega.n_obs)
                adata = mega[start:end]

                t1 = time.time()

                # ---------------- batch import obs ----------------
                global_cell_id = _append_obs_rows(
                    adata,
                    conn,
                    start_cell_id=global_cell_id,
                )

                # ---------------- import var (once) ----------------
                if not var_written:
                    _append_var(adata, conn)
                    var_written = True

                # ---------------- batch import X (CSRO) ----------------
                (
                    global_indptr_id,
                    global_indptr_offset,
                    global_data_id,
                ) = _append_x_hys(
                    adata,
                    conn,
                    base_cell_id=global_cell_id - adata.n_obs,
                    global_indptr_id=global_indptr_id,
                    global_indptr_offset=global_indptr_offset,
                    global_data_id=global_data_id,
                )

                t_write = time.time() - t1

                mini_batch_counter += 1

                # Commit once every commit_every mini-batches
                if mini_batch_counter % commit_every == 0:
                    conn.execute("COMMIT")
                    conn.execute("BEGIN TRANSACTION")

                del adata

            del mega

        # Final commit
        conn.execute("COMMIT")

    except Exception:
        conn.execute("ROLLBACK")
        raise

    # Primary keys: must be added after all data has been written
    conn.execute("ALTER TABLE obs ADD PRIMARY KEY (atlas_cell_id)")
    conn.execute("ALTER TABLE var ADD PRIMARY KEY (atlas_gene_id)")

    # Ordered import does not shuffle cells, so obsm can be imported normally
    _add_obsm_from_h5ad(
        h5ad_path,
        atlas,
        cells_per_block=cells_per_block,
    )
    _add_varm_from_h5ad(h5ad_path, atlas)

    try:
        adata_backed.file.close()
    except Exception:
        pass


# Merge one shuffle window and write it into DuckDB
def _write_shuffle_window_to_duckdb(
    window_adatas: list[AnnData],
    conn: DuckDBPyConnection,
    global_cell_id: int,
    global_indptr_id: int,
    global_indptr_offset: int,
    global_data_id: int,
    var_written: bool,
    source_x_scale: XScale,
):
    """Write one shuffle window into the Atlas database.

    This internal function is used in the random import workflow. It receives multiple AnnData
    blocks collected in one window, first merges them into one AnnData object,
    then randomly shuffles cell order within the window, and then sequentially
    writes ``obs``, ``var``, ``X_HyS_indptr``, and ``X_HyS_data``.

    Before writing the expression matrix, the function uniformly converts ``adata.X`` according to ``source_x_scale`` to
    the count scale, and finally writes it to the ``X_HyS_data.data_count`` field.

    Parameters
    ----------
    window_adatas
        List of AnnData blocks collected in one shuffle window.
    conn
        DuckDB database connection.
    global_cell_id
        Next global ``atlas_cell_id`` to write.
    global_indptr_id
        Next indptr row ID to write.
    global_indptr_offset
        Current cumulative number of nonzero values already written, used to relocate indptr.
    global_data_id
        Next ``X_HyS_data.id`` to write.
    var_written
        Whether the ``var`` table has already been written.
    source_x_scale
        Current scale of the input expression matrix, usually ``"count"`` or ``"log"``.

    Returns
    -------
    tuple
        Returns the updated global cursors and window statistics, in the following order:

        ``global_cell_id``、``global_indptr_id``、``global_indptr_offset``、
        ``global_data_id``、``var_written``、``window_cells``、``window_nnz``。

    Notes
    -----
    The ``var`` table is written only in the first window; subsequent windows only append ``obs`` and expression
    matrix-related tables.
    """

    # 1. Merge multiple batches within the window
    adata_window = sc.concat(
        window_adatas,
        axis=0,
        join="outer",
        merge="first",
        index_unique=None,
    )

    # 2. Shuffle all cells within the window uniformly
    if adata_window.n_obs > 1:
        perm = np.random.permutation(adata_window.n_obs)
        adata_window = adata_window[perm].copy()

    # Window statistics
    window_cells = adata_window.n_obs

    if sparse.issparse(adata_window.X):
        window_nnz = adata_window.X.nnz
    else:
        window_nnz = np.count_nonzero(adata_window.X)

    # 3. Write obs
    global_cell_id = _append_obs_rows(
        adata_window,
        conn,
        start_cell_id=global_cell_id,
    )

    # 4. Write var, only once
    if not var_written:
        _append_var(adata_window, conn)
        var_written = True

    # During import, uniformly convert the expression matrix to count before writing
    adata_window = _convert_x_to_count_inplace(
        adata_window,
        source_x_scale=source_x_scale,
    )

    # 5. Write X_HyS_data / X_HyS_indptr
    (
        global_indptr_id,
        global_indptr_offset,
        global_data_id,
    ) = _append_x_hys(
        adata_window,
        conn,
        base_cell_id=global_cell_id - adata_window.n_obs,
        global_indptr_id=global_indptr_id,
        global_indptr_offset=global_indptr_offset,
        global_data_id=global_data_id,
    )

    del adata_window

    gc.collect()

    return (
        global_cell_id,
        global_indptr_id,
        global_indptr_offset,
        global_data_id,
        var_written,
        window_cells,
        window_nnz,
    )


def _get_atlas_memory_limit(atlas: Atlas) -> str | int | None:
    """Get the memory limit parameter stored in the Atlas object.

    This function reads the memory limit value from the Atlas object, such as ``"32GB"``, ``"8G"``,
    ``16``, or ``None``.

    This is written as a separate helper so that the import module can remain compatible with different versions of the Atlas class:
    1. older versions may use ``__memory_limit``;
    2. newer versions may use ``__db_memory_limit``;
    3. some versions may provide the public attribute ``memory_limit``;
    4. the currently recommended version uses the public attribute ``db_memory_limit``.

    Parameters
    ----------
    atlas
        Atlas object. It is usually passed in by ``load_h5ad(..., atlas=atlas)`` or
        ``atlas.load_h5ad(...)``.

    Returns
    -------
    memory_limit
        Memory limit parameter stored in Atlas.

        Possible return values:
        - ``str``, such as ``"32GB"``, ``"8G"``, or ``"1024MB"``;
        - ``int``, such as ``32``, meaning 32GB;
        - ``None``, meaning no available memory limit is set.

    Notes
    -----
    This function is only responsible for reading the memory limit and does not apply the memory limit to DuckDB.
    The logic that actually executes DuckDB ``SET memory_limit`` should be implemented in the Atlas class
    inside ``_apply_memory_limit()``.

    The value returned here is mainly used later to estimate the Python import window size, for example:

    ``memory_limit -> memory_limit_bytes -> window_cells -> blocks_per_pool``。
    """

    # 1. First try to read private attributes.
    #    This makes it compatible with fields that may exist in older Atlas classes.
    for attr_name in ("_Atlas__memory_limit", "_Atlas__db_memory_limit"):
        try:
            value = getattr(atlas, attr_name)
        except Exception:
            value = None

        if value not in (None, "", "None"):
            return value

    # 2. Then try to read public attributes.
    #    Currently, atlas.db_memory_limit is recommended.
    for attr_name in ("memory_limit", "db_memory_limit"):
        try:
            value = getattr(atlas, attr_name)
        except RecursionError:
            # Prevent recursion errors caused by incorrectly implemented old properties.
            value = None
        except Exception:
            value = None

        if value not in (None, "", "None"):
            return value

    # 3. If none are found, assume no memory limit is set.
    return None


def _parse_memory_limit_to_bytes(memory_limit: str | int | None) -> int | None:
    """Convert the memory limit parameter to bytes.

    This function converts the ``db_memory_limit`` stored in Atlas into bytes,
    so that the import window size can be calculated later.

    Supported input forms include:

    - ``None``: means no memory limit and returns ``None``;
    - ``int``: interpreted as GB, for example ``32`` means ``32GB``;
    - numeric string without a unit: interpreted as GB, for example ``"32"`` means ``32GB``;
    - string with a unit: such as ``"32GB"``, ``"32G"``, ``"1024MB"``, or ``"512M"``.

    Parameters
    ----------
    memory_limit
        Memory limit value stored in Atlas.

        Common forms include:
        - ``"32GB"``
        - ``"16G"``
        - ``"8000MB"``
        - ``32``
        - ``None``

    Returns
    -------
    memory_limit_bytes
        Converted number of bytes.

        If ``memory_limit`` is ``None``, an empty string, or ``"None"``,
        then ``None`` is returned.

    Notes
    -----
    Note: here ``int`` is not interpreted as bytes, but as GB.
    This is to keep it consistent with the design of ``db_memory_limit`` in the Atlas class:

    ``db_memory_limit=32`` is equivalent to ``db_memory_limit="32GB"``.
    """

    if memory_limit is None:
        return None

    # Interpret int as GB, for example 32 -> 32GB.
    if isinstance(memory_limit, int):
        if memory_limit <= 0:
            raise ValueError("memory_limit must be > 0")
        return int(memory_limit * 1024 ** 3)

    if not isinstance(memory_limit, str):
        raise TypeError(
            f"memory_limit must be str, int, or None, current type: {type(memory_limit)}"
        )

    value_text = memory_limit.strip().upper().replace(" ", "")

    if value_text in {"", "NONE"}:
        return None

    units = {
        "KB": 1024,
        "K": 1024,
        "MB": 1024 ** 2,
        "M": 1024 ** 2,
        "GB": 1024 ** 3,
        "G": 1024 ** 3,
    }

    # Handle strings with units, such as "32GB", "16G", or "1024MB".
    for unit, factor in units.items():
        if value_text.endswith(unit):
            value = float(value_text[: -len(unit)])
            if value <= 0:
                raise ValueError("memory_limit must be > 0")
            return int(value * factor)

    # When no unit is provided, also interpret it as GB.
    value = float(value_text)
    if value <= 0:
        raise ValueError("memory_limit must be > 0")
    return int(value * 1024 ** 3)


def _inspect_x_from_backed(
    adata_backed: AnnData,
    sample_n: int = 5000,
    overhead_factor: float = 4.0,
) -> dict[str, Any]:
    """Pre-read a subset of cells while detecting the scale of X and estimating memory usage per cell.

    This function reads the first ``sample_n`` cells from an AnnData object opened in backed mode,
    and then completes two tasks based on this sample:

    1. Determine whether ``adata.X`` is currently on the count scale or log scale;
    2. Estimate roughly how much memory each cell will occupy during import.

    The benefits of doing this are:
    - no need to read the sample twice;
    - X scale detection and import-window estimation can share the same sample;
    - later, ``window_cells`` and ``blocks_per_pool`` can be automatically calculated
      based on ``estimated_bytes_per_cell`` and ``db_memory_limit``.

    Parameters
    ----------
    adata_backed
        AnnData object opened in backed mode.

        Usually from:

        ``adata_backed = sc.read_h5ad(h5ad_path, backed="r")``

    sample_n
        Number of cells to pre-read.

        The actual number of cells read is:

        ``min(sample_n, adata_backed.n_obs)``

        By default, the first 5000 cells are read.

    overhead_factor
        Memory overhead factor.

        Because the import process stores not only the original ``X`` but also creates temporary objects such as:

        - ``window_adatas``
        - ``sc.concat(...)`` or ``sparse.vstack(...)``
        - ``adata_window.copy()``
        - ``obs_df``
        - Arrow table
        - DuckDB register temporary objects

        Therefore, memory cannot be estimated only from ``X`` itself. Here it is multiplied by ``4.0`` by default,
        to more conservatively estimate the actual memory usage per cell during import.

    Returns
    -------
    info
        A dictionary containing the following fields:

        - ``x_scale``: detected X scale, usually ``"count"`` or ``"log"``;
        - ``sample_cells``: actual number of sampled cells;
        - ``nonzero_n``: number of nonzero expression values in the sample;
        - ``max``: maximum nonzero expression value in the sample;
        - ``q95``: 95th percentile of nonzero expression values in the sample;
        - ``frac_le_10``: fraction of nonzero expression values in the sample less than or equal to 10;
        - ``x_bytes``: base number of bytes for X in the sample estimated according to the target import structure;
        - ``x_bytes_per_cell``: base X memory estimate per cell;
        - ``estimated_bytes_per_cell``: per-cell memory estimate after multiplying by overhead_factor.

    Notes
    -----
    Memory estimation for sparse matrices follows the Atlas writing structure:

    - the CSR nonzero value array ``X.data`` is estimated as ``float32``;
    - ``indices`` are estimated as ``uint16``, corresponding to ``atlas_gene_id`` / ``USMALLINT``;
    - ``indptr`` is estimated as ``int64``.

    This estimate is not intended to be accurate at the Python-object level, but to obtain sufficiently stable
    ``window_cells`` and ``blocks_per_pool``.
    """

    n = min(sample_n, adata_backed.n_obs)

    if n <= 0:
        raise ValueError("[ERROR] there are no cells in the h5ad file, so X cannot be detected.")

    # Pre-read the first n cells.
    adata_sample = adata_backed[:n].to_memory()
    X = adata_sample.X

    if sparse.issparse(X):
        # Convert sparse matrices uniformly to CSR for easier statistics on data / indices / indptr.
        X = X.tocsr()

        # Nonzero expression values, used to determine count / log.
        values = np.asarray(X.data, dtype=np.float32)

        # Estimate the base memory of X according to the final Atlas writing structure:
        # data    -> float32
        # indices -> uint16
        # indptr  -> int64
        x_bytes = (
            X.data.size * np.dtype(np.float32).itemsize
            + X.indices.size * np.dtype(np.uint16).itemsize
            + X.indptr.size * np.dtype(np.int64).itemsize
        )
    else:
        # Estimate dense matrices as float32.
        X_arr = np.asarray(X)

        # Use only nonzero values to determine count / log.
        values = X_arr[X_arr != 0].astype(np.float32, copy=False)

        x_bytes = X_arr.size * np.dtype(np.float32).itemsize

    if values.size == 0:
        del adata_sample
        gc.collect()
        raise ValueError(
            "[ERROR] there are no nonzero expression values in the pre-read cells, so it is impossible to determine whether X is count or log."
        )

    vmax = float(np.max(values))
    q95 = float(np.percentile(values, 95))
    frac_le_10 = float(np.mean(values <= 10))

    # Heuristic rule:
    # count data usually contains larger integer counts;
    # log data usually has most nonzero values not very large.
    if vmax > 50 or q95 > 10:
        x_scale: XScale = "count"
    else:
        x_scale: XScale = "log"

    # Estimate the base X memory per cell.
    x_bytes_per_cell = float(x_bytes / n)

    # Add the empirical overhead factor for temporary objects during import.
    estimated_bytes_per_cell = max(1.0, x_bytes_per_cell * overhead_factor)

    logger.info("[INFO] X scale / memory precheck results:")
    logger.info(f"  - sample_cells = {n:,}")
    logger.info(f"  - nonzero_n    = {values.size:,}")
    logger.info(f"  - max          = {vmax:.4f}")
    logger.info(f"  - q95          = {q95:.4f}")
    logger.info(f"  - frac <= 10   = {frac_le_10:.4f}")
    logger.info(f"  - x_scale      = {x_scale}")
    logger.info(f"  - x_bytes_per_cell = {x_bytes_per_cell:.2f}")
    logger.info(f"  - estimated_bytes_per_cell = {estimated_bytes_per_cell:.2f}")

    del adata_sample
    gc.collect()

    return {
        "x_scale": x_scale,
        "sample_cells": n,
        "nonzero_n": int(values.size),
        "max": vmax,
        "q95": q95,
        "frac_le_10": frac_le_10,
        "x_bytes": int(x_bytes),
        "x_bytes_per_cell": x_bytes_per_cell,
        "estimated_bytes_per_cell": estimated_bytes_per_cell,
    }


def _estimate_window_cells_and_blocks_per_pool(
    *,
    memory_limit: str | int | None,
    cells_per_block: int,
    estimated_bytes_per_cell: float,
    memory_fraction: float = 0.05,      #
    default_blocks_per_pool: int = 20,  #
    min_blocks_per_pool: int = 5,       #
    max_blocks_per_pool: int = 100,     #
) -> tuple[int, int]:
    """Estimate the import window size and blocks_per_pool based on the memory limit.

    This function automatically calculates the following based on the Atlas ``db_memory_limit``, the user-provided ``cells_per_block``,
    and the sample-estimated ``estimated_bytes_per_cell``:

    1. ``window_cells``: the maximum number of cells in one import window;
    2. ``blocks_per_pool``: how many blocks are included in one window.

    Their relationship is:

    ``window_cells = cells_per_block * blocks_per_pool``

    Parameters
    ----------
    memory_limit
        Memory limit stored in Atlas.

        Common forms include:
        - ``"32GB"``
        - ``"16G"``
        - ``32``
        - ``None``

    cells_per_block
        Number of cells contained in each contiguous read block specified by the user.

        For example:

        ``cells_per_block = 500``

    estimated_bytes_per_cell
        Per-cell memory usage estimated by ``_inspect_x_from_backed()``.

        This value has already been multiplied by ``overhead_factor``, so it is more conservative than the memory of X alone.

    memory_fraction
        Memory fraction used to estimate the import window.

        For example, when ``memory_limit="32GB"`` and ``memory_fraction=0.25``,
        it means that at most about 8GB is used to estimate the import window.

        Using 1.0 is not recommended here because Python, NumPy, pandas, AnnData,
        Arrow, and DuckDB all consume additional memory.

    default_blocks_per_pool
        Default number of window blocks used when ``memory_limit`` is ``None``.

        The default value is 20, equivalent to the default behavior of older versions:

        ``window_cells = cells_per_block * 20``

    min_blocks_per_pool
        Minimum lower limit of ``blocks_per_pool`` when ``memory_limit`` is set.

    max_blocks_per_pool
        Maximum upper limit of ``blocks_per_pool``.

        This parameter avoids overly optimistic memory estimation that could make one window contain too many blocks,
        which could cause excessively high peak memory usage during ``sc.concat``, ``sparse.vstack``, or Arrow writing.

    Returns
    -------
    window_cells
        Actual number of cells used in one import window.

        This value will be realigned to:

        ``cells_per_block * blocks_per_pool``

    blocks_per_pool
        Number of blocks contained in one import window.

        The calculation is approximately:

        ``blocks_per_pool = window_cells // cells_per_block``

    Notes
    -----
    This function only estimates the import window size. It does not directly read h5ad or write to the database.

    Meaning in different import modes:

    - ``random`` mode: indicates how many blocks are included in one shuffle window;
    - multi-file ``random`` / ``order`` mode: indicates how many blocks are read from the global block pool each time;
    - ``order`` mode: ``window_cells`` is directly used as ``mega_batch_size``.
    """

    if cells_per_block <= 0:
        raise ValueError("cells_per_block must be > 0")

    if estimated_bytes_per_cell <= 0:
        raise ValueError("estimated_bytes_per_cell must be > 0")

    if min_blocks_per_pool <= 0:
        raise ValueError("min_blocks_per_pool must be > 0")

    if max_blocks_per_pool <= 0:
        raise ValueError("max_blocks_per_pool must be > 0")

    if min_blocks_per_pool > max_blocks_per_pool:
        raise ValueError("min_blocks_per_pool must be <= max_blocks_per_pool")

    memory_limit_bytes = _parse_memory_limit_to_bytes(memory_limit)

    # When no memory limit is set, keep the old default behavior.
    if memory_limit_bytes is None:
        blocks_per_pool = default_blocks_per_pool
        window_cells = cells_per_block * blocks_per_pool
        return window_cells, blocks_per_pool

    if not 0 < memory_fraction <= 1:
        raise ValueError("memory_fraction must be in the range (0, 1]")

    # Use only a fraction of memory_limit to estimate the import window to avoid excessive memory peaks.
    usable_memory_bytes = memory_limit_bytes * memory_fraction

    # Infer how many cells can fit in one window based on the estimated memory per cell.
    window_cells = int(usable_memory_bytes // estimated_bytes_per_cell)

    # Ensure at least one block.
    window_cells = max(cells_per_block, window_cells)

    # Calculate blocks_per_pool from window_cells and cells_per_block.
    blocks_per_pool = window_cells // cells_per_block
    blocks_per_pool = max(min_blocks_per_pool, blocks_per_pool)

    # Set an upper limit to avoid an overly large window.
    blocks_per_pool = min(blocks_per_pool, max_blocks_per_pool)

    # Realign window_cells to ensure it is exactly equal to cells_per_block * blocks_per_pool.
    window_cells = cells_per_block * blocks_per_pool

    logger.info("[INFO] Automatically estimated h5ad import window parameters:")
    logger.info(f"  - memory_limit = {memory_limit}")
    logger.info(f"  - memory_limit_bytes = {memory_limit_bytes:,}")
    logger.info(f"  - memory_fraction = {memory_fraction}")
    logger.info(f"  - cells_per_block = {cells_per_block:,}")
    logger.info(f"  - estimated_bytes_per_cell = {estimated_bytes_per_cell:.2f}")
    logger.info(f"  - window_cells = {window_cells:,}")
    logger.info(f"  - blocks_per_pool = {blocks_per_pool:,}")

    return window_cells, blocks_per_pool


def _detect_x_scale_from_backed(
    adata_backed: AnnData,
    sample_n: int = 5000,
) -> XScale:
    """Detect the storage scale of the input expression matrix.

    This internal function is a lightweight wrapper around ``_inspect_x_from_backed`` and only returns the X scale detection
    result. It pre-reads some cells from backed AnnData and uses the maximum nonzero expression value,
    percentiles, and low-value fraction to determine whether the input is more like count data or log data.

    Parameters
    ----------
    adata_backed
        AnnData object opened in backed mode.
    sample_n
        Number of cells used for pre-detection. The actual number read is
        ``min(sample_n, adata_backed.n_obs)``。

    Returns
    -------
    x_scale
        Detected expression matrix scale, usually ``"count"`` or ``"log"``.

    Notes
    -----
    This function does not modify AnnData or write to the database; the actual conversion is performed by
    ``_convert_x_to_count_inplace``.
    """

    x_info = _inspect_x_from_backed(
        adata_backed,
        sample_n=sample_n,
    )

    return x_info["x_scale"]


def _normalize_cells_per_block(cells_per_block, n_cells):
    """Normalize the cell block size used during import.

    This internal function is used by ``load_h5ad`` and its underlying import workflows. When the user does not explicitly specify
    ``cells_per_block``, the function estimates a default value based on the total number of cells and limits it to
    between ``512`` and ``4096``.

    Parameters
    ----------
    cells_per_block
        Block size provided by the user, or ``None``.
    n_cells
        Total number of cells in the current dataset to import.

    Returns
    -------
    int
        Normalized positive integer block size.

    Notes
    -----
    This function only generates the block size and does not check the memory window; the memory window is
    further estimated by ``_estimate_window_cells_and_blocks_per_pool``.
    """

    if cells_per_block is None:
        cells_per_block = int(0.001 * n_cells)
        cells_per_block = max(512, min(cells_per_block, 4096))

    cells_per_block = int(cells_per_block)
    logger.info(f"cells_per_block = {cells_per_block}")
    return cells_per_block

# Convert adata.X in place according to the input X scale; during import, write uniformly as count, with log conversion base e
def _convert_x_to_count_inplace(
    adata: AnnData,
    source_x_scale: XScale,
):
    """Convert the expression matrix to the count scale.

    This internal function is used to unify the expression matrix scale before import. The Atlas import workflow stores the expression matrix as
    count by default, so when the input ``adata.X`` is determined to be on the log scale, the function performs
    ``expm1`` conversion and clips negative values to 0, using e as the default conversion base.

    For sparse matrices, the function only converts the nonzero value array ``X.data`` without breaking the sparse structure; for dense
    matrices, the function converts the entire matrix to ``float32`` and processes it in place.

    Parameters
    ----------
    adata
        AnnData object to convert. The function updates its ``adata.X``.
    source_x_scale
        Current scale of the input expression matrix. Supports ``"count"`` and ``"log"``.

    Returns
    -------
    AnnData
        The same AnnData object after conversion.

    Notes
    -----
    When ``source_x_scale="count"``, the function directly returns the original object without copying.
    """

    if source_x_scale == "count":
        return adata

    X = adata.X

    # 1. sparse matrix: only modify nonzero values in X.data without breaking the sparse structure
    if sparse.issparse(X):
        if not sparse.isspmatrix_csr(X):
            X = X.tocsr()

        X.data = X.data.astype(np.float32, copy=False)

        if source_x_scale == "log":
            np.expm1(X.data, out=X.data)
            X.data[X.data < 0] = 0

        else:
            raise ValueError(f"Unsupported X scale: {source_x_scale}")

        adata.X = X
        return adata

    # 2. dense matrix: directly convert the entire matrix
    X = np.asarray(X, dtype=np.float32)

    if source_x_scale == "log":
        np.expm1(X, out=X)
        np.maximum(X, 0, out=X)

    else:
        raise ValueError(f"Unsupported X scale: {source_x_scale}")

    adata.X = X
    return adata


# Simply detect the underlying sparse format of h5ad.X
def _print_h5ad_x_format(h5ad_path: PathLike[str] | str):
    """Detect and print the underlying storage format of ``X`` in the h5ad file.

    This internal function directly reads the HDF5 structure of the h5ad file to determine whether ``X`` is a dense dataset or
    a sparse matrix group, and returns ``csr``, ``csc``,
    ``coo``, or another format name according to ``encoding-type``. It is only used for pre-import logging and diagnostics and does not read the full expression matrix.

    Parameters
    ----------
    h5ad_path
        Path to the h5ad file.

    Returns
    -------
    str or None
        Detected ``X`` storage format. Returns ``None`` if the file does not contain ``X``.

    Notes
    -----
    This function does not determine whether expression values are count or log; expression-scale detection is performed by
    ``_inspect_x_from_backed``.
    """

    h5ad_path = os.fspath(h5ad_path)

    with h5py.File(h5ad_path, "r") as f:
        if "X" not in f:
            logger.info("[INFO] h5ad.X format = None (the file does not contain X)")
            return None

        X = f["X"]

        # dense matrix
        if isinstance(X, h5py.Dataset):
            logger.info("[INFO] h5ad.X format = dense")
            return "dense"

        # sparse matrix group
        if isinstance(X, h5py.Group):
            encoding_type = X.attrs.get("encoding-type", "unknown")

            if isinstance(encoding_type, bytes):
                encoding_type = encoding_type.decode("utf-8")

            if encoding_type == "csr_matrix":
                logger.info("[INFO] h5ad.X format = CSR")
                return "csr"

            elif encoding_type == "csc_matrix":
                logger.info("[INFO] h5ad.X format = CSC")
                return "csc"

            elif encoding_type == "coo_matrix":
                logger.info("[INFO] h5ad.X format = COO")
                return "coo"

            else:
                logger.info(f"[INFO] h5ad.X format = unknown ({encoding_type})")
                return encoding_type

        logger.info("[INFO] h5ad.X format = unknown")
        return "unknown"


# Infer data types
def _infer_duckdb_type_from_series(series: pd.Series) -> str:

    """Infer DuckDB field type from a pandas Series.

    This internal function is used to create DuckDB table schemas from AnnData ``obs`` or ``var`` metadata.
    It maps pandas integer, floating, and boolean types to DuckDB ``BIGINT``,
    ``DOUBLE``, and ``BOOLEAN`` respectively; all other types are handled as ``VARCHAR``.

    Parameters
    ----------
    series
        pandas Series whose DuckDB type needs to be inferred.

    Returns
    -------
    str
        DuckDB field type name.

    Notes
    -----
    Strings, categorical variables, object types, and types that cannot be precisely mapped are written as ``VARCHAR``.
    """
    if pd.api.types.is_integer_dtype(series):
        return "BIGINT"
    if pd.api.types.is_float_dtype(series):
        return "DOUBLE"
    if pd.api.types.is_bool_dtype(series):
        return "BOOLEAN"
    return "VARCHAR"


# Create the obs table
def _create_obs_table_from_adata(conn: DuckDBPyConnection, adata: AnnData):
    """Create the Atlas ``obs`` table from AnnData obs metadata.

    This internal function reads column names and pandas dtypes from ``adata.obs``, infers the corresponding DuckDB field types, and creates a standard ``obs`` table containing
    ``atlas_cell_id`` and ``atlas_cell_name``.

    Similar to the Scanpy convention of storing cell annotations in ``adata.obs``, Atlas persists cell-level metadata into database tables,
    for reuse by subsequent QC, filtering, clustering, differential analysis, and plotting functions.

    Parameters
    ----------
    conn
        DuckDB connection object.

    adata
        Input AnnData object.

        It must contain ``obs`` and ``obs_names``; if the source data already contains Atlas system fields, they are skipped when creating the table.

    Notes
    -----
    This function only creates the table schema and does not write specific cell metadata. The writing process is completed by the upstream import function.

    Returns
    -------
    None
        The table schema is created directly in the DuckDB database and no object is returned.
    """

    # System reserved fields: uniformly created by scAtlasPy
    reserved_cols = {"atlas_cell_id", "atlas_cell_name"}

    # Force use of the required types
    cols = [
        "atlas_cell_id   INTEGER",
        "atlas_cell_name VARCHAR",
    ]

    for col in adata.obs.columns:
        # Skip old system fields already present in the source h5ad
        if col in reserved_cols:
            continue

        duck_type = _infer_duckdb_type_from_series(adata.obs[col])
        cols.append(f'"{col}" {duck_type}')

    ddl = f"""
    CREATE OR REPLACE TABLE obs (
        {", ".join(cols)}
    )
    """

    conn.execute(ddl)


# Create the var table
def _create_var_table_from_adata(conn: DuckDBPyConnection, adata: AnnData):
    """Create the Atlas ``var`` table from AnnData var metadata.

    This internal function reads column names and pandas dtypes from ``adata.var``, infers the corresponding DuckDB field types, and creates a standard ``var`` table containing
    ``atlas_gene_id`` and ``atlas_gene_name``.

    Similar to the Scanpy convention of storing gene annotations in ``adata.var``, Atlas persists gene-level metadata into database tables,
    for use by steps such as HVG, PCA loadings, marker genes, and feature plots.

    Parameters
    ----------
    conn
        DuckDB connection object.

    adata
        Input AnnData object.

        It must contain ``var`` and ``var_names``; if the source data already contains Atlas system fields, they are skipped when creating the table.

    Notes
    -----
    This function only creates the table schema and does not write specific gene metadata. The writing process is completed by the upstream import function.

    Returns
    -------
    None
        The table schema is created directly in the DuckDB database and no object is returned.
    """

    # System reserved fields: uniformly created by scAtlasPy
    reserved_cols = {"atlas_gene_id", "atlas_gene_name"}

    # Force use of the types you required
    cols = [
        "atlas_gene_id USMALLINT",
        "atlas_gene_name VARCHAR",
    ]

    for col in adata.var.columns:
        # Skip old system fields already present in the source h5ad
        if col in reserved_cols:
            continue

        duck_type = _infer_duckdb_type_from_series(adata.var[col])
        cols.append(f'"{col}" {duck_type}')

    ddl = f"""
    CREATE OR REPLACE TABLE var (
        {", ".join(cols)}
    )
    """

    conn.execute(ddl)


# Create the HyS storage structure
def _create_hys_tables(conn: DuckDBPyConnection):

    """Create HyS storage tables for the Atlas expression matrix.

    This internal function creates the two core tables used by the Atlas expression matrix:

    - ``X_HyS_indptr``: stores the cumulative nonzero positions of CSR ``indptr``;
    - ``X_HyS_data``: stores nonzero expression records, including ``id``, ``atlas_cell_id``,
      ``atlas_gene_id`` and ``data_count``.

    The function uses ``CREATE OR REPLACE TABLE``, so it overwrites old tables with the same names.

    Parameters
    ----------
    conn
        DuckDB database connection.

    Returns
    -------
    None
        The table schema is created directly in the DuckDB database and no object is returned.

    Notes
    -----
    ``X_HyS_indptr`` does not store the first 0; it only stores the cumulative indptr at each cell's end position;
    ``X_HyS_data.data_count`` is the imported count-scale expression value.
    """
    conn.execute(
        """ -- do not store the first 0 value
        CREATE OR REPLACE TABLE X_HyS_indptr (
            atlas_cell_id  INTEGER,  --   int32
            indptr BIGINT,           --   int64
        )
        """
    )
    conn.execute(
        """
        CREATE OR REPLACE TABLE X_HyS_data (
            id BIGINT,                --   int64
            atlas_cell_id  INTEGER,   --   int32
            atlas_gene_id  USMALLINT,  --  unsigned int16, between 0 and 65535
            data_count REAL            --  raw count, float32 single-precision floating point number (4 bytes)
        )
        """
    )


# Import the obs table
def _append_obs_rows(adata: AnnData, conn: DuckDBPyConnection, start_cell_id: int) -> int:

    """Append ``obs`` rows from one AnnData block.

    This internal function is used for backed h5ad chunked import. It copies ``adata.obs``, removes
    Atlas system fields that may already exist in the source, then regenerates the current database's ``atlas_cell_id`` and
    ``atlas_cell_name``, and finally appends them into the ``obs`` table.

    Parameters
    ----------
    adata
        Current AnnData block to write.
    conn
        DuckDB database connection.
    start_cell_id
        Starting ``atlas_cell_id`` used when writing the current obs block.

    Returns
    -------
    int
        Starting ``atlas_cell_id`` that should be used by the next block.

    Notes
    -----
    ``atlas_cell_name`` comes from the current block's ``adata.obs.index``.
    """
    n = adata.n_obs

    obs_df = adata.obs.copy()

    # Remove old system fields already present in the source h5ad
    for c in ["atlas_cell_id", "atlas_cell_name"]:
        if c in obs_df.columns:
            obs_df = obs_df.drop(columns=[c])

    # Regenerate atlas_cell_id / atlas_cell_name for the current database
    obs_df["atlas_cell_id"] = np.arange(
        start_cell_id,
        start_cell_id + n,
        dtype=np.int32,
    )

    obs_df["atlas_cell_name"] = adata.obs.index.astype(str)

    # Fix the column order: system fields are always placed first
    obs_df = obs_df[
        ["atlas_cell_id", "atlas_cell_name"]
        + [
            c for c in obs_df.columns
            if c not in ("atlas_cell_id", "atlas_cell_name")
        ]
    ]

    conn.register("obs_df", obs_df)
    conn.execute("INSERT INTO obs SELECT * FROM obs_df")
    conn.unregister("obs_df")

    return start_cell_id + n


# Import the var table
def _append_var(adata: AnnData, conn: DuckDBPyConnection):

    """Write AnnData ``var`` gene metadata.

    This internal function is used in the h5ad chunked import workflow. It copies ``adata.var``, removes
    Atlas system fields that may already exist in the source, regenerates ``atlas_gene_id`` and
    ``atlas_gene_name``, and inserts them into the ``var`` table.

    Parameters
    ----------
    adata
        AnnData object used to provide gene metadata.
    conn
        DuckDB database connection.

    Returns
    -------
    None
        Gene metadata is written directly into the ``var`` table and no object is returned.

    Notes
    -----
    During multi-block import, ``var`` only needs to be written once.
    """
    var_df = adata.var.copy()

    # Remove old system fields already present in the source h5ad
    for c in ["atlas_gene_id", "atlas_gene_name"]:
        if c in var_df.columns:
            var_df = var_df.drop(columns=[c])

    # Regenerate atlas_gene_id / atlas_gene_name for the current database
    var_df["atlas_gene_id"] = np.arange(
        adata.n_vars,
        dtype=np.uint16,
    )

    var_df["atlas_gene_name"] = adata.var.index.astype(str)

    # Fix the column order: system fields are always placed first
    var_df = var_df[
        ["atlas_gene_id", "atlas_gene_name"]
        + [
            c for c in var_df.columns
            if c not in ("atlas_gene_id", "atlas_gene_name")
        ]
    ]

    conn.register("var_df", var_df)
    conn.execute("INSERT INTO var SELECT * FROM var_df")
    conn.unregister("var_df")


# Import the X_HyS table
def _append_x_hys(
    adata: AnnData,
    conn: DuckDBPyConnection,
    *,
    base_cell_id: int,
    global_indptr_id: int,
    global_indptr_offset: int,
    global_data_id: int,
):

    """Append the HyS sparse expression matrix from one AnnData block.

    This internal function converts ``adata.X`` to a CSR matrix and appends it into the Atlas HyS tables:
    ``X_HyS_indptr`` stores the cumulative indptr for each cell, and ``X_HyS_data`` stores nonzero expression values
    and their corresponding cell/gene IDs. Expression values are written to the ``data_count`` field.

    Parameters
    ----------
    adata
        Current AnnData block to write. ``adata.X`` is required to already be on the count scale.
    conn
        DuckDB database connection.
    base_cell_id
        Atlas cell ID corresponding to the first row of the current AnnData block.
    global_indptr_id
        Next indptr row ID to write.
    global_indptr_offset
        Current cumulative number of nonzero values already written, used to relocate indptr.
    global_data_id
        Next ``X_HyS_data.id`` to write.

    Returns
    -------
    tuple[int, int, int]
        Updated ``global_indptr_id``, ``global_indptr_offset``, and
        ``global_data_id``。

    Notes
    -----
    ``atlas_gene_id`` is written as ``uint16``, corresponding to ``USMALLINT`` in the database.
    """
    X = adata.X

    if not sparse.issparse(X):
        X = sparse.csr_matrix(X)
    elif not sparse.isspmatrix_csr(X):
        X = X.tocsr()


    indptr = X.indptr.astype(np.int64, copy=False)
    indices = X.indices.astype(np.uint16, copy=False)
    data_count = X.data.astype(np.float32, copy=False)

    # ================= indptr =================
    row_nnz = np.diff(indptr)

    # Do not store indptr[0]; only store indptr[1:]
    adj_indptr = indptr[1:] + np.int64(global_indptr_offset)

    indptr_table = pa.table({
        "atlas_cell_id": pa.array(
            np.arange(
                global_indptr_id,
                global_indptr_id + len(adj_indptr),
                dtype=np.int32,
            ),
            type=pa.int32(),
        ),
        "indptr": pa.array(
            adj_indptr,
            type=pa.int64(),
        ),
    })

    conn.register("_indptr_arrow", indptr_table)
    conn.execute("""
        INSERT INTO X_HyS_indptr (
            atlas_cell_id,
            indptr
        )
        SELECT
            atlas_cell_id,
            indptr
        FROM _indptr_arrow
    """)
    conn.unregister("_indptr_arrow")

    global_indptr_id += len(adj_indptr)

    # ================= data_count =================
    nnz = len(data_count)

    if nnz > 0:

        cell_index = np.repeat(
            np.arange(
                base_cell_id,
                base_cell_id + adata.n_obs,
                dtype=np.int32,
            ),
            row_nnz,
        )

        data_table = pa.table({
            "id": pa.array(
                np.arange(
                    global_data_id,
                    global_data_id + nnz,
                    dtype=np.int64,
                ),
                type=pa.int64(),
            ),
            "atlas_cell_id": pa.array(
                cell_index,
                type=pa.int32(),
            ),
            "atlas_gene_id": pa.array(
                indices,
                type=pa.uint16(),
            ),
            "data_count": pa.array(
                data_count,
                type=pa.float32(),
            ),
        })

        conn.register("_data_arrow", data_table)
        conn.execute("""
            INSERT INTO X_HyS_data (
                id,
                atlas_cell_id,
                atlas_gene_id,
                data_count
            )
            SELECT
                id,
                atlas_cell_id,
                atlas_gene_id,
                data_count
            FROM _data_arrow
        """)
        conn.unregister("_data_arrow")

        global_data_id += nnz
        global_indptr_offset += nnz

    return global_indptr_id, global_indptr_offset, global_data_id


# Import obsm
def _add_obsm_from_h5ad(h5ad_path: PathLike[str] | str, atlas: Atlas, cells_per_block: int = 500):

    """Import ``obsm`` from an h5ad file into the Atlas database.

    This internal function directly reads the ``obsm`` group in the h5ad file, and for each ``obsm`` key
    creates an ``obsm_{key}`` table. Each table contains ``atlas_cell_id`` and several
    dimension columns such as ``dim_0`` and ``dim_1``.

    This function is suitable for import modes that preserve the original cell order. If the cell order has already been randomly reordered,
    ``obsm`` should not be imported directly in the original h5ad order.

    Parameters
    ----------
    h5ad_path
        Path to the h5ad file.
    atlas
        Atlas object. The object must already be connected to a DuckDB database.
    cells_per_block
        Number of cells in each batch when reading ``obsm`` arrays in batches.

    Returns
    -------
    None
        ``obsm`` results are written directly into the database and no object is returned.

    Notes
    -----
    If ``obsm`` does not exist in the h5ad file, the function logs this and skips it.
    """
    h5ad_path = os.fspath(h5ad_path)

    logger.info("Import obsm")
    conn = atlas.connection

    with h5py.File(h5ad_path, "r") as f:

        if "obsm" not in f:
            logger.info("  - obsm does not exist in the h5ad file, skipping")
            return

        obsm_grp = f["obsm"]

        for key in obsm_grp.keys():
            dset = obsm_grp[key]
            n_cells, k = dset.shape

            logger.info(f"  - obsm[{key}] shape={dset.shape}")

            cols = ", ".join([f"dim_{i} DOUBLE" for i in range(k)])
            conn.execute(f"""
                CREATE OR REPLACE TABLE obsm_{key} (
                    atlas_cell_id BIGINT,
                    {cols}
                )
            """)

            for start in range(0, n_cells, cells_per_block):

                end = min(start + cells_per_block, n_cells)
                block = dset[start:end]
                df = pd.DataFrame(
                    block,
                    columns=[f"dim_{i}" for i in range(k)]
                )
                df["atlas_cell_id"] = np.arange(start, end, dtype=np.int32)
                df = df[["atlas_cell_id"] + [c for c in df.columns if c != "atlas_cell_id"]]

                conn.register("obsm_df", df)
                conn.execute(f"INSERT INTO obsm_{key} SELECT * FROM obsm_df")
                conn.unregister("obsm_df")

    logger.info("obsm import completed")


# Import varm
def _add_varm_from_h5ad(h5ad_path: PathLike[str] | str, atlas: Atlas):

    """Import ``varm`` from an h5ad file into the Atlas database.

    This internal function directly reads the ``varm`` group in the h5ad file, and for each ``varm`` key
    creates a ``varm_{key}`` table. Each table contains ``atlas_gene_id`` and several
    dimension columns such as ``dim_0`` and ``dim_1``.

    ``varm`` is aligned to the gene dimension and does not depend on cell order, so both ordered import and random import can
    import it safely.

    Parameters
    ----------
    h5ad_path
        Path to the h5ad file.
    atlas
        Atlas object. The object must already be connected to a DuckDB database.

    Returns
    -------
    None
        ``varm`` results are written directly into the database and no object is returned.

    Notes
    -----
    If ``varm`` does not exist in the h5ad file, the function logs this and skips it.
    """
    h5ad_path = os.fspath(h5ad_path)

    logger.info("Import varm")

    conn = atlas.connection

    with h5py.File(h5ad_path, "r") as f:

        if "varm" not in f:
            logger.info("  - varm does not exist in the h5ad file, skipping")
            return

        varm_grp = f["varm"]

        for key in varm_grp.keys():
            dset = varm_grp[key]
            n_genes, k = dset.shape

            logger.info(f"  - varm[{key}] shape={dset.shape}")

            df = pd.DataFrame(
                dset[:],
                columns=[f"dim_{i}" for i in range(k)]
            )
            df["atlas_gene_id"] = np.arange(n_genes, dtype=np.uint16)
            df = df[["atlas_gene_id"] + [c for c in df.columns if c != "atlas_gene_id"]]

            conn.register("varm_df", df)
            conn.execute(f"""
                CREATE OR REPLACE TABLE varm_{key} AS
                SELECT * FROM varm_df
            """)
            conn.unregister("varm_df")

    logger.info("varm import completed")


# Import multiple data formats
def _read_smart(file_path: PathLike[str] | str):
    """Automatically select the Scanpy reading function based on the file suffix.

    This function checks the extension of the input file path and calls the corresponding ``scanpy.read_*`` function to read it as AnnData.

    It is suitable for small datasets or temporary conversion scenarios, and can uniformly handle common input formats such as h5ad, loom, Matrix Market, csv, text tables, Excel,
    10x h5, UMI-tools, and other common input formats.

    Parameters
    ----------
    file_path
        Input single-cell data file path.

    Returns
    -------
    adata
        The loaded AnnData object.

    Notes
    -----
    This function is only responsible for reading files and does not directly write to the Atlas database. To import into the database, continue by calling ``load_anndata`` or
    ``load_multi_format``。

    Examples
    --------
    Read an h5ad file::

        adata = _read_smart("example.h5ad")
    """

    # Get the file extension (lowercase form)
    file_path = os.fspath(file_path)
    file_ext = os.path.splitext(file_path)[1].lower()

    # Select the corresponding reading method based on the file extension
    if file_ext == '.h5ad':
        # h5ad format
        return sc.read_h5ad(file_path)
    elif file_ext == '.loom':
        # loom format
        return sc.read_loom(file_path)
    elif file_ext in ['.mtx', '.mtx.gz']:
        # mtx format (Matrix Market format)
        return sc.read_mtx(file_path)
    elif file_ext in ['.csv', '.csv.gz']:
        # csv format
        return sc.read_csv(file_path)
    elif file_ext in ['.txt', '.tsv', '.tab']:
        # text format, tab-delimited by default
        return sc.read_text(file_path)
    elif file_ext in ['.xlsx', '.xls']:
        # Excel format
        return sc.read_excel(file_path)
    elif file_ext == '.h5':
        # 10x Genomics h5 format
        return sc.read_10x_h5(file_path)
    elif 'umi_tools' in file_path.lower():
        # UMI-tools format
        return sc.read_umi_tools(file_path)
    else:
        # If the suffix is not recognized, try using the generic read function
        return sc.read(file_path)


# The following functions are used only in load_anndata
# Import obs
def _add_obs(adata:AnnData, atlas:Atlas):

    """Create and write the ``obs`` table from in-memory AnnData.

    This internal function is used by ``load_anndata``. It copies ``adata.obs``, writes the AnnData cell
    index into ``atlas_cell_name``, and generates ``atlas_cell_id`` according to the current row order,
    then creates or replaces the ``obs`` table in the Atlas database.

    Parameters
    ----------
    adata
        Input AnnData object.
    atlas
        Atlas object. The object must already be connected to a DuckDB database.

    Returns
    -------
    None
        The ``obs`` table is written directly into the database and no object is returned.

    Notes
    -----
    After writing, a primary key is added to ``obs.atlas_cell_id``.
    """
    logger.info("Import obs data")

    obs_df = adata.obs.copy()
    obs_df['atlas_cell_name'] = adata.obs.index

    obs_df['atlas_cell_id'] = range(len(obs_df))  # Add the id column

    obs_df = obs_df[['atlas_cell_id', 'atlas_cell_name'] + [col for col in obs_df.columns if col not in ['atlas_cell_id', 'atlas_cell_name']]]  # Directly specify column order
    logger.info(f"obs table data preparation completed, rows: {len(obs_df)}")

    atlas.connection.register('obs_df', obs_df)
    atlas.connection.execute("CREATE OR REPLACE TABLE obs AS SELECT * FROM obs_df")
    atlas.connection.execute("ALTER TABLE obs ADD PRIMARY KEY (atlas_cell_id)")  # Set the ID field as the primary key to ensure uniqueness
    atlas.connection.unregister('obs_df')
    logger.info("obs data imported successfully")


# Import var
def _add_var(adata:AnnData, atlas:Atlas):

    """Create and write the ``var`` table from in-memory AnnData.

    This internal function is used by ``load_anndata``. It writes the AnnData gene index into
    ``atlas_gene_name``, generates ``atlas_gene_id`` according to the current gene order, and creates or replaces
    the ``var`` table in the Atlas database.

    Parameters
    ----------
    adata
        Input AnnData object.
    atlas
        Atlas object. The object must already be connected to a DuckDB database.

    Returns
    -------
    None
        The ``var`` table is written directly into the database and no object is returned.

    Notes
    -----
    After writing, a primary key is added to ``var.atlas_gene_id``.
    """
    logger.info("Import var data")
    var_df = adata.var.reset_index().rename(columns={'index': 'atlas_gene_name'})
    var_df['atlas_gene_id'] = range(len(var_df))
    var_df = var_df[['atlas_gene_id', 'atlas_gene_name'] + [col for col in var_df.columns if col not in ['atlas_gene_id', 'atlas_gene_name']]]  # Directly specify column order
    logger.info(f"var table data preparation completed, rows: {len(var_df)}")

    atlas.connection.register('var_df', var_df)
    atlas.connection.execute("CREATE OR REPLACE TABLE var AS SELECT * FROM var_df")
    atlas.connection.execute("ALTER TABLE var ADD PRIMARY KEY (atlas_gene_id)")  # Set the ID field as the primary key to ensure uniqueness
    atlas.connection.unregister('var_df')
    logger.info("var data imported successfully")


# Import obsm
def _add_obsm(adata: AnnData, atlas: Atlas):

    """Import ``obsm`` matrices from in-memory AnnData.

    This internal function is used by ``load_anndata``. It iterates over two-dimensional matrices in ``adata.obsm``,
    creates or replaces an ``obsm_{key}`` table for each key, with fields including ``atlas_cell_id``
    and dimension columns such as ``dim_0`` and ``dim_1``.

    Parameters
    ----------
    adata
        Input AnnData object.
    atlas
        Atlas object. The object must already be connected to a DuckDB database.

    Returns
    -------
    None
        ``obsm`` results are written directly into the database and no object is returned.

    Notes
    -----
    If ``adata.obsm`` is empty, the function logs this and skips it.
    """
    logger.info("Import obsm ")

    conn = atlas.connection
    n_cells = adata.n_obs

    if not adata.obsm:
        logger.info("  - no obsm, skipping")
        return

    for key, mat in adata.obsm.items():
        logger.info(f"  - obsm[{key}] shape = {mat.shape}")

        # Force numpy conversion (avoid pandas sparse pitfalls)
        mat = np.asarray(mat)

        df = pd.DataFrame(
            mat,
            columns=[f"dim_{i}" for i in range(mat.shape[1])]
        )

        df.insert(0, "atlas_cell_id", np.arange(n_cells, dtype=np.int32))

        conn.register("obsm_df", df)
        conn.execute(f"""
            CREATE OR REPLACE TABLE obsm_{key} AS
            SELECT * FROM obsm_df
        """)
        conn.unregister("obsm_df")

    logger.info("obsm import completed (unified schema)")


# Import varm
def _add_varm(adata: AnnData, atlas: Atlas):

    """Import ``varm`` matrices from in-memory AnnData.

    This internal function is used by ``load_anndata``. It iterates over two-dimensional matrices in ``adata.varm``,
    creates or replaces a ``varm_{key}`` table for each key, with fields including ``atlas_gene_id``
    and dimension columns such as ``dim_0`` and ``dim_1``.

    Parameters
    ----------
    adata
        Input AnnData object.
    atlas
        Atlas object. The object must already be connected to a DuckDB database.

    Returns
    -------
    None
        ``varm`` results are written directly into the database and no object is returned.

    Notes
    -----
    If ``adata.varm`` is empty, the function logs this and skips it.
    """
    logger.info("Import varm (unified schema)")

    conn = atlas.connection
    n_genes = adata.n_vars

    if not adata.varm:
        logger.info("  - no varm, skipping")
        return

    for key, mat in adata.varm.items():
        logger.info(f"  - varm[{key}] shape = {mat.shape}")

        mat = np.asarray(mat)

        df = pd.DataFrame(
            mat,
            columns=[f"dim_{i}" for i in range(mat.shape[1])]
        )

        df.insert(0, "atlas_gene_id", np.arange(n_genes, dtype=np.uint16))

        conn.register("varm_df", df)
        conn.execute(f"""
            CREATE OR REPLACE TABLE varm_{key} AS
            SELECT * FROM varm_df
        """)
        conn.unregister("varm_df")

    logger.info("varm import completed (unified schema)")


# Import x_hys
def _add_x_hys_chunked(adata: AnnData, atlas: Atlas, chunk_size: int = 500):

    """Write the HyS expression matrix from in-memory AnnData in chunks.

    This internal function is used by ``load_anndata``. It converts ``adata.X`` to CSR in cell chunks,
    and writes it into ``X_HyS_indptr`` and ``X_HyS_data``. Nonzero expression values are written to
    the ``X_HyS_data.data_count`` field.

    Parameters
    ----------
    adata
        Input AnnData object. The function reads ``adata.X``.
    atlas
        Atlas object. The function connects to and writes into the corresponding DuckDB database.
    chunk_size
        Number of cells in each chunk, used to control peak memory and single-write size.

    Returns
    -------
    bool
        Returns ``True`` when writing succeeds. On exception, the transaction is rolled back and the error is re-raised.

    Notes
    -----
    This function rebuilds the ``X_HyS_indptr`` and ``X_HyS_data`` tables.
    """
    logger.info("Start importing X_HyS ")

    conn = atlas.connect("r+")
    atlas.connection = conn

    n_cells = adata.n_obs

    # ===================== Create tables =====================
    conn.execute(
        """ -- do not store the first 0 value
        CREATE OR REPLACE TABLE X_HyS_indptr(
            atlas_cell_id  INTEGER,
            indptr BIGINT
        )
        """
    )
    conn.execute(
        """
        CREATE OR REPLACE TABLE X_HyS_data (
            id BIGINT,
            atlas_cell_id INTEGER,
            atlas_gene_id USMALLINT,  --  unsigned int16, between 0 and 65535
            data_count REAL           --  raw count, float32 single-precision floating point number (4 bytes)
        )
        """
    )

    conn.execute("BEGIN TRANSACTION")

    try:
        total_chunks = (n_cells + chunk_size - 1) // chunk_size

        global_data_counter = np.int64(0)
        global_indptr_offset = np.int64(0)

        for chunk_idx in progress(range(total_chunks), desc="load"):
            start = chunk_idx * chunk_size
            end = min(start + chunk_size, n_cells)
            size = end - start

            # === Get data ===
            X_chunk = adata.X[start:end]

            if sparse.issparse(X_chunk):
                csr = X_chunk.tocsr()
            else:
                csr = sparse.csr_matrix(X_chunk)

            data_count = csr.data
            indices = csr.indices.astype(np.uint16)
            indptr = csr.indptr.astype(np.int64)

            # ================= indptr table =================
            adj_indptr = indptr[1:] + global_indptr_offset

            indptr_df = pd.DataFrame({
                "atlas_cell_id": np.arange(start, end, dtype=np.int32),
                "indptr": adj_indptr
            })

            conn.execute("INSERT INTO X_HyS_indptr SELECT * FROM indptr_df")

            global_indptr_offset = adj_indptr[-1]

            # ================= data_count expression values =================
            if len(data_count) > 0:
                nnz = len(data_count)
                data_ids = np.arange(
                    global_data_counter,
                    global_data_counter + nnz,
                    dtype=np.int64
                )

                # Construct cell_index directly within the chunk (CSR → COO)
                row_lengths = np.diff(indptr)
                cell_index = np.repeat(
                    np.arange(start, end, dtype=np.int32),
                    row_lengths
                )

                data_df = pd.DataFrame({
                    "id": data_ids,
                    "atlas_cell_id": cell_index,
                    "atlas_gene_id": indices,
                    "data_count": data_count
                })

                conn.execute("INSERT INTO X_HyS_data SELECT * FROM data_df")

                global_data_counter += nnz

            # === Clean up ===
            del X_chunk, csr, indptr_df
            if len(data_count) > 0:
                del data_df
            gc.collect()

        conn.execute("COMMIT")

        logger.info(
            f"Import completed: cells={n_cells:,}, nnz={global_data_counter:,}"
        )

        return True

    except Exception as e:
        conn.execute("ROLLBACK")
        logger.error(f"CSR import failed: {e}")
        raise
