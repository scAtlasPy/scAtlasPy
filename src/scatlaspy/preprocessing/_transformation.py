from _duckdb import DuckDBPyConnection
from ..data import Atlas, duckdb_memory_limit
from ..io import progress
from typing import Literal
from typing import Optional
import logging
from numbers import Number
import math
import os
import numpy as np
import pandas as pd
from datetime import datetime
import gc
logger = logging.getLogger('Atlas')


def _derived_data_table_name(data_name: str) -> str:
    return f"X_HyS_data_{data_name}"


def _has_table(conn: DuckDBPyConnection, table_name: str) -> bool:
    return conn.execute(
        """
        SELECT 1
        FROM information_schema.tables
        WHERE table_name = ?
        LIMIT 1
        """,
        [table_name],
    ).fetchone() is not None


def _has_column(conn: DuckDBPyConnection, table_name: str, column_name: str) -> bool:
    return conn.execute(
        """
        SELECT 1
        FROM information_schema.columns
        WHERE table_name = ?
          AND column_name = ?
        LIMIT 1
        """,
        [table_name, column_name],
    ).fetchone() is not None


def _require_expression_data(conn: DuckDBPyConnection, data_name: str) -> None:
    """Require an expression field on X_HyS_data or its derived value table."""

    if _has_column(conn, "X_HyS_data", data_name):
        return

    table_name = _derived_data_table_name(data_name)
    if _has_table(conn, table_name) and _has_column(conn, table_name, data_name):
        return

    raise ValueError(
        f"Expression field does not exist: {data_name}. "
        f"Expected X_HyS_data.{data_name} or {table_name}.{data_name}."
    )


def _expression_source_for_transform(conn: DuckDBPyConnection, data_name: str) -> tuple[str, str]:
    """Return a FROM clause and value expression for transforms needing cell/gene ids."""

    if _has_column(conn, "X_HyS_data", data_name):
        return "X_HyS_data x", f"x.{data_name}"

    table_name = _derived_data_table_name(data_name)
    if not (_has_table(conn, table_name) and _has_column(conn, table_name, data_name)):
        raise ValueError(
            f"Expression field does not exist: {data_name}. "
            f"Expected X_HyS_data.{data_name} or {table_name}.{data_name}."
        )

    if _has_column(conn, table_name, "atlas_cell_id") and _has_column(conn, table_name, "atlas_gene_id"):
        return f"{table_name} x", f"x.{data_name}"

    return f"X_HyS_data x JOIN {table_name} d ON x.id = d.id", f"d.{data_name}"


def _write_expression_transform_meta(
    conn: DuckDBPyConnection,
    *,
    data_name: str,
    source_data: str,
    transform: str,
    centered: bool,
) -> None:
    """Record how a derived expression field was generated."""

    conn.execute("""
        CREATE TABLE IF NOT EXISTS atlas_expression_transform_meta (
            data_name VARCHAR PRIMARY KEY,
            source_data VARCHAR,
            transform VARCHAR,
            centered BOOLEAN
        )
    """)
    conn.execute(
        """
        INSERT OR REPLACE INTO atlas_expression_transform_meta (
            data_name,
            source_data,
            transform,
            centered
        )
        VALUES (?, ?, ?, ?)
        """,
        [data_name, source_data, transform, bool(centered)],
    )


def normalize_total(
    atlas: Atlas,
    target_sum: float = 10_000,
    chunk_cells: int = 500_000,
    add_data: str = "data_normalize",
    use_data: str = "data_count",
) -> None:

    """Normalize by total expression per cell.

    This function normalizes the expression matrix in the Atlas database by the
    total expression of each cell. For each cell, it first computes the total
    expression of that cell in the ``use_data`` field, then scales all explicitly
    stored nonzero expression values in that cell to the scale defined by
    ``target_sum``, and writes the result to a derived expression table.

    This workflow is commonly used to adjust cells with different sequencing
    depths to a comparable expression scale. For example, the default parameters
    normalize the total expression of each cell to 10,000.

    The function processes data in chunks over the ``atlas_cell_id`` range. Each
    chunk only computes total counts within the current cell range and writes
    normalized expression records from the current chunk to a temporary target
    table. After all chunks are completed, the temporary table is renamed to the
    derived expression table for ``add_data``. The original ``X_HyS_data`` count
    table is not replaced.

    Parameters
    ----------
    atlas
        Atlas object. The object must already be connected to a DuckDB database,
        and the database must contain at least the ``obs`` table and the
        ``X_HyS_data`` table.

        The ``X_HyS_data`` table must contain ``atlas_cell_id``, ``id``, and the
        expression value field specified by ``use_data``.

    target_sum
        Target total expression per cell after normalization. The default value
        is ``10_000``.

        For each cell, the output value is approximately:

        ``x_normalized = x / cell_total * target_sum``

        where ``cell_total`` is the sum of expression values in the ``use_data``
        field for that cell.

    chunk_cells
        Number of cell IDs covered by each chunk when processing by
        ``atlas_cell_id`` range. The default value is ``500_000``.

        A larger value usually reduces the number of SQL loop iterations and
        improves runtime, but increases memory usage during aggregation and
        writing for a single chunk. A smaller value is more stable, but may run
        longer.

    add_data
        Name of the normalized result field to write to the derived expression
        table. The default value is ``"data_normalize"``.

    use_data
        Name of the expression value field read from the resolved expression source.
        The default value is ``"data_count"``. Common values include
        ``"data_count"``, ``"data_normalize"``, ``"data_log1p"``, and
        ``"data_scale"``.

    Returns
    -------
    None
        Results are written to a derived expression table. No object is
        returned.

    Notes
    -----
    This function only writes normalized values for explicitly stored nonzero
    expression records in the resolved expression source. For cells whose total expression is 0,
    no new expression records are generated.

    After completion, the function cleans temporary tables and executes a
    checkpoint to reduce the risk that later steps read an intermediate state or
    that too much DuckDB temporary space remains occupied.

    Examples
    --------
    Normalize raw counts to 10,000 per cell::

        sap.pp.normalize_total(atlas, target_sum=10_000)

    """

    start_time = datetime.now()

    conn = atlas.connection

    # 0. Set threads
    try:
        conn.execute(f"PRAGMA threads={os.cpu_count() or 1}")
    except Exception:
        pass

    # 1. Field check
    _require_expression_data(conn, use_data)
    source_from_sql, source_value = _expression_source_for_transform(conn, use_data)

    # 2. Write normalized values into a complete derived expression table.  The
    # extra cell/gene ids increase storage, but downstream steps can read the
    # derived table directly instead of joining back to X_HyS_data by id.
    # Process cells in chunks: a single full-table GROUP BY + JOIN can create
    # very large DuckDB temporary files on atlas-scale data.
    result_table = _derived_data_table_name(add_data)
    conn.execute(f""" DROP TABLE IF EXISTS {result_table} """)
    conn.execute(f"""
        ALTER TABLE X_HyS_data
        DROP COLUMN IF EXISTS {add_data}
    """)
    conn.execute(f"""
        CREATE TABLE {result_table} (
            id BIGINT,
            atlas_cell_id INTEGER,
            atlas_gene_id INTEGER,
            {add_data} REAL
        )
    """)

    min_cell, max_cell = conn.execute("""
        SELECT MIN(atlas_cell_id), MAX(atlas_cell_id)
        FROM X_HyS_data
        WHERE atlas_cell_id IS NOT NULL
    """).fetchone()

    if min_cell is None or max_cell is None:
        logger.warning("X_HyS_data is empty; normalize_total wrote an empty table")
        return

    n_chunks = math.ceil((max_cell - min_cell + 1) / chunk_cells)
    pbar = progress(
        range(n_chunks),
        total=n_chunks,
        desc="normalize_total",
        unit="chunk",
    )

    for i in pbar:
        c_start = min_cell + i * chunk_cells
        c_end = min(c_start + chunk_cells - 1, max_cell)

        conn.execute("DROP TABLE IF EXISTS _cell_sum_chunk")
        conn.execute(f"""
            CREATE TEMP TABLE _cell_sum_chunk AS
            SELECT
                x.atlas_cell_id,
                SUM({source_value}) AS total
            FROM {source_from_sql}
            WHERE x.atlas_cell_id BETWEEN {c_start} AND {c_end}
              AND {source_value} IS NOT NULL
            GROUP BY x.atlas_cell_id
            HAVING total > 0
        """)

        conn.execute(f"""
            INSERT INTO {result_table}
            SELECT
                x.id,
                x.atlas_cell_id,
                x.atlas_gene_id,
                CAST({source_value} * {float(target_sum)} / s.total AS REAL) AS {add_data}
            FROM {source_from_sql}
            JOIN _cell_sum_chunk AS s
              ON x.atlas_cell_id = s.atlas_cell_id
            WHERE x.atlas_cell_id BETWEEN {c_start} AND {c_end}
              AND {source_value} IS NOT NULL
        """)

    n_rows = conn.execute(f"SELECT COUNT(*) FROM {result_table}").fetchone()[0]
    logger.debug(f"normalize_total table {result_table} written, rows={n_rows:,}")

    # Memory cleanup
    _cleanup_transform_after_step(
        conn,
        temp_tables=["_cell_sum_chunk"],
        checkpoint=True,
        collect=True,
    )

    logger.info(f"normalize_total Done, elapsed time: {(datetime.now() - start_time).total_seconds():.2f} seconds")


def normalize_total_scale_factor(
    atlas: Atlas,
    target_sum: float = 10_000,
    add_obs_col: str = "scale_factor",
    use_data: str = "data_count",
    chunk_cells: int = 500_000,
) -> None:
    """Compute the normalization scale factor for each cell.

    This function precomputes the normalization scale factor for each cell in
    the ``obs`` table. For each cell, it calculates the total expression of the
    resolved ``use_data`` expression source, then computes:

    ``scale_factor = target_sum / cell_total``

    The result is written to the ``add_obs_col`` field in the ``obs`` table.

    This function itself does not modify the expression matrix and does not add
    any expression table. It is mainly used together with
    ``normalize_and_log1p`` so that later steps can write normalized-log values
    directly to a derived expression table, avoiding the need to first write a
    complete intermediate normalized matrix.

    The function processes data in chunks over the ``atlas_cell_id`` range. Each
    chunk only computes expression totals within the current cell range and
    writes the scale factor back to the ``obs`` table, making it suitable for
    larger datasets.

    Parameters
    ----------
    atlas
        Atlas object. The object must already be connected to a DuckDB database,
        and the database must contain at least the ``obs`` table and the
        ``X_HyS_data`` table.

        The ``obs`` table must contain the ``atlas_cell_id`` field; the database
        must contain the expression field specified by ``use_data`` either on
        ``X_HyS_data`` or in the corresponding derived expression table.

    target_sum
        Target total expression per cell after normalization. The default value
        is ``10_000``. A larger value makes the overall scale of later normalized
        expression values larger.

    add_obs_col
        Name of the scale factor field to write to the ``obs`` table. The
        default value is ``"scale_factor"``.

        If the column does not exist, the function adds it automatically. If it
        already exists, the function first resets the entire column to ``0`` and
        then writes the current calculation result.

    use_data
        Name of the expression value field read from the resolved expression source.
        The default value is ``"data_count"``. Common values include
        ``"data_count"``, ``"data_normalize"``, ``"data_log1p"``, and
        ``"data_scale"``.

    chunk_cells
        Number of cell IDs covered by each chunk when processing by
        ``atlas_cell_id`` range. The default value is ``500_000``.

    Returns
    -------
    None
        Results are written directly to the ``add_obs_col`` field in the
        ``obs`` table. No object is returned.

    Notes
    -----
    For cells whose total expression in the current ``use_data`` field is 0, the
    scale factor is written as ``0`` to avoid division by zero in later
    normalization.

    This function only writes cell-level metadata and does not change expression
    values.

    Examples
    --------
    Compute the default scale factor::

        sap.pp.normalize_total_scale_factor(atlas, target_sum=10_000)
    """

    start_time = datetime.now()

    conn = atlas.connection

    try:
        conn.execute(f"PRAGMA threads={os.cpu_count() or 1}")
    except Exception:
        pass

    # 0. Basic safety check
    _require_expression_data(conn, use_data)
    source_from_sql, source_value = _expression_source_for_transform(conn, use_data)

    # 1. Add the scale_factor field to obs
    conn.execute(f"""
        ALTER TABLE obs
        ADD COLUMN IF NOT EXISTS {add_obs_col} REAL
    """)

    # Initialize first to avoid keeping old values for empty cells or unmatched cells
    conn.execute(f"""
        UPDATE obs
        SET {add_obs_col} = 0
    """)

    # 2. Get the cell_id range
    min_cell, max_cell = conn.execute("""
        SELECT MIN(atlas_cell_id), MAX(atlas_cell_id)
        FROM obs
    """).fetchone()

    if min_cell is None or max_cell is None:
        logger.info("obs is empty; skipping")
        return

    n_chunks = math.ceil((max_cell - min_cell + 1) / chunk_cells)

    # 3. Compute total in chunks + write back to obs
    pbar = progress(
        range(n_chunks),
        total=n_chunks,
        desc="normalize_total_scale_factor",
        unit="chunk",
    )

    for i in pbar:

        c_start = min_cell + i * chunk_cells
        c_end = min(c_start + chunk_cells - 1, max_cell)

        # Only compute the cell sum for the current chunk
        conn.execute("DROP TABLE IF EXISTS _cell_sum_chunk")

        conn.execute(f"""
            CREATE TEMP TABLE _cell_sum_chunk AS
            SELECT
                x.atlas_cell_id,
                SUM({source_value}) AS total
            FROM {source_from_sql}
            WHERE x.atlas_cell_id BETWEEN {c_start} AND {c_end}
              AND {source_value} IS NOT NULL
            GROUP BY x.atlas_cell_id
        """)

        # Only update obs records corresponding to the current chunk
        conn.execute(f"""
            UPDATE obs
            SET {add_obs_col} =
                CASE
                    WHEN s.total > 0
                    THEN {float(target_sum)} / s.total
                    ELSE 0
                END
            FROM _cell_sum_chunk AS s
            WHERE obs.atlas_cell_id = s.atlas_cell_id
        """)

        # Clean up immediately after each chunk
        conn.execute("DROP TABLE IF EXISTS _cell_sum_chunk")

    # Memory cleanup
    _cleanup_transform_after_step(
        conn,
        temp_tables=["_cell_sum_chunk"],
        checkpoint=False,
        collect=True,
    )

    logger.info(f"normalize_total_scale_factor Done, elapsed time: {(datetime.now() - start_time).total_seconds():.2f} seconds")


@duckdb_memory_limit("5G")
def log1p(
    atlas: 'Atlas',
    base: Optional[Number] = None,
    add_data: str = "data_log1p",
    use_data: str = "data_normalize",
) -> None:
    """Apply a log1p transformation to the expression matrix.

    This function applies a log1p transformation to a specified expression field
    in the expression matrix and writes the result to a derived expression table.
    By default, it reads the ``data_normalize`` field, computes the natural
    logarithm ``ln(1 + x)``, and writes the result to the ``data_log1p`` field.

    This workflow is usually used after total-count normalization to compress the
    dynamic range of expression values and reduce the influence of highly
    expressed genes on downstream PCA or clustering.

    The function materializes the result with a single ``CREATE TABLE AS``
    statement, without a global sort.

    Parameters
    ----------
    atlas
        Atlas object. The object must already be connected to a DuckDB database,
        and the database must contain the ``X_HyS_data`` table.

    base
        Base of the logarithm. The default value is ``None``, meaning that the
        natural logarithm is used:

        ``ln(1 + x)``

        If a numeric value is provided, for example ``base=2``, the function
        computes:

        ``log_base(1 + x)``

    add_data
        Name of the log1p result field to write to the derived expression table.
        The default value is ``"data_log1p"``.

        If the field already exists, the function first drops the old field,
        then recreates and writes it.

    use_data
        Name of the expression value field read from the resolved expression source.
        The default value is ``"data_normalize"``. Common values include
        ``"data_count"``, ``"data_normalize"``, ``"data_log1p"``, and
        ``"data_scale"``.

    Returns
    -------
    None
        Results are written to a derived expression table. No object is
        returned.

    Notes
    -----
    This function does not change the original ``use_data`` field. It only adds
    or rebuilds the ``add_data`` field. For records where ``use_data`` is
    ``NULL``, ``add_data`` remains ``NULL``.

    Examples
    --------
    Apply a natural logarithm transformation to the normalized matrix::

        sap.pp.normalize_total(atlas)
        sap.pp.log1p(atlas)

    Use base 2 and write to a custom field::

        sap.pp.log1p(
            atlas,
            use_data="data_normalize",
            add_data="data_log1p_base2",
            base=2,
        )"""

    start_time = datetime.now()

    conn = atlas.connection

    conn.execute(f"PRAGMA threads = 10 ")

    # 0. Field existence check (important)
    _require_expression_data(conn, use_data)

    # 1. Store log1p values in a complete derived expression table. This
    # intentionally uses a single CTAS without ORDER BY, matching the fast
    # new-table benchmark pattern and avoiding an expensive global sort.
    result_table = _derived_data_table_name(add_data)
    conn.execute(f""" DROP TABLE IF EXISTS {result_table} """)
    conn.execute(f"""
        ALTER TABLE X_HyS_data
        DROP COLUMN IF EXISTS {add_data}
    """)

    from_sql, source_value = _expression_source_for_transform(conn, use_data)

    if base is None:
        log_expr = f"ln(1.0 + {source_value})"
    else:
        log_expr = f"log({float(base)}, 1.0 + {source_value})"

    conn.execute(f"""
        CREATE TABLE {result_table} AS
        SELECT
            x.id,
            x.atlas_cell_id,
            x.atlas_gene_id,
            CAST({log_expr} AS REAL) AS {add_data}
        FROM {from_sql}
        WHERE {source_value} IS NOT NULL
    """)

    n_rows = conn.execute(f"SELECT COUNT(*) FROM {result_table}").fetchone()[0]
    logger.debug(f"log1p table {result_table} written, rows={n_rows:,}")

    # Memory cleanup
    _cleanup_transform_after_step(
        conn,
        temp_tables=[],
        checkpoint=True,
        collect=True,
    )

    logger.info(f"log1p Done, elapsed time: {(datetime.now() - start_time).total_seconds():.2f} seconds")


def normalize_and_log1p(
    atlas: Atlas,
    target_sum: Optional[float] = 10_000,
    use_obs_col: str = "scale_factor",
    add_data: str = "data_log1p",
    use_data: str = "data_count",
    base: Optional[Number] = None,
) -> None:
    """Complete total-count normalization and log1p transformation in one workflow.

    This function combines total-count normalization and log1p transformation
    into a single workflow in the Atlas database. It first calls
    ``normalize_total_scale_factor`` to compute each cell's scale factor in the
    ``obs`` table. It then writes a derived expression table and directly computes:

    ``log(1 + x * scale_factor)``

    The result is written to the derived expression table for ``add_data``.

    Compared with running ``normalize_total`` first and then running ``log1p``,
    this function does not need to first write a complete intermediate
    normalized field. It is commonly used to generate ``data_log1p`` directly
    from ``data_count``.

    Parameters
    ----------
    atlas
        Atlas object. The object must already be connected to a DuckDB database,
        and the database must contain at least the ``obs`` table and the
        ``X_HyS_data`` table.

    target_sum
        Target total expression per cell after normalization. The default value
        is ``10_000``.

        This value is passed to ``normalize_total_scale_factor`` to compute the
        ``scale_factor`` for each cell.

    use_obs_col
        Name of the field in the ``obs`` table that stores the scale factor. The
        default value is ``"scale_factor"``.

        The function first writes each cell's scale factor into this field, then
        reads this field when updating the expression matrix in chunks.

    add_data
        Name of the normalized-and-log1p expression field to write to the
        derived expression table. The default value is ``"data_log1p"``.

        If the field already exists, the function first drops the old field and
        then recreates it.

    use_data
        Name of the raw expression field read from the resolved expression source. The
        default value is ``"data_count"``.

    base
        Base of the logarithm. The default value is ``None``, meaning that the
        natural logarithm e is used. If a numeric value is provided, for example
        ``base=2``, the corresponding-base log1p is computed.

    Returns
    -------
    None
        Results are written to the derived expression table for ``add_data``,
        and the corresponding scale factor is written to the ``use_obs_col``
        field in the ``obs`` table. No object is returned.

    Notes
    -----
    This function overwrites the old values in the ``use_obs_col`` field of the
    ``obs`` table and rebuilds the derived expression table for ``add_data``.

    Examples
    --------
    Compute the scale factor first, then write the log1p matrix::

        sap.pp.normalize_total_scale_factor(atlas, target_sum=10_000)
        sap.pp.normalize_and_log1p(atlas)
    """

    start_time = datetime.now()

    conn = atlas.connection
    try:
        conn.execute(f"PRAGMA threads={os.cpu_count()}")
    except:
        pass

    # 0. Field existence check (prevents silent bugs)
    _require_expression_data(conn, use_data)

    # 1. Call the function above: normalize_total -> compute scale_factor
    normalize_total_scale_factor(
        atlas=atlas,
        target_sum=target_sum,
        add_obs_col=use_obs_col,
        use_data=use_data,
    )

    # 2. Construct the log expression
    source_from_sql, source_value = _expression_source_for_transform(conn, use_data)

    if base is None:
        log_expr = f"ln(1.0 + {source_value} * o.{use_obs_col})"
    else:
        log_expr = f"log({float(base)}, 1.0 + {source_value} * o.{use_obs_col})"

    # 3. Store normalized-log values in an id,value derived table.
    result_table = _derived_data_table_name(add_data)
    conn.execute(f""" DROP TABLE IF EXISTS {result_table} """)
    conn.execute(f"""
        ALTER TABLE X_HyS_data
        DROP COLUMN IF EXISTS {add_data}
    """)

    conn.execute(f"""
        CREATE TABLE {result_table} AS
        SELECT
            x.id,
            x.atlas_cell_id,
            x.atlas_gene_id,
            CAST({log_expr} AS REAL) AS {add_data}
        FROM {source_from_sql}
        JOIN obs AS o
          ON x.atlas_cell_id = o.atlas_cell_id
        WHERE {source_value} IS NOT NULL
    """)

    # Memory cleanup
    _cleanup_transform_after_step(
        conn,
        temp_tables=["_cell_sum_chunk"],
        checkpoint=True,
        collect=True,
    )

    logger.info(f"normalize_and_log1p Done, elapsed time: {(datetime.now() - start_time).total_seconds():.2f} seconds")


def highly_variable_genes(
    atlas: Atlas,
    flavor: Literal["seurat", "cv", "var"] = "seurat",
    n_top_genes: int = 2000,
    add_var_col: str = "highly_variable_genes",
    use_data: str = "data_log1p",
    n_bins: int = 20,
    min_mean: float = 0.0125,
    max_mean: float = 3.0,
    min_disp: float = 0.5,
    max_disp: float = float("inf"),
    use_filtered: bool = True,
    obs_filter_col: str = "filter_cells",
    var_filter_col: str = "filter_genes",
    inplace: bool = True,
) -> None:
    """Identify highly variable genes and write them to the var table.

    This function identifies highly variable genes from the expression matrix in
    the Atlas database and writes the result to the ``var`` table. Highly
    variable genes are usually used in downstream workflows such as PCA,
    neighbor graph construction, clustering, and UMAP, and can reduce the impact
    of noisy and low-information genes on dimensionality-reduction results.

    The function supports three calculation flavors:

    - ``"seurat"``: a bin-normalized dispersion method using mean-binned dispersion standardization;
    - ``"cv"``: ranks genes by the coefficient of variation ``std / mean``;
    - ``"var"``: ranks genes by variance.

    The default is ``flavor="seurat"``. After calculation, the function writes a
    boolean column specified by ``add_var_col`` to the ``var`` table to mark the
    selected highly variable genes. Different flavors also write corresponding
    statistical fields, such as mean, variance, dispersion, normalized
    dispersion, or rank.

    Parameters
    ----------
    atlas
        Atlas object. The object must already be connected to a DuckDB database,
        and the database must contain at least the ``obs``, ``var``, and
        ``X_HyS_data`` tables.

    flavor
        Highly variable gene calculation method. Optional values are
        ``"seurat"``, ``"cv"``, and ``"var"``.

        ``"seurat"`` calls the Seurat-style mean-binning and normalized
        dispersion workflow; ``"cv"`` and ``"var"`` call the basic statistics
        workflow.

    n_top_genes
        Number of genes to mark as highly variable. The default value is
        ``2000``.

        When ``flavor="seurat"`` and ``n_top_genes`` is not ``None``, the first
        ``n_top_genes`` genes with the highest normalized dispersion are
        selected preferentially.

    add_var_col
        Name of the highly variable gene boolean marker column written to the
        ``var`` table. The default value is ``"highly_variable_genes"``.

    use_data
        Name of the expression field read from the resolved expression source. The
        default value is ``"data_log1p"``. Highly variable genes are usually
        recommended to be calculated from log1p-transformed expression values.

    n_bins
        Number of bins by mean expression when ``flavor="seurat"``. The default
        value is ``20``.

    min_mean
        Lower mean-expression bound in cutoff mode when ``flavor="seurat"``.

    max_mean
        Upper mean-expression bound in cutoff mode when ``flavor="seurat"``.

    min_disp
        Lower normalized-dispersion bound in cutoff mode when
        ``flavor="seurat"``.

    max_disp
        Upper normalized-dispersion bound in cutoff mode when
        ``flavor="seurat"``.

    use_filtered
        Whether to calculate only on filtered cells and genes. The default value
        is ``True``.

        When ``True``, the function preferentially uses the boolean columns
        specified by ``obs_filter_col`` and ``var_filter_col``. If the
        corresponding columns do not exist, it falls back to all cells or all
        genes.

    obs_filter_col
        Name of the boolean column in the ``obs`` table used to filter cells.
        The default value is ``"filter_cells"``.

    var_filter_col
        Name of the boolean column in the ``var`` table used to filter genes.
        The default value is ``"filter_genes"``.

    inplace
        Whether to write the result back to the ``var`` table. The default value
        is ``True``.

    Returns
    -------
    None
        Results are written directly to the highly variable gene marker column
        and related statistic columns in the ``var`` table. No object is
        returned.

    Notes
    -----
    This function does not automatically rebuild the minibatch read index. If
    later PCA or KMeans should use only the newly marked highly variable genes,
    call the following after running this function:

    ``atlas.build_read_index(use_hvg=True, gene_condition=...)``

    Examples
    --------
    Select 2000 highly variable genes using the default Seurat flavor::

        sap.pp.highly_variable_genes(atlas, n_top_genes=2000)

    Select 3000 highly variable genes on filtered cells and genes::

        sap.pp.filter_cells(atlas, min_genes=200)
        sap.pp.filter_genes(atlas, min_cells=3)
        sap.pp.highly_variable_genes(
            atlas,
            n_top_genes=3000,
            use_filtered=True,
            obs_filter_col="filter_cells",
            var_filter_col="filter_genes",
        )
    """

    start_time = datetime.now()

    if flavor in ["cv", "var"]:
         _highly_variable_genes_basic(
            atlas=atlas,
            flavor=flavor,
            n_top_genes=n_top_genes,
            add_var_col=add_var_col,
            use_data=use_data,
        )

    elif flavor == "seurat":
        _highly_variable_genes_seurat(
            atlas=atlas,
            n_top_genes=n_top_genes,
            add_var_col=add_var_col,
            use_data=use_data,
            n_bins=n_bins,
            min_mean=min_mean,
            max_mean=max_mean,
            min_disp=min_disp,
            max_disp=max_disp,
            use_filtered=use_filtered,
            obs_filter_col=obs_filter_col,
            var_filter_col=var_filter_col,
            inplace=inplace,
        )

    else:
        raise ValueError(
            f"Unsupported flavor: {flavor}. "
            "Optional values are: 'seurat', 'cv', 'var'"
        )

    logger.info(f"highly_variable_genes Done, elapsed time: {(datetime.now() - start_time).total_seconds():.2f} seconds")
    return None


def _highly_variable_genes_basic(
    atlas: Atlas,
    flavor: Literal["var", "cv"] = "cv",
    n_top_genes: int = 2000,
    add_var_col: str = "highly_variable_genes",
    use_data: str = "data_log1p"
) -> None:
    """Identify highly variable genes using basic statistics.

    This internal function supports ``highly_variable_genes(flavor="cv")`` and
    ``highly_variable_genes(flavor="var")``. It aggregates expression values in
    the ``X_HyS_data`` table by gene in the Atlas database, computes each gene's
    mean, variance, standard deviation, number of nonzero expression records,
    and ranking score across all cells, and then selects highly variable genes
    according to ``n_top_genes``.

    Unlike methods that only count nonzero expression values, this function also
    includes the 0 values not explicitly stored in the sparse table in the
    all-cell statistics. Specifically, it uses the total number of cells in the
    ``obs`` table as the denominator and derives the all-cell mean and variance
    of each gene from ``SUM(x)`` and ``SUM(x * x)``. The resulting statistics are
    closer to those computed on a complete dense expression matrix.

    After calculation, the function writes the statistics back to the ``var``
    table so that later visualization, PCA, ``scale``, or ``build_read_index``
    steps can reuse them.

    Parameters
    ----------
    atlas
        Atlas object. The object must already be connected to a DuckDB database,
        and the database must contain at least the ``obs``, ``var``, and
        ``X_HyS_data`` tables.

        The ``obs`` table is used to count the total number of cells; the
        ``var`` table must contain the ``atlas_gene_id`` field; the
        ``X_HyS_data`` table must contain ``atlas_gene_id`` and the expression
        field specified by ``use_data``.

    flavor
        Basic highly variable gene selection method. Optional values are
        ``"cv"`` and ``"var"``.

        ``"cv"`` uses the coefficient of variation ``std / mean`` as the
        ranking score; ``"var"`` uses variance as the ranking score.

    n_top_genes
        Number of genes to mark as highly variable. The default value is
        ``2000``.

        When this is ``None``, the function does not truncate to the top N
        genes, and instead writes all participating genes to the temporary
        highly variable gene set.

    add_var_col
        Name of the highly variable gene boolean marker column written to the
        ``var`` table. The default value is ``"highly_variable_genes"``.

    use_data
        Name of the expression field read from the resolved expression source. The
        default value is ``"data_log1p"``.

    Returns
    -------
    None
        Results are written directly to the ``var`` table. No object is
        returned.

    Notes
    -----
    This function adds or updates the following fields in the ``var`` table:

    - ``hvg_mean``: mean of each gene across all cells;
    - ``hvg_var``: variance of each gene across all cells;
    - ``hvg_std``: standard deviation of each gene across all cells;
    - ``hvg_score``: ranking score calculated according to ``flavor``;
    - ``hvg_nnz``: number of explicitly stored nonzero records for each gene in
      ``X_HyS_data``;
    - ``add_var_col``: boolean marker indicating whether the gene was selected
      as highly variable.

    This basic method does not read ``filter_cells`` or ``filter_genes``. If you
    need to use the Seurat-style workflow only on filtered cells and genes, call
    ``highly_variable_genes(flavor="seurat", use_filtered=True)``.

    Examples
    --------
    Select 2000 highly variable genes using the coefficient of variation::

        _highly_variable_genes_basic(atlas, flavor="cv", n_top_genes=2000)

    """

    conn = atlas.connection
    try:
        conn.execute(f"PRAGMA threads={os.cpu_count()}")
    except:
        pass

    # Check whether the field exists
    _require_expression_data(conn, use_data)
    source_from_sql, source_value = _expression_source_for_transform(conn, use_data)

    # Ensure the var table has reusable statistic columns
    conn.execute("""
        ALTER TABLE var
        ADD COLUMN IF NOT EXISTS hvg_mean REAL
    """)
    conn.execute("""
        ALTER TABLE var
        ADD COLUMN IF NOT EXISTS hvg_var REAL
    """)
    conn.execute("""
        ALTER TABLE var
        ADD COLUMN IF NOT EXISTS hvg_std REAL
    """)
    conn.execute("""
        ALTER TABLE var
        ADD COLUMN IF NOT EXISTS hvg_score REAL
    """)
    conn.execute("""
        ALTER TABLE var
        ADD COLUMN IF NOT EXISTS hvg_nnz BIGINT
    """)

    # Get the total number of cells N (the key for all-cell statistics)
    n_cells = conn.execute("""
        SELECT COUNT(*) FROM obs
    """).fetchone()[0]

    if n_cells == 0:
        raise ValueError("obs is empty; unable to calculate highly_variable_genes")

    # Compute all-cell mean / var / std for each gene; do not explicitly fill 0s, derive directly from sum / sumsq / N_cells

    conn.execute(f"""
        CREATE OR REPLACE TEMP TABLE _gene_stats AS
        WITH gene_sum AS (
            SELECT
                x.atlas_gene_id,
                COUNT(*) AS nnz,
                SUM({source_value}) AS sum_x,
                SUM(({source_value}) * ({source_value})) AS sum_x2
            FROM {source_from_sql}
            WHERE {source_value} IS NOT NULL
            GROUP BY x.atlas_gene_id
        )
        SELECT
            v.atlas_gene_id,
            COALESCE(g.nnz, 0) AS nnz,
            COALESCE(g.sum_x, 0.0) / {n_cells} AS mean,
            GREATEST(
                COALESCE(g.sum_x2, 0.0) / {n_cells}
                - POWER(COALESCE(g.sum_x, 0.0) / {n_cells}, 2),
                0.0
            ) AS var,
            SQRT(
                GREATEST(
                    COALESCE(g.sum_x2, 0.0) / {n_cells}
                    - POWER(COALESCE(g.sum_x, 0.0) / {n_cells}, 2),
                    0.0
                )
            ) AS std
        FROM var v
        LEFT JOIN gene_sum g
          ON v.atlas_gene_id = g.atlas_gene_id
    """)

    # Compute the ranking metric
    if flavor == "var":
        score_expr = "var"
    elif flavor == "cv":
        # CV = std / mean (avoid division by zero)
        score_expr = "CASE WHEN mean > 0 THEN std / mean ELSE 0 END"
    else:
        raise ValueError(f"Unsupported flavor: {flavor}")

    conn.execute(f"""
        CREATE OR REPLACE TEMP TABLE _gene_score AS
        SELECT
            atlas_gene_id,
            {score_expr} AS score
        FROM _gene_stats
    """)

    # Write statistics and score back to var for direct reuse in later plotting

    conn.execute("""
        UPDATE var
        SET
            hvg_mean = NULL,
            hvg_var = NULL,
            hvg_std = NULL,
            hvg_score = NULL,
            hvg_nnz = NULL
    """)

    conn.execute("""
        UPDATE var v
        SET
            hvg_mean = s.mean,
            hvg_var  = s.var,
            hvg_std  = s.std,
            hvg_nnz  = s.nnz
        FROM _gene_stats s
        WHERE v.atlas_gene_id = s.atlas_gene_id
    """)

    conn.execute("""
        UPDATE var v
        SET
            hvg_score = gs.score
        FROM _gene_score gs
        WHERE v.atlas_gene_id = gs.atlas_gene_id
    """)

    # Select top genes
    if n_top_genes is not None:

        conn.execute(f"""
            CREATE OR REPLACE TEMP TABLE _hvg AS
            SELECT atlas_gene_id
            FROM _gene_score
            ORDER BY score DESC
            LIMIT {int(n_top_genes)}
        """)
    else:

        conn.execute("""
            CREATE OR REPLACE TEMP TABLE _hvg AS
            SELECT atlas_gene_id
            FROM _gene_score
        """)

    # Write boolean results to the var table

    conn.execute(f"""
        ALTER TABLE var
        ADD COLUMN IF NOT EXISTS {add_var_col} BOOLEAN
    """)

    conn.execute(f"""
        UPDATE var
        SET {add_var_col} = FALSE
    """)

    conn.execute(f"""
        UPDATE var
        SET {add_var_col} = TRUE
        FROM _hvg
        WHERE var.atlas_gene_id = _hvg.atlas_gene_id
    """)

    # Memory cleanup
    _cleanup_transform_after_step(
        conn,
        temp_tables=["_gene_stats", "_gene_score", "_hvg"],
        checkpoint=False,
        collect=True,
    )


def _highly_variable_genes_seurat(
    atlas: Atlas,
    n_top_genes: int = 2000,
    add_var_col: str = "highly_variable_genes",
    use_data: str = "data_log1p",
    n_bins: int = 20,
    min_mean: float = 0.0125,
    max_mean: float = 3.0,
    min_disp: float = 0.5,
    max_disp: float = float("inf"),
    use_filtered: bool = True,
    obs_filter_col: str = "filter_cells",
    var_filter_col: str = "filter_genes",
    inplace: bool = True,
):
    """Identify highly variable genes using a Seurat-style method.

    This internal function supports ``highly_variable_genes(flavor="seurat")``.
    It first computes the mean and dispersion for each gene, then bins genes by
    mean expression, standardizes dispersion within each bin, and finally selects
    highly variable genes based on normalized dispersion.

    By default, the function assumes that ``use_data`` is a log1p-transformed
    expression field. Therefore, before computing Seurat-style means and
    dispersions, it applies ``EXP(use_data) - 1.0`` to explicitly stored
    expression values to approximately restore the original expression scale,
    and then includes the 0 values not explicitly stored in the sparse matrix in
    the all-cell statistics.

    The function supports calculation only on filtered cells and genes. When
    ``use_filtered=True``, it preferentially reads the boolean columns specified
    by ``obs_filter_col`` and ``var_filter_col``. If a filter column does not
    exist, the corresponding dimension automatically falls back to using all
    cells or all genes.

    Parameters
    ----------
    atlas
        Atlas object. The object must already be connected to a DuckDB database,
        and the database must contain at least the ``obs``, ``var``, and
        ``X_HyS_data`` tables.

        The ``obs`` table must contain the ``atlas_cell_id`` field; the ``var``
        table must contain ``atlas_gene_id`` and ``atlas_gene_name``; the
        ``X_HyS_data`` table must contain ``atlas_cell_id``, ``atlas_gene_id``,
        and the expression field specified by ``use_data``.

    n_top_genes
        Number of genes to mark as highly variable. The default value is
        ``2000``.

        When this is not ``None``, the function ignores ``min_mean``,
        ``max_mean``, ``min_disp``, and ``max_disp`` and directly selects the
        first ``n_top_genes`` genes with the highest normalized dispersion.

        When this is ``None``, the function enters cutoff mode and selects
        highly variable genes according to threshold ranges for means and
        normalized dispersions.

    add_var_col
        Name of the highly variable gene boolean marker column written to the
        ``var`` table. The default value is ``"highly_variable_genes"``.

    use_data
        Name of the expression field read from the resolved expression source. The
        default value is ``"data_log1p"``.

        This Seurat-style implementation should usually run on log1p-transformed
        data, because internally it uses ``EXP(use_data) - 1.0`` to restore the
        original scale.

    n_bins
        Number of bins by mean expression. The default value is ``20``.

        Binning is used to compare dispersion among genes with similar mean
        expression levels, reducing the influence of mean expression on
        dispersion ranking.

    min_mean
        Lower mean-expression bound for highly variable gene selection in cutoff
        mode.

    max_mean
        Upper mean-expression bound for highly variable gene selection in cutoff
        mode.

    min_disp
        Lower normalized-dispersion bound for highly variable gene selection in
        cutoff mode.

    max_disp
        Upper normalized-dispersion bound for highly variable gene selection in
        cutoff mode.

    use_filtered
        Whether to calculate only on filtered cells and genes. The default value
        is ``True``.

        When filter columns exist, only cells or genes with the corresponding
        column set to ``TRUE`` are kept. When a filter column does not exist, a
        log message is recorded and all objects in that dimension are used.

    obs_filter_col
        Name of the boolean column in the ``obs`` table used to filter cells.
        The default value is ``"filter_cells"``.

    var_filter_col
        Name of the boolean column in the ``var`` table used to filter genes.
        The default value is ``"filter_genes"``.

    inplace
        Whether to write the result back to the ``var`` table. The default value
        is ``True``.

        When ``True``, the function updates the database and returns ``None``.
        When ``False``, the function does not write back to the ``var`` table
        and instead returns a ``pandas.DataFrame`` containing the statistics.

    Returns
    -------
    None or pandas.DataFrame
        When ``inplace=True``, results are written directly to the ``var`` table
        and no object is returned.

        When ``inplace=False``, returns a gene-by-row DataFrame containing fields
        such as ``atlas_gene_id``, ``atlas_gene_name``, ``means``,
        ``dispersions``, ``dispersions_norm``, ``highly_variable_rank``, and
        ``add_var_col``.

    Notes
    -----
    When ``inplace=True``, the function adds or updates the following fields in
    the ``var`` table:

    - ``add_var_col``: boolean marker indicating whether the gene was selected
      as highly variable;
    - ``highly_variable_rank``: rank obtained by sorting normalized dispersion;
    - ``means``: Seurat-style mean based on ``log1p(mean_raw)``;
    - ``dispersions``: dispersion based on ``log(variance / mean)``;
    - ``dispersions_norm``: dispersion normalized after binning by mean.

    Before writing back, old results in the ``var`` table are cleared first to
    avoid stale ``TRUE`` markers remaining when ``use_filtered=True``.

    Examples
    --------
    Select 2000 highly variable genes on filtered cells and genes::

        _highly_variable_genes_seurat(
            atlas,
            n_top_genes=2000,
            use_filtered=True,
        )
    """

    conn = atlas.connection

    if conn is None:
        raise ValueError("atlas.connection is empty; please connect to the database first")

    try:
        conn.execute(f"PRAGMA threads={os.cpu_count()}")
    except Exception:
        pass

    # -------------------------------------------------
    # 0. Safe quoting for DuckDB fields
    # -------------------------------------------------
    def _q(name: str) -> str:
        """Add safe quoting to an SQL identifier.

        This internal helper is used to quote dynamic column names or table names
        when constructing DuckDB SQL. It first escapes existing double quotes in
        the name and then wraps the result in outer double quotes, preventing SQL
        parsing errors when the field name contains special characters.

        Parameters
        ----------
        name
            Column name, table name, or another SQL identifier that needs to be
            quoted.

        Returns
        -------
        quoted_name
            SQL identifier wrapped in double quotes.

        Notes
        -----
        This function only handles SQL identifier quoting. It does not check
        whether a field exists, and it should not be used to quote SQL string
        literals.
        """
        return '"' + name.replace('"', '""') + '"'

    # -------------------------------------------------
    # 1. Check base tables
    # -------------------------------------------------
    for table_name in ["obs", "var", "X_HyS_data"]:
        exists = conn.execute("""
            SELECT COUNT(*)
            FROM information_schema.tables
            WHERE table_name = ?
        """, [table_name]).fetchone()[0]

        if exists == 0:
            raise ValueError(f"Table does not exist in the database: {table_name}")

    # -------------------------------------------------
    # 2. Check the use_data field
    # -------------------------------------------------
    _require_expression_data(conn, use_data)
    source_from_sql, source_value = _expression_source_for_transform(conn, use_data)

    # -------------------------------------------------
    # 3. Check whether filter fields exist
    # -------------------------------------------------
    obs_cols = [
        r[0]
        for r in conn.execute("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'obs'
        """).fetchall()
    ]

    var_cols = [
        r[0]
        for r in conn.execute("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'var'
        """).fetchall()
    ]

    has_obs_filter = obs_filter_col in obs_cols
    has_var_filter = var_filter_col in var_cols

    if use_filtered:
        logger.debug("[rank_genes_groups] use_filtered=True")

        if has_obs_filter:
            logger.debug(f"[rank_genes_groups] using cells where obs.{obs_filter_col}=TRUE")
        else:
            logger.warning(f"{obs_filter_col} does not exist in obs; all cells will be used")

        if has_var_filter:
            logger.debug(f"[rank_genes_groups] using genes where var.{var_filter_col}=TRUE")
        else:
            logger.warning(f"{var_filter_col} does not exist in var; all genes will be used")
    else:
        logger.debug("[rank_genes_groups] use_filtered=False; using all cells / genes")

    # -------------------------------------------------
    # 4. Build temporary keep tables
    # -------------------------------------------------
    conn.execute("DROP TABLE IF EXISTS _hvg_obs_keep")
    conn.execute("DROP TABLE IF EXISTS _hvg_var_keep")

    if use_filtered and has_obs_filter:
        conn.execute(f"""
            CREATE TEMP TABLE _hvg_obs_keep AS
            SELECT atlas_cell_id
            FROM obs
            WHERE {_q(obs_filter_col)} = TRUE
        """)
    else:
        conn.execute("""
            CREATE TEMP TABLE _hvg_obs_keep AS
            SELECT atlas_cell_id
            FROM obs
        """)

    if use_filtered and has_var_filter:
        conn.execute(f"""
            CREATE TEMP TABLE _hvg_var_keep AS
            SELECT atlas_gene_id, atlas_gene_name
            FROM var
            WHERE {_q(var_filter_col)} = TRUE
        """)
    else:
        conn.execute("""
            CREATE TEMP TABLE _hvg_var_keep AS
            SELECT atlas_gene_id, atlas_gene_name
            FROM var
        """)

    n_cells = conn.execute("""
        SELECT COUNT(*)
        FROM _hvg_obs_keep
    """).fetchone()[0]

    n_genes = conn.execute("""
        SELECT COUNT(*)
        FROM _hvg_var_keep
    """).fetchone()[0]

    if n_cells == 0:
        raise ValueError("Number of cells for HVG is 0")

    if n_genes == 0:
        raise ValueError("Number of genes for HVG is 0")

    # -------------------------------------------------
    # 5. SQL aggregation of gene-level sum / sumsq
    #
    # Scanpy flavor='seurat' input is log-normalized data,
    # internally applying expm1(x) first.
    #
    # Therefore here:
    #     x_raw = EXP(use_data) - 1
    #
    # Then compute all-cell statistics including 0s:
    #     sum_x
    #     sum_x2
    # -------------------------------------------------

    conn.execute(f"""
        CREATE OR REPLACE TEMP TABLE _gene_sum AS
        SELECT
            atlas_gene_id,
            COUNT(*) AS nnz,
            SUM(x_raw) AS sum_x,
            SUM(x_raw * x_raw) AS sum_x2
        FROM (
            SELECT
                x.atlas_gene_id,
                EXP({source_value}) - 1.0 AS x_raw
            FROM {source_from_sql}
            JOIN _hvg_obs_keep AS o
              ON x.atlas_cell_id = o.atlas_cell_id
            JOIN _hvg_var_keep AS v
              ON x.atlas_gene_id = v.atlas_gene_id
            WHERE {source_value} IS NOT NULL
        ) AS t
        GROUP BY atlas_gene_id
    """)

    gene_df = conn.execute("""
        SELECT
            v.atlas_gene_id,
            v.atlas_gene_name,
            COALESCE(g.nnz, 0) AS nnz,
            COALESCE(g.sum_x, 0.0) AS sum_x,
            COALESCE(g.sum_x2, 0.0) AS sum_x2
        FROM _hvg_var_keep AS v
        LEFT JOIN _gene_sum AS g
          ON v.atlas_gene_id = g.atlas_gene_id
        ORDER BY v.atlas_gene_id
    """).fetchdf()

    # -------------------------------------------------
    # 6. Compute mean / variance / dispersion on a small Python table
    # -------------------------------------------------

    sum_x = gene_df["sum_x"].to_numpy(dtype=np.float64)
    sum_x2 = gene_df["sum_x2"].to_numpy(dtype=np.float64)

    # mean: all cells including 0s
    mean_raw = sum_x / float(n_cells)

    # variance: sample variance, aligned with Scanpy correction=1 as much as possible
    if n_cells > 1:
        var_raw = (sum_x2 - (sum_x ** 2) / float(n_cells)) / float(n_cells - 1)
    else:
        var_raw = np.zeros_like(mean_raw)

    var_raw = np.maximum(var_raw, 0.0)

    # Scanpy:
    # mean[mean == 0] = 1e-12
    mean_safe = mean_raw.copy()
    mean_safe[mean_safe == 0] = 1e-12

    dispersion = var_raw / mean_safe

    # Scanpy seurat:
    # dispersion[dispersion == 0] = nan
    # dispersion = log(dispersion)
    # mean = log1p(mean)
    dispersion = dispersion.astype(np.float64)
    dispersion[dispersion == 0] = np.nan

    dispersions = np.log(dispersion)
    means = np.log1p(mean_raw)

    gene_df["means"] = means
    gene_df["dispersions"] = dispersions

    # -------------------------------------------------
    # 7. Mean binning: Scanpy seurat uses pd.cut(means, bins=n_bins)
    # -------------------------------------------------

    work = gene_df[[
        "atlas_gene_id",
        "atlas_gene_name",
        "means",
        "dispersions"
    ]].copy()

    # Note:
    # pd.cut is consistent with Scanpy and uses equal-width bins;
    # not qcut.
    try:
        work["mean_bin"] = pd.cut(
            work["means"],
            bins=n_bins
        )
    except ValueError:
        # Extreme case where all means are identical
        work["mean_bin"] = pd.Series(["single_bin"] * len(work), index=work.index)

    # -------------------------------------------------
    # 8. Compute avg/dev within each bin
    #
    # Scanpy seurat:
    #     avg = mean(dispersions)
    #     dev = std(dispersions)
    #
    # Single-gene bin:
    #     dev = avg
    #     avg = 0
    # In this way, normalized dispersion = dispersion / dispersion = 1
    # -------------------------------------------------

    disp_stats = work.groupby("mean_bin", observed=True)["dispersions"].agg(
        avg="mean",
        dev="std",
        count="count",
    )

    # Single-gene bin: simulate Scanpy _postprocess_dispersions_seurat
    one_gene_bins = disp_stats["dev"].isna()

    if one_gene_bins.any():
        disp_stats.loc[one_gene_bins, "dev"] = disp_stats.loc[one_gene_bins, "avg"]
        disp_stats.loc[one_gene_bins, "avg"] = 0.0

    # Prevent dev from being 0
    disp_stats["dev"] = disp_stats["dev"].replace(0, np.nan)

    # Map back to each gene
    avg_map = disp_stats["avg"]
    dev_map = disp_stats["dev"]

    work["_disp_avg"] = work["mean_bin"].map(avg_map).astype(float)
    work["_disp_dev"] = work["mean_bin"].map(dev_map).astype(float)

    work["dispersions_norm"] = (
        (work["dispersions"] - work["_disp_avg"])
        / work["_disp_dev"]
    )

    # -------------------------------------------------
    # 9. Select HVG
    #
    # Scanpy behavior:
    # - If n_top_genes is not None, cutoffs are ignored
    # - Select n_top_genes with the highest normalized dispersion
    # - Ties may cause the count to be slightly larger; here, for engineering controllability, top N is selected strictly
    # -------------------------------------------------

    work["highly_variable_rank"] = np.nan
    work[add_var_col] = False

    if n_top_genes is not None:
        valid_score = work["dispersions_norm"].replace([np.inf, -np.inf], np.nan)

        rank_df = work.loc[valid_score.notna()].copy()

        rank_df = rank_df.sort_values(
            by=["dispersions_norm", "atlas_gene_id"],
            ascending=[False, True],
        ).reset_index(drop=True)

        top_n = min(int(n_top_genes), len(rank_df))

        rank_df["highly_variable_rank"] = np.arange(
            len(rank_df),
            dtype=np.float64,
        )

        top_ids = set(rank_df.loc[:top_n - 1, "atlas_gene_id"].tolist())

        rank_map = dict(zip(
            rank_df["atlas_gene_id"],
            rank_df["highly_variable_rank"],
        ))

        work["highly_variable_rank"] = work["atlas_gene_id"].map(rank_map)
        work[add_var_col] = work["atlas_gene_id"].isin(top_ids)

    else:
        # Scanpy cutoff mode: determine the range after nan_to_num
        score = work["dispersions_norm"].replace([np.inf, -np.inf], np.nan)
        score_for_cutoff = score.fillna(0.0)

        hv_mask = (
            (work["means"] > float(min_mean))
            & (work["means"] < float(max_mean))
            & (score_for_cutoff > float(min_disp))
            & (score_for_cutoff < float(max_disp))
        )

        work[add_var_col] = hv_mask.to_numpy()

        rank_df = work.loc[score.notna()].copy()
        rank_df = rank_df.sort_values(
            by=["dispersions_norm", "atlas_gene_id"],
            ascending=[False, True],
        ).reset_index(drop=True)

        rank_df["highly_variable_rank"] = np.arange(
            len(rank_df),
            dtype=np.float64,
        )

        rank_map = dict(zip(
            rank_df["atlas_gene_id"],
            rank_df["highly_variable_rank"],
        ))

        work["highly_variable_rank"] = work["atlas_gene_id"].map(rank_map)

    # -------------------------------------------------
    # 10. Merge back to gene_df
    # -------------------------------------------------
    gene_df = gene_df.drop(
        columns=[
            c for c in [
                "means",
                "dispersions",
                "dispersions_norm",
                "highly_variable_rank",
                add_var_col,
            ]
            if c in gene_df.columns
        ],
        errors="ignore",
    )

    gene_df = gene_df.merge(
        work[[
            "atlas_gene_id",
            "means",
            "dispersions",
            "dispersions_norm",
            "highly_variable_rank",
            add_var_col,
        ]],
        on="atlas_gene_id",
        how="left",
    )

    gene_df[add_var_col] = gene_df[add_var_col].fillna(False).astype(bool)

    hvg_count = int(gene_df[add_var_col].sum())

    # -------------------------------------------------
    # 11. Write back to var
    # -------------------------------------------------
    if inplace:

        conn.execute(f"""
            ALTER TABLE var
            ADD COLUMN IF NOT EXISTS {_q(add_var_col)} BOOLEAN
        """)

        conn.execute("""
            ALTER TABLE var
            ADD COLUMN IF NOT EXISTS highly_variable_rank REAL
        """)

        conn.execute("""
            ALTER TABLE var
            ADD COLUMN IF NOT EXISTS means REAL
        """)

        conn.execute("""
            ALTER TABLE var
            ADD COLUMN IF NOT EXISTS dispersions REAL
        """)

        conn.execute("""
            ALTER TABLE var
            ADD COLUMN IF NOT EXISTS dispersions_norm REAL
        """)

        write_df = gene_df[[
            "atlas_gene_id",
            add_var_col,
            "highly_variable_rank",
            "means",
            "dispersions",
            "dispersions_norm",
        ]].copy()

        # Clear old results in the full var table first to avoid stale TRUE values when use_filtered=True
        conn.execute(f"""
            UPDATE var
            SET
                {_q(add_var_col)} = FALSE,
                highly_variable_rank = NULL,
                means = NULL,
                dispersions = NULL,
                dispersions_norm = NULL
        """)

        conn.register("_hvg_seurat_py", write_df)

        conn.execute(f"""
            UPDATE var AS v
            SET
                {_q(add_var_col)} = p.{_q(add_var_col)},
                highly_variable_rank = p.highly_variable_rank,
                means = p.means,
                dispersions = p.dispersions,
                dispersions_norm = p.dispersions_norm
            FROM _hvg_seurat_py AS p
            WHERE v.atlas_gene_id = p.atlas_gene_id
        """)

        conn.unregister("_hvg_seurat_py")

    # 12. Unified cleanup of SQL temporary tables / pandas registered tables
    _cleanup_transform_after_step(
        conn,
        temp_tables=["_gene_sum", "_hvg_obs_keep", "_hvg_var_keep"],
        unregister_tables=["_hvg_seurat_py"],
        checkpoint=False,
        collect=False,
    )

    # If inplace=True, gene_df does not need to be returned, so delete large Python objects
    if inplace:
        try:
            del gene_df
        except Exception:
            pass
        try:
            del work
        except Exception:
            pass
        try:
            del rank_df
        except Exception:
            pass
        try:
            del write_df
        except Exception:
            pass
        try:
            del disp_stats
        except Exception:
            pass

    gc.collect()

    if not inplace:
        return gene_df


@duckdb_memory_limit("5G")
def scale(
    atlas: Atlas,
    use_data: str = "data_log1p",
    add_data: str = "data_scale",
    add_var_col: str = "zero_scale_transform",
    max_value: float | None = 10.0,
    use_hvg: bool = True,
    hvg_key: str = "highly_variable_genes",
    mode: Literal["center_and_scale", "center_only", "scale_only"] = "center_and_scale",
) -> None:
    """Center and/or variance-normalize the expression matrix by gene.

    This function transforms the expression matrix by gene in the Atlas database
    and writes the result to the derived expression table for ``add_data``. By
    default, it centers and standardizes the data as z-scores. It can also be
    configured to only center the data or only normalize each gene by its
    standard deviation. For PCA, ``mode="center_only"`` preserves gene-level
    variance and expression-strength differences after centering, while
    ``mode="center_and_scale"`` normalizes gene expression scales so selected
    genes contribute more comparably.

    For each target gene, the function first computes the mean and standard
    deviation across all cells based on ``use_data``. Then it transforms
    explicitly stored expression records according to ``mode``:

    - ``"center_and_scale"``: ``z = (x - mean_gene) / std_gene``;
    - ``"center_only"``: ``z = x - mean_gene``;
    - ``"scale_only"``: ``z = x / std_gene``.

    If ``max_value`` is not ``None``, the result is clipped to the range
    ``[-max_value, max_value]`` for modes that use variance normalization.

    Because 0 values not explicitly stored in the sparse matrix usually no
    longer equal 0 after centering, the function also writes the ``add_var_col``
    field to the ``var`` table. This field records the transformed fill value
    corresponding to the original 0 value for each gene. When minibatch dense
    reading uses ``data_scale``, this field is used to fill sparse zero
    positions.

    Parameters
    ----------
    atlas
        Atlas object. The object must already be connected to a DuckDB database,
        and the database must contain at least the ``obs``, ``var``, and
        ``X_HyS_data`` tables.

    use_data
        Name of the expression field read from the resolved expression source. The
        default value is ``"data_log1p"``.

    add_data
        Name of the scale result field written to a derived expression table. The
        default value is ``"data_scale"``.

    add_var_col
        Name of the zero-value scaled fill-value field written to the ``var``
        table. The default value is ``"zero_scale_transform"``.

    max_value
        Upper clipping bound for scaled expression values. The default value is
        ``10.0``.

        The current implementation uses ``[-max_value, max_value]`` as the
        clipping range.

    use_hvg
        Whether to calculate and write scale results only for the highly
        variable gene set. The default value is ``True``.

        When ``True``, the function only uses genes with ``hvg_key=TRUE`` in the
        ``var`` table. When ``False``, all genes are used.

    hvg_key
        Name of the boolean column in the ``var`` table that marks highly
        variable genes. The default value is ``"highly_variable_genes"``.

    mode
        Transformation mode. The default ``"center_and_scale"`` preserves the
        previous z-score behavior and normalizes gene expression scales. Use
        ``"center_only"`` to subtract gene means without dividing by standard
        deviation when PCA should preserve gene-level variance differences. Use
        ``"scale_only"`` to divide by standard deviation without subtracting
        gene means.

    Returns
    -------
    None
        No object is returned.
        Results are written to the derived expression table for ``add_data``.
        By default, this is ``data_scale``, which means z-score standardization
        is applied to data. Results are also written to the ``add_var_col`` field
        in the ``var`` table. By default, ``zero_scale_transform`` stores each
        gene's transformed zero value for later dense minibatch filling.

    Notes
    -----
    If ``use_hvg=True``, only highly variable genes receive ``data_scale``
    results. Non-target genes do not participate in later HVG read indexes based
    on ``data_scale``.

    After completion, if downstream minibatch, PCA, or KMeans needs to read the
    scaled data, run
    ``atlas.build_read_index(use_hvg=True, use_data=add_data, ...)``.

    Examples
    --------
    Standardize a log1p matrix::

        sap.pp.scale(atlas, use_data="data_log1p", add_data="data_scale")

    Scale only highly variable genes and clip extreme values::

        sap.pp.scale(
            atlas,
            use_hvg=True,
            hvg_key="highly_variable_genes",
            max_value=10,
        )

    Only center each gene without variance normalization::

        sap.pp.scale(atlas, mode="center_only")
        """

    start_time = datetime.now()
    conn = atlas.connection

    if mode not in {"center_and_scale", "center_only", "scale_only"}:
        raise ValueError(
            "mode must be one of 'center_and_scale', 'center_only', or 'scale_only'"
        )

    # 0. Parallelism
    try:
        n_threads = 4
        conn.execute(f"PRAGMA threads={n_threads}")
    except Exception:
        pass

    # 1. Input field checks
    _require_expression_data(conn, use_data)

    if conn.execute("""
        SELECT COUNT(*)
        FROM information_schema.columns
        WHERE table_name = 'X_HyS_data'
          AND column_name = 'id'
    """).fetchone()[0] == 0:
        raise ValueError("The id field does not exist in X_HyS_data; unable to write back by id chunks")

    if conn.execute("""
        SELECT COUNT(*)
        FROM information_schema.columns
        WHERE table_name = 'X_HyS_data'
          AND column_name = 'atlas_gene_id'
    """).fetchone()[0] == 0:
        raise ValueError("The atlas_gene_id field does not exist in X_HyS_data")

    # 2. Prepare output fields
    # Avoid updating the large expression table in place.  The scaled values are
    # materialized into a complete derived expression table.  Because scale is
    # usually restricted to HVGs, storing cell/gene ids here avoids an extra join
    # when build_read_index(use_data="data_scale") is called later.
    scale_table = f"X_HyS_data_{add_data}"
    conn.execute(f""" DROP TABLE IF EXISTS {scale_table} """)
    conn.execute(f""" ALTER TABLE X_HyS_data DROP COLUMN IF EXISTS {add_data} """)

    conn.execute(f""" ALTER TABLE var DROP COLUMN IF EXISTS {add_var_col} """)
    conn.execute(f""" ALTER TABLE var ADD COLUMN IF NOT EXISTS {add_var_col} REAL """)

    # 3. Prepare the target gene set
    conn.execute("DROP TABLE IF EXISTS _target_genes")

    if use_hvg:
        conn.execute(f"""
            CREATE TEMP TABLE _target_genes AS
            SELECT atlas_gene_id
            FROM var
            WHERE {hvg_key} = TRUE
        """)
    else:
        conn.execute("""
            CREATE TEMP TABLE _target_genes AS
            SELECT atlas_gene_id
            FROM var
        """)

    n_genes = conn.execute("""
        SELECT COUNT(*) FROM _target_genes
    """).fetchone()[0]

    if n_genes == 0:
        conn.execute("DROP TABLE IF EXISTS _target_genes")
        return

    # Get the total number of cells
    n_cells = conn.execute("""
        SELECT COUNT(*) FROM obs
    """).fetchone()[0]

    if n_cells == 0:
        conn.execute("DROP TABLE IF EXISTS _target_genes")
        raise ValueError("obs is empty; unable to calculate scale")

    # 4. Compute mean/std for all target genes at once
    t0 = datetime.now()

    conn.execute("DROP TABLE IF EXISTS _gene_stat")

    source_from_sql, source_value = _expression_source_for_transform(conn, use_data)

    if mode == "center_and_scale":
        transformed_expr = f"""
            CASE
                WHEN g.std > 0 THEN ({source_value} - g.mean) / g.std
                ELSE 0
            END
        """
        zero_expr = """
            CASE
                WHEN g.std > 0 THEN (0 - g.mean) / g.std
                ELSE 0
            END
        """
    elif mode == "center_only":
        transformed_expr = f"({source_value} - g.mean)"
        zero_expr = "(0 - g.mean)"
    else:
        transformed_expr = f"""
            CASE
                WHEN g.std > 0 THEN {source_value} / g.std
                ELSE 0
            END
        """
        zero_expr = """
            CASE
                WHEN g.std > 0 THEN 0 / g.std
                ELSE 0
            END
        """

    if max_value is not None and mode != "center_only":
        transformed_expr = f"""
            LEAST(
                {float(max_value)},
                GREATEST(
                    -{float(max_value)},
                    {transformed_expr}
                )
            )
        """
        zero_expr = f"""
            LEAST(
                {float(max_value)},
                GREATEST(
                    -{float(max_value)},
                    {zero_expr}
                )
            )
        """

    conn.execute(f"""
        CREATE TEMP TABLE _gene_stat AS
        SELECT
            x.atlas_gene_id,

            SUM({source_value}) / {n_cells} AS mean,

            SQRT(
                GREATEST(
                    SUM({source_value} * {source_value}) / {n_cells}
                    - POWER(SUM({source_value}) / {n_cells}, 2),
                    0.0
                )
            ) AS std

        FROM {source_from_sql}
        JOIN _target_genes t
          ON x.atlas_gene_id = t.atlas_gene_id
        WHERE {source_value} IS NOT NULL
        GROUP BY x.atlas_gene_id
    """)

    # 5. Update var: record the scaling factor for 0 values -> z-score
    t0 = datetime.now()

    conn.execute("BEGIN")
    try:
        conn.execute(f"""
            UPDATE var v
            SET {add_var_col} = CAST({zero_expr} AS REAL)
            FROM _gene_stat g
            WHERE v.atlas_gene_id = g.atlas_gene_id
        """)

        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise

    # 6. Materialize scaled nonzero values as a complete derived table.
    t0 = datetime.now()
    conn.execute(f"""
        CREATE TABLE {scale_table} AS
        SELECT
            x.id,
            x.atlas_cell_id,
            x.atlas_gene_id,
            CAST({transformed_expr} AS REAL) AS {add_data}
        FROM {source_from_sql}
        JOIN _gene_stat g
          ON x.atlas_gene_id = g.atlas_gene_id
        WHERE {source_value} IS NOT NULL
    """)
    n_scaled_rows = conn.execute(f"SELECT COUNT(*) FROM {scale_table}").fetchone()[0]
    logger.debug(
        f"scale table {scale_table} written, rows={n_scaled_rows:,}, "
        f"elapsed time: {(datetime.now() - t0).total_seconds():.2f} seconds"
    )
    _write_expression_transform_meta(
        conn,
        data_name=add_data,
        source_data=use_data,
        transform=f"scale:{mode}",
        centered=mode in {"center_and_scale", "center_only"},
    )

    # 8. Memory cleanup
    _cleanup_transform_after_step(
        conn,
        temp_tables=["_target_genes", "_gene_stat"],
        checkpoint=True,
        collect=True,
    )

    logger.info(f"scale Done, elapsed time: {(datetime.now() - start_time).total_seconds():.2f} seconds")


def sqrt(
    atlas: "Atlas",
    add_data: str = "data_sqrt",
    use_data: str = "data_count",
) -> None:
    """Apply a square-root transformation to the expression matrix.

    This function applies a square-root transformation to a specified expression
    field from the resolved expression source and writes the result to a complete
    derived expression table. By default, it reads ``data_count``, computes
    ``sqrt(x)``, and writes the result to ``data_sqrt``.

    The square-root transformation can be used as a simple variance-stabilizing
    or pre-visualization preprocessing method for count data. Compared with
    log1p, sqrt compresses low-expression counts more gently.

    The function materializes the result with a single ``CREATE TABLE AS``
    statement, without a global sort.

    Parameters
    ----------
    atlas
        Atlas object. The object must already be connected to a DuckDB database,
        and the database must contain the ``X_HyS_data`` table.

    add_data
        Name of the square-root transformation result field to write to a derived
        expression table. The default value is ``"data_sqrt"``.

    use_data
        Name of the expression field read from the resolved expression source. The
        default value is ``"data_count"``. Common values include
        ``"data_count"``, ``"data_normalize"``, ``"data_log1p"``, and
        ``"data_scale"``.

    Returns
    -------
    None
        Results are written to the derived expression table for ``add_data``. No
        object is returned.

    Notes
    -----
    This function does not modify the original ``use_data`` field. It rebuilds
    the derived expression table for ``add_data``. If ``use_data`` contains
    negative values, DuckDB's ``sqrt`` may produce invalid results or errors;
    therefore, this function should usually be applied to count fields or
    nonnegative normalized expression fields.

    Examples
    --------
    Apply a square-root transformation to raw counts::

        sap.pp.sqrt(atlas)

    Write to a custom derived field::

        sap.pp.sqrt(atlas, add_data="data_sqrt")
    """

    start_time = datetime.now()

    conn = atlas.connection

    try:
        conn.execute(f"PRAGMA threads={os.cpu_count()}")
    except Exception:
        pass

    if add_data is None:
        raise ValueError("add_data must be specified")

    # 0. Field existence check
    _require_expression_data(conn, use_data)
    source_from_sql, source_value = _expression_source_for_transform(conn, use_data)

    # 1. Construct the sqrt expression (0 is not specially handled)
    sqrt_expr = f"sqrt({source_value})"

    # 2. Store square-root values in a complete derived expression table. Keeping
    # cell and gene ids avoids an extra join in downstream expression readers.
    result_table = _derived_data_table_name(add_data)
    conn.execute(f""" DROP TABLE IF EXISTS {result_table} """)
    conn.execute(f"""
        ALTER TABLE X_HyS_data
        DROP COLUMN IF EXISTS {add_data}
    """)

    conn.execute(f"""
        CREATE TABLE {result_table} AS
        SELECT
            x.id,
            x.atlas_cell_id,
            x.atlas_gene_id,
            CAST({sqrt_expr} AS REAL) AS {add_data}
        FROM {source_from_sql}
        WHERE {source_value} IS NOT NULL
    """)

    n_rows = conn.execute(f"SELECT COUNT(*) FROM {result_table}").fetchone()[0]
    logger.debug(f"sqrt table {result_table} written, rows={n_rows:,}")

    # Memory cleanup
    _cleanup_transform_after_step(
        conn,
        temp_tables=[],
        checkpoint=True,
        collect=True,
    )

    logger.info(f"sqrt Done, elapsed time: {(datetime.now() - start_time).total_seconds():.2f} seconds")


def _cleanup_transform_after_step(
    conn: DuckDBPyConnection,
    temp_tables: list[str] | None = None,
    unregister_tables: list[str] | None = None,
    checkpoint: bool = False,
    collect: bool = True,
):
    """Clean up temporary resources generated by the current step.

    This internal function is used to release temporary resources after an
    expression-matrix transformation step. Multiple preprocessing functions may
    create DuckDB temporary tables, register temporary pandas/Arrow relations,
    or execute large-table ``UPDATE``, ``DROP``, or ``RENAME`` operations during
    execution. This function centralizes those cleanup actions so that, after
    each transformation step, temporary usage in both the database connection
    and the Python process is restored to a stable state as much as possible.

    The function only cleans resources. It does not modify official result
    fields in ``X_HyS_data``, ``obs``, or ``var``. It is usually called during
    finalization by internal workflows such as ``normalize_total``, ``log1p``,
    ``scale``, ``sqrt``, and highly variable gene calculation.

    Parameters
    ----------
    conn
        DuckDB database connection. The connection must still be valid and able
        to execute cleanup operations such as ``DROP TABLE``, ``CHECKPOINT``, or
        ``unregister``.

    temp_tables
        List of DuckDB temporary table names to drop. The default value is
        ``None``, which is treated as an empty list.

        The function executes ``DROP TABLE IF EXISTS`` for each name in the
        list. If a table no longer exists, or if dropping it fails, the cleanup
        exception is ignored.

    unregister_tables
        List of pandas/Arrow relation names to unregister from the DuckDB
        connection. The default value is ``None``, which is treated as an empty
        list.

        This parameter is commonly used to clean up temporary DataFrames
        registered in DuckDB through ``conn.register(...)``.

    checkpoint
        Whether to execute DuckDB ``CHECKPOINT`` after cleanup. The default is
        ``False``.

        For steps involving large-table reconstruction, column deletion, table
        renaming, or batch updates, setting this to ``True`` can flush data to
        disk earlier and release part of DuckDB's internal space.

    collect
        Whether to trigger Python garbage collection via ``gc.collect()`` after
        cleanup. The default is ``True``.

    Returns
    -------
    None
        This function only performs resource cleanup and returns no object.

    Notes
    -----
    Exceptions during the cleanup phase are caught and ignored to avoid
    overwriting the main results already produced by earlier preprocessing
    steps when cleanup of temporary objects fails. If you need to inspect
    temporary tables or the DuckDB connection state, manually check the
    temporary objects in the database before calling this function.

    This is an internal helper. Unless you need to extend the internal
    scAtlasPy workflow, it is generally not recommended to call it directly in
    user code.
    """

    if temp_tables is None:
        temp_tables = []

    if unregister_tables is None:
        unregister_tables = []

    # 1. Unregister temporary pandas / Arrow objects
    for t in unregister_tables:
        try:
            conn.unregister(t)
        except Exception:
            logger.debug(
                "Failed to unregister temporary object %s during transformation cleanup",
                t,
                exc_info=True,
            )

    # 2. Drop DuckDB temporary tables
    for t in temp_tables:
        try:
            conn.execute(f"DROP TABLE IF EXISTS {t}")
        except Exception:
            logger.debug(
                "Failed to drop temporary table %s during transformation cleanup",
                t,
                exc_info=True,
            )

    # 3. Checkpoint is recommended after large-table UPDATE / DROP / RENAME operations
    if checkpoint:
        try:
            conn.execute("CHECKPOINT")
        except Exception:
            logger.exception("DuckDB CHECKPOINT failed during transformation cleanup")
            raise

    # 4. Python-level garbage collection
    if collect:
        try:
            gc.collect()
        except Exception:
            logger.debug(
                "Python garbage collection failed during transformation cleanup",
                exc_info=True,
            )
