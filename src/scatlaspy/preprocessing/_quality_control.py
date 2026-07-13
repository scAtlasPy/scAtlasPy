import os
from datetime import datetime
from _duckdb import DuckDBPyConnection
from ..data import Atlas
from typing import Any
from typing import Optional
import logging
import math
import gc
from ..io import progress

logger = logging.getLogger('Atlas') # Get the logger
logger.addHandler(logging.NullHandler())


def filter_cells(
    atlas: Atlas,
    min_counts: Optional[int] = None,
    min_genes: Optional[int] = None,
    max_counts: Optional[int] = None,
    max_genes: Optional[int] = None,
    add_data: str = "filter_cells",
    chunk_cells: int = 500_000,   # Chunk size
) -> None:
    """Filter cells based on expression counts and the number of detected genes.

    This function computes each cell's total expression and number of detected genes from the expression matrix in chunks,
    then writes the cell filtering flag into the ``obs`` table according to the thresholds. The result is saved in the Atlas database.

    Parameters
    ----------
    atlas
        Atlas object. It usually needs to be connected to a DuckDB database and contain the ``obs``, ``var``, expression matrix,
        or result tables required by this function for reading or writing.

    min_counts
        Lower bound of total cell expression. Only cells with ``sum_expr >= min_counts`` pass this condition.
        If ``None``, this lower-bound condition is not used.

    min_genes
        Lower bound of the number of nonzero-expression genes detected in a cell. Only cells with
        ``nonzero_genes >= min_genes`` pass this condition.
        If ``None``, this lower-bound condition is not used.

    max_counts
        Upper bound of total cell expression. Only cells with ``sum_expr <= max_counts`` pass this condition.
        If ``None``, this upper-bound condition is not used.

    max_genes
        Upper bound of the number of nonzero-expression genes detected in a cell. Only cells with
        ``nonzero_genes <= max_genes`` pass this condition.
        If ``None``, this upper-bound condition is not used.

    add_data
        Name of the boolean filtering column written to the ``obs`` table. The default value is ``"filter_cells"``.
        If this column does not exist, the function automatically adds it;
        if this column already exists, the function first resets all values to ``FALSE``, then updates the cells that pass filtering
        to ``TRUE``.

    chunk_cells
        Number of cell IDs covered by each chunk when processing by ``atlas_cell_id`` range.
        A larger value usually runs faster, but increases memory usage during aggregation of a single chunk;
        a smaller value is more stable, but increases the number of SQL loops.

    Returns
    -------
    None
        The result is written directly into the Atlas database.
        In the ``obs`` table of the database, a field specified by ``add_data: str = "filter_cells"`` is added;
        it is ``true`` for cells that meet the filtering conditions and ``false`` otherwise.

    Examples
    --------
    Keep cells with at least 200 detected genes and a total expression no lower than 500::

        sap.pp.filter_cells(atlas, min_genes=200, min_counts=500)

    Set both lower and upper bounds, and write to a custom filtering column::

        sap.pp.filter_cells(
            atlas,
            min_counts=500,
            max_counts=50_000,
            min_genes=200,
            max_genes=6000,
            add_data="filter_cells",
        )
    """

    start_time = datetime.now()
    conn = atlas.connection

    # 0. DuckDB parameters
    try:
        th = os.cpu_count() or 1
        conn.execute(f"PRAGMA threads={th}")

    except Exception:
        pass


    # 1. Add obs filtering field
    conn.execute(f"""
        ALTER TABLE obs
        ADD COLUMN IF NOT EXISTS {add_data} BOOLEAN DEFAULT FALSE
    """)

    total_cells = conn.execute("SELECT COUNT(*) FROM obs").fetchone()[0]

    conn.execute(f"""
        UPDATE obs
        SET {add_data} = FALSE
    """)

    # 2. Build filtering conditions
    conds = []
    if min_counts is not None:
        conds.append(f"sum_expr >= {min_counts}")
    if max_counts is not None:
        conds.append(f"sum_expr <= {max_counts}")
    if min_genes is not None:
        conds.append(f"nonzero_genes >= {min_genes}")
    if max_genes is not None:
        conds.append(f"nonzero_genes <= {max_genes}")

    condition = " AND ".join(conds) if conds else "TRUE"

    # 3. Get the cell_id range
    min_cell, max_cell = conn.execute("""
        SELECT
            MIN(atlas_cell_id),
            MAX(atlas_cell_id)
        FROM obs
    """).fetchone()

    if min_cell is None or max_cell is None:
        logger.info("obs is empty; skipped.")
        return

    n_chunks = (max_cell - min_cell + chunk_cells) // chunk_cells

    keep_total = 0

    # 4. Chunked aggregation + chunked write-back
    pbar = progress(
        range(n_chunks),
        total=n_chunks,
        desc="filter_cells",
        unit="chunk",
    )

    for i in pbar:

        c_start = min_cell + i * chunk_cells
        c_end = min(c_start + chunk_cells - 1, max_cell)

        # Only create a temporary table for the current chunk; do not create the full keep_cells table anymore
        conn.execute("DROP TABLE IF EXISTS keep_cells_chunk")

        conn.execute(f"""
            CREATE TEMP TABLE keep_cells_chunk AS
            SELECT atlas_cell_id
            FROM (
                SELECT
                    atlas_cell_id,
                    SUM(data_count) AS sum_expr,
                    COUNT(*) AS nonzero_genes
                FROM X_HyS_data
                WHERE atlas_cell_id BETWEEN {c_start} AND {c_end}
                GROUP BY atlas_cell_id
            )
            WHERE {condition}
        """)

        # Number of retained cells in the current chunk
        keep_now = conn.execute("""
            SELECT COUNT(*) FROM keep_cells_chunk
        """).fetchone()[0]

        keep_total += keep_now

        # Only update TRUE cells within the current chunk
        conn.execute(f"""
            UPDATE obs
            SET {add_data} = TRUE
            WHERE atlas_cell_id IN (
                SELECT atlas_cell_id FROM keep_cells_chunk
            )
        """)

        # Clean up the small temporary table immediately after each chunk
        conn.execute("DROP TABLE IF EXISTS keep_cells_chunk")

    # 5. Summarize results
    removed = total_cells - keep_total

    logger.info(f"filter_cells Done, elapsed time: {(datetime.now() - start_time).total_seconds():.2f} seconds")
    logger.info(f"Retained cells = {keep_total} / {total_cells} , ({keep_total / total_cells * 100:.2f}%)")

    # Memory cleanup
    _cleanup_qc_after_step(
        conn,
        temp_tables=["keep_cells_chunk"],
        checkpoint=False,
        collect=True,
    )


def filter_genes(
    atlas: Atlas,
    min_counts: Optional[int] = None,
    min_cells: Optional[int] = None,
    max_counts: Optional[int] = None,
    max_cells: Optional[int] = None,
    add_data: str = "filter_genes"
) -> None:
    """Filter genes based on expression counts and the number of detected cells.

    This function computes each gene's total expression and the number of cells in which it is detected from the expression matrix,
    then writes the gene filtering flag into the ``var`` table according to the thresholds. The result is saved in the Atlas database.

    Genes are not directly removed.
    Instead, a boolean column is added or updated in the ``var`` table to mark which genes pass the filtering conditions.

    Parameters
    ----------
    atlas
        Atlas object. It usually needs to be connected to a DuckDB database and contain the
        ``var`` table and ``X_HyS_data`` table required by this function for reading or writing.

        The ``var`` table must contain the ``atlas_gene_id`` field;
        the ``X_HyS_data`` table must contain the ``atlas_gene_id`` and ``data_count`` fields.

    min_counts
        Lower bound of total gene expression. Only genes with ``sum_expr >= min_counts`` pass this condition.
        If ``None``, this lower-bound condition is not used.

    min_cells
        Lower bound of the number of cells in which the gene is detected. Only genes with
        ``nonzero_expr >= min_cells`` pass this condition.
        If ``None``, this lower-bound condition is not used.

    max_counts
        Upper bound of total gene expression. Only genes with ``sum_expr <= max_counts`` pass this condition.
        If ``None``, this upper-bound condition is not used.

    max_cells
        Upper bound of the number of cells in which the gene is detected. Only genes with
        ``nonzero_expr <= max_cells`` pass this condition.
        If ``None``, this upper-bound condition is not used.

    add_data
        Name of the boolean filtering column written to the ``var`` table. The default value is ``"filter_genes"``.

        If this column does not exist, the function automatically adds it;
        if this column already exists, the function rewrites it as ``TRUE`` or ``FALSE`` according to the current filtering conditions.

    Returns
    -------
    None
        The result is written directly into the Atlas database.
        The ``var`` table in the database adds or updates the field specified by ``add_data``;
        genes that meet the filtering conditions are marked as ``TRUE``, otherwise they are marked as ``FALSE``.

    Notes
    -----
    This function only writes filtering flags and does not delete the original data in ``var`` or ``X_HyS_data``.

    Examples
    --------
    Keep genes detected in at least 3 cells::

        sap.pp.filter_genes(
            atlas,
            min_cells=3,
        )

    Keep genes with total expression no lower than 10 and detected in at least 5 cells::

        sap.pp.filter_genes(
            atlas,
            min_counts=10,
            min_cells=5,
        )

    Check filtering result statistics::

        atlas.query(
            "SELECT filter_genes, COUNT(*) AS n_genes "
            "FROM var GROUP BY filter_genes"
        )

    Build a read index based on filtered genes::

        atlas.build_read_index(
            cell_condition="filter_cells",
            gene_condition="filter_genes",
            use_hvg=True,
            use_data="data_log1p",
        )

    """

    start_time = datetime.now()

    conn = atlas.connection

    # DuckDB multithreading
    try:
        th = os.cpu_count() or 1
        conn.execute(f"PRAGMA threads={th}")
    except Exception:
        pass

    # Count the number of genes
    n_genes = conn.execute("""
        SELECT COUNT(*) FROM var
    """).fetchone()[0]

    # Add filtering field
    conn.execute(f"""
        ALTER TABLE var
        ADD COLUMN IF NOT EXISTS {add_data} BOOLEAN DEFAULT FALSE
    """)

    # Build SQL conditions
    conds = []

    if min_counts is not None:
        conds.append(f"COALESCE(s.sum_expr, 0) >= {min_counts}")

    if max_counts is not None:
        conds.append(f"COALESCE(s.sum_expr, 0) <= {max_counts}")

    if min_cells is not None:
        conds.append(f"COALESCE(s.nonzero_expr, 0) >= {min_cells}")

    if max_cells is not None:
        conds.append(f"COALESCE(s.nonzero_expr, 0) <= {max_cells}")

    condition = " AND ".join(conds) if conds else "TRUE"

    # Aggregate X_HyS_data

    conn.execute("DROP TABLE IF EXISTS gene_filter_stats_tmp")

    # Only generate a small gene-level temporary table; the result size is approximately the number of genes, not the number of nnz entries
    conn.execute("""
        CREATE TEMP TABLE gene_filter_stats_tmp AS
        SELECT
            atlas_gene_id,
            SUM(data_count) AS sum_expr,
            COUNT(*) AS nonzero_expr
        FROM X_HyS_data
        GROUP BY atlas_gene_id
    """)

    # Pure SQL write-back to var

    conn.execute(f"""
        UPDATE var
        SET {add_data} =
            CASE
                WHEN {condition}
                THEN TRUE
                ELSE FALSE
            END
        FROM gene_filter_stats_tmp AS s
        WHERE var.atlas_gene_id = s.atlas_gene_id
    """)

    # Handle genes with completely zero expression; genes that never appear in CSR do not pass filtering by default
    conn.execute(f"""
        UPDATE var
        SET {add_data} = FALSE
        WHERE atlas_gene_id NOT IN (
            SELECT atlas_gene_id FROM gene_filter_stats_tmp
        )
    """)

    # Summarize results
    keep_count = conn.execute(f"""
        SELECT COUNT(*) FROM var
        WHERE {add_data} = TRUE
    """).fetchone()[0]

    conn.execute("DROP TABLE IF EXISTS gene_filter_stats_tmp")

    logger.info(f"filter_genes Done, elapsed time: {(datetime.now() - start_time).total_seconds():.2f} seconds")
    logger.info(f"Retained genes = {keep_count} / {n_genes} , ({keep_count / n_genes * 100:.2f}%)")

    # Memory cleanup
    _cleanup_qc_after_step(
        conn,
        temp_tables=["gene_filter_stats_tmp"],
        checkpoint=False,
        collect=True,
    )


def calculate_cell_total_counts(
    atlas: Atlas,
    add_data: str = "cell_total_counts",
    chunk_cells: int = 1_000_000,
) -> None:
    """Calculate total UMI counts for each cell.

    This function is used to calculate the total expression of each cell in the Atlas database, namely the sum of counts for each cell
    in the ``X_HyS_data.data_count`` field, and writes the result into the ``obs`` table.

    The result is usually used for quality control of single-cell data, such as checking sequencing depth for each cell,
    filtering low-quality cells, assisting normalization checks, and plotting QC distribution figures.

    The function processes the expression matrix by ``atlas_cell_id`` ranges in chunks. Each chunk
    only aggregates expression records within the current cell range and writes the result back to the ``obs`` table,
    avoiding excessive memory pressure caused by aggregating the full ``X_HyS_data`` table at once.

    Parameters
    ----------
    atlas
        Atlas object. The object must already be connected to a DuckDB database, and the database must contain at least
        the ``obs`` table and ``X_HyS_data`` table.

        The ``obs`` table must contain the ``atlas_cell_id`` field;
        the ``X_HyS_data`` table must contain the ``atlas_cell_id`` and ``data_count`` fields.

    add_data
        Name of the result column written to the ``obs`` table. The default value is ``"cell_total_counts"``.

        If this column does not exist, the function automatically adds it;
        if this column already exists, the function first resets all values in this column to ``0``, then writes each cell's
        total counts back to this column.

    chunk_cells
        Number of cell IDs covered by each chunk when processing by ``atlas_cell_id`` range.
        The default value is ``1_000_000``.

        A larger value usually reduces the number of SQL loops and improves speed,
        but increases memory usage during aggregation of a single chunk;
        a smaller value is more stable, but may take longer to run.

    Returns
    -------
    None
        The result is written directly into the ``add_data`` column of the ``obs`` table (total UMI counts for each cell),
        and no object is returned.

    Notes
    -----
    This function does not modify the expression matrix itself.
    It only adds or updates a cell-level QC metric column in the ``obs`` table.

    For cells that do not have any expression records in ``X_HyS_data``, their ``add_data`` value remains
    ``0``. Therefore, this function can safely handle cells with completely no nonzero expression records.

    Examples
    --------
    Write total UMI counts for each cell using the default column name::

        sap.pp.calculate_cell_total_counts(atlas)

    Adjust the chunk size to reduce memory pressure::

        sap.pp.calculate_cell_total_counts(
            atlas,
            add_data="cell_total_counts",
            chunk_cells=200_000,
        )

    Check result statistics::

        atlas.query(
            "SELECT "
            "MIN(cell_total_counts) AS min_counts, "
            "MAX(cell_total_counts) AS max_counts, "
            "AVG(cell_total_counts) AS mean_counts "
            "FROM obs"
        )

    Before plotting or filtering, you can first inspect the newly generated field in the ``obs`` table::
        atlas.head("obs")
    """

    start_time = datetime.now()

    conn = atlas.connection

    # DuckDB performance parameters
    try:
        th = os.cpu_count() or 1
        conn.execute(f"PRAGMA threads={th}")
    except Exception:
        pass

    # Ensure obs has the target column
    conn.execute(f"""
        ALTER TABLE obs
        ADD COLUMN IF NOT EXISTS {add_data} DOUBLE
    """)

    conn.execute(f"""
        UPDATE obs
        SET {add_data} = 0
    """)

    # Get the cell_id range
    min_cell, max_cell = conn.execute("""
        SELECT
            MIN(atlas_cell_id),
            MAX(atlas_cell_id)
        FROM obs
    """).fetchone()

    if min_cell is None or max_cell is None:
        logger.info("obs is empty; skipped.")
        return

    n_chunks = math.ceil((max_cell - min_cell + 1) / chunk_cells)

    total_updated_cells = 0

    # Chunked aggregation + chunked write-back
    pbar = progress(
        range(n_chunks),
        total=n_chunks,
        desc="calculate_cell_total_counts",
        unit="chunk",
    )

    for i in pbar:

        c_start = min_cell + i * chunk_cells
        c_end = min(c_start + chunk_cells - 1, max_cell)

        # Only create a small temporary table for the current chunk
        conn.execute("DROP TABLE IF EXISTS cell_total_counts_chunk")

        conn.execute(f"""
            CREATE TEMP TABLE cell_total_counts_chunk AS
            SELECT
                atlas_cell_id,
                SUM(data_count) AS total_counts
            FROM X_HyS_data
            WHERE atlas_cell_id BETWEEN {c_start} AND {c_end}
            GROUP BY atlas_cell_id
        """)

        n_now = conn.execute("""
            SELECT COUNT(*) FROM cell_total_counts_chunk
        """).fetchone()[0]

        total_updated_cells += n_now

        # Only write back cells with expression records in the current chunk
        conn.execute(f"""
            UPDATE obs
            SET {add_data} = t.total_counts
            FROM cell_total_counts_chunk t
            WHERE obs.atlas_cell_id = t.atlas_cell_id
        """)

        # Clean up immediately after each chunk
        conn.execute("DROP TABLE IF EXISTS cell_total_counts_chunk")

    logger.info(f"calculate_cell_total_counts Done, elapsed time: {(datetime.now() - start_time).total_seconds():.2f} seconds")

    # Memory cleanup
    _cleanup_qc_after_step(
        conn,
        temp_tables=["cell_total_counts_chunk"],
        checkpoint=False,
        collect=True,
    )


def calculate_gene_total_counts(
    atlas: 'Atlas',
    add_gene_total_counts: str = "gene_total_counts",
    add_gene_mean_counts: str = "gene_mean_counts",
) -> None:
    """Calculate total counts and mean counts for each gene.

    This function is used to calculate gene-level QC statistics in the Atlas database and write the results into
    the ``var`` table. The function aggregates ``data_count`` from the ``X_HyS_data`` table by ``atlas_gene_id``
    to obtain the total expression of each gene across all cells; it also computes the mean expression of each gene
    based on the total number of cells in the ``obs`` table.

    The results are usually used for gene quality control, gene filtering, highly expressed gene checks, data overview,
    and downstream visualization analysis.

    Parameters
    ----------
    atlas
        Atlas object. It usually needs to be connected to a DuckDB database, and the database must contain at least
        the ``obs`` table, ``var`` table, and ``X_HyS_data`` table.

        The ``obs`` table is used to count the total number of cells;
        the ``var`` table must contain the ``atlas_gene_id`` field;
        the ``X_HyS_data`` table must contain the ``atlas_gene_id`` and ``data_count`` fields.

    add_gene_total_counts
        Name of the gene total expression column written to the ``var`` table. The default value is
        ``"gene_total_counts"``.

        If this column does not exist, the function automatically adds it;
        if this column already exists, the function rewrites it with the currently calculated gene total counts.

    add_gene_mean_counts
        Name of the gene mean expression column written to the ``var`` table. The default value is
        ``"gene_mean_counts"``.

        This value is calculated as:

        ``gene_mean_counts = gene_total_counts / the total number of cells in the obs table``

        If this column does not exist, the function automatically adds it;
        if this column already exists, the function rewrites it with the currently calculated gene mean counts.

    Returns
    -------
    None
        The result is written directly into the ``var`` table in the Atlas database, and no object is returned.

        The ``var`` table in the database adds or updates two fields:

        1. ``add_gene_total_counts``: total counts for each gene;
        2. ``add_gene_mean_counts``: mean counts for each gene.

    Notes
    -----
    This function does not modify the expression matrix itself.
    It only adds or updates gene-level statistic columns in the ``var`` table.

    Examples
    --------
    Calculate gene total counts and mean counts using the default column names::

        sap.pp.calculate_gene_total_counts(atlas)
        atlas.head("var")

    Check gene statistic results::

        atlas.query(
            "SELECT atlas_gene_id, gene_total_counts, gene_mean_counts "
            "FROM var "
            "ORDER BY gene_total_counts DESC "
            "LIMIT 10"
        )

    Check the overall range of gene total counts::

        atlas.query(
            "SELECT "
            "MIN(gene_total_counts) AS min_counts, "
            "MAX(gene_total_counts) AS max_counts, "
            "AVG(gene_total_counts) AS mean_counts "
            "FROM var"
        )
    """

    start_time = datetime.now()

    conn = atlas.connection

    # DuckDB parallel setting
    try:
        th = os.cpu_count() or 1
        conn.execute(f"PRAGMA threads={th}")

    except:
        pass

    # Ensure the var table has the target columns
    cols = [r[1] for r in conn.execute("PRAGMA table_info(var)").fetchall()]

    if add_gene_total_counts not in cols:
        conn.execute(f"ALTER TABLE var ADD COLUMN {add_gene_total_counts} DOUBLE DEFAULT 0")
    if add_gene_mean_counts not in cols:
        conn.execute(f"ALTER TABLE var ADD COLUMN {add_gene_mean_counts} DOUBLE DEFAULT 0")

    # Total number of cells
    total_cells = conn.execute("SELECT COUNT(*) FROM obs").fetchone()[0]

    conn.execute("DROP TABLE IF EXISTS gene_stats_tmp")
    conn.execute("""
        CREATE TEMP TABLE gene_stats_tmp AS
        SELECT
            atlas_gene_id,
            SUM(data_count) AS total_counts
        FROM X_HyS_data
        GROUP BY atlas_gene_id
    """)

    conn.execute(f"""
        UPDATE var
        SET
            {add_gene_total_counts} = s.total_counts,
            {add_gene_mean_counts} = s.total_counts / {total_cells}
        FROM gene_stats_tmp AS s
        WHERE var.atlas_gene_id = s.atlas_gene_id
    """)

    # Fill zero-expression genes with zero
    conn.execute(f"""
        UPDATE var
        SET
            {add_gene_total_counts} = 0,
            {add_gene_mean_counts} = 0
        WHERE atlas_gene_id NOT IN (SELECT atlas_gene_id FROM gene_stats_tmp)
    """)

    # Memory cleanup
    conn.execute("DROP TABLE IF EXISTS gene_stats_tmp")

    _cleanup_qc_after_step(
        conn,
        temp_tables=["gene_stats_tmp"],
        checkpoint=False,
        collect=True,
    )

    logger.info(f"calculate_gene_total_counts Done, elapsed time: {(datetime.now() - start_time).total_seconds():.2f} seconds")


def calculate_qc_metrics(
    atlas: Atlas,
    qc_vars: dict[str, Any] | None=None,
    chunk_cells: int=100_000
) -> None:
    """Calculate commonly used single-cell QC metrics.

    This function is used to calculate cell-level and gene-level quality-control metrics in the Atlas database,
    and writes the calculated results directly into the ``obs`` table and ``var`` table in the Atlas database.

    The function mainly calculates two types of metrics:

    1. cell-wise QC metrics, written to the ``obs`` table;
    2. gene-wise QC metrics, written to the ``var`` table.

    For each cell, the function counts the total counts of the cell, the number of detected nonzero genes,
    and the counts and proportions of specified QC gene sets.
    For example, by default it calculates metrics related to mitochondrial genes and ribosomal genes.

    For each gene, the function counts the total counts of the gene across all cells, and how many
    cells have nonzero expression detected for that gene.

    The function computes cell-wise QC metrics by ``atlas_cell_id`` ranges in chunks,
    avoiding excessive memory pressure caused by aggregating all cells at once. The result size of gene-wise QC metrics
    is approximately equal to the number of genes, so it is calculated once outside the loop.

    Parameters
    ----------
    atlas
        Atlas object. The object must already be connected to a DuckDB database, and the database must contain at least
        the ``obs`` table, ``var`` table, and ``X_HyS_data`` table.

        The ``obs`` table must contain the ``atlas_cell_id`` field;
        the ``var`` table must contain the ``atlas_gene_id`` and ``atlas_gene_name`` fields;
        the ``X_HyS_data`` table must contain the ``atlas_cell_id``, ``atlas_gene_id``, and
        ``data_count`` fields.

    qc_vars
        Definition of QC gene sets. If ``None``, the default setting is used::

            {
                "mt": "MT-",
                "ribo": "^(RPS|RPL)"
            }

        The dictionary key is used as the QC metric name. For example, ``"mt"`` generates
        ``var.mt``, ``obs.total_counts_mt``, and ``obs.pct_counts_mt``;
        ``"ribo"`` generates ``var.ribo``, ``obs.total_counts_ribo``, and
        ``obs.pct_counts_ribo``.

        The dictionary value is a gene-name matching pattern:

        - If the string starts with ``"^"``, regular expression matching is used on ``atlas_gene_name``;
        - If the string does not start with ``"^"``, matching is performed by gene-name prefix.

        For example, ``"MT-"`` means matching genes that start with ``MT-``;
        ``"^(RPS|RPL)"`` means matching genes that start with ``RPS`` or ``RPL``.

    chunk_cells
        Number of cell IDs covered by each chunk when processing by ``atlas_cell_id`` range.
        The default value is ``100_000``.

        A larger value usually reduces the number of SQL loops and improves speed, but increases memory usage during aggregation of a single chunk;
        a smaller value is more stable, but may take longer to run.

    Returns
    -------
    None
        The result is written directly into the Atlas database, and no object is returned.

        The ``obs`` table adds or updates the following fields:

        - ``cell_total_counts``: total counts for each cell;
        - ``n_genes_by_counts``: number of nonzero-expression genes detected in each cell;
        - ``total_counts_{qc_key}``: sum of counts for this QC gene set in each cell;
        - ``pct_counts_{qc_key}``: proportion of counts from this QC gene set in the cell's total counts.

        The ``var`` table adds or updates the following fields:

        - ``{qc_key}``: whether the gene belongs to the specified QC gene set;
        - ``gene_total_counts``: sum of counts for this gene across all cells;
        - ``n_cells_by_counts``: number of cells in which nonzero expression of this gene is detected.

    Notes
    -----
    This function does not delete cells or genes, nor does it modify the expression matrix itself. It only adds or updates QC metric columns in the ``obs`` and
    ``var`` tables.

    Examples
    --------
    Calculate metrics using the default QC gene sets::

        sap.pp.calculate_qc_metrics(atlas)

    By default, mitochondrial and ribosomal related metrics are calculated::

        qc_vars = {
            "mt": "MT-",
            "ribo": "^(RPS|RPL)",
        }
        sap.pp.calculate_qc_metrics(atlas, qc_vars=qc_vars)

    Customize QC gene sets, for example calculating the proportion of hemoglobin genes::

        sap.pp.calculate_qc_metrics(
            atlas,
            qc_vars={
                "mt": "MT-",
                "ribo": "^(RPS|RPL)",
                "hb": "^(HBA|HBB)",
            },
        )

    Check cell-level QC metrics::

        atlas.query(
            "SELECT cell_total_counts, n_genes_by_counts, "
            "pct_counts_mt, pct_counts_ribo "
            "FROM obs LIMIT 5"
        )

    Check gene-level QC metrics::

        atlas.query(
            "SELECT atlas_gene_name, mt, ribo, gene_total_counts, n_cells_by_counts "
            "FROM var LIMIT 5"
        )

    """

    start_time = datetime.now()
    conn = atlas.connection

    if qc_vars is None:
        qc_vars = {
            "mt": "MT-",
            "ribo": "^(RPS|RPL)"
        }

    try:
        th = os.cpu_count() or 8
        conn.execute(f"PRAGMA threads={th}")
    except Exception:
        pass

    # Add QC flags to var
    for qc_key, pattern in qc_vars.items():
        conn.execute(f"""
            ALTER TABLE var
            ADD COLUMN IF NOT EXISTS {qc_key} BOOLEAN
        """)

        if pattern.startswith("^"):
            conn.execute(f"""
                UPDATE var
                SET {qc_key} = regexp_matches(atlas_gene_name, '{pattern}', 'i')
            """)
        else:
            conn.execute(f"""
                UPDATE var
                SET {qc_key} =
                    UPPER(atlas_gene_name) LIKE '{pattern.upper()}%'
            """)

    # Initialize obs columns
    conn.execute("""
        ALTER TABLE obs
        ADD COLUMN IF NOT EXISTS cell_total_counts REAL
    """)
    conn.execute("""
        ALTER TABLE obs
        ADD COLUMN IF NOT EXISTS n_genes_by_counts INTEGER
    """)

    for qc_key in qc_vars.keys():
        conn.execute(f"""
            ALTER TABLE obs
            ADD COLUMN IF NOT EXISTS total_counts_{qc_key} REAL
        """)
        conn.execute(f"""
            ALTER TABLE obs
            ADD COLUMN IF NOT EXISTS pct_counts_{qc_key} REAL
        """)

    # Initialize obs first to avoid keeping old values for completely empty cells
    conn.execute("""
        UPDATE obs
        SET
            cell_total_counts = 0,
            n_genes_by_counts = 0
    """)

    for qc_key in qc_vars.keys():
        conn.execute(f"""
            UPDATE obs
            SET
                total_counts_{qc_key} = 0,
                pct_counts_{qc_key} = 0
        """)

    # Calculate the cell_id range
    min_cell, max_cell = conn.execute("""
        SELECT MIN(atlas_cell_id), MAX(atlas_cell_id)
        FROM obs
    """).fetchone()

    if min_cell is None or max_cell is None:
        logger.info("obs is empty; skipped.")
        return

    n_chunks = math.ceil((max_cell - min_cell + 1) / chunk_cells)

    # Chunked processing
    pbar = progress(
        range(n_chunks),
        total=n_chunks,
        desc="calculate_qc_metrics",
        unit="chunk",
    )

    for i in pbar:

        c_start = min_cell + i * chunk_cells
        c_end = min(c_start + chunk_cells - 1, max_cell)

        qc_sum_expr = []
        for qc_key in qc_vars.keys():
            qc_sum_expr.append(
                f"SUM(CASE WHEN v.{qc_key} THEN x.data_count ELSE 0 END)"
                f" AS total_counts_{qc_key}"
            )
        qc_sum_sql = ",\n".join(qc_sum_expr)

        conn.execute("DROP TABLE IF EXISTS _cell_chunk")

        conn.execute(f"""
            CREATE TEMP TABLE _cell_chunk AS
            SELECT
                x.atlas_cell_id,
                SUM(x.data_count) AS cell_total_counts,
                COUNT(*)    AS n_genes_by_counts,
                {qc_sum_sql}
            FROM X_HyS_data x
            JOIN var v
              ON x.atlas_gene_id = v.atlas_gene_id
            WHERE x.atlas_cell_id BETWEEN {c_start} AND {c_end}
            GROUP BY x.atlas_cell_id
        """)

        set_expr = [
            "cell_total_counts = c.cell_total_counts",
            "n_genes_by_counts = c.n_genes_by_counts"
        ]

        for qc_key in qc_vars.keys():
            set_expr.append(
                f"total_counts_{qc_key} = c.total_counts_{qc_key}"
            )
            set_expr.append(
                f"""
                pct_counts_{qc_key} =
                CASE WHEN c.cell_total_counts > 0
                THEN 100.0 * c.total_counts_{qc_key} / c.cell_total_counts
                ELSE 0 END
                """
            )

        conn.execute(f"""
            UPDATE obs
            SET {",".join(set_expr)}
            FROM _cell_chunk c
            WHERE obs.atlas_cell_id = c.atlas_cell_id
        """)

        conn.execute("DROP TABLE IF EXISTS _cell_chunk")

    # gene-wise QC: calculate once outside the loop

    conn.execute("""
        ALTER TABLE var
        ADD COLUMN IF NOT EXISTS gene_total_counts REAL
    """)
    conn.execute("""
        ALTER TABLE var
        ADD COLUMN IF NOT EXISTS n_cells_by_counts INTEGER
    """)

    # The gene-level temporary table is small and does not require chunking
    conn.execute("DROP TABLE IF EXISTS _gene_qc")
    conn.execute("""
        CREATE TEMP TABLE _gene_qc AS
        SELECT
            atlas_gene_id,
            SUM(data_count) AS gene_total_counts,
            COUNT(*)  AS n_cells_by_counts
        FROM X_HyS_data
        GROUP BY atlas_gene_id
    """)

    # First set var to 0 to handle genes with completely zero expression
    conn.execute("""
        UPDATE var
        SET
            gene_total_counts = 0,
            n_cells_by_counts = 0
    """)

    conn.execute("""
        UPDATE var
        SET
            gene_total_counts = g.gene_total_counts,
            n_cells_by_counts = g.n_cells_by_counts
        FROM _gene_qc g
        WHERE var.atlas_gene_id = g.atlas_gene_id
    """)

    conn.execute("DROP TABLE IF EXISTS _gene_qc")

    # Memory cleanup
    _cleanup_qc_after_step(
        conn,
        temp_tables=["_cell_chunk", "_gene_qc"],
        checkpoint=False,
        collect=True,
    )

    logger.info(f"calculate_qc_metrics Done, elapsed time: {(datetime.now() - start_time).total_seconds():.2f} seconds")


def _cleanup_qc_after_step(
    conn: DuckDBPyConnection,
    temp_tables: list[str] | None = None,
    checkpoint: bool = False,
    collect: bool = True,
):
    """Clean temporary resources generated by the current step.

    This internal function belongs to the quality-control module and supports public APIs in the same module.

    It calculates cell/gene QC metrics at the database level and writes filtering flags back.

    It is usually not called directly as a user-facing entry point; when called directly, the input object, database connection,
    and related temporary tables must already have been prepared by upstream steps.

    Parameters
    ----------
    conn
        DuckDB database connection.

    temp_tables
        List of DuckDB temporary table names to drop. The default value is
        ``None``, which is treated as an empty list.

        When ``temp_tables=None``, this function does not attempt to drop any
        temporary tables. It still performs the optional ``CHECKPOINT`` and
        Python garbage collection steps controlled by ``checkpoint`` and
        ``collect``.

    checkpoint
        Whether to execute DuckDB ``CHECKPOINT`` after cleanup.

    collect
        Whether to trigger Python garbage collection after cleanup.

    Notes
    -----
    This is an internal helper; unless you need to extend the internal scAtlasPy workflow, it is generally not recommended to call it directly in user code.
    """

    if temp_tables is None:
        temp_tables = []

    for t in temp_tables:
        try:
            conn.execute(f"DROP TABLE IF EXISTS {t}")
        except Exception:
            pass

    if checkpoint:
        try:
            conn.execute("CHECKPOINT")
        except Exception:
            pass

    if collect:
        try:
            gc.collect()
        except Exception:
            pass
