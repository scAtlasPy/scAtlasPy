import duckdb
from os import PathLike, fspath
from ..io import progress
from datetime import datetime
import logging

logger = logging.getLogger("Atlas")
logger.addHandler(logging.NullHandler())


class FilterIndexBuilder:

    """Filter index builder.

    This class rebuilds continuous cell indices and gene indices based on the filtering
    markers in ``obs`` and ``var``, and generates the filtered HyS tables required for
    subsequent minibatch reading. It is the underlying implementation of
    ``Atlas.build_read_index``. Regular users usually do not need to instantiate it directly.

    Parameters
    ----------
    file_path
        Path to the Atlas ``.sasql`` database file.
    cell_condition
        Boolean column name or condition in ``obs`` used to filter cells;
        if ``None``, all cells are retained.
    gene_condition
        Boolean column name or condition in ``var`` used to filter genes;
        if ``None``, all genes are retained.
    use_hvg
        Whether to additionally restrict genes to highly variable genes.
    use_data
        Expression value column name read from the ``X_HyS_data`` table, such as
        ``"data_count"``, ``"data_normalize"``, ``"data_log1p"``, or ``"data_scale"``.

    Notes
    -----
    It is recommended to call this workflow through ``atlas.build_read_index(...)``.
    When using this class directly, make sure the Atlas database already contains
    basic tables such as ``obs``, ``var``, and ``X_HyS_data``.

    Examples
    --------
    Build a filtered index through an Atlas object::

        sap.pp.filter_cells(atlas, min_genes=200)
        sap.pp.filter_genes(atlas, min_cells=3)
        atlas.build_read_index(
            cell_condition="filter_cells",
            gene_condition="filter_genes",
            use_hvg=True,
            use_data="data_log1p",
        )

    Use the underlying builder directly, which is suitable for debugging or extending
    internal workflows::

        builder = FilterIndexBuilder(
            atlas.file_path,
            cell_condition="filter_cells",
            gene_condition="filter_genes",
            use_hvg=False,
            use_data="data_count",
        )
        builder.run()"""

    def __init__(
        self,
        file_path: PathLike[str] | str,
        *, # file_path can be passed positionally; parameters after * must be passed by name
        cell_condition: str | None = None,
        gene_condition: str | None = None,
        use_hvg: bool = True,
        use_data: str = "data_log1p",
    ):
        """Initialize the filter index builder.

        This constructor saves the Atlas database path, filtering conditions, HVG setting,
        and expression value column name, and creates a new DuckDB connection.
        The actual index rebuilding workflow is executed by ``run()``.

        Parameters
        ----------
        file_path
            Path to the Atlas ``.sasql`` database file.

        cell_condition
            Boolean column name in the ``obs`` table used to filter cells.
            If ``None``, all cells are retained.

        gene_condition
            Boolean column name in the ``var`` table used to filter genes.
            If ``None``, all genes are retained.

        use_hvg
            Whether to additionally apply ``highly_variable_genes=TRUE`` on top of
            the gene filtering condition.

        use_data
            Expression value column name read from the ``X_HyS_data`` table.

        Notes
        -----
        This class is usually called by ``Atlas.build_read_index(...)``.
        Regular users generally do not need to instantiate it directly.
        """

        self.file_path = fspath(file_path)       # Absolute path to the sasql file
        self.producer_num = 10           # Number of threads for minibatch streaming reading
        self.fetch_size = 500_0000    # Fetch size for minibatch streaming reading
        self.chunk_size = 1000_0000    # Amount of data processed each time

        self.cell_condition = cell_condition # Cell filtering condition filter_cells: indicates only cells with filter_cells = True are selected
        self.gene_condition = gene_condition # Gene filtering condition filter_genes: indicates only genes with filter_genes = True are selected
        self.use_hvg = use_hvg               # Whether to use HVG genes
        self.use_data = use_data       # Select which data column to process

        self.conn = duckdb.connect(file_path)
        self.conn.execute("PRAGMA preserve_insertion_order=true")
        # false does not force preservation of the input insertion order, allowing DuckDB to reorganize execution and write order for performance.
        # true tries to preserve the input order when writing through INSERT / COPY / SELECT


    def _has_column(self, table_name: str, column_name: str) -> bool:

        return self.conn.execute(
            """
            SELECT 1
            FROM information_schema.columns
            WHERE table_name = ?
              AND column_name = ?
            LIMIT 1
            """,
            [table_name, column_name],
        ).fetchone() is not None


    def _require_filter_column(
        self,
        *,
        table_name: str,
        column_name: str,
        function_name: str,
    ) -> None:

        if self._has_column(table_name, column_name):
            return

        raise ValueError(
            f"The {table_name} table is missing the filtering field {column_name}. "
            f"Please run sap.pp.{function_name}(...) first to generate this field, "
        )


    # External entry point
    def run(self):

        """Execute the complete read index construction workflow.

        This method performs the following steps in order:

        1. Rebuild ``obs.filter_cell_id``
        2. Rebuild ``var.filter_gene_id``
        3. Build the filtered expression matrix table ``X_HyS_data_filtered``
        4. Build the filtered indptr table ``X_HyS_indptr_filtered``

        After execution, the current DuckDB connection is closed.

        Returns
        -------
        None
            The result is written directly into the Atlas ``.sasql`` database.
        """

        start = datetime.now()

        self._rebuild_obs_filter_id()   # Reorder obs: filter cells + generate filter_cell_id
        self._rebuild_var_filter_id()   # Reorder var: filter genes + select HVG genes + generate filter_gene_id

        self._rebuild_x_hys_data_filtered()

        self._rebuild_x_hys_indptr_filtered()

        self.conn.close()

        logger.info(f"build_read_index Done, elapsed time: {(datetime.now() - start).total_seconds():.2f} seconds")


    # 1. Reorder obs: filter cells + generate filter_cell_id
    def _rebuild_obs_filter_id(self):

        """Rebuild ``obs.filter_cell_id``.

        This method first drops the old ``filter_cell_id`` column and then recreates it.
        It then selects the cells to retain according to ``cell_condition`` and generates
        continuous ``filter_cell_id`` values starting from 0 in ``atlas_cell_id`` order.

        Cells that do not pass the filtering condition keep ``filter_cell_id`` as ``NULL``.

        Returns
        -------
        None
            The result is written directly into the ``obs`` table.

        Raises
        ------
        ValueError
            Raised when the filtering column pointed to by ``cell_condition`` does not exist.
        """

        # If a filtering column is used, first confirm that the corresponding field exists in obs,
        # to avoid less understandable DuckDB errors
        if self.cell_condition is not None and self.cell_condition.isidentifier():
            self._require_filter_column(
                table_name="obs",
                column_name=self.cell_condition,
                function_name="filter_cells",
            )

        # Drop the old column
        self.conn.execute(""" ALTER TABLE obs DROP COLUMN IF EXISTS filter_cell_id """)

        # Add a new column
        self.conn.execute(""" ALTER TABLE obs ADD COLUMN filter_cell_id INTEGER """)

        # cell_condition=None means no cell filtering; otherwise filter cells by the specified boolean field
        if self.cell_condition is None:
            where_sql = "TRUE"
            logger.info("  -> Cell filtering is not used; keeping all cells")
        else:
            where_sql = f"{self.cell_condition}=TRUE"
            logger.info(f"  -> Using cell condition for filtering: {where_sql}")

        # Renumber only the cells that satisfy the condition
        self.conn.execute(f"""
        UPDATE obs
        SET filter_cell_id = sub.new_id
        FROM (
            SELECT
                atlas_cell_id,
                ROW_NUMBER() OVER (ORDER BY atlas_cell_id) - 1 AS new_id
            FROM obs
            WHERE {where_sql}
        ) AS sub
        WHERE obs.atlas_cell_id = sub.atlas_cell_id
        """)


    # 2. Reorder var: filter genes + select HVG genes + generate filter_gene_id
    def _rebuild_var_filter_id(self):

        """Rebuild ``var.filter_gene_id``.

        This method first drops the old ``filter_gene_id`` column and then recreates it.
        It then selects the genes to retain according to ``gene_condition`` and ``use_hvg``,
        and generates continuous ``filter_gene_id`` values starting from 0 in
        ``atlas_gene_id`` order.

        When ``use_hvg=True``, genes need to satisfy both:

        - ``gene_condition=TRUE``
        - ``highly_variable_genes=TRUE``

        Genes that do not pass the filtering condition keep ``filter_gene_id`` as ``NULL``.

        Returns
        -------
        None
            The result is written directly into the ``var`` table.

        Raises
        ------
        ValueError
            Raised when the filtering column pointed to by ``gene_condition`` does not exist.
        """

        # If a filtering column is used, first confirm that the corresponding field exists in var,
        # to avoid less understandable DuckDB errors
        if self.gene_condition is not None and self.gene_condition.isidentifier():
            self._require_filter_column(
                table_name="var",
                column_name=self.gene_condition,
                function_name="filter_genes",
            )

        # Drop the old column + add a new column
        self.conn.execute(""" ALTER TABLE var DROP COLUMN IF EXISTS filter_gene_id """)

        self.conn.execute(""" ALTER TABLE var ADD COLUMN filter_gene_id USMALLINT """)

        # gene_condition=None means no gene filtering; otherwise filter genes by the specified boolean field
        conditions = []

        if self.gene_condition is not None:
            conditions.append(f"({self.gene_condition})=TRUE")
            logger.info(f"  -> Using gene condition: {self.gene_condition}=TRUE")
        else:
            logger.info("  -> No gene filtering condition is used")

        # If HVG is enabled, additionally apply highly_variable_genes
        if self.use_hvg:
            conditions.append("highly_variable_genes=TRUE")
            logger.info("  -> Using the HVG gene subset")
        else:
            logger.info("  -> HVG filtering is not used; keeping all genes")

        condition = " AND ".join(conditions) if conditions else "TRUE"

        # Reorder gene_id
        self.conn.execute(f"""
        UPDATE var
        SET filter_gene_id = sub.new_id
        FROM (
            SELECT
                atlas_gene_id,
                ROW_NUMBER() OVER (ORDER BY atlas_gene_id) - 1 AS new_id
            FROM var
            WHERE {condition}
        ) AS sub
        WHERE var.atlas_gene_id = sub.atlas_gene_id
        """)


    # 3. Rebuild the new table: X_HyS_data_filtered
    def _rebuild_x_hys_data_filtered(self):

        """Build the filtered expression matrix table ``X_HyS_data_filtered``.

        This method reads expression records from the original ``X_HyS_data`` table and
        keeps only the cells and genes that pass filtering through ``obs.filter_cell_id``
        and ``var.filter_gene_id``.

        The output table contains:

        - ``filter_cell_id``: continuous cell index after filtering
        - ``filter_gene_id``: continuous gene index after filtering
        - ``data``: expression value column specified by ``use_data``
        - ``tid``: shard ID used for subsequent minibatch streaming reading

        In implementation, temporary mapping tables ``_obs_keep`` and ``_var_keep`` are
        created first, and then ``X_HyS_data.rowid`` is scanned and inserted in chunks
        to avoid processing an overly large expression matrix at once.

        Returns
        -------
        None
            The result is written directly into the ``X_HyS_data_filtered`` table.
        """

        conn = self.conn

        conn.execute("PRAGMA preserve_insertion_order = true")
        conn.execute("PRAGMA threads=10")

        # Create lightweight mapping tables in advance
        conn.execute("""
        CREATE OR REPLACE TEMP TABLE _obs_keep AS
        SELECT
            atlas_cell_id,
            CAST(filter_cell_id AS INTEGER) AS filter_cell_id
        FROM obs
        WHERE filter_cell_id IS NOT NULL
        """)

        conn.execute("""
        CREATE OR REPLACE TEMP TABLE _var_keep AS
        SELECT
            atlas_gene_id,
            CAST(filter_gene_id AS USMALLINT) AS filter_gene_id
        FROM var
        WHERE filter_gene_id IS NOT NULL
        """)

        # ANALYZE provides statistics for the DuckDB optimizer. It does not change data or create indexes,
        # but may make subsequent JOIN operations faster and more stable
        try:
            conn.execute("ANALYZE _obs_keep")
            conn.execute("ANALYZE _var_keep")
        except Exception:
            pass

        # Create the target table
        conn.execute("DROP TABLE IF EXISTS X_HyS_data_filtered")

        conn.execute("""
        CREATE TABLE X_HyS_data_filtered (
            filter_cell_id INTEGER,
            filter_gene_id USMALLINT,
            data REAL,
            tid TINYINT
        )
        """)

        # Get the rowid range
        min_id, max_id = conn.execute("""
            SELECT MIN(rowid), MAX(rowid)
            FROM X_HyS_data
        """).fetchone()

        if min_id is None:
            logger.debug(" X_HyS_data is an empty table, skipping")
            return

        total_rows = max_id - min_id + 1
        current = min_id

        logger.debug(f"rowid range: {min_id:,} ~ {max_id:,}")
        logger.debug(f"total rows to scan: {total_rows:,}")
        logger.debug(f"chunk_size = {self.chunk_size:,}")

        pbar = progress(
            total=total_rows,
            unit="rows",
            desc="build_read_index",
            ncols=130
        )

        # Insert sequentially in chunks
        while current <= max_id:
            end = min(current + self.chunk_size, max_id + 1)

            conn.execute(f"""
            INSERT INTO X_HyS_data_filtered
            SELECT
                obs.filter_cell_id,
                var.filter_gene_id,
                CAST(X.{self.use_data} AS REAL) AS data,
                CAST(0 AS TINYINT) AS tid
            FROM X_HyS_data AS X
            JOIN _obs_keep AS obs
              ON X.atlas_cell_id = obs.atlas_cell_id
            JOIN _var_keep AS var
              ON X.atlas_gene_id = var.atlas_gene_id
            WHERE X.rowid >= {current}
              AND X.rowid < {end}
            ORDER BY X.rowid
            """)

            pbar.update(end - current)
            current = end

        pbar.close()

        # Calculate tid
        conn.execute(f"""
            UPDATE X_HyS_data_filtered
            SET tid = CAST((rowid // {self.fetch_size}) % {self.producer_num} AS TINYINT)
        """)

        final_nnz = conn.execute("""
            SELECT COUNT(*)
            FROM X_HyS_data_filtered
        """).fetchone()[0]

        logger.info(f" X_HyS_data_filtered has been constructed successfully! nnz = {final_nnz:,}")

        # Clean up temporary tables
        conn.execute("DROP TABLE IF EXISTS _obs_keep")
        conn.execute("DROP TABLE IF EXISTS _var_keep")


    # 4. Rebuild the new table: X_HyS_indptr_filtered
    def _rebuild_x_hys_indptr_filtered(self):
        """Build the filtered CSR-like indptr table ``X_HyS_indptr_filtered``.

        This method counts the number of nonzero elements for each ``filter_cell_id``
        in ``X_HyS_data_filtered`` and completes all retained cells starting from
        the ``obs`` table. Even if some cells have no nonzero expression values,
        corresponding records are still kept in the indptr table.

        The final generated ``indptr`` represents the cumulative end position of each
        cell, namely the end pointer:

        - The start position of cell 0 is 0 by default
        - The end position of cell i is ``indptr[i]``
        - The start position of cell i should be obtained from the previous cell's
          ``indptr[i-1]``

        Returns
        -------
        None
            The result is written directly into the ``X_HyS_indptr_filtered`` table.
        """

        conn = self.conn

        # 1. First count the number of nonzero elements for each cell in the X table
        conn.execute("""
        CREATE OR REPLACE TEMP TABLE cell_nnz_raw AS
        SELECT
            CAST(filter_cell_id AS INTEGER) AS filter_cell_id,
            COUNT(*) AS cnt
        FROM X_HyS_data_filtered
        GROUP BY filter_cell_id
        """)

        # 2. Start from obs and complete all retained cells
        #    If a cell does not exist in X, it means it has no nonzero values, so cnt is filled with 0
        conn.execute("""
        CREATE OR REPLACE TEMP TABLE cell_nnz AS
        SELECT
            CAST(obs.filter_cell_id AS INTEGER) AS filter_cell_id,
            CAST(COALESCE(cell_nnz_raw.cnt, 0) AS BIGINT) AS cnt
        FROM obs
        LEFT JOIN cell_nnz_raw
          ON obs.filter_cell_id = cell_nnz_raw.filter_cell_id
        WHERE obs.filter_cell_id IS NOT NULL
        ORDER BY obs.filter_cell_id
        """)

        # 3. Generate prefix sum; here indptr stores the end position end_ptr of each cell
        conn.execute("""
        CREATE OR REPLACE TABLE X_HyS_indptr_filtered AS
        SELECT
            CAST(filter_cell_id AS INTEGER) AS filter_cell_id,
            CAST(SUM(cnt) OVER (ORDER BY filter_cell_id) AS BIGINT) AS indptr
        FROM cell_nnz
        ORDER BY filter_cell_id
        """)

        conn.execute("DROP TABLE IF EXISTS cell_nnz_raw")
        conn.execute("DROP TABLE IF EXISTS cell_nnz")
