from __future__ import annotations
import h5py
from . import progress
import numpy as np
import pandas as pd
from scipy import sparse
from anndata import AnnData
from datetime import datetime
from os import PathLike, fspath
from typing import TYPE_CHECKING
if TYPE_CHECKING:  # TYPE_CHECKING = imports for IDEs / type checkers; not executed at runtime to avoid circular imports
    from ..data import Atlas
from ..data._expression_source import resolve_expression_source
import logging
logger = logging.getLogger("Atlas")
logger.addHandler(logging.NullHandler())


def _get_axis_df(
    atlas: Atlas,
    table_name: str,
    id_column: str,
    columns: list[str] | str | None = None,
) -> pd.DataFrame:
    """Read a cell or gene metadata table and index it by the Atlas identifier."""

    conn = atlas.connection

    if conn is None:
        raise ValueError("atlas.connection is None. Please connect to the database first")

    table_exists = conn.execute("""
        SELECT COUNT(*)
        FROM information_schema.tables
        WHERE table_name = ?
    """, [table_name]).fetchone()[0]

    if table_exists == 0:
        raise ValueError(f"The {table_name} table does not exist in the database")

    table_columns = [
        row[0]
        for row in conn.execute("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = ?
            ORDER BY ordinal_position
        """, [table_name]).fetchall()
    ]

    if id_column not in table_columns:
        raise ValueError(f"The {id_column} field does not exist in the {table_name} table")

    if columns is None:
        select_columns = table_columns
    else:
        if isinstance(columns, str):
            columns = [columns]

        missing = [c for c in columns if c not in table_columns]
        if missing:
            raise ValueError(f"These fields do not exist in the {table_name} table: {missing}")

        select_columns = [id_column] + [
            c for c in columns
            if c != id_column
        ]

    select_sql = ", ".join([f'"{c}"' for c in select_columns])
    table_sql = '"' + table_name.replace('"', '""') + '"'

    df = conn.execute(f"""
        SELECT {select_sql}
        FROM {table_sql}
    """).df()

    return df.set_index(id_column, drop=True)


def write_h5ad(
    atlas: Atlas,
    out_h5ad_path: PathLike[str] | str,
    *,
    batch_cells: int = 1_000_000,
    use_data: str = "data_count",
) -> None:
    """Export an Atlas database to an h5ad file.

    This function reads ``obs``, ``var``, the sparse expression matrix,
    ``obsm_*`` result tables, and ``varm_*`` result tables from the Atlas DuckDB
    database, and writes them out as a standard AnnData ``.h5ad`` file for
    continued analysis in Scanpy or other tools that support AnnData.

    The expression matrix is reassembled into the CSR ``X`` in h5ad according to
    the internal Atlas HyS sparse structure. ``X.data`` comes from the expression
    source resolved by ``use_data``; this can be ``data_count`` in
    ``X_HyS_data`` or a derived expression table. ``X.indices`` comes from
    ``atlas_gene_id``.

    Parameters
    ----------
    atlas
        Atlas object. The object must already be connected to a DuckDB database,
        and the database must contain at least the ``obs``, ``var``,
        ``X_HyS_indptr``, and ``X_HyS_data`` tables.
    out_h5ad_path
        Output ``.h5ad`` file path.
    batch_cells
        Number of nonzero expression records processed per batch when writing
        expression matrix ``data`` and ``indices``.
        A larger value is usually faster, but increases per-batch memory usage.
    use_data
        Expression value field exported from the resolved expression source. The
        default is ``"data_count"``. Existing fields such as ``"data_log1p"``
        and ``"data_normalize"`` can also be used after the corresponding
        preprocessing steps.

    Returns
    -------
    None
        The result is written directly to the h5ad file specified by
        ``out_h5ad_path`` and no object is returned.

    Notes
    -----
    ``obsm_*`` tables are exported to h5ad ``obsm``. The ``obsm_`` prefix in the
    table name is removed.
    ``varm_*`` tables are exported to h5ad ``varm``. The ``varm_`` prefix in the
    table name is removed.

    Before export, the function resolves ``use_data`` from the base expression
    table or a derived expression table. If it cannot be resolved, an error is
    raised directly to avoid exporting an empty matrix or an incorrect field.

    Examples
    --------
    Export the current database::

        atlas.write_h5ad("./data/pbmc_export.h5ad")

    Use the object-style API and reduce per-batch memory usage::

        atlas.write_h5ad("./data/pbmc_export.h5ad", batch_cells=200_000)

    Export the log1p expression matrix::

        atlas.write_h5ad("./data/pbmc_log1p.h5ad", use_data="data_log1p")"""

    start_time = datetime.now()

    out_h5ad_path = fspath(out_h5ad_path)

    conn = atlas.connection

    if conn is None:
        raise ValueError("atlas.connection is None. Please connect to the database first")

    if not isinstance(use_data, str):
        raise TypeError("use_data must be str")

    if use_data == "":
        raise ValueError("use_data cannot be an empty string")

    # Safely quote SQL identifiers
    def _q(name: str) -> str:
        """Add safe quoting for DuckDB SQL identifiers.

        This internal helper is used to quote dynamic field names or table names,
        such as ``use_data``, ``obsm_*``, and ``varm_*`` tables. The function escapes
        double quotes in the name and wraps the result with double quotes to avoid
        SQL parsing failures caused by special characters or keywords.

        Parameters
        ----------
        name
            SQL identifier to quote.

        Returns
        -------
        str
            SQL identifier wrapped in double quotes.
        """
        return '"' + str(name).replace('"', '""') + '"'

    expr_source = resolve_expression_source(conn, use_data)

    # Read obs / var

    obs = conn.execute("SELECT * FROM obs ORDER BY atlas_cell_id").df()
    var = conn.execute("SELECT * FROM var ORDER BY atlas_gene_id").df()

    obs = obs.set_index("atlas_cell_id")
    var = var.set_index("atlas_gene_id")

    n_cells = obs.shape[0]
    n_genes = var.shape[0]

    # Read CSR indptr for the selected expression source.  Derived tables such as
    # data_scale may contain only a gene subset, so the original X_HyS_indptr
    # cannot be reused blindly.
    indptr_df = conn.execute(f"""
        WITH nnz_by_cell AS (
            SELECT
                {expr_source.cell_sql} AS atlas_cell_id,
                COUNT(*) AS cnt
            FROM {expr_source.from_sql}
            WHERE {expr_source.value_sql} IS NOT NULL
            GROUP BY {expr_source.cell_sql}
        ),
        complete AS (
            SELECT
                o.atlas_cell_id,
                COALESCE(n.cnt, 0) AS cnt
            FROM obs AS o
            LEFT JOIN nnz_by_cell AS n
              ON o.atlas_cell_id = n.atlas_cell_id
            ORDER BY o.atlas_cell_id
        )
        SELECT CAST(SUM(cnt) OVER (ORDER BY atlas_cell_id) AS BIGINT) AS indptr
        FROM complete
        ORDER BY atlas_cell_id
    """).df()

    indptr = np.empty(n_cells + 1, dtype=np.int64)
    indptr[0] = 0
    indptr[1:] = indptr_df["indptr"].to_numpy(dtype=np.int64)

    nnz = int(indptr[-1])

    # Create h5ad file
    with h5py.File(out_h5ad_path, "w") as f:

        # ---------- Root node attributes ----------
        f.attrs["encoding-type"] = "anndata"
        f.attrs["encoding-version"] = "0.1.0"

        gX = f.create_group("X")

        # AnnData CSR must be written in attrs, not as a dataset
        gX.attrs["encoding-type"] = "csr_matrix"
        gX.attrs["encoding-version"] = "0.1.0"
        gX.attrs["shape"] = (n_cells, n_genes)

        # ---------- datasets ----------
        d_data = gX.create_dataset(
            "data",
            shape=(nnz,),
            dtype="float32",
            chunks=(min(batch_cells, nnz),),
        )

        d_indices = gX.create_dataset(
            "indices",
            shape=(nnz,),
            dtype="uint16",
            chunks=(min(batch_cells, nnz),),
        )

        gX.create_dataset("indptr", data=indptr, dtype="int64")

        offset = 0

        for start in progress(
            range(0, nnz, batch_cells),
            desc="write_h5ad"
        ):
            end = min(start + batch_cells, nnz)

            rows = conn.execute(
                f"""
                SELECT atlas_gene_id, value
                FROM (
                    SELECT
                        {expr_source.gene_sql} AS atlas_gene_id,
                        {expr_source.value_sql} AS value,
                        ROW_NUMBER() OVER (ORDER BY {expr_source.cell_sql}, {expr_source.id_sql}) - 1 AS rn
                    FROM {expr_source.from_sql}
                    WHERE {expr_source.value_sql} IS NOT NULL
                ) AS q
                WHERE rn >= ? AND rn < ?
                ORDER BY rn
                """,
                [int(start), int(end)],
            ).fetchall()

            if not rows:
                continue

            idx, val = zip(*rows)

            d_indices[offset:offset + len(idx)] = idx
            d_data[offset:offset + len(val)] = val

            offset += len(idx)

        assert offset == nnz, f"nnz mismatch: {offset} != {nnz}"

        _write_dataframe(f, "obs", obs)
        _write_dataframe(f, "var", var)

        g_obsm = f.create_group("obsm")

        for (table_name,) in conn.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_name LIKE 'obsm_%'
        """).fetchall():

            key = table_name.replace("obsm_", "")

            value_cols = [
                row[0]
                for row in conn.execute("""
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_name = ?
                      AND column_name <> 'atlas_cell_id'
                    ORDER BY ordinal_position
                """, [table_name]).fetchall()
            ]

            if len(value_cols) == 0:
                continue

            select_values = ", ".join([
                f"t.{_q(c)} AS {_q(c)}"
                for c in value_cols
            ])

            df = conn.execute(f"""
                SELECT
                    {select_values}
                FROM obs AS o
                LEFT JOIN {_q(table_name)} AS t
                  ON o.atlas_cell_id = t.atlas_cell_id
                ORDER BY o.atlas_cell_id
            """).df()

            arr = df.to_numpy(dtype=np.float32)

            if arr.shape[0] != n_cells:
                raise ValueError(
                    f"obsm[{key}] row count error: {arr.shape[0]} != n_cells {n_cells}"
                )

            g_obsm.create_dataset(key, data=arr)

        g_varm = f.create_group("varm")

        def _q(name: str) -> str:
            return '"' + str(name).replace('"', '""') + '"'

        for (table_name,) in conn.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_name LIKE 'varm_%'
        """).fetchall():

            key = table_name.replace("varm_", "")

            # AnnData requires the first dimension of varm[key] to equal the number of rows in var.
            # Read numeric columns in the varm table except atlas_gene_id
            value_cols = [
                row[0]
                for row in conn.execute("""
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_name = ?
                      AND column_name <> 'atlas_gene_id'
                    ORDER BY ordinal_position
                """, [table_name]).fetchall()
            ]

            if len(value_cols) == 0:
                continue

            select_values = ", ".join([
                f"t.{_q(c)} AS {_q(c)}"
                for c in value_cols
            ])

            df = conn.execute(f"""
                SELECT
                    {select_values}
                FROM var AS v
                LEFT JOIN {_q(table_name)} AS t
                  ON v.atlas_gene_id = t.atlas_gene_id
                ORDER BY v.atlas_gene_id
            """).df()

            # ============================================================
            #   HVG genes      : original PCA loading
            #   non-HVG genes  : NaN
            # ============================================================
            arr = df.to_numpy(dtype=np.float32)

            if arr.shape[0] != n_genes:
                raise ValueError(
                    f"varm[{key}] row count error: {arr.shape[0]} != n_genes {n_genes}"
                )

            g_varm.create_dataset(key, data=arr)

    logger.info(f" write_h5ad Done, elapsed time: {(datetime.now() - start_time).total_seconds():.2f} seconds")


# Export the obs table in DuckDB as a pandas DataFrame
def get_obs_df(
    atlas: Atlas,
    columns: list[str] | str | None = None,
) -> pd.DataFrame:
    """Read the obs table from the Atlas database.

    This function reads all columns or selected columns from the ``obs`` table
    into a pandas DataFrame. It is suitable for quickly checking cell metadata,
    exporting statistical results, or merging with external analysis results.
    The returned result uses ``atlas_cell_id`` as the pandas index.

    Parameters
    ----------
    atlas
        Atlas object. The object must already be connected to a DuckDB database,
        and the database must contain the ``obs`` table.
    columns
        Column names to read from ``obs``. This can be a single string, a list of
        strings, or ``None``.
        If ``None``, all columns are read.

    Returns
    -------
    pandas.DataFrame
        Query result from ``obs`` indexed by ``atlas_cell_id``.

    Notes
    -----
    Even if ``atlas_cell_id`` is not explicitly included in ``columns``, the
    function automatically reads ``atlas_cell_id`` to set the DataFrame index.

    Examples
    --------
    Read all obs information::

        obs = atlas.get_obs_df()

    Read only clustering and automatic annotation columns::

            obs = atlas.get_obs_df(columns=["scatlas_cluster", "cell_type_auto"])"""

    start_time = datetime.now()

    df = _get_axis_df(
        atlas=atlas,
        table_name="obs",
        id_column="atlas_cell_id",
        columns=columns,
    )

    logger.info(f" get_obs_df Done, elapsed time: {(datetime.now() - start_time).total_seconds():.2f} seconds")

    return df


# Export the var table in DuckDB as a pandas DataFrame
def get_var_df(
    atlas: Atlas,
    columns: list[str] | str | None = None,
) -> pd.DataFrame:
    """Read the var table from the Atlas database.

    This function reads all columns or selected columns from the ``var`` table
    into a pandas DataFrame. It is suitable for checking gene metadata,
    exporting gene-level statistics, or aligning external gene-level results.
    The returned result uses ``atlas_gene_id`` as the pandas index.

    Parameters
    ----------
    atlas
        Atlas object. The object must already be connected to a DuckDB database,
        and the database must contain the ``var`` table.
    columns
        Column names to read from ``var``. This can be a single string, a list of
        strings, or ``None``.
        If ``None``, all columns are read.

    Returns
    -------
    pandas.DataFrame
        Query result from ``var`` indexed by ``atlas_gene_id``.

    Notes
    -----
    Even if ``atlas_gene_id`` is not explicitly included in ``columns``, the
    function automatically reads ``atlas_gene_id`` to set the DataFrame index.

    Examples
    --------
    Read all var information::

        var = atlas.get_var_df()

    Read only selected gene-level columns::

        var = atlas.get_var_df(columns=["atlas_gene_name", "highly_variable_genes"])
    """

    start_time = datetime.now()

    df = _get_axis_df(
        atlas=atlas,
        table_name="var",
        id_column="atlas_gene_id",
        columns=columns,
    )

    logger.info(f" get_var_df Done, elapsed time: {(datetime.now() - start_time).total_seconds():.2f} seconds")

    return df


def get_obsm_df(
    atlas: Atlas,
    table_name: str,
    atlas_cell_id: list[int] | np.ndarray | pd.Series | None = None,
    columns: list[str] | str | None = None,
) -> pd.DataFrame:
    """Read an ``obsm_*`` table from the Atlas database.

    This function reads cell-level multidimensional results, such as
    ``obsm_X_pca`` or ``obsm_X_umap``, into a pandas DataFrame. The returned
    result uses ``atlas_cell_id`` as the pandas index.

    Parameters
    ----------
    atlas
        Atlas object. The object must already be connected to a DuckDB database.
    table_name
        Name of the ``obsm_*`` table to read, for example ``"obsm_X_pca"`` or
        ``"obsm_X_umap"``.
    atlas_cell_id
        Optional Atlas cell IDs to select. If ``None``, all rows in the table
        are returned ordered by ``atlas_cell_id``. If a list or array is passed,
        the output follows the order of the provided IDs.
    columns
        Value columns to read from the ``obsm_*`` table. This can be a single
        string, a list of strings, or ``None``. If ``None``, all columns are
        read.

    Returns
    -------
    pandas.DataFrame
        Query result indexed by ``atlas_cell_id``.

    Examples
    --------
    Read all PCA coordinates::

        pca = atlas.get_obsm_df("obsm_X_pca")

    Read selected UMAP coordinates in a specific cell order::

        umap = atlas.get_obsm_df(
            "obsm_X_umap",
            atlas_cell_id=[10, 2, 7],
            columns=["umap1", "umap2"],
        )
    """

    start_time = datetime.now()
    conn = atlas.connection

    if conn is None:
        raise ValueError("atlas.connection is None. Please connect to the database first")

    if not isinstance(table_name, str) or not table_name:
        raise ValueError("table_name must be a non-empty string")

    if not table_name.startswith("obsm_"):
        raise ValueError(f"table_name must start with 'obsm_': {table_name}")

    table_exists = conn.execute("""
        SELECT COUNT(*)
        FROM information_schema.tables
        WHERE table_name = ?
    """, [table_name]).fetchone()[0]

    if table_exists == 0:
        raise ValueError(f"The {table_name} table does not exist in the database")

    table_columns = [
        row[0]
        for row in conn.execute("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = ?
            ORDER BY ordinal_position
        """, [table_name]).fetchall()
    ]

    if "atlas_cell_id" not in table_columns:
        raise ValueError(
            f"The atlas_cell_id field does not exist in the {table_name} table"
        )

    if columns is None:
        select_columns = table_columns
    else:
        if isinstance(columns, str):
            columns = [columns]

        missing = [c for c in columns if c not in table_columns]
        if missing:
            raise ValueError(f"These fields do not exist in the {table_name} table: {missing}")

        select_columns = ["atlas_cell_id"] + [
            c for c in columns
            if c != "atlas_cell_id"
        ]

    select_sql = ", ".join([f't."{c}"' for c in select_columns])
    table_sql = '"' + table_name.replace('"', '""') + '"'

    if atlas_cell_id is None:
        df = conn.execute(f"""
            SELECT {select_sql}
            FROM {table_sql} AS t
            ORDER BY t.atlas_cell_id
        """).df()
    else:
        cell_ids = [int(x) for x in list(atlas_cell_id)]

        if len(cell_ids) == 0:
            raise ValueError("atlas_cell_id cannot be empty")

        if len(cell_ids) != len(set(cell_ids)):
            raise ValueError("Duplicate values exist in atlas_cell_id. Please deduplicate them first")

        selected = pd.DataFrame(
            {
                "atlas_cell_id": np.asarray(cell_ids, dtype=np.int64),
                "_order": np.arange(len(cell_ids), dtype=np.int64),
            }
        )
        conn.register("_selected_obsm_cells", selected)
        try:
            df = conn.execute(f"""
                SELECT {select_sql}
                FROM _selected_obsm_cells AS s
                LEFT JOIN {table_sql} AS t
                  ON s.atlas_cell_id = t.atlas_cell_id
                ORDER BY s._order
            """).df()
        finally:
            conn.unregister("_selected_obsm_cells")

    logger.info(f" get_obsm_df Done, elapsed time: {(datetime.now() - start_time).total_seconds():.2f} seconds")

    return df.set_index("atlas_cell_id", drop=True)


def get_varm_df(
    atlas: Atlas,
    table_name: str,
    atlas_gene_id: list[int] | np.ndarray | pd.Series | None = None,
    columns: list[str] | str | None = None,
) -> pd.DataFrame:
    """Read a ``varm_*`` table from the Atlas database.

    This function reads gene-level multidimensional results, such as
    ``varm_PCs``, into a pandas DataFrame. The returned result uses
    ``atlas_gene_id`` as the pandas index.

    Parameters
    ----------
    atlas
        Atlas object. The object must already be connected to a DuckDB database.
    table_name
        Name of the ``varm_*`` table to read, for example ``"varm_PCs"``.
    atlas_gene_id
        Optional Atlas gene IDs to select. If ``None``, all rows in the table
        are returned ordered by ``atlas_gene_id``. If a list or array is passed,
        the output follows the order of the provided IDs.
    columns
        Value columns to read from the ``varm_*`` table. This can be a single
        string, a list of strings, or ``None``. If ``None``, all columns are
        read.

    Returns
    -------
    pandas.DataFrame
        Query result indexed by ``atlas_gene_id``.

    Examples
    --------
    Read all PCA loadings::

        pcs = atlas.get_varm_df("varm_PCs")

    Read selected PCs in a specific gene order::

        pcs = atlas.get_varm_df(
            "varm_PCs",
            atlas_gene_id=[10, 2, 7],
            columns=["pc0", "pc1"],
        )
    """

    start_time = datetime.now()
    conn = atlas.connection

    if conn is None:
        raise ValueError("atlas.connection is None. Please connect to the database first")

    if not isinstance(table_name, str) or not table_name:
        raise ValueError("table_name must be a non-empty string")

    if not table_name.startswith("varm_"):
        raise ValueError(f"table_name must start with 'varm_': {table_name}")

    table_exists = conn.execute("""
        SELECT COUNT(*)
        FROM information_schema.tables
        WHERE table_name = ?
    """, [table_name]).fetchone()[0]

    if table_exists == 0:
        raise ValueError(f"The {table_name} table does not exist in the database")

    table_columns = [
        row[0]
        for row in conn.execute("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = ?
            ORDER BY ordinal_position
        """, [table_name]).fetchall()
    ]

    if "atlas_gene_id" not in table_columns:
        raise ValueError(
            f"The atlas_gene_id field does not exist in the {table_name} table"
        )

    if columns is None:
        select_columns = table_columns
    else:
        if isinstance(columns, str):
            columns = [columns]

        missing = [c for c in columns if c not in table_columns]
        if missing:
            raise ValueError(f"These fields do not exist in the {table_name} table: {missing}")

        select_columns = ["atlas_gene_id"] + [
            c for c in columns
            if c != "atlas_gene_id"
        ]

    select_sql = ", ".join([f't."{c}"' for c in select_columns])
    table_sql = '"' + table_name.replace('"', '""') + '"'

    if atlas_gene_id is None:
        df = conn.execute(f"""
            SELECT {select_sql}
            FROM {table_sql} AS t
            ORDER BY t.atlas_gene_id
        """).df()
    else:
        gene_ids = [int(x) for x in list(atlas_gene_id)]

        if len(gene_ids) == 0:
            raise ValueError("atlas_gene_id cannot be empty")

        if len(gene_ids) != len(set(gene_ids)):
            raise ValueError("Duplicate values exist in atlas_gene_id. Please deduplicate them first")

        selected = pd.DataFrame(
            {
                "atlas_gene_id": np.asarray(gene_ids, dtype=np.int64),
                "_order": np.arange(len(gene_ids), dtype=np.int64),
            }
        )
        conn.register("_selected_varm_genes", selected)
        try:
            df = conn.execute(f"""
                SELECT {select_sql}
                FROM _selected_varm_genes AS s
                LEFT JOIN {table_sql} AS t
                  ON s.atlas_gene_id = t.atlas_gene_id
                ORDER BY s._order
            """).df()
        finally:
            conn.unregister("_selected_varm_genes")

    logger.info(f" get_varm_df Done, elapsed time: {(datetime.now() - start_time).total_seconds():.2f} seconds")

    return df.set_index("atlas_gene_id", drop=True)


def get_uns_df(
    atlas: Atlas,
    table_name: str,
    columns: list[str] | str | None = None,
) -> pd.DataFrame:
    """Read an ``uns_*`` table from the Atlas database.

    This function reads unstructured analysis result tables, such as
    ``uns_pca_stats`` or ``uns_umap_params``, into a pandas DataFrame.

    Parameters
    ----------
    atlas
        Atlas object. The object must already be connected to a DuckDB database.
    table_name
        Name of the ``uns_*`` table to read, for example ``"uns_pca_stats"``.
    columns
        Columns to read from the ``uns_*`` table. This can be a single string,
        a list of strings, or ``None``. If ``None``, all columns are read.

    Returns
    -------
    pandas.DataFrame
        Query result from the requested ``uns_*`` table.

    Examples
    --------
    Read PCA explained-variance statistics::

        pca_stats = atlas.get_uns_df("uns_pca_stats")

    Read stored UMAP parameters::

        umap_params = atlas.get_uns_df(
            "uns_umap_params",
            columns=["param_name", "param_value"],
        )
    """

    start_time = datetime.now()
    conn = atlas.connection

    if conn is None:
        raise ValueError("atlas.connection is None. Please connect to the database first")

    if not isinstance(table_name, str) or not table_name:
        raise ValueError("table_name must be a non-empty string")

    if not table_name.startswith("uns_"):
        raise ValueError(f"table_name must start with 'uns_': {table_name}")

    table_exists = conn.execute("""
        SELECT COUNT(*)
        FROM information_schema.tables
        WHERE table_name = ?
    """, [table_name]).fetchone()[0]

    if table_exists == 0:
        raise ValueError(f"The {table_name} table does not exist in the database")

    table_columns = [
        row[0]
        for row in conn.execute("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = ?
            ORDER BY ordinal_position
        """, [table_name]).fetchall()
    ]

    if columns is None:
        select_columns = table_columns
    else:
        if isinstance(columns, str):
            columns = [columns]

        missing = [c for c in columns if c not in table_columns]
        if missing:
            raise ValueError(f"These fields do not exist in the {table_name} table: {missing}")

        select_columns = list(columns)

    select_sql = ", ".join([f'"{c}"' for c in select_columns])
    table_sql = '"' + table_name.replace('"', '""') + '"'
    df = conn.execute(f"""
        SELECT {select_sql}
        FROM {table_sql}
    """).df()

    logger.info(f" get_uns_df Done, elapsed time: {(datetime.now() - start_time).total_seconds():.2f} seconds")

    return df


# Export a subset AnnData from DuckDB to memory according to an atlas_cell_id list
def get_anndata(
    atlas: Atlas,
    atlas_cell_ids: list[int] | np.ndarray | None,
    use_data: str = "data_count",
    include_obsm: bool = True,
    include_varm: bool = True,
) -> AnnData:
    """Construct an AnnData object from the Atlas database.

    This function exports an in-memory AnnData object from the Atlas database
    according to the user-provided ``atlas_cell_ids``. It preserves the order of
    the input cell IDs, reads the corresponding ``obs`` subset, the full ``var``,
    a sparse CSR ``X`` composed from the specified expression representation,
    and optionally reads ``obsm_*`` and ``varm_*`` result tables.

    This function is suitable for small-scale sampling export, local Scanpy analysis,
    model checking, or temporarily converting a group of cells in Atlas back to AnnData.

    Parameters
    ----------
    atlas
        Atlas object. The object must already be connected to a DuckDB database,
        and the database must contain at least the ``obs``, ``var``, and
        ``X_HyS_data`` tables.
    atlas_cell_ids
        List of Atlas cell IDs to export. It cannot be empty and cannot contain
        duplicate values.

        The order of cells in the returned AnnData object will be the same as
        the order of this list.
    use_data
        Expression field read from the resolved expression source. Common values
        include ``"data_count"``, ``"data_normalize"``, ``"data_log1p"``, and
        ``"data_scale"``.
    include_obsm
        Whether to write ``obsm_*`` result tables into the returned AnnData object.
    include_varm
        Whether to write ``varm_*`` result tables into the returned AnnData object.

    Returns
    -------
    AnnData
        AnnData object constructed from the Atlas database.

    Notes
    -----
    ``obsm_*`` tables are exported by left-joining according to the selected cell
    order. If some cells do not have embeddings, the corresponding positions are
    written as ``NaN``. ``varm_*`` tables are exported in full gene order.

    This function creates a temporary table ``_selected_cells`` to preserve the
    order of user-provided cells.

    Examples
    --------
    Export specified cells::

        cell_ids = [0, 1, 2, 3]
        adata = atlas.get_anndata(cell_ids, use_data="data_log1p")

    Export the first 5000 filtered cells and include UMAP/PCA::

        cell_ids = atlas.query(
            "SELECT atlas_cell_id FROM obs WHERE filter_cells = TRUE LIMIT 5000"
        )["atlas_cell_id"].tolist()
        adata = atlas.get_anndata(cell_ids, include_obsm=True, include_varm=True)
    """

    conn = atlas.connection

    if conn is None:
        raise ValueError("atlas.connection is None. Please connect to the database first")


    # 0. Basic checks
    if atlas_cell_ids is None or len(atlas_cell_ids) == 0:
        raise ValueError("atlas_cell_ids cannot be empty")

    atlas_cell_ids = [int(x) for x in atlas_cell_ids]

    if len(atlas_cell_ids) != len(set(atlas_cell_ids)):
        raise ValueError("Duplicate values exist in atlas_cell_ids. Please deduplicate them first")

    # Safe quoting for DuckDB identifiers
    def _q(name: str) -> str:
        """Add safe quoting for SQL identifiers.

        This internal helper is used in ``get_anndata`` to quote dynamic field names
        or table names, such as ``use_data``, ``obsm_*``, and ``varm_*``. The function
        escapes double quotes in the name and adds double quotes around the outside.

        Parameters
        ----------
        name
            SQL identifier to quote.

        Returns
        -------
        quoted_name
            SQL identifier wrapped in double quotes.

        Notes
        -----
        This function is only used for SQL identifiers, not for regular string values.
        """
        return '"' + name.replace('"', '""') + '"'

    expr_source = resolve_expression_source(conn, use_data)

    # 1. Create a temporary selected cell table to preserve the user input order
    selected_df = pd.DataFrame({
        "atlas_cell_id": atlas_cell_ids,
        "_cell_order": np.arange(len(atlas_cell_ids), dtype=np.int64),
    })

    conn.execute("DROP TABLE IF EXISTS _selected_cells")

    conn.register("_selected_cells_df", selected_df)
    conn.execute("""
        CREATE TEMP TABLE _selected_cells AS
        SELECT
            CAST(atlas_cell_id AS BIGINT) AS atlas_cell_id,
            CAST(_cell_order AS BIGINT) AS _cell_order
        FROM _selected_cells_df
    """)
    conn.unregister("_selected_cells_df")

    # 2. Read obs subset
    obs = conn.execute("""
        SELECT o.*
        FROM obs AS o
        JOIN _selected_cells AS s
          ON o.atlas_cell_id = s.atlas_cell_id
        ORDER BY s._cell_order
    """).df()

    if obs.shape[0] != len(atlas_cell_ids):
        found = set(obs["atlas_cell_id"].astype(int).tolist())
        missing = [x for x in atlas_cell_ids if x not in found]
        raise ValueError(
            f"{len(missing)} atlas_cell_id values do not exist in obs, "
            f"for example: {missing[:10]}"
        )

    if "atlas_cell_name" not in obs.columns:
        raise ValueError("The atlas_cell_name field does not exist in the obs table, so it cannot be used as the AnnData obs index")

    obs = obs.set_index("atlas_cell_name", drop=False)
    obs.index = obs.index.astype(str)

    var = conn.execute("""
        SELECT *
        FROM var
        ORDER BY atlas_gene_id
    """).df()

    if "atlas_gene_id" not in var.columns:
        raise ValueError("The atlas_gene_id field does not exist in the var table")

    if "atlas_gene_name" not in var.columns:
        raise ValueError("The atlas_gene_name field does not exist in the var table, so it cannot be used as the AnnData var index")

    var = var.set_index("atlas_gene_name", drop=False)
    var.index = var.index.astype(str)

    n_cells = obs.shape[0]
    n_genes = var.shape[0]

    # 4. Read X subset and assemble CSR
    x_sql = f"""
        SELECT
            s._cell_order AS row_id,
            {expr_source.gene_sql} AS col_id,
            {expr_source.value_sql} AS value
        FROM {expr_source.from_sql}
        JOIN _selected_cells AS s
          ON {expr_source.cell_sql} = s.atlas_cell_id
        WHERE {expr_source.value_sql} IS NOT NULL
        ORDER BY s._cell_order, {expr_source.gene_sql}
    """

    x_df = conn.execute(x_sql).df()

    if x_df.shape[0] == 0:
        X = sparse.csr_matrix((n_cells, n_genes), dtype=np.float32)
        nnz = 0
    else:
        rows = x_df["row_id"].to_numpy(dtype=np.int64)
        cols = x_df["col_id"].to_numpy(dtype=np.int64)
        vals = x_df["value"].to_numpy(dtype=np.float32)

        X = sparse.csr_matrix(
            (vals, (rows, cols)),
            shape=(n_cells, n_genes),
            dtype=np.float32,
        )

        X.sort_indices()
        nnz = X.nnz

    # 5. Create AnnData
    adata = AnnData(
        X=X,
        obs=obs,
        var=var,
    )

    # 6. Read obsm subset
    if include_obsm:

        obsm_tables = conn.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_name LIKE 'obsm_%'
            ORDER BY table_name
        """).fetchall()

        for (table_name,) in obsm_tables:
            key = table_name.replace("obsm_", "")

            # ============================================================
            #  Use _selected_cells as the base and LEFT JOIN the obsm table:
            #   cells with embeddings     -> keep original values
            #   cells without embeddings  -> NaN
            # ============================================================

            value_cols = [
                row[0]
                for row in conn.execute("""
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_name = ?
                      AND column_name <> 'atlas_cell_id'
                    ORDER BY ordinal_position
                """, [table_name]).fetchall()
            ]

            if len(value_cols) == 0:
                continue

            select_values = ", ".join([
                f"t.{_q(c)} AS {_q(c)}"
                for c in value_cols
            ])

            df = conn.execute(f"""
                SELECT
                    {select_values}
                FROM _selected_cells AS s
                LEFT JOIN {_q(table_name)} AS t
                  ON s.atlas_cell_id = t.atlas_cell_id
                ORDER BY s._cell_order
            """).df()

            arr = df.to_numpy(dtype=np.float32)

            if arr.shape[0] != n_cells:
                raise ValueError(
                    f"obsm[{key}] row count error: {arr.shape[0]} != selected cells {n_cells}"
                )

            adata.obsm[key] = arr

    # 7. Read full varm
    if include_varm:

        varm_tables = conn.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_name LIKE 'varm_%'
            ORDER BY table_name
        """).fetchall()

        for (table_name,) in varm_tables:
            key = table_name.replace("varm_", "")

            value_cols = [
                row[0]
                for row in conn.execute("""
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_name = ?
                      AND column_name <> 'atlas_gene_id'
                    ORDER BY ordinal_position
                """, [table_name]).fetchall()
            ]

            if len(value_cols) == 0:
                continue

            select_values = ", ".join([
                f"t.{_q(c)} AS {_q(c)}"
                for c in value_cols
            ])

            df = conn.execute(f"""
                SELECT
                    {select_values}
                FROM var AS v
                LEFT JOIN {_q(table_name)} AS t
                  ON v.atlas_gene_id = t.atlas_gene_id
                ORDER BY v.atlas_gene_id
            """).df()

            arr = df.to_numpy(dtype=np.float32)

            if arr.shape[0] != n_genes:
                raise ValueError(
                    f"varm[{key}] row count error: {arr.shape[0]} != genes {n_genes}"
                )

            adata.varm[key] = arr

    # 8. Clean up temporary table
    conn.execute("DROP TABLE IF EXISTS _selected_cells")

    logger.info(" AnnData export completed")
    logger.info(f"  - cells: {adata.n_obs:,}")
    logger.info(f"  - genes: {adata.n_vars:,}")

    return adata


# Write AnnData to h5ad
def _write_dataframe(f: h5py.File, key: str, df: pd.DataFrame):

    """Write a pandas DataFrame using the AnnData dataframe encoding.

    This internal function is used by ``write_h5ad`` and is responsible for writing
    pandas DataFrames such as ``obs`` or ``var`` into an already opened HDF5 file.
    The function creates the corresponding HDF5 group, writes the index, column data,
    column order, and AnnData dataframe encoding attributes, so that the exported file
    can be properly read by AnnData/Scanpy.

    Parameters
    ----------
    f
        Open HDF5 file handle.
    key
        HDF5 group name, such as ``"obs"`` or ``"var"``.
    df
        DataFrame to write into h5ad.

    Returns
    -------
    None
        The DataFrame is written directly into the HDF5 file and no object is returned.

    Notes
    -----
    String columns are written using UTF-8 variable-length strings. pandas categorical
    columns are first converted to strings. This implementation is a lightweight writer
    and does not expand the full AnnData category encoding for categorical columns.
    """
    g = f.create_group(key)

    # ---- AnnData dataframe metadata ----
    g.attrs["encoding-type"] = "dataframe"
    g.attrs["encoding-version"] = "0.2.0"

    # index
    index_name = df.index.name or "_index"
    index_data = np.array(df.index.astype(str).tolist(), dtype=object)

    g.create_dataset(
        index_name,
        data=index_data,
        dtype=h5py.string_dtype(encoding="utf-8"),
    )

    # columns
    colnames = []

    for col in df.columns:
        colnames.append(col)
        series = df[col]

        # pandas categorical -> string
        if pd.api.types.is_categorical_dtype(series):
            series = series.astype(str)

        arr = series.to_numpy()

        # ===== string columns =====
        if arr.dtype.kind in {"U", "O"}:
            data = np.array(series.astype(str).tolist(), dtype=object)

            g.create_dataset(
                col,
                data=data,
                dtype=h5py.string_dtype(encoding="utf-8"),
            )

        # ===== numeric columns =====
        else:
            g.create_dataset(col, data=arr)

    # AnnData spec attrs (key)
    g.attrs["column-order"] = np.array(colnames, dtype="S")
    g.attrs["_index"] = index_name
