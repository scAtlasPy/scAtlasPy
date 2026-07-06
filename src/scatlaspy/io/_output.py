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
import logging
logger = logging.getLogger("Atlas")
logger.addHandler(logging.NullHandler())


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
    the internal Atlas HyS sparse structure. ``X.data`` comes from the field
    specified by ``use_data`` in the ``X_HyS_data`` table, ``X.indices`` comes
    from ``atlas_gene_id``, and ``X.indptr`` comes from ``X_HyS_indptr``.

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
        Expression value field exported from the ``X_HyS_data`` table. The default
        is ``"data_count"``.
        Existing fields such as ``"data_log1p"`` and ``"data_normalize"`` can
        also be used.

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

    Before export, the function checks whether ``use_data`` exists in the
    ``X_HyS_data`` table. If it does not exist, an error is raised directly to
    avoid exporting an empty matrix or an incorrect field.

    Examples
    --------
    Export the current database::

        atlas.write_h5ad(r"F:\\data\\pbmc_export.h5ad")

    Use the object-style API and reduce per-batch memory usage::

        atlas.write_h5ad(r"F:\\data\\pbmc_export.h5ad", batch_cells=200000)

    Export the log1p expression matrix::

        atlas.write_h5ad(r"F:\\data\\pbmc_log1p.h5ad", use_data="data_log1p")"""

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

    x_field_exists = conn.execute(
        """
        SELECT COUNT(*)
        FROM information_schema.columns
        WHERE table_name = 'X_HyS_data'
          AND column_name = ?
        """,
        [use_data],
    ).fetchone()[0]

    if x_field_exists == 0:
        raise ValueError(f"The field does not exist in X_HyS_data: {use_data}")

    use_data_sql = _q(use_data)

    # Read obs / var

    obs = conn.execute("SELECT * FROM obs ORDER BY atlas_cell_id").df()
    var = conn.execute("SELECT * FROM var ORDER BY atlas_gene_id").df()

    obs = obs.set_index("atlas_cell_id")
    var = var.set_index("atlas_gene_id")

    n_cells = obs.shape[0]
    n_genes = var.shape[0]

    # Read CSR indptr
    indptr_df = conn.execute("""
        SELECT indptr
        FROM X_HyS_indptr
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
                SELECT atlas_gene_id, {use_data_sql}
                FROM X_HyS_data
                WHERE id >= ? AND id < ?
                ORDER BY id
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
    The returned result uses ``atlas_cell_id`` as the pandas index while also
    preserving the ``atlas_cell_id`` column itself.

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
        Query result from ``obs``. The default index is ``atlas_cell_id``.

    Notes
    -----
    Even if ``atlas_cell_id`` is not explicitly included in ``columns``, the
    function automatically places ``atlas_cell_id`` as the first column to set
    the DataFrame index.

    Examples
    --------
    Read all obs information::

        obs = atlas.get_obs_df()

    Read only clustering and automatic annotation columns::

        obs = atlas.get_obs_df(columns=["kmeans", "cell_type_auto"])"""

    start_time = datetime.now()

    conn = atlas.connection

    if conn is None:
        raise ValueError("atlas.connection is None. Please connect to the database first")

    # 1. Check whether the obs table exists
    obs_exists = conn.execute("""
        SELECT COUNT(*)
        FROM information_schema.tables
        WHERE table_name = 'obs'
    """).fetchone()[0]

    if obs_exists == 0:
        raise ValueError("The obs table does not exist in the database")

    # 2. Get all fields in obs
    obs_columns = [
        row[0]
        for row in conn.execute("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'obs'
            ORDER BY ordinal_position
        """).fetchall()
    ]

    if "atlas_cell_id" not in obs_columns:
        raise ValueError("The atlas_cell_id field does not exist in the obs table, so the pandas index cannot be set")

    # 3. Process columns
    if columns is None:
        select_columns = obs_columns
    else:
        if isinstance(columns, str):
            columns = [columns]

        # Check whether fields exist
        missing = [c for c in columns if c not in obs_columns]
        if missing:
            raise ValueError(f"These fields do not exist in the obs table: {missing}")

        # atlas_cell_id must be included and placed as the first column
        select_columns = ["atlas_cell_id"] + [
            c for c in columns
            if c != "atlas_cell_id"
        ]

    # 4. Query obs
    select_sql = ", ".join([f'"{c}"' for c in select_columns])

    sql = f"""
        SELECT {select_sql}
        FROM obs
    """

    df = conn.execute(sql).df()

    # 5. Use atlas_cell_id as the pandas index by default
    df = df.set_index("atlas_cell_id", drop=False)

    logger.info(f" get_obs_df Done, elapsed time: {(datetime.now() - start_time).total_seconds():.2f} seconds")

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
    a sparse CSR ``X`` composed from the specified expression field, and optionally
    reads ``obsm_*`` and ``varm_*`` result tables.

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
        Expression field read from the ``X_HyS_data`` table. Common values include
        ``"data_count"``, ``"data_normalize"``, ``"data_log1p"``, and ``"data_scale"``.
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

    # Check whether use_data exists
    x_field_exists = conn.execute(
        """
        SELECT COUNT(*)
        FROM information_schema.columns
        WHERE table_name = 'X_HyS_data'
          AND column_name = ?
        """,
        [use_data],
    ).fetchone()[0]

    if x_field_exists == 0:
        raise ValueError(f"The field does not exist in X_HyS_data: {use_data}")

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
            x.atlas_gene_id AS col_id,
            x.{_q(use_data)} AS value
        FROM X_HyS_data AS x
        JOIN _selected_cells AS s
          ON x.atlas_cell_id = s.atlas_cell_id
        ORDER BY s._cell_order, x.atlas_gene_id
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
