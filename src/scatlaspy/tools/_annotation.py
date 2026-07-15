from collections.abc import Mapping, Sequence
import pandas as pd
from ..data import Atlas


def manual_annotate_clusters(
    atlas: Atlas,
    cluster_to_cell_type: Mapping[str, str] | Sequence[str],
    groupby: str = "kmeans",
    obs_col: str = "cell_type_manual",
    table_name: str | None = "manual_cluster_annotation",
    unknown_label: str | None = None,
    return_df: bool = True,
) -> pd.DataFrame | None:
    """Manually add cell-type annotations to clusters.

    After inspecting marker genes, rank genes plots, and UMAP plots, the user
    manually specifies the cell type name corresponding to each cluster, and
    these labels are written back to the ``obs`` table.

    The function supports two input formats: one is an explicit dictionary that directly specifies the
    cell type corresponding to each cluster ID; the other is an ordered list, where labels are written
    sequentially according to the sorting order of the actual clusters in ``obs[groupby]``. At runtime, the function checks
    whether clusters exist, whether there are duplicates, and whether the list length is consistent with the number of clusters.

    Parameters
    ----------
    atlas
        Atlas object. It must already be connected to a DuckDB database containing the ``obs`` table.
        If it is not connected yet, the function automatically connects in ``r+`` mode.

    cluster_to_cell_type
        User-provided manual annotation result.

        If a dictionary is passed, the dictionary key is the cluster ID and the value is the cell type name.
        For example::

            {
                "0": "CD4 T cells",
                "1": "CD14+ Monocytes",
                "2": "B cells",
            }

        If a list or tuple is passed, labels are assigned sequentially according to the sorting order of cluster IDs
        in ``obs[groupby]``.
        For example::

            [
                "CD4 T cells",
                "CD14+ Monocytes",
                "B cells",
            ]

    groupby
        Column name in ``obs`` that stores clustering results.

        If kmeans clustering is used, this is usually ``"kmeans"``.
        If the current database uses KMeans clustering, it should be set to ``"kmeans"``.

    obs_col
        ``obs`` column name used to write manual cell-type annotations.
        By default, annotations are written to ``obs.cell_type_manual``.

    table_name
        Optional database table name used to save the mapping from cluster to cell type.
        If set to ``None``, no additional mapping table is saved.

    unknown_label
        Label used for clusters without manual annotations.

        If ``None``, unannotated clusters are written as ``NULL``.
        If set to ``"Unknown"``, unannotated clusters are written as ``"Unknown"``.

    return_df
        Whether to return the annotation result summary table.

    Returns
    -------
    pandas.DataFrame or None
        If ``return_df=True``, returns the manual annotation result and cell count for each cluster.
        If ``return_df=False``, returns ``None``.

    Notes
    -----
    This function modifies the ``obs`` table: if ``obs_col`` does not exist, a text column is automatically added;
    if it already exists, old values are first cleared and then the manual annotations provided in this run are written.

    When ``table_name`` is not ``None``, the function also saves a mapping table from cluster to cell type,
    making it easier to track the labels used in this manual annotation.

    Clusters without provided labels are written as ``NULL`` or the specified unknown label according to ``unknown_label``;
    clusters that are provided but do not exist in ``obs[groupby]`` raise an error directly.

    Examples
    --------
    Use an explicit dictionary for manual annotation::

        mapping = {
            "0": "CD4 T cells",
            "1": "CD14+ Monocytes",
            "2": "B cells",
            "3": "CD8 T cells",
            "4": "NK cells",
            "5": "FCGR3A+ Monocytes",
            "6": "Dendritic Cells",
            "7": "Megakaryocytes",
        }

        summary = sap.tl.manual_annotate_clusters(
            atlas,
            mapping,
            groupby="kmeans",
            obs_col="cell_type_manual",
        )

    Use an ordered list for manual annotation::

        new_cluster_names = [
            "CD4 T cells",
            "CD14+ Monocytes",
            "B cells",
            "CD8 T cells",
            "NK cells",
            "FCGR3A+ Monocytes",
            "Dendritic Cells",
            "Megakaryocytes",
        ]

        summary = sap.tl.manual_annotate_clusters(
            atlas,
            new_cluster_names,
            groupby="kmeans",
        )

    Plot the UMAP after manual annotation::
        sap.pl.umap(atlas, color="cell_type_manual")
    """

    # Get the database connection
    conn = atlas.connection
    if conn is None:
        atlas.connect("r+")
        conn = atlas.connection

    # Check whether the groupby column exists in the obs table
    obs_cols = [r[1] for r in conn.execute("PRAGMA table_info(obs)").fetchall()]

    if groupby not in obs_cols:
        raise ValueError(f"Clustering column does not exist in obs: {groupby!r}")

    group_col = _quote_identifier(groupby)
    obs_label_col = _quote_identifier(obs_col)

    # Read the cluster IDs that actually exist in obs: numeric clusters are sorted numerically
    cluster_df = conn.execute(
        f"""
        SELECT CAST({group_col} AS TEXT) AS cluster_id
        FROM obs
        WHERE {group_col} IS NOT NULL
        GROUP BY 1
        ORDER BY TRY_CAST(cluster_id AS INTEGER), cluster_id
        """
    ).df()

    if len(cluster_df) == 0:
        raise ValueError(f"No cluster was found in obs.{groupby}")

    cluster_ids = cluster_df["cluster_id"].astype(str).tolist()

    # ============================================================
    # Organize the user-provided manual annotations
    #    Supports two formats:
    #    1) Dictionary: {"0": "CD4 T cells", "1": "B cells"}
    #    2) List: ["CD4 T cells", "B cells", ...]
    # ============================================================

    if isinstance(cluster_to_cell_type, Mapping):
        mapping_df = pd.DataFrame({
            "cluster_id": [str(k) for k in cluster_to_cell_type.keys()],
            "cell_type": [str(v) for v in cluster_to_cell_type.values()],
        })

    else:
        if isinstance(cluster_to_cell_type, (str, bytes)):
            raise TypeError(
                "cluster_to_cell_type must be a dictionary or a list / tuple of cell type names; "
                "it cannot be a single string."
            )

        labels = [str(v) for v in cluster_to_cell_type]

        if len(labels) != len(cluster_ids):
            raise ValueError(
                "The number of manual annotation labels is inconsistent with the number of clusters: "
                f"you provided {len(labels)} labels, "
                f"but obs.{groupby} contains {len(cluster_ids)} clusters. "
                f"The current cluster order is: {cluster_ids}"
            )

        mapping_df = pd.DataFrame({
            "cluster_id": cluster_ids,
            "cell_type": labels,
        })

    # Check whether there are duplicated cluster IDs
    duplicated = mapping_df["cluster_id"][mapping_df["cluster_id"].duplicated()].tolist()

    if duplicated:
        raise ValueError(f"Duplicated cluster IDs exist in manual annotations: {duplicated}")

    # Check whether the clusters provided by the user actually exist in obs[groupby]
    unknown_clusters = sorted(set(mapping_df["cluster_id"]) - set(cluster_ids))

    if unknown_clusters:
        raise ValueError(
            f"Manual annotations contain clusters that do not exist in obs.{groupby}: {unknown_clusters}"
        )

    # Indicate which clusters do not have manual annotations
    #    These clusters will be written as NULL or unknown_label
    missing_clusters = sorted(set(cluster_ids) - set(mapping_df["cluster_id"]))

    if missing_clusters:
        if unknown_label is None:
            print(
                f"[manual_annotate_clusters] Note: the following clusters do not have manual annotations provided, "
                f"and will be written as NULL: {missing_clusters}"
            )
        else:
            print(
                f"[manual_annotate_clusters] Note: the following clusters do not have manual annotations provided, "
                f"and will be written as {unknown_label!r}: {missing_clusters}"
            )

    # If obs_col does not exist, create this column
    if obs_col not in obs_cols:
        conn.execute(f"ALTER TABLE obs ADD COLUMN {obs_label_col} TEXT")

    # Clear old annotations before each run
    #    This way, the next run overwrites the previous one and does not keep old labels
    if unknown_label is None:
        conn.execute(f"UPDATE obs SET {obs_label_col} = NULL")
    else:
        conn.execute(f"UPDATE obs SET {obs_label_col} = ?", [str(unknown_label)])

    # Register a temporary mapping table and write the manual annotations back to obs
    conn.register("_manual_annotation_tmp", mapping_df)

    try:
        conn.execute(
            f"""
            UPDATE obs AS o
            SET {obs_label_col} = m.cell_type
            FROM _manual_annotation_tmp AS m
            WHERE CAST(o.{group_col} AS TEXT) = CAST(m.cluster_id AS TEXT)
            """
        )

        # Save the mapping table from cluster -> cell type
        if table_name is not None:
            table = _quote_identifier(table_name)

            saved_mapping_df = mapping_df.copy()
            saved_mapping_df.insert(0, "groupby", groupby)
            saved_mapping_df["obs_col"] = obs_col

            conn.register("_manual_annotation_save_tmp", saved_mapping_df)

            try:
                conn.execute(f"DROP TABLE IF EXISTS {table}")
                conn.execute(f"CREATE TABLE {table} AS SELECT * FROM _manual_annotation_save_tmp")
            finally:
                conn.unregister("_manual_annotation_save_tmp")

    finally:
        conn.unregister("_manual_annotation_tmp")

    # Save the database state
    conn.execute("CHECKPOINT")

    # Whether to return the summary result
    if not return_df:
        return None

    return conn.execute(
        f"""
        SELECT
            CAST({group_col} AS TEXT) AS cluster_id,
            {obs_label_col} AS cell_type,
            COUNT(*) AS n_cells
        FROM obs
        GROUP BY 1, 2
        ORDER BY TRY_CAST(cluster_id AS INTEGER), cluster_id, cell_type
        """
    ).df()


def _quote_identifier(name: str) -> str:
    """Add safe quoting for DuckDB SQL identifiers.

    This internal helper is used to quote dynamically passed table names or field names, such as ``groupby``,
    ``obs_col``, and ``table_name``. The function escapes existing double quotes in the name and adds
    double quotes around the outside, avoiding SQL parsing failures when field names contain special characters, spaces, or DuckDB keywords.

    Parameters
    ----------
    name
        SQL table name, column name, or other identifier that needs to be quoted.

    Returns
    -------
    str
        SQL identifier after adding double quotes.

    Notes
    -----
    This function is only used for SQL identifiers and is not used to quote ordinary string values. String values should be passed using DuckDB
    parameter binding.
    """
    return '"' + str(name).replace('"', '""') + '"'
