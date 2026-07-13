import matplotlib.pyplot as plt
from datetime import datetime
from os import PathLike
from ..data import Atlas


def kmeans_cluster_size(
    atlas: Atlas,
    use_obs_col: str = "kmeans",
    figsize: tuple[float, float] | None=(7, 4),
    show_percent: bool = True,
    title: str | None = None,
    save_path: PathLike[str] | str | None = None
) -> None:

    """Plot the cell count distribution of K-means or another grouping column.

    This function reads the grouping column specified by ``use_obs_col`` from the
    ``obs`` table, counts the number of cells in each group, and displays the group
    sizes as a bar plot. By default, values are shown as percentages, but they can
    also be shown as raw cell counts.

    This plot is commonly used to check whether clustering results are overly
    imbalanced, whether very small clusters exist, or whether the group sizes under
    different clustering parameters meet expectations. Although the default column
    name is ``"kmeans"``, it can also be used for any ``obs`` grouping column such
    as ``leiden`` or ``cell_type``, as long as the column can be converted to integer
    labels.

    Parameters
    ----------
    atlas
        Atlas object. It must already be connected to a DuckDB database, and the
        database must contain the ``obs`` table.
    use_obs_col
        Column name in ``obs`` used to count group sizes. By default, ``"kmeans"``
        is read.
    figsize
        Matplotlib figure size. Defaults to ``(7, 4)``.
    show_percent
        Whether to display the y-axis as the percentage of cells. If ``False``, the
        raw number of cells in each group is displayed.
    title
        Figure title. If ``None``, ``"{use_obs_col} cluster distribution"`` is used
        automatically.
    save_path
        Path for saving the figure. If ``None``, the figure is only displayed and
        not saved.

    Returns
    -------
    None

    Examples
    --------
    Plot the default K-means cluster sizes::

        sap.tl.kmeans(atlas, n_clusters=20)
        sap.pl.kmeans_cluster_size(atlas)

    Plot a custom clustering column and display percentages::

        sap.pl.kmeans_cluster_size(
            atlas,
            use_obs_col="kmeans_50",
            show_percent=True,
            title="K-means 50 cluster size",
        )"""

    start = datetime.now()
    conn = atlas.connection

    # Check whether the kmeans column exists in obs
    obs_cols = [r[1] for r in conn.execute("PRAGMA table_info(obs)").fetchall()]
    if use_obs_col not in obs_cols:
        raise ValueError(
            f"The column does not exist in obs: {use_obs_col}\n"
            f"Please run sap.tl.kmeans(atlas, use_obs_col='{use_obs_col}') first"
        )

    # Count clusters
    df = conn.execute(f"""
        SELECT
            {use_obs_col} AS cluster,
            COUNT(*) AS n_cells
        FROM obs
        WHERE {use_obs_col} IS NOT NULL
        GROUP BY {use_obs_col}
        ORDER BY {use_obs_col}
    """).fetchdf()

    if len(df) == 0:
        raise ValueError(
            f"No available clustering results found in obs.{use_obs_col}.\n"
            f"Please run sap.tl.kmeans(atlas) first"
        )

    df["cluster"] = df["cluster"].astype(int).astype(str)
    df["pct"] = df["n_cells"] / df["n_cells"].sum() * 100

    # Plot
    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    ax.set_facecolor("white")

    y = df["pct"].to_numpy() if show_percent else df["n_cells"].to_numpy()

    ax.bar(
        df["cluster"].to_numpy(),
        y,
        width=0.75
    )

    ax.set_xlabel("KMeans cluster", fontsize=13)
    ax.set_ylabel("Percent of cells (%)" if show_percent else "Number of cells", fontsize=13)

    if title is None:
        title = f"{use_obs_col} cluster distribution"

    ax.set_title(title, fontsize=14, pad=10)

    ax.grid(True, axis="y", alpha=0.3)
    ax.grid(False, axis="x")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.tick_params(axis="both", labelsize=11)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()
