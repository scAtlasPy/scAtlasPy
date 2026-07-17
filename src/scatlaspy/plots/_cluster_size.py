import matplotlib.pyplot as plt
from datetime import datetime
from os import PathLike
from ..data import Atlas


def cluster_size(
    atlas: Atlas,
    use_obs_col: str = "scatlas_cluster",
    figsize: tuple[float, float] | None=(7, 4),
    show_percent: bool = True,
    title: str | None = None,
    save_path: PathLike[str] | str | None = None,
    dpi: int = 300,
) -> None:
    """Plot the cell count distribution of a grouping column.

    This function reads the grouping column specified by ``use_obs_col`` from the
    ``obs`` table, counts the number of cells in each group, and displays the group
    sizes as a bar plot. By default, values are shown as percentages, but they can
    also be shown as raw cell counts.

    This plot is commonly used to check whether clustering results are overly
    imbalanced, whether very small clusters exist, or whether the group sizes under
    different clustering parameters meet expectations. It can be used for any
    ``obs`` grouping column such as ``scatlas_cluster``, ``leiden``, or
    ``cell_type``, as long as the column can be displayed as discrete labels.

    Parameters
    ----------
    atlas
        Atlas object. It must already be connected to a DuckDB database, and the
        database must contain the ``obs`` table.
    use_obs_col
        Column name in ``obs`` used to count group sizes. By default,
        ``"scatlas_cluster"`` is read.
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
    dpi
        Resolution used when saving the figure.

    Returns
    -------
    None

    Examples
    --------
    Plot the default distilled Louvain cluster sizes::

        sap.tl.graph_clustering(atlas)
        sap.pl.cluster_size(atlas)

    Plot a custom grouping column and display percentages::

        sap.pl.cluster_size(
            atlas,
            use_obs_col="cell_type",
            show_percent=True,
            title="Cell type distribution",
        )"""

    start = datetime.now()
    conn = atlas.connection

    # Check whether the grouping column exists in obs.
    obs_cols = [r[1] for r in conn.execute("PRAGMA table_info(obs)").fetchall()]
    if use_obs_col not in obs_cols:
        raise ValueError(
            f"The column does not exist in obs: {use_obs_col}\n"
            "Please run a clustering or annotation step before plotting group sizes."
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
            "Please run a clustering or annotation step before plotting group sizes."
        )

    df["cluster"] = df["cluster"].astype(str)
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

    ax.set_xlabel(use_obs_col, fontsize=13)
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
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")

    plt.show()
