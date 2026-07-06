from ..data import Atlas
from matplotlib.lines import Line2D
import numpy as np
import re
import math
from os import PathLike
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from typing import Any
import logging
logger = logging.getLogger("Atlas")
logger.addHandler(logging.NullHandler())

# =====================================================
# Unified discrete categorical color palette pool
# -----------------------------------------------------
# Used for coloring obs categorical variables, such as:
# kmeans / cell_type / batch / organ, etc.
#
# These palettes together provide about 100 discrete colors:
# tab20(20) + tab20b(20) + tab20c(20)
# + Set3(12) + Paired(12) + Accent(8) + Dark2(8)
# =====================================================
DEFAULT_DISCRETE_PALETTES = (
    "tab20",
    "tab20b",
    "tab20c",
    "Set3",
    "Paired",
    "Accent",
    "Dark2",
)


# =====================================================
# General natural sorting for categorical labels
# -----------------------------------------------------
# Solves:
# embryo_1, embryo_10, embryo_11, embryo_2
#
# Sorted as:
# embryo_1, embryo_2, embryo_3, ..., embryo_10
#
# Also works for:
# cluster_1 / cluster_10
# batch2 / batch10
# group_3_day_2 / group_3_day_12
# =====================================================
_MISSING_CATEGORY_LABELS = {"", "na", "nan", "none", "<na>", "null"}


def umap(
        atlas: Atlas,
        color: str | list[str] = "kmeans",
        *,
        sample_n: int | None = None,
        where: str | None = None,

        # gene feature parameters
        use_data: str = "data_log1p",

        # Plotting parameters
        figsize: tuple[float, float] | None = (22, 8),
        point_size: float = 5,
        alpha: float = 0.85,
        cmap: str = "viridis",
        palette: str | list[str] | tuple[str, ...] | None = DEFAULT_DISCRETE_PALETTES,

        # legend / layout
        legend_loc: str | None = "right_margin",
        ncols: int = 3,
        frameon: bool = True,

        # Large-data / output parameters
        plot_batch_size: int = 200000,
        save_path: PathLike[str] | str | None = None,
) -> None:

    """Plot cell UMAP embeddings.

    This function reads UMAP coordinates from the ``obsm_X_umap`` table and draws one or more UMAP panels according to ``color``.
    ``color`` can be a field in the ``obs`` table or a gene name in ``var.atlas_gene_name``;
    when a list contains both obs fields and gene names, the mixed multi-panel plotting logic is used automatically.

    This plot is similar to Scanpy ``sc.pl.umap`` and is commonly used to inspect clustering, cell type annotations, QC metrics, or
    the distribution of marker gene expression in UMAP space.

    Parameters
    ----------
    atlas
        Atlas object. It must already be connected to a DuckDB database, and UMAP must have been run to generate the ``obsm_X_umap`` table.

    color
        Name or list of names used to color scatter points. Each element can be an ``obs`` table column name or
        a gene name. A single ``obs`` column uses obs plotting; one or more gene names use gene feature plotting;
        a mixed list uses multi-panel mixed plotting.

    sample_n
        Maximum number of cells sampled for plotting. If ``None``, all cells are used.

    where
        Additional SQL filtering condition used to restrict cells participating in plotting.
        If ``None``, no additional condition is added.

    use_data
        Expression value field read from ``X_HyS_data`` when ``color`` contains gene names, such as
        ``"data_log1p"``, ``"data_count"``, or ``"data_scale"``.

    figsize
        Figure size. If ``None``, the function default size is used.

    point_size
        Scatter point size.

    alpha
        Transparency of graphical elements.

    cmap
        Matplotlib colormap name used for continuous variables.

    palette
        Color list or palette used for discrete variables.

    legend_loc
        Legend location for discrete categories. ``"right_margin"`` places the legend in the right-side margin;
        ``"on_data"`` places category labels on the UMAP point cloud.

    ncols
        Number of subplots per row for multi-panel plotting.

    frameon
        Whether to show the axis frame.

    plot_batch_size
        Number of cells read from DuckDB per batch during large-data plotting. Mainly used for discrete obs streaming plots.

    save_path
        Path for saving the figure. If ``None``, the figure is only displayed.

    Returns
    -------
    None

    Examples
    --------
    Plot UMAP colored by K-means clusters::

        sap.tl.umap(atlas)
        sap.pl.umap(atlas, color="kmeans")

    Plot by marker gene expression and save the figure::

        sap.pl.umap(
            atlas,
            color="MS4A1",
            use_data="data_log1p",
            save_path=r"F:\\figures\\umap_MS4A1.png",
        )

    Plot obs grouping and gene expression at the same time::

        sap.pl.umap(
            atlas,
            color=["cell_type_auto", "MS4A1", "CD3D"],
            point_size=0.5,
        )"""

    conn = atlas.connection

    # Normalize parameters
    if isinstance(color, str):
        color_list = [color]
    else:
        color_list = list(color)

    if len(color_list) == 0:
        raise ValueError("color cannot be empty")

    if where is not None and str(where).strip() != "":
        logger.info(f"[UMAP] where = {where}")

    # Check whether obsm_X_umap exists
    tables = conn.execute("""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_name = 'obsm_X_umap'
    """).fetchdf()

    if len(tables) == 0:
        raise ValueError(
            "obsm_X_umap does not exist in the database.\n"
            "Please run sap.tl.umap(atlas) first"
        )

    # Get obs columns and gene names
    obs_cols = [r[1] for r in conn.execute("PRAGMA table_info(obs)").fetchall()]

    gene_df = conn.execute("""
        SELECT atlas_gene_name
        FROM var
    """).fetchdf()

    gene_set = set(gene_df["atlas_gene_name"].astype(str).tolist())

    # Determine color type
    obs_colors = []
    gene_colors = []

    for c in color_list:
        c = str(c)

        if c in obs_cols:
            obs_colors.append(c)

        elif c in gene_set:
            gene_colors.append(c)

        else:
            raise ValueError(
                f"color='{c}' is neither an obs column nor a gene name.\n"
                f"Please confirm that this field exists in obs or var."
            )

    # Single obs categorical plot
    if len(color_list) == 1 and len(obs_colors) == 1:
        _plot_umap_obs(
            atlas=atlas,
            color=obs_colors[0],
            sample_n=sample_n,
            where=where,
            legend_loc=legend_loc,
            figsize=figsize,
            point_size=point_size,
            alpha=alpha,
            cmap=cmap,
            palette=palette,
            frameon=frameon,
            save_path=save_path,
            plot_batch_size=plot_batch_size,
        )
        return None

    # Pure gene feature plot
    if len(obs_colors) == 0 and len(gene_colors) > 0:
        _plot_umap_features(
            atlas=atlas,
            genes=gene_colors,
            sample_n=sample_n,
            where=where,
            use_data=use_data,
            ncols=ncols,
            figsize=figsize,
            point_size=point_size,
            alpha=alpha,
            cmap=cmap,
            save_path=save_path,
        )
        return None

    # Mixed mode
    _plot_umap_mixed(
        atlas=atlas,
        obs_colors=obs_colors,
        gene_colors=gene_colors,
        sample_n=sample_n,
        where=where,
        use_data=use_data,
        ncols=ncols,
        figsize=figsize,
        point_size=point_size,
        alpha=alpha,
        cmap=cmap,
        palette=palette,
        frameon=frameon,
        save_path=save_path,
    )
    return None



# umap() - if color is an obs column -> plot_umap_obs()
def _plot_umap_obs(
        atlas: Atlas,
        color: str = "kmeans",
        sample_n: int | None = 50000,
        groups: list | None = None,
        where: str | None = None,
        legend_loc: str = "right_margin",
        title: str | None = None,
        figsize: tuple[float, float] | None=(22, 8),
        point_size: float = 1.0,
        alpha: float = 0.7,
        cmap: str = "viridis",
        palette: str | list[str] | tuple[str, ...] | None = DEFAULT_DISCRETE_PALETTES,
        frameon: bool = True,
        save_path: PathLike[str] | str | None = None,
        plot_batch_size: int = 200000,
        return_df: bool = False,
):

    """Plot UMAP by a single ``obs`` field.

    This internal function reads UMAP coordinates from ``obsm_X_umap`` and reads the
    cell-level field specified by ``color`` from ``obs`` for coloring.
    Numeric fields use a continuous colormap; string, boolean, or categorical fields use
    discrete colors and a legend.

    When the cell count for discrete categories is large and no DataFrame needs to be returned, this function delegates to
    ``_draw_umap_obs_streaming`` to read and plot data in batches, reducing the amount of data loaded at once.

    Parameters
    ----------
    atlas
        Atlas object. It must already be connected to a DuckDB database and contain the ``obs`` and ``obsm_X_umap`` tables.

    color
        ``obs`` column name used for coloring.

    sample_n
        Number of cells to sample; if ``None``, all available cells are usually used.

    groups
        List of category labels to keep when ``color`` is a discrete field. If ``None``, all categories are used.

    where
        Optional SQL filtering condition used to restrict cells participating in calculation or plotting.

    legend_loc
        Legend location.

    title
        Figure title.

    figsize
        Matplotlib figure size.

    point_size
        Scatter point size.

    alpha
        Plotting transparency.

    cmap
        Colormap used for continuous variables.

    palette
        Color scheme used for discrete categorical variables.

    frameon
        Whether to show the plot frame.

    save_path
        Path for saving the figure or result.

    plot_batch_size
        Number of cells read from the database per plotting batch.

    return_df
        Whether to return the DataFrame used for plotting. This is mainly used for debugging or external reuse of plotting data.

    Returns
    -------
    pandas.DataFrame | None
        Returns the loaded plotting data when ``return_df=True``; otherwise directly plots and returns ``None``.
    """

    conn = atlas.connection

    # Check tables and columns
    tables = conn.execute("""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_name IN ('obsm_X_umap', 'obs')
    """).fetchdf()["table_name"].tolist()

    if "obsm_X_umap" not in tables:
        raise ValueError("obsm_X_umap does not exist in the database. Please run sap.tl.umap(atlas) first")
    if "obs" not in tables:
        raise ValueError("obs does not exist in the database")

    obs_cols = [r[1] for r in conn.execute("PRAGMA table_info(obs)").fetchall()]
    umap_cols = [r[1] for r in conn.execute("PRAGMA table_info(obsm_X_umap)").fetchall()]

    if color not in obs_cols:
        raise ValueError(f"The column does not exist in obs: {color}")
    if "atlas_cell_id" not in obs_cols:
        raise ValueError("atlas_cell_id does not exist in obs")
    if "atlas_cell_id" not in umap_cols or "umap1" not in umap_cols or "umap2" not in umap_cols:
        raise ValueError("obsm_X_umap needs to contain atlas_cell_id / umap1 / umap2")

    # Build filtering conditions
    where_clauses = [f"o.{color} IS NOT NULL"]

    if where is not None and str(where).strip() != "":
        where_clauses.append(f"({where})")

    if groups is not None:
        groups_sql = ", ".join([f"'{str(g)}'" for g in groups])
        where_clauses.append(f"CAST(o.{color} AS TEXT) IN ({groups_sql})")

    where_sql = " AND ".join(where_clauses)

    # Fetch data, optionally with sampling
    if sample_n is None:
        query = f"""
            SELECT
                u.atlas_cell_id,
                u.umap1,
                u.umap2,
                CAST(o.{color} AS TEXT) AS color_label
            FROM obsm_X_umap u
            JOIN obs o
              ON u.atlas_cell_id = o.atlas_cell_id
            WHERE {where_sql}
            ORDER BY u.atlas_cell_id
        """
    else:
        query = f"""
            SELECT *
            FROM (
                SELECT
                    u.atlas_cell_id,
                    u.umap1,
                    u.umap2,
                    CAST(o.{color} AS TEXT) AS color_label
                FROM obsm_X_umap u
                JOIN obs o
                  ON u.atlas_cell_id = o.atlas_cell_id
                WHERE {where_sql}
            ) t
            USING SAMPLE {int(sample_n)} ROWS
            ORDER BY atlas_cell_id
        """

    # When sample_n=None, use full streaming plotting to avoid loading all data with fetchdf at once and exhausting memory
    if sample_n is None:
        return _draw_umap_obs_streaming(
            atlas=atlas,
            color=color,
            where_sql=where_sql,
            legend_loc=legend_loc,
            title=title,
            figsize=figsize,
            point_size=point_size,
            alpha=alpha,
            palette=palette,
            frameon=frameon,
            save_path=save_path,
            plot_batch_size=plot_batch_size
        )

    # When sample_n is not None, still use the original sampled plotting path
    plot_df = conn.execute(query).fetchdf()

    if len(plot_df) == 0:
        raise ValueError("No cells are available for plotting after filtering")

    # Use natural sorting by default
    # embryo_1, embryo_2, ..., embryo_10
    unique_labels = _sort_categories_natural(
        plot_df["color_label"].astype(str).unique().tolist()
    )

    # Use the unified large discrete color pool
    label_to_color = _build_discrete_color_map(
        labels=unique_labels,
        palette=palette,
    )

    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    ax.set_facecolor("white")

    for lab in unique_labels:
        sub = plot_df[plot_df["color_label"] == lab]
        ax.scatter(
            sub["umap1"].to_numpy(),
            sub["umap2"].to_numpy(),
            s=point_size,
            alpha=alpha,
            c=[label_to_color[lab]],
            linewidths=0,
            label=str(lab),
            rasterized=True,
        )

    # Title
    if title is None:
        title = color
    ax.set_title(title, fontsize=18, weight="normal", pad=10)

    ax.set_xlabel("UMAP1", fontsize=16)
    ax.set_ylabel("UMAP2", fontsize=16)

    # Legend / on-data labels
    if legend_loc == "right_margin":

        n_cat = len(unique_labels)
        max_label_len = max([len(str(c)) for c in unique_labels], default=0)

        if n_cat <= 14:
            legend_ncol = 1
            legend_fontsize = 20
        elif n_cat <= 30:
            legend_ncol = 2
            legend_fontsize = 20
        elif n_cat <= 60:
            legend_ncol = 4
            legend_fontsize = 20
        else:
            legend_ncol = 5
            legend_fontsize = 12

        if max_label_len >= 18:
            legend_fontsize = min(legend_fontsize, 15)
        if max_label_len >= 28:
            legend_fontsize = min(legend_fontsize, 15)

        leg = ax.legend(
            title=None,
            bbox_to_anchor=(1.03, 0.5),
            loc="center left",
            frameon=False,
            markerscale=8.0,
            fontsize=legend_fontsize,
            borderaxespad=0.0,
            ncol=legend_ncol,
            columnspacing=1.0,
            handletextpad=0.35,
            labelspacing=0.35,
            handlelength=0.8,
        )

        # Force-enlarge legend dots to match PCA
        for h in leg.legend_handles:
            if hasattr(h, "set_sizes"):
                h.set_sizes([100])

        leg.set_in_layout(False)

    elif legend_loc == "on_data":
        center_rows = []
        for lab in unique_labels:
            sub = plot_df[plot_df["color_label"] == lab]
            if len(sub) == 0:
                continue
            center_rows.append({
                "color_label": str(lab),
                "x_center": float(sub["umap1"].median()),
                "y_center": float(sub["umap2"].median()),
            })
        center_df = pd.DataFrame(center_rows)

        # Modified: apply simple collision avoidance to on_data label positions to prevent text crowding
        center_df = _spread_on_data_label_positions(center_df)
        for _, row in center_df.iterrows():
            ax.text(
                row["label_x"],
                row["label_y"],
                str(row["color_label"]),
                fontsize=14,
                weight="bold",
                color="black",
                ha="center",
                va="center",
                zorder=10,
            )
    else:
        raise ValueError("legend_loc only supports 'right_margin' or 'on_data'")

    # Style
    ax.grid(False)
    if not frameon:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        ax.spines["left"].set_linewidth(1.0)
        ax.spines["bottom"].set_linewidth(1.0)
        ax.tick_params(axis="both", labelsize=11, width=1.0, length=4)

    if legend_loc in ("right_margin", "on_data"):
        fig.subplots_adjust(
            left=0.08,
            right=0.70,
            bottom=0.10,
            top=0.90,
        )
    else:
        plt.tight_layout(pad=0.8)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

    if return_df:
        return plot_df

    return None


# umap() - if color is an obs column -> plot_umap_obs()
#          sample_n == None -> _draw_umap_obs_streaming()
def _draw_umap_obs_streaming(
        atlas: Atlas,
        color: str,
        where_sql: str,
        legend_loc: str = "right_margin",
        title: str | None = None,
        figsize: tuple[float, float] | None=(22, 8),
        point_size: float = 1.0,
        alpha: float = 0.7,
        palette: str | list[str] | tuple[str, ...] | None = DEFAULT_DISCRETE_PALETTES,
        frameon: bool = True,
        save_path: PathLike[str] | str | None = None,
        plot_batch_size: int = 200000
):

    """Plot UMAP for a discrete ``obs`` categorical variable in batches.

    This internal function is used for discrete categorical UMAP plotting in large-data scenarios.
    It first reads all category labels to fix the color mapping,
    then reads UMAP coordinates and category labels from DuckDB in batches according to ``plot_batch_size``,
    and scatters them batch by batch onto the same axes, avoiding loading all cells into memory at once.

    Parameters
    ----------
    atlas
        Atlas object. It must already be connected to a DuckDB database and contain the ``obs`` and ``obsm_X_umap`` tables.

    color
        Discrete ``obs`` column name used for coloring.

    where_sql
        Pre-composed SQL ``WHERE`` condition, excluding the ``WHERE`` keyword.

    legend_loc
        Legend location.

    title
        Figure title.

    figsize
        Matplotlib figure size.

    point_size
        Scatter point size.

    alpha
        Plotting transparency.

    palette
        Color scheme used for discrete categorical variables.

    frameon
        Whether to show the plot frame.

    save_path
        Path for saving the figure. If ``None``, the figure is only displayed.

    plot_batch_size
        Number of cells read from the database per plotting batch.

    Returns
    -------
    None
        The function directly plots the figure and does not return plotting data.
    """

    conn = atlas.connection

    # First fetch all categories to fix the colors
    label_df = conn.execute(f"""
        SELECT DISTINCT CAST(o.{color} AS TEXT) AS color_label
        FROM obsm_X_umap u
        JOIN obs o
          ON u.atlas_cell_id = o.atlas_cell_id
        WHERE {where_sql}
        ORDER BY color_label
    """).fetchdf()

    if len(label_df) == 0:
        raise ValueError("No cells are available for plotting after filtering")

    # Use natural sorting by default
    # embryo_1, embryo_2, ..., embryo_10
    # Do not directly use SQL string sorting results
    unique_labels = _sort_categories_natural(
        label_df["color_label"].astype(str).tolist()
    )

    # Use the unified large discrete color pool
    label_to_color = _build_discrete_color_map(
        labels=unique_labels,
        palette=palette,
    )

    # Create figure
    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    ax.set_facecolor("white")

    # Read in batches + plot in batches
    last_cell_id = -1
    total_drawn = 0

    while True:

        batch_df = conn.execute(f"""
            SELECT
                u.atlas_cell_id,
                u.umap1,
                u.umap2,
                CAST(o.{color} AS TEXT) AS color_label
            FROM obsm_X_umap u
            JOIN obs o
              ON u.atlas_cell_id = o.atlas_cell_id
            WHERE {where_sql}
              AND u.atlas_cell_id > {int(last_cell_id)}
            ORDER BY u.atlas_cell_id
            LIMIT {int(plot_batch_size)}
        """).fetchdf()

        if len(batch_df) == 0:
            break

        last_cell_id = int(batch_df["atlas_cell_id"].iloc[-1])
        total_drawn += len(batch_df)

        for lab in unique_labels:
            sub = batch_df[batch_df["color_label"].astype(str) == lab]

            if len(sub) == 0:
                continue

            ax.scatter(
                sub["umap1"].to_numpy(),
                sub["umap2"].to_numpy(),
                s=point_size,
                alpha=alpha,
                c=[label_to_color[lab]],
                linewidths=0,
                rasterized=True
            )

    # Title
    if title is None:
        title = color

    ax.set_title(title, fontsize=14, weight="normal", pad=8)
    ax.set_xlabel("UMAP1", fontsize=12)
    ax.set_ylabel("UMAP2", fontsize=12)

    # Legend
    if legend_loc == "right_margin":
        n_cat = len(unique_labels)
        max_label_len = max([len(str(c)) for c in unique_labels], default=0)

        if n_cat <= 14:
            legend_ncol = 1
            legend_fontsize = 20
        elif n_cat <= 30:
            legend_ncol = 2
            legend_fontsize = 20
        elif n_cat <= 60:
            legend_ncol = 4
            legend_fontsize = 20
        else:
            legend_ncol = 5
            legend_fontsize = 12

        if max_label_len >= 18:
            legend_fontsize = min(legend_fontsize, 15)
        if max_label_len >= 28:
            legend_fontsize = min(legend_fontsize, 15)

        legend_handles = [
            Line2D(
                [0], [0],
                marker="o",
                color="w",
                label=str(lab),
                markerfacecolor=label_to_color[lab],
                markersize=10,
            )
            for lab in unique_labels
        ]

        leg = ax.legend(
            handles=legend_handles,
            title=None,
            bbox_to_anchor=(1.03, 0.5),
            loc="center left",
            frameon=False,
            fontsize=legend_fontsize,
            borderaxespad=0.0,
            ncol=legend_ncol,
            columnspacing=1.0,
            handletextpad=0.35,
            labelspacing=0.35,
            handlelength=0.8,
        )

        leg.set_in_layout(False)

    elif legend_loc == "on_data":
        center_df = conn.execute(f"""
            SELECT
                CAST(o.{color} AS TEXT) AS color_label,
                MEDIAN(u.umap1) AS x_center,
                MEDIAN(u.umap2) AS y_center
            FROM obsm_X_umap u
            JOIN obs o
              ON u.atlas_cell_id = o.atlas_cell_id
            WHERE {where_sql}
            GROUP BY CAST(o.{color} AS TEXT)
        """).fetchdf()
        # Modified: apply simple collision avoidance to on_data label positions to prevent text crowding
        center_df = _spread_on_data_label_positions(center_df)
        for _, row in center_df.iterrows():
            ax.text(
                row["label_x"],
                row["label_y"],
                str(row["color_label"]),
                fontsize=14,
                weight="bold",
                color="black",
                ha="center",
                va="center",
                zorder=10,
            )

    else:
        raise ValueError("legend_loc only supports 'right_margin' or 'on_data'")

    ax.grid(False)

    ax.set_xticks([])
    ax.set_yticks([])

    # ax.set_aspect("equal", adjustable="box")
    ax.set_aspect("auto")

    ax.margins(0.02)

    if frameon:
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.0)
            spine.set_color("black")
    else:
        for spine in ax.spines.values():
            spine.set_visible(False)

    if legend_loc in ("right_margin", "on_data"):
        # fig.subplots_adjust(
        #     left=0.06,
        #     right=0.42,
        #     bottom=0.10,
        #     top=0.90,
        # )
        fig.subplots_adjust(
            left=0.08,
            right=0.70,
            bottom=0.10,
            top=0.90,
        )
    else:
        plt.tight_layout(pad=0.8)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

    return None



# umap() - if color is a gene name -> plot_umap_features()
def _plot_umap_features(
        atlas: Atlas,
        genes: str | list[str],
        sample_n: int | None = 50000,
        where: str | None = None,
        use_data: str = "data_scale",
        ncols: int = 3,
        figsize: tuple[float, float] | None=None,
        point_size: float = 8,
        alpha: float = 0.9,
        cmap: str = "viridis",
        save_path: PathLike[str] | str | None = None,
):

    """Plot UMAP feature plots by gene expression.

    This internal function resolves gene names from the ``var`` table, reads expression values from the ``use_data`` field in ``X_HyS_data``,
    merges them with ``obsm_X_umap`` coordinates, and then plots one or more gene feature UMAP panels.
    Undetected sparse expression values are filled with 0; when ``use_data="data_scale"``,
    a more appropriate zero-fill value is selected according to the distribution of that field
    for filling implicit zeros.

    Parameters
    ----------
    atlas
        Atlas object. It must already be connected to a DuckDB database and contain ``obsm_X_umap``, ``obs``,
        ``var``, and ``X_HyS_data`` tables.

    genes
        Gene name or list of gene names to plot; they must exist in ``var.atlas_gene_name``.

    sample_n
        Number of cells to sample; if ``None``, all available cells are usually used.

    where
        Optional SQL filtering condition used to restrict cells participating in calculation or plotting.

    use_data
        Expression value field read from ``X_HyS_data``, such as ``"data_log1p"``, ``"data_count"``
        or ``"data_scale"``.

    ncols
        Number of subplots per row for multi-panel plotting.

    figsize
        Matplotlib figure size.

    point_size
        Scatter point size.

    alpha
        Plotting transparency.

    cmap
        Matplotlib colormap used for continuous gene expression values.
    save_path
        Path for saving the figure. If ``None``, the figure is only displayed and not saved.

    Returns
    -------
    None

    Notes
    -----
    This function is an internal implementation path of ``sap.pl.umap``;
    users usually call it through ``sap.pl.umap`` by passing gene names.
    """

    start = datetime.now()
    conn = atlas.connection

    if isinstance(genes, str):
        genes = [genes]

    if len(genes) == 0:
        raise ValueError("genes cannot be empty")

    if where is not None and str(where).strip() != "":
        logger.info(f"[UMAP features] where = {where}")

    # Check tables and columns
    tables = conn.execute("""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_name IN ('obsm_X_umap', 'obs', 'var', 'X_HyS_data')
    """).fetchdf()["table_name"].tolist()

    if "obsm_X_umap" not in tables:
        raise ValueError("obsm_X_umap does not exist in the database. Please run sap.tl.umap(atlas) first")
    if "obs" not in tables:
        raise ValueError("obs does not exist in the database")
    if "var" not in tables:
        raise ValueError("var does not exist in the database")
    if "X_HyS_data" not in tables:
        raise ValueError("X_HyS_data does not exist in the database")

    umap_cols = [r[1] for r in conn.execute("PRAGMA table_info(obsm_X_umap)").fetchall()]
    obs_cols = [r[1] for r in conn.execute("PRAGMA table_info(obs)").fetchall()]
    x_cols = [r[1] for r in conn.execute("PRAGMA table_info(X_HyS_data)").fetchall()]
    var_cols = [r[1] for r in conn.execute("PRAGMA table_info(var)").fetchall()]

    if "atlas_cell_id" not in umap_cols or "umap1" not in umap_cols or "umap2" not in umap_cols:
        raise ValueError("obsm_X_umap needs to contain atlas_cell_id / umap1 / umap2")

    if "atlas_cell_id" not in obs_cols:
        raise ValueError("atlas_cell_id does not exist in obs")

    if "atlas_cell_id" not in x_cols or "atlas_gene_id" not in x_cols:
        raise ValueError("X_HyS_data needs to contain atlas_cell_id / atlas_gene_id")

    if use_data not in x_cols:
        raise ValueError(f"The field does not exist in X_HyS_data: {use_data}")

    if "atlas_gene_id" not in var_cols or "atlas_gene_name" not in var_cols:
        raise ValueError("var needs to contain atlas_gene_id / atlas_gene_name")

    # Filter with SQL first, then sample UMAP cells
    where_sql = ""

    if where is not None and str(where).strip() != "":
        where_sql = f"WHERE {where}"

    if sample_n is None:
        umap_query = f"""
            SELECT
                u.atlas_cell_id,
                u.umap1,
                u.umap2
            FROM obsm_X_umap u
            JOIN obs o
              ON u.atlas_cell_id = o.atlas_cell_id
            {where_sql}
            ORDER BY u.atlas_cell_id
        """
    else:
        umap_query = f"""
            SELECT *
            FROM (
                SELECT
                    u.atlas_cell_id,
                    u.umap1,
                    u.umap2
                FROM obsm_X_umap u
                JOIN obs o
                  ON u.atlas_cell_id = o.atlas_cell_id
                {where_sql}
            ) t
            USING SAMPLE {int(sample_n)} ROWS
            ORDER BY atlas_cell_id
        """

    umap_df = conn.execute(umap_query).fetchdf()

    if len(umap_df) == 0:
        raise ValueError("No cells are available for plotting after filtering / sampling")

    # Query gene_id
    gene_name_sql = ", ".join([f"'{str(g)}'" for g in genes])

    if use_data == "data_scale":
        if "zero_scale_transform" not in var_cols:
            raise ValueError(
                "zero_scale_transform does not exist in var.\n"
                "Please run the scale workflow first to write zero_scale_transform."
            )

        gene_map_df = conn.execute(f"""
            SELECT
                atlas_gene_id,
                atlas_gene_name,
                zero_scale_transform
            FROM var
            WHERE atlas_gene_name IN ({gene_name_sql})
        """).fetchdf()

    else:
        gene_map_df = conn.execute(f"""
            SELECT
                atlas_gene_id,
                atlas_gene_name
            FROM var
            WHERE atlas_gene_name IN ({gene_name_sql})
        """).fetchdf()

    if len(gene_map_df) == 0:
        raise ValueError("These genes were not found in var")

    if use_data == "data_scale":
        gene_map = {
            row["atlas_gene_name"]: (
                int(row["atlas_gene_id"]),
                float(row["zero_scale_transform"]) if pd.notna(row["zero_scale_transform"]) else 0.0
            )
            for _, row in gene_map_df.iterrows()
        }
    else:
        gene_map = {
            row["atlas_gene_name"]: int(row["atlas_gene_id"])
            for _, row in gene_map_df.iterrows()
        }

    missing_genes = [g for g in genes if g not in gene_map]
    if missing_genes:
        raise ValueError(f"These genes were not found in var: {missing_genes}")

    # Register temporary table of sampled cells
    conn.register("_umap_cells_tmp", umap_df[["atlas_cell_id"]])

    # Fetch expression gene by gene
    plot_data = {}

    for gene in genes:

        if use_data == "data_scale":
            gene_id, zero_fill = gene_map[gene]

            expr_df = conn.execute(f"""
                SELECT
                    c.atlas_cell_id,
                    COALESCE(x.{use_data}, {zero_fill}) AS expr
                FROM _umap_cells_tmp c
                LEFT JOIN X_HyS_data x
                  ON c.atlas_cell_id = x.atlas_cell_id
                 AND x.atlas_gene_id = {gene_id}
            """).fetchdf()

        else:
            gene_id = gene_map[gene]

            expr_df = conn.execute(f"""
                SELECT
                    c.atlas_cell_id,
                    COALESCE(x.{use_data}, 0.0) AS expr
                FROM _umap_cells_tmp c
                LEFT JOIN X_HyS_data x
                  ON c.atlas_cell_id = x.atlas_cell_id
                 AND x.atlas_gene_id = {gene_id}
            """).fetchdf()

        df = umap_df.merge(expr_df, on="atlas_cell_id", how="left")

        if use_data == "data_scale":
            _, zero_fill = gene_map[gene]
            df["expr"] = df["expr"].fillna(zero_fill)
        else:
            df["expr"] = df["expr"].fillna(0.0)

        # Draw high-expression points later to avoid being covered by low-expression points
        df = df.sort_values("expr", ascending=True).reset_index(drop=True)

        plot_data[gene] = df

    conn.unregister("_umap_cells_tmp")

    # Automatic layout
    n = len(genes)
    nrows = math.ceil(n / ncols)

    if figsize is None:
        figsize = (5.3 * ncols, 5.0 * nrows)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=figsize,
        facecolor="white"
    )

    if nrows == 1 and ncols == 1:
        axes = [[axes]]
    elif nrows == 1:
        axes = [axes]
    elif ncols == 1:
        axes = [[ax] for ax in axes]

    axes_flat = [ax for row in axes for ax in row]

    # Plot
    for ax, gene in zip(axes_flat, genes):
        df = plot_data[gene]

        sc = ax.scatter(
            df["umap1"].to_numpy(),
            df["umap2"].to_numpy(),
            c=df["expr"].to_numpy(),
            cmap=cmap,
            s=point_size,
            alpha=alpha,
            linewidths=0
        )

        cbar = plt.colorbar(sc, ax=ax, pad=0.02, fraction=0.046)
        cbar.ax.tick_params(labelsize=10)

        ax.set_title(gene, fontsize=18, weight="normal", pad=10)
        ax.set_xlabel("UMAP1", fontsize=16)
        ax.set_ylabel("UMAP2", fontsize=16)

        ax.set_facecolor("white")
        ax.grid(False)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.spines["left"].set_linewidth(1.0)
        ax.spines["bottom"].set_linewidth(1.0)
        ax.tick_params(axis="both", labelsize=11, width=1.0, length=4)

    # Hide extra subplots
    for ax in axes_flat[len(genes):]:
        ax.set_visible(False)

    plt.tight_layout(pad=1.0)
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

    return None


# umap() - mixed mode: obs categorical variables + gene feature variables in the same Figure
def _plot_umap_mixed(
        atlas: Atlas,
        obs_colors: list[str],
        gene_colors: list[str],
        sample_n: int | None = 50000,
        where: str | None = None,
        use_data: str = "data_log1p",
        ncols: int = 3,
        figsize: tuple[float, float] | None = None,
        point_size: float = 5,
        alpha: float = 0.85,
        cmap: str = "viridis",
        palette: str | list[str] | tuple[str, ...] | None = DEFAULT_DISCRETE_PALETTES,
        frameon: bool = True,
        save_path: PathLike[str] | str | None = None,
):
    """Plot a mixed-type multi-panel UMAP figure.

    This internal function supports cases where ``obs`` categorical variables and
    gene feature variables are passed to ``sap.pl.umap`` at the same time, such as ``color=["kmeans", "CD14", "NKG7"]``.
    The function draws different types of coloring variables into the same Figure, with each variable corresponding to
    an independent subplot, thereby achieving a multi-panel display effect similar to Scanpy ``sc.pl.umap``.

    Specifically, variables in ``obs_colors`` are plotted as discrete categorical variables,
    using the unified discrete color pool and legends; variables in ``gene_colors`` are plotted as continuous expression values
    using the expression field specified by ``use_data`` and a continuous colormap, and a colorbar is added for each gene
    feature subplot.

    This function does not overlay category labels and gene expression values on the same axes; instead, it places them
    in different panels of the same Figure. This avoids conflicts between discrete and continuous color
    mappings while maintaining a display style similar to Scanpy multi-variable UMAP visualization.

    Parameters
    ----------
    atlas
        Atlas object. Usually, UMAP should have already been computed, and the database should contain
        tables such as ``obsm_X_umap``, ``obs``, ``var``, and ``X_HyS_data``.

    obs_colors
        List of ``obs`` column names to plot, such as ``["kmeans"]``,
        ``["cell_type"]``, or ``["batch", "kmeans"]``.
        These variables are plotted as discrete categorical variables.

    gene_colors
        List of gene names to plot, such as ``["CD14", "NKG7"]``.
        These variables are plotted as continuous gene features.

    sample_n
        Number of cells sampled for plotting. If ``None``, all cells are used.
        For large datasets, it is recommended to set an appropriate integer, such as ``50000``,
        to avoid memory pressure from reading too many UMAP coordinates and expression values at once.

    where
        Optional SQL filtering condition used to restrict cells participating in plotting.
        For example, ``where="batch = 'sample1'"``.

    use_data
        Expression field read from ``X_HyS_data`` during gene feature plotting.
        Common values include ``"data_count"``, ``"data_normalize"``, ``"data_log1p"``
        and ``"data_scale"``.

    ncols
        Number of subplots displayed per row in a multi-panel figure.

    figsize
        Figure size. If ``None``, it is automatically set according to the number of subplots.

    point_size
        Scatter point size.

    alpha
        Scatter point transparency.

    cmap
        Colormap used for continuous gene feature expression values.

    palette
        ``obs`` Color scheme used for discrete categorical variables.

    frameon
        Whether to show the axis frame.

    save_path
        Path for saving the figure. If ``None``, the figure is only displayed and not saved.

    Returns
    -------
    None

    Notes
    -----
    This function is mainly used for mixed plotting scenarios involving ``obs`` categorical variables and gene features.
    If only a single ``obs`` categorical variable is plotted, ``_plot_umap_obs`` is still used;
    if only a gene feature list is plotted, ``_plot_umap_features`` is still used.
    Therefore, adding this function does not change the behavior of the existing single-type UMAP plotting paths.

    Examples
    --------
    Plot clustering labels and marker gene expression in the same Figure::

        sap.pl.umap(
            atlas,
            color=["kmeans", "CD14", "NKG7"],
            use_data="data_log1p",
            sample_n=50000,
        )

    Plot multiple obs categorical variables and multiple gene expressions::

        sap.pl.umap(
            atlas,
            color=["kmeans", "cell_type", "CD14", "NKG7"],
            use_data="data_log1p",
            ncols=2,
        )
    """

    conn = atlas.connection

    if len(obs_colors) == 0 and len(gene_colors) == 0:
        raise ValueError("obs_colors and gene_colors cannot both be empty")

    # -----------------------------------------------------
    # 1. Check basic tables
    # -----------------------------------------------------
    tables = conn.execute("""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_name IN ('obsm_X_umap', 'obs', 'var', 'X_HyS_data')
    """).fetchdf()["table_name"].tolist()

    if "obsm_X_umap" not in tables:
        raise ValueError("obsm_X_umap does not exist in the database. Please run sap.tl.umap(atlas) first")
    if "obs" not in tables:
        raise ValueError("obs does not exist in the database")
    if "var" not in tables:
        raise ValueError("var does not exist in the database")
    if "X_HyS_data" not in tables and len(gene_colors) > 0:
        raise ValueError("X_HyS_data does not exist in the database")

    obs_cols = [r[1] for r in conn.execute("PRAGMA table_info(obs)").fetchall()]
    var_cols = [r[1] for r in conn.execute("PRAGMA table_info(var)").fetchall()]
    x_cols = [r[1] for r in conn.execute("PRAGMA table_info(X_HyS_data)").fetchall()]

    for obs_col in obs_colors:
        if obs_col not in obs_cols:
            raise ValueError(f"The column does not exist in obs: {obs_col}")

    if len(gene_colors) > 0:
        if use_data not in x_cols:
            raise ValueError(f"The field does not exist in X_HyS_data: {use_data}")

        if "atlas_gene_id" not in var_cols or "atlas_gene_name" not in var_cols:
            raise ValueError("var needs to contain atlas_gene_id / atlas_gene_name")

    # -----------------------------------------------------
    # 2. Read UMAP + obs data
    # -----------------------------------------------------
    obs_select = ""

    if len(obs_colors) > 0:
        obs_select = ",\n                " + ",\n                ".join([
            f"CAST(o.{obs_col} AS TEXT) AS {obs_col}"
            for obs_col in obs_colors
        ])

    where_sql = ""

    if where is not None and str(where).strip() != "":
        where_sql = f"WHERE {where}"

    if sample_n is None:
        umap_query = f"""
            SELECT
                u.atlas_cell_id,
                u.umap1,
                u.umap2
                {obs_select}
            FROM obsm_X_umap u
            JOIN obs o
              ON u.atlas_cell_id = o.atlas_cell_id
            {where_sql}
            ORDER BY u.atlas_cell_id
        """
    else:
        umap_query = f"""
            SELECT *
            FROM (
                SELECT
                    u.atlas_cell_id,
                    u.umap1,
                    u.umap2
                    {obs_select}
                FROM obsm_X_umap u
                JOIN obs o
                  ON u.atlas_cell_id = o.atlas_cell_id
                {where_sql}
            ) t
            USING SAMPLE {int(sample_n)} ROWS
            ORDER BY atlas_cell_id
        """

    umap_df = conn.execute(umap_query).fetchdf()

    if len(umap_df) == 0:
        raise ValueError("No cells are available for plotting after filtering / sampling")

    # -----------------------------------------------------
    # 3. Read gene expression
    # -----------------------------------------------------
    gene_expr_data = {}

    if len(gene_colors) > 0:

        gene_name_sql = ", ".join([f"'{str(g)}'" for g in gene_colors])

        if use_data == "data_scale":
            if "zero_scale_transform" not in var_cols:
                raise ValueError(
                    "zero_scale_transform does not exist in var.\n"
                    "Please run the scale workflow first to write zero_scale_transform."
                )

            gene_map_df = conn.execute(f"""
                SELECT
                    atlas_gene_id,
                    atlas_gene_name,
                    zero_scale_transform
                FROM var
                WHERE atlas_gene_name IN ({gene_name_sql})
            """).fetchdf()

            gene_map = {
                row["atlas_gene_name"]: (
                    int(row["atlas_gene_id"]),
                    float(row["zero_scale_transform"]) if pd.notna(row["zero_scale_transform"]) else 0.0
                )
                for _, row in gene_map_df.iterrows()
            }

        else:
            gene_map_df = conn.execute(f"""
                SELECT
                    atlas_gene_id,
                    atlas_gene_name
                FROM var
                WHERE atlas_gene_name IN ({gene_name_sql})
            """).fetchdf()

            gene_map = {
                row["atlas_gene_name"]: int(row["atlas_gene_id"])
                for _, row in gene_map_df.iterrows()
            }

        missing_genes = [g for g in gene_colors if g not in gene_map]

        if missing_genes:
            raise ValueError(f"These genes were not found in var: {missing_genes}")

        conn.register("_umap_cells_tmp", umap_df[["atlas_cell_id"]])

        for gene in gene_colors:

            if use_data == "data_scale":
                gene_id, zero_fill = gene_map[gene]

                expr_df = conn.execute(f"""
                    SELECT
                        c.atlas_cell_id,
                        COALESCE(x.{use_data}, {zero_fill}) AS expr
                    FROM _umap_cells_tmp c
                    LEFT JOIN X_HyS_data x
                      ON c.atlas_cell_id = x.atlas_cell_id
                     AND x.atlas_gene_id = {gene_id}
                """).fetchdf()

            else:
                gene_id = gene_map[gene]

                expr_df = conn.execute(f"""
                    SELECT
                        c.atlas_cell_id,
                        COALESCE(x.{use_data}, 0.0) AS expr
                    FROM _umap_cells_tmp c
                    LEFT JOIN X_HyS_data x
                      ON c.atlas_cell_id = x.atlas_cell_id
                     AND x.atlas_gene_id = {gene_id}
                """).fetchdf()

            df = umap_df[["atlas_cell_id", "umap1", "umap2"]].merge(
                expr_df,
                on="atlas_cell_id",
                how="left"
            )

            if use_data == "data_scale":
                _, zero_fill = gene_map[gene]
                df["expr"] = df["expr"].fillna(zero_fill)
            else:
                df["expr"] = df["expr"].fillna(0.0)

            # Draw high-expression points later to avoid being covered by low-expression points
            df = df.sort_values("expr", ascending=True).reset_index(drop=True)

            gene_expr_data[gene] = df

        conn.unregister("_umap_cells_tmp")

    # -----------------------------------------------------
    # 4. Create multi-panel Figure
    # -----------------------------------------------------
    panel_names = obs_colors + gene_colors
    n_panels = len(panel_names)

    if n_panels == 0:
        raise ValueError("There is no panel available for plotting")

    ncols_eff = min(int(ncols), n_panels)
    nrows_eff = math.ceil(n_panels / ncols_eff)

    if figsize is None:
        figsize = (5.3 * ncols_eff, 5.0 * nrows_eff)

    fig, axes = plt.subplots(
        nrows_eff,
        ncols_eff,
        figsize=figsize,
        facecolor="white"
    )

    if nrows_eff == 1 and ncols_eff == 1:
        axes = [[axes]]
    elif nrows_eff == 1:
        axes = [axes]
    elif ncols_eff == 1:
        axes = [[ax] for ax in axes]

    axes_flat = [ax for row in axes for ax in row]

    # -----------------------------------------------------
    # 5. Plot obs categorical variables
    # -----------------------------------------------------
    ax_id = 0

    for obs_col in obs_colors:
        ax = axes_flat[ax_id]
        ax_id += 1

        df = umap_df[["umap1", "umap2", obs_col]].copy()
        df = df[df[obs_col].notna()].copy()
        df[obs_col] = df[obs_col].astype(str)

        unique_labels = _sort_categories_natural(
            df[obs_col].unique().tolist()
        )

        label_to_color = _build_discrete_color_map(
            labels=unique_labels,
            palette=palette,
        )

        for lab in unique_labels:
            sub = df[df[obs_col] == lab]

            if len(sub) == 0:
                continue

            ax.scatter(
                sub["umap1"].to_numpy(),
                sub["umap2"].to_numpy(),
                s=point_size,
                alpha=alpha,
                c=[label_to_color[lab]],
                linewidths=0,
                rasterized=True,
                label=str(lab),
            )

        ax.set_title(obs_col, fontsize=18, weight="normal", pad=10)

        legend_handles = [
            Line2D(
                [0], [0],
                marker="o",
                color="w",
                label=str(lab),
                markerfacecolor=label_to_color[lab],
                markersize=8,
            )
            for lab in unique_labels
        ]

        ax.legend(
            handles=legend_handles,
            title=None,
            bbox_to_anchor=(1.02, 0.5),
            loc="center left",
            frameon=False,
            fontsize=10,
            borderaxespad=0.0,
            handletextpad=0.35,
            labelspacing=0.35,
        )

    # -----------------------------------------------------
    # 6. Plot gene features
    # -----------------------------------------------------
    for gene in gene_colors:
        ax = axes_flat[ax_id]
        ax_id += 1

        df = gene_expr_data[gene]

        sc = ax.scatter(
            df["umap1"].to_numpy(),
            df["umap2"].to_numpy(),
            c=df["expr"].to_numpy(),
            cmap=cmap,
            s=point_size,
            alpha=alpha,
            linewidths=0,
            rasterized=True,
        )

        cbar = plt.colorbar(sc, ax=ax, pad=0.02, fraction=0.046)
        cbar.ax.tick_params(labelsize=10)

        ax.set_title(gene, fontsize=18, weight="normal", pad=10)

    # -----------------------------------------------------
    # 7. Apply unified style
    # -----------------------------------------------------
    for ax in axes_flat[:n_panels]:

        ax.set_xlabel("UMAP1", fontsize=14)
        ax.set_ylabel("UMAP2", fontsize=14)

        ax.set_facecolor("white")
        ax.grid(False)

        if not frameon:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.spines["left"].set_linewidth(1.0)
            ax.spines["bottom"].set_linewidth(1.0)
            ax.tick_params(axis="both", labelsize=10, width=1.0, length=4)

    # Hide extra subplots
    for ax in axes_flat[n_panels:]:
        ax.set_visible(False)

    plt.tight_layout(pad=1.0)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

    return None


def _natural_sort_key(value: Any):
    """Generate a natural sorting key for categorical labels.

    This internal helper is used to arrange discrete categories in UMAP legends in a more readable order, avoiding cases where
    ``cluster_10`` appears before ``cluster_2``. Labels that look like missing values, such as empty strings, ``NA``, and ``nan``,
    are placed at the end.

    Parameters
    ----------
    value
        Categorical label to sort. It can be a string, number, or any object that can be converted to a string.

    Returns
    -------
    tuple
        Sorting key that can be passed to ``sorted(..., key=...)``.

    Examples
    --------
    ``embryo_1 < embryo_2 < embryo_10`` and ``cluster_1 < cluster_2 < cluster_11``.
    """

    s = str(value).strip()

    # Put missing-value labels at the end
    if s.casefold() in _MISSING_CATEGORY_LABELS:
        return (1, ())

    parts = re.split(r"(\d+)", s)

    key = []
    for part in parts:
        if part == "":
            continue

        if part.isdigit():
            key.append((0, int(part)))
        else:
            key.append((1, part.casefold()))

    return (0, tuple(key))


def _sort_categories_natural(labels: Any) -> list[str]:
    """Deduplicate categorical labels and perform natural sorting.

    This internal helper first converts labels to strings and then sorts them using ``_natural_sort_key`` for use in discrete
    UMAP legends and grouped plotting order.

    Parameters
    ----------
    labels
        Sequence of categorical labels.

    Returns
    -------
    list[str]
        Deduplicated list of naturally sorted labels.
    """

    labels = [str(x) for x in list(labels)]

    # Deduplicate while preserving unique labels from the original list
    labels = list(dict.fromkeys(labels))

    return sorted(labels, key=_natural_sort_key)


def _build_discrete_color_map(labels: Any, palette: Any | None=None):
    """Build a color mapping for discrete categorical labels.

    This internal helper takes colors from one or more Matplotlib discrete palettes according to the order of ``labels``.
    When the number of categories exceeds the default color pool, ``hsv`` is used to provide additional colors, ensuring that every category has a corresponding color.

    Parameters
    ----------
    labels
        Sorted list of categorical labels.

    palette
        Matplotlib colormap name, sequence of colormap names, or ``None``. If ``None``,
        ``DEFAULT_DISCRETE_PALETTES`` is used.

    Returns
    -------
    dict
        Dictionary in the form ``{label: color}``, which can be directly used for Matplotlib scatter and legend.
    """

    labels = list(labels)

    # Use the large color pool by default
    if palette is None:
        palette_names = DEFAULT_DISCRETE_PALETTES

    # Compatible with the original palette="tab20" usage
    elif isinstance(palette, str):
        palette_names = (palette,)

    # Support palette=["tab20", "tab20b", ...]
    else:
        palette_names = tuple(palette)

    palette_colors = []

    for cmap_name in palette_names:
        cmap_obj = plt.get_cmap(cmap_name)

        # ListedColormap, such as tab20 / Set3, usually has .colors
        if hasattr(cmap_obj, "colors"):
            palette_colors.extend(list(cmap_obj.colors))

        # Fallback: if it is a continuous colormap, sample colors evenly
        else:
            n = getattr(cmap_obj, "N", 256)
            palette_colors.extend([
                cmap_obj(i / max(n - 1, 1))
                for i in range(n)
            ])

    # If the number of categories exceeds the color pool, continue filling with hsv
    if len(palette_colors) < len(labels):
        extra_n = len(labels) - len(palette_colors)
        hsv = plt.get_cmap("hsv")
        palette_colors.extend([
            hsv(i / max(extra_n, 1))
            for i in range(extra_n)
        ])

    return {
        lab: palette_colors[i]
        for i, lab in enumerate(labels)
    }


def _spread_on_data_label_positions(
        center_df: pd.DataFrame,
        x_col: str = "x_center",
        y_col: str = "y_center",
        min_dx_frac: float = 0.08,
        min_dy_frac: float = 0.06,
        step_frac: float = 0.018,
        max_iter: int = 200,
) -> pd.DataFrame:
    """Apply simple collision avoidance to category labels displayed directly on a UMAP plot.

    This internal helper iteratively adjusts label coordinates based on the center point of each category in UMAP space,
    reducing cases where multiple category labels crowd together.
    It only adjusts text label positions and does not change any cell point coordinates.

    Parameters
    ----------
    center_df
        DataFrame containing the center-point coordinates of each category.

    x_col
        Column name for the center-point x coordinate.

    y_col
        Column name for the center-point y coordinate.

    min_dx_frac
        Minimum allowed horizontal distance between labels, expressed as a fraction of the current UMAP x-axis span.

    min_dy_frac
        Minimum allowed vertical distance between labels, expressed as a fraction of the current UMAP y-axis span.

    step_frac
        Movement step for each collision-avoidance update, expressed as a fraction of the coordinate span.

    max_iter
        Maximum number of iterations.

    Returns
    -------
    pandas.DataFrame
        Input DataFrame with two added columns, ``label_x`` and ``label_y``.
    """

    center_df = center_df.copy()

    if len(center_df) <= 1:
        center_df["label_x"] = center_df[x_col].astype(float)
        center_df["label_y"] = center_df[y_col].astype(float)
        return center_df

    x = center_df[x_col].astype(float).to_numpy()
    y = center_df[y_col].astype(float).to_numpy()

    x_span = max(float(np.nanmax(x) - np.nanmin(x)), 1e-12)
    y_span = max(float(np.nanmax(y) - np.nanmin(y)), 1e-12)

    min_dx = min_dx_frac * x_span
    min_dy = min_dy_frac * y_span

    step_x = step_frac * x_span
    step_y = step_frac * y_span

    label_x = x.copy()
    label_y = y.copy()

    for _ in range(max_iter):
        moved = False

        for i in range(len(center_df)):
            for j in range(i + 1, len(center_df)):

                dx = label_x[j] - label_x[i]
                dy = label_y[j] - label_y[i]

                if abs(dx) < min_dx and abs(dy) < min_dy:

                    # If two labels almost completely overlap, assign a fixed direction
                    if abs(dx) < 1e-12 and abs(dy) < 1e-12:
                        direction = 1 if (i + j) % 2 == 0 else -1
                        dx = direction * 1e-6
                        dy = direction * 1e-6

                    # Push them slightly apart in both the horizontal and vertical directions
                    sx = step_x if dx >= 0 else -step_x
                    sy = step_y if dy >= 0 else -step_y

                    label_x[i] -= sx
                    label_x[j] += sx
                    label_y[i] -= sy
                    label_y[j] += sy

                    moved = True

        if not moved:
            break

    center_df["label_x"] = label_x
    center_df["label_y"] = label_y

    return center_df
