from ..data import Atlas
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from os import PathLike
import re
from typing import Any


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


def pca(
        atlas: Atlas,
        color: str | None = None,
        x_pc: int = 0,
        y_pc: int = 1,
        annotate_var_explained: bool = True,
        sample_n: int | None = None,
        use_data: str = "data_log1p",
        figsize: tuple[float, float] | None=(6, 5),
        point_size: float = 12,
        alpha: float = 0.8,
        cmap: str = "viridis",
        palette: str | list[str] | tuple[str, ...] | None = DEFAULT_DISCRETE_PALETTES,
        legend_loc: str | None = None,
        frameon: bool = True,
        save_path: PathLike[str] | str | None = None
) -> None:

    """Plot a cell PCA embedding scatter plot.

    This function reads cell PCA coordinates from the ``obsm_X_pca`` table and uses
    the two principal components specified by ``x_pc`` and ``y_pc`` to draw a
    two-dimensional scatter plot.
    ``color`` can be either a cell-level field in the ``obs`` table or a gene name
    in ``var.atlas_gene_name``. The former colors points by an obs column, while the
    latter reads the expression value of that gene from the ``use_data`` field in
    ``X_HyS_data`` and colors points with a continuous colorbar.

    This plot is similar to Scanpy ``sc.pl.pca`` and is commonly used to inspect PCA
    dimensionality reduction results, batch effects, QC metrics, or the distribution
    of marker genes in PCA space.

    Parameters
    ----------
    atlas
        Atlas object. It must already be connected to a DuckDB database and PCA must
        have been run to generate the ``obsm_X_pca`` table. If variance explained
        ratios need to be annotated on the axes, the ``uns_pca_stats`` table is also
        required.

    color
        Name used to color the scatter points. It can be an ``obs`` table column name
        or a gene name. If ``None``, a uniform gray color is used.

    x_pc
        PCA component index used for the x-axis, starting from 0.

    y_pc
        PCA component index used for the y-axis, starting from 0.

    annotate_var_explained
        Whether to annotate the variance explained ratio of each principal component on the axes.

    sample_n
        Maximum number of cells to sample for plotting. If ``None``, all cells are used.

    use_data
        Expression value field read from ``X_HyS_data`` when ``color`` is a gene name,
        such as ``"data_log1p"``, ``"data_count"``, or ``"data_scale"``.

    figsize
        Matplotlib figure size.

    point_size
        Scatter point size.

    alpha
        Transparency of graphical elements.

    cmap
        Matplotlib colormap name used for continuous variables or gene expression.

    palette
        Matplotlib palette name or sequence of palette names used for discrete
        ``obs`` categorical variables.

    legend_loc
        Legend location for discrete categories. The default value ``"right_margin"``
        places the legend in the right-side margin.

    frameon
        Whether to show the axis frame.

    save_path
        Path for saving the figure. If ``None``, the figure is only displayed and
        not saved.

    Returns
    -------
    None

    Examples
    --------
    Plot PC1 and PC2 colored by K-means clusters::

        sap.tl.pca(atlas)
        sap.pl.pca(atlas, color="kmeans")

    Plot PC2 and PC3 colored by a QC metric continuously::

        sap.pl.pca(
            atlas,
            color="pct_counts_mt",
            x_pc=1,
            y_pc=2,
            sample_n=200000,
        )

    Color by gene expression::

        sap.pl.pca(atlas, color="MS4A1", use_data="data_log1p")"""

    start = datetime.now()
    conn = atlas.connection

    if conn is None:
        raise ValueError("atlas.connection is None. Please connect to the database first")

    # Safe quoting for DuckDB fields
    def _q(name: str) -> str:
        """Add double-quote quoting for DuckDB SQL identifiers.

        This internal helper is used to safely concatenate column names, avoiding
        conflicts between ``obs`` fields, expression fields, or other SQL identifiers
        and SQL keywords. The function only handles identifier quoting and does not
        escape SQL values.

        Parameters
        ----------
        name
            Column name to use as a SQL identifier.

        Returns
        -------
        str
            SQL identifier with double quotes added and internal double quotes escaped.
        """
        return '"' + name.replace('"', '""') + '"'

    pcx = f"pc{x_pc}"
    pcy = f"pc{y_pc}"

    # Check whether the PCA table and PC columns exist
    pca_table_exists = conn.execute("""
        SELECT COUNT(*)
        FROM information_schema.tables
        WHERE table_name = 'obsm_X_pca'
    """).fetchone()[0]

    if pca_table_exists == 0:
        raise ValueError("obsm_X_pca does not exist in the database. Please run PCA first")

    obsm_cols = [
        r[1]
        for r in conn.execute("PRAGMA table_info(obsm_X_pca)").fetchall()
    ]

    if pcx not in obsm_cols or pcy not in obsm_cols:
        raise ValueError(
            f"The column does not exist in obsm_X_pca: {pcx} or {pcy}\n"
            f"Please confirm whether PCA has been computed, or whether x_pc / y_pc is out of range."
        )

    # Read explained variance ratio
    evr_map = {}

    pca_stats_exists = conn.execute("""
        SELECT COUNT(*)
        FROM information_schema.tables
        WHERE table_name = 'uns_pca_stats'
    """).fetchone()[0]

    if pca_stats_exists > 0:
        evr = conn.execute(f"""
            SELECT pc_index, variance_ratio
            FROM uns_pca_stats
            WHERE pc_index IN ({int(x_pc)}, {int(y_pc)})
            ORDER BY pc_index
        """).fetchdf()

        evr_map = dict(zip(evr["pc_index"], evr["variance_ratio"]))

    x_label = f"PC{x_pc + 1}"
    y_label = f"PC{y_pc + 1}"

    if annotate_var_explained:
        if x_pc in evr_map:
            x_label += f" ({evr_map[x_pc] * 100:.2f}%)"
        if y_pc in evr_map:
            y_label += f" ({evr_map[y_pc] * 100:.2f}%)"

    # First sample PCA coordinates by SQL
    if sample_n is None:
        pca_query = f"""
            SELECT atlas_cell_id, {_q(pcx)} AS {_q(pcx)}, {_q(pcy)} AS {_q(pcy)}
            FROM obsm_X_pca
        """
    else:
        pca_query = f"""
            SELECT atlas_cell_id, {_q(pcx)} AS {_q(pcx)}, {_q(pcy)} AS {_q(pcy)}
            FROM obsm_X_pca
            USING SAMPLE {int(sample_n)} ROWS
        """

    pca_df = conn.execute(pca_query).fetchdf()

    if pca_df.shape[0] == 0:
        raise ValueError("The PCA sampling result is empty, unable to plot")

    plot_df = pca_df.copy()

    color_kind = None

    if color is not None:

        # Get fields from obs / var / X_HyS_data
        obs_cols = [
            r[1]
            for r in conn.execute("PRAGMA table_info(obs)").fetchall()
        ]

        var_cols = [
            r[1]
            for r in conn.execute("PRAGMA table_info(var)").fetchall()
        ]

        x_cols = [
            r[1]
            for r in conn.execute("PRAGMA table_info(X_HyS_data)").fetchall()
        ]

        # color is a column name in the obs table
        if color in obs_cols:

            conn.register("_pca_cells_tmp", pca_df[["atlas_cell_id"]])

            obs_color_df = conn.execute(f"""
                SELECT
                    c.atlas_cell_id,
                    o.{_q(color)} AS color_value
                FROM _pca_cells_tmp AS c
                LEFT JOIN obs AS o
                  ON c.atlas_cell_id = o.atlas_cell_id
            """).fetchdf()

            conn.unregister("_pca_cells_tmp")

            plot_df = plot_df.merge(
                obs_color_df,
                on="atlas_cell_id",
                how="left"
            )

            color_kind = "obs"

        # color is a gene name in var.atlas_gene_name
        else:
            gene_row = conn.execute("""
                SELECT atlas_gene_id
                FROM var
                WHERE atlas_gene_name = ?
                LIMIT 1
            """, [color]).fetchone()

            if gene_row is not None:

                if use_data not in x_cols:
                    raise ValueError(
                        f"The expression field does not exist in X_HyS_data: {use_data}"
                    )

                gene_id = int(gene_row[0])

                conn.register("_pca_cells_tmp", pca_df[["atlas_cell_id"]])

                expr_df = conn.execute(f"""
                    SELECT
                        c.atlas_cell_id,
                        COALESCE(x.{_q(use_data)}, 0.0) AS color_value
                    FROM _pca_cells_tmp AS c
                    LEFT JOIN X_HyS_data AS x
                      ON c.atlas_cell_id = x.atlas_cell_id
                     AND x.atlas_gene_id = {gene_id}
                """).fetchdf()

                conn.unregister("_pca_cells_tmp")

                plot_df = plot_df.merge(
                    expr_df,
                    on="atlas_cell_id",
                    how="left"
                )

                plot_df["color_value"] = plot_df["color_value"].fillna(0.0)

                color_kind = "gene"

            # If it is a normal var column, raise an explicit error
            elif color in var_cols:
                raise ValueError(
                    f"color='{color}' is a column in the var table.\n"
                    f"However, each point in a PCA plot is a cell, while a var column is gene-level information, "
                    f"so it cannot be directly used to color cell PCA points.\n"
                    f"If you want to color by gene expression, please pass a gene name from var.atlas_gene_name."
                )

            else:
                raise ValueError(
                    f"Cannot find color='{color}'.\n"
                    f"It is neither an obs table column name nor a gene name in var.atlas_gene_name."
                )

    # Plot
    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    ax.set_facecolor("white")

    x = plot_df[pcx].to_numpy()
    y = plot_df[pcy].to_numpy()

    # Case A: no color specified, gray scatter points
    if color is None:
        ax.scatter(
            x,
            y,
            s=point_size,
            c="#bdbdbd",
            alpha=alpha,
            linewidths=0,
            rasterized=True,
        )

    # Case B: gene expression, continuous colorbar
    elif color_kind == "gene":
        sc_plot = ax.scatter(
            x,
            y,
            s=point_size,
            c=plot_df["color_value"].to_numpy(),
            cmap=cmap,
            alpha=alpha,
            linewidths=0,
            rasterized=True,
        )

        cbar = plt.colorbar(sc_plot, ax=ax, pad=0.02)
        cbar.set_label(color, fontsize=12)
        cbar.ax.tick_params(labelsize=10)

    # Case C: obs column
    #       numeric type -> continuous colorbar
    #       categorical / string / bool -> discrete legend
    elif color_kind == "obs":
        values = plot_df["color_value"]

        # bool is not treated as a continuous variable, but as a categorical variable
        is_bool = pd.api.types.is_bool_dtype(values)
        is_numeric = pd.api.types.is_numeric_dtype(values)

        # C1. numeric obs column: continuous colorbar
        if is_numeric and not is_bool:
            sc_plot = ax.scatter(
                x,
                y,
                s=point_size,
                c=values.to_numpy(),
                cmap=cmap,
                alpha=alpha,
                linewidths=0,
                rasterized=True,
            )

            cbar = plt.colorbar(sc_plot, ax=ax, pad=0.02)
            cbar.set_label(color, fontsize=12)
            cbar.ax.tick_params(labelsize=10)

        # C2. categorical obs column: discrete colors + legend
        else:
            values = values.astype("object").where(values.notna(), "NA")
            values_str = values.astype(str)

            # Use natural sorting by default
            # embryo_1, embryo_2, ..., embryo_10
            cats = _sort_categories_natural(pd.unique(values_str))

            # Explicitly specify category order
            values = pd.Series(
                pd.Categorical(
                    values_str,
                    categories=cats,
                    ordered=True,
                ),
                index=plot_df.index,
                name="color_value",
            )

            color_map = _build_discrete_color_map(
                labels=cats,
                palette=palette,
            )

            # Plot by category group, Scanpy-style legend
            for cat in cats:
                mask = values == cat

                ax.scatter(
                    plot_df.loc[mask, pcx].to_numpy(),
                    plot_df.loc[mask, pcy].to_numpy(),
                    s=point_size,
                    color=color_map[cat],
                    alpha=alpha,
                    linewidths=0,
                    label=str(cat),
                    rasterized=True,
                )

            if legend_loc == "right_margin":
                # Automatically adjust legend columns and font size according to the number of categories
                n_cat = len(cats)
                max_label_len = max([len(str(c)) for c in cats], default=0)

                if n_cat <= 14:
                    legend_ncol = 1  # Number of columns
                    legend_fontsize = 20  # Font size
                elif n_cat <= 30:
                    legend_ncol = 2
                    legend_fontsize = 20
                elif n_cat <= 60:
                    legend_ncol = 4
                    legend_fontsize = 20
                else:
                    legend_ncol = 5
                    legend_fontsize = 12

                if max_label_len >= 18: # Character length of the longest category name in the legend
                    legend_fontsize = min(legend_fontsize, 15)
                if max_label_len >= 28:
                    legend_fontsize = min(legend_fontsize, 15)

                leg = ax.legend(
                    title=None,
                    bbox_to_anchor=(1.03, 0.5),
                    loc="center left",
                    frameon=False,
                    markerscale=8.0, # Legend dots
                    fontsize=legend_fontsize,
                    borderaxespad=0.0,
                    ncol=legend_ncol,
                    columnspacing=1.0,
                    handletextpad=0.35,
                    labelspacing=0.35,
                    handlelength=0.8,
                )

                # Force enlarge scatter dots in the legend, making it more stable
                for h in leg.legend_handles:
                    if hasattr(h, "set_sizes"):
                        h.set_sizes([100])

                # Prevent tight_layout / layout systems from compressing the main plot for the legend
                leg.set_in_layout(False)

            elif legend_loc == "on_data":
                # Simple on_data: place category names near the median PCA coordinates of each category
                for cat in cats:
                    mask = values == cat
                    if mask.sum() == 0:
                        continue

                    x_med = np.median(plot_df.loc[mask, pcx].to_numpy())
                    y_med = np.median(plot_df.loc[mask, pcy].to_numpy())

                    ax.text(
                        x_med,
                        y_med,
                        str(cat),
                        fontsize=9,
                        weight="bold",
                        ha="center",
                        va="center",
                    )
            elif legend_loc is None:
                pass
            else:
                raise ValueError(
                    "legend_loc only supports 'right_margin', 'on_data', or None"
                )

    # Scanpy-style refinement
    ax.set_xlabel(x_label, fontsize=14)
    ax.set_ylabel(y_label, fontsize=14)

    if color is None:
        ax.set_title("PCA", fontsize=14, pad=8)
    else:
        ax.set_title(str(color), fontsize=14, pad=8)

    # Closer to Scanpy: do not draw grid by default
    ax.grid(False)

    if frameon:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.0)
        ax.spines["bottom"].set_linewidth(1.0)
    else:
        for spine in ax.spines.values():
            spine.set_visible(False)

    ax.tick_params(
        axis="both",
        labelsize=11,
        width=1.0,
        length=4,
    )

    ax.set_aspect("auto")

    # Control the height-width ratio of the PCA main plot frame to avoid becoming a narrow tall plot like Figure 2
    ax.set_box_aspect(0.75)

    # Do not let tight_layout squeeze the main plot narrow
    if legend_loc == "right_margin":
        # Manually leave space for the right-side legend so the main plot will not be compressed into a vertical strip
        fig.subplots_adjust(
            left=0.06,
            right=0.38,
            bottom=0.16,
            top=0.88,
        )
    else:
        plt.tight_layout(pad=0.8)

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", dpi=300)

    plt.show()


def pca_variance_ratio(
        atlas: Atlas,
        n_pcs: int = 30,
        *,
        log: bool = False,
        show: bool | None = None,
        figsize: tuple[float, float] | None = (7, 6),
        save_path: PathLike[str] | str | None = None,
) -> None:
    """Plot the variance explained ratio of each PCA component.

    This function reads ``variance_ratio`` for each principal component from the
    ``uns_pca_stats`` table and plots it in a style similar to Scanpy
    ``sc.pl.pca_variance_ratio``:
    the x-axis is the principal component ranking, the y-axis is the variance
    explained ratio, and each PC is labeled with vertical text.

    This plot is used to observe how much variation the first few principal components
    explain, helping determine whether the PCA dimensionality is sufficient or whether
    a single principal component explains an abnormally high proportion of variance.

    Parameters
    ----------
    atlas
        Atlas object. It must already be connected to a DuckDB database, and PCA must
        have been run to generate the ``uns_pca_stats`` table.

    n_pcs
        Number of leading PCA components to display.

    log
        Whether to use a logarithmic y-axis.

    show
        Whether to display the figure. If ``None``, the figure is displayed by default.

    figsize
        Matplotlib figure size.
    save_path
        Path for saving the figure. If ``None``, the figure is only displayed and
        not saved.

    Returns
    -------
    None

    Examples
    --------
    View the variance explained ratios of the first 30 principal components::

        sap.tl.pca(atlas)
        sap.pl.pca_variance_ratio(atlas, n_pcs=30)

    """

    conn = atlas.connection

    if conn is None:
        raise ValueError("atlas.connection is None. Please connect to the database first")

    # 1. Read PCA variance explained ratios
    df = conn.execute(f"""
        SELECT pc_index, variance_ratio
        FROM uns_pca_stats
        ORDER BY pc_index
        LIMIT {int(n_pcs)}
    """).fetchdf()

    if df.empty:
        raise ValueError("uns_pca_stats is empty. Please run PCA before plotting.")

    # 2. Prepare data
    # Scanpy style: the x-axis is ranking, starting from 0
    x = np.arange(df.shape[0])
    y = df["variance_ratio"].to_numpy(dtype=float)

    pc_labels = [
        f"PC{int(pc_index) + 1}"
        for pc_index in df["pc_index"].to_numpy()
    ]

    # 3. Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # 4. Label each PC with text instead of drawing a line
    for xi, yi, lab in zip(x, y, pc_labels):
        ax.text(
            xi,
            yi,
            lab,
            rotation=90,
            ha="center",
            va="bottom",
            fontsize=10,
            clip_on=False,
        )

    # 5. Axis style, as close to Scanpy as possible
    ax.set_title("variance ratio", fontsize=18, pad=8)
    ax.set_xlabel("ranking", fontsize=18)
    ax.set_ylabel("")

    # Show ranking on the x-axis
    ax.set_xlim(-0.8, len(x) - 0.2)

    # Try to include tick 20 on the right side, close to the Scanpy example
    if len(x) <= 20:
        ax.set_xticks(np.arange(0, len(x) + 1, 5))
    else:
        ax.set_xticks(np.arange(0, len(x), 5))

    # Leave some top space on the y-axis to avoid clipping the PC1 label
    y_max = float(np.nanmax(y))
    y_min = float(np.nanmin(y))

    if log:
        ax.set_yscale("log")
        positive_y = y[y > 0]
        if positive_y.size > 0:
            ax.set_ylim(
                float(np.nanmin(positive_y)) * 0.8,
                y_max * 1.25,
            )
    else:
        ax.set_ylim(
            max(0.0, y_min - y_max * 0.05),
            y_max * 1.20,
        )

    # Grid and border: Scanpy plots have a grid and a full border
    ax.grid(True, color="#cccccc", linewidth=0.8, alpha=0.9)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)

    ax.tick_params(
        axis="both",
        labelsize=14,
        width=1.0,
        length=4,
    )

    fig.tight_layout()

    # 6. Save figure
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", dpi=300)

    # 7. Show or close
    if show is None:
        show = True

    if show:
        plt.show()
    else:
        plt.close(fig)


def pca_variance_ratio_cumsum(
        atlas: Atlas,
        n_pcs: int = 30,
        *,
        log: bool = False,
        show: bool | None = None,
        figsize: tuple[float, float] | None=(16, 8),
        save_path: PathLike[str] | str | None = None,
) -> None:
    """Plot the cumulative PCA variance explained ratio.

    This function reads ``variance_ratio`` for each principal component from the
    ``uns_pca_stats`` table, computes the cumulative sum, and then draws a line plot.
    The x-axis is the principal component number, and the y-axis is the cumulative
    explained variance ratio.

    This plot is used to determine how many PCA dimensions should be retained to cover
    the major variation, for example checking whether the cumulative explained ratio
    of the first 30, 50, or 80 PCs reaches the expected level.

    Parameters
    ----------
    atlas
        Atlas object. It must already be connected to a DuckDB database, and PCA must
        have been run to generate the ``uns_pca_stats`` table.

    n_pcs
        Number of PCA components to display.

    log
        Whether to use a logarithmic y-axis. The cumulative explained ratio usually
        does not need a logarithmic axis; this parameter is kept for interface consistency.

    show
        Whether to display the figure immediately. If ``None``, the figure is displayed
        by default.

    figsize
        Matplotlib figure size.

    save_path
        Path for saving the figure. If ``None``, the figure is only displayed and
        not saved.

    Returns
    -------
    None

    Examples
    --------
    View the cumulative explained ratio of the first 50 principal components::

        sap.pl.pca_variance_ratio_cumsum(atlas, n_pcs=50)

    Save the cumulative explained ratio plot::

        sap.pl.pca_variance_ratio_cumsum(
            atlas,
            n_pcs=80,
            save_path=r"F:\\figures\\pca_cumsum.png",
            show=False,
        )"""

    # 1. Get database connection
    conn = atlas.connection

    # 2. Read PCA variance explained ratios from the uns_pca_stats table
    df = conn.execute(f"""
        SELECT pc_index, variance_ratio
        FROM uns_pca_stats
        ORDER BY pc_index
        LIMIT {int(n_pcs)}
    """).fetchdf()

    # If there is no data in the table, PCA has not been run yet, or PCA results have not been written to the database
    if df.empty:
        raise ValueError("uns_pca_stats is empty. Please run PCA before plotting.")

    # 3. Prepare plotting data
    # pc_index in the database usually starts from 0;
    # in Scanpy-style plots, it is usually displayed as PC1, PC2, PC3...
    x = df["pc_index"].to_numpy() + 1

    # Variance explained ratio of each PC
    y = df["variance_ratio"].to_numpy()

    # Cumulative variance explained ratio
    y_cum = np.cumsum(y)

    # 4. Create figure object
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(x, y_cum, marker="o")

    ax.set_xlabel("Principal component")
    ax.set_ylabel("Cumulative explained variance ratio")
    ax.set_title("Cumulative PCA variance ratio")

    # The cumulative explained ratio generally does not need log, but this parameter is kept for interface consistency
    if log:
        ax.set_yscale("log")

    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    # 5. Save figure
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", dpi=300)

    # 6. Show or close figure
    if show is None:
        show = True

    if show:
        plt.show()
    else:
        plt.close(fig)


    return None


# Plot PCA loadings
def pca_loadings(
        atlas: Atlas,
        components: int | tuple[int, ...] | list[int] = (1, 2),
        n_genes: int = 10,
        include_lowest: bool = True,
        figsize: tuple[float, float] | None = (14, 8),
        show: bool | None = None,
        save_path: PathLike[str] | str | None = None,
) -> None:
    """Plot PCA loadings.

    This function reads each gene's loading on the specified principal components
    from the ``varm_PCs`` table and plots the genes with the largest contributions
    for each PC. When ``include_lowest=True``, the lowest-loading side is also shown,
    which helps identify which genes drive the positive and negative directions.

    This plot is similar to ``scanpy.pl.pca_loadings`` and is commonly used to
    interpret the biological meaning of PCA axes, such as whether a principal component
    is dominated by specific markers, mitochondrial genes, ribosomal genes, or
    batch-related genes.

    Parameters
    ----------
    atlas
        Atlas object. It must already be connected to a DuckDB database, and PCA must
        have been run to generate the ``varm_PCs`` table. If the ``var`` table contains
        ``atlas_gene_name``, gene names are preferentially displayed in the plot.

    components
        Principal component numbers to display. Note that, as in Scanpy, this is
        1-based. For example, ``components=(1, 2)`` means PC1 and PC2.

    n_genes
        Number of genes to display on each side.
        When ``include_lowest=True``, each PC displays the top n_genes and bottom n_genes.

    include_lowest
        Whether to also show genes with the lowest loadings.

    figsize
        Figure size. If ``None``, it is automatically set according to the number of
        components.

    show
        Whether to display the figure. Defaults to True.

    save_path
        Path for saving the figure. If ``None``, the figure is only displayed and
        not saved.

    Returns
    -------
    None
        The function directly plots or saves the figure and does not return a figure.

    Examples
    --------
    Plot genes with the highest contributions on both sides of PC1 and PC2::

        sap.pl.pca_loadings(atlas, components=(1, 2), n_genes=10)

    Display only the highest-loading side of PC3 and save the figure::

        sap.pl.pca_loadings(
            atlas,
            components=3,
            n_genes=20,
            include_lowest=False,
            save_path=r"F:\\figures\\pc3_loadings.png",
            show=False,
        )
    """

    conn = atlas.connection

    if conn is None:
        raise ValueError("atlas.connection is None. Please connect to the database first")

    def _q(name: str) -> str:
        """Add double-quote quoting for DuckDB SQL identifiers.

        Parameters
        ----------
        name
            Column name to use as a SQL identifier.

        Returns
        -------
        str
            SQL identifier with double quotes added and internal double quotes escaped.
        """
        return '"' + name.replace('"', '""') + '"'

    # -------------------------------------------------
    # 1. Process components
    # -------------------------------------------------
    if isinstance(components, int):
        components = (components,)
    else:
        components = tuple(components)

    if len(components) == 0:
        raise ValueError("components cannot be empty")

    for comp in components:
        if comp < 1:
            raise ValueError("components uses Scanpy-style numbering and must start from 1; for example, use 1 for PC1")

    # -------------------------------------------------
    # 2. Check varm_PCs and var tables
    # -------------------------------------------------
    varm_exists = conn.execute("""
        SELECT COUNT(*)
        FROM information_schema.tables
        WHERE table_name = 'varm_PCs'
    """).fetchone()[0]

    if varm_exists == 0:
        raise ValueError("varm_PCs does not exist in the database. Please run PCA first")

    var_exists = conn.execute("""
        SELECT COUNT(*)
        FROM information_schema.tables
        WHERE table_name = 'var'
    """).fetchone()[0]

    if var_exists == 0:
        raise ValueError("The var table does not exist in the database")

    varm_cols = [
        r[1]
        for r in conn.execute("PRAGMA table_info('varm_PCs')").fetchall()
    ]

    var_cols = [
        r[1]
        for r in conn.execute("PRAGMA table_info('var')").fetchall()
    ]

    if "atlas_gene_id" not in varm_cols:
        raise ValueError("The atlas_gene_id field does not exist in the varm_PCs table")

    if "atlas_gene_id" not in var_cols:
        raise ValueError("The atlas_gene_id field does not exist in the var table")

    gene_name_col = "atlas_gene_name" if "atlas_gene_name" in var_cols else "atlas_gene_id"

    # -------------------------------------------------
    # 3. Automatically match PC column names
    # -------------------------------------------------
    def _find_pc_col(comp: int) -> str:
        """Find the loading column corresponding to the specified principal component in ``varm_PCs``.

        ``components`` uses Scanpy-style 1-based numbering, while database columns may
        use different naming styles such as ``pc0``, ``PC1``, or ``1``. This helper
        tries a set of common column names and returns the first match.

        Parameters
        ----------
        comp
            1-based principal component number, for example ``1`` means PC1.

        Returns
        -------
        str
            Loading column name that actually exists in ``varm_PCs``.
        """
        # comp is 1-based, while pc_index is 0-based
        pc_index = comp - 1

        candidates = [
            f"pc{pc_index}",
            f"PC{pc_index}",
            f"PC{comp}",
            f"pc{comp}",
            f"{pc_index}",
            f"{comp}",
        ]

        for c in candidates:
            if c in varm_cols:
                return c

        raise ValueError(
            f"Cannot find the loading column corresponding to PC{comp} in varm_PCs.\n"
            f"Tried these column names: {candidates}\n"
            f"Current fields in varm_PCs are: {varm_cols}"
        )

    # -------------------------------------------------
    # 4. Create figure
    # -------------------------------------------------
    n_components = len(components)

    if figsize is None:
        figsize = (5.0 * n_components, 4.2)

    fig, axes = plt.subplots(
        1,
        n_components,
        figsize=figsize,
        squeeze=False,
    )

    axes = axes.ravel()

    # -------------------------------------------------
    # 5. Draw a separate panel for each PC
    # -------------------------------------------------
    for ax, comp in zip(axes, components):

        pc_col = _find_pc_col(comp)

        df = conn.execute(f"""
            SELECT
                v.{_q(gene_name_col)} AS gene_name,
                p.{_q(pc_col)} AS loading
            FROM varm_PCs AS p
            JOIN var AS v
              ON p.atlas_gene_id = v.atlas_gene_id
            WHERE p.{_q(pc_col)} IS NOT NULL
        """).fetchdf()

        if df.empty:
            raise ValueError(f"The loading data for PC{comp} is empty")

        df["gene_name"] = df["gene_name"].astype(str)
        df["loading"] = df["loading"].astype(float)

        # Genes with the largest loadings
        top_df = (
            df.sort_values("loading", ascending=False)
              .head(int(n_genes))
              .copy()
        )

        if include_lowest:
            # Genes with the smallest loadings
            low_df = (
                df.sort_values("loading", ascending=True)
                  .head(int(n_genes))
                  .copy()
            )

            # To make the display more Scanpy-like, sort negative-direction genes from near 0 to most negative
            low_df = low_df.sort_values("loading", ascending=False).copy()

            plot_df = pd.concat([top_df, low_df], ignore_index=True)

            x_top = np.arange(len(top_df))
            x_low = np.arange(len(top_df) + 1, len(top_df) + 1 + len(low_df))

            # Positive-direction genes
            for xi, row in zip(x_top, top_df.itertuples(index=False)):
                ax.text(
                    xi,
                    row.loading,
                    row.gene_name,
                    rotation=90,
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    clip_on=False,
                )

            # Middle ellipsis
            y_mid = 0.0
            if len(plot_df) > 0:
                y_mid = float((plot_df["loading"].max() + plot_df["loading"].min()) / 2)

            ax.text(
                len(top_df),
                y_mid,
                "...",
                ha="center",
                va="center",
                fontsize=10,
            )

            # Negative-direction genes
            for xi, row in zip(x_low, low_df.itertuples(index=False)):
                ax.text(
                    xi,
                    row.loading,
                    row.gene_name,
                    rotation=90,
                    ha="center",
                    va="top",
                    fontsize=9,
                    clip_on=False,
                )

            x_all = np.concatenate([x_top, x_low])
            y_all = plot_df["loading"].to_numpy()

        else:
            plot_df = top_df.copy()
            x_all = np.arange(len(plot_df))
            y_all = plot_df["loading"].to_numpy()

            for xi, row in zip(x_all, plot_df.itertuples(index=False)):
                ax.text(
                    xi,
                    row.loading,
                    row.gene_name,
                    rotation=90,
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    clip_on=False,
                )

        # -------------------------------------------------
        # 6. Scanpy-like style
        # -------------------------------------------------
        ax.set_title(f"PC{comp}", fontsize=18, pad=8)
        ax.set_xlabel("ranking", fontsize=16)
        ax.set_ylabel("")

        if len(x_all) > 0:
            ax.set_xlim(-0.8, max(x_all) + 0.8)

        y_min = float(np.nanmin(y_all))
        y_max = float(np.nanmax(y_all))
        y_range = y_max - y_min

        if y_range == 0:
            y_range = abs(y_max) if y_max != 0 else 1.0

        ax.set_ylim(
            y_min - 0.15 * y_range,
            y_max + 0.15 * y_range,
        )

        ax.set_xticks([])

        ax.grid(
            True,
            axis="y",
            color="#cccccc",
            linewidth=0.8,
            alpha=0.9,
        )

        ax.grid(False, axis="x")

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.0)

        ax.tick_params(
            axis="both",
            labelsize=14,
            width=1.0,
            length=4,
        )

    fig.tight_layout()

    # -------------------------------------------------
    # 7. Save figure
    # -------------------------------------------------
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", dpi=300)

    # -------------------------------------------------
    # 8. Show or close
    # -------------------------------------------------
    if show is None:
        show = True

    if show:
        plt.show()
    else:
        plt.close(fig)


def _natural_sort_key(value: Any):
    """Generate a natural sorting key for categorical labels.

    This internal helper is used to arrange discrete category legends in a more
    readable order, avoiding cases where ``cluster_10`` appears before ``cluster_2``.
    Labels that look like missing values, such as empty strings, ``NA``, and ``nan``,
    are placed at the end.

    Parameters
    ----------
    value
        Categorical label to sort. It can be a string, number, or any object that
        can be converted to a string.

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

    This internal helper first converts labels to strings and then sorts them using
    ``_natural_sort_key``. It is used for PCA discrete coloring legends and category
    plotting order.

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

    # Deduplicate to avoid repeated categories
    labels = list(dict.fromkeys(labels))

    return sorted(labels, key=_natural_sort_key)


def _build_discrete_color_map(labels: Any, palette: Any | None=None):
    """Build a color mapping for discrete categorical labels.

    This internal helper takes colors from one or more Matplotlib discrete palettes
    according to the order of ``labels``. When the number of categories exceeds the
    default color pool, ``hsv`` is used to provide additional colors, ensuring that
    every category has a corresponding color.

    Parameters
    ----------
    labels
        Sorted list of categorical labels.

    palette
        Matplotlib colormap name, sequence of colormap names, or ``None``.
        If ``None``, ``DEFAULT_DISCRETE_PALETTES`` is used.

    Returns
    -------
    dict
        Dictionary in the form ``{label: color}``, which can be directly passed to
        Matplotlib scatter/legend.
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
