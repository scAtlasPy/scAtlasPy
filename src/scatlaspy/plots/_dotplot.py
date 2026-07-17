from ..data import Atlas
from ..data._expression_source import resolve_expression_source
import re
from os import PathLike
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colorbar import ColorbarBase
from typing import Any
from ._utils import estimate_dotplot_sample_per_group


# =====================================================
# Unified discrete categorical color palette pool
# -----------------------------------------------------
# Used for coloring obs categorical variables, such as:
# scatlas_cluster / cell_type / batch / organ, etc.
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


def dotplot(
    atlas: Atlas,
    genes: str | list[str],
    groupby: str = "scatlas_cluster",
    use_data: str = "data_log1p",
    sample_cells_per_group: int | str | None = "auto",
    groups: list | None = None,
    where: str | None = None,
    order: list | None = None,
    expression_cutoff: float = 0.0,
    standard_scale: str | None = None,
    colorbar_vmin: float | None = 0.0,
    colorbar_vmax: float | None = 5.0,
    font_size: int = 14,
    save_path: PathLike[str] | str | None = None,
    dpi: int = 300,
) -> None:

    """Plot a dotplot of gene expression across different cell groups.

    This function reads the expression of specified genes in each ``obs[groupby]``
    group from the resolved expression source, calculates the average expression and the percentage
    of expressing cells for each ``group x gene`` combination, and plots a dotplot:
    dot color represents average expression, and dot size represents the percentage
    of expressing cells.

    This plot is suitable for quickly comparing the expression patterns of multiple
    marker genes across different clusters, cell types, or sample groups.

    Parameters
    ----------
    atlas
        Atlas object. It must already be connected to a DuckDB database and contain
        the ``obs``, ``var``, and ``X_HyS_data`` tables.
    genes
        Gene name or list of gene names to display. They must exist in
        ``var.atlas_gene_name``.
    groupby
        Grouping column name in ``obs``, such as ``"scatlas_cluster"``,
        ``"leiden"``, or ``"cell_type"``.
    use_data
        Expression value field read from the resolved expression source, such as
        ``"data_log1p"``, ``"data_count"``, or ``"data_scale"``.
    sample_cells_per_group
        Maximum number of cells sampled from each group for plotting. The
        default ``"auto"`` estimates a per-group sample size from the number of
        groups and genes, targeting about 5 million sampled cell-gene values
        while keeping the result between 2,000 and 50,000 cells per group. If
        ``None``, all cells are used and statistics are aggregated in DuckDB.
    groups
        List of groups to display. If ``None``, all groups satisfying the conditions
        are used.
    where
        Additional SQL filtering condition. If ``None``, no additional condition is added.
    order
        Display order of groups. If ``None``, natural sorting is used.
    expression_cutoff
        Threshold for determining whether a cell expresses a gene. Cells with expression
        values greater than this threshold are counted toward the expression percentage.
    standard_scale
        Whether to standardize the average expression color values. Currently supports
        ``"var"``, which scales average expression to 0 to 1 within each gene.
        If ``None``, the raw average expression is used directly.
    colorbar_vmin
        Lower limit of the colorbar. If ``None``, it is automatically estimated from
        the current color values.
    colorbar_vmax
        Upper limit of the colorbar. If ``None``, it is automatically estimated from
        the current color values.
    font_size
        Font size for plotting.
    save_path
        Path for saving the figure. If ``None``, the figure is only displayed.
    dpi
        Resolution used when saving the figure.

    Returns
    -------
    None

    Examples
    --------
    View the expression of classic marker genes across distilled Louvain clusters::

        sap.pl.dotplot(
            atlas,
            genes=["MS4A1", "CD3D", "LYZ", "NKG7"],
            groupby="scatlas_cluster",
        )

    Display only specified clusters and save the figure::

        sap.pl.dotplot(
            atlas,
            genes=["MS4A1", "CD3D", "LYZ"],
            groupby="scatlas_cluster",
            groups=["0", "1", "2"],
            save_path="./figures/marker_dotplot.png",
        )"""

    conn = atlas.connection

    # Normalize parameters
    if isinstance(genes, str):
        genes = [genes]
    genes = [str(g) for g in genes]
    if len(genes) == 0:
        raise ValueError("genes cannot be empty")

    # Check columns
    obs_cols = [r[1] for r in conn.execute("PRAGMA table_info(obs)").fetchall()]
    var_cols = [r[1] for r in conn.execute("PRAGMA table_info(var)").fetchall()]
    expr_source = resolve_expression_source(conn, use_data)

    if groupby not in obs_cols:
        raise ValueError(f"Column does not exist in obs: {groupby}")
    if "atlas_cell_id" not in obs_cols:
        raise ValueError("atlas_cell_id does not exist in obs")
    if "atlas_gene_id" not in var_cols or "atlas_gene_name" not in var_cols:
        raise ValueError("atlas_gene_id / atlas_gene_name does not exist in var")

    # gene_name -> gene_id
    gene_name_sql = ", ".join([f"'{g}'" for g in genes])

    gene_map_df = conn.execute(f"""
        SELECT atlas_gene_id, atlas_gene_name
        FROM var
        WHERE atlas_gene_name IN ({gene_name_sql})
    """).fetchdf()

    if len(gene_map_df) == 0:
        raise ValueError("These genes were not found in var")

    gene_map = dict(zip(gene_map_df["atlas_gene_name"], gene_map_df["atlas_gene_id"]))
    missing_genes = [g for g in genes if g not in gene_map]
    if missing_genes:
        raise ValueError(f"These genes were not found in var: {missing_genes}")

    gene_map_df["atlas_gene_name"] = pd.Categorical(
        gene_map_df["atlas_gene_name"],
        categories=genes,
        ordered=True
    )
    gene_map_df = gene_map_df.sort_values("atlas_gene_name").reset_index(drop=True)

    # Build where clause
    where_clauses = [f"{groupby} IS NOT NULL"]

    if where is not None and str(where).strip() != "":
        where_clauses.append(f"({where})")

    if groups is not None:
        groups_sql = ", ".join([f"'{str(g)}'" for g in groups])
        where_clauses.append(f"CAST({groupby} AS TEXT) IN ({groups_sql})")

    where_sql = " AND ".join(where_clauses)

    # Group list
    group_df = conn.execute(f"""
        SELECT
            CAST({groupby} AS TEXT) AS group_label,
            COUNT(*) AS n_cells
        FROM obs
        WHERE {where_sql}
        GROUP BY 1
        ORDER BY 1
    """).fetchdf()

    if len(group_df) == 0:
        raise ValueError("No available group")

    # Sort numeric groups by numeric value to avoid 0,1,10,11,2
    def _group_sort_key(x: Any):
        """Generate a sorting key for dotplot group labels.

        Groups that can be converted to integers or floats are sorted numerically;
        other labels are sorted as strings, avoiding cases where ``"10"`` appears
        before ``"2"``.

        Parameters
        ----------
        x
            A single group label.

        Returns
        -------
        tuple
            Sorting key that can be passed to ``sorted(..., key=...)``.
        """
        try:
            return (0, int(x))
        except Exception:
            try:
                return (0, float(x))
            except Exception:
                return (1, str(x))

    group_df["__sort_key__"] = group_df["group_label"].map(_group_sort_key)
    group_df = (
        group_df
        .sort_values("__sort_key__")
        .drop(columns="__sort_key__")
        .reset_index(drop=True)
    )

    if order is not None:
        wanted = [str(x) for x in order]
        group_df = group_df[group_df["group_label"].isin(wanted)].copy()
        if len(group_df) == 0:
            raise ValueError("No available group after filtering by order")
        group_df["order_idx"] = group_df["group_label"].map({g: i for i, g in enumerate(wanted)})
        group_df = group_df.sort_values("order_idx").drop(columns="order_idx").reset_index(drop=True)

    group_labels = group_df["group_label"].astype(str).tolist()

    if isinstance(sample_cells_per_group, str) and sample_cells_per_group.lower().strip() == "auto":
        sample_cells_per_group = estimate_dotplot_sample_per_group(
            n_groups=len(group_labels),
            n_genes=len(genes),
        )

    # Sample cells from each group unless full-data aggregation was requested.
    # The full-data path stays in DuckDB and returns only group x gene statistics.
    if sample_cells_per_group is not None:
        sampled_parts = []
        for g in group_labels:
            q = f"""
                SELECT
                    atlas_cell_id,
                    CAST({groupby} AS TEXT) AS group_label
                FROM obs
                WHERE {where_sql}
                  AND CAST({groupby} AS TEXT) = '{g}'
                ORDER BY random()
                LIMIT {int(sample_cells_per_group)}
            """
            sampled_parts.append(conn.execute(q).fetchdf())

        cells_df = pd.concat(sampled_parts, ignore_index=True)
        if len(cells_df) == 0:
            raise ValueError("No cells remain after sampling")

        conn.register("_dotplot_cells_tmp", cells_df)
        cells_source_sql = "_dotplot_cells_tmp"
    else:
        conn.execute(f"""
            CREATE OR REPLACE TEMP VIEW _dotplot_cells_tmp AS
            SELECT
                atlas_cell_id,
                CAST({groupby} AS TEXT) AS group_label
            FROM obs
            WHERE {where_sql}
        """)
        cells_source_sql = "_dotplot_cells_tmp"

    conn.register("_dotplot_genes_tmp", gene_map_df[["atlas_gene_id", "atlas_gene_name"]])

    try:
        stat_df = conn.execute(f"""
            WITH cell_counts AS (
                SELECT
                    group_label,
                    COUNT(*) AS n_cells
                FROM {cells_source_sql}
                GROUP BY group_label
            ),
            group_gene_grid AS (
                SELECT
                    cc.group_label,
                    cc.n_cells,
                    g.atlas_gene_id,
                    g.atlas_gene_name AS gene
                FROM cell_counts cc
                CROSS JOIN _dotplot_genes_tmp g
            ),
            nonzero_stats AS (
                SELECT
                    c.group_label,
                    xexpr.atlas_gene_id,
                    SUM(
                        CASE
                            WHEN xexpr.expr IS NOT NULL
                            THEN xexpr.expr ELSE 0.0
                        END
                    ) AS sum_expr,
                    SUM(
                        CASE
                            WHEN xexpr.expr > {float(expression_cutoff)}
                            THEN 1 ELSE 0
                        END
                    ) AS n_expr
                FROM {cells_source_sql} c
                JOIN (
                    SELECT
                        {expr_source.cell_sql} AS atlas_cell_id,
                        {expr_source.gene_sql} AS atlas_gene_id,
                        {expr_source.value_sql} AS expr
                    FROM {expr_source.from_sql}
                    WHERE {expr_source.value_sql} IS NOT NULL
                ) AS xexpr
                  ON c.atlas_cell_id = xexpr.atlas_cell_id
                JOIN _dotplot_genes_tmp g
                  ON xexpr.atlas_gene_id = g.atlas_gene_id
                GROUP BY c.group_label, xexpr.atlas_gene_id
            )
            SELECT
                grid.group_label,
                grid.gene,
                COALESCE(s.sum_expr, 0.0) / NULLIF(grid.n_cells, 0) AS mean_expr,
                COALESCE(s.n_expr, 0.0) * 100.0 / NULLIF(grid.n_cells, 0) AS pct_expr,
                grid.n_cells AS n
            FROM group_gene_grid grid
            LEFT JOIN nonzero_stats s
              ON grid.group_label = s.group_label
             AND grid.atlas_gene_id = s.atlas_gene_id
        """).fetchdf()
    finally:
        try:
            conn.unregister("_dotplot_genes_tmp")
        except Exception:
            pass
        try:
            conn.unregister("_dotplot_cells_tmp")
        except Exception:
            pass
        try:
            conn.execute("DROP VIEW IF EXISTS _dotplot_cells_tmp")
        except Exception:
            pass

    if len(stat_df) == 0:
        raise ValueError("stat_df is empty, unable to plot")

    stat_df["gene"] = pd.Categorical(stat_df["gene"], categories=genes, ordered=True)
    stat_df["group_label"] = pd.Categorical(stat_df["group_label"], categories=group_labels, ordered=True)

    if standard_scale == "var":
        stat_df["mean_expr_scaled"] = (
            stat_df.groupby("gene", observed=True)["mean_expr"]
            .transform(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-12))
        )
        color_col = "mean_expr_scaled"
    else:
        color_col = "mean_expr"

    n_genes = len(genes)
    n_groups = len(group_labels)

    main_w = max(7.6, 0.46 * n_genes + 3.0)
    main_h = max(4.2, 0.50 * n_groups + 1.8)

    right_w = 3.0 if n_genes <= 8 else 3.1 if n_genes <= 16 else 3.3

    fig_w = main_w + right_w
    fig_h = main_h + 1.0

    # Left margin: keep it neither too empty nor clipped
    max_y_len = max(len(str(x)) for x in group_labels) if len(group_labels) > 0 else 10
    left_margin = min(0.24, max(0.09, 0.035 + 0.0085 * max_y_len))

    # Bottom margin: ensure gene names are fully displayed
    max_gene_len = max(len(str(x)) for x in genes) if len(genes) > 0 else 10
    bottom_margin = min(0.28, max(0.18, 0.06 + 0.009 * max_gene_len))

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
    gs = fig.add_gridspec(
        nrows=1, ncols=2,
        width_ratios=[main_w, right_w],
        wspace=0.03
    )

    ax = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[0, 1])

    ax.set_facecolor("white")
    ax_right.set_facecolor("white")
    ax_right.axis("off")

    # Main plot
    gene_to_x = {g: i for i, g in enumerate(genes)}
    group_to_y = {g: i for i, g in enumerate(group_labels)}

    x = stat_df["gene"].astype(str).map(gene_to_x).to_numpy()
    y = stat_df["group_label"].astype(str).map(group_to_y).to_numpy()

    pct = stat_df["pct_expr"].to_numpy()
    colors = stat_df[color_col].to_numpy()

    sizes = 4.0 + (pct / 100.0) ** 1.8 * 700.0

    if colorbar_vmin is None:
        vmin = float(np.nanmin(colors))
    else:
        vmin = float(colorbar_vmin)

    if colorbar_vmax is None:
        vmax = float(np.nanmax(colors))
    else:
        vmax = float(colorbar_vmax)

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    sizes = 4.0 + (pct / 100.0) ** 1.8 * 700.0

    ax.scatter(
        x,
        y,
        s=sizes,
        c=colors,
        cmap="Reds",
        norm=norm,  # Modified: the main plot and colorbar use the same color range
        edgecolors="#777777",
        linewidths=0.25
    )

    ax.set_xticks(np.arange(n_genes))
    ax.set_xticklabels(genes, rotation=90, fontsize=font_size)
    ax.set_yticks(np.arange(n_groups))
    ax.set_yticklabels(group_labels, fontsize=font_size)

    ax.set_xlim(-0.55, n_genes - 0.45)
    ax.set_ylim(-0.38, n_groups - 0.62)
    ax.invert_yaxis()

    ax.tick_params(axis="x", pad=2)
    ax.grid(False)

    # Black outer border
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
        spine.set_color("black")

    ax_size = ax_right.inset_axes([0.06, 0.50, 0.88, 0.28])
    ax_size.set_xlim(0, 1)
    ax_size.set_ylim(0, 1)
    ax_size.axis("off")

    ax_size.text(
        0.50, 0.98,
        "Fraction of cells\nin group (%)",
        ha="center", va="top",
        fontsize=font_size,
        transform=ax_size.transAxes
    )

    size_levels = [20, 40, 60, 80, 100]
    x_positions = np.linspace(0.14, 0.86, len(size_levels))

    y_circle = 0.44
    y_tick_top = 0.30
    y_tick_bottom = 0.22
    y_text = 0.08

    for x0, p in zip(x_positions, size_levels):
        s = 4.0 + (p / 100.0) ** 1.8 * 700.0

        ax_size.scatter(
            [x0], [y_circle],
            s=s,
            color="gray",
            edgecolors="none",
            transform=ax_size.transAxes
        )

        ax_size.plot(
            [x0, x0],
            [y_tick_top, y_tick_bottom],
            color="black",
            linewidth=1.0,
            transform=ax_size.transAxes
        )

        ax_size.text(
            x0, y_text,
            f"{p}",
            ha="center", va="center",
            fontsize=font_size,
            transform=ax_size.transAxes
        )

    # Lower right: Mean expression legend with increased spacing
    ax_cbar_box = ax_right.inset_axes([0.06, 0.12, 0.88, 0.18])
    ax_cbar_box.set_xlim(0, 1)
    ax_cbar_box.set_ylim(0, 1)
    ax_cbar_box.axis("off")

    ax_cbar_box.text(
        0.50, 0.98,
        "Mean expression\nin group",
        ha="center", va="top",
        fontsize=font_size,
        transform=ax_cbar_box.transAxes
    )

    cbar_ax = ax_cbar_box.inset_axes([0.12, 0.16, 0.76, 0.12])

    cb = ColorbarBase(
        cbar_ax,
        cmap=plt.get_cmap("Reds"),
        norm=norm,
        orientation="horizontal"
    )
    cb.ax.tick_params(labelsize=font_size, length=4, width=1.0)

    # Margins
    fig.subplots_adjust(
        left=left_margin,
        right=0.98,
        top=0.95,
        bottom=bottom_margin
    )

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")

    plt.show()


def _natural_sort_key(value: Any):
    """Generate a natural sorting key for categorical labels.

    This internal helper is used to arrange dotplot group labels in a more readable
    order, avoiding cases where ``cluster_10`` appears before ``cluster_2``.
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
    ``embryo_1 < embryo_2 < embryo_10`` and
    ``cluster_1 < cluster_2 < cluster_11``.
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
    ``_natural_sort_key``. It is used for the group display order in dotplot.

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
        Dictionary in the form ``{label: color}``, which can be used directly for
        Matplotlib plotting.
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
