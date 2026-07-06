from ..data import Atlas
import re
from os import PathLike
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colorbar import ColorbarBase
from scipy.stats import gaussian_kde
from typing import Any


# =====================================================
# Unified discrete category color pool
# -----------------------------------------------------
# Used for coloring categorical variables in obs, for example:
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
# Sorts as:
# embryo_1, embryo_2, embryo_3, ..., embryo_10
#
# Also applies to:
# cluster_1 / cluster_10
# batch2 / batch10
# group_3_day_2 / group_3_day_12
# =====================================================
_MISSING_CATEGORY_LABELS = {"", "na", "nan", "none", "<na>", "null"}


def violin(
        atlas: Atlas,
        genes: str | list[str],
        groupby: str = "kmeans",
        use_data: str = "data_log1p",
        sample_n_per_group: int | None = 2000,
        groups: list | None = None,
        where: str | None = None,
        order: list | None = None,
        save_path: PathLike[str] | str | None = None
) -> None:

    """Plot violin plots of gene expression across different cell groups.

    This function resolves gene names from ``var``, reads expression values from the ``use_data`` field in ``X_HyS_data``,
    and groups by ``obs[groupby]`` to draw standard violin plots. Each gene is plotted separately; the x-axis shows groups,
    and the y-axis shows expression values.
    This is suitable for checking the expression distribution of marker genes across different clusters or cell types.

    This function only plots gene expression and does not plot ``obs`` metrics.

    Parameters
    ----------
    atlas
        Atlas object. It must already be connected to a DuckDB database and contain the ``obs``, ``var``, and
        ``X_HyS_data`` tables.

    genes
        Gene name or list of gene names to display. They must exist in ``var.atlas_gene_name``.

    groupby
        Grouping column name in ``obs``, such as ``"kmeans"``, ``"leiden"``, or ``"cell_type"``.

    use_data
        Expression value field read from ``X_HyS_data``, such as ``"data_log1p"``, ``"data_count"``,
        or ``"data_scale"``.

    sample_n_per_group
        Maximum number of cells sampled per group for plotting. If ``None``, all cells are used.

    groups
        List of groups to display. If ``None``, all groups that satisfy the conditions are used.

    where
        Additional SQL filtering condition. If ``None``, no additional condition is added.

    order
        Display order of groups. If ``None``, natural sorting is used.

    save_path
        Path to save the figure. If ``None``, the figure is only displayed.

    Returns
    -------
    None
        The function draws the plot directly and does not return a figure.

    Examples
    --------
    Plot the expression distribution of a single gene across clusters::

        sap.pl.violin(atlas, genes="MS4A1", groupby="kmeans")

    Plot multiple marker genes at the same time::

        sap.pl.violin(
            atlas,
            genes=["MS4A1", "CD3D", "LYZ"],
            groupby="kmeans",
            use_data="data_log1p",
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
    x_cols = [r[1] for r in conn.execute("PRAGMA table_info(X_HyS_data)").fetchall()]

    if groupby not in obs_cols:
        raise ValueError(f"Column does not exist in obs: {groupby}")
    if "atlas_cell_id" not in obs_cols:
        raise ValueError("atlas_cell_id does not exist in obs")
    if "atlas_gene_id" not in var_cols or "atlas_gene_name" not in var_cols:
        raise ValueError("atlas_gene_id / atlas_gene_name does not exist in var")
    if use_data not in x_cols:
        raise ValueError(f"Field does not exist in X_HyS_data: {use_data}")

    # gene_name -> gene_id
    gene_name_sql = ", ".join([f"'{g}'" for g in genes])

    gene_map_df = conn.execute(f"""
        SELECT atlas_gene_id, atlas_gene_name
        FROM var
        WHERE atlas_gene_name IN ({gene_name_sql})
    """).fetchdf()

    if len(gene_map_df) == 0:
        raise ValueError("These genes cannot be found in var")

    gene_map = dict(zip(gene_map_df["atlas_gene_name"], gene_map_df["atlas_gene_id"]))
    missing_genes = [g for g in genes if g not in gene_map]
    if missing_genes:
        raise ValueError(f"These genes cannot be found in var: {missing_genes}")

    gene_map_df["atlas_gene_name"] = pd.Categorical(
        gene_map_df["atlas_gene_name"],
        categories=genes,
        ordered=True
    )
    gene_map_df = gene_map_df.sort_values("atlas_gene_name").reset_index(drop=True)

    # Prepare sampled cells by group
    where_clauses = [f"{groupby} IS NOT NULL"]
    if where is not None and str(where).strip() != "":
        where_clauses.append(f"({where})")

    if groups is not None:
        group_sql = ", ".join([f"'{g}'" for g in groups])
        where_clauses.append(f"CAST({groupby} AS TEXT) IN ({group_sql})")

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
        """Generate the sorting key for violin group labels.

        Groups that can be converted to integers or floats are sorted numerically, while other labels are sorted as strings, avoiding
        ``"10"`` appearing before ``"2"``.

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

    # Sample each group separately, then union
    sampled_parts = []
    for g in group_labels:
        if sample_n_per_group is None:
            q = f"""
                SELECT
                    atlas_cell_id,
                    CAST({groupby} AS TEXT) AS group_label
                FROM obs
                WHERE {where_sql}
                  AND CAST({groupby} AS TEXT) = '{g}'
            """
        else:
            q = f"""
                SELECT
                    atlas_cell_id,
                    CAST({groupby} AS TEXT) AS group_label
                FROM obs
                WHERE {where_sql}
                  AND CAST({groupby} AS TEXT) = '{g}'
                ORDER BY random()
                LIMIT {int(sample_n_per_group)}
            """
        sampled_parts.append(conn.execute(q).fetchdf())

    cells_df = pd.concat(sampled_parts, ignore_index=True)

    if len(cells_df) == 0:
        raise ValueError("No cells after sampling")

    # Register temporary tables
    conn.register("_violin_cells_tmp", cells_df)
    conn.register("_violin_genes_tmp", gene_map_df[["atlas_gene_id", "atlas_gene_name"]])

    # Fetch the long expression table (fill implicit zeros)
    plot_df = conn.execute(f"""
        SELECT
            c.group_label,
            g.atlas_gene_name AS gene,
            COALESCE(x.{use_data}, 0.0) AS expr
        FROM _violin_cells_tmp c
        CROSS JOIN _violin_genes_tmp g
        LEFT JOIN X_HyS_data x
            ON c.atlas_cell_id = x.atlas_cell_id
           AND g.atlas_gene_id = x.atlas_gene_id
    """).fetchdf()

    conn.unregister("_violin_cells_tmp")
    conn.unregister("_violin_genes_tmp")

    if len(plot_df) == 0:
        raise ValueError("plot_df is empty; cannot plot")

    plot_df["gene"] = pd.Categorical(plot_df["gene"], categories=genes, ordered=True)
    plot_df["group_label"] = pd.Categorical(plot_df["group_label"], categories=group_labels, ordered=True)
    plot_df = plot_df.sort_values(["gene", "group_label"]).reset_index(drop=True)

    # Plot
    n_panels = len(genes)
    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(4.2 * n_panels, 5.2),
        facecolor="white",
        squeeze=False
    )
    axes = axes.ravel()

    # scanpy-like palette
    scanpy_colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#bcbd22",
        "#17becf", "#7f7f7f", "#aec7e8", "#ffbb78"
    ]
    color_map = {g: scanpy_colors[i % len(scanpy_colors)] for i, g in enumerate(group_labels)}

    for ax, gene in zip(axes, genes):
        sub = plot_df[plot_df["gene"] == gene].copy()

        positions = np.arange(len(group_labels)) + 1

        violin_data = []
        for g in group_labels:
            vals = sub[sub["group_label"] == g]["expr"].to_numpy()
            violin_data.append(vals)

        vp = ax.violinplot(
            violin_data,
            positions=positions,
            widths=0.85,
            showmeans=False,
            showmedians=True,
            showextrema=False
        )

        for i, body in enumerate(vp["bodies"]):
            body.set_facecolor(color_map[group_labels[i]])
            body.set_edgecolor("#4a4a4a")
            body.set_alpha(0.9)
            body.set_linewidth(1.0)

        if "cmedians" in vp:
            vp["cmedians"].set_color("#2f2f2f")
            vp["cmedians"].set_linewidth(1.2)

        # jitter points
        for pos, g in zip(positions, group_labels):
            vals = sub[sub["group_label"] == g]["expr"].to_numpy()
            if len(vals) == 0:
                continue

            n_dot = min(250, len(vals))
            dot_idx = np.random.choice(len(vals), size=n_dot, replace=False)
            vals_dot = vals[dot_idx]
            jitter = (np.random.rand(n_dot) - 0.5) * 0.18

            ax.scatter(
                np.full(n_dot, pos) + jitter,
                vals_dot,
                s=3,
                c="#2f2f2f",
                alpha=0.6,
                linewidths=0
            )

        ax.set_title(gene, fontsize=14, weight="normal", pad=8)
        ax.set_xlabel(groupby, fontsize=12)
        ax.set_ylabel("expression", fontsize=12)
        ax.set_xticks(positions)
        ax.set_xticklabels(group_labels, fontsize=11)

        ax.set_facecolor("white")
        ax.grid(True, axis="y", color="#d9d9d9", linewidth=0.8, alpha=0.8)
        ax.grid(False, axis="x")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.0)
        ax.spines["bottom"].set_linewidth(1.0)
        ax.tick_params(axis="both", labelsize=11, width=1.0, length=4)

    plt.tight_layout(pad=1.0)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def stacked_violin(
        atlas: Atlas,
        genes: str | list[str],
        groupby: str = "cell_type_auto",
        use_data: str = "data_log1p",
        sample_n_per_group: int | None = 2000,
        groups: list | None = None,
        where: str | None = None,
        order: list | None = None,
        color_vmin: float | None = 0.0,
        color_vmax: float | None = 5.0,
        font_size: int = 14,
        save_path: PathLike[str] | str | None = None
) -> None:

    """Plot a stacked violin plot for multiple marker genes.

    This function reads expression values for multiple genes from ``X_HyS_data`` and groups by ``obs[groupby]`` to draw
    a stacked violin plot: each cell corresponds to a ``group x gene`` combination, the violin shape shows the expression distribution,
    and color intensity represents the median expression level of that combination.

    This plot is suitable for comparing expression patterns of multiple marker genes across different clusters, cell types, or sample groups.

    Parameters
    ----------
    atlas
        Atlas object. It must already be connected to a DuckDB database and contain the ``obs``, ``var``, and
        ``X_HyS_data`` tables.

    genes
        Gene name or list of gene names to display. They must exist in ``var.atlas_gene_name``.

    groupby
        Grouping column name in ``obs``, such as ``"kmeans"``, ``"leiden"``, or ``"cell_type"``.

    use_data
        Expression value field read from ``X_HyS_data``, such as ``"data_log1p"``, ``"data_count"``,
        or ``"data_scale"``.

    sample_n_per_group
        Maximum number of cells sampled per group for plotting. If ``None``, all cells are used.

    groups
        List of groups to display. If ``None``, all groups that satisfy the conditions are used.

    where
        Additional SQL filtering condition. If ``None``, no additional condition is added.

    order
        Display order of groups. If ``None``, natural sorting is used.

    color_vmin
        Minimum expression value for color mapping. If ``None``, the minimum median expression in the current data is used.

    color_vmax
        Maximum expression value for color mapping. If ``None``, the maximum median expression in the current data is used.

    font_size
        Plot font size.

    save_path
        Path to save the figure. If ``None``, the figure is only displayed.

    Returns
    -------
    None

    Examples
    --------
    Plot a stacked violin plot for multiple marker genes::

        sap.pl.stacked_violin(
            atlas,
            genes=["MS4A1", "CD3D", "LYZ", "NKG7"],
            groupby="kmeans",
        )

    Display by automatically annotated cell types and save the figure::

        sap.pl.stacked_violin(
            atlas,
            genes=["MS4A1", "CD3D", "LYZ"],
            groupby="cell_type_auto",
            save_path=r"F:\\figures\\stacked_violin.png",
        )"""

    conn = atlas.connection

    if isinstance(genes, str):
        genes = [genes]
    genes = [str(g) for g in genes]

    if len(genes) == 0:
        raise ValueError("genes cannot be empty")

    # Check columns
    obs_cols = [r[1] for r in conn.execute("PRAGMA table_info(obs)").fetchall()]
    var_cols = [r[1] for r in conn.execute("PRAGMA table_info(var)").fetchall()]
    x_cols = [r[1] for r in conn.execute("PRAGMA table_info(X_HyS_data)").fetchall()]

    if groupby not in obs_cols:
        raise ValueError(f"Column does not exist in obs: {groupby}")
    if "atlas_cell_id" not in obs_cols:
        raise ValueError("atlas_cell_id does not exist in obs")
    if "atlas_gene_id" not in var_cols or "atlas_gene_name" not in var_cols:
        raise ValueError("atlas_gene_id / atlas_gene_name does not exist in var")
    if use_data not in x_cols:
        raise ValueError(f"Field does not exist in X_HyS_data: {use_data}")

    # gene_name -> gene_id
    gene_name_sql = ", ".join([f"'{g}'" for g in genes])

    gene_map_df = conn.execute(f"""
        SELECT atlas_gene_id, atlas_gene_name
        FROM var
        WHERE atlas_gene_name IN ({gene_name_sql})
    """).fetchdf()

    if len(gene_map_df) == 0:
        raise ValueError("These genes cannot be found in var")

    gene_map = dict(zip(gene_map_df["atlas_gene_name"], gene_map_df["atlas_gene_id"]))
    missing_genes = [g for g in genes if g not in gene_map]
    if missing_genes:
        raise ValueError(f"These genes cannot be found in var: {missing_genes}")

    gene_map_df["atlas_gene_name"] = pd.Categorical(
        gene_map_df["atlas_gene_name"],
        categories=genes,
        ordered=True
    )
    gene_map_df = gene_map_df.sort_values("atlas_gene_name").reset_index(drop=True)

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
        """Generate the sorting key for stacked violin group labels.

        Groups that can be converted to integers or floats are sorted numerically, while other labels are sorted as strings, avoiding
        ``"10"`` appearing before ``"2"``.

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

    # Sample cells for each group
    sampled_parts = []
    for g in group_labels:
        if sample_n_per_group is None:
            q = f"""
                SELECT
                    atlas_cell_id,
                    CAST({groupby} AS TEXT) AS group_label
                FROM obs
                WHERE {where_sql}
                  AND CAST({groupby} AS TEXT) = '{g}'
            """
        else:
            q = f"""
                SELECT
                    atlas_cell_id,
                    CAST({groupby} AS TEXT) AS group_label
                FROM obs
                WHERE {where_sql}
                  AND CAST({groupby} AS TEXT) = '{g}'
                ORDER BY random()
                LIMIT {int(sample_n_per_group)}
            """
        sampled_parts.append(conn.execute(q).fetchdf())

    cells_df = pd.concat(sampled_parts, ignore_index=True)
    if len(cells_df) == 0:
        raise ValueError("No cells after sampling")

    # Register temporary tables
    conn.register("_sv_cells_tmp", cells_df)
    conn.register("_sv_genes_tmp", gene_map_df[["atlas_gene_id", "atlas_gene_name"]])

    # Fetch the long expression table (fill implicit zeros)
    expr_df = conn.execute(f"""
        SELECT
            c.group_label,
            g.atlas_gene_name AS gene,
            COALESCE(x.{use_data}, 0.0) AS expr
        FROM _sv_cells_tmp c
        CROSS JOIN _sv_genes_tmp g
        LEFT JOIN X_HyS_data x
            ON c.atlas_cell_id = x.atlas_cell_id
           AND g.atlas_gene_id = x.atlas_gene_id
    """).fetchdf()

    conn.unregister("_sv_cells_tmp")
    conn.unregister("_sv_genes_tmp")

    if len(expr_df) == 0:
        raise ValueError("expr_df is empty; cannot plot")

    expr_df["gene"] = pd.Categorical(expr_df["gene"], categories=genes, ordered=True)
    expr_df["group_label"] = pd.Categorical(expr_df["group_label"], categories=group_labels, ordered=True)

    # Median statistics (used for coloring)
    median_df = (
        expr_df
        .groupby(["group_label", "gene"], observed=True)["expr"]
        .median()
        .reset_index(name="median_expr")
    )

    # Layout
    n_genes = len(genes)
    n_groups = len(group_labels)

    main_w = max(7.4, 0.44 * n_genes + 2.8)
    main_h = max(4.4, 0.56 * n_groups + 2.0)

    right_w = 2.7
    fig_w = main_w + right_w
    fig_h = main_h + 0.8

    max_y_len = max(len(str(x)) for x in group_labels) if len(group_labels) > 0 else 10
    left_margin = min(0.25, max(0.09, 0.035 + 0.008 * max_y_len))

    max_gene_len = max(len(str(x)) for x in genes) if len(genes) > 0 else 10
    bottom_margin = min(0.28, max(0.18, 0.06 + 0.009 * max_gene_len))

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
    gs = fig.add_gridspec(
        1, 2,
        width_ratios=[main_w, right_w],
        wspace=0.03
    )

    ax = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[0, 1])
    ax_right.axis("off")
    ax.set_facecolor("white")
    ax_right.set_facecolor("white")

    # Color mapping
    if color_vmin is None:
        color_vmin = float(median_df["median_expr"].min())
    if color_vmax is None:
        color_vmax = float(median_df["median_expr"].max())

    norm = mpl.colors.Normalize(vmin=color_vmin, vmax=color_vmax)
    cmap = plt.get_cmap("Blues")

    gene_to_x = {g: i for i, g in enumerate(genes)}

    group_to_y = {g: (len(group_labels) - 1 - i) for i, g in enumerate(group_labels)}

    cell_half_height = 0.34
    cell_half_width = 0.33

    grouped_expr = expr_df.groupby(["group_label", "gene"], observed=True)

    for (grp, gene), sub in grouped_expr:
        vals = sub["expr"].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]

        if len(vals) == 0:
            continue

        x0 = gene_to_x[str(gene)]
        y0 = group_to_y[str(grp)]

        med_val = float(np.median(vals))
        facecolor = cmap(norm(med_val))

        vmin = float(np.min(vals))
        vmax = float(np.max(vals))

        # All-zero / constant distribution: draw a thin vertical line
        if vmax - vmin < 1e-12:
            ax.plot(
                [x0, x0],
                [y0 - cell_half_height * 0.35, y0 + cell_half_height * 0.35],
                color="#b0b0b0",
                linewidth=1.0,
                zorder=3
            )
            continue

        # KDE estimation
        try:
            kde = gaussian_kde(vals)
            ys = np.linspace(vmin, vmax, 120)
            dens = kde(ys)
        except Exception:
            ys = np.linspace(vmin, vmax, 120)
            dens = np.ones_like(ys)

        dens = np.asarray(dens, dtype=float)
        if np.max(dens) > 0:
            dens = dens / np.max(dens)
        else:
            dens = np.zeros_like(dens)

        ys_scaled = y0 + ((ys - vmin) / (vmax - vmin) - 0.5) * (2 * cell_half_height)

        widths = dens * cell_half_width
        x_left = x0 - widths
        x_right = x0 + widths

        poly_x = np.concatenate([x_left, x_right[::-1]])
        poly_y = np.concatenate([ys_scaled, ys_scaled[::-1]])

        ax.fill(
            poly_x, poly_y,
            facecolor=facecolor,
            edgecolor="#b0b0b0",
            linewidth=0.9,
            zorder=2
        )

        # Median horizontal line
        med_y = y0 + ((med_val - vmin) / (vmax - vmin) - 0.5) * (2 * cell_half_height)
        ax.plot(
            [x0 - cell_half_width * 0.45, x0 + cell_half_width * 0.45],
            [med_y, med_y],
            color="#808080",
            linewidth=0.9,
            zorder=3
        )

    # Main plot styling
    ax.set_xticks(np.arange(n_genes))
    ax.set_xticklabels(genes, rotation=90, fontsize=font_size)

    # Display y ticks from top to bottom in scanpy style
    y_tick_positions = [group_to_y[g] for g in group_labels]
    ax.set_yticks(y_tick_positions)
    ax.set_yticklabels(group_labels, fontsize=font_size)

    ax.set_xlim(-0.55, n_genes - 0.45)
    ax.set_ylim(-0.5, n_groups - 0.5)

    # Row separator lines
    for j in range(n_groups):
        ax.axhline(j + 0.5, color="#d0d0d0", linewidth=1.0, zorder=0)

    ax.grid(False)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
        spine.set_color("black")

    # Right-side colorbar description
    ax_cbar_box = ax_right.inset_axes([0.08, 0.08, 0.84, 0.22])
    ax_cbar_box.axis("off")
    ax_cbar_box.set_xlim(0, 1)
    ax_cbar_box.set_ylim(0, 1)

    ax_cbar_box.text(
        0.50, 0.96,
        "Median expression\nin group",
        ha="center", va="top",
        fontsize=font_size,
        transform=ax_cbar_box.transAxes
    )

    cbar_ax = ax_cbar_box.inset_axes([0.10, 0.18, 0.80, 0.18])

    cb = ColorbarBase(
        cbar_ax,
        cmap=cmap,
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
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def _natural_sort_key(value: Any):
    """Generate a natural sorting key for categorical labels.

    This internal helper orders group labels in a more readable way, avoiding ``cluster_10``
    appearing before ``cluster_2``. Missing-value-like labels such as empty strings, ``NA``, and ``nan`` are placed at the end.

    Parameters
    ----------
    value
        Categorical label to sort. It can be a string, a number, or an object convertible to a string.

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

    This internal helper first converts labels to strings and then sorts them by ``_natural_sort_key`` for use by violin
    and stacked violin group display order.

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
    When the number of categories exceeds the default color pool, ``hsv`` is used to add more colors so every category has a corresponding color.

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
        Dictionary in the form of ``{label: color}``, which can be used directly for Matplotlib plotting.
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

    # If the number of categories exceeds the color pool, use hsv to add more colors
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
