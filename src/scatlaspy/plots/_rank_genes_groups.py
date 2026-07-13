import math
from os import PathLike
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Any
from ..data import Atlas
from ..data._expression_source import resolve_expression_source


def rank_genes_groups(
        atlas: Atlas,
        use_table: str = "rank_genes_groups",
        groups: list | None = None,
        n_genes: int = 25,
        score_key: str = "scores",
        gene_label: str = "names",
        ncols: int = 4,
        figsize: tuple | None = None,
        save_path: PathLike[str] | str | None = None,
        show: bool = True,
) -> None:
    """Plot the top marker gene ranking for each group.

    This function reads the differential gene result table written by
    ``sap.tl.rank_genes_groups``, displays the top-ranked marker genes for each group,
    and shows the statistical score corresponding to ``score_key`` using scatter points
    and text labels in each subplot.
    It is similar to Scanpy ``sc.pl.rank_genes_groups`` and is suitable for quickly
    browsing candidate marker gene rankings for each cluster.

    Parameters
    ----------
    atlas
        Atlas object. It must already be connected to a DuckDB database, or be able
        to reconnect through ``atlas.connect("r+")``. The database must contain the
        differential gene result table specified by ``use_table``.

    use_table
        Differential gene result table name. Defaults to ``"rank_genes_groups"``.

    groups
        List of groups to display. If ``None``, all groups in the result table are displayed.

    n_genes
        Number of top genes to display for each group.

    score_key
        Score field name in the result table used for y-axis plotting, such as ``"scores"``.

    gene_label
        Field name in the result table used to display gene names. Defaults to ``"names"``.

    ncols
        Maximum number of subplot columns per row.

    figsize
        Matplotlib figure size. If ``None``, it is automatically estimated according
        to the number of groups and ``ncols``.

    save_path
        Path for saving the figure. If ``None``, the figure is not saved.

    show
        Whether to display the figure immediately. If ``False``, the current figure is
        closed, which is suitable for batch saving.

    Returns
    -------
    None

    Examples
    --------
    Calculate and plot the default differential gene ranking figure::

        sap.tl.rank_genes_groups(atlas, groupby="kmeans")
        sap.pl.rank_genes_groups(atlas)

    Read from a custom result table and display only selected groups::

        sap.pl.rank_genes_groups(
            atlas,
            use_table="rank_genes_groups_kmeans_top100",
            groups=["0", "1", "2"],
            n_genes=10,
        )"""

    conn = atlas.connection
    if conn is None:
        atlas.connect("r+")
        conn = atlas.connection

    # -------------------------------------------------
    # 1. Check whether the result table exists
    # -------------------------------------------------
    table_exists = conn.execute("""
        SELECT COUNT(*)
        FROM information_schema.tables
        WHERE table_name = ?
    """, [use_table]).fetchone()[0]

    if table_exists == 0:
        raise ValueError(
            f"The result table does not exist in the database: {use_table}."
            f"Please run sap.tl.rank_genes_groups(..., key_added='{use_table}') first"
        )

    # -------------------------------------------------
    # 2. Read the result table
    # -------------------------------------------------
    df = conn.execute(f"""
        SELECT *
        FROM "{use_table}"
    """).fetchdf()

    if len(df) == 0:
        raise ValueError(f"The {use_table} table is empty, unable to plot")

    required_cols = {"group", "rank", gene_label, score_key}
    missing = required_cols - set(df.columns)

    if len(missing) > 0:
        raise ValueError(
            f"The {use_table} table is missing required fields: {missing}."
            f"Current fields are: {list(df.columns)}"
        )

    # Convert group uniformly to string to avoid int / str confusion
    df["group"] = df["group"].astype(str)

    # Group sorting function: values that can be converted to numbers are sorted numerically; others are sorted as strings
    def _group_sort_key(x: Any):
        """Generate a sorting key for differential gene group labels.

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

    # -------------------------------------------------
    # 3. Filter groups
    # -------------------------------------------------
    all_groups = df["group"].dropna().unique().tolist()

    all_groups = sorted(all_groups, key=_group_sort_key)

    if groups is None:
        plot_groups = all_groups
    else:
        wanted = {str(g) for g in groups}
        plot_groups = [g for g in all_groups if g in wanted]

    if len(plot_groups) == 0:
        raise ValueError(
            f"No groups remain after filtering. Available groups: {all_groups}"
        )

    # -------------------------------------------------
    # 4. Prepare canvas
    # -------------------------------------------------
    n_panels = len(plot_groups)
    ncols = min(int(ncols), n_panels)
    nrows = math.ceil(n_panels / ncols)

    if figsize is None:
        figsize = (4.2 * ncols, 4.8 * nrows)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=figsize,
        facecolor="white",
        squeeze=False,
    )

    axes = axes.reshape(-1)

    # -------------------------------------------------
    # 5. Plot group by group
    # -------------------------------------------------
    for ax, group in zip(axes, plot_groups):

        plot_df = (
            df[df["group"] == group]
            .sort_values("rank")
            .head(int(n_genes))
            .copy()
        )

        if len(plot_df) == 0:
            ax.set_title(f"group {group}")
            ax.text(
                0.5,
                0.5,
                "No genes",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_axis_off()
            continue

        x = np.arange(len(plot_df))
        y = plot_df[score_key].to_numpy(dtype=float)

        ax.scatter(
            x,
            y,
            s=18,
            linewidths=0,
            alpha=0.9,
        )

        for xi, yi, gene in zip(x, y, plot_df[gene_label]):
            ax.text(
                xi,
                yi,
                str(gene),
                rotation=90,
                fontsize=9,
                ha="center",
                va="bottom",
            )

        # Try to use group vs reference as the title
        if "reference" in plot_df.columns:
            ref = str(plot_df["reference"].iloc[0])
            title = f"{group} vs. {ref}"
        else:
            title = f"group {group}"

        ax.set_title(title, fontsize=16)
        ax.set_xlabel("ranking", fontsize=13)
        ax.set_ylabel(score_key, fontsize=13)

        ax.tick_params(axis="both", labelsize=10)
        ax.set_facecolor("white")
        ax.grid(False)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Turn off extra panels
    for ax in axes[n_panels:]:
        ax.set_axis_off()

    plt.tight_layout()

    # -------------------------------------------------
    # 6. Save / show
    # -------------------------------------------------
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return None


def rank_genes_groups_volcano(
        atlas: Atlas,
        use_table: str = "rank_genes_groups",
        group: str | int = "0",
        lfc_key: str = "logfoldchanges",
        pval_key: str = "pvals_adj",
        gene_label: str = "names",
        pval_cutoff: float = 0.05,
        logfc_cutoff: float = 1.0,
        top_n: int = 8,
        figsize: tuple = (12, 10),
        y_cap: float |  None = None,
        xlim_abs: float | None = None,
        save_path: PathLike[str] | str | None = None,
        show: bool = True,
        label_fontsize: int = 7,
        label_offset_step: int = 12,
) -> None:
    """Plot a volcano plot for differential genes in a single group.

    This function reads the results of the specified ``group`` from the differential
    gene result table and plots a volcano plot based on log fold change and adjusted
    p-values.
    Significantly upregulated genes are shown in red, significantly downregulated
    genes are shown in blue, nonsignificant genes are shown in gray, and the most
    significant genes are automatically labeled.

    This plot is used to check whether marker genes of a cluster or cell type are
    significant relative to the reference group, whether the direction is clear, and
    whether extreme logFC or p-values exist.

    Parameters
    ----------
    atlas
        Atlas object. It must already be connected to a DuckDB database, or be able
        to reconnect through ``atlas.connect("r+")``. The database must contain the
        differential gene result table specified by ``use_table``.

    use_table
        Database table name for reading existing results.

    group
        Group label to plot in the volcano plot.

    lfc_key
        Field name in the result table storing log fold change.

    pval_key
        Field name in the result table storing adjusted p-values. Defaults to
        ``"pvals_adj"``.

    gene_label
        Field name in the result table used to display gene names.

    pval_cutoff
        Adjusted p-value threshold for significance.

    logfc_cutoff
        Absolute log fold change threshold for significance.

    top_n
        Number of genes to automatically label. The function preferentially selects
        part of the genes from significantly upregulated and significantly downregulated genes.

    figsize
        Figure size. If ``None``, the function default size is used.

    y_cap
        Display clipping upper limit for the y-axis ``-log10(padj)``. If
        ``None``, no clipping is applied.

    xlim_abs
        Absolute value of the symmetric x-axis display range. If ``None``, it is
        automatically estimated from the logFC distribution.

    save_path
        Path for saving the figure. If ``None``, the figure is only displayed or returned.

    show
        Whether to display the figure immediately. If ``None``, the current Matplotlib
        behavior is followed.

    label_fontsize
        Font size for automatically labeled gene names.

    label_offset_step
        Offset strength used to stagger labels when automatically labeling gene names.

    Returns
    -------
    None

    Examples
    --------
    Plot the default differential gene volcano plot::

        sap.tl.rank_genes_groups(atlas, groupby="kmeans")
        sap.pl.rank_genes_groups_volcano(atlas)

    Specify group, top genes, and save path::

        sap.pl.rank_genes_groups_volcano(
            atlas,
            group="0",
            top_n=20,
            save_path=r"F:\\figures\\rank_volcano.png",
        )
    """

    conn = atlas.connection
    if conn is None:
        atlas.connect("r+")
        conn = atlas.connection

    # 1. Check result table
    table_exists = conn.execute("""
        SELECT COUNT(*)
        FROM information_schema.tables
        WHERE table_name = ?
    """, [use_table]).fetchone()[0]

    if table_exists == 0:
        raise ValueError(
            f"The result table does not exist in the database: {use_table}."
            f"Please run sap.tl.rank_genes_groups(..., key_added='{use_table}') first"
        )

    # 2. Read results for the specified group
    df = conn.execute(f"""
        SELECT *
        FROM "{use_table}"
        WHERE CAST("group" AS TEXT) = CAST(? AS TEXT)
    """, [str(group)]).fetchdf()

    if len(df) == 0:
        raise ValueError(f"No result for group={group} found in the {use_table} table")

    required_cols = {lfc_key, pval_key, gene_label}
    missing = required_cols - set(df.columns)

    if len(missing) > 0:
        raise ValueError(
            f"The {use_table} table is missing fields required for the volcano plot: {missing}."
            f"Current fields are: {list(df.columns)}"
        )

    # 3. Clean abnormal values
    df = df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=[lfc_key, pval_key, gene_label]
    ).copy()

    if len(df) == 0:
        raise ValueError("No data available for plotting after cleaning NA / inf")

    # Ensure numeric columns are float
    df[lfc_key] = df[lfc_key].astype(float)
    df[pval_key] = df[pval_key].astype(float)

    # 4. Calculate -log10(padj)
    # First use a tiny value to avoid log10(0), then use y_cap for display clipping
    tiny = np.nextafter(0, 1)
    df["neg_log10_padj"] = -np.log10(
        df[pval_key].clip(lower=tiny)
    )

    if y_cap is not None:
        df["neg_log10_padj_plot"] = df["neg_log10_padj"].clip(upper=float(y_cap))
    else:
        df["neg_log10_padj_plot"] = df["neg_log10_padj"]

    y_max_real = float(np.nanmax(df["neg_log10_padj_plot"]))

    if y_cap is not None:
        y_upper = min(float(y_cap), y_max_real * 1.15 + 1.0)
    else:
        y_upper = y_max_real * 1.15 + 1.0

    # Provide at least some display space
    y_upper = max(y_upper, 5.0)

    # 5. Significance grouping
    df["significant"] = (
        (df[pval_key] < float(pval_cutoff))
        & (df[lfc_key].abs() >= float(logfc_cutoff))
    )

    # Color style aligned with Scanpy plots: upregulated red, downregulated blue, nonsignificant gray
    colors = np.where(
        df["significant"] & (df[lfc_key] > 0),
        "#d62728",
        np.where(
            df["significant"] & (df[lfc_key] < 0),
            "#1f77b4",
            "#9a9a9a",
        ),
    )

    # 6. Plot
    fig, ax = plt.subplots(figsize=figsize, facecolor="white")

    ax.scatter(
        df[lfc_key],
        df["neg_log10_padj_plot"],
        c=colors,
        s=8,
        alpha=0.75,
        linewidths=0,
    )

    # Threshold lines
    ax.axvline(-float(logfc_cutoff), color="#555555", linestyle="--", linewidth=1.0)
    ax.axvline(float(logfc_cutoff), color="#555555", linestyle="--", linewidth=1.0)
    ax.axhline(-np.log10(float(pval_cutoff)), color="#555555", linestyle="--", linewidth=1.0)

    # 7. Label top genes
    n_up = max(1, int(top_n) // 2)
    n_down = max(1, int(top_n) - n_up)

    top_up = (
        df[df["significant"] & (df[lfc_key] > 0)]
        .sort_values([pval_key, lfc_key], ascending=[True, False])
        .head(n_up)
    )

    top_down = (
        df[df["significant"] & (df[lfc_key] < 0)]
        .sort_values([pval_key, lfc_key], ascending=[True, True])
        .head(n_down)
    )

    top = pd.concat([top_up, top_down], axis=0)

    if len(top) == 0:
        top = (
            df.sort_values([pval_key, lfc_key], ascending=[True, False])
            .head(int(top_n))
        )

    for j, (_, row) in enumerate(top.iterrows()):

        x0 = float(row[lfc_key])
        y0 = float(row["neg_log10_padj_plot"])

        # Prevent labels from touching the plot frame
        y0 = min(y0, y_upper * 0.94)

        # Stagger labels slightly up and down to avoid overlap
        y_text = y0 - (j % 4) * (y_upper * (label_offset_step / 500.0))
        y_text = max(y_text, 0.1)

        if x0 >= 0:
            x_text = x0 + 0.08
            ha = "left"
        else:
            x_text = x0 - 0.08
            ha = "right"

        ax.text(
            x_text,
            y_text,
            str(row[gene_label]),
            fontsize=label_fontsize,
            ha=ha,
            va="bottom",
            clip_on=True,
        )

    # 8. Title
    if "reference" in df.columns:
        ref = str(df["reference"].iloc[0])
        title = f"group {group} vs {ref}"
        xlabel = f"log2 fold change: cluster {group} vs {ref}"
    else:
        title = f"group {group}"
        xlabel = "log2 fold change"

    ax.set_title(title, fontsize=20, pad=14)
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel("-log10 adjusted p-value", fontsize=16)

    ax.tick_params(axis="both", labelsize=13)

    # Keep only the left and bottom borders
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)

    # Use an adaptive upper limit for the y-axis
    ax.set_ylim(-1, y_upper)

    # Use a symmetric x-axis range so the figure visually resembles a standard volcano plot
    finite_lfc = (
        df[lfc_key]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .abs()
    )

    if xlim_abs is None:
        if len(finite_lfc) > 0:
            # Use the 99.5th percentile to prevent extreme outliers from making the plot too wide
            xlim_abs_use = float(np.nanpercentile(finite_lfc, 99.5))
            xlim_abs_use = max(xlim_abs_use, float(logfc_cutoff) * 2.5)
        else:
            xlim_abs_use = 5.0
    else:
        xlim_abs_use = float(xlim_abs)

    ax.set_xlim(-xlim_abs_use * 1.05, xlim_abs_use * 1.05)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return None


def rank_genes_groups_violin(
        atlas: Atlas,
        group: str = 0 ,
        use_table: str = "rank_genes_groups",
        groupby: str = "kmeans",
        reference: str | int | None = None,
        genes: list[str] | None = None,
        n_genes: int = 8,
        use_expr_field: str = "data_log1p",
        sample_cells_per_group: int = 2000,
        save_path: PathLike[str] | str | None = None
) -> None:

    """Plot violin plots of marker gene expression across different groups.

    This function automatically selects the top marker genes of the specified ``group``
    based on the ``rank_genes_groups`` results, or uses a manually specified gene list
    through ``genes``. The function reads expression values from the resolved
    ``use_expr_field`` expression source and plots violin plots grouped by
    ``obs[groupby]``, which are used to check whether candidate markers are
    specifically expressed in the target group.

    Parameters
    ----------
    atlas
        Atlas object. It must already be connected to a DuckDB database and contain
        ``obs``, ``var``, ``X_HyS_data``, and the differential gene result table specified
        by ``use_table``.

    group
        Target group label whose marker genes should be displayed.

    use_table
        Database table name for reading existing results.

    groupby
        Grouping column in ``obs``, such as ``"kmeans"``, ``"leiden"``, or ``"cell_type"``.

    reference
        Reference group for differential analysis. If ``None``, the default result
        matching ``group`` in the result table is used; when a value is provided, the
        corresponding ``reference`` is preferentially selected.

    genes
        Manually specified list of gene names to display. If ``None``, ``n_genes`` genes
        are selected by rank from the differential gene result table.

    n_genes
        Number of top marker genes automatically selected when ``genes`` is ``None``.

    use_expr_field
        Expression value field read from the resolved expression source, such as
        ``"data_log1p"`` or ``"data_count"``.

    sample_cells_per_group
        Maximum number of cells sampled for plotting from each ``groupby`` group.
        If ``None``, all cells are used.

    save_path
        Path for saving the figure. If ``None``, the figure is only displayed or returned.

    Returns
    -------
    None

    Examples
    --------
    Plot violin plots for the top markers of each group::

        sap.tl.rank_genes_groups(atlas, groupby="kmeans")
        sap.pl.rank_genes_groups_violin(atlas)

    Specify a group and expression field::

        sap.pl.rank_genes_groups_violin(
            atlas,
            group="0",
            n_genes=5,
            use_expr_field="data_log1p",
        )"""

    conn = atlas.connection

    # Check columns
    obs_cols = [r[1] for r in conn.execute("PRAGMA table_info(obs)").fetchall()]
    var_cols = [r[1] for r in conn.execute("PRAGMA table_info(var)").fetchall()]
    expr_source = resolve_expression_source(conn, use_expr_field)

    if groupby not in obs_cols:
        raise ValueError(f"The column does not exist in obs: {groupby}")
    if "atlas_cell_id" not in obs_cols:
        raise ValueError("atlas_cell_id does not exist in obs")
    if "atlas_gene_id" not in var_cols:
        raise ValueError("atlas_gene_id does not exist in var")

    table_exists = conn.execute("""
        SELECT COUNT(*)
        FROM information_schema.tables
        WHERE table_name = ?
    """, [use_table]).fetchone()[0]

    if table_exists == 0:
        raise ValueError(
            f"The result table does not exist in the database: {use_table}."
            f"Please run sap.tl.rank_genes_groups(..., key_added='{use_table}') first"
        )

    # =====================================================
    # 1. Gene list
    # =====================================================
    if genes is None:
        rank_df = conn.execute(f"""
            SELECT
                atlas_gene_id,
                names AS atlas_gene_name,
                rank
            FROM "{use_table}"
            WHERE CAST("group" AS TEXT) = CAST(? AS TEXT)
            ORDER BY rank
            LIMIT {int(n_genes)}
        """, [str(group)]).fetchdf()

        if len(rank_df) == 0:
            raise ValueError(f"No result for group={group} found in the {use_table} table")

        genes = rank_df["atlas_gene_name"].astype(str).tolist()

        gene_map_df = rank_df[["atlas_gene_id", "atlas_gene_name"]].drop_duplicates()

    else:
        if isinstance(genes, str):
            genes = [genes]

        if len(genes) == 0:
            raise ValueError("genes is empty")

        # gene_name -> gene_id
        gene_name_sql = ", ".join([f"'{g}'" for g in genes])

        if "atlas_gene_name" in var_cols:
            gene_name_col = "atlas_gene_name"
        elif "gene_name" in var_cols:
            gene_name_col = "gene_name"
        else:
            raise ValueError("atlas_gene_name or gene_name does not exist in var, so genes cannot be searched by gene name")

        gene_map_df = conn.execute(f"""
            SELECT
                atlas_gene_id,
                {gene_name_col} AS atlas_gene_name
            FROM var
            WHERE {gene_name_col} IN ({gene_name_sql})
        """).fetchdf()

        if len(gene_map_df) == 0:
            raise ValueError("These genes were not found in var")

        gene_map = dict(zip(gene_map_df["atlas_gene_name"], gene_map_df["atlas_gene_id"]))

        missing_genes = [g for g in genes if g not in gene_map]
        if missing_genes:
            raise ValueError(f"These genes were not found in var: {missing_genes}")

    if len(genes) == 0:
        raise ValueError("genes is empty")

    # =====================================================
    # 2. Sample target cells
    # =====================================================
    group_sql = f"'{group}'" if isinstance(group, str) else str(group)

    if reference is None:
        ref_from_result = conn.execute(f"""
            SELECT DISTINCT reference
            FROM "{use_table}"
            WHERE CAST("group" AS TEXT) = CAST(? AS TEXT)
            LIMIT 1
        """, [str(group)]).fetchone()

        if ref_from_result is not None:
            reference_in_result = str(ref_from_result[0])
        else:
            reference_in_result = "rest"

        if reference_in_result == "rest":
            # group vs rest
            group_cells_df = conn.execute(f"""
                SELECT atlas_cell_id
                FROM obs
                WHERE CAST({groupby} AS TEXT) = CAST({group_sql} AS TEXT)
                ORDER BY random()
                LIMIT {sample_cells_per_group}
            """).fetchdf()

            rest_cells_df = conn.execute(f"""
                SELECT atlas_cell_id
                FROM obs
                WHERE {groupby} IS NOT NULL
                  AND CAST({groupby} AS TEXT) != CAST({group_sql} AS TEXT)
                ORDER BY random()
                LIMIT {sample_cells_per_group}
            """).fetchdf()

            ref_label = "rest"

        else:
            # If the result table uses group vs a specific reference
            ref_sql = f"'{reference_in_result}'"

            group_cells_df = conn.execute(f"""
                SELECT atlas_cell_id
                FROM obs
                WHERE CAST({groupby} AS TEXT) = CAST({group_sql} AS TEXT)
                ORDER BY random()
                LIMIT {sample_cells_per_group}
            """).fetchdf()

            rest_cells_df = conn.execute(f"""
                SELECT atlas_cell_id
                FROM obs
                WHERE CAST({groupby} AS TEXT) = CAST({ref_sql} AS TEXT)
                ORDER BY random()
                LIMIT {sample_cells_per_group}
            """).fetchdf()

            ref_label = reference_in_result

    else:
        ref_sql = f"'{reference}'" if isinstance(reference, str) else str(reference)

        group_cells_df = conn.execute(f"""
            SELECT atlas_cell_id
            FROM obs
            WHERE CAST({groupby} AS TEXT) = CAST({group_sql} AS TEXT)
            ORDER BY random()
            LIMIT {sample_cells_per_group}
        """).fetchdf()

        rest_cells_df = conn.execute(f"""
            SELECT atlas_cell_id
            FROM obs
            WHERE CAST({groupby} AS TEXT) = CAST({ref_sql} AS TEXT)
            ORDER BY random()
            LIMIT {sample_cells_per_group}
        """).fetchdf()

        ref_label = str(reference)

    if len(group_cells_df) == 0:
        raise ValueError(f"group={group} has no cells")
    if len(rest_cells_df) == 0:
        raise ValueError("reference/rest has no cells")

    group_cells_df["group_label"] = str(group)
    rest_cells_df["group_label"] = ref_label

    cells_df = pd.concat([group_cells_df, rest_cells_df], ignore_index=True)

    # =====================================================
    # 3. Register sampled cells and genes
    # =====================================================
    conn.register("_violin_cells_tmp", cells_df)
    conn.register("_violin_genes_tmp", gene_map_df)

    # Fetch expression long table, including implicit zeros
    plot_df = conn.execute(f"""
        SELECT
            c.group_label,
            g.atlas_gene_name AS gene,
            COALESCE(xexpr.expr, 0.0) AS expr
        FROM _violin_cells_tmp c
        CROSS JOIN _violin_genes_tmp g
        LEFT JOIN (
            SELECT
                {expr_source.cell_sql} AS atlas_cell_id,
                {expr_source.gene_sql} AS atlas_gene_id,
                {expr_source.value_sql} AS expr
            FROM {expr_source.from_sql}
            WHERE {expr_source.value_sql} IS NOT NULL
        ) AS xexpr
          ON c.atlas_cell_id = xexpr.atlas_cell_id
         AND g.atlas_gene_id = xexpr.atlas_gene_id
    """).fetchdf()

    conn.unregister("_violin_cells_tmp")
    conn.unregister("_violin_genes_tmp")

    if len(plot_df) == 0:
        raise ValueError("plot_df is empty, unable to plot")

    # Preserve the input gene order
    plot_df["gene"] = pd.Categorical(plot_df["gene"], categories=genes, ordered=True)
    plot_df = plot_df.sort_values(["gene", "group_label"]).reset_index(drop=True)

    # =====================================================
    # 4. Plot
    # =====================================================
    fig, ax = plt.subplots(figsize=(1.25 * len(genes) + 2.5, 6), facecolor="white")

    labels = [str(group), ref_label]
    positions = np.arange(len(genes))

    width = 0.36
    pos_left = positions - width / 2
    pos_right = positions + width / 2

    color_map = {
        str(group): "#1f77b4",   # Blue
        ref_label: "#ff7f0e"     # Orange
    }

    for idx, gene in enumerate(genes):
        sub = plot_df[plot_df["gene"] == gene]

        vals_group = sub[sub["group_label"] == str(group)]["expr"].values
        vals_ref = sub[sub["group_label"] == ref_label]["expr"].values

        # violin 1: group
        if len(vals_group) > 0:
            vp1 = ax.violinplot(
                [vals_group],
                positions=[pos_left[idx]],
                widths=width,
                showmeans=False,
                showmedians=True,
                showextrema=False
            )
            for body in vp1["bodies"]:
                body.set_facecolor(color_map[str(group)])
                body.set_edgecolor("black")
                body.set_alpha(0.85)
            if "cmedians" in vp1:
                vp1["cmedians"].set_color("black")
                vp1["cmedians"].set_linewidth(1.0)

        # violin 2: ref/rest
        if len(vals_ref) > 0:
            vp2 = ax.violinplot(
                [vals_ref],
                positions=[pos_right[idx]],
                widths=width,
                showmeans=False,
                showmedians=True,
                showextrema=False
            )
            for body in vp2["bodies"]:
                body.set_facecolor(color_map[ref_label])
                body.set_edgecolor("black")
                body.set_alpha(0.85)
            if "cmedians" in vp2:
                vp2["cmedians"].set_color("black")
                vp2["cmedians"].set_linewidth(1.0)

        # Add a small number of scatter points for readability; subsample again after sampling
        n_dot = min(250, len(vals_group))
        if n_dot > 0:
            dot_idx = np.random.choice(len(vals_group), size=n_dot, replace=False)
            jitter = (np.random.rand(n_dot) - 0.5) * 0.08
            ax.scatter(
                np.full(n_dot, pos_left[idx]) + jitter,
                vals_group[dot_idx],
                s=3,
                c="black",
                alpha=0.6,
                linewidths=0
            )

        n_dot = min(250, len(vals_ref))
        if n_dot > 0:
            dot_idx = np.random.choice(len(vals_ref), size=n_dot, replace=False)
            jitter = (np.random.rand(n_dot) - 0.5) * 0.08
            ax.scatter(
                np.full(n_dot, pos_right[idx]) + jitter,
                vals_ref[dot_idx],
                s=3,
                c="black",
                alpha=0.6,
                linewidths=0
            )

    # Style refinement
    ax.set_title(f"{group} vs. {ref_label}", fontsize=20, pad=10)
    ax.set_xlabel("genes", fontsize=16)
    ax.set_ylabel("expression", fontsize=16)

    ax.set_xticks(positions)
    ax.set_xticklabels(genes, rotation=90, fontsize=11)

    ax.set_facecolor("white")
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.tick_params(axis="both", labelsize=11, width=1.0, length=4)

    # Legend
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=color_map[str(group)], edgecolor="black", label=str(group)),
        Patch(facecolor=color_map[ref_label], edgecolor="black", label=ref_label),
    ]
    ax.legend(handles=legend_handles, frameon=False, loc="upper right", fontsize=11)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()
