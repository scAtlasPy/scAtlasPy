from ..data import Atlas
import pandas as pd
import os
from os import PathLike
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from datetime import datetime


def highest_expr_genes(
        atlas: Atlas,
        n_top: int = 20,
        use_all_cells: bool = True,
        show_outliers: bool = True,
        max_outliers: int = 5000,
        figsize: tuple[float, float] | None=(12, 10),
        approx_quantile: bool = True,
        sample_cells: int | None = None,
        save_path: PathLike[str] | str | None = None
) -> None:

    """Plot a QC figure showing the percentage of total counts occupied by the highest expressed genes.

    This function calculates the percentage of each gene in each cell's total counts
    based on ``X_HyS_data.data_count``, and selects the top ``n_top`` genes with the
    highest average percentage. Each row in the figure corresponds to one gene.
    The x-axis represents the percentage of that gene in a single cell's total counts.
    The boxplot shows the distribution across cells, and optional hollow points show
    outlier cells.

    This plot is commonly used for post-import QC, helping detect whether a small
    number of genes "dominate" the expression counts, such as mitochondrial genes,
    ribosomal genes, hemoglobin genes, or other genes related to technical bias.

    Parameters
    ----------
    atlas
        Atlas object. It must already be connected to a DuckDB database, and the database
        must contain the ``obs``, ``var``, and ``X_HyS_data`` tables.
        ``X_HyS_data`` needs to contain the ``data_count`` field.

    n_top
        Number of genes with the highest average percentage to display.

    use_all_cells
        Whether to include all cells in the top-gene and boxplot statistics.
        If ``True``, cells where the gene is not detected are counted as 0, which is
        more consistent with the definition of "percentage of total counts across all cells".
        If ``False``, the calculation is mainly based on nonzero expression records,
        which is faster but does not include implicit zeros in the distribution.

    show_outliers
        Whether to additionally draw outlier cell points on the boxplot.

    max_outliers
        Maximum number of outlier points to draw for each gene, to avoid overly dense
        figures on large datasets.

    figsize
        Matplotlib figure size.

    approx_quantile
        Whether to use DuckDB's approximate quantile function to calculate quartiles.
        It is recommended to keep this as ``True`` for large datasets, as it can reduce
        memory and computational pressure. If ``False``, exact quantiles are used.

    sample_cells
        Number of cells sampled for plotting statistics. If ``None``, all cells satisfying
        the ``use_all_cells`` logic are used. When an integer is provided, approximate
        sampling based on the hash of ``atlas_cell_id`` is used, which is suitable for
        very large datasets.
    save_path
        Path for saving the figure. If ``None``, the figure is only displayed and not saved.

    Returns
    -------
    None
        The function directly plots the figure and does not return a statistics table.

    Notes
    -----
    If ``cell_total_counts`` or ``total_counts`` already exists in ``obs``, the function
    will reuse that column first. Otherwise, it temporarily aggregates each cell's total
    counts from ``X_HyS_data.data_count``.

    Examples
    --------
    Plot the top 20 genes with the highest average percentage::

        sap.pl.highest_expr_genes(atlas, n_top=20)

    Plot with sampling on large datasets and use approximate quantiles::

        sap.pl.highest_expr_genes(
            atlas,
            n_top=30,
            sample_cells=100000,
            approx_quantile=True,
        )
    """

    start = datetime.now()
    conn = atlas.connection

    try:
        conn.execute(f"PRAGMA threads={os.cpu_count()}")
    except Exception:
        pass

    # Use hash-based approximate sampling in large-data scenarios to avoid full-table ORDER BY RANDOM() sorting
    if sample_cells is not None:
        sample_cells = int(sample_cells)
        if sample_cells <= 0:
            sample_cells = None

    # If sample_cells is specified, create a lightweight TEMP VIEW instead of materializing random-sorted results
    if sample_cells is not None:
        n_cells_total = conn.execute("SELECT COUNT(*) FROM obs").fetchone()[0]

        if sample_cells < n_cells_total:
            sample_mod = max(1, int(n_cells_total // sample_cells))

            conn.execute(f"""
                CREATE OR REPLACE TEMP VIEW _sample_cells AS
                SELECT atlas_cell_id
                FROM obs
                WHERE (hash(atlas_cell_id) % {sample_mod}) = 0
            """)
        else:
            conn.execute("""
                CREATE OR REPLACE TEMP VIEW _sample_cells AS
                SELECT atlas_cell_id
                FROM obs
            """)

    # Prefer reusing obs.cell_total_counts to avoid scanning the entire X_HyS_data each time
    obs_cols = {r[1] for r in conn.execute("PRAGMA table_info('obs')").fetchall()}

    if "cell_total_counts" in obs_cols:
        conn.execute("""
            CREATE OR REPLACE TEMP VIEW _cell_total_counts AS 
            SELECT
                atlas_cell_id,
                cell_total_counts AS total_counts
            FROM obs
            WHERE cell_total_counts IS NOT NULL
        """)
    elif "total_counts" in obs_cols:
        conn.execute("""
            CREATE OR REPLACE TEMP VIEW _cell_total_counts AS
            SELECT
                atlas_cell_id,
                total_counts AS total_counts
            FROM obs
            WHERE total_counts IS NOT NULL
        """)
    else:
        # fallback: scan X only when no precomputed result exists
        conn.execute("""
            CREATE OR REPLACE TEMP TABLE _cell_total_counts AS
            SELECT
                atlas_cell_id,
                SUM(data_count) AS total_counts
            FROM X_HyS_data
            GROUP BY atlas_cell_id
        """)

    # Select top genes
    if use_all_cells:
        if sample_cells is not None:
            conn.execute("""
                CREATE OR REPLACE TEMP VIEW _all_cells AS
                SELECT atlas_cell_id
                FROM _sample_cells
            """)
        else:
            conn.execute("""
                CREATE OR REPLACE TEMP VIEW _all_cells AS
                SELECT atlas_cell_id
                FROM obs
            """)

        conn.execute("""
            CREATE OR REPLACE TEMP TABLE _all_gene_mean_pct AS
            WITH gene_pct_nonzero AS (
                SELECT
                    x.atlas_cell_id,
                    x.atlas_gene_id,
                    x.data_count * 100.0 / t.total_counts AS pct
                FROM X_HyS_data x
                JOIN _cell_total_counts t
                    ON x.atlas_cell_id = t.atlas_cell_id
                JOIN _all_cells c                      
                    ON x.atlas_cell_id = c.atlas_cell_id
                WHERE t.total_counts > 0
            ),
            gene_sum_pct AS (
                SELECT
                    atlas_gene_id,
                    SUM(pct) AS sum_pct
                FROM gene_pct_nonzero
                GROUP BY atlas_gene_id
            ),
            n_cells AS (
                SELECT COUNT(*) AS total_cells
                FROM _all_cells
            )
            SELECT
                v.atlas_gene_id,
                v.atlas_gene_name,
                COALESCE(g.sum_pct, 0.0) / n.total_cells AS mean_pct
            FROM var v
            CROSS JOIN n_cells n
            LEFT JOIN gene_sum_pct g
              ON v.atlas_gene_id = g.atlas_gene_id
        """)

        conn.execute(f"""
            CREATE OR REPLACE TEMP TABLE _top_expr_genes AS
            SELECT
                atlas_gene_id,
                atlas_gene_name,
                mean_pct
            FROM _all_gene_mean_pct
            ORDER BY mean_pct DESC
            LIMIT {int(n_top)}
        """)
    else:
        conn.execute(f"""
            CREATE OR REPLACE TEMP TABLE _top_expr_genes AS
            SELECT
                x.atlas_gene_id,
                v.atlas_gene_name,
                AVG(x.data_count * 100.0 / t.total_counts) AS mean_pct
            FROM X_HyS_data x
            JOIN _cell_total_counts t
                ON x.atlas_cell_id = t.atlas_cell_id
            JOIN var v
                ON x.atlas_gene_id = v.atlas_gene_id
            WHERE t.total_counts > 0
            GROUP BY x.atlas_gene_id, v.atlas_gene_name
            ORDER BY mean_pct DESC
            LIMIT {int(n_top)}
        """)

    # Directly calculate standard boxplot statistics in SQL
    qfunc = "approx_quantile" if approx_quantile else "quantile_cont"

    # When use_all_cells=False, boxplot statistics can be based on sampled cells to avoid overly heavy scanning/aggregation for 1e8 cells
    sample_join_sql = ""
    if sample_cells is not None:
        sample_join_sql = """
            JOIN _sample_cells s
                ON x.atlas_cell_id = s.atlas_cell_id
        """

    if use_all_cells:
        stats_df = conn.execute(f"""  
            WITH all_cells AS (
                SELECT atlas_cell_id
                FROM _all_cells 
            ),
            cell_gene_grid AS (
                SELECT
                    c.atlas_cell_id,
                    g.atlas_gene_id,
                    g.atlas_gene_name,
                    g.mean_pct
                FROM all_cells c
                CROSS JOIN _top_expr_genes g
            ),
            top_gene_pct AS (
                SELECT
                    x.atlas_cell_id,
                    x.atlas_gene_id,
                    x.data_count * 100.0 / t.total_counts AS pct
                FROM X_HyS_data x
                JOIN _cell_total_counts t
                    ON x.atlas_cell_id = t.atlas_cell_id
                JOIN _top_expr_genes g
                    ON x.atlas_gene_id = g.atlas_gene_id
                JOIN all_cells c           
                    ON x.atlas_cell_id = c.atlas_cell_id
                WHERE t.total_counts > 0
            ),
            full_values AS (
                SELECT
                    grid.atlas_gene_name,
                    grid.mean_pct,
                    COALESCE(p.pct, 0.0) AS pct
                FROM cell_gene_grid grid
                LEFT JOIN top_gene_pct p
                    ON grid.atlas_cell_id = p.atlas_cell_id
                   AND grid.atlas_gene_id = p.atlas_gene_id
            ),
            quartiles AS (
                SELECT
                    atlas_gene_name,
                    mean_pct,
                    COUNT(*) AS n,
                    {qfunc}(pct, 0.25) AS q1,     
                    {qfunc}(pct, 0.50) AS median, 
                    {qfunc}(pct, 0.75) AS q3     
                FROM full_values
                GROUP BY atlas_gene_name, mean_pct
            ),
            whiskers AS (
                SELECT
                    f.atlas_gene_name,
                    MIN(CASE
                            WHEN f.pct >= (q.q1 - 1.5 * (q.q3 - q.q1))
                            THEN f.pct
                        END) AS whisker_low,
                    MAX(CASE
                            WHEN f.pct <= (q.q3 + 1.5 * (q.q3 - q.q1))
                            THEN f.pct
                        END) AS whisker_high
                FROM full_values f
                JOIN quartiles q
                  ON f.atlas_gene_name = q.atlas_gene_name
                GROUP BY f.atlas_gene_name
            )
            SELECT
                q.atlas_gene_name,
                q.mean_pct,
                q.n,
                COALESCE(w.whisker_low, q.q1) AS whisker_low,
                q.q1,
                q.median,
                q.q3,
                COALESCE(w.whisker_high, q.q3) AS whisker_high
            FROM quartiles q
            LEFT JOIN whiskers w
              ON q.atlas_gene_name = w.atlas_gene_name
            ORDER BY q.mean_pct DESC, q.atlas_gene_name
        """).fetchdf()
    else:
        # Materialize the nonzero expression values of top genes into a temporary table, so the later quartiles / whiskers do not repeatedly rerun the same large CTE
        stats_df = conn.execute(f"""
                WITH quartiles AS (
                    SELECT
                        g.atlas_gene_id,
                        g.atlas_gene_name,
                        MAX(g.mean_pct) AS mean_pct,
                        COUNT(*) AS n,
                        {qfunc}(CAST(x.data_count * 100.0 / t.total_counts AS DOUBLE), 0.25) AS q1,
                        {qfunc}(CAST(x.data_count * 100.0 / t.total_counts AS DOUBLE), 0.50) AS median,
                        {qfunc}(CAST(x.data_count * 100.0 / t.total_counts AS DOUBLE), 0.75) AS q3
                    FROM X_HyS_data x
                    {sample_join_sql}                 -- If sample_cells is not empty, calculate the boxplot only on sampled cells
                    JOIN _top_expr_genes g
                        ON x.atlas_gene_id = g.atlas_gene_id
                    JOIN _cell_total_counts t
                        ON x.atlas_cell_id = t.atlas_cell_id
                    WHERE t.total_counts > 0
                    GROUP BY g.atlas_gene_id, g.atlas_gene_name
                ),
                whiskers AS (
                    SELECT
                        q.atlas_gene_name,
                        MIN(CASE
                                WHEN CAST(x.data_count * 100.0 / t.total_counts AS DOUBLE)
                                     >= (q.q1 - 1.5 * (q.q3 - q.q1))
                                THEN CAST(x.data_count * 100.0 / t.total_counts AS DOUBLE)
                            END) AS whisker_low,
                        MAX(CASE
                                WHEN CAST(x.data_count * 100.0 / t.total_counts AS DOUBLE)
                                     <= (q.q3 + 1.5 * (q.q3 - q.q1))
                                THEN CAST(x.data_count * 100.0 / t.total_counts AS DOUBLE)
                            END) AS whisker_high
                    FROM X_HyS_data x
                    {sample_join_sql}                 -- If sample_cells is not empty, calculate whiskers only on sampled cells
                    JOIN _top_expr_genes g
                        ON x.atlas_gene_id = g.atlas_gene_id
                    JOIN _cell_total_counts t
                        ON x.atlas_cell_id = t.atlas_cell_id
                    JOIN quartiles q
                        ON g.atlas_gene_id = q.atlas_gene_id
                    WHERE t.total_counts > 0
                    GROUP BY q.atlas_gene_name
                )
                SELECT
                    q.atlas_gene_name,
                    q.mean_pct,
                    q.n,
                    COALESCE(w.whisker_low, q.q1) AS whisker_low,
                    q.q1,
                    q.median,
                    q.q3,
                    COALESCE(w.whisker_high, q.q3) AS whisker_high
                FROM quartiles q
                LEFT JOIN whiskers w
                  ON q.atlas_gene_name = w.atlas_gene_name
                ORDER BY q.mean_pct DESC, q.atlas_gene_name
            """).fetchdf()

    # Extract outliers
    outlier_df = None

    if show_outliers:

        if use_all_cells:
            outlier_df = conn.execute(f"""  -- Changed to an f-string to support qfunc
                WITH all_cells AS (
                    SELECT atlas_cell_id
                    FROM _all_cells              -- Read from _all_cells; if sample_cells is not empty, sampled cells are used automatically
                ),
                cell_gene_grid AS (
                    SELECT
                        c.atlas_cell_id,
                        g.atlas_gene_id,
                        g.atlas_gene_name
                    FROM all_cells c
                    CROSS JOIN _top_expr_genes g
                ),
                top_gene_pct AS (
                    SELECT
                        x.atlas_cell_id,
                        x.atlas_gene_id,
                        x.data_count * 100.0 / t.total_counts AS pct
                    FROM X_HyS_data x
                    JOIN _cell_total_counts t
                        ON x.atlas_cell_id = t.atlas_cell_id
                    JOIN _top_expr_genes g
                        ON x.atlas_gene_id = g.atlas_gene_id
                    JOIN all_cells c              -- Restrict to _all_cells, supporting sample_cells
                        ON x.atlas_cell_id = c.atlas_cell_id
                    WHERE t.total_counts > 0
                ),
                full_values AS (
                    SELECT
                        grid.atlas_gene_name,
                        COALESCE(p.pct, 0.0) AS pct
                    FROM cell_gene_grid grid
                    LEFT JOIN top_gene_pct p
                        ON grid.atlas_cell_id = p.atlas_cell_id
                       AND grid.atlas_gene_id = p.atlas_gene_id
                ),
                bounds AS (
                    SELECT
                        atlas_gene_name,
                        {qfunc}(pct, 0.25) AS q1,  -- quantile_cont changed to switchable qfunc
                        {qfunc}(pct, 0.75) AS q3   -- quantile_cont changed to switchable qfunc
                    FROM full_values
                    GROUP BY atlas_gene_name
                ),
                outliers AS (
                    SELECT
                        f.atlas_gene_name,
                        f.pct,
                        ROW_NUMBER() OVER (
                            PARTITION BY f.atlas_gene_name
                            ORDER BY RANDOM()
                        ) AS rn
                    FROM full_values f
                    JOIN bounds b
                      ON f.atlas_gene_name = b.atlas_gene_name
                    WHERE f.pct < (b.q1 - 1.5 * (b.q3 - b.q1))
                       OR f.pct > (b.q3 + 1.5 * (b.q3 - b.q1))
                )
                SELECT atlas_gene_name, pct
                FROM outliers
                WHERE rn <= {int(max_outliers)}
            """).fetchdf()
        else:
            outlier_df = conn.execute(f"""
                WITH top_gene_values AS (
                    SELECT
                        g.atlas_gene_name,
                        x.data_count * 100.0 / t.total_counts AS pct
                    FROM X_HyS_data x
                    {sample_join_sql}                 -- If sample_cells is not empty, extract outliers only from sampled cells
                    JOIN _cell_total_counts t
                        ON x.atlas_cell_id = t.atlas_cell_id
                    JOIN _top_expr_genes g
                        ON x.atlas_gene_id = g.atlas_gene_id
                    WHERE t.total_counts > 0
                ),
                bounds AS (
                    SELECT
                        atlas_gene_name,
                        {qfunc}(pct, 0.25) AS q1, 
                        {qfunc}(pct, 0.75) AS q3   
                    FROM top_gene_values
                    GROUP BY atlas_gene_name
                ),
                outliers AS (
                    SELECT
                        f.atlas_gene_name,
                        f.pct,
                        ROW_NUMBER() OVER (
                            PARTITION BY f.atlas_gene_name
                            ORDER BY RANDOM()
                        ) AS rn
                    FROM top_gene_values f
                    JOIN bounds b
                      ON f.atlas_gene_name = b.atlas_gene_name
                    WHERE f.pct < (b.q1 - 1.5 * (b.q3 - b.q1))
                       OR f.pct > (b.q3 + 1.5 * (b.q3 - b.q1))
                )
                SELECT atlas_gene_name, pct
                FROM outliers
                WHERE rn <= {int(max_outliers)}
            """).fetchdf()

    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    ax.set_facecolor("white")

    stats_df = stats_df.sort_values("mean_pct", ascending=False).reset_index(drop=True)

    y_positions = list(range(len(stats_df), 0, -1))
    box_height = 0.78

    scanpy_colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#bcbd22", "#17becf", "#aec7e8",
        "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5", "#c49c94",
        "#f7b6d2", "#dbdb8d", "#9edae5", "#8c6d31", "#7f7f7f"
    ]

    for i, (y, row) in enumerate(zip(y_positions, stats_df.itertuples(index=False))):
        whisker_low = float(row.whisker_low)
        q1 = float(row.q1)
        median = float(row.median)
        q3 = float(row.q3)
        whisker_high = float(row.whisker_high)

        color = scanpy_colors[i % len(scanpy_colors)]

        ax.hlines(y, whisker_low, whisker_high, linewidth=1.1, color="#4a4a4a", zorder=2)
        ax.vlines(whisker_low, y - 0.11, y + 0.11, linewidth=1.1, color="#4a4a4a", zorder=2)
        ax.vlines(whisker_high, y - 0.11, y + 0.11, linewidth=1.1, color="#4a4a4a", zorder=2)

        rect = Rectangle(
            (q1, y - box_height / 2),
            max(q3 - q1, 1e-12),
            box_height,
            facecolor=color,
            edgecolor="#4a4a4a",
            linewidth=1.0,
            alpha=0.9,
            zorder=3
        )
        ax.add_patch(rect)

        ax.vlines(
            median,
            y - box_height / 2,
            y + box_height / 2,
            linewidth=1.4,
            color="#2f2f2f",
            zorder=4
        )

    # Optional outliers
    if show_outliers and outlier_df is not None and len(outlier_df) > 0:
        gene_to_y = {g: y for g, y in zip(stats_df["atlas_gene_name"], y_positions)}

        xs = outlier_df["pct"].to_numpy()
        ys = [gene_to_y[g] for g in outlier_df["atlas_gene_name"]]

        ax.scatter(
            xs,
            ys,
            s=8,
            facecolors="none",
            edgecolors="#2f2f2f",
            linewidths=0.5,
            alpha=0.7,
            zorder=5
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(stats_df["atlas_gene_name"].tolist(), fontsize=16)
    ax.set_xlabel("% of total counts", fontsize=18)
    ax.set_ylabel("")
    ax.set_title("Highest expressed genes", fontsize=12, weight="normal", pad=8)

    ax.grid(True, axis="x", color="#d9d9d9", linewidth=0.8, alpha=0.8)
    ax.grid(False, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)

    ax.tick_params(axis="x", labelsize=12, width=1.0, length=4)
    ax.tick_params(axis="y", width=1.0, length=4)

    plt.tight_layout(pad=0.8)
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

    # Clean up
    # Objects with the same name in DuckDB may be either VIEW or TABLE; directly using DROP VIEW may raise an error due to type mismatch
    def _safe_drop_temp(name: str):
        """Clean up temporary tables or temporary views created by ``highest_expr_genes``.

        Objects with the same name in DuckDB may be either VIEW or TABLE. This helper
        tries ``DROP VIEW`` and ``DROP TABLE`` separately to ensure intermediate objects
        are cleaned up after the function finishes, while preventing cleanup errors such
        as "object type mismatch" from affecting the plotting result.

        Parameters
        ----------
        name
            Name of the temporary object to clean up.
        """
        try:
            conn.execute(f"DROP VIEW IF EXISTS {name}")
        except Exception:
            pass

        try:
            conn.execute(f"DROP TABLE IF EXISTS {name}")
        except Exception:
            pass

    _safe_drop_temp("_sample_cells")
    _safe_drop_temp("_cell_total_counts")
    _safe_drop_temp("_all_cells")
    _safe_drop_temp("_top_expr_genes")
    _safe_drop_temp("_top_gene_values")
    _safe_drop_temp("_all_gene_mean_pct")


def violin_qc_metrics(
        atlas: Atlas,
        keys: str | list[str] | None = None,
        jitter: float = 0.4,
        multi_panel: bool = True,
        figsize: tuple[float, float] | None=None,
        use_filtered: bool = False,
        filter_key: str = "filter_cells",
        sample_n: int | None = 50000,
        random_state: int = 0,
        save_path: PathLike[str] | str | None = None
) -> None:

    """Plot violin plots for QC metrics in ``obs``.

    This function reads one or more cell-level QC metrics from the ``obs`` table and
    draws a violin plot for each metric. It also overlays slightly jittered scatter
    points to show the real cell distribution. By default, it plots
    ``n_genes_by_counts``, ``cell_total_counts``, and ``pct_counts_mt``.

    This plot is commonly used to inspect the overall distribution of QC metrics such
    as sequencing depth, number of detected genes, and mitochondrial percentage, and
    to help decide filtering thresholds.

    Parameters
    ----------
    atlas
        Atlas object. It must already be connected to a DuckDB database, and the
        ``obs`` table must contain the QC columns to be plotted.

    keys
        QC metric column names to plot. This can be a single string or a list of strings.
        If ``None``, the default QC metric list is used.

    jitter
        Jitter strength for scatter points in the violin plot.

    multi_panel
        Whether to use a horizontal multi-panel layout for multiple metrics.
        If ``False``, multiple metrics are arranged vertically.

    figsize
        Matplotlib figure size. If ``None``, it is estimated automatically according
        to the number of metrics.

    use_filtered
        Whether to plot only cells where ``filter_key = TRUE``.

    filter_key
        Boolean column name in ``obs`` indicating whether a cell passed filtering.
        Defaults to ``"filter_cells"``.

    sample_n
        Number of cells sampled from ``obs``. If ``None``, all cells satisfying the
        conditions are used.

    random_state
        Random seed; using a fixed integer improves reproducibility.
    save_path
        Path for saving the figure. If ``None``, the figure is only displayed and not saved.

    Notes
    -----
    Before plotting, you usually need to run ``sap.pp.calculate_qc_metrics`` or another
    preprocessing function that writes the corresponding QC columns. Columns that are
    entirely empty are skipped.

    Examples
    --------
    Plot the default QC metrics::

        sap.pl.violin_qc_metrics(atlas)

    Plot only filtered cells and specify metrics::

        sap.pl.violin_qc_metrics(
            atlas,
            keys=["cell_total_counts", "pct_counts_mt"],
            use_filtered=True,
        )
    """

    start = datetime.now()
    conn = atlas.connection

    if keys is None:
        keys = ["n_genes_by_counts", "cell_total_counts", "pct_counts_mt"]

    # Check whether obs columns exist
    obs_cols = [r[1] for r in conn.execute("PRAGMA table_info(obs)").fetchall()]

    missing = [k for k in keys if k not in obs_cols]
    if missing:
        raise ValueError(
            f"These columns do not exist in obs: {missing}\n"
            f"Please run sap.pp.calculate_qc_metrics(atlas) first"
        )

    if use_filtered and filter_key not in obs_cols:
        raise ValueError(f"The filtering column does not exist in obs: {filter_key}")

    # Sample first in SQL, then fetchdf
    select_cols = ", ".join(keys)

    where_clauses = []
    if use_filtered:
        where_clauses.append(f"{filter_key} = TRUE")

    # Keep only rows where at least one key is non-null to reduce invalid points
    non_null_cond = " OR ".join([f"{k} IS NOT NULL" for k in keys])
    where_clauses.append(f"({non_null_cond})")

    where_sql = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""

    if sample_n is None:
        # Full data, not recommended for large datasets
        query = f"""
            SELECT {select_cols}
            FROM obs
            {where_sql}
        """
    else:
        # Sample first in SQL, then fetchdf
        query = f"""
            SELECT {select_cols}
            FROM obs
            {where_sql}
            USING SAMPLE {int(sample_n)} ROWS
        """

    df = conn.execute(query).fetchdf()

    # Clean columns
    keep_keys = []
    for k in keys:
        if k in df.columns and df[k].notna().sum() > 0:
            keep_keys.append(k)

    if len(keep_keys) == 0:
        raise ValueError("No QC columns available for plotting; all columns are empty")

    df = df[keep_keys]
    keys = keep_keys

    # Automatic layout
    n = len(keys)

    if figsize is None:
        if multi_panel:
            figsize = (4.2 * n, 5.0)
        else:
            figsize = (6.0, 4.0 * n)

    if multi_panel:
        fig, axes = plt.subplots(1, n, figsize=figsize, facecolor="white")
        if n == 1:
            axes = [axes]
    else:
        fig, axes = plt.subplots(n, 1, figsize=figsize, facecolor="white")
        if n == 1:
            axes = [axes]

    violin_color = "#1f77b4"
    edge_color = "#4a4a4a"
    point_color = "#2f2f2f"

    # Draw violin + jitter column by column
    for ax, key in zip(axes, keys):
        vals = pd.to_numeric(df[key], errors="coerce").dropna().to_numpy()

        if len(vals) == 0:
            ax.set_visible(False)
            continue

        # violin
        parts = ax.violinplot(
            dataset=[vals],
            positions=[1],
            widths=0.9,
            showmeans=False,
            showmedians=True,
            showextrema=False
        )

        for pc in parts["bodies"]:
            pc.set_facecolor(violin_color)
            pc.set_edgecolor(edge_color)
            pc.set_alpha(0.9)
            pc.set_linewidth(1.0)

        if "cmedians" in parts:
            parts["cmedians"].set_color("#2f2f2f")
            parts["cmedians"].set_linewidth(1.2)

        # jitter points
        y_jitter = 1 + (pd.Series(range(len(vals))).sample(frac=1, random_state=random_state).rank().to_numpy() / len(vals) - 0.5) * jitter

        ax.scatter(
            y_jitter,
            vals,
            s=2,
            c=point_color,
            alpha=0.55,
            linewidths=0
        )

        ax.set_title(key, fontsize=12, weight="normal", pad=8)
        ax.set_xlim(0.5, 1.5)
        ax.set_xticks([1])
        ax.set_xticklabels([""])
        ax.set_ylabel("value", fontsize=11)

        ax.grid(True, axis="y", color="#d9d9d9", linewidth=0.8, alpha=0.8)
        ax.grid(False, axis="x")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.0)
        ax.spines["bottom"].set_linewidth(1.0)

        ax.tick_params(axis="y", labelsize=10, width=1.0, length=4)
        ax.tick_params(axis="x", width=1.0, length=0)

    plt.tight_layout(pad=1.0)
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def scatter_qc_metrics(
        atlas: Atlas,
        pairs: list[tuple[str, str]] | None=None,
        figsize: tuple[float, float] | None=(10, 4),
        use_filtered: bool = False,
        filter_key: str = "filter_cells",
        sample_n: int | None = 50000,
        point_size: float = 8,
        alpha: float = 0.7,
        save_path: PathLike[str] | str | None = None
) -> None:

    """Plot scatter plots for pairwise relationships between QC metrics in ``obs``.

    This function reads one or more pairs of QC metrics from the ``obs`` table and
    displays their relationships using scatter plots.
    By default, it plots ``cell_total_counts`` versus ``pct_counts_mt`` and
    ``cell_total_counts`` versus ``n_genes_by_counts``.

    This plot is suitable for helping decide filtering thresholds, such as identifying
    cells with low sequencing depth, high mitochondrial percentage, suspected doublets,
    or other QC-abnormal populations.

    Parameters
    ----------
    atlas
        Atlas object. It must already be connected to a DuckDB database, and the
        ``obs`` table must contain the QC columns to be plotted.

    pairs
        List of metric pairs to plot, for example
        ``[("cell_total_counts", "pct_counts_mt")]``.
        If ``None``, the two default QC relationships are used.

    figsize
        Matplotlib figure size.

    use_filtered
        Whether to plot only cells where ``filter_key = TRUE``.

    filter_key
        Boolean column name in ``obs`` indicating whether a cell passed filtering.
        Defaults to ``"filter_cells"``.

    sample_n
        Number of cells sampled from ``obs``. If ``None``, all cells satisfying the
        conditions are used.

    point_size
        Scatter point size.

    alpha
        Plotting transparency.

    save_path
        Path for saving the figure. If ``None``, the figure is only displayed and not saved.

    Notes
    -----
    Before plotting, you usually need to run ``sap.pp.calculate_qc_metrics`` or another
    preprocessing function that writes the corresponding QC columns. The function
    automatically filters cells where either column in the current metric pair is empty.

    Examples
    --------
    Plot the default QC scatter plots::

        sap.pl.scatter_qc_metrics(atlas)

    Customize QC metric pairs::

        sap.pl.scatter_qc_metrics(
            atlas,
            pairs=[("cell_total_counts", "n_genes_by_counts")],
            sample_n=100000,
        )
    """

    start = datetime.now()
    conn = atlas.connection

    if pairs is None:
        pairs = [
            ("cell_total_counts", "pct_counts_mt"),
            ("cell_total_counts", "n_genes_by_counts"),
        ]

    # Check whether obs columns exist
    obs_cols = [r[1] for r in conn.execute("PRAGMA table_info(obs)").fetchall()]

    needed_cols = set()
    for x, y in pairs:
        needed_cols.add(x)
        needed_cols.add(y)

    missing = [c for c in needed_cols if c not in obs_cols]
    if missing:
        raise ValueError(
            f"These columns do not exist in obs: {missing}\n"
            f"Please run sap.pp.calculate_qc_metrics(atlas) first"
        )

    if use_filtered and filter_key not in obs_cols:
        raise ValueError(f"The filtering column does not exist in obs: {filter_key}")

    # Sample first in SQL, then fetchdf
    select_cols = ", ".join(sorted(needed_cols))

    where_clauses = []

    if use_filtered:
        where_clauses.append(f"{filter_key} = TRUE")

    # At least ensure that x/y in each pair are not both empty
    pair_valid_clauses = []
    for x, y in pairs:
        pair_valid_clauses.append(f"({x} IS NOT NULL AND {y} IS NOT NULL)")

    if pair_valid_clauses:
        where_clauses.append("(" + " OR ".join(pair_valid_clauses) + ")")

    where_sql = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""

    if sample_n is None:
        query = f"""
            SELECT {select_cols}
            FROM obs
            {where_sql}
        """
    else:
        # Sample first in SQL
        query = f"""
            SELECT {select_cols}
            FROM obs
            {where_sql}
            USING SAMPLE {int(sample_n)} ROWS
        """

    df = conn.execute(query).fetchdf()

    # Automatic layout
    n = len(pairs)

    fig, axes = plt.subplots(1, n, figsize=figsize, facecolor="white")
    if n == 1:
        axes = [axes]

    # Draw each panel
    for ax, (x_col, y_col) in zip(axes, pairs):
        sub = df[[x_col, y_col]].copy()
        sub[x_col] = pd.to_numeric(sub[x_col], errors="coerce")
        sub[y_col] = pd.to_numeric(sub[y_col], errors="coerce")
        sub = sub.dropna()

        if len(sub) == 0:
            ax.set_visible(False)
            continue

        ax.scatter(
            sub[x_col].to_numpy(),
            sub[y_col].to_numpy(),
            s=point_size,
            c="#7f7f7f",      # Scanpy-like gray points
            alpha=alpha,
            linewidths=0
        )

        ax.set_xlabel(x_col, fontsize=16)
        ax.set_ylabel(y_col, fontsize=16)

        # Scanpy style: white background + light grid + remove the top and right borders
        ax.set_facecolor("white")
        ax.grid(True, color="#d9d9d9", linewidth=0.8, alpha=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.0)
        ax.spines["bottom"].set_linewidth(1.0)

        ax.tick_params(axis="both", labelsize=12, width=1.0, length=4)

    plt.tight_layout(pad=1.0)
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()
