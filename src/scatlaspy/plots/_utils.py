from __future__ import annotations

from os import PathLike
from typing import Any, Callable
import re

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np

DEFAULT_CELL_PLOT_SAMPLE_N = 1_000_000
DEFAULT_GROUP_PLOT_SAMPLE_N = 5_000
DEFAULT_EMBEDDING_FIGSIZE = (6.0, 5.0)
DEFAULT_EMBEDDING_PANEL_SIZE = (6.0, 5.6)
DOTPLOT_TARGET_CELL_GENE_VALUES = 5_000_000
DOTPLOT_MIN_SAMPLE_PER_GROUP = 2_000
DOTPLOT_MAX_SAMPLE_PER_GROUP = 50_000
_MISSING_CATEGORY_LABELS = {"", "na", "nan", "none", "<na>", "null"}


def default_point_size(n_points: int) -> float:
    """Estimate scatter point size from the number of plotted cells."""
    return 120_000 / max(int(n_points), 1)


def default_embedding_figsize() -> tuple[float, float]:
    """Return the default size for one PCA/UMAP embedding panel."""

    return DEFAULT_EMBEDDING_FIGSIZE


def estimate_embedding_grid_figsize(
    n_panels: int,
    ncols: int,
    *,
    panel_size: tuple[float, float] = DEFAULT_EMBEDDING_PANEL_SIZE,
) -> tuple[float, float]:
    """Estimate a figure size for multi-panel PCA/UMAP-like plots."""

    n_panels = max(int(n_panels), 1)
    ncols_eff = min(max(int(ncols), 1), n_panels)
    nrows_eff = int(np.ceil(n_panels / ncols_eff))
    width = float(panel_size[0]) * ncols_eff
    height = float(panel_size[1]) * nrows_eff
    return (width, height)


def quote_identifier(name: str) -> str:
    """Quote a DuckDB SQL identifier."""

    return '"' + str(name).replace('"', '""') + '"'


def natural_sort_key(value: Any) -> tuple[int, tuple[Any, ...]]:
    """Return a natural sorting key for categorical labels."""

    s = str(value).strip()
    if s.casefold() in _MISSING_CATEGORY_LABELS:
        return (1, ())

    key = []
    for part in re.split(r"(\d+)", s):
        if part == "":
            continue
        if part.isdigit():
            key.append((0, int(part)))
        else:
            key.append((1, part.casefold()))

    return (0, tuple(key))


def sort_categories_natural(labels: Any) -> list[str]:
    """Deduplicate labels and sort them naturally."""

    labels = [str(x) for x in list(labels)]
    labels = list(dict.fromkeys(labels))
    return sorted(labels, key=natural_sort_key)


def build_discrete_color_map(labels: Any, palette: Any | None) -> dict[str, Any]:
    """Build a deterministic discrete color map for category labels."""

    labels = list(labels)
    if palette is None:
        palette_names = ("tab20", "tab20b", "tab20c", "Set3", "Paired", "Accent", "Dark2")
    elif isinstance(palette, str):
        palette_names = (palette,)
    else:
        palette_names = tuple(palette)

    palette_colors = []
    for cmap_name in palette_names:
        cmap_obj = plt.get_cmap(cmap_name)
        if hasattr(cmap_obj, "colors"):
            palette_colors.extend(list(cmap_obj.colors))
        else:
            n = getattr(cmap_obj, "N", 256)
            palette_colors.extend([
                cmap_obj(i / max(n - 1, 1))
                for i in range(n)
            ])

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


def estimate_dotplot_sample_per_group(n_groups: int, n_genes: int) -> int:
    """Estimate a bounded per-group sample size for dotplot.

    Dotplot accuracy depends on the number of cells sampled per group, while the
    temporary work roughly scales with ``n_groups * n_genes * sample_per_group``.
    This estimator increases the sample size for small marker panels and reduces
    it for very broad group-by-gene panels.
    """

    n_groups = max(int(n_groups), 1)
    n_genes = max(int(n_genes), 1)
    sample_n = DOTPLOT_TARGET_CELL_GENE_VALUES // (n_groups * n_genes)
    sample_n = max(DOTPLOT_MIN_SAMPLE_PER_GROUP, int(sample_n))
    sample_n = min(DOTPLOT_MAX_SAMPLE_PER_GROUP, int(sample_n))
    return int(sample_n)


def draw_embedding_scatter_streaming(
    conn: Any,
    *,
    embedding_table: str,
    x_col: str,
    y_col: str,
    x_label: str,
    y_label: str,
    color: str | None = None,
    where_sql: str | None = None,
    title: str | None = None,
    figsize: tuple[float, float] | None = DEFAULT_EMBEDDING_FIGSIZE,
    point_size: float | None = None,
    alpha: float = 0.8,
    palette: Any | None = None,
    legend_loc: str | None = "right_margin",
    frameon: bool = True,
    hide_ticks: bool = True,
    ax: Any | None = None,
    show: bool = True,
    adjust_layout: bool = True,
    plot_batch_size: int = 200_000,
    save_path: PathLike[str] | str | None = None,
    dpi: int = 300,
    spread_label_positions: Callable[[Any], Any] | None = None,
) -> None:
    """Draw an embedding scatter plot in batches.

    This helper is shared by PCA and UMAP plotting. It handles the two
    atlas-scale-safe cases: uncolored embedding plots and plots colored by one
    categorical ``obs`` column. It avoids materializing all cell coordinates in
    pandas by reading and drawing one batch at a time.
    """

    e_table = quote_identifier(embedding_table)
    qx = quote_identifier(x_col)
    qy = quote_identifier(y_col)
    joins_obs = color is not None or (where_sql is not None and str(where_sql).strip() != "")
    from_sql = f"{e_table} e"
    if joins_obs:
        from_sql += " JOIN obs o ON e.atlas_cell_id = o.atlas_cell_id"

    clauses = []
    if color is not None:
        clauses.append(f"o.{quote_identifier(color)} IS NOT NULL")
    if where_sql is not None and str(where_sql).strip() != "":
        clauses.append(f"({where_sql})")

    base_where_sql = " AND ".join(clauses)
    where_clause = f"WHERE {base_where_sql}" if base_where_sql else ""

    total_points = conn.execute(f"""
        SELECT COUNT(*)
        FROM {from_sql}
        {where_clause}
    """).fetchone()[0]

    if int(total_points) == 0:
        raise ValueError("No cells are available for plotting")

    if point_size is None:
        point_size = default_point_size(int(total_points))

    if title is None:
        title = color if color is not None else embedding_table

    owns_figure = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    else:
        fig = ax.figure

    ax.set_facecolor("white")

    if color is None:
        _draw_plain_embedding_batches(
            conn,
            ax=ax,
            from_sql=from_sql,
            where_clause=where_clause,
            x_col=x_col,
            y_col=y_col,
            point_size=point_size,
            alpha=alpha,
            plot_batch_size=plot_batch_size,
        )
        unique_labels = []
        label_to_color = {}
    else:
        label_df = conn.execute(f"""
            SELECT DISTINCT CAST(o.{quote_identifier(color)} AS TEXT) AS color_label
            FROM {from_sql}
            {where_clause}
            ORDER BY color_label
        """).fetchdf()

        unique_labels = sort_categories_natural(
            label_df["color_label"].astype(str).tolist()
        )
        label_to_color = build_discrete_color_map(unique_labels, palette)

        _draw_categorical_embedding_batches(
            conn,
            ax=ax,
            from_sql=from_sql,
            where_clause=where_clause,
            x_col=x_col,
            y_col=y_col,
            color=color,
            unique_labels=unique_labels,
            label_to_color=label_to_color,
            point_size=point_size,
            alpha=alpha,
            plot_batch_size=plot_batch_size,
        )

    _style_streaming_embedding_axes(
        conn,
        ax=ax,
        from_sql=from_sql,
        where_clause=where_clause,
        x_col=x_col,
        y_col=y_col,
        color=color,
        x_label=x_label,
        y_label=y_label,
        title=title,
        frameon=frameon,
        hide_ticks=hide_ticks,
        legend_loc=legend_loc,
        unique_labels=unique_labels,
        label_to_color=label_to_color,
        spread_label_positions=spread_label_positions,
        adjust_layout=adjust_layout,
    )

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    if show and owns_figure:
        plt.show()


def draw_embedding_obs_continuous_streaming(
    conn: Any,
    *,
    embedding_table: str,
    x_col: str,
    y_col: str,
    x_label: str,
    y_label: str,
    color: str,
    where_sql: str | None = None,
    title: str | None = None,
    figsize: tuple[float, float] | None = DEFAULT_EMBEDDING_FIGSIZE,
    point_size: float | None = None,
    alpha: float = 0.8,
    cmap: str = "viridis",
    frameon: bool = True,
    hide_ticks: bool = True,
    ax: Any | None = None,
    show: bool = True,
    adjust_layout: bool = True,
    plot_batch_size: int = 200_000,
    range_sample_n: int = 100_000,
    range_quantiles: tuple[float, float] = (0.01, 0.99),
    save_path: PathLike[str] | str | None = None,
    dpi: int = 300,
) -> None:
    """Draw an embedding colored by one numeric ``obs`` column in batches."""

    e_table = quote_identifier(embedding_table)
    from_sql = f"{e_table} e JOIN obs o ON e.atlas_cell_id = o.atlas_cell_id"
    clauses = [f"o.{quote_identifier(color)} IS NOT NULL"]
    if where_sql is not None and str(where_sql).strip() != "":
        clauses.append(f"({where_sql})")
    where_clause = "WHERE " + " AND ".join(clauses)

    total_points = conn.execute(f"""
        SELECT COUNT(*)
        FROM {from_sql}
        {where_clause}
    """).fetchone()[0]
    if int(total_points) == 0:
        raise ValueError("No cells are available for plotting")
    if point_size is None:
        point_size = default_point_size(int(total_points))

    range_df = conn.execute(f"""
        SELECT color_value
        FROM (
            SELECT CAST(o.{quote_identifier(color)} AS DOUBLE) AS color_value
            FROM {from_sql}
            {where_clause}
        ) t
        USING SAMPLE {int(range_sample_n)} ROWS
    """).fetchdf()
    vmin, vmax = _estimate_continuous_range(
        range_df["color_value"].to_numpy(dtype=float),
        quantiles=range_quantiles,
    )

    owns_figure = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    else:
        fig = ax.figure
    ax.set_facecolor("white")

    scatter = _draw_continuous_embedding_batches(
        conn,
        ax=ax,
        from_sql=from_sql,
        where_clause=where_clause,
        x_col=x_col,
        y_col=y_col,
        value_sql=f"CAST(o.{quote_identifier(color)} AS DOUBLE)",
        point_size=point_size,
        alpha=alpha,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        plot_batch_size=plot_batch_size,
    )

    _style_streaming_embedding_axes(
        conn,
        ax=ax,
        from_sql=from_sql,
        where_clause=where_clause,
        x_col=x_col,
        y_col=y_col,
        color=None,
        x_label=x_label,
        y_label=y_label,
        title=str(color) if title is None else title,
        frameon=frameon,
        hide_ticks=hide_ticks,
        legend_loc=None,
        unique_labels=[],
        label_to_color={},
        spread_label_positions=None,
        adjust_layout=False,
    )
    cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label(str(color), fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    if adjust_layout:
        fig.tight_layout(pad=0.8)
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    if show and owns_figure:
        plt.show()


def draw_embedding_gene_expression_streaming(
    conn: Any,
    *,
    embedding_table: str,
    x_col: str,
    y_col: str,
    x_label: str,
    y_label: str,
    gene_id: int,
    gene_name: str,
    expr_source: Any,
    where_sql: str | None = None,
    title: str | None = None,
    figsize: tuple[float, float] | None = DEFAULT_EMBEDDING_FIGSIZE,
    point_size: float | None = None,
    alpha: float = 0.8,
    cmap: str = "viridis",
    frameon: bool = True,
    hide_ticks: bool = True,
    ax: Any | None = None,
    show: bool = True,
    adjust_layout: bool = True,
    plot_batch_size: int = 200_000,
    range_sample_n: int = 100_000,
    range_quantiles: tuple[float, float] = (0.01, 0.99),
    save_path: PathLike[str] | str | None = None,
    dpi: int = 300,
) -> None:
    """Draw an embedding colored by one sparse gene expression vector."""

    e_table = quote_identifier(embedding_table)
    joins_obs = where_sql is not None and str(where_sql).strip() != ""
    from_sql = f"{e_table} e"
    if joins_obs:
        from_sql += " JOIN obs o ON e.atlas_cell_id = o.atlas_cell_id"
    where_clause = f"WHERE ({where_sql})" if joins_obs else ""
    expr_join_sql = _gene_expression_join_sql(expr_source, int(gene_id))

    total_points = conn.execute(f"""
        SELECT COUNT(*)
        FROM {from_sql}
        {where_clause}
    """).fetchone()[0]
    if int(total_points) == 0:
        raise ValueError("No cells are available for plotting")
    if point_size is None:
        point_size = default_point_size(int(total_points))

    range_df = conn.execute(f"""
        WITH sampled AS (
            SELECT atlas_cell_id
            FROM (
                SELECT e.atlas_cell_id
                FROM {from_sql}
                {where_clause}
            ) t
            USING SAMPLE {int(range_sample_n)} ROWS
        )
        SELECT COALESCE(xexpr.expr, 0.0) AS color_value
        FROM sampled
        LEFT JOIN ({expr_join_sql}) xexpr
          ON sampled.atlas_cell_id = xexpr.atlas_cell_id
    """).fetchdf()
    vmin, vmax = _estimate_continuous_range(
        range_df["color_value"].to_numpy(dtype=float),
        quantiles=range_quantiles,
    )

    owns_figure = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    else:
        fig = ax.figure
    ax.set_facecolor("white")

    scatter = _draw_gene_expression_embedding_batches(
        conn,
        ax=ax,
        from_sql=from_sql,
        where_clause=where_clause,
        x_col=x_col,
        y_col=y_col,
        expr_join_sql=expr_join_sql,
        point_size=point_size,
        alpha=alpha,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        plot_batch_size=plot_batch_size,
    )

    _style_streaming_embedding_axes(
        conn,
        ax=ax,
        from_sql=from_sql,
        where_clause=where_clause,
        x_col=x_col,
        y_col=y_col,
        color=None,
        x_label=x_label,
        y_label=y_label,
        title=str(gene_name) if title is None else title,
        frameon=frameon,
        hide_ticks=hide_ticks,
        legend_loc=None,
        unique_labels=[],
        label_to_color={},
        spread_label_positions=None,
        adjust_layout=False,
    )
    cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label(str(gene_name), fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    if adjust_layout:
        fig.tight_layout(pad=0.8)
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    if show and owns_figure:
        plt.show()


def _draw_plain_embedding_batches(
    conn: Any,
    *,
    ax: Any,
    from_sql: str,
    where_clause: str,
    x_col: str,
    y_col: str,
    point_size: float,
    alpha: float,
    plot_batch_size: int,
) -> None:
    """Draw uncolored embedding batches."""

    last_cell_id = -1
    while True:
        batch_df = conn.execute(f"""
            SELECT
                e.atlas_cell_id,
                e.{quote_identifier(x_col)} AS x,
                e.{quote_identifier(y_col)} AS y
            FROM {from_sql}
            {where_clause}
            {"AND" if where_clause else "WHERE"} e.atlas_cell_id > {int(last_cell_id)}
            ORDER BY e.atlas_cell_id
            LIMIT {int(plot_batch_size)}
        """).fetchdf()

        if len(batch_df) == 0:
            break

        last_cell_id = int(batch_df["atlas_cell_id"].iloc[-1])
        ax.scatter(
            batch_df["x"].to_numpy(),
            batch_df["y"].to_numpy(),
            s=point_size,
            c="#bdbdbd",
            alpha=alpha,
            linewidths=0,
            rasterized=True,
        )


def _draw_categorical_embedding_batches(
    conn: Any,
    *,
    ax: Any,
    from_sql: str,
    where_clause: str,
    x_col: str,
    y_col: str,
    color: str,
    unique_labels: list[str],
    label_to_color: dict[str, Any],
    point_size: float,
    alpha: float,
    plot_batch_size: int,
) -> None:
    """Draw categorical embedding batches."""

    last_cell_id = -1
    while True:
        batch_df = conn.execute(f"""
            SELECT
                e.atlas_cell_id,
                e.{quote_identifier(x_col)} AS x,
                e.{quote_identifier(y_col)} AS y,
                CAST(o.{quote_identifier(color)} AS TEXT) AS color_label
            FROM {from_sql}
            {where_clause}
            {"AND" if where_clause else "WHERE"} e.atlas_cell_id > {int(last_cell_id)}
            ORDER BY e.atlas_cell_id
            LIMIT {int(plot_batch_size)}
        """).fetchdf()

        if len(batch_df) == 0:
            break

        last_cell_id = int(batch_df["atlas_cell_id"].iloc[-1])
        batch_labels = batch_df["color_label"].astype(str)

        for lab in unique_labels:
            sub = batch_df[batch_labels == lab]
            if len(sub) == 0:
                continue

            ax.scatter(
                sub["x"].to_numpy(),
                sub["y"].to_numpy(),
                s=point_size,
                alpha=alpha,
                c=[label_to_color[lab]],
                linewidths=0,
                rasterized=True,
            )


def _draw_continuous_embedding_batches(
    conn: Any,
    *,
    ax: Any,
    from_sql: str,
    where_clause: str,
    x_col: str,
    y_col: str,
    value_sql: str,
    point_size: float,
    alpha: float,
    cmap: str,
    vmin: float,
    vmax: float,
    plot_batch_size: int,
) -> Any:
    """Draw embedding batches colored by a continuous SQL expression."""

    last_cell_id = -1
    scatter = None
    while True:
        batch_df = conn.execute(f"""
            SELECT
                e.atlas_cell_id,
                e.{quote_identifier(x_col)} AS x,
                e.{quote_identifier(y_col)} AS y,
                {value_sql} AS color_value
            FROM {from_sql}
            {where_clause}
            {"AND" if where_clause else "WHERE"} e.atlas_cell_id > {int(last_cell_id)}
            ORDER BY e.atlas_cell_id
            LIMIT {int(plot_batch_size)}
        """).fetchdf()

        if len(batch_df) == 0:
            break

        last_cell_id = int(batch_df["atlas_cell_id"].iloc[-1])
        scatter = ax.scatter(
            batch_df["x"].to_numpy(),
            batch_df["y"].to_numpy(),
            s=point_size,
            c=batch_df["color_value"].to_numpy(dtype=float),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            alpha=alpha,
            linewidths=0,
            rasterized=True,
        )

    if scatter is None:
        raise ValueError("No cells are available for plotting")
    return scatter


def _draw_gene_expression_embedding_batches(
    conn: Any,
    *,
    ax: Any,
    from_sql: str,
    where_clause: str,
    x_col: str,
    y_col: str,
    expr_join_sql: str,
    point_size: float,
    alpha: float,
    cmap: str,
    vmin: float,
    vmax: float,
    plot_batch_size: int,
) -> Any:
    """Draw embedding batches colored by one sparse expression vector."""

    last_cell_id = -1
    scatter = None
    while True:
        batch_df = conn.execute(f"""
            SELECT
                e.atlas_cell_id,
                e.{quote_identifier(x_col)} AS x,
                e.{quote_identifier(y_col)} AS y,
                COALESCE(xexpr.expr, 0.0) AS color_value
            FROM {from_sql}
            LEFT JOIN ({expr_join_sql}) xexpr
              ON e.atlas_cell_id = xexpr.atlas_cell_id
            {where_clause}
            {"AND" if where_clause else "WHERE"} e.atlas_cell_id > {int(last_cell_id)}
            ORDER BY e.atlas_cell_id
            LIMIT {int(plot_batch_size)}
        """).fetchdf()

        if len(batch_df) == 0:
            break

        last_cell_id = int(batch_df["atlas_cell_id"].iloc[-1])
        batch_df = batch_df.sort_values("color_value", ascending=True)
        scatter = ax.scatter(
            batch_df["x"].to_numpy(),
            batch_df["y"].to_numpy(),
            s=point_size,
            c=batch_df["color_value"].to_numpy(dtype=float),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            alpha=alpha,
            linewidths=0,
            rasterized=True,
        )

    if scatter is None:
        raise ValueError("No cells are available for plotting")
    return scatter


def _gene_expression_join_sql(expr_source: Any, gene_id: int) -> str:
    """Return a sparse expression subquery for one gene.

    Missing sparse entries are intentionally handled outside this subquery with
    ``COALESCE(..., 0.0)``. This is appropriate for count and log1p expression
    but not for scaled expression, whose implicit-zero value is gene-specific.
    """

    return f"""
        SELECT
            {expr_source.cell_sql} AS atlas_cell_id,
            {expr_source.value_sql} AS expr
        FROM {expr_source.from_sql}
        WHERE {expr_source.value_sql} IS NOT NULL
          AND {expr_source.gene_sql} = {int(gene_id)}
    """


def _estimate_continuous_range(
    values: Any,
    *,
    quantiles: tuple[float, float],
) -> tuple[float, float]:
    """Estimate a robust color range from sampled continuous values."""

    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0, 1.0

    q_low, q_high = quantiles
    vmin = float(np.nanquantile(arr, q_low))
    vmax = float(np.nanquantile(arr, q_high))

    if not np.isfinite(vmin) or not np.isfinite(vmax):
        vmin = float(np.nanmin(arr))
        vmax = float(np.nanmax(arr))

    if vmax <= vmin:
        pad = abs(vmin) * 0.05 if vmin != 0 else 1.0
        vmin -= pad
        vmax += pad

    return vmin, vmax


def is_numeric_duckdb_type(dtype: Any) -> bool:
    """Return whether a DuckDB type name should be plotted continuously."""

    dtype = str(dtype).upper()
    if "BOOL" in dtype:
        return False
    return any(
        key in dtype
        for key in ("INT", "FLOAT", "DOUBLE", "REAL", "DECIMAL", "NUMERIC")
    )


def is_numeric_obs_column(conn: Any, obs_col: str) -> bool:
    """Return whether an ``obs`` column has a numeric DuckDB type."""

    for row in conn.execute("PRAGMA table_info(obs)").fetchall():
        if row[1] == obs_col:
            return is_numeric_duckdb_type(row[2])
    return False


def _style_streaming_embedding_axes(
    conn: Any,
    *,
    ax: Any,
    from_sql: str,
    where_clause: str,
    x_col: str,
    y_col: str,
    color: str | None,
    x_label: str,
    y_label: str,
    title: str,
    frameon: bool,
    hide_ticks: bool,
    legend_loc: str | None,
    unique_labels: list[str],
    label_to_color: dict[str, Any],
    spread_label_positions: Callable[[Any], Any] | None,
    adjust_layout: bool,
) -> None:
    """Apply common styling and categorical legends to streamed embeddings."""

    ax.set_title(title, fontsize=14, weight="normal", pad=8)
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.grid(False)
    if hide_ticks:
        ax.set_xticks([])
        ax.set_yticks([])
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

    if color is None or legend_loc is None:
        if adjust_layout:
            plt.tight_layout(pad=0.8)
        return

    if legend_loc == "right_margin":
        _draw_right_margin_legend(ax, unique_labels, label_to_color)
        if adjust_layout:
            ax.figure.subplots_adjust(left=0.08, right=0.70, bottom=0.10, top=0.90)
        return

    if legend_loc == "on_data":
        center_df = conn.execute(f"""
            SELECT
                CAST(o.{quote_identifier(color)} AS TEXT) AS color_label,
                MEDIAN(e.{quote_identifier(x_col)}) AS x_center,
                MEDIAN(e.{quote_identifier(y_col)}) AS y_center
            FROM {from_sql}
            {where_clause}
            GROUP BY CAST(o.{quote_identifier(color)} AS TEXT)
        """).fetchdf()
        if spread_label_positions is not None:
            center_df = spread_label_positions(center_df)
            x_name = "label_x"
            y_name = "label_y"
        else:
            x_name = "x_center"
            y_name = "y_center"

        for _, row in center_df.iterrows():
            ax.text(
                row[x_name],
                row[y_name],
                str(row["color_label"]),
                fontsize=14,
                weight="bold",
                color="black",
                ha="center",
                va="center",
                zorder=10,
            )
        if adjust_layout:
            ax.figure.subplots_adjust(left=0.08, right=0.70, bottom=0.10, top=0.90)
        return

    raise ValueError("legend_loc only supports 'right_margin', 'on_data', or None")


def _draw_right_margin_legend(
    ax: Any,
    unique_labels: list[str],
    label_to_color: dict[str, Any],
) -> None:
    """Draw a categorical legend in the right margin."""

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
            markersize=6,
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
    leg.set_in_layout(True)
