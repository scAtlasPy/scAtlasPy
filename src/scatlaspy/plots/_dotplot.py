from ..data import Atlas
import re
from os import PathLike
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colorbar import ColorbarBase
from typing import Any


# =====================================================
# 统一离散分类颜色池
# -----------------------------------------------------
# 用于 obs 分类变量上色，例如：
# kmeans / cell_type / batch / organ 等
#
# 这些 palette 拼起来大约有 100 个离散颜色：
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
# 通用分类标签自然排序
# -----------------------------------------------------
# 解决：
# embryo_1, embryo_10, embryo_11, embryo_2
#
# 排成：
# embryo_1, embryo_2, embryo_3, ..., embryo_10
#
# 同样适用于：
# cluster_1 / cluster_10
# batch2 / batch10
# group_3_day_2 / group_3_day_12
# =====================================================
_MISSING_CATEGORY_LABELS = {"", "na", "nan", "none", "<na>", "null"}


def _natural_sort_key(value: Any):
    """
    分类标签自然排序 key。

    Examples
    --------
    embryo_1  < embryo_2  < embryo_10
    cluster_1 < cluster_2 < cluster_11
    """

    s = str(value).strip()

    # 缺失值标签放最后
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
    """
    对分类标签做默认自然排序。

    不额外暴露参数，所有离散分类变量默认使用这个排序。
    """

    labels = [str(x) for x in list(labels)]

    # 去重，同时保留原始列表中的唯一标签
    labels = list(dict.fromkeys(labels))

    return sorted(labels, key=_natural_sort_key)


def _build_discrete_color_map(labels: Any, palette: Any | None=None):
    """构建内部中间数据结构。

    该内部函数属于UMAP/表达可视化模块，用于支撑同一模块中的公共 API。

    读取 UMAP、obs、var 和表达矩阵，绘制 UMAP、feature plot、violin、dotplot 和 stacked violin。

    它通常不会作为用户入口直接调用；直接调用时需要保证输入对象、数据库连接和相关临时表已经由上游步骤准备好。

    Parameters
    ----------
    labels
        分类标签列表。

    palette
        离散分类变量使用的颜色方案。

    Returns
    -------
    result
        构建得到的内部对象，通常是 DataFrame、Arrow Table 或更新后的游标元组。

    Notes
    -----
    这是内部 helper；除非需要扩展 scAtlasPy 内部流程，一般不建议在用户代码中直接调用。
    """

    labels = list(labels)

    #  默认使用大颜色池
    if palette is None:
        palette_names = DEFAULT_DISCRETE_PALETTES

    # 兼容原来的 palette="tab20" 写法
    elif isinstance(palette, str):
        palette_names = (palette,)

    # 支持 palette=["tab20", "tab20b", ...]
    else:
        palette_names = tuple(palette)

    palette_colors = []

    for cmap_name in palette_names:
        cmap_obj = plt.get_cmap(cmap_name)

        # ListedColormap，比如 tab20 / Set3，通常有 .colors
        if hasattr(cmap_obj, "colors"):
            palette_colors.extend(list(cmap_obj.colors))

        # 兜底：如果是连续 colormap，就均匀取色
        else:
            n = getattr(cmap_obj, "N", 256)
            palette_colors.extend([
                cmap_obj(i / max(n - 1, 1))
                for i in range(n)
            ])

    # 如果类别数超过颜色池，继续用 hsv 补足
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


# dotplot 热图
def dotplot(
        atlas: Atlas,
        genes: str | list[str],
        groupby: str = "kmeans",
        use_data: str = "data_log1p",
        sample_cells_per_group: int | None = None,
        groups: list | None = None,
        where: str | None = None,
        order: list | None = None,
        expression_cutoff: float = 0.0,
        standard_scale: str | None = None,
        colorbar_vmin: float | None = 0.0,
        colorbar_vmax: float | None = 5.0,
        font_size: int = 14,
        save_path: PathLike[str] | str | None = None
):

    """绘制基因在不同分组中的 dotplot。

    该函数从 Atlas 数据库抽样读取指定基因在各分组中的表达，计算平均表达和表达比例，并绘制类似 Scanpy ``sc.pl.dotplot`` 的点图。

    Parameters
    ----------
    atlas
        Atlas 对象。通常需要已经连接到 DuckDB 数据库，并包含该函数读取或写入所需的 ``obs``、``var``、表达矩阵或结果表。
    genes
        需要展示的基因名称列表。
    groupby
        ``obs`` 中的分组列名，例如 ``"kmeans"``、``"leiden"`` 或 ``"cell_type"``。
    use_data
        读取的表达矩阵或结果表名称。常用值包括 ``"data"``、``"data_normalize"``、``"data_log1p"`` 和
        ``"data_scale"``。
    sample_cells_per_group
        每个分组抽样用于绘图的细胞数量。
    groups
        需要计算、展示或保留的分组列表。为 ``None`` 时使用全部分组。
    where
        额外 SQL 过滤条件。为 ``None`` 时不添加额外条件。
    order
        分组或基因展示顺序。为 ``None`` 时使用默认顺序。
    expression_cutoff
        判断基因是否表达的阈值。
    standard_scale
        是否按变量或分组对颜色值做标准化。
    colorbar_vmin
        颜色条下限。
    colorbar_vmax
        颜色条上限。
    font_size
        绘图字体大小。
    save_path
        图片保存路径。为 ``None`` 时只显示或返回图对象。

    Returns
    -------
    None

    Examples
    --------
    查看经典 marker genes 在 K-means cluster 中的表达::

        sap.pl.dotplot(
            atlas,
            genes=["MS4A1", "CD3D", "LYZ", "NKG7"],
            groupby="kmeans",
        )

    只展示指定 cluster，并保存图片::

        sap.pl.dotplot(
            atlas,
            genes=["MS4A1", "CD3D", "LYZ"],
            groupby="kmeans",
            groups=["0", "1", "2"],
            save_path=r"F:\\figures\\marker_dotplot.png",
        )"""

    conn = atlas.connection

    # 参数标准化
    if isinstance(genes, str):
        genes = [genes]
    genes = [str(g) for g in genes]
    if len(genes) == 0:
        raise ValueError("genes 不能为空")

    # 检查列
    obs_cols = [r[1] for r in conn.execute("PRAGMA table_info(obs)").fetchall()]
    var_cols = [r[1] for r in conn.execute("PRAGMA table_info(var)").fetchall()]
    x_cols = [r[1] for r in conn.execute("PRAGMA table_info(X_HyS_data)").fetchall()]

    if groupby not in obs_cols:
        raise ValueError(f"obs 中不存在列: {groupby}")
    if "atlas_cell_id" not in obs_cols:
        raise ValueError("obs 中不存在 atlas_cell_id")
    if "atlas_gene_id" not in var_cols or "atlas_gene_name" not in var_cols:
        raise ValueError("var 中不存在 atlas_gene_id / atlas_gene_name")
    if use_data not in x_cols:
        raise ValueError(f"X_HyS_data 中不存在字段: {use_data}")

    # gene_name -> gene_id
    gene_name_sql = ", ".join([f"'{g}'" for g in genes])

    gene_map_df = conn.execute(f"""
        SELECT atlas_gene_id, atlas_gene_name
        FROM var
        WHERE atlas_gene_name IN ({gene_name_sql})
    """).fetchdf()

    if len(gene_map_df) == 0:
        raise ValueError("var 中找不到这些基因")

    gene_map = dict(zip(gene_map_df["atlas_gene_name"], gene_map_df["atlas_gene_id"]))
    missing_genes = [g for g in genes if g not in gene_map]
    if missing_genes:
        raise ValueError(f"var 中找不到这些基因: {missing_genes}")

    gene_map_df["atlas_gene_name"] = pd.Categorical(
        gene_map_df["atlas_gene_name"],
        categories=genes,
        ordered=True
    )
    gene_map_df = gene_map_df.sort_values("atlas_gene_name").reset_index(drop=True)

    # 构造 where
    where_clauses = [f"{groupby} IS NOT NULL"]

    if where is not None and str(where).strip() != "":
        where_clauses.append(f"({where})")

    if groups is not None:
        groups_sql = ", ".join([f"'{str(g)}'" for g in groups])
        where_clauses.append(f"CAST({groupby} AS TEXT) IN ({groups_sql})")

    where_sql = " AND ".join(where_clauses)

    # group 列表
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
        raise ValueError("没有可用 group")

    # 数字型 group 按数值排序，避免 0,1,10,11,2
    def _group_sort_key(x: Any):
        """生成分组或标签的自然排序键。

        该内部函数属于UMAP/表达可视化模块，用于支撑同一模块中的公共 API。

        读取 UMAP、obs、var 和表达矩阵，绘制 UMAP、feature plot、violin、dotplot 和 stacked violin。

        它通常不会作为用户入口直接调用；直接调用时需要保证输入对象、数据库连接和相关临时表已经由上游步骤准备好。

        Parameters
        ----------
        x
            需要排序、格式化或转换的单个输入值。

        Returns
        -------
        sort_key
            可用于自然排序的键。

        Notes
        -----
        这是内部 helper；除非需要扩展 scAtlasPy 内部流程，一般不建议在用户代码中直接调用。
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
            raise ValueError("order 过滤后没有可用 group")
        group_df["order_idx"] = group_df["group_label"].map({g: i for i, g in enumerate(wanted)})
        group_df = group_df.sort_values("order_idx").drop(columns="order_idx").reset_index(drop=True)

    group_labels = group_df["group_label"].astype(str).tolist()

    # 每个 group 抽样细胞
    sampled_parts = []
    for g in group_labels:
        if sample_cells_per_group is None:
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
                LIMIT {int(sample_cells_per_group)}
            """
        sampled_parts.append(conn.execute(q).fetchdf())

    cells_df = pd.concat(sampled_parts, ignore_index=True)
    if len(cells_df) == 0:
        raise ValueError("抽样后没有细胞")

    # 注册临时表
    conn.register("_dotplot_cells_tmp", cells_df)
    conn.register("_dotplot_genes_tmp", gene_map_df[["atlas_gene_id", "atlas_gene_name"]])

    # 取表达长表（补隐式 0）
    expr_df = conn.execute(f"""
        SELECT
            c.group_label,
            g.atlas_gene_name AS gene,
            COALESCE(x.{use_data}, 0.0) AS expr
        FROM _dotplot_cells_tmp c
        CROSS JOIN _dotplot_genes_tmp g
        LEFT JOIN X_HyS_data x
            ON c.atlas_cell_id = x.atlas_cell_id
           AND g.atlas_gene_id = x.atlas_gene_id
    """).fetchdf()

    conn.unregister("_dotplot_cells_tmp")
    conn.unregister("_dotplot_genes_tmp")

    if len(expr_df) == 0:
        raise ValueError("expr_df 为空，无法作图")

    expr_df["gene"] = pd.Categorical(expr_df["gene"], categories=genes, ordered=True)
    expr_df["group_label"] = pd.Categorical(expr_df["group_label"], categories=group_labels, ordered=True)

    # 聚合统计
    stat_df = (
        expr_df
        .groupby(["group_label", "gene"], observed=True)
        .agg(
            mean_expr=("expr", "mean"),
            pct_expr=("expr", lambda x: (x > expression_cutoff).mean() * 100.0),
            n=("expr", "size")
        )
        .reset_index()
    )

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

    # 左边距：控制不要太空，也不要截断
    max_y_len = max(len(str(x)) for x in group_labels) if len(group_labels) > 0 else 10
    left_margin = min(0.24, max(0.09, 0.035 + 0.0085 * max_y_len))

    # 底部边距：gene 名显示完整
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

    # 主图
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
        norm=norm,  # 修改：主图和 colorbar 使用同一个颜色范围
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

    # 黑色外框
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

    # 右侧下部：Mean expression legend（拉开）
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

    # 边距
    fig.subplots_adjust(
        left=left_margin,
        right=0.98,
        top=0.95,
        bottom=bottom_margin
    )

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()
