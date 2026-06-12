from ..data import Atlas
from matplotlib.lines import Line2D
import re
import math
from os import PathLike
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
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


# UMAP 可视化入口
def umap(
        atlas: Atlas,
        color: str | list[str] = "kmeans",
        *,
        sample_n: int | None = 50000,
        where: str | None = None,

        # gene feature 参数
        use_data: str = "data_log1p",

        # 图形参数
        figsize: tuple[float, float] | None = (22, 8),
        point_size: float = 1.0,
        alpha: float = 0.7,
        cmap: str = "viridis",
        palette: str | list[str] | tuple[str, ...] | None = DEFAULT_DISCRETE_PALETTES,

        # legend / layout
        legend_loc: str | None = "right_margin",
        ncols: int = 3,
        frameon: bool = True,

        # 大数据 / 输出参数
        plot_batch_size: int = 200000,
        save_path: PathLike[str] | str | None = None,
        return_df: bool = False,
):

    """绘制 UMAP embedding。

    该函数读取 ``obsm_X_umap`` 或指定 UMAP 结果表，并按 ``obs`` 列、表达值或其他变量着色，绘制二维 UMAP 散点图。它类似 Scanpy 的 ``sc.pl.umap``。

    Parameters
    ----------
    atlas
        Atlas 对象。通常需要已经连接到 DuckDB 数据库，并包含该函数读取或写入所需的 ``obs``、``var``、表达矩阵或结果表。
    color
        用于给散点上色的 ``obs`` 列名或数值列名。
    sample_n
        绘图时最多抽样的细胞数量。为 ``None`` 时使用全部细胞。
    where
        额外 SQL 过滤条件。为 ``None`` 时不添加额外条件。
    use_data
        读取的表达矩阵或结果表名称。常用值包括 ``"data"``、``"data_normalize"``、``"data_log1p"`` 和
        ``"data_scale"``。
    figsize
        图形大小。为 ``None`` 时使用函数默认尺寸。
    point_size
        散点大小。
    alpha
        图形元素透明度。
    cmap
        连续变量使用的 Matplotlib colormap 名称。
    palette
        离散变量使用的颜色列表或调色板。
    legend_loc
        图例位置。
    ncols
        参数。用于控制该函数的输入、输出或计算细节；默认值适合常规 Atlas 工作流。
    frameon
        是否显示坐标轴边框。
    plot_batch_size
        参数。用于控制该函数的输入、输出或计算细节；默认值适合常规 Atlas 工作流。
    save_path
        图片保存路径。为 ``None`` 时只显示或返回图对象。
    return_df
        是否返回结果 DataFrame。

    Returns
    -------
    matplotlib.figure.Figure 或 None
        当 ``return_fig=True`` 或函数实现返回图对象时返回 Figure；否则通常直接显示图形。

    Examples
    --------
    按 K-means cluster 绘制 UMAP::

        sap.tl.umap(atlas)
        sap.pl.umap(atlas, color="kmeans")

    按 marker gene 表达绘制，并保存图片::

        sap.pl.umap(
            atlas,
            color="MS4A1",
            use_data="data_log1p",
            save=r"F:\\figures\\umap_MS4A1.png",
        )

    使用自定义 UMAP 表和较小点大小::

        sap.pl.umap(
            atlas,
            use_table="obsm_X_umap_n45_d02",
            color="cell_type_auto",
            point_size=0.5,
        )"""

    conn = atlas.connection

    # 参数标准化
    if isinstance(color, str):
        color_list = [color]
    else:
        color_list = list(color)

    if len(color_list) == 0:
        raise ValueError("color 不能为空")

    if where is not None and str(where).strip() != "":
        print(f"[UMAP] where = {where}")

    # 检查 obsm_X_umap 是否存在
    tables = conn.execute("""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_name = 'obsm_X_umap'
    """).fetchdf()

    if len(tables) == 0:
        raise ValueError(
            "数据库中不存在 obsm_X_umap。\n"
            "请先运行 sap.tl.umap(atlas)"
        )

    # 获取 obs 列和 gene 名
    obs_cols = [r[1] for r in conn.execute("PRAGMA table_info(obs)").fetchall()]

    gene_df = conn.execute("""
        SELECT atlas_gene_name
        FROM var
    """).fetchdf()

    gene_set = set(gene_df["atlas_gene_name"].astype(str).tolist())

    # 判断 color 类型
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
                f"color='{c}' 既不是 obs 列，也不是 gene 名。\n"
                f"请确认 obs 或 var 中存在该字段。"
            )

    # 单个 obs 分类图
    if len(color_list) == 1 and len(obs_colors) == 1:
        return _plot_umap_obs(
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
            return_df=return_df,
        )

    # 纯 gene feature 图
    if len(obs_colors) == 0 and len(gene_colors) > 0:
        return _plot_umap_features(
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
        )

    # 混合模式
    result = {}

    for obs_col in obs_colors:
        result[obs_col] = _plot_umap_obs(
            atlas=atlas,
            color=obs_col,
            sample_n=sample_n,
            where=where,
            legend_loc=legend_loc,
            figsize=figsize,
            point_size=point_size,
            alpha=alpha,
            cmap=cmap,
            palette=palette,
            frameon=frameon,
            save_path=None,
            plot_batch_size=plot_batch_size,
            return_df=return_df,
        )

    if len(gene_colors) > 0:
        result["genes"] = _plot_umap_features(
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
        )

    return result



# umap() ─ 如果 color 是 obs 列 → plot_umap_obs()
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

    """绘制中间分析结果。

    该内部函数属于UMAP/表达可视化模块，用于支撑同一模块中的公共 API。

    读取 UMAP、obs、var 和表达矩阵，绘制 UMAP、feature plot、violin、dotplot 和 stacked violin。

    它通常不会作为用户入口直接调用；直接调用时需要保证输入对象、数据库连接和相关临时表已经由上游步骤准备好。

    当前实现中会访问或生成的关键表包括：``obs``、``obsm_X_umap``。

    Parameters
    ----------
    atlas
        Atlas 对象。通常要求已经连接数据库，并包含该函数所需的 ``obs``、``var``、``X_HyS_data`` 或
        embedding 结果表。

    color
        用于着色的 ``obs`` 列名、基因名或它们的列表。

    sample_n
        抽样细胞数量；为 ``None`` 时通常使用全部可用细胞。

    groups
        需要分析或绘制的分组；为 ``None`` 时使用全部分组。

    where
        可选 SQL 过滤条件，用于限制参与计算或绘图的细胞。

    legend_loc
        图例位置。

    title
        图标题。

    figsize
        matplotlib 图像大小。

    point_size
        散点大小。

    alpha
        绘图透明度。

    cmap
        连续变量使用的 colormap。

    palette
        离散分类变量使用的颜色方案。

    frameon
        是否显示图框。

    save_path
        图像或结果保存路径。

    plot_batch_size
        绘图时分批读取数据库的细胞数量。

    return_df
        是否返回用于绘图或分析的 DataFrame。

    Returns
    -------
    result
        函数返回结果。具体类型取决于参数设置和内部执行路径。

    Notes
    -----
    这是内部 helper；除非需要扩展 scAtlasPy 内部流程，一般不建议在用户代码中直接调用。
    """

    conn = atlas.connection

    # 检查表和列
    tables = conn.execute("""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_name IN ('obsm_X_umap', 'obs')
    """).fetchdf()["table_name"].tolist()

    if "obsm_X_umap" not in tables:
        raise ValueError("数据库中不存在 obsm_X_umap，请先运行 sap.tl.umap(atlas)")
    if "obs" not in tables:
        raise ValueError("数据库中不存在 obs")

    obs_cols = [r[1] for r in conn.execute("PRAGMA table_info(obs)").fetchall()]
    umap_cols = [r[1] for r in conn.execute("PRAGMA table_info(obsm_X_umap)").fetchall()]

    if color not in obs_cols:
        raise ValueError(f"obs 中不存在列: {color}")
    if "atlas_cell_id" not in obs_cols:
        raise ValueError("obs 中不存在 atlas_cell_id")
    if "atlas_cell_id" not in umap_cols or "umap1" not in umap_cols or "umap2" not in umap_cols:
        raise ValueError("obsm_X_umap 需要包含 atlas_cell_id / umap1 / umap2")

    # 构造过滤条件
    where_clauses = [f"o.{color} IS NOT NULL"]

    if where is not None and str(where).strip() != "":
        where_clauses.append(f"({where})")

    if groups is not None:
        groups_sql = ", ".join([f"'{str(g)}'" for g in groups])
        where_clauses.append(f"CAST(o.{color} AS TEXT) IN ({groups_sql})")

    where_sql = " AND ".join(where_clauses)

    # 取数据（可抽样）
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

    # sample_n=None 时，走全量 streaming 绘图，避免一次性 fetchdf 爆内存
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

    # sample_n 不是 None 时，仍然走原来的抽样绘图
    plot_df = conn.execute(query).fetchdf()

    if len(plot_df) == 0:
        raise ValueError("筛选后没有可绘制的细胞")

    # 默认使用自然排序
    # embryo_1, embryo_2, ..., embryo_10
    unique_labels = _sort_categories_natural(
        plot_df["color_label"].astype(str).unique().tolist()
    )

    # 使用统一大离散颜色池
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

    # 标题
    if title is None:
        title = color
    ax.set_title(title, fontsize=18, weight="normal", pad=10)

    ax.set_xlabel("UMAP1", fontsize=16)
    ax.set_ylabel("UMAP2", fontsize=16)

    # 图例 / on-data 标签
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

        # 强制放大 legend 圆点，和 PCA 一致
        for h in leg.legend_handles:
            if hasattr(h, "set_sizes"):
                h.set_sizes([100])

        leg.set_in_layout(False)

    elif legend_loc == "on_data":
        for lab in unique_labels:
            sub = plot_df[plot_df["color_label"] == lab]
            x_center = sub["umap1"].mean()
            y_center = sub["umap2"].mean()
            ax.text(
                x_center,
                y_center,
                str(lab),
                fontsize=12,
                weight="bold",
                ha="center",
                va="center"
            )
    else:
        raise ValueError("legend_loc 只能是 'right_margin' 或 'on_data'")

    # 样式
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

    if legend_loc == "right_margin":
        fig.subplots_adjust(
            left=0.06,
            right=0.38,
            bottom=0.16,
            top=0.88,
        )
    else:
        plt.tight_layout(pad=0.8)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

    if return_df:
        return plot_df

    return None


# umap() ─ 如果 color 是 obs 列 → plot_umap_obs()
#          sample_n == None → _draw_umap_obs_streaming()
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

    """执行 ``_draw_umap_obs_streaming`` 的核心功能。

    该内部函数属于UMAP/表达可视化模块，用于支撑同一模块中的公共 API。

    读取 UMAP、obs、var 和表达矩阵，绘制 UMAP、feature plot、violin、dotplot 和 stacked violin。

    它通常不会作为用户入口直接调用；直接调用时需要保证输入对象、数据库连接和相关临时表已经由上游步骤准备好。

    当前实现中会访问或生成的关键表包括：``obs``、``obsm_X_umap``。

    Parameters
    ----------
    atlas
        Atlas 对象。通常要求已经连接数据库，并包含该函数所需的 ``obs``、``var``、``X_HyS_data`` 或
        embedding 结果表。

    color
        用于着色的 ``obs`` 列名、基因名或它们的列表。

    where_sql
        已经拼接好的 SQL WHERE 条件。

    legend_loc
        图例位置。

    title
        图标题。

    figsize
        matplotlib 图像大小。

    point_size
        散点大小。

    alpha
        绘图透明度。

    palette
        离散分类变量使用的颜色方案。

    frameon
        是否显示图框。

    save_path
        图像或结果保存路径。

    plot_batch_size
        绘图时分批读取数据库的细胞数量。

    Returns
    -------
    result
        函数返回结果。具体类型取决于参数设置和内部执行路径。

    Notes
    -----
    这是内部 helper；除非需要扩展 scAtlasPy 内部流程，一般不建议在用户代码中直接调用。
    """

    conn = atlas.connection

    # 先取全部类别，用于固定颜色
    label_df = conn.execute(f"""
        SELECT DISTINCT CAST(o.{color} AS TEXT) AS color_label
        FROM obsm_X_umap u
        JOIN obs o
          ON u.atlas_cell_id = o.atlas_cell_id
        WHERE {where_sql}
        ORDER BY color_label
    """).fetchdf()

    if len(label_df) == 0:
        raise ValueError("筛选后没有可绘制的细胞")

    # 默认使用自然排序
    # embryo_1, embryo_2, ..., embryo_10
    # 不要直接使用 SQL 的字符串排序结果
    unique_labels = _sort_categories_natural(
        label_df["color_label"].astype(str).tolist()
    )

    # 使用统一大离散颜色池
    label_to_color = _build_discrete_color_map(
        labels=unique_labels,
        palette=palette,
    )

    # 建图
    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    ax.set_facecolor("white")

    # 分批读取 + 分批画图
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

    # 标题
    if title is None:
        title = color

    ax.set_title(title, fontsize=14, weight="normal", pad=8)
    ax.set_xlabel("UMAP1", fontsize=12)
    ax.set_ylabel("UMAP2", fontsize=12)

    # 图例
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
                AVG(u.umap1) AS x_center,
                AVG(u.umap2) AS y_center
            FROM obsm_X_umap u
            JOIN obs o
              ON u.atlas_cell_id = o.atlas_cell_id
            WHERE {where_sql}
            GROUP BY CAST(o.{color} AS TEXT)
        """).fetchdf()

        for _, row in center_df.iterrows():
            ax.text(
                row["x_center"],
                row["y_center"],
                str(row["color_label"]),
                fontsize=12,
                weight="bold",
                ha="center",
                va="center"
            )

    else:
        raise ValueError("legend_loc 只能是 'right_margin' 或 'on_data'")

    ax.grid(False)

    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_aspect("equal", adjustable="box")

    ax.margins(0.02)

    if frameon:
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.0)
            spine.set_color("black")
    else:
        for spine in ax.spines.values():
            spine.set_visible(False)

    if legend_loc == "right_margin":
        fig.subplots_adjust(
            left=0.06,
            right=0.42,
            bottom=0.10,
            top=0.90,
        )
    else:
        plt.tight_layout(pad=0.8)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

    return None



# umap() ─ 如果 color 是 gene 名 → plot_umap_features()
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
):

    """绘制中间分析结果。

    该内部函数属于UMAP/表达可视化模块，用于支撑同一模块中的公共 API。

    读取 UMAP、obs、var 和表达矩阵，绘制 UMAP、feature plot、violin、dotplot 和 stacked violin。

    它通常不会作为用户入口直接调用；直接调用时需要保证输入对象、数据库连接和相关临时表已经由上游步骤准备好。

    当前实现中会访问或生成的关键表包括：``X_HyS_data``、``obs``、``obsm_X_umap``、``var``。

    Parameters
    ----------
    atlas
        Atlas 对象。通常要求已经连接数据库，并包含该函数所需的 ``obs``、``var``、``X_HyS_data`` 或
        embedding 结果表。

    genes
        需要绘制或分析的基因名称列表。

    sample_n
        抽样细胞数量；为 ``None`` 时通常使用全部可用细胞。

    where
        可选 SQL 过滤条件，用于限制参与计算或绘图的细胞。

    use_data
        绘制 gene feature 或表达分布时读取的 ``X_HyS_data`` 表达字段。

    ncols
        多面板绘图时每行的子图数量。

    figsize
        matplotlib 图像大小。

    point_size
        散点大小。

    alpha
        绘图透明度。

    Returns
    -------
    result
        函数返回结果。具体类型取决于参数设置和内部执行路径。

    Notes
    -----
    这是内部 helper；除非需要扩展 scAtlasPy 内部流程，一般不建议在用户代码中直接调用。
    """

    start = datetime.now()
    conn = atlas.connection

    if isinstance(genes, str):
        genes = [genes]

    if len(genes) == 0:
        raise ValueError("genes 不能为空")

    if where is not None and str(where).strip() != "":
        print(f"[UMAP features] where = {where}")

    # 检查表和列
    tables = conn.execute("""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_name IN ('obsm_X_umap', 'obs', 'var', 'X_HyS_data')
    """).fetchdf()["table_name"].tolist()

    if "obsm_X_umap" not in tables:
        raise ValueError("数据库中不存在 obsm_X_umap，请先运行 sap.tl.umap(atlas)")
    if "obs" not in tables:
        raise ValueError("数据库中不存在 obs")
    if "var" not in tables:
        raise ValueError("数据库中不存在 var")
    if "X_HyS_data" not in tables:
        raise ValueError("数据库中不存在 X_HyS_data")

    umap_cols = [r[1] for r in conn.execute("PRAGMA table_info(obsm_X_umap)").fetchall()]
    obs_cols = [r[1] for r in conn.execute("PRAGMA table_info(obs)").fetchall()]
    x_cols = [r[1] for r in conn.execute("PRAGMA table_info(X_HyS_data)").fetchall()]
    var_cols = [r[1] for r in conn.execute("PRAGMA table_info(var)").fetchall()]

    if "atlas_cell_id" not in umap_cols or "umap1" not in umap_cols or "umap2" not in umap_cols:
        raise ValueError("obsm_X_umap 需要包含 atlas_cell_id / umap1 / umap2")

    if "atlas_cell_id" not in obs_cols:
        raise ValueError("obs 中不存在 atlas_cell_id")

    if "atlas_cell_id" not in x_cols or "atlas_gene_id" not in x_cols:
        raise ValueError("X_HyS_data 需要包含 atlas_cell_id / atlas_gene_id")

    if use_data not in x_cols:
        raise ValueError(f"X_HyS_data 中不存在字段: {use_data}")

    if "atlas_gene_id" not in var_cols or "atlas_gene_name" not in var_cols:
        raise ValueError("var 需要包含 atlas_gene_id / atlas_gene_name")

    # SQL 先过滤，再抽样 UMAP 细胞
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
        raise ValueError("筛选 / 抽样后没有可绘制的细胞")

    # 查询 gene_id
    gene_name_sql = ", ".join([f"'{str(g)}'" for g in genes])

    if use_data == "data_scale":
        if "zero_scale_transform" not in var_cols:
            raise ValueError(
                "var 中不存在 zero_scale_transform。\n"
                "请先运行 scale 流程写入 zero_scale_transform。"
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
        raise ValueError("var 中找不到这些基因")

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
        raise ValueError(f"var 中找不到这些基因: {missing_genes}")

    # 注册抽样细胞临时表
    conn.register("_umap_cells_tmp", umap_df[["atlas_cell_id"]])

    # 逐个 gene 取表达
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

        # 高表达点后画，避免被低表达点盖住
        df = df.sort_values("expr", ascending=True).reset_index(drop=True)

        plot_data[gene] = df

    conn.unregister("_umap_cells_tmp")

    # 自动布局
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

    # 作图
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

    # 多余子图隐藏
    for ax in axes_flat[len(genes):]:
        ax.set_visible(False)

    plt.tight_layout(pad=1.0)
    plt.show()

    return plot_data

