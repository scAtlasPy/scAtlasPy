from ..data import Atlas
from matplotlib.lines import Line2D
import os
import math
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colorbar import ColorbarBase
from scipy.stats import gaussian_kde


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


def _build_discrete_color_map(labels, palette=None):
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
        sample_n: int | None = 50000,
        where: str | None = None,
        use_expr_field: str = "data_log1p",
        ncols: int = 3,
        figsize=(22, 8),
        point_size: float = 1.0,
        alpha: float =  0.7 ,
        cmap: str = "viridis",
        palette: str | list[str] | tuple[str, ...] | None = DEFAULT_DISCRETE_PALETTES,
        legend_loc: str = "right_margin",
        frameon: bool = True,
        save_path: str | None = None,
        plot_batch_size: int = 200000,
        return_df: bool = False,
):

    """绘制 UMAP embedding。

    该函数从 ``obsm_X_umap`` 中读取 UMAP 坐标，并根据 ``color`` 参数自动判断是 obs 分类/连续变量还是 gene
    expression feature。

    当 ``color`` 为 obs 列时绘制分类或连续 UMAP；当 ``color`` 为基因名时，从 ``X_HyS_data``
    中读取表达值并绘制 feature plot；混合输入会分别生成结果。

    Parameters
    ----------
    atlas
        Atlas 对象。通常要求已经连接数据库，并包含该函数所需的 ``obs``、``var``、``X_HyS_data`` 或
        embedding 结果表。

    color
        用于着色的 ``obs`` 列名、基因名或它们的列表。

    sample_n
        抽样细胞数量；为 ``None`` 时通常使用全部可用细胞。

    where
        可选 SQL 过滤条件，用于限制参与计算或绘图的细胞。

    use_expr_field
        绘制 gene feature 或表达分布时读取的 ``X_HyS_data`` 表达字段。

    ncols
        多面板绘图时每行的子图数量。

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

    legend_loc
        图例位置。

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
    绘图前通常需要先运行对应的 ``sap.tl`` 或 ``sap.pp`` 计算步骤，确保结果表和统计列已经存在。

    Examples
    --------
    调用该函数：::

        sap.pl.umap(...)
    """
    print("\n==== sap.pl.umap ====")

    conn = atlas.connection

    # 参数标准化
    if isinstance(color, str):
        color_list = [color]
    else:
        color_list = list(color)

    if len(color_list) == 0:
        raise ValueError("color 不能为空")

    print(f"[UMAP] color = {color_list}")
    print(f"[UMAP] sample_n = {sample_n}")

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
            use_expr_field=use_expr_field,
            ncols=ncols,
            figsize=figsize,
            point_size=point_size,
            alpha=alpha
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
            use_expr_field=use_expr_field,
            ncols=ncols,
            figsize=figsize,
            point_size=point_size,
            alpha=alpha
        )

    return result



# umap() ─ 如果 color 是 obs 列 → plot_umap_obs()
def _plot_umap_obs(
        atlas,
        color: str = "kmeans",
        sample_n: int | None = 50000,
        groups: list | None = None,
        where: str | None = None,
        legend_loc: str = "right_margin",
        title: str | None = None,
        figsize=(22, 8),
        point_size: float = 1.0,
        alpha: float = 0.7,
        cmap: str = "viridis",
        palette: str | list[str] | tuple[str, ...] | None = DEFAULT_DISCRETE_PALETTES,
        frameon: bool = True,
        save_path: str | None = None,
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
    print(f"\n==== _plot_umap_obs (color={color}) ====")
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

    # 调色板;尽量按数字排序；如果不是纯数字，再按字符串排序
    def _sort_label(x):
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
        except:
            return (1, str(x))

    unique_labels = sorted(
        plot_df["color_label"].astype(str).unique().tolist(),
        key=_sort_label
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

    print("[UMAP] Done")

    if return_df:
        return plot_df

    return None


# umap() ─ 如果 color 是 obs 列 → plot_umap_obs()
#          sample_n == None → _draw_umap_obs_streaming()
def _draw_umap_obs_streaming(
        atlas,
        color: str,
        where_sql: str,
        legend_loc: str = "right_margin",
        title: str | None = None,
        figsize=(22, 8),
        point_size: float = 1.0,
        alpha: float = 0.7,
        palette: str | list[str] | tuple[str, ...] | None = DEFAULT_DISCRETE_PALETTES,
        frameon: bool = True,
        save_path: str | None = None,
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
    print("\n==== plot_umap_obs_streaming_full ====")

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

    # 按数字顺序排序；如果不是纯数字，再按字符串排序
    def _sort_label(x):
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
        except:
            return (1, str(x))

    # 不要直接使用 SQL 的字符串排序结果
    unique_labels = sorted(
        label_df["color_label"].astype(str).tolist(),
        key=_sort_label
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

        print(f"[UMAP streaming] drawn cells = {total_drawn:,}")

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
                markersize=10,  # ✅【修改】圆点大小
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

    print(f"[UMAP streaming] Done, total drawn = {total_drawn:,}")

    return None



# umap() ─ 如果 color 是 gene 名 → plot_umap_features()
def _plot_umap_features(
        atlas,
        genes,
        sample_n: int | None = 50000,
        where: str | None = None,
        use_expr_field: str = "data_scale",
        ncols: int = 3,
        figsize=None,
        point_size: float = 8,
        alpha: float = 0.9
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

    use_expr_field
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
    print("\n==== _plot_umap_features ====")
    start = datetime.now()
    conn = atlas.connection

    if isinstance(genes, str):
        genes = [genes]

    if len(genes) == 0:
        raise ValueError("genes 不能为空")

    print(f"[UMAP features] genes = {genes}")
    print(f"[UMAP features] sample_n = {sample_n}")
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

    if use_expr_field not in x_cols:
        raise ValueError(f"X_HyS_data 中不存在字段: {use_expr_field}")

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

    print(f"[UMAP features] plotted cells = {len(umap_df):,}")

    # 查询 gene_id
    gene_name_sql = ", ".join([f"'{str(g)}'" for g in genes])

    if use_expr_field == "data_scale":
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

    if use_expr_field == "data_scale":
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

        if use_expr_field == "data_scale":
            gene_id, zero_fill = gene_map[gene]

            expr_df = conn.execute(f"""
                SELECT
                    c.atlas_cell_id,
                    COALESCE(x.{use_expr_field}, {zero_fill}) AS expr
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
                    COALESCE(x.{use_expr_field}, 0.0) AS expr
                FROM _umap_cells_tmp c
                LEFT JOIN X_HyS_data x
                  ON c.atlas_cell_id = x.atlas_cell_id
                 AND x.atlas_gene_id = {gene_id}
            """).fetchdf()

        df = umap_df.merge(expr_df, on="atlas_cell_id", how="left")

        if use_expr_field == "data_scale":
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
            cmap="viridis",
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

    print(f"[UMAP features] Done in {(datetime.now() - start).total_seconds():.2f}s")

    return plot_data


# 普通 violin
def violin(
        atlas,
        genes,
        groupby: str = "kmeans",
        use_expr_field: str = "data_log1p",
        sample_n_per_group: int | None = 2000,
        groups: list | None = None,
        where: str | None = None,
        order: list | None = None,
        save_path: str | None = None
):

    """按分组绘制基因表达小提琴图。

    该函数从 ``obs`` 中读取分组信息，并从 ``X_HyS_data`` 中读取指定基因的表达值，按 group 绘制表达分布。

    功能上类似 Scanpy 的 ``sc.pl.violin``，但数据直接来自 Atlas 数据库，并支持每组抽样以控制绘图规模。

    Parameters
    ----------
    atlas
        Atlas 对象。通常要求已经连接数据库，并包含该函数所需的 ``obs``、``var``、``X_HyS_data`` 或
        embedding 结果表。

    genes
        需要绘制或分析的基因名称列表。

    groupby
        ``obs`` 中用于分组的列名。

    use_expr_field
        绘制 gene feature 或表达分布时读取的 ``X_HyS_data`` 表达字段。

    sample_n_per_group
        每个分组最多抽样的细胞数量。

    groups
        需要分析或绘制的分组；为 ``None`` 时使用全部分组。

    where
        可选 SQL 过滤条件，用于限制参与计算或绘图的细胞。

    order
        分组显示顺序；为 ``None`` 时按自然排序生成。

    save_path
        图像或结果保存路径。

    Returns
    -------
    result
        函数返回结果。具体类型取决于参数设置和内部执行路径。

    Notes
    -----
    绘图前通常需要先运行对应的 ``sap.tl`` 或 ``sap.pp`` 计算步骤，确保结果表和统计列已经存在。

    Examples
    --------
    调用该函数：::

        sap.pl.violin(...)
    """
    print(f"\n==== violin (groupby={groupby}) ====")
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
    if use_expr_field not in x_cols:
        raise ValueError(f"X_HyS_data 中不存在字段: {use_expr_field}")

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

    # 准备 group 抽样细胞
    where_clauses = [f"{groupby} IS NOT NULL"]
    if where is not None and str(where).strip() != "":
        where_clauses.append(f"({where})")

    if groups is not None:
        group_sql = ", ".join([f"'{g}'" for g in groups])
        where_clauses.append(f"CAST({groupby} AS TEXT) IN ({group_sql})")

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
    def _group_sort_key(x):
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
    print(f"-> groups = {group_labels}")

    # 每个 group 单独抽样，再 union
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
        raise ValueError("抽样后没有细胞")

    # 注册临时表
    conn.register("_violin_cells_tmp", cells_df)
    conn.register("_violin_genes_tmp", gene_map_df[["atlas_gene_id", "atlas_gene_name"]])

    # 取表达长表（补隐式 0）
    plot_df = conn.execute(f"""
        SELECT
            c.group_label,
            g.atlas_gene_name AS gene,
            COALESCE(x.{use_expr_field}, 0.0) AS expr
        FROM _violin_cells_tmp c
        CROSS JOIN _violin_genes_tmp g
        LEFT JOIN X_HyS_data x
            ON c.atlas_cell_id = x.atlas_cell_id
           AND g.atlas_gene_id = x.atlas_gene_id
    """).fetchdf()

    conn.unregister("_violin_cells_tmp")
    conn.unregister("_violin_genes_tmp")

    if len(plot_df) == 0:
        raise ValueError("plot_df 为空，无法作图")

    plot_df["gene"] = pd.Categorical(plot_df["gene"], categories=genes, ordered=True)
    plot_df["group_label"] = pd.Categorical(plot_df["group_label"], categories=group_labels, ordered=True)
    plot_df = plot_df.sort_values(["gene", "group_label"]).reset_index(drop=True)

    # 作图
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

    return plot_df



# dotplot 热图
def dotplot(
        atlas,
        genes,
        groupby: str = "kmeans",
        use_expr_field: str = "data_log1p",
        sample_n_per_group: int | None = 2000,
        groups: list | None = None,
        where: str | None = None,
        order: list | None = None,
        expression_cutoff: float = 0.0,
        standard_scale: str | None = None,
        colorbar_vmin: float | None = 0.0,
        colorbar_vmax: float | None = 5.0,
        font_size: int = 14,
        save_path: str | None = None
):

    """按分组绘制基因表达 dotplot。

    该函数计算每个 group 中每个基因的平均表达量和表达细胞比例，并用颜色表示表达强度、点大小表示表达比例。

    功能上类似 Scanpy 的 ``sc.pl.dotplot``，适合比较 marker 基因在多个细胞类型或 cluster 中的表达模式。

    Parameters
    ----------
    atlas
        Atlas 对象。通常要求已经连接数据库，并包含该函数所需的 ``obs``、``var``、``X_HyS_data`` 或
        embedding 结果表。

    genes
        需要绘制或分析的基因名称列表。

    groupby
        ``obs`` 中用于分组的列名。

    use_expr_field
        绘制 gene feature 或表达分布时读取的 ``X_HyS_data`` 表达字段。

    sample_n_per_group
        每个分组最多抽样的细胞数量。

    groups
        需要分析或绘制的分组；为 ``None`` 时使用全部分组。

    where
        可选 SQL 过滤条件，用于限制参与计算或绘图的细胞。

    order
        分组显示顺序；为 ``None`` 时按自然排序生成。

    expression_cutoff
        dotplot 中判断细胞是否表达某基因的阈值。

    standard_scale
        dotplot 中是否按 gene 或 group 标准化平均表达。

    colorbar_vmin
        颜色条下限。

    colorbar_vmax
        颜色条上限。

    font_size
        绘图中文字大小。

    save_path
        图像或结果保存路径。

    Returns
    -------
    result
        函数返回结果。具体类型取决于参数设置和内部执行路径。

    Notes
    -----
    绘图前通常需要先运行对应的 ``sap.tl`` 或 ``sap.pp`` 计算步骤，确保结果表和统计列已经存在。

    Examples
    --------
    调用该函数：::

        sap.pl.dotplot(...)
    """
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
    if use_expr_field not in x_cols:
        raise ValueError(f"X_HyS_data 中不存在字段: {use_expr_field}")

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
    def _group_sort_key(x):
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
        raise ValueError("抽样后没有细胞")

    # 注册临时表
    conn.register("_dotplot_cells_tmp", cells_df)
    conn.register("_dotplot_genes_tmp", gene_map_df[["atlas_gene_id", "atlas_gene_name"]])

    # 取表达长表（补隐式 0）
    expr_df = conn.execute(f"""
        SELECT
            c.group_label,
            g.atlas_gene_name AS gene,
            COALESCE(x.{use_expr_field}, 0.0) AS expr
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

    ax.scatter(
        x,
        y,
        s=sizes,
        c=colors,
        cmap="Reds",
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

    if colorbar_vmin is None:
        vmin = float(np.nanmin(colors))
    else:
        vmin = float(colorbar_vmin)

    if colorbar_vmax is None:
        vmax = float(np.nanmax(colors))
    else:
        vmax = float(colorbar_vmax)

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

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

    return stat_df




# 堆叠提琴图 violin gene 在 x 轴，group 在 y 轴，每个 group × gene 的格子里画一个小提琴形状，并用 median expression 控制颜色深浅
def stacked_violin(
        atlas,
        genes,
        groupby: str = "cell_type_auto",
        use_expr_field: str = "data_log1p",
        sample_n_per_group: int | None = 2000,
        groups: list | None = None,
        where: str | None = None,
        order: list | None = None,
        color_vmin: float | None = 0.0,
        color_vmax: float | None = 5.0,
        font_size: int = 14,
        save_path: str | None = None
):

    """按分组绘制堆叠小提琴图。

    该函数为多个基因和多个分组构建紧凑的 stacked violin 图，用于展示 marker 基因在不同 cluster 或细胞类型中的表达分布。

    表达值从 Atlas 的 ``X_HyS_data`` 表中按需读取，并可通过每组抽样控制绘图数据量。

    Parameters
    ----------
    atlas
        Atlas 对象。通常要求已经连接数据库，并包含该函数所需的 ``obs``、``var``、``X_HyS_data`` 或
        embedding 结果表。

    genes
        需要绘制或分析的基因名称列表。

    groupby
        ``obs`` 中用于分组的列名。

    use_expr_field
        绘制 gene feature 或表达分布时读取的 ``X_HyS_data`` 表达字段。

    sample_n_per_group
        每个分组最多抽样的细胞数量。

    groups
        需要分析或绘制的分组；为 ``None`` 时使用全部分组。

    where
        可选 SQL 过滤条件，用于限制参与计算或绘图的细胞。

    order
        分组显示顺序；为 ``None`` 时按自然排序生成。

    color_vmin
        颜色映射下限。

    color_vmax
        颜色映射上限。

    font_size
        绘图中文字大小。

    save_path
        图像或结果保存路径。

    Returns
    -------
    result
        函数返回结果。具体类型取决于参数设置和内部执行路径。

    Notes
    -----
    绘图前通常需要先运行对应的 ``sap.tl`` 或 ``sap.pp`` 计算步骤，确保结果表和统计列已经存在。

    Examples
    --------
    调用该函数：::

        sap.pl.stacked_violin(...)
    """
    conn = atlas.connection

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
    if use_expr_field not in x_cols:
        raise ValueError(f"X_HyS_data 中不存在字段: {use_expr_field}")

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
    def _group_sort_key(x):
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
        raise ValueError("抽样后没有细胞")

    # 注册临时表
    conn.register("_sv_cells_tmp", cells_df)
    conn.register("_sv_genes_tmp", gene_map_df[["atlas_gene_id", "atlas_gene_name"]])

    # 取表达长表（补隐式 0）
    expr_df = conn.execute(f"""
        SELECT
            c.group_label,
            g.atlas_gene_name AS gene,
            COALESCE(x.{use_expr_field}, 0.0) AS expr
        FROM _sv_cells_tmp c
        CROSS JOIN _sv_genes_tmp g
        LEFT JOIN X_HyS_data x
            ON c.atlas_cell_id = x.atlas_cell_id
           AND g.atlas_gene_id = x.atlas_gene_id
    """).fetchdf()

    conn.unregister("_sv_cells_tmp")
    conn.unregister("_sv_genes_tmp")

    if len(expr_df) == 0:
        raise ValueError("expr_df 为空，无法作图")

    expr_df["gene"] = pd.Categorical(expr_df["gene"], categories=genes, ordered=True)
    expr_df["group_label"] = pd.Categorical(expr_df["group_label"], categories=group_labels, ordered=True)

    # 中位数统计（用于着色）
    median_df = (
        expr_df
        .groupby(["group_label", "gene"], observed=True)["expr"]
        .median()
        .reset_index(name="median_expr")
    )

    # 布局
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

    # 颜色映射
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

        # 全 0 / 常数分布：画一个细竖线
        if vmax - vmin < 1e-12:
            ax.plot(
                [x0, x0],
                [y0 - cell_half_height * 0.35, y0 + cell_half_height * 0.35],
                color="#b0b0b0",
                linewidth=1.0,
                zorder=3
            )
            continue

        # KDE 估计
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

        # 中位数横线
        med_y = y0 + ((med_val - vmin) / (vmax - vmin) - 0.5) * (2 * cell_half_height)
        ax.plot(
            [x0 - cell_half_width * 0.45, x0 + cell_half_width * 0.45],
            [med_y, med_y],
            color="#808080",
            linewidth=0.9,
            zorder=3
        )

    # 主图美化
    ax.set_xticks(np.arange(n_genes))
    ax.set_xticklabels(genes, rotation=90, fontsize=font_size)

    # y tick 也按 scanpy 风格从上到下显示
    y_tick_positions = [group_to_y[g] for g in group_labels]
    ax.set_yticks(y_tick_positions)
    ax.set_yticklabels(group_labels, fontsize=font_size)

    ax.set_xlim(-0.55, n_genes - 0.45)
    ax.set_ylim(-0.5, n_groups - 0.5)

    # 行分隔线
    for j in range(n_groups):
        ax.axhline(j + 0.5, color="#d0d0d0", linewidth=1.0, zorder=0)

    ax.grid(False)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
        spine.set_color("black")

    # 右侧色条说明
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

    return expr_df, median_df

