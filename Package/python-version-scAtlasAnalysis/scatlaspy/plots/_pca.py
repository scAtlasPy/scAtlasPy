from ..data import Atlas
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from os import PathLike, fspath
import re
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

    # 去重，避免重复 category
    labels = list(dict.fromkeys(labels))

    return sorted(labels, key=_natural_sort_key)


def _build_discrete_color_map(labels: Any, palette: Any | None=None):
    """构建内部中间数据结构。

    该内部函数属于PCA 可视化模块，用于支撑同一模块中的公共 API。

    读取 PCA embedding 和方差解释率表，绘制 PCA 散点图和 variance ratio 图。

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


# 可视化 / 对外入口层 ;  用 PCA 的前两个主成分（PC1, PC2）做二维散点图
def pca(
        atlas: Atlas,
        color: str | None = None,
        x_pc: int = 0,
        y_pc: int = 1,
        annotate_var_explained: bool = True,
        sample_n: int | None = None,
        use_data: str = "data_log1p",
        figsize: tuple[float, float] | None=(6, 5), # (6, 5) (22, 8)
        point_size: float = 12,
        alpha: float = 0.8,
        cmap: str = "viridis",
        palette: str | list[str] | tuple[str, ...] | None = DEFAULT_DISCRETE_PALETTES,
        legend_loc: str | None = None,  # str = "right_margin",
        frameon: bool = True,
        return_df: bool = False,
):

    """绘制 PCA 细胞 embedding。

    该函数读取 ``obsm_X_pca`` 中的细胞 PCA 坐标，并按指定 ``obs`` 列或数值变量着色，绘制二维 PCA 散点图。它类似 Scanpy 的 ``sc.pl.pca``。

    Parameters
    ----------
    atlas
        Atlas 对象。通常需要已经连接到 DuckDB 数据库，并包含该函数读取或写入所需的 ``obs``、``var``、表达矩阵或结果表。
    color
        用于给散点上色的 ``obs`` 列名或数值列名。
    x_pc
        横轴使用的 PCA 主成分编号，从 0 开始。
    y_pc
        纵轴使用的 PCA 主成分编号，从 0 开始。
    annotate_var_explained
        是否在坐标轴上标注每个主成分解释的方差比例。
    sample_n
        绘图时最多抽样的细胞数量。为 ``None`` 时使用全部细胞。
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
    frameon
        是否显示坐标轴边框。
    return_df
        是否返回结果 DataFrame。

    Returns
    -------
    matplotlib.figure.Figure 或 None
        当 ``return_fig=True`` 或函数实现返回图对象时返回 Figure；否则通常直接显示图形。

    Examples
    --------
    绘制 PC1 和 PC2，并按 K-means cluster 着色::

        sap.tl.pca(atlas)
        sap.pl.pca(atlas, color="kmeans")

    绘制 PC2 和 PC3，并返回用于检查的 DataFrame::

        df = sap.pl.pca(
            atlas,
            color="pct_counts_mt",
            x_pc=1,
            y_pc=2,
            sample_n=200000,
            return_df=True,
        )"""

    start = datetime.now()
    conn = atlas.connection

    if conn is None:
        raise ValueError("atlas.connection 为空，请先连接数据库")

    # DuckDB 字段安全引用
    def _q(name: str) -> str:
        """为 SQL 标识符添加安全引用。

        该内部函数属于PCA 可视化模块，用于支撑同一模块中的公共 API。

        读取 PCA embedding 和方差解释率表，绘制 PCA 散点图和 variance ratio 图。

        它通常不会作为用户入口直接调用；直接调用时需要保证输入对象、数据库连接和相关临时表已经由上游步骤准备好。

        Parameters
        ----------
        name
            对象名称、列名或 SQL 标识符，具体含义由调用位置决定。

        Returns
        -------
        quoted_name
            加双引号后的 SQL 标识符。

        Notes
        -----
        这是内部 helper；除非需要扩展 scAtlasPy 内部流程，一般不建议在用户代码中直接调用。
        """
        return '"' + name.replace('"', '""') + '"'

    pcx = f"pc{x_pc}"
    pcy = f"pc{y_pc}"

    # 检查 PCA 表和 PC 列是否存在
    pca_table_exists = conn.execute("""
        SELECT COUNT(*)
        FROM information_schema.tables
        WHERE table_name = 'obsm_X_pca'
    """).fetchone()[0]

    if pca_table_exists == 0:
        raise ValueError("数据库中不存在 obsm_X_pca，请先运行 PCA")

    obsm_cols = [
        r[1]
        for r in conn.execute("PRAGMA table_info(obsm_X_pca)").fetchall()
    ]

    if pcx not in obsm_cols or pcy not in obsm_cols:
        raise ValueError(
            f"obsm_X_pca 中不存在列: {pcx} 或 {pcy}\n"
            f"请确认 PCA 是否已经计算，或者 x_pc / y_pc 是否超出范围。"
        )

    # 读取 explained variance ratio
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

    # SQL 先抽样 PCA 坐标
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
        raise ValueError("PCA 抽样结果为空，无法绘图")

    plot_df = pca_df.copy()

    color_kind = None

    if color is not None:

        # 获取 obs / var / X_HyS_data 字段
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

        # color 是 obs 表列名
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

        # color 是 var.atlas_gene_name 基因名
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
                        f"X_HyS_data 中不存在表达字段: {use_data}"
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

            # 如果是 var 普通列，明确报错
            elif color in var_cols:
                raise ValueError(
                    f"color='{color}' 是 var 表中的列。\n"
                    f"但是 PCA 图的每个点是 cell，var 列是 gene-level 信息，"
                    f"不能直接用于 cell PCA 上色。\n"
                    f"如果想按基因表达上色，请传入 var.atlas_gene_name 中的基因名。"
                )

            else:
                raise ValueError(
                    f"找不到 color='{color}'。\n"
                    f"它既不是 obs 表列名，也不是 var.atlas_gene_name 中的基因名。"
                )

    # 绘图
    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    ax.set_facecolor("white")

    x = plot_df[pcx].to_numpy()
    y = plot_df[pcy].to_numpy()

    # 情况 A：不指定 color，灰色散点
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

    # 情况 B：基因表达，连续色条
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

    # 情况 C：obs 列
    #       数值型 → 连续色条
    #       分类/字符串/布尔 → 离散 legend
    elif color_kind == "obs":
        values = plot_df["color_value"]

        # bool 不按连续变量处理，而是按分类变量处理
        is_bool = pd.api.types.is_bool_dtype(values)
        is_numeric = pd.api.types.is_numeric_dtype(values)

        # C1. obs 数值列：连续色条
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

        # C2. obs 分类列：离散颜色 + legend
        else:
            values = values.astype("object").where(values.notna(), "NA")
            values_str = values.astype(str)

            # 默认使用自然排序
            # embryo_1, embryo_2, ..., embryo_10
            cats = _sort_categories_natural(pd.unique(values_str))

            # 显式指定 category 顺序
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

            # 按类别分组绘图，Scanpy 风格 legend
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
                # 根据类别数量自动调整 legend 列数和字体
                n_cat = len(cats)
                max_label_len = max([len(str(c)) for c in cats], default=0)

                if n_cat <= 14:
                    legend_ncol = 1  # 列数
                    legend_fontsize = 20  # 字体大小
                elif n_cat <= 30:
                    legend_ncol = 2
                    legend_fontsize = 20
                elif n_cat <= 60:
                    legend_ncol = 4
                    legend_fontsize = 20
                else:
                    legend_ncol = 5
                    legend_fontsize = 12

                if max_label_len >= 18: # 图例中所有类别名称里，最长那个名称的字符长度
                    legend_fontsize = min(legend_fontsize, 15)
                if max_label_len >= 28:
                    legend_fontsize = min(legend_fontsize, 15)

                leg = ax.legend(
                    title=None,
                    bbox_to_anchor=(1.03, 0.5),
                    loc="center left",
                    frameon=False,
                    markerscale=8.0, # 图例圆点
                    fontsize=legend_fontsize,
                    borderaxespad=0.0,
                    ncol=legend_ncol,
                    columnspacing=1.0,
                    handletextpad=0.35,
                    labelspacing=0.35,
                    handlelength=0.8,
                )

                # 强制放大 legend 里的 scatter 圆点，更稳定
                for h in leg.legend_handles:
                    if hasattr(h, "set_sizes"):
                        h.set_sizes([100])

                # 不让 tight_layout / layout 系统为了 legend 压缩主图
                leg.set_in_layout(False)

            elif legend_loc == "on_data":
                # 简单 on_data：把类别名放到该类 PCA 坐标中位数附近
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
                    "legend_loc 只支持 'right_margin'、'on_data' 或 None"
                )

    # Scanpy 风格美化
    ax.set_xlabel(x_label, fontsize=14)
    ax.set_ylabel(y_label, fontsize=14)

    if color is None:
        ax.set_title("PCA", fontsize=14, pad=8)
    else:
        ax.set_title(str(color), fontsize=14, pad=8)

    # 更接近 Scanpy，默认不画网格
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

    # 控制 PCA 主图框的高宽比例，避免变成图2那种瘦高图
    ax.set_box_aspect(0.75)

    # 不要让 tight_layout 把主图挤窄
    if legend_loc == "right_margin":
        # 手动给右侧 legend 留空间，主图不会被压成竖条
        fig.subplots_adjust(
            left=0.06,
            right=0.38,
            bottom=0.16,
            top=0.88,
        )
    else:
        plt.tight_layout(pad=0.8)

    plt.show()

    if return_df:
        return plot_df


# 画 pca_variance_ratio
def pca_variance_ratio(
        atlas: Atlas,
        n_pcs: int = 30,
        *,
        log: bool = False,
        show: bool | None = None,
        save: bool | PathLike[str] | str | None = None,
        figsize: tuple[float, float] | None = (7, 6),
        return_fig: bool = False,
):
    """绘制 PCA 方差解释比例，Scanpy-like 风格。

    该函数读取 ``uns_pca_stats`` 中每个主成分的 explained variance ratio，
    并以 Scanpy ``sc.pl.pca_variance_ratio`` 类似的方式绘图：
    横轴为 ranking，纵轴为 variance ratio，每个 PC 以竖排文本标注。

    Parameters
    ----------
    atlas
        Atlas 对象。

    n_pcs
        展示的 PCA 主成分数量。

    log
        是否使用 y 轴对数坐标。

    show
        是否显示图像。

    save
        图片保存路径。

    figsize
        图形大小。

    return_fig
        是否返回 ``fig, ax``。
    """

    conn = atlas.connection

    if conn is None:
        raise ValueError("atlas.connection 为空，请先连接数据库")

    # 1. 读取 PCA 方差解释率
    df = conn.execute(f"""
        SELECT pc_index, variance_ratio
        FROM uns_pca_stats
        ORDER BY pc_index
        LIMIT {int(n_pcs)}
    """).fetchdf()

    if df.empty:
        raise ValueError("uns_pca_stats 为空，请先运行 PCA 后再绘图。")

    # 2. 准备数据
    # Scanpy 风格：x 轴是 ranking，从 0 开始
    x = np.arange(df.shape[0])
    y = df["variance_ratio"].to_numpy(dtype=float)

    pc_labels = [
        f"PC{int(pc_index) + 1}"
        for pc_index in df["pc_index"].to_numpy()
    ]

    # 3. 创建图像
    fig, ax = plt.subplots(figsize=figsize)

    # 4. 用文字标注每个 PC，而不是画折线
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

    # 5. 坐标轴样式，尽量接近 Scanpy
    ax.set_title("variance ratio", fontsize=18, pad=8)
    ax.set_xlabel("ranking", fontsize=18)
    ax.set_ylabel("")

    # x 轴显示 ranking
    ax.set_xlim(-0.8, len(x) - 0.2)

    # 尽量让右侧有 20 这个刻度，接近 Scanpy 示例
    if len(x) <= 20:
        ax.set_xticks(np.arange(0, len(x) + 1, 5))
    else:
        ax.set_xticks(np.arange(0, len(x), 5))

    # y 轴留一点上方空间，避免 PC1 标签被裁掉
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

    # 网格和边框：Scanpy 图是有网格和完整边框的
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

    # 6. 保存图像
    if save:
        default_name = "pca_variance_ratio"

        if save is True:
            save_path = f"{default_name}.png"

        elif isinstance(save, (str, PathLike)):
            save = fspath(save)

            if save.startswith("."):
                save_path = f"{default_name}{save}"

            elif save.startswith("_"):
                save_path = f"{default_name}{save}"

            else:
                save_path = save

        else:
            raise ValueError("save 只支持 bool 或 str。")

        fig.savefig(save_path, bbox_inches="tight", dpi=300)

    # 7. 显示或关闭
    if show is None:
        show = True

    if show:
        plt.show()
    else:
        plt.close(fig)

    if return_fig:
        return fig, ax


# 画累计解释方差
def pca_variance_ratio_cumsum(
        atlas: Atlas,
        n_pcs: int = 30,
        *,
        log: bool = False,
        show: bool | None = None,
        save: bool | PathLike[str] | str | None = None,
        figsize: tuple[float, float] | None=(16, 8),
        return_fig: bool = False,
):
    """绘制 PCA 累积方差解释比例。

    该函数读取 PCA 方差解释比例并计算累积和，帮助判断 PCA 维度选择是否足以覆盖主要变化。

    Parameters
    ----------
    atlas
        Atlas 对象。通常需要已经连接到 DuckDB 数据库，并包含该函数读取或写入所需的 ``obs``、``var``、表达矩阵或结果表。
    n_pcs
        展示的 PCA 主成分数量。
    log
        是否使用对数坐标或对数显示。
    show
        是否立即显示图形。为 ``None`` 时遵循 Matplotlib 当前行为。
    save
        图片保存路径。为 ``None`` 时不保存。
    figsize
        图形大小。为 ``None`` 时使用函数默认尺寸。
    return_fig
        是否返回 Matplotlib Figure 对象。

    Returns
    -------
    matplotlib.figure.Figure 或 None
        当 ``return_fig=True`` 或函数实现返回图对象时返回 Figure；否则通常直接显示图形。

    Examples
    --------
    查看前 50 个主成分的累积解释比例::

        sap.pl.pca_variance_ratio_cumsum(atlas, n_pcs=50)

    返回 Figure 以便进一步修改::

        fig = sap.pl.pca_variance_ratio_cumsum(
            atlas,
            n_pcs=80,
            return_fig=True,
        )"""

    # 1. 获取数据库连接
    conn = atlas.connection

    # 2. 从 uns_pca_stats 表中读取 PCA 方差解释率
    df = conn.execute(f"""
        SELECT pc_index, variance_ratio
        FROM uns_pca_stats
        ORDER BY pc_index
        LIMIT {int(n_pcs)}
    """).fetchdf()

    # 如果表中没有数据，说明还没有运行 PCA，或者 PCA 结果没有写入数据库
    if df.empty:
        raise ValueError("uns_pca_stats 为空，请先运行 PCA 后再绘图。")

    # 3. 准备绘图数据
    # 数据库中的 pc_index 通常从 0 开始；
    # Scanpy 风格绘图中通常显示为 PC1, PC2, PC3...
    x = df["pc_index"].to_numpy() + 1

    # 每个 PC 的方差解释率
    y = df["variance_ratio"].to_numpy()

    # 累计方差解释率
    y_cum = np.cumsum(y)

    # 4. 创建图像对象
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(x, y_cum, marker="o")

    ax.set_xlabel("Principal component")
    ax.set_ylabel("Cumulative explained variance ratio")
    ax.set_title("Cumulative PCA variance ratio")

    # 一般累计解释率不需要 log，但保留这个参数以统一接口
    if log:
        ax.set_yscale("log")

    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    # 5. 保存图像
    if save:
        default_name = "pca_variance_ratio_cumsum"

        # save=True：保存为默认 png 文件
        if save is True:
            save_path = f"{default_name}.png"

        # save 是字符串：根据字符串形式判断保存路径
        elif isinstance(save, (str, PathLike)):
            save = fspath(save)

            # save=".pdf" / ".png" / ".svg"
            if save.startswith("."):
                save_path = f"{default_name}{save}"

            # save="_test.png"
            elif save.startswith("_"):
                save_path = f"{default_name}{save}"

            # save="my_pca_cum.png"
            else:
                save_path = save

        else:
            raise ValueError("save 只支持 bool 或 str。")

        fig.savefig(save_path, bbox_inches="tight", dpi=300)

    # 6. 显示或关闭图像
    if show is None:
        show = True

    if show:
        plt.show()
    else:
        plt.close(fig)

    # 7. 是否返回 fig, ax
    if return_fig:
        return fig, ax


# 画 PCA loadings
def pca_loadings(
        atlas: Atlas,
        components: int | tuple[int, ...] | list[int] = (1, 2),
        n_genes: int = 10,
        include_lowest: bool = True,
        figsize: tuple[float, float] | None = (14, 8),
        show: bool | None = None,
        save: bool | PathLike[str] | str | None = None,
        return_fig: bool = False,
):
    """绘制 PCA loadings 图，类似 scanpy.pl.pca_loadings。

    该函数从 ``varm_PCs`` 表中读取每个基因在指定 PC 上的 loading，
    并展示 loading 最大的基因；当 ``include_lowest=True`` 时，
    同时展示 loading 最小的基因。

    Parameters
    ----------
    atlas
        Atlas 对象。需要已经运行过 PCA，并包含 ``varm_PCs`` 表。

    components
        需要展示的主成分编号。注意这里和 Scanpy 一样，从 1 开始。
        例如 ``components=(1, 2)`` 表示 PC1 和 PC2。

    n_genes
        每侧展示的基因数量。
        当 ``include_lowest=True`` 时，每个 PC 会展示 top n_genes 和 bottom n_genes。

    include_lowest
        是否同时展示 loading 最小的基因。

    figsize
        图像大小。为 ``None`` 时自动根据 components 数量设置。

    show
        是否显示图像。默认为 True。

    save
        图片保存路径。

    return_fig
        是否返回 ``fig, axes``。
    """

    conn = atlas.connection

    if conn is None:
        raise ValueError("atlas.connection 为空，请先连接数据库")

    def _q(name: str) -> str:
        return '"' + name.replace('"', '""') + '"'

    # -------------------------------------------------
    # 1. 处理 components
    # -------------------------------------------------
    if isinstance(components, int):
        components = (components,)
    else:
        components = tuple(components)

    if len(components) == 0:
        raise ValueError("components 不能为空")

    for comp in components:
        if comp < 1:
            raise ValueError("components 使用 Scanpy 风格编号，必须从 1 开始，例如 PC1 写 1")

    # -------------------------------------------------
    # 2. 检查 varm_PCs 和 var 表
    # -------------------------------------------------
    varm_exists = conn.execute("""
        SELECT COUNT(*)
        FROM information_schema.tables
        WHERE table_name = 'varm_PCs'
    """).fetchone()[0]

    if varm_exists == 0:
        raise ValueError("数据库中不存在 varm_PCs 表，请先运行 PCA")

    var_exists = conn.execute("""
        SELECT COUNT(*)
        FROM information_schema.tables
        WHERE table_name = 'var'
    """).fetchone()[0]

    if var_exists == 0:
        raise ValueError("数据库中不存在 var 表")

    varm_cols = [
        r[1]
        for r in conn.execute("PRAGMA table_info('varm_PCs')").fetchall()
    ]

    var_cols = [
        r[1]
        for r in conn.execute("PRAGMA table_info('var')").fetchall()
    ]

    if "atlas_gene_id" not in varm_cols:
        raise ValueError("varm_PCs 表中不存在 atlas_gene_id 字段")

    if "atlas_gene_id" not in var_cols:
        raise ValueError("var 表中不存在 atlas_gene_id 字段")

    gene_name_col = "atlas_gene_name" if "atlas_gene_name" in var_cols else "atlas_gene_id"

    # -------------------------------------------------
    # 3. 自动匹配 PC 列名
    # -------------------------------------------------
    def _find_pc_col(comp: int) -> str:
        # comp 是 1-based，pc_index 是 0-based
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
            f"varm_PCs 中找不到 PC{comp} 对应的 loading 列。\n"
            f"尝试过这些列名: {candidates}\n"
            f"当前 varm_PCs 字段为: {varm_cols}"
        )

    # -------------------------------------------------
    # 4. 创建图像
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
    # 5. 每个 PC 单独画一个 panel
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
            raise ValueError(f"PC{comp} 的 loading 数据为空")

        df["gene_name"] = df["gene_name"].astype(str)
        df["loading"] = df["loading"].astype(float)

        # loading 最大的基因
        top_df = (
            df.sort_values("loading", ascending=False)
              .head(int(n_genes))
              .copy()
        )

        if include_lowest:
            # loading 最小的基因
            low_df = (
                df.sort_values("loading", ascending=True)
                  .head(int(n_genes))
                  .copy()
            )

            # 为了显示上更像 Scanpy，负向基因从接近 0 到最负排列
            low_df = low_df.sort_values("loading", ascending=False).copy()

            plot_df = pd.concat([top_df, low_df], ignore_index=True)

            x_top = np.arange(len(top_df))
            x_low = np.arange(len(top_df) + 1, len(top_df) + 1 + len(low_df))

            # 正向基因
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

            # 中间省略号
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

            # 负向基因
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
        # 6. Scanpy-like 样式
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
    # 7. 保存图像
    # -------------------------------------------------
    if save:
        default_name = "pca_loadings"

        if save is True:
            save_path = f"{default_name}.png"

        elif isinstance(save, (str, PathLike)):
            save = fspath(save)

            if save.startswith("."):
                save_path = f"{default_name}{save}"
            elif save.startswith("_"):
                save_path = f"{default_name}{save}"
            else:
                save_path = save

        else:
            raise ValueError("save 只支持 bool 或 str。")

        fig.savefig(save_path, bbox_inches="tight", dpi=300)

    # -------------------------------------------------
    # 8. 显示或关闭
    # -------------------------------------------------
    if show is None:
        show = True

    if show:
        plt.show()
    else:
        plt.close(fig)

    if return_fig:
        return fig, axes