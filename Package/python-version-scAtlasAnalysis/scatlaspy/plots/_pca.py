from ..data import Atlas
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime


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
    """
    根据 labels 构建离散颜色映射。

    Parameters
    ----------
    labels
        分类标签列表，例如 cats。

    palette
        可以是：
        - None：使用 DEFAULT_DISCRETE_PALETTES
        - str：例如 "tab20"
        - list/tuple：例如 ["tab20", "tab20b", "Set3"]

    Returns
    -------
    color_map
        dict: label -> color
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
        atlas,
        color: str | None = None,
        x_pc: int = 0,
        y_pc: int = 1,
        annotate_var_explained: bool = True,
        sample_n: int | None = 500000,
        use_expr_field: str = "data_log1p",
        figsize=(22, 8),
        point_size: float = 1.0,
        alpha: float = 0.7,
        cmap: str = "viridis",
        palette: str | list[str] | tuple[str, ...] | None = DEFAULT_DISCRETE_PALETTES,
        legend_loc: str = "right_margin",
        frameon: bool = True,
        return_df: bool = False,
):

    print("\n==== pca plot ====")
    start = datetime.now()
    conn = atlas.connection

    if conn is None:
        raise ValueError("atlas.connection 为空，请先连接数据库")

    # DuckDB 字段安全引用
    def _q(name: str) -> str:
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

            print(f"[COLOR] obs column: {color}")

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

                print(f"[COLOR] gene expression: {color}")

                if use_expr_field not in x_cols:
                    raise ValueError(
                        f"X_HyS_data 中不存在表达字段: {use_expr_field}"
                    )

                gene_id = int(gene_row[0])

                conn.register("_pca_cells_tmp", pca_df[["atlas_cell_id"]])

                expr_df = conn.execute(f"""
                    SELECT
                        c.atlas_cell_id,
                        COALESCE(x.{_q(use_expr_field)}, 0.0) AS color_value
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
            values = values.astype(str).astype("category")

            cats = list(values.cat.categories)

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

    print(f"Done in {(datetime.now() - start).total_seconds():.2f}s")

    if return_df:
        return plot_df


# 画 pca_variance_ratio
def pca_variance_ratio(
        atlas: Atlas,
        n_pcs: int = 30,
        *,
        log: bool = False,
        show: bool | None = None,
        save: bool | str | None = None,
        figsize=(16, 8),
        return_fig: bool = False,
):
    """
    从 Atlas 数据库中绘制 PCA 主成分方差解释率图。

    Parameters
    ----------
    atlas
        Atlas 对象，内部需要包含 DuckDB 数据库连接 atlas.connection。

    n_pcs
        要展示的主成分数量。例如 n_pcs=30 表示展示前 30 个 PC。

    log
        是否使用对数坐标显示 y 轴。
        如果 log=True，则 y 轴使用 log scale。

    show
        是否显示图像。
        如果 show=True，则直接显示图像。
        如果 show=False，则不显示图像。
        如果 show=None，则默认显示图像。

    save
        是否保存图像。
        如果 save=True，则保存为默认文件名 'pca_variance_ratio.png'。
        如果 save 是字符串，则根据字符串保存文件。

        示例：
            save=True              -> 保存为 pca_variance_ratio.png
            save=".pdf"            -> 保存为 pca_variance_ratio.pdf
            save="_test.png"       -> 保存为 pca_variance_ratio_test.png
            save="my_pca.png"      -> 保存为 my_pca.png

    figsize
        图像大小，默认是 (16, 8)。

    return_fig
        是否返回 matplotlib 的 fig 和 ax 对象。
        如果 return_fig=True，则返回 fig, ax，方便后续继续修改图像。
    """

    # 获取数据库连接
    conn = atlas.connection

    # 2. 从 uns_pca_stats 表中读取 PCA 方差解释率
    # uns_pca_stats 表中通常包含：
    # pc_index        ：主成分编号，通常从 0 开始
    # variance_ratio  ：每个主成分解释的方差比例
    df = conn.execute(f"""
        SELECT pc_index, variance_ratio
        FROM uns_pca_stats
        ORDER BY pc_index
        LIMIT {int(n_pcs)}
    """).fetchdf()


    if df.empty:
        raise ValueError("uns_pca_stats 为空，请先运行 PCA 后再绘图。")

    # 3. 准备绘图数据
    # 数据库中的 pc_index 一般从 0 开始：
    # 0, 1, 2, ...
    # 但 Scanpy 绘图习惯显示为：
    # PC1, PC2, PC3, ...
    # 所以这里 +1
    x = df["pc_index"].to_numpy() + 1

    # y 轴为每个主成分的方差解释率
    y = df["variance_ratio"].to_numpy()

    # 4. 创建图像对象
    # 使用 fig, ax 的方式，而不是直接 plt.figure()
    # 这样更方便后续控制 show、save 和 return_fig
    fig, ax = plt.subplots(figsize=figsize)

    # 绘制折线图
    ax.plot(x, y, marker="o")

    # 设置坐标轴标签和标题
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Explained variance ratio")
    ax.set_title("PCA variance ratio")

    # 如果 log=True，则使用对数坐标
    if log:
        ax.set_yscale("log")

    # 添加网格线，让图更容易阅读
    ax.grid(True, alpha=0.3)

    # 自动调整布局，避免标签或标题被裁剪
    fig.tight_layout()

    # 5. 保存图像
    # 这里模仿 Scanpy 的 save 风格：
    # save=True       -> 使用默认文件名
    # save=".pdf"     -> 默认文件名 + 后缀
    # save="_xxx.png" -> 默认文件名 + 自定义后缀
    # save="xxx.png"  -> 使用完整自定义文件名
    if save:
        default_name = "pca_variance_ratio"

        # save=True：保存为默认 png 文件
        if save is True:
            save_path = f"{default_name}.png"

        # save 是字符串：根据字符串形式判断保存路径
        elif isinstance(save, str):

            # 例如 save=".pdf" / ".png" / ".svg"
            # 保存为 pca_variance_ratio.pdf / pca_variance_ratio.png / pca_variance_ratio.svg
            if save.startswith("."):
                save_path = f"{default_name}{save}"

            # 例如 save="_test.png"
            # 保存为 pca_variance_ratio_test.png
            elif save.startswith("_"):
                save_path = f"{default_name}{save}"

            # 例如 save="my_pca.png"
            # 直接使用用户提供的完整文件名
            else:
                save_path = save

        else:
            raise ValueError("save 只支持 bool 或 str。")

        # 保存图像
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"[pca_variance_ratio] 图像已保存到: {save_path}")

    # 6. 显示或关闭图像
    # 如果 show=None，则默认显示图像
    if show is None:
        show = True

    # show=True：显示图像
    if show:
        plt.show()

    # show=False：不显示图像，并关闭 fig，避免占用内存
    else:
        plt.close(fig)

    # 7. 是否返回 fig, ax
    # 如果 return_fig=True，返回 fig 和 ax，方便外部继续修改图像
    if return_fig:
        return fig, ax


# 画累计解释方差
def pca_variance_ratio_cumsum(
        atlas: Atlas,
        n_pcs: int = 30,
        *,
        log: bool = False,
        show: bool | None = None,
        save: bool | str | None = None,
        figsize=(16, 8),
        return_fig: bool = False,
):
    """
    从 Atlas 数据库中绘制 PCA 累计方差解释率图。

    Parameters
    ----------
    atlas
        Atlas 对象，内部需要包含 DuckDB 数据库连接 atlas.connection。

    n_pcs
        要展示的主成分数量。例如 n_pcs=30 表示展示前 30 个 PC。

    log
        是否使用对数坐标显示 y 轴。
        一般累计方差解释率不建议使用 log=True，但这里保留接口，方便统一风格。

    show
        是否显示图像。
        如果 show=True，则直接显示图像。
        如果 show=False，则不显示图像。
        如果 show=None，则默认显示图像。

    save
        是否保存图像。
        如果 save=True，则保存为默认文件名 'pca_variance_ratio_cumsum.png'。
        如果 save 是字符串，则根据字符串保存文件。

        示例：
            save=True              -> 保存为 pca_variance_ratio_cumsum.png
            save=".pdf"            -> 保存为 pca_variance_ratio_cumsum.pdf
            save="_test.png"       -> 保存为 pca_variance_ratio_cumsum_test.png
            save="my_pca_cum.png"  -> 保存为 my_pca_cum.png

    figsize
        图像大小，默认是 (16, 8)。

    return_fig
        是否返回 matplotlib 的 fig 和 ax 对象。
        如果 return_fig=True，则返回 fig, ax，方便后续继续修改图像。
    """

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
        elif isinstance(save, str):

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
        print(f"[pca_variance_ratio_cumsum] 图像已保存到: {save_path}")

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

