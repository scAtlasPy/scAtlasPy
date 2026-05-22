import numpy as np
from ..data import Atlas
import matplotlib.pyplot as plt
from datetime import datetime


''' 可视化 / 对外入口层 '''
# 用 PCA 的前两个主成分（PC1, PC2）做二维散点图
def pca(
        atlas,
        color: str | None = None,          # ✅ 修改：支持 obs 列名 或 gene name
        x_pc: int = 0,
        y_pc: int = 1,
        annotate_var_explained: bool = True,
        sample_n: int | None = 500000,
        use_expr_field: str = "data_log1p",
        figsize=(14, 4.2),                 # ✅ 修改：从 (6, 6) 改宽，更接近 Scanpy 横向布局
        point_size: float = 1.0,            # ✅ 修改：从 8 改小，避免点太大
        alpha: float = 0.7,                 # ✅ 修改：从 0.9 改低一点，更接近 Scanpy 密度感
        cmap: str = "viridis",             # ✅ 新增：连续变量 colormap
        palette: str = "tab20",            # ✅ 新增：分类变量 palette
        legend_loc: str = "right_margin",  # ✅ 新增：分类 legend 位置
        frameon: bool = True,              # ✅ 新增：Scanpy 风格 frame
        return_df: bool = False,           # ✅ 新增：是否返回绘图数据
):
    """
    数据库版 Scanpy 风格 PCA 图。

    color 支持两种：
    1. obs 表列名：
        - celltype
        - BATCH
        - kmeans
        - leiden
        - 其他 obs 中的连续/分类字段

    2. var.atlas_gene_name 中的基因名：
        - CST3
        - NKG7
        - MS4A1
        - 其他基因名

    不支持：
    - var 表普通列名直接给 cell PCA 上色
      因为 PCA 图的点是 cell，var 列是 gene-level 信息。
    """

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from datetime import datetime

    print("\n==== pca plot ====")
    start = datetime.now()
    conn = atlas.connection

    if conn is None:
        raise ValueError("atlas.connection 为空，请先连接数据库")

    # =====================================================
    # ✅ 修改 1：DuckDB 字段安全引用
    # =====================================================
    def _q(name: str) -> str:
        return '"' + name.replace('"', '""') + '"'

    pcx = f"pc{x_pc}"
    pcy = f"pc{y_pc}"

    # =====================================================
    # 0️⃣ 检查 PCA 表和 PC 列是否存在
    # =====================================================
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

    # =====================================================
    # 1️⃣ 读取 explained variance ratio
    # =====================================================
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

    # =====================================================
    # 2️⃣ SQL 先抽样 PCA 坐标
    # =====================================================
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

    # =====================================================
    # ✅ 修改 2：Scanpy 风格解析 color
    #     优先级：
    #       1. obs 表列名
    #       2. var.atlas_gene_name 基因名
    # =====================================================
    color_kind = None

    if color is not None:

        # -------------------------------------------------
        # 2.1 获取 obs / var / X_CSRO_data 字段
        # -------------------------------------------------
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
            for r in conn.execute("PRAGMA table_info(X_CSRO_data)").fetchall()
        ]

        # =================================================
        # ✅ 修改 3：color 是 obs 表列名
        # =================================================
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

        # =================================================
        # ✅ 修改 4：color 是 var.atlas_gene_name 基因名
        # =================================================
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
                        f"X_CSRO_data 中不存在表达字段: {use_expr_field}"
                    )

                gene_id = int(gene_row[0])

                conn.register("_pca_cells_tmp", pca_df[["atlas_cell_id"]])

                expr_df = conn.execute(f"""
                    SELECT
                        c.atlas_cell_id,
                        COALESCE(x.{_q(use_expr_field)}, 0.0) AS color_value
                    FROM _pca_cells_tmp AS c
                    LEFT JOIN X_CSRO_data AS x
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

            # =================================================
            # ✅ 修改 5：如果是 var 普通列，明确报错
            # =================================================
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

    # =====================================================
    # 3️⃣ 绘图
    # =====================================================
    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    ax.set_facecolor("white")

    x = plot_df[pcx].to_numpy()
    y = plot_df[pcy].to_numpy()

    # =====================================================
    # 情况 A：不指定 color，灰色散点
    # =====================================================
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

    # =====================================================
    # 情况 B：基因表达，连续色条
    # =====================================================
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

    # =====================================================
    # 情况 C：obs 列
    #       数值型 → 连续色条
    #       分类/字符串/布尔 → 离散 legend
    # =====================================================
    elif color_kind == "obs":
        values = plot_df["color_value"]

        # bool 不按连续变量处理，而是按分类变量处理
        is_bool = pd.api.types.is_bool_dtype(values)
        is_numeric = pd.api.types.is_numeric_dtype(values)

        # -------------------------
        # C1. obs 数值列：连续色条
        # -------------------------
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

        # -------------------------
        # C2. obs 分类列：离散颜色 + legend
        # -------------------------
        else:
            values = values.astype("object").where(values.notna(), "NA")
            values = values.astype(str).astype("category")

            cats = list(values.cat.categories)
            cmap_obj = plt.get_cmap(palette)

            color_map = {
                cat: cmap_obj(i % cmap_obj.N)
                for i, cat in enumerate(cats)
            }

            # ✅ 修改：按类别分组绘图，Scanpy 风格 legend
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
                # ✅ 修改：legend 改成两列，并且放在右侧中部，接近 Scanpy 横向布局
                leg = ax.legend(
                    title=color,
                    bbox_to_anchor=(1.04, 0.5),   # ✅ 修改：从右上改为右侧居中
                    loc="center left",            # ✅ 修改：从 upper left 改为 center left
                    frameon=False,
                    markerscale=2,
                    fontsize=9,
                    title_fontsize=10,
                    borderaxespad=0.0,
                    ncol=2,                       # ✅ 修改：关键，legend 分两列显示
                    columnspacing=1.2,            # ✅ 修改：两列间距
                    handletextpad=0.4,            # ✅ 修改：点和文字间距
                )

                # ✅ 修改：关键，不让 tight_layout / layout 系统为了 legend 压缩主图
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

    # =====================================================
    # 4️⃣ Scanpy 风格美化
    # =====================================================
    ax.set_xlabel(x_label, fontsize=14)
    ax.set_ylabel(y_label, fontsize=14)

    if color is None:
        ax.set_title("PCA", fontsize=14, pad=8)
    else:
        ax.set_title(str(color), fontsize=14, pad=8)

    # ✅ 修改：更接近 Scanpy，默认不画网格
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

    # ✅ 修改：控制 PCA 主图框的高宽比例，避免变成图2那种瘦高图
    ax.set_box_aspect(0.75)

    # =====================================================
    # ✅ 修改：不要让 tight_layout 把主图挤窄
    # =====================================================
    if legend_loc == "right_margin":
        # ✅ 修改：手动给右侧 legend 留空间，主图不会被压成竖条
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


# 1️⃣ 画 pca_variance_ratio（最像 sc.pl.pca_variance_ratio）
# 单个PC贡献 👉 每个PC单独贡献多少信息
# 🎯 这个图的核心用途
# 👉 找：
# ✔️ “elbow point（拐点）”
def pca_variance_ratio(atlas: Atlas, n_pcs=50, log=True, figsize=(16, 8)):
    """
    直接从数据库 uns_pca_stats 画 explained variance ratio
    类似：sc.pl.pca_variance_ratio(adata, log=True, n_pcs=50)
    """
    conn = atlas.connection

    df = conn.execute(f"""
        SELECT pc_index, variance, variance_ratio
        FROM uns_pca_stats
        ORDER BY pc_index
        LIMIT {n_pcs}
    """).fetchdf()

    x = df["pc_index"].to_numpy() + 1  # Scanpy 习惯从 PC1 开始
    y = df["variance_ratio"].to_numpy()

    plt.figure(figsize=figsize)
    plt.plot(x, y, marker="o")
    plt.xlabel("Principal component")
    plt.ylabel("Explained variance ratio")
    plt.title("PCA variance ratio")

    if log:
        plt.yscale("log")

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# 2️⃣ 画累计解释方差（这个也很常用）
# 看“总信息覆盖多少”
# 👉 横轴：PC1 → PC50
# 👉 纵轴：累计解释了多少数据信息
def pca_variance_ratio_cumsum(atlas: Atlas, n_pcs=50, figsize=(16, 8)):
    """
    画累计 explained variance ratio
    """
    conn = atlas.connection

    df = conn.execute(f"""
        SELECT pc_index, variance_ratio
        FROM uns_pca_stats
        ORDER BY pc_index
        LIMIT {n_pcs}
    """).fetchdf()

    x = df["pc_index"].to_numpy() + 1
    y = df["variance_ratio"].to_numpy()
    y_cum = np.cumsum(y)

    plt.figure(figsize=figsize)
    plt.plot(x, y_cum, marker="o")
    plt.xlabel("Principal component")
    plt.ylabel("Cumulative explained variance ratio")
    plt.title("Cumulative PCA variance ratio")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

