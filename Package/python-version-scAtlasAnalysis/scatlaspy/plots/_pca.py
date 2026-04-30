import numpy as np
from ..data import Atlas
import matplotlib.pyplot as plt
from datetime import datetime


''' 可视化 / 对外入口层 '''
# 用 PCA 的前两个主成分（PC1, PC2）做二维散点图
# 然后用某个基因（这里是 CST3）的表达值给点上色。
def pca(
        atlas: Atlas,
        color: str | None = None,  # 例如 "CST3"
        x_pc: int = 0,
        y_pc: int = 1,
        annotate_var_explained: bool = True,
        sample_n: int | None = 50000,  # SQL先抽样，适合大数据
        use_expr_field: str = "data_log1p",  # 给基因着色时用哪个表达列
        figsize=(6, 6),
        point_size: float = 8,
        alpha: float = 0.9
):
    """
    数据库版 Scanpy 风格 PCA 图

    参数
    ----
    atlas : Atlas
    color : str | None
        按什么上色：
        - None: 单色散点
        - 基因名: 按该基因表达上色，比如 "CST3"
    x_pc / y_pc : int
        画哪两个主成分，0-based
    annotate_var_explained : bool
        是否在坐标轴上标注 explained variance ratio
    sample_n : int | None
        抽样多少细胞。None 表示不抽样
    use_expr_field : str
        基因表达取哪个字段，比如 "data" / "data_log1p" / "data_scale"
    """

    print("\n==== pca plot ====")
    start = datetime.now()
    conn = atlas.connection

    pcx = f"pc{x_pc}"
    pcy = f"pc{y_pc}"

    # -------------------------------------------------
    # 0️⃣ 检查表和列是否存在
    # -------------------------------------------------
    obsm_cols = [r[1] for r in conn.execute("PRAGMA table_info(obsm_X_pca)").fetchall()]
    if pcx not in obsm_cols or pcy not in obsm_cols:
        raise ValueError(
            f"obsm_X_pca 中不存在列: {pcx} 或 {pcy}\n"
            f"请先运行 atlas.pca()"
        )

    # -------------------------------------------------
    # 1️⃣ 读 explained variance ratio
    # -------------------------------------------------
    evr = conn.execute(f"""
        SELECT pc_index, variance_ratio
        FROM uns_pca_stats
        WHERE pc_index IN ({x_pc}, {y_pc})
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

    # -------------------------------------------------
    # 2️⃣ SQL 先抽样细胞
    # -------------------------------------------------
    if sample_n is None:
        pca_query = f"""
            SELECT atlas_cell_id, {pcx}, {pcy}
            FROM obsm_X_pca
        """
    else:
        pca_query = f"""
            SELECT atlas_cell_id, {pcx}, {pcy}
            FROM obsm_X_pca
            USING SAMPLE {int(sample_n)} ROWS
        """

    pca_df = conn.execute(pca_query).fetchdf()

    # -------------------------------------------------
    # 3️⃣ 如果按基因上色：取该基因表达
    # -------------------------------------------------
    if color is not None:
        # 先查 gene id
        gene_row = conn.execute(f"""
            SELECT atlas_gene_id
            FROM var
            WHERE atlas_gene_name = '{color}'
            LIMIT 1
        """).fetchone()

        if gene_row is None:
            raise ValueError(f"var 中找不到基因: {color}")

        gene_id = gene_row[0]

        # 对抽样到的细胞取该基因表达，没有就补0
        conn.register("_pca_cells_tmp", pca_df[["atlas_cell_id"]])

        expr_df = conn.execute(f"""
            SELECT
                c.atlas_cell_id,
                COALESCE(x.{use_expr_field}, 0.0) AS expr
            FROM _pca_cells_tmp c
            LEFT JOIN X_CSRO_data x
              ON c.atlas_cell_id = x.atlas_cell_id
             AND x.atlas_gene_id = {int(gene_id)}
        """).fetchdf()

        conn.unregister("_pca_cells_tmp")

        plot_df = pca_df.merge(expr_df, on="atlas_cell_id", how="left")
        plot_df["expr"] = plot_df["expr"].fillna(0.0)

    else:
        plot_df = pca_df.copy()

    # -------------------------------------------------
    # 4️⃣ 画图
    # -------------------------------------------------
    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    ax.set_facecolor("white")

    if color is None:
        ax.scatter(
            plot_df[pcx].to_numpy(),
            plot_df[pcy].to_numpy(),
            s=point_size,
            c="#7f7f7f",
            alpha=alpha,
            linewidths=0
        )
    else:
        sc = ax.scatter(
            plot_df[pcx].to_numpy(),
            plot_df[pcy].to_numpy(),
            s=point_size,
            c=plot_df["expr"].to_numpy(),
            cmap="viridis",
            alpha=alpha,
            linewidths=0
        )
        cbar = plt.colorbar(sc, ax=ax, pad=0.02)
        cbar.set_label(color, fontsize=12)

    ax.set_xlabel(x_label, fontsize=16)
    ax.set_ylabel(y_label, fontsize=16)
    ax.set_title(color if color is not None else "PCA", fontsize=14, pad=8)

    ax.grid(True, color="#d9d9d9", linewidth=0.8, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.tick_params(axis="both", labelsize=12, width=1.0, length=4)

    plt.tight_layout(pad=0.8)
    plt.show()

    print(f"Done in {(datetime.now() - start).total_seconds():.2f}s")



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

