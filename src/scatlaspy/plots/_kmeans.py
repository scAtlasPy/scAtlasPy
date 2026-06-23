import matplotlib.pyplot as plt
from datetime import datetime
from ..data import Atlas


# KMeans 聚类结果图,看数量（cluster 大小）
def kmeans_cluster_size(
        atlas: Atlas,
        use_obs_col: str = "kmeans",
        figsize: tuple[float, float] | None=(7, 4),
        show_percent: bool = True,
        title: str | None = None
):

    """绘制 K-means cluster 的细胞数量。

    该函数读取 ``obs`` 中的聚类列，统计每个 cluster 的细胞数，并绘制柱状图。适合检查聚类是否过度不均衡或是否存在很小的 cluster。

    Parameters
    ----------
    atlas
        Atlas 对象。通常需要已经连接到 DuckDB 数据库，并包含该函数读取或写入所需的 ``obs``、``var``、表达矩阵或结果表。
    use_obs_col
        读取或写入 ``obs`` 的列名。
    figsize
        图形大小。为 ``None`` 时使用函数默认尺寸。
    show_percent
        是否在图中展示百分比信息。
    title
        图标题。为 ``None`` 时使用默认标题。

    Returns
    -------
    None

    Examples
    --------
    绘制默认 K-means 聚类大小::

        sap.tl.kmeans(atlas, n_clusters=20)
        sap.pl.kmeans_cluster_size(atlas)

    绘制自定义聚类列并显示百分比::

        sap.pl.kmeans_cluster_size(
            atlas,
            use_obs_col="kmeans_50",
            show_percent=True,
            title="K-means 50 cluster size",
        )"""

    start = datetime.now()
    conn = atlas.connection

    # 检查 obs 中是否存在 kmeans 列
    obs_cols = [r[1] for r in conn.execute("PRAGMA table_info(obs)").fetchall()]
    if use_obs_col not in obs_cols:
        raise ValueError(
            f"obs 中不存在列: {use_obs_col}\n"
            f"请先运行 sap.tl.kmeans(atlas, use_obs_col='{use_obs_col}')"
        )

    # 统计 cluster 数量
    df = conn.execute(f"""
        SELECT
            {use_obs_col} AS cluster,
            COUNT(*) AS n_cells
        FROM obs
        WHERE {use_obs_col} IS NOT NULL
        GROUP BY {use_obs_col}
        ORDER BY {use_obs_col}
    """).fetchdf()

    if len(df) == 0:
        raise ValueError(
            f"obs.{use_obs_col} 中没有可用聚类结果。\n"
            f"请先运行 sap.tl.kmeans(atlas)"
        )

    df["cluster"] = df["cluster"].astype(int).astype(str)
    df["pct"] = df["n_cells"] / df["n_cells"].sum() * 100

    # 画图
    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    ax.set_facecolor("white")

    y = df["pct"].to_numpy() if show_percent else df["n_cells"].to_numpy()

    ax.bar(
        df["cluster"].to_numpy(),
        y,
        width=0.75
    )

    ax.set_xlabel("KMeans cluster", fontsize=13)
    ax.set_ylabel("Percent of cells (%)" if show_percent else "Number of cells", fontsize=13)

    if title is None:
        title = f"{use_obs_col} cluster distribution"

    ax.set_title(title, fontsize=14, pad=10)

    ax.grid(True, axis="y", alpha=0.3)
    ax.grid(False, axis="x")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.tick_params(axis="both", labelsize=11)

    plt.tight_layout()
    plt.show()

