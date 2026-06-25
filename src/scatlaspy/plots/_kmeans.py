import matplotlib.pyplot as plt
from datetime import datetime
from os import PathLike
from ..data import Atlas


def kmeans_cluster_size(
        atlas: Atlas,
        use_obs_col: str = "kmeans",
        figsize: tuple[float, float] | None=(7, 4),
        show_percent: bool = True,
        title: str | None = None,
        save_path: PathLike[str] | str | None = None
):

    """绘制 K-means 或其他分组列的细胞数量分布。

    该函数从 ``obs`` 表读取 ``use_obs_col`` 指定的分组列，统计每个分组包含的细胞数，
    并用柱状图展示分组规模。默认按百分比显示，也可以切换为原始细胞数。

    该图常用于检查聚类结果是否过度不均衡、是否存在很小的 cluster，或确认不同
    聚类参数下的分组规模是否符合预期。虽然默认列名是 ``"kmeans"``，但也可以用于
    ``leiden``、``cell_type`` 等任意 ``obs`` 分组列，只要该列能转换为整数标签。

    Parameters
    ----------
    atlas
        Atlas 对象。要求已经连接 DuckDB 数据库，并且数据库中包含 ``obs`` 表。
    use_obs_col
        ``obs`` 中用于统计分组大小的列名。默认读取 ``"kmeans"``。
    figsize
        Matplotlib 图像大小。默认 ``(7, 4)``。
    show_percent
        是否把 y 轴显示为细胞百分比。为 ``False`` 时显示每个分组的原始细胞数量。
    title
        图标题。为 ``None`` 时自动使用 ``"{use_obs_col} cluster distribution"``。
    save_path
        图片保存路径。为 ``None`` 时只显示图片，不保存。

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
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()
