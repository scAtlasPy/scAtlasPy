import matplotlib.pyplot as plt
from datetime import datetime
from ..data import Atlas


# KMeans 聚类结果图,看数量（cluster 大小）
def kmeans_cluster_size(
        atlas: Atlas,
        obs_col: str = "kmeans",
        figsize=(7, 4),
        show_percent: bool = True,
        title: str | None = None
):

    print("\n==== plot_kmeans ====")
    start = datetime.now()
    conn = atlas.connection

    # 检查 obs 中是否存在 kmeans 列
    obs_cols = [r[1] for r in conn.execute("PRAGMA table_info(obs)").fetchall()]
    if obs_col not in obs_cols:
        raise ValueError(
            f"obs 中不存在列: {obs_col}\n"
            f"请先运行 sap.tl.kmeans(atlas, obs_col='{obs_col}')"
        )

    # 统计 cluster 数量
    df = conn.execute(f"""
        SELECT
            {obs_col} AS cluster,
            COUNT(*) AS n_cells
        FROM obs
        WHERE {obs_col} IS NOT NULL
        GROUP BY {obs_col}
        ORDER BY {obs_col}
    """).fetchdf()

    if len(df) == 0:
        raise ValueError(
            f"obs.{obs_col} 中没有可用聚类结果。\n"
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
        title = f"{obs_col} cluster distribution"

    ax.set_title(title, fontsize=14, pad=10)

    ax.grid(True, axis="y", alpha=0.3)
    ax.grid(False, axis="x")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.tick_params(axis="both", labelsize=11)

    plt.tight_layout()
    plt.show()

    print(f"Done in {(datetime.now() - start).total_seconds():.2f}s")

    return df

