import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from datetime import datetime
import math

# 🧠 这个函数的两个模式，你怎么选？
# ✅ use_all_cells=True
# 适合：
# 你想和 Scanpy 图尽量一致
# 数据规模还可以接受
# 特点：
# 会构造 obs × top_genes
# 如果 top_genes=20，其实不大，通常能接受
# ✅ use_all_cells=False
# 适合：
# 超大数据
# 更关注趋势，不追求和 Scanpy 完全一致
# 特点：
# 更快
# 更省内存

# 🚀 你这个数据库版的优势
# 💥 1. 不需要把整个矩阵转 dense
# Scanpy 小数据可以 dense，大数据很危险。
# 你这里始终在：
# CSR长表 + SQL聚合
# 上做。
#
# 💥 2. 很适合做大数据 benchmark
# 这张图其实很适合写进你论文的 supplementary：
# Scanpy：小数据能画
# 你：大数据也能画

# 在 DuckDB 里先聚合，再只取 top 基因，再画图

# 导入完数据，直接画图:
# “最高表达基因占比图（highest expressed genes）”
# 👉 用来检查 有没有少数基因“垄断”表达（技术偏差）
def highest_expr_genes_sql(
        atlas,
        n_top: int = 20,
        use_all_cells: bool = True,     # ✅ 更接近 Scanpy 语义
        show_outliers: bool = False,    # ✅ 是否绘制离群点
        max_outliers: int = 5000,       # ✅ 每个基因最多绘制多少个离群点
        figsize=(12, 10)
):
    """
    数据库版 sc.pl.highest_expr_genes（SQL分位数版，Scanpy-like style）

    参数
    ----
    atlas : Atlas
        Atlas 对象
    n_top : int
        取平均占比最高的前 n 个基因
    use_all_cells : bool
        True  -> 更接近 Scanpy：所有细胞参与，未表达补 0
        False -> 仅统计非零表达细胞（更快，更适合超大数据）
    show_outliers : bool
        是否绘制离群点
    max_outliers : int
        每个基因最多绘制多少个离群点（防止内存爆炸）
    figsize : tuple
        图大小
    """
    # 日常 / 大数据 / 正式流程
    #  use_all_cells=False,
    #  show_outliers=False,

    # 小数据 / 和 Scanpy 对齐 / 做展示图
    #     use_all_cells=True,
    #     show_outliers=True,
    #     max_outliers=5000,

    print("\n==== plot_highest_expr_genes_sql (final aligned to scanpy) ====")
    start = datetime.now()
    conn = atlas.connection

    try:
        conn.execute(f"PRAGMA threads={os.cpu_count()}")
    except Exception:
        pass

    # -------------------------------------------------
    # 1️⃣ 每个细胞总 counts
    # -------------------------------------------------
    conn.execute("""
        CREATE OR REPLACE TEMP TABLE _cell_total_counts AS
        SELECT
            atlas_cell_id,
            SUM(data) AS total_counts
        FROM X_CSRO_data
        GROUP BY atlas_cell_id
    """)

    # -------------------------------------------------
    # 2️⃣ 选 top genes
    #     ✅【关键修改】当 use_all_cells=True 时，
    #     top gene 也按“所有细胞平均 fraction”来算
    # -------------------------------------------------
    if use_all_cells:
        conn.execute("""
            CREATE OR REPLACE TEMP TABLE _all_cells AS
            SELECT atlas_cell_id
            FROM obs
        """)

        conn.execute("""
            CREATE OR REPLACE TEMP TABLE _all_gene_mean_pct AS
            WITH gene_pct_nonzero AS (
                SELECT
                    x.atlas_cell_id,
                    x.atlas_gene_id,
                    x.data * 100.0 / t.total_counts AS pct
                FROM X_CSRO_data x
                JOIN _cell_total_counts t
                    ON x.atlas_cell_id = t.atlas_cell_id
                WHERE t.total_counts > 0
            ),
            gene_sum_pct AS (
                SELECT
                    atlas_gene_id,
                    SUM(pct) AS sum_pct
                FROM gene_pct_nonzero
                GROUP BY atlas_gene_id
            ),
            n_cells AS (
                SELECT COUNT(*) AS total_cells
                FROM _all_cells
            )
            SELECT
                v.atlas_gene_id,
                v.atlas_gene_name,
                COALESCE(g.sum_pct, 0.0) / n.total_cells AS mean_pct
            FROM var v
            CROSS JOIN n_cells n
            LEFT JOIN gene_sum_pct g
              ON v.atlas_gene_id = g.atlas_gene_id
        """)

        conn.execute(f"""
            CREATE OR REPLACE TEMP TABLE _top_expr_genes AS
            SELECT
                atlas_gene_id,
                atlas_gene_name,
                mean_pct
            FROM _all_gene_mean_pct
            ORDER BY mean_pct DESC
            LIMIT {int(n_top)}
        """)
    else:
        # ✅ 保留你原来的稀疏快版逻辑
        conn.execute(f"""
            CREATE OR REPLACE TEMP TABLE _top_expr_genes AS
            SELECT
                x.atlas_gene_id,
                v.atlas_gene_name,
                AVG(x.data * 100.0 / t.total_counts) AS mean_pct
            FROM X_CSRO_data x
            JOIN _cell_total_counts t
                ON x.atlas_cell_id = t.atlas_cell_id
            JOIN var v
                ON x.atlas_gene_id = v.atlas_gene_id
            WHERE t.total_counts > 0
            GROUP BY x.atlas_gene_id, v.atlas_gene_name
            ORDER BY mean_pct DESC
            LIMIT {int(n_top)}
        """)

    # -------------------------------------------------
    # 3️⃣ SQL 直接计算标准 boxplot 统计量
    # -------------------------------------------------
    if use_all_cells:
        stats_df = conn.execute("""
            WITH all_cells AS (
                SELECT atlas_cell_id
                FROM obs
            ),
            cell_gene_grid AS (
                SELECT
                    c.atlas_cell_id,
                    g.atlas_gene_id,
                    g.atlas_gene_name,
                    g.mean_pct
                FROM all_cells c
                CROSS JOIN _top_expr_genes g
            ),
            top_gene_pct AS (
                SELECT
                    x.atlas_cell_id,
                    x.atlas_gene_id,
                    x.data * 100.0 / t.total_counts AS pct
                FROM X_CSRO_data x
                JOIN _cell_total_counts t
                    ON x.atlas_cell_id = t.atlas_cell_id
                JOIN _top_expr_genes g
                    ON x.atlas_gene_id = g.atlas_gene_id
                WHERE t.total_counts > 0
            ),
            full_values AS (
                SELECT
                    grid.atlas_gene_name,
                    grid.mean_pct,
                    COALESCE(p.pct, 0.0) AS pct
                FROM cell_gene_grid grid
                LEFT JOIN top_gene_pct p
                    ON grid.atlas_cell_id = p.atlas_cell_id
                   AND grid.atlas_gene_id = p.atlas_gene_id
            ),
            quartiles AS (
                SELECT
                    atlas_gene_name,
                    mean_pct,
                    COUNT(*) AS n,
                    quantile_cont(pct, 0.25) AS q1,
                    quantile_cont(pct, 0.50) AS median,
                    quantile_cont(pct, 0.75) AS q3
                FROM full_values
                GROUP BY atlas_gene_name, mean_pct
            ),
            whiskers AS (
                SELECT
                    f.atlas_gene_name,
                    MIN(CASE
                            WHEN f.pct >= (q.q1 - 1.5 * (q.q3 - q.q1))
                            THEN f.pct
                        END) AS whisker_low,
                    MAX(CASE
                            WHEN f.pct <= (q.q3 + 1.5 * (q.q3 - q.q1))
                            THEN f.pct
                        END) AS whisker_high
                FROM full_values f
                JOIN quartiles q
                  ON f.atlas_gene_name = q.atlas_gene_name
                GROUP BY f.atlas_gene_name
            )
            SELECT
                q.atlas_gene_name,
                q.mean_pct,
                q.n,
                COALESCE(w.whisker_low, q.q1) AS whisker_low,
                q.q1,
                q.median,
                q.q3,
                COALESCE(w.whisker_high, q.q3) AS whisker_high
            FROM quartiles q
            LEFT JOIN whiskers w
              ON q.atlas_gene_name = w.atlas_gene_name
            ORDER BY q.mean_pct DESC, q.atlas_gene_name
        """).fetchdf()
    else:
        stats_df = conn.execute("""
            WITH top_gene_values AS (
                SELECT
                    g.atlas_gene_name,
                    g.mean_pct,
                    x.data * 100.0 / t.total_counts AS pct
                FROM X_CSRO_data x
                JOIN _cell_total_counts t
                    ON x.atlas_cell_id = t.atlas_cell_id
                JOIN _top_expr_genes g
                    ON x.atlas_gene_id = g.atlas_gene_id
                WHERE t.total_counts > 0
            ),
            quartiles AS (
                SELECT
                    atlas_gene_name,
                    mean_pct,
                    COUNT(*) AS n,
                    quantile_cont(pct, 0.25) AS q1,
                    quantile_cont(pct, 0.50) AS median,
                    quantile_cont(pct, 0.75) AS q3
                FROM top_gene_values
                GROUP BY atlas_gene_name, mean_pct
            ),
            whiskers AS (
                SELECT
                    f.atlas_gene_name,
                    MIN(CASE
                            WHEN f.pct >= (q.q1 - 1.5 * (q.q3 - q.q1))
                            THEN f.pct
                        END) AS whisker_low,
                    MAX(CASE
                            WHEN f.pct <= (q.q3 + 1.5 * (q.q3 - q.q1))
                            THEN f.pct
                        END) AS whisker_high
                FROM top_gene_values f
                JOIN quartiles q
                  ON f.atlas_gene_name = q.atlas_gene_name
                GROUP BY f.atlas_gene_name
            )
            SELECT
                q.atlas_gene_name,
                q.mean_pct,
                q.n,
                COALESCE(w.whisker_low, q.q1) AS whisker_low,
                q.q1,
                q.median,
                q.q3,
                COALESCE(w.whisker_high, q.q3) AS whisker_high
            FROM quartiles q
            LEFT JOIN whiskers w
              ON q.atlas_gene_name = w.atlas_gene_name
            ORDER BY q.mean_pct DESC, q.atlas_gene_name
        """).fetchdf()

    # -------------------------------------------------
    # 3.5️⃣ 可选：提取离群点
    # -------------------------------------------------
    outlier_df = None

    if show_outliers:
        print("-> extracting outliers ...")

        if use_all_cells:
            outlier_df = conn.execute(f"""
                WITH all_cells AS (
                    SELECT atlas_cell_id
                    FROM obs
                ),
                cell_gene_grid AS (
                    SELECT
                        c.atlas_cell_id,
                        g.atlas_gene_id,
                        g.atlas_gene_name
                    FROM all_cells c
                    CROSS JOIN _top_expr_genes g
                ),
                top_gene_pct AS (
                    SELECT
                        x.atlas_cell_id,
                        x.atlas_gene_id,
                        x.data * 100.0 / t.total_counts AS pct
                    FROM X_CSRO_data x
                    JOIN _cell_total_counts t
                        ON x.atlas_cell_id = t.atlas_cell_id
                    JOIN _top_expr_genes g
                        ON x.atlas_gene_id = g.atlas_gene_id
                    WHERE t.total_counts > 0
                ),
                full_values AS (
                    SELECT
                        grid.atlas_gene_name,
                        COALESCE(p.pct, 0.0) AS pct
                    FROM cell_gene_grid grid
                    LEFT JOIN top_gene_pct p
                        ON grid.atlas_cell_id = p.atlas_cell_id
                       AND grid.atlas_gene_id = p.atlas_gene_id
                ),
                bounds AS (
                    SELECT
                        atlas_gene_name,
                        quantile_cont(pct, 0.25) AS q1,
                        quantile_cont(pct, 0.75) AS q3
                    FROM full_values
                    GROUP BY atlas_gene_name
                ),
                outliers AS (
                    SELECT
                        f.atlas_gene_name,
                        f.pct,
                        ROW_NUMBER() OVER (
                            PARTITION BY f.atlas_gene_name
                            ORDER BY RANDOM()
                        ) AS rn
                    FROM full_values f
                    JOIN bounds b
                      ON f.atlas_gene_name = b.atlas_gene_name
                    WHERE f.pct < (b.q1 - 1.5 * (b.q3 - b.q1))
                       OR f.pct > (b.q3 + 1.5 * (b.q3 - b.q1))
                )
                SELECT atlas_gene_name, pct
                FROM outliers
                WHERE rn <= {int(max_outliers)}
            """).fetchdf()
        else:
            outlier_df = conn.execute(f"""
                WITH top_gene_values AS (
                    SELECT
                        g.atlas_gene_name,
                        x.data * 100.0 / t.total_counts AS pct
                    FROM X_CSRO_data x
                    JOIN _cell_total_counts t
                        ON x.atlas_cell_id = t.atlas_cell_id
                    JOIN _top_expr_genes g
                        ON x.atlas_gene_id = g.atlas_gene_id
                    WHERE t.total_counts > 0
                ),
                bounds AS (
                    SELECT
                        atlas_gene_name,
                        quantile_cont(pct, 0.25) AS q1,
                        quantile_cont(pct, 0.75) AS q3
                    FROM top_gene_values
                    GROUP BY atlas_gene_name
                ),
                outliers AS (
                    SELECT
                        f.atlas_gene_name,
                        f.pct,
                        ROW_NUMBER() OVER (
                            PARTITION BY f.atlas_gene_name
                            ORDER BY RANDOM()
                        ) AS rn
                    FROM top_gene_values f
                    JOIN bounds b
                      ON f.atlas_gene_name = b.atlas_gene_name
                    WHERE f.pct < (b.q1 - 1.5 * (b.q3 - b.q1))
                       OR f.pct > (b.q3 + 1.5 * (b.q3 - b.q1))
                )
                SELECT atlas_gene_name, pct
                FROM outliers
                WHERE rn <= {int(max_outliers)}
            """).fetchdf()

    # -------------------------------------------------
    # 4️⃣ 画图（Scanpy-like style）
    # -------------------------------------------------
    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    ax.set_facecolor("white")

    stats_df = stats_df.sort_values("mean_pct", ascending=False).reset_index(drop=True)

    y_positions = list(range(len(stats_df), 0, -1))
    box_height = 0.78

    scanpy_colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#bcbd22", "#17becf", "#aec7e8",
        "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5", "#c49c94",
        "#f7b6d2", "#dbdb8d", "#9edae5", "#8c6d31", "#7f7f7f"
    ]

    for i, (y, row) in enumerate(zip(y_positions, stats_df.itertuples(index=False))):
        whisker_low = float(row.whisker_low)
        q1 = float(row.q1)
        median = float(row.median)
        q3 = float(row.q3)
        whisker_high = float(row.whisker_high)

        color = scanpy_colors[i % len(scanpy_colors)]

        ax.hlines(y, whisker_low, whisker_high, linewidth=1.1, color="#4a4a4a", zorder=2)
        ax.vlines(whisker_low, y - 0.11, y + 0.11, linewidth=1.1, color="#4a4a4a", zorder=2)
        ax.vlines(whisker_high, y - 0.11, y + 0.11, linewidth=1.1, color="#4a4a4a", zorder=2)

        rect = Rectangle(
            (q1, y - box_height / 2),
            max(q3 - q1, 1e-12),
            box_height,
            facecolor=color,
            edgecolor="#4a4a4a",
            linewidth=1.0,
            alpha=0.9,
            zorder=3
        )
        ax.add_patch(rect)

        ax.vlines(
            median,
            y - box_height / 2,
            y + box_height / 2,
            linewidth=1.4,
            color="#2f2f2f",
            zorder=4
        )

    # 可选 outliers
    if show_outliers and outlier_df is not None and len(outlier_df) > 0:
        gene_to_y = {g: y for g, y in zip(stats_df["atlas_gene_name"], y_positions)}

        xs = outlier_df["pct"].to_numpy()
        ys = [gene_to_y[g] for g in outlier_df["atlas_gene_name"]]

        ax.scatter(
            xs,
            ys,
            s=8,
            facecolors="none",
            edgecolors="#2f2f2f",
            linewidths=0.5,
            alpha=0.7,
            zorder=5
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(stats_df["atlas_gene_name"].tolist(), fontsize=16)
    ax.set_xlabel("% of total counts", fontsize=18)
    ax.set_ylabel("")
    ax.set_title("Highest expressed genes", fontsize=12, weight="normal", pad=8)

    ax.grid(True, axis="x", color="#d9d9d9", linewidth=0.8, alpha=0.8)
    ax.grid(False, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)

    ax.tick_params(axis="x", labelsize=12, width=1.0, length=4)
    ax.tick_params(axis="y", width=1.0, length=4)

    plt.tight_layout(pad=0.8)
    plt.show()

    # -------------------------------------------------
    # 5️⃣ 清理
    # -------------------------------------------------
    conn.execute("DROP TABLE IF EXISTS _cell_total_counts")
    conn.execute("DROP TABLE IF EXISTS _top_expr_genes")
    conn.execute("DROP TABLE IF EXISTS _all_cells")
    conn.execute("DROP TABLE IF EXISTS _all_gene_mean_pct")

    print(f"Done in {(datetime.now() - start).total_seconds():.2f}s")





# 可视化QC指标 , 画 QC 小提琴图
def violin_qc_metrics(
        atlas,
        keys=None,
        jitter: float = 0.4,
        multi_panel: bool = True,
        figsize=None,
        use_filtered: bool = False,
        filter_key: str = "filter_cells",
        sample_n: int | None = 50000,
        random_state: int = 0
):
    """
    Scanpy 风格 QC violin plot（SQL 先抽样版，大数据更友好）

    参数
    ----
    atlas : Atlas
        Atlas 对象
    keys : list[str] | None
        要画的 obs 列名
        默认:
        ["n_genes_by_counts", "cell_total_counts", "pct_counts_mt"]
    jitter : float
        散点抖动幅度，类似 scanpy 的 jitter=0.4
    multi_panel : bool
        True -> 多个 panel 横向排列
        False -> 单图纵向排列
    figsize : tuple | None
        图大小；None 时自动估计
    use_filtered : bool
        是否只画通过过滤的细胞（obs.filter_cells = TRUE）
    filter_key : str
        过滤列名，默认 "filter_cells"
    sample_n : int | None
        SQL 中先抽样的细胞数
        None 表示不抽样，全量作图（大数据不推荐）
    random_state : int
        随机种子（目前主要用于保持接口一致）
    """

    print("\n==== violin_qc_metrics (SQL first sampling) ====")
    start = datetime.now()
    conn = atlas.connection

    if keys is None:
        keys = ["n_genes_by_counts", "cell_total_counts", "pct_counts_mt"]

    # -------------------------------------------------
    # 0️⃣ 检查 obs 列是否存在
    # -------------------------------------------------
    obs_cols = [r[1] for r in conn.execute("PRAGMA table_info(obs)").fetchall()]

    missing = [k for k in keys if k not in obs_cols]
    if missing:
        raise ValueError(
            f"obs 中不存在这些列: {missing}\n"
            f"请先运行 sap.pp.calculate_qc_metrics(atlas)"
        )

    if use_filtered and filter_key not in obs_cols:
        raise ValueError(f"obs 中不存在过滤列: {filter_key}")

    # -------------------------------------------------
    # 1️⃣ SQL 先抽样，再 fetchdf
    # -------------------------------------------------
    select_cols = ", ".join(keys)

    where_clauses = []
    if use_filtered:
        where_clauses.append(f"{filter_key} = TRUE")

    # 只保留至少有一个 key 非空的行（减少无效点）
    non_null_cond = " OR ".join([f"{k} IS NOT NULL" for k in keys])
    where_clauses.append(f"({non_null_cond})")

    where_sql = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""

    if sample_n is None:
        # ✅ 全量（大数据不推荐）
        query = f"""
            SELECT {select_cols}
            FROM obs
            {where_sql}
        """
    else:
        # ✅【关键优化】先在 SQL 中抽样，再 fetchdf
        # DuckDB USING SAMPLE 是 reservoir/system sample 风格，适合这里
        query = f"""
            SELECT {select_cols}
            FROM obs
            {where_sql}
            USING SAMPLE {int(sample_n)} ROWS
        """

    df = conn.execute(query).fetchdf()

    # -------------------------------------------------
    # 2️⃣ 清理列：去掉全空列
    # -------------------------------------------------
    keep_keys = []
    for k in keys:
        if k in df.columns and df[k].notna().sum() > 0:
            keep_keys.append(k)

    if len(keep_keys) == 0:
        raise ValueError("没有可绘制的 QC 列（全部为空）")

    df = df[keep_keys]
    keys = keep_keys

    # -------------------------------------------------
    # 3️⃣ 自动布局
    # -------------------------------------------------
    n = len(keys)

    if figsize is None:
        if multi_panel:
            figsize = (4.2 * n, 5.0)
        else:
            figsize = (6.0, 4.0 * n)

    if multi_panel:
        fig, axes = plt.subplots(1, n, figsize=figsize, facecolor="white")
        if n == 1:
            axes = [axes]
    else:
        fig, axes = plt.subplots(n, 1, figsize=figsize, facecolor="white")
        if n == 1:
            axes = [axes]

    violin_color = "#1f77b4"
    edge_color = "#4a4a4a"
    point_color = "#2f2f2f"

    # -------------------------------------------------
    # 4️⃣ 逐列画 violin + jitter
    # -------------------------------------------------
    for ax, key in zip(axes, keys):
        vals = pd.to_numeric(df[key], errors="coerce").dropna().to_numpy()

        if len(vals) == 0:
            ax.set_visible(False)
            continue

        # violin
        parts = ax.violinplot(
            dataset=[vals],
            positions=[1],
            widths=0.9,
            showmeans=False,
            showmedians=True,
            showextrema=False
        )

        for pc in parts["bodies"]:
            pc.set_facecolor(violin_color)
            pc.set_edgecolor(edge_color)
            pc.set_alpha(0.9)
            pc.set_linewidth(1.0)

        if "cmedians" in parts:
            parts["cmedians"].set_color("#2f2f2f")
            parts["cmedians"].set_linewidth(1.2)

        # jitter points
        # 不再依赖 numpy 随机数，直接用 matplotlib + pandas index 生成均匀抖动
        y_jitter = 1 + (pd.Series(range(len(vals))).sample(frac=1, random_state=random_state).rank().to_numpy() / len(vals) - 0.5) * jitter

        ax.scatter(
            y_jitter,
            vals,
            s=2,
            c=point_color,
            alpha=0.55,
            linewidths=0
        )

        ax.set_title(key, fontsize=12, weight="normal", pad=8)
        ax.set_xlim(0.5, 1.5)
        ax.set_xticks([1])
        ax.set_xticklabels([""])
        ax.set_ylabel("value", fontsize=11)

        ax.grid(True, axis="y", color="#d9d9d9", linewidth=0.8, alpha=0.8)
        ax.grid(False, axis="x")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.0)
        ax.spines["bottom"].set_linewidth(1.0)

        ax.tick_params(axis="y", labelsize=10, width=1.0, length=4)
        ax.tick_params(axis="x", width=1.0, length=0)

    plt.tight_layout(pad=1.0)
    plt.show()

    print(f"Done in {(datetime.now() - start).total_seconds():.2f}s")


# QC 散点图（scatter plot）
# 用来发现“异常细胞”的关系图（不是看分布，是看相关性）
def scatter_qc_metrics(
        atlas,
        pairs=None,
        figsize=(10, 4),
        use_filtered: bool = False,
        filter_key: str = "filter_cells",
        sample_n: int | None = 50000,
        random_state: int = 0,
        point_size: float = 8,
        alpha: float = 0.7
):
    """
    Scanpy 风格 QC scatter plot（SQL 先抽样版，大数据更友好）

    参数
    ----
    atlas : Atlas
        Atlas 对象
    pairs : list[tuple[str, str]] | None
        要画的 (x, y) 列对
        默认:
        [
            ("cell_total_counts", "pct_counts_mt"),
            ("cell_total_counts", "n_genes_by_counts")
        ]
    figsize : tuple
        图大小
    use_filtered : bool
        是否只画通过过滤的细胞（obs.filter_cells = TRUE）
    filter_key : str
        过滤列名，默认 "filter_cells"
    sample_n : int | None
        SQL 中先抽样的细胞数
        None 表示不抽样，全量作图（大数据不推荐）
    random_state : int
        随机种子（主要用于接口一致性）
    point_size : float
        散点大小
    alpha : float
        散点透明度
    """

    print("\n==== scatter_qc_metrics (SQL first sampling) ====")
    start = datetime.now()
    conn = atlas.connection

    if pairs is None:
        pairs = [
            ("cell_total_counts", "pct_counts_mt"),
            ("cell_total_counts", "n_genes_by_counts"),
        ]

    # -------------------------------------------------
    # 0️⃣ 检查 obs 列是否存在
    # -------------------------------------------------
    obs_cols = [r[1] for r in conn.execute("PRAGMA table_info(obs)").fetchall()]

    needed_cols = set()
    for x, y in pairs:
        needed_cols.add(x)
        needed_cols.add(y)

    missing = [c for c in needed_cols if c not in obs_cols]
    if missing:
        raise ValueError(
            f"obs 中不存在这些列: {missing}\n"
            f"请先运行 sap.pp.calculate_qc_metrics(atlas)"
        )

    if use_filtered and filter_key not in obs_cols:
        raise ValueError(f"obs 中不存在过滤列: {filter_key}")

    # -------------------------------------------------
    # 1️⃣ SQL 先抽样，再 fetchdf
    # -------------------------------------------------
    select_cols = ", ".join(sorted(needed_cols))

    where_clauses = []

    if use_filtered:
        where_clauses.append(f"{filter_key} = TRUE")

    # 至少保证每个 pair 的 x/y 不是全空
    pair_valid_clauses = []
    for x, y in pairs:
        pair_valid_clauses.append(f"({x} IS NOT NULL AND {y} IS NOT NULL)")

    if pair_valid_clauses:
        where_clauses.append("(" + " OR ".join(pair_valid_clauses) + ")")

    where_sql = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""

    if sample_n is None:
        query = f"""
            SELECT {select_cols}
            FROM obs
            {where_sql}
        """
    else:
        # ✅ SQL 先抽样
        query = f"""
            SELECT {select_cols}
            FROM obs
            {where_sql}
            USING SAMPLE {int(sample_n)} ROWS
        """

    df = conn.execute(query).fetchdf()

    # -------------------------------------------------
    # 2️⃣ 自动布局
    # -------------------------------------------------
    n = len(pairs)

    fig, axes = plt.subplots(1, n, figsize=figsize, facecolor="white")
    if n == 1:
        axes = [axes]

    # -------------------------------------------------
    # 3️⃣ 逐图绘制
    # -------------------------------------------------
    for ax, (x_col, y_col) in zip(axes, pairs):
        sub = df[[x_col, y_col]].copy()
        sub[x_col] = pd.to_numeric(sub[x_col], errors="coerce")
        sub[y_col] = pd.to_numeric(sub[y_col], errors="coerce")
        sub = sub.dropna()

        if len(sub) == 0:
            ax.set_visible(False)
            continue

        ax.scatter(
            sub[x_col].to_numpy(),
            sub[y_col].to_numpy(),
            s=point_size,
            c="#7f7f7f",      # Scanpy-like 灰点
            alpha=alpha,
            linewidths=0
        )

        ax.set_xlabel(x_col, fontsize=16)
        ax.set_ylabel(y_col, fontsize=16)

        # Scanpy 风格：白底 + 淡网格 + 去掉右上边框
        ax.set_facecolor("white")
        ax.grid(True, color="#d9d9d9", linewidth=0.8, alpha=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.0)
        ax.spines["bottom"].set_linewidth(1.0)

        ax.tick_params(axis="both", labelsize=12, width=1.0, length=4)

    plt.tight_layout(pad=1.0)
    plt.show()

    print(f"Done in {(datetime.now() - start).total_seconds():.2f}s")


# 高变基因（HVG, Highly Variable Genes）选择图
def highly_variable_genes_plot(
        atlas,
        hvg_key: str = "highly_variable_genes",
        mean_key: str = "hvg_mean",
        var_key: str = "hvg_var",
        std_key: str = "hvg_std",
        score_key: str = "hvg_score",
        sample_other: int | None = 20000,
        figsize = None,
        point_size_hvg: float = 8,
        point_size_other: float = 6,
        alpha_hvg: float = 0.9,
        alpha_other: float = 0.6
):
    """
    数据库版 HVG 可视化（直接复用 var 中已存统计量，不再扫描 X_CSRO_data）

    参数
    ----
    atlas : Atlas
        Atlas 对象
    hvg_key : str
        var 中 HVG 布尔列名
    mean_key / var_key / std_key / score_key : str
        var 中已保存的 HVG 统计列名
    sample_other : int | None
        非 HVG 基因可选抽样数量，减少绘图点数
    figsize : tuple
        图大小
    """

    print("\n==== highly_variable_genes_plot ====")
    start = datetime.now()
    conn = atlas.connection

    # -------------------------------------------------
    # 0️⃣ 检查 var 中列是否存在
    # -------------------------------------------------
    var_cols = [r[1] for r in conn.execute("PRAGMA table_info(var)").fetchall()]

    needed = [hvg_key, mean_key, var_key, std_key, score_key, "atlas_gene_name"]
    missing = [c for c in needed if c not in var_cols]
    if missing:
        raise ValueError(
            f"var 中不存在这些列: {missing}\n"
            f"请先运行修改后的 sap.pp.highly_variable_genes(atlas)"
        )

    # -------------------------------------------------
    # 1️⃣ 直接从 var 读取 gene-level 结果
    # -------------------------------------------------
    df = conn.execute(f"""
        SELECT
            atlas_gene_id,
            atlas_gene_name,
            COALESCE({hvg_key}, FALSE) AS is_hvg,
            COALESCE({mean_key}, 0.0)  AS mean_expr,
            COALESCE({var_key}, 0.0)   AS var_expr,
            COALESCE({std_key}, 0.0)   AS std_expr,
            COALESCE({score_key}, 0.0) AS score_expr
        FROM var
        ORDER BY atlas_gene_id
    """).fetchdf()

    # -------------------------------------------------
    # 2️⃣ 构造左图显示量
    # -------------------------------------------------
    # 对 flavor="var"：左图直接显示 hvg_score (= var)
    # 对 flavor="cv" ：左图显示 hvg_score (= std/mean)
    # 这样可直接对齐你自己的排序逻辑
    df["var_norm_display"] = df["score_expr"]

    # 非 HVG 可选抽样
    if sample_other is not None:
        df_hvg = df[df["is_hvg"]].copy()
        df_other = df[~df["is_hvg"]].copy()

        if len(df_other) > sample_other:
            df_other = df_other.sample(sample_other, random_state=0)

        plot_df = pd.concat([df_hvg, df_other], axis=0, ignore_index=True)
    else:
        plot_df = df.copy()

    plot_hvg = plot_df[plot_df["is_hvg"]].copy()
    plot_other = plot_df[~plot_df["is_hvg"]].copy()

    # -------------------------------------------------
    # 3️⃣ 画图（Scanpy-like style）
    # -------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=figsize, facecolor="white")

    # ---------- 左图 ----------
    ax = axes[0]

    if len(plot_other) > 0:
        ax.scatter(
            plot_other["mean_expr"].to_numpy(),
            plot_other["var_norm_display"].to_numpy(),
            s=point_size_other,
            c="#8c8c8c",
            alpha=alpha_other,
            linewidths=0,
            label="other genes"
        )

    if len(plot_hvg) > 0:
        ax.scatter(
            plot_hvg["mean_expr"].to_numpy(),
            plot_hvg["var_norm_display"].to_numpy(),
            s=point_size_hvg,
            c="black",
            alpha=alpha_hvg,
            linewidths=0,
            label="highly variable genes"
        )

    ax.set_xlabel("mean expressions of genes", fontsize=16)
    ax.set_ylabel("variances of genes (normalized)", fontsize=16)

    ax.legend(
        frameon=True,
        fontsize=11,
        markerscale=1.0,
        loc="upper left",
        borderpad=0.4,
        handlelength=1.2,
        handletextpad=0.4
    )

    ax.grid(True, color="#d9d9d9", linewidth=0.8, alpha=0.8)
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.tick_params(axis="both", labelsize=12, width=1.0, length=4)

    # ---------- 右图 ----------
    ax = axes[1]

    if len(plot_other) > 0:
        ax.scatter(
            plot_other["mean_expr"].to_numpy(),
            plot_other["var_expr"].to_numpy(),
            s=point_size_other,
            c="#8c8c8c",
            alpha=alpha_other,
            linewidths=0
        )

    if len(plot_hvg) > 0:
        ax.scatter(
            plot_hvg["mean_expr"].to_numpy(),
            plot_hvg["var_expr"].to_numpy(),
            s=point_size_hvg,
            c="black",
            alpha=alpha_hvg,
            linewidths=0
        )

    ax.set_xlabel("mean expressions of genes", fontsize=16)
    ax.set_ylabel("variances of genes (not normalized)", fontsize=16)

    ax.grid(True, color="#d9d9d9", linewidth=0.8, alpha=0.8)
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.tick_params(axis="both", labelsize=12, width=1.0, length=4)

    plt.tight_layout(pad=1.0)
    plt.show()

    print(f"Done in {(datetime.now() - start).total_seconds():.2f}s")



def highly_variable_genes_plot_seurat(
        atlas,
        hvg_key: str = "highly_variable_genes",
        sample_other: int | None = 20000,
        save: str | None = None,
):
    """
    数据库版 Seurat-like HVG 可视化。

    适配 highly_variable_genes_seurat() 的输出结果。
    直接读取 var 表中的：
        - means
        - dispersions
        - dispersions_norm
        - highly_variable_genes
        - highly_variable_rank

    不扫描 X_CSRO_data。
    """

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from datetime import datetime

    print("\n==== highly_variable_genes_plot_like_seurat ====")
    start = datetime.now()

    conn = atlas.connection

    if conn is None:
        raise ValueError("atlas.connection 为空，请先连接数据库")

    # -------------------------------------------------
    # 0. DuckDB 字段安全引用
    # -------------------------------------------------
    def _q(name: str) -> str:
        return '"' + name.replace('"', '""') + '"'

    # -------------------------------------------------
    # 1. 检查 var 中列是否存在
    # -------------------------------------------------
    var_cols = [
        r[0]
        for r in conn.execute("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'var'
        """).fetchall()
    ]

    needed = [
        "atlas_gene_id",
        "atlas_gene_name",
        hvg_key,
        "means",
        "dispersions",
        "dispersions_norm",
    ]

    missing = [c for c in needed if c not in var_cols]

    if missing:
        raise ValueError(
            f"var 中不存在这些列: {missing}\n"
            f"请先运行 highly_variable_genes_seurat(atlas)"
        )

    # -------------------------------------------------
    # 2. 读取 var 中已经保存好的 HVG 结果
    # -------------------------------------------------
    df = conn.execute(f"""
        SELECT
            atlas_gene_id,
            atlas_gene_name,
            COALESCE({_q(hvg_key)}, FALSE) AS is_hvg,
            means,
            dispersions,
            dispersions_norm
        FROM var
        ORDER BY atlas_gene_id
    """).fetchdf()

    # -------------------------------------------------
    # 3. 清理 nan / inf
    # -------------------------------------------------
    for col in ["means", "dispersions", "dispersions_norm"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    df["is_hvg"] = df["is_hvg"].fillna(False).astype(bool)

    df = df[df["means"].notna()].copy()

    if len(df) == 0:
        raise ValueError(
            "var.means 全为空，无法绘图。请先运行 highly_variable_genes_seurat(atlas)。"
        )

    print(f"[INFO] genes for plot = {len(df):,}")
    print(f"[INFO] HVGs = {int(df['is_hvg'].sum()):,}")

    # -------------------------------------------------
    # 4. 非 HVG 可选抽样
    # -------------------------------------------------
    if sample_other is not None:
        df_hvg = df[df["is_hvg"]].copy()
        df_other = df[~df["is_hvg"]].copy()

        if len(df_other) > sample_other:
            df_other = df_other.sample(
                n=int(sample_other),
                random_state=0,
            )

        plot_df = pd.concat(
            [df_hvg, df_other],
            axis=0,
            ignore_index=True,
        )
    else:
        plot_df = df.copy()

    plot_hvg = plot_df[plot_df["is_hvg"]].copy()
    plot_other = plot_df[~plot_df["is_hvg"]].copy()

    # -------------------------------------------------
    # 5. 画图
    # -------------------------------------------------
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11, 4.5),
        facecolor="white",
    )

    # =================================================
    # 左图：normalized dispersions
    # =================================================
    ax = axes[0]

    other_norm = plot_other[plot_other["dispersions_norm"].notna()]
    hvg_norm = plot_hvg[plot_hvg["dispersions_norm"].notna()]

    if len(other_norm) > 0:
        ax.scatter(
            other_norm["means"].to_numpy(),
            other_norm["dispersions_norm"].to_numpy(),
            s=6,
            c="#9a9a9a",
            alpha=0.55,
            linewidths=0,
            label="other genes",
        )

    if len(hvg_norm) > 0:
        ax.scatter(
            hvg_norm["means"].to_numpy(),
            hvg_norm["dispersions_norm"].to_numpy(),
            s=8,
            c="black",
            alpha=0.9,
            linewidths=0,
            label="highly variable genes",
        )

    ax.set_xlabel("means of genes", fontsize=14)
    ax.set_ylabel("dispersions of genes (normalized)", fontsize=14)

    ax.legend(
        frameon=True,
        fontsize=10,
        markerscale=1.2,
        loc="upper left",
    )

    ax.grid(True, color="#d9d9d9", linewidth=0.8, alpha=0.8)
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=11)

    # =================================================
    # 右图：raw dispersions
    # =================================================
    ax = axes[1]

    other_disp = plot_other[plot_other["dispersions"].notna()]
    hvg_disp = plot_hvg[plot_hvg["dispersions"].notna()]

    if len(other_disp) > 0:
        ax.scatter(
            other_disp["means"].to_numpy(),
            other_disp["dispersions"].to_numpy(),
            s=6,
            c="#9a9a9a",
            alpha=0.55,
            linewidths=0,
        )

    if len(hvg_disp) > 0:
        ax.scatter(
            hvg_disp["means"].to_numpy(),
            hvg_disp["dispersions"].to_numpy(),
            s=8,
            c="black",
            alpha=0.9,
            linewidths=0,
        )

    ax.set_xlabel("means of genes", fontsize=14)
    ax.set_ylabel("dispersions of genes (not normalized)", fontsize=14)

    ax.grid(True, color="#d9d9d9", linewidth=0.8, alpha=0.8)
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=11)

    plt.tight_layout(pad=1.0)

    if save is not None:
        plt.savefig(save, dpi=300, bbox_inches="tight")
        print(f"[INFO] figure saved to: {save}")

    plt.show()

    print(f"Done in {(datetime.now() - start).total_seconds():.2f}s")


