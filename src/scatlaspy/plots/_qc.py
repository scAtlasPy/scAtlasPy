from ..data import Atlas
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from datetime import datetime


# 最高表达基因占比图（highest expressed genes）: 用来检查 有没有少数基因“垄断”表达（技术偏差）
def highest_expr_genes(
        atlas: Atlas,
        n_top: int = 20,
        use_all_cells: bool = True,
        show_outliers: bool = True,    # 是否绘制离群点
        max_outliers: int = 5000,       # 每个基因最多绘制多少个离群点
        figsize: tuple[float, float] | None=(12, 10),
        approx_quantile: bool = True,   # 大数据默认用近似分位数，避免 OOM
        sample_cells: int | None = None # 大数据绘图时可抽样细胞，适合 1e8 cells 小内存场景
):

    """绘制最高表达基因统计图。

    该函数从表达矩阵中统计每个基因的表达贡献，展示 top genes 的表达量或比例。

    它常用于导入后 QC，帮助识别线粒体、核糖体或其他可能主导总表达量的基因。

    Parameters
    ----------
    atlas
        Atlas 对象。通常要求已经连接数据库，并包含该函数所需的 ``obs``、``var``、``X_HyS_data`` 或
        embedding 结果表。

    n_top
        需要展示的 top 项数量。

    use_all_cells
        是否使用所有细胞计算统计量。

    show_outliers
        是否显示离群点。

    max_outliers
        最多显示的离群点数量。

    figsize
        matplotlib 图像大小。

    approx_quantile
        用于识别离群表达的近似分位数。

    sample_cells
        用于估计统计量的抽样细胞数量。

    Notes
    -----
    绘图前通常需要先运行对应的 ``sap.tl`` 或 ``sap.pp`` 计算步骤，确保结果表和统计列已经存在。

    Examples
    --------
    调用该函数：::

        sap.pl.highest_expr_genes(...)
    """

    start = datetime.now()
    conn = atlas.connection

    try:
        conn.execute(f"PRAGMA threads={os.cpu_count()}")
    except Exception:
        pass

    # 大数据场景下用 hash 近似抽样，避免 ORDER BY RANDOM() 全表排序
    if sample_cells is not None:
        sample_cells = int(sample_cells)
        if sample_cells <= 0:
            sample_cells = None

    # 如果指定 sample_cells，创建一个轻量 TEMP VIEW，不物化随机排序结果
    if sample_cells is not None:
        n_cells_total = conn.execute("SELECT COUNT(*) FROM obs").fetchone()[0]

        if sample_cells < n_cells_total:
            sample_mod = max(1, int(n_cells_total // sample_cells))

            conn.execute(f"""
                CREATE OR REPLACE TEMP VIEW _sample_cells AS
                SELECT atlas_cell_id
                FROM obs
                WHERE (hash(atlas_cell_id) % {sample_mod}) = 0
            """)
        else:
            conn.execute("""
                CREATE OR REPLACE TEMP VIEW _sample_cells AS
                SELECT atlas_cell_id
                FROM obs
            """)

    # 优先复用 obs.cell_total_counts，避免每次扫描整个 X_HyS_data
    obs_cols = {r[1] for r in conn.execute("PRAGMA table_info('obs')").fetchall()}

    if "cell_total_counts" in obs_cols:
        conn.execute("""
            CREATE OR REPLACE TEMP VIEW _cell_total_counts AS 
            SELECT
                atlas_cell_id,
                cell_total_counts AS total_counts
            FROM obs
            WHERE cell_total_counts IS NOT NULL
        """)
    elif "total_counts" in obs_cols:
        conn.execute("""
            CREATE OR REPLACE TEMP VIEW _cell_total_counts AS
            SELECT
                atlas_cell_id,
                total_counts AS total_counts
            FROM obs
            WHERE total_counts IS NOT NULL
        """)
    else:
        # fallback：没有预计算结果时才扫描 X
        conn.execute("""
            CREATE OR REPLACE TEMP TABLE _cell_total_counts AS
            SELECT
                atlas_cell_id,
                SUM(data_count) AS total_counts
            FROM X_HyS_data
            GROUP BY atlas_cell_id
        """)

    # 选 top genes
    if use_all_cells:
        if sample_cells is not None:
            conn.execute("""
                CREATE OR REPLACE TEMP VIEW _all_cells AS
                SELECT atlas_cell_id
                FROM _sample_cells
            """)
        else:
            conn.execute("""
                CREATE OR REPLACE TEMP VIEW _all_cells AS
                SELECT atlas_cell_id
                FROM obs
            """)

        conn.execute("""
            CREATE OR REPLACE TEMP TABLE _all_gene_mean_pct AS
            WITH gene_pct_nonzero AS (
                SELECT
                    x.atlas_cell_id,
                    x.atlas_gene_id,
                    x.data_count * 100.0 / t.total_counts AS pct
                FROM X_HyS_data x
                JOIN _cell_total_counts t
                    ON x.atlas_cell_id = t.atlas_cell_id
                JOIN _all_cells c                      
                    ON x.atlas_cell_id = c.atlas_cell_id
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
        conn.execute(f"""
            CREATE OR REPLACE TEMP TABLE _top_expr_genes AS
            SELECT
                x.atlas_gene_id,
                v.atlas_gene_name,
                AVG(x.data_count * 100.0 / t.total_counts) AS mean_pct
            FROM X_HyS_data x
            JOIN _cell_total_counts t
                ON x.atlas_cell_id = t.atlas_cell_id
            JOIN var v
                ON x.atlas_gene_id = v.atlas_gene_id
            WHERE t.total_counts > 0
            GROUP BY x.atlas_gene_id, v.atlas_gene_name
            ORDER BY mean_pct DESC
            LIMIT {int(n_top)}
        """)

    # SQL 直接计算标准 boxplot 统计量
    qfunc = "approx_quantile" if approx_quantile else "quantile_cont"

    # use_all_cells=False 时，boxplot 统计可基于抽样细胞，避免 1e8 cells 下扫描/聚合过重
    sample_join_sql = ""
    if sample_cells is not None:
        sample_join_sql = """
            JOIN _sample_cells s
                ON x.atlas_cell_id = s.atlas_cell_id
        """

    if use_all_cells:
        stats_df = conn.execute(f"""  
            WITH all_cells AS (
                SELECT atlas_cell_id
                FROM _all_cells 
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
                    x.data_count * 100.0 / t.total_counts AS pct
                FROM X_HyS_data x
                JOIN _cell_total_counts t
                    ON x.atlas_cell_id = t.atlas_cell_id
                JOIN _top_expr_genes g
                    ON x.atlas_gene_id = g.atlas_gene_id
                JOIN all_cells c           
                    ON x.atlas_cell_id = c.atlas_cell_id
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
                    {qfunc}(pct, 0.25) AS q1,     
                    {qfunc}(pct, 0.50) AS median, 
                    {qfunc}(pct, 0.75) AS q3     
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
        # 把 top genes 的非零表达值物化成临时表,这样后面 quartiles / whiskers 不会反复重跑同一个大 CTE
        stats_df = conn.execute(f"""
                WITH quartiles AS (
                    SELECT
                        g.atlas_gene_id,
                        g.atlas_gene_name,
                        MAX(g.mean_pct) AS mean_pct,
                        COUNT(*) AS n,
                        {qfunc}(CAST(x.data_count * 100.0 / t.total_counts AS DOUBLE), 0.25) AS q1,
                        {qfunc}(CAST(x.data_count * 100.0 / t.total_counts AS DOUBLE), 0.50) AS median,
                        {qfunc}(CAST(x.data_count * 100.0 / t.total_counts AS DOUBLE), 0.75) AS q3
                    FROM X_HyS_data x
                    {sample_join_sql}                 -- 如果 sample_cells 不为空，则只对抽样细胞计算 boxplot
                    JOIN _top_expr_genes g
                        ON x.atlas_gene_id = g.atlas_gene_id
                    JOIN _cell_total_counts t
                        ON x.atlas_cell_id = t.atlas_cell_id
                    WHERE t.total_counts > 0
                    GROUP BY g.atlas_gene_id, g.atlas_gene_name
                ),
                whiskers AS (
                    SELECT
                        q.atlas_gene_name,
                        MIN(CASE
                                WHEN CAST(x.data_count * 100.0 / t.total_counts AS DOUBLE)
                                     >= (q.q1 - 1.5 * (q.q3 - q.q1))
                                THEN CAST(x.data_count * 100.0 / t.total_counts AS DOUBLE)
                            END) AS whisker_low,
                        MAX(CASE
                                WHEN CAST(x.data_count * 100.0 / t.total_counts AS DOUBLE)
                                     <= (q.q3 + 1.5 * (q.q3 - q.q1))
                                THEN CAST(x.data_count * 100.0 / t.total_counts AS DOUBLE)
                            END) AS whisker_high
                    FROM X_HyS_data x
                    {sample_join_sql}                 -- 如果 sample_cells 不为空，则只对抽样细胞计算 whisker
                    JOIN _top_expr_genes g
                        ON x.atlas_gene_id = g.atlas_gene_id
                    JOIN _cell_total_counts t
                        ON x.atlas_cell_id = t.atlas_cell_id
                    JOIN quartiles q
                        ON g.atlas_gene_id = q.atlas_gene_id
                    WHERE t.total_counts > 0
                    GROUP BY q.atlas_gene_name
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

    # 提取离群点
    outlier_df = None

    if show_outliers:

        if use_all_cells:
            outlier_df = conn.execute(f"""  -- 改成 f-string，支持 qfunc
                WITH all_cells AS (
                    SELECT atlas_cell_id
                    FROM _all_cells              -- 从 _all_cells 读取；如果 sample_cells 不为空则自动使用抽样细胞
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
                        x.data_count * 100.0 / t.total_counts AS pct
                    FROM X_HyS_data x
                    JOIN _cell_total_counts t
                        ON x.atlas_cell_id = t.atlas_cell_id
                    JOIN _top_expr_genes g
                        ON x.atlas_gene_id = g.atlas_gene_id
                    JOIN all_cells c              -- 限制到 _all_cells，支持 sample_cells
                        ON x.atlas_cell_id = c.atlas_cell_id
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
                        {qfunc}(pct, 0.25) AS q1,  -- quantile_cont 改为可切换 qfunc
                        {qfunc}(pct, 0.75) AS q3   -- quantile_cont 改为可切换 qfunc
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
                        x.data_count * 100.0 / t.total_counts AS pct
                    FROM X_HyS_data x
                    {sample_join_sql}                 -- 如果 sample_cells 不为空，则只提取抽样细胞离群点
                    JOIN _cell_total_counts t
                        ON x.atlas_cell_id = t.atlas_cell_id
                    JOIN _top_expr_genes g
                        ON x.atlas_gene_id = g.atlas_gene_id
                    WHERE t.total_counts > 0
                ),
                bounds AS (
                    SELECT
                        atlas_gene_name,
                        {qfunc}(pct, 0.25) AS q1, 
                        {qfunc}(pct, 0.75) AS q3   
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

    # 清理
    # DuckDB 中同名对象可能是 VIEW，也可能是 TABLE，直接 DROP VIEW 可能因类型不匹配报错
    def _safe_drop_temp(name: str):
        """清理当前步骤产生的临时资源。

        该内部函数属于QC 可视化模块，用于支撑同一模块中的公共 API。

        读取 QC 指标和表达矩阵，绘制最高表达基因、violin、scatter 和 HVG 诊断图。

        它通常不会作为用户入口直接调用；直接调用时需要保证输入对象、数据库连接和相关临时表已经由上游步骤准备好。

        Parameters
        ----------
        name
            对象名称、列名或 SQL 标识符，具体含义由调用位置决定。

        Notes
        -----
        这是内部 helper；除非需要扩展 scAtlasPy 内部流程，一般不建议在用户代码中直接调用。
        """
        try:
            conn.execute(f"DROP VIEW IF EXISTS {name}")
        except Exception:
            pass

        try:
            conn.execute(f"DROP TABLE IF EXISTS {name}")
        except Exception:
            pass

    _safe_drop_temp("_sample_cells")
    _safe_drop_temp("_cell_total_counts")
    _safe_drop_temp("_all_cells")
    _safe_drop_temp("_top_expr_genes")
    _safe_drop_temp("_top_gene_values")
    _safe_drop_temp("_all_gene_mean_pct")


# 可视化QC指标 , 画 QC 小提琴图
def violin_qc_metrics(
        atlas: Atlas,
        keys: str | list[str]=None,
        jitter: float = 0.4,
        multi_panel: bool = True,
        figsize: tuple[float, float] | None=None,
        use_filtered: bool = False,
        filter_key: str = "filter_cells",
        sample_n: int | None = 50000,
        random_state: int = 0
):

    """绘制 QC 指标小提琴图。

    该函数从 ``obs`` 中读取一个或多个 QC 指标，并绘制分布图，可选择抽样、过滤细胞和多面板布局。

    常用于检查 ``total_counts``、``n_genes_by_counts``、线粒体比例等指标的分布。

    Parameters
    ----------
    atlas
        Atlas 对象。通常要求已经连接数据库，并包含该函数所需的 ``obs``、``var``、``X_HyS_data`` 或
        embedding 结果表。

    keys
        需要绘制的 QC 指标列名。

    jitter
        小提琴图中散点抖动强度。

    multi_panel
        是否为多个指标使用多面板布局。

    figsize
        matplotlib 图像大小。

    use_filtered
        是否只使用通过过滤的细胞或基因。

    filter_key
        表示过滤状态的列名。

    sample_n
        抽样细胞数量；为 ``None`` 时通常使用全部可用细胞。

    random_state
        随机种子；固定整数可以提高结果复现性。

    Notes
    -----
    绘图前通常需要先运行对应的 ``sap.tl`` 或 ``sap.pp`` 计算步骤，确保结果表和统计列已经存在。

    Examples
    --------
    调用该函数：::

        sap.pl.violin_qc_metrics(...)
    """

    start = datetime.now()
    conn = atlas.connection

    if keys is None:
        keys = ["n_genes_by_counts", "cell_total_counts", "pct_counts_mt"]

    # 检查 obs 列是否存在
    obs_cols = [r[1] for r in conn.execute("PRAGMA table_info(obs)").fetchall()]

    missing = [k for k in keys if k not in obs_cols]
    if missing:
        raise ValueError(
            f"obs 中不存在这些列: {missing}\n"
            f"请先运行 sap.pp.calculate_qc_metrics(atlas)"
        )

    if use_filtered and filter_key not in obs_cols:
        raise ValueError(f"obs 中不存在过滤列: {filter_key}")

    # SQL 先抽样，再 fetchdf
    select_cols = ", ".join(keys)

    where_clauses = []
    if use_filtered:
        where_clauses.append(f"{filter_key} = TRUE")

    # 只保留至少有一个 key 非空的行（减少无效点）
    non_null_cond = " OR ".join([f"{k} IS NOT NULL" for k in keys])
    where_clauses.append(f"({non_null_cond})")

    where_sql = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""

    if sample_n is None:
        # 全量（大数据不推荐）
        query = f"""
            SELECT {select_cols}
            FROM obs
            {where_sql}
        """
    else:
        # 先在 SQL 中抽样，再 fetchdf
        query = f"""
            SELECT {select_cols}
            FROM obs
            {where_sql}
            USING SAMPLE {int(sample_n)} ROWS
        """

    df = conn.execute(query).fetchdf()

    # 清理列
    keep_keys = []
    for k in keys:
        if k in df.columns and df[k].notna().sum() > 0:
            keep_keys.append(k)

    if len(keep_keys) == 0:
        raise ValueError("没有可绘制的 QC 列（全部为空）")

    df = df[keep_keys]
    keys = keep_keys

    # 自动布局
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

    # 逐列画 violin + jitter
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


# QC 散点图（scatter plot），用来发现“异常细胞”的关系图
def scatter_qc_metrics(
        atlas: Atlas,
        pairs: list[tuple[str, str]] | None=None,
        figsize: tuple[float, float] | None=(10, 4),
        use_filtered: bool = False,
        filter_key: str = "filter_cells",
        sample_n: int | None = 50000,
        point_size: float = 8,
        alpha: float = 0.7
):

    """绘制 QC 指标散点图。

    该函数读取 ``obs`` 中的 QC 指标对，例如 total counts 与检测基因数，并用散点图展示关系。

    它适合辅助决定细胞过滤阈值，发现低质量细胞、双细胞或异常测序深度。

    Parameters
    ----------
    atlas
        Atlas 对象。通常要求已经连接数据库，并包含该函数所需的 ``obs``、``var``、``X_HyS_data`` 或
        embedding 结果表。

    pairs
        scatter QC 图中需要绘制的指标对。

    figsize
        matplotlib 图像大小。

    use_filtered
        是否只使用通过过滤的细胞或基因。

    filter_key
        表示过滤状态的列名。

    sample_n
        抽样细胞数量；为 ``None`` 时通常使用全部可用细胞。

    random_state
        随机种子；固定整数可以提高结果复现性。

    point_size
        散点大小。

    alpha
        绘图透明度。

    Notes
    -----
    绘图前通常需要先运行对应的 ``sap.tl`` 或 ``sap.pp`` 计算步骤，确保结果表和统计列已经存在。

    Examples
    --------
    调用该函数：::

        sap.pl.scatter_qc_metrics(...)
    """

    start = datetime.now()
    conn = atlas.connection

    if pairs is None:
        pairs = [
            ("cell_total_counts", "pct_counts_mt"),
            ("cell_total_counts", "n_genes_by_counts"),
        ]

    # 检查 obs 列是否存在
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

    # SQL 先抽样，再 fetchdf
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
        # SQL 先抽样
        query = f"""
            SELECT {select_cols}
            FROM obs
            {where_sql}
            USING SAMPLE {int(sample_n)} ROWS
        """

    df = conn.execute(query).fetchdf()

    # 自动布局
    n = len(pairs)

    fig, axes = plt.subplots(1, n, figsize=figsize, facecolor="white")
    if n == 1:
        axes = [axes]

    # 逐图绘制
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


