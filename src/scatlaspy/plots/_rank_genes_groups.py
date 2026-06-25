import math
from os import PathLike
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Any
from ..data import Atlas


def rank_genes_groups(
        atlas: Atlas,
        use_table: str = "rank_genes_groups",
        groups: list | None = None,
        n_genes: int = 25,
        score_key: str = "scores",
        gene_label: str = "names",
        ncols: int = 4,
        figsize: tuple | None = None,
        save_path: PathLike[str] | str | None = None,
        show: bool = True,
):
    """绘制每个分组的 top marker gene 排名图。

    该函数读取 ``sap.tl.rank_genes_groups`` 写入的差异基因结果表，按分组展示排名靠前的
    marker genes，并在每个子图中用散点和文字标签展示 ``score_key`` 对应的统计分数。
    它类似 Scanpy 的 ``sc.pl.rank_genes_groups``，适合快速浏览每个 cluster 的候选
    marker gene 排名。

    Parameters
    ----------
    atlas
        Atlas 对象。要求已经连接 DuckDB 数据库，或可以通过 ``atlas.connect("r+")``
        重新连接。数据库中需要包含 ``use_table`` 指定的差异基因结果表。
    use_table
        差异基因结果表名，默认 ``"rank_genes_groups"``。
    groups
        需要展示的分组列表。为 ``None`` 时展示结果表中的全部分组。
    n_genes
        每个分组展示的 top gene 数量。
    score_key
        结果表中用于 y 轴绘图的分数字段名，例如 ``"scores"``。
    gene_label
        结果表中用于显示基因名的字段名，默认 ``"names"``。
    ncols
        子图每行最多显示的列数。
    figsize
        Matplotlib 图像大小。为 ``None`` 时根据分组数量和 ``ncols`` 自动估计。
    save_path
        图片保存路径。为 ``None`` 时不保存。
    show
        是否立即显示图形。为 ``False`` 时关闭当前 figure，适合批量保存。

    Returns
    -------
    None

    Examples
    --------
    计算并绘制默认差异基因排名图::

        sap.tl.rank_genes_groups(atlas, groupby="kmeans")
        sap.pl.rank_genes_groups(atlas)

    从自定义结果表读取，并只展示部分分组::

        sap.pl.rank_genes_groups(
            atlas,
            use_table="rank_genes_groups_kmeans_top100",
            groups=["0", "1", "2"],
            n_genes=10,
        )"""

    conn = atlas.connection
    if conn is None:
        atlas.connect("r+")
        conn = atlas.connection

    # -------------------------------------------------
    # 1. 检查结果表是否存在
    # -------------------------------------------------
    table_exists = conn.execute("""
        SELECT COUNT(*)
        FROM information_schema.tables
        WHERE table_name = ?
    """, [use_table]).fetchone()[0]

    if table_exists == 0:
        raise ValueError(
            f"数据库中不存在结果表: {use_table}。"
            f"请先运行 sap.tl.rank_genes_groups(..., key_added='{use_table}')"
        )

    # -------------------------------------------------
    # 2. 读取结果表
    # -------------------------------------------------
    df = conn.execute(f"""
        SELECT *
        FROM "{use_table}"
    """).fetchdf()

    if len(df) == 0:
        raise ValueError(f"{use_table} 表为空，无法绘图")

    required_cols = {"group", "rank", gene_label, score_key}
    missing = required_cols - set(df.columns)

    if len(missing) > 0:
        raise ValueError(
            f"{use_table} 表缺少必要字段: {missing}。"
            f"当前字段为: {list(df.columns)}"
        )

    # group 统一转字符串，避免 int / str 混乱
    df["group"] = df["group"].astype(str)

    # group 排序函数，能转成数字的按数字排，不能转数字的按字符串排
    def _group_sort_key(x: Any):
        """生成差异基因分组标签的排序键。

        能转换为整数或浮点数的分组按数值排序，其他标签按字符串排序，避免
        ``"10"`` 排在 ``"2"`` 前面。

        Parameters
        ----------
        x
            单个分组标签。

        Returns
        -------
        tuple
            可传给 ``sorted(..., key=...)`` 的排序键。
        """
        try:
            return (0, int(x))
        except Exception:
            try:
                return (0, float(x))
            except Exception:
                return (1, str(x))

    # -------------------------------------------------
    # 3. 过滤 groups
    # -------------------------------------------------
    all_groups = df["group"].dropna().unique().tolist()

    all_groups = sorted(all_groups, key=_group_sort_key)

    if groups is None:
        plot_groups = all_groups
    else:
        wanted = {str(g) for g in groups}
        plot_groups = [g for g in all_groups if g in wanted]

    if len(plot_groups) == 0:
        raise ValueError(
            f"groups 过滤后为空。可用 groups: {all_groups}"
        )

    # -------------------------------------------------
    # 4. 准备画布
    # -------------------------------------------------
    n_panels = len(plot_groups)
    ncols = min(int(ncols), n_panels)
    nrows = math.ceil(n_panels / ncols)

    if figsize is None:
        figsize = (4.2 * ncols, 4.8 * nrows)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=figsize,
        facecolor="white",
        squeeze=False,
    )

    axes = axes.reshape(-1)

    # -------------------------------------------------
    # 5. 逐 group 作图
    # -------------------------------------------------
    for ax, group in zip(axes, plot_groups):

        plot_df = (
            df[df["group"] == group]
            .sort_values("rank")
            .head(int(n_genes))
            .copy()
        )

        if len(plot_df) == 0:
            ax.set_title(f"group {group}")
            ax.text(
                0.5,
                0.5,
                "No genes",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_axis_off()
            continue

        x = np.arange(len(plot_df))
        y = plot_df[score_key].to_numpy(dtype=float)

        ax.scatter(
            x,
            y,
            s=18,
            linewidths=0,
            alpha=0.9,
        )

        for xi, yi, gene in zip(x, y, plot_df[gene_label]):
            ax.text(
                xi,
                yi,
                str(gene),
                rotation=90,
                fontsize=9,
                ha="center",
                va="bottom",
            )

        # 标题尽量使用 group vs reference
        if "reference" in plot_df.columns:
            ref = str(plot_df["reference"].iloc[0])
            title = f"{group} vs. {ref}"
        else:
            title = f"group {group}"

        ax.set_title(title, fontsize=16)
        ax.set_xlabel("ranking", fontsize=13)
        ax.set_ylabel(score_key, fontsize=13)

        ax.tick_params(axis="both", labelsize=10)
        ax.set_facecolor("white")
        ax.grid(False)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # 多余 panel 关掉
    for ax in axes[n_panels:]:
        ax.set_axis_off()

    plt.tight_layout()

    # -------------------------------------------------
    # 6. 保存 / 显示
    # -------------------------------------------------
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return None


def rank_genes_groups_volcano(
        atlas: Atlas,
        use_table: str = "rank_genes_groups",
        group: str | int = "0",
        lfc_key: str = "logfoldchanges",
        pval_key: str = "pvals_adj",
        gene_label: str = "names",
        pval_cutoff: float = 0.05,
        logfc_cutoff: float = 1.0,
        top_n: int = 8,
        figsize: tuple = (12, 10),
        y_cap: float |  None = None,
        xlim_abs: float | None = None,
        save_path: PathLike[str] | str | None = None,
        show: bool = True,
        label_fontsize: int = 7,
        label_offset_step: int = 12,
):
    """绘制单个分组的差异基因火山图。

    该函数从差异基因结果表中读取指定 ``group`` 的结果，根据 log fold change 和校正p 值绘制火山图。
    显著上调基因显示为红色，显著下调基因显示为蓝色，不显著基因显示为
    灰色，并自动标注最显著的一批基因。

    该图用于检查某个 cluster 或细胞类型相对于参考组的 marker genes 是否显著、方向是否清楚，
    以及是否存在极端 logFC 或 p 值。

    Parameters
    ----------
    atlas
        Atlas 对象。要求已经连接 DuckDB 数据库，或可以通过 ``atlas.connect("r+")``
        重新连接。数据库中需要包含 ``use_table`` 指定的差异基因结果表。
    use_table
        读取已有结果的数据库表名。
    group
        需要绘制火山图的分组标签。
    lfc_key
        结果表中保存 log fold change 的字段名。
    pval_key
        结果表中保存校正 p 值的字段名，默认 ``"pvals_adj"``。
    gene_label
        结果表中用于显示基因名的字段名。
    pval_cutoff
        显著性判断的校正 p 值阈值。
    logfc_cutoff
        显著性判断的绝对 log fold change 阈值。
    top_n
        自动标注的基因数量。函数会优先在显著上调和显著下调基因中各取一部分。
    figsize
        图形大小。为 ``None`` 时使用函数默认尺寸。
    y_cap
        y 轴 ``-log10(padj)`` 的显示截断上限。为 ``None`` 时不截断。
    xlim_abs
        x 轴左右对称显示范围的绝对值。为 ``None`` 时根据 logFC 分布自动估计。
    save_path
        图片保存路径。为 ``None`` 时只显示或返回图对象。
    show
        是否立即显示图形。为 ``None`` 时遵循 Matplotlib 当前行为。
    label_fontsize
        自动标注基因名的字体大小。
    label_offset_step
        自动标注基因名时用于错开标签的偏移强度。

    Returns
    -------
    None

    Examples
    --------
    绘制默认差异基因火山图::

        sap.tl.rank_genes_groups(atlas, groupby="kmeans")
        sap.pl.rank_genes_groups_volcano(atlas)

    指定分组、top genes 和保存路径::

        sap.pl.rank_genes_groups_volcano(
            atlas,
            group="0",
            top_n=20,
            save_path=r"F:\\figures\\rank_volcano.png",
        )"""

    conn = atlas.connection
    if conn is None:
        atlas.connect("r+")
        conn = atlas.connection

    # 1. 检查结果表
    table_exists = conn.execute("""
        SELECT COUNT(*)
        FROM information_schema.tables
        WHERE table_name = ?
    """, [use_table]).fetchone()[0]

    if table_exists == 0:
        raise ValueError(
            f"数据库中不存在结果表: {use_table}。"
            f"请先运行 sap.tl.rank_genes_groups(..., key_added='{use_table}')"
        )

    # 2. 读取指定 group 的结果
    df = conn.execute(f"""
        SELECT *
        FROM "{use_table}"
        WHERE CAST("group" AS TEXT) = CAST(? AS TEXT)
    """, [str(group)]).fetchdf()

    if len(df) == 0:
        raise ValueError(f"{use_table} 表中没有 group={group} 的结果")

    required_cols = {lfc_key, pval_key, gene_label}
    missing = required_cols - set(df.columns)

    if len(missing) > 0:
        raise ValueError(
            f"{use_table} 表缺少火山图必要字段: {missing}。"
            f"当前字段为: {list(df.columns)}"
        )

    # 3. 清理异常值
    df = df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=[lfc_key, pval_key, gene_label]
    ).copy()

    if len(df) == 0:
        raise ValueError("清理 NA / inf 后没有可绘制数据")

    # 确保数值列是 float
    df[lfc_key] = df[lfc_key].astype(float)
    df[pval_key] = df[pval_key].astype(float)

    # 4. 计算 -log10(padj)
    # 先用极小值避免 log10(0)，再用 y_cap 做显示截断
    tiny = np.nextafter(0, 1)
    df["neg_log10_padj"] = -np.log10(
        df[pval_key].clip(lower=tiny)
    )

    if y_cap is not None:
        df["neg_log10_padj_plot"] = df["neg_log10_padj"].clip(upper=float(y_cap))
    else:
        df["neg_log10_padj_plot"] = df["neg_log10_padj"]

    y_max_real = float(np.nanmax(df["neg_log10_padj_plot"]))

    if y_cap is not None:
        y_upper = min(float(y_cap), y_max_real * 1.15 + 1.0)
    else:
        y_upper = y_max_real * 1.15 + 1.0

    # 至少给一点显示空间
    y_upper = max(y_upper, 5.0)

    # 5. 显著性分组
    df["significant"] = (
        (df[pval_key] < float(pval_cutoff))
        & (df[lfc_key].abs() >= float(logfc_cutoff))
    )

    # 颜色对齐 Scanpy 图风格：上调红，下调蓝，不显著灰
    colors = np.where(
        df["significant"] & (df[lfc_key] > 0),
        "#d62728",
        np.where(
            df["significant"] & (df[lfc_key] < 0),
            "#1f77b4",
            "#9a9a9a",
        ),
    )

    # 6. 作图
    fig, ax = plt.subplots(figsize=figsize, facecolor="white")

    ax.scatter(
        df[lfc_key],
        df["neg_log10_padj_plot"],
        c=colors,
        s=8,
        alpha=0.75,
        linewidths=0,
    )

    # 阈值线
    ax.axvline(-float(logfc_cutoff), color="#555555", linestyle="--", linewidth=1.0)
    ax.axvline(float(logfc_cutoff), color="#555555", linestyle="--", linewidth=1.0)
    ax.axhline(-np.log10(float(pval_cutoff)), color="#555555", linestyle="--", linewidth=1.0)

    # 7. 标注 top genes
    n_up = max(1, int(top_n) // 2)
    n_down = max(1, int(top_n) - n_up)

    top_up = (
        df[df["significant"] & (df[lfc_key] > 0)]
        .sort_values([pval_key, lfc_key], ascending=[True, False])
        .head(n_up)
    )

    top_down = (
        df[df["significant"] & (df[lfc_key] < 0)]
        .sort_values([pval_key, lfc_key], ascending=[True, True])
        .head(n_down)
    )

    top = pd.concat([top_up, top_down], axis=0)

    if len(top) == 0:
        top = (
            df.sort_values([pval_key, lfc_key], ascending=[True, False])
            .head(int(top_n))
        )

    for j, (_, row) in enumerate(top.iterrows()):

        x0 = float(row[lfc_key])
        y0 = float(row["neg_log10_padj_plot"])

        # 标签不要顶到图框
        y0 = min(y0, y_upper * 0.94)

        # 上下错开一点，避免多个标签压在一起
        y_text = y0 - (j % 4) * (y_upper * (label_offset_step / 500.0))
        y_text = max(y_text, 0.1)

        if x0 >= 0:
            x_text = x0 + 0.08
            ha = "left"
        else:
            x_text = x0 - 0.08
            ha = "right"

        ax.text(
            x_text,
            y_text,
            str(row[gene_label]),
            fontsize=label_fontsize,
            ha=ha,
            va="bottom",
            clip_on=True,
        )

    # 8. 标题
    if "reference" in df.columns:
        ref = str(df["reference"].iloc[0])
        title = f"group {group} vs {ref}"
        xlabel = f"log2 fold change: cluster {group} vs {ref}"
    else:
        title = f"group {group}"
        xlabel = "log2 fold change"

    ax.set_title(title, fontsize=20, pad=14)
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel("-log10 adjusted p-value", fontsize=16)

    ax.tick_params(axis="both", labelsize=13)

    # 只保留左/下边框
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)

    # y 轴使用自适应上限
    ax.set_ylim(-1, y_upper)

    # x 轴对称显示，视觉上更像标准火山图
    finite_lfc = (
        df[lfc_key]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .abs()
    )

    if xlim_abs is None:
        if len(finite_lfc) > 0:
            # 用 99.5 分位，避免极端离群点把图拉得过宽
            xlim_abs_use = float(np.nanpercentile(finite_lfc, 99.5))
            xlim_abs_use = max(xlim_abs_use, float(logfc_cutoff) * 2.5)
        else:
            xlim_abs_use = 5.0
    else:
        xlim_abs_use = float(xlim_abs)

    ax.set_xlim(-xlim_abs_use * 1.05, xlim_abs_use * 1.05)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return None


def rank_genes_groups_violin(
        atlas: Atlas,
        group: str = 0 ,
        use_table: str = "rank_genes_groups",
        groupby: str = "kmeans",
        reference: str | int | None = None,
        genes: list[str] | None = None,
        n_genes: int = 8,
        use_expr_field: str = "data_log1p",
        sample_cells_per_group: int = 2000,
        save_path: PathLike[str] | str | None = None
):

    """绘制 marker genes 在不同分组中的表达 violin 图。

    该函数基于 ``rank_genes_groups`` 结果自动选择指定 ``group`` 的 top marker genes，
    也可以通过 ``genes`` 手动指定基因列表。函数会从 ``X_HyS_data`` 的
    ``use_expr_field`` 字段读取表达值，按 ``obs[groupby]`` 分组绘制 violin 图，
    用于检查候选 marker 是否在目标分组中特异表达。

    Parameters
    ----------
    atlas
        Atlas 对象。要求已经连接 DuckDB 数据库，并包含 ``obs``、``var``、
        ``X_HyS_data`` 和 ``use_table`` 指定的差异基因结果表。
    group
        需要展示 marker genes 的目标分组标签。
    use_table
        读取已有结果的数据库表名。
    groupby
        ``obs`` 中的分组列名，例如 ``"kmeans"``、``"leiden"`` 或 ``"cell_type"``。
    reference
        差异分析参考组。为 ``None`` 时使用结果表中匹配 ``group`` 的默认结果；
        传入值时会优先筛选对应 ``reference``。
    genes
        手动指定需要展示的基因名称列表。为 ``None`` 时从差异基因结果表按排名选择
        ``n_genes`` 个基因。
    n_genes
        当 ``genes`` 为 ``None`` 时，自动选择的 top marker gene 数量。
    use_expr_field
        从 ``X_HyS_data`` 读取的表达值字段，例如 ``"data_log1p"`` 或 ``"data_count"``。
    sample_cells_per_group
        每个 ``groupby`` 分组最多抽样用于绘图的细胞数量。为 ``None`` 时使用全部细胞。
    save_path
        图片保存路径。为 ``None`` 时只显示或返回图对象。

    Returns
    -------
    None

    Examples
    --------
    绘制每组 top marker 的 violin 图::

        sap.tl.rank_genes_groups(atlas, groupby="kmeans")
        sap.pl.rank_genes_groups_violin(atlas)

    指定分组和表达字段::

        sap.pl.rank_genes_groups_violin(
            atlas,
            group="0",
            n_genes=5,
            use_expr_field="data_log1p",
        )"""

    conn = atlas.connection

    # 检查列
    obs_cols = [r[1] for r in conn.execute("PRAGMA table_info(obs)").fetchall()]
    var_cols = [r[1] for r in conn.execute("PRAGMA table_info(var)").fetchall()]
    x_cols = [r[1] for r in conn.execute("PRAGMA table_info(X_HyS_data)").fetchall()]

    if groupby not in obs_cols:
        raise ValueError(f"obs 中不存在列: {groupby}")
    if "atlas_cell_id" not in obs_cols:
        raise ValueError("obs 中不存在 atlas_cell_id")
    if "atlas_gene_id" not in var_cols:
        raise ValueError("var 中不存在 atlas_gene_id")
    if use_expr_field not in x_cols:
        raise ValueError(f"X_HyS_data 中不存在字段: {use_expr_field}")

    table_exists = conn.execute("""
        SELECT COUNT(*)
        FROM information_schema.tables
        WHERE table_name = ?
    """, [use_table]).fetchone()[0]

    if table_exists == 0:
        raise ValueError(
            f"数据库中不存在结果表: {use_table}。"
            f"请先运行 sap.tl.rank_genes_groups(..., key_added='{use_table}')"
        )

    # =====================================================
    # 1. gene 列表
    # =====================================================
    if genes is None:
        rank_df = conn.execute(f"""
            SELECT
                atlas_gene_id,
                names AS atlas_gene_name,
                rank
            FROM "{use_table}"
            WHERE CAST("group" AS TEXT) = CAST(? AS TEXT)
            ORDER BY rank
            LIMIT {int(n_genes)}
        """, [str(group)]).fetchdf()

        if len(rank_df) == 0:
            raise ValueError(f"{use_table} 表中找不到 group={group} 的结果")

        genes = rank_df["atlas_gene_name"].astype(str).tolist()

        gene_map_df = rank_df[["atlas_gene_id", "atlas_gene_name"]].drop_duplicates()

    else:
        if isinstance(genes, str):
            genes = [genes]

        if len(genes) == 0:
            raise ValueError("genes 为空")

        # gene_name -> gene_id
        gene_name_sql = ", ".join([f"'{g}'" for g in genes])

        if "atlas_gene_name" in var_cols:
            gene_name_col = "atlas_gene_name"
        elif "gene_name" in var_cols:
            gene_name_col = "gene_name"
        else:
            raise ValueError("var 中不存在 atlas_gene_name 或 gene_name，无法按基因名查找")

        gene_map_df = conn.execute(f"""
            SELECT
                atlas_gene_id,
                {gene_name_col} AS atlas_gene_name
            FROM var
            WHERE {gene_name_col} IN ({gene_name_sql})
        """).fetchdf()

        if len(gene_map_df) == 0:
            raise ValueError("var 中找不到这些基因")

        gene_map = dict(zip(gene_map_df["atlas_gene_name"], gene_map_df["atlas_gene_id"]))

        missing_genes = [g for g in genes if g not in gene_map]
        if missing_genes:
            raise ValueError(f"var 中找不到这些基因: {missing_genes}")

    if len(genes) == 0:
        raise ValueError("genes 为空")

    # =====================================================
    # 2. 抽样目标细胞
    # =====================================================
    group_sql = f"'{group}'" if isinstance(group, str) else str(group)

    if reference is None:
        ref_from_result = conn.execute(f"""
            SELECT DISTINCT reference
            FROM "{use_table}"
            WHERE CAST("group" AS TEXT) = CAST(? AS TEXT)
            LIMIT 1
        """, [str(group)]).fetchone()

        if ref_from_result is not None:
            reference_in_result = str(ref_from_result[0])
        else:
            reference_in_result = "rest"

        if reference_in_result == "rest":
            # group vs rest
            group_cells_df = conn.execute(f"""
                SELECT atlas_cell_id
                FROM obs
                WHERE CAST({groupby} AS TEXT) = CAST({group_sql} AS TEXT)
                ORDER BY random()
                LIMIT {sample_cells_per_group}
            """).fetchdf()

            rest_cells_df = conn.execute(f"""
                SELECT atlas_cell_id
                FROM obs
                WHERE {groupby} IS NOT NULL
                  AND CAST({groupby} AS TEXT) != CAST({group_sql} AS TEXT)
                ORDER BY random()
                LIMIT {sample_cells_per_group}
            """).fetchdf()

            ref_label = "rest"

        else:
            # 如果结果表里是 group vs 某个 reference
            ref_sql = f"'{reference_in_result}'"

            group_cells_df = conn.execute(f"""
                SELECT atlas_cell_id
                FROM obs
                WHERE CAST({groupby} AS TEXT) = CAST({group_sql} AS TEXT)
                ORDER BY random()
                LIMIT {sample_cells_per_group}
            """).fetchdf()

            rest_cells_df = conn.execute(f"""
                SELECT atlas_cell_id
                FROM obs
                WHERE CAST({groupby} AS TEXT) = CAST({ref_sql} AS TEXT)
                ORDER BY random()
                LIMIT {sample_cells_per_group}
            """).fetchdf()

            ref_label = reference_in_result

    else:
        ref_sql = f"'{reference}'" if isinstance(reference, str) else str(reference)

        group_cells_df = conn.execute(f"""
            SELECT atlas_cell_id
            FROM obs
            WHERE CAST({groupby} AS TEXT) = CAST({group_sql} AS TEXT)
            ORDER BY random()
            LIMIT {sample_cells_per_group}
        """).fetchdf()

        rest_cells_df = conn.execute(f"""
            SELECT atlas_cell_id
            FROM obs
            WHERE CAST({groupby} AS TEXT) = CAST({ref_sql} AS TEXT)
            ORDER BY random()
            LIMIT {sample_cells_per_group}
        """).fetchdf()

        ref_label = str(reference)

    if len(group_cells_df) == 0:
        raise ValueError(f"group={group} 没有细胞")
    if len(rest_cells_df) == 0:
        raise ValueError(f"reference/rest 没有细胞")

    group_cells_df["group_label"] = str(group)
    rest_cells_df["group_label"] = ref_label

    cells_df = pd.concat([group_cells_df, rest_cells_df], ignore_index=True)

    # =====================================================
    # 3. 注册采样细胞和基因
    # =====================================================
    conn.register("_violin_cells_tmp", cells_df)
    conn.register("_violin_genes_tmp", gene_map_df)

    # 取表达长表（含隐式 0）
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

    # gene 顺序保持传入顺序
    plot_df["gene"] = pd.Categorical(plot_df["gene"], categories=genes, ordered=True)
    plot_df = plot_df.sort_values(["gene", "group_label"]).reset_index(drop=True)

    # =====================================================
    # 4. 作图
    # =====================================================
    fig, ax = plt.subplots(figsize=(1.25 * len(genes) + 2.5, 6), facecolor="white")

    labels = [str(group), ref_label]
    positions = np.arange(len(genes))

    width = 0.36
    pos_left = positions - width / 2
    pos_right = positions + width / 2

    color_map = {
        str(group): "#1f77b4",   # 蓝
        ref_label: "#ff7f0e"     # 橙
    }

    for idx, gene in enumerate(genes):
        sub = plot_df[plot_df["gene"] == gene]

        vals_group = sub[sub["group_label"] == str(group)]["expr"].values
        vals_ref = sub[sub["group_label"] == ref_label]["expr"].values

        # violin 1: group
        if len(vals_group) > 0:
            vp1 = ax.violinplot(
                [vals_group],
                positions=[pos_left[idx]],
                widths=width,
                showmeans=False,
                showmedians=True,
                showextrema=False
            )
            for body in vp1["bodies"]:
                body.set_facecolor(color_map[str(group)])
                body.set_edgecolor("black")
                body.set_alpha(0.85)
            if "cmedians" in vp1:
                vp1["cmedians"].set_color("black")
                vp1["cmedians"].set_linewidth(1.0)

        # violin 2: ref/rest
        if len(vals_ref) > 0:
            vp2 = ax.violinplot(
                [vals_ref],
                positions=[pos_right[idx]],
                widths=width,
                showmeans=False,
                showmedians=True,
                showextrema=False
            )
            for body in vp2["bodies"]:
                body.set_facecolor(color_map[ref_label])
                body.set_edgecolor("black")
                body.set_alpha(0.85)
            if "cmedians" in vp2:
                vp2["cmedians"].set_color("black")
                vp2["cmedians"].set_linewidth(1.0)

        # 少量散点增强可读性（抽样后再小抽一点）
        n_dot = min(250, len(vals_group))
        if n_dot > 0:
            dot_idx = np.random.choice(len(vals_group), size=n_dot, replace=False)
            jitter = (np.random.rand(n_dot) - 0.5) * 0.08
            ax.scatter(
                np.full(n_dot, pos_left[idx]) + jitter,
                vals_group[dot_idx],
                s=3,
                c="black",
                alpha=0.6,
                linewidths=0
            )

        n_dot = min(250, len(vals_ref))
        if n_dot > 0:
            dot_idx = np.random.choice(len(vals_ref), size=n_dot, replace=False)
            jitter = (np.random.rand(n_dot) - 0.5) * 0.08
            ax.scatter(
                np.full(n_dot, pos_right[idx]) + jitter,
                vals_ref[dot_idx],
                s=3,
                c="black",
                alpha=0.6,
                linewidths=0
            )

    # 美化
    ax.set_title(f"{group} vs. {ref_label}", fontsize=20, pad=10)
    ax.set_xlabel("genes", fontsize=16)
    ax.set_ylabel("expression", fontsize=16)

    ax.set_xticks(positions)
    ax.set_xticklabels(genes, rotation=90, fontsize=11)

    ax.set_facecolor("white")
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.tick_params(axis="both", labelsize=11, width=1.0, length=4)

    # 图例
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=color_map[str(group)], edgecolor="black", label=str(group)),
        Patch(facecolor=color_map[ref_label], edgecolor="black", label=ref_label),
    ]
    ax.legend(handles=legend_handles, frameon=False, loc="upper right", fontsize=11)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()
