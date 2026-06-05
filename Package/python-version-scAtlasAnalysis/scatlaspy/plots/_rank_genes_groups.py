import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 画 rank_genes_groups 排名图
def rank_genes_groups(
        atlas,
        key: str = "rank_genes_groups",
        groups: list | None = None,
        n_genes: int = 25,
        score_key: str = "scores",
        gene_label: str = "names",
        ncols: int = 4,
        figsize: tuple | None = None,
        save_path: str | None = None,
        show: bool = True,
        return_fig: bool = False,
):
    """绘制差异表达基因排名图。

    该函数读取 ``sap.tl.rank_genes_groups`` 生成的结果表，按 group 展示 top marker genes 及其得分。

    它用于快速浏览每个 cluster 或细胞类型最具代表性的 marker。

    Parameters
    ----------
    atlas
        Atlas 对象。通常要求已经连接数据库，并包含该函数所需的 ``obs``、``var``、``X_HyS_data`` 或
        embedding 结果表。

    key
        结果键名、表名前缀或 HDF5 group 名称。

    groups
        需要分析或绘制的分组；为 ``None`` 时使用全部分组。

    n_genes
        每个分组保留或绘制的基因数量。

    score_key
        ``var`` 中保存得分的列名。

    gene_label
        绘图时用于显示基因名的列名。

    ncols
        多面板绘图时每行的子图数量。

    figsize
        matplotlib 图像大小。

    save_path
        图像或结果保存路径。

    show
        是否立即显示图像。

    return_fig
        是否返回 matplotlib Figure 对象。

    Returns
    -------
    result
        函数返回结果。具体类型取决于参数设置和内部执行路径。

    Notes
    -----
    绘图前通常需要先运行对应的 ``sap.tl`` 或 ``sap.pp`` 计算步骤，确保结果表和统计列已经存在。

    Examples
    --------
    调用该函数：::

        sap.pl.rank_genes_groups(...)
    """



    print(f"\n==== pl.rank_genes_groups (key={key}) ====")

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
    """, [key]).fetchone()[0]

    if table_exists == 0:
        raise ValueError(
            f"数据库中不存在结果表: {key}。"
            f"请先运行 sap.tl.rank_genes_groups(..., key_added='{key}')"
        )

    # -------------------------------------------------
    # 2. 读取结果表
    # -------------------------------------------------
    df = conn.execute(f"""
        SELECT *
        FROM "{key}"
    """).fetchdf()

    if len(df) == 0:
        raise ValueError(f"{key} 表为空，无法绘图")

    required_cols = {"group", "rank", gene_label, score_key}
    missing = required_cols - set(df.columns)

    if len(missing) > 0:
        raise ValueError(
            f"{key} 表缺少必要字段: {missing}。"
            f"当前字段为: {list(df.columns)}"
        )

    # group 统一转字符串，避免 int / str 混乱
    df["group"] = df["group"].astype(str)

    # ✅ 修改：group 排序函数，能转成数字的按数字排，不能转数字的按字符串排
    def _group_sort_key(x):
        """生成分组或标签的自然排序键。

        该内部函数属于差异表达可视化模块，用于支撑同一模块中的公共 API。

        读取 marker 结果表，绘制排名图、火山图和 marker 表达小提琴图。

        它通常不会作为用户入口直接调用；直接调用时需要保证输入对象、数据库连接和相关临时表已经由上游步骤准备好。

        Parameters
        ----------
        x
            需要排序、格式化或转换的单个输入值。

        Returns
-------
        sort_key
            可用于自然排序的键。

        Notes
        -----
        这是内部 helper；除非需要扩展 scAtlasPy 内部流程，一般不建议在用户代码中直接调用。
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

    # ✅ 修改：按数值顺序排序 group
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

    print(f"-> groups = {plot_groups}")
    print(f"-> n_genes = {n_genes}")
    print(f"-> score_key = {score_key}")

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
        print(f"-> saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    print("pl.rank_genes_groups 完成 ✅")

    if return_fig:
        return fig

    return None



# 绘制火山图
def rank_genes_groups_volcano(
        atlas,
        key: str = "rank_genes_groups",
        group: str | int = "0",
        lfc_key: str = "logfoldchanges",
        pval_key: str = "pvals_adj",
        gene_label: str = "names",
        pval_cutoff: float = 0.05,
        logfc_cutoff: float = 1.0,
        top_n: int = 8,
        figsize: tuple = (12, 10),
        y_cap: float |  None = None,
        xlim_abs: float | None = None,     # ✅ 修改：控制 x 轴是否对称显示
        save_path: str | None = None,
        show: bool = True,
        return_fig: bool = False,
        label_fontsize: int = 7,
        label_offset_step: int = 12,
):
    """绘制差异表达基因火山图。

    该函数读取某个 group 的差异表达结果，用 log fold change 作为横轴，用 p 值显著性作为纵轴，并标注 top genes。

    它适合检查单个 cluster 与参考组之间的显著上调/下调 marker。

    Parameters
    ----------
    atlas
        Atlas 对象。通常要求已经连接数据库，并包含该函数所需的 ``obs``、``var``、``X_HyS_data`` 或
        embedding 结果表。

    key
        结果键名、表名前缀或 HDF5 group 名称。

    group
        需要绘制或分析的单个分组名称。

    lfc_key
        火山图中使用的 log fold change 列名。

    pval_key
        火山图中使用的 p 值列名。

    gene_label
        绘图时用于显示基因名的列名。

    pval_cutoff
        显著性 p 值阈值。

    logfc_cutoff
        log fold change 阈值。

    top_n
        保留、标注或评分时使用的 top 项数量。

    figsize
        matplotlib 图像大小。

    y_cap
        火山图 y 轴上限。

    xlim_abs
        火山图 x 轴绝对范围。

    save_path
        图像或结果保存路径。

    show
        是否立即显示图像。

    return_fig
        是否返回 matplotlib Figure 对象。

    label_fontsize
        基因标签字体大小。

    label_offset_step
        多个基因标签之间的偏移步长。

    Returns
    -------
    result
        函数返回结果。具体类型取决于参数设置和内部执行路径。

    Notes
    -----
    绘图前通常需要先运行对应的 ``sap.tl`` 或 ``sap.pp`` 计算步骤，确保结果表和统计列已经存在。

    Examples
    --------
    调用该函数：::

        sap.pl.rank_genes_groups_volcano(...)
    """

    print(f"\n==== pl.rank_genes_groups_volcano (group={group}) ====")

    conn = atlas.connection
    if conn is None:
        atlas.connect("r+")
        conn = atlas.connection

    # 1. 检查结果表
    table_exists = conn.execute("""
        SELECT COUNT(*)
        FROM information_schema.tables
        WHERE table_name = ?
    """, [key]).fetchone()[0]

    if table_exists == 0:
        raise ValueError(
            f"数据库中不存在结果表: {key}。"
            f"请先运行 sap.tl.rank_genes_groups(..., key_added='{key}')"
        )

    # 2. 读取指定 group 的结果
    df = conn.execute(f"""
        SELECT *
        FROM "{key}"
        WHERE CAST("group" AS TEXT) = CAST(? AS TEXT)
    """, [str(group)]).fetchdf()

    if len(df) == 0:
        raise ValueError(f"{key} 表中没有 group={group} 的结果")

    required_cols = {lfc_key, pval_key, gene_label}
    missing = required_cols - set(df.columns)

    if len(missing) > 0:
        raise ValueError(
            f"{key} 表缺少火山图必要字段: {missing}。"
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

    # ✅ 修改：y 轴自适应，不要无脑拉到 y_cap
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
    # ✅ 修改：左右两侧分别选择 top genes，避免全部堆在同一侧
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

    # ✅ 修改：不用箭头拉到图外，直接在点附近轻微错开
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

    # ✅ 修改：y 轴使用自适应上限
    ax.set_ylim(-1, y_upper)

    # ✅ 修改：x 轴对称显示，视觉上更像标准火山图
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
        print(f"-> saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    print("pl.rank_genes_groups_volcano 完成 ✅")

    if return_fig:
        return fig

    return None



# rank_genes_groups 排名图 对应的 提琴图
def rank_genes_groups_violin(
        atlas,
        group = 0 ,
        key: str = "rank_genes_groups",
        groupby: str = "kmeans",
        reference: str | int | None = None,
        genes: list[str] | None = None,
        n_genes: int = 8,
        use_expr_field: str = "data_log1p",
        sample_n_per_group: int = 2000,
        save_path: str | None = None
):

    """绘制差异表达基因小提琴图。

    该函数选择某个 group 的 top marker genes，并从表达矩阵中读取这些基因在不同分组中的表达分布。

    它用于验证 marker gene 是否真正集中表达于目标 cluster 或细胞类型。

    Parameters
    ----------
    atlas
        Atlas 对象。通常要求已经连接数据库，并包含该函数所需的 ``obs``、``var``、``X_HyS_data`` 或
        embedding 结果表。

    group
        需要绘制或分析的单个分组名称。

    key
        结果键名、表名前缀或 HDF5 group 名称。

    groupby
        ``obs`` 中用于分组的列名。

    reference
        差异表达分析中的参考组，可以是 ``"rest"`` 或某个具体分组。

    genes
        需要绘制或分析的基因名称列表。

    n_genes
        每个分组保留或绘制的基因数量。

    use_expr_field
        绘制 gene feature 或表达分布时读取的 ``X_HyS_data`` 表达字段。

    sample_n_per_group
        每个分组最多抽样的细胞数量。

    save_path
        图像或结果保存路径。

    Returns
    -------
    result
        函数返回结果。具体类型取决于参数设置和内部执行路径。

    Notes
    -----
    绘图前通常需要先运行对应的 ``sap.tl`` 或 ``sap.pp`` 计算步骤，确保结果表和统计列已经存在。

    Examples
    --------
    调用该函数：::

        sap.pl.rank_genes_groups_violin(...)
    """
    print(f"\n==== rank_genes_groups_violin (group={group}, key={key}, reference={reference}) ====")
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

    # ✅ 修改：检查 rank_genes_groups 计算结果表是否存在
    table_exists = conn.execute("""
        SELECT COUNT(*)
        FROM information_schema.tables
        WHERE table_name = ?
    """, [key]).fetchone()[0]

    if table_exists == 0:
        raise ValueError(
            f"数据库中不存在结果表: {key}。"
            f"请先运行 sap.tl.rank_genes_groups(..., key_added='{key}')"
        )

    # =====================================================
    # 1. gene 列表
    # =====================================================
    if genes is None:
        # ✅ 修改：不再依赖 rank_result，而是从 rank_genes_groups 表读取 top genes
        rank_df = conn.execute(f"""
            SELECT
                atlas_gene_id,
                names AS atlas_gene_name,
                rank
            FROM "{key}"
            WHERE CAST("group" AS TEXT) = CAST(? AS TEXT)
            ORDER BY rank
            LIMIT {int(n_genes)}
        """, [str(group)]).fetchdf()

        if len(rank_df) == 0:
            raise ValueError(f"{key} 表中找不到 group={group} 的结果")

        genes = rank_df["atlas_gene_name"].astype(str).tolist()

        # ✅ 修改：直接从结果表得到 gene_id 和 gene_name
        gene_map_df = rank_df[["atlas_gene_id", "atlas_gene_name"]].drop_duplicates()

    else:
        if isinstance(genes, str):
            genes = [genes]

        if len(genes) == 0:
            raise ValueError("genes 为空")

        # gene_name -> gene_id
        gene_name_sql = ", ".join([f"'{g}'" for g in genes])

        # ✅ 修改：兼容 var 里有 atlas_gene_name 或 gene_name 的情况
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

    print(f"-> genes = {genes}")

    # =====================================================
    # 2. 抽样目标细胞
    # =====================================================
    group_sql = f"'{group}'" if isinstance(group, str) else str(group)

    if reference is None:
        # ✅ 修改：如果 reference=None，优先从结果表读取 reference
        ref_from_result = conn.execute(f"""
            SELECT DISTINCT reference
            FROM "{key}"
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
                LIMIT {sample_n_per_group}
            """).fetchdf()

            rest_cells_df = conn.execute(f"""
                SELECT atlas_cell_id
                FROM obs
                WHERE {groupby} IS NOT NULL
                  AND CAST({groupby} AS TEXT) != CAST({group_sql} AS TEXT)
                ORDER BY random()
                LIMIT {sample_n_per_group}
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
                LIMIT {sample_n_per_group}
            """).fetchdf()

            rest_cells_df = conn.execute(f"""
                SELECT atlas_cell_id
                FROM obs
                WHERE CAST({groupby} AS TEXT) = CAST({ref_sql} AS TEXT)
                ORDER BY random()
                LIMIT {sample_n_per_group}
            """).fetchdf()

            ref_label = reference_in_result

    else:
        ref_sql = f"'{reference}'" if isinstance(reference, str) else str(reference)

        group_cells_df = conn.execute(f"""
            SELECT atlas_cell_id
            FROM obs
            WHERE CAST({groupby} AS TEXT) = CAST({group_sql} AS TEXT)
            ORDER BY random()
            LIMIT {sample_n_per_group}
        """).fetchdf()

        rest_cells_df = conn.execute(f"""
            SELECT atlas_cell_id
            FROM obs
            WHERE CAST({groupby} AS TEXT) = CAST({ref_sql} AS TEXT)
            ORDER BY random()
            LIMIT {sample_n_per_group}
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

    return plot_df