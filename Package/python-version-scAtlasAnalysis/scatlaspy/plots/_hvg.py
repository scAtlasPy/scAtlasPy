from ..data import Atlas
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from os import PathLike
from typing import Literal


# 高变基因（HVG, Highly Variable Genes）选择图
def _highly_variable_genes_plot(
        atlas: Atlas,
        hvg_key: str = "highly_variable_genes",
        mean_key: str = "hvg_mean",
        var_key: str = "hvg_var",
        std_key: str = "hvg_std",
        score_key: str = "hvg_score",
        sample_other: int | None = 20000,
        figsize: tuple[float, float] | None = None,
        point_size_hvg: float = 8,
        point_size_other: float = 6,
        alpha_hvg: float = 0.9,
        alpha_other: float = 0.6
):

    """绘制高变基因筛选结果。

    该函数读取 ``var`` 中的均值、方差、标准差、高变得分和 HVG 标记，绘制高变基因诊断散点图。

    它用于检查 HVG 选择是否合理，以及高变基因是否覆盖预期的表达均值范围。

    Parameters
    ----------
    atlas
        Atlas 对象。通常要求已经连接数据库，并包含该函数所需的 ``obs``、``var``、``X_HyS_data`` 或
        embedding 结果表。

    hvg_key
        ``var`` 中表示高变基因的布尔列名。

    mean_key
        ``var`` 中保存基因均值的列名。

    var_key
        ``var`` 中保存基因方差的列名。

    std_key
        ``var`` 中保存基因标准差的列名。

    score_key
        ``var`` 中保存得分的列名。

    sample_other
        从非目标点中抽样用于绘图的数量。

    figsize
        matplotlib 图像大小。

    point_size_hvg
        高变基因点大小。

    point_size_other
        非高变基因点大小。

    alpha_hvg
        高变基因点透明度。

    alpha_other
        非高变基因点透明度。

    Notes
    -----
    绘图前通常需要先运行对应的 ``sap.tl`` 或 ``sap.pp`` 计算步骤，确保结果表和统计列已经存在。

    Examples
    --------
    调用该函数：::

        sap.pl.highly_variable_genes_plot(...)
    """
    print("\n==== highly_variable_genes_plot ====")
    start = datetime.now()
    conn = atlas.connection

    # 检查 var 中列是否存在
    var_cols = [r[1] for r in conn.execute("PRAGMA table_info(var)").fetchall()]

    needed = [hvg_key, mean_key, var_key, std_key, score_key, "atlas_gene_name"]
    missing = [c for c in needed if c not in var_cols]
    if missing:
        raise ValueError(
            f"var 中不存在这些列: {missing}\n"
            f"请先运行修改后的 sap.pp.highly_variable_genes(atlas)"
        )

    # 直接从 var 读取 gene-level 结果
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



# 高变基因（HVG, Highly Variable Genes）选择图 ： seurat 版本
def _highly_variable_genes_plot_seurat(
        atlas: Atlas,
        hvg_key: str = "highly_variable_genes",
        sample_other: int | None = 20000,
        save: PathLike[str] | str | None = None,
):

    """绘制 Seurat 风格高变基因筛选结果。

    该函数读取 Seurat 风格 HVG 统计字段，展示 normalized dispersion 与 mean 的关系。

    它适合检查 ``highly_variable_genes_seurat`` 的分箱和筛选结果。

    Parameters
    ----------
    atlas
        Atlas 对象。通常要求已经连接数据库，并包含该函数所需的 ``obs``、``var``、``X_HyS_data`` 或
        embedding 结果表。

    hvg_key
        ``var`` 中表示高变基因的布尔列名。

    sample_other
        从非目标点中抽样用于绘图的数量。

    save
        图像保存设置，可为布尔值、扩展名或文件名。

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

        sap.pl.highly_variable_genes_plot_seurat(...)
    """
    print("\n==== highly_variable_genes_plot_like_seurat ====")
    start = datetime.now()

    conn = atlas.connection

    if conn is None:
        raise ValueError("atlas.connection 为空，请先连接数据库")

    # DuckDB 字段安全引用
    def _q(name: str) -> str:
        """为 SQL 标识符添加安全引用。

        该内部函数属于QC 可视化模块，用于支撑同一模块中的公共 API。

        读取 QC 指标和表达矩阵，绘制最高表达基因、violin、scatter 和 HVG 诊断图。

        它通常不会作为用户入口直接调用；直接调用时需要保证输入对象、数据库连接和相关临时表已经由上游步骤准备好。

        Parameters
        ----------
        name
            对象名称、列名或 SQL 标识符，具体含义由调用位置决定。

        Returns
-------
        quoted_name
            加双引号后的 SQL 标识符。

        Notes
        -----
        这是内部 helper；除非需要扩展 scAtlasPy 内部流程，一般不建议在用户代码中直接调用。
        """
        return '"' + name.replace('"', '""') + '"'

    # 检查 var 中列是否存在
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

    # 读取 var 中已经保存好的 HVG 结果
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

    # 清理 nan / inf
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

    # 非 HVG 可选抽样
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

    # 画图
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11, 4.5),
        facecolor="white",
    )

    # 左图：normalized dispersions
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

    # 右图：raw dispersions
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


# 高变基因（HVG, Highly Variable Genes）选择图
def highly_variable_genes(
        atlas: Atlas,
        flavor: Literal["seurat", "cv", "var"] = "seurat",

        # 通用参数：两个底层函数都支持
        hvg_key: str = "highly_variable_genes",
        sample_other: int | None = 20000,

        # cv / var 版本参数：只传给 highly_variable_genes_plot()
        mean_key: str = "hvg_mean",
        var_key: str = "hvg_var",
        std_key: str = "hvg_std",
        score_key: str = "hvg_score",
        figsize: tuple[float, float] | None = None,
        point_size_hvg: float = 8,
        point_size_other: float = 6,
        alpha_hvg: float = 0.9,
        alpha_other: float = 0.6,

        # seurat 版本参数：只传给 highly_variable_genes_plot_seurat()
        save: PathLike[str] | str | None = None,
):
    """
    绘制高变基因结果。

    该函数是 HVG 绘图的统一入口，不自动判断 flavor，
    而是根据用户显式指定的 flavor 调用已有的两个绘图函数。

    Parameters
    ----------
    atlas
        Atlas 对象。

    flavor
        HVG 绘图类型。

        - "seurat":
            调用 highly_variable_genes_plot_seurat()

        - "cv":
            调用 highly_variable_genes_plot()

        - "var":
            调用 highly_variable_genes_plot()

    hvg_key
        var 表中表示高变基因的布尔字段名。

    sample_other
        非高变基因抽样数量。为 None 时绘制全部非高变基因。

    mean_key
        cv / var 版本中，基因均值字段名。

    var_key
        cv / var 版本中，基因方差字段名。

    std_key
        cv / var 版本中，基因标准差字段名。

    score_key
        cv / var 版本中，HVG score 字段名。

    figsize
        cv / var 版本图像大小。

    point_size_hvg
        cv / var 版本高变基因点大小。

    point_size_other
        cv / var 版本非高变基因点大小。

    alpha_hvg
        cv / var 版本高变基因点透明度。

    alpha_other
        cv / var 版本非高变基因点透明度。

    save
        seurat 版本保存路径。

    Returns
    -------
    result
        底层绘图函数返回结果。
    """

    flavor = str(flavor).lower().strip()

    if flavor == "seurat":
        return _highly_variable_genes_plot_seurat(
            atlas=atlas,
            hvg_key=hvg_key,
            sample_other=sample_other,
            save=save,
        )

    elif flavor in ["cv", "var"]:
        return _highly_variable_genes_plot(
            atlas=atlas,
            hvg_key=hvg_key,
            mean_key=mean_key,
            var_key=var_key,
            std_key=std_key,
            score_key=score_key,
            sample_other=sample_other,
            figsize=figsize,
            point_size_hvg=point_size_hvg,
            point_size_other=point_size_other,
            alpha_hvg=alpha_hvg,
            alpha_other=alpha_other,
        )

    else:
        raise ValueError(
            f"不支持的 flavor: {flavor}. "
            "可选值为: 'seurat', 'cv', 'var'"
        )
