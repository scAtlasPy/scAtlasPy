from ..data import Atlas
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from os import PathLike
from typing import Literal


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

        save_path: PathLike[str] | str | None = None,
):
    """绘制高变基因筛选结果诊断图。

    该函数是 HVG 绘图的公开入口，会根据 ``flavor`` 调用对应的底层绘图逻辑：
    ``"seurat"`` 读取 ``means``、``dispersions`` 和 ``dispersions_norm``；
    ``"cv"`` 或 ``"var"`` 读取 ``hvg_mean``、``hvg_var``、``hvg_std`` 和``hvg_score``。
    所有风格都会根据 ``hvg_key`` 高亮已经标记为高变的基因。

    该图类似 Scanpy 的 ``sc.pl.highly_variable_genes``，主要用于确认高变基因筛选
    结果是否合理，而不是重新计算高变基因。

    Parameters
    ----------
    atlas
        Atlas 对象。要求已经连接 DuckDB 数据库，并且 ``var`` 表中已经包含对应
        ``flavor`` 所需的 HVG 统计列。
    flavor
        绘图风格。``"seurat"`` 使用 Seurat 风格 dispersion 结果；
        ``"cv"`` 和 ``"var"`` 使用均值、方差、标准差和高变得分结果。
    hvg_key
        ``var`` 中标记高变基因的列名。
    sample_other
        从非高变基因中抽样展示的数量。为 ``None`` 时绘制全部非高变基因。
    mean_key
        ``var`` 中保存均值的列名。
    var_key
        ``var`` 中保存方差的列名。
    std_key
        ``var`` 中保存标准差的列名。
    score_key
        ``var`` 中保存高变基因评分的列名。
    figsize
        CV/方差风格图的 Matplotlib 图像大小；Seurat 风格使用底层函数固定尺寸。
    point_size_hvg
        高变基因散点大小。
    point_size_other
        非高变基因散点大小。
    alpha_hvg
        高变基因散点透明度。
    alpha_other
        非高变基因散点透明度。
    save_path
        图片保存路径。为 ``None`` 时只显示图片，不保存。

    Returns
    -------
    None

    Examples
    --------
    绘制默认高变基因结果::

        sap.pp.highly_variable_genes(atlas, n_top_genes=2000)
        sap.pl.highly_variable_genes(atlas)

    绘制 CV 风格结果::

        sap.pl.highly_variable_genes(
            atlas,
            flavor="cv",
            hvg_key="highly_variable_genes",
        )

    保存 Seurat 风格图::

        sap.pl.highly_variable_genes(
            atlas,
            flavor="seurat",
            hvg_key="highly_variable_genes",
            sample_other=50000,
            save_path=r"F:\\figures\\hvg.png",
        )"""

    flavor = str(flavor).lower().strip()

    if flavor == "seurat":
        return _highly_variable_genes_plot_seurat(
            atlas=atlas,
            hvg_key=hvg_key,
            sample_other=sample_other,
            save_path=save_path,
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
            save_path=save_path,
        )

    else:
        raise ValueError(
            f"不支持的 flavor: {flavor}. "
            "可选值为: 'seurat', 'cv', 'var'"
        )


# HVG： cv / var 版本
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
        alpha_other: float = 0.6,
        save_path: PathLike[str] | str | None = None
):

    """绘制 cv/var风格的高变基因筛选诊断图。

    该内部绘图函数从 ``var`` 表读取每个基因的均值、方差、标准差、高变得分和
    HVG 标记，并绘制两张诊断散点图：左图展示归一化后的高变得分与平均表达的
    关系，右图展示原始方差与平均表达的关系。高变基因会被单独高亮显示。

    该图用于检查 CV 或方差风格的 HVG 选择是否合理，例如高变基因是否集中在
    期望的表达区间、非高变基因背景是否过密，以及 ``score_key`` 是否能有效区分
    高变和非高变基因。

    Parameters
    ----------
    atlas
        Atlas 对象。要求已经连接 DuckDB 数据库，并且 ``var`` 表中包含 HVG 统计列。

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
        从非高变基因中抽样用于绘图的数量。为 ``None`` 时绘制全部非高变基因。

    figsize
        Matplotlib 图像大小。为 ``None`` 时使用 Matplotlib 默认尺寸。

    point_size_hvg
        高变基因点大小。

    point_size_other
        非高变基因点大小。

    alpha_hvg
        高变基因点透明度。

    alpha_other
        非高变基因点透明度。
    save_path
        图片保存路径。为 ``None`` 时只显示图片，不保存。

    Notes
    -----
    绘图前需要先运行会写入 ``hvg_mean``、``hvg_var``、``hvg_std``、``hvg_score`` 和
    ``hvg_key`` 的高变基因计算流程。该函数只负责读取结果并绘图，不会重新计算 HVG。

    Examples
    --------
    绘制 CV/方差风格 HVG 结果::

        sap.pl.highly_variable_genes(atlas, flavor="cv")
    """

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
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

# HVG： seurat 版本
def _highly_variable_genes_plot_seurat(
        atlas: Atlas,
        hvg_key: str = "highly_variable_genes",
        sample_other: int | None = 20000,
        save_path: PathLike[str] | str | None = None,
):

    """绘制 Seurat 风格的高变基因筛选诊断图。

    该内部绘图函数从 ``var`` 表读取 Seurat 风格 HVG 结果，包括 ``means``、
    ``dispersions``、``dispersions_norm`` 和 ``hvg_key``。函数会绘制两张散点图：
    左图展示 normalized dispersion 与平均表达的关系，
    右图展示原始 dispersion与平均表达的关系，并高亮 ``hvg_key`` 标记的高变基因。

    该图适合检查 Seurat 风格分箱归一化后的 dispersion 是否合理，以及最终选出的
    高变基因是否分布在预期的表达范围内。

    Parameters
    ----------
    atlas
        Atlas 对象。要求已经连接 DuckDB 数据库，并且 ``var`` 表中包含 Seurat 风格
        HVG 统计列。

    hvg_key
        ``var`` 中表示高变基因的布尔列名。

    sample_other
        从非高变基因中抽样用于绘图的数量。为 ``None`` 时绘制全部非高变基因。

    save_path
        图片保存路径。为 ``None`` 时只显示图片，不写入文件。

    Returns
    -------
    None
        函数直接绘图，并在 ``save_path`` 不为 ``None`` 时保存图片。

    Notes
    -----
    绘图前需要先运行 Seurat 风格的高变基因计算流程，确保 ``var`` 表中已经存在
    ``means``、``dispersions`` 和 ``dispersions_norm``。

    Examples
    --------
    绘制默认 Seurat 风格 HVG 结果::

        sap.pl.highly_variable_genes(atlas, flavor="seurat")
    """

    start = datetime.now()

    conn = atlas.connection

    if conn is None:
        raise ValueError("atlas.connection 为空，请先连接数据库")

    # DuckDB 字段安全引用
    def _q(name: str) -> str:
        """为 DuckDB SQL 标识符添加双引号引用。

        该内部 helper 用于安全拼接 ``var`` 表列名，避免列名中包含特殊字符或与 SQL
        关键字冲突。函数只处理标识符引用，不处理 SQL 值的转义。

        Parameters
        ----------
        name
            需要作为 SQL 标识符使用的列名。

        Returns
        -------
        str
            已加双引号并转义内部双引号的 SQL 标识符。
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

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()
