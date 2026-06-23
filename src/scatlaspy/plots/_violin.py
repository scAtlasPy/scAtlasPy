from ..data import Atlas
import re
from os import PathLike
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colorbar import ColorbarBase
from scipy.stats import gaussian_kde
from typing import Any


# =====================================================
# 统一离散分类颜色池
# -----------------------------------------------------
# 用于 obs 分类变量上色，例如：
# kmeans / cell_type / batch / organ 等
#
# 这些 palette 拼起来大约有 100 个离散颜色：
# tab20(20) + tab20b(20) + tab20c(20)
# + Set3(12) + Paired(12) + Accent(8) + Dark2(8)
# =====================================================
DEFAULT_DISCRETE_PALETTES = (
    "tab20",
    "tab20b",
    "tab20c",
    "Set3",
    "Paired",
    "Accent",
    "Dark2",
)


# =====================================================
# 通用分类标签自然排序
# -----------------------------------------------------
# 解决：
# embryo_1, embryo_10, embryo_11, embryo_2
#
# 排成：
# embryo_1, embryo_2, embryo_3, ..., embryo_10
#
# 同样适用于：
# cluster_1 / cluster_10
# batch2 / batch10
# group_3_day_2 / group_3_day_12
# =====================================================
_MISSING_CATEGORY_LABELS = {"", "na", "nan", "none", "<na>", "null"}


def _natural_sort_key(value: Any):
    """
    分类标签自然排序 key。

    Examples
    --------
    embryo_1  < embryo_2  < embryo_10
    cluster_1 < cluster_2 < cluster_11
    """

    s = str(value).strip()

    # 缺失值标签放最后
    if s.casefold() in _MISSING_CATEGORY_LABELS:
        return (1, ())

    parts = re.split(r"(\d+)", s)

    key = []
    for part in parts:
        if part == "":
            continue

        if part.isdigit():
            key.append((0, int(part)))
        else:
            key.append((1, part.casefold()))

    return (0, tuple(key))


def _sort_categories_natural(labels: Any) -> list[str]:
    """
    对分类标签做默认自然排序。

    不额外暴露参数，所有离散分类变量默认使用这个排序。
    """

    labels = [str(x) for x in list(labels)]

    # 去重，同时保留原始列表中的唯一标签
    labels = list(dict.fromkeys(labels))

    return sorted(labels, key=_natural_sort_key)


def _build_discrete_color_map(labels: Any, palette: Any | None=None):
    """构建内部中间数据结构。

    该内部函数属于UMAP/表达可视化模块，用于支撑同一模块中的公共 API。

    读取 UMAP、obs、var 和表达矩阵，绘制 UMAP、feature plot、violin、dotplot 和 stacked violin。

    它通常不会作为用户入口直接调用；直接调用时需要保证输入对象、数据库连接和相关临时表已经由上游步骤准备好。

    Parameters
    ----------
    labels
        分类标签列表。

    palette
        离散分类变量使用的颜色方案。

    Returns
    -------
    result
        构建得到的内部对象，通常是 DataFrame、Arrow Table 或更新后的游标元组。

    Notes
    -----
    这是内部 helper；除非需要扩展 scAtlasPy 内部流程，一般不建议在用户代码中直接调用。
    """

    labels = list(labels)

    #  默认使用大颜色池
    if palette is None:
        palette_names = DEFAULT_DISCRETE_PALETTES

    # 兼容原来的 palette="tab20" 写法
    elif isinstance(palette, str):
        palette_names = (palette,)

    # 支持 palette=["tab20", "tab20b", ...]
    else:
        palette_names = tuple(palette)

    palette_colors = []

    for cmap_name in palette_names:
        cmap_obj = plt.get_cmap(cmap_name)

        # ListedColormap，比如 tab20 / Set3，通常有 .colors
        if hasattr(cmap_obj, "colors"):
            palette_colors.extend(list(cmap_obj.colors))

        # 兜底：如果是连续 colormap，就均匀取色
        else:
            n = getattr(cmap_obj, "N", 256)
            palette_colors.extend([
                cmap_obj(i / max(n - 1, 1))
                for i in range(n)
            ])

    # 如果类别数超过颜色池，继续用 hsv 补足
    if len(palette_colors) < len(labels):
        extra_n = len(labels) - len(palette_colors)
        hsv = plt.get_cmap("hsv")
        palette_colors.extend([
            hsv(i / max(extra_n, 1))
            for i in range(extra_n)
        ])

    return {
        lab: palette_colors[i]
        for i, lab in enumerate(labels)
    }



# 普通 violin
def violin(
        atlas: Atlas,
        genes: str | list[str],
        groupby: str = "kmeans",
        use_data: str = "data_log1p",
        sample_n_per_group: int | None = 2000,
        groups: list | None = None,
        where: str | None = None,
        order: list | None = None,
        save_path: PathLike[str] | str | None = None
):

    """绘制基因或 obs 指标的 violin 图。

    该函数从 Atlas 数据库读取指定基因表达或 ``obs`` 指标，并按分组绘制 violin 图。用途类似 Scanpy 的 ``sc.pl.violin``。

    Parameters
    ----------
    atlas
        Atlas 对象。通常需要已经连接到 DuckDB 数据库，并包含该函数读取或写入所需的 ``obs``、``var``、表达矩阵或结果表。
    genes
        需要展示的基因名称列表。
    groupby
        ``obs`` 中的分组列名，例如 ``"kmeans"``、``"leiden"`` 或 ``"cell_type"``。
    use_data
        读取的表达矩阵或结果表名称。常用值包括 ``"data"``、``"data_normalize"``、``"data_log1p"`` 和
        ``"data_scale"``。
    sample_n_per_group
        每个分组抽样用于绘图的细胞数量。
    groups
        需要计算、展示或保留的分组列表。为 ``None`` 时使用全部分组。
    where
        额外 SQL 过滤条件。为 ``None`` 时不添加额外条件。
    order
        分组或基因展示顺序。为 ``None`` 时使用默认顺序。
    save_path
        图片保存路径。为 ``None`` 时只显示或返回图对象。

    Returns
    -------
    matplotlib.figure.Figure 或 None
        当 ``return_fig=True`` 或函数实现返回图对象时返回 Figure；否则通常直接显示图形。

    Examples
    --------
    绘制单个基因在 cluster 中的表达分布::

        sap.pl.violin(atlas, keys=["MS4A1"], groupby="kmeans")

    同时绘制多个 QC 指标::

        sap.pl.violin(
            atlas,
            keys=["n_genes_by_counts", "pct_counts_mt"],
            groupby="kmeans",
            use_data="data_log1p",
        )"""

    conn = atlas.connection

    # 参数标准化
    if isinstance(genes, str):
        genes = [genes]
    genes = [str(g) for g in genes]

    if len(genes) == 0:
        raise ValueError("genes 不能为空")

    # 检查列
    obs_cols = [r[1] for r in conn.execute("PRAGMA table_info(obs)").fetchall()]
    var_cols = [r[1] for r in conn.execute("PRAGMA table_info(var)").fetchall()]
    x_cols = [r[1] for r in conn.execute("PRAGMA table_info(X_HyS_data)").fetchall()]

    if groupby not in obs_cols:
        raise ValueError(f"obs 中不存在列: {groupby}")
    if "atlas_cell_id" not in obs_cols:
        raise ValueError("obs 中不存在 atlas_cell_id")
    if "atlas_gene_id" not in var_cols or "atlas_gene_name" not in var_cols:
        raise ValueError("var 中不存在 atlas_gene_id / atlas_gene_name")
    if use_data not in x_cols:
        raise ValueError(f"X_HyS_data 中不存在字段: {use_data}")

    # gene_name -> gene_id
    gene_name_sql = ", ".join([f"'{g}'" for g in genes])

    gene_map_df = conn.execute(f"""
        SELECT atlas_gene_id, atlas_gene_name
        FROM var
        WHERE atlas_gene_name IN ({gene_name_sql})
    """).fetchdf()

    if len(gene_map_df) == 0:
        raise ValueError("var 中找不到这些基因")

    gene_map = dict(zip(gene_map_df["atlas_gene_name"], gene_map_df["atlas_gene_id"]))
    missing_genes = [g for g in genes if g not in gene_map]
    if missing_genes:
        raise ValueError(f"var 中找不到这些基因: {missing_genes}")

    gene_map_df["atlas_gene_name"] = pd.Categorical(
        gene_map_df["atlas_gene_name"],
        categories=genes,
        ordered=True
    )
    gene_map_df = gene_map_df.sort_values("atlas_gene_name").reset_index(drop=True)

    # 准备 group 抽样细胞
    where_clauses = [f"{groupby} IS NOT NULL"]
    if where is not None and str(where).strip() != "":
        where_clauses.append(f"({where})")

    if groups is not None:
        group_sql = ", ".join([f"'{g}'" for g in groups])
        where_clauses.append(f"CAST({groupby} AS TEXT) IN ({group_sql})")

    where_sql = " AND ".join(where_clauses)

    # group 列表
    group_df = conn.execute(f"""
        SELECT
            CAST({groupby} AS TEXT) AS group_label,
            COUNT(*) AS n_cells
        FROM obs
        WHERE {where_sql}
        GROUP BY 1
        ORDER BY 1
    """).fetchdf()

    if len(group_df) == 0:
        raise ValueError("没有可用 group")

    # 数字型 group 按数值排序，避免 0,1,10,11,2
    def _group_sort_key(x: Any):
        """生成分组或标签的自然排序键。

        该内部函数属于UMAP/表达可视化模块，用于支撑同一模块中的公共 API。

        读取 UMAP、obs、var 和表达矩阵，绘制 UMAP、feature plot、violin、dotplot 和 stacked violin。

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

    group_df["__sort_key__"] = group_df["group_label"].map(_group_sort_key)
    group_df = (
        group_df
        .sort_values("__sort_key__")
        .drop(columns="__sort_key__")
        .reset_index(drop=True)
    )

    if order is not None:
        wanted = [str(x) for x in order]
        group_df = group_df[group_df["group_label"].isin(wanted)].copy()
        if len(group_df) == 0:
            raise ValueError("order 过滤后没有可用 group")
        group_df["order_idx"] = group_df["group_label"].map({g: i for i, g in enumerate(wanted)})
        group_df = group_df.sort_values("order_idx").drop(columns="order_idx").reset_index(drop=True)

    group_labels = group_df["group_label"].astype(str).tolist()

    # 每个 group 单独抽样，再 union
    sampled_parts = []
    for g in group_labels:
        if sample_n_per_group is None:
            q = f"""
                SELECT
                    atlas_cell_id,
                    CAST({groupby} AS TEXT) AS group_label
                FROM obs
                WHERE {where_sql}
                  AND CAST({groupby} AS TEXT) = '{g}'
            """
        else:
            q = f"""
                SELECT
                    atlas_cell_id,
                    CAST({groupby} AS TEXT) AS group_label
                FROM obs
                WHERE {where_sql}
                  AND CAST({groupby} AS TEXT) = '{g}'
                ORDER BY random()
                LIMIT {int(sample_n_per_group)}
            """
        sampled_parts.append(conn.execute(q).fetchdf())

    cells_df = pd.concat(sampled_parts, ignore_index=True)

    if len(cells_df) == 0:
        raise ValueError("抽样后没有细胞")

    # 注册临时表
    conn.register("_violin_cells_tmp", cells_df)
    conn.register("_violin_genes_tmp", gene_map_df[["atlas_gene_id", "atlas_gene_name"]])

    # 取表达长表（补隐式 0）
    plot_df = conn.execute(f"""
        SELECT
            c.group_label,
            g.atlas_gene_name AS gene,
            COALESCE(x.{use_data}, 0.0) AS expr
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

    plot_df["gene"] = pd.Categorical(plot_df["gene"], categories=genes, ordered=True)
    plot_df["group_label"] = pd.Categorical(plot_df["group_label"], categories=group_labels, ordered=True)
    plot_df = plot_df.sort_values(["gene", "group_label"]).reset_index(drop=True)

    # 作图
    n_panels = len(genes)
    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(4.2 * n_panels, 5.2),
        facecolor="white",
        squeeze=False
    )
    axes = axes.ravel()

    # scanpy-like palette
    scanpy_colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#bcbd22",
        "#17becf", "#7f7f7f", "#aec7e8", "#ffbb78"
    ]
    color_map = {g: scanpy_colors[i % len(scanpy_colors)] for i, g in enumerate(group_labels)}

    for ax, gene in zip(axes, genes):
        sub = plot_df[plot_df["gene"] == gene].copy()

        positions = np.arange(len(group_labels)) + 1

        violin_data = []
        for g in group_labels:
            vals = sub[sub["group_label"] == g]["expr"].to_numpy()
            violin_data.append(vals)

        vp = ax.violinplot(
            violin_data,
            positions=positions,
            widths=0.85,
            showmeans=False,
            showmedians=True,
            showextrema=False
        )

        for i, body in enumerate(vp["bodies"]):
            body.set_facecolor(color_map[group_labels[i]])
            body.set_edgecolor("#4a4a4a")
            body.set_alpha(0.9)
            body.set_linewidth(1.0)

        if "cmedians" in vp:
            vp["cmedians"].set_color("#2f2f2f")
            vp["cmedians"].set_linewidth(1.2)

        # jitter points
        for pos, g in zip(positions, group_labels):
            vals = sub[sub["group_label"] == g]["expr"].to_numpy()
            if len(vals) == 0:
                continue

            n_dot = min(250, len(vals))
            dot_idx = np.random.choice(len(vals), size=n_dot, replace=False)
            vals_dot = vals[dot_idx]
            jitter = (np.random.rand(n_dot) - 0.5) * 0.18

            ax.scatter(
                np.full(n_dot, pos) + jitter,
                vals_dot,
                s=3,
                c="#2f2f2f",
                alpha=0.6,
                linewidths=0
            )

        ax.set_title(gene, fontsize=14, weight="normal", pad=8)
        ax.set_xlabel(groupby, fontsize=12)
        ax.set_ylabel("expression", fontsize=12)
        ax.set_xticks(positions)
        ax.set_xticklabels(group_labels, fontsize=11)

        ax.set_facecolor("white")
        ax.grid(True, axis="y", color="#d9d9d9", linewidth=0.8, alpha=0.8)
        ax.grid(False, axis="x")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.0)
        ax.spines["bottom"].set_linewidth(1.0)
        ax.tick_params(axis="both", labelsize=11, width=1.0, length=4)

    plt.tight_layout(pad=1.0)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


# 堆叠提琴图 violin gene 在 x 轴，group 在 y 轴，每个 group × gene 的格子里画一个小提琴形状，并用 median expression 控制颜色深浅
def stacked_violin(
        atlas: Atlas,
        genes: str | list[str],
        groupby: str = "cell_type_auto",
        use_data: str = "data_log1p",
        sample_n_per_group: int | None = 2000,
        groups: list | None = None,
        where: str | None = None,
        order: list | None = None,
        color_vmin: float | None = 0.0,
        color_vmax: float | None = 5.0,
        font_size: int = 14,
        save_path: PathLike[str] | str | None = None
):

    """绘制 stacked violin marker 表达图。

    该函数按分组展示多个基因的表达分布，并将每个基因的 violin 图堆叠排列，适合展示细胞类型 marker 在 cluster 或注释类别中的表达模式。

    Parameters
    ----------
    atlas
        Atlas 对象。通常需要已经连接到 DuckDB 数据库，并包含该函数读取或写入所需的 ``obs``、``var``、表达矩阵或结果表。
    genes
        需要展示的基因名称列表。
    groupby
        ``obs`` 中的分组列名，例如 ``"kmeans"``、``"leiden"`` 或 ``"cell_type"``。
    use_data
        读取的表达矩阵或结果表名称。常用值包括 ``"data"``、``"data_normalize"``、``"data_log1p"`` 和
        ``"data_scale"``。
    sample_n_per_group
        每个分组抽样用于绘图的细胞数量。
    groups
        需要计算、展示或保留的分组列表。为 ``None`` 时使用全部分组。
    where
        额外 SQL 过滤条件。为 ``None`` 时不添加额外条件。
    order
        分组或基因展示顺序。为 ``None`` 时使用默认顺序。
    color_vmin
        参数。用于控制该函数的输入、输出或计算细节；默认值适合常规 Atlas 工作流。
    color_vmax
        参数。用于控制该函数的输入、输出或计算细节；默认值适合常规 Atlas 工作流。
    font_size
        绘图字体大小。
    save_path
        图片保存路径。为 ``None`` 时只显示或返回图对象。

    Returns
    -------
    None

    Examples
    --------
    绘制多个 marker genes 的 stacked violin::

        sap.pl.stacked_violin(
            atlas,
            genes=["MS4A1", "CD3D", "LYZ", "NKG7"],
            groupby="kmeans",
        )

    按自动注释细胞类型展示，并交换坐标轴::

        sap.pl.stacked_violin(
            atlas,
            genes=["MS4A1", "CD3D", "LYZ"],
            groupby="cell_type_auto",
            swap_axes=True,
            save=r"F:\\figures\\stacked_violin.png",
        )"""
    conn = atlas.connection

    if isinstance(genes, str):
        genes = [genes]
    genes = [str(g) for g in genes]

    if len(genes) == 0:
        raise ValueError("genes 不能为空")

    # 检查列
    obs_cols = [r[1] for r in conn.execute("PRAGMA table_info(obs)").fetchall()]
    var_cols = [r[1] for r in conn.execute("PRAGMA table_info(var)").fetchall()]
    x_cols = [r[1] for r in conn.execute("PRAGMA table_info(X_HyS_data)").fetchall()]

    if groupby not in obs_cols:
        raise ValueError(f"obs 中不存在列: {groupby}")
    if "atlas_cell_id" not in obs_cols:
        raise ValueError("obs 中不存在 atlas_cell_id")
    if "atlas_gene_id" not in var_cols or "atlas_gene_name" not in var_cols:
        raise ValueError("var 中不存在 atlas_gene_id / atlas_gene_name")
    if use_data not in x_cols:
        raise ValueError(f"X_HyS_data 中不存在字段: {use_data}")

    # gene_name -> gene_id
    gene_name_sql = ", ".join([f"'{g}'" for g in genes])

    gene_map_df = conn.execute(f"""
        SELECT atlas_gene_id, atlas_gene_name
        FROM var
        WHERE atlas_gene_name IN ({gene_name_sql})
    """).fetchdf()

    if len(gene_map_df) == 0:
        raise ValueError("var 中找不到这些基因")

    gene_map = dict(zip(gene_map_df["atlas_gene_name"], gene_map_df["atlas_gene_id"]))
    missing_genes = [g for g in genes if g not in gene_map]
    if missing_genes:
        raise ValueError(f"var 中找不到这些基因: {missing_genes}")

    gene_map_df["atlas_gene_name"] = pd.Categorical(
        gene_map_df["atlas_gene_name"],
        categories=genes,
        ordered=True
    )
    gene_map_df = gene_map_df.sort_values("atlas_gene_name").reset_index(drop=True)

    where_clauses = [f"{groupby} IS NOT NULL"]

    if where is not None and str(where).strip() != "":
        where_clauses.append(f"({where})")

    if groups is not None:
        groups_sql = ", ".join([f"'{str(g)}'" for g in groups])
        where_clauses.append(f"CAST({groupby} AS TEXT) IN ({groups_sql})")

    where_sql = " AND ".join(where_clauses)

    # group 列表
    group_df = conn.execute(f"""
        SELECT
            CAST({groupby} AS TEXT) AS group_label,
            COUNT(*) AS n_cells
        FROM obs
        WHERE {where_sql}
        GROUP BY 1
        ORDER BY 1
    """).fetchdf()

    if len(group_df) == 0:
        raise ValueError("没有可用 group")

    # 数字型 group 按数值排序，避免 0,1,10,11,2
    def _group_sort_key(x: Any):
        """生成分组或标签的自然排序键。

        该内部函数属于UMAP/表达可视化模块，用于支撑同一模块中的公共 API。

        读取 UMAP、obs、var 和表达矩阵，绘制 UMAP、feature plot、violin、dotplot 和 stacked violin。

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

    group_df["__sort_key__"] = group_df["group_label"].map(_group_sort_key)
    group_df = (
        group_df
        .sort_values("__sort_key__")
        .drop(columns="__sort_key__")
        .reset_index(drop=True)
    )

    if order is not None:
        wanted = [str(x) for x in order]
        group_df = group_df[group_df["group_label"].isin(wanted)].copy()
        if len(group_df) == 0:
            raise ValueError("order 过滤后没有可用 group")
        group_df["order_idx"] = group_df["group_label"].map({g: i for i, g in enumerate(wanted)})
        group_df = group_df.sort_values("order_idx").drop(columns="order_idx").reset_index(drop=True)

    group_labels = group_df["group_label"].astype(str).tolist()

    # 每个 group 抽样细胞
    sampled_parts = []
    for g in group_labels:
        if sample_n_per_group is None:
            q = f"""
                SELECT
                    atlas_cell_id,
                    CAST({groupby} AS TEXT) AS group_label
                FROM obs
                WHERE {where_sql}
                  AND CAST({groupby} AS TEXT) = '{g}'
            """
        else:
            q = f"""
                SELECT
                    atlas_cell_id,
                    CAST({groupby} AS TEXT) AS group_label
                FROM obs
                WHERE {where_sql}
                  AND CAST({groupby} AS TEXT) = '{g}'
                ORDER BY random()
                LIMIT {int(sample_n_per_group)}
            """
        sampled_parts.append(conn.execute(q).fetchdf())

    cells_df = pd.concat(sampled_parts, ignore_index=True)
    if len(cells_df) == 0:
        raise ValueError("抽样后没有细胞")

    # 注册临时表
    conn.register("_sv_cells_tmp", cells_df)
    conn.register("_sv_genes_tmp", gene_map_df[["atlas_gene_id", "atlas_gene_name"]])

    # 取表达长表（补隐式 0）
    expr_df = conn.execute(f"""
        SELECT
            c.group_label,
            g.atlas_gene_name AS gene,
            COALESCE(x.{use_data}, 0.0) AS expr
        FROM _sv_cells_tmp c
        CROSS JOIN _sv_genes_tmp g
        LEFT JOIN X_HyS_data x
            ON c.atlas_cell_id = x.atlas_cell_id
           AND g.atlas_gene_id = x.atlas_gene_id
    """).fetchdf()

    conn.unregister("_sv_cells_tmp")
    conn.unregister("_sv_genes_tmp")

    if len(expr_df) == 0:
        raise ValueError("expr_df 为空，无法作图")

    expr_df["gene"] = pd.Categorical(expr_df["gene"], categories=genes, ordered=True)
    expr_df["group_label"] = pd.Categorical(expr_df["group_label"], categories=group_labels, ordered=True)

    # 中位数统计（用于着色）
    median_df = (
        expr_df
        .groupby(["group_label", "gene"], observed=True)["expr"]
        .median()
        .reset_index(name="median_expr")
    )

    # 布局
    n_genes = len(genes)
    n_groups = len(group_labels)

    main_w = max(7.4, 0.44 * n_genes + 2.8)
    main_h = max(4.4, 0.56 * n_groups + 2.0)

    right_w = 2.7
    fig_w = main_w + right_w
    fig_h = main_h + 0.8

    max_y_len = max(len(str(x)) for x in group_labels) if len(group_labels) > 0 else 10
    left_margin = min(0.25, max(0.09, 0.035 + 0.008 * max_y_len))

    max_gene_len = max(len(str(x)) for x in genes) if len(genes) > 0 else 10
    bottom_margin = min(0.28, max(0.18, 0.06 + 0.009 * max_gene_len))

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
    gs = fig.add_gridspec(
        1, 2,
        width_ratios=[main_w, right_w],
        wspace=0.03
    )

    ax = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[0, 1])
    ax_right.axis("off")
    ax.set_facecolor("white")
    ax_right.set_facecolor("white")

    # 颜色映射
    if color_vmin is None:
        color_vmin = float(median_df["median_expr"].min())
    if color_vmax is None:
        color_vmax = float(median_df["median_expr"].max())

    norm = mpl.colors.Normalize(vmin=color_vmin, vmax=color_vmax)
    cmap = plt.get_cmap("Blues")

    gene_to_x = {g: i for i, g in enumerate(genes)}

    group_to_y = {g: (len(group_labels) - 1 - i) for i, g in enumerate(group_labels)}

    cell_half_height = 0.34
    cell_half_width = 0.33

    grouped_expr = expr_df.groupby(["group_label", "gene"], observed=True)

    for (grp, gene), sub in grouped_expr:
        vals = sub["expr"].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]

        if len(vals) == 0:
            continue

        x0 = gene_to_x[str(gene)]
        y0 = group_to_y[str(grp)]

        med_val = float(np.median(vals))
        facecolor = cmap(norm(med_val))

        vmin = float(np.min(vals))
        vmax = float(np.max(vals))

        # 全 0 / 常数分布：画一个细竖线
        if vmax - vmin < 1e-12:
            ax.plot(
                [x0, x0],
                [y0 - cell_half_height * 0.35, y0 + cell_half_height * 0.35],
                color="#b0b0b0",
                linewidth=1.0,
                zorder=3
            )
            continue

        # KDE 估计
        try:
            kde = gaussian_kde(vals)
            ys = np.linspace(vmin, vmax, 120)
            dens = kde(ys)
        except Exception:
            ys = np.linspace(vmin, vmax, 120)
            dens = np.ones_like(ys)

        dens = np.asarray(dens, dtype=float)
        if np.max(dens) > 0:
            dens = dens / np.max(dens)
        else:
            dens = np.zeros_like(dens)

        ys_scaled = y0 + ((ys - vmin) / (vmax - vmin) - 0.5) * (2 * cell_half_height)

        widths = dens * cell_half_width
        x_left = x0 - widths
        x_right = x0 + widths

        poly_x = np.concatenate([x_left, x_right[::-1]])
        poly_y = np.concatenate([ys_scaled, ys_scaled[::-1]])

        ax.fill(
            poly_x, poly_y,
            facecolor=facecolor,
            edgecolor="#b0b0b0",
            linewidth=0.9,
            zorder=2
        )

        # 中位数横线
        med_y = y0 + ((med_val - vmin) / (vmax - vmin) - 0.5) * (2 * cell_half_height)
        ax.plot(
            [x0 - cell_half_width * 0.45, x0 + cell_half_width * 0.45],
            [med_y, med_y],
            color="#808080",
            linewidth=0.9,
            zorder=3
        )

    # 主图美化
    ax.set_xticks(np.arange(n_genes))
    ax.set_xticklabels(genes, rotation=90, fontsize=font_size)

    # y tick 也按 scanpy 风格从上到下显示
    y_tick_positions = [group_to_y[g] for g in group_labels]
    ax.set_yticks(y_tick_positions)
    ax.set_yticklabels(group_labels, fontsize=font_size)

    ax.set_xlim(-0.55, n_genes - 0.45)
    ax.set_ylim(-0.5, n_groups - 0.5)

    # 行分隔线
    for j in range(n_groups):
        ax.axhline(j + 0.5, color="#d0d0d0", linewidth=1.0, zorder=0)

    ax.grid(False)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
        spine.set_color("black")

    # 右侧色条说明
    ax_cbar_box = ax_right.inset_axes([0.08, 0.08, 0.84, 0.22])
    ax_cbar_box.axis("off")
    ax_cbar_box.set_xlim(0, 1)
    ax_cbar_box.set_ylim(0, 1)

    ax_cbar_box.text(
        0.50, 0.96,
        "Median expression\nin group",
        ha="center", va="top",
        fontsize=font_size,
        transform=ax_cbar_box.transAxes
    )

    cbar_ax = ax_cbar_box.inset_axes([0.10, 0.18, 0.80, 0.18])

    cb = ColorbarBase(
        cbar_ax,
        cmap=cmap,
        norm=norm,
        orientation="horizontal"
    )
    cb.ax.tick_params(labelsize=font_size, length=4, width=1.0)

    # 边距
    fig.subplots_adjust(
        left=left_margin,
        right=0.98,
        top=0.95,
        bottom=bottom_margin
    )

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


