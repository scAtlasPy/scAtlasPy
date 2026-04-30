# 1️⃣ KMeans（聚类）
# 属于：无监督学习
# 作用：
# 把数据分成 K 类（cluster）
# 输入：
# 高维数据（比如 PCA 后 50维）
# 输出：
# 每个点的 cluster label（0,1,2,...）
# 2️⃣ UMAP（降维 + 可视化）
# 属于：非线性降维
# 作用：
# 把高维数据 → 映射到 2D / 3D
# 目标：
# 尽量保持“邻近关系”（谁和谁相似）
# 输出：
# 每个点在2D空间的位置


# 在单细胞 / embedding 里，经典 pipeline 是：
# 原始数据 (高维)
#     ↓
# PCA（降维，去噪）
#     ↓
# KMeans / Leiden（聚类）
#     ↓
# UMAP（可视化）


# KMeans 的结果本身是“看不见的”
# labels = [0,1,1,2,0,...]
#
# 👉 你根本不知道：
#
# cluster 长什么样
# 有没有分开
# 有没有混在一起

# 用 KMeans 的结果给 UMAP 上色
# plt.scatter(umap[:,0], umap[:,1], c=labels)
#
# 👉 结果：
#
# 每个 cluster 一个颜色
# 看 cluster 是否分开

# UMAP 只是“图”，没有分类
# 必须组合：
# UMAP（形状） + KMeans（颜色） = 可解释结果
# UMAP = 地图
# KMeans = 行政区划

# UMAP可视化，拟定两个方案：
# 1，采样指定数量的细胞，调用传统UMAP
# 2，在全部数据上，以minibatch的方式训练ParameterizedUMAP，https://umap-learn.readthedocs.io/en/latest/parametric_umap.html

# | 方法             | 是否有模型  | 能否泛化 | 是否可 minibatch |
# | UMAP            | ❌ 无模型  | ❌ 不可 | ❌             |
# | Parametric UMAP | ✅ 神经网络 | ✅ 可以 | ✅             |
# 普通 UMAP = “画一张地图”
# Parametric UMAP = “学一个画地图的函数”

# 1. 用神经网络学习 embedding
# X → MLP → Z(2D)
# ✔ 2. minibatch 训练（关键）
# 每次只取一小批数据更新网络
# 类似：
# for batch in data:
#     loss = UMAP_loss(batch)
#     backprop()

# 损失函数（简化）：
# Loss = 正样本（邻居）拉近 + 负样本（非邻居）推远

# repeat:
#     1. 取一小批数据 batch
#     2. 构建邻居关系
#     3. 计算 UMAP loss
#     4. 反向传播更新神经网络

# 模型结构（默认）
# 通常是：
# Input (n_genes)
#    ↓
# Dense 512
#    ↓
# ReLU
#    ↓
# Dense 256
#    ↓
# ReLU
#    ↓
# Dense 2

# UMAP 在你 pipeline 中的作用 = 把 PCA embedding + KMeans label 变成 2D 可视化地图


# MiniBatchKMeans = KMeans 的“流式 / 小批量版本”
#
# 普通 KMeans：每次用全量数据更新中心
# MiniBatchKMeans：每次只用**一个 batch（小块数据）**更新中心
# 循环：
#     取一小批数据（batch）
#     用这批数据 → 更新中心（增量更新）

from datetime import datetime
import math
import pandas as pd
import matplotlib.pyplot as plt
from ..data import Atlas



# Scanpy 风格 UMAP 可视化入口。
# 只负责画图，不计算 UMAP，不训练 KMeans。
# 统一入口
# Scanpy 风格 UMAP 可视化入口。
# 只负责画图，不计算 UMAP，不训练 KMeans。
def umap(
        atlas: Atlas,
        color: str | list[str] = "kmeans",
        sample_n: int | None = 50000,
        where: str | None = None,
        use_expr_field: str = "data_log1p",
        ncols: int = 3,
        figsize=None,
        point_size: float = 8,
        alpha: float = 0.9,
        legend_loc: str = "right_margin",
        frameon: bool = False,
        save_path: str | None = None,
        # ✅【新增】sample_n=None 全量绘图时，每批读取多少细胞
        plot_batch_size: int = 200000
):
    """
    Scanpy 风格 UMAP 可视化入口（简化版）

    支持：
    - SQL过滤（where）
    - SQL抽样（sample_n）
    - obs 分类 / gene feature 混合绘图

    用法
    ----
    sap.pl.umap(
        atlas,
        color="cell_type_auto",
        where="cell_type_auto_confidence IN ('high','medium')",
        sample_n=100000
    )
    """

    print("\n==== sap.pl.umap ====")

    conn = atlas.connection

    # -------------------------------------------------
    # 0️⃣ 参数标准化
    # -------------------------------------------------
    if isinstance(color, str):
        color_list = [color]
    else:
        color_list = list(color)

    if len(color_list) == 0:
        raise ValueError("color 不能为空")

    print(f"[UMAP] color = {color_list}")
    print(f"[UMAP] sample_n = {sample_n}")

    if where is not None and str(where).strip() != "":
        print(f"[UMAP] where = {where}")

    # -------------------------------------------------
    # 1️⃣ 检查 obsm_X_umap 是否存在
    # -------------------------------------------------
    tables = conn.execute("""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_name = 'obsm_X_umap'
    """).fetchdf()

    if len(tables) == 0:
        raise ValueError(
            "数据库中不存在 obsm_X_umap。\n"
            "请先运行 sap.tl.umap(atlas)"
        )

    # -------------------------------------------------
    # 2️⃣ 获取 obs 列和 gene 名
    # -------------------------------------------------
    obs_cols = [r[1] for r in conn.execute("PRAGMA table_info(obs)").fetchall()]

    gene_df = conn.execute("""
        SELECT atlas_gene_name
        FROM var
    """).fetchdf()

    gene_set = set(gene_df["atlas_gene_name"].astype(str).tolist())

    # -------------------------------------------------
    # 3️⃣ 判断 color 类型
    # -------------------------------------------------
    obs_colors = []
    gene_colors = []

    for c in color_list:
        c = str(c)

        if c in obs_cols:
            obs_colors.append(c)

        elif c in gene_set:
            gene_colors.append(c)

        else:
            raise ValueError(
                f"color='{c}' 既不是 obs 列，也不是 gene 名。\n"
                f"请确认 obs 或 var 中存在该字段。"
            )

    # -------------------------------------------------
    # 4️⃣ 单个 obs 分类图
    # -------------------------------------------------
    if len(color_list) == 1 and len(obs_colors) == 1:
        return plot_umap_obs(
            atlas=atlas,
            color=obs_colors[0],
            sample_n=sample_n,
            where=where,
            legend_loc=legend_loc,
            point_size=point_size,
            alpha=alpha,
            frameon=frameon,
            save_path=save_path,
            plot_batch_size=plot_batch_size,  # ✅【新增】
        )

    # -------------------------------------------------
    # 5️⃣ 纯 gene feature 图
    # -------------------------------------------------
    if len(obs_colors) == 0 and len(gene_colors) > 0:
        return plot_umap_features(
            atlas=atlas,
            genes=gene_colors,
            sample_n=sample_n,
            where=where,
            use_expr_field=use_expr_field,
            ncols=ncols,
            figsize=figsize,
            point_size=point_size,
            alpha=alpha
        )

    # -------------------------------------------------
    # 6️⃣ 混合模式
    # -------------------------------------------------
    result = {}

    for obs_col in obs_colors:
        result[obs_col] = plot_umap_obs(
            atlas=atlas,
            color=obs_col,
            sample_n=sample_n,
            where=where,
            legend_loc=legend_loc,
            point_size=point_size,
            alpha=alpha,
            frameon=frameon,
            save_path=None,
            plot_batch_size=plot_batch_size,  # ✅【新增】
        )

    if len(gene_colors) > 0:
        result["genes"] = plot_umap_features(
            atlas=atlas,
            genes=gene_colors,
            sample_n=sample_n,
            where=where,
            use_expr_field=use_expr_field,
            ncols=ncols,
            figsize=figsize,
            point_size=point_size,
            alpha=alpha
        )

    return result



#  UMAP 分类变量图
#  数据库版 Scanpy 风格 UMAP categorical plot
def plot_umap_obs(
        atlas,
        color: str = "kmeans",
        sample_n: int | None = 50000,
        groups: list | None = None,
        where: str | None = None,
        legend_loc: str = "right_margin",   # "right_margin" | "on_data"
        title: str | None = None,
        point_size: float = 8,
        alpha: float = 0.9,
        frameon: bool = False,
        save_path: str | None = None,
        # ✅【新增】全量绘图分批读取
        plot_batch_size: int = 200000
):
    """
    数据库版 Scanpy 风格 UMAP categorical plot
    支持按 obs 中任意分类列上色，例如：
        - kmeans
        - leiden
        - cell_type_auto
        - cell_type_auto_confidence

    参数
    ----
    atlas : Atlas
    color : str
        obs 中的分类列名
    sample_n : int | None
        抽样多少细胞；None 表示全量
    groups : list | None
        只显示指定类别
    where : str | None
        额外的 obs 过滤条件
    legend_loc : str
        "right_margin" -> 右侧图例
        "on_data"      -> 在簇中心直接标文字
    title : str | None
        图标题；None 时默认用 color
    point_size : float
    alpha : float
    frameon : bool
        是否保留坐标轴边框
    save_path : str | None

    返回
    ----
    plot_df : DataFrame
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    print(f"\n==== plot_umap_obs (color={color}) ====")
    conn = atlas.connection

    # -------------------------------------------------
    # 0️⃣ 检查表和列
    # -------------------------------------------------
    tables = conn.execute("""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_name IN ('obsm_X_umap', 'obs')
    """).fetchdf()["table_name"].tolist()

    if "obsm_X_umap" not in tables:
        raise ValueError("数据库中不存在 obsm_X_umap，请先运行 sap.tl.umap(atlas)")
    if "obs" not in tables:
        raise ValueError("数据库中不存在 obs")

    obs_cols = [r[1] for r in conn.execute("PRAGMA table_info(obs)").fetchall()]
    umap_cols = [r[1] for r in conn.execute("PRAGMA table_info(obsm_X_umap)").fetchall()]

    if color not in obs_cols:
        raise ValueError(f"obs 中不存在列: {color}")
    if "atlas_cell_id" not in obs_cols:
        raise ValueError("obs 中不存在 atlas_cell_id")
    if "atlas_cell_id" not in umap_cols or "umap1" not in umap_cols or "umap2" not in umap_cols:
        raise ValueError("obsm_X_umap 需要包含 atlas_cell_id / umap1 / umap2")

    # -------------------------------------------------
    # 1️⃣ 构造过滤条件
    # -------------------------------------------------
    where_clauses = [f"o.{color} IS NOT NULL"]

    if where is not None and str(where).strip() != "":
        where_clauses.append(f"({where})")

    if groups is not None:
        groups_sql = ", ".join([f"'{str(g)}'" for g in groups])
        where_clauses.append(f"CAST(o.{color} AS TEXT) IN ({groups_sql})")

    where_sql = " AND ".join(where_clauses)

    # -------------------------------------------------
    # 2️⃣ 取数据（可抽样）
    # -------------------------------------------------
    if sample_n is None:
        query = f"""
            SELECT
                u.atlas_cell_id,
                u.umap1,
                u.umap2,
                CAST(o.{color} AS TEXT) AS color_label
            FROM obsm_X_umap u
            JOIN obs o
              ON u.atlas_cell_id = o.atlas_cell_id
            WHERE {where_sql}
            ORDER BY u.atlas_cell_id
        """
    else:
        query = f"""
            SELECT *
            FROM (
                SELECT
                    u.atlas_cell_id,
                    u.umap1,
                    u.umap2,
                    CAST(o.{color} AS TEXT) AS color_label
                FROM obsm_X_umap u
                JOIN obs o
                  ON u.atlas_cell_id = o.atlas_cell_id
                WHERE {where_sql}
            ) t
            USING SAMPLE {int(sample_n)} ROWS
            ORDER BY atlas_cell_id
        """

    # ✅【新增】sample_n=None 时，走全量 streaming 绘图，避免一次性 fetchdf 爆内存
    if sample_n is None:
        return _draw_umap_obs_streaming(
            atlas=atlas,
            color=color,
            where_sql=where_sql,
            legend_loc=legend_loc,
            title=title,
            point_size=point_size,
            alpha=alpha,
            frameon=frameon,
            save_path=save_path,
            plot_batch_size=plot_batch_size
        )

    # ✅ sample_n 不是 None 时，仍然走原来的抽样绘图
    plot_df = conn.execute(query).fetchdf()

    if len(plot_df) == 0:
        raise ValueError("筛选后没有可绘制的细胞")

    # -------------------------------------------------
    # 3️⃣ 调色板
    # -------------------------------------------------
    # 尽量按数字排序；如果不是纯数字，再按字符串排序
    def _sort_label(x):
        try:
            return (0, int(x))
        except:
            return (1, str(x))

    unique_labels = sorted(
        plot_df["color_label"].astype(str).unique().tolist(),
        key=_sort_label
    )

    palette = []
    for cmap_name in ["tab20", "tab20b", "tab20c", "Set3", "Paired", "Accent", "Dark2"]:
        cmap = plt.get_cmap(cmap_name)
        if hasattr(cmap, "colors"):
            palette.extend(list(cmap.colors))

    if len(palette) < len(unique_labels):
        hsv = plt.get_cmap("hsv")
        palette.extend([hsv(i / max(len(unique_labels), 1)) for i in range(len(unique_labels))])

    palette = palette[:len(unique_labels)]
    label_to_color = {lab: palette[i] for i, lab in enumerate(unique_labels)}

    # -------------------------------------------------
    # 4️⃣ 作图
    # -------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.2, 5.8), facecolor="white")
    ax.set_facecolor("white")

    for lab in unique_labels:
        sub = plot_df[plot_df["color_label"] == lab]
        ax.scatter(
            sub["umap1"].to_numpy(),
            sub["umap2"].to_numpy(),
            s=point_size,
            alpha=alpha,
            c=[label_to_color[lab]],
            linewidths=0,
            label=str(lab)
        )

    # 标题
    if title is None:
        title = color
    ax.set_title(title, fontsize=18, weight="normal", pad=10)

    ax.set_xlabel("UMAP1", fontsize=16)
    ax.set_ylabel("UMAP2", fontsize=16)

    # 图例 / on-data 标签
    if legend_loc == "right_margin":
        legend_handles = [
            Line2D(
                [0], [0],
                marker="o",
                color="w",
                label=str(lab),
                markerfacecolor=label_to_color[lab],
                markersize=9
            )
            for lab in unique_labels
        ]
        ax.legend(
            handles=legend_handles,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=False,
            borderaxespad=0.0,
            handlelength=0.8,
            handletextpad=0.4,
            fontsize=11
        )

    elif legend_loc == "on_data":
        for lab in unique_labels:
            sub = plot_df[plot_df["color_label"] == lab]
            x_center = sub["umap1"].mean()
            y_center = sub["umap2"].mean()
            ax.text(
                x_center,
                y_center,
                str(lab),
                fontsize=12,
                weight="bold",
                ha="center",
                va="center"
            )
    else:
        raise ValueError("legend_loc 只能是 'right_margin' 或 'on_data'")

    # 样式
    ax.grid(False)
    if not frameon:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        ax.spines["left"].set_linewidth(1.0)
        ax.spines["bottom"].set_linewidth(1.0)
        ax.tick_params(axis="both", labelsize=11, width=1.0, length=4)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

    print("[UMAP] Done")
    return plot_df


def _draw_umap_obs_streaming(
        atlas,
        color: str,
        where_sql: str,
        legend_loc: str = "right_margin",
        title: str | None = None,
        point_size: float = 0.3,
        alpha: float = 0.5,
        frameon: bool = False,
        save_path: str | None = None,
        plot_batch_size: int = 200000
):
    """
    小内存全量 UMAP 分类图。

    sample_n=None 时使用：
    - 不一次性 fetchdf 全量
    - 每次只读取 plot_batch_size 行
    - 分批 scatter 到同一张图
    """

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    print("\n==== plot_umap_obs_streaming_full ====")

    conn = atlas.connection

    # -------------------------------------------------
    # 1️⃣ 先取全部类别，用于固定颜色
    # -------------------------------------------------
    label_df = conn.execute(f"""
        SELECT DISTINCT CAST(o.{color} AS TEXT) AS color_label
        FROM obsm_X_umap u
        JOIN obs o
          ON u.atlas_cell_id = o.atlas_cell_id
        WHERE {where_sql}
        ORDER BY color_label
    """).fetchdf()

    if len(label_df) == 0:
        raise ValueError("筛选后没有可绘制的细胞")

    unique_labels = label_df["color_label"].astype(str).tolist()

    # -------------------------------------------------
    # 2️⃣ 调色板
    # -------------------------------------------------
    palette = []

    for cmap_name in ["tab20", "tab20b", "tab20c", "Set3", "Paired", "Accent", "Dark2"]:
        cmap = plt.get_cmap(cmap_name)
        if hasattr(cmap, "colors"):
            palette.extend(list(cmap.colors))

    if len(palette) < len(unique_labels):
        hsv = plt.get_cmap("hsv")
        palette.extend([
            hsv(i / max(len(unique_labels), 1))
            for i in range(len(unique_labels))
        ])

    palette = palette[:len(unique_labels)]

    label_to_color = {
        lab: palette[i]
        for i, lab in enumerate(unique_labels)
    }

    # -------------------------------------------------
    # 3️⃣ 建图
    # -------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.0, 6.5), facecolor="white")
    ax.set_facecolor("white")

    # -------------------------------------------------
    # 4️⃣ 分批读取 + 分批画图
    # -------------------------------------------------
    last_cell_id = -1
    total_drawn = 0

    while True:

        batch_df = conn.execute(f"""
            SELECT
                u.atlas_cell_id,
                u.umap1,
                u.umap2,
                CAST(o.{color} AS TEXT) AS color_label
            FROM obsm_X_umap u
            JOIN obs o
              ON u.atlas_cell_id = o.atlas_cell_id
            WHERE {where_sql}
              AND u.atlas_cell_id > {int(last_cell_id)}
            ORDER BY u.atlas_cell_id
            LIMIT {int(plot_batch_size)}
        """).fetchdf()

        if len(batch_df) == 0:
            break

        last_cell_id = int(batch_df["atlas_cell_id"].iloc[-1])
        total_drawn += len(batch_df)

        for lab in unique_labels:
            sub = batch_df[batch_df["color_label"].astype(str) == lab]

            if len(sub) == 0:
                continue

            ax.scatter(
                sub["umap1"].to_numpy(),
                sub["umap2"].to_numpy(),
                s=point_size,
                alpha=alpha,
                c=[label_to_color[lab]],
                linewidths=0,
                rasterized=True
            )

        print(f"[UMAP streaming] drawn cells = {total_drawn:,}")

    # -------------------------------------------------
    # 5️⃣ 标题
    # -------------------------------------------------
    if title is None:
        title = color

    ax.set_title(title, fontsize=18, weight="normal", pad=10)
    ax.set_xlabel("UMAP1", fontsize=16)
    ax.set_ylabel("UMAP2", fontsize=16)

    # -------------------------------------------------
    # 6️⃣ 图例
    # -------------------------------------------------
    if legend_loc == "right_margin":
        legend_handles = [
            Line2D(
                [0], [0],
                marker="o",
                color="w",
                label=str(lab),
                markerfacecolor=label_to_color[lab],
                markersize=9
            )
            for lab in unique_labels
        ]

        ax.legend(
            handles=legend_handles,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=False,
            borderaxespad=0.0,
            handlelength=0.8,
            handletextpad=0.4,
            fontsize=11
        )

    elif legend_loc == "on_data":
        center_df = conn.execute(f"""
            SELECT
                CAST(o.{color} AS TEXT) AS color_label,
                AVG(u.umap1) AS x_center,
                AVG(u.umap2) AS y_center
            FROM obsm_X_umap u
            JOIN obs o
              ON u.atlas_cell_id = o.atlas_cell_id
            WHERE {where_sql}
            GROUP BY CAST(o.{color} AS TEXT)
        """).fetchdf()

        for _, row in center_df.iterrows():
            ax.text(
                row["x_center"],
                row["y_center"],
                str(row["color_label"]),
                fontsize=12,
                weight="bold",
                ha="center",
                va="center"
            )

    else:
        raise ValueError("legend_loc 只能是 'right_margin' 或 'on_data'")

    # -------------------------------------------------
    # 7️⃣ 样式
    # -------------------------------------------------
    ax.grid(False)

    if not frameon:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        ax.spines["left"].set_linewidth(1.0)
        ax.spines["bottom"].set_linewidth(1.0)
        ax.tick_params(axis="both", labelsize=11, width=1.0, length=4)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

    print(f"[UMAP streaming] Done, total drawn = {total_drawn:,}")

    return None



# UMAP 基因表达图
# 把已经算好的 UMAP 坐标，按多个基因表达值着色，画成 Scanpy 那种 sc.pl.umap(..., color=[...]) 风格图。
# UMAP 基因表达图
# 把已经算好的 UMAP 坐标，按多个基因表达值着色，
# 画成 Scanpy 那种 sc.pl.umap(..., color=[...]) 风格图。
def plot_umap_features(
        atlas,
        genes,
        sample_n: int | None = 50000,          # SQL先抽样，适合大数据
        where: str | None = None,              # ✅【新增】：SQL过滤条件
        use_expr_field: str = "data_scale",    # "data" / "data_log1p" / "data_scale"
        ncols: int = 3,
        figsize=None,
        point_size: float = 8,
        alpha: float = 0.9
):
    """
    数据库版 Scanpy 风格 UMAP feature plot

    支持：
    1. SQL 过滤 where
    2. SQL 抽样 sample_n
    3. 多基因 feature plot
    4. data_scale 缺失值自动补 zero_scale_transform
    """

    import math
    import pandas as pd
    import matplotlib.pyplot as plt
    from datetime import datetime

    print("\n==== plot_umap_features ====")
    start = datetime.now()
    conn = atlas.connection

    if isinstance(genes, str):
        genes = [genes]

    if len(genes) == 0:
        raise ValueError("genes 不能为空")

    print(f"[UMAP features] genes = {genes}")
    print(f"[UMAP features] sample_n = {sample_n}")
    if where is not None and str(where).strip() != "":
        print(f"[UMAP features] where = {where}")

    # -------------------------------------------------
    # 0️⃣ 检查表和列
    # -------------------------------------------------
    tables = conn.execute("""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_name IN ('obsm_X_umap', 'obs', 'var', 'X_CSRO_data')
    """).fetchdf()["table_name"].tolist()

    if "obsm_X_umap" not in tables:
        raise ValueError("数据库中不存在 obsm_X_umap，请先运行 sap.tl.umap(atlas)")
    if "obs" not in tables:
        raise ValueError("数据库中不存在 obs")
    if "var" not in tables:
        raise ValueError("数据库中不存在 var")
    if "X_CSRO_data" not in tables:
        raise ValueError("数据库中不存在 X_CSRO_data")

    umap_cols = [r[1] for r in conn.execute("PRAGMA table_info(obsm_X_umap)").fetchall()]
    obs_cols = [r[1] for r in conn.execute("PRAGMA table_info(obs)").fetchall()]
    x_cols = [r[1] for r in conn.execute("PRAGMA table_info(X_CSRO_data)").fetchall()]
    var_cols = [r[1] for r in conn.execute("PRAGMA table_info(var)").fetchall()]

    if "atlas_cell_id" not in umap_cols or "umap1" not in umap_cols or "umap2" not in umap_cols:
        raise ValueError("obsm_X_umap 需要包含 atlas_cell_id / umap1 / umap2")

    if "atlas_cell_id" not in obs_cols:
        raise ValueError("obs 中不存在 atlas_cell_id")

    if "atlas_cell_id" not in x_cols or "atlas_gene_id" not in x_cols:
        raise ValueError("X_CSRO_data 需要包含 atlas_cell_id / atlas_gene_id")

    if use_expr_field not in x_cols:
        raise ValueError(f"X_CSRO_data 中不存在字段: {use_expr_field}")

    if "atlas_gene_id" not in var_cols or "atlas_gene_name" not in var_cols:
        raise ValueError("var 需要包含 atlas_gene_id / atlas_gene_name")

    # -------------------------------------------------
    # 1️⃣ SQL 先过滤，再抽样 UMAP 细胞
    # -------------------------------------------------
    where_sql = ""

    if where is not None and str(where).strip() != "":
        # ✅ 这里允许用户写：
        # where="cell_type_auto_confidence IN ('high','medium')"
        # 或者：
        # where="o.cell_type_auto_confidence IN ('high','medium')"
        where_sql = f"WHERE {where}"

    if sample_n is None:
        umap_query = f"""
            SELECT
                u.atlas_cell_id,
                u.umap1,
                u.umap2
            FROM obsm_X_umap u
            JOIN obs o
              ON u.atlas_cell_id = o.atlas_cell_id
            {where_sql}
            ORDER BY u.atlas_cell_id
        """
    else:
        umap_query = f"""
            SELECT *
            FROM (
                SELECT
                    u.atlas_cell_id,
                    u.umap1,
                    u.umap2
                FROM obsm_X_umap u
                JOIN obs o
                  ON u.atlas_cell_id = o.atlas_cell_id
                {where_sql}
            ) t
            USING SAMPLE {int(sample_n)} ROWS
            ORDER BY atlas_cell_id
        """

    umap_df = conn.execute(umap_query).fetchdf()

    if len(umap_df) == 0:
        raise ValueError("筛选 / 抽样后没有可绘制的细胞")

    print(f"[UMAP features] plotted cells = {len(umap_df):,}")

    # -------------------------------------------------
    # 2️⃣ 查询 gene_id
    # -------------------------------------------------
    gene_name_sql = ", ".join([f"'{str(g)}'" for g in genes])

    if use_expr_field == "data_scale":
        if "zero_scale_transform" not in var_cols:
            raise ValueError(
                "var 中不存在 zero_scale_transform。\n"
                "请先运行 scale 流程写入 zero_scale_transform。"
            )

        gene_map_df = conn.execute(f"""
            SELECT
                atlas_gene_id,
                atlas_gene_name,
                zero_scale_transform
            FROM var
            WHERE atlas_gene_name IN ({gene_name_sql})
        """).fetchdf()

    else:
        gene_map_df = conn.execute(f"""
            SELECT
                atlas_gene_id,
                atlas_gene_name
            FROM var
            WHERE atlas_gene_name IN ({gene_name_sql})
        """).fetchdf()

    if len(gene_map_df) == 0:
        raise ValueError("var 中找不到这些基因")

    if use_expr_field == "data_scale":
        gene_map = {
            row["atlas_gene_name"]: (
                int(row["atlas_gene_id"]),
                float(row["zero_scale_transform"]) if pd.notna(row["zero_scale_transform"]) else 0.0
            )
            for _, row in gene_map_df.iterrows()
        }
    else:
        gene_map = {
            row["atlas_gene_name"]: int(row["atlas_gene_id"])
            for _, row in gene_map_df.iterrows()
        }

    missing_genes = [g for g in genes if g not in gene_map]
    if missing_genes:
        raise ValueError(f"var 中找不到这些基因: {missing_genes}")

    # -------------------------------------------------
    # 3️⃣ 注册抽样细胞临时表
    # -------------------------------------------------
    conn.register("_umap_cells_tmp", umap_df[["atlas_cell_id"]])

    # -------------------------------------------------
    # 4️⃣ 逐个 gene 取表达
    # -------------------------------------------------
    plot_data = {}

    for gene in genes:

        if use_expr_field == "data_scale":
            gene_id, zero_fill = gene_map[gene]

            expr_df = conn.execute(f"""
                SELECT
                    c.atlas_cell_id,
                    COALESCE(x.{use_expr_field}, {zero_fill}) AS expr
                FROM _umap_cells_tmp c
                LEFT JOIN X_CSRO_data x
                  ON c.atlas_cell_id = x.atlas_cell_id
                 AND x.atlas_gene_id = {gene_id}
            """).fetchdf()

        else:
            gene_id = gene_map[gene]

            expr_df = conn.execute(f"""
                SELECT
                    c.atlas_cell_id,
                    COALESCE(x.{use_expr_field}, 0.0) AS expr
                FROM _umap_cells_tmp c
                LEFT JOIN X_CSRO_data x
                  ON c.atlas_cell_id = x.atlas_cell_id
                 AND x.atlas_gene_id = {gene_id}
            """).fetchdf()

        df = umap_df.merge(expr_df, on="atlas_cell_id", how="left")

        if use_expr_field == "data_scale":
            _, zero_fill = gene_map[gene]
            df["expr"] = df["expr"].fillna(zero_fill)
        else:
            df["expr"] = df["expr"].fillna(0.0)

        # ✅ 高表达点后画，避免被低表达点盖住
        df = df.sort_values("expr", ascending=True).reset_index(drop=True)

        plot_data[gene] = df

    conn.unregister("_umap_cells_tmp")

    # -------------------------------------------------
    # 5️⃣ 自动布局
    # -------------------------------------------------
    n = len(genes)
    nrows = math.ceil(n / ncols)

    if figsize is None:
        figsize = (5.3 * ncols, 5.0 * nrows)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=figsize,
        facecolor="white"
    )

    if nrows == 1 and ncols == 1:
        axes = [[axes]]
    elif nrows == 1:
        axes = [axes]
    elif ncols == 1:
        axes = [[ax] for ax in axes]

    axes_flat = [ax for row in axes for ax in row]

    # -------------------------------------------------
    # 6️⃣ 作图
    # -------------------------------------------------
    for ax, gene in zip(axes_flat, genes):
        df = plot_data[gene]

        sc = ax.scatter(
            df["umap1"].to_numpy(),
            df["umap2"].to_numpy(),
            c=df["expr"].to_numpy(),
            cmap="viridis",
            s=point_size,
            alpha=alpha,
            linewidths=0
        )

        cbar = plt.colorbar(sc, ax=ax, pad=0.02, fraction=0.046)
        cbar.ax.tick_params(labelsize=10)

        ax.set_title(gene, fontsize=18, weight="normal", pad=10)
        ax.set_xlabel("UMAP1", fontsize=16)
        ax.set_ylabel("UMAP2", fontsize=16)

        ax.set_facecolor("white")
        ax.grid(False)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.spines["left"].set_linewidth(1.0)
        ax.spines["bottom"].set_linewidth(1.0)
        ax.tick_params(axis="both", labelsize=11, width=1.0, length=4)

    # 多余子图隐藏
    for ax in axes_flat[len(genes):]:
        ax.set_visible(False)

    plt.tight_layout(pad=1.0)
    plt.show()

    print(f"[UMAP features] Done in {(datetime.now() - start).total_seconds():.2f}s")

    return plot_data



# marker 排名图
# 数据库版 Scanpy 风格 rank_genes_groups 排名图（cluster vs. rest）
def plot_rank_genes_groups(
        atlas,
        groupby: str = "kmeans",
        use_expr_field: str = "data_log1p",
        n_genes: int = 25,
        mask_var: str | None = "highly_variable_genes",
        method: str = "t-test",   # "t-test" | "fusion"
        groups: list | None = None,
        reference: str | int | None = None,
        # ✅ 新增：None -> vs rest；否则 -> vs reference;
        #     groups=[0],
        #     reference=1,  指定：0 vs 1
        #     groups=[0, 2, 3],
        #     reference=1,  指定多个：0 vs 1, 2 vs 1, 3 vs 1
        save_path: str | None = None
):
    """
    数据库版 Scanpy 风格 rank_genes_groups 排名图

    支持两种模式
    ----------
    1) reference is None:
       group vs. rest
       例如：
           0 vs. rest
           1 vs. rest

    2) reference is not None:
       group vs. reference
       例如：
           0 vs. 1
           3 vs. 5

    参数
    ----
    atlas : Atlas
    groupby : str
        obs 中的分组列，例如 "kmeans" / "leiden"
    use_expr_field : str
        X_CSRO_data 中的表达字段，例如 "data_log1p"
    n_genes : int
        每个 group 展示前多少个 marker
    mask_var : str | None
        var 中的布尔列，例如 "highly_variable_genes"
        None 表示使用全部基因
    method : str
        "t-test" -> 仅按 t_like_score 排名
        "fusion" -> 融合 t_like_score + log2fc + pct_diff
    groups : list | None
        指定要分析哪些 group；None 表示全部
    reference : str | int | None
        None 表示 group vs. rest
        指定值表示 group vs. reference
    save_path : str | None
        保存图片路径

    返回
    ----
    result_dict : dict
        {
            group_id: DataFrame(marker ranking),
            ...
        }
    """

    import os
    import math
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from datetime import datetime

    print(f"\n==== plot_rank_genes_groups (method={method}, reference={reference}) ====")
    start = datetime.now()
    conn = atlas.connection

    # -------------------------------------------------
    # 0️⃣ 并行
    # -------------------------------------------------
    try:
        conn.execute(f"PRAGMA threads={os.cpu_count()}")
    except Exception:
        pass

    # -------------------------------------------------
    # 1️⃣ 检查列
    # -------------------------------------------------
    obs_cols = [r[1] for r in conn.execute("PRAGMA table_info(obs)").fetchall()]
    var_cols = [r[1] for r in conn.execute("PRAGMA table_info(var)").fetchall()]
    x_cols = [r[1] for r in conn.execute("PRAGMA table_info(X_CSRO_data)").fetchall()]

    if groupby not in obs_cols:
        raise ValueError(f"obs 中不存在列: {groupby}")

    if use_expr_field not in x_cols:
        raise ValueError(f"X_CSRO_data 中不存在字段: {use_expr_field}")

    if mask_var is not None and mask_var not in var_cols:
        raise ValueError(f"var 中不存在列: {mask_var}")

    # -------------------------------------------------
    # 2️⃣ group 信息
    # -------------------------------------------------
    group_df = conn.execute(f"""
        SELECT {groupby} AS grp, COUNT(*) AS n_cells
        FROM obs
        WHERE {groupby} IS NOT NULL
        GROUP BY 1
        ORDER BY 1
    """).fetchdf()

    if len(group_df) == 0:
        raise ValueError(f"obs.{groupby} 中没有可用分组")

    if groups is not None:
        group_df = group_df[group_df["grp"].isin(set(groups))].reset_index(drop=True)

    if len(group_df) == 0:
        raise ValueError("groups 过滤后没有可用 group")

    group_list = group_df["grp"].tolist()

    if reference is not None:
        if reference not in set(conn.execute(f"""
            SELECT DISTINCT {groupby}
            FROM obs
            WHERE {groupby} IS NOT NULL
        """).fetchnumpy()[groupby]):
            # 上面这个判断在某些类型下可能麻烦，所以再走一次稳妥逻辑
            ref_exist = conn.execute(f"""
                SELECT COUNT(*)
                FROM obs
                WHERE {groupby} IS NOT NULL
                  AND CAST({groupby} AS TEXT) = CAST('{reference}' AS TEXT)
            """).fetchone()[0]
            if ref_exist == 0:
                raise ValueError(f"reference={reference} 不在 obs.{groupby} 中")

        # 避免 group == reference
        group_list = [g for g in group_list if str(g) != str(reference)]
        if len(group_list) == 0:
            raise ValueError("groups 与 reference 去重后为空")

    print(f"-> groupby = {groupby}")
    print(f"-> groups = {group_list}")
    print(f"-> reference = {reference}")

    # -------------------------------------------------
    # 3️⃣ 候选基因集合
    # -------------------------------------------------
    var_where = "1=1"
    if mask_var is not None:
        var_where = f"COALESCE({mask_var}, FALSE)=TRUE"

    conn.execute("DROP TABLE IF EXISTS _rg_gene_set")
    conn.execute(f"""
        CREATE TEMP TABLE _rg_gene_set AS
        SELECT atlas_gene_id, atlas_gene_name
        FROM var
        WHERE {var_where}
    """)

    gene_count = conn.execute("SELECT COUNT(*) FROM _rg_gene_set").fetchone()[0]
    if gene_count == 0:
        raise ValueError("候选基因集合为空，请检查 mask_var 设置")

    print(f"-> candidate genes = {gene_count}")

    # -------------------------------------------------
    # 4️⃣ 各 group × gene 聚合统计
    #     一次性算好，后面避免重复扫大表
    # -------------------------------------------------
    conn.execute("DROP TABLE IF EXISTS _rg_group")
    conn.execute(f"""
        CREATE TEMP TABLE _rg_group AS
        SELECT
            o.{groupby} AS grp,
            x.atlas_gene_id,
            SUM(x.{use_expr_field}) AS sum_expr,
            SUM(x.{use_expr_field} * x.{use_expr_field}) AS sumsq_expr,
            COUNT(*) AS nnz
        FROM X_CSRO_data x
        JOIN obs o
            ON x.atlas_cell_id = o.atlas_cell_id
        JOIN _rg_gene_set gs
            ON x.atlas_gene_id = gs.atlas_gene_id
        WHERE o.{groupby} IS NOT NULL
        GROUP BY 1, 2
    """)

    # -------------------------------------------------
    # 5️⃣ 每个 group 的细胞数
    # -------------------------------------------------
    conn.execute("DROP TABLE IF EXISTS _rg_group_n")
    conn.execute(f"""
        CREATE TEMP TABLE _rg_group_n AS
        SELECT
            {groupby} AS grp,
            COUNT(*) AS n_cells
        FROM obs
        WHERE {groupby} IS NOT NULL
        GROUP BY 1
    """)

    # -------------------------------------------------
    # 6️⃣ 全局总和（仅用于 vs rest）
    # -------------------------------------------------
    if reference is None:
        total_cells = int(group_df["n_cells"].sum())

        conn.execute("DROP TABLE IF EXISTS _rg_total")
        conn.execute(f"""
            CREATE TEMP TABLE _rg_total AS
            SELECT
                gs.atlas_gene_id,
                gs.atlas_gene_name,
                COALESCE(SUM(x.{use_expr_field}), 0) AS sum_expr,
                COALESCE(SUM(x.{use_expr_field} * x.{use_expr_field}), 0) AS sumsq_expr,
                COUNT(x.{use_expr_field}) AS nnz
            FROM _rg_gene_set gs
            LEFT JOIN X_CSRO_data x
                ON gs.atlas_gene_id = x.atlas_gene_id
            LEFT JOIN obs o
                ON x.atlas_cell_id = o.atlas_cell_id
            WHERE o.{groupby} IS NOT NULL OR o.{groupby} IS NULL
            GROUP BY 1, 2
        """)

    # -------------------------------------------------
    # 7️⃣ 工具函数：z-score
    # -------------------------------------------------
    def _zscore(series: pd.Series) -> np.ndarray:
        x = series.values.astype(float)
        std = x.std()
        if std < 1e-9:
            return np.zeros_like(x)
        return (x - x.mean()) / (std + 1e-9)

    # -------------------------------------------------
    # 8️⃣ 逐 group 计算 marker ranking
    # -------------------------------------------------
    result_dict = {}

    for grp in group_list:

        grp_sql = f"'{grp}'" if isinstance(grp, str) else str(grp)

        if reference is None:
            # =================================================
            # 模式 A：group vs rest
            # =================================================
            df = conn.execute(f"""
                WITH base AS (
                    SELECT
                        t.atlas_gene_id,
                        t.atlas_gene_name,

                        COALESCE(g.sum_expr, 0) AS sum_in,
                        COALESCE(g.sumsq_expr, 0) AS sumsq_in,
                        COALESCE(g.nnz, 0) AS nnz_in,

                        t.sum_expr AS sum_total,
                        t.sumsq_expr AS sumsq_total,
                        t.nnz AS nnz_total,

                        n1.n_cells AS n_in
                    FROM _rg_total t
                    LEFT JOIN _rg_group g
                        ON t.atlas_gene_id = g.atlas_gene_id
                       AND g.grp = {grp_sql}
                    LEFT JOIN _rg_group_n n1
                        ON CAST(n1.grp AS TEXT) = CAST({grp_sql} AS TEXT)
                )
                SELECT
                    atlas_gene_id,
                    atlas_gene_name,

                    sum_in * 1.0 / NULLIF(n_in, 0) AS mean_in,
                    (sum_total - sum_in) * 1.0 / NULLIF({total_cells} - n_in, 0) AS mean_ref,

                    nnz_in * 1.0 / NULLIF(n_in, 0) AS pct_in,
                    (nnz_total - nnz_in) * 1.0 / NULLIF({total_cells} - n_in, 0) AS pct_ref,

                    CASE
                        WHEN n_in > 0 AND ({total_cells} - n_in) > 0 THEN
                            (
                                (sum_in * 1.0 / NULLIF(n_in, 0))
                                -
                                ((sum_total - sum_in) * 1.0 / NULLIF({total_cells} - n_in, 0))
                            )
                        ELSE 0
                    END AS t_like_score
                FROM base
            """).fetchdf()

            title_suffix = "rest"

        else:
            # =================================================
            # 模式 B：group vs reference
            # =================================================
            ref_sql = f"'{reference}'" if isinstance(reference, str) else str(reference)

            df = conn.execute(f"""
                WITH n_in AS (
                    SELECT n_cells AS n_in
                    FROM _rg_group_n
                    WHERE CAST(grp AS TEXT) = CAST({grp_sql} AS TEXT)
                ),
                n_ref AS (
                    SELECT n_cells AS n_ref
                    FROM _rg_group_n
                    WHERE CAST(grp AS TEXT) = CAST({ref_sql} AS TEXT)
                ),
                g_in AS (
                    SELECT
                        gs.atlas_gene_id,
                        gs.atlas_gene_name,
                        COALESCE(g.sum_expr, 0) AS sum_in,
                        COALESCE(g.sumsq_expr, 0) AS sumsq_in,
                        COALESCE(g.nnz, 0) AS nnz_in
                    FROM _rg_gene_set gs
                    LEFT JOIN _rg_group g
                        ON gs.atlas_gene_id = g.atlas_gene_id
                       AND CAST(g.grp AS TEXT) = CAST({grp_sql} AS TEXT)
                ),
                g_ref AS (
                    SELECT
                        gs.atlas_gene_id,
                        COALESCE(g.sum_expr, 0) AS sum_ref,
                        COALESCE(g.sumsq_expr, 0) AS sumsq_ref,
                        COALESCE(g.nnz, 0) AS nnz_ref
                    FROM _rg_gene_set gs
                    LEFT JOIN _rg_group g
                        ON gs.atlas_gene_id = g.atlas_gene_id
                       AND CAST(g.grp AS TEXT) = CAST({ref_sql} AS TEXT)
                )
                SELECT
                    i.atlas_gene_id,
                    i.atlas_gene_name,

                    i.sum_in * 1.0 / NULLIF(n_in.n_in, 0) AS mean_in,
                    r.sum_ref * 1.0 / NULLIF(n_ref.n_ref, 0) AS mean_ref,

                    i.nnz_in * 1.0 / NULLIF(n_in.n_in, 0) AS pct_in,
                    r.nnz_ref * 1.0 / NULLIF(n_ref.n_ref, 0) AS pct_ref,

                    CASE
                        WHEN n_in.n_in > 0 AND n_ref.n_ref > 0 THEN
                            (i.sum_in * 1.0 / NULLIF(n_in.n_in, 0))
                            -
                            (r.sum_ref * 1.0 / NULLIF(n_ref.n_ref, 0))
                        ELSE 0
                    END AS t_like_score
                FROM g_in i
                JOIN g_ref r
                    ON i.atlas_gene_id = r.atlas_gene_id
                CROSS JOIN n_in
                CROSS JOIN n_ref
            """).fetchdf()

            title_suffix = str(reference)

        if len(df) == 0:
            result_dict[grp] = df
            continue

        # -------------------------------------------------
        # 9️⃣ 统一后处理
        # -------------------------------------------------
        df["log2fc"] = np.log2((df["mean_in"] + 1e-9) / (df["mean_ref"] + 1e-9))
        df["pct_diff"] = df["pct_in"] - df["pct_ref"]

        # 只保留更像 marker 的正向基因
        df = df[
            (df["mean_in"] > df["mean_ref"]) &
            (df["pct_in"] >= df["pct_ref"])
        ].copy()

        if len(df) == 0:
            result_dict[grp] = df
            continue

        # -------------------------------------------------
        # 🔥 method 控制排序
        # -------------------------------------------------
        if method == "t-test":
            df = df.sort_values(
                ["t_like_score", "log2fc", "pct_diff"],
                ascending=False
            ).reset_index(drop=True)
            df["final_score"] = df["t_like_score"]

        elif method == "fusion":
            t_z = _zscore(df["t_like_score"])
            fc_z = _zscore(df["log2fc"])
            pct_z = _zscore(df["pct_diff"])

            df["final_score"] = (
                0.5 * t_z +
                0.3 * fc_z +
                0.2 * pct_z
            )

            df = df.sort_values(
                ["final_score", "t_like_score", "log2fc", "pct_diff"],
                ascending=False
            ).reset_index(drop=True)

        else:
            raise ValueError("method 必须是 't-test' 或 'fusion'")

        df["comparison"] = f"{grp} vs. {title_suffix}"
        result_dict[grp] = df

    # -------------------------------------------------
    # 🔟 作图
    # -------------------------------------------------
    n_panels = len(result_dict)
    if n_panels == 0:
        raise ValueError("没有可绘制的 group 结果")

    ncols = min(4, n_panels)
    nrows = math.ceil(n_panels / ncols)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.2 * ncols, 4.8 * nrows),
        facecolor="white"
    )

    axes = np.array(axes).reshape(-1)

    for ax, (grp, df) in zip(axes, result_dict.items()):
        plot_df = df.head(n_genes).copy()

        if len(plot_df) == 0:
            ax.set_title(f"{grp} vs. {title_suffix}")
            ax.text(0.5, 0.5, "No markers", ha="center", va="center")
            ax.set_axis_off()
            continue

        x = np.arange(len(plot_df))
        y = plot_df["final_score"].values

        ax.scatter(x, y, s=18, linewidths=0)

        for xi, yi, gene in zip(x, y, plot_df["atlas_gene_name"]):
            ax.text(xi, yi, gene, rotation=90, fontsize=9, ha="center", va="bottom")

        ax.set_title(plot_df["comparison"].iloc[0], fontsize=18)
        ax.set_xlabel("ranking", fontsize=16)
        ax.set_ylabel("score", fontsize=16)

        ax.set_facecolor("white")
        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for ax in axes[n_panels:]:
        ax.set_axis_off()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

    print(f"-> done in {(datetime.now() - start).total_seconds():.2f}s ✅")

    return result_dict


# marker violin
# rank_genes_groups 排名图 对应的  提琴图
def plot_rank_genes_groups_violin(
        atlas,
        group,
        groupby: str = "kmeans",
        reference: str | int | None = None,
        genes: list[str] | None = None,
        rank_result: dict | None = None,
        n_genes: int = 8,
        use_expr_field: str = "data_log1p",
        sample_n_per_group: int = 2000,
        save_path: str | None = None
):
    """
    数据库版 Scanpy 风格 rank_genes_groups_violin

    功能
    ----
    画某个 group 的 top marker genes 在：
        1) group vs. rest
        2) group vs. reference
    中的表达分布（violin）

    支持两种输入 gene 方式
    --------------------
    1. genes=None 且提供 rank_result
       -> 自动取 rank_result[group] 的前 n_genes 个 marker

    2. 显式传 genes
       -> 直接画指定基因

    参数
    ----
    atlas : Atlas
    groupby : str
        obs 中分组列，例如 kmeans / leiden
    group : str | int
        要看的目标 group，例如 0
    reference : str | int | None
        None -> group vs. rest
        指定值 -> group vs. reference
    genes : list[str] | None
        指定要画的基因；None 时自动从 rank_result 中取
    rank_result : dict | None
        plot_rank_genes_groups() 的返回结果
    n_genes : int
        当 genes=None 时，自动取前多少个 gene
    use_expr_field : str
        X_CSRO_data 中表达字段，例如 data_log1p
    sample_n_per_group : int
        每组最多抽样多少细胞来画图
    save_path : str | None
        保存路径

    返回
    ----
    plot_df : DataFrame
        长表，含 group_label / gene / expr
    """

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    print(f"\n==== plot_rank_genes_groups_violin (group={group}, reference={reference}) ====")
    conn = atlas.connection

    # -------------------------------------------------
    # 0️⃣ 检查列
    # -------------------------------------------------
    obs_cols = [r[1] for r in conn.execute("PRAGMA table_info(obs)").fetchall()]
    var_cols = [r[1] for r in conn.execute("PRAGMA table_info(var)").fetchall()]
    x_cols = [r[1] for r in conn.execute("PRAGMA table_info(X_CSRO_data)").fetchall()]

    if groupby not in obs_cols:
        raise ValueError(f"obs 中不存在列: {groupby}")
    if "atlas_cell_id" not in obs_cols:
        raise ValueError("obs 中不存在 atlas_cell_id")
    if "atlas_gene_id" not in var_cols or "atlas_gene_name" not in var_cols:
        raise ValueError("var 中不存在 atlas_gene_id / atlas_gene_name")
    if use_expr_field not in x_cols:
        raise ValueError(f"X_CSRO_data 中不存在字段: {use_expr_field}")

    # -------------------------------------------------
    # 1️⃣ gene 列表
    # -------------------------------------------------
    if genes is None:
        if rank_result is None:
            raise ValueError("genes=None 时，必须提供 rank_result")
        if group not in rank_result:
            raise ValueError(f"rank_result 中不存在 group={group}")

        rank_df = rank_result[group]
        if len(rank_df) == 0:
            raise ValueError(f"group={group} 在 rank_result 中为空")

        genes = rank_df["atlas_gene_name"].astype(str).head(n_genes).tolist()

    if isinstance(genes, str):
        genes = [genes]

    if len(genes) == 0:
        raise ValueError("genes 为空")

    print(f"-> genes = {genes}")

    # -------------------------------------------------
    # 2️⃣ gene_name -> gene_id
    # -------------------------------------------------
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

    # -------------------------------------------------
    # 3️⃣ 抽样目标细胞
    # -------------------------------------------------
    group_sql = f"'{group}'" if isinstance(group, str) else str(group)

    if reference is None:
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

    # -------------------------------------------------
    # 4️⃣ 注册采样细胞和基因
    # -------------------------------------------------
    conn.register("_violin_cells_tmp", cells_df)
    conn.register("_violin_genes_tmp", gene_map_df)

    # -------------------------------------------------
    # 5️⃣ 取表达长表（含隐式 0）
    # -------------------------------------------------
    plot_df = conn.execute(f"""
        SELECT
            c.group_label,
            g.atlas_gene_name AS gene,
            COALESCE(x.{use_expr_field}, 0.0) AS expr
        FROM _violin_cells_tmp c
        CROSS JOIN _violin_genes_tmp g
        LEFT JOIN X_CSRO_data x
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

    # -------------------------------------------------
    # 6️⃣ 作图
    # -------------------------------------------------
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

    # -------------------------------------------------
    # 7️⃣ 美化
    # -------------------------------------------------
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



# 普通 violin
# 验证自动注释是否靠谱， 数据库版 Scanpy 风格 violin plot
def violin(
        atlas,
        genes,
        groupby: str = "kmeans",
        use_expr_field: str = "data_log1p",
        sample_n_per_group: int | None = 2000,
        groups: list | None = None,
        where: str | None = None,
        order: list | None = None,
        save_path: str | None = None
):
    """
    数据库版 Scanpy 风格 violin plot

    功能
    ----
    比较一个或多个基因在不同 group 中的表达分布。
    可直接用于：
        - kmeans / leiden
        - annotate_clusters 写回的 cell_type_auto
        - 其他 obs 分类列

    参数
    ----
    atlas : Atlas
    genes : str | list[str]
        要画的基因名，例如 ["CST3", "NKG7", "PPBP"]
    groupby : str
        obs 中的分组列，例如 "kmeans" / "cell_type_auto"
    use_expr_field : str
        X_CSRO_data 中表达字段，例如 "data_log1p"
    sample_n_per_group : int | None
        每个 group 最多抽样多少细胞；None 表示不抽样
    groups : list | None
        只画指定 groups
    where : str | None
        obs 过滤条件，例如 "cell_type_auto_confidence = 'high'"
    order : list | None
        指定 group 显示顺序
    save_path : str | None
        保存路径

    返回
    ----
    plot_df : DataFrame
        长表：group_label / gene / expr
    """

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    print(f"\n==== violin (groupby={groupby}) ====")
    conn = atlas.connection

    # -------------------------------------------------
    # 0️⃣ 参数标准化
    # -------------------------------------------------
    if isinstance(genes, str):
        genes = [genes]
    genes = [str(g) for g in genes]

    if len(genes) == 0:
        raise ValueError("genes 不能为空")

    # -------------------------------------------------
    # 1️⃣ 检查列
    # -------------------------------------------------
    obs_cols = [r[1] for r in conn.execute("PRAGMA table_info(obs)").fetchall()]
    var_cols = [r[1] for r in conn.execute("PRAGMA table_info(var)").fetchall()]
    x_cols = [r[1] for r in conn.execute("PRAGMA table_info(X_CSRO_data)").fetchall()]

    if groupby not in obs_cols:
        raise ValueError(f"obs 中不存在列: {groupby}")
    if "atlas_cell_id" not in obs_cols:
        raise ValueError("obs 中不存在 atlas_cell_id")
    if "atlas_gene_id" not in var_cols or "atlas_gene_name" not in var_cols:
        raise ValueError("var 中不存在 atlas_gene_id / atlas_gene_name")
    if use_expr_field not in x_cols:
        raise ValueError(f"X_CSRO_data 中不存在字段: {use_expr_field}")

    # -------------------------------------------------
    # 2️⃣ gene_name -> gene_id
    # -------------------------------------------------
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

    # -------------------------------------------------
    # 3️⃣ 准备 group 抽样细胞
    # -------------------------------------------------
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

    if order is not None:
        wanted = [str(x) for x in order]
        group_df = group_df[group_df["group_label"].isin(wanted)].copy()
        if len(group_df) == 0:
            raise ValueError("order 过滤后没有可用 group")
        group_df["order_idx"] = group_df["group_label"].map({g: i for i, g in enumerate(wanted)})
        group_df = group_df.sort_values("order_idx").drop(columns="order_idx").reset_index(drop=True)

    group_labels = group_df["group_label"].astype(str).tolist()
    print(f"-> groups = {group_labels}")

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

    # -------------------------------------------------
    # 4️⃣ 注册临时表
    # -------------------------------------------------
    conn.register("_violin_cells_tmp", cells_df)
    conn.register("_violin_genes_tmp", gene_map_df[["atlas_gene_id", "atlas_gene_name"]])

    # -------------------------------------------------
    # 5️⃣ 取表达长表（补隐式 0）
    # -------------------------------------------------
    plot_df = conn.execute(f"""
        SELECT
            c.group_label,
            g.atlas_gene_name AS gene,
            COALESCE(x.{use_expr_field}, 0.0) AS expr
        FROM _violin_cells_tmp c
        CROSS JOIN _violin_genes_tmp g
        LEFT JOIN X_CSRO_data x
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

    # -------------------------------------------------
    # 6️⃣ 作图
    # -------------------------------------------------
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

    return plot_df



# 热图
def dotplot(
        atlas,
        genes,
        groupby: str = "cell_type_auto",
        use_expr_field: str = "data_log1p",
        sample_n_per_group: int | None = 2000,
        groups: list | None = None,
        where: str | None = None,
        order: list | None = None,
        expression_cutoff: float = 0.0,
        standard_scale: str | None = None,   # None | "var"
        colorbar_vmin: float | None = 0.0,
        colorbar_vmax: float | None = 5.0,
        font_size: int = 14,
        save_path: str | None = None
):
    """
    更接近 Scanpy 风格的 dotplot（最终版）

    点大小  -> Fraction of cells in group (%)
    点颜色  -> Mean expression in group
    """

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from matplotlib.colorbar import ColorbarBase

    conn = atlas.connection

    # -------------------------------------------------
    # 0️⃣ 参数标准化
    # -------------------------------------------------
    if isinstance(genes, str):
        genes = [genes]
    genes = [str(g) for g in genes]
    if len(genes) == 0:
        raise ValueError("genes 不能为空")

    # -------------------------------------------------
    # 1️⃣ 检查列
    # -------------------------------------------------
    obs_cols = [r[1] for r in conn.execute("PRAGMA table_info(obs)").fetchall()]
    var_cols = [r[1] for r in conn.execute("PRAGMA table_info(var)").fetchall()]
    x_cols = [r[1] for r in conn.execute("PRAGMA table_info(X_CSRO_data)").fetchall()]

    if groupby not in obs_cols:
        raise ValueError(f"obs 中不存在列: {groupby}")
    if "atlas_cell_id" not in obs_cols:
        raise ValueError("obs 中不存在 atlas_cell_id")
    if "atlas_gene_id" not in var_cols or "atlas_gene_name" not in var_cols:
        raise ValueError("var 中不存在 atlas_gene_id / atlas_gene_name")
    if use_expr_field not in x_cols:
        raise ValueError(f"X_CSRO_data 中不存在字段: {use_expr_field}")

    # -------------------------------------------------
    # 2️⃣ gene_name -> gene_id
    # -------------------------------------------------
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

    # -------------------------------------------------
    # 3️⃣ 构造 where
    # -------------------------------------------------
    where_clauses = [f"{groupby} IS NOT NULL"]

    if where is not None and str(where).strip() != "":
        where_clauses.append(f"({where})")

    if groups is not None:
        groups_sql = ", ".join([f"'{str(g)}'" for g in groups])
        where_clauses.append(f"CAST({groupby} AS TEXT) IN ({groups_sql})")

    where_sql = " AND ".join(where_clauses)

    # -------------------------------------------------
    # 4️⃣ group 列表
    # -------------------------------------------------
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

    if order is not None:
        wanted = [str(x) for x in order]
        group_df = group_df[group_df["group_label"].isin(wanted)].copy()
        if len(group_df) == 0:
            raise ValueError("order 过滤后没有可用 group")
        group_df["order_idx"] = group_df["group_label"].map({g: i for i, g in enumerate(wanted)})
        group_df = group_df.sort_values("order_idx").drop(columns="order_idx").reset_index(drop=True)

    group_labels = group_df["group_label"].astype(str).tolist()

    # -------------------------------------------------
    # 5️⃣ 每个 group 抽样细胞
    # -------------------------------------------------
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

    # -------------------------------------------------
    # 6️⃣ 注册临时表
    # -------------------------------------------------
    conn.register("_dotplot_cells_tmp", cells_df)
    conn.register("_dotplot_genes_tmp", gene_map_df[["atlas_gene_id", "atlas_gene_name"]])

    # -------------------------------------------------
    # 7️⃣ 取表达长表（补隐式 0）
    # -------------------------------------------------
    expr_df = conn.execute(f"""
        SELECT
            c.group_label,
            g.atlas_gene_name AS gene,
            COALESCE(x.{use_expr_field}, 0.0) AS expr
        FROM _dotplot_cells_tmp c
        CROSS JOIN _dotplot_genes_tmp g
        LEFT JOIN X_CSRO_data x
            ON c.atlas_cell_id = x.atlas_cell_id
           AND g.atlas_gene_id = x.atlas_gene_id
    """).fetchdf()

    conn.unregister("_dotplot_cells_tmp")
    conn.unregister("_dotplot_genes_tmp")

    if len(expr_df) == 0:
        raise ValueError("expr_df 为空，无法作图")

    expr_df["gene"] = pd.Categorical(expr_df["gene"], categories=genes, ordered=True)
    expr_df["group_label"] = pd.Categorical(expr_df["group_label"], categories=group_labels, ordered=True)

    # -------------------------------------------------
    # 8️⃣ 聚合统计
    # -------------------------------------------------
    stat_df = (
        expr_df
        .groupby(["group_label", "gene"], observed=True)
        .agg(
            mean_expr=("expr", "mean"),
            pct_expr=("expr", lambda x: (x > expression_cutoff).mean() * 100.0),
            n=("expr", "size")
        )
        .reset_index()
    )

    if standard_scale == "var":
        stat_df["mean_expr_scaled"] = (
            stat_df.groupby("gene", observed=True)["mean_expr"]
            .transform(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-12))
        )
        color_col = "mean_expr_scaled"
    else:
        color_col = "mean_expr"

    # -------------------------------------------------
    # 9️⃣ 布局：整体放大 + 右侧更松弛
    # -------------------------------------------------
    n_genes = len(genes)
    n_groups = len(group_labels)

    main_w = max(7.6, 0.46 * n_genes + 3.0)
    main_h = max(4.2, 0.50 * n_groups + 1.8)

    right_w = 3.0 if n_genes <= 8 else 3.1 if n_genes <= 16 else 3.3

    fig_w = main_w + right_w
    fig_h = main_h + 1.0

    # 左边距：控制不要太空，也不要截断
    max_y_len = max(len(str(x)) for x in group_labels) if len(group_labels) > 0 else 10
    left_margin = min(0.24, max(0.09, 0.035 + 0.0085 * max_y_len))

    # 底部边距：gene 名显示完整
    max_gene_len = max(len(str(x)) for x in genes) if len(genes) > 0 else 10
    bottom_margin = min(0.28, max(0.18, 0.06 + 0.009 * max_gene_len))

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
    gs = fig.add_gridspec(
        nrows=1, ncols=2,
        width_ratios=[main_w, right_w],
        wspace=0.03
    )

    ax = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[0, 1])

    ax.set_facecolor("white")
    ax_right.set_facecolor("white")
    ax_right.axis("off")

    # -------------------------------------------------
    # 🔟 主图
    # -------------------------------------------------
    gene_to_x = {g: i for i, g in enumerate(genes)}
    group_to_y = {g: i for i, g in enumerate(group_labels)}

    x = stat_df["gene"].astype(str).map(gene_to_x).to_numpy()
    y = stat_df["group_label"].astype(str).map(group_to_y).to_numpy()

    pct = stat_df["pct_expr"].to_numpy()
    colors = stat_df[color_col].to_numpy()

    # ✅ 小点更小，大点更大
    sizes = 4.0 + (pct / 100.0) ** 1.8 * 700.0

    ax.scatter(
        x,
        y,
        s=sizes,
        c=colors,
        cmap="Reds",
        edgecolors="#777777",
        linewidths=0.25
    )

    ax.set_xticks(np.arange(n_genes))
    ax.set_xticklabels(genes, rotation=90, fontsize=font_size)
    ax.set_yticks(np.arange(n_groups))
    ax.set_yticklabels(group_labels, fontsize=font_size)

    ax.set_xlim(-0.55, n_genes - 0.45)
    ax.set_ylim(-0.38, n_groups - 0.62)
    ax.invert_yaxis()

    ax.tick_params(axis="x", pad=2)
    ax.grid(False)

    # 黑色外框
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
        spine.set_color("black")

    # -------------------------------------------------
    # 1️⃣1️⃣ 右侧上部：Fraction legend（修复不拥挤）
    # -------------------------------------------------
    ax_size = ax_right.inset_axes([0.06, 0.50, 0.88, 0.28])
    ax_size.set_xlim(0, 1)
    ax_size.set_ylim(0, 1)
    ax_size.axis("off")

    ax_size.text(
        0.50, 0.98,
        "Fraction of cells\nin group (%)",
        ha="center", va="top",
        fontsize=font_size,
        transform=ax_size.transAxes
    )

    size_levels = [20, 40, 60, 80, 100]
    x_positions = np.linspace(0.14, 0.86, len(size_levels))

    # ✅ 标题、圆点、刻度线、数字再拉开一点
    y_circle = 0.44
    y_tick_top = 0.30
    y_tick_bottom = 0.22
    y_text = 0.08

    for x0, p in zip(x_positions, size_levels):
        s = 4.0 + (p / 100.0) ** 1.8 * 700.0

        ax_size.scatter(
            [x0], [y_circle],
            s=s,
            color="gray",
            edgecolors="none",
            transform=ax_size.transAxes
        )

        ax_size.plot(
            [x0, x0],
            [y_tick_top, y_tick_bottom],
            color="black",
            linewidth=1.0,
            transform=ax_size.transAxes
        )

        ax_size.text(
            x0, y_text,
            f"{p}",
            ha="center", va="center",
            fontsize=font_size,
            transform=ax_size.transAxes
        )

    # -------------------------------------------------
    # 1️⃣2️⃣ 右侧下部：Mean expression legend（拉开）
    # -------------------------------------------------
    ax_cbar_box = ax_right.inset_axes([0.06, 0.12, 0.88, 0.18])
    ax_cbar_box.set_xlim(0, 1)
    ax_cbar_box.set_ylim(0, 1)
    ax_cbar_box.axis("off")

    ax_cbar_box.text(
        0.50, 0.98,
        "Mean expression\nin group",
        ha="center", va="top",
        fontsize=font_size,
        transform=ax_cbar_box.transAxes
    )

    cbar_ax = ax_cbar_box.inset_axes([0.12, 0.16, 0.76, 0.12])

    if colorbar_vmin is None:
        vmin = float(np.nanmin(colors))
    else:
        vmin = float(colorbar_vmin)

    if colorbar_vmax is None:
        vmax = float(np.nanmax(colors))
    else:
        vmax = float(colorbar_vmax)

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    cb = ColorbarBase(
        cbar_ax,
        cmap=plt.get_cmap("Reds"),
        norm=norm,
        orientation="horizontal"
    )
    cb.ax.tick_params(labelsize=font_size, length=4, width=1.0)

    # -------------------------------------------------
    # 1️⃣3️⃣ 边距
    # -------------------------------------------------
    fig.subplots_adjust(
        left=left_margin,
        right=0.98,
        top=0.95,
        bottom=bottom_margin
    )

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

    return stat_df




# 数据库版 Scanpy 风格 stacked_violin
def stacked_violin(
        atlas,
        genes,
        groupby: str = "cell_type_auto",
        use_expr_field: str = "data_log1p",
        sample_n_per_group: int | None = 2000,
        groups: list | None = None,
        where: str | None = None,
        order: list | None = None,
        color_vmin: float | None = 0.0,
        color_vmax: float | None = 5.0,
        font_size: int = 14,
        save_path: str | None = None
):
    """
    数据库版 Scanpy 风格 stacked_violin

    功能
    ----
    - 横轴：genes
    - 纵轴：groups / cell types
    - 每个格子内画一个“小提琴”，表示该 group 中该 gene 的表达分布
    - 颜色表示该 group 中该 gene 的 median expression
    """

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from matplotlib.colorbar import ColorbarBase
    from scipy.stats import gaussian_kde

    conn = atlas.connection

    # -------------------------------------------------
    # 0️⃣ 参数标准化
    # -------------------------------------------------
    if isinstance(genes, str):
        genes = [genes]
    genes = [str(g) for g in genes]

    if len(genes) == 0:
        raise ValueError("genes 不能为空")

    # -------------------------------------------------
    # 1️⃣ 检查列
    # -------------------------------------------------
    obs_cols = [r[1] for r in conn.execute("PRAGMA table_info(obs)").fetchall()]
    var_cols = [r[1] for r in conn.execute("PRAGMA table_info(var)").fetchall()]
    x_cols = [r[1] for r in conn.execute("PRAGMA table_info(X_CSRO_data)").fetchall()]

    if groupby not in obs_cols:
        raise ValueError(f"obs 中不存在列: {groupby}")
    if "atlas_cell_id" not in obs_cols:
        raise ValueError("obs 中不存在 atlas_cell_id")
    if "atlas_gene_id" not in var_cols or "atlas_gene_name" not in var_cols:
        raise ValueError("var 中不存在 atlas_gene_id / atlas_gene_name")
    if use_expr_field not in x_cols:
        raise ValueError(f"X_CSRO_data 中不存在字段: {use_expr_field}")

    # -------------------------------------------------
    # 2️⃣ gene_name -> gene_id
    # -------------------------------------------------
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

    # -------------------------------------------------
    # 3️⃣ 构造 where
    # -------------------------------------------------
    where_clauses = [f"{groupby} IS NOT NULL"]

    if where is not None and str(where).strip() != "":
        where_clauses.append(f"({where})")

    if groups is not None:
        groups_sql = ", ".join([f"'{str(g)}'" for g in groups])
        where_clauses.append(f"CAST({groupby} AS TEXT) IN ({groups_sql})")

    where_sql = " AND ".join(where_clauses)

    # -------------------------------------------------
    # 4️⃣ group 列表
    # -------------------------------------------------
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

    if order is not None:
        wanted = [str(x) for x in order]
        group_df = group_df[group_df["group_label"].isin(wanted)].copy()
        if len(group_df) == 0:
            raise ValueError("order 过滤后没有可用 group")
        group_df["order_idx"] = group_df["group_label"].map({g: i for i, g in enumerate(wanted)})
        group_df = group_df.sort_values("order_idx").drop(columns="order_idx").reset_index(drop=True)

    group_labels = group_df["group_label"].astype(str).tolist()

    # -------------------------------------------------
    # 5️⃣ 每个 group 抽样细胞
    # -------------------------------------------------
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

    # -------------------------------------------------
    # 6️⃣ 注册临时表
    # -------------------------------------------------
    conn.register("_sv_cells_tmp", cells_df)
    conn.register("_sv_genes_tmp", gene_map_df[["atlas_gene_id", "atlas_gene_name"]])

    # -------------------------------------------------
    # 7️⃣ 取表达长表（补隐式 0）
    # -------------------------------------------------
    expr_df = conn.execute(f"""
        SELECT
            c.group_label,
            g.atlas_gene_name AS gene,
            COALESCE(x.{use_expr_field}, 0.0) AS expr
        FROM _sv_cells_tmp c
        CROSS JOIN _sv_genes_tmp g
        LEFT JOIN X_CSRO_data x
            ON c.atlas_cell_id = x.atlas_cell_id
           AND g.atlas_gene_id = x.atlas_gene_id
    """).fetchdf()

    conn.unregister("_sv_cells_tmp")
    conn.unregister("_sv_genes_tmp")

    if len(expr_df) == 0:
        raise ValueError("expr_df 为空，无法作图")

    expr_df["gene"] = pd.Categorical(expr_df["gene"], categories=genes, ordered=True)
    expr_df["group_label"] = pd.Categorical(expr_df["group_label"], categories=group_labels, ordered=True)

    # -------------------------------------------------
    # 8️⃣ 中位数统计（用于着色）
    # -------------------------------------------------
    median_df = (
        expr_df
        .groupby(["group_label", "gene"], observed=True)["expr"]
        .median()
        .reset_index(name="median_expr")
    )

    # -------------------------------------------------
    # 9️⃣ 布局
    # -------------------------------------------------
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

    # -------------------------------------------------
    # 🔟 颜色映射
    # -------------------------------------------------
    if color_vmin is None:
        color_vmin = float(median_df["median_expr"].min())
    if color_vmax is None:
        color_vmax = float(median_df["median_expr"].max())

    norm = mpl.colors.Normalize(vmin=color_vmin, vmax=color_vmax)
    cmap = plt.get_cmap("Blues")

    # -------------------------------------------------
    # 1️⃣1️⃣ 画 stacked violin
    # 每个格子一个“迷你小提琴”
    # -------------------------------------------------
    gene_to_x = {g: i for i, g in enumerate(genes)}

    # ✅ 修正上下方向：直接按 scanpy 风格从上到下排，不再 invert_yaxis
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

    # -------------------------------------------------
    # 1️⃣2️⃣ 主图美化
    # -------------------------------------------------
    ax.set_xticks(np.arange(n_genes))
    ax.set_xticklabels(genes, rotation=90, fontsize=font_size)

    # ✅ y tick 也按 scanpy 风格从上到下显示
    y_tick_positions = [group_to_y[g] for g in group_labels]
    ax.set_yticks(y_tick_positions)
    ax.set_yticklabels(group_labels, fontsize=font_size)

    ax.set_xlim(-0.55, n_genes - 0.45)
    ax.set_ylim(-0.5, n_groups - 0.5)

    # ❌ 不再翻转
    # ax.invert_yaxis()

    # 行分隔线
    for j in range(n_groups):
        ax.axhline(j + 0.5, color="#d0d0d0", linewidth=1.0, zorder=0)

    ax.grid(False)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
        spine.set_color("black")

    # -------------------------------------------------
    # 1️⃣3️⃣ 右侧色条说明
    # -------------------------------------------------
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

    # -------------------------------------------------
    # 1️⃣4️⃣ 边距
    # -------------------------------------------------
    fig.subplots_adjust(
        left=left_margin,
        right=0.98,
        top=0.95,
        bottom=bottom_margin
    )

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

    return expr_df, median_df

