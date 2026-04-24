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
import numpy as np
import pandas as pd
import umap as umap_lib
from ..data import Atlas

# 数据库版 UMAP embedding（Scanpy风格 + 大数据友好 + SQL落库）
def umap(
        atlas: Atlas,
        fit_sample_n: int | None = 50000,
        transform_batch_size: int = 50000,
        n_components: int = 2,
        n_neighbors: int = 15,
        min_dist: float = 0.5,
        metric: str = "euclidean",
        random_state: int = 42,
        n_jobs: int = 1,
        table_name: str = "obsm_X_umap",
        save_params_table: str = "uns_umap_params"
):
    """
    UMAP 计算入口，对齐 Scanpy 的 tl.umap 风格。

    功能
    ----
    1. 从 obsm_X_pca 读取 PCA embedding
    2. SQL 先抽样拟合 UMAP
    3. 对全量 PCA embedding 分块 transform
    4. 写入：
       - obsm_X_umap
       - uns_umap_params

    前提
    ----
    请先运行：
        sap.tl.pca(atlas)

    用法
    ----
    sap.tl.umap(atlas)

    sap.tl.umap(
        atlas,
        fit_sample_n=50000,
        n_neighbors=15,
        min_dist=0.5
    )
    """

    print("\n==== sap.tl.umap ====")
    start = datetime.now()
    conn = atlas.connection

    # -------------------------------------------------
    # 0️⃣ 检查 obsm_X_pca
    # -------------------------------------------------
    tables = conn.execute("""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_name = 'obsm_X_pca'
    """).fetchdf()

    if len(tables) == 0:
        raise ValueError(
            "数据库中不存在 obsm_X_pca。\n"
            "请先运行 sap.tl.pca(atlas)"
        )

    pca_cols = [r[1] for r in conn.execute("PRAGMA table_info(obsm_X_pca)").fetchall()]
    pc_cols = [c for c in pca_cols if c.startswith("pc")]

    if len(pc_cols) == 0:
        raise ValueError("obsm_X_pca 中不存在 pc 列")

    # ✅ 保证 pc0, pc1, pc2 ... 按数字顺序排列
    pc_cols = sorted(pc_cols, key=lambda x: int(x.replace("pc", "")))

    pc_cols_sql = ", ".join(pc_cols)

    # -------------------------------------------------
    # 1️⃣ 拟合 UMAP：SQL 先抽样
    # -------------------------------------------------
    if fit_sample_n is None:
        fit_query = f"""
            SELECT cell_id, {pc_cols_sql}
            FROM obsm_X_pca
            ORDER BY cell_id
        """
    else:
        fit_query = f"""
            SELECT cell_id, {pc_cols_sql}
            FROM obsm_X_pca
            USING SAMPLE {int(fit_sample_n)} ROWS
        """

    fit_df = conn.execute(fit_query).fetchdf()

    if len(fit_df) == 0:
        raise ValueError("拟合 UMAP 的样本为空")

    X_fit = fit_df.drop(columns=["cell_id"]).to_numpy(dtype=np.float32)

    print(f"[UMAP] fit sample shape = {X_fit.shape}")

    # -------------------------------------------------
    # 2️⃣ 拟合 UMAP 模型
    # -------------------------------------------------
    reducer = umap_lib.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
        n_jobs=n_jobs
    )

    reducer.fit(X_fit)

    # -------------------------------------------------
    # 3️⃣ 建输出表
    # -------------------------------------------------
    conn.execute(f"DROP TABLE IF EXISTS {table_name}")
    conn.execute(f"""
        CREATE TABLE {table_name} (
            cell_id BIGINT,
            umap1 FLOAT,
            umap2 FLOAT
        )
    """)

    conn.execute(f"DROP TABLE IF EXISTS {save_params_table}")
    conn.execute(f"""
        CREATE TABLE {save_params_table} (
            param_name VARCHAR,
            param_value VARCHAR
        )
    """)

    params_df = pd.DataFrame({
        "param_name": [
            "n_components",
            "n_neighbors",
            "min_dist",
            "metric",
            "random_state",
            "fit_sample_n",
            "transform_batch_size",
            "input_table",
            "output_table"
        ],
        "param_value": [
            str(n_components),
            str(n_neighbors),
            str(min_dist),
            str(metric),
            str(random_state),
            str(fit_sample_n),
            str(transform_batch_size),
            "obsm_X_pca",
            table_name
        ]
    })

    conn.append(save_params_table, params_df)

    # -------------------------------------------------
    # 4️⃣ 全量分块 transform
    # -------------------------------------------------
    total_n = conn.execute("SELECT COUNT(*) FROM obsm_X_pca").fetchone()[0]
    print(f"[UMAP] total cells = {total_n}")

    offset = 0

    while offset < total_n:
        batch_df = conn.execute(f"""
            SELECT cell_id, {pc_cols_sql}
            FROM obsm_X_pca
            ORDER BY cell_id
            LIMIT {int(transform_batch_size)}
            OFFSET {int(offset)}
        """).fetchdf()

        if len(batch_df) == 0:
            break

        X_batch = batch_df.drop(columns=["cell_id"]).to_numpy(dtype=np.float32)
        X_umap = reducer.transform(X_batch).astype(np.float32)

        out_df = pd.DataFrame({
            "cell_id": batch_df["cell_id"].to_numpy(dtype=np.int64),
            "umap1": X_umap[:, 0],
            "umap2": X_umap[:, 1]
        })

        conn.append(table_name, out_df)

        offset += len(batch_df)
        print(f"[UMAP] transformed {offset}/{total_n}")

    print("[UMAP] Done")
    print(f"耗时: {(datetime.now() - start).total_seconds():.2f} 秒")

    return reducer

