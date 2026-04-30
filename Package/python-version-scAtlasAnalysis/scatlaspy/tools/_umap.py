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


import umap as umap_lib
from sklearn.manifold import trustworthiness
from sklearn.neighbors import NearestNeighbors


# ============================================
# 评估函数：KNN overlap
# ============================================
def knn_overlap(X_high, X_low, k=15):
    """
    比较高维空间和低维UMAP空间的近邻重叠率。

    X_high: PCA空间，例如 X_fit
    X_low : UMAP空间，例如 X_fit_umap
    k     : 近邻数量

    返回：
        overlap score，范围大约 0~1，越高越好
    """

    n = X_high.shape[0]

    # 防止样本太少
    k = min(k, n - 1)

    nn_high = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nn_low = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")

    nn_high.fit(X_high)
    nn_low.fit(X_low)

    high_idx = nn_high.kneighbors(X_high, return_distance=False)[:, 1:]
    low_idx = nn_low.kneighbors(X_low, return_distance=False)[:, 1:]

    overlap = []

    for i in range(n):
        s1 = set(high_idx[i])
        s2 = set(low_idx[i])
        overlap.append(len(s1 & s2) / k)

    return float(np.mean(overlap))


# ============================================
# 数据库版 UMAP embedding
# Scanpy风格 + 大数据友好 + SQL落库 + 评估
# ============================================
def umap(
        atlas,
        fit_sample_n: int | None = 50000,
        transform_batch_size: int = 50000,
        n_components: int = 2,
        n_neighbors: int = 15,
        min_dist: float = 0.5,
        metric: str = "euclidean",
        random_state: int = 42,
        n_jobs: int = 1,
        table_name: str = "obsm_X_umap",
        save_params_table: str = "uns_umap_params",

        # ✅【新增】评估参数
        eval_sample_n: int = 5000,
        save_eval_table: str = "uns_umap_eval"
):
    """
    UMAP 计算入口，对齐 Scanpy 的 tl.umap 风格。

    功能
    ----
    1. 从 obsm_X_pca 读取 PCA embedding
    2. SQL 先抽样拟合 UMAP
    3. 训练后计算 UMAP embedding 质量指标：
       - trustworthiness
       - knn_overlap
    4. 对全量 PCA embedding 分块 transform
    5. 写入：
       - obsm_X_umap
       - uns_umap_params
       - uns_umap_eval

    前提
    ----
    请先运行：
        sap.tl.pca(atlas)
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

    pca_cols = [
        r[1]
        for r in conn.execute("PRAGMA table_info(obsm_X_pca)").fetchall()
    ]

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
            SELECT atlas_cell_id, {pc_cols_sql}
            FROM obsm_X_pca
            ORDER BY atlas_cell_id
        """
    else:
        fit_query = f"""
            SELECT atlas_cell_id, {pc_cols_sql}
            FROM obsm_X_pca
            USING SAMPLE {int(fit_sample_n)} ROWS
        """

    fit_df = conn.execute(fit_query).fetchdf()

    if len(fit_df) == 0:
        raise ValueError("拟合 UMAP 的样本为空")

    X_fit = fit_df.drop(columns=["atlas_cell_id"]).to_numpy(dtype=np.float32)

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

    print("[UMAP] Fitting reducer...")
    reducer.fit(X_fit)

    # -------------------------------------------------
    # 2.5️⃣【新增】训练后评估 UMAP embedding 质量
    # -------------------------------------------------
    print("[UMAP] Evaluating embedding quality ...")

    X_fit_umap = reducer.transform(X_fit).astype(np.float32)

    # ✅ 只在子样本上评估，避免 eval_n × eval_n 距离矩阵爆内存
    eval_n = min(eval_sample_n, X_fit.shape[0])

    rng = np.random.default_rng(random_state)

    eval_idx = rng.choice(
        X_fit.shape[0],
        size=eval_n,
        replace=False
    )

    X_eval = X_fit[eval_idx]
    X_eval_umap = X_fit_umap[eval_idx]

    print(f"[UMAP] eval sample shape = {X_eval.shape}")

    eval_k = min(n_neighbors, eval_n - 1)

    trustworthiness_score = trustworthiness(
        X_eval,
        X_eval_umap,
        n_neighbors=eval_k
    )

    knn_overlap_score = knn_overlap(
        X_eval,
        X_eval_umap,
        k=eval_k
    )

    print(f"[UMAP] trustworthiness = {trustworthiness_score:.4f}")
    print(f"[UMAP] knn_overlap     = {knn_overlap_score:.4f}")

    # ✅ 简单自动评价
    if trustworthiness_score < 0.80:
        print("⚠️ UMAP局部结构保持较弱，建议增大 fit_sample_n / 调整 n_neighbors / 检查 PCA")
    elif trustworthiness_score < 0.90:
        print("✅ UMAP局部结构保持正常")
    else:
        print("🔥 UMAP局部结构保持很好")

    if knn_overlap_score < 0.20:
        print("⚠️ KNN重叠率偏低，低维空间近邻和PCA空间差异较大")
    elif knn_overlap_score < 0.40:
        print("✅ KNN重叠率正常，单细胞UMAP中常见")
    else:
        print("🔥 KNN重叠率较高，局部邻域保持很好")

    # -------------------------------------------------
    # 3️⃣ 建输出表
    # -------------------------------------------------
    conn.execute(f"DROP TABLE IF EXISTS {table_name}")
    conn.execute(f"""
        CREATE TABLE {table_name} (
            atlas_cell_id BIGINT,
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
            "output_table",
            "eval_sample_n"
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
            table_name,
            str(eval_sample_n)
        ]
    })

    conn.append(save_params_table, params_df)

    # -------------------------------------------------
    # 3.5️⃣【新增】保存评估结果
    # -------------------------------------------------
    conn.execute(f"DROP TABLE IF EXISTS {save_eval_table}")
    conn.execute(f"""
        CREATE TABLE {save_eval_table} (
            metric_name VARCHAR,
            metric_value DOUBLE
        )
    """)

    eval_df = pd.DataFrame({
        "metric_name": [
            "trustworthiness",
            "knn_overlap",
            "eval_sample_n",
            "eval_n_neighbors"
        ],
        "metric_value": [
            float(trustworthiness_score),
            float(knn_overlap_score),
            float(eval_n),
            float(eval_k)
        ]
    })

    conn.append(save_eval_table, eval_df)

    # -------------------------------------------------
    # 4️⃣ 全量分块 transform
    # -------------------------------------------------
    total_n = conn.execute("SELECT COUNT(*) FROM obsm_X_pca").fetchone()[0]
    print(f"[UMAP] total cells = {total_n}")

    offset = 0

    while offset < total_n:

        batch_df = conn.execute(f"""
            SELECT atlas_cell_id, {pc_cols_sql}
            FROM obsm_X_pca
            ORDER BY atlas_cell_id
            LIMIT {int(transform_batch_size)}
            OFFSET {int(offset)}
        """).fetchdf()

        if len(batch_df) == 0:
            break

        X_batch = batch_df.drop(columns=["atlas_cell_id"]).to_numpy(dtype=np.float32)
        X_umap = reducer.transform(X_batch).astype(np.float32)

        out_df = pd.DataFrame({
            "atlas_cell_id": batch_df["atlas_cell_id"].to_numpy(dtype=np.int64),
            "umap1": X_umap[:, 0],
            "umap2": X_umap[:, 1]
        })

        conn.append(table_name, out_df)

        offset += len(batch_df)

        print(f"[UMAP] transformed {offset}/{total_n}")

    print("[UMAP] Done")
    print(f"耗时: {(datetime.now() - start).total_seconds():.2f} 秒")

    return reducer






#todo minibatch umap
# 在全部数据上，以 minibatch 的方式训练 Parametric UMAP
# DuckDB / CSR
#    ↓
# Streaming PCA 得到 obsm_X_pca
#    ↓
# 用全部或大样本 X_pca 构建 UMAP graph
#    ↓
# ParametricUMAP 用 edge mini-batch 训练 encoder
#    ↓
# encoder.transform(X_pca_batch)
#    ↓
# 分批写入 obsm_X_umap

# ParametricUMAP.fit 内部发生了什么
# ① 构建 kNN graph（重点‼️）
# X_fit  (n_cells × n_pcs)
#    ↓
# kNN 搜索（n_neighbors）
#    ↓
# 邻接图 graph
#
# 👉 本质就是：
#
# neighbors = NearestNeighbors(n_neighbors=15)
# graph = neighbors.fit(X_fit)
#
# 但内部是用 UMAP 自己优化过的 NNDescent（更快）
#
# ② 转成 UMAP fuzzy graph
# kNN graph
#    ↓
# UMAP fuzzy simplicial set
#
# 👉 就是 UMAP 的核心概率图：
#
# 邻居 → 概率连接
# 非邻居 → 负样本
# ③ 用 graph 训练神经网络（minibatch）
# graph edges
#    ↓ (随机采样)
# (edge_i, edge_j)
#    ↓
# loss = 拉近邻居 + 推远非邻居
#    ↓
# backprop 更新 encoder
#
# 👉 这一步才是真正的：
#
# minibatch training


# self.embedder = ParametricUMAP(...)
# self.embedder.fit(X_fit)
# 1️⃣ 构建 kNN graph
# 2️⃣ 构建 fuzzy graph
# 3️⃣ minibatch 训练 NN
from datetime import datetime
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from ..data import Atlas


# =========================================================
# 数据库版 Parametric UMAP
# =========================================================
class StreamingParametricUMAP:
    """
    数据库版 Parametric UMAP

    核心目标
    --------
    把已经计算好的 PCA embedding：

        obsm_X_pca
            atlas_cell_id, pc0, pc1, pc2, ...

    转换成 UMAP embedding：

        obsm_X_umap
            atlas_cell_id, umap1, umap2

    和普通 UMAP 的区别
    ------------------
    普通 UMAP：
        直接优化每个细胞的二维坐标。

    Parametric UMAP：
        训练一个神经网络 encoder：

            X_pca -> X_umap

        也就是学习一个“映射函数”。

    重要提醒
    --------
    这个类的“小内存大数据”主要体现在 transform_all 阶段：

        全量 obsm_X_pca
            ↓
        分批读取
            ↓
        分批 transform
            ↓
        分批写入 obsm_X_umap

    但是 fit 阶段仍然需要把 X_fit 读入内存，并且 ParametricUMAP.fit()
    内部会基于 X_fit 构建 UMAP graph。

    所以：
        fit_sample_n=200000  -> 推荐，比较安全
        fit_sample_n=None    -> 全量训练，可能很吃内存
    """

    def __init__(
            self,
            n_components: int = 2,
            n_neighbors: int = 15,
            min_dist: float = 0.5,
            metric: str = "euclidean",
            random_state: int = 42,

            # ParametricUMAP 内部训练时使用的 batch size
            # 注意：这里不是 cell batch，而是 graph edge batch。
            batch_size: int = 2048,

            # 神经网络训练轮数
            n_training_epochs: int = 10,

            # encoder 神经网络结构
            # 例如：
            # input_dim=50
            # 50 -> 256 -> 128 -> 2
            hidden_units: tuple[int, ...] = (256, 128),

            # 是否启用早停
            # 如果 loss 很久不下降，就提前停止训练
            early_stopping: bool = True,
            early_stopping_patience: int = 5,
            early_stopping_min_delta: float = 1e-3,

            verbose: bool = True,

            # ✅【新增】评估指标最多抽样多少细胞
            eval_sample_n: int = 10000,

    ):
        # UMAP 输出维度，通常是 2
        self.n_components = n_components

        # UMAP 近邻数
        # 越大越强调全局结构，越小越强调局部结构
        self.n_neighbors = n_neighbors

        # UMAP 点之间最小距离
        # 越小 cluster 越紧
        self.min_dist = min_dist

        # 距离度量
        # 对 PCA embedding 一般用 euclidean
        self.metric = metric

        # 随机种子，保证结果尽量可复现
        self.random_state = random_state

        # ParametricUMAP 内部 edge minibatch 大小
        self.batch_size = batch_size

        # 神经网络训练 epoch 数
        self.n_training_epochs = n_training_epochs

        # encoder 隐藏层结构
        self.hidden_units = hidden_units

        # 早停参数
        self.early_stopping = early_stopping
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta

        self.verbose = verbose

        # ✅【新增】避免 trustworthiness / knn_overlap 全量评估爆内存
        self.eval_sample_n = eval_sample_n

        # 训练好的 ParametricUMAP 对象
        self.embedder = None

        # tf.keras encoder 网络
        self.encoder = None

        # PCA 列名，例如 ["pc0", "pc1", ..., "pc49"]
        self.pc_cols = None

        # ✅【新增】训练质量指标
        self.trustworthiness_score_ = None
        self.knn_overlap_score_ = None

    # -------------------------------------------------
    # 0️⃣ 检查 PCA 表
    # -------------------------------------------------
    def _check_pca_table(
            self,
            atlas: Atlas,
            pca_table: str = "obsm_X_pca"
    ):
        """
        检查数据库中是否存在 obsm_X_pca，
        并自动识别 pc0, pc1, pc2 ... 这些 PCA 列。

        为什么需要这个函数？
        -------------------
        Parametric UMAP 的输入不是原始基因表达矩阵，
        而是 PCA 后的低维矩阵。

        也就是：

            obsm_X_pca
                atlas_cell_id
                pc0
                pc1
                ...
        """
        conn = atlas.connection

        # 检查 pca_table 是否存在
        tables = conn.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_name = ?
        """, [pca_table]).fetchdf()

        if len(tables) == 0:
            raise ValueError(
                f"数据库中不存在 {pca_table}。\n"
                f"请先运行 sap.tl.pca(atlas)"
            )

        # 读取表结构
        pca_cols = [
            r[1]
            for r in conn.execute(f"PRAGMA table_info({pca_table})").fetchall()
        ]

        # 必须有 atlas_cell_id，用来写回 obsm_X_umap
        if "atlas_cell_id" not in pca_cols:
            raise ValueError(f"{pca_table} 中不存在 atlas_cell_id")

        # 自动寻找 pc 列
        pc_cols = [c for c in pca_cols if c.startswith("pc")]

        if len(pc_cols) == 0:
            raise ValueError(f"{pca_table} 中不存在 pc0 / pc1 / ... 列")

        # 保证 pc0, pc1, pc2 ... 的顺序正确
        # 否则如果字符串排序，可能出现 pc0, pc1, pc10, pc11, pc2
        pc_cols = sorted(pc_cols, key=lambda x: int(x.replace("pc", "")))

        self.pc_cols = pc_cols

        print(f"[ParametricUMAP] input table = {pca_table}")
        print(f"[ParametricUMAP] n_pcs = {len(pc_cols)}")

        return pc_cols

    # -------------------------------------------------
    # 1️⃣ 构建 encoder 神经网络
    # -------------------------------------------------
    def _build_encoder(self, input_dim: int):
        """
        构建 ParametricUMAP 使用的神经网络 encoder。

        输入
        ----
        X_pca:
            shape = (n_cells, n_pcs)

        输出
        ----
        X_umap:
            shape = (n_cells, n_components)

        举例
        ----
        如果 input_dim=50, n_components=2：

            50维 PCA
                ↓
            Dense(256)
                ↓
            BatchNorm
                ↓
            Dense(128)
                ↓
            BatchNorm
                ↓
            Dense(2)
                ↓
            UMAP 坐标
        """
        import tensorflow as tf

        layers = [
            # 输入层，告诉模型每个细胞有多少个 PCA 特征
            tf.keras.layers.InputLayer(input_shape=(input_dim,))
        ]

        # 构建隐藏层
        for units in self.hidden_units:
            layers.append(tf.keras.layers.Dense(units, activation="relu"))

            # BatchNormalization 可以让训练更稳定
            # 对大样本神经网络训练通常有帮助
            layers.append(tf.keras.layers.BatchNormalization())

        # 输出层：输出 UMAP 坐标
        # n_components=2 时，就是 umap1, umap2
        layers.append(tf.keras.layers.Dense(self.n_components))

        encoder = tf.keras.Sequential(
            layers,
            name="scAtlasPy_ParametricUMAP_Encoder"
        )

        print("[ParametricUMAP] encoder summary:")
        encoder.summary()

        return encoder

    # -------------------------------------------------
    # 2️⃣ 读取训练数据
    # -------------------------------------------------
    def _load_fit_data(
            self,
            atlas: Atlas,
            pca_table: str = "obsm_X_pca",
            fit_sample_n: int | None = 200000
    ):
        """
        从 obsm_X_pca 读取用于训练 ParametricUMAP 的数据。

        fit_sample_n
        ------------
        int:
            SQL 随机抽样指定数量细胞训练。

        None:
            读取全量 obsm_X_pca 训练。

        为什么默认不建议 None？
        -----------------------
        因为 ParametricUMAP.fit(X_fit) 内部会基于 X_fit 构建 UMAP graph。

        graph 构建阶段不是完全 streaming 的，
        所以 X_fit 太大时会产生明显内存压力。
        """
        conn = atlas.connection

        pc_cols_sql = ", ".join(self.pc_cols)

        if fit_sample_n is None:
            # 全量训练
            query = f"""
                SELECT atlas_cell_id, {pc_cols_sql}
                FROM {pca_table}
                ORDER BY atlas_cell_id
            """
        else:
            # 抽样训练
            # 这里是 SQL 层抽样，不需要把全量 PCA 拉到 Python 后再抽样
            query = f"""
                SELECT atlas_cell_id, {pc_cols_sql}
                FROM {pca_table}
                USING SAMPLE {int(fit_sample_n)} ROWS
            """

        fit_df = conn.execute(query).fetchdf()

        if len(fit_df) == 0:
            raise ValueError("ParametricUMAP 训练样本为空")

        # 去掉 cell id，只保留 PCA 数值矩阵
        X_fit = fit_df.drop(columns=["atlas_cell_id"]).to_numpy(dtype=np.float32)

        print(f"[ParametricUMAP] X_fit shape = {X_fit.shape}")

        return X_fit

    # -------------------------------------------------
    # 3️⃣ 训练 Parametric UMAP
    # -------------------------------------------------
    def fit(
            self,
            atlas: Atlas,
            pca_table: str = "obsm_X_pca",
            fit_sample_n: int | None = 200000
    ):
        """
        训练 ParametricUMAP。

        关键点
        ------
        这里你没有自己写：

            for batch in X:
                partial_fit(batch)

        因为 ParametricUMAP 不是这样训练的。

        它内部会：
            1. 基于 X_fit 构建 UMAP graph
            2. 从 graph 中采样 edges
            3. 用 edge minibatch 训练 encoder 神经网络

        所以真正触发 graph 构建和 minibatch 训练的是：

            self.embedder.fit(X_fit)
        """
        from umap.parametric_umap import ParametricUMAP
        import tensorflow as tf

        print("\n==== StreamingParametricUMAP.fit ====")
        start = datetime.now()

        # 检查 PCA 表，得到 pc 列
        pc_cols = self._check_pca_table(atlas, pca_table=pca_table)
        input_dim = len(pc_cols)

        # todo 读取训练数据
        #  每轮的输入是一样的 ?
        # 不建议在同一个 fit 里每轮换 X_fit ❌
        # 因为 ParametricUMAP 的核心不是简单监督学习，而是学习：
        # 这个 X_fit 上的邻居关系
        # 如果每轮换 X_fit：
        # epoch1 graph A
        # epoch2 graph B
        # 那 loss 的目标一直变，训练会不稳定。
        X_fit = self._load_fit_data(
            atlas=atlas,
            pca_table=pca_table,
            fit_sample_n=fit_sample_n
        )

        # 构建 encoder 网络
        self.encoder = self._build_encoder(input_dim=input_dim)

        # Keras 训练参数
        keras_fit_kwargs = {}

        # 早停机制
        if self.early_stopping:
            keras_fit_kwargs["callbacks"] = [
                tf.keras.callbacks.EarlyStopping(
                    monitor="loss",
                    min_delta=self.early_stopping_min_delta,
                    patience=self.early_stopping_patience,
                    verbose=1,
                    restore_best_weights=True
                )
            ]

        # 创建 ParametricUMAP 对象
        self.embedder = ParametricUMAP(
            encoder=self.encoder,

            # 输入维度
            dims=(input_dim,),

            # UMAP 参数
            n_components=self.n_components,
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            metric=self.metric,
            random_state=self.random_state,

            # # UMAP / ParametricUMAP 训练轮数
            # n_epochs=self.n_training_epochs,

            # edge minibatch 大小
            batch_size=self.batch_size,

            # Keras fit 参数
            keras_fit_kwargs=keras_fit_kwargs,

            verbose=self.verbose
        )

        # 🔥 真正控制 Keras fit 的训练轮数
        self.embedder.n_training_epochs = self.n_training_epochs
        print(f"[ParametricUMAP]  self.embedder.n_training_epochs  = { self.embedder.n_training_epochs }")

        print("[ParametricUMAP] Start fitting ...")

        # 核心：内部构建 graph + 训练神经网络
        self.embedder.fit(X_fit)

        # 训练后评估效果
        print("[ParametricUMAP] Evaluating embedding quality ...")

        X_fit_umap = self.embedder.transform(X_fit).astype(np.float32)

        # ✅【修改】评估指标只在子样本上计算，避免 200k × 200k 距离矩阵爆内存
        eval_n = min(self.eval_sample_n, X_fit.shape[0])

        rng = np.random.default_rng(self.random_state)
        eval_idx = rng.choice(
            X_fit.shape[0],
            size=eval_n,
            replace=False
        )

        X_eval = X_fit[eval_idx]
        X_eval_umap = X_fit_umap[eval_idx]

        print(f"[ParametricUMAP] eval sample shape = {X_eval.shape}")

        # 训练后评估效果
        self.trustworthiness_score_ = self.compute_trustworthiness(
            X_eval,
            X_eval_umap,
            n_neighbors=self.n_neighbors
        )
        # 训练后评估效果
        self.knn_overlap_score_ = self.knn_overlap(
            X_eval,
            X_eval_umap,
            k=self.n_neighbors
        )

        print(
            f"[ParametricUMAP] Fit done in "
            f"{(datetime.now() - start).total_seconds():.2f}s"
        )

        return self


    # 评估 UMAP 是否保持原空间的“近邻关系”
    def compute_trustworthiness(self, X_high, X_low, n_neighbors=15):
        score = trustworthiness(X_high, X_low, n_neighbors=n_neighbors)
        print(f"[UMAP] Trustworthiness = {score:.4f}")
        return score
    # 原空间最近的邻居
    # ↓
    # 在 UMAP 空间还近吗？
    # 📈 范围
    # 0 ~ 1
    # 0.98+  → 非常好
    # 0.95+  → 好
    # <0.90  → 有问题

    def knn_overlap(self, X_high, X_low, k=15):

        nn_high = NearestNeighbors(n_neighbors=k + 1).fit(X_high)
        nn_low = NearestNeighbors(n_neighbors=k + 1).fit(X_low)

        idx_high = nn_high.kneighbors(return_distance=False)[:, 1:]
        idx_low = nn_low.kneighbors(return_distance=False)[:, 1:]

        overlaps = []

        for i in range(len(idx_high)):
            overlap = len(set(idx_high[i]) & set(idx_low[i])) / k
            overlaps.append(overlap)

        score = float(np.mean(overlaps))
        print(f"[UMAP] KNN overlap = {score:.4f}")
        return score


    # -------------------------------------------------
    # 4️⃣ 创建输出表 obsm_X_umap
    # -------------------------------------------------
    def _create_umap_table(
            self,
            atlas: Atlas,
            table_name: str = "obsm_X_umap"
    ):
        """
        创建 UMAP 输出表。

        默认二维：
            atlas_cell_id, umap1, umap2

        如果 n_components > 2：
            atlas_cell_id, umap1, umap2, umap3, ...
        """
        conn = atlas.connection

        # 覆盖旧结果
        conn.execute(f"DROP TABLE IF EXISTS {table_name}")

        if self.n_components == 2:
            conn.execute(f"""
                CREATE TABLE {table_name} (
                    atlas_cell_id BIGINT,
                    umap1 FLOAT,
                    umap2 FLOAT
                )
            """)
        else:
            cols = ",\n".join([
                f"umap{i + 1} FLOAT"
                for i in range(self.n_components)
            ])

            conn.execute(f"""
                CREATE TABLE {table_name} (
                    atlas_cell_id BIGINT,
                    {cols}
                )
            """)

        print(f"[ParametricUMAP] created table: {table_name}")

    # -------------------------------------------------
    # 5️⃣ 分批读取 PCA
    # -------------------------------------------------
    def _iter_pca_batches(
            self,
            atlas: Atlas,
            pca_table: str = "obsm_X_pca",
            transform_batch_size: int = 50000
    ):
        """
        分批读取 obsm_X_pca。

        这里不用 OFFSET，而是用 keyset pagination：

            WHERE atlas_cell_id > last_cell_id
            ORDER BY atlas_cell_id
            LIMIT batch_size

        为什么不用 OFFSET？
        ------------------
        OFFSET 越往后越慢，因为数据库需要跳过越来越多的行。

        keyset pagination 更适合大表。
        """
        conn = atlas.connection
        pc_cols_sql = ", ".join(self.pc_cols)

        last_cell_id = -1

        while True:
            batch_df = conn.execute(f"""
                SELECT atlas_cell_id, {pc_cols_sql}
                FROM {pca_table}
                WHERE atlas_cell_id > {int(last_cell_id)}
                ORDER BY atlas_cell_id
                LIMIT {int(transform_batch_size)}
            """).fetchdf()

            if len(batch_df) == 0:
                break

            last_cell_id = int(batch_df["atlas_cell_id"].iloc[-1])

            yield batch_df

    # -------------------------------------------------
    # 6️⃣ 写入一个 batch 的 UMAP 结果
    # -------------------------------------------------
    def _write_umap_batch(
            self,
            atlas: Atlas,
            cell_ids,
            X_umap,
            table_name: str = "obsm_X_umap"
    ):
        """
        把一个 batch 的 UMAP 坐标 append 到 DuckDB。

        输入：
            cell_ids:
                当前 batch 的 atlas_cell_id

            X_umap:
                当前 batch 的 UMAP 坐标
                shape = (batch_size, n_components)
        """
        X_umap = X_umap.astype(np.float32)

        out_df = pd.DataFrame({
            "atlas_cell_id": np.asarray(cell_ids, dtype=np.int64)
        })

        for i in range(self.n_components):
            out_df[f"umap{i + 1}"] = X_umap[:, i].astype(np.float32)

        atlas.connection.append(table_name, out_df)

    # -------------------------------------------------
    # 7️⃣ transform 全量细胞
    # -------------------------------------------------
    def transform_all(
            self,
            atlas: Atlas,
            pca_table: str = "obsm_X_pca",
            table_name: str = "obsm_X_umap",
            transform_batch_size: int = 50000
    ):
        """
        使用训练好的 ParametricUMAP 模型，对全量 PCA embedding 分批 transform。

        这是本类真正小内存大数据友好的部分。

        流程
        ----
        obsm_X_pca
            ↓ 分批读取
        X_batch
            ↓ encoder 前向传播
        X_umap
            ↓ append
        obsm_X_umap
        """
        if self.embedder is None:
            raise ValueError("请先运行 fit() 训练 ParametricUMAP")

        print("\n==== StreamingParametricUMAP.transform_all ====")
        start = datetime.now()

        conn = atlas.connection

        # 再检查一次 PCA 表，保证 pc_cols 正确
        self._check_pca_table(atlas, pca_table=pca_table)

        # 创建输出表
        self._create_umap_table(atlas, table_name=table_name)

        total_n = conn.execute(f"SELECT COUNT(*) FROM {pca_table}").fetchone()[0]
        done = 0

        for batch_df in tqdm(
                self._iter_pca_batches(
                    atlas=atlas,
                    pca_table=pca_table,
                    transform_batch_size=transform_batch_size
                ),
                total=max(1, int(np.ceil(total_n / transform_batch_size)))
        ):
            cell_ids = batch_df["atlas_cell_id"].to_numpy(dtype=np.int64)

            X_batch = batch_df.drop(columns=["atlas_cell_id"]).to_numpy(dtype=np.float32)

            # 神经网络前向传播：
            # PCA embedding -> UMAP 坐标
            X_umap = self.embedder.transform(X_batch).astype(np.float32)

            # self.compute_trustworthiness(X_batch, X_umap) # todo 评估结果的指标
            # self.knn_overlap(X_batch, X_umap)

            # 写入 DuckDB
            self._write_umap_batch(
                atlas=atlas,
                cell_ids=cell_ids,
                X_umap=X_umap,
                table_name=table_name
            )

            done += len(batch_df)

        print(
            f"[ParametricUMAP] transformed {done:,}/{total_n:,} cells "
            f"in {(datetime.now() - start).total_seconds():.2f}s"
        )

        return self

    # -------------------------------------------------
    # 8️⃣ 写参数表
    # -------------------------------------------------
    def _write_params(
            self,
            atlas: Atlas,
            fit_sample_n,
            transform_batch_size,
            pca_table: str,
            table_name: str,
            save_params_table: str = "uns_umap_params"
    ):
        """
        保存 UMAP 参数到 uns_umap_params。

        作用：
            方便之后知道 obsm_X_umap 是怎么生成的。
        """
        conn = atlas.connection

        conn.execute(f"DROP TABLE IF EXISTS {save_params_table}")
        conn.execute(f"""
            CREATE TABLE {save_params_table} (
                param_name VARCHAR,
                param_value VARCHAR
            )
        """)

        params = {
            "method": "ParametricUMAP",
            "input_table": pca_table,
            "output_table": table_name,
            "n_components": self.n_components,
            "n_neighbors": self.n_neighbors,
            "min_dist": self.min_dist,
            "metric": self.metric,
            "random_state": self.random_state,
            "batch_size_edge": self.batch_size,
            "n_training_epochs": self.n_training_epochs,
            "hidden_units": str(self.hidden_units),
            "fit_sample_n": fit_sample_n,
            "transform_batch_size": transform_batch_size,
            "eval_sample_n": self.eval_sample_n,
            "trustworthiness": self.trustworthiness_score_,
            "knn_overlap": self.knn_overlap_score_,
        }

        params_df = pd.DataFrame({
            "param_name": list(params.keys()),
            "param_value": [str(v) for v in params.values()]
        })

        conn.append(save_params_table, params_df)

        print(f"[ParametricUMAP] params written to {save_params_table}")

    # -------------------------------------------------
    # 9️⃣ 保存模型
    # -------------------------------------------------
    def save_model(self, model_path: str):
        """
        保存 ParametricUMAP 模型。

        注意
        ----
        不能只用 pickle 保存 ParametricUMAP，
        因为它里面包含 TensorFlow / Keras 神经网络。

        所以要用：
            self.embedder.save(model_path)
        """
        if self.embedder is None:
            raise ValueError("请先训练模型再保存")

        os.makedirs(model_path, exist_ok=True)

        self.embedder.save(model_path)

        print(f"[ParametricUMAP] model saved to: {model_path}")

    # -------------------------------------------------
    # 🔟 一键运行
    # -------------------------------------------------
    def run(
            self,
            atlas: Atlas,
            pca_table: str = "obsm_X_pca",
            table_name: str = "obsm_X_umap",
            fit_sample_n: int | None = 200000,
            transform_batch_size: int = 50000,
            save_params_table: str = "uns_umap_params",
            model_path: str | None = None
    ):
        """
        一键执行完整流程：

            1. fit()
                抽样 / 全量读取 X_fit
                构建 UMAP graph
                训练 ParametricUMAP encoder

            2. transform_all()
                分批读取全量 obsm_X_pca
                分批 transform
                写入 obsm_X_umap

            3. _write_params()
                写入 uns_umap_params

            4. save_model()
                可选保存模型
        """
        print("\n==== StreamingParametricUMAP.run ====")
        start = datetime.now()

        self.fit(
            atlas=atlas,
            pca_table=pca_table,
            fit_sample_n=fit_sample_n
        )

        self.transform_all(
            atlas=atlas,
            pca_table=pca_table,
            table_name=table_name,
            transform_batch_size=transform_batch_size
        )

        self._write_params(
            atlas=atlas,
            fit_sample_n=fit_sample_n,
            transform_batch_size=transform_batch_size,
            pca_table=pca_table,
            table_name=table_name,
            save_params_table=save_params_table
        )

        if model_path is not None:
            self.save_model(model_path)

        print(
            f"[ParametricUMAP] All done in "
            f"{(datetime.now() - start).total_seconds():.2f}s ✅"
        )

        return self




# =========================================================
# 外部入口函数：sap.tl.parametric_umap(...)
# =========================================================
def parametric_umap(
        atlas: Atlas,
        fit_sample_n: int | None = 200000,
        transform_batch_size: int = 50000,

        n_components: int = 2,
        n_neighbors: int = 15,
        min_dist: float = 0.5,
        metric: str = "euclidean",
        random_state: int = 42,

        batch_size: int = 2048,
        n_training_epochs: int = 5,
        eval_sample_n: int = 10000,
        hidden_units: tuple[int, ...] = (256, 128),

        early_stopping: bool = True,
        early_stopping_patience: int = 5,
        early_stopping_min_delta: float = 1e-3,

        pca_table: str = "obsm_X_pca",
        table_name: str = "obsm_X_umap",
        save_params_table: str = "uns_umap_params",
        model_path: str | None = None,

        verbose: bool = True
):
    """
    Scanpy 风格 Parametric UMAP 计算入口。

    推荐用法
    --------
    sap.tl.parametric_umap(
        atlas,
        fit_sample_n=200000,
        transform_batch_size=50000,
        n_neighbors=15,
        min_dist=0.5,
        batch_size=2048,
        n_training_epochs=2,
        model_path="parametric_umap_model"
    )

    参数重点
    --------
    fit_sample_n:
        用多少细胞训练 ParametricUMAP。

        - 200000：推荐起步
        - None：全量训练，不推荐大数据直接用

    batch_size:
        ParametricUMAP 内部 edge minibatch 大小。

    transform_batch_size:
        transform 全量细胞时，每次从 DuckDB 读取多少细胞。
    """

    runner = StreamingParametricUMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
        batch_size=batch_size,
        n_training_epochs=n_training_epochs,
        hidden_units=hidden_units,
        early_stopping=early_stopping,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
        verbose=verbose,
        eval_sample_n = eval_sample_n,
    )

    runner.run(
        atlas=atlas,
        pca_table=pca_table,
        table_name=table_name,
        fit_sample_n=fit_sample_n,
        transform_batch_size=transform_batch_size,
        save_params_table=save_params_table,
        model_path=model_path
    )

    return runner

# 参数	                 控制什么	            本质
# fit_sample_n	        用多少细胞构图	    数据规模（graph大小）
# batch_size	        每步训练用多少 edge	训练粒度（SGD batch）
# n_training_epochs	    训练多少轮	        训练强度（迭代次数）

# 1️⃣ fit_sample_n（最关键）
# 👉 控制：UMAP graph 的规模
# X_fit = sample cells
# → build KNN graph
# → 生成 edge
# 📊 本质
# cell → graph → edge
# 如果：
# fit_sample_n = 200k
# 那 graph 大概：
# 节点：200k
# 边：200k × n_neighbors（比如15）≈ 300万条边

# 推荐（你这个场景）
# 你是：
# 小内存 + 超大数据
# 👉 推荐：
# fit_sample_n = 100k ~ 300k   ✅ 最优

# 2️⃣ batch_size（edge minibatch）
# 👉 控制：每一步训练用多少“边”
# batch_size = 2048
# 意味着：
# 每次训练：
# 随机采 2048 条 graph edge
# → 喂给神经网络
# → 更新参数

# 3️⃣_training_epochs（训练轮数）
# 👉 控制：训练多少轮
# 1 epoch = 遍历 graph edge 一轮
# 📊 本质
# 训练强度
# 🎯 影响
# 影响	说明
# embedding质量	↑
# 过拟合风险	↑
# 时间	↑
# 推荐
# n_training_epochs = 10 ~ 30
#
# 👉 你现在：
#
# 10 = 偏快（OK）
# 20 = 更稳（推荐）

# 🧠 三者关系（重点理解）
# 👉 整体 pipeline
# 1️⃣ fit_sample_n 决定 graph
#         ↓
# 2️⃣ graph → edges
#         ↓
# 3️⃣ 每 step 用 batch_size 条 edge
#         ↓
# 4️⃣ 重复 n_training_epochs 次
# 📊 用一句话总结
# fit_sample_n = 数据规模
# batch_size   = 每次训练吃多少
# epochs       = 训练多少次