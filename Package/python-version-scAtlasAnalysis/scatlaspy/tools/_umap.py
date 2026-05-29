import umap as umap_lib
from sklearn.manifold import trustworthiness
from sklearn.neighbors import NearestNeighbors
from datetime import datetime
import numpy as np
import pandas as pd

# 评估函数：KNN overlap
def knn_overlap(X_high, X_low, k=15):

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


# UMAP 抽样训练
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
        eval_sample_n: int = 5000,
        save_eval_table: str = "uns_umap_eval"
):
    """
       基于 PCA embedding 计算 UMAP，并将结果写入数据库。

       该函数从数据库表 ``obsm_X_pca`` 中读取已经计算好的 PCA 坐标，
       先使用部分细胞抽样拟合 UMAP 模型，然后将全量细胞分批转换到
       UMAP 空间，并将二维坐标保存到 ``obsm_X_umap`` 表中。

       与 Scanpy 中 ``sc.tl.umap`` 的用途类似，本函数用于计算细胞的
       UMAP 低维表示。但为了适配大规模单细胞数据，本函数采用
       “抽样拟合 + 全量分块 transform” 的方式，以降低内存占用。

       Parameters
       ----------
       atlas
           Atlas 对象。

           要求数据库中已经存在 ``obsm_X_pca`` 表。该表通常由
           ``sap.tl.pca(atlas)`` 生成，至少应包含：

           - ``atlas_cell_id``：细胞 ID
           - ``pc0, pc1, pc2, ...``：PCA 坐标列

       fit_sample_n
           用于拟合 UMAP 模型的细胞抽样数量。

           - ``int``：从 ``obsm_X_pca`` 中随机抽取指定数量的细胞拟合 UMAP。
           - ``None``：使用全部细胞拟合 UMAP。

           较大的 ``fit_sample_n`` 通常可以获得更稳定的 UMAP 结构，
           但会增加内存占用和计算时间。对于百万级细胞数据，可以设置为
           ``500000`` 或更高；如果希望更接近全量 UMAP，可设置为 ``None``。

       transform_batch_size
           全量细胞 UMAP transform 时的分块大小。

           UMAP 模型拟合完成后，函数会从 ``obsm_X_pca`` 中按批读取
           PCA 坐标，并调用 ``reducer.transform`` 将其转换到 UMAP 空间。

           较大的值通常速度更快，但占用更多内存；较小的值更稳但速度较慢。

       n_components
           UMAP 输出维度。

           默认值为 ``2``，表示生成二维 UMAP 坐标，并写入：

           - ``umap1``
           - ``umap2``

           当前绘图函数通常默认使用二维 UMAP，因此一般保持 ``2`` 即可。

       n_neighbors
           UMAP 构建局部邻域时使用的近邻数。

           该参数控制 UMAP 更关注局部结构还是整体结构：

           - 较小值，例如 ``15``：更强调局部结构，簇可能更紧密、更分开。
           - 较大值，例如 ``45`` 或 ``50``：更强调整体结构，簇之间关系更平滑。

       min_dist
           UMAP 低维空间中点与点之间允许的最小距离。

           该参数影响簇的紧密程度：

           - 较小值，例如 ``0.1`` 或 ``0.2``：簇更紧凑。
           - 较大值，例如 ``0.5``：点更分散，簇形状更松。

       metric
           计算 UMAP 邻域时使用的距离度量。

           对于 PCA embedding，通常使用 ``"euclidean"``。
           如果未来输入为文本向量或其他 embedding，也可以考虑使用
           ``"cosine"`` 等距离度量。

       random_state
           随机种子。

           用于控制 UMAP 拟合和评估抽样过程中的随机性。
           设置为固定整数时，结果更容易复现。若设置为 ``None``，
           每次运行可能产生略有不同的结果。

       n_jobs
           UMAP 拟合和 transform 使用的线程数。

           - ``1``：单线程，结果通常更稳定、更可复现。
           - 大于 ``1``：可能加快计算，但结果可能存在轻微随机差异。

       table_name
           保存 UMAP 坐标的数据库表名。

           默认保存到 ``obsm_X_umap``。表结构为：

           - ``atlas_cell_id``
           - ``umap1``
           - ``umap2``

           如果希望保存多个不同参数的 UMAP 结果，可以修改该参数，
           例如 ``"obsm_X_umap_n45_d02"``。

       save_params_table
           保存本次 UMAP 参数的数据库表名。

           默认保存到 ``uns_umap_params``。该表记录本次运行使用的
           UMAP 参数，例如：

           - ``n_components``
           - ``n_neighbors``
           - ``min_dist``
           - ``metric``
           - ``random_state``
           - ``fit_sample_n``
           - ``transform_batch_size``

       eval_sample_n
           用于评估 UMAP embedding 质量的细胞数量。

           函数会从拟合样本中再抽取最多 ``eval_sample_n`` 个细胞，
           计算 UMAP 质量指标，包括：

           - ``trustworthiness``：低维空间对高维局部结构的保持程度。
           - ``knn_overlap``：PCA 空间和 UMAP 空间近邻集合的重叠比例。

           较大的值评估更稳定，但会增加计算时间和内存开销。

       save_eval_table
           保存 UMAP 评估结果的数据库表名。

           默认保存到 ``uns_umap_eval``。该表包含：

           - ``trustworthiness``
           - ``knn_overlap``
           - ``eval_sample_n``
           - ``eval_n_neighbors``

       Returns
       -------
       reducer
           拟合完成的 ``umap.UMAP`` 对象。

           该对象可用于后续对新的 PCA embedding 进行 transform，
           或用于检查 UMAP 模型参数。

       Notes
       -----
       本函数采用的是：

       ``抽样 fit UMAP -> 全量 transform``

       而不是 Scanpy 默认的全量 neighbor graph UMAP。因此在大规模数据上
       更节省内存，但生成的 UMAP 图形可能与 Scanpy 的全量 UMAP 不完全一致。

       如果希望结果更接近 Scanpy 的全量 UMAP，可以增大 ``fit_sample_n``，
       或设置 ``fit_sample_n=None`` 使用全部细胞拟合。但这会显著增加
       内存和计算时间。

       Examples
       --------
       使用默认参数计算 UMAP：

       sap.tl.umap(atlas)

       使用 50 万细胞拟合 UMAP，并分批 transform 全量数据：

       sap.tl.umap(
       ...     atlas,
       ...     fit_sample_n=500000,
       ...     transform_batch_size=100000
       ... )

       调整 UMAP 结构参数：

       sap.tl.umap(
       ...     atlas,
       ...     fit_sample_n=500000,
       ...     n_neighbors=45,
       ...     min_dist=0.2,
       ...     random_state=42
       ... )

       保存为不同名称的 UMAP 结果表：

        sap.tl.umap(
       ...     atlas,
       ...     table_name="obsm_X_umap_n45_d02",
       ...     save_params_table="uns_umap_params_n45_d02",
       ...     save_eval_table="uns_umap_eval_n45_d02"
       ... )

       See Also
       --------
       sap.tl.pca
           计算 PCA embedding，并写入 ``obsm_X_pca``。
       sap.pl.umap
           绘制已经计算好的 UMAP embedding。
       """

    print("\n==== sap.tl.umap ====")
    start = datetime.now()
    conn = atlas.connection

    # 检查 obsm_X_pca
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

    # 保证 pc0, pc1, pc2 ... 按数字顺序排列
    pc_cols = sorted(pc_cols, key=lambda x: int(x.replace("pc", "")))
    pc_cols_sql = ", ".join(pc_cols)

    # 拟合 UMAP：SQL 先抽样
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

    # 拟合 UMAP 模型
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

    # 训练后评估 UMAP embedding 质量
    print("[UMAP] Evaluating embedding quality ...")

    X_fit_umap = reducer.transform(X_fit).astype(np.float32)

    # 只在子样本上评估，避免 eval_n × eval_n 距离矩阵爆内存
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

    # 简单自动评价
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

    # 建输出表
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

    # 保存评估结果
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

    # 全量分块 transform
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
