import umap as umap_lib
from sklearn.manifold import trustworthiness
from sklearn.neighbors import NearestNeighbors
from datetime import datetime
import numpy as np
import pandas as pd
from ..data import Atlas
import logging
logger = logging.getLogger('Atlas')

# 评估函数：KNN overlap
def knn_overlap(X_high: np.ndarray, X_low: np.ndarray, k: int=15):

    """计算高维和低维空间的近邻重叠率。

    该函数分别在高维坐标和低维 embedding 中计算每个样本的 k 近邻集合，
    然后返回两个近邻集合的平均重叠比例。它用于评估 UMAP embedding 对 PCA
    局部邻域结构的保留程度。

    Parameters
    ----------
    X_high
        高维空间中的坐标矩阵，例如 PCA 坐标。
    X_low
        低维空间中的坐标矩阵，例如 UMAP 坐标。
    k
        计算重叠率时使用的近邻数量。

    Returns
    -------
    float
        所有样本的平均 kNN overlap，取值范围通常为 ``0`` 到 ``1``。

    Examples
    --------
    比较 PCA 和 UMAP 坐标的局部邻域一致性::

        score = knn_overlap(X_pca, X_umap, k=15)
        print(score)

    在 UMAP 评估流程中和 trustworthiness 一起使用::

        tw = trustworthiness(X_pca, X_umap, n_neighbors=15)
        overlap = knn_overlap(X_pca, X_umap, k=15)"""
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
        atlas: Atlas,
        fit_sample_n: int | None = None,
        transform_batch_size: int = 50000,
        n_components: int = 2,
        n_pcs: int | None = None,
        n_neighbors: int = 15,
        min_dist: float = 0.5,
        spread: float = 1.0,
        metric: str = "euclidean",
        random_state: int = 42,
        n_jobs: int = 1,
        add_table: str = "obsm_X_umap",
        save_params_table: str = "uns_umap_params",
        eval_sample_n: int = 5000,
        save_eval_table: str = "uns_umap_eval"
):
    """基于 PCA embedding 计算 UMAP。

    该函数从 ``obsm_X_pca`` 读取 PCA 坐标，先抽样拟合 UMAP 模型，再把全量细胞分批 transform 到 UMAP 空间，并将坐标写入数据库。它类似 Scanpy 的 ``sc.tl.umap``，但采用抽样拟合和全量分块转换以降低内存占用。

    Parameters
    ----------
    atlas
        Atlas 对象。通常需要已经连接到 DuckDB 数据库，并包含该函数读取或写入所需的 ``obs``、``var``、表达矩阵或结果表。
    fit_sample_n
        用于拟合模型的细胞抽样数量。为 ``None`` 时使用全部细胞。
    transform_batch_size
        模型拟合后 transform 全量数据时的分块大小。
    n_components
        输出维度或参与计算的主成分数量。
    n_pcs
        用于 UMAP 的 PCA 维度数量。为 ``None`` 时使用 ``obsm_X_pca`` 中全部 PC。
    n_neighbors
        UMAP 构建局部邻域时使用的近邻数。
    min_dist
        UMAP 低维空间中点与点之间允许的最小距离。
    spread
        UMAP 低维空间的整体铺开尺度。通常要求 ``spread >= min_dist``。
    metric
        距离度量。PCA embedding 通常使用 ``"euclidean"``。
    random_state
        随机种子。设为固定整数时结果更容易复现。
    n_jobs
        计算使用的线程数。
    add_table
        写入数据库的结果表名。
    save_params_table
        保存本次运行参数的数据库表名。
    eval_sample_n
        用于评估 embedding 质量的细胞抽样数量。
    save_eval_table
        保存评估结果的数据库表名。

    Returns
    -------
    umap.UMAP
        拟合完成的 UMAP 对象，可继续用于 transform 或检查模型参数。

    Examples
    --------
    使用默认参数计算二维 UMAP::

        sap.tl.pca(atlas)
        sap.tl.umap(atlas)

    使用 50 万细胞拟合 UMAP，并分批转换全量数据::

        sap.tl.umap(
            atlas,
            fit_sample_n=500000,
            transform_batch_size=100000,
            n_neighbors=45,
            min_dist=0.2,
            random_state=42,
        )

    保存为不同名称的 UMAP 结果表::

        sap.tl.umap(
            atlas,
            add_table="obsm_X_umap_n45_d02",
            save_params_table="uns_umap_params_n45_d02",
            save_eval_table="uns_umap_eval_n45_d02",
        )"""

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

    if n_pcs is not None:
        n_pcs = int(n_pcs)

        if n_pcs <= 0:
            raise ValueError("n_pcs 必须大于 0")

        if n_pcs > len(pc_cols):
            raise ValueError(
                f"n_pcs={n_pcs} 超过 obsm_X_pca 中可用 PC 数量: {len(pc_cols)}"
            )

        pc_cols = pc_cols[:n_pcs]

    pc_cols_sql = ", ".join(pc_cols)

    logger.info(f"[UMAP] using n_pcs = {len(pc_cols)}")
    logger.info(f"[UMAP] n_neighbors = {n_neighbors}")
    logger.info(f"[UMAP] min_dist = {min_dist}")
    logger.info(f"[UMAP] spread = {spread}")

    if float(spread) < float(min_dist):
        raise ValueError(
            f"spread 必须大于或等于 min_dist，当前 spread={spread}, min_dist={min_dist}"
        )

    # 拟合 UMAP：SQL 先抽样
    if fit_sample_n is None:
        fit_query = f"""
            SELECT atlas_cell_id, {pc_cols_sql}
            FROM obsm_X_pca
            ORDER BY atlas_cell_id
        """
    else:
        seed = 0 if random_state is None else int(random_state)

        fit_query = f"""
            SELECT atlas_cell_id, {pc_cols_sql}
            FROM obsm_X_pca
            ORDER BY hash(atlas_cell_id + {seed})
            LIMIT {int(fit_sample_n)}
        """

    fit_df = conn.execute(fit_query).fetchdf()

    if len(fit_df) == 0:
        raise ValueError("拟合 UMAP 的样本为空")

    X_fit = fit_df.drop(columns=["atlas_cell_id"]).to_numpy(dtype=np.float32)

    # 拟合 UMAP 模型
    reducer = umap_lib.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        spread=spread,
        metric=metric,
        random_state=random_state,
        n_jobs=n_jobs
    )

    reducer.fit(X_fit)

    # 训练后评估 UMAP embedding 质量
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

    logger.info(f"[UMAP] trustworthiness = {trustworthiness_score:.4f}")
    logger.info(f"[UMAP] knn_overlap     = {knn_overlap_score:.4f}")

    # 简单自动评价
    if trustworthiness_score < 0.80:
        logger.info(" UMAP局部结构保持较弱，建议增大 fit_sample_n / 调整 n_neighbors / 检查 PCA")
    elif trustworthiness_score < 0.90:
        logger.info(" UMAP局部结构保持正常")
    else:
        logger.info(" UMAP局部结构保持很好")

    if knn_overlap_score < 0.20:
        logger.info(" KNN重叠率偏低，低维空间近邻和PCA空间差异较大")
    elif knn_overlap_score < 0.40:
        logger.info(" KNN重叠率正常，单细胞UMAP中常见")
    else:
        logger.info(" KNN重叠率较高，局部邻域保持很好")

    # 建输出表
    conn.execute(f"DROP TABLE IF EXISTS {add_table}")
    conn.execute(f"""
        CREATE TABLE {add_table} (
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
            "n_pcs",
            "n_neighbors",
            "min_dist",
            "spread",
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
            str(len(pc_cols)),
            str(n_neighbors),
            str(min_dist),
            str(spread),
            str(metric),
            str(random_state),
            str(fit_sample_n),
            str(transform_batch_size),
            "obsm_X_pca",
            add_table,
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

        conn.append(add_table, out_df)

        offset += len(batch_df)

        logger.info(f"[UMAP] transformed {offset}/{total_n}")

    logger.info(f"UMAP Done, 耗时: {(datetime.now() - start).total_seconds():.2f} 秒")

    return reducer