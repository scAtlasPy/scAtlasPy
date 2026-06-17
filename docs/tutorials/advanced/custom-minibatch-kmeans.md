# 自定义 MiniBatchKMeans

本页说明如何从零实现 MiniBatchKMeans，理解 multi-pass 训练模式。

## 开始前需要完成

与 {doc}`train-logistic-regression` 相同，需要先完成 `build_read_index()`。

## 案例：自定义 MiniBatchKMeans

如果想实现自己的聚类算法，multi-pass 提供了理想的训练数据流：

```python
import numpy as np

n_clusters = 10
n_genes = None  # 从第一个 batch 推断
centroids = None
counts = None

for batch_id, X_batch in enumerate(
    atlas.get_minibatch_dense(
        pass_mode="multi-pass",
        batch_size=2048,
        buffer_batch_num=5,
        max_batches=1000,
    )
):
    X_batch = np.asarray(X_batch, dtype=np.float64)

    if centroids is None:
        n_genes = X_batch.shape[1]
        # 随机初始化中心
        rng = np.random.default_rng(42)
        centroids = X_batch[rng.choice(X_batch.shape[0], n_clusters, replace=False)].copy()
        counts = np.zeros(n_clusters)

    # E-step：分配最近中心
    distances = np.linalg.norm(X_batch[:, None, :] - centroids[None, :, :], axis=2)
    assignments = distances.argmin(axis=1)

    # M-step：更新中心（mini-batch 风格）
    lr = 1.0 / (1.0 + counts[assignments])
    for k in range(n_clusters):
        mask = assignments == k
        if mask.sum() > 0:
            centroids[k] = (1 - lr[mask][:, None]) * centroids[k] + (lr[mask][:, None] * X_batch[mask]).sum(axis=0)
            counts[k] += mask.sum()
```

## 与内置工具的关系

```python
# PCA fit 使用 multi-pass，内部实现和案例 1 类似
sap.tl.pca(atlas, fit_batches=1000, buffer_batch_num=5)

# KMeans fit 也使用 multi-pass
sap.tl.kmeans(atlas, fit_batches=1000, buffer_batch_num=5, n_clusters=10)
```

两个函数都开放了 `fit_batches` 和 `buffer_batch_num` 参数，和直接使用 `get_minibatch_dense(multi-pass)` 的参数含义完全一致。

## 下一步

了解如何用 SQL 直接查询数据库中的结果，参考 {doc}`sql-query-cases`。
