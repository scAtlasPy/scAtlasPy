# 在线协方差矩阵

本页说明如何单次遍历构建基因-基因协方差矩阵，用于自定义降维。

## 开始前需要完成

与 {doc}`welford-online-statistics` 相同，需要先完成 `build_read_index()`。

## 案例：在线计算基因-基因协方差矩阵

如果只是想做 PCA 之外的自定义降维，可以单次遍历构建协方差矩阵：

```python
import numpy as np

n_total = 0
mean = None
cov_sum = None  # 外积累加

for X_batch in atlas.get_minibatch_dense(
    pass_mode="single-pass",
    batch_size=2048,
):
    X_batch = np.asarray(X_batch, dtype=np.float64)
    n_batch = X_batch.shape[0]

    if mean is None:
        mean = X_batch.mean(axis=0)
        n_total = n_batch
        centered = X_batch - mean
        cov_sum = centered.T @ centered
        continue

    # 更新均值
    delta = (X_batch.mean(axis=0) - mean) * n_batch / (n_total + n_batch)
    mean = mean + delta
    n_total += n_batch

    # 更新协方差
    centered = X_batch - mean
    cov_sum += centered.T @ centered

cov_matrix = cov_sum / max(n_total - 1, 1)
print(f"Covariance matrix shape: {cov_matrix.shape}")
```

```{note}
协方差矩阵大小是 `n_genes × n_genes`。如果 HVG 选了 2000 个基因，矩阵 2000×2000（约 32 MB float64），可接受。如果选了 8000 个基因，矩阵 8000×8000（约 512 MB），需注意内存。
```

## 下一步

继续阅读 {doc}`collect-full-matrix` 了解如何收集所有 batch 做一次性分析。
