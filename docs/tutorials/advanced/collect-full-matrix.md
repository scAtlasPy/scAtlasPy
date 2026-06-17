# 收集所有 batch 做一次性分析

本页说明如何收集所有 batch，vstack 为完整 dense 矩阵用于一次性分析。

## 开始前需要完成

与 {doc}`welford-online-statistics` 相同，需要先完成 `build_read_index()`。

## 案例：收集所有 batch 做一次性分析

如果过滤后矩阵可以放进内存，直接收集所有 batch：

```python
import numpy as np

batches = []
for X_batch in atlas.get_minibatch_dense(
    pass_mode="single-pass",
    batch_size=2048,
):
    batches.append(np.asarray(X_batch, dtype=np.float32))

X_full = np.vstack(batches)
print(f"Full dense matrix: {X_full.shape}, memory: {X_full.nbytes / 1e9:.2f} GB")
```

```{warning}
只有在你确认过滤后矩阵足够小（比如 <10 万细胞 × <2000 基因 ≈ 800 MB）时才这样做。
大数据场景下 `np.vstack` 会导致 OOM。
```

## 下一步

继续阅读 {doc}`model-full-prediction` 了解如何用已训练模型对流式 batch 做全量预测。
