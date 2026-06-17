# Welford 在线统计算法

本页说明如何用单次遍历（single-pass）计算每个基因的均值和方差。本页也包含 `get_minibatch_dense` 的底层机制说明和关键参数，这些知识适用于所有单次遍历案例。

面向希望把自己的算法接入 scAtlasPy 的开发者。

## 开始前需要完成

```python
sap.pp.filter_cells(atlas, min_genes=200, min_counts=500)
sap.pp.filter_genes(atlas, min_cells=3)
sap.pp.normalize_and_log1p(atlas)
sap.pp.highly_variable_genes(atlas)
sap.pp.scale(atlas)

atlas.build_read_index(
    cell_condition="filter_cells",
    gene_condition="filter_genes",
    use_hvg=True,
    use_data="data_scale",
)
```

## 底层机制简述

了解底层有助于写出高效代码。`get_minibatch_dense` 在 single-pass 模式下的工作流程：

1. **10 个生产者线程**按 `tid` 分片并行从 DuckDB 读取 `X_HyS_data_filtered` 表
2. **1 个消费者线程**从共享队列取出数据，通过 RingBuffer 按序组装成 CSR，再转为 dense 宽表
3. 宽表中隐式零值（某细胞未表达某基因）用 `var.zero_scale_transform` 填充
4. 最后通过 `out_queue` 将 batch 逐个 yield 给调用者

关键点：**每个 batch 只读一遍，按细胞顺序输出**。不涉及随机打乱。

## 基础用法

```python
for batch_id, X_batch in enumerate(
    atlas.get_minibatch_dense(
        pass_mode="single-pass",
        batch_size=2048,
    )
):
    print(f"batch {batch_id}: shape={X_batch.shape}, dtype={X_batch.dtype}")
    # X_batch 是 (n_cells, n_genes) 的 float32 numpy 数组

    if batch_id >= 2:  # 调试时只看前几个 batch
        break
```

最后一个 batch 的细胞数可能小于 `batch_size`，这是正常现象（总细胞数通常不是 batch_size 的整数倍）。

## 关键参数

| 参数 | 默认值 | 含义 |
|---|---|---|
| `pass_mode` | `"single-pass"` | 按顺序遍历一遍 |
| `batch_size` | `2048` | 每个 batch 的细胞数。更大 = 更高吞吐但更多内存 |
| `max_batches` | `None` | 最多输出多少个 batch。`None` = 遍历全部 |
| `buffer_batch_num` | `5` | single-pass 下仅影响内部 RingBuffer 的预取缓冲，不影响输出顺序 |

### batch_size 与内存的关系

一个 batch 的内存占用 ≈ `batch_size × n_genes × 4 bytes`（float32）。

举例：HVG 选 2000 个基因，`batch_size=2048`，每个 batch 约 16 MB。如果设 `batch_size=8192`，约 64 MB。根据你的可用内存选择合适的值。

### max_batches 的用法

```python
# 只看前 50 个 batch（比如快速验证代码逻辑）
for X_batch in atlas.get_minibatch_dense(
    pass_mode="single-pass",
    batch_size=2048,
    max_batches=50,
):
    # 处理逻辑...
```

## 案例：在线计算每个基因的均值和方差

Welford 在线算法只需要单次遍历，内存恒定：

```python
import numpy as np

n_total = 0
mean = None
m2 = None

for X_batch in atlas.get_minibatch_dense(
    pass_mode="single-pass",
    batch_size=2048,
):
    X_batch = np.asarray(X_batch, dtype=np.float64)
    n_batch = X_batch.shape[0]

    batch_mean = X_batch.mean(axis=0)
    batch_m2 = ((X_batch - batch_mean) ** 2).sum(axis=0)

    if mean is None:
        mean = batch_mean
        m2 = batch_m2
        n_total = n_batch
        continue

    # Welford 并行合并公式
    delta = batch_mean - mean
    new_total = n_total + n_batch
    mean = mean + delta * n_batch / new_total
    m2 = m2 + batch_m2 + delta**2 * n_total * n_batch / new_total
    n_total = new_total

variance = m2 / max(n_total - 1, 1)
std = np.sqrt(variance)

print(f"Processed {n_total} cells, {len(mean)} genes")
print(f"Mean range: [{mean.min():.4f}, {mean.max():.4f}]")
print(f"Std range:  [{std.min():.4f}, {std.max():.4f}]")
```

**注意**：如果 `use_data="data_scale"`（z-score 标准化），理论上每个基因的均值应接近 0、标准差应接近 1。如果偏离太大，可能是 scale 步骤有问题。

**为什么用 float64**：长期累加的在线统计算法对精度敏感。初始数据是 float32，转 float64 避免累积误差。

## 性能与调试

### 输出速度

minibatch 系统每输出若干个 batch 会在控制台打印速度信息：

```
[Speed] output_batches=5, [ current=2.35 batch/s, 4812 cells/s, ][ avg=2.41 batch/s, 4935 cells/s ]
```

- `current`：最近一个 batch 的瞬时速度
- `avg`：自开始以来的平均速度

### 常见问题

| 问题 | 可能原因 | 解决方法 |
|---|---|---|
| 遍历卡住 | 生产者线程提前退出 | 检查 `build_read_index` 是否成功运行 |
| 速度很慢 | batch_size 太小 | 增大 `batch_size` 到 4096 或 8192 |
| 内存不足 | batch_size 太大 | 减小 `batch_size` 到 1024 |
| 最后一个 batch 行为异常 | 细胞数不足一个 batch | 正常现象，代码自动处理 |

## 下一步

继续阅读 {doc}`online-covariance-matrix` 了解如何单次遍历构建基因-基因协方差矩阵。
