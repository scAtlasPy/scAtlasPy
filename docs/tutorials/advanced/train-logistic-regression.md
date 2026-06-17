# 训练 Logistic Regression

本页说明如何用 multi-pass 模式训练逻辑回归分类器。本页也包含 ShuffleBuffer 机制说明和关键参数，这些知识适用于所有多轮训练案例。

## 开始前需要完成

和 {doc}`welford-online-statistics` 一样，需要先完成 QC、过滤、标准化、HVG、scale 和 `build_read_index()`。multi-pass 的核心前提是 `build_read_index()` 已经定义了要读取的细胞、基因和表达字段。

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

## 底层机制：ShuffleBuffer

multi-pass 和 single-pass 的核心区别在于 **ShuffleBuffer**：

```
Producer Threads (×10)
    ↓
  Queue
    ↓
Consumer Thread (×1)
    ↓
组装为 dense batch
    ↓
┌─────────────────────────────┐
│  ShuffleBuffer               │
│  缓存 buffer_batch_num 个 batch │
│  攒满 → 随机打乱 → 逐个输出      │
│  输出完 → 重置 → 继续攒          │
└─────────────────────────────┘
    ↓
  out_queue → yield X_batch
```

关键行为：

- **不跨 pass 记忆**：每个 pass 内部独立攒 batch、独立打乱。不同 pass 之间不共享 ShuffleBuffer 状态。
- **随机性来源**：batch 内的细胞顺序由 ShuffleBuffer 打乱决定。不同 pass 会从数据库重新读取（消费者重新从 RingBuffer 拿数据），但由于 ShuffleBuffer 的存在，即使同一 pass 内的相邻 batch 也不是严格顺序的。
- **自动循环**：外部 `get_minibatch_dense` 的 `while True` 循环负责反复创建新的 fetcher，直到达到 `max_batches`。

## 基础用法

```python
for batch_id, X_batch in enumerate(
    atlas.get_minibatch_dense(
        pass_mode="multi-pass",
        batch_size=2048,
        buffer_batch_num=5,
        max_batches=1000,
    )
):
    # 训练一步
    train_step(X_batch)
```

`max_batches=1000` 表示总共输出 1000 个 batch 后停止，防止无限循环。

你需要自己定义 `train_step()`。它接收一个 `(batch_size, n_genes)` 的 float32 dense numpy 数组。

## 关键参数

| 参数 | 默认值 | 含义 |
|---|---|---|
| `pass_mode` | `"single-pass"` | 设为 `"multi-pass"` 开启多轮随机遍历 |
| `batch_size` | `2048` | 每个 batch 的细胞数 |
| `buffer_batch_num` | `5` | 攒多少个 batch 后打乱。越大越随机，内存占用越高 |
| `max_batches` | `None` | 总输出 batch 数上限。**multi-pass 下建议一定设上限**，否则无限循环 |

### buffer_batch_num 的选择

| `buffer_batch_num` | 打乱粒度 | 内存占用（2000 基因 × 2048 batch_size） |
|---|---|---|
| 1 | 无打乱（batch 按原始顺序） | ~16 MB |
| 5 | 5 个 batch 范围内打乱 | ~80 MB |
| 10 | 10 个 batch 范围内打乱 | ~160 MB |
| 20 | 20 个 batch 范围内打乱 | ~320 MB |

一般推荐 5–10。太小则随机性不够，太大则内存占用高但随机性收益递减。

### max_batches 的选择

`max_batches` 的实际含义是"总共训练多少步"，而不是"遍历多少轮完整数据"。

- 如果 `max_batches=1000`，`batch_size=2048`，总共看到约 200 万个细胞样本（同一细胞可能被多次看到）
- 对于 SGD 类算法，通常设几千到几万
- 如果全量细胞数是 N，想遍历约 10 个 epoch：`max_batches ≈ 10 × N / batch_size`

## 案例：训练 Logistic Regression（含真实标签）

以下是一个完整的端到端示例，从 obs 表读取标签并与 minibatch 同步：

```python
import numpy as np
from sklearn.linear_model import SGDClassifier

# Step 1：从 obs 表读取所有通过过滤的细胞的标签
obs_df = atlas.get_obs_df()
filtered_cells = obs_df[obs_df["filter_cells"].notna()]  # filter_cell_id 不为空的细胞
labels = filtered_cells["cell_type"].values  # 假设 obs 中有 cell_type 列

# Step 2：创建模型
model = SGDClassifier(loss="log_loss", random_state=42)
classes = np.unique(labels).tolist()

# Step 3：训练
batch_size = 2048
cell_offset = 0

for batch_id, X_batch in enumerate(
    atlas.get_minibatch_dense(
        pass_mode="multi-pass",
        batch_size=batch_size,
        buffer_batch_num=5,
        max_batches=2000,
    )
):
    n_cells = X_batch.shape[0]

    # 标签与 batch 的 cell 顺序一致（都是按 filter_cell_id 排序）
    y_batch = labels[cell_offset : cell_offset + n_cells]

    # 如果是 multi-pass 的新一轮，需要重置 offset
    # 判断方式：如果当前 offset + n_cells > 总细胞数，说明开始新 pass
    if cell_offset + n_cells > len(labels):
        cell_offset = 0
        y_batch = labels[cell_offset : cell_offset + n_cells]

    model.partial_fit(X_batch, y_batch, classes=classes)

    cell_offset += n_cells

    if (batch_id + 1) % 100 == 0:
        print(f"batch {batch_id + 1}: accuracy = {model.score(X_batch, y_batch):.3f}")

print(f"Training done. Total batches: {batch_id + 1}")
```

```{note}
multi-pass 下每次遍历完所有细胞后会开启新一轮，`cell_offset` 在每轮开始时从 0 重新开始。代码中需要处理这个回绕逻辑。后续 scAtlasPy 会提供同步标签读取接口简化这个流程。
```

## 调试与验证

### 检查 batch 是否真的被随机化

```python
# 输出前 10 个 batch 的第一个细胞的第一个基因值，观察是否有规律
values = []
for batch_id, X_batch in enumerate(
    atlas.get_minibatch_dense(
        pass_mode="multi-pass",
        batch_size=2048,
        buffer_batch_num=5,
        max_batches=10,
    )
):
    values.append(float(X_batch[0, 0]))
    print(f"batch {batch_id}: X[0,0] = {X_batch[0, 0]:.4f}, shape = {X_batch.shape}")

# 如果值都一样，说明 buffer_batch_num 太小或 batch 没有被随机化
print(f"Unique X[0,0] values: {len(set(values))} / 10")
```

### 控制台输出解读

训练时会看到类似输出：

```
[Speed] output_batches=5, [ current=2.35 batch/s, 4812 cells/s, ][ avg=2.41 batch/s, 4935 cells/s ]
[get_minibatch_dense] multi-pass start pass=2, produced=120, remain=880
```

- 每个 pass 开始时打印当前 pass 编号和已输出/剩余 batch 数
- 达到 `max_batches` 后自动停止并打印 `reach max_batches`

## 常见问题

| 问题 | 可能原因 | 解决方法 |
|---|---|---|
| 训练一直不收敛 | `buffer_batch_num` 太小导致数据不够随机 | 增大到 10–20 |
| 内存不足 | `batch_size` × `buffer_batch_num` 太大 | 减小 batch_size 或 buffer_batch_num |
| 速度慢 | batch_size 太小，吞吐不够 | 增大 batch_size |
| 忘记设 max_batches 导致无限循环 | 默认 `max_batches=None` | 务必设一个上限 |

## 当前限制

- `Atlas.get_minibatch_csr()` 当前没有向调用者 `yield X_batch`。教程只推荐使用 `get_minibatch_dense()`。
- 标签读取当前需要用户自己从 obs 表查询并保证顺序对齐。后续会提供同步标签读取接口。

## 下一步

继续阅读 {doc}`train-pytorch-nn` 了解如何用 PyTorch 训练神经网络。
