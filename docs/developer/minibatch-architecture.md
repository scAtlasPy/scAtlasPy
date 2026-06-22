# Minibatch 底层架构

本页面向算法开发者，说明 `get_minibatch_dense()` 在 single-pass 和 multi-pass 模式下的内部实现机制。

## Single-pass 模式

`get_minibatch_dense` 在 single-pass 模式下的工作流程：

1. **10 个生产者线程**按 `tid` 分片并行从 DuckDB 读取 `X_HyS_data_filtered` 表
2. **1 个消费者线程**从共享队列取出数据，通过 RingBuffer 按序组装成 CSR，再转为 dense 宽表
3. 宽表中隐式零值（某细胞未表达某基因）用 `var.zero_scale_transform` 填充
4. 最后通过 `out_queue` 将 batch 逐个 yield 给调用者

关键点：**每个 batch 只读一遍，按细胞顺序输出**。不涉及随机打乱。

### 速度信息解读

minibatch 系统每输出若干个 batch 会在控制台打印速度信息：

```
[Speed] output_batches=5, [ current=2.35 batch/s, 4812 cells/s, ][ avg=2.41 batch/s, 4935 cells/s ]
```

- `current`：最近一个 batch 的瞬时速度
- `avg`：自开始以来的平均速度

---

## Multi-pass 模式与 ShuffleBuffer

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

### `buffer_batch_num` 与打乱粒度

| `buffer_batch_num` | 打乱粒度 | 内存占用（2000 基因 × 2048 batch_size） |
|---|---|---|
| 1 | 无打乱（batch 按原始顺序） | ~16 MB |
| 5 | 5 个 batch 范围内打乱 | ~80 MB |
| 10 | 10 个 batch 范围内打乱 | ~160 MB |
| 20 | 20 个 batch 范围内打乱 | ~320 MB |

一般推荐 5–10。太小则随机性不够，太大则内存占用高但随机性收益递减。

### `max_batches` 的选择

`max_batches` 的实际含义是"总共训练多少步"，而不是"遍历多少轮完整数据"。

- 如果 `max_batches=1000`，`batch_size=2048`，总共看到约 200 万个细胞样本（同一细胞可能被多次看到）
- 对于 SGD 类算法，通常设几千到几万
- 如果全量细胞数是 N，想遍历约 10 个 epoch：`max_batches ≈ 10 × N / batch_size`

### 控制台输出解读

训练时会看到类似输出：

```
[Speed] output_batches=5, [ current=2.35 batch/s, 4812 cells/s, ][ avg=2.41 batch/s, 4935 cells/s ]
[get_minibatch_dense] multi-pass start pass=2, produced=120, remain=880
```

- 每个 pass 开始时打印当前 pass 编号和已输出/剩余 batch 数
- 达到 `max_batches` 后自动停止并打印 `reach max_batches`

---

## 常见问题

| 问题 | 可能原因 | 解决方法 |
|---|---|---|
| 遍历卡住 | 生产者线程提前退出 | 检查 `build_read_index` 是否成功运行 |
| 速度很慢 | batch_size 太小 | 增大 `batch_size` 到 4096 或 8192 |
| 内存不足 | batch_size 太大 | 减小 `batch_size` 到 1024 |
| 最后一个 batch 行为异常 | 细胞数不足一个 batch | 正常现象，代码自动处理 |
| 训练一直不收敛 | `buffer_batch_num` 太小导致数据不够随机 | 增大到 10–20 |
| 内存不足（训练） | `batch_size` × `buffer_batch_num` 太大 | 减小 batch_size 或 buffer_batch_num |
| 忘记设 max_batches 导致无限循环 | 默认 `max_batches=None` | 务必设一个上限 |

---

## 当前限制

- `Atlas.get_minibatch_csr()` 当前没有向调用者 `yield X_batch`。建议只使用 `get_minibatch_dense()`。
- 标签读取当前需要用户自己从 obs 表查询并保证顺序对齐。后续会提供同步标签读取接口。

## 相关页面

- {doc}`../tutorials/advanced/stream-mean-and-variance` — single-pass 实战案例
- {doc}`../tutorials/advanced/train-logistic-regression-with-minibatches` — multi-pass 实战案例
