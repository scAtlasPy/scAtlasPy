# 模型全量预测

本页说明如何用已训练的 sklearn 模型对流式 batch 做全量预测。

## 开始前需要完成

与 {doc}`welford-online-statistics` 相同，需要先完成 `build_read_index()`。此外需要有一个已训练好的模型。

## 案例：用模型做全量预测

```python
# 假设你已经有一个训练好的 sklearn 模型
from sklearn.linear_model import LogisticRegression

model: LogisticRegression  # 已训练的模型
all_preds = []

for X_batch in atlas.get_minibatch_dense(
    pass_mode="single-pass",
    batch_size=2048,
):
    preds = model.predict(X_batch)
    all_preds.append(preds)

all_preds = np.concatenate(all_preds)
print(f"Predicted {len(all_preds)} cells")
```

## 与 built-in 工具的关系

内置工具的训练和预测使用了不同的 pass_mode：

| 工具 | 训练阶段 | 预测/转换阶段 |
|---|---|---|
| `sap.tl.pca` | `multi-pass` | `single-pass` |
| `sap.tl.kmeans` | `multi-pass` | `single-pass` |

```python
# PCA 的 fit（内部 multi-pass）-> transform（内部 single-pass）
sap.tl.pca(atlas, fit_batches=1000)

# KMeans 的 fit（内部 multi-pass）-> predict（内部 single-pass）
sap.tl.kmeans(atlas, fit_batches=1000)
```

这些工具在训练阶段使用 multi-pass 获取随机化 batch，预测/转换阶段使用 single-pass 按序处理全量细胞。如果你要实现类似的 "训练 + 推理" 模式，可以参考这个设计。

## 下一步

如果你的算法需要反复随机读取数据做迭代训练，参考 {doc}`train-logistic-regression`。
