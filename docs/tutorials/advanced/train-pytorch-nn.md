# 训练 PyTorch 神经网络

本页说明如何用 multi-pass 模式训练简单 MLP 神经网络，并切换到 single-pass 做预测。

## 开始前需要完成

与 {doc}`train-logistic-regression` 相同，需要先完成 `build_read_index()`。

## 案例：训练简单神经网络（PyTorch）

```python
import torch
import torch.nn as nn

# 简单 MLP
model = nn.Sequential(
    nn.Linear(n_genes, 128),
    nn.ReLU(),
    nn.Linear(128, n_classes),
)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

model.train()
for batch_id, X_batch in enumerate(
    atlas.get_minibatch_dense(
        pass_mode="multi-pass",
        batch_size=2048,
        buffer_batch_num=5,
        max_batches=5000,
    )
):
    X_tensor = torch.from_numpy(np.asarray(X_batch, dtype=np.float32))
    y_tensor = get_label_tensor(batch_id, batch_size)  # 需要自己实现标签获取

    optimizer.zero_grad()
    loss = criterion(model(X_tensor), y_tensor)
    loss.backward()
    optimizer.step()

    if (batch_id + 1) % 200 == 0:
        print(f"batch {batch_id + 1}: loss = {loss.item():.4f}")

# 预测阶段切换到 single-pass
model.eval()
all_preds = []
for X_batch in atlas.get_minibatch_dense(
    pass_mode="single-pass",
    batch_size=2048,
):
    with torch.no_grad():
        X_tensor = torch.from_numpy(np.asarray(X_batch, dtype=np.float32))
        preds = model(X_tensor).argmax(dim=1).numpy()
    all_preds.append(preds)

all_preds = np.concatenate(all_preds)
```

```{tip}
训练用 multi-pass + 随机打乱，预测用 single-pass + 顺序遍历。这和内置 PCA/KMeans 的设计模式一致。
```

## 下一步

继续阅读 {doc}`custom-minibatch-kmeans` 了解如何从零实现 MiniBatchKMeans。
