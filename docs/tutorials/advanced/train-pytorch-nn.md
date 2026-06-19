# 训练 PyTorch 神经网络

本页说明如何用 multi-pass 模式训练简单 MLP 神经网络，并切换到 single-pass 做预测。

## 开始前需要完成

与 {doc}`train-logistic-regression` 相同，需要先完成 `build_read_index()`。

## 案例：训练简单神经网络（PyTorch）

以下是一个完整的端到端示例，从 obs 表读取标签并与 minibatch 同步：

```python
import torch
import torch.nn as nn
import numpy as np

# Step 1：从 obs 表读取所有通过过滤的细胞的标签
obs_df = atlas.get_obs_df()
filtered_cells = obs_df[obs_df["filter_cells"].notna()]
labels = filtered_cells["cell_type"].values  # 假设 obs 中有 cell_type 列
all_classes = np.unique(labels)

class_to_idx = {c: i for i, c in enumerate(all_classes)}
y_all = np.array([class_to_idx[c] for c in labels])
n_classes = len(all_classes)

# Step 2：创建简单 MLP
n_genes = None  # 从第一个 batch 推断
model = None
optimizer = None
criterion = nn.CrossEntropyLoss()

# Step 3：训练
batch_size = 2048
cell_offset = 0

model.train()
for batch_id, X_batch in enumerate(
    atlas.get_minibatch_dense(
        pass_mode="multi-pass",
        batch_size=batch_size,
        buffer_batch_num=5,
        max_batches=5000,
    )
):
    n_cells = X_batch.shape[0]

    if model is None:
        n_genes = X_batch.shape[1]
        model = nn.Sequential(
            nn.Linear(n_genes, 128),
            nn.ReLU(),
            nn.Linear(128, n_classes),
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 标签与 batch 的 cell 顺序一致
    if cell_offset + n_cells > len(y_all):
        cell_offset = 0
    y_batch = y_all[cell_offset : cell_offset + n_cells]

    X_tensor = torch.from_numpy(np.asarray(X_batch, dtype=np.float32))
    y_tensor = torch.tensor(y_batch, dtype=torch.long)

    optimizer.zero_grad()
    loss = criterion(model(X_tensor), y_tensor)
    loss.backward()
    optimizer.step()

    cell_offset += n_cells

    if (batch_id + 1) % 200 == 0:
        print(f"batch {batch_id + 1}: loss = {loss.item():.4f}")

# 预测阶段切换到 single-pass
model.eval()
all_preds = []
for X_batch in atlas.get_minibatch_dense(
    pass_mode="single-pass",
    batch_size=batch_size,
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
