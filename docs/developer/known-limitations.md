# 已知限制

本页记录当前版本需要明确告知用户和开发者的限制，避免教程承诺还没有稳定支持的能力。

- `Atlas.minibatch_CSR()` 当前源码没有向调用者 `yield X_batch`，教程主线只写 `minibatch_dense()`。
- `export_duckdb_to_h5ad()` 当前主路径导出 `X_HyS_data.data`，不是任意表达字段。
- `sap.tl.umap()` 当前基于 `obsm_X_pca` 抽样拟合并 transform 全量 PCA，不是 Scanpy `neighbors + graph UMAP` 的完全等价实现。
- `sap.tl.kmeans()` 是 MiniBatchKMeans 聚类，不是 Leiden 或 Louvain。
- 旧脚本中出现但没有从当前源码导出的函数名不能写进用户教程。

## 第一版官网暂不处理的内容

- 不把根目录旧 `docs/` 迁移为新官网正文。
- 不承诺固定 benchmark 数字。
- 不把 `start_to_end.py` 中未验证的旧函数名写入用户教程。

