# 性能参数

这一页说明哪些参数会影响运行速度、内存占用和结果稳定性。普通使用者可以先使用教程默认值；当数据集变大或服务器资源有限时，再回到本页调整。

## 主要参数

- DuckDB `memory_limit`：限制数据库可使用的内存。
- DuckDB `PRAGMA threads`：控制数据库查询线程数。
- h5ad 导入的 `batch_size`：每次读取多少细胞。
- 大 h5ad 导入的 `mega_batch_factor`：控制更大读取块的组合方式。
- `build_read_index()` 的 `cell_condition`、`gene_condition`、`use_hvg`、`use_data`：控制使用哪些细胞、哪些基因和哪个表达字段构建过滤矩阵，直接影响后续 PCA 和 minibatch 读取的数据规模。
- `minibatch_dense()` 的 `batch_size`、`buffer_batch_num`、`max_batches`：`batch_size` 影响单批内存占用和吞吐，`buffer_batch_num` 影响 multi-pass 下随机化程度，`max_batches` 控制总输出 batch 数上限。
- PCA / KMeans 的 `fit_batches`：控制训练时读取多少批数据。
- UMAP 的 `fit_sample_n`、`transform_batch_size`：控制拟合样本量和分批 transform 大小。

## 建议起点

| 场景 | 建议 |
|---|---|
| quickstart 或小数据 | `SET memory_limit = '4GB'` |
| 中等规模数据 | `SET memory_limit = '30GB'` 或 `'60GB'` |
| 大 h5ad 导入 | 先使用 `batch_size=4096` |
| PCA 或 KMeans 不稳定 | 适当增加 `fit_batches` |
| UMAP 太慢 | 降低 `fit_sample_n` 或先抽样画图 |

任何 benchmark 数字都必须来自当前源码和当前数据重新测试。本页不承诺固定性能。

