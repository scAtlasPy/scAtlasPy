# Tutorials

教程按实际研究任务组织，而不是按软件内部模块组织。教程分为两级：

- **Basic**：面向普通用户的完整基础分析流程，从导入数据到细胞类型注释。
- **Advanced**：面向可视化检查、大数据读取、自定义模型训练和 SQL 查询的进阶用法。

第一次使用建议按 Basic 顺序阅读。

::::{grid} 1 2 2 2
:gutter: 2

:::{grid-item-card} Basic
:link: basic/index
:link-type: doc
准备数据 → 质量控制与预处理 → 聚类与细胞类型注释，覆盖标准单细胞分析全流程。
:::

:::{grid-item-card} Advanced
:link: advanced/index
:link-type: doc
重连数据库、画图检查结果、单次/多轮分批读取矩阵、SQL 案例查询。
:::

::::

```{toctree}
:hidden:
:maxdepth: 2

basic/index
advanced/index
```

## 该读哪一页

| 你的目标 | 推荐页面 | 看完后应能做到 |
|---|---|---|
| 第一次把数据放进平台 | {doc}`basic/preparing-data` | 创建 `.sasql` 数据库，用 `load_h5ad()` 导入数据，`describe()` 检查状态 |
| 跑常规单细胞流程 | {doc}`basic/quality-control-preprocessing` → {doc}`basic/clustering-cell-type-annotation` | 完成 QC、过滤、标准化、HVG、PCA、聚类、UMAP 和细胞类型注释 |
| 重连已有数据库继续分析 | {doc}`advanced/reconnect-database` | 重连 `.sasql`，查看表结构和分析进度，从中断处继续 |
| 判断分析结果是否合理 | {doc}`advanced/qc-plots` → {doc}`advanced/umap-plots` | 画 QC、HVG、PCA、UMAP 和 marker gene 图 |
| 数据太大，普通读入不稳定 | {doc}`basic/preparing-data` | 选择 `load_type` 参数，用流式导入写入数据库 |
| 想接入自己的算法 | {doc}`advanced/welford-online-statistics` 或 {doc}`advanced/train-logistic-regression` | 获取分批表达矩阵，用于自定义模型或统计 |
| 用 SQL 直接查询结果 | {doc}`advanced/sql-query-cases` | 按 cluster 统计、查询基因表达、导出中间结果 |
| 从其他平台迁移脚本 | {doc}`../how-to/migrate-from-other-platforms` | 将 Scanpy 脚本改写为 scAtlasPy 写法 |
| 导出结果到其他平台 | {doc}`../how-to/export-to-other-platforms` | 导出 h5ad 或 AnnData 给 Scanpy/cellxgene 使用 |

## 教程中的变量怎么理解

- `atlas`：一个分析项目，对应一个 `.sasql` 数据库文件。
- `obs`：细胞信息表，例如样本、批次、cluster、细胞类型等。
- `var`：基因信息表，例如基因名、是否高变基因等。
- `X_HyS_data`：表达矩阵中的非零表达值（长表格式）。
- `X_HyS_indptr`：CSR 行指针表，与 `X_HyS_data` 共同构成稀疏表达矩阵。
- `data`：原始表达值，存储在 `X_HyS_data.data`。
- `data_log1p`：标准化并取 `log1p` 后的表达值，常用于画基因表达图。
- `data_scale`：scale 后的表达值，常用于 PCA。
