# scAtlasPy

面向大规模单细胞转录组数据的 Python 分析平台。

scAtlasPy 适合这样的情况：你的研究数据集规模较大，完整读入内存不稳定，或者你希望把导入、质控、标准化、降维、聚类、可视化和导出流程保存成可复用的分析数据库。它保留接近 Scanpy 的使用习惯，但把中间结果保存在 `.sasql` 文件中，方便反复检查和继续分析。

::::{grid} 1 2 3 3
:gutter: 2

:::{grid-item-card} 安装环境
:link: installation
:link-type: doc
配置运行环境，并用 PBMC3k 示例确认最小流程可以跑通。
:::


:::{grid-item-card} 分析教程
:link: tutorials/index
:link-type: doc
按研究任务学习：导入数据、分块导入、预处理聚类和画图。
:::

:::{grid-item-card} 平台协作
:link: how-to/index
:link-type: doc
从其他平台迁移、数据来回转换、导出结果到其他平台。
:::

:::{grid-item-card} API 速查
:link: api/index
:link-type: doc
查看 `sap.io`、`sap.pp`、`sap.tl`、`sap.pl` 的公开函数。
:::

:::{grid-item-card} 开发者说明
:link: developer/index
:link-type: doc
了解数据模型、批读取、性能参数和贡献规范。
:::

:::{grid-item-card} 更新日志
:link: release-notes/index
:link-type: doc
查看各版本的更新内容、新功能和修复记录。
:::
::::

## 平台定位

scAtlasPy 是一个基于 DuckDB 的单细胞转录组分析平台。它保留接近 Scanpy 的使用习惯，但把中间结果保存在数据库中，适合需要持久化分析过程、反复检查结果、尤其是*数据规模超出内存容量*的研究任务。

### 数据规模与平台推荐

细胞数不是唯一的决定因素——可用内存、是否需要持久化、是否多批次合并同样重要。下表以 HVG 选取 2000–5000 个基因后的典型内存占用为参考：

| 数据规模（细胞数） | 推荐 | 说明 |
|---|---|---|
| < 20 万 | Scanpy / scAtlasPy 均可 | 16GB 内存即可用 Scanpy 完整分析，生态工具链更成熟；如果需要将中间结果持久化、中断后继续分析或 SQL 查询，选 scAtlasPy |
| 20 万 – 100 万 | scAtlasPy 推荐，Scanpy 也可 | Scanpy 需要 32GB+ 内存且 dense 化后压力增大；scAtlasPy 流式读写保持较低内存占用，增量 PCA/KMeans 无需一次性加载全量矩阵 |
| 100 万 – 500 万 | scAtlasPy | 流式分块计算避免 OOM；随机窗口导入可在导入时打乱细胞顺序；UMAP 支持分批拟合 |
| > 500 万 | scAtlasPy | 多文件合并导入（`list_random`）；SQL 层抽样查询无需加载全量数据；全流程分块处理，内存可控 |



### 适合什么研究任务

- 单细胞数据规模较大，内存不足以完整加载。
- 希望分析中间结果持久化保存，随时继续聚类、可视化、marker gene 查询或导出。
- 需要将多个样本/批次的数据合并到统一数据库中分析。
- 需要批量读取表达矩阵接入自定义机器学习算法。

## 先读哪一部分

如果你主要关心自己的生物学问题，先完成 {doc}`installation`，再按需要读 {doc}`tutorials/basic/quality-control-preprocessing` 和 {doc}`tutorials/advanced/qc-plots` 等教程。

如果你已经会用 Scanpy，读 {doc}`how-to/migrate-from-other-platforms`，重点看哪些步骤会写入数据库字段，以及什么时候需要导出回 h5ad。

如果你想通过 scAtlasPy 获取数据用于你的算法，读 {doc}`tutorials/advanced/welford-online-statistics`、{doc}`tutorials/advanced/train-logistic-regression` 和 {doc}`tutorials/advanced/sql-query-cases`。

如果你想修复或扩展平台本身，读 {doc}`developer/data-model`、{doc}`developer/performance` 和 {doc}`developer/documentation`。

```{toctree}
:hidden:
:maxdepth: 2

installation
tutorials/index
how-to/index
api/index
developer/index
release-notes/index
references
```
