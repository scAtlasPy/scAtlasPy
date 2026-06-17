# Advanced Tutorials

进阶教程面向需要可视化检查、自定义算法接入、SQL 查询和数据库管理的用户。每个页面聚焦一个具体任务，不需要按顺序阅读。

## 重连数据库

::::{grid} 1 2 2 2
:gutter: 2

:::{grid-item-card} 重连数据库继续使用
:link: reconnect-database
:link-type: doc
连接已有 `.sasql` 数据库，查看表结构、已有字段，继续分析或导出。
:::

::::

## 画图和解释结果

::::{grid} 1 2 2 2
:gutter: 2

:::{grid-item-card} 质控图
:link: qc-plots
:link-type: doc
绘制 highest_expr_genes、violin_qc_metrics、scatter_qc_metrics。含通用参数说明和画图决策速查表。
:::

:::{grid-item-card} 高变基因图
:link: hvg-plots
:link-type: doc
绘制 highly_variable_genes 判断 HVG 选择是否合理。
:::

:::{grid-item-card} PCA 图
:link: pca-plots
:link-type: doc
方差解释率图和 PCA 散点图，判断降维质量和批次效应。
:::

:::{grid-item-card} UMAP 图
:link: umap-plots
:link-type: doc
按聚类、基因、样本上色，全量流式绘图，混合模式。
:::

:::{grid-item-card} Marker Gene 排名与验证
:link: marker-gene-ranking
:link-type: doc
rank_genes_groups、violin、dotplot、stacked_violin 全套验证流程。
:::

::::

## 单次遍历数据

::::{grid} 1 2 2 2
:gutter: 2

:::{grid-item-card} Welford 在线统计算法
:link: welford-online-statistics
:link-type: doc
单次遍历计算基因均值与方差，含底层 minibatch 机制说明。
:::

:::{grid-item-card} 在线协方差矩阵
:link: online-covariance-matrix
:link-type: doc
单次遍历构建基因-基因协方差矩阵，用于自定义降维。
:::

:::{grid-item-card} 收集全量矩阵
:link: collect-full-matrix
:link-type: doc
收集所有 batch，vstack 为完整 dense 矩阵用于一次性分析。
:::

:::{grid-item-card} 模型全量预测
:link: model-full-prediction
:link-type: doc
用已训练的 sklearn 模型对流式 batch 做全量预测。
:::

::::

## 多轮训练读取

::::{grid} 1 2 2 2
:gutter: 2

:::{grid-item-card} 训练 Logistic Regression
:link: train-logistic-regression
:link-type: doc
从 obs 表读取标签，与 minibatch 同步，SGD 训练分类器。含 ShuffleBuffer 机制说明。
:::

:::{grid-item-card} 训练 PyTorch 神经网络
:link: train-pytorch-nn
:link-type: doc
多轮随机读取训练 MLP，切换到 single-pass 做预测。
:::

:::{grid-item-card} 自定义 MiniBatchKMeans
:link: custom-minibatch-kmeans
:link-type: doc
从零实现 MiniBatchKMeans，理解 multi-pass 训练模式。
:::

::::

## SQL 查询

::::{grid} 1 2 2 2
:gutter: 2

:::{grid-item-card} SQL 查询案例
:link: sql-query-cases
:link-type: doc
完整案例：按 cluster 汇总、查询 marker gene 表达、提取绘图数据、导出 CSV。
:::

::::

```{toctree}
:hidden:
:maxdepth: 1

reconnect-database
qc-plots
hvg-plots
pca-plots
umap-plots
marker-gene-ranking
welford-online-statistics
online-covariance-matrix
collect-full-matrix
model-full-prediction
train-logistic-regression
train-pytorch-nn
custom-minibatch-kmeans
sql-query-cases
```
