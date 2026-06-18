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

:::{grid-item-card} 画图的参数说明
:link: plot-parameter-guide
:link-type: doc
通用参数说明、画图决策速查表，以及 QC、HVG、PCA、UMAP 各类图的画法和解读。
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
单次遍历计算基因均值与方差。
:::

:::{grid-item-card} 在线协方差矩阵
:link: online-covariance-matrix
:link-type: doc
单次遍历构建基因-基因协方差矩阵，用于自定义降维。
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
从 obs 表读取标签，与 minibatch 同步，SGD 训练分类器。
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
plot-parameter-guide
marker-gene-ranking
welford-online-statistics
online-covariance-matrix
model-full-prediction
train-logistic-regression
train-pytorch-nn
custom-minibatch-kmeans
sql-query-cases
```
