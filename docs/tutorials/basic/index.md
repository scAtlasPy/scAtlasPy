# Basic Tutorials

基础教程面向普通用户，按分析流程顺序覆盖从数据导入到细胞类型注释的完整过程。第一次使用建议按顺序阅读。

::::{grid} 1 2 2 2
:gutter: 2

:::{grid-item-card} Preparing Data
:link: preparing-data
:link-type: doc
创建 Atlas 数据库、导入 h5ad 或非 h5ad 数据、连接已有数据库继续分析。
:::

:::{grid-item-card} 质量控制与预处理
:link: quality-control-preprocessing
:link-type: doc
QC 指标、过滤细胞和基因、标准化、log1p、高变基因、scale 和构建过滤矩阵。
:::

:::{grid-item-card} 聚类与细胞类型注释
:link: clustering-cell-type-annotation
:link-type: doc
PCA 降维、KMeans 聚类、UMAP 可视化、marker gene 排名和自动细胞类型注释。
:::

:::{grid-item-card} Notebook Output Example
:link: notebook-output-example
:link-type: doc
Render a Jupyter notebook with executed code, tables, and figures.
:::

:::{grid-item-card} PBMC3K end-to-end workflow
:link: pbmc3k-basic-workflow
:link-type: doc
Run a complete on-disk PBMC3K analysis from h5ad import through QC, preprocessing, PCA, clustering, UMAP, marker ranking, and manual annotation.
:::

::::

```{toctree}
:hidden:
:maxdepth: 1

preparing-data
quality-control-preprocessing
clustering-cell-type-annotation
notebook-output-example
pbmc3k-basic-workflow
```
