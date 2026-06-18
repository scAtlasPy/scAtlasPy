# API

API 页面用于查函数名和参数，不建议作为入门教程。第一次使用请先阅读 {doc}`../installation` 和 {doc}`../tutorials/index`。

scAtlasPy 通常这样导入：

```python
import scatlaspy as sap
```

常用入口：

- `sap.Atlas`：创建或打开分析数据库。
- `sap.io`：导入、导出数据。
- `sap.pp`：质控、过滤、标准化、高变基因、scale。
- `sap.tl`：PCA、聚类、UMAP、手动注释。
- `sap.pl`：QC 图、PCA 图、UMAP 图、marker gene 图。

```{note}
多数函数会把结果写入 Atlas 数据库，而不是返回新的 AnnData 对象。需要检查结果时，可以使用 `atlas.query(...)`。
```

```{toctree}
:maxdepth: 2

atlas
io
preprocessing
tools
plotting
```
