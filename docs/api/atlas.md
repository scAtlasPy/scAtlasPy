# Atlas 数据库对象

`Atlas` 表示一个分析项目，对应一个 `.sasql` 数据库文件。普通使用者通常只需要知道：先创建 `atlas`，再把它传给导入、预处理、聚类和画图函数。

```python
import scatlaspy as sap

# 创建或打开 ./data/run01.sasql
atlas = sap.Atlas("run01", "./data")
```

```{eval-rst}
.. currentmodule:: scatlaspy

.. autosummary::
   :toctree: generated/
   :nosignatures:

   Atlas
```

## 常用方法

- `close()`：关闭数据库连接。
- `execute_sql(sql)`：执行 SQL 命令，例如设置内存上限。
- `query(query)`：执行查询并返回 pandas DataFrame。
- `filter_build_index(...)`：指定后续批读取或 PCA 使用哪些细胞、基因和表达字段。
- `minibatch_dense(...)`：分批读取 dense 表达矩阵，主要给算法开发者使用。

