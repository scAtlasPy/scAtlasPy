# PCA 图

本页说明如何绘制 PCA 方差解释率图和散点图，判断降维质量和批次效应。

## 开始前需要完成

需要已完成 PCA 降维步骤（参考 {doc}`../basic/clustering-cell-type-annotation`）。

## 方差解释率

```python
# 各主成分的方差解释率
sap.pl.pca_variance_ratio(atlas, n_pcs=50)

# 累计方差解释率
sap.pl.pca_variance_ratio_cumsum(atlas, n_pcs=50)
```

**怎么读图**：

- **方差解释率图**：第一个 PC 通常解释最多变异。观察曲线是否快速下降——如果前十来个 PC 解释了大量变异，后续趋于平缓，这是正常形态。
- **累计图**：在单细胞数据中，前 50 个 PC 累计解释 15–30% 的变异属于正常范围。如果前几个 PC 的解释率极低（<1%），可能需要检查数据过滤是否合理。
- **批次效应判断**：如果 PC1 解释了异常高的变异（>50%），很可能不是生物信号而是批次效应。

**保存**（注意：这两个函数使用 `save` 参数，不是 `save_path`）：

```python
sap.pl.pca_variance_ratio(atlas, n_pcs=50, save=".pdf")
sap.pl.pca_variance_ratio_cumsum(atlas, n_pcs=50, save="my_pca_cum.png")
```

## PCA 散点图

```python
# 按聚类结果上色
sap.pl.pca(atlas, color="kmeans", sample_n=100000)

# 按某个基因表达上色
sap.pl.pca(atlas, color="CST3", use_data="data_log1p")

# 指定不同的主成分维度
sap.pl.pca(atlas, color="kmeans", x_pc=1, y_pc=2)

# 返回数据用于自定义分析
plot_df = sap.pl.pca(atlas, color="kmeans", return_df=True)
```

**怎么读图**：

- 按 `kmeans` 上色：观察不同 cluster 在 PCA 空间是否初步分开。如果所有 cluster 在 PC1/PC2 上完全混合，后续 UMAP 也不会有好的分离。
- 按基因表达上色：确认关注的基因是否驱动了主要变异方向。如果某个 marker gene 的表达和 PC1 高度一致，说明它是数据中的主要变异来源。

**`legend_loc` 选项**：

| 值 | 效果 |
|---|---|
| `"right_margin"` | 图例在右侧（默认），适合类别数适中 |
| `"on_data"` | 标签直接标在各类别的中位数位置，适合类别较多 |
| `None` | 不显示图例 |

## 下一步

继续阅读 {doc}`umap-plots` 了解 UMAP 可视化方法。
