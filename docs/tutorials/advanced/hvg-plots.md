# 高变基因图

本页说明如何绘制高变基因图，判断 HVG 选择是否合理。

## 开始前需要完成

需要已完成标准化、高变基因选择步骤（参考 {doc}`../basic/quality-control-preprocessing`）。

## 高变基因图

`sap.pl.highly_variable_genes()` 是 HVG 绘图的统一入口，通过 `flavor` 参数切换绘图风格：

```python
# Seurat v3 风格（默认）
sap.pl.highly_variable_genes(atlas, flavor="seurat")

# 标准方差/变异系数风格
sap.pl.highly_variable_genes(atlas, flavor="cv")

# 方差风格
sap.pl.highly_variable_genes(atlas, flavor="var")
```

**`flavor` 参数**：

| `flavor` | 说明 |
|---|---|
| `"seurat"`（默认） | Seurat v3 风格，展示标准化方差 vs 均值 |
| `"cv"` | 变异系数风格，展示 normalized variance vs mean |
| `"var"` | 方差风格，与 `"cv"` 调用同一底层实现 |

**怎么读图**：

- 高变基因（高亮显示）应该在中等表达量区域集中出现。如果高变基因集中在极高表达量区域，说明可能在原始 counts 空间选择了高变基因，应该先做标准化。
- **判断标准**：高变基因应该在 mean 的中段分散开，而不是全部挤在最右侧。如果你调整了 `n_top_genes`，可以观察图来确认选择的高变基因是否合理。

```{note}
`sap.pp.highly_variable_genes()` 是计算高变基因的预处理函数，结果写入 `var.highly_variable_genes`。
`sap.pl.highly_variable_genes()` 是画图函数，从 `var` 表中读取已有结果进行绘制。
```

## 下一步

确认高变基因选择合理后，继续阅读 {doc}`pca-plots` 了解 PCA 降维质量的判断方法。
