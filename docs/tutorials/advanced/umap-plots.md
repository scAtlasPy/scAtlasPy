# UMAP 图

UMAP 是 scAtlasPy 最核心的可视化工具。同一个函数可以根据 `color` 参数自动切换绘制模式。

## 开始前需要完成

需要已完成 UMAP 降维步骤（参考 {doc}`../basic/clustering-cell-type-annotation`）。

## 按 obs 列上色

```python
# 按 cluster
sap.pl.umap(atlas, color="kmeans", sample_n=100000, legend_loc="right_margin")

# 按样本/批次
sap.pl.umap(atlas, color="sample_id", sample_n=100000)

# 按细胞类型注释
sap.pl.umap(atlas, color="cell_type_manual", sample_n=100000)

# 加 where 条件筛选特定细胞
sap.pl.umap(atlas, color="kmeans", sample_n=50000, where="kmeans IN (0, 1, 2)")
```

**怎么读图**：

- cluster 是否在 UMAP 空间分开？如果分开好，说明聚类和降维结果一致。
- 不同样本/批次是否混合？如果某个样本形成孤立的岛，可能有批次效应。
- 用 `where` 条件放大查看特定 cluster 的内部结构。

**图例位置**：

```python
# 右侧图例
sap.pl.umap(atlas, color="kmeans", legend_loc="right_margin")

# 标签直接标在图内各 cluster 中心
sap.pl.umap(atlas, color="kmeans", legend_loc="on_data")
```

## 按基因表达上色

```python
# 单个基因
sap.pl.umap(
    atlas,
    color="CST3",
    use_data="data_log1p",
    sample_n=100000,
)

# 多个基因并排
sap.pl.umap(
    atlas,
    color=["CST3", "NKG7", "PPBP", "MS4A1"],
    use_data="data_log1p",
    sample_n=100000,
    ncols=2,
)
```

**怎么读图**：

- 基因表达是否局限在特定的 UMAP 区域？一个好的 marker gene 应该在某个 cluster 区域明显高表达，其余区域低表达。
- 多个基因时放在一起对比，确认每个基因的高表达区域是否对应预期的细胞类型。

**`ncols` 参数**：控制每行放几个子图。基因较多时建议 `ncols=3` 或 `ncols=4`。

## 混合模式

```python
# kmeans + 两个基因，自动判断类型分别画图
sap.pl.umap(
    atlas,
    color=["kmeans", "CST3", "NKG7"],
    use_data="data_log1p",
)
```

color 列表中的元素会自动判断是 obs 列还是基因名，然后分别调用对应的绘图逻辑。

## 全量流式绘图（避免 OOM）

```python
# sample_n=None 时走全量 streaming 绘图，分批次画点，适合展示完整 UMAP 给报告用
sap.pl.umap(
    atlas,
    color="cell_type_manual",
    sample_n=None,
    plot_batch_size=200000,
)
```

`plot_batch_size` 控制每批读取和绘制多少个细胞。内存不足时可以调小。

## UMAP 上色参数速查

| 你想看什么 | 改哪里 | 示例 |
|---|---|---|
| 按 cluster 上色 | `color="kmeans"` | `color="kmeans"` |
| 按样本/批次 | obs 列名 | `color="sample_id"` |
| 看一个基因 | 基因名 | `color="MS4A1"` |
| 看多个基因并排 | `color=[...]` + `ncols` | `color=["IL7R", "LYZ", "NKG7"], ncols=3` |
| 只看特定 cluster | `where` | `where="kmeans IN (0, 1, 2)"` |
| 数据太大画图慢 | `sample_n` | `sample_n=30000` |
| 保存图片 | `save_path` | `save_path="umap.png"` |
| 标签标在图内 | `legend_loc` | `legend_loc="on_data"` |

## 下一步

继续阅读 {doc}`marker-gene-ranking` 了解如何找到每个 cluster 的 marker gene。
