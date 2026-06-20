# Marker Gene 排名与验证

本页说明如何找到每个 cluster 的 marker gene，并用多种图形验证 marker 是否真的在该 cluster 高表达。


## 工作流程

Marker gene 分析分为两步：
1. **计算**：`sap.tl.rank_genes_groups()` 按分组统计差异表达，结果写入数据库表
2. **画图**：`sap.pl.rank_genes_groups()` 从数据库读取结果绘制排名图

## 步骤 1：计算 Marker gene 排名

```python
# 按 kmeans 分组，计算每个 cluster 的差异表达基因
rank_result = sap.tl.rank_genes_groups(
    atlas,
    groupby="kmeans",
    use_data="data_log1p",
    n_genes=10,
    reference="rest",          # "rest" = vs 其余所有细胞
    mask_var="highly_variable_genes",  # 只在高变基因中搜索
)
```

**参数说明**：

| 参数 | 含义 |
|---|---|
| `groupby` | 按哪个 obs 列分组比较 |
| `use_data` | 使用哪个表达字段计算差异 |
| `n_genes` | 每个 group 保留前多少个 marker |
| `reference` | 参考组；`"rest"` = vs 其余细胞；也可指定某个 cluster |
| `mask_var` | 限制在哪些基因中搜索 marker（默认 `None` 使用全部基因） |
| `corr_method` | 多重检验校正方法（默认 `"benjamini-hochberg"`） |
| `add_table` | 结果表名称前缀（默认 `"rank_genes_groups"`） |

结果会写入以 `add_table` 为前缀的数据库表，`return_df=True` 时同时返回 pandas DataFrame。

## 步骤 2：画排名图

```python
# 从数据库读取已有的排名结果并绘制
sap.pl.rank_genes_groups(
    atlas,
    use_table="rank_genes_groups",
    groups=[0, 1, 2],       # 展示哪些 cluster；None = 全部
    n_genes=10,              # 每个 cluster 展示前几个基因
    ncols=4,                # 每行几个子图
)
```

**参数说明**：

| 参数 | 含义 |
|---|---|
| `use_table` | 排名结果表名称前缀（与 `add_table` 对应） |
| `groups` | 要展示的 cluster 列表；`None` = 全部 |
| `n_genes` | 每个 group 展示前多少个 marker |
| `score_key` | `var` 中保存得分的列名（默认 `"scores"`） |
| `gene_label` | 绘图时显示基因名的列（默认 `"names"`） |
| `ncols` | 每行子图数 |

## Marker gene 验证图

拿到排名结果后，用以下图验证 marker 是否真的在该 cluster 高表达。

### Rank genes 提琴图

```python
# 查看 cluster 0 的 top marker 在各个 group 中的表达
sap.pl.rank_genes_groups_violin(
    atlas,
    group=0,
    groupby="kmeans",
    use_table="rank_genes_groups",
    n_genes=8,
    use_expr_field="data_log1p",
)
```

蓝色是该 cluster，橙色是 reference/rest。一个好的 marker 应该在蓝色明显高于橙色。

```{note}
`rank_genes_groups_violin` 使用 `use_expr_field` 参数（不是 `use_data`），这是少数例外。
```

### 自定义基因列表的小提琴图

```python
# 指定你想看的基因和分组
sap.pl.violin(
    atlas,
    genes=["IL7R", "NKG7", "PPBP"],
    groupby="kmeans",
    use_data="data_log1p",
)
```

适合验证你自己确定的 marker gene 列表。

### Dotplot

```python
marker_genes = ["IL7R", "CD79A", "MS4A1", "CD8A", "LYZ", "NKG7", "PPBP"]

sap.pl.dotplot(
    atlas,
    genes=marker_genes,
    groupby="kmeans",
    use_data="data_log1p",
    standard_scale="var",
)
```

**怎么读图**：

- **圆点大小** = 表达该基因的细胞比例
- **圆点颜色** = 平均表达量
- 一个好的 marker gene 应该在某个 cluster 行有又大又红的点，其他行几乎没有

**`standard_scale="var"`**（默认 `None`）：跨基因归一化颜色，让不同表达量级的基因可以在同一个色条上比较。

### Stacked violin

```python
sap.pl.stacked_violin(
    atlas,
    genes=marker_genes,
    groupby="kmeans",
    use_data="data_log1p",
)
```

**怎么读图**：

- 每个格子是一个 gene × group 的组合
- 格子里的小提琴显示表达分布，颜色深浅表示中位数高低
- 适合同时观察多个基因在多个 group 中的完整表达分布
- 好的 marker 在其目标 group 中应该有一个深色的宽分布，其他 group 中颜色浅

## KMeans 聚类大小图

```python
sap.pl.kmeans_cluster_size(atlas, use_obs_col="kmeans")
```

**怎么读图**：展示每个 cluster 的细胞数量。如果某个 cluster 特别大或特别小（极端情况如 1 个 cluster 占了 80% 的细胞），可能需要调整 `n_clusters` 或检查数据是否存在批次效应。

## 下一步

- 如果对底层数据读取机制感兴趣，阅读 {doc}`welford-online-statistics`
- 用 SQL 直接查询 marker gene 表达：{doc}`sql-query-cases`
