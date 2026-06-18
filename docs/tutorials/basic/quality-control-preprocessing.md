# 质量控制与预处理

本教程覆盖单细胞分析流程的前半部分：计算 QC 指标、过滤细胞和基因、标准化、log1p 转换、高变基因选择、scale 和构建过滤矩阵索引。完成这些步骤后，数据就准备好进入 PCA、聚类和 UMAP。

本教程使用 **pbmc3k**（3k PBMC）数据集作为示例。

## 1. 打开已有 Atlas 数据库

```python
import scatlaspy as sap

atlas = sap.Atlas("run01", "./data")

# 根据服务器内存调整内存限制
atlas.execute_sql("SET memory_limit = '60GB'")
```

如果你的数据还没有导入，先参考 {doc}`preparing-data` 完成导入。

## 2. 计算 QC 指标

```python
sap.pp.calculate_qc_metrics(
    atlas,
    qc_vars={
        "mt": "MT-",
        "ribo": "^(RPS|RPL)",
    },
)
```

这一步会计算并写入以下指标：

- `obs` 表：`cell_total_counts`、`n_genes_by_counts`、`total_counts_mt`、`pct_counts_mt`、`total_counts_ribo`、`pct_counts_ribo`
- `var` 表：`mt` (bool)、`ribo` (bool)、`gene_total_counts`、`n_cells_by_counts`

参数说明：

- `"MT-"` 用于识别人类线粒体基因。小鼠数据通常需要改成 `"mt-"`。
- `"^(RPS|RPL)"` 用于识别核糖体蛋白基因。

```{note}
如果你的数据不是人类，请根据物种调整 `qc_vars`。如果不关心线粒体或核糖体比例，可以省略对应的 key。
```

### QC 可视化检查

计算完 QC 指标后，用以下图快速检查数据质量，判断后续过滤阈值是否合理。

**最高表达基因占比**：

```python
sap.pl.highest_expr_genes(atlas, n_top=20)
```

![最高表达基因](../_static/pbmc3k/highest_expr_genes.png)

每个基因的箱线图显示它在各细胞中占总 counts 的百分比。如果前 1–2 个基因占比显著高于其他（比如占了 30% 以上），可能存在核糖体 RNA 污染等技术偏差。

**QC 指标小提琴图**：

```python
sap.pl.violin_qc_metrics(
    atlas,
    keys=["n_genes_by_counts", "cell_total_counts", "pct_counts_mt"],
    sample_n=50000,
)
```

![QC 小提琴图](../_static/pbmc3k/violin_qc_metrics.png)

- `n_genes_by_counts`：每个细胞检测到的基因数。典型 PBMC 中位数约 500–1500，低于 200 说明数据质量可能有问题。
- `cell_total_counts`：每个细胞的总 UMI 数，分布应有明显峰值，没有极端高值尾巴。
- `pct_counts_mt`：线粒体基因占比。人类 PBMC 通常 <5–10%，超过 20% 可能是死细胞或受损细胞。

**QC 散点图**：

```python
sap.pl.scatter_qc_metrics(atlas, sample_n=50000)
```

![QC 散点图](../_static/pbmc3k/scatter_qc_metrics.png)

- 左图 `cell_total_counts` vs `pct_counts_mt`：如果线粒体比例随 counts 降低而升高，说明低测序深度细胞倾向于受损，属正常现象，但要确认过滤阈值合理。
- 右图 `cell_total_counts` vs `n_genes_by_counts`：应为正相关。如果 counts 高但基因数很低，可能是技术噪音。

## 3. 标记细胞和基因过滤结果

根据上一步 QC 图的分布情况确定阈值后，标记过滤结果：

```python
# 标记通过过滤条件的细胞
sap.pp.filter_cells(atlas, min_genes=200, min_counts=500)

# 标记通过过滤条件的基因
sap.pp.filter_genes(atlas, min_cells=3)
```

参数说明：

- `min_genes=200`：每个细胞至少检测到 200 个基因。
- `min_counts=500`：每个细胞总 UMI 或 reads 数至少为 500。
- `min_cells=3`：一个基因至少在 3 个细胞中被检测到。

这些函数不会直接删除细胞或基因，而是在表中写入过滤标记（`filter_cells` 列和 `filter_genes` 列）。这样你可以随时调整阈值后重新分析，不必重新导入原始数据。

## 4. 标准化、log1p 和高变基因

```python
# 对每个细胞做总量标准化，并写入 log1p 表达值
sap.pp.normalize_and_log1p(atlas, target_sum=1e4)

# 选择最有助于区分细胞状态的高变基因
sap.pp.highly_variable_genes(atlas, n_top_genes=2000)
```

这一步对应常见 Scanpy 流程中的：

- 总量标准化：让不同细胞的测序深度更可比。结果写入 `X_HyS_data.data_log1p`。
- `log1p` 转换：降低极高表达值的影响。默认使用自然对数（底数 $e$，即 `ln(1+x)`）。
- 选择高变基因：保留最能区分细胞状态的基因。结果写入 `var.highly_variable_genes`。

如果你更喜欢 Seurat v3 风格的高变基因选择：

```python
sap.pp.highly_variable_genes_seurat(atlas, n_top_genes=2000)
```

### 高变基因可视化

```python
sap.pl.highly_variable_genes(atlas, flavor="seurat")
```

![高变基因图](../_static/pbmc3k/highly_variable_genes.png)

高变基因（橙色高亮）应该在中等表达量区域集中出现。如果高变基因全部集中在极高表达量区域（最右侧），说明可能在原始 counts 空间做了选择，应该先标准化。如果调整了 `n_top_genes`，可以用此图确认选择范围是否合理。

### 分步控制

如果你需要更精细的控制，也可以分步操作：

```python
sap.pp.normalize_total(atlas, target_sum=1e4)
sap.pp.log1p(atlas)
```

其他可用变换：
- `sap.pp.expm1(atlas)`：log1p 的逆运算，将 `data_log1p` 还原为 `data_normalize`
- `sap.pp.sqrt(atlas)`：sqrt 变换，替代 log 变换的另一种选择

## 5. Scale

```python
# 对表达值进行 z-score 标准化，为 PCA 做准备
sap.pp.scale(atlas)
```

`scale()` 是按 id 分块处理的大数据安全版，结果写入 `X_HyS_data.data_scale` 和 `var.zero_scale_transform`。

常用可调参数：

| 参数 | 含义 | 常见修改 |
|---|---|---|
| `target_sum=1e4` | 每个细胞标准化后的总表达量 | 一般保持默认 |
| `n_top_genes=2000` | 选择多少个高变基因 | 1000、2000、3000 都常见 |

## 6. 构建过滤后的矩阵索引

```python
atlas.build_read_index(
    cell_condition="filter_cells",
    gene_condition="filter_genes",
    use_hvg=True,
    use_data="data_log1p",
)
```

这一步告诉 scAtlasPy：后续 PCA、聚类和 minibatch 读取使用通过细胞过滤、通过基因过滤、且属于高变基因的 `data_log1p` 表达矩阵。

参数说明：

| 参数 | 含义 |
|---|---|
| `cell_condition="filter_cells"` | 只使用 `obs.filter_cells` 为 TRUE 的细胞 |
| `gene_condition="filter_genes"` | 只使用 `var.filter_genes` 为 TRUE 的基因 |
| `use_hvg=True` | 进一步限制在高变基因 |
| `use_data="data_log1p"` | 读取 log1p 转换后的表达值 |

如果你想临时分析所有通过过滤的基因，可以把 `use_hvg=False`。如果你想用 scale 后表达值，可以把 `use_data="data_scale"`。

## 下一步

完成质量控制与预处理后，继续阅读 {doc}`clustering-cell-type-annotation`，进行 PCA 降维、聚类、UMAP 可视化和细胞类型注释。
