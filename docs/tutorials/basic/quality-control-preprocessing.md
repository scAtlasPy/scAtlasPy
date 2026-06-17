# 质量控制与预处理

本教程覆盖单细胞分析流程的前半部分：计算 QC 指标、过滤细胞和基因、标准化、log1p 转换、高变基因选择、scale 和构建过滤矩阵索引。完成这些步骤后，数据就准备好进入 PCA、聚类和 UMAP。

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

### 快速版与小数据

如果你的数据规模较小、可以完整放入内存，也可以使用快速版：

```python
sap.pp.calculate_qc_metrics_fast(
    atlas,
    qc_vars={"mt": "MT-", "ribo": "^(RPS|RPL)"},
)
```

对大多数大数据场景，主线推荐 `calculate_qc_metrics()`（分块安全版）。

## 3. 标记细胞和基因过滤结果

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

这些函数不会直接删除细胞或基因，而是在表中写入过滤标记（`filter_cells` 列和 `filter_genes` 列）。这样你可以随时调整阈值后重新分析。

小数据也可以使用 `filter_cells_fast()`，但大数据场景推荐使用 `filter_cells()`。

## 4. 标准化、log1p 和高变基因

```python
# 对每个细胞做总量标准化，并写入 log1p 表达值
sap.pp.normalize_and_log1p(atlas, target_sum=1e4)

# 选择最有助于区分细胞状态的高变基因
sap.pp.highly_variable_genes(atlas, n_top_genes=2000)
```

这一步对应常见 Scanpy 流程中的：

- 总量标准化：让不同细胞的测序深度更可比。结果写入 `X_HyS_data.data_log1p`。
- `log1p` 转换：降低极高表达值的影响。
- 选择高变基因：保留最能区分细胞状态的基因。结果写入 `var.highly_variable_genes`。

如果你更喜欢 Seurat v3 风格的高变基因选择：

```python
sap.pp.highly_variable_genes_seurat(atlas, n_top_genes=2000)
```

### 分步控制

如果你需要更精细的控制，也可以分步操作：

```python
sap.pp.normalize_total(atlas, target_sum=1e4)
sap.pp.log1p(atlas)
```

对应的快速版（小数据）为 `normalize_total_fast()` 和 `log1p_fast()`。

其他可用变换：
- `sap.pp.expm1(atlas)`：log1p 的逆运算，将 `data_log1p` 还原为 `data_normalize`
- `sap.pp.sqrt(atlas)` / `sap.pp.sqrt_fast(atlas)`：sqrt 变换，替代 log 变换的另一种选择

## 5. Scale

```python
# 对表达值进行 z-score 标准化，为 PCA 做准备
sap.pp.scale(atlas)
```

`scale()` 是大数据安全版（按 id 分块），结果写入 `X_HyS_data.data_scale` 和 `var.zero_scale_transform`。

如果你的数据规模较小、可以完整放入内存，也可以使用快速版：

```python
sap.pp.scale_fast(atlas)
```

```{note}
`scale()` 和 `scale_fast()` 的区别：
- `scale()`：按 id 分块处理，内存安全，适合大数据。
- `scale_fast()`：全量读入内存，速度更快但内存占用更高，适合小到中等数据。
选择建议：不确定时先用 `scale()`。
```

常用可调参数：

| 参数 | 含义 | 常见修改 |
|---|---|---|
| `target_sum=1e4` | 每个细胞标准化后的总表达量 | 一般保持默认 |
| `n_top_genes=2000` | 选择多少个高变基因 | 1000、2000、3000 都常见 |

## 6. 构建过滤后的矩阵索引

```python
atlas.filter_build_index(
    cell_condition="filter_cells",
    gene_condition="filter_genes",
    use_hvg=True,
    select_data="data_scale",
)
```

这一步告诉 scAtlasPy：后续 PCA、聚类和 minibatch 读取使用通过细胞过滤、通过基因过滤、且属于高变基因的 `data_scale` 表达矩阵。

参数说明：

| 参数 | 含义 |
|---|---|
| `cell_condition="filter_cells"` | 只使用 `obs.filter_cells` 为 TRUE 的细胞 |
| `gene_condition="filter_genes"` | 只使用 `var.filter_genes` 为 TRUE 的基因 |
| `use_hvg=True` | 进一步限制在高变基因 |
| `select_data="data_scale"` | 读取 scale 后的表达值 |

如果你想临时分析所有通过过滤的基因，可以把 `use_hvg=False`。如果你想用 log 表达值而不是 scale 后表达值，可以把 `select_data="data_log1p"`。

## 下一步

完成质量控制与预处理后，继续阅读 {doc}`clustering-cell-type-annotation`，进行 PCA 降维、聚类、UMAP 可视化和细胞类型注释。
