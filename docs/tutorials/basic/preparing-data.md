# Preparing Data

本教程回答一个问题：怎样使用 `load_h5ad()` 创建新的 Atlas 数据库并导入 h5ad 数据。

导入完成后，数据会保存为一个 `.sasql` 数据库文件。后续质控、过滤、标准化、降维、聚类和绘图都围绕这个 Atlas 数据库继续进行。

## 创建 Atlas 数据库并导入数据

### 1. 创建新的 Atlas 数据库

```python
import scatlaspy as sap

atlas = sap.Atlas("pbmc_demo", "./data")

# 设置 DuckDB 可使用的内存上限；通常不要超过服务器可用内存
atlas.execute_sql("SET memory_limit = '8GB'")
```

这段代码会在 `./data` 目录下创建：

```text
pbmc_demo.sasql
```

| 位置 | 含义 | 建议 |
|---|---|---|
| `"pbmc_demo"` | Atlas 数据库名称 | 使用项目、样本或批次名称 |
| `"./data"` | 数据库保存目录 | 放在空间充足的磁盘 |

```{note}
建议一个分析项目对应一个新的 Atlas 数据库。不要把不相关的数据反复导入同一个 `.sasql` 文件。
```

### 2. 使用 `load_h5ad()` 导入数据

`load_h5ad()` 是统一入口，通过 `load_type` 选择策略（默认 `"random"`）：

| `load_type` | 含义 | 何时选用 |
|---|---|---|
| `"order"` | 顺序导入，保留原始细胞顺序 | 需要保留行顺序、导入 obsm/varm |
| `"random"`（默认） | 随机窗口导入，打乱细胞顺序 | 单个 h5ad，希望随机化 |
| `"list_random"` | 多文件全局随机混合导入 | 多个样本/批次合并 |

#### 方式 A：顺序导入

```python
atlas.load_h5ad("pbmc3k.h5ad", load_type="order", store_type="count")
```

#### 方式 B：随机窗口导入（默认）

```python
atlas.load_h5ad("large.h5ad", store_type="count")
# 等价于 load_type="random"，cells_per_block=500，blocks_per_pool=10
```

#### 方式 C：多个文件合并导入

```python
atlas.load_h5ad(
    ["sample1.h5ad", "sample2.h5ad", "sample3.h5ad"],
    load_type="list_random",
    store_type="count",
)
```

多文件导入时需要注意：
- 每个 h5ad 文件应使用同一套基因。
- 基因顺序最好已经一致；否则导入前需要先整理。
- 样本、批次、实验条件等信息应保存在各自 h5ad 的 `obs` 中，方便导入后按列查询。

### 3. 关键参数说明

| 参数 | 默认值 | 含义 |
|---|---|---|
| `load_type` | `"random"` | 导入策略：`"order"` / `"random"` / `"list_random"` |
| `store_type` | `"count"` | 目标表达尺度。`"count"` 存原始 counts，`"log"` 存 log1p 值。自动检测源数据并转换 |
| `cells_per_block` | `500` | 每个连续读取 block 的细胞数。越大读速越快，内存越大 |
| `blocks_per_pool` | `10` | 每次攒多少个 block 后写入。影响随机化程度和单次内存 |

```{note}
`store_type` 表示你希望写入数据库的表达尺度。代码会抽样判断源 `X` 是 count 还是 log，并在写入前自动做 `log1p` 或 `expm1` 转换。
```

### 4. 导入后检查

`load_h5ad()` 导入完成时会自动打印摘要信息，无需手动验证：

```text
✔ 全部数据成功导入 DuckDB（顺序导入，含 obsm / varm）
  - cells: 3,000
  - nnz:   2,500,000
  - store_type: count
```

如果想确认数据库状态，可以随时调用 `describe()`：

```python
print(atlas.describe())
```

输出示例：

```text
file_name    : ./data/pbmc_demo.sasql
tables      : 6
table names : obs, var, X_HyS_indptr, X_HyS_data, obsm_X_pca, obsm_X_umap
n_cells     : 3,000
n_genes     : 32,738
```

导入后数据库包含这些表：

| 表名 | 说明 |
|---|---|
| `obs` | 细胞元数据，含 `atlas_cell_id`、`atlas_cell_name` |
| `var` | 基因元数据，含 `atlas_gene_id`、`atlas_gene_name` |
| `X_HyS_indptr` | CSR 行指针 |
| `X_HyS_data` | 表达矩阵非零值（长表），字段含 `data` |
| `obsm_{key}` | 细胞级降维结果 |
| `varm_{key}` | 基因级降维结果 |

### 5. 清理重复基因名

```python
atlas.gene_names_duplicated()
```

### 6. 其他数据格式

非 h5ad 格式（`.loom`、`.mtx`、`.csv`、`.xlsx` 等），使用 `load_multi_format()`：

```python
atlas.load_multi_format("input.loom")
```

该函数自动识别文件后缀，调用对应 Scanpy 读取函数，再写入 Atlas 数据库。适合可完整读入内存的小数据。

详见 {doc}`../../api/io` 中 `load_multi_format` 的文档。

### 7. 完成后关闭连接

```python
atlas.close()
```

## 下一步

导入完成后，继续阅读 {doc}`quality-control-preprocessing`，完成质控、过滤和标准化。

如果想重新连接已有数据库继续分析，参考 {doc}`../advanced/reconnect-database`。
