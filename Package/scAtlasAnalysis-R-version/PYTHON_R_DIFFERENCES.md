# Python vs R 版本 scAtlas 差异清单

本文档记录 Python 版本 (scatlaspy) 和 R 版本 (scAtlasAnalysis-R) 之间的差异，便于两个版本的数据互通和功能对齐。

---

## 1. 数据库表结构

### 1.1 表结构对比

| 表名 | Python (`input.py`) | R (`atlas.R`) | 差异 |
|------|---------------------|---------------|------|
| **obs** | `id INTEGER PRIMARY KEY, + obs字段` | `id INTEGER PRIMARY KEY, cell_id VARCHAR` | Python动态添加obs字段 |
| **var** | `id INTEGER PRIMARY KEY, + var字段` | `id INTEGER PRIMARY KEY, gene_id VARCHAR` | Python动态添加var字段 |
| **X_CSR_indptr** | `id INTEGER, cell_id VARCHAR, indptr BIGINT` | 同左 | 一致 |
| **X_CSR_data** | `id BIGINT, indices USMALLINT, data REAL` | `id BIGINT, cell_index BIGINT, indices INTEGER, data REAL` | **Python缺少cell_index** |
| **uns** | 无 | `id, key, value_type, value_string, value_real` | R有，Python无 |

### 1.2 X_CSR_data 详细对比

| 字段 | Python 类型 | R 类型 | 说明 |
|------|-------------|--------|------|
| id | BIGINT PRIMARY KEY | BIGINT PRIMARY KEY | 一致 |
| cell_index | **无** | BIGINT NOT NULL | Python缺少此列 |
| indices | USMALLINT | INTEGER NOT NULL | 类型不一致 |
| data | REAL | REAL NOT NULL | 一致 |

### 1.3 索引对比

| 索引 | Python | R |
|------|--------|---|
| idx_csr_cell | **无** | `CREATE INDEX idx_csr_cell ON X_CSR_data(cell_index)` |
| idx_csr_gene | **无** | `CREATE INDEX idx_csr_gene ON X_CSR_data(indices)` |

---

## 2. 预处理函数对比

### 2.1 filter_cells / filter_cells_CSR_ultrafast

| 属性 | Python | R |
|------|--------|---|
| 函数名 | `filter_cells_CSR_ultrafast` | `filter_cells` |
| 过滤列名 | `add_key="filter_cells_1"` | `col_name="filter_cells_1"` |
| 返回值 | 无 (in-place修改) | 新atlas对象 (view) |
| 并行设置 | `PRAGMA threads` | `SET threads` |

### 2.2 filter_genes / filter_genes_CSR

| 属性 | Python | R |
|------|--------|---|
| 函数名 | `filter_genes_CSR` | `filter_genes` |
| 过滤列名 | `add_key="filter_genes_1"` | `col_name="filter_genes_1"` |

### 2.3 normalize_total / normalize_total_scale_factor

| 属性 | Python | R |
|------|--------|---|
| 函数名 | `normalize_total_scale_factor` | `normalize_total` |
| 缩放因子列 | `add_key="scale_factor"` | `col_name="scale_factor"` |
| 归一化输出列 | `add_field="data_normalize"` | **无** |
| target_sum | `target_sum=10000` | `target_sum=10000` |

### 2.4 log1p

| 属性 | Python | R |
|------|--------|---|
| 函数名 | `log1p_chunked` | `log1p` |
| 输入字段 | `select_data="data"` | `field="data"` |
| 输出字段 | `add_field="log1p_factor"` | `col_name="log1p_factor"` |
| chunk_size | `100_000_000` | `100_000_000` |
| 默认对数 | `ln(1.0 + data)` | `ln(1.0 + field)` |
| 并行设置 | `PRAGMA threads` | `SET threads` |

### 2.5 scale

| 属性 | Python | R |
|------|--------|---|
| 函数名 | `scale_gene_chunked` | `scale` |
| 输出字段 | `add_field="X_scale"` | `col_name="X_scale"` |
| 默认裁剪 | `max_value=10.0` | `max_value=NULL` |
| gene分块大小 | `gene_chunk_size=512` | `gene_chunk_size=512` |
| HVG支持 | `use_hvg=False` | `use_hvg=FALSE` |
| HVG列名 | `hvg_key="highly_variable"` | `hvg_key="highly_variable"` |
| 并行设置 | `PRAGMA threads` | `SET threads` |

### 2.6 exp1

| 属性 | Python | R |
|------|--------|---|
| 函数名 | `exp1_chunked` | `exp1` |
| 输入字段 | `select_data="log1p_factor"` | `field="log1p_factor"` |
| 输出字段 | `add_field="exp1_factor"` | `col_name="exp1_factor"` |
| chunk_size | `100_000_000` | `100_000_000` |
| 默认公式 | `exp(data) - 1.0` | `exp(field) - 1.0` |
| 并行设置 | `PRAGMA threads` | `SET threads` |

### 2.7 highly_variable_genes

| 属性 | Python | R |
|------|--------|---|
| 函数名 | `highly_variable_genes` | `highly_variable_genes` |
| 输出列 | `add_key="highly_variable_genes"` | `col_name="highly_variable_genes"` |
| flavor | `flavor="var"|"cv"` | `flavor="var"|"cv"` |
| n_top_genes | `n_top_genes=NULL` | `n_top=NULL` |

### 2.8 calculate_qc_metrics

| 属性 | Python | R |
|------|--------|---|
| 函数 | `calculate_qc_metrics` | `calculate_qc_metrics` |
| obs列 | `total_counts, n_genes_by_counts, total_counts_mt, pct_counts_mt` | 同左 |
| var列 | `total_counts, n_cells_by_counts` | 同左 |
| mt前缀 | `mt_prefix="MT-"` | `mt_prefix="MT-"` |

### 2.9 normalize_and_log1p

| 属性 | Python | R |
|------|--------|---|
| 函数名 | `normalize_and_log1p` | `normalize_and_log1p` |
| 输出字段 | `add_field="X_log1p"` | `col_name="log1p_factor"` **不一致** |
| scale_key | `scale_key="scale_factor"` | `scale_key="scale_factor"` |
| target_sum | `target_sum=10000` | `target_sum=10000` |

### 2.10 calculate_cell_total_counts

| 属性 | Python | R |
|------|--------|---|
| 函数名 | `calculate_cell_total_counts` | `calculate_cell_total_counts` |
| 输出列 | `add_key="cell_total_counts"` | `col_name="total_counts"` **不一致** |

### 2.11 calculate_gene_total_counts

| 属性 | Python | R |
|------|--------|---|
| 函数名 | `calculate_gene_total_counts` | `calculate_gene_total_counts` |
| 总数列 | `add_key1="gene_total_counts"` | `col_total="total_counts"` **不一致** |
| 均值列 | `add_key2="gene_means_counts"` | `col_mean="mean_counts"` **不一致** |

---

## 3. 列名对照表

### 3.1 obs表（细胞元数据）

| 功能 | Python 列名 | R 列名 | 状态 |
|------|-------------|--------|------|
| 细胞过滤标记 | `filter_cells_1` | `filter_cells_1` | 一致 |
| 归一化缩放因子 | `scale_factor` | `scale_factor` | 一致 |
| 细胞总表达量(QC) | `total_counts` | `total_counts` | 一致 |
| 细胞总表达量(Calculate) | `cell_total_counts` | `total_counts` | **不一致** |
| 表达基因数 | `n_genes_by_counts` | `n_genes_by_counts` | 一致 |
| 线粒体表达量 | `total_counts_mt` | `total_counts_mt` | 一致 |
| 线粒体占比 | `pct_counts_mt` | `pct_counts_mt` | 一致 |

### 3.2 var表（基因元数据）

| 功能 | Python 列名 | R 列名 | 状态 |
|------|-------------|--------|------|
| 基因过滤标记 | `filter_genes_1` | `filter_genes_1` | 一致 |
| 高变基因标记 | `highly_variable_genes` | `highly_variable_genes` | 一致 |
| 线粒体基因标记 | `mt` | `mt` | 一致 |
| 基因总表达量(QC) | `total_counts` | `total_counts` | 一致 |
| 基因总表达量(Calculate) | `gene_total_counts` | `total_counts` | **不一致** |
| 基因平均表达量(Calculate) | `gene_means_counts` | `mean_counts` | **不一致** |
| 表达细胞数 | `n_cells_by_counts` | `n_cells_by_counts` | 一致 |

### 3.3 X_CSR_data表（稀疏矩阵数据）

| 功能 | Python 列名 | R 列名 | 状态 |
|------|-------------|--------|------|
| 原始表达值 | `data` | `data` | 一致 |
| 归一化后数据 | `data_normalize` | **无** | - |
| log变换 | `log1p_factor` | `log1p_factor` | 一致 |
| 归一化+log | `X_log1p` | `log1p_factor` | **不一致** |
| scale标准化 | `X_scale` | `X_scale` | 一致 |
| exp逆变换 | `exp1_factor` | `exp1_factor` | 一致 |

---

## 4. 函数名称差异（命名不同但功能相同）

| 功能 | Python 函数名 | R 函数名 |
|------|---------------|----------|
| 细胞过滤 | `filter_cells_CSR_ultrafast` | `filter_cells` |
| 基因过滤 | `filter_genes_CSR` | `filter_genes` |
| 归一化(仅scale) | `normalize_total_scale_factor` | `normalize_total` |
| log变换 | `log1p_chunked` | `log1p` |
| exp变换 | `exp1_chunked` | `exp1` |
| scale标准化 | `scale_gene_chunked` | `scale` |
| 高变基因 | `highly_variable_genes` | `highly_variable_genes` |
| 质控指标 | `calculate_qc_metrics` | `calculate_qc_metrics` |
| 归一化+log组合 | `normalize_and_log1p` | `normalize_and_log1p` |
| 细胞总计数 | `calculate_cell_total_counts` | `calculate_cell_total_counts` |
| 基因总计数 | `calculate_gene_total_counts` | `calculate_gene_total_counts` |

---

## 5. 各自独有函数

### 5.1 Python 独有函数（R版本无对应）

- `add_X_CSR_chunk_append`
- `add_varm_from_h5ad`
- `add_X_CSR_from_10x_h5`
- `load_AnnData_from_h5ad`
- `initialize_csr_tables`
- `inspect_h5ad_structure`

### 5.2 R 独有函数（Python版本无对应）

- **无**（所有R函数在Python都有对应）

---

## 6. 并行设置差异

| 数据库 | Python 语法 | R 语法 |
|--------|-------------|--------|
| DuckDB | `PRAGMA threads=N` | `SET threads = N` |

> 注：DuckDB 两种语法都支持，但 `SET threads` 是标准SQL语法。

---

## 7. 数据迁移注意事项

### 7.1 Python -> R

1. Python的 `X_CSR_data` 表缺少 `cell_index` 列
2. 需要补充：`ALTER TABLE X_CSR_data ADD COLUMN cell_index BIGINT`
3. 需要更新cell_index值（根据indptr计算）

### 7.2 R -> Python

1. R的 `uns` 表在Python中不存在
2. 需要创建兼容的元数据存储方案

---

## 8. 改进建议

### 8.1 R版本已完成的改进

- [x] log1p: 添加DuckDB并行设置 (`SET threads`)
- [x] log1p: 添加字段存在性检查
- [x] log1p: 添加NULL值过滤
- [x] log1p: chunk_size参数化，默认1亿
- [x] scale: 添加DuckDB并行设置
- [x] scale: 添加gene分块处理
- [x] scale: 添加HVG支持
- [x] scale: 使用LEAST/GREATEST进行裁剪
- [x] exp1: 添加DuckDB并行设置
- [x] exp1: 每个chunk输出进度
- [x] 列名: filter_cells -> filter_cells_1
- [x] 列名: filter_genes -> filter_genes_1
- [x] 列名: highly_variable -> highly_variable_genes
- [x] log1p: col_name X_log1p -> log1p_factor
- [x] exp1: col_name X_exp1 -> exp1_factor, field X_log1p -> log1p_factor
- [x] atlas_create_indexes: 移除X表索引（X表不存在）
- [x] atlas_maintain_tables: 移除ANALYZE X

### 8.2 建议后续改进

| 优先级 | 项目 | 说明 |
|--------|------|------|
| 高 | Python列名对齐 | `normalize_and_log1p`: `X_log1p` -> `log1p_factor` |
| 高 | Python列名对齐 | `calculate_cell_total_counts`: `cell_total_counts` -> `total_counts` |
| 高 | Python列名对齐 | `calculate_gene_total_counts`: `gene_total_counts`/`gene_means_counts` -> `total_counts`/`mean_counts` |
| 中 | Python添加cell_index | X_CSR_data表添加cell_index列 |
| 低 | R添加uns表 | 可选，Python可考虑添加 |

---

## 9. 返回值风格差异

| 操作 | Python 风格 | R 风格 |
|------|-------------|--------|
| filter_cells | 原地修改，返回NULL | 返回新atlas视图对象 |
| filter_genes | 原地修改，返回NULL | 返回新atlas视图对象 |
| normalize_total | 原地修改，返回NULL | 原地修改，返回invisible(self) |
| log1p | 原地修改，返回NULL | 原地修改，返回invisible(self) |
| exp1 | 原地修改，返回NULL | 原地修改，返回invisible(self) |
| scale | 原地修改，返回NULL | 原地修改，返回invisible(self) |
| 索引子集 a[1:100] | 不支持 | 返回新atlas视图对象 |

---

*文档版本: 2.0 (2026-01-09)*
