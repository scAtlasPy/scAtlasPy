# scAtlas (R版本)

基于 DuckDB 的单细胞图谱数据管理工具。支持对大规模单细胞数据进行高效的存储、查询和分析。

## 目录

1. [快速开始](#快速开始)
2. [安装](#安装)
3. [核心概念](#核心概念)
4. [API参考](#api参考)
   - [数据库](#数据库)
   - [数据访问](#数据访问)
   - [筛选](#筛选)
   - [转换](#转换)
   - [质控指标](#质控指标)
   - [输入输出](#输入输出)
   - [分批查询](#分批查询)
   - [数据库优化](#数据库优化)
   - [上下文管理器](#上下文管理器)
5. [示例工作流](#示例工作流)
6. [数据库结构](#数据库结构)
7. [Python版本兼容性](#python版本兼容性)
8. [注意事项](#注意事项)

---

## 快速开始

```r
library(scAtlas)

# 1. 创建数据库并加载数据
atlas <- atlas("pbmc", mode = "r+")
atlas <- load_data(atlas, "pbmc.h5")

# 2. 查看数据摘要
atlas                                    # 查看摘要
obs(atlas)                               # 细胞元数据
var(atlas)                               # 基因元数据
X(atlas)                                 # 表达矩阵

# 3. 预处理
atlas <- filter_cells(atlas, min_counts = 200)
atlas <- normalize_and_log1p(atlas, target_sum = 10000)
atlas <- highly_variable_genes(atlas, n_top = 2000)

# 4. 关闭连接（重要！）
atlas_close(atlas)
```

---

## 安装

```r
# 安装依赖包
install.packages(c("DBI", "duckdb", "Matrix", "jsonlite", "parallel"))

# 安装scAtlas包
cd scAtlasAnalysis-R-version
R CMD INSTALL .
```

---

## 核心概念

### Atlas对象

atlas对象是一个S3类，其背后是DuckDB数据库文件。数据库采用CSR稀疏矩阵格式存储表达数据，支持处理上亿细胞的数据。

```r
atlas(name, path = ".", mode = c("r", "r+"))
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `name` | 字符 | - | 数据库名称（不带扩展名） |
| `path` | 字符 | `.` | 数据库目录路径 |
| `mode` | 字符 | - | 访问模式：`"r"`只读或`"r+"`读写 |

**返回：** 包含以下字段的列表对象（S3类`atlas`）：

| 字段 | 类型 | 说明 |
|------|------|------|
| `name` | 字符 | 数据库名称 |
| `path` | 字符 | 数据库路径 |
| `mode` | 字符 | 访问模式 |
| `con` | DBIConnection | DuckDB数据库连接 |
| `obs_cell_id` | 字符向量或NULL | 筛选后的细胞ID，NULL表示全部 |
| `var_gene_id` | 整数向量或NULL | 筛选后的基因ID，NULL表示全部 |
| `is_view` | 逻辑 | 是否为视图 |

### 操作模式

函数分为两种操作模式：

| 类型 | 函数 | 行为 | 返回值 |
|------|------|------|--------|
| **不可变（Immutable）** | `filter_cells()`, `filter_genes()`, `a[1:100]` | 不修改原对象 | 新的atlas对象 |
| **原地（In-place）** | `normalize_total()`, `log1p()`, `scale()`, `exp1()`, `highly_variable_genes()` | 修改数据库 | invisible(self) |

---

## API参考

### 数据库

```r
# 创建或打开数据库
atlas(name, path = ".", mode = c("r", "r+"))

# 关闭连接（重要！完成后必须调用）
atlas_close(a)

# 检查数据库是否存在
atlas_exists(name, path = ".")

# 删除数据库
delete_atlas(name, path = ".")

# 检查atlas对象是否有效
atlas_is_valid(a)
```

### 数据访问

```r
# 获取细胞元数据（返回data.frame）
obs(a)                           # 所有列
obs(a, columns = "cell_id")      # 特定列

# 获取基因元数据（返回data.frame）
var(a)
var(a, columns = "gene_id")

# 获取表达矩阵（返回dgCMatrix稀疏矩阵或矩阵）
X(a)                             # 稀疏矩阵（默认）
X(a, sparse = FALSE)             # 密集矩阵
X(a, genes = c("gene_A", "gene_B"))  # 特定基因

# 执行SQL查询
query(a, "SELECT COUNT(*) FROM obs")
```

### 筛选

```r
# 按表达量筛选细胞
# 在obs表添加filter_cells_1列（BOOLEAN类型）
filter_cells(a, min_counts = 100, min_genes = 50,
             max_counts = NULL, max_genes = NULL,
             col_name = "filter_cells_1")

# 按表达量筛选基因
# 在var表添加filter_genes_1列（BOOLEAN类型）
filter_genes(a, min_counts = 10, min_cells = 3,
             max_counts = NULL, max_cells = NULL,
             col_name = "filter_genes_1")

# 按索引子集（返回新视图）
a[1:100]          # 前100个细胞（按id）
a["cell_123"]     # 特定细胞（按cell_id）
a[c("cell_1", "cell_2", "cell_3")]  # 多个细胞
```

**筛选参数说明：**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `min_counts` | 细胞最小总表达量 | NULL |
| `min_genes` | 细胞最小表达基因数 | NULL |
| `max_counts` | 细胞最大总表达量 | NULL |
| `max_genes` | 细胞最大表达基因数 | NULL |
| `min_cells` | 基因最小表达细胞数 | NULL |
| `max_cells` | 基因最大表达细胞数 | NULL |
| `col_name` | 输出列名 | 参见各函数默认值 |

### 转换

#### 归一化

```r
# 计算每个细胞的缩放因子（添加到obs表的scale_factor列）
normalize_total(a, target_sum = 10000,
                col_name = "scale_factor",
                field = "data")
```

#### 组合归一化+Log

```r
# 组合操作：先normalize_total，再对结果log1p
# 输出到X_CSR_data表的log1p_factor列
normalize_and_log1p(a, target_sum = 10000,
                    scale_key = "scale_factor",
                    col_name = "log1p_factor",
                    field = "data",
                    base = NULL,
                    chunk_size = 100000000)
```

#### Log变换

```r
# Log1p变换：对X_CSR_data的data列进行log(1+x)变换
# 输出到X_CSR_data表的log1p_factor列
log1p(a)                      # 自然对数ln(1+x)
log1p(a, base = 2)            # 以2为底的对数log2(1+x)

# 参数：
# - base: 对数底，NULL表示自然对数
# - col_name: 输出列名，默认"log1p_factor"
# - field: 输入字段，默认"data"
# - chunk_size: 分块大小，默认100000000
```

#### 逆Log变换

```r
# Exp1变换：log1p的逆操作，计算exp(x)-1
# 输入log1p_factor列，输出exp1_factor列
exp1(a)
exp1(a, base = NULL)          # 自然指数exp(x)-1
exp1(a, base = 2)             # 2的幂：2^x - 1

# 参数：
# - base: 指数底，NULL表示自然指数
# - col_name: 输出列名，默认"exp1_factor"
# - field: 输入字段，默认"log1p_factor"
# - chunk_size: 分块大小，默认100000000
```

#### 高变基因

```r
# 识别高变基因（添加到var表的highly_variable_genes列）
highly_variable_genes(a, flavor = c("var", "cv"),
                      n_top = NULL,
                      col_name = "highly_variable_genes",
                      field = "data")

# 参数：
# - flavor: "var"按方差排序，"cv"按变异系数排序
# - n_top: 选择top N基因，NULL表示全部
# - col_name: 输出列名，默认"highly_variable_genes"
# - field: 输入字段，默认"data"
```

#### Z-Score标准化

```r
# Z-score标准化：按基因计算均值和标准差
# 输出到X_CSR_data表的X_scale列
scale(a)                      # 无裁剪
scale(a, max_value = 10)      # 裁剪到[-10, 10]

# 参数：
# - max_value: 裁剪阈值，NULL表示不裁剪
# - col_name: 输出列名，默认"X_scale"
# - field: 输入字段，默认"data"
# - gene_chunk_size: 基因分块大小，默认512
# - use_hvg: 是否只对高变基因标准化，默认FALSE
# - hvg_key: 高变基因列名，默认"highly_variable"
```

### 质控指标

```r
# 计算细胞和基因的质控指标
calculate_qc_metrics(a, mt_prefix = "MT-")

# 添加的列：
# obs表：total_counts, n_genes_by_counts, total_counts_mt, pct_counts_mt
# var表：total_counts, n_cells_by_counts
# var表：mt列（根据mt_prefix标记线粒体基因）

# 每个细胞的总数（添加到obs表的total_counts列）
calculate_cell_total_counts(a, col_name = "total_counts")

# 每个基因的总数和均值（添加到var表的两列）
calculate_gene_total_counts(a, col_name1 = "gene_total_counts",
                           col_name2 = "gene_means_counts")
```

### 输入输出(实际上这块还不太稳定)

```r
# 加载数据（自动检测格式）
load_data(a, "data.h5")        # H5, H5AD, 10X H5
load_data(a, "data.rds")       # RDS
load_data(a, "data.mtx")       # 10X MTX目录
load_data(a, "data.csv")       # CSV矩阵
load_data(a, "data.tsv")       # TSV矩阵

# 保存为H5AD
save_h5ad(a, "output.h5ad",
          write_obs = TRUE,
          write_var = TRUE,
          write_X = TRUE)
```

### 分批查询

```r
# 分批查询表达矩阵，支持三种模式

# 1. 顺序模式：按顺序获取所有数据
query_minibatch(a, batch_size = 2048, mode = "order")

# 2. 不放回随机：随机打乱后按批次获取，每批不重复
query_minibatch(a, batch_size = 2048, mode = "random_no_replace")

# 3. 有放回随机：无限循环，每次随机选择batch_size个细胞
query_minibatch(a, batch_size = 2048, mode = "random_replace",
                callback = function(i, adata) {
                  # 处理每个批次
                  # 抛出错误可停止循环
                })

# 带回调函数
batches <- query_minibatch(a, batch_size = 2048,
                          callback = function(i, adata) {
                            message(sprintf("批次 %d: %d x %d",
                                           i, nrow(adata$X), ncol(adata$X)))
                          })
```

### 数据库优化

```r
# 设置性能参数
atlas_optimize_settings(threads = 4, memory_limit = "8GB")

# 创建索引
atlas_create_indexes(a)           # 创建所有推荐索引
atlas_create_indexes(a, phase = 1) # 只创建第一阶段索引

# 表维护
atlas_maintain_tables(a)          # 更新统计信息

# 一键优化（执行所有优化步骤）
atlas_optimize(a)
```

### 上下文管理器

```r
# 使用with()进行自动清理
results <- with(atlas("mydata", mode = "r"), {
  X()                              # 获取表达矩阵
  obs()                            # 获取细胞元数据
  query_minibatch(a, batch_size = 1000)  # 分批查询
})
# 离开with块时会自动调用atlas_close()
```

---

## 示例工作流

```r
library(scAtlas)

# ---------------------------------------------------------
# 1. 初始化
# ---------------------------------------------------------
atlas <- atlas("pbmc分析", mode = "r+")
atlas <- load_data(atlas, "pbmc数据.h5")
atlas

# ---------------------------------------------------------
# 2. 质控
# ---------------------------------------------------------
# 计算质控指标
atlas <- calculate_qc_metrics(atlas, mt_prefix = "MT-")

# 筛选低质量细胞（保留表达量>=200且基因数>=50的细胞）
atlas <- filter_cells(atlas, min_counts = 200, min_genes = 50)

# 筛选低表达基因（保留在>=3个细胞中表达的基因）
atlas <- filter_genes(atlas, min_cells = 3)

# ---------------------------------------------------------
# 3. 归一化
# ---------------------------------------------------------
# 归一化并log转换（目标总和10000）
atlas <- normalize_and_log1p(atlas, target_sum = 10000)

# ---------------------------------------------------------
# 4. 特征选择
# ---------------------------------------------------------
# 识别top 2000高变基因
atlas <- highly_variable_genes(atlas, n_top = 2000)

# 缩放数据（Z-score，裁剪到[-10, 10]）
atlas <- scale(atlas, max_value = 10)

# ---------------------------------------------------------
# 5. 获取结果
# ---------------------------------------------------------
# 获取处理后的数据
X处理后 <- X(atlas)
obs数据 <- obs(atlas)
var数据 <- var(atlas)

# ---------------------------------------------------------
# 6. 清理
# ---------------------------------------------------------
atlas_close(atlas)
```

---

## 数据库结构

数据库采用DuckDB存储，文件扩展名为`.sasql`。内部包含以下表：

```
<名称>.sasql/
|
├── obs                        # 细胞元数据表
│   ├── id INTEGER PRIMARY KEY     # 主键，从0开始递增
│   ├── cell_id VARCHAR NOT NULL   # 细胞标识符
│   ├── scale_factor REAL          # 归一化缩放因子（由normalize_total添加）
│   ├── filter_cells_1 BOOLEAN     # 细胞过滤标记（由filter_cells添加）
│   ├── total_counts REAL          # 细胞总表达量（由calculate_cell_total_counts添加）
│   ├── n_genes_by_counts INTEGER  # 表达基因数（由calculate_qc_metrics添加）
│   ├── total_counts_mt REAL       # 线粒体基因总表达（由calculate_qc_metrics添加）
│   └── pct_counts_mt REAL         # 线粒体基因占比（由calculate_qc_metrics添加）
|
├── var                        # 基因元数据表
│   ├── id INTEGER PRIMARY KEY     # 主键，从0开始递增
│   ├── gene_id VARCHAR NOT NULL   # 基因标识符
│   ├── filter_genes_1 BOOLEAN     # 基因过滤标记（由filter_genes添加）
│   ├── highly_variable_genes BOOLEAN  # 高变基因标记（由highly_variable_genes添加）
│   ├── total_counts REAL          # 基因总表达量（由calculate_gene_total_counts添加）
│   ├── n_cells_by_counts INTEGER  # 表达细胞数（由calculate_qc_metrics添加）
│   └── mt BOOLEAN                 # 线粒体基因标记（由calculate_qc_metrics添加）
|
├── X_CSR_indptr             # CSR稀疏矩阵行指针表
│   ├── id INTEGER PRIMARY KEY     # 主键
│   ├── cell_id VARCHAR NOT NULL   # 对应细胞ID
│   └── indptr BIGINT NOT NULL     # 行指针（CSR格式）
|
├── X_CSR_data               # CSR稀疏矩阵数据表（核心存储）
│   ├── id BIGINT PRIMARY KEY      # 主键，全局唯一
│   ├── cell_index BIGINT NOT NULL # 细胞索引（对应obs.id）
│   ├── indices INTEGER NOT NULL   # 基因索引（对应var.id）
│   ├── data REAL NOT NULL         # 表达值
│   ├── log1p_factor REAL          # log变换结果（由log1p添加）
│   ├── exp1_factor REAL           # 逆log变换结果（由exp1添加）
│   └── X_scale REAL               # Z-score标准化结果（由scale添加）
|
└── uns                      # 非结构化元数据表（键值对）
    ├── id INTEGER PRIMARY KEY
    ├── key VARCHAR UNIQUE NOT NULL
    ├── value_type VARCHAR NOT NULL
    ├── value_string TEXT
    └── value_real DOUBLE
```

### 索引

系统自动创建以下索引以优化查询性能：

| 索引名 | 表 | 字段 |
|--------|-----|------|
| idx_csr_cell | X_CSR_data | cell_index |
| idx_csr_gene | X_CSR_data | indices |

---

## Python版本兼容性

本R版本与Python版本（scatlaspy）设计为功能对等，数据互通。

### 列名对应关系

| 功能 | Python列名 | R列名 | 状态 |
|------|------------|-------|------|
| 细胞过滤 | `filter_cells_1` | `filter_cells_1` | 一致 |
| 基因过滤 | `filter_genes_1` | `filter_genes_1` | 一致 |
| 归一化缩放 | `scale_factor` | `scale_factor` | 一致 |
| Log变换 | `log1p_factor` | `log1p_factor` | 一致 |
| Exp变换 | `exp1_factor` | `exp1_factor` | 一致 |
| Z-score标准化 | `X_scale` | `X_scale` | 一致 |
| 高变基因 | `highly_variable_genes` | `highly_variable_genes` | 一致 |
| 基因总表达 | `gene_total_counts` | `gene_total_counts` | 一致 |

### 函数对应关系

| R函数 | Python函数 | 说明 |
|-------|-----------|------|
| `filter_cells` | `filter_cells_CSR_ultrafast` | 功能相同，参数名不同 |
| `filter_genes` | `filter_genes_CSR` | 功能相同，参数名不同 |
| `normalize_total` | `normalize_total_scale_factor` | 功能相同 |
| `log1p` | `log1p_chunked` | 功能相同 |
| `exp1` | `exp1_chunked` | 功能相同 |
| `scale` | `scale_gene_chunked` | 功能相同 |
| `highly_variable_genes` | `highly_variable_genes` | 功能相同 |
| `calculate_qc_metrics` | `calculate_qc_metrics` | 功能相同 |
| `calculate_gene_total_counts` | `calculate_gene_total_counts` | 功能相同 |

### 主要差异

| 方面 | R版本 | Python版本 |
|------|-------|-----------|
| 返回值风格 | filter函数返回新atlas视图 | 所有操作原地修改 |
| 并行设置语法 | `SET threads = N` | `PRAGMA threads=N` |
| 数据库表结构 | X_CSR_data含cell_index | X_CSR_data不含cell_index |
| 临时表前缀 | `_scale_stats`等 | `keep_cells`等 |

---

## 注意事项

1. **始终关闭连接**：使用`atlas_close()`完成所有操作后关闭连接，否则数据库文件可能被锁定，无法被其他进程访问。

2. **load_data后重新赋值**：`load_data()`函数返回带有刷新细胞/基因ID的新对象，必须重新赋值：
   ```r
   a <- load_data(a, "data.h5")  # 正确：ID会刷新
   load_data(a, "data.h5")       # 错误：ID不会刷新
   ```

3. **视图是轻量的**：`filter_cells()`、`filter_genes()`和索引子集操作创建的是视图，不是数据副本。它们共享同一个底层数据库，内存占用极低。

4. **SQL访问**：使用`query()`进行自定义SQL查询。数据库使用DuckDB，支持完整SQL语法，包括聚合、连接、子查询等。

5. **内存管理**：包使用DuckDB的外核查询引擎，可以处理大于内存的数据。通过`atlas_optimize_settings()`可调整内存限制：
   ```r
   atlas_optimize_settings(memory_limit = "16GB")
   ```

6. **大数据分块处理**：对于超大规模数据（亿级细胞），`log1p()`、`exp1()`等转换操作会自动分块处理，避免内存溢出。默认块大小为1亿条记录。

---

## 支持的输入格式

| 格式 | 扩展名 | 说明 |
|------|--------|------|
| 10X H5 | `.h5` | 10X Genomics HDF5格式，支持Cell Ranger输出 |
| AnnData | `.h5ad` | H5AD格式，scanpy标准格式 |
| RDS | `.rds` | R序列化对象，需包含obs、var和X |
| 10X MTX | `.mtx` | Matrix Market格式，需配合genes.tsv和barcodes.tsv |
| CSV | `.csv` | 逗号分隔的表达矩阵 |
| TSV | `.tsv` | 制表符分隔的表达矩阵 |
| Loom | `.loom` | Loom格式（需要loomR包） |

---

## 技术细节

### DuckDB并行设置

R版本使用DuckDB的`SET threads`语法启用并行查询：

```r
# 在函数内部自动设置
DBI::dbExecute(con, "SET threads = 8")
```

### 稀疏矩阵存储

采用CSR（Compressed Sparse Row）格式存储表达矩阵：

- **X_CSR_indptr**：存储每行的起始位置
- **X_CSR_data**：存储非零值的(行索引, 列索引, 值)三元组

这种格式对于单细胞数据（高度稀疏）非常高效，存储空间约为密集矩阵的1%-10%。

### 类型安全

| 字段 | DuckDB类型 | R类型 |
|------|-----------|-------|
| id | BIGINT | integer64 |
| cell_id | VARCHAR | character |
| gene_id | VARCHAR | character |
| data/indices | REAL/INTEGER | double/integer |
| flags | BOOLEAN | logical |
