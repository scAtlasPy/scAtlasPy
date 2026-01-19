# scAtlas Python vs R 版本详细对比文档

> 文档版本: 1.0
> 整理日期: 2026-01-12
> 项目地址: `{项目根目录}`

---

## 目录

1. [概述](#一概述)
2. [数据库结构对比](#二数据库结构对比)
3. [核心类实现对比](#三核心类实现对比)
4. [预处理函数对比](#四预处理函数对比)
5. [IO模块对比](#五io模块对比)
6. [函数返回值风格对比](#六函数返回值风格对比)
7. [数据格式转换对比](#七数据格式转换对比)
8. [列名对照表](#八列名对照表)
9. [SQL语法差异](#九sql语法差异)
10. [开发进度追踪](#十开发进度追踪)
11. [完整源码对照](#十一完整源码对照)

---

## 一、概述

### 1.1 版本对应关系

| 组件 | Python 版本 | R 版本 |
|------|-------------|--------|
| **包名** | `scatlaspy` | `scAtlas` |
| **核心类** | `Atlas` | `atlas` (S3) |
| **预处理模块** | `_quality_control.py`, `_transformation.py` | `preprocessing.R` |
| **IO模块** | `input.py`, `output.py` | `io.R` |
| **数据库文件** | `.sasql` | `.sasql` |

### 1.2 核心差异一览

| 方面 | Python | R |
|------|--------|---|
| **对象系统** | 类 (class) | S3 对象 (list) |
| **返回风格** | 原地修改 (in-place) | 不可变 (immutable) |
| **索引子集** | 不支持 | `atlas[1:100]` |
| **数据库连接** | `atlas.connection` | `atlas$con` |
| **并行设置** | `PRAGMA threads=N` | `SET threads = N` |
| **主键类型** | 无 cell_index | 有 cell_index 列 |

---

## 二、数据库结构对比

### 2.1 表结构总览

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         数据库表结构对比                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Python 版本 (scatlaspy)              R 版本 (scAtlas)                 │
│  ────────────────────────────────     ──────────────────────────────   │
│                                                                         │
│  obs                         vs      obs                                │
│  ├─ id (INTEGER PRIMARY KEY)          ├─ id (INTEGER PRIMARY KEY)      │
│  ├─ cell_id (VARCHAR)                 ├─ cell_id (VARCHAR)             │
│  └─ [动态添加列]                       └─ [动态添加列]                   │
│                                                                         │
│  var                         vs      var                                │
│  ├─ id (INTEGER PRIMARY KEY)          ├─ id (INTEGER PRIMARY KEY)      │
│  ├─ gene_id (VARCHAR)                 ├─ gene_id (VARCHAR)             │
│  └─ [动态添加列]                       └─ [动态添加列]                   │
│                                                                         │
│  X_CSR_indptr               vs      X_CSR_indptr                        │
│  ├─ id (BIGINT PRIMARY KEY)           ├─ id (BIGINT PRIMARY KEY)       │
│  ├─ cell_id (VARCHAR)                 ├─ cell_id (VARCHAR)             │
│  └─ indptr (BIGINT)                   └─ indptr (BIGINT)               │
│                                                                         │
│  X_CSR_data                 vs      X_CSR_data                          │
│  ├─ id (BIGINT PRIMARY KEY)           ├─ id (BIGINT PRIMARY KEY)       │
│  ├─ indices (USMALLINT)               ├─ cell_index (BIGINT) ⬅️ 差异   │
│  ├─ data (REAL)                       ├─ indices (INTEGER) ⬅️ 类型差异 │
│  └─ [动态添加列]                       └─ data (REAL)                   │
│                                         └─ [动态添加列]                  │
│                                                                         │
│  无 uns 表                  vs      uns                                 │
│                                         ├─ id (INTEGER PRIMARY KEY)     │
│                                         ├─ key (VARCHAR)                │
│                                         ├─ value_type (VARCHAR)         │
│                                         ├─ value_string (TEXT)          │
│                                         └─ value_real (DOUBLE)          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 X_CSR_data 表详细对比

**Python 版本:**

```sql
-- 文件: Package/python-version-scAtlasAnalysis/scatlaspy/io/input.py

CREATE TABLE X_CSR_data (
    id BIGINT PRIMARY KEY,
    indices USMALLINT NOT NULL,
    data REAL NOT NULL
    -- 注意：Python 版本没有 cell_index 列！
)
```

**R 版本:**

```r
# 文件: Package/scAtlasAnalysis-R-version/R/atlas.R (第122-128行)

DBI::dbExecute(con, "CREATE TABLE X_CSR_data (
    id BIGINT PRIMARY KEY,
    cell_index BIGINT NOT NULL,
    indices INTEGER NOT NULL,
    data REAL NOT NULL
)")
```

**差异分析:**

| 字段 | Python 类型 | R 类型 | 影响 |
|------|-------------|--------|------|
| `id` | BIGINT PRIMARY KEY | BIGINT PRIMARY KEY | 一致 |
| `cell_index` | **无** | BIGINT NOT NULL | R 可直接按细胞查询 |
| `indices` | USMALLINT | INTEGER | R 支持更多基因 |
| `data` | REAL | REAL NOT NULL | 一致 |

### 2.3 索引对比

**Python 版本:**

```python
# 无自动创建的索引
```

**R 版本:**

```r
# 文件: Package/scAtlasAnalysis-R-version/R/atlas.R (第131-132行)

DBI::dbExecute(con, "CREATE INDEX idx_csr_cell ON X_CSR_data(cell_index)")
DBI::dbExecute(con, "CREATE INDEX idx_csr_gene ON X_CSR_data(indices)")
```

**索引说明:**

| 索引 | R 版本 | Python 版本 | 用途 |
|------|--------|-------------|------|
| `idx_csr_cell` | ✅ | ❌ | 按细胞索引快速查询 |
| `idx_csr_gene` | ✅ | ❌ | 按基因索引快速查询 |

---

## 三、核心类实现对比

### 3.1 类定义对比

**Python 版本 - Atlas 类:**

Python 版本使用私有属性（双下划线前缀），实际实现见源文件 `scatlaspy/data/_atlas.py`。

**R 版本 - S3 对象:**

```r
# 文件: Package/scAtlasAnalysis-R-version/R/atlas.R (第44-86行)

atlas <- function(name, path = ".", mode = c("r", "r+")) {
  mode <- match.arg(mode)

  db_file <- file.path(path, paste0(name, ".sasql"))

  # 创建或连接数据库
  if (!file.exists(db_file)) {
    if (mode == "r") {
      stop(sprintf("数据库文件不存在: %s", db_file))
    }
    con <- .create_db(db_file)
  } else {
    con <- .connect_db(db_file, read_only = (mode == "r"))
  }

  # 加载细胞/基因 ID
  obs_r <- tryCatch(DBI::dbGetQuery(con, "SELECT cell_id FROM obs"),
                    error = function(e) data.frame(cell_id = character(0)))
  var_r <- tryCatch(DBI::dbGetQuery(con, "SELECT gene_id FROM var"),
                    error = function(e) data.frame(gene_id = character(0)))

  # 构建 atlas 对象（S3 类）
  obj <- list(
    name = name,
    path = path,
    mode = mode,
    con = con,
    obs_cell_id = if (nrow(obs_r) > 0) obs_r$cell_id else NULL,
    var_gene_id = if (nrow(var_r) > 0) var_r$gene_id else NULL,
    is_view = FALSE
  )
  class(obj) <- "atlas"
  obj
}
```

### 3.2 数据访问方法对比

**Python - obs 方法:**

Python 版本当前无直接的 `obs` / `X` 属性实现，需通过 `query()` 方法访问数据。

**R - obs() 函数:**

```r
# 文件: Package/scAtlasAnalysis-R-version/R/atlas.R

#' @export
obs.atlas <- function(x, columns = NULL, ...) {
  cell_ids <- x$obs_cell_id

  if (is.null(cell_ids)) {
    if (is.null(columns)) {
      DBI::dbReadTable(x$con, "obs")
    } else {
      cols <- c("id", "cell_id", columns)
      cols <- cols[cols %in% DBI::dbListFields(x$con, "obs")]
      query <- paste0("SELECT ", paste(cols, collapse = ", "), " FROM obs")
      DBI::dbGetQuery(x$con, query)
    }
  } else {
    ids_str <- paste(cell_ids, collapse = "','")
    if (is.null(columns)) {
      query <- sprintf("SELECT * FROM obs WHERE cell_id IN ('%s')", ids_str)
    } else {
      cols <- c("id", "cell_id", columns)
      cols <- cols[cols %in% DBI::dbListFields(x$con, "obs")]
      query <- sprintf("SELECT %s FROM obs WHERE cell_id IN ('%s')",
                       paste(cols, collapse = ", "), ids_str)
    }
    DBI::dbGetQuery(x$con, query)
  }
}
```

### 3.3 表达式矩阵获取对比

**Python - X 方法:**

Python 版本当前无直接的 `X` 属性实现，可通过 `query_minibatch()` 或自定义查询获取数据。

**R - X() 函数:**

```r
# 文件: Package/scAtlasAnalysis-R-version/R/atlas.R

#' @export
X.atlas <- function(x, genes = NULL, sparse = TRUE, ...) {
  gene_ids <- x$var_gene_id
  cell_ids <- x$obs_cell_id

  # 获取 indptr
  if (is.null(cell_ids)) {
    indptr_df <- DBI::dbReadTable(x$con, "X_CSR_indptr")
  } else {
    ids_str <- paste(cell_ids, collapse = "','")
    indptr_df <- DBI::dbGetQuery(x$con,
      sprintf("SELECT * FROM X_CSR_indptr WHERE cell_id IN ('%s')", ids_str))
  }
  indptr <- indptr_df$indptr

  # 获取 data
  if (is.null(genes)) {
    data_query <- "SELECT cell_index, indices, data FROM X_CSR_data ORDER BY cell_index, indices"
  } else {
    gene_ids_str <- paste(genes - 1, collapse = ",")
    data_query <- sprintf(
      "SELECT cell_index, indices, data FROM X_CSR_data WHERE indices IN (%s) ORDER BY cell_index, indices",
      gene_ids_str
    )
  }

  data_df <- DBI::dbGetQuery(x$con, data_query)

  # 构建稀疏矩阵
  if (nrow(data_df) == 0) {
    return(Matrix::Matrix(0, nrow = length(indptr) - 1, ncol = length(gene_ids), sparse = TRUE))
  }

  i <- data_df$cell_index + 1
  j <- data_df$indices + 1
  x_vals <- data_df$data

  Matrix::sparseMatrix(i = i, j = j, x = x_vals,
                       dims = c(length(indptr) - 1, length(gene_ids)),
                       index1 = FALSE)
}
```

---

## 四、预处理函数对比

### 4.1 细胞过滤函数对比

**Python - filter_cells_CSR_ultrafast():**

```python
# 文件: Package/python-version-scAtlasAnalysis/scatlaspy/preprocessing/_quality_control.py (第22-104行)

def filter_cells_CSR_ultrafast(atlas: 'Atlas',
                               min_counts=None, min_genes=None,
                               max_counts=None, max_genes=None,
                               add_key="filter_cells_1"):
    """
    使用 DuckDB 原生并行过滤细胞

    Args:
        atlas: Atlas 对象
        min_counts: 最小总表达量
        min_genes: 最小表达基因数
        max_counts: 最大总表达量
        max_genes: 最大表达基因数
        add_key: 输出列名
    """
    from datetime import datetime
    start = datetime.now()

    conn = atlas.connect("r+")
    th = os.cpu_count()
    conn.execute(f"PRAGMA threads={th}")
    print(f"DuckDB threads = {th}")

    # 预先添加列
    conn.execute(f"""
        ALTER TABLE obs
        ADD COLUMN IF NOT EXISTS {add_key} BOOLEAN DEFAULT FALSE
    """)

    # 统计细胞数
    total_cells = conn.execute("SELECT COUNT(*) FROM obs").fetchone()[0]
    print(f"总细胞数 = {total_cells:,}")

    # 构建 SQL 过滤条件
    conds = []
    if min_counts is not None: conds.append(f"sum_expr >= {min_counts}")
    if max_counts is not None: conds.append(f"sum_expr <= {max_counts}")
    if min_genes  is not None: conds.append(f"nonzero_genes >= {min_genes}")
    if max_genes  is not None: conds.append(f"nonzero_genes <= {max_genes}")
    condition = " AND ".join(conds) if conds else "TRUE"

    # Step1：计算需要保留的 cell_index
    conn.execute("DROP TABLE IF EXISTS keep_cells")
    conn.execute(f"""
        CREATE TEMP TABLE keep_cells AS
        SELECT cell_index
        FROM (
            SELECT
                cell_index,
                SUM(data) AS sum_expr,
                COUNT(*) AS nonzero_genes
            FROM X_CSR_data
            GROUP BY cell_index
        ) WHERE {condition}
    """)

    # Step2：更新 obs 表
    conn.execute(f"UPDATE obs SET {add_key}=FALSE")
    conn.execute(f"""
        UPDATE obs SET {add_key}=TRUE
        WHERE id IN (SELECT cell_index FROM keep_cells)
    """)

    # Step3：统计结果
    keep_cells = conn.execute(f"SELECT COUNT(*) FROM keep_cells").fetchone()[0]
    removed = total_cells - keep_cells
    print(f"保留细胞 = {keep_cells:,}")
    print(f"过滤细胞 = {removed:,} ({removed/total_cells*100:.2f}%)")
```

**R - filter_cells.atlas():**

```r
# 文件: Package/scAtlasAnalysis-R-version/R/preprocessing.R (第46-115行)

#' @export
filter_cells.atlas <- function(a,
                               min_counts = NULL,
                               min_genes = NULL,
                               max_counts = NULL,
                               max_genes = NULL,
                               col_name = "filter_cells_1",
                               ...) {
  if (!inherits(a, "atlas")) stop("Argument must be an atlas object")
  if (is.null(a$con) || !DBI::dbIsValid(a$con)) stop("Database connection is closed")

  if (all(sapply(list(min_counts, min_genes, max_counts, max_genes), is.null))) {
    stop("At least one filter condition must be specified")
  }

  con <- a$con
  total <- DBI::dbGetQuery(con, "SELECT COUNT(*) AS n FROM obs")$n[1]
  message(sprintf("[filter_cells] Total cells: %s", format(total, big.mark = ",")))

  # 添加 filter 列
  DBI::dbExecute(con, sprintf("
    ALTER TABLE obs
    ADD COLUMN IF NOT EXISTS %s BOOLEAN DEFAULT FALSE
  ", col_name))

  # 构建过滤条件
  conds <- c()
  if (!is.null(min_counts)) conds <- c(conds, sprintf("sum_expr >= %d", min_counts))
  if (!is.null(max_counts)) conds <- c(conds, sprintf("sum_expr <= %d", max_counts))
  if (!is.null(min_genes)) conds <- c(conds, sprintf("nonzero >= %d", min_genes))
  if (!is.null(max_genes)) conds <- c(conds, sprintf("nonzero <= %d", max_genes))
  condition <- paste(conds, collapse = " AND ")

  # 查找符合条件的细胞
  DBI::dbExecute(con, "DROP TABLE IF EXISTS _filter_keep")
  DBI::dbExecute(con, sprintf("
    CREATE TEMP TABLE _filter_keep AS
    SELECT cell_index
    FROM (
      SELECT cell_index, SUM(data) AS sum_expr, COUNT(*) AS nonzero
      FROM X_CSR_data
      GROUP BY cell_index
    ) WHERE %s
  ", condition))

  # 更新 filter 列
  DBI::dbExecute(con, sprintf("UPDATE obs SET %s = FALSE", col_name))
  DBI::dbExecute(con, sprintf("
    UPDATE obs SET %s = TRUE
    WHERE id IN (SELECT cell_index FROM _filter_keep)
  ", col_name))

  # 获取过滤后的细胞 ID
  keep_ids <- DBI::dbGetQuery(con, sprintf(
    "SELECT cell_id FROM obs WHERE %s = TRUE", col_name
  ))$cell_id

  n_keep <- length(keep_ids)
  message(sprintf("[filter_cells] Kept: %s (%.1f%%)",
         format(n_keep, big.mark = ","), 100 * n_keep / total))

  # 清理并返回新视图
  DBI::dbExecute(con, "DROP TABLE IF EXISTS _filter_keep")
  .new_atlas_view(a, obs_cell_id = keep_ids)
}
```

### 4.2 归一化函数对比

**Python - normalize_total_scale_factor():**

```python
# 文件: Package/python-version-scAtlasAnalysis/scatlaspy/preprocessing/_transformation.py (第137-220行)

def normalize_total_scale_factor(
                            atlas: Atlas,
                            target_sum: Optional[float] = 10000,
                            add_key: str = "scale_factor",
                            select_data: str = "data"
                        ) -> None:
    """
    高性能 normalize_total（Scanpy 等价）：
    - 不修改 X_CSR_data
    - 只计算每个 cell 的 scale_factor
    """
    print("==== normalize_total (scale_factor only) ====")
    start = datetime.now()

    conn = atlas.connection

    # 1. 检查字段是否存在
    col_exists = conn.execute(f"""
        SELECT COUNT(*)
        FROM information_schema.columns
        WHERE table_name = 'X_CSR_data'
          AND column_name = '{select_data}'
    """).fetchone()[0]

    if col_exists == 0:
        raise ValueError(f"X_CSR_data 中不存在字段: {select_data}")

    # 2. 计算每个 cell 的 total
    conn.execute(f"""
        CREATE OR REPLACE TEMP TABLE _cell_sum AS
        SELECT
            cell_index,
            SUM({select_data}) AS total
        FROM X_CSR_data
        GROUP BY cell_index
    """)

    # 3. 确定 target_sum
    if target_sum is None:
        target_sum = conn.execute(
            "SELECT median(total) FROM _cell_sum"
        ).fetchone()[0]

    # 4. 写入 scale_factor 到 obs
    conn.execute(f"""
        ALTER TABLE obs
        ADD COLUMN IF NOT EXISTS {add_key} DOUBLE
    """)
    conn.execute(f"""
        UPDATE obs
        SET {add_key} = {float(target_sum)}
    """)

    print(f"target_sum = {target_sum}")
    print("normalize_total 完成")
```

**R - normalize_total():**

```r
# R 版本使用类似的逻辑，但参数名略有不同
normalize_total <- function(a, target_sum = 10000,
                            col_name = "scale_factor",
                            field = "data") {

  con <- a$con

  # 1. 计算每个细胞的总表达量
  DBI::dbExecute(con, "
    CREATE OR REPLACE TEMP TABLE _cell_sum AS
    SELECT cell_index, SUM(data) AS total
    FROM X_CSR_data
    GROUP BY cell_index
  ")

  # 2. 添加 scale_factor 列
  DBI::dbExecute(con, sprintf("
    ALTER TABLE obs
    ADD COLUMN IF NOT EXISTS %s DOUBLE
  ", col_name))

  # 3. 更新所有细胞的 scale_factor
  DBI::dbExecute(con, sprintf("
    UPDATE obs
    SET %s = %f
  ", col_name, as.numeric(target_sum)))

  invisible(a)
}
```

### 4.3 Log 变换函数对比

**Python - log1p_chunked():**

```python
def log1p_chunked(atlas: 'Atlas',
                  add_field: str = "log1p_factor",
                  select_data: str = "data",
                  chunk_size: int = 100_000_000) -> None:
    """
    分块执行 log1p 变换
    """
    conn = atlas.connection
    conn.execute(f"PRAGMA threads={os.cpu_count()}")

    # 获取总行数
    total_rows = conn.execute("SELECT COUNT(*) FROM X_CSR_data").fetchone()[0]
    n_chunks = math.ceil(total_rows / chunk_size)

    print(f"log1p: 处理 {total_rows} 行，分 {n_chunks} 块")

    for i in range(n_chunks):
        offset = i * chunk_size

        conn.execute(f"""
            UPDATE X_CSR_data SET {add_field} = LN(1.0 + COALESCE({select_data}, 0))
            FROM (SELECT id FROM X_CSR_data LIMIT {chunk_size} OFFSET {offset}) AS sub
            WHERE X_CSR_data.id = sub.id
        """)

        print(f"    块 {i+1}/{n_chunks} 完成")
```

**R - log1p():**

```r
log1p <- function(a, base = NULL, col_name = "log1p_factor",
                  field = "data", chunk_size = 100000000) {

  con <- a$con

  # 删除旧列并添加新列
  DBI::dbExecute(con, sprintf("ALTER TABLE X_CSR_data DROP COLUMN IF EXISTS %s", col_name))
  DBI::dbExecute(con, sprintf("ALTER TABLE X_CSR_data ADD COLUMN %s REAL", col_name))

  # 设置并行
  DBI::dbExecute(con, "SET threads = 4")

  # 分块处理
  total_rows <- DBI::dbGetQuery(con, "SELECT COUNT(*) as n FROM X_CSR_data")$n
  n_chunks <- ceiling(total_rows / chunk_size)

  cat(sprintf("  log1p: 处理 %d 行，分 %d 块\n", total_rows, n_chunks))

  for (i in seq_len(n_chunks)) {
    offset <- (i - 1) * chunk_size

    if (is.null(base)) {
      update_sql <- sprintf(
        "UPDATE X_CSR_data SET %s = LN(1.0 + COALESCE(%s, 0))
         FROM (SELECT id FROM X_CSR_data LIMIT %d OFFSET %d) AS sub
         WHERE X_CSR_data.id = sub.id",
        col_name, field, chunk_size, offset
      )
    } else {
      update_sql <- sprintf(
        "UPDATE X_CSR_data SET %s = LOG(%s, 1.0 + COALESCE(%s, 0))
         FROM (SELECT id FROM X_CSR_data LIMIT %d OFFSET %d) AS sub
         WHERE X_CSR_data.id = sub.id",
        col_name, base, field, chunk_size, offset
      )
    }

    DBI::dbExecute(con, update_sql)
    cat(sprintf("    块 %d/%d 完成\n", i, n_chunks))
  }

  invisible(a)
}
```

### 4.4 预处理函数对照表

| 功能 | Python 函数 | R 函数 | 实现差异 |
|------|-------------|--------|----------|
| 细胞过滤 | `filter_cells_CSR_ultrafast()` | `filter_cells()` | R 返回视图 |
| 基因过滤 | `filter_genes_CSR()` | `filter_genes()` | R 返回视图 |
| 归一化 | `normalize_total_scale_factor()` | `normalize_total()` | 一致 |
| 归一化+Log | `normalize_and_log1p()` | `normalize_and_log1p()` | 列名不同 |
| Log 变换 | `log1p_chunked()` | `log1p()` | 参数名不同 |
| Scale 标准化 | `scale_gene_chunked()` | `scale()` | 参数名不同 |
| 逆 Log | `exp1_chunked()` | `exp1()` | 一致 |
| 高变基因 | `highly_variable_genes()` | `highly_variable_genes()` | 一致 |
| 质控指标 | `calculate_qc_metrics()` | `calculate_qc_metrics()` | 一致 |

---

## 五、IO模块对比

### 5.1 数据导入函数对比

**Python - 加载数据:**

```python
# Python 版本使用 scanpy 的接口读取数据，然后导入到数据库
from scatlaspy.io.input import load_AnnData

adata = sc.read_h5ad("data.h5ad")
load_AnnData(adata, atlas)
```

**R - load_data():**

```r
# 文件: Package/scAtlasAnalysis-R-version/R/io.R (第82-132行)

#' Load data from file (auto-detect format)
#' @export
load_data <- function(a, file) {
  if (!inherits(a, "atlas")) stop("Argument must be an atlas object")
  if (a$mode != "r+") stop("Requires mode = 'r+'")
  if (is.null(a$con) || !DBI::dbIsValid(a$con)) stop("Connection closed")
  if (!file.exists(file)) stop(sprintf("File not found: %s", file))

  fmt <- .detect_format(file)
  cat(sprintf("[load_data] Loading: %s\n", basename(file)))
  set_progress("load_data", status = sprintf("Format: %s", fmt))

  switch(fmt,
    "h5" = .load_h5(a, file),
    "rds" = .load_rds(a, file),
    "csv" = .load_csv(a, file, ","),
    "tsv" = .load_csv(a, file, "\t"),
    "mtx" = .load_mtx(a, file),
    stop(sprintf("Unsupported format: %s", fmt))
  )

  DBI::dbExecute(a$con, "PRAGMA force_checkpoint")

  # 刷新 cell/gene IDs
  obs_r <- DBI::dbGetQuery(a$con, "SELECT cell_id FROM obs")
  var_r <- DBI::dbGetQuery(a$con, "SELECT gene_id FROM var")
  a$obs_cell_id <- if (nrow(obs_r) > 0) obs_r$cell_id else NULL
  a$var_gene_id <- if (nrow(var_r) > 0) var_r$gene_id else NULL

  invisible(a)
}
```

### 5.2 H5 导入详细对比

**Python - 10X H5 导入:**

Python 版本支持通过 `load_AnnData_chunk` 函数分批导入 H5/H5AD 文件，支持超大文件。

**R - 10X H5 导入:**

```r
# 文件: Package/scAtlasAnalysis-R-version/R/io.R (第211-323行)

.load_10x_h5 <- function(con, h5f, prefix = "matrix/") {
  set_progress("load_data", status = sprintf("10X format (%s*) detected", prefix))

  shape <- h5f[[paste0(prefix, "shape")]][]
  data <- h5f[[paste0(prefix, "data")]][]
  indices <- h5f[[paste0(prefix, "indices")]][]
  indptr <- h5f[[paste0(prefix, "indptr")]][]
  barcodes <- h5f[[paste0(prefix, "barcodes")]][]

  # 获取基因 ID
  gene_ids <- NULL
  gene_paths <- c(
    paste0(prefix, "features/name"),
    paste0(prefix, "gene_names"),
    paste0(prefix, "genes")
  )
  for (gp in gene_paths) {
    gene_ids <- tryCatch(h5f[[gp]][], error = function(e) NULL)
    if (!is.null(gene_ids)) {
      set_progress("load_data", status = sprintf("Found gene IDs at: %s", gp))
      break
    }
  }
  if (is.null(gene_ids)) {
    gene_ids <- paste0("gene_", seq_len(shape[1]))
  }

  n_cells <- length(barcodes)
  n_genes <- length(gene_ids)
  nnz <- length(data)

  set_progress("load_data", status = sprintf("cells=%s, genes=%s, nnz=%s",
    format(n_cells, big.mark = ","),
    format(n_genes, big.mark = ","),
    format(nnz, big.mark = ",")))

  # 创建 CSR 表
  DBI::dbExecute(con, "DROP TABLE IF EXISTS X_CSR_indptr")
  DBI::dbExecute(con, "DROP TABLE IF EXISTS X_CSR_data")

  DBI::dbExecute(con, "CREATE TABLE X_CSR_indptr (
    id BIGINT PRIMARY KEY,
    cell_id VARCHAR,
    indptr BIGINT
  )")

  DBI::dbExecute(con, "CREATE TABLE X_CSR_data (
    id BIGINT PRIMARY KEY,
    indices USMALLINT,
    data FLOAT,
    cell_index BIGINT
  )")

  # 写入 indptr
  indptr_df <- data.frame(
    id = seq_len(n_cells) - 1,
    cell_id = barcodes,
    indptr = as.integer(indptr)
  )
  DBI::dbWriteTable(con, "X_CSR_indptr", indptr_df, append = TRUE)

  # 写入 data（分批）
  batch_size <- 1000000
  n_batches <- ceiling(nnz / batch_size)

  for (batch_idx in seq_len(n_batches)) {
    start_idx <- (batch_idx - 1) * batch_size + 1
    end_idx <- min(batch_idx * batch_size, nnz)

    data_df <- data.frame(
      id = (start_idx:end_idx) - 1,
      indices = indices[start_idx:end_idx],
      data = as.numeric(data[start_idx:end_idx]),
      cell_index = rep(seq_len(n_cells) - 1, diff(indptr))[start_idx:end_idx]
    )

    DBI::dbAppendTable(con, "X_CSR_data", data_df)
  }
}
```

### 5.3 支持的文件格式

| 格式 | Python | R | 说明 |
|------|--------|---|------|
| H5 | ✅ | ✅ | 10X H5 格式 |
| H5AD | ✅ | ✅ | AnnData 格式 |
| RDS | ❌ | ✅ | R 序列化格式 |
| MTX | ✅ | ✅ | 10X MTX 格式 |
| CSV | ✅ | ✅ | 逗号分隔 |
| TSV | ✅ | ✅ | 制表符分隔 |

---

## 六、函数返回值风格对比

### 6.1 返回风格差异

**Python 风格：原地修改**

```python
# Python: 操作原地修改数据库，返回 None

atlas.filter_cells_CSR_ultrafast(min_counts=100)
# 返回: None (in-place 修改)

atlas.normalize_total_scale_factor(target_sum=10000)
# 返回: None

atlas.log1p_chunked()
# 返回: None
```

**R 风格：不可变/返回视图**

```r
# R: 操作返回新的 atlas 对象（视图）

atlas <- filter_cells(atlas, min_counts = 100)
# 返回: 新的 atlas 对象（视图，不复制数据）

atlas <- normalize_total(atlas, target_sum = 10000)
# 返回: invisible(self)（修改自身）

atlas <- log1p(atlas)
# 返回: invisible(self)
```

### 6.2 返回值类型对比表

| 函数 | Python 返回 | R 返回 |
|------|-------------|--------|
| `filter_cells()` | `None` | 新 `atlas` 对象 |
| `filter_genes()` | `None` | 新 `atlas` 对象 |
| `normalize_total()` | `None` | `invisible(self)` |
| `log1p()` | `None` | `invisible(self)` |
| `scale()` | `None` | `invisible(self)` |
| `exp1()` | `None` | `invisible(self)` |
| `highly_variable_genes()` | `None` | `invisible(self)` |

### 6.3 视图系统对比

**R 视图系统实现:**

```r
# 文件: Package/scAtlasAnalysis-R-version/R/atlas.R

.new_atlas_view <- function(a, obs_cell_id = NULL, var_gene_id = NULL) {
  """创建新的 atlas 视图"""
  obj <- a
  if (!is.null(obs_cell_id)) {
    obj$obs_cell_id <- obs_cell_id
  }
  if (!is.null(var_gene_id)) {
    obj$var_gene_id <- var_gene_id
  }
  obj$is_view <- TRUE
  class(obj) <- "atlas"
  obj
}
```

**Python 索引子集（R 特有）:**

```r
# 文件: Package/scAtlasAnalysis-R-version/R/atlas.R

#' @export
`[.atlas` <- function(x, i) {
  if (is.numeric(i)) {
    # 整数索引
    if (is.null(x$obs_cell_id)) {
      cell_ids <- DBI::dbGetQuery(x$con,
        sprintf("SELECT cell_id FROM obs LIMIT 1 OFFSET %d", i - 1))$cell_id
    } else {
      cell_ids <- x$obs_cell_id[i]
    }
  } else if (is.character(i)) {
    # 字符索引（cell_id）
    cell_ids <- i
  } else {
    stop("索引必须为整数或字符向量")
  }

  # 创建新视图
  x$obs_cell_id <- cell_ids
  x$is_view <- TRUE
  x
}
```

---

## 七、数据格式转换对比

### 7.1 CSR 稀疏矩阵结构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         CSR 稀疏矩阵格式                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  原始矩阵 (5x4, 稀疏度 60%):                                            │
│  ┌─────┬─────┬─────┬─────┐                                              │
│  │  0  │ 1.5 │  0  │ 2.1 │                                              │
│  ├─────┼─────┼─────┼─────┤                                              │
│  │ 0.3 │  0  │ 0.7 │  0  │                                              │
│  ├─────┼─────┼─────┼─────┤                                              │
│  │  0  │  0  │ 3.2 │  0  │                                              │
│  ├─────┼─────┼─────┼─────┤                                              │
│  │ 1.1 │ 0.4 │  0  │ 0.8 │                                              │
│  ├─────┼─────┼─────┼─────┤                                              │
│  │  0  │ 2.3 │  0  │  0  │                                              │
│  └─────┴─────┴─────┴─────┘                                              │
│                                                                         │
│  CSR 存储格式:                                                           │
│  ┌────────────────┬────────────────┬────────────────┐                   │
│  │ indptr (6个)   │ data (8个)     │ indices (8个)  │                   │
│  ├────────────────┼────────────────┼────────────────┤                   │
│  │ [0, 2, 4, 6,   │ [1.5, 2.1,     │ [1, 3, 0, 2,   │                   │
│  │  8, 10]        │  0.3, 0.7,     │  2, 0, 1, 3,   │                   │
│  │                │  3.2, 1.1,     │  1, 3]         │                   │
│  │                │  0.4, 0.8,     │                │                   │
│  │                │  2.3]          │                │                   │
│  └────────────────┴────────────────┴────────────────┘                   │
│                                                                         │
│  Python (scipy.sparse):           R (Matrix):                          │
│  ┌────────────────────────┐       ┌────────────────────────┐          │
│  │ indptr: ndarray        │       │ indptr: integer        │          │
│  │ indices: ndarray       │       │ i: integer             │          │
│  │ data: ndarray          │       │ x: numeric             │          │
│  │ shape: tuple           │       │ Dim: integer vector    │          │
│  └────────────────────────┘       └────────────────────────┘          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 7.2 数据类型对比

| 数据类型 | Python (DuckDB) | R (DuckDB) |
|----------|-----------------|------------|
| 细胞 ID | VARCHAR | VARCHAR |
| 基因 ID | VARCHAR | VARCHAR |
| 表达值 | REAL | REAL / FLOAT |
| 索引 | USMALLINT (Python) | INTEGER / BIGINT (R) |
| 行指针 | BIGINT | BIGINT |
| 布尔值 | BOOLEAN | BOOLEAN |

### 7.3 数据转换示例

**Python - 构建稀疏矩阵:**

```python
from scipy import sparse
import numpy as np

def build_sparse_matrix_from_db(conn, n_cells, n_genes):
    """从数据库构建 CSR 矩阵"""
    # 读取 indptr
    indptr = np.array(conn.execute(
        "SELECT indptr FROM X_CSR_indptr ORDER BY id"
    ).fetchone())

    # 读取 data 和 indices
    data_df = conn.execute(
        "SELECT indices, data FROM X_CSR_data ORDER BY id"
    ).df()

    # 构建 CSR 矩阵
    matrix = sparse.csr_matrix(
        (data_df['data'].values, data_df['indices'].values, indptr),
        shape=(n_cells, n_genes)
    )
    return matrix
```

**R - 构建稀疏矩阵:**

```r
build_sparse_matrix_from_db <- function(con, n_cells, n_genes) {
  # 读取 indptr
  indptr_df <- DBI::dbReadTable(con, "X_CSR_indptr")
  indptr <- indptr_df$indptr

  # 读取 data
  data_df <- DBI::dbGetQuery(con,
    "SELECT cell_index, indices, data FROM X_CSR_data ORDER BY cell_index, indices")

  # 构建 dgCMatrix
  if (nrow(data_df) == 0) {
    return(Matrix::Matrix(0, nrow = n_cells, ncol = n_genes, sparse = TRUE))
  }

  i <- data_df$cell_index + 1
  j <- data_df$indices + 1
  x_vals <- data_df$data

  Matrix::sparseMatrix(i = i, j = j, x = x_vals,
                       dims = c(n_cells, n_genes),
                       index1 = FALSE)
}
```

---

## 八、列名对照表

### 8.1 obs 表（细胞元数据）

| 功能 | Python 列名 | R 列名 | 状态 |
|------|-------------|--------|------|
| 细胞 ID | `cell_id` | `cell_id` | ✅ 一致 |
| 过滤标记 | `filter_cells_1` | `filter_cells_1` | ✅ 一致 |
| 归一化因子 | `scale_factor` | `scale_factor` | ✅ 一致 |
| 总表达量(QC) | `total_counts` | `total_counts` | ✅ 一致 |
| 总表达量(Calc) | `cell_total_counts` | `total_counts` | ❌ 不一致 |
| 表达基因数 | `n_genes_by_counts` | `n_genes_by_counts` | ✅ 一致 |
| 线粒体表达量 | `total_counts_mt` | `total_counts_mt` | ✅ 一致 |
| 线粒体占比 | `pct_counts_mt` | `pct_counts_mt` | ✅ 一致 |

### 8.2 var 表（基因元数据）

| 功能 | Python 列名 | R 列名 | 状态 |
|------|-------------|--------|------|
| 基因 ID | `gene_id` | `gene_id` | ✅ 一致 |
| 过滤标记 | `filter_genes_1` | `filter_genes_1` | ✅ 一致 |
| 高变基因标记 | `highly_variable_genes` | `highly_variable_genes` | ✅ 一致 |
| 线粒体基因标记 | `mt` | `mt` | ✅ 一致 |
| 基因总表达量(QC) | `total_counts` | `total_counts` | ✅ 一致 |
| 基因总表达量(Calc) | `gene_total_counts` | `total_counts` | ❌ 不一致 |
| 基因平均表达量 | `gene_means_counts` | `mean_counts` | ❌ 不一致 |
| 表达细胞数 | `n_cells_by_counts` | `n_cells_by_counts` | ✅ 一致 |

### 8.3 X_CSR_data 表（稀疏矩阵数据）

| 功能 | Python 列名 | R 列名 | 状态 |
|------|-------------|--------|------|
| 主键 | `id` | `id` | ✅ 一致 |
| 细胞索引 | **无** | `cell_index` | ❌ 缺失 |
| 基因索引 | `indices` | `indices` | ✅ 一致 |
| 原始表达值 | `data` | `data` | ✅ 一致 |
| 归一化数据 | `data_normalize` | **无** | - |
| Log 变换 | `log1p_factor` | `log1p_factor` | ✅ 一致 |
| 归一化+Log | `X_log1p` | `log1p_factor` | ❌ 不一致 |
| Scale 标准化 | `X_scale` | `X_scale` | ✅ 一致 |
| 逆 Log | `exp1_factor` | `exp1_factor` | ✅ 一致 |

---

## 九、SQL语法差异

### 9.1 并行查询设置

**Python:**

```python
# PRAGMA 语法
conn.execute(f"PRAGMA threads={os.cpu_count()}")
conn.execute("PRAGMA memory_limit='32GB'")
```

**R:**

```r
# SET 语法
DBI::dbExecute(con, "SET threads = 4")
DBI::dbExecute(con, "SET memory_limit = '32GB'")
```

### 9.2 临时表命名

**Python:**

```python
conn.execute("DROP TABLE IF EXISTS keep_cells")
conn.execute("CREATE TEMP TABLE keep_cells AS ...")
```

**R:**

```r
DBI::dbExecute(con, "DROP TABLE IF EXISTS _filter_keep")
DBI::dbExecute(con, "CREATE TEMP TABLE _filter_keep AS ...")
```

### 9.3 查询语法差异

**Python:**

```python
# 获取单个值
result = conn.execute("SELECT COUNT(*) FROM obs").fetchone()[0]

# 获取多行
df = conn.execute("SELECT * FROM obs").df()
```

**R:**

```r
# 获取单个值
result <- DBI::dbGetQuery(con, "SELECT COUNT(*) as n FROM obs")$n

# 获取多行
df <- DBI::dbReadTable(con, "obs")
```

---

## 十、开发进度追踪

### 10.1 已完成功能

| 功能 | Python | R | 说明 |
|------|--------|---|------|
| 数据库创建 | ✅ | ✅ | `atlas()` / `atlas()` |
| 数据加载 H5 | ✅ | ✅ | `load_AnnData()` / `load_data()` |
| 数据加载 H5AD | ✅ | ✅ | `load_AnnData()` / `load_data()` |
| 数据加载 RDS | ❌ | ✅ | `load_data()` |
| 数据加载 MTX | ✅ | ✅ | `load_AnnData()` / `load_data()` |
| 细胞过滤 | ✅ | ✅ | `filter_cells_*()` / `filter_cells()` |
| 基因过滤 | ✅ | ✅ | `filter_genes_*()` / `filter_genes()` |
| 归一化 | ✅ | ✅ | `normalize_total_*()` / `normalize_total()` |
| Log 变换 | ✅ | ✅ | `log1p_chunked()` / `log1p()` |
| Scale 标准化 | ✅ | ✅ | `scale_gene_chunked()` / `scale()` |
| 高变基因 | ✅ | ✅ | `highly_variable_genes()` / `highly_variable_genes()` |
| 质控指标 | ✅ | ✅ | `calculate_qc_metrics()` / `calculate_qc_metrics()` |
| 分批查询 | ✅ | ✅ | `query_minibatch()` / `query_minibatch()` |
| 索引子集 | ❌ | ✅ | `atlas[1:100]` |
| 导出 H5AD | ✅ | ✅ | `save_h5ad()` / `save_h5ad()` |

### 10.2 待改进项目

| 优先级 | 项目 | 说明 |
|--------|------|------|
| 高 | Python 列名对齐 | `X_log1p` → `log1p_factor` |
| 高 | Python 列名对齐 | `cell_total_counts` → `total_counts` |
| 高 | Python 列名对齐 | `gene_total_counts` → `total_counts` |
| 中 | Python 添加 cell_index | X_CSR_data 表添加 cell_index 列 |
| 中 | R 添加 uns 表 | 可选 |
| 低 | 文档完善 | 补充示例和使用说明 |

---

## 十一、完整源码对照

### 11.1 Python _atlas.py

实际源码请直接查看源文件 `scatlaspy/data/_atlas.py`。

### 11.2 R atlas.R 完整核心部分

```r
# Package/scAtlasAnalysis-R-version/R/atlas.R

#' Create Atlas Object
#'
#' @param name Database name (without extension)
#' @param path Database directory path
#' @param mode Access mode: "r" (read-only) or "r+" (read-write)
#'
#' @return atlas object
#' @export
atlas <- function(name, path = ".", mode = c("r", "r+")) {
  mode <- match.arg(mode)

  db_file <- file.path(path, paste0(name, ".sasql"))

  if (!file.exists(db_file)) {
    if (mode == "r") {
      stop(sprintf("Database file not found: %s", db_file))
    }
    con <- .create_db(db_file)
  } else {
    con <- .connect_db(db_file, read_only = (mode == "r"))
  }

  obs_r <- tryCatch(DBI::dbGetQuery(con, "SELECT cell_id FROM obs"),
                    error = function(e) data.frame(cell_id = character(0)))
  var_r <- tryCatch(DBI::dbGetQuery(con, "SELECT gene_id FROM var"),
                    error = function(e) data.frame(gene_id = character(0)))

  obj <- list(
    name = name,
    path = path,
    mode = mode,
    con = con,
    obs_cell_id = if (nrow(obs_r) > 0) obs_r$cell_id else NULL,
    var_gene_id = if (nrow(var_r) > 0) seq_len(nrow(var_r)) - 1 else NULL,
    is_view = FALSE
  )
  class(obj) <- "atlas"
  obj
}

#' Print atlas summary
#' @export
print.atlas <- function(x, ...) {
  cat("<atlas>\n")
  cat(sprintf("  Name: %s\n", x$name))
  cat(sprintf("  Path: %s\n", x$path))
  cat(sprintf("  Mode: %s\n", x$mode))

  n_cells <- if (is.null(x$obs_cell_id)) {
    DBI::dbGetQuery(x$con, "SELECT COUNT(*) FROM obs")$COUNT
  } else length(x$obs_cell_id)

  n_genes <- if (is.null(x$var_gene_id)) {
    DBI::dbGetQuery(x$con, "SELECT COUNT(*) FROM var")$COUNT
  } else length(x$var_gene_id)

  cat(sprintf("  Cells: %s\n", format(n_cells, big.mark = ",")))
  cat(sprintf("  Genes: %s\n", format(n_genes, big.mark = ",")))
  invisible(x)
}

#' Close database connection
#' @export
atlas_close <- function(a) {
  if (!is.null(a$con) && DBI::dbIsValid(a$con)) {
    DBI::dbDisconnect(a$con, shutdown = TRUE)
    a$con <- NULL
  }
  invisible(a)
}

#' Get cell metadata
#' @export
obs.atlas <- function(x, columns = NULL, ...) {
  # 实现见前文
}

#' Get gene metadata
#' @export
var.atlas <- function(x, columns = NULL, ...) {
  # 实现见前文
}

#' Get expression matrix
#' @export
X.atlas <- function(x, genes = NULL, sparse = TRUE, ...) {
  # 实现见前文
}

#' Index subset
#' @export
`[.atlas` <- function(x, i) {
  # 实现见前文
}

#' Minibatch query
#' @export
query_minibatch <- function(a, batch_size = 2048,
                            mode = c("order", "random_no_replace", "random_replace"),
                            callback = NULL) {
  # 实现见前文
}
```

---

## 附录

### A. 文件路径对照

| 描述 | Python 路径 | R 路径 |
|------|-------------|--------|
| 核心类 | `scatlaspy/data/_atlas.py` | `scAtlasAnalysis-R-version/R/atlas.R` |
| 预处理 | `scatlaspy/preprocessing/*.py` | `scAtlasAnalysis-R-version/R/preprocessing.R` |
| IO | `scatlaspy/io/*.py` | `scAtlasAnalysis-R-version/R/io.R` |
| Benchmark | `benchmark/Scripts/scatlaspy/*.py` | `benchmark/Scripts/scatlas-R/*.R` |

### B. 包依赖

**Python 依赖:**
```
scanpy>=1.9.0
anndata>=0.8.0
scipy>=1.9.0
numpy>=1.23.0
duckdb>=0.8.0
psutil>=5.9.0
```

**R 依赖:**
```
Depends: R (>= 3.5.0)
Imports: DBI, duckdb (>= 0.8.0), Matrix, jsonlite, parallel, data.table
```

---

*文档版本: 1.1*
*最后更新: 2026-01-19*
*更新说明: 修复路径硬编码问题*
