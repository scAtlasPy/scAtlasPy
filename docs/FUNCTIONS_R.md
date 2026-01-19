# scAtlasAnalysis（R包）重要函数源码



## 目录

1. [核心类 Atlas](#1-核心类-atlas)
2. [质控函数（Quality Control）](#2-质控函数-quality-control)
3. [转换函数（Transformation）](#3-转换函数-transformation)
4. [输入函数（Input/IO）](#4-输入函数-inputio)
5. [输出函数（Output/IO）](#5-输出函数-outputio)

---

## 1. 核心类 Atlas

### 1.1 构造函数/atlas.R:44-86）

这里我把Python版本的`_create()`和`connect()`, 还有`__init__()`函数都合并到R版本的`atlas()`函数中了
```r
atlas <- function(name, path = ".", mode = c("r", "r+")) {
  mode <- match.arg(mode)

  # 参数验证
  if (!is.character(name) || nchar(name) == 0) {
    stop("'name' 必须是非空字符向量")
  }

  # 构建数据库文件路径
  db_file <- file.path(path, paste0(name, ".sasql"))

  # 创建或连接数据库
  if (!file.exists(db_file)) {
    if (mode == "r") {
      stop(sprintf("数据库文件不存在: %s", db_file))
    }
    con <- .create_db(db_file)
    message(sprintf("[atlas] 创建新数据库: %s", db_file))
  } else {
    con <- .connect_db(db_file, read_only = (mode == "r"))
    message(sprintf("[atlas] 打开现有数据库: %s", db_file))
  }

  # 加载细胞/基因ID（用于视图筛选）
  obs_r <- tryCatch(DBI::dbGetQuery(con, "SELECT cell_id FROM obs"),
                    error = function(e) data.frame(cell_id = character(0)))
  var_r <- tryCatch(DBI::dbGetQuery(con, "SELECT gene_id FROM var"),
                    error = function(e) data.frame(gene_id = character(0)))

  # 构建atlas对象（S3类）
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

### 1.2 数据库连接/atlas.R:227-231）

```r
query <- function(a, sql, ...) {
  if (!inherits(a, "atlas")) stop("参数必须是atlas对象")
  if (is.null(a$con) || !DBI::dbIsValid(a$con)) stop("连接已关闭")
  DBI::dbGetQuery(a$con, sql)
}
```

### 1.3 视图/子集化/atlas.R:254-277）

```r
`[.atlas` <- function(a, item) {
  if (!inherits(a, "atlas")) stop("参数必须是atlas对象")

  all_ids <- a$obs_cell_id
  n <- length(all_ids)

  # 根据索引类型筛选
  if (is.numeric(item)) {
    # 整数索引
    if (length(item) == 1) {
      if (item >= 1 && item <= n) keep <- all_ids[item] else keep <- character(0)
    } else {
      valid <- item[item >= 1 & item <= n]
      keep <- if (length(valid) > 0) all_ids[valid] else character(0)
    }
  } else if (is.character(item)) {
    # 字符型索引
    keep <- item[item %in% all_ids]
  } else {
    stop("不支持的索引类型")
  }

  .new_view(a, obs_cell_id = keep)
}
```

### 1.4 分批查询/atlas.R:463-583）

```r
query_minibatch <- function(a, mode = c("order", "random_replace", "random_no_replace"),
                            batch_size = 2048, drop_last = TRUE,
                            callback = NULL, verbose = TRUE) {
  mode <- match.arg(mode)
  if (!inherits(a, "atlas")) stop("参数必须是atlas对象")
  if (is.null(a$con) || !DBI::dbIsValid(a$con)) stop("连接已关闭")

  con <- a$con

  # 获取所有cell_id
  if (is.null(a$obs_cell_id)) {
    all_cells_df <- DBI::dbGetQuery(con, "SELECT id, cell_id FROM obs ORDER BY id")
    all_cell_ids <- all_cells_df$cell_id
    all_cell_id_map <- all_cells_df$id
  } else {
    all_cells_df <- DBI::dbGetQuery(con,
      sprintf("SELECT id, cell_id FROM obs WHERE cell_id IN (%s) ORDER BY id",
              paste0("'", a$obs_cell_id, "'", collapse = ",")))
    all_cell_ids <- a$obs_cell_id
    all_cell_id_map <- setNames(all_cells_df$id, all_cells_df$cell_id)
  }

  total_num <- length(all_cell_ids)

  # 获取var表（所有批次相同）
  sub_var <- DBI::dbGetQuery(con, "SELECT * FROM var")
  gene_id <- sub_var$gene_id
  sub_var <- sub_var[, -1, drop = FALSE]

  results <- list()
  batch_num <- 0

  if (mode == "order") {
    num_batches <- if (drop_last) total_num %/% batch_size else ceiling(total_num / batch_size)

    for (i in seq_len(num_batches)) {
      offset <- (i - 1) * batch_size
      cur_size <- if (!drop_last && i == num_batches && total_num %% batch_size != 0) {
        total_num %% batch_size
      } else batch_size

      batch_cell_ids <- all_cell_ids[(offset + 1):(offset + cur_size)]
      batch_data <- .build_batch(a, batch_cell_ids, gene_id, con)

      adata <- list(X = batch_data$X, obs = batch_data$obs, var = sub_var,
                    obs_names = batch_data$cell_id, var_names = gene_id, shape = dim(batch_data$X))
      class(adata) <- "SimpleAnnData"

      batch_num <- batch_num + 1
      if (is.null(callback)) results[[batch_num]] <- adata else callback(batch_num, adata)
    }
  }

  if (is.null(callback)) invisible(results) else invisible(NULL)
}
```

### 1.5 SQL执行/atlas.R:227-231）

```r
query <- function(a, sql, ...) {
  if (!inherits(a, "atlas")) stop("参数必须是atlas对象")
  if (is.null(a$con) || !DBI::dbIsValid(a$con)) stop("连接已关闭")
  DBI::dbGetQuery(a$con, sql)
}
```

---

## 2. 质控函数（Quality Control）

### 2.1 filter_cells_CSR_ultrafast/preprocessing.R:46-115）

```r
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

  # Add filter column to obs table
  DBI::dbExecute(con, sprintf("
    ALTER TABLE obs
    ADD COLUMN IF NOT EXISTS %s BOOLEAN DEFAULT FALSE
  ", col_name))

  # Build filter conditions
  conds <- c()
  if (!is.null(min_counts)) conds <- c(conds, sprintf("sum_expr >= %d", min_counts))
  if (!is.null(max_counts)) conds <- c(conds, sprintf("sum_expr <= %d", max_counts))
  if (!is.null(min_genes)) conds <- c(conds, sprintf("nonzero >= %d", min_genes))
  if (!is.null(max_genes)) conds <- c(conds, sprintf("nonzero <= %d", max_genes))
  condition <- paste(conds, collapse = " AND ")

  # Find cells meeting criteria (using CSR data for efficiency)
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

  # Update filter column
  DBI::dbExecute(con, sprintf("UPDATE obs SET %s = FALSE", col_name))
  DBI::dbExecute(con, sprintf("
    UPDATE obs SET %s = TRUE
    WHERE id IN (SELECT cell_index FROM _filter_keep)
  ", col_name))

  # Get filtered cell IDs
  keep_ids <- DBI::dbGetQuery(con, sprintf(
    "SELECT cell_id FROM obs WHERE %s = TRUE", col_name
  ))$cell_id

  n_keep <- length(keep_ids)
  message(sprintf("[filter_cells] Kept: %s (%.1f%%)",
         format(n_keep, big.mark = ","), 100 * n_keep / total))

  DBI::dbExecute(con, "DROP TABLE IF EXISTS _filter_keep")

  # Return new atlas object (view with filtered cells)
  .new_atlas_view(a, obs_cell_id = keep_ids)
}
```

### 2.2 filter_genes_CSR/preprocessing.R:145-207）

```r
filter_genes.atlas <- function(a,
                               min_counts = NULL,
                               min_cells = NULL,
                               max_counts = NULL,
                               max_cells = NULL,
                               col_name = "filter_genes_1",
                               ...) {
  if (!inherits(a, "atlas")) stop("Argument must be an atlas object")
  if (is.null(a$con) || !DBI::dbIsValid(a$con)) stop("Database connection is closed")

  if (all(sapply(list(min_counts, min_cells, max_counts, max_cells), is.null))) {
    stop("At least one filter condition must be specified")
  }

  con <- a$con
  total <- DBI::dbGetQuery(con, "SELECT COUNT(*) AS n FROM var")$n[1]
  message(sprintf("[filter_genes] Total genes: %s", format(total, big.mark = ",")))

  # Add filter column
  DBI::dbExecute(con, sprintf("
    ALTER TABLE var
    ADD COLUMN IF NOT EXISTS %s BOOLEAN DEFAULT FALSE
  ", col_name))

  # Build conditions
  conds <- c()
  if (!is.null(min_counts)) conds <- c(conds, sprintf("sum_expr >= %d", min_counts))
  if (!is.null(max_counts)) conds <- c(conds, sprintf("sum_expr <= %d", max_counts))
  if (!is.null(min_cells)) conds <- c(conds, sprintf("nonzero >= %d", min_cells))
  if (!is.null(max_cells)) conds <- c(conds, sprintf("nonzero <= %d", max_cells))
  condition <- paste(conds, collapse = " AND ")

  # Find genes meeting criteria
  DBI::dbExecute(con, "DROP TABLE IF EXISTS _filter_keep")
  DBI::dbExecute(con, sprintf("
    CREATE TEMP TABLE _filter_keep AS
    SELECT indices AS gene_index
    FROM (
      SELECT indices, SUM(data) AS sum_expr, COUNT(*) AS nonzero
      FROM X_CSR_data
      GROUP BY indices
    ) WHERE %s
  ", condition))

  # Update filter column
  DBI::dbExecute(con, sprintf("UPDATE var SET %s = FALSE", col_name))
  DBI::dbExecute(con, sprintf("
    UPDATE var SET %s = TRUE
    WHERE id IN (SELECT gene_index FROM _filter_keep)
  ", col_name))

  keep_ids <- DBI::dbGetQuery(con, sprintf(
    "SELECT gene_id FROM var WHERE %s = TRUE", col_name
  ))$gene_id

  n_keep <- length(keep_ids)
  message(sprintf("[filter_genes] Kept: %s (%.1f%%)",
         format(n_keep, big.mark = ","), 100 * n_keep / total))

  DBI::dbExecute(con, "DROP TABLE IF EXISTS _filter_keep")

  .new_atlas_view(a, var_gene_id = keep_ids)
}
```

### 2.3 calculate_qc_metrics/preprocessing.R:819-984）

```r
calculate_qc_metrics.atlas <- function(a,
                                       mt_prefix = "MT-",
                                       mt_key = "mt",
                                       ...) {
  if (!inherits(a, "atlas")) stop("Argument must be an atlas object")
  if (is.null(a$con) || !DBI::dbIsValid(a$con)) stop("Database connection is closed")

  con <- a$con
  message("[calculate_qc_metrics] Computing QC metrics...")

  start_time <- Sys.time()

  # Step 1: Mark mitochondrial genes in var
  message("  Step 1: Marking mitochondrial genes...")
  DBI::dbExecute(con, sprintf("
    ALTER TABLE var
    ADD COLUMN IF NOT EXISTS %s BOOLEAN
  ", mt_key))

  DBI::dbExecute(con, sprintf("
    UPDATE var
    SET %s = CASE
      WHEN gene_id LIKE '%s%%' THEN TRUE
      ELSE FALSE
    END
  ", mt_key, mt_prefix))

  n_mt <- DBI::dbGetQuery(con, sprintf(
    "SELECT COUNT(*) AS n FROM var WHERE %s = TRUE", mt_key
  ))$n[1]

  # Step 2: Cell-wise QC (total_counts, n_genes_by_counts)
  message("  Step 2: Computing cell-wise QC metrics...")

  DBI::dbExecute(con, "
    CREATE OR REPLACE TEMP TABLE _cell_basic AS
    SELECT
      cell_index AS id,
      SUM(data) AS total_counts,
      COUNT(*) AS n_genes_by_counts
    FROM X_CSR_data
    WHERE data IS NOT NULL
    GROUP BY cell_index
  ")

  DBI::dbExecute(con, "ALTER TABLE obs ADD COLUMN IF NOT EXISTS total_counts REAL")
  DBI::dbExecute(con, "ALTER TABLE obs ADD COLUMN IF NOT EXISTS n_genes_by_counts INTEGER")

  DBI::dbExecute(con, "
    UPDATE obs
    SET
      total_counts = c.total_counts,
      n_genes_by_counts = c.n_genes_by_counts
    FROM _cell_basic c
    WHERE obs.id = c.id
  ")

  n_cells <- DBI::dbGetQuery(con, "SELECT COUNT(*) AS n FROM obs")$n[1]

  # Step 3: Mitochondrial QC (total_counts_mt, pct_counts_mt)
  message("  Step 3: Computing mitochondrial QC metrics...")

  if (n_mt > 0) {
    DBI::dbExecute(con, sprintf("
      CREATE OR REPLACE TEMP TABLE _cell_mt AS
      SELECT
        x.cell_index AS id,
        SUM(x.data) AS total_counts_mt
      FROM X_CSR_data x
      JOIN var v ON x.indices = v.id
      WHERE v.%s = TRUE
      GROUP BY x.cell_index
    ", mt_key))

    DBI::dbExecute(con, "ALTER TABLE obs ADD COLUMN IF NOT EXISTS total_counts_mt REAL")
    DBI::dbExecute(con, "ALTER TABLE obs ADD COLUMN IF NOT EXISTS pct_counts_mt REAL")

    DBI::dbExecute(con, "
      UPDATE obs
      SET
        total_counts_mt = COALESCE(m.total_counts_mt, 0),
        pct_counts_mt = CASE
          WHEN obs.total_counts > 0
          THEN 100.0 * COALESCE(m.total_counts_mt, 0) / obs.total_counts
          ELSE 0
        END
      FROM _cell_mt m
      WHERE obs.id = m.id
    ")
  }

  # Step 4: Gene-wise QC (total_counts, n_cells_by_counts)
  message("  Step 4: Computing gene-wise QC metrics...")

  DBI::dbExecute(con, "
    CREATE OR REPLACE TEMP TABLE _gene_qc AS
    SELECT
      indices AS id,
      SUM(data) AS total_counts,
      COUNT(DISTINCT cell_index) AS n_cells_by_counts
    FROM X_CSR_data
    WHERE data IS NOT NULL
    GROUP BY indices
  ")

  DBI::dbExecute(con, "ALTER TABLE var ADD COLUMN IF NOT EXISTS total_counts REAL")
  DBI::dbExecute(con, "ALTER TABLE var ADD COLUMN IF NOT EXISTS n_cells_by_counts INTEGER")

  DBI::dbExecute(con, "
    UPDATE var
    SET
      total_counts = g.total_counts,
      n_cells_by_counts = g.n_cells_by_counts
    FROM _gene_qc g
    WHERE var.id = g.id
  ")

  # Clean up
  DBI::dbExecute(con, "DROP TABLE IF EXISTS _cell_basic")
  DBI::dbExecute(con, "DROP TABLE IF EXISTS _cell_mt")
  DBI::dbExecute(con, "DROP TABLE IF EXISTS _gene_qc")

  invisible(a)
}
```

### 2.4 calculate_cell_total_counts/preprocessing.R:1011-1062）

```r
calculate_cell_total_counts.atlas <- function(a,
                                               col_name = "total_counts",
                                               batch_size = 2048,
                                               ...) {
  if (!inherits(a, "atlas")) stop("Argument must be an atlas object")
  if (is.null(a$con) || !DBI::dbIsValid(a$con)) stop("Database connection is closed")

  con <- a$con
  message("[calculate_cell_total_counts] Computing total counts per cell...")

  start_time <- Sys.time()

  # Add column if not exists
  col_check <- DBI::dbGetQuery(con, sprintf("
    SELECT COUNT(*) AS n FROM information_schema.columns
    WHERE table_name = 'obs' AND column_name = '%s'
  ", col_name))$n[1]

  if (col_check == 0) {
    DBI::dbExecute(con, sprintf("
      ALTER TABLE obs ADD COLUMN %s REAL
    ", col_name))
  }

  # Compute total counts using SQL aggregation
  DBI::dbExecute(con, "
    CREATE OR REPLACE TEMP TABLE _cell_total AS
    SELECT
      cell_index AS id,
      SUM(data) AS total
    FROM X_CSR_data
    WHERE data IS NOT NULL
    GROUP BY cell_index
  ")

  DBI::dbExecute(con, sprintf("
    UPDATE obs
    SET %s = COALESCE(t.total, 0)
    FROM _cell_total t
    WHERE obs.id = t.id
  ", col_name))

  n_cells <- DBI::dbGetQuery(con, "SELECT COUNT(*) AS n FROM obs WHERE NOT isnan(total_counts)")$n[1]

  DBI::dbExecute(con, "DROP TABLE IF EXISTS _cell_total")

  elapsed <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
  message(sprintf("[calculate_cell_total_counts] Computed for %s cells in %.2f seconds",
         format(n_cells, big.mark = ","), elapsed))

  invisible(a)
}
```

### 2.5 calculate_gene_total_counts/preprocessing.R:1089-1154）

```r
calculate_gene_total_counts.atlas <- function(a,
                                               col_total = "total_counts",
                                               col_mean = "mean_counts",
                                               ...) {
  if (!inherits(a, "atlas")) stop("Argument must be an atlas object")
  if (is.null(a$con) || !DBI::dbIsValid(a$con)) stop("Database connection is closed")

  con <- a$con
  message("[calculate_gene_total_counts] Computing total and mean per gene...")

  start_time <- Sys.time()

  # Add columns if not exist
  col_check <- DBI::dbGetQuery(con, sprintf("
    SELECT COUNT(*) AS n FROM information_schema.columns
    WHERE table_name = 'var' AND column_name = '%s'
  ", col_total))$n[1]

  if (col_check == 0) {
    DBI::dbExecute(con, sprintf("
      ALTER TABLE var ADD COLUMN %s REAL DEFAULT 0.0
    ", col_total))
  }

  col_check <- DBI::dbGetQuery(con, sprintf("
    SELECT COUNT(*) AS n FROM information_schema.columns
    WHERE table_name = 'var' AND column_name = '%s'
  ", col_mean))$n[1]

  if (col_check == 0) {
    DBI::dbExecute(con, sprintf("
      ALTER TABLE var ADD COLUMN %s REAL DEFAULT 0.0
    ", col_mean))
  }

  # Compute gene statistics using SQL
  DBI::dbExecute(con, "
    CREATE OR REPLACE TEMP TABLE _gene_stats AS
    SELECT
      indices AS id,
      SUM(data) AS total,
      AVG(data) AS mean_val
    FROM X_CSR_data
    WHERE data IS NOT NULL
    GROUP BY indices
  ")

  DBI::dbExecute(con, sprintf("
    UPDATE var
    SET
      %s = COALESCE(s.total, 0),
      %s = COALESCE(s.mean_val, 0)
    FROM _gene_stats s
    WHERE var.id = s.id
  ", col_total, col_mean))

  DBI::dbExecute(con, "DROP TABLE IF EXISTS _gene_stats")

  elapsed <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
  message(sprintf("[calculate_gene_total_counts] Computed for all genes in %.2f seconds", elapsed))

  invisible(a)
}
```

---

## 3. 转换函数（Transformation）

### 3.1 normalize_total_scale_factor/preprocessing.R:236-283）

```r
normalize_total.atlas <- function(a,
                                  target_sum = 10000,
                                  col_name = "scale_factor",
                                  field = "data",
                                  ...) {
  if (!inherits(a, "atlas")) stop("Argument must be an atlas object")
  if (is.null(a$con) || !DBI::dbIsValid(a$con)) stop("Database connection is closed")

  con <- a$con
  message("[normalize_total] Computing scale factors...")

  # Compute total counts per cell
  DBI::dbExecute(con, "DROP TABLE IF EXISTS _cell_sum")
  DBI::dbExecute(con, sprintf("
    CREATE TEMP TABLE _cell_sum AS
    SELECT cell_index, SUM(%s) AS total
    FROM X_CSR_data
    GROUP BY cell_index
  ", field))

  # Use median if target_sum is NULL
  if (is.null(target_sum)) {
    target_sum <- DBI::dbGetQuery(con, "SELECT median(total) AS m FROM _cell_sum")$m[1]
    message(sprintf("  Using median: %s", format(target_sum, digits = 6)))
  }

  # Add or update scale factor column
  DBI::dbExecute(con, sprintf("
    ALTER TABLE obs
    ADD COLUMN IF NOT EXISTS %s REAL
  ", col_name))

  DBI::dbExecute(con, sprintf("
    UPDATE obs
    SET %s = CASE WHEN s.total > 0 THEN %f / s.total ELSE 0 END
    FROM _cell_sum AS s
    WHERE obs.id = s.cell_index
  ", col_name, as.numeric(target_sum)))

  n <- DBI::dbGetQuery(con, sprintf(
    "SELECT COUNT(*) AS n FROM obs WHERE %s > 0", col_name
  ))$n[1]

  message(sprintf("[normalize_total] Scale factors computed: %s cells", format(n, big.mark = ",")))

  DBI::dbExecute(con, "DROP TABLE IF EXISTS _cell_sum")
  invisible(a)
}
```

### 3.2 log1p_chunked/preprocessing.R:311-393）

```r
log1p.atlas <- function(a,
                        base = NULL,
                        col_name = "log1p_factor",
                        field = "data",
                        chunk_size = 100000000,
                        ...) {
  if (!inherits(a, "atlas")) stop("Argument must be an atlas object")
  if (is.null(a$con) || !DBI::dbIsValid(a$con)) stop("Database connection is closed")

  con <- a$con
  message("[log1p] Applying log(1+x) transformation...")

  # Enable DuckDB parallel writes
  tryCatch({
    n_threads <- parallel::detectCores()
    DBI::dbExecute(con, sprintf("SET threads = %d", n_threads))
  }, error = function(e) {
    message("  Could not set DuckDB threads: ", e$message)
  })

  # Check if input field exists
  field_exists <- DBI::dbGetQuery(con, sprintf("
    SELECT COUNT(*) AS cnt
    FROM information_schema.columns
    WHERE table_name = 'X_CSR_data'
      AND column_name = '%s'
  ", field))$cnt[1]

  if (field_exists == 0) {
    stop(sprintf("Field '%s' does not exist in X_CSR_data", field))
  }

  # Determine log expression
  if (is.null(base)) {
    log_expr <- sprintf("ln(1.0 + %s)", field)
    message("  Using natural logarithm (ln)")
  } else {
    log_expr <- sprintf("log(%f, 1.0 + %s)", as.numeric(base), field)
  }

  # Add output column
  DBI::dbExecute(con, sprintf("
    ALTER TABLE X_CSR_data
    ADD COLUMN IF NOT EXISTS %s REAL
  ", col_name))

  # Get id range for chunked updates
  id_range <- DBI::dbGetQuery(con, "SELECT MIN(id) AS min, MAX(id) AS max FROM X_CSR_data")
  min_id <- id_range$min[1]
  max_id <- id_range$max[1]
  n_chunks <- ceiling((max_id - min_id + 1) / chunk_size)

  message(sprintf("  Processing %s records in %d chunks...",
         format(n_total, big.mark = ","), n_chunks))

  # Update in chunks to avoid memory issues
  for (i in seq_len(n_chunks)) {
    start_id <- min_id + (i - 1) * chunk_size
    end_id <- min(start_id + chunk_size - 1, max_id)

    DBI::dbExecute(con, sprintf("
      UPDATE X_CSR_data
      SET %s = %s
      WHERE id BETWEEN %s AND %s
        AND %s IS NOT NULL
    ", col_name, log_expr, as.character(start_id), as.character(end_id), field))
  }

  message(sprintf("[log1p] Transformation complete: %s", col_name))
  invisible(a)
}
```

### 3.3 scale_gene_chunked/preprocessing.R:533-681）

```r
scale.atlas <- function(a,
                        max_value = NULL,
                        col_name = "X_scale",
                        field = "data",
                        gene_chunk_size = 512,
                        use_hvg = FALSE,
                        hvg_key = "highly_variable",
                        ...) {
  if (!inherits(a, "atlas")) stop("Argument must be an atlas object")
  if (is.null(a$con) || !DBI::dbIsValid(a$con)) stop("Database connection is closed")

  con <- a$con
  message("[scale] Computing z-scores (DuckDB OLAP optimized)...")

  # Enable DuckDB parallel writes
  tryCatch({
    n_threads <- parallel::detectCores()
    DBI::dbExecute(con, sprintf("SET threads = %d", n_threads))
  }, error = function(e) {
    message("  Could not set DuckDB threads: ", e$message)
  })

  # Check if input field exists
  field_exists <- DBI::dbGetQuery(con, sprintf("
    SELECT COUNT(*) AS cnt
    FROM information_schema.columns
    WHERE table_name = 'X_CSR_data'
      AND column_name = '%s'
  ", field))$cnt[1]

  if (field_exists == 0) {
    stop(sprintf("Field '%s' does not exist in X_CSR_data", field))
  }

  # Add output column
  DBI::dbExecute(con, sprintf("
    ALTER TABLE X_CSR_data
    ADD COLUMN IF NOT EXISTS %s REAL
  ", col_name))

  # Get gene list
  if (use_hvg) {
    gene_ids <- DBI::dbGetQuery(con, sprintf("
      SELECT id FROM var
      WHERE %s = TRUE
      ORDER BY id
    ", hvg_key))$id
  } else {
    gene_ids <- DBI::dbGetQuery(con, "
      SELECT DISTINCT indices
      FROM X_CSR_data
      ORDER BY indices
    ")$indices
  }

  n_genes <- length(gene_ids)
  n_chunks <- ceiling(n_genes / gene_chunk_size)

  # Create temporary table for scaled values
  DBI::dbExecute(con, "
    CREATE TEMP TABLE _X_scale_tmp (
      id BIGINT,
      indices INTEGER,
      val REAL
    )
  ")

  # Process genes in chunks
  for (i in seq_len(n_chunks)) {
    chunk_start <- (i - 1) * gene_chunk_size
    chunk_genes <- gene_ids[(chunk_start + 1):min(chunk_start + gene_chunk_size, n_genes)]
    gene_list_sql <- paste(chunk_genes, collapse = ",")

    # Compute gene-wise statistics
    DBI::dbExecute(con, sprintf("
      CREATE OR REPLACE TEMP TABLE _gene_stat AS
      SELECT
        indices,
        AVG(%s) AS mean,
        STDDEV_POP(%s) AS std
      FROM X_CSR_data
      WHERE indices IN (%s)
        AND %s IS NOT NULL
      GROUP BY indices
    ", field, field, gene_list_sql, field))

    # Compute z-scores with optional clipping
    if (is.null(max_value)) {
      DBI::dbExecute(con, sprintf("
        INSERT INTO _X_scale_tmp
        SELECT
          x.id,
          x.indices,
          CASE
            WHEN g.std > 0 THEN (x.%s - g.mean) / g.std
            ELSE 0
          END
        FROM X_CSR_data x
        JOIN _gene_stat g ON x.indices = g.indices
        WHERE x.indices IN (%s)
          AND x.%s IS NOT NULL
      ", field, gene_list_sql, field))
    } else {
      DBI::dbExecute(con, sprintf("
        INSERT INTO _X_scale_tmp
        SELECT
          x.id,
          x.indices,
          CASE
            WHEN g.std > 0 THEN
              LEAST(%.6f, GREATEST(-%.6f, (x.%s - g.mean) / g.std))
            ELSE 0
          END
        FROM X_CSR_data x
        JOIN _gene_stat g ON x.indices = g.indices
        WHERE x.indices IN (%s)
          AND x.%s IS NOT NULL
      ", as.numeric(max_value), as.numeric(max_value), field, gene_list_sql, field))
    }
  }

  # Merge scaled values back to X_CSR_data
  message("  Merging scaled values back to X_CSR_data...")
  DBI::dbExecute(con, sprintf("
    UPDATE X_CSR_data x
    SET %s = t.val
    FROM _X_scale_tmp t
    WHERE x.id = t.id AND x.indices = t.indices
  ", col_name))

  # Cleanup
  DBI::dbExecute(con, "DROP TABLE IF EXISTS _X_scale_tmp")
  DBI::dbExecute(con, "DROP TABLE IF EXISTS _gene_stat")

  message(sprintf("[scale] Z-score complete: %s", col_name))
  invisible(a)
}
```

### 3.4 highly_variable_genes/preprocessing.R:422-505）

```r
highly_variable_genes.atlas <- function(a,
                                        flavor = c("var", "cv"),
                                        n_top = NULL,
                                        col_name = "highly_variable_genes",
                                        field = "data",
                                        ...) {
  if (!inherits(a, "atlas")) stop("Argument must be an atlas object")
  flavor <- match.arg(flavor)
  if (is.null(a$con) || !DBI::dbIsValid(a$con)) stop("Database connection is closed")

  con <- a$con
  message(sprintf("[highly_variable_genes] flavor = %s", flavor))

  # Step 1: Compute gene-level statistics
  message("  Step 1: Computing gene statistics...")
  DBI::dbExecute(con, "DROP TABLE IF EXISTS _hvg_stats")
  DBI::dbExecute(con, sprintf("
    CREATE TEMP TABLE _hvg_stats AS
    SELECT
      indices AS gene_index,
      AVG(%s) AS mean,
      VAR_POP(%s) AS var,
      STDDEV_POP(%s) AS std
    FROM X_CSR_data
    WHERE %s IS NOT NULL
    GROUP BY indices
  ", field, field, field, field))

  # Step 2: Compute score
  message("  Step 2: Computing scores...")
  if (flavor == "var") {
    score_expr <- "var"
  } else {
    score_expr <- "CASE WHEN mean > 0 THEN std / mean ELSE 0 END"
  }

  DBI::dbExecute(con, sprintf("
    CREATE TEMP TABLE _hvg_score AS
    SELECT gene_index, %s AS score FROM _hvg_stats
  ", score_expr))

  # Step 3: Select top genes
  if (!is.null(n_top)) {
    message(sprintf("  Step 3: Selecting top %s genes...", format(n_top, big.mark = ",")))
    DBI::dbExecute(con, sprintf("
      CREATE TEMP TABLE _hvg_top AS
      SELECT gene_index FROM _hvg_score
      ORDER BY score DESC LIMIT %d
    ", as.integer(n_top)))
  }

  # Step 4: Write to var table
  message(sprintf("  Step 4: Writing to var.%s", col_name))
  DBI::dbExecute(con, sprintf("
    ALTER TABLE var
    ADD COLUMN IF NOT EXISTS %s BOOLEAN
  ", col_name))

  DBI::dbExecute(con, sprintf("UPDATE var SET %s = FALSE", col_name))
  DBI::dbExecute(con, sprintf("
    UPDATE var SET %s = TRUE
    WHERE id IN (SELECT gene_index FROM _hvg_top)
  ", col_name))

  # Count
  n_hvg <- DBI::dbGetQuery(con, sprintf(
    "SELECT SUM(CASE WHEN %s THEN 1 ELSE 0 END) AS n FROM var", col_name
  ))$n[1]

  message(sprintf("[highly_variable_genes] Found %s HVG", format(n_hvg, big.mark = ",")))

  # Clean up
  DBI::dbExecute(con, "DROP TABLE IF EXISTS _hvg_stats")
  DBI::dbExecute(con, "DROP TABLE IF EXISTS _hvg_score")
  DBI::dbExecute(con, "DROP TABLE IF EXISTS _hvg_top")

  invisible(a)
}
```

### 3.5 normalize_and_log1p/preprocessing.R:1188-1262）

```r
normalize_and_log1p.atlas <- function(a,
                                       target_sum = 10000,
                                       scale_key = "scale_factor",
                                       col_name = "log1p_factor",
                                       field = "data",
                                       base = NULL,
                                       chunk_size = 100000000,
                                       ...) {
  if (!inherits(a, "atlas")) stop("Argument must be an atlas object")
  if (is.null(a$con) || !DBI::dbIsValid(a$con)) stop("Database connection is closed")

  con <- a$con
  message("[normalize_and_log1p] Normalizing and log-transforming...")

  start_time <- Sys.time()

  # Step 1: Compute scale factors (normalize_total)
  message("  Step 1: Computing scale factors...")
  normalize_total.atlas(a, target_sum = target_sum, col_name = scale_key, field = field)

  # Step 2: Apply log(1 + data * scale_factor)
  message("  Step 2: Applying log1p with scale factors...")

  # Determine log expression
  if (is.null(base)) {
    log_expr <- sprintf("ln(1.0 + x.%s * o.%s)", field, scale_key)
  } else {
    log_expr <- sprintf("log(%f, 1.0 + x.%s * o.%s)", as.numeric(base), field, scale_key)
  }

  # Add output column
  DBI::dbExecute(con, sprintf("
    ALTER TABLE X_CSR_data
    ADD COLUMN IF NOT EXISTS %s REAL
  ", col_name))

  # Get id range
  id_range <- DBI::dbGetQuery(con, "SELECT MIN(id) AS min, MAX(id) AS max FROM X_CSR_data")
  min_id <- id_range$min[1]
  max_id <- id_range$max[1]

  n_total <- max_id - min_id + 1
  n_chunks <- ceiling(n_total / chunk_size)

  # Update in chunks with JOIN to obs for scale factors
  for (i in seq_len(n_chunks)) {
    start_id <- min_id + (i - 1) * chunk_size
    end_id <- min(start_id + chunk_size - 1, max_id)

    DBI::dbExecute(con, sprintf("
      UPDATE X_CSR_data AS x
      SET %s = %s
      FROM obs AS o
      WHERE x.cell_index = o.id
        AND x.id BETWEEN %s AND %s
        AND x.%s IS NOT NULL
    ", col_name, log_expr, as.character(start_id), as.character(end_id), field))
  }

  elapsed <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
  message(sprintf("[normalize_and_log1p] Complete in %.2f seconds", elapsed))

  invisible(a)
}
```

---

## 4. 输入函数（Input/IO）

### 4.1 load_data/io.R:84-132）

```r
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

  # Get final statistics
  n_obs <- DBI::dbGetQuery(a$con, "SELECT COUNT(*) as n FROM obs")$n
  n_vars <- DBI::dbGetQuery(a$con, "SELECT COUNT(*) as n FROM var")$n
  nnz <- DBI::dbGetQuery(a$con, "SELECT COUNT(*) as n FROM X_CSR_data")$n

  # Refresh cell/gene IDs - return new atlas object with updated IDs
  obs_r <- DBI::dbGetQuery(a$con, "SELECT cell_id FROM obs")
  var_r <- DBI::dbGetQuery(a$con, "SELECT gene_id FROM var")
  a$obs_cell_id <- if (nrow(obs_r) > 0) obs_r$cell_id else NULL
  a$var_gene_id <- if (nrow(var_r) > 0) var_r$gene_id else NULL

  invisible(a)
}
```

---

## 5. 输出函数（Output/IO）

### 5.1 save_h5ad/io.R:627-679）

```r
save_h5ad <- function(a, file, write_obs = TRUE, write_var = TRUE, write_X = TRUE) {
  if (!inherits(a, "atlas")) stop("Argument must be an atlas object")
  if (!requireNamespace("hdf5r", quietly = TRUE)) {
    stop("Package 'hdf5r' is required")
  }

  message(sprintf("[save_h5ad] %s", file))

  # Get filtered IDs
  cell_ids <- if (is.null(a$obs_cell_id)) {
    DBI::dbGetQuery(a$con, "SELECT cell_id FROM obs ORDER BY id")$cell_id
  } else a$obs_cell_id

  gene_ids <- if (is.null(a$var_gene_id)) {
    DBI::dbGetQuery(a$con, "SELECT gene_id FROM var ORDER BY id")$gene_id
  } else a$var_gene_id

  if (file.exists(file)) unlink(file)
  h5f <- hdf5r::H5File$new(file, mode = "w")

  # Write obs
  if (write_obs) {
    obs_df <- obs(a)
    obs_df[] <- lapply(obs_df, function(x) if (is.logical(x)) as.integer(x) else x)
    h5f$create_dataset("/obs", robj = as.matrix(obs_df))
  }

  # Write var
  if (write_var) {
    var_df <- var(a)
    var_df[] <- lapply(var_df, function(x) if (is.logical(x)) as.integer(x) else x)
    h5f$create_dataset("/var", robj = as.matrix(var_df))
  }

  # Write X
  if (write_X) {
    message("  X...")
    X_data <- X(a)
    h5g <- h5f$create_group("/X")
    h5g$create_dataset("format", robj = "CSR")
    h5g$create_dataset("shape", robj = c(length(cell_ids), length(gene_ids)))
    h5g$create_dataset("data", robj = X_data$data)
    h5g$create_dataset("indices", robj = X_data$indices)
    h5g$create_dataset("indptr", robj = X_data$indptr)
  }

  h5f$close()
  message("[save_h5ad] Complete")
  invisible(a)
}
```

---

## 返回类型汇总表

| 函数分类 | 函数名 | 返回类型 | 设计风格 |
|---------|-------|---------|---------|
| **Core Atlas** | `atlas()` | `atlas`对象 | 构造函数 |
| | `query()` | `data.frame` | 查询结果 |
| | `[.atlas` | `atlas`视图 | 不可变视图 |
| | `query_minibatch()` | `list`或`invisible` | 分批结果 |
| **QC** | `filter_cells()` | `atlas`视图 | **不可变（返回新对象）** |
| | `filter_genes()` | `atlas`视图 | **不可变（返回新对象）** |
| | `calculate_qc_metrics()` | `invisible(a)` | 原地操作 |
| | `calculate_cell_total_counts()` | `invisible(a)` | 原地操作 |
| | `calculate_gene_total_counts()` | `invisible(a)` | 原地操作 |
| **Transformation** | `normalize_total()` | `invisible(a)` | 原地操作 |
| | `log1p()` | `invisible(a)` | 原地操作 |
| | `scale()` | `invisible(a)` | 原地操作 |
| | `highly_variable_genes()` | `invisible(a)` | 原地操作 |
| | `normalize_and_log1p()` | `invisible(a)` | 原地操作 |
| **Input** | `load_data()` | `invisible(a)` | 加载数据 |
| **Output** | `save_h5ad()` | `invisible(a)` | 保存数据 |

---

## R vs Python 实现对比要点

### 1. 筛选操作（filter_cells/filter_genes）

| 特性 | Python | R |
|-----|--------|---|
| 返回值 | `None`（原地修改） | 新`atlas`视图对象 |
| 筛选标记 | 在obs/var表添加布尔列 | 在obs/var表添加布尔列 |
| 筛选应用 | 通过`__obs_cell_id`列表 | 通过`obs_cell_id`列表 |
| 性能 | O(1)新对象，IN查询可能有瓶颈 | O(1)新对象，支持管道操作 |

### 2. 转换操作（normalize/scale/log1p）

| 特性 | Python | R |
|-----|--------|---|
| 返回值 | `None` | `invisible(a)` |
| 缩放因子存储 | obs表 | obs表 |
| 转换值存储 | X_CSR_data新字段 | X_CSR_data新字段 |
| 分块处理 | ✓ | ✓ |
| 并行处理 | DuckDB自动 | 手动设置threads |

### 3. 关键差异

- **R使用`invisible(a)`**：使结果不打印但可继续用于管道操作
- **Python返回`None`**：直接修改自身，不支持链式调用
- **R的筛选返回新对象**：符合R的函数式编程惯用法
- **两者都使用DuckDB**：SQL操作基本一致
