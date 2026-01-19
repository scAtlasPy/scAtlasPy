# atlas.R - Atlas核心（S3对象）
# =============================================================================
#
# 功能：管理和分析单细胞图谱数据的数据库交互
# 对应Python版本：scatlaspy/data/_atlas.py
#
# 主要功能：
# 1. 创建/连接DuckDB数据库
# 2. SQL查询（query）
# 3. 分批查询（query_minibatch）
# 4. 视图系统（基于obs_cell_id筛选）
#
# =============================================================================

# -----------------------------------------------------------------------------
# 构造函数：创建或打开数据库
# -----------------------------------------------------------------------------

#' 创建Atlas对象
#'
#' 创建或打开一个由DuckDB支持的单细胞图谱数据库。
#'
#' @param name 字符型。数据库名称（不带扩展名）。
#' @param path 字符型。数据库目录路径。默认：当前目录。
#' @param mode 字符型。访问模式："r"（只读）或 "r+"（读写）。
#'
#' @return atlas对象，包含以下字段：
#'   - name: 数据库名称
#'   - path: 数据库路径
#'   - mode: 访问模式
#'   - con: DuckDB连接
#'   - obs_cell_id: 筛选后的细胞ID（NULL表示全部）
#'   - var_gene_id: 筛选后的基因ID（NULL表示全部）
#'   - is_view: 是否为视图
#'
#' @export
#'
#' @examples
#' # 创建新数据库
#' atlas <- atlas("my_data", path = ".", mode = "r+")
#'
#' # 打开现有数据库（只读）
#' atlas <- atlas("my_data", path = ".", mode = "r")
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
  # 注意：加载数据后需要重新赋值以刷新这些ID
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


# -----------------------------------------------------------------------------
# 内部函数：创建数据库
# -----------------------------------------------------------------------------

# 创建新的DuckDB数据库并初始化表结构
.create_db <- function(db_file) {
  # 创建目录
  dir.create(dirname(db_file), recursive = TRUE, showWarnings = FALSE)

  # 连接数据库
  con <- DBI::dbConnect(duckdb::duckdb(db_file))
  .set_pragma(con)

  # 创建表结构
  # obs表：细胞元数据
  DBI::dbExecute(con, "CREATE TABLE obs (
    id INTEGER PRIMARY KEY,
    cell_id VARCHAR NOT NULL
  )")

  # var表：基因元数据
  DBI::dbExecute(con, "CREATE TABLE var (
    id INTEGER PRIMARY KEY,
    gene_id VARCHAR NOT NULL
  )")

  # X_CSR_indptr表：CSR稀疏矩阵的行指针
  DBI::dbExecute(con, "CREATE TABLE X_CSR_indptr (
    id BIGINT PRIMARY KEY,
    cell_id VARCHAR NOT NULL,
    indptr BIGINT NOT NULL
  )")

  # X_CSR_data表：CSR稀疏矩阵的数据
  DBI::dbExecute(con, "CREATE TABLE X_CSR_data (
    id BIGINT PRIMARY KEY,
    cell_index BIGINT NOT NULL,
    indices INTEGER NOT NULL,
    data REAL NOT NULL
  )")

  # 创建索引以加速查询
  DBI::dbExecute(con, "CREATE INDEX idx_csr_cell ON X_CSR_data(cell_index)")
  DBI::dbExecute(con, "CREATE INDEX idx_csr_gene ON X_CSR_data(indices)")

  # uns表：非结构化元数据（键值对）
  DBI::dbExecute(con, "CREATE TABLE uns (
    id INTEGER PRIMARY KEY,
    key VARCHAR UNIQUE NOT NULL,
    value_type VARCHAR NOT NULL,
    value_string TEXT,
    value_real DOUBLE
  )")

  con
}


# -----------------------------------------------------------------------------
# 内部函数：连接数据库
# -----------------------------------------------------------------------------

# 连接到现有数据库
.connect_db <- function(db_file, read_only = TRUE) {
  con <- DBI::dbConnect(duckdb::duckdb(db_file), read_only = read_only)
  .set_pragma(con)

  # 验证数据库结构
  required <- c("obs", "var", "X_CSR_indptr", "X_CSR_data")
  existing <- DBI::dbListTables(con)
  missing <- setdiff(required, existing)
  if (length(missing) > 0) {
    stop(sprintf("数据库缺少必需的表: %s", paste(missing, collapse = ", ")))
  }

  con
}


# -----------------------------------------------------------------------------
# 内部函数：设置性能参数
# -----------------------------------------------------------------------------

# 设置DuckDB的性能参数（线程数和内存限制）
.set_pragma <- function(con) {
  # 从全局选项获取配置
  threads <- getOption("scatlas.threads", parallel::detectCores())
  memory_limit <- getOption("scatlas.memory_limit", "32GB")

  # 设置DuckDB参数
  DBI::dbExecute(con, sprintf("PRAGMA threads=%d", threads))
  DBI::dbExecute(con, sprintf("PRAGMA memory_limit='%s'", memory_limit))
}


# -----------------------------------------------------------------------------
# S3方法：打印
# -----------------------------------------------------------------------------

#' @export
print.atlas <- function(x, ...) {
  cat("<atlas>\n")
  cat(sprintf("  名称: %s\n", x$name))
  cat(sprintf("  路径: %s\n", x$path))
  cat(sprintf("  模式: %s\n", x$mode))

  # 获取细胞和基因数量
  n_cells <- if (is.null(x$obs_cell_id)) {
    DBI::dbGetQuery(x$con, "SELECT COUNT(*) FROM obs")$COUNT
  } else length(x$obs_cell_id)

  n_genes <- if (is.null(x$var_gene_id)) {
    DBI::dbGetQuery(x$con, "SELECT COUNT(*) FROM var")$COUNT
  } else length(x$var_gene_id)

  cat(sprintf("  细胞数: %s\n", format(n_cells, big.mark = ",")))
  cat(sprintf("  基因数: %s\n", format(n_genes, big.mark = ",")))
  if (isTRUE(x$is_view)) cat("  视图: 是\n")

  invisible(x)
}


# -----------------------------------------------------------------------------
# S3方法：SQL查询
# -----------------------------------------------------------------------------

#' 执行SQL查询
#'
#' @param a atlas对象
#' @param sql 字符型。SQL查询字符串。
#' @return 数据框
#'
#' @export
#'
#' @examples
#' atlas <- atlas("my_data")
#' result <- query(atlas, "SELECT COUNT(*) FROM obs")
query <- function(a, sql, ...) {
  if (!inherits(a, "atlas")) stop("参数必须是atlas对象")
  if (is.null(a$con) || !DBI::dbIsValid(a$con)) stop("连接已关闭")
  DBI::dbGetQuery(a$con, sql)
}


# -----------------------------------------------------------------------------
# S3方法：视图系统（索引操作）
# -----------------------------------------------------------------------------

#' 按索引子集化Atlas
#'
#' 创建基于细胞筛选的新atlas视图。支持整数、切片、字符串或列表索引。
#'
#' @param a atlas对象
#' @param item 整数、切片、字符型或逻辑型索引。
#'
#' @return 带有筛选细胞的新atlas对象（视图）。
#'
#' @export
#'
#' @examples
#' atlas <- atlas("my_data")
#' atlas_subset <- atlas[1:100]  # 前100个细胞
#' atlas_cell <- atlas["cell_123"]  # 特定细胞
#' atlas_cells <- atlas[c("cell_1", "cell_2", "cell_3")]  # 多个细胞
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


# -----------------------------------------------------------------------------
# 关闭连接
# -----------------------------------------------------------------------------

#' 关闭Atlas连接
#'
#' 关闭数据库连接。完成后必须调用以释放文件锁。
#'
#' @param a atlas对象
#' @return NULL
#'
#' @export
atlas_close <- function(a) {
  if (!inherits(a, "atlas")) return(invisible(NULL))
  if (is.null(a$con) || !DBI::dbIsValid(a$con)) return(invisible(NULL))
  DBI::dbDisconnect(a$con)
  a$con <- NULL
  invisible(NULL)
}


# -----------------------------------------------------------------------------
# 检查连接是否有效
# -----------------------------------------------------------------------------

#' 检查Atlas连接是否有效
#'
#' @param a atlas对象
#' @return 逻辑型。如果连接有效则返回TRUE。
#'
#' @export
atlas_is_valid <- function(a) {
  if (!inherits(a, "atlas")) FALSE
  else !is.null(a$con) && DBI::dbIsValid(a$con)
}


# -----------------------------------------------------------------------------
# 内部函数：创建新视图
# -----------------------------------------------------------------------------

# 创建新的视图对象（不可变筛选）
.new_view <- function(a, obs_cell_id = NULL, var_gene_id = NULL) {
  # 与现有筛选取交集
  if (is.null(obs_cell_id)) {
    obs_cell_id <- a$obs_cell_id
  } else if (!is.null(a$obs_cell_id)) {
    obs_cell_id <- intersect(a$obs_cell_id, obs_cell_id)
  }

  if (is.null(var_gene_id)) {
    var_gene_id <- a$var_gene_id
  } else if (!is.null(a$var_gene_id)) {
    var_gene_id <- intersect(a$var_gene_id, var_gene_id)
  }

  obj <- list(
    name = a$name, path = a$path, mode = a$mode, con = a$con,
    obs_cell_id = obs_cell_id, var_gene_id = var_gene_id, is_view = TRUE
  )
  class(obj) <- "atlas"
  obj
}


# -----------------------------------------------------------------------------
# S3方法：分批查询
# -----------------------------------------------------------------------------

# 内部函数：构建批次数据
.build_batch <- function(a, cell_ids_batch, gene_id, con) {
  # 根据给定的cell_ids构建批次数据（返回稀疏矩阵和obs）

  n_cells <- length(cell_ids_batch)

  if (n_cells == 0) {
    return(list(
      X = Matrix::sparseMatrix(i = integer(0), p = c(0L), dims = c(0, length(gene_id))),
      obs = data.frame(),
      cell_id = character(0)
    ))
  }

  # 获取该批次的obs
  ids_str <- paste0("'", cell_ids_batch, "'", collapse = ",")
  obs_df <- DBI::dbGetQuery(con,
    sprintf("SELECT * FROM obs WHERE cell_id IN (%s)", ids_str))

  # 确保按cell_ids_batch的顺序排列
  obs_df <- obs_df[match(cell_ids_batch, obs_df$cell_id), , drop = FALSE]

  sub_obs <- obs_df[, -1, drop = FALSE]
  cell_id <- sub_obs$cell_id

  # 获取indptr（需要按id排序）
  indptr_df <- DBI::dbGetQuery(con,
    sprintf("SELECT id, indptr FROM X_CSR_indptr WHERE cell_id IN (%s) ORDER BY id", ids_str))
  indptr <- indptr_df$indptr[match(obs_df$id, indptr_df$id)]

  # 获取数据
  n_data <- tail(indptr, 1)
  if (n_data > 0) {
    # indptr是累积值，第一个细胞的indptr应该是0（经过归一化后）
    # 但我们需要找到数据行的起始位置
    # 由于indptr表存储的是累积值，且已经加上全局偏移，
    # 我们需要用cell_index来筛选数据（cell_index是全局的cell id）

    # 获取该批次所有cell的全局id
    batch_global_ids <- as.integer(obs_df$id)
    min_global_id <- min(batch_global_ids)
    max_global_id <- max(batch_global_ids)

    # 使用cell_index来筛选属于该批次的数据
    # cell_index = 0 对应全局 cell id 0，cell_index = 1000 对应全局 cell id 1000
    data_df <- DBI::dbGetQuery(con,
      sprintf("SELECT cell_index, indices, data FROM X_CSR_data
               WHERE cell_index >= %d AND cell_index <= %d
               ORDER BY cell_index, id",
              min_global_id, max_global_id))
    indices <- as.integer(data_df$indices)
    data_vals <- data_df$data

    # 将全局cell_index映射到批次内本地索引（0到n_cells-1）
    # 创建一个从全局id到本地索引的映射
    global_to_local <- setNames(seq_len(n_cells) - 1, as.character(batch_global_ids))

    # 直接用全局cell_index查找本地索引
    cell_idx <- global_to_local[as.character(data_df$cell_index)]

    # 转换为整数（如果有NA，警告）
    if (any(is.na(cell_idx))) {
      warning(sprintf("有%d个cell_index无法映射到本地索引", sum(is.na(cell_idx))))
    }
    cell_idx <- as.integer(cell_idx)
  } else {
    indices <- integer(0)
    data_vals <- numeric(0)
    cell_idx <- integer(0)
  }

  # 构建稀疏矩阵（使用i, j, x形式）
  X <- Matrix::sparseMatrix(
    i = cell_idx,
    j = indices,
    x = data_vals,
    dims = c(n_cells, length(gene_id)),
    index1 = FALSE
  )

  list(X = X, obs = sub_obs, cell_id = cell_id)
}


#' 分批查询
#'
#' 分批迭代遍历atlas，返回AnnData对象。
#'
#' @param a atlas对象
#' @param mode 字符型。"order"=顺序，"random_replace"=有放回随机（无限循环），
#'             "random_no_replace"=不放回随机（每个样本只出现一次）。
#' @param batch_size 整数。每批次的细胞数。
#' @param drop_last 逻辑型。如果为TRUE，丢弃最后一个不完整批次（仅对order/random_no_replace有效）。
#' @param callback 函数(i, adata)。每个批次调用的回调函数。
#'                 如果为NULL，返回所有AnnData对象的列表。
#' @param verbose 逻辑型。是否打印进度信息。
#'
#' @return 如果callback为NULL，返回AnnData对象列表；否则返回invisible。
#'
#' @export
#'
#' @examples
#' atlas <- atlas("my_data")
#'
#' # 顺序模式
#' query_minibatch(atlas, batch_size = 100, mode = "order")
#'
#' # 有放回随机模式（无限循环，需要手动break）
#' query_minibatch(atlas, batch_size = 100, mode = "random_replace", callback = function(i, adata) {
#'   if (i >= 10) stop("stop")  # 手动停止
#' })
#'
#' # 不放回随机模式
#' query_minibatch(atlas, batch_size = 100, mode = "random_no_replace")
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
    all_cell_id_map <- all_cells_df$id  # 用于映射
  } else {
    # 对于视图，需要保持顺序
    all_cells_df <- DBI::dbGetQuery(con,
      sprintf("SELECT id, cell_id FROM obs WHERE cell_id IN (%s) ORDER BY id",
              paste0("'", a$obs_cell_id, "'", collapse = ",")))
    all_cell_ids <- a$obs_cell_id  # 使用保存的顺序
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
    # ========== 顺序模式 ==========
    num_batches <- if (drop_last) total_num %/% batch_size else ceiling(total_num / batch_size)

    for (i in seq_len(num_batches)) {
      offset <- (i - 1) * batch_size
      cur_size <- if (!drop_last && i == num_batches && total_num %% batch_size != 0) {
        total_num %% batch_size
      } else batch_size

      if (verbose) cat(sprintf("\n--- 批次 %d (%d 个细胞) ---\n", i, cur_size))

      batch_cell_ids <- all_cell_ids[(offset + 1):(offset + cur_size)]
      batch_data <- .build_batch(a, batch_cell_ids, gene_id, con)

      adata <- list(X = batch_data$X, obs = batch_data$obs, var = sub_var,
                    obs_names = batch_data$cell_id, var_names = gene_id, shape = dim(batch_data$X))
      class(adata) <- "SimpleAnnData"

      batch_num <- batch_num + 1
      if (is.null(callback)) results[[batch_num]] <- adata else callback(batch_num, adata)
      if (verbose) cat(sprintf("  批次 %d: %d x %d 矩阵\n", batch_num, nrow(adata$X), ncol(adata$X)))
    }

  } else if (mode == "random_no_replace") {
    # ========== 不放回随机模式 ==========
    # 随机打乱所有索引
    shuffled_indices <- sample(seq_len(total_num))

    # 计算批次数
    num_batches <- if (drop_last) total_num %/% batch_size else ceiling(total_num / batch_size)

    for (i in seq_len(num_batches)) {
      start_idx <- (i - 1) * batch_size + 1
      end_idx <- min(i * batch_size, total_num)

      if (start_idx > total_num) break

      cur_size <- if (!drop_last && i == num_batches && total_num %% batch_size != 0) {
        total_num %% batch_size
      } else {
        end_idx - start_idx + 1
      }

      batch_indices <- shuffled_indices[start_idx:(start_idx + cur_size - 1)]
      batch_cell_ids <- all_cell_ids[batch_indices]

      if (verbose) cat(sprintf("\n--- 批次 %d (%d 个细胞, random_no_replace) ---\n", i, cur_size))

      batch_data <- .build_batch(a, batch_cell_ids, gene_id, con)

      adata <- list(X = batch_data$X, obs = batch_data$obs, var = sub_var,
                    obs_names = batch_data$cell_id, var_names = gene_id, shape = dim(batch_data$X))
      class(adata) <- "SimpleAnnData"

      batch_num <- batch_num + 1
      if (is.null(callback)) results[[batch_num]] <- adata else callback(batch_num, adata)
      if (verbose) cat(sprintf("  批次 %d: %d x %d 矩阵\n", batch_num, nrow(adata$X), ncol(adata$X)))
    }

  } else if (mode == "random_replace") {
    # ========== 有放回随机模式（无限循环） ==========
    i <- 0
    while (TRUE) {
      # 随机选择batch_size个索引（有放回，确保批次内不重复）
      batch_indices <- sample(seq_len(total_num), size = batch_size, replace = FALSE)
      batch_cell_ids <- all_cell_ids[batch_indices]

      i <- i + 1
      if (verbose) cat(sprintf("\n--- 批次 %d (%d 个细胞, random_replace) ---\n", i, batch_size))

      batch_data <- .build_batch(a, batch_cell_ids, gene_id, con)

      adata <- list(X = batch_data$X, obs = batch_data$obs, var = sub_var,
                    obs_names = batch_data$cell_id, var_names = gene_id, shape = dim(batch_data$X))
      class(adata) <- "SimpleAnnData"

      batch_num <- batch_num + 1
      if (is.null(callback)) results[[batch_num]] <- adata else callback(batch_num, adata)
      if (verbose) cat(sprintf("  批次 %d: %d x %d 矩阵\n", batch_num, nrow(adata$X), ncol(adata$X)))

      # 用户可以通过callback中抛出错误来停止循环
    }
  }

  if (verbose) cat(sprintf("\n[query_minibatch] 总批次数: %d\n", batch_num))
  if (is.null(callback)) invisible(results) else invisible(NULL)
}


# -----------------------------------------------------------------------------
# S3方法：获取数据（obs/var/X）
# -----------------------------------------------------------------------------

#' 获取细胞元数据
#'
#' @param a atlas对象
#' @param columns 字符型向量。要返回的列名，NULL表示所有列。
#' @param filtered 逻辑型。是否应用视图筛选。
#' @return 数据框
#'
#' @export
obs <- function(a, columns = NULL, filtered = TRUE, ...) {
  if (!inherits(a, "atlas")) stop("参数必须是atlas对象")
  cols <- if (is.null(columns)) "*" else paste0('"', columns, '"', collapse = ", ")
  sql <- sprintf("SELECT %s FROM obs", cols)
  if (filtered && !is.null(a$obs_cell_id)) {
    ids <- paste0("'", a$obs_cell_id, "'", collapse = ",")
    sql <- sprintf("%s WHERE cell_id IN (%s)", sql, ids)
  }
  DBI::dbGetQuery(a$con, sql)
}


#' 获取基因元数据
#'
#' @param a atlas对象
#' @param columns 字符型向量。要返回的列名，NULL表示所有列。
#' @param filtered 逻辑型。是否应用视图筛选。
#' @return 数据框
#'
#' @export
var <- function(a, columns = NULL, filtered = TRUE, ...) {
  if (!inherits(a, "atlas")) stop("参数必须是atlas对象")
  cols <- if (is.null(columns)) "*" else paste0('"', columns, '"', collapse = ", ")
  sql <- sprintf("SELECT %s FROM var", cols)
  if (filtered && !is.null(a$var_gene_id)) {
    ids <- paste0("'", a$var_gene_id, "'", collapse = ",")
    sql <- sprintf("%s WHERE gene_id IN (%s)", sql, ids)
  }
  DBI::dbGetQuery(a$con, sql)
}


#' 获取表达矩阵
#'
#' @param a atlas对象
#' @param field 字符型。要检索的数据字段（默认："data"）。
#' @param sparse 逻辑型。是否返回稀疏矩阵（默认：TRUE）。
#' @return dgCMatrix或矩阵
#'
#' @export
X <- function(a, field = "data", sparse = TRUE, ...) {
  if (!inherits(a, "atlas")) stop("参数必须是atlas对象")
  con <- a$con

  # 获取总基因数（从var表）
  n_genes_result <- DBI::dbGetQuery(con, "SELECT COUNT(*) as n FROM var")
  n_genes <- as.integer(n_genes_result$n[1])

  # 获取细胞数
  n_cells <- DBI::dbGetQuery(con, "SELECT COUNT(*) as n FROM obs")$n

  # 获取indptr
  if (is.null(a$obs_cell_id)) {
    indptr_df <- DBI::dbGetQuery(con, "SELECT id, indptr FROM X_CSR_indptr ORDER BY id")
  } else {
    ids <- paste0("'", a$obs_cell_id, "'", collapse = ",")
    indptr_df <- DBI::dbGetQuery(con,
      sprintf("SELECT id, indptr FROM X_CSR_indptr WHERE cell_id IN (%s) ORDER BY id", ids))
  }
  indptr <- indptr_df$indptr
  obs_ids <- indptr_df$id

  n_cells <- length(indptr)
  if (n_cells == 0) {
    if (sparse) {
      return(Matrix::sparseMatrix(i = integer(0), p = c(0L), dims = c(0, n_genes)))
    } else {
      return(matrix(0, nrow = 0, ncol = n_genes))
    }
  }

  # 计算每行的非零元素数和行索引
  # 使用 numeric 类型避免整数溢出
  nonzero_per_row <- as.numeric(diff(as.numeric(indptr)))
  row_indices <- rep(seq_len(n_cells) - 1, nonzero_per_row)

  # 获取数据
  base_sql <- sprintf("SELECT cell_index, indices, %s FROM X_CSR_data", field)
  if (is.null(a$obs_cell_id) && is.null(a$var_gene_id)) {
    data_df <- DBI::dbGetQuery(con, sprintf("%s ORDER BY id", base_sql))
  } else {
    cell_cond <- if (is.null(a$obs_cell_id)) "TRUE" else {
      ids <- paste0("'", a$obs_cell_id, "'", collapse = ",")
      sprintf("cell_index IN (SELECT id FROM obs WHERE cell_id IN (%s))", ids)
    }
    gene_cond <- if (is.null(a$var_gene_id)) "TRUE" else {
      ids <- paste0("'", a$var_gene_id, "'", collapse = ",")
      sprintf("indices + 1 IN (SELECT id FROM var WHERE gene_id IN (%s))", ids)
    }
    data_df <- DBI::dbGetQuery(con, sprintf("%s WHERE %s AND %s ORDER BY id",
                                            base_sql, cell_cond, gene_cond))

    # 如果有var筛选，需要重新计算n_genes
    if (!is.null(a$var_gene_id)) {
      n_genes <- length(a$var_gene_id)
    }
  }

  # 使用i, j, x形式构建稀疏矩阵
  # i: 行索引 (0-based)
  # j: 列索引 (0-based)
  # x: 值
  mat <- Matrix::sparseMatrix(
    i = as.integer(data_df$cell_index),
    j = as.integer(data_df$indices),
    x = data_df[[field]],
    dims = c(n_cells, n_genes),
    index1 = FALSE
  )

  if (sparse) mat else as.matrix(mat)
}


# -----------------------------------------------------------------------------
# 工具函数
# -----------------------------------------------------------------------------

#' 检查数据库是否存在
#'
#' @param name 字符型。数据库名称。
#' @param path 字符型。数据库目录路径。
#' @return 逻辑型
#'
#' @export
atlas_exists <- function(name, path = ".") {
  file.exists(file.path(path, paste0(name, ".sasql")))
}


#' 删除数据库
#'
#' @param name 字符型。数据库名称。
#' @param path 字符型。数据库目录路径。
#' @return NULL
#'
#' @export
delete_atlas <- function(name, path = ".") {
  db_file <- file.path(path, paste0(name, ".sasql"))
  if (file.exists(db_file)) {
    file.remove(db_file)
    message(sprintf("[atlas] 已删除: %s", db_file))
  } else {
    warning(sprintf("未找到: %s", db_file))
  }
  invisible(NULL)
}


# -----------------------------------------------------------------------------
# 数据库优化函数
# -----------------------------------------------------------------------------

#' 优化数据库设置
#'
#' 优化DuckDB配置以提高查询性能。
#'
#' @param a atlas对象
#' @param threads 整数。使用的CPU线程数，NULL使用所有可用核心。
#' @param memory_limit 字符型。内存限制，如"48GB"，NULL使用默认设置。
#' @return invisible(NULL)
#'
#' @export
atlas_optimize_settings <- function(a, threads = NULL, memory_limit = NULL) {
  if (!inherits(a, "atlas")) stop("参数必须是atlas对象")
  if (is.null(a$con) || !DBI::dbIsValid(a$con)) stop("连接已关闭")

  con <- a$con

  # 设置线程数
  if (!is.null(threads)) {
    DBI::dbExecute(con, sprintf("PRAGMA threads=%d", threads))
    message(sprintf("[atlas] 设置线程数: %d", threads))
  } else {
    n_threads <- parallel::detectCores()
    DBI::dbExecute(con, sprintf("PRAGMA threads=%d", n_threads))
    message(sprintf("[atlas] 自动设置线程数: %d", n_threads))
  }

  # 设置内存限制
  if (!is.null(memory_limit)) {
    DBI::dbExecute(con, sprintf("PRAGMA memory_limit='%s'", memory_limit))
    message(sprintf("[atlas] 设置内存限制: %s", memory_limit))
  }

  # 启用进度条
  DBI::dbExecute(con, "PRAGMA enable_progress_bar=true")

  message("[atlas] 数据库配置优化完成")
  invisible(NULL)
}


#' 创建数据库索引
#'
#' 为所有表创建索引以提高查询性能。
#'
#' @param a atlas对象
#' @param phase 整数。要创建的索引阶段(1-4)，NULL表示创建所有阶段。
#' @return invisible(NULL)
#'
#' @export
atlas_create_indexes <- function(a, phase = NULL) {
  if (!inherits(a, "atlas")) stop("参数必须是atlas对象")
  if (is.null(a$con) || !DBI::dbIsValid(a$con)) stop("连接已关闭")

  con <- a$con

  # 分阶段索引（仅对存在的表创建索引）
  indexes <- list(
    `1` = list(
      name = "第一阶段：核心索引",
      sql = c(
        "CREATE INDEX IF NOT EXISTS idx_obs_cell_id ON obs(cell_id)",
        "CREATE INDEX IF NOT EXISTS idx_var_gene_id ON var(gene_id)",
        "CREATE INDEX IF NOT EXISTS idx_csr_indptr_cell ON X_CSR_indptr(cell_id)",
        "CREATE INDEX IF NOT EXISTS idx_csr_data_indices ON X_CSR_data(indices)"
      )
    ),
    `2` = list(
      name = "第二阶段：复合索引",
      sql = c(
        "CREATE INDEX IF NOT EXISTS idx_obs_cell_id_id ON obs(cell_id, id)",
        "CREATE INDEX IF NOT EXISTS idx_csr_indptr_combo ON X_CSR_indptr(cell_id, indptr)",
        "CREATE INDEX IF NOT EXISTS idx_csr_data_combo ON X_CSR_data(indices, data)"
      )
    ),
    `3` = list(
      name = "第三阶段：辅助索引",
      sql = c(
        "CREATE INDEX IF NOT EXISTS idx_obs_id ON obs(id)",
        "CREATE INDEX IF NOT EXISTS idx_var_id ON var(id)",
        "CREATE INDEX IF NOT EXISTS idx_csr_indptr_id ON X_CSR_indptr(id)",
        "CREATE INDEX IF NOT EXISTS idx_csr_data_id ON X_CSR_data(id)"
      )
    ),
    `4` = list(
      name = "第四阶段：CSR专用索引",
      sql = c(
        "CREATE INDEX IF NOT EXISTS idx_csr_data_covering ON X_CSR_data(indices, id, data)",
        "CREATE INDEX IF NOT EXISTS idx_csr_indptr_range ON X_CSR_indptr(indptr, cell_id)"
      )
    )
  )

  phases_to_run <- if (is.null(phase)) seq_len(length(indexes)) else phase

  for (p in phases_to_run) {
    if (p %in% names(indexes)) {
      phase_info <- indexes[[p]]
      message(sprintf("[atlas] %s...", phase_info$name))

      for (sql in phase_info$sql) {
        tryCatch({
          DBI::dbExecute(con, sql)
        }, error = function(e) {
          # 静默忽略表不存在的错误
        })
      }
    }
  }

  message("[atlas] 索引创建完成")
  invisible(NULL)
}


#' 表维护操作
#'
#' 更新表统计信息以优化查询计划。
#'
#' @param a atlas对象
#' @return invisible(NULL)
#'
#' @export
atlas_maintain_tables <- function(a) {
  if (!inherits(a, "atlas")) stop("参数必须是atlas对象")
  if (is.null(a$con) || !DBI::dbIsValid(a$con)) stop("连接已关闭")

  con <- a$con

  # 更新统计信息（仅对存在的表）
  tryCatch({
    DBI::dbExecute(con, "ANALYZE obs")
    DBI::dbExecute(con, "ANALYZE var")
    DBI::dbExecute(con, "ANALYZE X_CSR_indptr")
    DBI::dbExecute(con, "ANALYZE X_CSR_data")
    message("[atlas] 统计信息更新完成")
  }, error = function(e) {
    warning(sprintf("[atlas] 表维护时出错: %s", e$message))
  })

  invisible(NULL)
}


#' 综合数据库优化
#'
#' 执行完整的数据库优化，包括配置优化、索引创建和表维护。
#'
#' @param a atlas对象
#' @param threads 整数。使用的CPU线程数。
#' @param memory_limit 字符型。内存限制，如"48GB"。
#' @param create_indexes 逻辑型。是否创建索引。
#' @return invisible(NULL)
#'
#' @export
atlas_optimize <- function(a, threads = NULL, memory_limit = NULL, create_indexes = TRUE) {
  if (!inherits(a, "atlas")) stop("参数必须是atlas对象")
  if (is.null(a$con) || !DBI::dbIsValid(a$con)) stop("连接已关闭")

  message("[atlas] 开始执行数据库综合优化...")

  # 1. 数据库配置优化
  atlas_optimize_settings(a, threads = threads, memory_limit = memory_limit)

  # 2. 创建索引
  if (create_indexes) {
    atlas_create_indexes(a)
  }

  # 3. 表维护
  atlas_maintain_tables(a)

  message("[atlas] 数据库综合优化完成!")
  invisible(NULL)
}


# -----------------------------------------------------------------------------
# 上下文管理器
# -----------------------------------------------------------------------------

#' 上下文管理器：自动管理atlas连接
#'
#' 使用with()语法自动关闭连接。
#'
#' @param data atlas对象
#' @param expr 要执行的表达式
#' @return 表达式的返回值
#'
#' @examples
#' \dontrun{
#' # 自动关闭连接
#' result <- with(atlas("my_data"), {
#'   query(a, "SELECT COUNT(*) FROM obs")
#' })
#' }
#'
#' @export
with.atlas <- function(data, expr, ...) {
  on.exit(atlas_close(data), add = TRUE)
  eval.parent(substitute(expr))
}
