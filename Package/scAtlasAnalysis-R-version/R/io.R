# io.R - 输入输出函数
# =============================================================================
#
# 功能：数据的导入和导出
# 支持格式：H5、H5AD、10X H5、RDS、MTX、CSV、TSV
#
# =============================================================================

# Global progress state for tracking - 使用延迟初始化的环境
.scatlas_progress <- local({
  env <- new.env(parent = emptyenv())
  env$label <- NULL
  env$current <- NULL
  env$total <- NULL
  env
})

#' Set progress callback for load operations
#' @param label Progress label
#' @param current Current item
#' @param total Total items
#' @param status Status message
#' @export
set_progress <- function(label, current = NULL, total = NULL, status = NULL) {
  if (!is.null(label)) {
    .scatlas_progress$label <- label
  }

  # Build progress string
  if (!is.null(total) && total > 0) {
    pct <- if (!is.null(current)) round(current / total * 100, 1) else 0
    bar_width <- 30
    filled <- round(bar_width * min(current / total, 1))
    bar <- paste(rep("=", filled), collapse = "")
    empty <- paste(rep("-", bar_width - filled), collapse = "")

    progress_str <- sprintf("\r[%s%s] %s%%", bar, empty, pct)

    if (!is.null(status)) {
      cat(progress_str, " ", status, "\n", sep = "")
    } else {
      cat(progress_str, "\r")
    }
  } else if (!is.null(status)) {
    cat("  ", status, "\n", sep = "")
  }

  .scatlas_progress$current <- current
  .scatlas_progress$total <- total

  # Flush output
  flush.console()
}

#' Reset progress
#' @export
reset_progress <- function() {
  .scatlas_progress$label <- NULL
  .scatlas_progress$current <- NULL
  .scatlas_progress$total <- NULL
  cat("\n")
}

# Detect file format
.detect_format <- function(file) {
  ext <- tools::file_ext(file)
  if (ext %in% c("h5", "h5ad", "hdf5")) "h5"
  else if (ext %in% c("rds", "RDS")) "rds"
  else if (ext %in% c("csv", "CSV")) "csv"
  else if (ext %in% c("tsv", "TSV", "tab", "TAB")) "tsv"
  else if (ext %in% c("mtx", "MTX", "gz")) "mtx"
  else "unknown"
}

# Get current offset for CSR data (for multi-batch loading)
# Returns the number of existing rows in X_CSR_data
.get_csr_offset <- function(con) {
  result <- DBI::dbGetQuery(con, "SELECT COUNT(*) as cnt FROM X_CSR_data")
  as.integer(result$cnt)
}

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

  # Get file size for display
  file_size <- file.info(file)$size
  size_str <- if (!is.na(file_size)) {
    if (file_size > 1e9) sprintf("%.1f GB", file_size / 1e9)
    else if (file_size > 1e6) sprintf("%.1f MB", file_size / 1e6)
    else sprintf("%.1f KB", file_size / 1e3)
  } else ""
  set_progress("load_data", status = sprintf("Size: %s", size_str))

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

  cat("\n")
  set_progress("load_data", status = sprintf("[OK] cells=%s, genes=%s, nnz=%s",
    format(n_obs, big.mark = ","),
    format(n_vars, big.mark = ","),
    format(nnz, big.mark = ",")))

  # Refresh cell/gene IDs - return new atlas object with updated IDs
  obs_r <- DBI::dbGetQuery(a$con, "SELECT cell_id FROM obs")
  var_r <- DBI::dbGetQuery(a$con, "SELECT gene_id FROM var")
  a$obs_cell_id <- if (nrow(obs_r) > 0) obs_r$cell_id else NULL
  a$var_gene_id <- if (nrow(var_r) > 0) var_r$gene_id else NULL

  invisible(a)
}

# Load from H5/H5AD
.load_h5 <- function(a, file) {
  if (!requireNamespace("hdf5r", quietly = TRUE)) {
    stop("Package 'hdf5r' is required for H5 files")
  }

  set_progress("load_data", status = "Reading H5 file...")
  con <- a$con
  h5f <- hdf5r::H5File$new(file, mode = "r")
  on.exit(h5f$close(), add = TRUE)

  .read <- function(path) tryCatch(h5f[[path]][], error = function(e) NULL)

  # 检测标准10X格式 (matrix/...)
  if ("matrix" %in% names(h5f)) {
    set_progress("load_data", status = "10X format (standard) detected")
    .load_10x_h5(con, h5f, prefix = "matrix/")
  # 检测非标准10X格式 (根目录下的barcodes, data, indices, indptr)
  } else if (all(c("barcodes", "data", "indices", "indptr") %in% names(h5f))) {
    set_progress("load_data", status = "10X format (flat) detected")
    .load_10x_h5(con, h5f, prefix = "")
  # 检测带前缀的10X格式 (如 mm10/)
  } else {
    # 查找包含所有必需CSR组件的组
    top_names <- names(h5f)
    for (name in top_names) {
      grp <- h5f[[name]]
      if (inherits(grp, "H5Group")) {
        grp_names <- names(grp)
        if (all(c("barcodes", "data", "indices", "indptr") %in% grp_names)) {
          set_progress("load_data", status = sprintf("10X format (prefixed: %s/) detected", name))
          .load_10x_h5(con, h5f, prefix = paste0(name, "/"))
          return(invisible(a))
        }
      }
    }
  }

  # AnnData format (非10X格式)
  if ("obs" %in% names(h5f)) {
    obs_df <- as.data.frame(.read("/obs"))

    # 处理 h5ad 结构化数组中的 index 列（这是细胞ID）
    if ("index" %in% colnames(obs_df)) {
      # index 列是行索引，提取为 cell_id
      obs_df$cell_id <- as.character(obs_df$index)
      obs_df$index <- NULL  # 删除 index 列
    } else if (!"cell_id" %in% colnames(obs_df)) {
      obs_df$cell_id <- rownames(obs_df) %||% seq_len(nrow(obs_df))
    }

    # 添加 id 列（放在最前面）
    obs_df <- data.frame(id = seq_len(nrow(obs_df)) - 1, obs_df)
    DBI::dbWriteTable(con, "obs", obs_df, overwrite = TRUE, append = FALSE)
    set_progress("load_data", status = sprintf("obs: %s rows, %s cols",
      format(nrow(obs_df), big.mark = ","), ncol(obs_df) - 1))
  }
  if ("var" %in% names(h5f)) {
    var_df <- as.data.frame(.read("/var"))

    # 处理 h5ad 结构化数组中的 index 列（这是基因ID）
    if ("index" %in% colnames(var_df)) {
      var_df$gene_id <- as.character(var_df$index)
      var_df$index <- NULL
    } else if (!"gene_id" %in% colnames(var_df)) {
      var_df$gene_id <- rownames(var_df) %||% seq_len(nrow(var_df))
    }

    # 添加 id 列
    var_df <- data.frame(id = seq_len(nrow(var_df)) - 1, var_df)
    DBI::dbWriteTable(con, "var", var_df, overwrite = TRUE, append = FALSE)
    set_progress("load_data", status = sprintf("var: %s rows, %s cols",
      format(nrow(var_df), big.mark = ","), ncol(var_df) - 1))
  }
  if ("X" %in% names(h5f)) .load_x_from_h5(con, h5f, "/X")
}

# Load from 10X H5
# 支持标准格式 (matrix/*) 和非标准格式 (prefix/* 或 flat)
.load_10x_h5 <- function(con, h5f, prefix = "matrix/") {
  set_progress("load_data", status = sprintf("10X format (%s*) detected, reading metadata...", prefix))

  shape <- h5f[[paste0(prefix, "shape")]][]
  data <- h5f[[paste0(prefix, "data")]][]
  indices <- h5f[[paste0(prefix, "indices")]][]
  indptr <- h5f[[paste0(prefix, "indptr")]][]
  barcodes <- h5f[[paste0(prefix, "barcodes")]][]

  # 基因ID可能在 features/name 或 genes
  # 注意：对于嵌套路径，不能用 names(h5f) 检查，需要用 tryCatch
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
    set_progress("load_data", status = "Using synthetic gene IDs")
    gene_ids <- paste0("gene_", seq_len(shape[1]))
  }

  n_cells <- length(barcodes)
  n_genes <- length(gene_ids)
  nnz <- length(data)

  set_progress("load_data", status = sprintf("cells=%s, genes=%s, nnz=%s",
    format(n_cells, big.mark = ","),
    format(n_genes, big.mark = ","),
    format(nnz, big.mark = ",")))

  # Get current offset for global indexing
  offset <- .get_csr_offset(con)

  # Write obs/var
  set_progress("load_data", status = "Writing obs table...")
  DBI::dbWriteTable(con, "obs",
    data.frame(id = seq_len(n_cells) - 1, cell_id = barcodes),
    overwrite = TRUE, append = FALSE)

  set_progress("load_data", status = "Writing var table...")
  DBI::dbWriteTable(con, "var",
    data.frame(id = seq_len(n_genes) - 1, gene_id = gene_ids),
    overwrite = TRUE, append = FALSE)

  # Write CSR - 先创建表（指定类型）
  set_progress("load_data", status = "Creating CSR indptr table...")

  # 删除旧表
  DBI::dbExecute(con, "DROP TABLE IF EXISTS X_CSR_indptr")
  DBI::dbExecute(con, "DROP TABLE IF EXISTS X_CSR_data")

  # 创建表
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

  # 插入 indptr
  set_progress("load_data", status = "Writing CSR indptr...")
  indptr_df <- data.frame(
    id = seq_len(n_cells) - 1,
    cell_id = barcodes,
    indptr = as.integer(indptr[seq_len(n_cells)] + offset)
  )
  DBI::dbWriteTable(con, "X_CSR_indptr", indptr_df, append = TRUE)

  # 插入 data（分批）
  set_progress("load_data", status = "Writing CSR data...")
  cell_indices <- rep(seq_len(n_cells) - 1, diff(indptr))

  batch_size <- 1000000
  n_batches <- ceiling(nnz / batch_size)

  for (batch_idx in seq_len(n_batches)) {
    start_idx <- (batch_idx - 1) * batch_size + 1
    end_idx <- min(batch_idx * batch_size, nnz)

    data_df <- data.frame(
      id = (start_idx:end_idx) - 1,
      indices = indices[start_idx:end_idx],
      data = as.numeric(data[start_idx:end_idx]),
      cell_index = cell_indices[start_idx:end_idx]
    )

    DBI::dbAppendTable(con, "X_CSR_data", data_df)

    if (batch_idx %% 50 == 0 || batch_idx == n_batches) {
      set_progress("load_data", status = sprintf("Writing CSR data: batch %d/%d",
        batch_idx, n_batches))
    }
  }

  set_progress("load_data", status = sprintf("Written: %s entries (offset: %d)",
    format(nnz, big.mark = ","), offset))
}

# Load X from H5 (AnnData format)
.load_x_from_h5 <- function(con, h5f, path) {
  set_progress("load_data", status = "Reading X matrix...")

  X_item <- h5f[[path]]

  # 检查是否是 CSR Group (data, indices, indptr)
  if (inherits(X_item, "H5Group") || (is.list(X_item) && all(c("data", "indices", "indptr") %in% names(X_item)))) {
    # CSR 格式：直接读取各个组件
    set_progress("load_data", status = "Reading CSR data...")
    indptr <- X_item[["indptr"]][]
    indices <- X_item[["indices"]][]
    data_vals <- X_item[["data"]][]
    set_progress("load_data", status = sprintf("CSR: %s indptr, %s indices, %s data",
      format(length(indptr), big.mark = ","),
      format(length(indices), big.mark = ","),
      format(length(data_vals), big.mark = ",")))
  } else {
    # 尝试作为数据集读取并转换为 dgCMatrix
    set_progress("load_data", status = "Converting X to CSR...")
    X_data <- X_item[]
    if (is.null(X_data)) return()

    if (is.list(X_data) && all(c("data", "indices", "indptr") %in% names(X_data))) {
      indptr <- X_data$indptr; indices <- X_data$indices; data_vals <- X_data$data
    } else {
      X_mat <- as(X_data, "dgCMatrix")
      indptr <- X_mat@p; indices <- X_mat@i; data_vals <- X_mat@x
    }
  }

  # Get current offset for global indexing
  offset <- .get_csr_offset(con)

  cell_ids <- DBI::dbGetQuery(con, "SELECT cell_id FROM obs ORDER BY id")$cell_id
  n_cells <- length(indptr) - 1
  nnz <- length(data_vals)

  set_progress("load_data", status = sprintf("X: %s x %s, nnz=%s (offset: %d)",
    format(n_cells, big.mark = ","),
    format(max(indices) + 1, big.mark = ","),
    format(nnz, big.mark = ","),
    offset))

  # Normalize indptr: ensure it starts from 0
  indptr_offset <- if (length(indptr) > 1 && indptr[1] != 0) indptr[1] else 0
  indptr_normalized <- indptr - indptr_offset

  # Use indptr[1] to indptr[n_cells] (skip last element which is total)
  indptr_vals <- as.integer(indptr_normalized[seq_len(n_cells)] + offset)
  cell_indices <- rep(seq_len(n_cells) - 1, diff(indptr_normalized)) + offset

  set_progress("load_data", status = "Writing X_CSR_indptr...")

  # 先删除旧表
  DBI::dbExecute(con, "DROP TABLE IF EXISTS X_CSR_indptr")

  # 创建表（指定类型）
  DBI::dbExecute(con, "CREATE TABLE X_CSR_indptr (
    id BIGINT PRIMARY KEY,
    cell_id VARCHAR,
    indptr BIGINT
  )")

  # 插入数据
  indptr_df <- data.frame(
    id = seq_len(n_cells) - 1,
    cell_id = cell_ids,
    indptr = indptr_vals
  )
  DBI::dbWriteTable(con, "X_CSR_indptr", indptr_df, append = TRUE)

  set_progress("load_data", status = "Writing X_CSR_data...")

  # 先删除旧表
  DBI::dbExecute(con, "DROP TABLE IF EXISTS X_CSR_data")

  # 创建表（指定类型，与 Python 版本一致）
  DBI::dbExecute(con, "CREATE TABLE X_CSR_data (
    id BIGINT PRIMARY KEY,
    indices USMALLINT,
    data FLOAT,
    cell_index BIGINT
  )")

  # 插入数据（分批处理以避免内存问题）
  batch_size <- 1000000
  n_batches <- ceiling(nnz / batch_size)

  for (batch_idx in seq_len(n_batches)) {
    start_idx <- (batch_idx - 1) * batch_size + 1
    end_idx <- min(batch_idx * batch_size, nnz)

    data_df <- data.frame(
      id = (start_idx:end_idx) - 1,
      indices = indices[start_idx:end_idx],
      data = as.numeric(data_vals[start_idx:end_idx]),
      cell_index = cell_indices[start_idx:end_idx]
    )

    # 使用 dbAppendTable 插入数据
    DBI::dbAppendTable(con, "X_CSR_data", data_df)

    if (batch_idx %% 10 == 0 || batch_idx == n_batches) {
      set_progress("load_data", status = sprintf("Writing X_CSR_data: batch %d/%d",
        batch_idx, n_batches))
    }
  }
}

# Load from RDS
.load_rds <- function(a, file) {
  set_progress("load_data", status = sprintf("Reading: %s", basename(file)))
  obj <- readRDS(file)
  con <- a$con

  if (is.list(obj) && !is.data.frame(obj)) {
    # AnnData-like object
    if (!is.null(obj$obs)) {
      obs_df <- as.data.frame(obj$obs)
      if (!"cell_id" %in% colnames(obs_df)) obs_df$cell_id <- rownames(obj$obs) %||% seq_len(nrow(obj$obs))
      obs_df <- data.frame(id = seq_len(nrow(obs_df)) - 1, obs_df)
      DBI::dbWriteTable(con, "obs", obs_df, overwrite = TRUE, append = FALSE)
      set_progress("load_data", status = sprintf("obs: %s rows", format(nrow(obs_df), big.mark = ",")))
    }
    if (!is.null(obj$var)) {
      var_df <- as.data.frame(obj$var)
      if (!"gene_id" %in% colnames(var_df)) var_df$gene_id <- rownames(obj$var) %||% seq_len(nrow(var_df))
      var_df <- data.frame(id = seq_len(nrow(var_df)) - 1, var_df)
      DBI::dbWriteTable(con, "var", var_df, overwrite = TRUE, append = FALSE)
      set_progress("load_data", status = sprintf("var: %s rows", format(nrow(var_df), big.mark = ",")))
    }
    if (!is.null(obj$X)) .load_x(con, obj$X)
  } else if (is.matrix(obj) || inherits(obj, "dgCMatrix")) {
    # Just a matrix
    n_cells <- nrow(obj); n_genes <- ncol(obj)
    DBI::dbWriteTable(con, "obs",
      data.frame(id = seq_len(n_cells) - 1, cell_id = paste0("cell_", seq_len(n_cells))),
      overwrite = TRUE, append = FALSE)
    DBI::dbWriteTable(con, "var",
      data.frame(id = seq_len(n_genes) - 1, gene_id = paste0("gene_", seq_len(n_genes))),
      overwrite = TRUE, append = FALSE)
    .load_x(con, obj)
    set_progress("load_data", status = sprintf("%s x %s", format(n_cells, big.mark = ","), format(n_genes, big.mark = ",")))
  }
}

# Load matrix to CSR
.load_x <- function(con, X_mat) {
  set_progress("load_data", status = "Processing X matrix...")

  if (!inherits(X_mat, "dgCMatrix")) X_mat <- as(X_mat, "dgCMatrix")
  indptr <- X_mat@p; indices <- X_mat@i; data_vals <- X_mat@x

  # Get current offset for global indexing
  offset <- .get_csr_offset(con)

  cell_ids <- DBI::dbGetQuery(con, "SELECT cell_id FROM obs ORDER BY id")$cell_id
  n_cells <- length(indptr) - 1
  nnz <- length(data_vals)

  set_progress("load_data", status = sprintf("X: %s x %s, nnz=%s (offset: %d)",
    format(n_cells, big.mark = ","),
    format(max(indices) + 1, big.mark = ","),
    format(nnz, big.mark = ","),
    offset))

  # Normalize indptr: ensure it starts from 0
  indptr_offset <- if (length(indptr) > 1 && indptr[1] != 0) indptr[1] else 0
  indptr_normalized <- indptr - indptr_offset

  # Use indptr[1] to indptr[n_cells] (skip last element which is total)
  indptr_vals <- as.integer(indptr_normalized[seq_len(n_cells)] + offset)
  cell_indices <- rep(seq_len(n_cells) - 1, diff(indptr_normalized)) + offset

  # 创建表（指定类型）
  set_progress("load_data", status = "Creating CSR tables...")

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
  set_progress("load_data", status = "Writing X_CSR_indptr...")
  indptr_df <- data.frame(
    id = seq_len(n_cells) - 1,
    cell_id = cell_ids,
    indptr = indptr_vals
  )
  DBI::dbWriteTable(con, "X_CSR_indptr", indptr_df, append = TRUE)

  # 写入 data（分批）
  set_progress("load_data", status = "Writing X_CSR_data...")

  batch_size <- 1000000
  n_batches <- ceiling(nnz / batch_size)

  for (batch_idx in seq_len(n_batches)) {
    start_idx <- (batch_idx - 1) * batch_size + 1
    end_idx <- min(batch_idx * batch_size, nnz)

    data_df <- data.frame(
      id = (start_idx:end_idx) - 1,
      indices = indices[start_idx:end_idx],
      data = as.numeric(data_vals[start_idx:end_idx]),
      cell_index = cell_indices[start_idx:end_idx]
    )

    DBI::dbAppendTable(con, "X_CSR_data", data_df)

    if (batch_idx %% 50 == 0 || batch_idx == n_batches) {
      set_progress("load_data", status = sprintf("Writing X_CSR_data: batch %d/%d",
        batch_idx, n_batches))
    }
  }
}

# Load from CSV/TSV
.load_csv <- function(a, file, delim) {
  set_progress("load_data", status = sprintf("Reading: %s", basename(file)))
  con <- a$con

  X_mat <- tryCatch(as.matrix(read.csv(file, header = TRUE, row.names = 1)),
    error = function(e) NULL)
  if (is.null(X_mat)) {
    X_mat <- tryCatch(as.matrix(read.delim(file, header = TRUE, row.names = 1)),
      error = function(e) NULL)
  }
  if (is.null(X_mat)) stop("Could not read matrix from CSV/TSV")

  n_cells <- nrow(X_mat); n_genes <- ncol(X_mat)
  obs_df <- data.frame(id = seq_len(n_cells) - 1, cell_id = rownames(X_mat))
  var_df <- data.frame(id = seq_len(n_genes) - 1, gene_id = colnames(X_mat))

  DBI::dbWriteTable(con, "obs", obs_df, overwrite = TRUE, append = FALSE)
  DBI::dbWriteTable(con, "var", var_df, overwrite = TRUE, append = FALSE)
  .load_x(con, X_mat)

  set_progress("load_data", status = sprintf("%s x %s", format(n_cells, big.mark = ","), format(n_genes, big.mark = ",")))
}

# Load from 10X MTX
.load_mtx <- function(a, file) {
  set_progress("load_data", status = sprintf("Reading: %s", basename(file)))
  con <- a$con
  mtx_dir <- dirname(file)

  # Find barcodes/genes files
  barcodes_file <- list.files(mtx_dir, pattern = "barcodes", full.names = TRUE, ignore.case = TRUE)[1]
  genes_file <- list.files(mtx_dir, pattern = "genes|features", full.names = TRUE, ignore.case = TRUE)[1]

  # Read matrix
  X_mat <- tryCatch({
    if (grepl("\\.gz$", file)) {
      gz_con <- gzfile(file, "rt"); on.exit(gz_con, add = TRUE)
      Matrix::readMM(gz_con)
    } else Matrix::readMM(file)
  }, error = function(e) stop(sprintf("Failed to read MTX: %s", e$message)))

  n_cells <- nrow(X_mat); n_genes <- ncol(X_mat)
  set_progress("load_data", status = sprintf("%s x %s", format(n_cells, big.mark = ","), format(n_genes, big.mark = ",")))

  # Cell/gene IDs
  cell_ids <- paste0("cell_", seq_len(n_cells))
  if (!is.null(barcodes_file) && file.exists(barcodes_file)) {
    barcodes <- tryCatch(readLines(barcodes_file), error = function(e) NULL)
    if (!is.null(barcodes)) cell_ids <- barcodes
  }

  gene_ids <- paste0("gene_", seq_len(n_genes))
  if (!is.null(genes_file) && file.exists(genes_file)) {
    genes <- tryCatch(readLines(genes_file), error = function(e) NULL)
    if (!is.null(genes)) gene_ids <- sub("\t.*", "", genes)
  }

  DBI::dbWriteTable(con, "obs",
    data.frame(id = seq_along(cell_ids) - 1, cell_id = cell_ids),
    overwrite = TRUE, append = FALSE)
  DBI::dbWriteTable(con, "var",
    data.frame(id = seq_along(gene_ids) - 1, gene_id = gene_ids),
    overwrite = TRUE, append = FALSE)

  .load_x(con, X_mat)
}

# Alias for backward compatibility
load_h5ad <- function(a, file, ...) load_data(a, file, ...)

#' Save to H5AD
#' @export
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
    message(sprintf("  obs: %s", format(nrow(obs_df), big.mark = ",")))
  }

  # Write var
  if (write_var) {
    var_df <- var(a)
    var_df[] <- lapply(var_df, function(x) if (is.logical(x)) as.integer(x) else x)
    h5f$create_dataset("/var", robj = as.matrix(var_df))
    message(sprintf("  var: %s", format(nrow(var_df), big.mark = ",")))
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
    message(sprintf("    %s non-zero", format(length(X_data$data), big.mark = ",")))
  }

  h5f$close()
  message("[save_h5ad] Complete")
  invisible(a)
}
