#!/usr/bin/env Rscript
# scAtlas (R) 性能基准测试
# 与 Scanpy/Seurat Benchmark 保持一致
#
# 使用方式:
#   # 单独运行一个 operator
#   Rscript scatlas_benchmark.R --operator log1p --dataset ../../Dataset/20k_PBMC.h5
#
#   # 运行所有 operators
#   Rscript run_all_isolated.R --dataset ../../Dataset/20k_PBMC.h5

library(optparse)
library(data.table)

# 添加 scatlas-R 包路径
if (!'/home/senpeng/zspbenchmark/Package/scAtlasAnalysis' %in% .libPaths()) {
  .libPaths(c('/home/senpeng/zspbenchmark/Package/scAtlasAnalysis', .libPaths()))
}
library(scAtlas)

# ==================== 核心：内存监控 ====================

get_memory_mb <- function() {
  pid <- Sys.getpid()
  status_file <- paste0("/proc/", pid, "/status")
  if (file.exists(status_file)) {
    lines <- readLines(status_file)
    rss_line <- grep("^VmRSS:", lines, value = TRUE)
    if (length(rss_line) > 0) {
      rss_kb <- as.numeric(strsplit(rss_line, "\\s+")[[1]][2])
      return(rss_kb / 1024)
    }
  }
  return(NA)
}

# ==================== 基因统计信息缓存 ====================

.gene_stats_cache <- new.env(parent = emptyenv())

get_gene_stats <- function(a) {
  cache_key <- paste0(a$name, "_", a$path)
  if (exists(cache_key, envir = .gene_stats_cache)) {
    return(get(cache_key, envir = .gene_stats_cache))
  }

  # 使用 SQL 查询计算每个基因的表达细胞数，避免加载整个矩阵
  cat("  计算基因表达统计（首次）...\n")
  result_sql <- DBI::dbGetQuery(a$con, "
    SELECT indices, COUNT(*) as n_cells
    FROM X_CSR_data
    WHERE data > 0
    GROUP BY indices
    ORDER BY n_cells
  ")

  n_genes <- nrow(result_sql)
  sorted_indices <- result_sql$indices

  # 选取不同位置的基因
  mid <- as.integer(n_genes / 2)
  result <- list(
    sorted_idx = sorted_indices,
    genes_1 = sorted_indices[max(1, mid)],
    genes_2 = sorted_indices[max(1, mid):min(n_genes, mid + 1)],
    genes_3 = sorted_indices[max(1, mid + 2):min(n_genes, mid + 4)]
  )
  assign(cache_key, result, envir = .gene_stats_cache)
  return(result)
}

# ==================== 工具函数 ====================

# 检查数据库是否有效（是否有数据）
is_db_valid <- function(db_file) {
  if (!file.exists(db_file)) return(FALSE)
  tryCatch({
    con <- DBI::dbConnect(duckdb::duckdb(db_file), read_only = TRUE)
    on.exit(DBI::dbDisconnect(con), add = TRUE)
    # 检查obs表是否有数据
    n_obs <- DBI::dbGetQuery(con, "SELECT COUNT(*) as n FROM obs")$n
    n_vars <- DBI::dbGetQuery(con, "SELECT COUNT(*) as n FROM var")$n
    # 有效数据库至少应该有 100 个细胞和基因（排除空数据库）
    n_obs > 100 && n_vars > 100
  }, error = function(e) FALSE)
}

load_data_atlas <- function(db_name, db_path, data_file, reuse_db = TRUE) {
  cat("Loading data:", data_file, "\n")

  db_file <- file.path(db_path, paste0(db_name, ".sasql"))
  lock_file <- file.path(db_path, paste0(db_name, ".importing"))

  # 第一步：获取导入权限
  # 等待其他进程完成导入
  wait_count <- 0
  while (file.exists(lock_file)) {
    wait_count <- wait_count + 1
    if (wait_count %% 10 == 0) {
      cat("  Waiting for other import to complete...", wait_count, "s\n")
    }
    Sys.sleep(1)
  }

  # 创建锁文件（表示我正在导入）
  file.create(lock_file)
  on.exit(file.remove(lock_file), add = TRUE)

  # 第二步：检查是否有有效数据库
  db_valid <- is_db_valid(db_file)

  if (file.exists(db_file) && reuse_db && db_valid) {
    cat("  Reusing existing database...\n")
    a <- atlas(db_name, path = db_path, mode = "r+")
  } else {
    # 删除旧的数据库文件
    if (file.exists(db_file)) {
      cat("  Removing old/invalid database...\n")
      file.remove(db_file)
    }

    cat("  Creating new database and importing...\n")

    # 创建数据库
    a <- atlas(db_name, path = db_path, mode = "r+")

    # 导入数据
    load_data(a, data_file)

    # 刷新缓存的 ID（R 赋值传值，需要手动更新）
    obs_r <- DBI::dbGetQuery(a$con, "SELECT cell_id FROM obs")
    var_r <- DBI::dbGetQuery(a$con, "SELECT gene_id FROM var")
    a$obs_cell_id <- if (nrow(obs_r) > 0) obs_r$cell_id else NULL
    a$var_gene_id <- if (nrow(var_r) > 0) var_r$gene_id else NULL

    DBI::dbExecute(a$con, "PRAGMA force_checkpoint")

    cat("  Import completed.\n")
  }

  # 第三步：删除锁文件（导入已完成）
  # on.exit 会在函数结束时自动删除，这里不需要手动删除
  invisible(NULL)

  # 设置 DuckDB 内存限制
  cat("  Setting memory limit to 32GB...\n")
  atlas_optimize_settings(a, memory_limit = "32GB")

  # 获取数据量
  n_obs <- nrow(obs(a))
  n_vars <- nrow(var(a))
  cat("  Data size:", n_obs, "cells x", n_vars, "genes\n")
  return(a)
}

save_result <- function(result, dataset_path, output_dir = NULL, result_file = NULL) {
  dataset_name <- tools::file_path_sans_ext(basename(dataset_path))
  if (is.null(result_file)) {
    result_file <- paste0("scatlas_results_", dataset_name, ".csv")
  }
  if (!is.null(output_dir)) {
    result_file <- file.path(output_dir, result_file)
  }
  result$Dataset <- basename(dataset_path)
  df <- as.data.frame(result)
  cols <- c("Operator", "Time_s", "Peak_Memory_MiB", "Dataset")
  df <- df[, cols, drop = FALSE]
  if (!dir.exists(dirname(result_file))) {
    dir.create(dirname(result_file), recursive = TRUE, showWarnings = FALSE)
  }
  if (file.exists(result_file)) {
    fwrite(df, result_file, append = TRUE, col.names = FALSE)
  } else {
    fwrite(df, result_file)
  }
  cat("Result saved:", result_file, "\n")
}

# ==================== Operators (与 Scanpy/Seurat 一致) ====================

filter_cells_min_genes_200 <- function(a) {
  # 只执行 filter_cells，不调用 nrow(obs()) 避免 X() 构建稀疏矩阵失败
  filter_cells(a, min_genes = 200)
  invisible(TRUE)
}

filter_cells_max_genes_6000 <- function(a) {
  filter_cells(a, max_genes = 6000)
  invisible(TRUE)
}

filter_cells_min_counts_500 <- function(a) {
  filter_cells(a, min_counts = 500)
  invisible(TRUE)
}

filter_cells_max_counts_40000 <- function(a) {
  filter_cells(a, max_counts = 40000)
  invisible(TRUE)
}

filter_genes_min_cells_3 <- function(a) {
  filter_genes(a, min_cells = 3)
  invisible(TRUE)
}

filter_genes_max_cells_1000000 <- function(a) {
  filter_genes(a, max_cells = 1000000)
  invisible(TRUE)
}

filter_genes_min_counts_10 <- function(a) {
  filter_genes(a, min_counts = 10)
  invisible(TRUE)
}

filter_genes_max_counts_100000 <- function(a) {
  filter_genes(a, max_counts = 100000)
  invisible(TRUE)
}

log1p_transform <- function(a) {
  log1p(a)
  invisible(TRUE)
}

scale_transform <- function(a) {
  scale(a)
  invisible(TRUE)
}

expm1_transform <- function(a) {
  exp1(a)
  invisible(TRUE)
}

sqrt_transform <- function(a) {
  # 使用 SQL 直接更新（先删除旧列，允许覆盖）
  DBI::dbExecute(a$con, "ALTER TABLE X_CSR_data DROP COLUMN IF EXISTS X_sqrt")
  DBI::dbExecute(a$con, "ALTER TABLE X_CSR_data ADD COLUMN X_sqrt REAL")
  DBI::dbExecute(a$con, "UPDATE X_CSR_data SET X_sqrt = SQRT(data) WHERE data IS NOT NULL")
  invisible(TRUE)
}

pca <- function(a) {
  cat("  Note: PCA not directly supported in scatlas, skipping...\n")
  invisible(TRUE)
}

sequential_iteration <- function(a) {
  batch_size <- 2048
  n_cells <- nrow(obs(a))
  count <- 0
  for (i in seq(1, n_cells, by = batch_size)) {
    end_idx <- min(i + batch_size - 1, n_cells)
    # 只查询计数，不构建矩阵
    result <- DBI::dbGetQuery(a$con,
      sprintf("SELECT COUNT(*) as n FROM X_CSR_data WHERE cell_index >= %d AND cell_index < %d",
              i - 1, end_idx))
    count <- count + result$n
  }
  invisible(count)
}

shuffled_iteration <- function(a) {
  batch_size <- 2048
  n_cells <- nrow(obs(a))
  indices <- sample(1:n_cells)
  count <- 0
  for (i in seq(1, length(indices), by = batch_size)) {
    end_idx <- min(i + batch_size - 1, length(indices))
    batch_indices <- indices[i:end_idx]
    indices_str <- paste(batch_indices, collapse = ",")
    result <- DBI::dbGetQuery(a$con,
      sprintf("SELECT COUNT(*) as n FROM X_CSR_data WHERE cell_index IN (%s)", indices_str))
    count <- count + result$n
  }
  invisible(count)
}

random_minibatch_iteration <- function(a) {
  batch_size <- 2048
  n_cells <- nrow(obs(a))
  n_batches <- 100
  count <- 0
  for (i in 1:n_batches) {
    batch_indices <- sample(1:n_cells, size = batch_size, replace = FALSE)
    indices_str <- paste(batch_indices, collapse = ",")
    result <- DBI::dbGetQuery(a$con,
      sprintf("SELECT COUNT(*) as n FROM X_CSR_data WHERE cell_index IN (%s)", indices_str))
    count <- count + result$n
  }
  invisible(count)
}

query_by_gene_names <- function(a) {
  # 获取前4个基因的 id
  gene_ids <- DBI::dbGetQuery(a$con, "SELECT id FROM var LIMIT 4")
  if (nrow(gene_ids) > 0) {
    indices_str <- paste(gene_ids$id, collapse = ",")
    result <- DBI::dbGetQuery(a$con,
      sprintf("SELECT COUNT(*) as n FROM X_CSR_data WHERE indices IN (%s)", indices_str))
    invisible(result$n)
  }
  invisible(0)
}

query_by_expression_1gene <- function(a) {
  stats <- get_gene_stats(a)
  gene <- stats$genes_1
  # 使用 SQL 查询
  result <- DBI::dbGetQuery(a$con,
    sprintf("SELECT COUNT(*) as n FROM X_CSR_data WHERE indices = %d AND data > 0.5", gene))
  invisible(result$n)
}

query_by_expression_2genes <- function(a) {
  stats <- get_gene_stats(a)
  genes <- stats$genes_2
  # 使用 SQL 查询两个基因都表达的细胞
  result <- DBI::dbGetQuery(a$con,
    sprintf("SELECT COUNT(*) as n FROM X_CSR_data
             WHERE indices IN (%d, %d) AND data > 0.5
             GROUP BY cell_index
             HAVING COUNT(DISTINCT indices) = 2", genes[1], genes[2]))
  invisible(result$n)
}

query_by_expression_3genes <- function(a) {
  stats <- get_gene_stats(a)
  genes <- stats$genes_3
  # 使用 SQL 查询三个基因都表达的细胞
  result <- DBI::dbGetQuery(a$con,
    sprintf("SELECT COUNT(*) as n FROM (
              SELECT cell_index FROM X_CSR_data
              WHERE indices IN (%d, %d, %d) AND data > 0.5
              GROUP BY cell_index
              HAVING COUNT(DISTINCT indices) = 3
            ) t", genes[1], genes[2], genes[3]))
  invisible(result$n)
}

# ==================== Operator 注册表 ====================

OPERATORS <- list(
  filter_cells_min_genes_200,
  filter_cells_max_genes_6000,
  filter_cells_min_counts_500,
  filter_cells_max_counts_40000,
  filter_genes_min_cells_3,
  filter_genes_max_cells_1000000,
  filter_genes_min_counts_10,
  filter_genes_max_counts_100000,
  log1p_transform,
  scale_transform,
  expm1_transform,
  sqrt_transform,
  pca,
  sequential_iteration,
  shuffled_iteration,
  random_minibatch_iteration,
  query_by_gene_names,
  query_by_expression_1gene,
  query_by_expression_2genes,
  query_by_expression_3genes
)

OPERATOR_NAMES <- c(
  "filter_cells_min_genes_200",
  "filter_cells_max_genes_6000",
  "filter_cells_min_counts_500",
  "filter_cells_max_counts_40000",
  "filter_genes_min_cells_3",
  "filter_genes_max_cells_1000000",
  "filter_genes_min_counts_10",
  "filter_genes_max_counts_100000",
  "log1p",
  "scale",
  "expm1",
  "sqrt",
  "pca",
  "sequential_iteration",
  "shuffled_iteration",
  "random_minibatch_iteration",
  "query_by_gene_names",
  "query_by_expression_1gene",
  "query_by_expression_2genes",
  "query_by_expression_3genes"
)

# ==================== 主函数 ====================

run_single_operator <- function(args) {
  # 解析路径
  data_file <- args$dataset
  db_path <- dirname(data_file)
  db_name <- tools::file_path_sans_ext(basename(data_file))

  force_reimport <- isTRUE(args[["force-reimport"]])
  a <- load_data_atlas(db_name, db_path, data_file, reuse_db = !force_reimport)

  # 预计算基因统计（不计入 operator 时间）
  invisible(get_gene_stats(a))

  if (!(args$operator %in% OPERATOR_NAMES)) {
    cat("Error: Unknown operator:", args$operator, "\n")
    cat("\nAvailable operators:\n")
    for (name in OPERATOR_NAMES) {
      cat("  -", name, "\n")
    }
    atlas_close(a)
    quit(status = 1)
  }

  op_idx <- which(OPERATOR_NAMES == args$operator)
  func <- OPERATORS[[op_idx]]

  cat("\n[Running]", args$operator, "\n")

  # 启动内存监控
  peak_mem <- 0
  mem_file <- paste0("/tmp/scatlas_mem_", Sys.getpid(), ".txt")
  monitor_script <- paste0("
    pid=", Sys.getpid(), "
    while kill -0 $pid 2>/dev/null; do
      grep VmRSS /proc/$pid/status 2>/dev/null | awk '{print $2}' >> ", mem_file, "
      sleep 0.5
    done
  ")
  system(paste0("(", monitor_script, ") &"))
  monitor_pid <- as.integer(strsplit(system("echo $!", intern = TRUE), "\n")[[1]])

  # 执行 operator
  start_time <- Sys.time()
  result <- tryCatch({
    func(a)
    duration <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))

    # 停止内存监控
    system(paste0("kill ", monitor_pid, " 2>/dev/null"))
    Sys.sleep(0.2)

    # 计算峰值内存
    if (file.exists(mem_file)) {
      mem_samples <- as.numeric(readLines(mem_file))
      peak_mem <- max(mem_samples, na.rm = TRUE) / 1024  # KB to MB
      file.remove(mem_file)
    } else {
      peak_mem <- get_memory_mb()
    }

    list(
      Operator = args$operator,
      Time_s = round(duration, 4),
      Peak_Memory_MiB = round(peak_mem, 2)
    )
  }, error = function(e) {
    system(paste0("kill ", monitor_pid, " 2>/dev/null"))
    if (file.exists(mem_file)) file.remove(mem_file)
    if (grepl("out of memory|Killed", e$message, ignore.case = TRUE)) {
      cat("  OOM: Out of memory\n")
      list(
        Operator = args$operator,
        Time_s = "OOM",
        Peak_Memory_MiB = "OOM"
      )
    } else {
      stop(e)
    }
  })

  save_result(result, args$dataset, args[["output-dir"]], args[["result-file"]])
  cat("  -> Done. Time:", result$Time_s, "s, Peak Memory:", result$Peak_Memory_MiB, "MiB\n")

  atlas_close(a)
}

list_operators <- function() {
  cat("Available operators:\n")
  for (name in OPERATOR_NAMES) {
    cat("  -", name, "\n")
  }
}

main <- function() {
  option_list <- list(
    make_option(c("--operator"), help = "Operator to run"),
    make_option(c("--dataset"), help = "Dataset file path"),
    make_option(c("--output-dir"), help = "Output directory", default = NULL),
    make_option(c("--result-file"), help = "Result file name", default = NULL),
    make_option(c("--force-reimport"), action = "store_true", help = "Force reimport data", default = FALSE),
    make_option(c("--list"), action = "store_true", help = "List all operators")
  )

  parser <- OptionParser(usage = "%prog [options]", option_list = option_list,
                         description = "scAtlas (R) Performance Benchmark")
  args <- parse_args(parser)

  if (isTRUE(args$list)) {
    list_operators()
    return()
  }

  if (is.null(args$dataset)) {
    print_help(parser)
    cat("\nError: --dataset is required\n")
    quit(status = 1)
  }

  if (is.null(args$operator)) {
    print_help(parser)
    cat("\nError: --operator is required\n")
    quit(status = 1)
  }

  run_single_operator(args)
}

if (!interactive()) {
  main()
}
