#!/usr/bin/env Rscript
library(Seurat)
library(optparse)
library(data.table)
library(SeuratDisk)
library(Matrix)


# Define %||% operator for NULL handling
`%||%` <- function(x, y) if (is.null(x)) y else x

# Gene stats cache
.gene_stats_cache <- new.env(parent = emptyenv())

ensure_meta_cols <- function(obj) {
  # 修复 rownames 不一致问题（Seurat v5 兼容性）
  counts <- tryCatch({
    get_counts(obj)
  }, error = function(e) NULL)
  if (!is.null(counts)) {
    current_rownames <- rownames(obj)
    expected_rownames <- rownames(counts)
    if (!identical(current_rownames, expected_rownames)) {
      cat("  Fixing rownames mismatch...\n")
      rownames(obj) <- expected_rownames
    }
  }

  # 确保必要的 metadata 列存在
  if (!"nFeature_RNA" %in% colnames(obj@meta.data)) {
    counts <- get_counts(obj)
    obj$nFeature_RNA <- Matrix::colSums(counts > 0)
  }
  if (!"nCount_RNA" %in% colnames(obj@meta.data)) {
    counts <- get_counts(obj)
    obj$nCount_RNA <- Matrix::colSums(counts)
  }
  obj
}

get_counts <- function(obj, assay = "RNA") {
  # Get counts data - handle Seurat v5 multi-layer assays
  assay_obj <- obj[[assay]]
  if ("counts" %in% names(assay_obj)) {
    return(assay_obj$counts)
  }
  # Fallback: try to get from layers
  layers <- Layers(assay_obj, search = "counts")
  if (length(layers) > 0) {
    return(LayerData(assay_obj, layer = layers[1]))
  }
  stop("No counts layer found")
}

get_gene_stats <- function(obj) {
  cache_key <- object.size(obj)
  if (exists(as.character(cache_key), envir = .gene_stats_cache)) {
    return(get(as.character(cache_key), envir = .gene_stats_cache))
  }
  counts <- get_counts(obj)
  gene_n_cells <- Matrix::rowSums(counts > 0)
  sorted_idx <- order(gene_n_cells, decreasing = FALSE)
  n_genes <- length(sorted_idx)
  result <- list(
    sorted_idx = sorted_idx,
    genes_1 = rownames(obj)[sorted_idx[as.integer(n_genes / 2) - 1]],
    genes_2 = rownames(obj)[sorted_idx[as.integer(n_genes / 2):(as.integer(n_genes / 2) + 1)]],
    genes_3 = rownames(obj)[sorted_idx[(as.integer(n_genes / 2) + 2):(as.integer(n_genes / 2) + 4)]]
  )
  assign(as.character(cache_key), result, envir = .gene_stats_cache)
  return(result)
}

clear_gene_stats_cache <- function() {
  rm(list = ls(envir = .gene_stats_cache), envir = .gene_stats_cache)
}

get_data_format <- function(path) {
  ext <- tools::file_ext(path)
  return(ext)
}

load_data <- function(file_path) {
  cat("Loading data:", file_path, "\n")
  ext <- get_data_format(file_path)
  if (ext == "rds") {
    obj <- readRDS(file_path)
  } else if (ext == "h5seurat") {
    # 直接读取 h5seurat 文件（Seurat 专用格式）
    obj <- tryCatch({
      SeuratDisk::LoadH5Seurat(file_path)
    }, error = function(e) {
      cat("  LoadH5Seurat failed, trying alternative method...\n")
      # 备选方案：使用 Connect 直接读取数据
      h5 <- SeuratDisk::Connect(file_path, force = FALSE)

      # 读取 counts 矩阵 (稀疏格式)
      counts_indptr <- h5[["assays/RNA/counts/indptr"]][]
      counts_indices <- h5[["assays/RNA/counts/indices"]][]
      counts_data <- h5[["assays/RNA/counts/data"]][]
      n_cells <- length(counts_indptr) - 1
      n_genes_actual <- max(counts_indices) + 1

      # 读取 genes (从 meta.features，编码格式)
      fn <- h5[["assays/RNA/meta.features/feature_name"]]
      categories <- fn[["categories"]][]
      codes <- fn[["codes"]][]
      genes <- categories[codes + 1]  # codes 是0索引的

      # 读取 cells
      cells <- h5[["cell.names"]][]
      cells <- as.character(cells)

      h5$close_all()

      # 重新构建稀疏矩阵
      counts <- Matrix::sparseMatrix(
        i = counts_indices + 1,
        p = counts_indptr,
        x = counts_data,
        dims = c(n_genes_actual, n_cells),
        dimnames = list(genes[1:n_genes_actual], cells)
      )

      # 创建 Seurat 对象
      obj <- CreateSeuratObject(counts = counts)
      cat("  Loaded", dim(obj)[1], "genes x", dim(obj)[2], "cells\n")
      obj
    })
    # 确保必要的 metadata 列存在（Seurat v5 兼容性）
    obj <- ensure_meta_cols(obj)
  } else if (ext == "h5ad") {
    # SeuratDisk 读取 h5ad 需要先转换为 h5seurat 格式
    h5seurat_path <- gsub("\\.h5ad$", ".h5seurat", file_path)
    if (!file.exists(h5seurat_path)) {
      cat("  Converting h5ad to h5seurat format...\n")
      SeuratDisk::Convert(file_path, dest = h5seurat_path, overwrite = TRUE, verbose = FALSE)
    }
    obj <- SeuratDisk::LoadH5Seurat(h5seurat_path)
    obj <- ensure_meta_cols(obj)
  } else if (ext == "h5") {
    # 使用 Read10X_h5 读取，并显式指定类型以避免稀疏矩阵问题
    raw_data <- Read10X_h5(file_path, use.names = TRUE)
    if (inherits(raw_data, "list")) {
      # 10x v3 可能返回包含多个矩阵的 list
      raw_data <- raw_data$Counts %||% raw_data$count %||% raw_data[[1]]
    }
    # 重新构建稀疏矩阵以确保格式正确（Seurat v5 兼容性）
    if (inherits(raw_data, "dgCMatrix")) {
      cat("  Reconstructing sparse matrix...\n")
      raw_data <- Matrix(raw_data, sparse = TRUE)
    }
    obj <- CreateSeuratObject(counts = raw_data)
  } else {
    stop("Unsupported file format: ", ext)
  }
  cat("Data size:", dim(obj)[1], "genes x", dim(obj)[2], "cells\n")
  return(obj)
}

save_result <- function(result, dataset_path, output_dir = NULL, result_file = NULL) {
  dataset_name <- tools::file_path_sans_ext(basename(dataset_path))
  if (is.null(result_file)) {
    result_file <- paste0("seurat_results_", dataset_name, ".csv")
  }
  if (!is.null(output_dir)) {
    result_file <- file.path(output_dir, result_file)
  }
  result$Dataset <- basename(dataset_path)
  df <- as.data.frame(result)
  cols <- c("Operator", "Time_s", "Peak_Memory_MiB", "Dataset")
  df <- df[, cols, drop = FALSE]
  if (file.exists(result_file)) {
    fwrite(df, result_file, append = TRUE, col.names = FALSE)
  } else {
    fwrite(df, result_file)
  }
  cat("Result saved:", result_file, "\n")
}

handle_oom <- function(op_name, dataset_path, output_dir = NULL, result_file = NULL) {
  result <- data.frame(
    Operator = op_name,
    Time_s = "OOM",
    Peak_Memory_MiB = "OOM",
    Dataset = basename(dataset_path)
  )
  if (is.null(result_file)) {
    result_file <- paste0("seurat_results_", tools::file_path_sans_ext(basename(dataset_path)), ".csv")
  }
  if (!is.null(output_dir)) {
    result_file <- file.path(output_dir, result_file)
  }
  df <- as.data.frame(result)
  if (file.exists(result_file)) {
    fwrite(df, result_file, append = TRUE, col.names = FALSE)
  } else {
    fwrite(df, result_file)
  }
  cat("OOM result saved\n")
}

get_rss_mb <- function() {
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

run_benchmark <- function(op_func, op_name, ...) {
  pid <- Sys.getpid()
  # 使用固定的临时文件路径
  mem_file <- paste0("/tmp/seurat_mem_", pid, ".txt")

  # 写入监控脚本 - 使用 VmHWM (High Water Mark) 记录历史最大物理内存
  monitor_script <- sprintf(
    "#!/bin/bash
pid=%d
peak=0
touch %s
while kill -0 $pid 2>/dev/null; do
  hwm=$(grep VmHWM /proc/$pid/status 2>/dev/null | awk '{print $2}')
  if [ -n \"$hwm\" ]; then
    hwm_mb=$((hwm / 1024))
    if [ $hwm_mb -gt $peak ]; then
      peak=$hwm_mb
    fi
    echo $peak > %s
  fi
  sleep 0.01
done
rm -f %s
",
    pid, mem_file, mem_file, mem_file
  )
  script_file <- paste0("/tmp/seurat_mem_script_", pid, ".sh")
  writeLines(monitor_script, script_file)
  system2("bash", args = c(script_file), wait = FALSE)

  peak_mem <- get_rss_mb()
  start_time <- Sys.time()
  result <- tryCatch({
    op_func(...)
    Time_s <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
    list(
      Operator = op_name,
      Time_s = round(Time_s, 4),
      Peak_Memory_MiB = round(peak_mem, 2)
    )
  }, error = function(e) {
    if (grepl("memory|cannot allocate", e$message, ignore.case = TRUE)) {
      return(list(Operator = op_name, Time_s = "OOM", Peak_Memory_MiB = "OOM"))
    }
    stop(e)
  })

  # 读取监控进程的峰值内存
  Sys.sleep(0.1)
  if (file.exists(mem_file)) {
    peak_from_file <- tryCatch({
      as.numeric(tail(readLines(mem_file), 1))
    }, error = function(e) NA)
    if (!is.na(peak_from_file) && peak_from_file > result$Peak_Memory_MiB) {
      result$Peak_Memory_MiB <- round(peak_from_file, 2)
    }
    try(file.remove(mem_file), silent = TRUE)
  }
  try(file.remove(script_file), silent = TRUE)

  return(result)
}

# Operators
filter_cells_min_genes_200 <- function(obj) { subset(obj, subset = nFeature_RNA >= 200) }
filter_cells_max_genes_6000 <- function(obj) { subset(obj, subset = nFeature_RNA <= 6000) }
filter_cells_min_counts_500 <- function(obj) { subset(obj, subset = nCount_RNA >= 500) }
filter_cells_max_counts_40000 <- function(obj) { subset(obj, subset = nCount_RNA <= 40000) }

filter_genes_min_cells_3 <- function(obj) {
  genes_to_keep <- which(Matrix::rowSums(obj) >= 3)
  subset(obj, features = genes_to_keep)
}
filter_genes_max_cells_1000000 <- function(obj) {
  genes_to_keep <- which(Matrix::rowSums(obj) <= 1000000)
  subset(obj, features = genes_to_keep)
}
filter_genes_min_counts_10 <- function(obj) {
  genes_to_keep <- which(Matrix::rowSums(obj) >= 10)
  subset(obj, features = genes_to_keep)
}
filter_genes_max_counts_100000 <- function(obj) {
  genes_to_keep <- which(Matrix::rowSums(obj) <= 100000)
  subset(obj, features = genes_to_keep)
}

log1p <- function(obj) {
  obj <- NormalizeData(obj, normalization.method = "LogNormalize", verbose = FALSE)
  obj
}

scale <- function(obj) {
  if (!"data" %in% names(obj[["RNA"]])) {
    obj <- NormalizeData(obj, verbose = FALSE)
  }
  # Get features from RNA assay
  rna_features <- rownames(obj[["RNA"]])
  features <- head(rna_features, min(2000, length(rna_features)))
  ScaleData(obj, features = features, do.scale = TRUE, do.center = TRUE, verbose = FALSE)
}

expm1 <- function(obj) {
  counts <- get_counts(obj)
  if (inherits(counts, "dgCMatrix")) {
    counts@x <- exp(counts@x) - 1
    obj[["RNA"]]$counts <- counts
  } else {
    new_counts <- as.matrix(exp(counts) - 1)
    obj[["RNA"]]$counts <- as(new_counts, "dgCMatrix")
  }
  obj
}

sqrt_transform <- function(obj) {
  counts <- get_counts(obj)
  if (inherits(counts, "dgCMatrix")) {
    counts@x <- sqrt(counts@x)
    obj[["RNA"]]$counts <- counts
  } else {
    new_counts <- as.matrix(sqrt(counts))
    obj[["RNA"]]$counts <- as(new_counts, "dgCMatrix")
  }
  obj
}

pca <- function(obj) {
  if (!"data" %in% names(obj[["RNA"]])) {
    obj <- NormalizeData(obj, verbose = FALSE)
  }
  if (!"scale.data" %in% names(obj[["RNA"]])) {
    obj <- ScaleData(obj, do.scale = FALSE, do.center = TRUE, verbose = FALSE)
  }
  features <- rownames(obj)
  RunPCA(obj, features = features, npcs = 50, verbose = FALSE)
}

sequential_iteration <- function(obj) {
  counts <- get_counts(obj)
  n_cells <- dim(counts)[2]
  batch_size <- 2048
  for (i in seq(1, n_cells, by = batch_size)) {
    end_idx <- min(i + batch_size - 1, n_cells)
    m <- as.matrix(counts[, i:end_idx])
  }
}

shuffled_iteration <- function(obj) {
  counts <- get_counts(obj)
  n_cells <- dim(counts)[2]
  batch_size <- 2048
  indices <- sample(1:n_cells)
  for (i in seq(1, n_cells, by = batch_size)) {
    end_idx <- min(i + batch_size - 1, n_cells)
    batch_indices <- indices[i:end_idx]
    m <- as.matrix(counts[, batch_indices])
  }
}

random_minibatch_iteration <- function(obj) {
  counts <- get_counts(obj)
  n_cells <- dim(counts)[2]
  batch_size <- 2048
  n_batches <- 100
  for (i in 1:n_batches) {
    batch_indices <- sample(1:n_cells, size = batch_size, replace = FALSE)
    m <- as.matrix(counts[, batch_indices])
  }
}

query_by_gene_names <- function(obj) {
  genes <- rownames(obj)[1:4]
  counts <- get_counts(obj)
  m <- as.matrix(counts[genes, ])
}

query_by_expression_1gene <- function(obj) {
  stats <- get_gene_stats(obj)
  gene <- stats$genes_1
  counts <- get_counts(obj)
  expr <- as.numeric(counts[gene, ])
  mask <- expr > 0.5
  m <- as.matrix(counts[gene, which(mask)])
}

query_by_expression_2genes <- function(obj) {
  stats <- get_gene_stats(obj)
  genes <- stats$genes_2
  counts <- get_counts(obj)
  expr1 <- as.numeric(counts[genes[1], ])
  expr2 <- as.numeric(counts[genes[2], ])
  mask <- expr1 > 0.5 & expr2 > 0.5
  m <- as.matrix(counts[genes[1], which(mask)])
}

query_by_expression_3genes <- function(obj) {
  stats <- get_gene_stats(obj)
  genes <- stats$genes_3
  counts <- get_counts(obj)
  expr1 <- as.numeric(counts[genes[1], ])
  expr2 <- as.numeric(counts[genes[2], ])
  expr3 <- as.numeric(counts[genes[3], ])
  mask <- expr1 > 0.5 & expr2 > 0.5 & expr3 > 0.5
  m <- as.matrix(counts[genes[1], which(mask)])
}

OPERATORS <- list(
  filter_cells_min_genes_200,
  filter_cells_max_genes_6000,
  filter_cells_min_counts_500,
  filter_cells_max_counts_40000,
  filter_genes_min_cells_3,
  filter_genes_max_cells_1000000,
  filter_genes_min_counts_10,
  filter_genes_max_counts_100000,
  log1p,
  scale,
  expm1,
  sqrt_transform,
  pca,
  query_by_gene_names,
  query_by_expression_1gene,
  query_by_expression_2genes,
  query_by_expression_3genes,
  sequential_iteration,
  shuffled_iteration,
  random_minibatch_iteration
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
  "query_by_gene_names",
  "query_by_expression_1gene",
  "query_by_expression_2genes",
  "query_by_expression_3genes",
  "sequential_iteration",
  "shuffled_iteration",
  "random_minibatch_iteration"
)

main <- function() {
  option_list <- list(
    make_option(c("--operator"), type = "character", help = "Operator name"),
    make_option(c("--dataset"), type = "character", help = "Dataset file path"),
    make_option(c("--memory-limit"), type = "numeric", default = 200, help = "Memory limit (GB)"),
    make_option(c("--output-dir"), type = "character", default = NULL, help = "Output directory"),
    make_option(c("--result-file"), type = "character", default = NULL, help = "Result filename"),
    make_option(c("--list"), action = "store_true", default = FALSE, help = "List operators"),
    make_option(c("--run-all"), action = "store_true", default = FALSE, help = "Run all operators")
  )
  parser <- OptionParser(option_list = option_list)
  args <- parse_args(parser)
  if (args$list) {
    cat("Available operators:\n")
    for (name in OPERATOR_NAMES) { cat("  -", name, "\n") }
    quit(status = 0)
  }
  if (!is.null(args$run_all) && args$run_all) {
    if (is.null(args$dataset)) { cat("Error: --run-all requires --dataset\n"); quit(status = 1) }
    run_all_operators(args); quit(status = 0)
  }
  if (is.null(args$operator) || is.null(args$dataset)) {
    print_help(parser); cat("\nError: --operator and --dataset are required\n"); quit(status = 1)
  }
  if (!file.exists(args$dataset)) {
    cat("Error: Dataset file not found:", args$dataset, "\n"); quit(status = 1)
  }
  run_single_operator(args)
}

run_single_operator <- function(args) {
  cat("DEBUG: args$output_dir =", args$output_dir, "\n")
  cat("DEBUG: args$result_file =", args$result_file, "\n")
  obj <- load_data(args$dataset)
  invisible(get_gene_stats(obj))
  op_idx <- which(OPERATOR_NAMES == args$operator)
  if (length(op_idx) == 0) {
    cat("Error: Unknown operator:", args$operator, "\n")
    cat("Available operators:\n")
    for (name in sort(OPERATOR_NAMES)) { cat("  -", name, "\n") }
    quit(status = 1)
  }
  op_func <- OPERATORS[[op_idx]]
  cat("\n[Running]", args$operator, "\n")
  result <- tryCatch({
    run_benchmark(op_func, args$operator, obj)
  }, error = function(e) {
    msg <- tolower(e$message)
    if (grepl("memory|cannot allocate", msg)) {
      cat("OOM: Time_s=OOM, Peak_Memory_MiB=OOM\n")
      handle_oom(args$operator, args$dataset, args$output_dir, args$result_file)
      quit(status = 1)
    }
    stop(e)
  })
  save_result(result, args$dataset, args$output_dir, args$result_file)
  cat("  -> Done. Time:", result$Time_s, "s, Peak memory:", result$Peak_Memory_MiB, "MiB\n")
}

run_all_operators <- function(args) {
  obj <- load_data(args$dataset)
  invisible(get_gene_stats(obj))
  cat("\nRunning", length(OPERATORS), "operators...\n")
  cat(paste(rep("=", 60), collapse = ""), "\n")
  for (i in seq_along(OPERATORS)) {
    name <- OPERATOR_NAMES[i]
    op_func <- OPERATORS[[i]]
    cat("\n[", i, "/", length(OPERATORS), "] ", name, "...\n", sep = "")
    result <- tryCatch({
      run_benchmark(op_func, name, obj)
    }, error = function(e) {
      msg <- tolower(e$message)
      if (grepl("memory|cannot allocate", msg)) {
        cat("  OOM\n")
        handle_oom(name, args$dataset, args$output_dir, args$result_file)
        return(list(Operator = name, Time_s = "OOM", Peak_Memory_MiB = "OOM"))
      }
      stop(e)
    })
    if (result$Time_s != "OOM") {
      cat("  -> Done. Time:", result$Time_s, "s, Peak memory:", result$Peak_Memory_MiB, "MiB\n")
    }
    save_result(result, args$dataset, args$output_dir, args$result_file)
    gc(verbose = FALSE)
    Sys.sleep(0.3)
  }
  cat("\n", paste(rep("=", 60), collapse = ""), "\n", sep = "")
  cat("All done!\n")
}

main()
