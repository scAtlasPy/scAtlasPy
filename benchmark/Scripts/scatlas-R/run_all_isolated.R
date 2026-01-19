#!/usr/bin/env Rscript
#
# Run all scAtlas-R operators in isolated processes for one dataset
#

# ==================== Configuration ====================
# 使用相对路径，基于脚本所在目录
get_script_dir <- function() {
  for (i in sys.nframe():1) {
    f <- sys.frame(i)
    if (!is.null(f$ofile)) {
      return(dirname(f$ofile))
    }
  }
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg) > 0) {
    return(dirname(sub("^--file=", "", file_arg[1])))
  }
  return(".")
}
SCRIPT_DIR <- get_script_dir()
BENCHMARK_DIR <- file.path(SCRIPT_DIR, "..", "..")

# 测试1: 20k_PBMC数据集
# DATASET_PATH <- file.path(BENCHMARK_DIR, "Dataset", "20k_PBMC.h5")

# 测试2: 130k_thymus_atlas数据集
# DATASET_PATH <- file.path(BENCHMARK_DIR, "Dataset", "130k_thymus_atlas_HTA07.A01.v02.entire_data_raw_count.h5ad")

# 测试3: 500k_drugscreen数据集
# DATASET_PATH <- file.path(BENCHMARK_DIR, "Dataset", "500k_drugscreen.h5")

# 测试4: 1M_neurons数据集
DATASET_PATH <- file.path(BENCHMARK_DIR, "Dataset", "1M_neurons_filtered_gene_bc_matrices_h5.h5")
OUTPUT_DIR <- file.path(BENCHMARK_DIR, "Results", "Lastest_Results", "100G_OOM")
RESULT_FILE <- paste0("R-scatlas_results_", tools::file_path_sans_ext(basename(DATASET_PATH)), ".csv")
# =================================================

save_oom_result <- function(op_name, dataset_path, output_dir = NULL, result_file = NULL) {
  if (is.null(result_file)) {
    result_file <- paste0("R-scatlas_results_", tools::file_path_sans_ext(basename(dataset_path)), ".csv")
  }
  if (!is.null(output_dir)) {
    result_file <- file.path(output_dir, result_file)
  }
  result <- data.frame(
    Operator = op_name,
    Time_s = "OOM",
    Peak_Memory_MiB = "OOM",
    Dataset = basename(dataset_path)
  )
  if (file.exists(result_file)) {
    data.table::fwrite(result, result_file, append = TRUE, col.names = FALSE)
  } else {
    data.table::fwrite(result, result_file)
  }
  cat("  OOM result saved\n")
}

main <- function() {
  if (!file.exists(DATASET_PATH)) {
    cat("Error: Dataset file not found:", DATASET_PATH, "\n")
    quit(status = 1)
  }

  this_dir <- getwd()
  main_script <- file.path(this_dir, "scatlas_benchmark.R")
  rscript_path <- Sys.which("Rscript")

  cmd <- c(
    rscript_path,
    main_script,
    "--dataset", DATASET_PATH
  )

  if (!is.null(OUTPUT_DIR) && OUTPUT_DIR != "") {
    cmd <- c(cmd, "--output-dir", OUTPUT_DIR)
  }
  if (!is.null(RESULT_FILE) && RESULT_FILE != "") {
    cmd <- c(cmd, "--result-file", RESULT_FILE)
  }

  operators <- c(
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

  cat("Running", length(operators), "operators in isolated processes...\n")
  cat(paste(rep("=", 60), collapse = ""), "\n")

  for (i in seq_along(operators)) {
    op <- operators[i]
    full_cmd <- c(cmd, "--operator", op)
    cat("\n[", i, "/", length(operators), "] ", op, "...\n", sep = "")
    cat(paste(rep("-", 40), collapse = ""), "\n")

    result <- system2(rscript_path, args = full_cmd[-1], stdout = TRUE, stderr = TRUE, wait = TRUE)

    # 检测 OOM
    oom_detected <- any(grepl("^OOM:", result))
    killed_detected <- any(grepl("Killed", result))
    if (oom_detected || killed_detected) {
      save_oom_result(op, DATASET_PATH, OUTPUT_DIR, RESULT_FILE)
      cat("  OOM (memory limit exceeded or killed)\n")
    } else if (any(grepl("^Error", result))) {
      cat("  Failed:", grep("^Error", result, value = TRUE)[1], "\n")
    } else {
      # 显示最后几行输出
      for (line in tail(result[result != ""], 3)) {
        cat("  ", line, "\n")
      }
    }
  }

  cat("\n", paste(rep("=", 60), collapse = ""), "\n", sep = "")
  cat("All done!\n")
}

if (!interactive()) {
  main()
}
