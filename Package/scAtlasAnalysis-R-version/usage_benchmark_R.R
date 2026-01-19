# benchmark_R.R - R包完整功能测试
# =============================================================================
# 测试 scAtlas 包的所有核心功能
# 与Python版本 scatlaspy 保持功能对等
# =============================================================================

library(scAtlas)
library(jsonlite)

# 配置
DATA_FILE <- "/home/senpeng/zspbenchmark/benchmark/Dataset/20k_PBMC.h5"
DB_PATH <- "/home/senpeng/scatlas_R"
DATABASE_NAME <- "benchmark"

# 计时函数
time_it <- function(name, expr, verbose = TRUE) {
  gc()
  start <- Sys.time()
  result <- expr
  end <- Sys.time()
  elapsed <- as.numeric(difftime(end, start, units = "secs"))
  if (verbose) {
    cat(sprintf("  %s: %.2f 秒\n", name, elapsed))
  }
  list(name = name, seconds = elapsed, result = result)
}

# 记录结果
results <- list()

cat("==============================================\n")
cat("       R scAtlas 完整功能测试\n")
cat("==============================================\n\n")

# ============================================================================
# 第一部分：数据加载测试
# ============================================================================
cat("=== 第1部分：数据加载 ===\n\n")

# 1. 创建数据库并加载数据
cat("[1-1] 加载 H5 数据...\n")
a <- atlas(DATABASE_NAME, path = DB_PATH, mode = "r+")
t <- time_it("load_data (H5)", load_data(a, DATA_FILE))
results[[length(results) + 1]] <- data.frame(
  category = "Data Loading",
  test = t$name,
  seconds = t$seconds,
  stringsAsFactors = FALSE
)
cat("\n")

# 打印数据库信息
cat("数据库信息:\n")
print(a)
cat("\n")

# ============================================================================
# 第二部分：基本查询测试
# ============================================================================
cat("=== 第2部分：基本查询 ===\n\n")

# 2. 查询 obs 数量
cat("[2-1] 查询 obs 数量...\n")
t <- time_it("nrow(obs)", nrow(obs(a)))
results[[length(results) + 1]] <- data.frame(
  category = "Basic Query",
  test = t$name,
  seconds = t$seconds,
  stringsAsFactors = FALSE
)

# 3. 查询 var 数量
cat("[2-2] 查询 var 数量...\n")
t <- time_it("nrow(var)", nrow(var(a)))
results[[length(results) + 1]] <- data.frame(
  category = "Basic Query",
  test = t$name,
  seconds = t$seconds,
  stringsAsFactors = FALSE
)

# 4. SQL 查询
cat("[2-3] SQL 查询...\n")
t <- time_it("query (SQL)", {
  query(a, "SELECT COUNT(*) as n FROM obs")
  query(a, "SELECT COUNT(*) as n FROM var")
  query(a, "SELECT COUNT(*) as n FROM X_CSR_data")
})
results[[length(results) + 1]] <- data.frame(
  category = "Basic Query",
  test = t$name,
  seconds = t$seconds,
  stringsAsFactors = FALSE
)

# 5. 获取 obs 表（前10列）
cat("[2-4] 获取 obs 表...\n")
t <- time_it("obs()", head(obs(a), 10))
results[[length(results) + 1]] <- data.frame(
  category = "Basic Query",
  test = t$name,
  seconds = t$seconds,
  stringsAsFactors = FALSE
)

# 6. 获取 var 表
cat("[2-5] 获取 var 表...\n")
t <- time_it("var()", head(var(a), 10))
results[[length(results) + 1]] <- data.frame(
  category = "Basic Query",
  test = t$name,
  seconds = t$seconds,
  stringsAsFactors = FALSE
)

# 7. 获取 X 矩阵
cat("[2-6] 获取 X 矩阵 (稀疏)...\n")
t <- time_it("X() sparse", {
  X_mat <- X(a)
  dim(X_mat)
})
results[[length(results) + 1]] <- data.frame(
  category = "Basic Query",
  test = t$name,
  seconds = t$seconds,
  stringsAsFactors = FALSE
)

cat("\n")

# ============================================================================
# 第三部分：视图系统测试
# ============================================================================
cat("=== 第3部分：视图系统 ===\n\n")

# 8. 整数索引筛选
cat("[3-1] 整数索引筛选 [1:100]...\n")
t <- time_it("atlas[1:100]", {
  a_view <- a[1:100]
  nrow(obs(a_view))
})
results[[length(results) + 1]] <- data.frame(
  category = "View System",
  test = t$name,
  seconds = t$seconds,
  stringsAsFactors = FALSE
)

# 9. 字符索引筛选
cat("[3-2] 字符索引筛选...\n")
cell_ids <- obs(a)$cell_id[1:50]
t <- time_it("atlas[cell_ids]", {
  a_view <- a[cell_ids]
  nrow(obs(a_view))
})
results[[length(results) + 1]] <- data.frame(
  category = "View System",
  test = t$name,
  seconds = t$seconds,
  stringsAsFactors = FALSE
)

# 10. 视图链式操作
cat("[3-3] 视图链式操作...\n")
t <- time_it("chain views", {
  a_v1 <- a[1:500]
  a_v2 <- a_v1[1:100]
  nrow(obs(a_v2))
})
results[[length(results) + 1]] <- data.frame(
  category = "View System",
  test = t$name,
  seconds = t$seconds,
  stringsAsFactors = FALSE
)

cat("\n")

# ============================================================================
# 第四部分：筛选操作测试
# ============================================================================
cat("=== 第4部分：筛选操作 ===\n\n")

# 11. filter_cells (单条件)
# 默认在 obs 表添加 filter_cells_1 列（与Python版本一致）
cat("[4-1] filter_cells (min_counts=100)...\n")
t <- time_it("filter_cells", {
  a_filt <- filter_cells(a, min_counts = 100)
  nrow(obs(a_filt))
})
results[[length(results) + 1]] <- data.frame(
  category = "Filtering",
  test = t$name,
  seconds = t$seconds,
  stringsAsFactors = FALSE
)

# 验证 filter_cells_1 列存在
check_col <- DBI::dbGetQuery(a$con, "
  SELECT COUNT(*) as cnt FROM information_schema.columns
  WHERE table_name = 'obs' AND column_name = 'filter_cells_1'
")
if (check_col$cnt[1] > 0) {
  cat("  -> filter_cells_1 列已创建\n")
}

# 12. filter_cells (多条件)
cat("[4-2] filter_cells (多条件)...\n")
t <- time_it("filter_cells (multi)", {
  a_filt2 <- filter_cells(a, min_counts = 500, min_genes = 200, max_counts = 50000)
  nrow(obs(a_filt2))
})
results[[length(results) + 1]] <- data.frame(
  category = "Filtering",
  test = t$name,
  seconds = t$seconds,
  stringsAsFactors = FALSE
)

# 13. filter_genes
# 默认在 var 表添加 filter_genes_1 列（与Python版本一致）
cat("[4-3] filter_genes (min_counts=10, min_cells=3)...\n")
t <- time_it("filter_genes", {
  a_filt <- filter_genes(a, min_counts = 10, min_cells = 3)
  nrow(var(a_filt))
})
results[[length(results) + 1]] <- data.frame(
  category = "Filtering",
  test = t$name,
  seconds = t$seconds,
  stringsAsFactors = FALSE
)

# 验证 filter_genes_1 列存在
check_col <- DBI::dbGetQuery(a$con, "
  SELECT COUNT(*) as cnt FROM information_schema.columns
  WHERE table_name = 'var' AND column_name = 'filter_genes_1'
")
if (check_col$cnt[1] > 0) {
  cat("  -> filter_genes_1 列已创建\n")
}

# 14. filter_genes (多条件)
cat("[4-4] filter_genes (多条件)...\n")
t <- time_it("filter_genes (multi)", {
  a_filt2 <- filter_genes(a, min_counts = 100, min_cells = 10, max_cells = 5000)
  nrow(var(a_filt2))
})
results[[length(results) + 1]] <- data.frame(
  category = "Filtering",
  test = t$name,
  seconds = t$seconds,
  stringsAsFactors = FALSE
)

cat("\n")

# ============================================================================
# 第五部分：归一化与转换测试
# ============================================================================
cat("=== 第5部分：归一化与转换 ===\n\n")
# 注意：直接在已有数据上添加新列，不删除重建数据库

# 15. normalize_total
cat("[5-1] normalize_total (target_sum=10000)...\n")
t <- time_it("normalize_total", normalize_total(a, target_sum = 10000))
results[[length(results) + 1]] <- data.frame(
  category = "Normalization",
  test = t$name,
  seconds = t$seconds,
  stringsAsFactors = FALSE
)

# 16. log1p (自然对数)
# 输出到 log1p_factor 列（与Python版本 scatlaspy 一致）
cat("[5-2] log1p (自然对数)...\n")
t <- time_it("log1p", log1p(a))
results[[length(results) + 1]] <- data.frame(
  category = "Transformation",
  test = t$name,
  seconds = t$seconds,
  stringsAsFactors = FALSE
)

# 验证 log1p_factor 列存在
check_col <- DBI::dbGetQuery(a$con, "
  SELECT COUNT(*) as cnt FROM information_schema.columns
  WHERE table_name = 'X_CSR_data' AND column_name = 'log1p_factor'
")
if (check_col$cnt[1] > 0) {
  cat("  -> log1p_factor 列已创建\n")
}

# 17. exp1 (逆转换)
# 输入 log1p_factor 列，输出 exp1_factor 列（与Python版本一致）
cat("[5-3] exp1 (逆转换)...\n")
t <- time_it("exp1", exp1(a))
results[[length(results) + 1]] <- data.frame(
  category = "Transformation",
  test = t$name,
  seconds = t$seconds,
  stringsAsFactors = FALSE
)

# 验证 exp1_factor 列存在
check_col <- DBI::dbGetQuery(a$con, "
  SELECT COUNT(*) as cnt FROM information_schema.columns
  WHERE table_name = 'X_CSR_data' AND column_name = 'exp1_factor'
")
if (check_col$cnt[1] > 0) {
  cat("  -> exp1_factor 列已创建\n")
}

# 18. normalize_and_log1p (组合操作)
cat("[5-4] normalize_and_log1p (组合)...\n")
t <- time_it("normalize_and_log1p", normalize_and_log1p(a, target_sum = 10000))
results[[length(results) + 1]] <- data.frame(
  category = "Combined",
  test = t$name,
  seconds = t$seconds,
  stringsAsFactors = FALSE
)

# 验证 exp1_factor 列存在
check_col <- DBI::dbGetQuery(a$con, "
  SELECT COUNT(*) as cnt FROM information_schema.columns
  WHERE table_name = 'X_CSR_data' AND column_name = 'exp1_factor'
")
if (check_col$cnt[1] > 0) {
  cat("  -> exp1_factor 列已创建\n")
}

cat("\n")

# ============================================================================
# 第六部分：缩放与特征选择测试
# ============================================================================
cat("=== 第6部分：缩放与特征选择 ===\n\n")

# 19. scale (Z-score)
cat("[6-1] scale (Z-score)...\n")
t <- time_it("scale", scale(a))
results[[length(results) + 1]] <- data.frame(
  category = "Scaling",
  test = t$name,
  seconds = t$seconds,
  stringsAsFactors = FALSE
)

# 20. scale (带截断)
cat("[6-2] scale (max_value=10)...\n")
t <- time_it("scale (clip)", scale(a, max_value = 10))
results[[length(results) + 1]] <- data.frame(
  category = "Scaling",
  test = t$name,
  seconds = t$seconds,
  stringsAsFactors = FALSE
)

# 21. highly_variable_genes (按方差)
# 默认在 var 表添加 highly_variable_genes 列（与Python版本一致）
cat("[6-3] highly_variable_genes (按方差, n_top=2000)...\n")
t <- time_it("HVG (var)", highly_variable_genes(a, flavor = "var", n_top = 2000))
results[[length(results) + 1]] <- data.frame(
  category = "Feature Selection",
  test = t$name,
  seconds = t$seconds,
  stringsAsFactors = FALSE
)

# 验证 highly_variable_genes 列存在
check_col <- DBI::dbGetQuery(a$con, "
  SELECT COUNT(*) as cnt FROM information_schema.columns
  WHERE table_name = 'var' AND column_name = 'highly_variable_genes'
")
if (check_col$cnt[1] > 0) {
  cat("  -> highly_variable_genes 列已创建\n")
}

# 22. highly_variable_genes (按变异系数)
cat("[6-4] highly_variable_genes (按CV, n_top=2000)...\n")
t <- time_it("HVG (cv)", highly_variable_genes(a, flavor = "cv", n_top = 2000))
results[[length(results) + 1]] <- data.frame(
  category = "Feature Selection",
  test = t$name,
  seconds = t$seconds,
  stringsAsFactors = FALSE
)

cat("\n")

# ============================================================================
# 第七部分：QC 指标计算测试
# ============================================================================
cat("=== 第7部分：QC 指标计算 ===\n\n")
# 注意：直接在已有数据上添加新列，不删除重建数据库

# 23. calculate_qc_metrics
cat("[7-1] calculate_qc_metrics...\n")
t <- time_it("calculate_qc_metrics", calculate_qc_metrics(a, mt_prefix = "MT-"))
results[[length(results) + 1]] <- data.frame(
  category = "QC Metrics",
  test = t$name,
  seconds = t$seconds,
  stringsAsFactors = FALSE
)

# 24. calculate_cell_total_counts
cat("[7-2] calculate_cell_total_counts...\n")
t <- time_it("calculate_cell_counts", calculate_cell_total_counts(a))
results[[length(results) + 1]] <- data.frame(
  category = "QC Metrics",
  test = t$name,
  seconds = t$seconds,
  stringsAsFactors = FALSE
)

# 25. calculate_gene_total_counts
cat("[7-3] calculate_gene_total_counts...\n")
t <- time_it("calculate_gene_counts", calculate_gene_total_counts(a))
results[[length(results) + 1]] <- data.frame(
  category = "QC Metrics",
  test = t$name,
  seconds = t$seconds,
  stringsAsFactors = FALSE
)

cat("\n")

# ============================================================================
# 第八部分：数据库优化测试
# ============================================================================
cat("=== 第8部分：数据库优化 ===\n\n")

# 26. atlas_optimize_settings
cat("[8-1] atlas_optimize_settings...\n")
t <- time_it("optimize_settings", atlas_optimize_settings(a))
results[[length(results) + 1]] <- data.frame(
  category = "Optimization",
  test = t$name,
  seconds = t$seconds,
  stringsAsFactors = FALSE
)

# 27. atlas_create_indexes
cat("[8-2] atlas_create_indexes...\n")
t <- time_it("create_indexes", atlas_create_indexes(a))
results[[length(results) + 1]] <- data.frame(
  category = "Optimization",
  test = t$name,
  seconds = t$seconds,
  stringsAsFactors = FALSE
)

# 28. atlas_maintain_tables
cat("[8-3] atlas_maintain_tables...\n")
t <- time_it("maintain_tables", atlas_maintain_tables(a))
results[[length(results) + 1]] <- data.frame(
  category = "Optimization",
  test = t$name,
  seconds = t$seconds,
  stringsAsFactors = FALSE
)

cat("\n")

# ============================================================================
# 第九部分：分批查询测试（仅顺序模式）
# ============================================================================
cat("=== 第9部分：分批查询 ===\n\n")

# 29. query_minibatch (顺序模式)
cat("[9-1] query_minibatch (顺序模式, batch_size=1000)...\n")
t <- time_it("query_minibatch (order)", {
  batches <- query_minibatch(a, mode = "order", batch_size = 1000,
                             callback = NULL, verbose = FALSE)
  length(batches)
})
results[[length(results) + 1]] <- data.frame(
  category = "Minibatch",
  test = t$name,
  seconds = t$seconds,
  stringsAsFactors = FALSE
)

# 30. query_minibatch (带回调)
cat("[9-2] query_minibatch (回调模式)...\n")
t <- time_it("query_minibatch (callback)", {
  n_calls <- 0
  query_minibatch(a, mode = "order", batch_size = 2000,
                  callback = function(i, adata) {
                    n_calls <<- n_calls + 1
                  }, verbose = FALSE)
  n_calls
})
results[[length(results) + 1]] <- data.frame(
  category = "Minibatch",
  test = t$name,
  seconds = t$seconds,
  stringsAsFactors = FALSE
)

cat("\n")

# ============================================================================
# 第十部分：数据导出测试
# ============================================================================
cat("=== 第10部分：数据导出 ===\n\n")

# 31. save_h5ad
cat("[10-1] save_h5ad...\n")
t <- time_it("save_h5p", save_h5ad(a, "/tmp/benchmark_export.h5ad"))
results[[length(results) + 1]] <- data.frame(
  category = "Export",
  test = t$name,
  seconds = t$seconds,
  stringsAsFactors = FALSE
)

cat("\n")

# ============================================================================
# 第十一部分：上下文管理器测试
# ============================================================================
cat("=== 第11部分：上下文管理器 ===\n\n")

# 32. with() 上下文管理器
cat("[11-1] with() 上下文管理器...\n")
t <- time_it("with() context", {
  result <- with(atlas(DATABASE_NAME, path = DB_PATH, mode = "r"), {
    nrow(obs(.))
  })
  result
})
results[[length(results) + 1]] <- data.frame(
  category = "Context Manager",
  test = t$name,
  seconds = t$seconds,
  stringsAsFactors = FALSE
)

cat("\n")

# ============================================================================
# 清理与结果保存
# ============================================================================
cat("=== 清理 ===\n")
atlas_close(a)

# 删除测试数据库
delete_atlas(DATABASE_NAME, path = DB_PATH)
if (file.exists("/tmp/benchmark_export.h5ad")) {
  file.remove("/tmp/benchmark_export.h5ad")
}

# 合并结果
results_df <- do.call(rbind, results)

# 保存结果
output_file <- "/home/senpeng/scatlas_R/scAtlasAnalysis-R-version/benchmark_R_results.csv"
write.csv(results_df, output_file, row.names = FALSE)

cat("\n==============================================\n")
cat("           测试完成！\n")
cat("==============================================\n\n")

# 按类别汇总
cat("按类别汇总:\n")
summary_df <- aggregate(seconds ~ category, data = results_df, FUN = sum)
summary_df <- summary_df[order(summary_df$seconds, decreasing = TRUE), ]
print(summary_df)

cat(sprintf("\n结果已保存到: %s\n", output_file))
cat("\n详细结果:\n")
print(results_df)
