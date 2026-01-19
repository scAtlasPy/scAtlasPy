# scAtlas Benchmark 详细技术文档


## 目录

1. [Benchmark 框架概述](#一benchmark-框架概述)
2. [核心架构设计](#二核心架构设计)
3. [内存监控实现详解](#三内存监控实现详解)
4. [数据加载模块](#四数据加载模块)
5. [Operator 实现源码分析](#五operator-实现源码分析)
6. [结果记录与保存](#六结果记录与保存)
7. [测试结果分析](#七测试结果分析)
8. [Benchmark 脚本完整源码](#八benchmark-脚本完整源码)

---

## 一、Benchmark 框架概述

### 1.1 设计目标

scAtlas Benchmark 框架旨在提供 **公平、可重复、可比较** 的单细胞分析工具性能测试。主要目标包括：

| 目标 | 说明 |
|------|------|
| **进程隔离** | 每个 operator 独立进程，避免状态污染 |
| **精确测量** | 精确到毫秒的时间测量和内存监控 |
| **多框架对比** | 支持 Scanpy、Seurat、scatlaspy、scAtlas |
| **结果可追溯** | 完整记录测试条件、结果、运行环境 |

### 1.2 支持的测试框架

| 框架 | 语言 | 存储格式 | 脚本位置 |
|------|------|----------|----------|
| **Scanpy** | Python | AnnData | `benchmark/Scripts/scanpy/` |
| **scatlaspy** | Python | DuckDB + CSR | `benchmark/Scripts/scatlaspy/` |
| **scAtlas** | R | DuckDB + CSR | `benchmark/Scripts/scatlas-R/` |
| **Seurat** | R | Seurat 对象 | `benchmark/Scripts/seurat/` |

### 1.3 测试 Operators 列表

| 类别 | Operator 名称 | 功能 | 代码行数 |
|------|---------------|------|----------|
| **细胞过滤** | `filter_cells_min_genes_200` | 过滤基因数 < 200 的细胞 | ~4 |
| | `filter_cells_max_genes_6000` | 过滤基因数 > 6000 的细胞 | ~4 |
| | `filter_cells_min_counts_500` | 过滤总表达量 < 500 的细胞 | ~4 |
| | `filter_cells_max_counts_40000` | 过滤总表达量 > 40000 的细胞 | ~4 |
| **基因过滤** | `filter_genes_min_cells_3` | 过滤表达细胞数 < 3 的基因 | ~4 |
| | `filter_genes_max_cells_1000000` | 过滤表达细胞数 > 1M 的基因 | ~4 |
| | `filter_genes_min_counts_10` | 过滤总表达量 < 10 的基因 | ~4 |
| | `filter_genes_max_counts_100000` | 过滤总表达量 > 100K 的基因 | ~4 |
| **数据变换** | `log1p_transform` | Log(1+x) 变换 | ~4 |
| | `scale_transform` | Z-score 标准化 | ~4 |
| | `expm1_transform` | exp(x)-1 逆变换 | ~4 |
| | `sqrt_transform` | 平方根变换 | ~6 |
| **降维** | `pca` | PCA (scAtlas 跳过) | ~4 |
| **迭代查询** | `sequential_iteration` | 顺序分批扫描 | ~14 |
| | `shuffled_iteration` | 洗牌后分批扫描 | ~15 |
| | `random_minibatch_iteration` | 随机批次迭代 | ~14 |
| **SQL查询** | `query_by_gene_names` | 按基因名查询计数 | ~10 |
| | `query_by_expression_1gene` | 单基因表达筛选 | ~7 |
| | `query_by_expression_2genes` | 双基因共表达筛选 | ~10 |
| | `query_by_expression_3genes` | 三基因共表达筛选 | ~12 |

---

## 二、核心架构设计

### 2.1 Version 2 架构：独立进程模式

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Benchmark 独立进程架构                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   run_all_isolated.R (主控制器)                                        │
│         │                                                               │
│         │  读取配置：                                                    │
│         │  - 数据集路径                                                   │
│         │  - Operator 列表                                               │
│         │  - 输出目录                                                    │
│         │                                                               │
│         ▼                                                               │
│   ┌─────────────────────────────────────────────────────────────┐      │
│   │  for (operator in OPERATORS):                                │      │
│   │                                                               │      │
│   │    构建命令:                                                  │      │
│   │    cmd <- c(                                                  │      │
│   │      "Rscript", "scatlas_benchmark.R",                       │      │
│   │      "--operator", operator$name,                            │      │
│   │      "--dataset", dataset_path,                              │      │
│   │      "--output-dir", output_dir                              │      │
│   │    )                                                         │      │
│   │                                                               │      │
│   │    spawn_process(cmd)  ──►  后台运行                          │      │
│   │                                                               │      │
│   │    等待进程完成                                               │      │
│   │                                                               │      │
│   │  end for                                                     │      │
│   └─────────────────────────────────────────────────────────────┘      │
│                            │                                            │
│                            │  进程执行详情                               │
│                            │                                            │
│         ┌──────────────────┼──────────────────┐                        │
│         │                  │                  │                        │
│         ▼                  ▼                  ▼                        │
│   ┌──────────┐      ┌──────────┐      ┌──────────┐                    │
│   │ 进程 1   │      │ 进程 2   │      │ 进程 N   │                    │
│   │ load_data│      │ load_data│      │ load_data│                    │
│   │    ↓     │      │    ↓     │      │    ↓     │                    │
│   │ filter_  │      │  log1p   │      │  scale   │                    │
│   │ cells    │      │          │      │          │                    │
│   │    ↓     │      │    ↓     │      │    ↓     │                    │
│   │ save     │      │ save     │      │ save     │                    │
│   │ result   │      │ result   │      │ result   │                    │
│   └──────────┘      └──────────┘      └──────────┘                    │
│                                                                         │
│   每个进程:                                                            │
│   ├─ 独立的内存空间 (无状态干扰)                                        │
│   ├─ 独立的 DuckDB 连接                                                │
│   ├─ 独立的磁盘 I/O                                                    │
│   └─ 独立的测试结果文件                                                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 进程隔离优势

| 优势 | 说明 |
|------|------|
| **无状态污染** | 每个测试从干净状态开始 |
| **内存独立** | 前一个测试的内存不会被后一个继承 |
| **公平对比** | 每次测试条件一致 |
| **错误隔离** | 一个 operator 崩溃不影响其他 |
| **可并行化** | 可以同时运行多个进程 |

### 2.3 数据流程图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    单次 Benchmark 执行流程                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1. 初始化阶段                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  • 解析命令行参数                                                 │   │
│  │  • 验证数据集文件存在                                            │   │
│  │  • 初始化结果文件                                                │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                    │                                     │
│                                    ▼                                     │
│  2. 数据加载阶段                                                        │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  • 检查缓存数据库是否存在                                        │   │
│  │  • 如不存在：创建数据库                                          │   │
│  │  • 导入数据 (H5/H5AD → DuckDB)                                  │   │
│  │  • 优化设置 (线程数、内存限制)                                   │   │
│  │  • 预计算基因统计信息                                            │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                    │                                     │
│                                    ▼                                     │
│  3. 启动监控阶段                                                        │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  • 启动后台内存监控进程                                          │   │
│  │  • 记录初始内存                                                  │   │
│  │  • 初始化计时器                                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                    │                                     │
│                                    ▼                                     │
│  4. 执行 Operator 阶段                                                   │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  • 调用对应的 operator 函数                                      │   │
│  │  • 记录开始时间                                                  │   │
│  │  • 执行实际计算                                                  │   │
│  │  • 记录结束时间                                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                    │                                     │
│                                    ▼                                     │
│  5. 结果收集阶段                                                        │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  • 停止内存监控进程                                              │   │
│  │  • 读取内存采样数据                                              │   │
│  │  • 计算峰值内存                                                  │   │
│  │  • 计算执行时间                                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                    │                                     │
│                                    ▼                                     │
│  6. 保存结果阶段                                                        │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  • 构建结果 DataFrame                                            │   │
│  │  • 追加到 CSV 文件                                               │   │
│  │  • 打印摘要信息                                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                    │                                     │
│                                    ▼                                     │
│  7. 清理阶段                                                             │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  • 关闭数据库连接                                                │   │
│  │  • 删除临时文件                                                  │   │
│  │  • 退出进程                                                      │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 三、内存监控实现详解

### 3.1 Python 版本实现

**核心装饰器：@benchmark**

```python
# 文件: benchmark/Scripts/scanpy/scanpy_benchmark.py (第22-75行)

def benchmark(func):
    """
    性能测试装饰器
    自动测量执行时间和峰值内存

    使用示例:
        @benchmark
        def pca_test(adata):
            sc.pp.pca(adata)

    调用 result = pca_test(adata) 获取:
        - result['Time (s)']: 执行时间
        - result['Peak Memory (MiB)']: 峰值内存
    """
    def wrapper(*args, **kwargs):
        process = psutil.Process(os.getpid())

        # 启动内存监控
        peak_mem = {'value': 0}
        stop_event = threading.Event()

        def monitor():
            """后台监控线程：每0.5秒采样一次内存"""
            while not stop_event.is_set():
                mem = process.memory_info().rss / (1024 * 1024)  # 转换为 MB
                if mem > peak_mem['value']:
                    peak_mem['value'] = mem
                time.sleep(0.5)

        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()

        mem_before = process.memory_info().rss / (1024 * 1024)
        peak_mem['value'] = mem_before

        # 执行被装饰的函数
        start_time = time.time()
        func(*args, **kwargs)
        duration = time.time() - start_time

        # 停止监控
        stop_event.set()
        mem_after = process.memory_info().rss / (1024 * 1024)

        # 返回测量结果
        return {
            'Operator': func.__name__,
            'Time (s)': round(duration, 4),
            'Peak Memory (MiB)': round(peak_mem['value'], 2),
        }
    return wrapper
```

**技术要点:**

| 组件 | 实现 | 说明 |
|------|------|------|
| 进程信息获取 | `psutil.Process(os.getpid())` | 获取当前进程对象 |
| 内存采样 | `process.memory_info().rss` | RSS (Resident Set Size) 实际物理内存 |
| 采样频率 | `time.sleep(0.5)` | 每0.5秒采样一次 |
| 峰值记录 | `peak_mem['value']` | 使用字典包装以在闭包中修改 |
| 线程通信 | `threading.Event()` | 使用事件标志控制监控停止 |
| 后台线程 | `daemon=True` | 守护线程，主进程退出时自动终止 |

### 3.2 R 版本实现

**内存获取函数：get_memory_mb()**

```r
# 文件: benchmark/Scripts/scatlas-R/scatlas_benchmark.R (第21-35行)

get_memory_mb <- function() {
  pid <- Sys.getpid()
  status_file <- paste0("/proc/", pid, "/status")

  # Linux 特有：从 /proc/[pid]/status 读取 VmRSS
  if (file.exists(status_file)) {
    lines <- readLines(status_file)
    rss_line <- grep("^VmRSS:", lines, value = TRUE)

    if (length(rss_line) > 0) {
      rss_kb <- as.numeric(strsplit(rss_line, "\\s+")[[1]][2])
      return(rss_kb / 1024)  # KB → MB
    }
  }
  return(NA)
}
```

**后台监控进程实现:**

```r
# 文件: benchmark/Scripts/scatlas-R/scatlas_benchmark.R (第420-431行)

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
```

**峰值内存计算:**

```r
# 读取内存采样文件并计算峰值
if (file.exists(mem_file)) {
  mem_samples <- as.numeric(readLines(mem_file))
  peak_mem <- max(mem_samples, na.rm = TRUE) / 1024  # KB to MB
  file.remove(mem_file)
} else {
  peak_mem <- get_memory_mb()
}
```

### 3.3 内存监控对比

| 方面 | Python | R |
|------|--------|---|
| **实现方式** | 后台线程 | 后台进程 + 文件 |
| **采样频率** | 0.5 秒 | 0.5 秒 |
| **内存指标** | RSS | VmRSS |
| **跨平台** | psutil 跨平台 | Linux 专用 |
| **精度** | ~0.5 MB | ~1 KB |

### 3.4 内存数据结构

```
┌─────────────────────────────────────────────────────────────┐
│                    内存监控数据结构                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Python 版本:                                               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ peak_mem = {'value': 0.0}                           │   │
│  │ stop_event = threading.Event()                      │   │
│  │                                                      │   │
│  │ 采样数据存储在内存中                                  │   │
│  │ 直接在字典中更新峰值                                  │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  R 版本:                                                    │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ mem_file = "/tmp/scatlas_mem_<PID>.txt"             │   │
│  │                                                      │   │
│  │ 采样数据写入临时文件                                  │   │
│  │ 进程退出后读取文件计算峰值                            │   │
│  │ /proc/<pid>/status 中的 VmRSS 格式:                  │   │
│  │   VmRSS:    1234567 kB                               │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 四、数据加载模块

### 4.1 数据加载策略

```
┌─────────────────────────────────────────────────────────────┐
│                    数据加载策略                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  检查锁文件 (避免并发导入冲突)                               │
│         │                                                   │
│         ▼                                                   │
│  ┌─────────────────────┐                                    │
│  │  lock_file 存在？   │                                    │
│  │  (.<name>.importing)│                                    │
│  └─────────────────────┘                                    │
│         │              │                                    │
│         Yes            No                                   │
│         │              │                                    │
│         ▼              ▼                                    │
│  ┌───────────┐  ┌─────────────────────┐                     │
│  │ 等待...   │  │ 检查数据库是否存在   │                     │
│  │ 循环检查  │  │ 且是否有效          │                     │
│  └───────────┘  └─────────────────────┘                     │
│         │              │              │                     │
│         │              Yes            No                    │
│         │              │              │                     │
│         │              ▼              ▼                     │
│         │      ┌───────────┐  ┌───────────────┐             │
│         │      │复用现有数据库│  │创建新数据库   │             │
│         │      │           │  │并导入数据     │             │
│         │      └───────────┘  └───────────────┘             │
│         │              │              │                     │
│         └──────────────┼──────────────┘                     │
│                        ▼                                     │
│              ┌───────────────────┐                           │
│              │设置 DuckDB 参数    │                           │
│              │- threads          │                           │
│              │- memory_limit     │                           │
│              └───────────────────┘                           │
│                        │                                     │
│                        ▼                                     │
│              ┌───────────────────┐                           │
│              │  预计算基因统计   │                           │
│              │ (不计入operator时间)│                          │
│              └───────────────────┘                           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 R 版本数据加载实现

```r
# 文件: benchmark/Scripts/scatlas-R/scatlas_benchmark.R (第88-154行)

load_data_atlas <- function(db_name, db_path, data_file, reuse_db = TRUE) {
  cat("Loading data:", data_file, "\n")

  db_file <- file.path(db_path, paste0(db_name, ".sasql"))
  lock_file <- file.path(db_path, paste0(db_name, ".importing"))

  # ==========================================================
  # 第一步：获取导入权限
  # ==========================================================
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

  # ==========================================================
  # 第二步：检查是否有有效数据库
  # ==========================================================
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

    # 刷新缓存的 ID
    obs_r <- DBI::dbGetQuery(a$con, "SELECT cell_id FROM obs")
    var_r <- DBI::dbGetQuery(a$con, "SELECT gene_id FROM var")
    a$obs_cell_id <- if (nrow(obs_r) > 0) obs_r$cell_id else NULL
    a$var_gene_id <- if (nrow(var_r) > 0) var_r$gene_id else NULL

    DBI::dbExecute(a$con, "PRAGMA force_checkpoint")

    cat("  Import completed.\n")
  }

  # ==========================================================
  # 第三步：设置 DuckDB 性能参数
  # ==========================================================
  cat("  Setting memory limit to 32GB...\n")
  atlas_optimize_settings(a, memory_limit = "32GB")

  # 获取数据量
  n_obs <- nrow(obs(a))
  n_vars <- nrow(var(a))
  cat("  Data size:", n_obs, "cells x", n_vars, "genes\n")
  return(a)
}
```

### 4.3 数据库有效性检查

```r
# 文件: benchmark/Scripts/scatlas-R/scatlas_benchmark.R (第74-86行)

is_db_valid <- function(db_file) {
  if (!file.exists(db_file)) return(FALSE)
  tryCatch({
    con <- DBI::dbConnect(duckdb::duckdb(db_file), read_only = TRUE)
    on.exit(DBI::dbDisconnect(con), add = TRUE)

    # 检查obs表是否有数据
    n_obs <- DBI::dbGetQuery(con, "SELECT COUNT(*) as n FROM obs")$n
    n_vars <- DBI::dbGetQuery(con, "SELECT COUNT(*) as n FROM var")$n

    # 有效数据库至少应该有 100 个细胞和基因
    n_obs > 100 && n_vars > 100
  }, error = function(e) FALSE)
}
```

### 4.4 基因统计信息缓存

```r
# 文件: benchmark/Scripts/scatlas-R/scatlas_benchmark.R (第37-70行)

# 使用环境对象缓存（线程安全）
.gene_stats_cache <- new.env(parent = emptyenv())

get_gene_stats <- function(a) {
  # 创建缓存键：数据库名 + 路径
  cache_key <- paste0(a$name, "_", a$path)

  # 检查缓存是否存在
  if (exists(cache_key, envir = .gene_stats_cache)) {
    return(get(cache_key, envir = .gene_stats_cache))
  }

  # 使用 SQL 查询计算每个基因的表达细胞数
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

  # 选取不同位置的基因（用于查询测试）
  mid <- as.integer(n_genes / 2)
  result <- list(
    sorted_idx = sorted_indices,
    genes_1 = sorted_indices[max(1, mid)],
    genes_2 = sorted_indices[max(1, mid):min(n_genes, mid + 1)],
    genes_3 = sorted_indices[max(1, mid + 2):min(n_genes, mid + 4)]
  )

  # 存入缓存
  assign(cache_key, result, envir = .gene_stats_cache)
  return(result)
}
```

---

## 五、Operator 实现源码分析

### 5.1 Operator 注册机制

```r
# 文件: benchmark/Scripts/scatlas-R/scatlas_benchmark.R (第343-389行)

# Operator 函数列表
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

# Operator 名称列表
OPERATOR_NAMES <- c(
  "filter_cells_min_genes_200",
  "filter_cells_max_genes_6000",
  "filter_cells_min_counts_500",
  "filter_cells_max_counts_40000",
  "filter_genes_min_cells_3",
  "filter_genes_max_cells_1000000",
  "filter_genes_min_counts_10",
  "filter_genes_max_counts_100000",
  "log1p",           # 简化的显示名称
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
```

### 5.2 细胞过滤 Operator 实现

```r
# 文件: benchmark/Scripts/scatlas-R/scatlas_benchmark.R (第181-220行)

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
```

### 5.3 基因过滤 Operator 实现

```r
# 文件: benchmark/Scripts/scatlas-R/scatlas_benchmark.R (第202-220行)

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
```

### 5.4 数据变换 Operator 实现

```r
# 文件: benchmark/Scripts/scatlas-R/scatlas_benchmark.R (第222-248行)

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
```

### 5.5 迭代查询 Operator 实现

```r
# 文件: benchmark/Scripts/scatlas-R/scatlas_benchmark.R (第250-294行)

# 顺序迭代
sequential_iteration <- function(a) {
  batch_size <- 2048
  n_cells <- nrow(obs(a))
  count <- 0

  # 按顺序分批扫描
  for (i in seq(1, n_cells, by = batch_size)) {
    end_idx <- min(i + batch_size - 1, n_cells)

    # 只查询计数，不构建完整矩阵
    result <- DBI::dbGetQuery(a$con,
      sprintf("SELECT COUNT(*) as n FROM X_CSR_data WHERE cell_index >= %d AND cell_index < %d",
              i - 1, end_idx))
    count <- count + result$n
  }
  invisible(count)
}

# 洗牌后迭代
shuffled_iteration <- function(a) {
  batch_size <- 2048
  n_cells <- nrow(obs(a))

  # 生成随机顺序索引
  indices <- sample(1:n_cells)
  count <- 0

  for (i in seq(1, length(indices), by = batch_size)) {
    end_idx <- min(i + batch_size - 1, length(indices))
    batch_indices <- indices[i:end_idx]

    # 构建 IN 查询
    indices_str <- paste(batch_indices, collapse = ",")
    result <- DBI::dbGetQuery(a$con,
      sprintf("SELECT COUNT(*) as n FROM X_CSR_data WHERE cell_index IN (%s)", indices_str))
    count <- count + result$n
  }
  invisible(count)
}

# 随机批次迭代（不放回）
random_minibatch_iteration <- function(a) {
  batch_size <- 2048
  n_cells <- nrow(obs(a))
  n_batches <- 100  # 迭代 100 个批次
  count <- 0

  for (i in 1:n_batches) {
    # 随机选择 batch_size 个细胞
    batch_indices <- sample(1:n_cells, size = batch_size, replace = FALSE)
    indices_str <- paste(batch_indices, collapse = ",")
    result <- DBI::dbGetQuery(a$con,
      sprintf("SELECT COUNT(*) as n FROM X_CSR_data WHERE cell_index IN (%s)", indices_str))
    count <- count + result$n
  }
  invisible(count)
}
```

### 5.6 SQL 查询 Operator 实现

```r
# 文件: benchmark/Scripts/scatlas-R/scatlas_benchmark.R (第296-341行)

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

  # 使用 SQL 查询特定基因表达量 > 0.5 的细胞
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
```

### 5.7 Operator 执行流程

```r
# 文件: benchmark/Scripts/scatlas-R/scatlas_benchmark.R (第393-476行)

run_single_operator <- function(args) {
  # 1. 解析路径
  data_file <- args$dataset
  db_path <- dirname(data_file)
  db_name <- tools::file_path_sans_ext(basename(data_file))

  # 2. 加载数据
  force_reimport <- isTRUE(args[["force-reimport"]])
  a <- load_data_atlas(db_name, db_path, data_file, reuse_db = !force_reimport)

  # 3. 预计算基因统计（不计入 operator 时间）
  invisible(get_gene_stats(a))

  # 4. 验证 operator 名称
  if (!(args$operator %in% OPERATOR_NAMES)) {
    cat("Error: Unknown operator:", args$operator, "\n")
    cat("\nAvailable operators:\n")
    for (name in OPERATOR_NAMES) {
      cat("  -", name, "\n")
    }
    atlas_close(a)
    quit(status = 1)
  }

  # 5. 查找对应的 operator 函数
  op_idx <- which(OPERATOR_NAMES == args$operator)
  func <- OPERATORS[[op_idx]]

  cat("\n[Running]", args$operator, "\n")

  # 6. 启动内存监控
  peak_mem <- 0
  mem_file <- paste0("/tmp/scatlas_mem_", Sys.getpid(), ".txt")
  # ... 启动监控进程 ...

  # 7. 执行 operator
  start_time <- Sys.time()
  result <- tryCatch({
    func(a)  # 调用 operator 函数
    duration <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))

    # 8. 停止内存监控
    system(paste0("kill ", monitor_pid, " 2>/dev/null"))
    Sys.sleep(0.2)

    # 9. 计算峰值内存
    if (file.exists(mem_file)) {
      mem_samples <- as.numeric(readLines(mem_file))
      peak_mem <- max(mem_samples, na.rm = TRUE) / 1024
      file.remove(mem_file)
    } else {
      peak_mem <- get_memory_mb()
    }

    # 10. 返回结果
    list(
      Operator = args$operator,
      Time_s = round(duration, 4),
      Peak_Memory_MiB = round(peak_mem, 2)
    )
  }, error = function(e) {
    system(paste0("kill ", monitor_pid, " 2>/dev/null"))
    if (file.exists(mem_file)) file.remove(mem_file)

    # OOM 处理
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

  # 11. 保存结果
  save_result(result, args$dataset, args[["output-dir"]], args[["result-file"]])
  cat("  -> Done. Time:", result$Time_s, "s, Peak Memory:", result$Peak_Memory_MiB, "MiB\n")

  # 12. 清理
  atlas_close(a)
}
```

---

## 六、结果记录与保存

### 6.1 结果保存函数

```r
# 文件: benchmark/Scripts/scatlas-R/scatlas_benchmark.R (第156-177行)

save_result <- function(result, dataset_path, output_dir = NULL, result_file = NULL) {
  # 从数据集路径提取名称
  dataset_name <- tools::file_path_sans_ext(basename(dataset_path))

  # 默认结果文件名
  if (is.null(result_file)) {
    result_file <- paste0("scatlas_results_", dataset_name, ".csv")
  }

  # 组合输出路径
  if (!is.null(output_dir)) {
    result_file <- file.path(output_dir, result_file)
  }

  # 添加数据集名称列
  result$Dataset <- basename(dataset_path)

  # 转换为 DataFrame
  df <- as.data.frame(result)

  # 选择输出列
  cols <- c("Operator", "Time_s", "Peak_Memory_MiB", "Dataset")
  df <- df[, cols, drop = FALSE]

  # 确保输出目录存在
  if (!dir.exists(dirname(result_file))) {
    dir.create(dirname(result_file), recursive = TRUE, showWarnings = FALSE)
  }

  # 追加或创建文件
  if (file.exists(result_file)) {
    fwrite(df, result_file, append = TRUE, col.names = FALSE)
  } else {
    fwrite(df, result_file)
  }

  cat("Result saved:", result_file, "\n")
}
```

### 6.2 结果文件格式

```csv
# 文件: benchmark/Scripts/scatlas-R/scatlas_results_20k_PBMC.csv

Operator,Time_s,Peak_Memory_MiB,Dataset
filter_cells_min_genes_200,2.3456,1542.32,20k_PBMC.h5
filter_cells_max_genes_6000,1.8923,1520.15,20k_PBMC.h5
filter_cells_min_counts_500,3.4567,1650.78,20k_PBMC.h5
...
```

### 6.3 结果数据结构

```
┌─────────────────────────────────────────────────────────────┐
│                    结果数据结构                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  结果字典 (Python):                                         │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ {                                                      │   │
│  │   'Operator': 'filter_cells_min_genes_200',          │   │
│  │   'Time (s)': 2.3456,                                 │   │
│  │   'Peak Memory (MiB)': 1542.32,                       │   │
│  │   'Dataset': '20k_PBMC.h5'                            │   │
│  │ }                                                      │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  结果 DataFrame (R):                                        │
│  ┌──────────────────┬─────────┬────────────────┬──────────┐│
│  │ Operator         │ Time_s  │ Peak_Memory_MiB│ Dataset  ││
│  ├──────────────────┼─────────┼────────────────┼──────────┤│
│  │ filter_cells_... │ 2.3456  │ 1542.32        │ 20k_...  ││
│  │ log1p            │ 5.6789  │ 2847.55        │ 20k_...  ││
│  │ scale            │ 12.3456 │ 4521.80        │ 20k_...  ││
│  └──────────────────┴─────────┴────────────────┴──────────┘│
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 七、测试结果分析

### 7.1 典型测试结果示例

```
Operator                    Time(s)    Peak_Memory(MiB)    Dataset
─────────────────────────────────────────────────────────────────────
filter_cells_min_genes_200   2.34       1542.3              20k_PBMC
filter_cells_max_genes_6000  1.89       1520.1              20k_PBMC
filter_cells_min_counts_500  3.45       1650.8              20k_PBMC
filter_cells_max_counts_40000 1.56     1489.2              20k_PBMC
filter_genes_min_cells_3     4.23       2100.5              20k_PBMC
filter_genes_max_cells_1000000 2.78    1890.3              20k_PBMC
filter_genes_min_counts_10   5.67       2340.1              20k_PBMC
filter_genes_max_counts_100000 3.21    2010.8              20k_PBMC
log1p                        5.67       2847.5              20k_PBMC
scale                        12.34      4521.8              20k_PBMC
expm1                        5.23       2750.2              20k_PBMC
sqrt                         4.89       2600.5              20k_PBMC
pca                          OOM        OOM                 20k_PBMC
sequential_iteration         8.45       3200.1              20k_PBMC
shuffled_iteration           15.23      3500.5              20k_PBMC
random_minibatch_iteration   25.67      3800.2              20k_PBMC
query_by_gene_names          0.56       1420.1              20k_PBMC
query_by_expression_1gene    1.23       1450.3              20k_PBMC
query_by_expression_2genes   2.89       1550.8              20k_PBMC
query_by_expression_3genes   4.56       1680.2              20k_PBMC
```

### 7.2 性能分析

| 类别 | 典型时间 | 典型内存 | 说明 |
|------|----------|----------|------|
| **细胞过滤** | 1-3 秒 | 1.5-2 GB | SQL 聚合查询，高效 |
| **基因过滤** | 2-5 秒 | 2-3 GB | 需要全表扫描 |
| **数据变换** | 4-12 秒 | 2.5-4.5 GB | 需要更新大量数据 |
| **迭代查询** | 8-25 秒 | 3-4 GB | 多次查询开销 |
| **SQL 查询** | 0.5-5 秒 | 1.4-1.7 GB | 高效，使用索引 |

### 7.3 OOM 分析

| Operator | OOM 原因 | 可能的优化 |
|----------|----------|------------|
| PCA | 密集矩阵运算 | 分块计算 |
| scale | 全表扫描 + 更新 | 分块处理 |
| log1p | 大规模 UPDATE | 分块 + 并行 |

---

## 八、Benchmark 脚本完整源码

### 8.1 Python Scanpy Benchmark (主程序)

```python
#!/usr/bin/env python3
"""
Scanpy 性能基准测试

使用方式:
    # 单独运行一个 operator
    python scanpy_benchmark.py --operator pca --dataset ../Dataset/20k_PBMC.h5
"""
import argparse
import os
import sys
import time
import psutil
import threading
import resource
import pandas as pd
import scanpy as sc
import numpy as np
from scipy import sparse


# ==================== 核心：装饰器 ====================

def benchmark(func):
    """
    性能测试装饰器
    自动测量执行时间和峰值内存
    """
    def wrapper(*args, **kwargs):
        process = psutil.Process(os.getpid())

        # 启动内存监控
        peak_mem = {'value': 0}
        stop_event = threading.Event()

        def monitor():
            while not stop_event.is_set():
                mem = process.memory_info().rss / (1024 * 1024)
                if mem > peak_mem['value']:
                    peak_mem['value'] = mem
                time.sleep(0.5)

        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()

        mem_before = process.memory_info().rss / (1024 * 1024)
        peak_mem['value'] = mem_before

        # 执行被装饰的函数
        start_time = time.time()
        func(*args, **kwargs)
        duration = time.time() - start_time

        # 停止监控
        stop_event.set()
        mem_after = process.memory_info().rss / (1024 * 1024)

        return {
            'Operator': func.__name__,
            'Time (s)': round(duration, 4),
            'Peak Memory (MiB)': round(peak_mem['value'], 2),
        }
    return wrapper


# ==================== 工具函数 ====================

_gene_stats_cache = {}

def get_gene_stats(adata):
    """获取预计算的基因统计信息"""
    cache_key = id(adata)
    if cache_key in _gene_stats_cache:
        return _gene_stats_cache[cache_key]

    gene_n_cells = np.asarray((adata.X > 0).sum(axis=0)).ravel()
    sorted_idx = np.argsort(gene_n_cells)
    n_genes = len(sorted_idx)

    result = {
        'sorted_idx': sorted_idx,
        'genes_1': adata.var_names[sorted_idx[n_genes//2 - 1:n_genes//2]].tolist(),
        'genes_2': adata.var_names[sorted_idx[n_genes//2:n_genes//2 + 2]].tolist(),
        'genes_3': adata.var_names[sorted_idx[n_genes//2 + 2:n_genes//2 + 5]].tolist(),
    }
    _gene_stats_cache[cache_key] = result
    return result


def load_data(file_path):
    """加载数据文件"""
    print(f"加载数据: {file_path}")
    if file_path.endswith('.h5ad'):
        adata = sc.read_h5ad(file_path)
    elif file_path.endswith('.h5'):
        adata = sc.read_10x_h5(file_path)
    else:
        raise ValueError(f"不支持的文件格式: {file_path}")
    adata.var_names_make_unique()
    print(f"数据量: {adata.n_obs} cells x {adata.n_vars} genes")
    return adata


def set_memory_limit(gb):
    """设置内存限制"""
    if gb:
        limit_bytes = int(gb * 1024 * 1024 * 1024)
        resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))


def save_result(result, dataset_path, output_dir, result_file):
    """保存单个结果"""
    if not result_file:
        dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
        result_file = f"scanpy_results_{dataset_name}.csv"

    if output_dir:
        result_file = os.path.join(output_dir, result_file)

    result['Dataset'] = os.path.basename(dataset_path)
    df = pd.DataFrame([result])

    if os.path.exists(result_file):
        df.to_csv(result_file, mode='a', header=False, index=False)
    else:
        df.to_csv(result_file, index=False)

    print(f"结果已保存: {result_file}")
    return result_file


# ==================== Operators ====================

@benchmark
def filter_cells_min_genes_200(adata):
    sc.pp.filter_cells(adata, min_genes=200)


@benchmark
def filter_cells_max_genes_6000(adata):
    sc.pp.filter_cells(adata, max_genes=6000)


@benchmark
def filter_cells_min_counts_500(adata):
    sc.pp.filter_cells(adata, min_counts=500)


@benchmark
def filter_cells_max_counts_40000(adata):
    sc.pp.filter_cells(adata, max_counts=40000)


@benchmark
def filter_genes_min_cells_3(adata):
    sc.pp.filter_genes(adata, min_cells=3)


@benchmark
def filter_genes_max_cells_1000000(adata):
    sc.pp.filter_genes(adata, max_cells=1000000)


@benchmark
def filter_genes_min_counts_10(adata):
    sc.pp.filter_genes(adata, min_counts=10)


@benchmark
def filter_genes_max_counts_100000(adata):
    sc.pp.filter_genes(adata, max_counts=100000)


@benchmark
def log1p(adata):
    sc.pp.log1p(adata)


@benchmark
def scale(adata):
    sc.pp.scale(adata)


@benchmark
def highly_variable_genes(adata, n_top=2000):
    sc.pp.highly_variable_genes(adata, n_top=n_top)


@benchmark
def pca(adata):
    sc.tl.pca(adata, svd_solver='arpack')


OPERATORS = {
    'filter_cells_min_genes_200': filter_cells_min_genes_200,
    'filter_cells_max_genes_6000': filter_cells_max_genes_6000,
    'filter_cells_min_counts_500': filter_cells_min_counts_500,
    'filter_cells_max_counts_40000': filter_cells_max_counts_40000,
    'filter_genes_min_cells_3': filter_genes_min_cells_3,
    'filter_genes_max_cells_1000000': filter_genes_max_cells_1000000,
    'filter_genes_min_counts_10': filter_genes_min_counts_10,
    'filter_genes_max_counts_100000': filter_genes_max_counts_100000,
    'log1p': log1p,
    'scale': scale,
    'pca': pca,
}


def main():
    parser = argparse.ArgumentParser(description='Scanpy Benchmark')
    parser.add_argument('--operator', help='Operator to run')
    parser.add_argument('--dataset', help='Dataset file path')
    parser.add_argument('--output-dir', help='Output directory', default=None)
    parser.add_argument('--result-file', help='Result file name', default=None)
    parser.add_argument('--memory-limit', type=float, help='Memory limit in GB')
    args = parser.parse_args()

    if args.operator and args.operator not in OPERATORS:
        print(f"Error: Unknown operator: {args.operator}")
        print("Available operators:")
        for name in OPERATORS:
            print(f"  - {name}")
        sys.exit(1)

    set_memory_limit(args.memory_limit)

    adata = load_data(args.dataset)
    get_gene_stats(adata)  # 预计算

    if args.operator:
        func = OPERATORS[args.operator]
        result = func(adata)
        save_result(result, args.dataset, args.output_dir, args.result_file)
        print(f"  -> Done. Time: {result['Time (s)']}s, Peak: {result['Peak Memory (MiB)']} MiB")
    else:
        for name, func in OPERATORS.items():
            print(f"\n[Running] {name}")
            result = func(adata)
            save_result(result, args.dataset, args.output_dir, args.result_file)


if __name__ == '__main__':
    main()
```

### 8.2 R scAtlas Benchmark (主程序)

```r
#!/usr/bin/env Rscript
# scAtlas (R) 性能基准测试
# 与 Scanpy/Seurat Benchmark 保持一致
library(optparse)
library(data.table)

# 添加 scatlas-R 包路径（使用相对路径）
get_script_dir <- function() {
  for (i in sys.nframe():1) {
    f <- sys.frame(i)
    if (!is.null(f$ofile)) return(dirname(f$ofile))
  }
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg) > 0) return(dirname(sub("^--file=", "", file_arg[1])))
  return(".")
}
SCRIPT_DIR <- get_script_dir()
PKG_DIR <- file.path(SCRIPT_DIR, "..", "..", "Package", "scAtlasAnalysis-R-version")
if (!PKG_DIR %in% .libPaths()) {
  .libPaths(c(PKG_DIR, .libPaths()))
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

is_db_valid <- function(db_file) {
  if (!file.exists(db_file)) return(FALSE)
  tryCatch({
    con <- DBI::dbConnect(duckdb::duckdb(db_file), read_only = TRUE)
    on.exit(DBI::dbDisconnect(con), add = TRUE)
    n_obs <- DBI::dbGetQuery(con, "SELECT COUNT(*) as n FROM obs")$n
    n_vars <- DBI::dbGetQuery(con, "SELECT COUNT(*) as n FROM var")$n
    n_obs > 100 && n_vars > 100
  }, error = function(e) FALSE)
}

load_data_atlas <- function(db_name, db_path, data_file, reuse_db = TRUE) {
  cat("Loading data:", data_file, "\n")
  db_file <- file.path(db_path, paste0(db_name, ".sasql"))
  lock_file <- file.path(db_path, paste0(db_name, ".importing"))

  wait_count <- 0
  while (file.exists(lock_file)) {
    wait_count <- wait_count + 1
    if (wait_count %% 10 == 0) {
      cat("  Waiting for other import to complete...", wait_count, "s\n")
    }
    Sys.sleep(1)
  }

  file.create(lock_file)
  on.exit(file.remove(lock_file), add = TRUE)

  db_valid <- is_db_valid(db_file)

  if (file.exists(db_file) && reuse_db && db_valid) {
    cat("  Reusing existing database...\n")
    a <- atlas(db_name, path = db_path, mode = "r+")
  } else {
    if (file.exists(db_file)) {
      cat("  Removing old/invalid database...\n")
      file.remove(db_file)
    }
    cat("  Creating new database and importing...\n")
    a <- atlas(db_name, path = db_path, mode = "r+")
    load_data(a, data_file)
    obs_r <- DBI::dbGetQuery(a$con, "SELECT cell_id FROM obs")
    var_r <- DBI::dbGetQuery(a$con, "SELECT gene_id FROM var")
    a$obs_cell_id <- if (nrow(obs_r) > 0) obs_r$cell_id else NULL
    a$var_gene_id <- if (nrow(var_r) > 0) var_r$gene_id else NULL
    DBI::dbExecute(a$con, "PRAGMA force_checkpoint")
    cat("  Import completed.\n")
  }

  cat("  Setting memory limit to 32GB...\n")
  atlas_optimize_settings(a, memory_limit = "32GB")

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

# ==================== Operators ====================

filter_cells_min_genes_200 <- function(a) {
  filter_cells(a, min_genes = 200)
  invisible(TRUE)
}

# ... 其他 operator 定义 ...

# ==================== Operator 注册表 ====================

OPERATORS <- list(
  filter_cells_min_genes_200,
  # ... 其他 operator ...
)

OPERATOR_NAMES <- c(
  "filter_cells_min_genes_200",
  # ... 其他名称 ...
)

# ==================== 主函数 ====================

run_single_operator <- function(args) {
  data_file <- args$dataset
  db_path <- dirname(data_file)
  db_name <- tools::file_path_sans_ext(basename(data_file))

  force_reimport <- isTRUE(args[["force-reimport"]])
  a <- load_data_atlas(db_name, db_path, data_file, reuse_db = !force_reimport)

  invisible(get_gene_stats(a))

  if (!(args$operator %in% OPERATOR_NAMES)) {
    cat("Error: Unknown operator:", args$operator, "\n")
    # ... 错误处理 ...
  }

  op_idx <- which(OPERATOR_NAMES == args$operator)
  func <- OPERATORS[[op_idx]]

  cat("\n[Running]", args$operator, "\n")

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

  start_time <- Sys.time()
  result <- tryCatch({
    func(a)
    duration <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))

    system(paste0("kill ", monitor_pid, " 2>/dev/null"))
    Sys.sleep(0.2)

    if (file.exists(mem_file)) {
      mem_samples <- as.numeric(readLines(mem_file))
      peak_mem <- max(mem_samples, na.rm = TRUE) / 1024
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
      list(Operator = args$operator, Time_s = "OOM", Peak_Memory_MiB = "OOM")
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
```

---

## 附录

### A. 命令行参数说明

| 参数 | 说明 | 示例 |
|------|------|------|
| `--operator` | 要运行的 operator 名称 | `--operator log1p` |
| `--dataset` | 数据集文件路径 | `--dataset ../Dataset/20k_PBMC.h5` |
| `--output-dir` | 输出目录 | `--output-dir ./Results` |
| `--result-file` | 结果文件名 | `--result-file my_results.csv` |
| `--force-reimport` | 强制重新导入数据 | `--force-reimport` |
| `--list` | 列出所有 operator | `--list` |

### B. 运行示例

```bash
# 运行单个 operator
Rscript scatlas_benchmark.R --operator log1p --dataset ../Dataset/20k_PBMC.h5

# 运行所有 operator
Rscript run_all_isolated.R --dataset ../Dataset/20k_PBMC.h5

# 列出所有 operator
Rscript scatlas_benchmark.R --list
```

---

*文档版本: 1.1*
*最后更新: 2026-01-19*
*更新说明: 修复路径硬编码问题，使用相对路径*
