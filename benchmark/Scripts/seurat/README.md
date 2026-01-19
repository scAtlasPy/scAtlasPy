# Seurat Benchmark

Seurat 单细胞分析性能测试框架。

## 目录结构

```
seurat/
├── seurat_benchmark.R   # 主程序（R 脚本）
├── run_all_isolated.R   # 批量运行脚本（每个 operator 独立进程）
└── README.md            # 本说明文档
```

## 快速开始

### 方式一：运行全部测试（推荐）

```bash
cd /home/senpeng/zspbenchmark/benchmark/Scripts/seurat
Rscript run_all_isolated.R
```

每个 operator 独立进程，避免相互影响。配置在脚本顶部修改。

### 方式二：运行单个 operator

```bash
Rscript seurat_benchmark.R --operator pca --dataset ../Dataset/20k_PBMC.h5
```

### 方式三：批量运行全部（在同一进程）

```bash
Rscript seurat_benchmark.R --dataset ../Dataset/20k_PBMC.h5 --run-all
```

## 命令选项

| 选项 | 说明 |
|------|------|
| `--operator` | 要运行的 operator 名称 |
| `--dataset` | 数据集文件路径 |
| `--memory-limit` | 内存限制（GB），默认 100 |
| `--output-dir` | 输出目录 |
| `--result-file` | 结果文件名 |
| `--list` | 列出所有 operators |
| `--run-all` | 运行所有 operators |

## 可用的 Operators

```bash
Rscript seurat_benchmark.R --list
```

- filter_cells_min_genes_200
- filter_cells_max_genes_6000
- filter_cells_min_counts_500
- filter_cells_max_counts_40000
- filter_genes_min_cells_3
- filter_genes_max_cells_1000000
- filter_genes_min_counts_10
- filter_genes_max_counts_100000
- log1p
- scale
- expm1
- sqrt
- pca
- sequential_iteration
- shuffled_iteration
- random_minibatch_iteration
- query_by_gene_names
- query_by_expression_1gene
- query_by_expression_2genes
- query_by_expression_3genes

## 配置说明

在 `run_all_isolated.R` 顶部修改：

```r
DATASET_PATH <- "/path/to/data.rds"  # 数据集路径
OUTPUT_DIR <- "../results"           # 输出目录
RESULT_FILE <- "seurat_results.csv"  # 结果文件名
MEMORY_LIMIT_GB <- 100               # 内存限制
```

## 输出结果

结果保存为 CSV 文件，多个测试结果追加到同一文件：

```csv
Operator,Time_s,Peak_Memory_MiB,Dataset
pca,17.84,1500.97,20k_PBMC.h5
log1p,0.15,1082.44,20k_PBMC.h5
```

## 数据集格式

| 格式 | 后缀 |
|------|------|
| Seurat Object | `.rds` |
| AnnData | `.h5ad` (需 SeuratDisk) |
| 10x Genomics | `.h5` |

## 与 Scanpy 版本的区别

| 特性 | Scanpy | Seurat |
|------|--------|--------|
| 语言 | Python | R |
| 内存监控 | psutil | gc() |
| 进程隔离 | subprocess | system2() |
| 数据结构 | AnnData | Seurat Object |
