# Scanpy Benchmark

Scanpy 单细胞分析性能测试框架。

## 目录结构

```
scanpy/
├── scanpy_benchmark.py    # 主程序（包含所有 operators 和 @benchmark 装饰器）
├── run_all_isolated.py    # 批量运行脚本（每个 operator 独立进程）
└── README.md              # 本说明文档
```

## 核心：@benchmark 装饰器

使用 `@benchmark` 装饰器，自动测量执行时间和峰值内存：

```python
@benchmark
def pca(adata):
    sc.pp.pca(adata.copy())

# 调用时返回测量结果
result = pca(adata)
# result['Time (s)'] - 执行时间
# result['Peak Memory (MiB)'] - 峰值内存
```

## 快速开始

### 方式一：运行全部测试（推荐）

```bash
cd /home/senpeng/zspbenchmark/benchmark/Scripts/scanpy
python run_all_isolated.py
```

每个 operator 独立进程，避免相互影响。配置在脚本顶部修改。

### 方式二：运行单个 operator

```bash
python scanpy_benchmark.py --operator pca --dataset ../Dataset/20k_PBMC.h5
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
python scanpy_benchmark.py --list
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
- random_minibatch_iteration(*n_batches=100*)
- query_by_gene_names
- query_by_expression_1gene
- query_by_expression_2genes
- query_by_expression_3genes

## 配置说明(已改为用相对路径)

在 `run_all_isolated.py` 顶部修改：

```python
DATASET_PATH = "/path/to/data.h5"  # 数据集路径
OUTPUT_DIR = "../results"           # 输出目录
RESULT_FILE = "scanpy_results.csv"  # 结果文件名
MEMORY_LIMIT_GB = 100               # 内存限制
```

## 输出结果

结果保存为 CSV 文件，多个测试结果追加到同一文件：

```csv
Operator,Time (s),Peak Memory (MiB),Memory Before (MiB),Memory After (MiB),Dataset
pca,17.84,1500.97,1082.0,714.89,20k_PBMC.h5
log1p,0.15,1082.44,663.74,663.75,20k_PBMC.h5
```

## 添加新的 Operator

在 `scanpy_benchmark.py` 中添加：

```python
@benchmark
def my_operator(adata):
    """我的新 operator"""
    sc.pp.highly_variable_genes(adata.copy())

# 注册到 OPERATORS 字典
OPERATORS = {
    ...
    'my_operator': my_operator,
}
```

## 数据集格式

| 格式 | 后缀 |
|------|------|
| AnnData | `.h5ad` |
| 10x Genomics | `.h5` |
