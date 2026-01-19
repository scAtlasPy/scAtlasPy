# scAtlas - 单细胞图谱数据管理工具

> 一个基于 DuckDB + CSR 稀疏矩阵的高性能单细胞数据分析框架

---

## 简介

scAtlas 是一个专为大规模单细胞转录组数据设计的分析与管理框架。通过创新的存储架构（DuckDB + CSR 稀疏矩阵），实现了超大规模数据集的高效存储和快速查询，支持 Python 和 R 双语言接口。

## 项目结构

```
scAtlas/
├── Package/                          # 核心代码包
│   ├── python-version-scAtlasAnalysis/   # Python 实现 (scatlaspy)
│   │   └── scatlaspy/
│   │       ├── data/          # Atlas 核心类
│   │       ├── preprocessing/ # 预处理模块
│   │       ├── io/            # 输入输出模块
│   │       ├── tools/         # 工具函数
│   │       └── plots/         # 可视化模块
│   │
│   └── scAtlasAnalysis-R-version/      # R 实现
│       └── R/
│           ├── atlas.R        # Atlas 核心类
│           ├── load_data.R    # 数据加载
│           ├── filter.R       # 过滤操作
│           └── transform.R    # 数据变换
│
├── benchmark/                      # 性能基准测试框架
│   ├── Scripts/                   # 测试脚本
│   │   ├── scanpy/              # Scanpy 基准测试
│   │   ├── scatlaspy/           # scatlaspy 基准测试
│   │   ├── scatlas-R/           # R scAtlas 基准测试
│   │   └── seurat/              # Seurat 基准测试
│   ├── Dataset/                  # 测试数据集
│   └── Results/                  # 测试结果
│
└── docs/                          # 文档
    ├── BENCHMARK_DETAILED.md     # 基准测试详细文档
    ├── FUNCTIONS_PYTHON.md       # Python API 文档
    ├── FUNCTIONS_R.md            # R API 文档
    └── PYTHON_R_COMPARISON.md    # Python/R 对比文档
```

## 快速开始

### 安装

```bash
# Python 版本
cd Package/python-version-scAtlasAnalysis
pip install -e Package/python-version-scAtlasAnalysis

# R 版本
cd Package/scAtlasAnalysis-R-version
R CMD INSTALL .
```

### Python 示例

```python
from scatlaspy import Atlas, load_AnnData

# 创建数据库并加载数据
atlas = Atlas("pbmc_analysis", path="./data")
load_Annata(adata, atlas)

# 数据预处理
filter_cells(atlas, min_genes=200)
normalize_total(atlas, target_sum=10000)
log1p(atlas)

# 分批查询
for adata_batch in atlas.query_minibatch(batch_size=2048):
    process(adata_batch)

atlas.close()
```

### R 示例

```r
library(scAtlas)

# 创建数据库并加载数据
atlas <- Atlas$new("pbmc_analysis", path = "./data")
load_AnnData(adata, atlas)

# 数据预处理
filter_cells(atlas, min_genes = 200)
normalize_total(atlas, target_sum = 10000)
log1p(atlas)

# 分批查询
for (batch in atlas$query_minibatch(batch_size = 2048)) {
    process(batch)
}

atlas$close()
```

## 支持的数据格式

| 格式 | 说明 | Python | R |
|------|------|--------|---|
| `.h5ad` | AnnData 格式 | ✓ | ✓ |
| `.h5` | 10X Genomics HDF5 | ✓ | ✓ |
| `.mtx` | Matrix Market | ✓ | ✓ |
| `.loom` | Loom 格式 | ✓ | - |
| `.rds` | R 序列化对象 | - | ✓ |

## 基准测试

scAtlas 提供了完整的性能对比框架，支持与主流工具的公平比较：

| 框架 | 语言 | 存储格式 | 位置 |
|------|------|----------|------|
| **scatlaspy** | Python | DuckDB + CSR | `benchmark/Scripts/scatlaspy/` |
| **scAtlas** | R | DuckDB + CSR | `benchmark/Scripts/scatlas-R/` |
| **Scanpy** | Python | AnnData | `benchmark/Scripts/scanpy/` |
| **Seurat** | R | Seurat 对象 | `benchmark/Scripts/seurat/` |

运行基准测试：(需要在run_all_isolated.py/R中自主修改数据集路径)

```bash
# Python scatlaspy
cd benchmark/Scripts/scatlaspy
python run_all_isolated.py

# R scAtlas
cd benchmark/Scripts/scatlas-R
Rscript run_all_isolated.R

# R Seurat
cd benchmark/Scripts/seurat
Rscript run_all_isolated.R
```

## 核心 API

### 数据管理 (Atlas 类)

| 方法 | 说明 |
|------|------|
| `Atlas()` | 创建数据库连接 |
| `query_minibatch()` | 分批查询数据 |
| `query_by_names()` | 按基因名查询 |
| `query_by_expression()` | 按表达量查询 |
| `close()` | 关闭连接 |

### 预处理函数

| 函数 | 说明 |
|------|------|
| `filter_cells()` | 按基因数过滤细胞 |
| `filter_genes()` | 按细胞数过滤基因 |
| `normalize_total()` | 归一化 |
| `log1p()` | Log 变换 |
| `scale()` | Z-score 标准化 |
| `highly_variable_genes()` | 识别高变基因 |

## 技术架构

### 存储设计

```
┌─────────────────────────────────────┐
│           DuckDB 数据库              │
├─────────────────────────────────────┤
│  obs (细胞元数据)                    │
│  var (基因元数据)                    │
│  X_CSR_indptr (CSR 行指针)           │
│  X_CSR_data (CSR 稀疏数据)           │
│  uns_raw (非结构化数据)              │
└─────────────────────────────────────┘
```



## 相关文档

- [分支管理规范](README.md)
- [基准测试详细文档](docs/BENCHMARK_DETAILED.md)
- [Python API 文档](docs/FUNCTIONS_PYTHON.md)
- [R API 文档](docs/FUNCTIONS_R.md)
- [Python/R 对比文档](docs/PYTHON_R_COMPARISON.md)

---

*最后更新: 2026-01-19*
