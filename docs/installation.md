# Installation

本页说明如何安装配置 scAtlasPy，并确认当前环境可以完整运行。下面的命令假设你已经拿到了项目代码，并且当前目录记为 `<repo>`。


## 环境要求

推荐使用 Python 3.10 或更高版本。当前源码包的运行依赖在仓库根目录的 `pyproject.toml` 中声明，主要包括：

- `duckdb`：保存和查询 Atlas 数据库。
- `scanpy`、`anndata`：读取和导出常见单细胞数据格式。
- `numpy`、`pandas`、`scipy`、`scikit-learn`：矩阵计算、表格处理和机器学习。
- `matplotlib`、`umap-learn`：可视化和 UMAP。


## 安装运行依赖

进入仓库根目录：

```bash
cd <repo>
```

以 editable mode 安装当前源码包和运行依赖：

```bash
python -m pip install -e .
```

editable mode 会让 Python 直接使用 `src/scatlaspy/` 中的源码。修改源码后通常不需要重新安装。

如果你只是临时运行源码而不安装包，也可以手动设置 `PYTHONPATH`：

```bash
export PYTHONPATH=$PWD/src:$PYTHONPATH
```

如果你不在仓库根目录，也可以写成绝对路径：

```bash
export PYTHONPATH=<repo>/src:$PYTHONPATH
```

## 检查环境

先确认 `scatlaspy` 可以导入：

```bash
python -c "import scatlaspy as sap; print(sap.Atlas)"
```

如果命令能正常输出 `sap.Atlas`，说明包已经可以被当前 Python 环境找到。

## 可选：用 PBMC3k 流程测试安装环境

PBMC3k 是一个常用的单细胞 RNA-seq 入门数据集，包含约 3,000 个外周血单个核细胞。PBMC 中通常能看到 T 细胞、B 细胞、NK 细胞、单核细胞等免疫细胞类型，因此它经常被用来测试与演示单细胞数据的质控、标准化、降维和聚类流程。

此处用于确认 scAtlasPy 是否可以在当前环境中正常运行。完成后，通常会得到一个 scAtlasPy 数据库文件和一个从数据库导出的 h5ad 文件。

你可以在 Python 脚本或 notebook 中运行一个小型 PBMC3k 流程，确认 scAtlasPy 可以创建 Atlas 数据库、导入 h5ad、执行 QC 和导出结果。运行过程中可以打印类似日志：

```text
[quickstart] load Scanpy PBMC3k dataset
[quickstart] create Atlas database
[quickstart] import h5ad into Atlas
[quickstart] calculate QC metrics
[quickstart] filter cells and genes
[quickstart] normalize and log1p
[quickstart] find highly variable genes
[quickstart] scale
[quickstart] export h5ad: data/quickstart_out.h5ad
[quickstart] done
```

看到 `[quickstart] done` 表示示例流程已经完整跑完。

示例流程通常会在当前目录创建：

```text
data/quickstart_pbmc3k.h5ad
data/quickstart_demo.sasql
data/quickstart_out.h5ad
```

其中：

- `quickstart_pbmc3k.h5ad` 是示例输入数据。
- `quickstart_demo.sasql` 是 scAtlasPy 保存分析结果的 Atlas 数据库。
- `quickstart_out.h5ad` 是从 Atlas 数据库导出的 h5ad 文件。

```{warning}
如果你的示例脚本会删除同名的 `data/quickstart_demo.sasql` 后重新创建数据库，请确认该文件不是你需要保留的分析结果。
```


## 下一步

如果安装或 quickstart 运行失败，优先检查依赖是否安装完整、包是否通过 `python -m pip install -e .` 安装到当前环境，或者 `PYTHONPATH` 是否指向 `<repo>/src`。

- 将你的数据集文件写入 Atlas 数据库，阅读 {doc}`tutorials/basic/preparing-data`。
- 继续完成标准单细胞流程：{doc}`tutorials/basic/quality-control-preprocessing` → {doc}`tutorials/basic/clustering-cell-type-annotation`。
- 学习如何画 QC、HVG、PCA、UMAP 和 marker gene 图：{doc}`tutorials/advanced/plot-parameter-guide`。
- 数据集太大时，在导入教程中选择流式导入：{doc}`tutorials/basic/preparing-data`。
- 已经熟悉 Scanpy 的用户，查看迁移对照：{doc}`how-to/migrate-from-other-platforms`。
