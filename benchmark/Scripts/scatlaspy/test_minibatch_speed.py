import argparse
import os
import sys
import time
import psutil
import threading
import pandas as pd
import numpy as np
import scanpy as sc
import random

# 添加 scatlaspy 到路径 - 使用相对路径
# 结构: /path/to/zspbenchmark/
#       ├── benchmark/          <- 代码仓库
#       └── Package/            <- scatlaspy 包
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# ROOT_DIR = benchmark 目录 (需要去掉两级: Scripts/scatlaspy)
ROOT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
# PARENT_DIR = zspbenchmark 目录 (去掉三级)
PARENT_DIR = os.path.dirname(ROOT_DIR)
sys.path.insert(0, os.path.join(PARENT_DIR, 'Package', 'python-version-scAtlasAnalysis'))
from scatlaspy import Atlas
from scatlaspy.preprocessing import (
    filter_cells_CSR_ultrafast,
    filter_genes_CSR,
    calculate_qc_metrics,
    normalize_total_scale_factor,
    normalize_total_new_chunked,
    log1p_chunked,
    exp1_chunked,
    normalize_and_log1p,
    highly_variable_genes,
    scale_gene_chunked,
)
from scatlaspy.io import load_AnnData

def load_data(file_path, reuse_db=True):
    """加载数据文件到 Atlas 数据库

    Args:
        file_path: 数据文件路径
        reuse_db: 如果数据库已存在，是否复用（不重新导入）
    """
    print(f"加载数据: {file_path}")
    path = os.path.dirname(file_path)
    name = os.path.splitext(os.path.basename(file_path))[0]

    db_file = os.path.join(path, f"{name}.sasql")

    # 如果数据库已存在且选择复用
    if os.path.exists(db_file) and reuse_db:
        print("  复用已存在的数据库...")
        atlas = Atlas(name, path)
        # 限制 DuckDB 内存使用 #@@
 #       atlas.connection.execute("PRAGMA memory_limit='32GB'")
    else:
        # 需要重新导入
        if os.path.exists(db_file):
            print("  删除旧数据库，重新导入...")
            os.remove(db_file)
        else:
            print("  新建数据库...")

        atlas = Atlas(name, path)
        # 限制 DuckDB 内存使用#@@
  #      atlas.connection.execute("PRAGMA memory_limit='32GB'")

        # 根据文件类型选择加载方式
        ext = os.path.splitext(file_path)[1].lower()

        if ext == '.h5':
            # 10x h5 格式
            print("  使用 scanpy 读取 10x h5 文件...")
            adata = sc.read_10x_h5(file_path)
        elif ext == '.h5ad':
            # anndata h5ad 格式
            print("  使用 scanpy 读取 h5ad 文件...")
            adata = sc.read_h5ad(file_path)
        else:
            raise ValueError(f"不支持的文件格式: {ext}")

        # 加载到 Atlas
        print("  加载到 scatlaspy 数据库...")
        load_AnnData(adata, atlas)
        print("  数据导入完成")

    # 获取数据量
    n_obs = atlas.connection.execute("SELECT COUNT(*) FROM obs").fetchone()[0]
    n_vars = atlas.connection.execute("SELECT COUNT(*) FROM var").fetchone()[0]
    print(f"  数据量: {n_obs} cells x {n_vars} genes")

    return atlas




def sequential_iteration(atlas):
    """顺序遍历数据集（直接使用 CSR 专用函数）"""
    batch_size = 2048
    total_cells = atlas.connection.execute("SELECT COUNT(*) FROM obs").fetchone()[0]
    count = 0
    # 直接调用 CSR 专用的 minibatch 函数
    for adata in atlas.minibatch_scan_order_cursor_csr_df_arrow_onlylie(total_cells, batch_size, drop_last=True):
        count += adata.n_obs
    print(f"  遍历了 {count} 个细胞")

def main():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
    dataset_path = os.path.join(ROOT_DIR, 'Dataset', '130k_thymus_atlas_HTA07.A01.v02.entire_data_raw_count.h5ad')
    atlas = load_data(dataset_path)
    start_time = time.time()
    result = sequential_iteration(atlas)
    duration = time.time() - start_time
    print(f"  顺序遍历耗时: {duration:.2f} 秒")

if __name__ == '__main__':
    main()