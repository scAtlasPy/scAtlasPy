#!/usr/bin/env python3
"""导入 h5ad 文件到 scatlaspy 数据库

用法:
    cd benchmark/Scripts/scatlaspy
    python3 import_h5ad.py

或者一行命令:
    cd benchmark/Scripts/scatlaspy
    python3 -c "
        import sys
        sys.path.insert(0, '../../Package/python-version-scAtlasAnalysis')
        from scatlaspy import Atlas
        from scatlaspy.io import load_AnnData
        import anndata as ad

        a = Atlas('test', path='.')
        adata = ad.read_h5ad('../Dataset/130k_thymus_atlas_HTA07.A01.v02.entire_data_raw_count.h5ad')
        load_AnnData(adata, a)
        print(f'cells: {len(a.obs):,}')
        print(f'genes: {len(a.var):,}')
    "
"""
import os
import sys

# 使用相对路径
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
from scatlaspy.io import load_AnnData
import anndata as ad

# 配置 - 使用相对路径
DATA_FILE = os.path.join(ROOT_DIR, 'Dataset', '130k_thymus_atlas_HTA07.A01.v02.entire_data_raw_count.h5ad')

print(f"导入数据: {DATA_FILE}")

# 创建数据库到当前目录
a = Atlas("test", path=".")
print(f"数据库创建成功: test.sasql")

# 读取 h5ad 文件
print("读取 h5ad 文件...")
adata = ad.read_h5ad(DATA_FILE)
print(f"  {adata.shape[0]:,} cells x {adata.shape[1]:,} genes")

# 导入数据
print("导入数据到数据库...")
load_AnnData(adata, a)

print("\n导入完成!")
print(f"  cells: {len(a.obs):,}")
print(f"  genes: {len(a.var):,}")
