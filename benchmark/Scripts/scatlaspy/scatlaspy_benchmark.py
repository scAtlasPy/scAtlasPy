#!/usr/bin/env python3
"""
scAtlas (Python) 性能基准测试
与 Scanpy/Seurat Benchmark 保持一致

使用方式:
    # 单独运行一个 operator
    python scatlaspy_benchmark.py --operator filter_cells --dataset ../Dataset/20k_PBMC.h5

  
"""
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


# ==================== 核心：装饰器 ====================

def benchmark(func):
    """性能测试装饰器，自动测量执行时间和峰值内存"""
    def wrapper(*args, **kwargs):
        process = psutil.Process(os.getpid())
        peak_mem = {'value': 0}
        stop_event = threading.Event()

        def monitor():
            while not stop_event.is_set():
                mem = process.memory_info().rss / (1024 * 1024)
                if mem > peak_mem['value']:
                    peak_mem['value'] = mem
                time.sleep(0.05)

        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
        mem_before = process.memory_info().rss / (1024 * 1024)

        start_time = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start_time
       

        stop_event.set()
        peak_delta = peak_mem['value'] - mem_before

        thread.join(timeout=0.5)

        return {
            'Operator': func.__name__,
            'Time (s)': round(duration, 4),
            'Peak Memory (MiB)': round(peak_mem['value'], 2),
            'peak_delta':round(peak_delta,2),
        }
    return wrapper


# ==================== 工具函数 ====================

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


def save_result(result, dataset_path, output_dir, result_file):
    """保存单个结果"""
    if not result_file:
        dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
        result_file = f"scatlaspy_results_{dataset_name}.csv"

    if output_dir:
        result_file = os.path.join(output_dir, result_file)

    result['Dataset'] = os.path.basename(dataset_path)
    df = pd.DataFrame([result])

    os.makedirs(os.path.dirname(result_file) if os.path.dirname(result_file) else '.', exist_ok=True)
    if os.path.exists(result_file):
        df.to_csv(result_file, mode='a', header=False, index=False)
    else:
        df.to_csv(result_file, index=False)

    print(f"结果已保存: {result_file}")
    return result_file


# ==================== 基因统计信息缓存 ====================

_gene_stats_cache = {}


def get_gene_stats(atlas):
    """获取预计算的基因统计信息（只计算一次）

    按每个基因在多少细胞中表达来排序，选取不同位置的基因
    与 scanpy benchmark 保持一致
    """
    cache_key = id(atlas)

    if cache_key in _gene_stats_cache:
        return _gene_stats_cache[cache_key]

    print("  计算基因表达统计（首次）...")

    # 计算每个基因在多少细胞中表达（data > 0）
    # 返回按表达细胞数排序后的 gene indices
    sorted_result = atlas.connection.execute("""
        SELECT indices
        FROM X_CSR_data
        WHERE data > 0
        GROUP BY indices
        ORDER BY COUNT(DISTINCT cell_index)
    """).fetchall()

    sorted_indices = [r[0] for r in sorted_result]
    n_genes = len(sorted_indices)

    # 选取不同位置的基因，完全不重叠
    # genes_1: 中间偏左 1 个
    # genes_2: 中间 2 个
    # genes_3: 中间偏右 3 个
    result = {
        'genes_1': [sorted_indices[n_genes // 2 - 1]],  # 中间偏左 1 个
        'genes_2': sorted_indices[n_genes // 2 : n_genes // 2 + 2],  # 中间 2 个
        'genes_3': sorted_indices[n_genes // 2 + 2 : n_genes // 2 + 5],  # 中间偏右 3 个
    }

    _gene_stats_cache[cache_key] = result
    return result


# ==================== Operators (与 Scanpy/Seurat 一致) ====================

@benchmark
def filter_cells_min_genes_200(atlas):
    """过滤细胞：最少 200 个基因"""    
    filter_cells_CSR_ultrafast(atlas, min_genes=200, add_key="filter_cells_min_genes")
 
@benchmark
def filter_cells_max_genes_6000(atlas):
    """过滤细胞：最多 6000 个基因"""
    mem_before = psutil.Process().memory_info().rss / (1024*1024)
    print(f"  执行前: {mem_before:.0f} MiB")
    filter_cells_CSR_ultrafast(atlas, max_genes=6000, add_key="filter_cells_max_genes")
    mem_after = psutil.Process().memory_info().rss / (1024*1024)
    print(f"  执行后: {mem_after:.0f} MiB")


@benchmark
def filter_cells_min_counts_500(atlas):
    """过滤细胞：最少 500 个 counts"""
    mem_before = psutil.Process().memory_info().rss / (1024*1024)
    print(f"  执行前: {mem_before:.0f} MiB")
    filter_cells_CSR_ultrafast(atlas, min_counts=500, add_key="filter_cells_min_counts")
    mem_after = psutil.Process().memory_info().rss / (1024*1024)
    print(f"  执行后: {mem_after:.0f} MiB")


@benchmark
def filter_cells_max_counts_40000(atlas):
    """过滤细胞：最多 40000 个 counts"""
    filter_cells_CSR_ultrafast(atlas, max_counts=40000, add_key="filter_cells_max_counts")


@benchmark
def filter_genes_min_cells_3(atlas):
    """过滤基因：最少 3 个细胞表达"""
    filter_genes_CSR(atlas, min_cells=3, add_key="filter_genes_min_cells")


@benchmark
def filter_genes_max_cells_1000000(atlas):
    """过滤基因：最多 1000000 个细胞表达"""
    filter_genes_CSR(atlas, max_cells=1000000, add_key="filter_genes_max_cells")


@benchmark
def filter_genes_min_counts_10(atlas):
    """过滤基因：最少 10 个 counts"""
    filter_genes_CSR(atlas, min_counts=10, add_key="filter_genes_min_counts")


@benchmark
def filter_genes_max_counts_100000(atlas):
    """过滤基因：最多 100000 个 counts"""
    filter_genes_CSR(atlas, max_counts=100000, add_key="filter_genes_max_counts")


@benchmark
def log1p(atlas):
    """Log1p 变换"""
    log1p_chunked(atlas, add_field="X_log1p")


@benchmark
def scale(atlas):
    """标准化"""
    scale_gene_chunked(atlas, select_data="data", add_field="X_scale", max_value=10)


@benchmark
def expm1(atlas):
    """Expm1 变换（log1p的逆运算）- 直接在原始 data 上操作"""
    # 直接对 data 字段执行 exp(x) - 1
    exp1_chunked(atlas, add_field="X_expm1", select_data="data")


@benchmark
def sqrt(atlas):
    """Sqrt 变换"""
    # 使用 SQL 直接更新（先删除旧列，允许覆盖）
    # conn.execute("""
    #     ALTER TABLE X_CSR_data
    #     ADD COLUMN IF NOT EXISTS X_sqrt REAL
    # """)
    atlas.connection.execute("""
        ALTER TABLE X_CSR_data
        DROP COLUMN IF EXISTS X_sqrt
    """)
    atlas.connection.execute("""
        ALTER TABLE X_CSR_data
        ADD COLUMN X_sqrt REAL
    """)
    atlas.connection.execute("""
        UPDATE X_CSR_data
        SET X_sqrt = SQRT(data)
        WHERE data IS NOT NULL
    """)


@benchmark
def pca(atlas):
    """PCA 降维 - scatlaspy 不直接支持 PCA，跳过"""
    print("  Note: PCA not directly supported in scatlaspy, skipping...")
    pass


@benchmark
def sequential_iteration(atlas):
    """顺序遍历数据集（直接使用 CSR 专用函数）"""
    batch_size = 8192
    total_cells = atlas.connection.execute("SELECT COUNT(*) FROM obs").fetchone()[0]
    count = 0
    # 直接调用 CSR 专用的 minibatch 函数
    for adata in atlas.minibatch_scan_order_cursor_csr_df_arrow_onlylie(total_cells, batch_size, drop_last=True):
        count += adata.n_obs
    print(f"  遍历了 {count} 个细胞")


@benchmark
def shuffled_iteration(atlas):
    """随机顺序遍历数据集（不放回，使用 CSR 格式）"""
    batch_size = 2048
    total_cells = atlas.connection.execute("SELECT COUNT(*) FROM obs").fetchone()[0]

    # 生成随机顺序
    indices = np.random.permutation(total_cells)
    count = 0

    for i in range(0, len(indices), batch_size):
        batch_indices = indices[i:i + batch_size]
        indices_str = ','.join(map(str, batch_indices))

        result = atlas.connection.execute(f"""
            SELECT x.indices, x.data
            FROM X_CSR_data x
            JOIN X_CSR_indptr i ON x.cell_index = i.id
            WHERE i.id IN ({indices_str})
        """).fetchall()
        count += len(result)

    print(f"  遍历了 {count} 条记录")


@benchmark
def random_minibatch_iteration(atlas):
    """随机小批量访问（使用 CSR 格式，有放回）"""
    batch_size = 2048
    n_batches = 100
    total_cells = atlas.connection.execute("SELECT COUNT(*) FROM obs").fetchone()[0]

    for _ in range(n_batches):
        # 随机选择 batch_size 个索引
        indices = np.random.choice(total_cells, size=batch_size, replace=False)
        indices_str = ','.join(map(str, indices))

        result = atlas.connection.execute(f"""
            SELECT x.indices, x.data
            FROM X_CSR_data x
            JOIN X_CSR_indptr i ON x.cell_index = i.id
            WHERE i.id IN ({indices_str})
        """).fetchall()
        _ = len(result)


@benchmark
def query_by_gene_names(atlas):
    """按基因名称查询（读取实际数据）"""
    # 获取前4个基因的 id
    gene_ids = atlas.connection.execute("SELECT id FROM var LIMIT 4").fetchall()
    if gene_ids:
        indices_str = ','.join(str(g[0]) for g in gene_ids)
        result = atlas.connection.execute(f"""
            SELECT * FROM X_CSR_data WHERE indices IN ({indices_str})
        """).fetchall()
        _ = len(result)
        print(f"  查询到 {len(result)} 条记录")


@benchmark
def query_by_gene_names_count(atlas):
    """按基因名称查询（只计数）"""
    # 获取前4个基因的 id
    gene_ids = atlas.connection.execute("SELECT id FROM var LIMIT 4").fetchall()
    if gene_ids:
        indices_str = ','.join(str(g[0]) for g in gene_ids)
        count = atlas.connection.execute(f"""
            SELECT COUNT(*) FROM X_CSR_data WHERE indices IN ({indices_str})
        """).fetchone()[0]
        print(f"  计数: {count}")


@benchmark
def query_by_expression_1gene(atlas):
    """按表达值查询（单基因）- 读取数据"""
    stats = get_gene_stats(atlas)
    gene_idx = stats['genes_1'][0]
    result = atlas.connection.execute(f"""
        SELECT data FROM X_CSR_data WHERE indices = {gene_idx} AND data > 0.5
    """).fetchall()
    _ = len(result)
    print(f"  基因 id: {gene_idx}, 查询到 {len(result)} 条记录")


@benchmark
def query_by_expression_1gene_count(atlas):
    """按表达值查询（单基因）- 只计数"""
    stats = get_gene_stats(atlas)
    gene_idx = stats['genes_1'][0]
    count = atlas.connection.execute(f"""
        SELECT COUNT(*) FROM X_CSR_data WHERE indices = {gene_idx} AND data > 0.5
    """).fetchone()[0]
    print(f"  基因 id: {gene_idx}, 计数: {count}")


@benchmark
def query_by_expression_2genes(atlas):
    """按表达值查询（双基因）- 读取数据"""
    stats = get_gene_stats(atlas)
    gene_idx1, gene_idx2 = stats['genes_2']
    result = atlas.connection.execute(f"""
        SELECT data FROM X_CSR_data
        WHERE indices IN ({gene_idx1}, {gene_idx2}) AND data > 0.5
    """).fetchall()
    _ = len(result)
    print(f"  基因 id: {gene_idx1}, {gene_idx2}, 查询到 {len(result)} 条记录")


@benchmark
def query_by_expression_2genes_count(atlas):
    """按表达值查询（双基因）- 只计数"""
    stats = get_gene_stats(atlas)
    gene_idx1, gene_idx2 = stats['genes_2']
    count = atlas.connection.execute(f"""
        SELECT COUNT(*) FROM X_CSR_data
        WHERE indices IN ({gene_idx1}, {gene_idx2}) AND data > 0.5
    """).fetchone()[0]
    print(f"  基因 id: {gene_idx1}, {gene_idx2}, 计数: {count}")


@benchmark
def query_by_expression_3genes(atlas):
    """按表达值查询（三基因）- 读取数据"""
    stats = get_gene_stats(atlas)
    gene_idx1, gene_idx2, gene_idx3 = stats['genes_3']
    result = atlas.connection.execute(f"""
        SELECT data FROM X_CSR_data
        WHERE indices IN ({gene_idx1}, {gene_idx2}, {gene_idx3}) AND data > 0.5
    """).fetchall()
    _ = len(result)
    print(f"  基因 id: {gene_idx1}, {gene_idx2}, {gene_idx3}, 查询到 {len(result)} 条记录")


@benchmark
def query_by_expression_3genes_count(atlas):
    """按表达值查询（三基因）- 只计数"""
    stats = get_gene_stats(atlas)
    gene_idx1, gene_idx2, gene_idx3 = stats['genes_3']
    count = atlas.connection.execute(f"""
        SELECT COUNT(*) FROM X_CSR_data
        WHERE indices IN ({gene_idx1}, {gene_idx2}, {gene_idx3}) AND data > 0.5
    """).fetchone()[0]
    print(f"  基因 id: {gene_idx1}, {gene_idx2}, {gene_idx3}, 计数: {count}")


# ==================== Operator 注册表 ====================

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
    'expm1': expm1,
    'sqrt': sqrt,
    'pca': pca,
    'sequential_iteration': sequential_iteration,
 #   'shuffled_iteration': shuffled_iteration,
  #  'random_minibatch_iteration': random_minibatch_iteration,

    # query_by_gene_names - 读取数据
    'query_by_gene_names': query_by_gene_names,
    # query_by_expression - 读取数据
    'query_by_expression_1gene': query_by_expression_1gene,
    'query_by_expression_2genes': query_by_expression_2genes,
    'query_by_expression_3genes': query_by_expression_3genes,
  #  query_by_gene_names_count - 只计数
    'query_by_gene_names_count': query_by_gene_names_count,
    # query_by_expression - 只计数
    # 'query_by_expression_1gene_count': query_by_expression_1gene_count,
    # 'query_by_expression_2genes_count': query_by_expression_2genes_count,
    # 'query_by_expression_3genes_count': query_by_expression_3genes_count,
}


# ==================== 主函数 ====================

def run_single_operator(args):
    """运行单个 operator"""
    atlas = load_data(args.dataset, reuse_db=not args.force_reimport)
    # 预计算基因统计（不计入 operator 时间）
    _ = get_gene_stats(atlas)
    if args.operator not in OPERATORS:
        print(f"错误: 未知 operator: {args.operator}")
        print(f"\n可用 operators:")
        for name in sorted(OPERATORS.keys()):
            print(f"  - {name}")
        sys.exit(1)

    func = OPERATORS[args.operator]
    print(f"\n[运行] {args.operator}")

    try:
        result = func(atlas)
        save_result(result, args.dataset, args.output_dir, args.result_file)
        print(f"  -> 完成. 时间: {result['Time (s)']}s, 峰值内存: {result['Peak Memory (MiB)']} MiB")
    except MemoryError:
        print("OOM: Time (s)=OOM, Peak Memory (MiB)=OOM")
        sys.exit(1)

    atlas.close()




def list_operators(args):
    """列出所有 operators"""
    print("可用 operators:\n")
    for name in sorted(OPERATORS.keys()):
        print(f"  - {name}")


def main():
    parser = argparse.ArgumentParser(
        description="scAtlas (Python) 性能基准测试",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--operator', help='要运行的 operator 名称')
    parser.add_argument('--dataset', required=False, help='数据集文件路径')
    parser.add_argument('--output-dir', help='输出目录')
    parser.add_argument('--result-file', help='结果文件名')
    parser.add_argument('--list', action='store_true', help='列出所有 operators')
    parser.add_argument('--run-all', action='store_true', help='运行所有 operators')
    parser.add_argument('--force-reimport', action='store_true',
                        help='强制重新导入数据（删除旧数据库）')

    args = parser.parse_args()

    if args.list:
        list_operators(args)
        return

    if not args.dataset:
        parser.print_help()
        print("\n错误: 需要指定 --dataset")
        sys.exit(1)

    if args.operator:
        run_single_operator(args)


if __name__ == '__main__':
    main()
