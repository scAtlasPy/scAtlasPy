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

        # 返回测量结果
        return {
            'Operator': func.__name__,
            'Time (s)': round(duration, 4),
            'Peak Memory (MiB)': round(peak_mem['value'], 2),
       #     'Memory Before (MiB)': round(mem_before, 2),
       #     'Memory After (MiB)': round(mem_after, 2),
        }
    return wrapper


# ==================== 工具函数 ====================

# 基因统计信息缓存（避免重复计算）
_gene_stats_cache = {}


def get_gene_stats(adata):
    """获取预计算的基因统计信息（只计算一次）"""
    cache_key = id(adata)

    if cache_key in _gene_stats_cache:
        return _gene_stats_cache[cache_key]

    # 计算每个基因在多少细胞中表达
    gene_n_cells = np.asarray((adata.X > 0).sum(axis=0)).ravel()
    sorted_idx = np.argsort(gene_n_cells)
    n_genes = len(sorted_idx)

    # 选取不同位置的基因，完全不重叠
    # genes_1: 中间偏左 1 个
    # genes_2: 中间 2 个
    # genes_3: 中间偏右 3 个
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
# 每个 operator 都是用 @benchmark 装饰的函数

@benchmark
def filter_cells_min_genes_200(adata):
    """过滤细胞：最少 200 个基因"""
#    sc.pp.filter_cells(adata, min_genes=200)
    mem_before = psutil.Process().memory_info().rss / (1024*1024)
    print(f"  执行前: {mem_before:.0f} MiB")
    sc.pp.filter_cells(adata, min_genes=200)
    mem_after = psutil.Process().memory_info().rss / (1024*1024)
    print(f"  执行后: {mem_after:.0f} MiB")

@benchmark
def filter_cells_max_genes_6000(adata):
    """过滤细胞：最多 6000 个基因"""
    sc.pp.filter_cells(adata, max_genes=6000)


@benchmark
def filter_cells_min_counts_500(adata):
    """过滤细胞：最少 500 个 counts"""
    #sc.pp.filter_cells(adata, min_counts=500)
    sc.pp.filter_cells(adata, min_counts=500)


@benchmark
def filter_cells_max_counts_40000(adata):
    """过滤细胞：最多 40000 个 counts"""
    sc.pp.filter_cells(adata, max_counts=40000)


@benchmark
def filter_genes_min_cells_3(adata):
    """过滤基因：最少 3 个细胞表达"""
    sc.pp.filter_genes(adata, min_cells=3)


@benchmark
def filter_genes_max_cells_1000000(adata):
    """过滤基因：最多 1000000 个细胞表达"""
    sc.pp.filter_genes(adata, max_cells=1000000)


@benchmark
def filter_genes_min_counts_10(adata):
    """过滤基因：最少 10 个 counts"""
    sc.pp.filter_genes(adata, min_counts=10)


@benchmark
def filter_genes_max_counts_100000(adata):
    """过滤基因：最多 100000 个 counts"""
    sc.pp.filter_genes(adata, max_counts=100000)


@benchmark
def log1p(adata):
    """Log1p 变换"""
    sc.pp.log1p(adata)


@benchmark
def scale(adata):
    """标准化"""
    sc.pp.scale(adata, max_value=10)


@benchmark
def expm1(adata):
    """Expm1 变换（原地，避免 densify）"""
    adata_copy = adata
    if sparse.issparse(adata_copy.X):
        adata_copy.X.data = np.expm1(adata_copy.X.data)
    else:
        adata_copy.X = np.expm1(adata_copy.X)


@benchmark
def sqrt(adata):
    """Sqrt 变换"""
    sc.pp.sqrt(adata)


@benchmark
def pca(adata):
    """PCA 降维"""
    sc.pp.pca(adata)


@benchmark
def sequential_iteration(adata):
    """顺序遍历数据集"""
    batch_size = 2048
    for i in range(0, adata.n_obs, batch_size):
        _ = adata[i:i + batch_size, :].X.toarray()


@benchmark
def shuffled_iteration(adata):
    """随机顺序遍历数据集"""
    batch_size = 2048
    indices = np.random.permutation(adata.n_obs)
    for i in range(0, adata.n_obs, batch_size):
        _ = adata[indices[i:i + batch_size], :].X.toarray()


@benchmark
def random_minibatch_iteration(adata):
    """随机小批量访问"""
    batch_size = 2048
    n_batches = 100
    for _ in range(n_batches):
        indices = np.random.choice(adata.n_obs, size=batch_size, replace=False)
        _ = adata[indices, :].X.toarray()


@benchmark
def query_by_gene_names(adata):
    """按基因名称查询"""
    gene_list = adata.var_names[:4].tolist()
    _ = pd.DataFrame(adata[:, gene_list].X)


@benchmark
def query_by_expression_1gene(adata):
    """按表达值查询（单基因）"""
    stats = get_gene_stats(adata)
    gene = stats['genes_1'][0]  # 1个基因

    x = adata[:, gene].X.toarray().ravel()
    mask = x > 0.5
   # _ = pd.DataFrame(adata[mask, gene].X[:100], columns=[gene])
    _ = pd.DataFrame(adata[mask, gene].X, columns=[gene])


@benchmark
def query_by_expression_2genes(adata):
    """按表达值查询（双基因）"""
    stats = get_gene_stats(adata)
    genes = stats['genes_2']

    x1 = adata[:, genes[0]].X.toarray().ravel()
    x2 = adata[:, genes[1]].X.toarray().ravel()
    mask = (x1 > 0.5) & (x2 > 0.5)
    _ = pd.DataFrame(adata[mask, genes[0]].X, columns=[genes[0]])


@benchmark
def query_by_expression_3genes(adata):
    """按表达值查询（三基因）"""
    stats = get_gene_stats(adata)
    genes = stats['genes_3']

    x1 = adata[:, genes[0]].X.toarray().ravel()
    x2 = adata[:, genes[1]].X.toarray().ravel()
    x3 = adata[:, genes[2]].X.toarray().ravel()
    mask = (x1 > 0.5) & (x2 > 0.5) & (x3 > 0.5)
    _ = pd.DataFrame(adata[mask, genes[0]].X, columns=[genes[0]])


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
    'query_by_gene_names': query_by_gene_names,
    'query_by_expression_1gene': query_by_expression_1gene,
    'query_by_expression_2genes': query_by_expression_2genes,
    'query_by_expression_3genes': query_by_expression_3genes,
    'sequential_iteration': sequential_iteration,
    'shuffled_iteration': shuffled_iteration,
    'random_minibatch_iteration': random_minibatch_iteration,
}


# ==================== 主函数 ====================

def run_single_operator(args):
    """运行单个 operator"""
    set_memory_limit(args.memory_limit)
    adata = load_data(args.dataset)

    # 预计算基因统计（不计入 operator 时间）
    _ = get_gene_stats(adata)

    if args.operator not in OPERATORS:
        print(f"错误: 未知 operator: {args.operator}")
        print(f"\n可用 operators:")
        for name in sorted(OPERATORS.keys()):
            print(f"  - {name}")
        sys.exit(1)

    func = OPERATORS[args.operator]
    print(f"\n[运行] {args.operator}")

    try:
        result = func(adata)  # @benchmark 装饰器自动测量
        save_result(result, args.dataset, args.output_dir, args.result_file)
        print(f"  -> 完成. 时间: {result['Time (s)']}s, 峰值内存: {result['Peak Memory (MiB)']} MiB")
    except MemoryError:
        # OOM 时打印特殊标记，让 run_all_isolated.py 能识别
        print("OOM: Time (s)=OOM, Peak Memory (MiB)=OOM")
        print("结果已保存")
        sys.exit(1)


def list_operators(args):
    """列出所有 operators"""
    print("可用 operators:\n")
    for name in sorted(OPERATORS.keys()):
        print(f"  - {name}")


def main():
    parser = argparse.ArgumentParser(
        description="Scanpy 性能基准测试",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--operator', help='要运行的 operator 名称')
    parser.add_argument('--dataset', required=False, help='数据集文件路径')
    parser.add_argument('--memory-limit', type=float, default=200, help='内存限制 (GB)，默认 100')
    parser.add_argument('--output-dir', help='输出目录')
    parser.add_argument('--result-file', help='结果文件名')
    parser.add_argument('--list', action='store_true', help='列出所有 operators')

    args = parser.parse_args()

    if args.list:
        list_operators(args)
        return

    if not args.operator or not args.dataset:
        parser.print_help()
        print("\n错误: 需要指定 --operator 和 --dataset")
        sys.exit(1)

    if not os.path.exists(args.dataset):
        print(f"错误: 数据集文件不存在: {args.dataset}")
        sys.exit(1)

    run_single_operator(args)


if __name__ == '__main__':
    main()
