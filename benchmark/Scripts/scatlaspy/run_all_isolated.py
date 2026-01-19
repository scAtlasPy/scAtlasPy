#!/usr/bin/env python3
"""
批量运行所有 operators，每个独立进程
复用同一个 .sasql 数据库文件，避免重复导入
"""
import os
import sys
import pandas as pd

# 添加路径 - 使用相对路径
# 结构: /path/to/zspbenchmark/
#       ├── benchmark/          <- 代码仓库
#       └── Package/            <- scatlaspy 包
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# ROOT_DIR = benchmark 目录 (需要去掉两级: Scripts/scatlaspy)
ROOT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
# PARENT_DIR = zspbenchmark 目录 (去掉三级)
PARENT_DIR = os.path.dirname(ROOT_DIR)

sys.path.insert(0, os.path.join(PARENT_DIR, 'Package', 'python-version-scAtlasAnalysis'))
sys.path.insert(0, SCRIPT_DIR)

# ==================== 在这里配置 ====================
# 使用相对路径
DATASET_PATH = os.path.join(ROOT_DIR, 'Dataset', '130k_thymus_atlas_HTA07.A01.v02.entire_data_raw_count.h5ad')  # 数据集路径
OUTPUT_DIR = os.path.join(ROOT_DIR, 'Results', 'Lastest_Results', '100G_OOM')  # 输出目录
# =================================================

# 自动生成结果文件名
RESULT_FILE = f"scatlaspy_results_{os.path.basename(DATASET_PATH)}.csv"

# 数据库文件路径ss
DB_NAME = os.path.splitext(os.path.basename(DATASET_PATH))[0]
DB_FILE = os.path.join(os.path.dirname(DATASET_PATH), f"{DB_NAME}.sasql")


def ensure_database():
    """确保数据库已存在，如果不存在则导入数据"""
    if os.path.exists(DB_FILE):
        # 检查数据库是否完整（是否有数据）
        try:
            import duckdb
            conn = duckdb.connect(DB_FILE, read_only=True)
            n_obs = conn.execute("SELECT COUNT(*) FROM obs").fetchone()[0]
            conn.close()
            if n_obs > 0:
                print(f"数据库已存在: {DB_FILE}")
                print(f"  已有 {n_obs} 条 obs 记录，将复用")
                return True
        except:
            pass

    print(f"数据库不存在或不完整，需要先导入数据...")
    print(f"  数据集: {DATASET_PATH}")
    print(f"  输出: {DB_FILE}")
    print("-" * 60)

    # 直接调用导入逻辑
    from scatlaspy_benchmark import load_data
    atlas = load_data(DATASET_PATH, reuse_db=False)
    atlas.close()

    print("-" * 60)
    print("数据导入完成")
    return True


def save_oom_result(op_name, dataset_path, output_dir, result_file):
    """保存 OOM 结果"""
    if not result_file:
        result_file = f"scatlaspy_results_{os.path.splitext(os.path.basename(dataset_path))[0]}.csv"
    if output_dir:
        result_file = os.path.join(output_dir, result_file)

    result = {
        'Operator': op_name,
        'Time (s)': 'OOM',
        'Peak Memory (MiB)': 'OOM',
        'Dataset': os.path.basename(dataset_path),
    }
    df = pd.DataFrame([result])
    if os.path.exists(result_file):
        df.to_csv(result_file, mode='a', header=False, index=False)
    else:
        df.to_csv(result_file, index=False)
    print(f"  OOM 结果已保存")


def main():
    import subprocess

    if not os.path.exists(DATASET_PATH):
        print(f"错误: 数据集文件不存在: {DATASET_PATH}")
        sys.exit(1)

    # 确保数据库已存在
    if not ensure_database():
        sys.exit(1)

    # 构建命令行（不复用 --force-reimport）
    cmd = [
        sys.executable,
        'scatlaspy_benchmark.py',
        '--dataset', DATASET_PATH,
    ]

    if OUTPUT_DIR:
        cmd.extend(['--output-dir', OUTPUT_DIR])
    if RESULT_FILE:
        cmd.extend(['--result-file', RESULT_FILE])

    # 获取所有 operators
    from scatlaspy_benchmark import OPERATORS
    operators = list(OPERATORS.keys())

    print(f"\n共 {len(operators)} 个 operators，将为每个创建独立进程...")
    print("复用数据库: " + DB_FILE)
    print("=" * 60)

    for i, op in enumerate(operators, 1):
        full_cmd = cmd + ['--operator', op]
        print(f"\n[{i}/{len(operators)}] {op}...")
        print("-" * 40)

        # 实时显示输出（不捕获）
        result = subprocess.run(full_cmd)

        if result.returncode != 0:
            print(f"  失败 (返回码: {result.returncode})")

    print("\n" + "=" * 60)
    print("全部完成！")


if __name__ == '__main__':
    main()
