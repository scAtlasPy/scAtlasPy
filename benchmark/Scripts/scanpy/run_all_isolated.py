#!/usr/bin/env python3
"""
批量运行所有 operators，每个独立进程
"""
import os
import sys
import subprocess
import pandas as pd

# ==================== 在这里配置 ====================
# 使用相对路径，基于脚本所在目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(SCRIPT_DIR, '..', '..', 'Dataset', 'HBCA__subsample_20__hvg_2000.h5ad')  # 数据集路径
OUTPUT_DIR = os.path.join(SCRIPT_DIR, '..', '..', 'Results', 'Lastest_Results', '100G_OOM')  # 输出目录
RESULT_FILE = "scanpy_results_HBCA__subsample_20__hvg_2000.h5ad.csv"  # 结果文件名
MEMORY_LIMIT_GB = 200  # 内存限制 (GB)
# =================================================


def save_oom_result(op_name, dataset_path, output_dir, result_file):
    """保存 OOM 结果"""
    if not result_file:
        result_file = f"scanpy_results_{os.path.splitext(os.path.basename(dataset_path))[0]}.csv"
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
    if not os.path.exists(DATASET_PATH):
        print(f"错误: 数据集文件不存在: {DATASET_PATH}")
        sys.exit(1)

    # 构建命令行
    cmd = [
        sys.executable,
        'scanpy_benchmark.py',
        '--dataset', DATASET_PATH,
        '--memory-limit', str(MEMORY_LIMIT_GB),
    ]

    if OUTPUT_DIR:
        cmd.extend(['--output-dir', OUTPUT_DIR])
    if RESULT_FILE:
        cmd.extend(['--result-file', RESULT_FILE])

    # 获取所有 operators
    from scanpy_benchmark import OPERATORS
    operators = list(OPERATORS.keys())

    print(f"共 {len(operators)} 个 operators，将为每个创建独立进程...")
    print("=" * 60)

    for i, op in enumerate(operators, 1):
        full_cmd = cmd + ['--operator', op]
        print(f"\n[{i}/{len(operators)}] {op}...")

        result = subprocess.run(full_cmd, capture_output=True, text=True)

        # 检测 OOM
        if "OOM:" in result.stdout:
            save_oom_result(op, DATASET_PATH, OUTPUT_DIR, RESULT_FILE)
            print(f"  OOM (内存不足)")
        elif result.returncode != 0:
            print(f"  失败: {result.stderr[:200]}")
        else:
            for line in result.stdout.strip().split('\n')[-3:]:
                if line.strip():
                    print(f"  {line}")

    print("\n" + "=" * 60)
    print("全部完成！")


if __name__ == '__main__':
    main()
