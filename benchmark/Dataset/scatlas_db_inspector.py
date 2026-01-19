#!/usr/bin/env python3
"""
scatlas_db_inspector.py - 检查 scAtlas SASQL 数据库结构

用法:
    python scatlas_db_inspector.py <path/to/database.sasql>
    python scatlas_db_inspector.py <path/to/database.sasql> --tables
    python scatlas_db_inspector.py <path/to/database.sasql> --sample obs 10
    python scatlas_db_inspector.py <path/to/database.sasql> --schema
"""

import argparse
import sys
from pathlib import Path

try:
    import duckdb
except ImportError:
    print("需要安装 duckdb: pip install duckdb")
    sys.exit(1)


def format_number(n: int) -> str:
    """格式化大数字"""
    if n >= 1_000_000:
        return f"{n:,} ({n/1_000_000:.1f}M)"
    elif n >= 1_000:
        return f"{n:,} ({n/1_000:.1f}K)"
    return f"{n:,}"


def get_table_info(con, table_name: str) -> dict:
    """获取表的详细信息"""
    # 获取行数
    count = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]

    # 获取列信息
    columns = con.execute(f"PRAGMA table_info({table_name})").fetchall()
    col_info = []
    for col in columns:
        col_info.append({
            'cid': col[0],
            'name': col[1],
            'type': col[2],
            'notnull': col[3],
            'default': col[4],
            'pk': col[5]
        })

    return {
        'name': table_name,
        'rows': count,
        'columns': col_info
    }


def print_table_schema(info: dict):
    """打印表结构"""
    print(f"\n{'='*60}")
    print(f"表: {info['name']}")
    print(f"{'='*60}")
    print(f"行数: {format_number(info['rows'])}")
    print(f"\n列信息:")
    print(f"  {'序号':<4} {'名称':<25} {'类型':<15} {'主键':<6}")
    print(f"  {'-'*4} {'-'*25} {'-'*15} {'-'*6}")

    for i, col in enumerate(info['columns']):
        pk = "是" if col['pk'] else ""
        print(f"  {col['cid']:<4} {col['name']:<25} {col['type']:<15} {pk:<6}")


def print_database_overview(con):
    """打印数据库概览"""
    print("\n" + "="*60)
    print("scAtlas 数据库概览")
    print("="*60)

    # 获取所有表
    tables = con.execute("SHOW TABLES").fetchall()
    table_names = [t[0] for t in tables]

    print(f"\n表列表 ({len(table_names)} 个表):")
    for t in table_names:
        count = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        print(f"  - {t}: {format_number(count)} 行")

    # 关键表统计
    key_tables = ['obs', 'var', 'X_CSR_indptr', 'X_CSR_data']
    for table in key_tables:
        if table in table_names:
            info = get_table_info(con, table)
            print_table_schema(info)


def print_specific_table(con, table_name: str, limit: int = 10):
    """打印指定表的内容"""
    if limit > 0:
        print(f"\n{'-'*60}")
        print(f"表: {table_name} (前 {limit} 行)")
        print(f"{'-'*60}")

        try:
            result = con.execute(f"SELECT * FROM {table_name} LIMIT {limit}").fetchall()
            if result:
                # 获取列名
                columns = con.execute(f"PRAGMA table_info({table_name})").fetchall()
                col_names = [c[1] for c in columns]

                # 打印列名
                print("  " + " | ".join(f"{c:<15}" for c in col_names[:8]))
                print("  " + "-" * (17 * min(len(col_names), 8)))

                # 打印数据
                for row in result:
                    row_str = " | ".join(str(v)[:15] for v in row[:8])
                    print(f"  {row_str}")
            else:
                print("  (空表)")
        except Exception as e:
            print(f"  读取失败: {e}")


def print_schema_only(con):
    """只打印所有表的结构"""
    tables = con.execute("SHOW TABLES").fetchall()
    for t in tables:
        info = get_table_info(con, t[0])
        print_table_schema(info)


def print_csrs_structure(con):
    """打印 CSR 稀疏矩阵结构"""
    print("\n" + "="*60)
    print("CSR 稀疏矩阵结构")
    print("="*60)

    # indptr 统计
    indptr_stats = con.execute("""
        SELECT
            MIN(indptr) as min,
            MAX(indptr) as max,
            AVG(indptr) as avg,
            COUNT(*) as n
        FROM X_CSR_indptr
    """).fetchone()

    # 读取 indptr 样本
    indptr_sample = con.execute("SELECT id, indptr FROM X_CSR_indptr ORDER BY id LIMIT 5").fetchall()

    print(f"\nX_CSR_indptr (行指针):")
    print(f"  范围: {indptr_stats[0]} ~ {indptr_stats[1]}")
    print(f"  平均: {indptr_stats[2]:.1f}")
    print(f"  细胞数: {format_number(indptr_stats[3])}")
    print(f"  样本: {indptr_sample}")

    # data 统计
    data_stats = con.execute("""
        SELECT
            COUNT(*) as total,
            COUNT(DISTINCT cell_index) as cells_with_data,
            MIN(data) as min_val,
            MAX(data) as max_val,
            AVG(data) as avg_val
        FROM X_CSR_data
    """).fetchone()

    print(f"\nX_CSR_data (稀疏数据):")
    print(f"  总非零元素: {format_number(data_stats[0])}")
    print(f"  有数据的细胞: {format_number(data_stats[1])}")
    print(f"  值范围: {data_stats[2]:.4f} ~ {data_stats[3]:.4f}")
    print(f"  平均值: {data_stats[4]:.4f}")

    # 稀疏度计算
    n_cells = indptr_stats[3]
    n_genes = con.execute("SELECT COUNT(*) FROM var").fetchone()[0]
    theoretical = n_cells * n_genes
    sparsity = (1 - data_stats[0] / theoretical) * 100 if theoretical > 0 else 0
    print(f"  稀疏度: {sparsity:.4f}%")


def main():
    parser = argparse.ArgumentParser(
        description="检查 scAtlas SASQL 数据库结构",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    %(prog)s db.sasql                      # 查看概览
    %(prog)s db.sasql --tables             # 只列所有表
    %(prog)s db.sasql --sample obs 5       # 查看 obs 表前5行
    %(prog)s db.sasql --schema             # 只看表结构
    %(prog)s db.sasql --csr                # 查看 CSR 结构
        """
    )

    parser.add_argument('database', type=str, help='数据库文件路径 (.sasql)')
    parser.add_argument('--tables', action='store_true', help='只列出所有表')
    parser.add_argument('--sample', nargs='+', metavar=('TABLE', 'N'),
                        help='显示表的前N行，例如: --sample obs 10')
    parser.add_argument('--schema', action='store_true', help='只显示表结构')
    parser.add_argument('--csr', action='store_true', help='显示 CSR 稀疏矩阵结构')
    parser.add_argument('--full', action='store_true', help='显示完整信息（概览+CSR）')

    args = parser.parse_args()

    db_path = Path(args.database)
    if not db_path.exists():
        print(f"错误: 文件不存在: {db_path}")
        sys.exit(1)

    try:
        con = duckdb.connect(str(db_path), read_only=True)

        if args.tables:
            tables = con.execute("SHOW TABLES").fetchall()
            print("数据库中的表:")
            for t in tables:
                count = con.execute(f"SELECT COUNT(*) FROM {t[0]}").fetchone()[0]
                print(f"  - {t[0]}: {format_number(count)} 行")

        elif args.sample:
            table_name = args.sample[0]
            n = int(args.sample[1]) if len(args.sample) > 1 else 10
            print_specific_table(con, table_name, n)

        elif args.schema:
            print_schema_only(con)

        elif args.csr:
            print_csrs_structure(con)

        elif args.full:
            print_database_overview(con)
            print_csrs_structure(con)

        else:
            # 默认显示概览
            print_database_overview(con)

        con.close()

    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
