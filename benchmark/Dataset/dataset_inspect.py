import scanpy as sc
import numpy as np
import scipy.sparse as sp
import os


def inspect_sc_dataset(path, sample_n=200_000):
    print("=" * 80)
    print(f"数据集路径: {path}")
    print("=" * 80)

    # ---------- 读取数据 ----------
    if path.endswith(".h5ad"):
        adata = sc.read_h5ad(path)
    elif path.endswith(".h5"):
        adata = sc.read_10x_h5(path)
    else:
        raise ValueError("不支持的文件格式")

    # ---------- 基本信息 ----------
    print("\n【基本信息】")
    print(f"细胞数 (n_obs): {adata.n_obs:,}")
    print(f"基因数 (n_vars): {adata.n_vars:,}")
    print(f"文件大小: {os.path.getsize(path) / 1024**3:.2f} GB")

    # ---------- 矩阵结构 ----------
    print("\n【表达矩阵结构】")
    print(f"X 类型: {type(adata.X)}")
    print(f"是否稀疏矩阵: {sp.issparse(adata.X)}")

    if sp.issparse(adata.X):
        nnz = adata.X.nnz
        density = nnz / (adata.n_obs * adata.n_vars)
        print(f"非零元素数量 (nnz): {nnz:,}")
        print(f"稀疏密度 nnz/(cells×genes): {density:.6f}")
    else:
        print("警告：表达矩阵为 dense，内存消耗极高")

    # ---------- 数值检查 ----------
    print("\n【数值类型检查】")
    data = adata.X.data if sp.issparse(adata.X) else adata.X.ravel()
    data = data[:sample_n]

    has_decimal = np.any(data % 1 != 0)
    has_negative = np.any(data < 0)

    print(f"是否包含小数: {has_decimal}")
    print(f"是否包含负值: {has_negative}")

    # ---------- AnnData 结构 ----------
    print("\n【AnnData 结构信息】")
    print(f"是否存在 adata.raw: {adata.raw is not None}")
    print(f"layers: {list(adata.layers.keys())}")
    print(f"uns 中的前若干键: {list(adata.uns.keys())[:10]}")

    # ---------- 结果解释 ----------
    print("\n【数据集性质判断】")

    if has_negative:
        print("表达矩阵包含负值，数据很可能经过 scale / z-score 处理")
    elif has_decimal:
        print("表达矩阵包含小数，数据很可能经过 log 或 normalize 处理")
    else:
        print("表达矩阵为整数 counts，属于 raw 或仅做过 cell filtering")

    if adata.n_vars > 30000:
        print("基因数量非常大，属于未做或极轻度 gene filtering 的数据")
    elif adata.n_vars <= 3000:
        print("基因数量很小，属于 HVG 或高度处理后的数据")
    else:
        print("基因数量处于中等范围，可能做过一定程度的过滤")

    if sp.issparse(adata.X):
        if density > 0.03:
            print("稀疏密度较高，该数据在 filter / scale / PCA 操作中内存开销会很大")
        else:
            print("稀疏密度正常，矩阵结构较友好")

    print("\n【综合结论】")
    if (not has_decimal) and (not has_negative) and adata.n_vars > 20000:
        print("该数据集可归类为：filtered raw counts（仅做过 cell filtering）")
    elif has_decimal and not has_negative:
        print("该数据集可归类为：log-normalized 数据")
    elif has_negative:
        print("该数据集可归类为：scaled / centered 数据")
    else:
        print("该数据集为混合或自定义处理流程产物")

    print("=" * 80)


# ======================
# 使用示例
# ======================
# inspect_sc_dataset("your_dataset.h5ad")
# inspect_sc_dataset("filtered_feature_bc_matrix.h5")
#inspect_sc_dataset("")
#inspect_sc_dataset("")

#inspect_sc_dataset("/home/senpeng/zspbenchmark/benchmark/Dataset/20k_PBMC.h5")

#inspect_sc_dataset("/home/senpeng/zspbenchmark/benchmark/Dataset/130k_thymus_atlas_HTA07.A01.v02.entire_data_raw_count.h5ad")

#inspect_sc_dataset("/home/senpeng/zspbenchmark/benchmark/Dataset/480k_TabulaSapiens.h5ad")

#inspect_sc_dataset("/home/senpeng/zspbenchmark/benchmark/Dataset/500k_drugscreen.h5")

inspect_sc_dataset("/home/senpeng/zspbenchmark/benchmark/Dataset/1M_neurons_filtered_gene_bc_matrices_h5.h5")
