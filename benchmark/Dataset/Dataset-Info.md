

# 1. 20K_PBMC.h5
【基本信息】
细胞数 (n_obs): 23,837
基因数 (n_vars): 36,601
文件大小: 0.07 GB

【表达矩阵结构】
X 类型: <class 'scipy.sparse._csr.csr_matrix'>
是否稀疏矩阵: True
非零元素数量 (nnz): 54,878,285
稀疏密度 nnz/(cells×genes): 0.062901

【数值类型检查】
是否包含小数: False
是否包含负值: False

【AnnData 结构信息】
是否存在 adata.raw: False
layers: []
uns 中的前若干键: []

【数据集性质判断】
表达矩阵为整数 counts，属于 raw 或仅做过 cell filtering
基因数量非常大，属于未做或极轻度 gene filtering 的数据
稀疏密度较高，该数据在 filter / scale / PCA 操作中内存开销会很大

【综合结论】
该数据集可归类为：filtered raw counts（仅做过 cell filtering）

# 2. 130k_thymus_atlas_HTA07.A01.v02.entire_data_raw_count.h5ad
【基本信息】
细胞数 (n_obs): 272,554
基因数 (n_vars): 33,694
文件大小: 1.46 GB

【表达矩阵结构】
X 类型: <class 'scipy.sparse._csr.csr_matrix'>
是否稀疏矩阵: True
非零元素数量 (nnz): 589,152,814
稀疏密度 nnz/(cells×genes): 0.064154

【数值类型检查】
是否包含小数: False
是否包含负值: False

【AnnData 结构信息】
是否存在 adata.raw: False
layers: []
uns 中的前若干键: []

【数据集性质判断】
表达矩阵为整数 counts，属于 raw 或仅做过 cell filtering
基因数量非常大，属于未做或极轻度 gene filtering 的数据
稀疏密度较高，该数据在 filter / scale / PCA 操作中内存开销会很大

【综合结论】
该数据集可归类为：filtered raw counts（仅做过 cell filtering）

# 3. 480k_TabulaSapiens.h5ad
【基本信息】
细胞数 (n_obs): 483,152
基因数 (n_vars): 58,870
文件大小: 38.28 GB

【表达矩阵结构】
X 类型: <class 'scipy.sparse._csr.csr_matrix'>
是否稀疏矩阵: True
非零元素数量 (nnz): 1,271,192,991
稀疏密度 nnz/(cells×genes): 0.044692

【数值类型检查】
是否包含小数: True
是否包含负值: False

【AnnData 结构信息】
是否存在 adata.raw: True
layers: ['decontXcounts', 'raw_counts']
uns 中的前若干键: ['_scvi', '_training_mode', 'compartment_colors', 'dendrogram_cell_type_tissue', 'dendrogram_computational_compartment_assignment', 'dendrogram_consensus_prediction', 'dendrogram_tissue_cell_type', 'donor_colors', 'donor_method_colors', 'hvg']

【数据集性质判断】
表达矩阵包含小数，数据很可能经过 log 或 normalize 处理
基因数量非常大，属于未做或极轻度 gene filtering 的数据
稀疏密度较高，该数据在 filter / scale / PCA 操作中内存开销会很大

【综合结论】
该数据集可归类为：log-normalized 数据
# 4. 500k_drugscreen.h5
【基本信息】
细胞数 (n_obs): 525,251
基因数 (n_vars): 36,601
文件大小: 2.18 GB

【表达矩阵结构】
X 类型: <class 'scipy.sparse._csr.csr_matrix'>
是否稀疏矩阵: True
非零元素数量 (nnz): 1,665,295,706
稀疏密度 nnz/(cells×genes): 0.086623

【数值类型检查】
是否包含小数: False
是否包含负值: False

【AnnData 结构信息】
是否存在 adata.raw: False
layers: []
uns 中的前若干键: []

【数据集性质判断】
表达矩阵为整数 counts，属于 raw 或仅做过 cell filtering
基因数量非常大，属于未做或极轻度 gene filtering 的数据
稀疏密度较高，该数据在 filter / scale / PCA 操作中内存开销会很大

【综合结论】
该数据集可归类为：filtered raw counts（仅做过 cell filtering）
# 5.1M_neurons_filtered_gene_bc_matrices_h5.h5
【基本信息】
细胞数 (n_obs): 1,306,127
基因数 (n_vars): 27,998
文件大小: 3.93 GB

【表达矩阵结构】
X 类型: <class 'scipy.sparse._csr.csr_matrix'>
是否稀疏矩阵: True
非零元素数量 (nnz): 2,624,828,308
稀疏密度 nnz/(cells×genes): 0.071778

【数值类型检查】
是否包含小数: False
是否包含负值: False

【AnnData 结构信息】
是否存在 adata.raw: False
layers: []
uns 中的前若干键: []

【数据集性质判断】
表达矩阵为整数 counts，属于 raw 或仅做过 cell filtering
基因数量处于中等范围，可能做过一定程度的过滤
稀疏密度较高，该数据在 filter / scale / PCA 操作中内存开销会很大

【综合结论】
该数据集可归类为：filtered raw counts（仅做过 cell filtering）