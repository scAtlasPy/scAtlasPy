import numpy as np
from tqdm import tqdm
from ..data import Atlas
import matplotlib.pyplot as plt
from sklearn.decomposition import IncrementalPCA
import pandas as pd
from datetime import datetime
# 使用 scikit-learn 里的 IncrementalPCA，专门用于 大数据 / 分批训练（streaming） 的 PCA

''' 第 1 层：算法执行层 '''
# 一个“流式 PCA 引擎”；支持 batch-by-batch 训练 + 推理
class StreamingPCA:

    # 初始化
    def __init__(self, n_components = 50):

        self.n_components = n_components # PCA 目标维度
        self.ipca = IncrementalPCA(n_components=n_components) # 创建 sklearn 的增量 PCA 模型

        # 假设你有一个表达矩阵（已经标准化）：
        # cell × gene：
        #       g1   g2
        # c1    2    2
        # c2    3    3
        # c3    4    4
        # c4    5    5
        # g1 和 g2 完全一样（强相关）

        # PCA 会找一个新方向：
        # PC1 ≈ (g1 + g2) / √2
        # PC2 ≈ (g1 - g2) / √2

        # 训练完成后保存的结果（对齐 Scanpy）
        self.components_ = None                 # 现在还没训练 → 没有结果                 → 方向（往哪里投影）
        # adata.varm["PCs"]    特征向量
        # self.components_.shape = (n_components, n_genes)
        # components_ =  👉 每个 PC 是“基因的线性组合方向”
        # PC1: [0.707, 0.707]  PC1：两个基因一起涨 → 最重要方向
        # PC2: [0.707, -0.707] PC2：一个涨一个跌 → 几乎没信息
        # 🎯 components_ = 坐标轴
        # 👉 新坐标系长啥样


        self.explained_variance_ = None         # 每个主成分的“方差大小”,这个方向有多重要    → 强度（这个方向多重要）
        # adata.uns["pca"]["variance"]   特征值
        # explained_variance_ = [10.0, 0.01]  👉 每个主成分上的“数据方差大小”
        # PC1：数据 spread 很大 → 信息多
        # PC2：几乎没有变化 → 信息少
        # 就像：
        # PC1 = 主干道路（很多车）
        # PC2 = 小巷子（几乎没人）
        # 📏 variance = 每个轴有多重要
        # 👉 哪个轴“更有用”


        self.explained_variance_ratio_ = None   # 每个主成分解释的数据比例（百分比）         → 占比（解释了多少信息）
        # adata.uns["pca"]["variance_ratio"]  特征值归一化
        # explained_variance_ratio_ = [0.999, 0.001]
        # 👉 每个 PC 解释了“多少比例的信息”
        # PC1：解释了 99.9% 的信息
        # PC2：几乎没用
        # 📊 ratio = 占总信息多少
        # 👉 保留了多少信息

        # explained_variance_ratio_ = explained_variance_ / sum(explained_variance_)

    # 新建 obsm_X_pca 表
    def _create_pca_table(self, atlas:Atlas, n_components=50, table_name="obsm_X_pca"):

        atlas.connection.execute(f""" DROP TABLE IF EXISTS {table_name}; """)

        cols = ",\n".join([f"pc{i} FLOAT" for i in range(n_components)])

        sql = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            cell_id INTEGER,
            {cols}
        );
        """
        atlas.connection.execute(sql)
        print("obsm_X_pca 新建完成")

    # 新建 varm_PCs 表
    def _create_pcs_table(self, atlas:Atlas,  n_components=50, table_name="varm_PCs"):

        atlas.connection.execute(f""" DROP TABLE IF EXISTS {table_name}; """)

        cols = ",\n".join([f"pc{i} FLOAT" for i in range(n_components)])

        sql = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            gene_id USMALLINT,
            {cols}
        );
        """
        atlas.connection.execute(sql)
        print("varm_PCs 新建完成")

    # 新建 uns_pca_stats 表
    def _create_pca_stats_table(self, atlas:Atlas, table_name="uns_pca_stats"):

        atlas.connection.execute(f""" DROP TABLE IF EXISTS {table_name}; """)

        sql = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            pc_index USMALLINT,
            variance REAL,           --  float 32 单精度浮点数（4字节）
            variance_ratio REAL      --  float 32 单精度浮点数（4字节）
        );
        """
        atlas.connection.execute(sql)
        print("uns_pca_stats 新建完成")

    # 写 obsm_X_pca 表
    def _writer_obsm_X_pca(self, atlas: Atlas, X_batch, cell_offset, table_name="obsm_X_pca"):

        n = X_batch.shape[0]

        # cell_id
        cell_ids = np.arange(cell_offset, cell_offset + n, dtype=np.int64)
        # float32（节省空间）
        X_batch = X_batch.astype(np.float32)

        # 构建 DataFrame
        df = pd.DataFrame(
            X_batch,
            columns=[f"pc{i}" for i in range(X_batch.shape[1])]
        )

        df.insert(0, "cell_id", cell_ids)

        atlas.connection.append(table_name, df)

        return cell_offset + n

    # 写 varm_PCs 表
    def _writer_varm_PCs(self, atlas: Atlas, table_name="varm_PCs"):

        # components_.shape     = (PC, gene)
        # components_.T.shape   = (gene, PC)
        # 结果示例
        # varm_PCs（基因 × PCA权重）
        # gene_id	 pc0	 pc1	pc2
        #  0	    0.2	    0.1	    0.6
        #  1	    0.3	    -0.2	0.1

        pcs = self.components_.T.astype(np.float32)  # (n_genes, n_components)
        df = pd.DataFrame(
            pcs,
            columns=[f"pc{i}" for i in range(pcs.shape[1])]
        )
        # 插入 gene_id
        df.insert(0, "gene_id", np.arange(pcs.shape[0], dtype=np.int32))
        atlas.connection.append(table_name, df)

    # 写 uns_pca_stats 表
    def _writer_uns_pca_stats(self, atlas: Atlas, table_name="uns_pca_stats"):

        pc_index = np.arange(len(self.explained_variance_), dtype=np.int32)

        df = pd.DataFrame({
            "pc_index": np.arange(len(self.explained_variance_), dtype=np.int32),
            "variance": self.explained_variance_.astype(np.float32),
            "variance_ratio": self.explained_variance_ratio_.astype(np.float32)
        })

        atlas.connection.append(table_name, df)

    # 训练 PCA
    def fit(self,  atlas: Atlas ):

        print("[PCA] Start fitting...")

        batch_count = 0
        prev_components = None # 上一轮 components_
        buffer = [] # 减少 partial_fit 次数

        for X_batch in tqdm( atlas.minibatch_dense() ) :  # 获取minibatch
            print(f"[PCA] 当前的批次编号 : {batch_count}")
            # self.ipca.partial_fit(X_batch)

            # todo 方法 4 （ check_input=False） +  方法 3
            #  跳过 数据校验（validate_data） （ 稍快一些 ）
            self.ipca.partial_fit(X_batch, check_input=False)

            # todo 方法 3 ：减少 partial_fit 次数， 用大 batch 进行 fit
            #  transfer 是否可以用类似的方法 ？ 在输出端进行控制会不会比较好一些 ；
            # buffer.append(X_batch)
            # if len(buffer) == 200 :
            #     X_big = np.vstack(buffer) # 纵向拼接 成一个大的batch
            #     self.ipca.partial_fit(X_big)
            #     buffer = [] # 清空

            # todo  pca
            #   830000 * 2000  -->  830000 * 50
            #   原始                         batch/s=1    710 s
            #  buffer                                     耗时        不校验 （ 稍快一些 ）
            #   5   [Consumer] batch 405, batch/s=1.51   295.48 s    208.09
            #   10  [Consumer] batch 405, batch/s=2.16   211.59 s
            #   20  [Consumer] batch 405, batch/s=2.78   167.53 s
            #   50  [Consumer] batch 405, batch/s=3.60   143.95 s    140.77
            #   100 [Consumer] batch 405, batch/s=4.49   138.48 s              1.6 GB 内存
            #   200 [Consumer] batch 405, batch/s=9.03   135.57 s

            # 原始速度 [Consumer] batch 405, batch/s=158.65

            # todo 方法 1
            #  ：Explained Variance 阈值
            #  好像没有什么用
            # if self.ipca.explained_variance_ratio_.sum() > 0.95:
            #     print(" 方法1：Explained Variance 阈值 达标")
            #     break

            # todo 方法 2 ：当前 components_ vs 上一轮 components_
            #  好像没什么用
            # 当前的 diff : 2.00101800954968
            # [PCA] 当前的批次编号 : 405
            # 406it [11:46,  1.74s/it]
            # 当前的 diff : 0.05175935483896717   最后 2 批的精度对比
            # [PCA] Fit done
            # curr = self.ipca.components_  # 当前 components_
            # if prev_components is not None:
            #     diff = np.linalg.norm(curr - prev_components)
            #     print( f"当前的 diff : {diff} " )
            #     if diff < 1e-3: # 优化小于 0.001
            #         print(" 方法 2 ：当前 components_ vs 上一轮 components_ 达标")
            #         break
            # prev_components = curr.copy()



            batch_count += 1

        # 保存结果
        self.components_ = self.ipca.components_.astype(np.float32)                              # 方向（往哪里投影）
        self.explained_variance_ = self.ipca.explained_variance_.astype(np.float32)              # 强度（这个方向多重要）
        self.explained_variance_ratio_ = self.ipca.explained_variance_ratio_.astype(np.float32)  # 占比（解释了多少信息）

        print("[PCA] Fit done")

        return self

    # 降维
    def transform(self, atlas: Atlas):

        print("[PCA] Start transforming...")

        cell_offset = 0  # 🔥关键：全局递增

        for X_batch in tqdm(atlas.minibatch_dense()):
            X_pca = self.ipca.transform(X_batch)
            # transform 内部其实做的就是：
            # X_pca = X_batch @ components_.T
            # @ 表示 矩阵乘法
            # components_
            # 👉 定义了“新坐标轴”
            # X @ components_.T
            # 👉 把数据投影到这些新坐标轴上

            # 维度
            # X_batch.shape        = (n_cells, n_genes)
            # components_.shape    = (n_components, n_genes)
            # components_.T.shape  = (n_genes, n_components)

            # # ✅ 只写 obsm（每个batch）
            cell_offset = self._writer_obsm_X_pca(
                atlas,
                X_pca,
                cell_offset
            )

        print("[PCA] Transform done")

    # 主函数
    def fit_transform(self, atlas: Atlas):

        print("[PCA] Fit + Transform (multi-pass)")

        # 1️⃣ 训练
        self.fit(atlas)

        # # ✅ 写一次模型结果
        self._writer_varm_PCs(atlas)
        self._writer_uns_pca_stats(atlas)

        # 2️⃣ transform（写 obsm）
        self.transform(atlas)
        return self

    # fit_transform(atlas)
    #
    # ↓ 第1步（fit）
    # 学出：
    # components_
    # variance
    #
    # ↓ 第2步（写一次）
    # 写：
    # varm_PCs
    # uns_pca_stats
    #
    # ↓ 第3步（transform）
    # for 每个batch:
    #     写 obsm_X_pca


    # 获取结果（类似 scanpy）
    def get_results(self):
        return {
            "components": self.components_,
            "explained_variance": self.explained_variance_,
            "explained_variance_ratio": self.explained_variance_ratio_
        }


    # 从数据库读取 PCA components，并恢复到 self.components_
    def load_components(self, atlas, table_name="varm_PCs"):
        """
        从数据库读取 PCA components，并恢复到 self.components_
        """
        conn = atlas.connection

        # 1️⃣ 读取整张表
        df = conn.execute(f"""
            SELECT * FROM {table_name}
            ORDER BY gene_id
        """).fetchdf()

        # 2️⃣ 去掉 gene_id
        pcs = df.drop(columns=["gene_id"]).values

        # 3️⃣ 转置回 PCA 原始格式
        # (gene, pc) -> (pc, gene)
        components_ = pcs.T.astype(np.float32)

        print(f"[Load] components_ shape = {components_.shape}")

        return components_


    # 外部执行 run 函数
    def run(self,atlas: Atlas):

        # 建表
        self._create_pca_table(atlas)
        self._create_pcs_table(atlas)
        self._create_pca_stats_table(atlas)

        # 运行PCA
        self.fit_transform(atlas)

        # 对比信息
        components = self.load_components(atlas)
        if np.array_equal(components, self.components_):
            print(" components 提取正确")
        if np.allclose(components, self.components_):
            print(" components 提取正确")


#  PCA 总入口（Scanpy 风格）
def pca(
        atlas: Atlas,
        n_components: int = 50,
        color: str | None = "CST3",
        plot_variance_ratio: bool = True,
        plot_variance_ratio_cumsum: bool = True,
        plot_embedding: bool = True,
        x_pc: int = 0,
        y_pc: int = 1,
        sample_n: int | None = 50000,
        use_expr_field: str = "data_log1p"
):
    """
    PCA 总入口（Scanpy 风格）

    用法
    ----
    sap.pl.pca(atlas)

    或
    ----
    sap.pl.pca(
        atlas,
        n_components=50,
        color="CST3"
    )
    """
    import time

    t_start = time.time()

    print("\n==== sap.pl.pca ====")

    # 1️⃣ 初始化 PCA 对象
    pca_runner = StreamingPCA(n_components=n_components)

    # 2️⃣ 运行 PCA：建表 + fit + transform + 写库
    pca_runner.run(atlas)

    t_end = time.time()
    print(f"[PCA] total time = {t_end - t_start:.2f} seconds")

    return pca_runner


# StreamingPCA（IncrementalPCA）
# = 近似 SVD（但仍然在做 SVD） → 慢但严格
#
# OnePassPCA
# = 幂迭代（XᵀX Q） → 无 SVD → 快但近似

# StreamingPCA = “我先学完，再用”
# OnePassPCA = “我一边学，一边用”

# 🟥 StreamingPCA
# ✔ 对齐 sklearn / scanpy
# ✔ 有 variance / ratio
# ✔ 数学严格
# ❌ 慢
# ❌ 两遍 IO

# 🟩 OnePassPCA
# ✔ 单 pass
# ✔ 无 SVD
# ✔ streaming 友好
# ✔ 超快
# ❌ 没有 variance（需要额外算）

# 高变基因（≤5000），并且需要精确的方差解释率，使用 StreamingPCA（IncrementalPCA）是最稳妥、最标准的选择。
# 全基因（未筛选）上直接做 PCA，或者数据量极大且只有一次遍历机会，那么 OnePassPCA 是更优的工程方案。
# 将 OnePassPCA 作为一个可选的加速模式，当 n_genes > 5000 时自动切换


# todo 改进
#  用一个随机初始化的“方向猜想器” Q，随着数据流不断修正猜想，同时用当前的猜想把数据压缩存盘。
#  通过多猜几个方向（过采样）和反复强化（幂迭代），让最终猜想无限接近真实主成分。
#  2840130 x 2000
#  精度太低了 Batch 68] proj_diff = 1.49e+00

class OnePassPCA:

    # 初始化
    def __init__(self, n_components=50, oversample=10, power_iter=2, warmup_batches=10):
        """
        单 pass PCA（Streaming Randomized PCA）
        参数：
        - n_components: PCA维度
        - oversample: 过采样（提升精度）
        - power_iter: power iteration 次数（建议 2~3）
        - warmup_batches: 前多少 batch 只训练不写入
        """
        self.k = n_components     # PCA维度         ，目标维度
        self.p = oversample       # 过采样（提高精度），多学习的维度
        # 真实主子空间 ≈ 前 k 个特征
        # 但我们用随机方法估计 → 会有误差
        # 👉 多给一点维度（k+p），最后再扔掉多余的方向 ，只保留前 k 个。；这是 randomized SVD 标准做法。

        # 初始时，Q 是随机乱猜的（_init_Q 里用随机数初始化），就像你闭着眼随便指了 60 个方向。

        # 过采样 (p)：我们最终只要 k 个主成分，但故意多维护 k+p 个方向。
        # 这就像你想找房间里最亮的 5 盏灯，但你先找出最亮的 10 盏，再从里面挑前 5 盏——这样更不容易漏掉真正亮的。



        self.l = self.k + self.p  # 实际子空间维度

        self.power_iter = power_iter  # 幂迭代次数（提高精度）
        self.warmup = warmup_batches  # 前几个 batch 只训练不输出

        self.Q = None
        self.components_ = None

        # 🔥新增
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None

        # 🔥新增（用于统计 variance）
        self._var_accum = None
        self._n_samples = 0

        # 🔥新增（early stop）
        self.tol = 1e-4
        self.patience = 3


    # 初始化子空间
    def _init_Q(self, n_genes):

        Q = np.random.randn(n_genes, self.l).astype(np.float32) # 生成随机矩阵
        # Q (genes × l )     self.l = self.k + self.p  # 实际子空间维度
        Q, _ = np.linalg.qr(Q) # QR分解： Q → 正交基
        return Q

    # 更新子空间（核心）
    def _update_Q(self, X, Q):
        # 不断用协方差矩阵作用 Q，让 Q 收敛到主特征向量（PCA方向）

        for _ in range(self.power_iter): # 循环 power_iter 次： 提高精度
            # 做 2 次：方向已经非常接近真实主成分（误差小到忽略不计）。

            Q = X.T @ (X @ Q)        # Q ← (X^T X) Q :  Power Iteration（幂迭代）
            # X^T X 的特征向量 = PCA主成分方向;
            # 不断强化最大特征值对应方向
            # 每次更新：
            # Q 往“数据最主要变化方向”靠近

            Q, _ = np.linalg.qr(Q)  # QR分解： Q → 正交基 ;  防止：数值爆炸、向量变共线
            # QR 返回两个东西：
            # Q（你要的）
            # R（你不要的） _ 的意思是：我不关心这个变量（丢掉）

            # QR 分解复杂度：O(n_genes × l²)
            # 当前 2000 × 60² ≈ 7.2M ops 👉 很快 ✅（远比 SVD 快）

            # 把 Q 里面“乱七八糟的向量”，变成“互相垂直且长度为1的一组向量”
            # ❌ 原来：方向可能重复、歪的
            # ✅ 现在：一组“标准正交基” ✔ 互相垂直 ✔ 线性无关  ✔ 稳定
            # 👉 shape 不变，但内容变了！

            # 💥 如果不做 QR 会怎样？
            # Q 的列向量会：
            # ❌ 越来越相似（collapse）
            # ❌ 数值爆炸
            # ❌ 失去 rank

            # 👉 最终：
            # PCA 完全坏掉 💀

            # ✅ 做了 QR：
            # 每一步都保证：
            # ✔ 子空间是正交的
            # ✔ 数值稳定
            # ✔ 不会退化

            # QR 分解：
            # 第一步（归一化）：把每个箭头缩放到长度等于 1（保持方向不变，只改长度）。这叫单位化。
            # 第二步（正交化）：让箭头之间互相垂直（正交）。如果第二个箭头有点歪，跟第一个不垂直，就把它掰成与第一个垂直的方向，同时尽量保持它原来指向的大致区域。

            # 幂迭代 (power_iter)：每一次 _update_Q 里循环 2~3 次 Q = X.T @ (X @ Q)，这相当于把“重要方向”的信号放大，把噪声方向抑制掉。
            # 多做几次，Q 就更贴近真实主成分。

        return Q

    # 建表（obsm）
    def _create_pca_table(self, atlas: Atlas, table_name="obsm_X_pca"):
        atlas.connection.execute(f"DROP TABLE IF EXISTS {table_name};")

        cols = ",\n".join([f"pc{i} FLOAT" for i in range(self.k)])

        sql = f"""
        CREATE TABLE {table_name} (
            cell_id BIGINT,
            {cols}
        );
        """
        atlas.connection.execute(sql)
        print("[DB] obsm_X_pca created")

    # 建表（varm）
    def _create_pcs_table(self, atlas: Atlas, table_name="varm_PCs"):
        atlas.connection.execute(f"DROP TABLE IF EXISTS {table_name};")

        cols = ",\n".join([f"pc{i} FLOAT" for i in range(self.k)])

        sql = f"""
        CREATE TABLE {table_name} (
            gene_id INTEGER,
            {cols}
        );
        """
        atlas.connection.execute(sql)
        print("[DB] varm_PCs created")

    # 新建 uns_pca_stats 表
    def _create_pca_stats_table(self, atlas:Atlas, table_name="uns_pca_stats"):

        atlas.connection.execute(f""" DROP TABLE IF EXISTS {table_name}; """)

        sql = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            pc_index USMALLINT,
            variance REAL,           --  float 32 单精度浮点数（4字节）
            variance_ratio REAL      --  float 32 单精度浮点数（4字节）
        );
        """
        atlas.connection.execute(sql)
        print("uns_pca_stats 新建完成")

    # 写 obsm
    def _writer_obsm(self, atlas, X_pca, offset):

        n = X_pca.shape[0]

        df = pd.DataFrame(
            X_pca.astype(np.float32),
            columns=[f"pc{i}" for i in range(self.k)]
        )

        df.insert(0, "cell_id", np.arange(offset, offset + n, dtype=np.int64))

        atlas.connection.append("obsm_X_pca", df)

        return offset + n

    # 写 varm
    def _writer_varm(self, atlas):

        pcs = self.Q[:, :self.k]          # (gene, k)

        df = pd.DataFrame(
            pcs.astype(np.float32),
            columns=[f"pc{i}" for i in range(self.k)]
        )

        df.insert(0, "gene_id", np.arange(pcs.shape[0], dtype=np.int32))

        atlas.connection.append("varm_PCs", df)

    # 写 uns_pca_stats
    def _writer_uns_pca_stats(self, atlas: Atlas, table_name="uns_pca_stats"):

        pc_index = np.arange(self.k, dtype=np.int32)

        df = pd.DataFrame({
            "pc_index": pc_index,
            "variance": self.explained_variance_.astype(np.float32),
            "variance_ratio": self.explained_variance_ratio_.astype(np.float32)
        })

        atlas.connection.append(table_name, df)

    # 主函数（单 pass）
    def fit_transform(self, atlas: Atlas):

        print("[OnePassPCA] Start")

        # 建表
        self._create_pca_table(atlas)
        self._create_pcs_table(atlas)
        self._create_pca_stats_table(atlas)

        offset = 0
        i = 0
        first = True

        # 🔥新增：variance 统计
        self._var_accum = None
        self._n_samples = 0

        # 🔥新增：early stop
        self._no_improve = 0

        for X in tqdm(atlas.minibatch_dense()):

            X = X.astype(np.float32)

            # 初始化 Q
            if first:
                self.Q = self._init_Q(X.shape[1]) #👉  用 gene 数初始化
                first = False

            # ========= warmup =========

            # 前 warmup_batches（比如 10 批）只用来修正猜测方向 Q，不输出任何压缩结果。
            # 让 Q 先“学一点结构”
            # 再开始输出
            if i < self.warmup:

                # 方差统计（即使不写入，也要统计，保证方差准确）
                Z = X @ self.Q
                if self._var_accum is None:
                    self._var_accum = np.zeros(self.l, dtype=np.float64)
                self._var_accum += np.sum(Z ** 2, axis=0)
                self._n_samples += X.shape[0]

                self.Q = self._update_Q(X, self.Q)
                i += 1
                continue

            # ========= 正式批次 =========

            # ========= 1️⃣ transform =========
            Z = X @ self.Q  # (cells, l)
            X_pca = Z[:, :self.k]
            # X_pca = X @ self.Q[:, :self.k]    # 👉 用当前 Q 投影 ，  # 只用前 k 个维度压缩 ， 不需要过采样的 L 维
            # self.Q [行选择, 列选择]
            # 行选择   选中“所有行” 所有 gene
            # 列选择   从第 0 列 到 第 k-1 列
            # 取 Q 的：
            # ✔ 所有 gene
            # ✔ 前 k 个主成分方向

            offset = self._writer_obsm(atlas, X_pca, offset) # 立即转换，并存储

            # 🔥新增：variance 统计（必须在更新前，用当前 Q）
            # Z = X @ self.Q  # (cells, l)

            if self._var_accum is None:
                self._var_accum = np.zeros(self.l, dtype=np.float64)

            self._var_accum += np.sum(Z ** 2, axis=0)
            self._n_samples += X.shape[0]


            # ========= 2️⃣ 更新子空间 =========
            Q_old = self.Q.copy()  # 🔥新增
            self.Q = self._update_Q(X, self.Q)
            # 让 Q 更贴近真正的主要变化方向
            # 里面有个 power_iter（幂迭代）循环，做 2~3 次，是为了强化重要方向、压制噪声方向。

            # 🔥新增：早停 early stop（放在 warmup 之后才有意义）
            # 计算投影矩阵差异
            diff = np.linalg.norm(self.Q @ self.Q.T - Q_old @ Q_old.T, ord='fro')
            print(f"[Batch {i}] proj_diff = {diff:.2e}")  # 临时添加
            # [Batch 19] proj_diff = 1.34e+00
            # 数学上：proj_diff = || Q Q^T - Q_old Q_old^T ||_F 的取值在 [0, √(2l)] 之间（l = k + p）。
            # 当两个正交子空间完全正交时，该值达到最大约 √(2l)。您的 l 为 60（假设 k=50, p=10），则最大可能值约为 √120 ≈ 10.95。
            # 1.35 相当于最大值的约 12%，说明子空间在每次更新中仍然发生了显著的方向旋转，而不是微调。

            # 主成分方向将严重偏离真实 PCA，第一个主成分几乎肯定指向“均值方向”，而非生物学差异方向。
            # 方差解释率也会失真（第一个主成分方差极大，其余极小）。
            # 下游聚类/UMAP 可能会受影响，但有时仍然能区分细胞类型（因为均值方向可能被 Harmony 等工具校正掉）。


            # 子空间角度差
            # diff = np.linalg.norm( self.Q.T @ Q_old - np.eye(self.l) )

            if diff < self.tol:
                self._no_improve += 1
            else:
                self._no_improve = 0

            if self._no_improve >= self.patience:
                print(f"[Early Stop] converged (diff={diff:.2e})")
                break

            i += 1

        # ========= 最终 components =========
        self.components_ = self.Q[:, :self.k].T   # (k, gene) # 最终的主成分方向
        # components 在整个过程中一直在变，
        # 而且：越到后面越好，最终的 components 是最好的。

        # 🔥新增：explained_variance
        var = self._var_accum[:self.k] / (self._n_samples - 1)

        self.explained_variance_ = var.astype(np.float64)

        total_var = np.sum(self.explained_variance_)
        self.explained_variance_ratio_ = ( self.explained_variance_ / total_var ).astype(np.float64)
        # 这不是标准 PCA 的 explained_variance_ratio_。标准定义是 var_i / sum_of_all_gene_variances（总方差是所有基因的方差之和）。
        # 你的定义会导致 ratio 之和为 1，而实际可能只解释了一小部分总方差。


        # ========= 写 varm =========
        self._writer_varm(atlas)

        # ========= 写 uns_pca_stats =========
        self._writer_uns_pca_stats(atlas)

        print("[OnePassPCA] Done")

        return self

# todo
#   2840130 x 2000
#  buffer      StreamingPCA                     耗时
#   20  [Consumer] batch 405, batch/s  = 2    601.74 s
#   100 [Consumer] batch 1385, batch/s = 3.42  471.80 s


# todo  pca
#   830000 * 2000  -->  830000 * 50
#  buffer      StreamingPCA                    耗时         OnePassPCA                         耗时
#   5   [Consumer] batch 405, batch/s=1.51   295.48 s    [Consumer] batch 404, batch/s=27.44  15.37 s
#   10  [Consumer] batch 405, batch/s=2.16   211.59 s    [Consumer] batch 405, batch/s=35.96  11.66 s
#   20  [Consumer] batch 405, batch/s=2.78   167.53 s    [Consumer] batch 405, batch/s=42.36  9.98 s
#   50  [Consumer] batch 405, batch/s=3.60   143.95 s    [Consumer] batch 405, batch/s=72.47  6.20 s
#   100 [Consumer] batch 405, batch/s=4.49   138.48 s    [Consumer] batch 405, batch/s=76.58  6.45 s
#   200 [Consumer] batch 405, batch/s=9.03   135.57 s    [Consumer] batch 405, batch/s=89.50  6.81 s

#   👉 Randomized Streaming PCA（随机化 + 单遍扫描）
# 用一个低维子空间 Q 逼近 X 的主方向

# 初始化 Q（随机正交矩阵）
# for 每个 batch X:
#     1️⃣ 用当前 Q 做降维（transform）
#     2️⃣ 用 X 更新 Q（让 Q 更接近真实主成分）
# 最后：
#     Q ≈ PCA components

# 缺点 / 风险
# ⚠️ 1. 不是真正精确 PCA
# 因为：
# 没有全局 covariance
# 误差来源：
# batch 顺序
# warmup 不足
# power_iter 不够
# ⚠️ 2. 早期 batch 质量差
# 即使 warmup：
# 前面几个 batch 的 embedding 仍然偏差较大

# 可以优化的点（很关键）

# ⭐ 优化1：加入学习率（防止震荡）
# 现在是：
# Q = X.T @ (X @ Q)
# 可以改成：
# Q = (1 - lr) * Q + lr * (X.T @ (X @ Q))
# 👉 更稳定（类似 SGD）

# ⭐ 优化2：batch weighting
# 不同 batch size：
# 大 batch 应该影响更大

# todo
#  ⭐ Power Iteration（幂迭代）
#   找最大特征向量
# 想象：  有一个矩阵 A = X^T X
# 你想找： A 最大的方向

# 随便找一个向量 v
# 不断做：
# v ← A v
# v ← normalize(v)

# v 会越来越接近 最大特征向量

# 你代码在做同样的事情
# 只不过：
# v → 变成 Q（多个方向）
# Q = X.T @ (X @ Q)     让 Q 往 PCA方向靠近

# 再加上：
# Q, _ = np.linalg.qr(Q)
# 👉 保证：
# 每个方向互相正交（不重复）