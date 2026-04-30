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
            atlas_cell_id INTEGER,
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
            atlas_gene_id USMALLINT,
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

        # atlas_cell_id
        cell_ids = np.arange(cell_offset, cell_offset + n, dtype=np.int64)
        # float32（节省空间）
        X_batch = X_batch.astype(np.float32)

        # 构建 DataFrame
        df = pd.DataFrame(
            X_batch,
            columns=[f"pc{i}" for i in range(X_batch.shape[1])]
        )

        df.insert(0, "atlas_cell_id", cell_ids)

        atlas.connection.append(table_name, df)

        return cell_offset + n

    # 写 varm_PCs 表
    def _writer_varm_PCs(self, atlas: Atlas, table_name="varm_PCs"):

        # components_.shape     = (PC, gene)
        # components_.T.shape   = (gene, PC)
        # 结果示例
        # varm_PCs（基因 × PCA权重）
        # atlas_gene_id	 pc0	 pc1	pc2
        #  0	    0.2	    0.1	    0.6
        #  1	    0.3	    -0.2	0.1

        pcs = self.components_.T.astype(np.float32)  # (n_genes, n_components)
        df = pd.DataFrame(
            pcs,
            columns=[f"pc{i}" for i in range(pcs.shape[1])]
        )
        # 插入 atlas_gene_id
        df.insert(0, "atlas_gene_id", np.arange(pcs.shape[0], dtype=np.int32))
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
        epoch_num_need = 10  # 设置训练轮次
        epoch_num_now = 0 # 当前训练轮次

        while epoch_num_now < epoch_num_need:

            for X_batch in tqdm( atlas.minibatch_dense(pass_mode = "multi-pass") ) :  # 获取minibatch
                print(f"[PCA] 当前的批次编号 : {batch_count}")

                self.ipca.partial_fit(X_batch)

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

                batch_count += 1
                print(f"[PCA] 当前的训练轮次 : { epoch_num_now}")
                epoch_num_now +=1

        # 保存结果
        self.components_ = self.ipca.components_.astype(np.float32)                              # 方向（往哪里投影）
        self.explained_variance_ = self.ipca.explained_variance_.astype(np.float32)              # 强度（这个方向多重要）
        self.explained_variance_ratio_ = self.ipca.explained_variance_ratio_.astype(np.float32)  # 占比（解释了多少信息）

        print("[PCA] Fit done")


        cum_ratio = np.cumsum(self.explained_variance_ratio_)

        print("[PCA] 累计解释方差比例（前 {} 个主成分）：{:.4f}".format(
            len(self.explained_variance_ratio_),
            self.explained_variance_ratio_.sum()
        ))

        print("[PCA] 前 10 个主成分的累计解释方差比例：")
        print(cum_ratio[:10])

        print("[PCA] 最终累计解释方差比例：{:.4f}".format(cum_ratio[-1]))

        total_ratio = cum_ratio[-1]

        # ✅ 自动评价
        if total_ratio < 0.1:
            print("⚠️ PCA解释比例较低，可能需要检查数据或增加主成分数")
        elif total_ratio < 0.2:
            print("⚠️ PCA解释比例一般（单细胞中常见）")
        elif total_ratio < 0.4:
            print("✅ PCA解释比例正常")
        else:
            print("🔥 PCA解释比例较高，结构较明显")

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

        print("[PCA] Fit + Transform")

        # 1️⃣ 训练
        self.fit(atlas)

        # # ✅ 写一次模型结果
        self._writer_varm_PCs(atlas)
        self._writer_uns_pca_stats(atlas)

        # 2️⃣ transform（写 obsm）
        self.transform(atlas)
        return self


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
            ORDER BY atlas_gene_id
        """).fetchdf()

        # 2️⃣ 去掉 atlas_gene_id
        pcs = df.drop(columns=["atlas_gene_id"]).values

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
def pca( atlas: Atlas, n_components: int = 50 ):
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


# todo 2700 x 32738
# 1 轮
# [PCA] 累计解释方差比例（前 50 个主成分）：0.2892
# [PCA] 前 10 个主成分的累计解释方差比例：
# [0.03334165 0.04753095 0.05819701 0.06609452 0.0731115  0.07977109
#  0.08617575 0.09239872 0.09848478 0.10447507]
# [PCA] 最终累计解释方差比例：0.2892

# 10 轮
# [PCA] 累计解释方差比例（前 50 个主成分）：0.2892
# [PCA] 前 10 个主成分的累计解释方差比例：
# [0.03334164 0.04753093 0.058197   0.0660945  0.07311148 0.07977106
#  0.08617574 0.0923987  0.09848477 0.10447505]
# [PCA] 最终累计解释方差比例：0.2892

# 全量的 PCA
# [PCA] 累计解释方差比例（前 50 个主成分）：0.2877
# [PCA] 前 10 个主成分的累计解释方差比例：
# [0.03334164 0.04753095 0.058197   0.06609449 0.07311138 0.07977074
#  0.08617501 0.09239761 0.09848297 0.10447221]
# [PCA] 最终累计解释方差比例：0.2877

# todo 819200 细胞
# [PCA] 累计解释方差比例（前 50 个主成分）：0.4531
# [PCA] 前 10 个主成分的累计解释方差比例：
# [0.0724051  0.1290006  0.16795102 0.20260295 0.22874527 0.25148278
#  0.2708412  0.2882735  0.30078217 0.311649  ]
# [PCA] 最终累计解释方差比例：0.4531
# 🔥 PCA解释比例较高，结构较明显



import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA
import pandas as pd


class SimplePCA:

    def __init__(self, n_components=50):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)

        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None

    # ================================
    # 1️⃣ 收集全量数据
    # ================================
    def _collect_all_data(self, atlas):

        print("[PCA] Collecting all data into memory...")

        X_list = []

        for X_batch in tqdm(atlas.minibatch_dense()):
            X_list.append(X_batch)

        X = np.vstack(X_list)

        print(f"[PCA] Full matrix shape = {X.shape}")

        return X

    # ================================
    # 2️⃣ Fit（一次性 PCA）
    # ================================
    def fit(self, atlas):

        X = self._collect_all_data(atlas)

        print("[PCA] Start full PCA fit...")
        self.pca.fit(X)

        self.components_ = self.pca.components_.astype(np.float32)
        self.explained_variance_ = self.pca.explained_variance_.astype(np.float32)
        self.explained_variance_ratio_ = self.pca.explained_variance_ratio_.astype(np.float32)

        print("[PCA] Fit done")

        # ================================
        # 📊 解释方差评估（🔥你要的部分）
        # ================================
        cum_ratio = np.cumsum(self.explained_variance_ratio_)

        print("[PCA] 累计解释方差比例（前 {} 个主成分）：{:.4f}".format(
            len(self.explained_variance_ratio_),
            self.explained_variance_ratio_.sum()
        ))

        print("[PCA] 前 10 个主成分的累计解释方差比例：")
        print(cum_ratio[:10])

        print("[PCA] 最终累计解释方差比例：{:.4f}".format(cum_ratio[-1]))

        total_ratio = cum_ratio[-1]

        # ✅ 自动评价
        if total_ratio < 0.1:
            print("⚠️ PCA解释比例较低，可能需要检查数据或增加主成分数")
        elif total_ratio < 0.2:
            print("⚠️ PCA解释比例一般（单细胞中常见）")
        elif total_ratio < 0.4:
            print("✅ PCA解释比例正常")
        else:
            print("🔥 PCA解释比例较高，结构较明显")

        return X

    # ================================
    # 3️⃣ 写 obsm
    # ================================
    def _write_obsm(self, atlas, X_pca):

        atlas.connection.execute("DROP TABLE IF EXISTS obsm_X_pca")

        cols = ",\n".join([f"pc{i} FLOAT" for i in range(self.n_components)])

        atlas.connection.execute(f"""
            CREATE TABLE obsm_X_pca (
                atlas_cell_id INTEGER,
                {cols}
            )
        """)

        df = pd.DataFrame(
            X_pca.astype(np.float32),
            columns=[f"pc{i}" for i in range(self.n_components)]
        )

        df.insert(0, "atlas_cell_id", np.arange(len(df)))

        atlas.connection.append("obsm_X_pca", df)

    # ================================
    # 4️⃣ 写 varm
    # ================================
    def _write_varm(self, atlas):

        atlas.connection.execute("DROP TABLE IF EXISTS varm_PCs")

        cols = ",\n".join([f"pc{i} FLOAT" for i in range(self.n_components)])

        atlas.connection.execute(f"""
            CREATE TABLE varm_PCs (
                atlas_gene_id USMALLINT,
                {cols}
            )
        """)

        pcs = self.components_.T  # (gene × pc)

        df = pd.DataFrame(
            pcs,
            columns=[f"pc{i}" for i in range(self.n_components)]
        )

        df.insert(0, "atlas_gene_id", np.arange(pcs.shape[0]))

        atlas.connection.append("varm_PCs", df)

    # ================================
    # 5️⃣ 写 uns
    # ================================
    def _write_uns(self, atlas):

        atlas.connection.execute("DROP TABLE IF EXISTS uns_pca_stats")

        atlas.connection.execute("""
            CREATE TABLE uns_pca_stats (
                pc_index USMALLINT,
                variance REAL,
                variance_ratio REAL
            )
        """)

        df = pd.DataFrame({
            "pc_index": np.arange(len(self.explained_variance_)),
            "variance": self.explained_variance_,
            "variance_ratio": self.explained_variance_ratio_
        })

        atlas.connection.append("uns_pca_stats", df)

    # ================================
    # 6️⃣ 主流程
    # ================================
    def run(self, atlas):

        # 1️⃣ fit
        X = self.fit(atlas)

        # 2️⃣ transform
        print("[PCA] Transform...")
        X_pca = self.pca.transform(X)

        # 3️⃣ 写库
        self._write_obsm(atlas, X_pca)
        self._write_varm(atlas)
        self._write_uns(atlas)

        print("[PCA] Done ✅")


# ================================
# 🎯 Scanpy 风格入口
# ================================
def pca_simple(atlas, n_components=50):

    print("\n==== sap.tl.pca (simple) ====")

    runner = SimplePCA(n_components=n_components)
    runner.run(atlas)

    return runner