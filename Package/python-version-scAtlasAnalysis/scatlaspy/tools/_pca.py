from ..data import Atlas
from sklearn.decomposition import IncrementalPCA
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA
import pandas as pd
import time


# 流式 PCA ；支持 minibatch 训练 + 推理
class StreamingPCA:

    # 初始化
    def __init__(self,
                 n_components = 30,
                 fit_batches: int = 1000,
                 buffer_batch_num: int = 5,
                 ):

        self.n_components = n_components # PCA 目标维度
        self.ipca = IncrementalPCA(n_components=n_components) # 创建 sklearn 的增量 PCA 模型
        self.fit_batches = fit_batches
        self.buffer_batch_num = buffer_batch_num

        self.components_ = None                 # components_ = 坐标轴  → 方向（往哪里投影）
        self.explained_variance_ = None         # variance = 每个轴有多重要   → 强度（这个方向多重要）
        self.explained_variance_ratio_ = None   # ratio = 占总信息多少        → 占比（解释了多少信息）


    # 新建 obsm_X_pca 表
    def _create_pca_table(self, atlas:Atlas, n_components = 30, table_name="obsm_X_pca"):

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
    def _create_pcs_table(self, atlas:Atlas,  n_components = 30, table_name="varm_PCs"):

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

        cell_ids = np.arange(cell_offset, cell_offset + n, dtype=np.int32) # atlas_cell_id

        X_batch = X_batch.astype(np.float32) # float32（节省空间）

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
    def fit(self, atlas: Atlas):

        print("[PCA] Start fitting...")
        print(f"[PCA] fit_batches = {self.fit_batches}")
        print(f"[PCA] buffer_batch_num = {self.buffer_batch_num}")

        batch_count = 0

        for X_batch in tqdm(
                atlas.minibatch_dense(
                    pass_mode="multi-pass",
                    buffer_batch_num=self.buffer_batch_num,
                    max_batches=self.fit_batches,
                ),
                total=self.fit_batches,
                desc="[PCA] partial_fit batches"
        ):
            self.ipca.partial_fit(X_batch)

            batch_count += 1

            if batch_count % 10 == 0:
                print(f"[PCA] partial_fit batch = {batch_count}/{self.fit_batches}")

        if batch_count == 0:
            raise RuntimeError("[PCA] 没有获得任何 minibatch，无法训练 PCA")

        # 保存结果
        self.components_ = self.ipca.components_.astype(np.float32)
        self.explained_variance_ = self.ipca.explained_variance_.astype(np.float32)
        self.explained_variance_ratio_ = self.ipca.explained_variance_ratio_.astype(np.float32)

        print("[PCA] Fit done")
        print(f"[PCA] actual fitted batches = {batch_count}")

        cum_ratio = np.cumsum(self.explained_variance_ratio_)

        print("[PCA] 累计解释方差比例（前 {} 个主成分）：{:.4f}".format(
            len(self.explained_variance_ratio_),
            self.explained_variance_ratio_.sum()
        ))

        print("[PCA] 前 10 个主成分的累计解释方差比例：")
        print(cum_ratio[:10])

        print("[PCA] 最终累计解释方差比例：{:.4f}".format(cum_ratio[-1]))

        total_ratio = cum_ratio[-1]

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

        for X_batch in tqdm(atlas.minibatch_dense( pass_mode="single-pass")):

            X_pca = self.ipca.transform(X_batch)

            # 只写 obsm（每个batch）
            cell_offset = self._writer_obsm_X_pca(
                atlas,
                X_pca,
                cell_offset
            )

        print("[PCA] Transform done")

    # 主函数
    def fit_transform(self, atlas: Atlas):

        print("[PCA] Fit + Transform")

        # 训练
        self.fit(atlas)

        # 写一次模型结果
        self._writer_varm_PCs(atlas)
        self._writer_uns_pca_stats(atlas)

        # transform（写 obsm）
        self.transform(atlas)
        return self

    # 获取结果
    def get_results(self):
        return {
            "components": self.components_,
            "explained_variance": self.explained_variance_,
            "explained_variance_ratio": self.explained_variance_ratio_
        }

    # 从数据库读取 PCA components，并恢复到 self.components_
    def load_components(self, atlas, table_name="varm_PCs"):

        conn = atlas.connection

        # 读取整张表
        df = conn.execute(f"""
            SELECT * FROM {table_name}
            ORDER BY atlas_gene_id
        """).fetchdf()

        # 去掉 atlas_gene_id
        pcs = df.drop(columns=["atlas_gene_id"]).values

        # 转置回 PCA 原始格式；(gene, pc) -> (pc, gene)
        components_ = pcs.T.astype(np.float32)

        print(f"[Load] components_ shape = {components_.shape}")

        return components_

    def run(self, atlas: Atlas):

        # 建表；建表维度必须和本次 PCA 输出维度 self.n_components 对齐
        self._create_pca_table(
            atlas,
            n_components=self.n_components
        )

        # varm_PCs 表维度必须和 self.components_.T 的列数一致
        self._create_pcs_table(
            atlas,
            n_components=self.n_components
        )

        self._create_pca_stats_table(atlas)

        # 运行PCA
        self.fit_transform(atlas)

        # 对比信息
        components = self.load_components(atlas)
        if np.array_equal(components, self.components_):
            print(" components 提取正确")
        if np.allclose(components, self.components_):
            print(" components 提取正确")


# 流式 PCA 入口
def pca(
        atlas: Atlas,
        n_components: int = 50,
        fit_batches: int = 1000,
        buffer_batch_num: int = 5,
):

    t_start = time.time()

    print("\n==== sap.tl.pca ====")

    pca_runner = StreamingPCA(
        n_components=n_components,
        fit_batches=fit_batches,
        buffer_batch_num=buffer_batch_num,
    )

    pca_runner.run(atlas)

    t_end = time.time()
    print(f"[PCA] total time = {t_end - t_start:.2f} seconds")

    return pca_runner



# 全量pca
class SimplePCA:

    def __init__(self, n_components = 30):

        self.n_components = n_components
        self.pca = PCA(n_components=n_components)

        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None

    # 收集全量数据
    def _collect_all_data(self, atlas):

        print("[PCA] Collecting all data into memory...")

        X_list = []

        for X_batch in tqdm(atlas.minibatch_dense()):
            X_list.append(X_batch)

        X = np.vstack(X_list)

        print(f"[PCA] Full matrix shape = {X.shape}")

        return X

    # Fit（一次性 PCA）
    def fit(self, atlas):

        X = self._collect_all_data(atlas)

        print("[PCA] Start full PCA fit...")
        self.pca.fit(X)

        self.components_ = self.pca.components_.astype(np.float32)
        self.explained_variance_ = self.pca.explained_variance_.astype(np.float32)
        self.explained_variance_ratio_ = self.pca.explained_variance_ratio_.astype(np.float32)

        print("[PCA] Fit done")

        # 解释方差评估
        cum_ratio = np.cumsum(self.explained_variance_ratio_)

        print("[PCA] 累计解释方差比例（前 {} 个主成分）：{:.4f}".format(
            len(self.explained_variance_ratio_),
            self.explained_variance_ratio_.sum()
        ))

        print("[PCA] 前 10 个主成分的累计解释方差比例：")
        print(cum_ratio[:10])

        print("[PCA] 最终累计解释方差比例：{:.4f}".format(cum_ratio[-1]))

        total_ratio = cum_ratio[-1]

        # 自动评价
        if total_ratio < 0.1:
            print("⚠️ PCA解释比例较低，可能需要检查数据或增加主成分数")
        elif total_ratio < 0.2:
            print("⚠️ PCA解释比例一般（单细胞中常见）")
        elif total_ratio < 0.4:
            print("✅ PCA解释比例正常")
        else:
            print("🔥 PCA解释比例较高，结构较明显")

        return X

    # 写 obsm
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

    # 写 varm
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

    # 写 uns
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

    # 主流程
    def run(self, atlas):

        # fit
        X = self.fit(atlas)

        # transform
        print("[PCA] Transform...")
        X_pca = self.pca.transform(X)

        # 写库
        self._write_obsm(atlas, X_pca)
        self._write_varm(atlas)
        self._write_uns(atlas)

        print("[PCA] Done ✅")


# 全量pca 入口
def pca_simple(atlas, n_components = 30):

    print("\n==== sap.tl.pca (simple) ====")

    runner = SimplePCA(n_components=n_components)
    runner.run(atlas)

    return runner