from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm
from ..data import Atlas
from sklearn.decomposition import IncrementalPCA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import umap as umap_lib

# MiniBatchKMeans = KMeans 的“流式 / 小批量版本”
#
# 普通 KMeans：每次用全量数据更新中心
# MiniBatchKMeans：每次只用**一个 batch（小块数据）**更新中心
# 循环：
#     取一小批数据（batch）
#     用这批数据 → 更新中心（增量更新）

# PCA + kmeans 一起执行
# 🌈 训练模块
class StreamingPCAMiniBatchKMeans:

    # 初始化
    def __init__(self,
                 n_components=50,
                 n_clusters=2,
                 batch_size=2048):

        # PCA参数（来自你训练好的PCA）
        self.ipca = IncrementalPCA(n_components=n_components)  # 创建 sklearn 的增量 PCA 模型
        self.components_ = None  # 现在还没训练 → 没有结果       # 🎯 components_ = 坐标轴      → 方向（往哪里投影）
        self.explained_variance_ = None  # 每个主成分的“方差大小”,这个方向有多重要    → 强度（这个方向多重要）
        self.explained_variance_ratio_ = None  # 每个主成分解释的数据比例（百分比）         → 占比（解释了多少信息）

        self.n_clusters = n_clusters # 目标 聚类数

        self.kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            batch_size=batch_size,
            init="k-means++",
            n_init="auto"
        )


    # 预先 建表
    def _create_tables(self, atlas,
                       cluster_table="obs_cluster",
                       center_table="kmeans_centers"):

        atlas.connection.execute(f"DROP TABLE IF EXISTS {cluster_table}")
        atlas.connection.execute(f"DROP TABLE IF EXISTS {center_table}")

        atlas.connection.execute(f"""
            CREATE TABLE {cluster_table} (
                atlas_cell_id INTEGER,
                cluster_id INTEGER
            );
        """)

        atlas.connection.execute(f"""
            CREATE TABLE {center_table} (
                cluster_id INTEGER,
                pc_index INTEGER,
                value FLOAT
            );
        """)

    # 写 cluster
    def _write_clusters(self, atlas, cell_ids, labels, table_name):

        df = pd.DataFrame({
            "atlas_cell_id": cell_ids,
            "cluster_id": labels.astype(np.int32)
        })

        atlas.connection.append(table_name, df)


    # 写 centers
    def _write_centers(self, atlas, table_name):

        C = self.kmeans.cluster_centers_

        rows = []
        for i in range(C.shape[0]):
            for j in range(C.shape[1]):
                rows.append((i, j, float(C[i, j])))

        df = pd.DataFrame(rows,
                          columns=["cluster_id", "pc_index", "value"])

        atlas.connection.append(table_name, df)


    # 1. 训练 PCA
    def _fit_PCA(self,  atlas: Atlas ):

        print("[PCA] Start fitting...")

        batch_count = 0
        for X_batch in tqdm( atlas.minibatch_dense() ) :  # 获取minibatch
            self.ipca.partial_fit(X_batch)
            batch_count += 1

        # 保存结果
        self.components_ = self.ipca.components_.astype(np.float32)                              # 方向（往哪里投影）
        self.explained_variance_ = self.ipca.explained_variance_.astype(np.float32)              # 强度（这个方向多重要）
        self.explained_variance_ratio_ = self.ipca.explained_variance_ratio_.astype(np.float32)  # 占比（解释了多少信息）

        print("[PCA] Fit done")

        return self


    # 2. 转换 pca + minibatch kmeans 聚类 训练
    def fit_kmeans(self, atlas):

        print("[Pipeline] Start 转换 pca + minibatch kmeans 聚类 训练 ")
        self.components_ = self.load_components(atlas)

        for X_batch in tqdm(atlas.minibatch_dense()):

            # 1️⃣ PCA
            X_pca = X_batch @ self.components_.T


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
            # cell_offset = self.writer_obsm_X_pca(
            #     atlas,
            #     X_pca,
            #     cell_offset
            # )

            # 2️⃣ KMeans online train
            self.kmeans.partial_fit(X_pca)

            # print(f"cluster_centers : {self.kmeans.cluster_centers_}")

    # 3. 转换 pca + minibatch kmeans 聚类 预测
    def predict_kmeans(
            self,
            atlas,
            cluster_table="obs_cluster",
            write_to_obs: bool = True,
            obs_col: str = "kmeans"
    ):
        """
        PCA + MiniBatchKMeans 预测，并把 cluster label 写入数据库

        功能
        ----
        1. 可选写入独立表 obs_cluster
        2. 推荐：同步写入 obs.kmeans，供后续 groupby 作图/分析直接使用

        参数
        ----
        atlas : Atlas
        cluster_table : str
            独立保存聚类结果的表名
        write_to_obs : bool
            是否把结果同步写入 obs 表
        obs_col : str
            写入 obs 的列名，默认 "kmeans"
        """

        print("[Pipeline] Start minibatch kmeans 聚类 转换 ")

        import numpy as np
        import pandas as pd

        conn = atlas.connection

        # -------------------------------------------------
        # 0️⃣ 准备：建独立表
        # -------------------------------------------------
        conn.execute(f"DROP TABLE IF EXISTS {cluster_table}")
        conn.execute(f"""
            CREATE TABLE {cluster_table} (
                atlas_cell_id BIGINT,
                cluster_id INTEGER
            )
        """)

        # -------------------------------------------------
        # 1️⃣ 准备：obs 中增加 kmeans 列
        # -------------------------------------------------
        if write_to_obs:
            obs_cols = [r[1] for r in conn.execute("PRAGMA table_info(obs)").fetchall()]
            if obs_col not in obs_cols:
                conn.execute(f"ALTER TABLE obs ADD COLUMN {obs_col} INTEGER")

            # 先清空旧结果（可选，但推荐）
            conn.execute(f"UPDATE obs SET {obs_col} = NULL")

        # -------------------------------------------------
        # 2️⃣ 读取 PCA components（只加载一次）
        # -------------------------------------------------
        self.components_ = self.load_components(atlas)

        cell_offset = 0

        for X_batch in tqdm(atlas.minibatch_dense()):
            # 1️⃣ PCA transform
            X_pca = X_batch @ self.components_.T

            # 2️⃣ predict
            labels = self.kmeans.predict(X_pca).astype(np.int32)

            # 3️⃣ 当前 batch 对应的 atlas_cell_id
            n = len(labels)
            atlas_cell_ids = np.arange(cell_offset, cell_offset + n, dtype=np.int64)

            batch_df = pd.DataFrame({
                "atlas_cell_id": atlas_cell_ids,
                "cluster_id": labels
            })

            # 4️⃣ 写独立表
            conn.append(cluster_table, batch_df)

            # 5️⃣ 同步写回 obs.kmeans
            if write_to_obs:
                conn.register("_kmeans_batch_tmp", batch_df)

                conn.execute(f"""
                    UPDATE obs
                    SET {obs_col} = t.cluster_id
                    FROM _kmeans_batch_tmp t
                    WHERE obs.atlas_cell_id = t.atlas_cell_id
                """)

                conn.unregister("_kmeans_batch_tmp")

            cell_offset += n

        print("[Pipeline] Done")

        # -------------------------------------------------
        # 3️⃣ 保存 centers
        # -------------------------------------------------
        conn.execute("DROP TABLE IF EXISTS kmeans_centers")

        C = self.kmeans.cluster_centers_
        rows = []
        for i in range(C.shape[0]):
            for j in range(C.shape[1]):
                rows.append((i, j, float(C[i, j])))

        df_centers = pd.DataFrame(rows, columns=["cluster_id", "pc_index", "value"])
        conn.execute("""
            CREATE TABLE kmeans_centers (
                cluster_id INTEGER,
                pc_index INTEGER,
                value FLOAT
            )
        """)
        conn.append("kmeans_centers", df_centers)

        print("[Pipeline] centers written")


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


    # 运行主函数
    def run(self,atlas):

        # self._create_tables(atlas) # 建表
        # self._fit_PCA(atlas)  # pca 训练

        self.fit_kmeans(atlas) #  kmeans 训练

        self.predict_kmeans(atlas) #  kmeans 转换

        # self.umap_cluster_visualization(atlas)  #  kmeans 转换 + umap

        return self


#  🌈 外面的入口函数 kmeans()
def kmeans(
        atlas: Atlas,
        n_components: int = 50,
        n_clusters: int = 10,
        batch_size: int = 2048,
        obs_col: str = "kmeans",
        cluster_table: str = "obs_cluster",
        write_to_obs: bool = True
):
    """
    MiniBatchKMeans 计算入口，对齐 Scanpy 的 tl 风格。

    前提
    ----
    请先运行：
        sap.tl.pca(atlas)

    功能
    ----
    1. 从 varm_PCs 读取 PCA components
    2. 流式读取 dense batch
    3. 投影到 PCA 空间
    4. MiniBatchKMeans partial_fit
    5. predict 全量细胞 cluster
    6. 写入：
       - obs_cluster
       - obs.kmeans
       - kmeans_centers
    """
    import time

    t_start = time.time()

    print("\n==== sap.tl.kmeans ====")

    conn = atlas.connection

    # ✅ 检查 PCA components 是否存在
    tables = [r[0] for r in conn.execute("SHOW TABLES").fetchall()]
    if "varm_PCs" not in tables:
        raise ValueError(
            "数据库中不存在 varm_PCs。\n"
            "请先运行 sap.tl.pca(atlas)，再运行 sap.tl.kmeans(atlas)。"
        )

    runner = StreamingPCAMiniBatchKMeans(
        n_components=n_components,
        n_clusters=n_clusters,
        batch_size=batch_size
    )

    runner.fit_kmeans(atlas)

    runner.predict_kmeans(
        atlas,
        cluster_table=cluster_table,
        write_to_obs=write_to_obs,
        obs_col=obs_col
    )

    t_end = time.time()
    print(f"[KMeans] total time = {t_end - t_start:.2f} seconds")

    return runner
