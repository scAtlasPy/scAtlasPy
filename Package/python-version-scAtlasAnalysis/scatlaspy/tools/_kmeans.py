from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm
from ..data import Atlas
import numpy as np
import pandas as pd


# MiniBatchKMeans = KMeans 的“流式 / 小批量版本”
#
# 普通 KMeans：每次用全量数据更新中心
# MiniBatchKMeans：每次只用**一个 batch（小块数据）**更新中心
# 循环：
#     取一小批数据（batch）
#     用这批数据 → 更新中心（增量更新）


class StreamingPCAMiniBatchKMeans:

    # 初始化
    def __init__(
            self,
            n_components=50,
            n_clusters=2,
            batch_size=2048,
            fit_batches: int = 1000,        # ✅ 新增：指定 KMeans 训练阶段使用多少个 minibatch
            buffer_batch_num: int = 5,      # ✅ 新增：multi-pass 时 ShuffleBuffer 的 batch 数，和 PCA 的设计保持一致
            random_state: int | None = 0,   # ✅ 新增：结果更稳定
    ):

        # =====================================================
        # ✅ 修改：这里不再创建 IncrementalPCA
        # -----------------------------------------------------
        # PCA 已经提前通过 sap.tl.pca(atlas) 训练完成，
        # 这里 KMeans 只需要从数据库 varm_PCs 读取 PCA components。
        # =====================================================

        # PCA参数（来自你训练好的PCA）
        # 后续会从 varm_PCs 读取 components_
        self.components_ = None  # 🎯 components_ = 坐标轴 → 方向（往哪里投影）

        self.n_components = n_components
        self.n_clusters = n_clusters  # 目标 聚类数
        self.batch_size = batch_size

        # ✅ 新增：KMeans 训练使用多少个 minibatch
        self.fit_batches = fit_batches

        # ✅ 新增：multi-pass 时 ShuffleBuffer 的 batch 数
        self.buffer_batch_num = buffer_batch_num

        # ✅ 新增：随机种子，方便结果可复现
        self.random_state = random_state

        self.kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            batch_size=batch_size,
            init="k-means++",
            n_init="auto",
            random_state=random_state,
        )

    # 预先 建表
    def _create_tables(
            self,
            atlas,
            cluster_table="obs_cluster",
            center_table="kmeans_centers"
    ):

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
    def _write_centers(self, atlas, table_name="kmeans_centers"):

        conn = atlas.connection

        conn.execute(f"DROP TABLE IF EXISTS {table_name}")

        conn.execute(f"""
            CREATE TABLE {table_name} (
                cluster_id INTEGER,
                pc_index INTEGER,
                value FLOAT
            )
        """)

        C = self.kmeans.cluster_centers_

        rows = []
        for i in range(C.shape[0]):
            for j in range(C.shape[1]):
                rows.append((i, j, float(C[i, j])))

        df = pd.DataFrame(
            rows,
            columns=["cluster_id", "pc_index", "value"]
        )

        conn.append(table_name, df)

        print("[Pipeline] centers written")

    # 2. 转换 pca + minibatch kmeans 聚类 训练
    def fit_kmeans(self, atlas):

        print("[Pipeline] Start 转换 pca + minibatch kmeans 聚类 训练 ")
        print(f"[KMeans] n_clusters = {self.n_clusters}")
        print(f"[KMeans] fit_batches = {self.fit_batches}")
        print(f"[KMeans] buffer_batch_num = {self.buffer_batch_num}")

        # 读取 PCA components
        self.components_ = self.load_components(atlas)

        # 如果用户传入的 n_components 和数据库里实际 PCA 维度不同，给一个提示
        real_components = self.components_.shape[0]
        if real_components != self.n_components:
            print(
                f"[WARN] 传入 n_components={self.n_components}，"
                f"但 varm_PCs 中实际 components 数量={real_components}。"
                f"后续将以数据库中的 {real_components} 个 PC 为准。"
            )
            self.n_components = real_components

        batch_count = 0

        # =====================================================
        # ✅ 修改：像 PCA 一样，指定训练 minibatch 数
        # -----------------------------------------------------
        # 原来：
        #     for X_batch in tqdm(atlas.minibatch_dense()):
        #
        # 现在：
        #     pass_mode="multi-pass"
        #     max_batches=self.fit_batches
        #
        # 含义：
        #     KMeans 不再只训练一遍全量数据；
        #     而是训练指定数量的 minibatch。
        #
        # 例如：
        #     fit_batches=1000
        #     表示 self.kmeans.partial_fit(X_pca) 执行 1000 次。
        # =====================================================
        for X_batch in tqdm(
                atlas.minibatch_dense(
                    pass_mode="multi-pass",
                    buffer_batch_num=self.buffer_batch_num,
                    max_batches=self.fit_batches,
                ),
                total=self.fit_batches,
                desc="[KMeans] partial_fit batches"
        ):

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

            # 2️⃣ KMeans online train
            self.kmeans.partial_fit(X_pca)

            batch_count += 1

            if batch_count % 10 == 0:
                print(f"[KMeans] partial_fit batch = {batch_count}/{self.fit_batches}")

            # print(f"cluster_centers : {self.kmeans.cluster_centers_}")

        if batch_count == 0:
            raise RuntimeError("[KMeans] 没有获得任何 minibatch，无法训练 KMeans")

        print("[KMeans] Fit done")
        print(f"[KMeans] actual fitted batches = {batch_count}")

        return self

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
        if self.components_ is None:
            self.components_ = self.load_components(atlas)

        cell_offset = 0
        predict_batch_count = 0

        # =====================================================
        # ✅ 修改：预测阶段明确 single-pass
        # -----------------------------------------------------
        # 训练阶段可以 multi-pass，因为只需要训练指定数量的 batch。
        # 但是预测阶段必须每个 cell 都预测一次，所以应该 single-pass。
        # =====================================================
        for X_batch in tqdm(
                atlas.minibatch_dense(
                    pass_mode="single-pass",
                ),
                desc="[KMeans] predict batches"
        ):

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
            predict_batch_count += 1

            if predict_batch_count % 20 == 0:
                print(
                    f"[KMeans] predicted cells = {cell_offset:,}, "
                    f"batches = {predict_batch_count}"
                )

        print("[Pipeline] Done")
        print(f"[KMeans] total predicted cells = {cell_offset:,}")
        print(f"[KMeans] total predict batches = {predict_batch_count}")

        # -------------------------------------------------
        # 3️⃣ 保存 centers
        # -------------------------------------------------
        self._write_centers(atlas, table_name="kmeans_centers")

        return self

    # 从数据库读取 PCA components，并恢复到 self.components_
    def load_components(self, atlas, table_name="varm_PCs"):
        """
        从数据库读取 PCA components，并恢复到 self.components_
        """
        conn = atlas.connection

        # 1️⃣ 读取整张表
        df = conn.execute(f"""
            SELECT *
            FROM {table_name}
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
    def run(
            self,
            atlas,
            cluster_table="obs_cluster",
            write_to_obs: bool = True,
            obs_col: str = "kmeans"
    ):

        # self._create_tables(atlas) # 建表

        self.fit_kmeans(atlas) #  kmeans 训练

        self.predict_kmeans(
            atlas,
            cluster_table=cluster_table,
            write_to_obs=write_to_obs,
            obs_col=obs_col
        ) #  kmeans 转换

        return self


#  🌈 外面的入口函数 kmeans()
def kmeans(
        atlas: Atlas,
        n_components: int = 50,
        n_clusters: int = 10,
        batch_size: int = 2048,
        fit_batches: int = 1000,        # ✅ 新增：指定 KMeans 训练阶段使用多少个 minibatch
        buffer_batch_num: int = 5,      # ✅ 新增：multi-pass 时 ShuffleBuffer 的 batch 数
        obs_col: str = "kmeans",
        cluster_table: str = "obs_cluster",
        write_to_obs: bool = True,
        random_state: int | None = 0,   # ✅ 新增：结果更稳定
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

    新增
    ----
    fit_batches:
        指定 KMeans 训练阶段使用多少个 minibatch。
        例如 fit_batches=1000，表示 KMeans partial_fit 训练 1000 次。

    buffer_batch_num:
        multi-pass 时 ShuffleBuffer 的 batch 数，和 PCA 的设计保持一致。
    """
    import time

    t_start = time.time()

    print("\n==== sap.tl.kmeans ====")
    print(f"[KMeans] n_components = {n_components}")
    print(f"[KMeans] n_clusters = {n_clusters}")
    print(f"[KMeans] batch_size = {batch_size}")
    print(f"[KMeans] fit_batches = {fit_batches}")
    print(f"[KMeans] buffer_batch_num = {buffer_batch_num}")

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
        batch_size=batch_size,
        fit_batches=fit_batches,
        buffer_batch_num=buffer_batch_num,
        random_state=random_state,
    )

    runner.run(
        atlas,
        cluster_table=cluster_table,
        write_to_obs=write_to_obs,
        obs_col=obs_col,
    )

    t_end = time.time()
    print(f"[KMeans] total time = {t_end - t_start:.2f} seconds")

    return runner