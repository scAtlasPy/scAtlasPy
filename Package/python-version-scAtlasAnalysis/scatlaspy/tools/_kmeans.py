from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm
from ..data import Atlas
import numpy as np
import pandas as pd
import time

# MiniBatchKMeans
class StreamingPCAMiniBatchKMeans:

    def __init__(
            self,
            n_components=50,
            n_clusters=2,
            batch_size=2048,
            fit_batches: int = 1000,        #  KMeans 训练阶段使用多少个 minibatch
            buffer_batch_num: int = 5,      #  multi-pass 时 ShuffleBuffer 的 batch 数，和 PCA 的设计保持一致
    ):

        # PCA参数（来自你训练好的PCA），从 varm_PCs 读取 components_
        self.components_ = None  # 🎯 components_ = 坐标轴 → 方向（往哪里投影）
        self.n_components = n_components
        self.n_clusters = n_clusters  # 目标 聚类数量
        self.batch_size = batch_size
        self.fit_batches = fit_batches  # KMeans 训练使用多少个 minibatch
        self.buffer_batch_num = buffer_batch_num # multi-pass 时 ShuffleBuffer 的 batch 数
        self.kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            batch_size=batch_size,
            init="k-means++",
            n_init="auto",
        )
        # self.kmeans = MiniBatchKMeans(
        #     n_clusters=n_clusters,
        #     batch_size=batch_size,
        #     init="random",  # ✅ 修改：避免第一次 partial_fit 卡在 k-means++ 初始化
        #     n_init=1,  # ✅ 修改：不要用 auto
        #     random_state=42,  # ✅ 新增：保证可复现
        #     reassignment_ratio=0,  # ✅ 新增：减少运行中重新分配中心带来的额外开销
        # )

    # 写 obs_cluster
    def _write_clusters(self, atlas, cell_ids, labels, table_name):

        df = pd.DataFrame({
            "atlas_cell_id": cell_ids,
            "cluster_id": labels.astype(np.int32)
        })

        atlas.connection.append(table_name, df)

    # 写 kmeans_centers
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

    # 转换 pca + minibatch kmeans 聚类 训练
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

        # minibatch kmeans 聚类 训练
        for X_batch in tqdm(
                atlas.minibatch_dense(
                    batch_size=self.batch_size,
                    pass_mode="multi-pass",
                    buffer_batch_num=self.buffer_batch_num,
                    max_batches=self.fit_batches,
                ),
                total=self.fit_batches,
                desc="[KMeans] partial_fit batches"
        ):

            t0 = time.time()
            print("[DEBUG] got X_batch", X_batch.shape, X_batch.dtype)

            X_pca = X_batch @ self.components_.T  # pca 转换
            print(f"[DEBUG] PCA projection time = {time.time() - t0:.2f}s")

            # todo
            X_pca = np.ascontiguousarray(X_pca, dtype=np.float32)

            if not np.isfinite(X_pca).all():
                raise ValueError(
                    f"X_pca 中存在 NaN/Inf: "
                    f"min={np.nanmin(X_pca)}, max={np.nanmax(X_pca)}"
                )

            t1 = time.time()
            self.kmeans.partial_fit(X_pca)   # KMeans 训练
            print(f"[DEBUG] kmeans partial_fit time = {time.time() - t1:.2f}s")

            batch_count += 1

            if batch_count % 10 == 0:
                print(f"[KMeans] partial_fit batch = {batch_count}/{self.fit_batches}")

        if batch_count == 0:
            raise RuntimeError("[KMeans] 没有获得任何 minibatch，无法训练 KMeans")

        print("[KMeans] Fit done")
        print(f"[KMeans] actual fitted batches = {batch_count}")

        return self

    # 转换 pca + minibatch kmeans 聚类 预测
    def predict_kmeans(
            self,
            atlas,
            cluster_table="obs_cluster",
            write_to_obs: bool = True,
            obs_col: str = "kmeans"
    ):

        print("[Pipeline] Start minibatch kmeans 聚类 转换 ")

        conn = atlas.connection

        conn.execute(f"DROP TABLE IF EXISTS {cluster_table}")
        conn.execute(f"""
            CREATE TABLE {cluster_table} (
                atlas_cell_id BIGINT,
                cluster_id INTEGER
            )
        """)

        # obs 中增加 kmeans 列
        if write_to_obs:
            obs_cols = [r[1] for r in conn.execute("PRAGMA table_info(obs)").fetchall()]
            if obs_col not in obs_cols:
                conn.execute(f"ALTER TABLE obs ADD COLUMN {obs_col} INTEGER")
            # 先清空旧结果
            conn.execute(f"UPDATE obs SET {obs_col} = NULL")

        # 读取 PCA components
        if self.components_ is None:
            self.components_ = self.load_components(atlas)

        cell_offset = 0
        predict_batch_count = 0

        # 转换阶段 使用 single-pass
        for X_batch in tqdm(
                atlas.minibatch_dense(
                    batch_size=self.batch_size,
                    pass_mode="single-pass",
                ),
                desc="[KMeans] predict batches"
        ):

            # PCA transform
            X_pca = X_batch @ self.components_.T

            # kmeans transform
            labels = self.kmeans.predict(X_pca).astype(np.int32)

            # 当前 batch 对应的 atlas_cell_id
            n = len(labels)
            atlas_cell_ids = np.arange(cell_offset, cell_offset + n, dtype=np.int32)

            batch_df = pd.DataFrame({
                "atlas_cell_id": atlas_cell_ids,
                "cluster_id": labels
            })

            # 写表
            conn.append(cluster_table, batch_df)

            # 同步写回 obs.kmeans
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

        # 结果完整性检查
        # check_df = conn.execute(f"""
        # SELECT
        #     (SELECT COUNT(*) FROM obs WHERE filter_cell_id IS NOT NULL) AS filtered_cell_n,
        #     (SELECT COUNT(*) FROM {cluster_table}) AS cluster_n,
        #     (SELECT COUNT(DISTINCT atlas_cell_id) FROM {cluster_table}) AS cluster_unique_n,
        #     (SELECT COUNT(*) FROM obs WHERE {obs_col} IS NOT NULL) AS obs_labeled_n
        # """).fetchdf()
        #
        # print(check_df)

        # 保存 centers
        self._write_centers(atlas, table_name="kmeans_centers")

        return self

    # 从数据库读取 PCA components，并恢复到 self.components_
    def load_components(self, atlas, table_name="varm_PCs"):

        conn = atlas.connection

        df = conn.execute(f"""
            SELECT *
            FROM {table_name}
            ORDER BY atlas_gene_id
        """).fetchdf()

        # 去掉 atlas_gene_id
        pcs = df.drop(columns=["atlas_gene_id"]).values

        # 转置回 PCA 原始格式 (gene, pc) -> (pc, gene)
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

        #  kmeans 训练
        self.fit_kmeans(atlas)

        #  kmeans 转换
        self.predict_kmeans(
            atlas,
            cluster_table=cluster_table,
            write_to_obs=write_to_obs,
            obs_col=obs_col
        )

        return self


#  入口函数
def kmeans(
        atlas: Atlas,
        n_components: int = 30,
        n_clusters: int = 10,
        batch_size: int = 2048,
        fit_batches: int = 1000,        # 指定 KMeans 训练阶段使用多少个 minibatch
        buffer_batch_num: int = 5,      # multi-pass 时 ShuffleBuffer 的 batch 数
        obs_col: str = "kmeans",
        cluster_table: str = "obs_cluster",
        write_to_obs: bool = True,
):

    t_start = time.time()

    print("\n==== sap.tl.kmeans ====")
    print(f"[KMeans] n_components = {n_components}")
    print(f"[KMeans] n_clusters = {n_clusters}")
    print(f"[KMeans] batch_size = {batch_size}")
    print(f"[KMeans] fit_batches = {fit_batches}")
    print(f"[KMeans] buffer_batch_num = {buffer_batch_num}")

    conn = atlas.connection

    # 检查 PCA components 是否存在
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