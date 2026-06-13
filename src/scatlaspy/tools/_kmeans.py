from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm
from ..data import Atlas
import numpy as np
import pandas as pd
import time
import logging

logger = logging.getLogger('Atlas')

# MiniBatchKMeans
class StreamingKMeans:

    """基于 PCA embedding 的流式 MiniBatchKMeans 聚类器。

    该类读取 Atlas 中的 PCA loadings 和表达矩阵 minibatch，将表达矩阵投影到 PCA
    空间后使用 ``MiniBatchKMeans.partial_fit`` 训练聚类模型，并把聚类标签写回
    ``obs`` 或聚类结果表。它是 ``sap.tl.kmeans`` 的底层实现。

    Parameters
    ----------
    n_components
        使用的 PCA 主成分数量，需要与 ``varm_PCs`` 中可用的 PC 列数匹配。
    n_clusters
        K-means 聚类数。
    batch_size
        每个 minibatch 的细胞数量。
    fit_batches
        用于训练 MiniBatchKMeans 的 minibatch 数量上限。
    buffer_batch_num
        ``multi-pass`` 读取时 shuffle buffer 中缓存的 batch 数量。

    Notes
    -----
    推荐通过 ``sap.tl.kmeans`` 调用该流程。直接使用本类时，需要先完成 PCA 并构建读取索引。

    Examples
    --------
    推荐的公共 API 用法::

        sap.tl.pca(atlas, n_components=50)
        sap.tl.kmeans(atlas, n_components=30, n_clusters=20)

    直接使用底层类，适合调试聚类训练过程::

        model = StreamingKMeans(n_components=30, n_clusters=20, batch_size=4096)
        model.run(atlas, use_obs_col="kmeans_20", write_to_obs=True)"""
    def __init__(
            self,
            n_components: int=50,
            n_clusters: int=2,
            batch_size: int=2048,
            fit_batches: int = 1000,        #  KMeans 训练阶段使用多少个 minibatch
            buffer_batch_num: int = 5,      #  multi-pass 时 ShuffleBuffer 的 batch 数，和 PCA 的设计保持一致
    ):
        """初始化流式 MiniBatchKMeans 聚类器。

        该方法保存 PCA 投影维度、聚类数量和 minibatch 参数，并创建 sklearn 的 ``MiniBatchKMeans`` 模型。

        后续 ``fit_kmeans`` 会从 Atlas 中读取 PCA loadings 和表达矩阵 minibatch，先把表达矩阵投影到 PCA
        空间，再用 ``partial_fit`` 进行流式训练。

        Parameters
        ----------
        n_components
            使用的 PCA 主成分数量。

            需要与 ``varm_PCs`` 表中的 PC 列数保持一致。

        n_clusters
            KMeans 聚类数量。

        batch_size
            每个 minibatch 中的细胞数量。

            较大的值通常训练更快，但会增加单批投影和聚类时的内存占用。

        fit_batches
            用于训练 MiniBatchKMeans 的 minibatch 数量上限。

        buffer_batch_num
            ``multi-pass`` 读取时 shuffle buffer 中缓存的 batch 数量。

        Notes
        -----
        该对象只初始化模型和参数，不会立即读取数据库或写入聚类标签。
        """

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
    def _write_clusters(self, atlas: Atlas, cell_ids: np.ndarray, labels: np.ndarray, table_name: str):

        """将计算结果写入数据库表。

        该内部函数属于KMeans 聚类模块，用于支撑同一模块中的公共 API。

        基于 PCA loadings 和 Atlas minibatch 流式训练 MiniBatchKMeans，并写入聚类标签。

        它通常不会作为用户入口直接调用；直接调用时需要保证输入对象、数据库连接和相关临时表已经由上游步骤准备好。

        Parameters
        ----------
        atlas
            Atlas 对象。通常要求已经连接数据库，并包含该函数所需的 ``obs``、``var``、``X_HyS_data`` 或
            embedding 结果表。

        cell_ids
            当前 batch 对应的 Atlas 细胞 ID 数组。

        labels
            分类标签列表。

        table_name
            数据库表名。

        Notes
        -----
        这是内部 helper；除非需要扩展 scAtlasPy 内部流程，一般不建议在用户代码中直接调用。
        """
        df = pd.DataFrame({
            "atlas_cell_id": cell_ids,
            "cluster_id": labels.astype(np.int32)
        })

        atlas.connection.append(table_name, df)


    # 写 kmeans_centers
    def _write_centers(self, atlas: Atlas, table_name: str="kmeans_centers"):

        """将计算结果写入数据库表。

        该内部函数属于KMeans 聚类模块，用于支撑同一模块中的公共 API。

        基于 PCA loadings 和 Atlas minibatch 流式训练 MiniBatchKMeans，并写入聚类标签。

        它通常不会作为用户入口直接调用；直接调用时需要保证输入对象、数据库连接和相关临时表已经由上游步骤准备好。

        当前实现中会访问或生成的关键表包括：``kmeans_centers``。

        Parameters
        ----------
        atlas
            Atlas 对象。通常要求已经连接数据库，并包含该函数所需的 ``obs``、``var``、``X_HyS_data`` 或
            embedding 结果表。

        table_name
            数据库表名。

        Notes
        -----
        这是内部 helper；除非需要扩展 scAtlasPy 内部流程，一般不建议在用户代码中直接调用。
        """
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


    # 转换 pca + minibatch kmeans 聚类 训练
    def fit_kmeans(self, atlas: Atlas):

        """执行 ``fit_kmeans`` 的核心功能。

        基于 PCA loadings 和 Atlas minibatch 流式训练 MiniBatchKMeans，并写入聚类标签。

        函数会直接读取或写入 Atlas 数据库中的相关表，并尽量通过 SQL、分块读取或流式计算减少内存占用。

        整体用法和 Scanpy 中相近的 ``sap.tl.fit_kmeans`` 风格 API 类似，但结果保存在 Atlas
        数据库表中，便于后续步骤复用。

        当前实现中会访问或生成的关键表包括：``varm_PCs``。

        Parameters
        ----------
        atlas
            Atlas 对象。通常要求已经连接数据库，并包含该函数所需的 ``obs``、``var``、``X_HyS_data`` 或
            embedding 结果表。

        Returns
-------
        result
            函数返回结果。具体类型取决于参数设置和内部执行路径。

        Notes
        -----
        运行前请确认前序步骤已经生成所需的表达字段、过滤索引或 embedding 表。

        Examples
        --------
        调用该函数：::

            sap.tl.fit_kmeans(...)
        """

        # 读取 PCA components
        self.components_ = self.load_components(atlas)

        # 如果用户传入的 n_components 和数据库里实际 PCA 维度不同，给一个提示
        real_components = self.components_.shape[0]
        if real_components != self.n_components:
            self.n_components = real_components

        batch_count = 0

        # minibatch kmeans 聚类 训练
        for X_batch in tqdm(
                atlas.get_minibatch_dense(
                    batch_size=self.batch_size,
                    pass_mode="multi-pass",
                    buffer_batch_num=self.buffer_batch_num,
                    max_batches=self.fit_batches,
                ),
                total=self.fit_batches,
                desc="KMeans fit"
        ):

            t0 = time.time()

            X_pca = X_batch @ self.components_.T  # pca 转换

            X_pca = np.ascontiguousarray(X_pca, dtype=np.float32)

            if not np.isfinite(X_pca).all():
                raise ValueError(
                    f"X_pca 中存在 NaN/Inf: "
                    f"min={np.nanmin(X_pca)}, max={np.nanmax(X_pca)}"
                )

            t1 = time.time()
            self.kmeans.partial_fit(X_pca)   # KMeans 训练

            batch_count += 1

            if batch_count % 10 == 0:
                logger.info(f"[KMeans] partial_fit batch = {batch_count}/{self.fit_batches}")

        if batch_count == 0:
            raise RuntimeError("[KMeans] 没有获得任何 minibatch，无法训练 KMeans")

        return self


    # 转换 pca + minibatch kmeans 聚类 预测
    def predict_kmeans(
            self,
            atlas: Atlas,
            use_cluster_table: str="obs_cluster",
            write_to_obs: bool = True,
            use_obs_col: str = "kmeans"
    ):

        """执行 ``predict_kmeans`` 的核心功能。

        基于 PCA loadings 和 Atlas minibatch 流式训练 MiniBatchKMeans，并写入聚类标签。

        函数会直接读取或写入 Atlas 数据库中的相关表，并尽量通过 SQL、分块读取或流式计算减少内存占用。

        整体用法和 Scanpy 中相近的 ``sap.tl.predict_kmeans`` 风格 API 类似，但结果保存在 Atlas
        数据库表中，便于后续步骤复用。

        当前实现中会访问或生成的关键表包括：``kmeans_centers``、``obs``。

        Parameters
        ----------
        atlas
            Atlas 对象。通常要求已经连接数据库，并包含该函数所需的 ``obs``、``var``、``X_HyS_data`` 或
            embedding 结果表。

        use_cluster_table
            保存细胞聚类标签的数据库表名。

        write_to_obs
            是否将结果同步写入 ``obs`` 表。

        use_obs_col
            ``obs`` 中用于写入或读取结果的列名。

        Returns
-------
        result
            函数返回结果。具体类型取决于参数设置和内部执行路径。

        Notes
        -----
        运行前请确认前序步骤已经生成所需的表达字段、过滤索引或 embedding 表。

        Examples
        --------
        调用该函数：::

            sap.tl.predict_kmeans(...)
        """

        conn = atlas.connection

        conn.execute(f"DROP TABLE IF EXISTS {use_cluster_table}")
        conn.execute(f"""
            CREATE TABLE {use_cluster_table} (
                atlas_cell_id BIGINT,
                cluster_id INTEGER
            )
        """)

        # obs 中增加 kmeans 列
        if write_to_obs:
            obs_cols = [r[1] for r in conn.execute("PRAGMA table_info(obs)").fetchall()]
            if use_obs_col not in obs_cols:
                conn.execute(f"ALTER TABLE obs ADD COLUMN {use_obs_col} INTEGER")
            # 先清空旧结果
            conn.execute(f"UPDATE obs SET {use_obs_col} = NULL")

        # 读取 PCA components
        if self.components_ is None:
            self.components_ = self.load_components(atlas)

        cell_offset = 0
        predict_batch_count = 0

        # 转换阶段 使用 single-pass
        for X_batch in tqdm(
                atlas.get_minibatch_dense(
                    batch_size=self.batch_size,
                    pass_mode="single-pass",
                ),
                desc="KMeans predict"
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
            conn.append(use_cluster_table, batch_df)

            # 同步写回 obs.kmeans
            if write_to_obs:
                conn.register("_kmeans_batch_tmp", batch_df)

                conn.execute(f"""
                    UPDATE obs
                    SET {use_obs_col} = t.cluster_id
                    FROM _kmeans_batch_tmp t
                    WHERE obs.atlas_cell_id = t.atlas_cell_id
                """)

                conn.unregister("_kmeans_batch_tmp")

            cell_offset += n
            predict_batch_count += 1

            if predict_batch_count % 20 == 0:
                logger.info(
                    f"[KMeans] predicted cells = {cell_offset:,}, "
                    f"batches = {predict_batch_count}"
                )

        # 保存 centers
        self._write_centers(atlas, table_name="kmeans_centers")

        return self


    # 从数据库读取 PCA components，并恢复到 self.components_
    def load_components(self, atlas: Atlas, table_name: str="varm_PCs"):

        """执行 ``load_components`` 的核心功能。

        基于 PCA loadings 和 Atlas minibatch 流式训练 MiniBatchKMeans，并写入聚类标签。

        函数会直接读取或写入 Atlas 数据库中的相关表，并尽量通过 SQL、分块读取或流式计算减少内存占用。

        整体用法和 Scanpy 中相近的 ``sap.tl.load_components`` 风格 API 类似，但结果保存在 Atlas
        数据库表中，便于后续步骤复用。

        当前实现中会访问或生成的关键表包括：``varm_PCs``。

        Parameters
        ----------
        atlas
            Atlas 对象。通常要求已经连接数据库，并包含该函数所需的 ``obs``、``var``、``X_HyS_data`` 或
            embedding 结果表。

        table_name
            数据库表名。

        Returns
-------
        result
            函数返回结果。具体类型取决于参数设置和内部执行路径。

        Notes
        -----
        运行前请确认前序步骤已经生成所需的表达字段、过滤索引或 embedding 表。

        Examples
        --------
        调用该函数：::

            sap.tl.load_components(...)
        """
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

        return components_


    # 运行主函数
    def run(
            self,
            atlas: Atlas,
            use_cluster_table: str="obs_cluster",
            write_to_obs: bool = True,
            use_obs_col: str = "kmeans"
    ):
        """训练并写入流式 KMeans 聚类结果。

        该方法是 ``StreamingKMeans`` 的主流程入口，会先调用 ``fit_kmeans`` 训练模型，
        再调用 ``predict_kmeans`` 为全量细胞预测聚类标签。

        结果默认写入独立的聚类结果表，并可同步回 ``obs`` 表中的指定列，便于后续 UMAP 着色、差异基因分析和
        cluster-level 可视化。

        Parameters
        ----------
        atlas
            Atlas 对象。

            要求数据库中已经存在 PCA loadings 表，并且过滤索引和 minibatch 读取流程可以正常使用。

        use_cluster_table
            保存聚类结果的数据库表名。

        write_to_obs
            是否把聚类标签同步写入 ``obs`` 表。

        use_obs_col
            写入 ``obs`` 时使用的列名。

        Returns
        -------
        self
            当前 ``StreamingKMeans`` 对象。

        Examples
        --------
        运行完整 KMeans 流程：::

            model.run(atlas, use_cluster_table="obs_cluster", use_obs_col="kmeans")
        """

        #  kmeans 训练
        self.fit_kmeans(atlas)

        #  kmeans 转换
        self.predict_kmeans(
            atlas,
            use_cluster_table=use_cluster_table,
            write_to_obs=write_to_obs,
            use_obs_col=use_obs_col
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
        use_obs_col: str = "kmeans",
        use_cluster_table: str = "obs_cluster",
        write_to_obs: bool = True,
):

    """基于 PCA embedding 进行 MiniBatch K-means 聚类。

    该函数读取 ``obsm_X_pca`` 中的 PCA 坐标，拟合 MiniBatchKMeans 模型，并将聚类标签写入 ``obs`` 或聚类统计表。适合在大规模细胞数据上快速生成粗粒度 cluster。

    Parameters
    ----------
    atlas
        Atlas 对象。通常需要已经连接到 DuckDB 数据库，并包含该函数读取或写入所需的 ``obs``、``var``、表达矩阵或结果表。
    n_components
        输出维度或参与计算的主成分数量。
    n_clusters
        K-means 聚类数。
    batch_size
        每个小批量包含的细胞数量。较大值通常更快，但会增加内存占用。
    fit_batches
        用于拟合模型的小批量数量。较大值通常更稳定，但计算时间更长。
    buffer_batch_num
        预取缓冲区中的批次数量。较大值可提高吞吐，但会占用更多内存。
    use_obs_col
        读取或写入 ``obs`` 的列名。
    use_cluster_table
        保存聚类统计结果的数据库表名。
    write_to_obs
        是否把结果写回 ``obs`` 表。

    Returns
    -------
    Any
        函数返回底层实现产生的结果。

    Examples
    --------
    使用前 30 个主成分聚成 20 类::

        sap.tl.pca(atlas, n_components=50)
        sap.tl.kmeans(atlas, n_components=30, n_clusters=20)

    写入自定义 obs 列，并保存 cluster 统计表::

        sap.tl.kmeans(
            atlas,
            n_clusters=50,
            use_obs_col="kmeans_50",
            use_cluster_table="obs_cluster_kmeans_50",
        )"""

    t_start = time.time()

    conn = atlas.connection

    # 检查 PCA components 是否存在
    tables = [r[0] for r in conn.execute("SHOW TABLES").fetchall()]
    if "varm_PCs" not in tables:
        raise ValueError(
            "数据库中不存在 varm_PCs。\n"
            "请先运行 sap.tl.pca(atlas)，再运行 sap.tl.kmeans(atlas)。"
        )

    runner = StreamingKMeans(
        n_components=n_components,
        n_clusters=n_clusters,
        batch_size=batch_size,
        fit_batches=fit_batches,
        buffer_batch_num=buffer_batch_num,
    )

    runner.run(
        atlas,
        use_cluster_table=use_cluster_table,
        write_to_obs=write_to_obs,
        use_obs_col=use_obs_col,
    )

    t_end = time.time()

    logger.info(f" KMeans Done, 耗时 = {t_end - t_start:.2f} seconds")
