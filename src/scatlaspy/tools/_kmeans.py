from sklearn.cluster import MiniBatchKMeans
from ..io import progress
from ..data import Atlas
import numpy as np
import pandas as pd
import time
import logging

logger = logging.getLogger('Atlas')

# MiniBatchKMeans
class StreamingKMeans:

    """基于 PCA embedding 的流式 MiniBatchKMeans 聚类器。

    该类读取 Atlas 数据库中的 PCA loadings 表 ``varm_PCs``，并通过
    ``atlas.get_minibatch_dense`` 分批读取表达矩阵。每个 minibatch 会先投影到
    PCA 空间，再用于 sklearn ``MiniBatchKMeans`` 的流式训练或预测。

    完整流程分为两步：

    - ``fit_kmeans``：读取 minibatch，投影到 PCA 空间，并用
      ``partial_fit`` 训练 KMeans 中心；
    - ``predict_kmeans``：再次读取全量 minibatch，预测每个细胞的 cluster，
      并写入独立结果表和可选的 ``obs`` 列。

    该类是 ``sap.tl.kmeans`` 的底层实现。普通用户通常直接调用公共函数
    ``kmeans``。

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
    运行前需要先完成 PCA，确保数据库中存在 ``varm_PCs`` 表，并且 Atlas 的
    dense minibatch 读取索引已经构建完成。

    Examples
    --------
    推荐的公共 API 用法::

        sap.tl.pca(atlas, n_components=50)
        sap.tl.kmeans(atlas, n_components=30, n_clusters=20)
    """

    def __init__(
            self,
            n_components: int=50,
            n_clusters: int=2,
            batch_size: int=2048,
            fit_batches: int = 1000,
            buffer_batch_num: int = 5,
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

    # 写 obs_cluster
    def _write_clusters(self, atlas: Atlas, cell_ids: np.ndarray, labels: np.ndarray, table_name: str):

        """将一个 batch 的 KMeans 聚类标签写入结果表。

        该内部函数把当前 batch 的 ``atlas_cell_id`` 和 KMeans 预测得到的
        ``cluster_id`` 组成 DataFrame，并追加写入 ``obs_cluster`` 风格的结果表。

        Parameters
        ----------
        atlas
            Atlas 对象。要求对象已经连接到 DuckDB 数据库。
        cell_ids
            当前 batch 对应的 ``atlas_cell_id`` 数组。
        labels
            当前 batch 预测得到的 cluster 标签数组。
        table_name
            保存聚类标签的结果表名。

        Returns
        -------
        None
            聚类标签直接追加写入数据库表，不返回对象。

        Notes
        -----
        该 helper 当前只追加写入独立结果表；同步写回 ``obs`` 的逻辑在
        ``predict_kmeans`` 中完成。
        """
        df = pd.DataFrame({
            "atlas_cell_id": cell_ids,
            "cluster_id": labels.astype(np.int32)
        })

        atlas.connection.append(table_name, df)


    # 写 kmeans_centers
    def _write_centers(self, atlas: Atlas, table_name: str="kmeans_centers"):

        """将 KMeans 聚类中心写入数据库表。

        该内部函数读取 ``self.kmeans.cluster_centers_``，并将每个 cluster 在每个
        PCA 维度上的中心坐标展开成长表，写入 ``kmeans_centers`` 风格的表。
        输出表包含 ``cluster_id``、``pc_index`` 和 ``value`` 三列。

        Parameters
        ----------
        atlas
            Atlas 对象。要求对象已经连接到 DuckDB 数据库。
        table_name
            保存 KMeans 中心的表名。默认值为 ``"kmeans_centers"``。

        Returns
        -------
        None
            聚类中心直接写入数据库表，不返回对象。

        Notes
        -----
        调用前需要先完成 ``fit_kmeans``，否则 ``cluster_centers_`` 尚未生成。
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

        """分批训练 MiniBatchKMeans 模型。

        该方法先从 ``varm_PCs`` 表读取 PCA loadings，然后通过
        ``atlas.get_minibatch_dense(pass_mode="multi-pass")`` 分批读取表达矩阵。
        每个 batch 会先乘以 PCA loadings 转换到 PCA 空间，再调用
        ``MiniBatchKMeans.partial_fit`` 更新聚类中心。

        该方法只训练 KMeans 模型，不写入细胞聚类标签。标签写入由
        ``predict_kmeans`` 或 ``run`` 完成。

        Parameters
        ----------
        atlas
            Atlas 对象。要求已经连接到 DuckDB 数据库，数据库中存在
            ``varm_PCs`` 表，并且 dense minibatch 读取流程可用。

        Returns
        -------
        StreamingKMeans
            当前 ``StreamingKMeans`` 对象，便于链式调用。

        Notes
        -----
        如果数据库中的 PCA 维度与初始化时传入的 ``n_components`` 不一致，当前
        实现会使用数据库中实际读取到的 PCA 维度。

        """

        # 读取 PCA components
        self.components_ = self.load_components(atlas)

        # 如果用户传入的 n_components 和数据库里实际 PCA 维度不同，给一个提示
        real_components = self.components_.shape[0]
        if real_components != self.n_components:
            self.n_components = real_components

        batch_count = 0

        # minibatch kmeans 聚类 训练
        for X_batch in progress(
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
            add_obs_col: str = "kmeans"
    ):

        """为全量细胞预测 KMeans 聚类标签并写入数据库。

        该方法使用已经训练好的 ``MiniBatchKMeans`` 模型，对全量表达矩阵执行
        single-pass minibatch 读取。每个 batch 会先投影到 PCA 空间，再调用
        ``self.kmeans.predict`` 得到 cluster 标签。

        预测结果会写入 ``use_cluster_table`` 指定的独立结果表；当
        ``write_to_obs=True`` 时，也会同步写回 ``obs`` 表中的 ``add_obs_col`` 列。
        预测结束后，函数还会把聚类中心写入 ``kmeans_centers`` 表。

        Parameters
        ----------
        atlas
            Atlas 对象。要求已经连接到 DuckDB 数据库，并且 dense minibatch 读取
            流程可用。
        use_cluster_table
            保存细胞聚类标签的数据库表名。默认值为 ``"obs_cluster"``。
        write_to_obs
            是否将聚类标签同步写入 ``obs`` 表。默认值为 ``True``。
        add_obs_col
            写入 ``obs`` 表的聚类标签列名。默认值为 ``"kmeans"``。

        Returns
        -------
        StreamingKMeans
            当前 ``StreamingKMeans`` 对象。

        Notes
        -----
        调用前需要先运行 ``fit_kmeans``。如果 ``self.components_`` 为空，函数会
        自动从 ``varm_PCs`` 表读取 PCA loadings。

        Examples
        --------
        在已训练模型上写入聚类标签::

            model.fit_kmeans(atlas)
            model.predict_kmeans(atlas, add_obs_col="kmeans_20")
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
            if add_obs_col not in obs_cols:
                conn.execute(f"ALTER TABLE obs ADD COLUMN {add_obs_col} INTEGER")
            # 先清空旧结果
            conn.execute(f"UPDATE obs SET {add_obs_col} = NULL")

        # 读取 PCA components
        if self.components_ is None:
            self.components_ = self.load_components(atlas)

        cell_offset = 0
        predict_batch_count = 0

        # 转换阶段 使用 single-pass
        for X_batch in progress(
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
                    SET {add_obs_col} = t.cluster_id
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

        """从数据库读取 PCA loadings 供 KMeans 投影使用。

        该方法读取 ``varm_PCs`` 表，按 ``atlas_gene_id`` 排序后去掉基因 ID 列，
        并转置为 ``(n_components, n_genes)`` 形状。返回的矩阵会用于
        ``X_batch @ components_.T``，将表达矩阵投影到 PCA 空间。

        Parameters
        ----------
        atlas
            Atlas 对象。要求对象已经连接到 DuckDB 数据库。
        table_name
            读取的 PCA loadings 表名。默认值为 ``"varm_PCs"``。

        Returns
        -------
        numpy.ndarray
            PCA loadings 数组，形状为 ``(n_components, n_genes)``，类型为
            ``float32``。

        Notes
        -----
        该方法依赖 ``sap.tl.pca`` 已经生成 ``varm_PCs`` 表。

        Examples
        --------
        读取 PCA loadings::

            components = model.load_components(atlas)
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
            add_obs_col: str = "kmeans"
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

        add_obs_col
            写入 ``obs`` 时使用的列名。

        Returns
        -------
        self
            当前 ``StreamingKMeans`` 对象。

        Examples
        --------
        运行完整 KMeans 流程：::

            model.run(atlas, use_cluster_table="obs_cluster", add_obs_col="kmeans")
        """

        #  kmeans 训练
        self.fit_kmeans(atlas)

        #  kmeans 转换
        self.predict_kmeans(
            atlas,
            use_cluster_table=use_cluster_table,
            write_to_obs=write_to_obs,
            add_obs_col=add_obs_col
        )

        return self


#  入口函数
def kmeans(
        atlas: Atlas,
        n_components: int = 30,
        n_clusters: int = 10,
        batch_size: int = 2048,
        fit_batches: int = 1000,
        add_obs_col: str = "kmeans",
        use_cluster_table: str = "obs_cluster",
):

    """基于 PCA embedding 进行 MiniBatch K-means 聚类。

    该函数是 scAtlasPy 的 KMeans 公共入口。它读取 ``varm_PCs`` 中保存的 PCA
    loadings，并通过 Atlas dense minibatch 接口分批读取表达矩阵。
    训练阶段：会将每个 minibatch 投影到 PCA 空间并流式更新 ``MiniBatchKMeans``；
    预测阶段：会再次分批读取全量细胞，预测 cluster 标签并写入数据库。

    运行完成后通常会生成或更新三类结果：

    - ``use_cluster_table``：每个细胞的 ``cluster_id``；
    - ``obs[add_obs_col]``：可选的 obs 聚类标签列；
    - ``kmeans_centers``：每个 cluster 在 PCA 空间中的中心坐标。

    Parameters
    ----------
    atlas
        Atlas 对象。要求对象已经连接到 DuckDB 数据库，并且数据库中已经存在
        PCA loadings 表 ``varm_PCs``。
    n_components
        用于 KMeans 聚类的 PCA 主成分数量。默认值为 ``30``。
        如果 ``varm_PCs`` 中实际可用的维度与该值不同，底层类会使用数据库中
        实际读取到的 PCA 维度。
    n_clusters
        KMeans 聚类数量。默认值为 ``10``。
    batch_size
        每个 minibatch 包含的细胞数量。较大值通常更快，但会增加单批内存占用。
    fit_batches
        训练阶段最多读取多少个 minibatch。较大值通常使聚类中心更稳定，但会
        增加运行时间。
    add_obs_col
        写入 ``obs`` 表的聚类标签列名。默认值为 ``"kmeans"``。
    use_cluster_table
        保存每个细胞聚类标签的独立结果表名。默认值为 ``"obs_cluster"``。

    Returns
    -------
    None
        聚类结果直接写入 Atlas 数据库，不返回对象。

    Notes
    -----
    该函数不会自动运行 PCA。如果数据库中不存在 ``varm_PCs``，函数会报错并提示
    先运行 ``sap.tl.pca(atlas)``。

    Examples
    --------
    使用前 30 个主成分聚成 20 类::

        sap.tl.pca(atlas, n_components=50)
        sap.tl.kmeans(atlas, n_components=30, n_clusters=20)
    """

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
    )

    runner.run(
        atlas,
        use_cluster_table=use_cluster_table,
        add_obs_col=add_obs_col,
    )

    t_end = time.time()

    logger.info(f" KMeans Done, 耗时 = {t_end - t_start:.2f} seconds")
