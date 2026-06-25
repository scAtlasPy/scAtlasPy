from ..data import Atlas
from sklearn.decomposition import IncrementalPCA
import numpy as np
from ..io import progress
import pandas as pd
import time
import logging
logger = logging.getLogger('Atlas')


class StreamingPCA:
    """面向 Atlas 数据库的流式 PCA 模型。

    该类封装 sklearn 的 ``IncrementalPCA``，用于在不一次性加载完整表达矩阵的
    情况下训练 PCA。它会通过 Atlas minibatch 读取接口分批获取 dense 表达矩阵，
    先使用 ``partial_fit`` 学习主成分，再将全量细胞分批投影到 PCA 空间。

    运行完成后，结果会写入 Atlas 数据库中的三张表：

    - ``obsm_X_pca``：每个细胞的 PCA 坐标；
    - ``varm_PCs``：每个基因在各主成分上的 loadings；
    - ``uns_pca_stats``：每个主成分的方差和方差解释比例。

    该类是 ``sap.tl.pca`` 的底层实现。普通用户通常直接调用公共函数
    ``pca``，只有在需要调试训练过程或自定义流式参数时才需要直接使用本类。

    Parameters
    ----------
    n_components
        需要计算的 PCA 主成分数量。
    fit_batches
        用于拟合 ``IncrementalPCA`` 的 minibatch 数量上限。
    buffer_batch_num
        ``multi-pass`` 读取时 shuffle buffer 中缓存的 batch 数量。
    batch_size
        每个 minibatch 包含的细胞数量。

    Notes
    -----
    运行前需要先通过 ``atlas.build_read_index(...)`` 构建读取索引，使
    ``atlas.get_minibatch_dense`` 能够按预期读出表达矩阵。

    Examples
    --------
    推荐的公共 API 用法::

        atlas.build_read_index(use_hvg=True)
        sap.tl.pca(atlas, n_components=50)
    """

    # 初始化
    def __init__(self,
                 n_components: int = 30,
                 fit_batches: int = 1000,
                 buffer_batch_num: int = 5,
                 batch_size: int = 2048,
                 ):

        """初始化流式 PCA 计算器。

        该方法只保存 PCA 参数并创建 sklearn ``IncrementalPCA`` 对象，不会立即
        读取数据库，也不会写入任何结果表。实际训练和写库发生在 ``fit``、
        ``transform`` 或 ``run`` 阶段。

        Parameters
        ----------
        n_components
            需要计算的 PCA 主成分数量。该值决定 ``obsm_X_pca``、``varm_PCs``
            和 ``uns_pca_stats`` 中的输出维度。
        fit_batches
            训练阶段最多读取多少个 minibatch 进行 ``partial_fit``。
        buffer_batch_num
            ``multi-pass`` 读取时预取或 shuffle buffer 中缓存的 minibatch 数量。
        batch_size
            每个 minibatch 中的细胞数量。

        Notes
        -----
        较大的 ``batch_size`` 和 ``fit_batches`` 通常能提高 PCA 稳定性，但会增加
        训练时间和单批内存占用。
        """
        self.n_components = n_components # PCA 目标维度
        self.ipca = IncrementalPCA(n_components=n_components) # 创建 sklearn 的增量 PCA 模型
        self.fit_batches = fit_batches
        self.buffer_batch_num = buffer_batch_num
        self.batch_size = batch_size

        self.components_ = None                 # components_ = 坐标轴      → 方向（往哪里投影）
        self.explained_variance_ = None         # variance = 每个轴有多重要   → 强度（这个方向多重要）
        self.explained_variance_ratio_ = None   # ratio = 占总信息多少        → 占比（解释了多少信息）


    # 新建 obsm_X_pca 表
    def _create_pca_table(self, atlas:Atlas, n_components: int = 30, table_name: str="obsm_X_pca"):

        """创建细胞 PCA 坐标结果表。

        该内部函数用于创建 ``obsm_X_pca`` 风格的结果表。表中每一行对应一个
        细胞，包含 ``atlas_cell_id`` 以及 ``pc0``、``pc1`` 等 PCA 坐标列。
        如果同名表已经存在，会先删除旧表再重新创建，避免不同 PCA 维度的旧结果
        混入新结果。

        Parameters
        ----------
        atlas
            Atlas 对象。要求对象已经连接到 DuckDB 数据库。
        n_components
            需要创建的 PCA 坐标列数量。
        table_name
            结果表名。默认值为 ``"obsm_X_pca"``。

        Returns
        -------
        None
            表结构直接在 Atlas 数据库中创建，不返回对象。

        Notes
        -----
        这是内部建表 helper，通常由 ``run`` 自动调用。
        """
        atlas.connection.execute(f""" DROP TABLE IF EXISTS {table_name}; """)

        cols = ",\n".join([f"pc{i} FLOAT" for i in range(n_components)])

        sql = f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
            atlas_cell_id INTEGER,
            {cols}
        );
        """
        atlas.connection.execute(sql)


    # 新建 varm_PCs 表
    def _create_pcs_table(self, atlas:Atlas,  n_components: int = 30, table_name: str="varm_PCs"):

        """创建基因 PCA loadings 结果表。

        该内部函数用于创建 ``varm_PCs`` 表。表中每一行对应一个基因，包含
        ``atlas_gene_id`` 以及 ``pc0``、``pc1`` 等主成分 loadings 列。
        该表后续会被 KMeans、UMAP 前处理或其他需要 PCA 投影的流程复用。

        Parameters
        ----------
        atlas
            Atlas 对象。要求对象已经连接到 DuckDB 数据库。
        n_components
            需要创建的 PCA loading 列数量。
        table_name
            结果表名。默认值为 ``"varm_PCs"``。

        Returns
        -------
        None
            表结构直接在 Atlas 数据库中创建，不返回对象。

        Notes
        -----
        这是内部建表 helper，通常由 ``run`` 自动调用。
        """
        atlas.connection.execute(f""" DROP TABLE IF EXISTS {table_name}; """)

        cols = ",\n".join([f"pc{i} FLOAT" for i in range(n_components)])

        sql = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            atlas_gene_id USMALLINT,
            {cols}
        );
        """
        atlas.connection.execute(sql)


    # 新建 uns_pca_stats 表
    def _create_pca_stats_table(self, atlas:Atlas, table_name: str="uns_pca_stats"):

        """创建 PCA 方差统计结果表。

        该内部函数用于创建 ``uns_pca_stats`` 表。表中每一行对应一个主成分，
        记录 ``pc_index``、该主成分解释的方差 ``variance``，以及方差解释比例
        ``variance_ratio``。

        Parameters
        ----------
        atlas
            Atlas 对象。要求对象已经连接到 DuckDB 数据库。
        table_name
            结果表名。默认值为 ``"uns_pca_stats"``。

        Returns
        -------
        None
            表结构直接在 Atlas 数据库中创建，不返回对象。

        Notes
        -----
        这是内部建表 helper，通常由 ``run`` 自动调用。
        """
        atlas.connection.execute(f""" DROP TABLE IF EXISTS {table_name}; """)

        sql = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            pc_index USMALLINT,
            variance REAL, 
            variance_ratio REAL 
        );
        """
        atlas.connection.execute(sql)


    # 写 obsm_X_pca 表
    def _writer_obsm_x_pca(self, atlas: Atlas, X_batch: np.ndarray, cell_offset: int, table_name: str= "obsm_X_pca"):

        """将一个 batch 的 PCA 坐标写入细胞结果表。

        该内部函数把当前 batch 的 PCA 投影结果转换为 ``float32`` DataFrame，
        按 ``cell_offset`` 生成连续的 ``atlas_cell_id``，并追加写入
        ``obsm_X_pca`` 风格的结果表。

        Parameters
        ----------
        atlas
            Atlas 对象。要求对象已经连接到 DuckDB 数据库。
        X_batch
            当前 batch 的 PCA 坐标矩阵，形状为
            ``(n_cells_in_batch, n_components)``。
        cell_offset
            当前 batch 第一个细胞对应的全局 ``atlas_cell_id`` 起点。
        table_name
            写入的 PCA 坐标表名。默认值为 ``"obsm_X_pca"``。

        Returns
        -------
        int
            下一个 batch 应使用的 ``cell_offset``。

        Notes
        -----
        该函数假设 minibatch 读取顺序与 ``atlas_cell_id`` 的顺序一致。
        """
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
    def _writer_varm_pcs(self, atlas: Atlas, table_name: str= "varm_PCs"):

        """将 PCA loadings 写入 ``varm_PCs`` 表。

        该内部函数读取 ``self.components_``，将 sklearn 中的
        ``(n_components, n_genes)`` 结构转置为 ``(n_genes, n_components)``，
        然后按基因写入 ``varm_PCs`` 风格的结果表。

        Parameters
        ----------
        atlas
            Atlas 对象。要求对象已经连接到 DuckDB 数据库。
        table_name
            写入的 loadings 表名。默认值为 ``"varm_PCs"``。

        Returns
        -------
        None
            PCA loadings 直接追加写入数据库表，不返回对象。

        Notes
        -----
        调用前需要确保 ``fit`` 已经完成，并且 ``self.components_`` 不为空。
        """
        pcs = self.components_.T.astype(np.float32)  # (n_genes, n_components)
        df = pd.DataFrame(
            pcs,
            columns=[f"pc{i}" for i in range(pcs.shape[1])]
        )
        # 插入 atlas_gene_id
        df.insert(0, "atlas_gene_id", np.arange(pcs.shape[0], dtype=np.int32))
        atlas.connection.append(table_name, df)


    # 写 uns_pca_stats 表
    def _writer_uns_pca_stats(self, atlas: Atlas, table_name: str="uns_pca_stats"):

        """将 PCA 方差解释统计写入 ``uns_pca_stats`` 表。

        该内部函数把 ``self.explained_variance_`` 和
        ``self.explained_variance_ratio_`` 整理成 DataFrame，并追加写入数据库。

        Parameters
        ----------
        atlas
            Atlas 对象。要求对象已经连接到 DuckDB 数据库。
        table_name
            写入的 PCA 统计表名。默认值为 ``"uns_pca_stats"``。

        Returns
        -------
        None
            PCA 统计结果直接追加写入数据库表，不返回对象。

        Notes
        -----
        调用前需要确保 ``fit`` 已经完成，并且方差统计数组已经生成。
        """
        pc_index = np.arange(len(self.explained_variance_), dtype=np.int32)

        df = pd.DataFrame({
            "pc_index": np.arange(len(self.explained_variance_), dtype=np.int32),
            "variance": self.explained_variance_.astype(np.float32),
            "variance_ratio": self.explained_variance_ratio_.astype(np.float32)
        })

        atlas.connection.append(table_name, df)


    # 训练 PCA
    def fit(self, atlas: Atlas):

        """分批拟合 IncrementalPCA 模型。

        该方法通过 ``atlas.get_minibatch_dense(pass_mode="multi-pass")`` 分批读取
        dense 表达矩阵，并使用 ``IncrementalPCA.partial_fit`` 进行流式训练。
        训练完成后，会把主成分、方差和方差解释比例保存到当前对象的属性中。

        该方法只训练模型，不写入 ``obsm_X_pca``、``varm_PCs`` 或
        ``uns_pca_stats`` 表。写库由 ``fit_transform`` 或 ``run`` 完成。

        Parameters
        ----------
        atlas
            Atlas 对象。要求已经连接数据库，并且已经构建可用于 dense minibatch
            读取的索引。

        Returns
        -------
        StreamingPCA
            当前 ``StreamingPCA`` 对象，便于链式调用。

        Notes
        -----
        如果没有读到任何 minibatch，函数会抛出 ``RuntimeError``，避免生成空的
        PCA 模型。

        """

        batch_count = 0

        for X_batch in progress(
                atlas.get_minibatch_dense(
                    pass_mode="multi-pass",
                    buffer_batch_num=self.buffer_batch_num,
                    max_batches=self.fit_batches,
                    batch_size=self.batch_size,
                ),
                total=self.fit_batches,
                desc="PCA"
        ):
            self.ipca.partial_fit(X_batch)

            batch_count += 1

            if batch_count % 10 == 0:
                logger.info(f"[PCA] partial_fit batch = {batch_count}/{self.fit_batches}")

        if batch_count == 0:
            raise RuntimeError("[PCA] 没有获得任何 minibatch，无法训练 PCA")

        # 保存结果
        self.components_ = self.ipca.components_.astype(np.float32)
        self.explained_variance_ = self.ipca.explained_variance_.astype(np.float32)
        self.explained_variance_ratio_ = self.ipca.explained_variance_ratio_.astype(np.float32)

        cum_ratio = np.cumsum(self.explained_variance_ratio_)

        logger.info("[PCA] 累计解释方差比例（前 {} 个主成分）：{:.4f}".format(
            len(self.explained_variance_ratio_),
            self.explained_variance_ratio_.sum()
        ))

        logger.info("[PCA] 前 10 个主成分的累计解释方差比例：")
        logger.info(cum_ratio[:10])

        logger.info("[PCA] 最终累计解释方差比例：{:.4f}".format(cum_ratio[-1]))

        total_ratio = cum_ratio[-1]

        if total_ratio < 0.1:
            logger.info(" PCA解释比例较低，可能需要检查数据或增加主成分数")
        elif total_ratio < 0.2:
            logger.info(" PCA解释比例一般（单细胞中常见）")
        elif total_ratio < 0.4:
            logger.info(" PCA解释比例正常")
        else:
            logger.info(" PCA解释比例较高，结构较明显")

        return self


    def transform(self, atlas: Atlas):
        """将全量细胞分批投影到 PCA 空间。

        该方法通过 ``atlas.get_minibatch_dense(pass_mode="single-pass")`` 读取全量
        表达矩阵，并使用已经拟合好的 ``IncrementalPCA`` 模型计算每个 batch 的
        PCA 坐标。每个 batch 的结果会立即追加写入 ``obsm_X_pca`` 表。

        Parameters
        ----------
        atlas
            Atlas 对象。要求已经连接数据库，并且已经构建可用于 dense minibatch
            读取的索引。

        Returns
        -------
        None
            PCA 坐标直接写入 ``obsm_X_pca`` 表，不返回对象。

        Notes
        -----
        调用前需要先运行 ``fit``，并确保 ``_create_pca_table`` 已经创建好目标表。

        """

        cell_offset = 0  # 全局递增

        for X_batch in progress(atlas.get_minibatch_dense(pass_mode="single-pass")):

            X_pca = self.ipca.transform(X_batch)

            # 只写 obsm（每个batch）
            cell_offset = self._writer_obsm_x_pca(
                atlas,
                X_pca,
                cell_offset
            )


    # 主函数
    def fit_transform(self, atlas: Atlas):

        """训练 PCA 模型并写入全部 PCA 结果。

        该方法先调用 ``fit`` 完成 IncrementalPCA 训练，然后写入基因 loadings
        和 PCA 方差统计，最后调用 ``transform`` 把全量细胞投影到 PCA 空间。

        Parameters
        ----------
        atlas
            Atlas 对象。要求已经连接数据库，并且 dense minibatch 读取流程可用。

        Returns
        -------
        StreamingPCA
            当前 ``StreamingPCA`` 对象。

        Notes
        -----
        调用前应先创建 ``obsm_X_pca``、``varm_PCs`` 和 ``uns_pca_stats`` 表；
        ``run`` 方法会自动完成这些建表步骤。

        """

        # 训练
        self.fit(atlas)

        # 写一次模型结果
        self._writer_varm_pcs(atlas)
        self._writer_uns_pca_stats(atlas)

        # transform（写 obsm）
        self.transform(atlas)
        return self


    # 获取结果
    def get_results(self):
        """获取当前 PCA 模型保存在内存中的结果。

        该方法返回 ``fit`` 后保存在当前对象中的 PCA 主成分、方差和方差解释比例。
        它不会访问数据库，也不会读取 ``obsm_X_pca``、``varm_PCs`` 或
        ``uns_pca_stats`` 表。

        Returns
        -------
        dict
            包含以下键：

            - ``"components"``：PCA loadings，形状为 ``(n_components, n_genes)``；
            - ``"explained_variance"``：每个主成分解释的方差；
            - ``"explained_variance_ratio"``：每个主成分的方差解释比例。

        Notes
        -----
        调用前需要先运行 ``fit`` 或 ``fit_transform``，否则返回的数组可能为
        ``None``。

        Examples
        --------
        查看前几个主成分的解释比例::

            model.fit(atlas)
            result = model.get_results()
            result["explained_variance_ratio"][:5]
        """
        return {
            "components": self.components_,
            "explained_variance": self.explained_variance_,
            "explained_variance_ratio": self.explained_variance_ratio_
        }


    # 从数据库读取 PCA components，并恢复到 self.components_
    def load_components(self, atlas: Atlas, table_name: str="varm_PCs"):

        """从数据库读取 PCA loadings。

        该方法读取 ``varm_PCs`` 风格的表，按 ``atlas_gene_id`` 排序后去掉基因
        ID 列，并转置回 sklearn ``IncrementalPCA`` 使用的
        ``(n_components, n_genes)`` 形状。

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
        该方法常用于检查数据库中保存的 PCA loadings 是否与内存中的
        ``self.components_`` 一致，也可供 KMeans 等后续流程读取 PCA 投影矩阵。

        Examples
        --------
        从数据库恢复 PCA loadings::

            components = model.load_components(atlas)
        """
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

        return components_


    def run(self, atlas: Atlas):
        """执行完整流式 PCA 计算流程。

        该方法会先创建 PCA 坐标表、PC loadings 表和 PCA 统计表，然后调用
        ``fit_transform`` 完成 IncrementalPCA 的训练和全量细胞投影。运行结束后，
        结果写入 Atlas 数据库。

        Parameters
        ----------
        atlas
            Atlas 对象。要求已经完成过滤索引构建，并能够通过
            ``atlas.get_minibatch_dense`` 读取表达矩阵 minibatch。

        Returns
        -------
        None
            PCA 坐标、loadings 和方差解释比例直接写入 Atlas 数据库表。

        Examples
        --------
        通过公共 API 运行 PCA::

            atlas.build_read_index(use_hvg=True)
            sap.tl.pca(atlas, n_components=50)
        """

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
            logger.info(" components 提取正确")
        if np.allclose(components, self.components_):
            logger.info(" components 提取正确")


def pca(
        atlas: Atlas,
        n_components: int = 50,
        fit_batches: int = 1000,
        batch_size: int = 2048,
):
    """基于 Atlas 表达矩阵计算 PCA。

    该函数是 scAtlasPy 的 PCA 公共入口。它从 Atlas 的 dense minibatch 读取接口
    中分批读取表达矩阵，使用 sklearn ``IncrementalPCA`` 流式拟合主成分，
    再把全量细胞投影到 PCA 空间，并将结果写入数据库。

    运行完成后会生成或覆盖以下结果表：

    - ``obsm_X_pca``：细胞 PCA 坐标；
    - ``varm_PCs``：基因 PCA loadings；
    - ``uns_pca_stats``：PCA 方差和方差解释比例。

    该流程类似 Scanpy 的 ``sc.tl.pca``，但为了适配大规模数据，训练和投影都
    通过 minibatch 分块完成。

    Parameters
    ----------
    atlas
        Atlas 对象。要求对象已经连接到 DuckDB 数据库，并且已经通过
        ``atlas.build_read_index(...)`` 构建好可用于 dense minibatch 读取的索引。
    n_components
        需要计算并保存的 PCA 主成分数量。默认值为 ``50``。
    fit_batches
        用于拟合 ``IncrementalPCA`` 的 minibatch 数量上限。较大的值通常更稳定，
        但训练时间更长。
    batch_size
        每个 minibatch 包含的细胞数量。较大的值通常吞吐更高，但会增加单批
        内存占用。

    Returns
    -------
    None
        PCA 结果直接写入 Atlas 数据库，不返回对象。

    Notes
    -----
    如果希望 PCA 只基于高变基因或过滤后的基因运行，需要在调用本函数前通过
    ``atlas.build_read_index`` 构建对应读取索引。

    Examples
    --------
    在默认读取索引上计算 50 个主成分::

        atlas.build_read_index(use_hvg=True)
        sap.tl.pca(atlas, n_components=50)
    """

    t_start = time.time()

    pca_runner = StreamingPCA(
        n_components=n_components,
        fit_batches=fit_batches,
        batch_size=batch_size,
    )

    pca_runner.run(atlas)

    t_end = time.time()
    logger.info(f" PCA Done, total time = {t_end - t_start:.2f} seconds")
