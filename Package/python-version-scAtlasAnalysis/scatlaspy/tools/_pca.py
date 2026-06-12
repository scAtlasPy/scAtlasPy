from ..data import Atlas
from sklearn.decomposition import IncrementalPCA
import numpy as np
from tqdm import tqdm
import pandas as pd
import time
import logging
logger = logging.getLogger('Atlas')

# 流式 PCA ；支持 minibatch 训练 + 推理
class StreamingPCA:
    """?? Atlas ?????? PCA ???

    ???? sklearn ``IncrementalPCA``?????????????????????? PCA?
    ???? Atlas minibatch ??????????????? ``partial_fit`` ??????
    ??????????? PCA ????????? ``obsm_X_pca``?``varm_PCs`` ?
    ``uns_pca_stats``?

    Parameters
    ----------
    n_components
        ????? PCA ??????
    fit_batches
        ???? IncrementalPCA ? minibatch ?????
    buffer_batch_num
        ``multi-pass`` ??? shuffle buffer ???? batch ???

    Notes
    -----
    ???? ``sap.tl.pca`` ??????????????????????????

    Examples
    --------
    ????? API ??::

        atlas.build_read_index(use_hvg=True)
        sap.tl.pca(atlas, n_components=50)

    ???????? PCA ?::

        model = StreamingPCA(n_components=50, fit_batches=1000)
        model.run(atlas)
        results = model.get_results()
        results["explained_variance_ratio"][:5]"""

    # 初始化
    def __init__(self,
                 n_components: int = 30,
                 fit_batches: int = 1000,
                 buffer_batch_num: int = 5,
                 ):

        """初始化对象。

        该内部函数属于PCA 计算模块，用于支撑同一模块中的公共 API。

        计算 PCA scores、PC loadings 和方差解释率，写入 ``obsm_X_pca``、``varm_PCs`` 和
        ``uns_pca_stats``。

        它通常不会作为用户入口直接调用；直接调用时需要保证输入对象、数据库连接和相关临时表已经由上游步骤准备好。

        Parameters
        ----------
        n_components
            输出维度或 PCA 主成分数量。

        fit_batches
            用于流式拟合模型的 minibatch 数量上限。

        buffer_batch_num
            shuffle buffer 中缓存的 minibatch 数量。

        Notes
        -----
        这是内部 helper；除非需要扩展 scAtlasPy 内部流程，一般不建议在用户代码中直接调用。
        """
        self.n_components = n_components # PCA 目标维度
        self.ipca = IncrementalPCA(n_components=n_components) # 创建 sklearn 的增量 PCA 模型
        self.fit_batches = fit_batches
        self.buffer_batch_num = buffer_batch_num

        self.components_ = None                 # components_ = 坐标轴  → 方向（往哪里投影）
        self.explained_variance_ = None         # variance = 每个轴有多重要   → 强度（这个方向多重要）
        self.explained_variance_ratio_ = None   # ratio = 占总信息多少        → 占比（解释了多少信息）


    # 新建 obsm_X_pca 表
    def _create_pca_table(self, atlas:Atlas, n_components: int = 30, table_name: str="obsm_X_pca"):

        """创建 Atlas 工作流所需的数据库表。

        该内部函数属于PCA 计算模块，用于支撑同一模块中的公共 API。

        计算 PCA scores、PC loadings 和方差解释率，写入 ``obsm_X_pca``、``varm_PCs`` 和
        ``uns_pca_stats``。

        它通常不会作为用户入口直接调用；直接调用时需要保证输入对象、数据库连接和相关临时表已经由上游步骤准备好。

        当前实现中会访问或生成的关键表包括：``obsm_X_pca``。

        Parameters
        ----------
        atlas
            Atlas 对象。通常要求已经连接数据库，并包含该函数所需的 ``obs``、``var``、``X_HyS_data`` 或
            embedding 结果表。

        n_components
            输出维度或 PCA 主成分数量。

        table_name
            数据库表名。

        Notes
        -----
        这是内部 helper；除非需要扩展 scAtlasPy 内部流程，一般不建议在用户代码中直接调用。
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

        """创建 Atlas 工作流所需的数据库表。

        该内部函数属于PCA 计算模块，用于支撑同一模块中的公共 API。

        计算 PCA scores、PC loadings 和方差解释率，写入 ``obsm_X_pca``、``varm_PCs`` 和
        ``uns_pca_stats``。

        它通常不会作为用户入口直接调用；直接调用时需要保证输入对象、数据库连接和相关临时表已经由上游步骤准备好。

        当前实现中会访问或生成的关键表包括：``varm_PCs``。

        Parameters
        ----------
        atlas
            Atlas 对象。通常要求已经连接数据库，并包含该函数所需的 ``obs``、``var``、``X_HyS_data`` 或
            embedding 结果表。

        n_components
            输出维度或 PCA 主成分数量。

        table_name
            数据库表名。

        Notes
        -----
        这是内部 helper；除非需要扩展 scAtlasPy 内部流程，一般不建议在用户代码中直接调用。
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

        """创建 Atlas 工作流所需的数据库表。

        该内部函数属于PCA 计算模块，用于支撑同一模块中的公共 API。

        计算 PCA scores、PC loadings 和方差解释率，写入 ``obsm_X_pca``、``varm_PCs`` 和
        ``uns_pca_stats``。

        它通常不会作为用户入口直接调用；直接调用时需要保证输入对象、数据库连接和相关临时表已经由上游步骤准备好。

        当前实现中会访问或生成的关键表包括：``uns_pca_stats``。

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
        atlas.connection.execute(f""" DROP TABLE IF EXISTS {table_name}; """)

        sql = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            pc_index USMALLINT,
            variance REAL,           --  float 32 单精度浮点数（4字节）
            variance_ratio REAL      --  float 32 单精度浮点数（4字节）
        );
        """
        atlas.connection.execute(sql)


    # 写 obsm_X_pca 表
    def _writer_obsm_x_pca(self, atlas: Atlas, X_batch: np.ndarray, cell_offset: int, table_name: str= "obsm_X_pca"):

        """将计算结果写入数据库表。

        该内部函数属于PCA 计算模块，用于支撑同一模块中的公共 API。

        计算 PCA scores、PC loadings 和方差解释率，写入 ``obsm_X_pca``、``varm_PCs`` 和
        ``uns_pca_stats``。

        它通常不会作为用户入口直接调用；直接调用时需要保证输入对象、数据库连接和相关临时表已经由上游步骤准备好。

        当前实现中会访问或生成的关键表包括：``obsm_X_pca``。

        Parameters
        ----------
        atlas
            Atlas 对象。通常要求已经连接数据库，并包含该函数所需的 ``obs``、``var``、``X_HyS_data`` 或
            embedding 结果表。

        X_batch
            当前 batch 的表达矩阵或 embedding 矩阵。

        cell_offset
            顺序写入细胞结果时使用的全局细胞偏移量。

        table_name
            数据库表名。

        Returns
        -------
        result
            函数返回结果。具体类型取决于参数设置和内部执行路径。

        Notes
        -----
        这是内部 helper；除非需要扩展 scAtlasPy 内部流程，一般不建议在用户代码中直接调用。
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

        """将计算结果写入数据库表。

        该内部函数属于PCA 计算模块，用于支撑同一模块中的公共 API。

        计算 PCA scores、PC loadings 和方差解释率，写入 ``obsm_X_pca``、``varm_PCs`` 和
        ``uns_pca_stats``。

        它通常不会作为用户入口直接调用；直接调用时需要保证输入对象、数据库连接和相关临时表已经由上游步骤准备好。

        当前实现中会访问或生成的关键表包括：``varm_PCs``。

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

        """将计算结果写入数据库表。

        该内部函数属于PCA 计算模块，用于支撑同一模块中的公共 API。

        计算 PCA scores、PC loadings 和方差解释率，写入 ``obsm_X_pca``、``varm_PCs`` 和
        ``uns_pca_stats``。

        它通常不会作为用户入口直接调用；直接调用时需要保证输入对象、数据库连接和相关临时表已经由上游步骤准备好。

        当前实现中会访问或生成的关键表包括：``uns_pca_stats``。

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
        pc_index = np.arange(len(self.explained_variance_), dtype=np.int32)

        df = pd.DataFrame({
            "pc_index": np.arange(len(self.explained_variance_), dtype=np.int32),
            "variance": self.explained_variance_.astype(np.float32),
            "variance_ratio": self.explained_variance_ratio_.astype(np.float32)
        })

        atlas.connection.append(table_name, df)


    # 训练 PCA
    def fit(self, atlas: Atlas):

        """执行 ``fit`` 的核心功能。

        计算 PCA scores、PC loadings 和方差解释率，写入 ``obsm_X_pca``、``varm_PCs`` 和
        ``uns_pca_stats``。

        函数会直接读取或写入 Atlas 数据库中的相关表，并尽量通过 SQL、分块读取或流式计算减少内存占用。

        整体用法和 Scanpy 中相近的 ``sap.tl.fit`` 风格 API 类似，但结果保存在 Atlas 数据库表中，便于后续步骤复用。

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

            sap.tl.fit(...)
        """

        batch_count = 0

        for X_batch in tqdm(
                atlas.get_minibatch_dense(
                    pass_mode="multi-pass",
                    buffer_batch_num=self.buffer_batch_num,
                    max_batches=self.fit_batches,
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


    # transform
    def transform(self, atlas: Atlas):

        """执行 ``transform`` 的核心功能。

        计算 PCA scores、PC loadings 和方差解释率，写入 ``obsm_X_pca``、``varm_PCs`` 和
        ``uns_pca_stats``。

        函数会直接读取或写入 Atlas 数据库中的相关表，并尽量通过 SQL、分块读取或流式计算减少内存占用。

        整体用法和 Scanpy 中相近的 ``sap.tl.transform`` 风格 API 类似，但结果保存在 Atlas
        数据库表中，便于后续步骤复用。

        Parameters
        ----------
        atlas
            Atlas 对象。通常要求已经连接数据库，并包含该函数所需的 ``obs``、``var``、``X_HyS_data`` 或
            embedding 结果表。

        Notes
        -----
        运行前请确认前序步骤已经生成所需的表达字段、过滤索引或 embedding 表。

        Examples
        --------
        调用该函数：::

            sap.tl.transform(...)
        """

        cell_offset = 0  # 关键：全局递增

        for X_batch in tqdm(atlas.get_minibatch_dense(pass_mode="single-pass")):

            X_pca = self.ipca.transform(X_batch)

            # 只写 obsm（每个batch）
            cell_offset = self._writer_obsm_x_pca(
                atlas,
                X_pca,
                cell_offset
            )


    # 主函数
    def fit_transform(self, atlas: Atlas):

        """执行 ``fit_transform`` 的核心功能。

        计算 PCA scores、PC loadings 和方差解释率，写入 ``obsm_X_pca``、``varm_PCs`` 和
        ``uns_pca_stats``。

        函数会直接读取或写入 Atlas 数据库中的相关表，并尽量通过 SQL、分块读取或流式计算减少内存占用。

        整体用法和 Scanpy 中相近的 ``sap.tl.fit_transform`` 风格 API 类似，但结果保存在 Atlas
        数据库表中，便于后续步骤复用。

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

            sap.tl.fit_transform(...)
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
        """获取数据库或对象中的内部信息。

        计算 PCA scores、PC loadings 和方差解释率，写入 ``obsm_X_pca``、``varm_PCs`` 和
        ``uns_pca_stats``。

        函数会直接读取或写入 Atlas 数据库中的相关表，并尽量通过 SQL、分块读取或流式计算减少内存占用。

        整体用法和 Scanpy 中相近的 ``sap.tl.get_results`` 风格 API 类似，但结果保存在 Atlas
        数据库表中，便于后续步骤复用。

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

            sap.tl.get_results(...)
        """
        return {
            "components": self.components_,
            "explained_variance": self.explained_variance_,
            "explained_variance_ratio": self.explained_variance_ratio_
        }


    # 从数据库读取 PCA components，并恢复到 self.components_
    def load_components(self, atlas: Atlas, table_name: str="varm_PCs"):

        """执行 ``load_components`` 的核心功能。

        计算 PCA scores、PC loadings 和方差解释率，写入 ``obsm_X_pca``、``varm_PCs`` 和
        ``uns_pca_stats``。

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
        """?????? PCA ?????

        ??????? PCA ????PC loadings ?? PCA ????????
        ``fit_transform`` ?? IncrementalPCA ?????????????????
        ???? Atlas ????

        Parameters
        ----------
        atlas
            Atlas ?????????????????????
            ``atlas.get_minibatch_dense`` ?????? minibatch?

        Returns
        -------
        None
            PCA ???loadings ??????????? Atlas ?????

        Examples
        --------
        ???? API ?? PCA::

            atlas.build_read_index(use_hvg=True)
            sap.tl.pca(atlas, n_components=50)

        ?????????????::

            model = StreamingPCA(n_components=30, fit_batches=500)
            model.run(atlas)
            atlas.head("obsm_X_pca")
            atlas.head("uns_pca_stats")"""

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


# 流式 PCA 入口
def pca(
        atlas: Atlas,
        n_components: int = 50,
        fit_batches: int = 1000,
        buffer_batch_num: int = 5,
):

    """基于 Atlas 表达矩阵计算 PCA。

    该函数从 Atlas 的小批量读取接口中读取表达矩阵，使用增量 PCA 方式拟合主成分，并把细胞坐标和方差解释比例写入数据库。它类似 Scanpy 的 ``sc.tl.pca``，但面向大规模数据采用分块训练。

    Parameters
    ----------
    atlas
        Atlas 对象。通常需要已经连接到 DuckDB 数据库，并包含该函数读取或写入所需的 ``obs``、``var``、表达矩阵或结果表。
    n_components
        输出维度或参与计算的主成分数量。
    fit_batches
        用于拟合模型的小批量数量。较大值通常更稳定，但计算时间更长。
    buffer_batch_num
        预取缓冲区中的批次数量。较大值可提高吞吐，但会占用更多内存。

    Returns
    -------
    Any
        函数返回底层实现产生的结果。

    Examples
    --------
    在默认读取索引上计算 50 个主成分::

        atlas.build_read_index(use_hvg=True)
        sap.tl.pca(atlas, n_components=50)

    使用更多拟合批次提高稳定性::

        sap.tl.pca(
            atlas,
            n_components=80,
            fit_batches=2000,
            buffer_batch_num=8,
        )"""

    t_start = time.time()

    pca_runner = StreamingPCA(
        n_components=n_components,
        fit_batches=fit_batches,
        buffer_batch_num=buffer_batch_num,
    )

    pca_runner.run(atlas)

    t_end = time.time()
    print(f" PCA Done, total time = {t_end - t_start:.2f} seconds")
