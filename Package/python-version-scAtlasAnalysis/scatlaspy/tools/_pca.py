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
            print(" PCA解释比例较低，可能需要检查数据或增加主成分数")
        elif total_ratio < 0.2:
            print(" PCA解释比例一般（单细胞中常见）")
        elif total_ratio < 0.4:
            print(" PCA解释比例正常")
        else:
            print(" PCA解释比例较高，结构较明显")

        return self

    # 降维
    def transform(self, atlas: Atlas):

        print("[PCA] Start transforming...")

        cell_offset = 0  # 关键：全局递增

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



# todo scanpy pca
# ============================================================
# Scanpy / ARPACK PCA：一次性载入 dense 矩阵后调用 sc.tl.pca
# ============================================================

def _choose_exact_batch_size(n_cells: int, preferred: int = 2048, max_batch_size: int = 20000):
    """
    选择一个可以整除 n_cells 的 batch_size，避免 minibatch_dense 丢最后 partial batch。

    例如：
        n_cells = 186961
        186961 = 6031 * 31
        会选择 6031，而不是 2048。
    """
    n_cells = int(n_cells)
    preferred = int(preferred)
    max_batch_size = int(max_batch_size)

    upper = min(n_cells, max_batch_size)

    divisors = []
    for b in range(1, upper + 1):
        if n_cells % b == 0:
            divisors.append(b)

    if len(divisors) == 0:
        return preferred

    # 选择 <= max_batch_size 的最大整除 batch_size
    return max(divisors)


class ScanpyArpackPCA:
    """
    使用 Scanpy 的 sc.tl.pca(..., svd_solver="arpack") 计算 PCA。

    注意：
    这个方法会把当前 atlas.minibatch_dense() 输出的矩阵一次性拼成 dense X，
    所以只适合中小数据或验证用，不适合超大数据。
    """

    def __init__(
            self,
            n_components: int = 30,
            preferred_batch_size: int = 2048,
            max_exact_batch_size: int = 20000,
            svd_solver: str = "arpack",
            random_state: int = 42,
            strict_n_obs: bool = True,
    ):
        self.n_components = int(n_components)
        self.preferred_batch_size = int(preferred_batch_size)
        self.max_exact_batch_size = int(max_exact_batch_size)
        self.svd_solver = svd_solver
        self.random_state = random_state
        self.strict_n_obs = bool(strict_n_obs)

        self.adata_ = None

    # --------------------------------------------------------
    # 建表：保持和原 StreamingPCA 一样的表结构
    # --------------------------------------------------------
    def _create_pca_table(self, atlas: Atlas, n_components=30, table_name="obsm_X_pca"):
        atlas.connection.execute(f"""DROP TABLE IF EXISTS {table_name};""")

        cols = ",\n".join([f"pc{i} FLOAT" for i in range(n_components)])

        sql = f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                atlas_cell_id INTEGER,
                {cols}
            );
        """
        atlas.connection.execute(sql)
        print("obsm_X_pca 新建完成")

    def _create_pcs_table(self, atlas: Atlas, n_components=30, table_name="varm_PCs"):
        atlas.connection.execute(f"""DROP TABLE IF EXISTS {table_name};""")

        cols = ",\n".join([f"pc{i} FLOAT" for i in range(n_components)])

        sql = f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                atlas_gene_id USMALLINT,
                {cols}
            );
        """
        atlas.connection.execute(sql)
        print("varm_PCs 新建完成")

    def _create_pca_stats_table(self, atlas: Atlas, table_name="uns_pca_stats"):
        atlas.connection.execute(f"""DROP TABLE IF EXISTS {table_name};""")

        sql = f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                pc_index USMALLINT,
                variance REAL,
                variance_ratio REAL
            );
        """
        atlas.connection.execute(sql)
        print("uns_pca_stats 新建完成")

    # --------------------------------------------------------
    # 获取 cell / gene 映射
    # --------------------------------------------------------
    def _get_cell_ids_in_matrix_order(self, atlas: Atlas):
        """
        minibatch_dense 通常按照 filter_cell_id 顺序输出。
        所以这里优先按照 obs.filter_cell_id 排序取 atlas_cell_id。
        """
        conn = atlas.connection

        obs_cols = conn.execute("PRAGMA table_info(obs)").fetchdf()["name"].tolist()

        if "filter_cell_id" in obs_cols:
            df = conn.execute("""
                SELECT atlas_cell_id
                FROM obs
                WHERE filter_cell_id IS NOT NULL
                ORDER BY filter_cell_id
            """).fetchdf()
        else:
            df = conn.execute("""
                SELECT atlas_cell_id
                FROM obs
                ORDER BY atlas_cell_id
            """).fetchdf()

        return df["atlas_cell_id"].to_numpy(dtype=np.int32)

    def _get_gene_ids_in_matrix_order(self, atlas: Atlas):
        """
        minibatch_dense 通常按照 filter_gene_id 顺序输出。
        所以这里优先按照 var.filter_gene_id 排序取 atlas_gene_id。
        """
        conn = atlas.connection

        var_cols = conn.execute("PRAGMA table_info(var)").fetchdf()["name"].tolist()

        if "filter_gene_id" in var_cols:
            df = conn.execute("""
                SELECT atlas_gene_id
                FROM var
                WHERE filter_gene_id IS NOT NULL
                ORDER BY filter_gene_id
            """).fetchdf()
        else:
            df = conn.execute("""
                SELECT atlas_gene_id
                FROM var
                ORDER BY atlas_gene_id
            """).fetchdf()

        return df["atlas_gene_id"].to_numpy(dtype=np.int32)

    # --------------------------------------------------------
    # 从 minibatch_dense 收集 dense 矩阵
    # --------------------------------------------------------
    def _collect_dense_matrix(self, atlas: Atlas):
        conn = atlas.connection

        cell_ids = self._get_cell_ids_in_matrix_order(atlas)
        n_cells = len(cell_ids)

        batch_size = _choose_exact_batch_size(
            n_cells=n_cells,
            preferred=self.preferred_batch_size,
            max_batch_size=self.max_exact_batch_size,
        )

        print("\n==== collect dense matrix for Scanpy PCA ====")
        print(f"n_cells = {n_cells:,}")
        print(f"preferred_batch_size = {self.preferred_batch_size}")
        print(f"chosen batch_size = {batch_size}")
        print(f"n_cells % batch_size = {n_cells % batch_size}")

        batches = []
        total_rows = 0
        n_genes = None

        for X_batch in tqdm(
                atlas.minibatch_dense(
                    batch_size=batch_size,
                    pass_mode="single-pass",
                ),
                desc="[Scanpy PCA] collect dense batches"
        ):
            X_batch = np.asarray(X_batch, dtype=np.float32)

            if n_genes is None:
                n_genes = X_batch.shape[1]

            batches.append(X_batch)
            total_rows += X_batch.shape[0]

        if len(batches) == 0:
            raise RuntimeError("[Scanpy PCA] 没有从 minibatch_dense 获得任何 batch")

        X = np.vstack(batches).astype(np.float32, copy=False)

        print("\n==== collected dense matrix ====")
        print("X shape:", X.shape)
        print("expected n_cells:", n_cells)
        print("actual rows:", total_rows)

        if self.strict_n_obs and X.shape[0] != n_cells:
            raise RuntimeError(
                f"[Scanpy PCA] 收集到的行数和 obs 不一致："
                f"X.shape[0]={X.shape[0]}, obs cells={n_cells}。\n"
                f"这通常说明 minibatch_dense 丢了最后 partial batch。"
            )

        if not np.isfinite(X).all():
            raise ValueError(
                f"[Scanpy PCA] X 中存在 NaN/Inf: "
                f"min={np.nanmin(X)}, max={np.nanmax(X)}"
            )

        return X, cell_ids[:X.shape[0]]

    # --------------------------------------------------------
    # 写结果表
    # --------------------------------------------------------
    def _write_obsm_X_pca(self, atlas: Atlas, X_pca, cell_ids, table_name="obsm_X_pca"):
        X_pca = np.asarray(X_pca, dtype=np.float32)

        df = pd.DataFrame(
            X_pca,
            columns=[f"pc{i}" for i in range(X_pca.shape[1])]
        )

        df.insert(0, "atlas_cell_id", cell_ids.astype(np.int32))

        atlas.connection.append(table_name, df)

        print(f"[Scanpy PCA] obsm_X_pca written: {len(df):,} cells")

    def _write_varm_PCs(self, atlas: Atlas, adata, table_name="varm_PCs"):
        if "PCs" not in adata.varm:
            raise ValueError("adata.varm 中不存在 PCs，sc.tl.pca 可能没有成功运行")

        pcs = np.asarray(adata.varm["PCs"], dtype=np.float32)

        gene_ids = self._get_gene_ids_in_matrix_order(atlas)

        if len(gene_ids) != pcs.shape[0]:
            print(
                f"[WARN] gene_ids 数量 {len(gene_ids)} 与 PCs 行数 {pcs.shape[0]} 不一致，"
                f"改用 np.arange。"
            )
            gene_ids = np.arange(pcs.shape[0], dtype=np.int32)

        df = pd.DataFrame(
            pcs,
            columns=[f"pc{i}" for i in range(pcs.shape[1])]
        )

        df.insert(0, "atlas_gene_id", gene_ids.astype(np.int32))

        atlas.connection.append(table_name, df)

        print(f"[Scanpy PCA] varm_PCs written: {len(df):,} genes")

    def _write_uns_pca_stats(self, atlas: Atlas, adata, table_name="uns_pca_stats"):
        if "pca" not in adata.uns:
            raise ValueError("adata.uns 中不存在 pca，sc.tl.pca 可能没有成功运行")

        variance = np.asarray(adata.uns["pca"]["variance"], dtype=np.float32)
        variance_ratio = np.asarray(adata.uns["pca"]["variance_ratio"], dtype=np.float32)

        df = pd.DataFrame({
            "pc_index": np.arange(len(variance), dtype=np.int32),
            "variance": variance,
            "variance_ratio": variance_ratio,
        })

        atlas.connection.append(table_name, df)

        print("[Scanpy PCA] uns_pca_stats written")

    # --------------------------------------------------------
    # 主运行逻辑
    # --------------------------------------------------------
    def run(self, atlas: Atlas):
        import scanpy as sc
        import anndata as ad

        t_start = time.time()

        print("\n==== sap.tl.pca_scanpy_arpack ====")
        print(f"[Scanpy PCA] n_components = {self.n_components}")
        print(f"[Scanpy PCA] svd_solver = {self.svd_solver}")
        print(f"[Scanpy PCA] random_state = {self.random_state}")

        self._create_pca_table(
            atlas,
            n_components=self.n_components,
            table_name="obsm_X_pca"
        )

        self._create_pcs_table(
            atlas,
            n_components=self.n_components,
            table_name="varm_PCs"
        )

        self._create_pca_stats_table(
            atlas,
            table_name="uns_pca_stats"
        )

        # 1. 收集当前 SQL 流程中的 dense X
        X, cell_ids = self._collect_dense_matrix(atlas)

        if self.n_components >= min(X.shape):
            raise ValueError(
                f"n_components={self.n_components} 必须小于 min(X.shape)={min(X.shape)}"
            )

        # 2. 构建 AnnData，并调用 Scanpy PCA
        adata = ad.AnnData(X=X)

        print("\n[Scanpy PCA] running sc.tl.pca ...")

        sc.tl.pca(
            adata,
            n_comps=self.n_components,
            svd_solver=self.svd_solver,
            random_state=self.random_state,
        )

        self.adata_ = adata

        X_pca = np.asarray(adata.obsm["X_pca"], dtype=np.float32)

        print("[Scanpy PCA] X_pca shape:", X_pca.shape)

        # 3. 写入 SQL 表
        self._write_obsm_X_pca(
            atlas,
            X_pca=X_pca,
            cell_ids=cell_ids,
            table_name="obsm_X_pca"
        )

        self._write_varm_PCs(
            atlas,
            adata=adata,
            table_name="varm_PCs"
        )

        self._write_uns_pca_stats(
            atlas,
            adata=adata,
            table_name="uns_pca_stats"
        )

        cum_ratio = np.cumsum(adata.uns["pca"]["variance_ratio"])

        print("[Scanpy PCA] 累计解释方差比例（前 {} 个主成分）：{:.4f}".format(
            len(cum_ratio),
            float(cum_ratio[-1])
        ))

        print("[Scanpy PCA] 前 10 个主成分的累计解释方差比例：")
        print(cum_ratio[:10])

        t_end = time.time()
        print(f"[Scanpy PCA] total time = {t_end - t_start:.2f} seconds")

        return self


def pca_scanpy_arpack(
        atlas: Atlas,
        n_components: int = 30,
        preferred_batch_size: int = 2048,
        max_exact_batch_size: int = 20000,
        svd_solver: str = "arpack",
        random_state: int = 42,
        strict_n_obs: bool = True,
):
    """
    Scanpy PCA 入口函数。

    等价目标：
        sc.tl.pca(
            adata,
            n_comps=n_components,
            svd_solver="arpack",
            random_state=42,
        )

    结果写入：
        obsm_X_pca
        varm_PCs
        uns_pca_stats

    注意：
        该方法会一次性拼接 dense X，适合中小数据或验证流程。
        超大数据请继续使用 sap.tl.pca 的 StreamingPCA。
    """

    runner = ScanpyArpackPCA(
        n_components=n_components,
        preferred_batch_size=preferred_batch_size,
        max_exact_batch_size=max_exact_batch_size,
        svd_solver=svd_solver,
        random_state=random_state,
        strict_n_obs=strict_n_obs,
    )

    runner.run(atlas)

    return runner