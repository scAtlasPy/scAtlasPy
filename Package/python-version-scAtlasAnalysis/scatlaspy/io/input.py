import time
import gc
from tqdm import tqdm
from ..data import Atlas
import os
from anndata import AnnData
from scipy import sparse
import logging
import h5py
import numpy as np
import pandas as pd
import scanpy as sc
# 获取日志记录器
logger = logging.getLogger('Atlas')


# todo 大文件读取， 只支持 h5ad 格式的导入 ######################################################

# 主入口函数
def load_big_h5ad_to_duckdb(
    h5ad_path: str,
    atlas,
    batch_size: int = 1024,
):
    """
    从 h5ad 文件流式导入 DuckDB（CSR-only，全局游标版）

    关键工程原则：
      - scanpy backed：避免一次性加载 X
      - mega-batch   ：顺序磁盘 IO（HDF5 友好） ; mega-batch = “一次从磁盘顺序读多大”
      - mini-batch   ： 每批导入的细胞数量 ; mini-batch = “一次在内存里处理多大”
      - CSR-only     ：不存稠密矩阵
      - 全局游标     ：
    """


    mega_batch_size = batch_size * 8  # mega-batch：控制磁盘 IO 行为（性能参数，通常 = mini-batch × 常数）

    # --------------------------------------------------------
    #  1️⃣ 连接数据库
    # --------------------------------------------------------
    conn = atlas.connect("r+")
    atlas.connection = conn

    # ========================================================
    # 全局游标（★ 本版本的核心 ★）
    # ========================================================
    global_cell_id = 0          # obs.id 起始游标
    global_indptr_id = 0        # X_CSR_indptr.id 起始游标
    global_indptr_offset = 0    # 全局 nnz 偏移（CSR indptr 语义）
    global_data_id = 0          # X_CSR_data.id 起始游标

    obs_written = False
    var_written = False

    # --------------------------------------------------------
    # 2️⃣ backed 打开（只用于 schema + X）
    # --------------------------------------------------------
    adata_backed = sc.read_h5ad(h5ad_path, backed="r") # backed="r" 磁盘后端，只读（lazy loading）

    # 不把 X / obs / var 真正读进内存 ； 只有在切片或 .to_memory() 时才触发磁盘 IO
    n_cells = adata_backed.n_obs

    # --------------------------------------------------------
    # 3️⃣ 动态建表
    # --------------------------------------------------------
    # _create_obs_var_tables(conn)
    # _create_csr_tables(conn)

    # todo 建表（动态 schema）
    _create_obs_table_from_adata(conn, adata_backed[:1])
    _create_var_table_from_adata(conn, adata_backed[:1])
    ''' # 从 h5ad 文件中，只抽取 1 个 cell，
     # 用它来读取 obs / var 的字段结构（列名 + dtype），
     # 然后在 DuckDB 里创建 obs / var 表结构，但不导入任何数据。
    '''

    _create_csr_tables(conn)


    print(f"[INFO] 数据集维度: {adata_backed.n_obs:,} × {adata_backed.n_vars:,}")
    print("[INFO] 使用 scanpy backed + mega-batch + 全局游标模式")

    # ========================================================
    # 4️⃣ mega-batch / mini-batch 导入
    # ========================================================
    for mega_start in tqdm(
        range(0, n_cells, mega_batch_size),
        desc="Mega-batch（磁盘顺序读取）",
    ):
        mega_end = min(mega_start + mega_batch_size, n_cells)

        # ★ 真正触发磁盘读取的地方
        mega = adata_backed[mega_start:mega_end].to_memory()

        # ====================================================
        # mini-batch
        # ====================================================
        for start in range(0, mega.n_obs, batch_size):
            end = min(start + batch_size, mega.n_obs)
            adata = mega[start:end]

            # ---------------- batch 导入 obs ----------------
            global_cell_id = _append_obs_rows(
                adata,
                conn,
                start_cell_id=global_cell_id,
            )

            # ---------------- 导入 var（一次） ----------------
            if not var_written:
                _append_var(adata, conn)
                var_written = True

            # ---------------- batch 导入 X（CSRO） ----------------
            (
                global_indptr_id,
                global_indptr_offset,
                global_data_id,
            ) = _append_X_CSR_chunk(
                adata,
                conn,
                base_cell_id=global_cell_id - adata.n_obs,
                global_indptr_id=global_indptr_id,
                global_indptr_offset=global_indptr_offset,
                global_data_id=global_data_id,
            )

        # mega-batch 结束，释放内存
        del mega
        gc.collect()

    # --------------------------------------------------------
    # 5️⃣ 主键（非常重要：必须在数据写完之后）
    # --------------------------------------------------------
    conn.execute("ALTER TABLE obs ADD PRIMARY KEY (id)")
    conn.execute("ALTER TABLE var ADD PRIMARY KEY (id)")
    conn.execute("ALTER TABLE X_CSR_indptr ADD PRIMARY KEY (id)")
    conn.execute("ALTER TABLE X_CSR_data ADD PRIMARY KEY (id)")

    # ====================================================
    # 6️⃣ 导入 obsm / varm
    # ====================================================
    add_obsm_from_h5ad(h5ad_path, atlas)
    add_varm_from_h5ad(h5ad_path, atlas)

    print("✔ 全部数据成功导入 DuckDB（含 obsm / varm）")

# 推断数据类型
def _infer_duckdb_type_from_series(s: pd.Series) -> str:
    """根据 pandas dtype 推断 DuckDB 类型"""
    if pd.api.types.is_integer_dtype(s):
        return "BIGINT"
    if pd.api.types.is_float_dtype(s):
        return "DOUBLE"
    if pd.api.types.is_bool_dtype(s):
        return "BOOLEAN"
    return "VARCHAR"

# 建表
def _create_obs_table_from_adata(conn, adata):
    cols = ["id BIGINT", "cell_id VARCHAR"]

    for col in adata.obs.columns:
        duck_type = _infer_duckdb_type_from_series(adata.obs[col])
        cols.append(f'"{col}" {duck_type}')

    ddl = f"""
    CREATE OR REPLACE TABLE obs (
        {", ".join(cols)}
    )
    """
    conn.execute(ddl)

def _create_var_table_from_adata(conn, adata):
    cols = ["id BIGINT", "gene_id VARCHAR"]

    for col in adata.var.columns:
        duck_type = _infer_duckdb_type_from_series(adata.var[col])
        cols.append(f'"{col}" {duck_type}')

    ddl = f"""
    CREATE OR REPLACE TABLE var (
        {", ".join(cols)}
    )
    """
    conn.execute(ddl)

def _create_csr_tables(conn):
    """CSR-only 存储结构"""
    conn.execute(
        """
        CREATE OR REPLACE TABLE X_CSR_indptr (
            id BIGINT,
            cell_id VARCHAR,
            indptr BIGINT
        )
        """
    )
    conn.execute(
        """
        CREATE OR REPLACE TABLE X_CSR_data (
            id BIGINT,
            indices USMALLINT,  --  gene id
            data REAL,
            cell_index BIGINT   --  cell id
        )
        """
    )

# 导入
def _append_obs_rows(adata, conn, start_cell_id: int) -> int:
    n = adata.n_obs

    obs_df = adata.obs.copy()
    obs_df["cell_id"] = adata.obs.index.astype(str)
    obs_df["id"] = np.arange(start_cell_id, start_cell_id + n, dtype=np.int64)

    obs_df = obs_df[
        ["id", "cell_id"] + [c for c in obs_df.columns if c not in ("id", "cell_id")]
    ]

    conn.register("obs_df", obs_df)
    conn.execute("INSERT INTO obs SELECT * FROM obs_df")
    conn.unregister("obs_df")

    return start_cell_id + n

def _append_var(adata, conn):
    var_df = adata.var.copy()
    var_df["gene_id"] = adata.var.index.astype(str)
    var_df["id"] = np.arange(adata.n_vars, dtype=np.int64)

    var_df = var_df[
        ["id", "gene_id"] + [c for c in var_df.columns if c not in ("id", "gene_id")]
    ]

    conn.register("var_df", var_df)
    conn.execute("INSERT INTO var SELECT * FROM var_df")
    conn.unregister("var_df")

def _append_X_CSR_chunk(
    adata,
    conn,
    *,
    base_cell_id: int,
    global_indptr_id: int,
    global_indptr_offset: int,
    global_data_id: int,
):
    """
    导入一个 mini-batch 的 CSR 数据（不使用 COUNT(*)）

    CSR 语义：
      - indptr 仅存 indptr[1:]
      - indptr 为【全局 nnz 偏移】
      - cell_index 为【全局 cell id】
    """

    X = adata.X

    # 确保 CSR
    if not sparse.issparse(X):
        X = sparse.csr_matrix(X)
    else:
        X = X.tocsr()

    indptr = X.indptr.astype(np.int64)
    indices = X.indices.astype(np.uint16)
    data = X.data.astype(np.float32)

    # ================= indptr =================
    # 不存 indptr[0]
    row_nnz = np.diff(indptr)
    adj_indptr = indptr[1:] + global_indptr_offset

    indptr_df = pd.DataFrame(
        {
            "id": np.arange(
                global_indptr_id,
                global_indptr_id + len(adj_indptr),
                dtype=np.int64,
            ),
            "cell_id": adata.obs.index.astype(str),
            "indptr": adj_indptr,
        }
    )

    conn.register("indptr_df", indptr_df)
    conn.execute("INSERT INTO X_CSR_indptr SELECT * FROM indptr_df")
    conn.unregister("indptr_df")

    global_indptr_id += len(adj_indptr)

    # ================= data =================
    nnz = len(data)
    if nnz > 0:
        cell_index = np.repeat(
            np.arange(base_cell_id, base_cell_id + adata.n_obs, dtype=np.int64),
            row_nnz,
        )

        data_df = pd.DataFrame(
            {
                "id": np.arange(global_data_id, global_data_id + nnz, dtype=np.int64),
                "indices": indices,
                "data": data,
                "cell_index": cell_index,
            }
        )

        conn.register("data_df", data_df)
        conn.execute("INSERT INTO X_CSR_data SELECT * FROM data_df")
        conn.unregister("data_df")

        global_data_id += nnz
        global_indptr_offset += nnz

    return global_indptr_id, global_indptr_offset, global_data_id


def add_obsm_from_h5ad(h5ad_path: str, atlas, batch_size=100_000):
    logger.info("导入 obsm（h5py 直读，支持超大规模）")

    conn = atlas.connection

    with h5py.File(h5ad_path, "r") as f:

        if "obsm" not in f:
            logger.info("  - h5ad 中不存在 obsm，跳过")
            return

        obsm_grp = f["obsm"]

        for key in obsm_grp.keys():
            dset = obsm_grp[key]
            n_cells, k = dset.shape

            logger.info(f"  - obsm[{key}] shape={dset.shape}")

            cols = ", ".join([f"dim_{i} DOUBLE" for i in range(k)])
            conn.execute(f"""
                CREATE OR REPLACE TABLE obsm_{key} (
                    cell_index BIGINT,
                    {cols}
                )
            """)

            for start in range(0, n_cells, batch_size):
                end = min(start + batch_size, n_cells)

                block = dset[start:end]
                df = pd.DataFrame(
                    block,
                    columns=[f"dim_{i}" for i in range(k)]
                )
                df["cell_index"] = np.arange(start, end, dtype=np.int64)
                df = df[["cell_index"] + [c for c in df.columns if c != "cell_index"]]

                conn.register("obsm_df", df)
                conn.execute(f"INSERT INTO obsm_{key} SELECT * FROM obsm_df")
                conn.unregister("obsm_df")

    logger.info("obsm 导入完成")

def add_varm_from_h5ad(h5ad_path: str, atlas):
    logger.info("导入 varm（h5py 直读）")

    conn = atlas.connection

    with h5py.File(h5ad_path, "r") as f:

        if "varm" not in f:
            logger.info("  - h5ad 中不存在 varm，跳过")
            return

        varm_grp = f["varm"]

        for key in varm_grp.keys():
            dset = varm_grp[key]
            n_genes, k = dset.shape

            logger.info(f"  - varm[{key}] shape={dset.shape}")

            df = pd.DataFrame(
                dset[:],
                columns=[f"dim_{i}" for i in range(k)]
            )
            df["gene_index"] = np.arange(n_genes, dtype=np.int64)
            df = df[["gene_index"] + [c for c in df.columns if c != "gene_index"]]

            conn.register("varm_df", df)
            conn.execute(f"""
                CREATE OR REPLACE TABLE varm_{key} AS
                SELECT * FROM varm_df
            """)
            conn.unregister("varm_df")

    logger.info("varm 导入完成")

# todo 小文件读取 ： 支持多种数据格式的导入；  ######################################################

def load_small_to_duckdb( file_path , atlas:Atlas):
    '''
    :param file_path:
    :param atlas:
    :return:
    '''
    print("开始导入数据")
    adata = read_smart(file_path)
    load_AnnData(adata,atlas)
    print("✔ 全部数据成功导入 DuckDB ")

def read_smart(file_path, **kwargs):
    """
    根据文件后缀名智能选择相应的scanpy读取方法

    参数:
        file_path: 文件路径
        **kwargs: 传递给具体读取函数的额外参数
    返回:
        AnnData对象
    """
    # 获取文件后缀名（小写形式）
    file_ext = os.path.splitext(file_path)[1].lower()

    # 根据文件后缀选择对应的读取方法
    if file_ext == '.h5ad':
        # h5ad格式 - scanpy原生格式
        return sc.read_h5ad(file_path, **kwargs)
    elif file_ext == '.loom':
        # loom格式
        return sc.read_loom(file_path, **kwargs)
    elif file_ext in ['.mtx', '.mtx.gz']:
        # mtx格式 (Matrix Market格式)
        return sc.read_mtx(file_path, **kwargs)
    elif file_ext in ['.csv', '.csv.gz']:
        # csv格式
        return sc.read_csv(file_path, **kwargs)
    elif file_ext in ['.txt', '.tsv', '.tab']:
        # 文本格式，默认制表符分隔
        return sc.read_text(file_path, **kwargs)
    elif file_ext in ['.xlsx', '.xls']:
        # Excel格式
        return sc.read_excel(file_path, **kwargs)
    elif file_ext == '.h5':
        # 10x Genomics h5格式
        return sc.read_10x_h5(file_path, **kwargs)
    elif 'umi_tools' in file_path.lower():
        # UMI-tools格式
        return sc.read_umi_tools(file_path, **kwargs)
    else:
        # 如果不认识的后缀，尝试使用通用的read函数
        return sc.read(file_path, **kwargs)

def load_AnnData(adata:AnnData, atlas:Atlas, var_names_clean = True):
    """
    将anndata数据存入数据库中

    :param adata: AnnData对象，包含单细胞数据
    :param atlas: Atlas数据库实例
    :return: None
    """
    try:
        # 1. 准备数据表
        logger.info("准备数据表...")

        if hasattr(adata, 'obs'):
            add_obs(adata, atlas)  # 细胞表数据（对应obs）,
        else:
            print("Skipping obs layer")

        if hasattr(adata, 'var'):
            add_var(adata, atlas)  # 基因表数据（对应var）
        else:
            print("Skipping var layer")

        if hasattr(adata, 'X'):
            start_time = time.time()
            add_X_CSR_chunked(adata,atlas,chunk_size=4096) # 分块导入X表数据
            end_time = time.time()
            logger.info("######## X表的导入用时为： " + str(end_time - start_time))
        else:
            print("Skipping X layer")

        if hasattr(adata, 'obsm'):
            add_obsm(adata,atlas)
        else:
            print("Skipping obsm layer")

        if hasattr(adata, 'varm'):
            add_varm(adata,atlas)
        else:
            print("Skipping varm layer")

        # 显示表结构
        logger.debug("数据库表结构:")
        tables = atlas.connection.execute("SHOW TABLES")
        if tables:
            logger.debug(f"数据库中的表: {tables}")

        logger.info("AnnData数据成功加载到数据库")

    except Exception as e:
        logger.error(f"加载数据失败: {str(e)}")
        logger.exception("加载数据异常详情:")
        raise

def add_obs(adata:AnnData, atlas:Atlas):

    logger.info("导入obs数据")
    # obs_df = adata.obs.reset_index().rename(columns={'index': 'cell_id'})
    # 方法1：推荐使用，最简洁
    obs_df = adata.obs.copy()
    obs_df['cell_id'] = adata.obs.index

    obs_df['id'] = range(len(obs_df))  # 添加 id 列

    obs_df = obs_df[['id', 'cell_id'] + [col for col in obs_df.columns if col not in ['id', 'cell_id']]]  # 直接指定列的顺序
    logger.info(f"obs表数据准备完成，行数: {len(obs_df)}")

    atlas.connection.register('obs_df', obs_df)
    atlas.connection.execute("CREATE OR REPLACE TABLE obs AS SELECT * FROM obs_df")
    atlas.connection.execute("ALTER TABLE obs ADD PRIMARY KEY (id)")  # 设置ID字段为主码，保证唯一性
    atlas.connection.unregister('obs_df')
    logger.info("导入obs数据成功")

def add_var(adata:AnnData, atlas:Atlas):
    #  todo 是否需要添加 一列递增的ID作为主码，保证数据的唯一性。??? 基因名是否需要进行清洗操作
    # add var data into duckdb
    logger.info("导入var数据")
    var_df = adata.var.reset_index().rename(columns={'index': 'gene_id'})
    var_df['id'] = range(len(var_df))
    var_df = var_df[['id', 'gene_id'] + [col for col in var_df.columns if col not in ['id', 'gene_id']]]  # 直接指定列的顺序
    logger.info(f"var表数据准备完成，行数: {len(var_df)}")

    atlas.connection.register('var_df', var_df)
    atlas.connection.execute("CREATE OR REPLACE TABLE var AS SELECT * FROM var_df")
    atlas.connection.execute("ALTER TABLE var ADD PRIMARY KEY (id)")  # 设置ID字段为主码，保证唯一性
    atlas.connection.unregister('var_df')
    logger.info("导入var数据成功")

def add_obsm(adata: AnnData, atlas: Atlas):
    """
    small / big 通用 obsm schema
    obsm_{key}(cell_index, dim_0, dim_1, ...)
    """
    logger.info("导入 obsm（统一 schema）")

    conn = atlas.connection
    n_cells = adata.n_obs

    if not adata.obsm:
        logger.info("  - 无 obsm，跳过")
        return

    for key, mat in adata.obsm.items():
        logger.info(f"  - obsm[{key}] shape = {mat.shape}")

        # 强制 numpy（避免 pandas 稀疏坑）
        mat = np.asarray(mat)

        df = pd.DataFrame(
            mat,
            columns=[f"dim_{i}" for i in range(mat.shape[1])]
        )

        # ★ 关键：cell_index = obs.id 的语义
        df.insert(0, "cell_index", np.arange(n_cells, dtype=np.int64))

        conn.register("obsm_df", df)
        conn.execute(f"""
            CREATE OR REPLACE TABLE obsm_{key} AS
            SELECT * FROM obsm_df
        """)
        conn.unregister("obsm_df")

    logger.info("obsm 导入完成（统一 schema）")

def add_varm(adata: AnnData, atlas: Atlas):
    """
    small / big 通用 varm schema
    varm_{key}(gene_index, dim_0, dim_1, ...)
    """
    logger.info("导入 varm（统一 schema）")

    conn = atlas.connection
    n_genes = adata.n_vars

    if not adata.varm:
        logger.info("  - 无 varm，跳过")
        return

    for key, mat in adata.varm.items():
        logger.info(f"  - varm[{key}] shape = {mat.shape}")

        mat = np.asarray(mat)

        df = pd.DataFrame(
            mat,
            columns=[f"dim_{i}" for i in range(mat.shape[1])]
        )

        # ★ gene_index = var.id 的语义
        df.insert(0, "gene_index", np.arange(n_genes, dtype=np.int64))

        conn.register("varm_df", df)
        conn.execute(f"""
            CREATE OR REPLACE TABLE varm_{key} AS
            SELECT * FROM varm_df
        """)
        conn.unregister("varm_df")

    logger.info("varm 导入完成（统一 schema）")

def add_X_CSR_chunked( adata: AnnData, atlas: Atlas, chunk_size: int = 2048):
    """
    仅导入 CSR 结构：
      - X_CSR_indptr(id, cell_id, indptr)
      - X_CSR_data(id, indices, data, cell_index)

    并在导入完成后，直接构建 cell_index
    """
    logger.info("开始导入 CSR 表达矩阵（无 X 稠密表）")

    conn = atlas.connect("r+")
    atlas.connection = conn

    cell_ids = adata.obs.index
    n_cells = adata.n_obs

    # ===================== 建表 =====================
    conn.execute("""
        CREATE OR REPLACE TABLE X_CSR_indptr (
            id BIGINT PRIMARY KEY,
            cell_id VARCHAR,
            indptr BIGINT
        )
    """)

    conn.execute("""
        CREATE OR REPLACE TABLE X_CSR_data (
            id BIGINT PRIMARY KEY,
            indices USMALLINT,
            data REAL,
            cell_index BIGINT
        )
    """)

    conn.execute("BEGIN TRANSACTION")

    try:
        total_chunks = (n_cells + chunk_size - 1) // chunk_size

        global_data_counter = np.int64(0)
        global_indptr_offset = np.int64(0)

        for chunk_idx in tqdm(range(total_chunks), desc="导入 CSR chunks"):
            start = chunk_idx * chunk_size
            end = min(start + chunk_size, n_cells)
            size = end - start

            # === 取数据 ===
            X_chunk = adata.X[start:end]

            if sparse.issparse(X_chunk):
                csr = X_chunk.tocsr()
            else:
                csr = sparse.csr_matrix(X_chunk)

            data = csr.data
            indices = csr.indices.astype(np.int64)
            indptr = csr.indptr.astype(np.int64)

            # ================= indptr 表 =================
            adj_indptr = indptr[1:] + global_indptr_offset

            indptr_df = pd.DataFrame({
                "id": np.arange(start, end, dtype=np.int64),
                "cell_id": cell_ids[start:end],
                "indptr": adj_indptr
            })

            conn.execute("INSERT INTO X_CSR_indptr SELECT * FROM indptr_df")

            global_indptr_offset = adj_indptr[-1]

            # ================= data 表 =================
            if len(data) > 0:
                nnz = len(data)
                data_ids = np.arange(
                    global_data_counter,
                    global_data_counter + nnz,
                    dtype=np.int64
                )

                # 直接在 chunk 内构造 cell_index（CSR → COO）
                row_lengths = np.diff(indptr)
                cell_index = np.repeat(
                    np.arange(start, end, dtype=np.int64),
                    row_lengths
                )

                data_df = pd.DataFrame({
                    "id": data_ids,
                    "indices": indices,
                    "data": data,
                    "cell_index": cell_index
                })

                conn.execute("INSERT INTO X_CSR_data SELECT * FROM data_df")

                global_data_counter += nnz

            # === 清理 ===
            del X_chunk, csr, indptr_df
            if len(data) > 0:
                del data_df
            gc.collect()

        conn.execute("COMMIT")

        logger.info(
            f"导入完成：cells={n_cells:,}, nnz={global_data_counter:,}"
        )
        print(f"✔ 导入完成：cells={n_cells:,}, nnz={global_data_counter:,}")

        return True

    except Exception as e:
        conn.execute("ROLLBACK")
        logger.error(f"CSR 导入失败: {e}")
        raise


# todo 基因名清洗 ：先导入，再清洗， var表  ######################################

def clean_genes_in_database(atlas: Atlas, gene_id_column: str = "gene_id"):
    """
    在数据库层面清洗基因名，直接操作数据库表
    参数:
    ----------
    atlas : Atlas
        Atlas数据库对象
        gene_id_column : str  基因ID列名，默认为'gene_id'
        在数据库中添加后缀模式：为重复基因添加 _1, _2, _3 等后缀
        仅处理 var 表
    """
    logger.info(f"开始在数据库 var表 中清洗基因名 ")

    # 检查表是否存在
    tables = atlas.connection.execute("SHOW TABLES").df()
    if 'var' not in tables['name'].values:
        logger.error("var表 不存在，请先导入数据")
        return False

    logger.info("开始添加后缀模式...")

    # 1. 构建带后缀的临时 var 表（var_with_suffix）
    atlas.connection.execute(f"""
        CREATE OR REPLACE TEMPORARY TABLE var_with_suffix AS
        WITH ranked_genes AS (
            SELECT *,
                   ROW_NUMBER() OVER (
                       PARTITION BY {gene_id_column}
                       ORDER BY id
                   ) AS rn
            FROM var
        )
        SELECT
            id,
            CASE
                WHEN rn = 1 THEN {gene_id_column} -- 第一个出现的基因名保持不变         
                ELSE {gene_id_column} || '_' || (rn - 1)::VARCHAR -- 后续重复基因添加后缀：gene_1, gene_2, ...
            END AS {gene_id_column}
        FROM ranked_genes
        ORDER BY id
    """)

    # 创建带后缀的基因名临时表 var_with_suffix
    # id | gene_id
    # ---|---------
    # 1  | TP53     -- 第一个TP53保持原名
    # 2  | EGFR     -- EGFR只有一个，保持原名
    # 3  | TP53_1   -- 第二个TP53添加_1后缀
    # 4  | BRAF     -- BRAF只有一个，保持原名
    # 5  | TP53_2   -- 第三个TP53添加_2后缀

    # 2. 记录 gene_id 实际发生变化的映射关系（仅用于日志）
    gene_mapping = atlas.connection.execute(f"""
        SELECT
            v.{gene_id_column}  AS original_gene_id,
            vs.{gene_id_column} AS new_gene_id
        FROM var v
        JOIN var_with_suffix vs
            ON v.id = vs.id
        WHERE v.{gene_id_column} != vs.{gene_id_column}
    """).df()
    # 获取基因名映射关系。gene_mapping数据框
    #    original_gene_id  new_gene_id
    # 0              TP53       TP53_1
    # 1              TP53       TP53_2

    # 3.  根据是否存在重复基因输出不同日志
    if len(gene_mapping) > 0:
        logger.info(
            f"发现 {len(gene_mapping)} 个重复基因，已成功添加后缀"
        )
    else:
        logger.info("未发现重复基因，var 表保持不变")

    # 4. 更新var表
    if(len(gene_mapping)>0):
        atlas.connection.execute("DROP TABLE var")
        atlas.connection.execute("ALTER TABLE var_with_suffix RENAME TO var")
        atlas.connection.execute("ALTER TABLE var ADD PRIMARY KEY (id)")
        logger.info("var表已更新")

    logger.info("清洗基因名 已完成!")
    return True

# 示例
# var 表
# id | gene_id
# ---|--------
# 1  | TP53
# 2  | EGFR
# 3  | TP53    -- 重复基因
# 4  | BRAF
# 5  | TP53    -- 重复基因

# 添加后缀模式：为重复基因添加 _1, _2, _3 等后缀
# 更新后的var表：
# id | gene_id
# ---|---------
# 1  | TP53
# 2  | EGFR
# 3  | TP53_1
# 4  | BRAF
# 5  | TP53_2