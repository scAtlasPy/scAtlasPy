import json
import time
import gc
from typing import Counter
from collections import defaultdict
from matplotlib.backend_tools import ToolForward
from tqdm import tqdm
from ..data import Atlas
import os
from anndata import AnnData
from scipy import sparse
from datetime import datetime
import logging
import h5py
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from functools import partial
import scanpy as sc
# 获取日志记录器
logger = logging.getLogger('Atlas')


# todo 大文件读取 ######################################################

def inspect_h5ad_structure(h5ad_path: str):
    """
    检查h5ad文件的结构
    """
    with h5py.File(h5ad_path, 'r') as f:
        print("=== H5AD文件结构 ===")

        def print_structure(name, obj, indent=0):
            spaces = " " * indent
            if isinstance(obj, h5py.Group):
                print(f"{spaces}{name}/")
                # 打印组的属性
                if obj.attrs:
                    print(f"{spaces}  属性: {dict(obj.attrs)}")
                # 递归打印子项
                for key in obj.keys():
                    print_structure(key, obj[key], indent + 2)
            elif isinstance(obj, h5py.Dataset):
                print(f"{spaces}{name}: {obj.shape} {obj.dtype}")
                # 打印数据集的属性
                if obj.attrs:
                    print(f"{spaces}  属性: {dict(obj.attrs)}")

        for key in f.keys():
            print_structure(key, f[key])


def load_AnnData_from_h5ad(h5ad_path: str, atlas: Atlas, batch_size=2048):
    """
    分批次读取h5ad文件并存入数据库

    """
    atlas.connection = atlas.connect("r+")
    # todo X表的 cell_id 要是真实的； X表的字段名先用gene_1 ； obs 和 var 表的字段不见了


    try:
        # 1. 读取var和varm信息
        logger.info("读取var和varm信息...")
        with h5py.File(h5ad_path, 'r') as f:
            # 1. 读取var信息（一次性）
            if 'var' in f:
                add_var_from_h5ad(f, atlas)

            # 处理varm
            if 'varm' in f:
                add_varm_from_h5ad(f, atlas)

            # 2. 获取总细胞数
            obs_group = f['obs']
            n_cells = obs_group['_index'].shape[0]
            logger.info(f"总细胞数: {n_cells}")

            var_group = f['var']
            n_genes = var_group['_index'].shape[0]
            logger.info(f"总细胞数: {n_genes}")

            # 3. 初始化CSR表
            initialize_csr_tables(atlas)

            # 4. 分批次处理数据
            n_batches = (n_cells + batch_size - 1) // batch_size

            for batch_idx in tqdm(range(n_batches)):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, n_cells)

                # # 读取和处理当前批次
                # process_batch(h5ad_path, atlas, start_idx, end_idx, batch_idx)

                # 处理当前批次
                process_batch_simple(f, atlas, start_idx, end_idx, batch_idx)

                gc.collect()

            logger.info("X表创建完成")

        # 5. 处理其他数据
        logger.info("处理其他数据...")
        with h5py.File(h5ad_path, 'r') as f:
            if 'obsm' in f:
                add_obsm_from_h5ad(f, atlas)
            if 'obsp' in f:
                add_obsp_from_h5ad(f, atlas)
            if 'uns' in f:
                add_uns_from_h5ad(f, atlas)

        logger.info("所有数据成功加载到数据库")

    except Exception as e:
        logger.error(f"加载数据失败: {str(e)}")
        raise


def process_batch_simple(f, atlas: Atlas, start_idx: int, end_idx: int, batch_idx: int):
    """处理一个批次的数据（简化版）"""
    # 读取obs数据
    obs_group = f['obs']

    # 获取细胞索引
    if '_index' in obs_group:
        cell_ids = obs_group['_index'][start_idx:end_idx]
    elif 'index' in obs_group:
        cell_ids = obs_group['index'][start_idx:end_idx]
    else:
        cell_ids = [f"cell_{i}" for i in range(start_idx, end_idx)]

    print("已获取 cell_ids", cell_ids)

    # # 处理字节字符串
    # if cell_ids and isinstance(cell_ids[0], bytes):
    #     cell_ids = [cid.decode('utf-8') for cid in cell_ids]

    # 读取obs数据
    obs_data = {}
    for key in obs_group.keys():
        if isinstance(obs_group[key], h5py.Dataset) and key not in ['_index', 'index']:
            data = obs_group[key][start_idx:end_idx]
            if data.dtype.kind == 'S':
                data = [d.decode('utf-8') for d in data]
            obs_data[key] = data

    # 创建obs DataFrame
    obs_df = pd.DataFrame(obs_data) if obs_data else pd.DataFrame()
    obs_df['cell_id'] = cell_ids
    obs_df['id'] = range(start_idx, end_idx)
    obs_df = obs_df[['id', 'cell_id'] + [col for col in obs_df.columns if col not in ['id', 'cell_id']]]

    # 存储到数据库
    if batch_idx == 0:
        atlas.connection.register('obs_df', obs_df)
        atlas.connection.execute("CREATE OR REPLACE TABLE obs AS SELECT * FROM obs_df")
        atlas.connection.execute("ALTER TABLE obs ADD PRIMARY KEY (id)")
        atlas.connection.unregister('obs_df')
    else:
        atlas.connection.register('obs_df', obs_df)
        atlas.connection.execute("INSERT INTO obs SELECT * FROM obs_df")
        atlas.connection.unregister('obs_df')

    # 读取并处理X数据
    x_item = f['X']
    if isinstance(x_item, h5py.Dataset):
        # 密集矩阵
        X_batch = x_item[start_idx:end_idx, :]
    else:
        # 稀疏矩阵
        data = x_item['data'][:]
        indices = x_item['indices'][:]
        indptr = x_item['indptr'][:]

        batch_start_ptr = indptr[start_idx]
        batch_end_ptr = indptr[end_idx]

        batch_data = data[batch_start_ptr:batch_end_ptr]
        batch_indices = indices[batch_start_ptr:batch_end_ptr]
        batch_indptr = indptr[start_idx:end_idx + 1] - batch_start_ptr

        n_cells = end_idx - start_idx
        n_genes = x_item.attrs['shape'][1]

        X_batch = sparse.csr_matrix((batch_data, batch_indices, batch_indptr), shape=(n_cells, n_genes))

    # 处理X数据
    process_X_dense(X_batch, atlas, start_idx, end_idx, batch_idx)
    process_X_csr(X_batch, atlas, start_idx, end_idx, batch_idx)

def initialize_csr_tables(atlas: Atlas):
    """初始化CSR相关表"""
    atlas.connection.execute("""
        CREATE OR REPLACE TABLE X_CSR_indptr (
            id INTEGER PRIMARY KEY,
            cell_id VARCHAR,
            indptr BIGINT
        )
    """)

    atlas.connection.execute("""
        CREATE OR REPLACE TABLE X_CSR_data (
            id BIGINT PRIMARY KEY,
            indices USMALLINT,  -- 改为 INT16 无符号整数
            data REAL   -- 表达值 32 位的FLOAT
        )
    """)


def process_batch(h5ad_path: str, atlas: Atlas, start_idx: int, end_idx: int, batch_idx: int):
    """处理一个批次的数据"""
    # 读取obs数据（read_obs_batch已经包含cell_id）
    obs_df = read_obs_batch(h5ad_path, start_idx, end_idx)
    if obs_df is not None:
        # 添加 id 列
        obs_df['id'] = range(start_idx, end_idx)

        # 调整列顺序，确保 id 和 cell_id 在最前面
        obs_df = obs_df[['id', 'cell_id'] + [col for col in obs_df.columns if col not in ['id', 'cell_id']]]

        # 存储到数据库
        if batch_idx == 0:
            atlas.connection.register('obs_df', obs_df)
            atlas.connection.execute("CREATE OR REPLACE TABLE obs AS SELECT * FROM obs_df")
            atlas.connection.execute("ALTER TABLE obs ADD PRIMARY KEY (id)")
            atlas.connection.unregister('obs_df')
        else:
            atlas.connection.register('obs_df', obs_df)
            atlas.connection.execute("INSERT INTO obs SELECT * FROM obs_df")
            atlas.connection.unregister('obs_df')

    # 读取X数据
    X_data = read_X_batch(h5ad_path, start_idx, end_idx)
    if X_data is not None:
        process_X_dense(X_data, atlas, start_idx, end_idx, batch_idx)
        process_X_csr(X_data, atlas, start_idx, end_idx, batch_idx)


def read_obs_batch(h5ad_path: str, start_idx: int, end_idx: int):
    """读取指定范围的obs数据，包含原始细胞ID"""
    try:
        with h5py.File(h5ad_path, 'r') as f:
            if 'obs' not in f:
                return None

            obs_group = f['obs']
            obs_data = {}

            # 提取细胞索引（cell_id）
            cell_ids = []
            if 'index' in obs_group and isinstance(obs_group['index'], h5py.Dataset):
                cell_ids = obs_group['index'][start_idx:end_idx]
            elif '_index' in obs_group.attrs:
                index_attr = obs_group.attrs['_index']
                if isinstance(index_attr, np.ndarray):
                    cell_ids = index_attr[start_idx:end_idx]

            # 处理字节字符串
            if cell_ids and isinstance(cell_ids[0], bytes):
                cell_ids = [cid.decode('utf-8') for cid in cell_ids]

            # 如果没有找到索引，使用默认
            if not cell_ids:
                cell_ids = [f"cell_{i}" for i in range(start_idx, end_idx)]

            # 读取其他数据
            for key in obs_group.keys():
                if isinstance(obs_group[key], h5py.Dataset) and key != 'index':
                    data = obs_group[key][start_idx:end_idx]
                    if data.dtype.kind == 'S':
                        data = [d.decode('utf-8') for d in data]
                    obs_data[key] = data

            # 创建DataFrame，包含cell_id
            if obs_data or cell_ids:
                obs_df = pd.DataFrame(obs_data) if obs_data else pd.DataFrame()
                obs_df['cell_id'] = cell_ids
                return obs_df
            else:
                return None

    except Exception as e:
        logger.error(f"读取obs批次失败: {str(e)}")
        return None


def read_X_batch(h5ad_path: str, start_idx: int, end_idx: int):
    """读取指定范围的X数据"""
    try:
        with h5py.File(h5ad_path, 'r') as f:
            if 'X' not in f:
                return None

            x_item = f['X']

            # 处理稀疏矩阵
            if isinstance(x_item, h5py.Group) and 'data' in x_item:
                data = x_item['data'][:]
                indices = x_item['indices'][:]
                indptr = x_item['indptr'][:]

                batch_start_ptr = indptr[start_idx]
                batch_end_ptr = indptr[end_idx]

                batch_data = data[batch_start_ptr:batch_end_ptr]
                batch_indices = indices[batch_start_ptr:batch_end_ptr]
                batch_indptr = indptr[start_idx:end_idx + 1] - batch_start_ptr

                n_cells = end_idx - start_idx
                n_genes = x_item.attrs.get('shape', [0, 1])[1]

                return sparse.csr_matrix((batch_data, batch_indices, batch_indptr),
                                         shape=(n_cells, n_genes))
            elif isinstance(x_item, h5py.Dataset):
                # 密集矩阵
                return x_item[start_idx:end_idx, :]
            else:
                return None
    except Exception as e:
        logger.error(f"读取X批次失败: {str(e)}")
        return None


def process_X_dense(X_batch, atlas: Atlas, start_idx: int, end_idx: int, batch_idx: int):
    """处理稠密格式的X数据"""
    if X_batch is None:
        return

    # 获取基因名
    gene_query = atlas.connection.execute("SELECT gene_id FROM var ORDER BY id").fetchall()
    gene_names = [row[0] for row in gene_query]

    # 转换为密集矩阵
    if not isinstance(X_batch, np.ndarray):
        X_dense = X_batch.toarray()
    else:
        X_dense = X_batch

    # 创建DataFrame
    cell_ids = [f"cell_{i}" for i in range(start_idx, end_idx)]
    df_X = pd.DataFrame(X_dense, columns=gene_names)
    df_X['cell_id'] = cell_ids
    df_X['id'] = range(start_idx, end_idx)

    cols = ['id', 'cell_id'] + gene_names
    df_X = df_X[cols]

    # 插入数据库
    temp_table = f"temp_X_batch_{batch_idx}"
    atlas.connection.register(temp_table, df_X)

    if batch_idx == 0:
        create_table_sql = f"""
        CREATE OR REPLACE TABLE X (
            id BIGINT PRIMARY KEY,
            cell_id VARCHAR,
            {', '.join([f'"{gene}" FLOAT' for gene in gene_names])}
        )
        """
        atlas.connection.execute(create_table_sql)

    atlas.connection.execute(f"INSERT INTO X SELECT * FROM {temp_table}")
    atlas.connection.unregister(temp_table)


def process_X_csr(csr_matrix, atlas: Atlas, start_idx: int, end_idx: int, batch_idx: int):
    """处理CSR格式的X数据"""
    if csr_matrix is None:
        return

    if not isinstance(csr_matrix, sparse.csr_matrix):
        csr_matrix = sparse.csr_matrix(csr_matrix)

    data_array = csr_matrix.data
    indices_array = csr_matrix.indices.astype(np.int64)
    indptr_array = csr_matrix.indptr.astype(np.int64)

    # 处理indptr
    if len(indptr_array) > 0:
        offset_query = atlas.connection.execute(
            "SELECT COALESCE(MAX(indptr), 0) FROM X_CSR_indptr"
        ).fetchone()
        global_indptr_offset = offset_query[0] if offset_query else 0

        adjusted_indptr = indptr_array[1:] + global_indptr_offset

        cell_ids = [f"cell_{i}" for i in range(start_idx, end_idx)]
        index_df = pd.DataFrame({
            'id': range(start_idx, end_idx),
            'cell_id': cell_ids,
            'indptr': adjusted_indptr
        })

        temp_indptr = f"temp_indptr_{batch_idx}"
        atlas.connection.register(temp_indptr, index_df)
        atlas.connection.execute(f"INSERT INTO X_CSR_indptr SELECT * FROM {temp_indptr}")
        atlas.connection.unregister(temp_indptr)

    # 处理data
    if len(data_array) > 0:
        max_id_query = atlas.connection.execute(
            "SELECT COALESCE(MAX(id), -1) FROM X_CSR_data"
        ).fetchone()
        global_data_counter = max_id_query[0] + 1 if max_id_query else 0

        data_ids = np.arange(global_data_counter, global_data_counter + len(data_array), dtype=np.int64)

        data_df = pd.DataFrame({
            'id': data_ids,
            'indices': indices_array,
            'data': data_array
        })

        temp_data = f"temp_data_{batch_idx}"
        atlas.connection.register(temp_data, data_df)
        atlas.connection.execute(f"INSERT INTO X_CSR_data SELECT * FROM {temp_data}")
        atlas.connection.unregister(temp_data)

def add_var_from_h5ad(f, atlas: Atlas):
    """处理varm数据"""

    logger.info("读取var信息...")
    if 'var' in f:
        var_group = f['var']

        # 获取基因索引（gene_id）
        if '_index' in var_group:
            gene_ids = var_group['_index'][:]
        elif 'index' in var_group:
            gene_ids = var_group['index'][:]
        else:
            # 获取基因数量
            x_item = f['X']
            if isinstance(x_item, h5py.Dataset):
                n_genes = x_item.shape[1]
            else:
                n_genes = x_item.attrs['shape'][1]
            gene_ids = [f"gene_{i}" for i in range(n_genes)]

        print(" 已获取 gene_ids:", gene_ids)

        # # 处理字节字符串
        # if gene_ids and isinstance(gene_ids[0], bytes):
        #     gene_ids = [g.decode('utf-8') for g in gene_ids]

        # 创建var DataFrame
        var_data = {}
        for key in var_group.keys():
            if isinstance(var_group[key], h5py.Dataset) and key not in ['_index', 'index']:
                data = var_group[key][:]
                if data.dtype.kind == 'S':
                    data = [d.decode('utf-8') for d in data]
                var_data[key] = data

        var_df = pd.DataFrame(var_data, index=gene_ids)
        var_df = var_df.reset_index().rename(columns={'index': 'gene_id'})
        var_df['id'] = range(len(var_df))
        var_df = var_df[['id', 'gene_id'] + [col for col in var_df.columns if col not in ['id', 'gene_id']]]

        # 存储var表
        atlas.connection.register('var_df', var_df)
        atlas.connection.execute("CREATE OR REPLACE TABLE var AS SELECT * FROM var_df")
        atlas.connection.execute("ALTER TABLE var ADD PRIMARY KEY (id)") # 设置ID字段为主码，保证唯一性
        atlas.connection.unregister('var_df')
        logger.info(f"var表创建完成: {len(var_df)} 行")

def add_varm_from_h5ad(f, atlas: Atlas):
    """处理varm数据"""
    if 'varm' not in f:
        print(" varm 表不存在！  ")
        return

    varm_group = f['varm']
    for key in varm_group.keys():
        if isinstance(varm_group[key], h5py.Dataset):
            varm_data = varm_group[key][:]
            varm_df = pd.DataFrame(varm_data)

            temp_table = f"temp_varm_{key}"
            atlas.connection.register(temp_table, varm_df)
            atlas.connection.execute(f"CREATE OR REPLACE TABLE varm_{key} AS SELECT * FROM {temp_table}")
            atlas.connection.unregister(temp_table)
            logger.info(f" varm_{key} 创建完成: {len(temp_table)} 行")


def add_obsm_from_h5ad(f, atlas: Atlas):
    """处理obsm数据"""
    if 'obsm' not in f:
        return

    obsm_group = f['obsm']
    for key in obsm_group.keys():
        if isinstance(obsm_group[key], h5py.Dataset):
            obsm_data = obsm_group[key][:]
            obsm_df = pd.DataFrame(obsm_data)

            temp_table = f"temp_obsm_{key}"
            atlas.connection.register(temp_table, obsm_df)
            atlas.connection.execute(f"CREATE OR REPLACE TABLE obsm_{key} AS SELECT * FROM {temp_table}")
            atlas.connection.unregister(temp_table)


def add_obsp_from_h5ad(f, atlas: Atlas):
    """处理obsp数据，使用稀疏格式存储"""
    if 'obsp' not in f:
        return

    obsp_group = f['obsp']
    for key in obsp_group.keys():
        try:
            item = obsp_group[key]

            # 处理稀疏矩阵
            if isinstance(item, h5py.Group) and 'data' in item:
                # 读取稀疏矩阵组件
                data = item['data'][:]
                indices = item['indices'][:]
                indptr = item['indptr'][:]
                shape = item.attrs.get('shape', [0, 0])

                # 转换为稀疏矩阵
                sparse_matrix = sparse.csr_matrix((data, indices, indptr), shape=shape)
            elif isinstance(item, h5py.Dataset):
                # 密集矩阵，转换为稀疏矩阵
                dense_matrix = item[:]
                sparse_matrix = sparse.csr_matrix(dense_matrix)
            else:
                continue

            # 获取CSR组件
            n_cells = sparse_matrix.shape[0]
            n_nonzero = len(sparse_matrix.data)

            # 存储indptr表
            indptr_df = pd.DataFrame({
                'cell_id': [f"cell_{i}" for i in range(n_cells)],
                'indptr': sparse_matrix.indptr[1:]  # 跳过第一个0
            })

            temp_indptr = f"temp_obsp_{key}_indptr"
            atlas.connection.register(temp_indptr, indptr_df)
            atlas.connection.execute(f"CREATE OR REPLACE TABLE obsp_{key}_indptr AS SELECT * FROM {temp_indptr}")
            atlas.connection.unregister(temp_indptr)

            # 存储data表（如果有非零值）
            if n_nonzero > 0:
                data_df = pd.DataFrame({
                    'id': range(n_nonzero),
                    'indices': sparse_matrix.indices,
                    'data': sparse_matrix.data
                })

                temp_data = f"temp_obsp_{key}_data"
                atlas.connection.register(temp_data, data_df)
                atlas.connection.execute(f"CREATE OR REPLACE TABLE obsp_{key}_data AS SELECT * FROM {temp_data}")
                atlas.connection.unregister(temp_data)

        except Exception as e:
            logger.error(f"处理obsp/{key}失败: {str(e)}")
            continue


def add_uns_from_h5ad(f, atlas: Atlas):
    """处理uns数据"""
    if 'uns' not in f:
        return

    # 创建uns表
    atlas.connection.execute("CREATE OR REPLACE TABLE uns_raw (key TEXT, value TEXT)")

    def process_uns_item(item, key_path=""):
        if isinstance(item, h5py.Dataset):
            data = item[()]
            serialized_value = make_json_serializable(data)

            atlas.connection.execute(
                "INSERT INTO uns_raw VALUES (?, ?)",
                (key_path, serialized_value)
            )
        elif isinstance(item, h5py.Group):
            for key in item.keys():
                new_key_path = f"{key_path}.{key}" if key_path else key
                process_uns_item(item[key], new_key_path)

    process_uns_item(f['uns'])


def make_json_serializable(value):
    """转换为JSON可序列化的格式"""
    if isinstance(value, np.ndarray):
        return value.tolist()
    elif isinstance(value, (np.int64, np.int32)):
        return int(value)
    elif isinstance(value, (np.float64, np.float32)):
        return float(value)
    elif isinstance(value, bytes):
        return value.decode('utf-8')
    elif isinstance(value, dict):
        return {k: make_json_serializable(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [make_json_serializable(v) for v in value]
    else:
        return value

# todo 大文件读取 ######################################################

def h5ad_reader(file, batch_size=1024):
    """
    引擎级 h5ad 分批读取器
    """
    with h5py.File(file, "r") as f:
        X = f["X"]
        obs = f["obs"]
        var = f["var"]

        n_cells = obs["_index"].shape[0]
        n_genes = var["_index"].shape[0]

        print(f"dataset shape = ({n_cells}, {n_genes})")

        # 选择 X reader
        if isinstance(X, h5py.Dataset):
            X_reader = partial(read_dense, X=X)
        elif isinstance(X, h5py.Group):
            X_reader = partial(
                read_csr,
                n_cols=n_genes,
                data=X["data"],
                indices=X["indices"],
                indptr=X["indptr"]
            )
        else:
            raise RuntimeError(f"Unknown X type: {type(X)}")

        # ===== once-only parts =====
        var_df = read_meta(var)

        varm_dict = read_varm(f["varm"]) if "varm" in f else None
        uns_dict = read_uns_safe(f["uns"]) if "uns" in f else None

        # ===== batch loop =====
        for start in range(0, n_cells, batch_size):
            end = min(start + batch_size, n_cells)

            sub_X = X_reader(start, end)
            sub_obs = read_meta(obs, start, end)

            sub_obsm = (
                read_obsm(f["obsm"], start, end)
                if "obsm" in f else None
            )

            adata = sc.AnnData(
                X=sub_X,
                obs=sub_obs,
                var=var_df,
                obsm=sub_obsm,
                varm=varm_dict
            )

            # uns 只在第一个 batch 附加
            if start == 0 and uns_dict is not None:
                adata.uns = uns_dict

            yield adata


def read_uns_safe(uns_group, max_array_size=10_000):
    """
    安全读取 uns：
    - 只读取 Dataset
    - 跳过 Group
    - 跳过过大的数组
    """
    res = {}

    for k, v in uns_group.items():

        # 1️⃣ Group：直接跳过
        if isinstance(v, h5py.Group):
            logger.info(f"Skip uns[{k}] (Group)")
            continue

        # 2️⃣ Dataset：尝试读取
        if isinstance(v, h5py.Dataset):
            try:
                # 标量
                if v.shape == ():
                    res[k] = v[()]

                # 小数组
                elif v.size <= max_array_size:
                    res[k] = v[()].tolist()

                # 大数组
                else:
                    logger.info(f"Skip uns[{k}] (large array: {v.shape})")

            except Exception as e:
                logger.warning(f"Skip uns[{k}] (read failed): {e}")

    return res


def read_obsm(obsm_group, start, end):
    """
    obsm: cell-level embedding，按 batch 读取
    """
    res = {}
    for k, v in obsm_group.items():
        res[k] = v[start:end]
    return res


def read_varm(varm_group):
    """
    varm: gene-level matrix, 一次性读入
    """
    res = {}
    for k, v in varm_group.items():
        res[k] = v[()]
    return res


# def h5ad_reader(file, batch_size=1024):
#     # 使用 h5py 打开 h5ad 文件
#     # 注意：这里只是建立文件句柄，不会把任何数据读入内存（惰性加载）
#     with h5py.File(file, "r") as f:
#
#         # 获取 h5ad 中的三个核心对象
#         # X：表达矩阵（dense 或 sparse）
#         # obs：细胞元数据
#         # var：基因元数据
#         X = f["X"]
#         obs = f["obs"]
#         var = f["var"]
#
#         # 细胞数和基因数
#         n_cells = obs["_index"].shape[0]
#         n_genes = var["_index"].shape[0]
#
#         print(f"the shape of dataset is ({n_cells, n_genes})")
#
#         # X 可能是稠密矩阵，也可能是 CSR 稀疏矩阵
#         if isinstance(X, h5py.Dataset):  # dense matrix
#             X_reader = read_dense
#         elif isinstance(X, h5py.Group):  # csr sparse matrix
#             X_reader = partial(
#                 read_csr,
#                 n_cols=n_genes,
#                 data=X["data"],
#                 indices=X["indices"],
#                 indptr=X["indptr"]
#             )
#         else:
#             raise RuntimeError(f"unknown data type {type(X)}")
#
#         # 基因元数据通常较小，可以一次性全部读入内存
#         var_df = read_meta(var)
#
#         # 按 batch_size 逐批读取细胞数据
#         for start in range(0, n_cells, batch_size):
#             end = start + batch_size
#             sub_X = X_reader(start, end) # 读取表达矩阵的子块（只读当前 batch 的行）
#             sub_obs = read_meta(obs, start, end) # 读取 obs 的子块
#             yield sc.AnnData(X=sub_X, obs=sub_obs, var=var_df)
#             # 这一段的本质是一个 生成器（generator）。可以 for adata in h5ad_reader(...):


def read_meta(meta, start=None, end=None):
    """
    读取 obs 或 var 元数据

    如果 start / end 都为 None：
        - 读取整个表（适合 var）
    否则：
        - 只读取 slice(start, end) 对应的行（适合 obs）

    :param meta: h5py.Group，对应 obs 或 var
    :param start: 起始行
    :param end: 结束行
    :return: pandas.DataFrame，index 为 "_index"
    """
    res = dict()
    # 构造切片对象
    selected_cells = slice(start, end)

    for k, v in meta.items():
        # 普通数值 / 字符串列
        if isinstance(v, h5py.Dataset):
            # 这里是真正发生磁盘 → 内存读取的地方
            res[k] = v[selected_cells]
        # 分类变量（categorical），AnnData 的标准存储方式
        if isinstance(v, h5py.Group):
            values = v["codes"][selected_cells] # codes 是整数编码
            categories = {i: e for i, e in enumerate(v["categories"][()])} # categories 是编码到真实值的映射表
            res[k] = np.vectorize(categories.get)(values) # 将 codes 映射回真实类别
    res = pd.DataFrame(res)
    return res.set_index("_index")  # 使用 _index 作为行索引（细胞名 / 基因名）


def read_csr(start, end, indptr, indices, data, n_cols):
    """
    从 h5ad 中的 CSR 结构中，读取 [start:end) 行对应的子矩阵
    """

    # indptr 的切片，长度 = 行数 + 1
    sub_indptr = indptr[start:end + 1]

    # 找到该 batch 在 data / indices 中对应的范围
    sub_indices = indices[sub_indptr[0]:sub_indptr[-1]]
    sub_data = data[sub_indptr[0]:sub_indptr[-1]]

    # CSR 要求 indptr 从 0 开始，因此需要整体平移
    sub_indptr = sub_indptr - sub_indptr[0]

    # 构造一个新的 CSR 矩阵
    return csr_matrix(
        (sub_data, sub_indices, sub_indptr),
        shape=(sub_indptr.shape[0] - 1, n_cols)
    )

# 读取稠密矩阵的子块
def read_dense(start, end, X):
    return X[start:end,:] # 读取稠密矩阵的子块

# todo 大文件读取  ######################################################
def load_AnnData_chunk(
    h5ad_path: str,
    atlas: Atlas,
    batch_size: int = 4096
):
    """
    超大 h5ad → DuckDB
    - obs / X / obsm : batch
    - var / varm / uns : once
    - no obsp
    """

    logger.info("==== 开始加载超大 h5ad（chunk 模式） ====")

    # ================== 1. 打开 backed AnnData ==================
    adata_backed = sc.read_h5ad(h5ad_path, backed="r") # lazy 加载，，不会爆内存
    # 打开一个 h5ad 文件的“只读磁盘视图（backed view）”， 不把表达矩阵 X 读进内存
    atlas.connection = atlas.connect("r+")

    try:
        # ================== 2. 一次性结构数据 ==================
        logger.info("导入 var / varm / uns（一次性）")

        add_var(adata_backed, atlas)

        if hasattr(adata_backed, "varm"):
            add_varm(adata_backed, atlas)

        if hasattr(adata_backed, "uns"): # uns 只存 metadata（默认）
            add_uns_safe(adata_backed, atlas)

        # ================== 3. 初始化 CSR 表（只建一次） ==================
        logger.info("初始化 CSR 表结构")

        atlas.connection.execute("""
            CREATE OR REPLACE TABLE X_CSR_indptr (
                id BIGINT PRIMARY KEY,
                cell_id VARCHAR,
                indptr BIGINT
            )
        """)

        atlas.connection.execute("""
            CREATE OR REPLACE TABLE X_CSR_data (
                id BIGINT PRIMARY KEY,
                indices USMALLINT,
                data REAL,
                cell_index BIGINT
            )
        """)

        # 全局 offset（跨 batch）
        global_data_counter = np.int64(0)
        global_indptr_offset = np.int64(0)
        global_cell_offset = np.int64(0)

        # ================== 4. batch 读取 ==================
        for i, adata_chunk in enumerate(
            h5ad_reader(h5ad_path, batch_size)
        ):
            logger.info(f"---- batch {i} | cells={adata_chunk.n_obs}")

            # ---------- obs ----------
            add_obs_chunk(
                adata_chunk,
                atlas,
                cell_offset=global_cell_offset
            )

            # ---------- X ----------
            nnz, global_data_counter, global_indptr_offset = \
                add_X_CSR_chunk_append(
                    adata_chunk,
                    atlas,
                    cell_offset=global_cell_offset,
                    data_offset=global_data_counter,
                    indptr_offset=global_indptr_offset
                )

            # ---------- obsm ----------
            add_obsm_chunk(
                adata_chunk,
                atlas,
                cell_offset=global_cell_offset
            )

            global_cell_offset += adata_chunk.n_obs

            del adata_chunk
            gc.collect()

        logger.info("✅ 超大 h5ad 成功加载完成")

    except Exception as e:
        logger.error("加载失败", exc_info=True)
        raise


def add_obs_chunk(adata, atlas, cell_offset: int):
    """
    将 adata.obs 的一个 chunk 写入数据库
    - obs 所有列一律 VARCHAR
    - id BIGINT
    """

    # ============ 1. 构造 obs_df ============
    obs_df = adata.obs.copy()

    # cell_id
    obs_df["cell_id"] = adata.obs.index

    # 全局 cell id
    obs_df["id"] = np.arange(
        cell_offset,
        cell_offset + adata.n_obs,
        dtype=np.int64
    )

    # id 放第一列（和你原逻辑一致）
    cols = ["id"] + [c for c in obs_df.columns if c != "id"]
    obs_df = obs_df[cols]

    # ============ 2. 强制 obs 列类型规范化 ============
    for c in obs_df.columns:
        if c == "id":
            continue

        # bytes → str
        obs_df[c] = obs_df[c].map(
            lambda x: (
                x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else x
            )
        )

        # category / object / number → string
        obs_df[c] = obs_df[c].astype("string")

    # ============ 3. 注册临时表 ============
    atlas.connection.register("obs_df", obs_df)

    # ============ 4. 建表 or 插入 ============
    if cell_offset == 0:
        # ---- 首批：显式建 schema（防止 DuckDB 推断 ENUM） ----
        col_defs = []
        for c in obs_df.columns:
            if c == "id":
                col_defs.append("id BIGINT")
            else:
                col_defs.append(f"{c} VARCHAR")

        atlas.connection.execute(f"""
            CREATE TABLE obs (
                {', '.join(col_defs)}
            )
        """)

        atlas.connection.execute("""
            INSERT INTO obs
            SELECT * FROM obs_df
        """)

        atlas.connection.execute("""
            ALTER TABLE obs ADD PRIMARY KEY (id)
        """)
    else:
        atlas.connection.execute("""
            INSERT INTO obs
            SELECT * FROM obs_df
        """)

    # ============ 5. 清理 ============
    atlas.connection.unregister("obs_df")




def add_X_CSR_chunk_append(
    adata,
    atlas,
    cell_offset,
    data_offset,
    indptr_offset
):
    X = adata.X
    if not sparse.issparse(X):
        X = sparse.csr_matrix(X)

    csr = X.tocsr()

    data = csr.data
    indices = csr.indices
    indptr = csr.indptr

    # ===== indptr =====
    adj_indptr = indptr[1:] + indptr_offset

    indptr_df = pd.DataFrame({
        "id": np.arange(
            cell_offset,
            cell_offset + adata.n_obs,
            dtype=np.int64
        ),
        "cell_id": adata.obs.index,
        "indptr": adj_indptr
    })

    atlas.connection.register("indptr_df", indptr_df)
    atlas.connection.execute("""
        INSERT INTO X_CSR_indptr SELECT * FROM indptr_df
    """)
    atlas.connection.unregister("indptr_df")

    # ===== data =====
    nnz = len(data)
    if nnz > 0:
        cell_index = np.repeat(
            np.arange(
                cell_offset,
                cell_offset + adata.n_obs,
                dtype=np.int64
            ),
            np.diff(indptr)
        )

        data_df = pd.DataFrame({
            "id": np.arange(
                data_offset,
                data_offset + nnz,
                dtype=np.int64
            ),
            "indices": indices,
            "data": data,
            "cell_index": cell_index
        })

        atlas.connection.register("data_df", data_df)
        atlas.connection.execute("""
            INSERT INTO X_CSR_data SELECT * FROM data_df
        """)
        atlas.connection.unregister("data_df")

    return nnz, data_offset + nnz, adj_indptr[-1]

def add_obsm_chunk(adata, atlas, cell_offset):

    for key in adata.obsm.keys():
        df = pd.DataFrame(adata.obsm[key])
        df["cell_id"] = np.arange(
            cell_offset,
            cell_offset + adata.n_obs
        )

        atlas.connection.register("obsm_df", df)

        if cell_offset == 0:
            atlas.connection.execute(
                f"CREATE TABLE obsm_{key} AS SELECT * FROM obsm_df"
            )
        else:
            atlas.connection.execute(
                f"INSERT INTO obsm_{key} SELECT * FROM obsm_df"
            )

        atlas.connection.unregister("obsm_df")

def add_uns_safe(adata, atlas, allow_keys=None):
    allow_keys = allow_keys or [
        "params", "log1p", "rank_genes_groups"
    ]

    atlas.connection.execute("""
        CREATE TABLE IF NOT EXISTS uns_raw (
            key TEXT,
            value TEXT,
            data_type TEXT
        )
    """)

    for key in adata.uns.keys():
        if key not in allow_keys:
            logger.info(f"Skip uns[{key}]")
            continue

        value = adata.uns[key]
        serialized = json.dumps(make_json_serializable(value))

        atlas.connection.execute(
            "INSERT INTO uns_raw VALUES (?, ?, ?)",
            (key, serialized, type(value).__name__)
        )


# todo 大文件读取 ######################################################

"""" 先利用scanpy的接口，后续再直接利用源格式进行导入 """
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

#######################################################
# load data from object

def load_AnnData(adata:AnnData, atlas:Atlas, var_names_clean = True):
    """
    将anndata数据存入数据库中

    :param adata: AnnData对象，包含单细胞数据
    :param atlas: Atlas数据库实例
    :return: None
    """
    # 获取数据库连接
    atlas.connection = atlas.connect("r+")

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
            add_X_as_chunk_CSR_only(adata,atlas,chunk_size=4096) # 分块导入X表数据
            # add_X_as_chunk_CSR(adata,atlas,chunk_size=4096) # 分块导入X表数据 + CSR数据
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

        if hasattr(adata, 'obsp'):
            add_obsp(adata,atlas)
        else:
            print("Skipping obsp layer")

        if hasattr(adata, 'uns'):
            add_uns(adata,atlas)
        else:
            print("Skipping uns layer")




        # # 4. 验证数据
        # logger.info("验证数据是否正确插入...")
        #
        # cells_count = atlas.connection.execute("SELECT COUNT(*) FROM obs").fetchone()[0]
        # genes_count = atlas.connection.execute("SELECT COUNT(*) FROM var").fetchone()[0]
        # expression_count = atlas.connection.execute("SELECT COUNT(*) FROM expression").fetchone()[0]
        #
        # logger.info(f"cells表插入行数: {cells_count}")
        # logger.info(f"genes表插入行数: {genes_count}")
        # logger.info(f"expression表插入行数: {expression_count}")

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
    finally:
        # 确保连接关闭
        # atlas.close()
        logger.debug("数据库连接已关闭")

##########################################
# add meta data

def add_X(adata:AnnData,atlas:Atlas):
    # TODO 建表，插入数据
    # add obs data into duckdb
    # 表达矩阵数据（对应X） todo 是否需要添加 一列递增的ID作为主码，保证数据的唯一性。??? 基因名是否需要进行清洗操作？
    #  todo 采用长格式--只存储非零值； 宽格式：（id , cell_id , gene_1, gene_2 ,..., gene_n）
    # 生成宽格式数据
    logger.info("导入表达矩阵 X 数据")

    cell_ids = [f"cell_{i}" for i in range(adata.n_obs)]
    gene_names = [f"gene_{i}" for i in range(adata.n_vars)]

    X = adata.X
    if not isinstance(X, np.ndarray):  # 转换成密集矩阵
        X = X.toarray()

    df_X = pd.DataFrame(X, columns=gene_names)
    df_X['cell_id'] = cell_ids
    df_X['id'] = range(len(cell_ids))

    # 重新排列列顺序: id, cell_id, 然后是基因
    cols = ['id', 'cell_id'] + gene_names
    df_X = df_X[cols]

    # 动态创建表并插入数据
    atlas.connection.register('temp_df', df_X)
    atlas.connection.execute("CREATE OR REPLACE TABLE X AS SELECT * FROM df_X")
    atlas.connection.execute("ALTER TABLE X ADD PRIMARY KEY (id)")  # 设置ID字段为主码，保证唯一性
    logger.info(f"成功创建表 expression，包含 {len(df_X)} 行数据")

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

def add_obsm(adata:AnnData, atlas:Atlas):

    logger.info("导入obsm数据")

    for key in adata.obsm.keys():
        obsm_df = pd.DataFrame(adata.obsm[key])
        atlas.connection.register(f'obsm_{key}_df', obsm_df)
        atlas.connection.execute(f"CREATE OR REPLACE TABLE obsm_{key} AS SELECT * FROM obsm_{key}_df")
        atlas.connection.unregister(f'obsm_{key}_df')

    logger.info("导入obsm数据成功")

def add_varm(adata:AnnData, atlas:Atlas):

    logger.info("导入varm数据")

    for key in adata.varm.keys():
        varm_df = pd.DataFrame(adata.varm[key])
        atlas.connection.register(f'varm_{key}_df', varm_df)
        atlas.connection.execute(f"CREATE OR REPLACE TABLE varm_{key} AS SELECT * FROM varm_{key}_df")
        atlas.connection.unregister(f'varm_{key}_df')

    logger.info("导入varm数据成功")

def add_obsp(adata: AnnData, atlas: Atlas):
    """导入obsp数据，使用稀疏格式存储"""
    logger.info("导入obsp数据")

    for key in adata.obsp.keys():
        try:
            # 获取稀疏矩阵
            sparse_matrix = adata.obsp[key]

            # 确保是CSR格式
            if not isinstance(sparse_matrix, sparse.csr_matrix):
                sparse_matrix = sparse_matrix.tocsr()

            # 获取CSR组件
            n_cells = sparse_matrix.shape[0]
            n_nonzero = len(sparse_matrix.data)

            logger.info(f"处理obsp/{key}: {n_cells}x{n_cells}矩阵, {n_nonzero}个非零元素")

            # 存储indptr表
            indptr_df = pd.DataFrame({
                'cell_id': [f"cell_{i}" for i in range(n_cells)],
                'indptr': sparse_matrix.indptr[1:]  # 跳过第一个0
            })

            temp_indptr = f"temp_obsp_{key}_indptr"
            atlas.connection.register(temp_indptr, indptr_df)
            atlas.connection.execute(f"CREATE OR REPLACE TABLE obsp_{key}_indptr AS SELECT * FROM {temp_indptr}")
            atlas.connection.unregister(temp_indptr)

            # 存储data表（如果有非零值）
            if n_nonzero > 0:
                data_df = pd.DataFrame({
                    'id': range(n_nonzero),
                    'indices': sparse_matrix.indices,
                    'data': sparse_matrix.data
                })

                temp_data = f"temp_obsp_{key}_data"
                atlas.connection.register(temp_data, data_df)
                atlas.connection.execute(f"CREATE OR REPLACE TABLE obsp_{key}_data AS SELECT * FROM {temp_data}")
                atlas.connection.unregister(temp_data)

        except Exception as e:
            logger.error(f"处理obsp/{key}失败: {str(e)}")
            continue

    logger.info("导入obsp数据成功")

def add_X_as_chunk(adata:AnnData, atlas:Atlas, chunk_size=2048):
    """
    优化版本：使用 DuckDB 的批量插入功能
    """
    logger.info("开始导入表达矩阵数据")

    # 生成细胞ID和基因名
    cell_ids = adata.obs.index
    # cell_ids =  [f"cell_{i}" for i in range(adata.n_obs)]
    gene_names = [f"gene_{i}" for i in range(adata.n_vars)]

    # 创建表
    create_table_sql = f"""
    CREATE OR REPLACE TABLE X (
        id INTEGER PRIMARY KEY,
        cell_id VARCHAR,
        {', '.join([f'"{gene}" FLOAT' for gene in gene_names])}
    )
    """
    atlas.connection.execute(create_table_sql)

    # 开始事务
    atlas.connection.execute("BEGIN TRANSACTION")

    try:
        # 计算总块数
        total_cells = adata.n_obs
        total_chunks = (total_cells + chunk_size - 1) // chunk_size # 向上取整
        # adata.X=adata.X.astype(np.float32) # 将数据转换为float32以减少内存占用
        adata._X = adata.X.astype(np.float32)  # 绕过视图检查（谨慎使用）

        # 分块处理数据
        for chunk_idx in tqdm(range(total_chunks), desc="导入数据块"): #  Python 进度条库 加上 文字提示
            # 计算当前块的起始和结束位置
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, total_cells)

            # 切片 ：获取数据块
            chunk_X = adata.X[start_idx:end_idx]

            # 转换为密集矩阵（如果是稀疏矩阵）
            if not isinstance(chunk_X, np.ndarray):
                chunk_X = chunk_X.toarray()

            # 创建当前块的 DataFrame
            chunk_df = pd.DataFrame(chunk_X, columns=gene_names)

            # 添加元数据列
            chunk_df['cell_id'] = cell_ids[start_idx:end_idx]
            chunk_df['id'] = range(start_idx, end_idx)

            # 重新排列列顺序
            cols = ['id', 'cell_id'] + gene_names
            chunk_df = chunk_df[cols]

            # # 将数据块插入到数据库
            # con.register(f'temp_chunk_{chunk_idx}', chunk_df)
            # con.execute(f"INSERT INTO expression SELECT * FROM temp_chunk_{chunk_idx}")
            #
            # # 清理临时表
            # con.execute(f"DROP VIEW IF EXISTS temp_chunk_{chunk_idx}")

            # 使用 DuckDB 的 df() 方法直接插入
            atlas.connection.execute(f"INSERT INTO X SELECT * FROM chunk_df")

            # 强制垃圾回收
            del chunk_X, chunk_df
            gc.collect()


        # 提交事务
        atlas.connection.execute("COMMIT")
        logger.info(f"成功导入 {total_cells} 行数据")
        return True

    except Exception as e:
        # 回滚事务
        atlas.connection.execute("ROLLBACK")
        logger.error(f"导入失败: {str(e)}")
        return False

# todo 添加 CSR 格式的数据表
def add_X_as_chunk_CSR(adata:AnnData, atlas:Atlas, chunk_size=2048):
    """
    优化版本：使用 DuckDB 的批量插入功能
    """
    logger.info("开始导入表达矩阵数据")

    # 生成细胞ID和基因名
    cell_ids = adata.obs.index
    # cell_ids =  [f"cell_{i}" for i in range(adata.n_obs)]
    gene_names = [f"gene_{i}" for i in range(adata.n_vars)]

    # 创建三个表：稠密矩阵表、CSR索引表、CSR数据表
    # 1. X表 稠密矩阵表 - 存储完整的表达矩阵（便于某些查询）
    create_table_sql = f"""
    CREATE OR REPLACE TABLE X (
        id BIGINT PRIMARY KEY,  -- 改为BIGINT
        cell_id VARCHAR,
        {', '.join([f'"{gene}" FLOAT' for gene in gene_names])}
    )
    """
    atlas.connection.execute(create_table_sql)

    # 2.X_CSR_indptr 表 CSR索引表 - 存储每个细胞在CSR数据中的起始位置，没有存第一个非0值
    create_X_CSR_indptr = f"""
        CREATE OR REPLACE TABLE X_CSR_indptr (
            id BIGINT PRIMARY KEY,  -- 改为BIGINT
            cell_id VARCHAR,
            indptr BIGINT -- 该细胞在CSR数据表中的起始位置
        )
        """
    atlas.connection.execute(create_X_CSR_indptr)

    # 3. X_CSR_data表  CSR数据表 - 存储所有非零表达值及其对应的基因索引                 indices BIGINT, -- 基因的列索引
    create_X_CSR_data = f"""
            CREATE OR REPLACE TABLE X_CSR_data (
                id BIGINT PRIMARY KEY,
                indices USMALLINT,  -- 改为 INT16 无符号整数
                data REAL   -- 表达值 32 位的FLOAT
            )
            """
    atlas.connection.execute(create_X_CSR_data)

    # 开始事务
    atlas.connection.execute("BEGIN TRANSACTION")

    time_X = 0 # 导入X表用时
    time_X_CSR_indptr = 0  # 导入 CSR_indptr表 用时
    time_X_CSR_data = 0  # 导入 CSR_data表 用时

    try:
        # 计算总块数
        total_cells = adata.n_obs
        total_chunks = (total_cells + chunk_size - 1) // chunk_size # 向上取整
        # adata.X=adata.X.astype(np.float32) # 将数据转换为float32以减少内存占用
        adata._X = adata.X.astype(np.float32)  # 绕过视图检查（谨慎使用）

        isFirst = True  # 第一批标记，用于指导 csr 格式拼接
        global_data_counter = np.int64(0)  # 全局数据计数器，用于跟踪CSR数据表中的ID
        global_indptr_offset = np.int64(0)  # 全局indptr偏移量

        # 分块处理数据
        for chunk_idx in tqdm(range(total_chunks), desc="导入数据块"): #  Python 进度条库 加上 文字提示
            # 计算当前块的起始和结束位置
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, total_cells)
            cells_in_this_chunk  = end_idx - start_idx # 当前块的实际细胞数

            # 切片 ：获取数据块
            chunk_X = adata.X[start_idx:end_idx]

            # 转换为密集矩阵（如果是稀疏矩阵）
            if not isinstance(chunk_X, np.ndarray):
                csr_matrix = chunk_X.tocsr()  # todo 如果 X 已经是稀疏矩阵但需要 CSR 格式
                chunk_X = chunk_X.toarray()
            else:
                csr_matrix = sparse.csr_matrix(chunk_X) # todo 如果 X 是稠密矩阵



            # 创建当前块的 DataFrame
            chunk_df = pd.DataFrame(chunk_X, columns=gene_names)
            # 添加元数据列
            chunk_df['cell_id'] = cell_ids[start_idx:end_idx]
            chunk_df['id'] = range(start_idx, end_idx)
            # 重新排列列顺序
            cols = ['id', 'cell_id'] + gene_names
            chunk_df = chunk_df[cols]

            time_X_s = datetime.now()
            atlas.connection.execute(f"INSERT INTO X SELECT * FROM chunk_df") # 使用 DuckDB 的 df() 方法直接插入
            time_X_e = datetime.now()
            time_X_temp = (time_X_e - time_X_s).total_seconds()  # 导入X表用时
            print(f"批次：{chunk_idx} ，大小：{chunk_size }; 导入X表用时 为  {time_X_temp:2f} 秒")
            time_X = time_X + time_X_temp


            # ====== 提取当前块的 csr_matrix 信息，存入数据表中 X_CSR_indptr 和 X_CSR_data 中

            time_X_CSR_indptr_s = datetime.now()

            data_array = csr_matrix.data  # 非零表达值数组
            indices_array = csr_matrix.indices.astype(np.int64) # 非零值对应的基因索引数组
            indptr_array = csr_matrix.indptr.astype(np.int64)  # 每个细胞在data数组中的起始位置，即前i-1行的非零元素个数
            print(f"当前的indptr_array 的最后一个值为 {indptr_array[-1]} ")

            adjusted_indptr = indptr_array[1:] + global_indptr_offset # 去掉第一个元素（总是0），并加上全局偏移量

            index_df = pd.DataFrame({ # 创建X_CSR_indptr表的DataFrame
                'id': range(start_idx, end_idx),
                'cell_id': cell_ids[start_idx:end_idx],
                'indptr': adjusted_indptr
            })
            global_indptr_offset = adjusted_indptr[-1] # 取最后一个数字

            print(f"当前的global_indptr_offset 的值为 {global_indptr_offset} ")
            atlas.connection.execute("INSERT INTO X_CSR_indptr SELECT * FROM index_df")  # 插入X_CSR_indptr表
            print(f"最后一个 indptr 值为 {index_df['indptr'].iloc[-1] } ")

            time_X_CSR_indptr_e = datetime.now()
            time_X_CSR_indptr_temp = (time_X_CSR_indptr_e - time_X_CSR_indptr_s).total_seconds()  # 导入X表用时
            print(f"批次：{chunk_idx} ，大小：{chunk_size}; 导入 X_CSR_indptr 表用时 为  {time_X_CSR_indptr_temp:2f} 秒")
            time_X_CSR_indptr = time_X_CSR_indptr + time_X_CSR_indptr_temp

            # =================== 处理 X_CSR_data 表数据 ================================================
            time_X_CSR_data_s = datetime.now()

            if len(data_array) > 0:
                # global_data_counter = 0  # 全局数据计数器，为当前块的非零数据生成全局唯一的ID
                data_ids = np.arange(global_data_counter, global_data_counter + len(data_array), dtype=np.int64) # 生成int64类型的ID范围

                data_df = pd.DataFrame({
                    'id': data_ids,
                    'indices': indices_array,
                    'data': data_array
                })

                # 插入X_CSR_data表
                atlas.connection.execute("INSERT INTO X_CSR_data SELECT * FROM data_df")

                # 更新全局计数器
                global_data_counter += len(data_array)

                time_X_CSR_data_e = datetime.now()
                time_X_CSR_data_temp = (time_X_CSR_data_e - time_X_CSR_data_s).total_seconds()
                print(f"批次：{chunk_idx} ，大小：{chunk_size}; 导入 time_X_CSR_data 表用时 为  {time_X_CSR_data_temp:2f} 秒")
                time_X_CSR_data = time_X_CSR_data + time_X_CSR_data_temp

            # 强制垃圾回收
            del chunk_X, chunk_df, csr_matrix, index_df
            if len(data_array) > 0:
                del data_df
            gc.collect()

        # 提交事务
        atlas.connection.execute("COMMIT")
        logger.info(f"成功导入 {total_cells} 行数据")
        logger.info(f"X_CSR_indptr 共存储 {total_cells} 个值")
        logger.info(f"X_CSR_data 数据表共存储 {global_data_counter} 个非零值")

        print(f"成功导入 {total_cells} 行数据")
        print(f"批次大小：{chunk_size}; 导入 time_X  表 总用时 为  {time_X :2f} 秒")
        print(f"批次大小：{chunk_size}; 导入 time_X_CSR_indptr 表 总用时 为  {time_X_CSR_indptr:2f} 秒")
        print(f"批次大小：{chunk_size}; 导入 time_X_CSR_data 表 总用时 为  {time_X_CSR_data:2f} 秒")

        return True

    except Exception as e:
        # 回滚事务
        atlas.connection.execute("ROLLBACK")
        logger.error(f"导入失败: {str(e)}")
        return False

# todo 在 CSR_data 中添加 cell_index ， 把 CSR + COO 的优势结合起来
def build_CSR_cell_index_simple(atlas):
    """
    X_CSR_indptr 表结构：
        id, cell_id, indptr
    存储的是 CSR indptr[1:]，缺首个 0，
    且 indptr 最后一项已经是 nnz。

    X_CSR_data：
        id, indices, data
    """
    import numpy as np
    import pyarrow as pa

    conn = atlas.connect("r+")
    atlas.connection = conn

    # === 1. 读 indptr[1:]（按 id 排序）===
    rows = conn.execute("""
        SELECT indptr FROM X_CSR_indptr ORDER BY id
    """).fetchall()

    indptr_1 = np.array([r[0] for r in rows], dtype=np.int64)
    n_cells = len(indptr_1)

    # === 2. 构造完整 indptr ===
    # indptr_full = [0, indptr[1], indptr[2], ..., nnz]
    indptr_full = np.concatenate(([0], indptr_1))

    # === 3. 构造 cell_index ===
    diffs = np.diff(indptr_full)
    cell_index = np.repeat(np.arange(n_cells, dtype=np.int64), diffs)

    nnz = indptr_full[-1]
    assert len(cell_index) == nnz

    # === 4. 注册 Arrow 表（id, cell_index）===
    ids = np.arange(nnz, dtype=np.int64)
    arrow_tbl = pa.Table.from_arrays(
        [pa.array(ids), pa.array(cell_index)],
        names=["id", "cell_index"]
    )
    conn.register("tmp_cell_index", arrow_tbl)

    # === 5. 添加并更新 cell_index 列 ===
    conn.execute("""
        ALTER TABLE X_CSR_data
        ADD COLUMN IF NOT EXISTS cell_index BIGINT
    """)

    conn.execute("""
        UPDATE X_CSR_data AS x
        SET cell_index = t.cell_index
        FROM tmp_cell_index t
        WHERE x.id = t.id
    """)

    print(f"构建完成：cells={n_cells:,}, nnz={nnz:,}")

# todo 添加 CSRO 格式的数据表 , 不加载含0的大 X 宽表
def add_X_as_chunk_CSR_only(
        adata: AnnData,
        atlas: Atlas,
        chunk_size: int = 2048):
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


def add_uns(adata:AnnData, atlas:Atlas):
    """
    annsql 的处理方式
    :param adata:
    :param atlas:
    :return:
    """
    logger.info("开始导入 uns 表数据")
    try:
        atlas.connection.execute("CREATE TABLE uns_raw (key TEXT, value TEXT, data_type TEXT)")
    except Exception as e:
        print(f"Error creating uns_raw table: {e}")
    for key, value in adata.uns.items():
        try:
            serialized_value = make_json_serializable(value)
        except TypeError as e:
            print(f"Error serializing key {key}: {e}")
            continue
        if isinstance(value, dict):
            data_type = 'dict'
        elif isinstance(value, list):
            data_type = 'list'
        elif isinstance(value, (int, float, str)):
            data_type = 'scalar'
        elif isinstance(value, np.ndarray):
            data_type = 'array'
            value = value.tolist()
        else:
            data_type = 'unknown'
        try:
            atlas.connection.execute("INSERT INTO uns_raw VALUES (?, ?, ?)", (key, serialized_value, data_type))
            logger.info("导入 uns 表数据成功")
        except Exception as e:
            print(f"Error inserting key {key}: {e}")

def make_json_serializable(value):
    """
    Converts a given value into a JSON serializable format.

    Parameters:
        value (any): The value to be converted.
    Returns:
        JSON: The converted value in a JSON serializable format.
    """

    if isinstance(value, np.ndarray):
        return value.tolist()
    elif isinstance(value, (np.int64, np.int32)):
        return int(value)
    elif isinstance(value, (np.float64, np.float32)):
        return float(value)
    elif isinstance(value, dict):
        return {k: make_json_serializable(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [make_json_serializable(v) for v in value]
    else:
        return value

#### 基因名清洗（1）：在导入之前修改adata.var 和 adata.X ，再进行导入   ######################################
def clean_gene_names(adata, mode="add_suffix", gene_id_column=None, inplace=True):
    """
    基因名清洗函数

    参数:
    ----------
    adata : AnnData
        AnnData对象
    mode : str
        清洗模式，可选:
        - "add_suffix" / "加后缀": 对重复基因ID，第一个保留原名，其他添加后缀 _1, _2, _3
        - "keep_first" / "只保留一个": 对重复基因ID，只保留第一个，其余删除
        - "mean" / "取平均值": 对重复基因ID，取表达值的平均值合并
        - "median" / "取中位数": 对重复基因ID，取表达值的中位数合并
    gene_id_column : str, optional
        如果基因ID不在var_names中，指定包含基因ID的列名
    inplace : bool
        是否原地修改adata

    返回:
    -------
    AnnData
        处理后的AnnData对象
    """
    # 验证模式名称
    valid_modes = ["add_suffix", "keep_first", "mean", "median"]
    if mode not in valid_modes:
        raise ValueError(f"不支持的mode: {mode}，请选择: {valid_modes} 模式")

    if not inplace:
        adata = adata.copy()  # 是否原地修改adata

    # 获取基因名列表
    # 修改adata.var_names会同步更新adata.var.index，但反向修改adata.var.index不会自动更新adata.var_names；推荐通过var_names属性统一管理索引
    if gene_id_column and gene_id_column in adata.var.columns:
        gene_names = adata.var[gene_id_column].values
        use_column = True
    else:
        gene_names = adata.var_names.values
        use_column = False

    print(f"原始数据: {adata.n_vars} 个基因, {adata.n_obs} 个细胞")

    # 统计重复基因
    gene_counts = pd.Series(gene_names).value_counts() # 返回一个新的Series，索引是唯一的基因名，值是对应的出现次数,例子 TP53 2
    duplicate_genes = gene_counts[gene_counts > 1] # 创建一个布尔掩码，标识哪些基因出现次数大于1； TP53 True

    if len(duplicate_genes) == 0:
        print("没有发现重复基因名")
        return adata if inplace else adata

    print(f"发现 {len(duplicate_genes)} 个重复基因，总计 {duplicate_genes.sum()} 次出现")
    print(f"重复基因示例: {list(duplicate_genes.index[:5])}")

    # 根据模式调用相应的处理函数
    if mode == "add_suffix":
        adata = _add_suffix_mode(adata, gene_names, duplicate_genes, use_column, gene_id_column)
    elif mode == "keep_first":
        adata = _keep_first_mode(adata, gene_names, duplicate_genes, use_column, gene_id_column)
    elif mode == "mean":
        adata = _mean_mode(adata, gene_names, duplicate_genes, use_column, gene_id_column)
    elif mode == "median":
        adata = _median_mode(adata, gene_names, duplicate_genes, use_column, gene_id_column)

    return adata if inplace else adata


def _add_suffix_mode(adata, gene_names, duplicate_genes, use_column, gene_id_column):
    """模式1: 为重复基因添加后缀 - add_suffix"""

    # 创建新的基因名
    new_gene_names = []
    gene_counter = defaultdict(int)

    for gene in gene_names:
        if gene in duplicate_genes:
            gene_counter[gene] += 1
            if gene_counter[gene] == 1:
                new_gene_names.append(gene)  # 第一个保留原名
            else:
                new_gene_names.append(f"{gene}_{gene_counter[gene] - 1}")  # 后续添加后缀
        else:
            new_gene_names.append(gene)

    # 更新基因名
    if use_column:
        adata.var[gene_id_column] = new_gene_names
        # 同时更新var_names为清洗后的基因名
        adata.var_names = new_gene_names
    else:
        adata.var_names = new_gene_names

    print(f"add_suffix模式完成: 保留了所有 {adata.n_vars} 个基因")
    print(f"重复基因已添加后缀，示例: {[name for name in new_gene_names if '_' in name][:5]}")

    return adata


def _keep_first_mode(adata, gene_names, duplicate_genes, use_column, gene_id_column):
    """模式2: 只保留每个重复基因的第一个出现 - keep_first"""

    # 记录要保留的基因索引
    keep_indices = []
    seen_genes = set()

    for i, gene in enumerate(gene_names):
        if gene not in seen_genes:
            keep_indices.append(i)
            seen_genes.add(gene)  # 通过set来实现唯一性，只保留每个重复基因的第一个出现

    # 筛选数据
    adata = adata[:, keep_indices]

    print(f"keep_first模式完成: 从 {len(gene_names)} 个基因减少到 {adata.n_vars} 个基因")
    print(f"删除了 {len(gene_names) - len(keep_indices)} 个重复基因")

    return adata


def _mean_mode(adata, gene_names, duplicate_genes, use_column, gene_id_column):
    """模式3: 对重复基因取平均值合并 - mean"""
    return _aggregate_mode(adata, gene_names, duplicate_genes, np.mean, "mean")


def _median_mode(adata, gene_names, duplicate_genes, use_column, gene_id_column):
    """模式4: 对重复基因取中位数合并 - median"""
    return _aggregate_mode(adata, gene_names, duplicate_genes, np.median, "median")


def _aggregate_mode(adata, gene_names, duplicate_genes, agg_func, mode_name):
    """通用的聚合函数，用于mean和median模式"""

    # 按基因名分组
    gene_groups = defaultdict(list)
    for i, gene in enumerate(gene_names): # enumerate(gene_names)：将基因名列表转化为(索引, 基因名)元组的迭代器。如[("TP53",0), ("BRCA1",1)]。
        gene_groups[gene].append(i)
    # 生成的gene_groups字典结构,例如
    # {
    #     "TP53": [0, 2],  # 基因TP53出现在索引0和2的位置
    #     "BRCA1": [1],  # 基因BRCA1出现在索引1的位置
    # }

    # 准备新的表达矩阵
    new_X = []
    new_gene_names = []
    new_var_data = []

    single_genes_processed = 0
    duplicate_genes_processed = 0

    # 遍历基因分组字典，处理每个基因及其对应的多个位置索引
    for gene, indices in gene_groups.items():  # 例如: gene="TP53", indices=[0, 2, 5]

        if len(indices) == 1:  # 检查是否为非重复基因（只有一个位置索引）=== 非重复基因处理：直接保留原始数据 ===
            gene_expression = adata.X[:, indices[0]]  # 获取该基因列的所有细胞表达值
            if hasattr(gene_expression, 'toarray'):# 如果是稀疏矩阵，转换为稠密矩阵并展平为一维数组
                gene_expression = gene_expression.toarray().flatten()

            new_X.append(gene_expression) # 将处理后的表达数据添加到新表达矩阵列表中
            new_gene_names.append(gene) # 将该基因名添加到新基因名列表中
            new_var_data.append(adata.var.iloc[indices[0]]) # 获取该基因在原始数据中的元数据（使用第一个出现位置的元数据）
            single_genes_processed += 1 # 更新非重复基因计数器

        else: # === 重复基因处理：进行表达值聚合 ===
            gene_data = adata.X[:, indices]    # 提取该重复基因在所有出现位置上的表达数据 # 形状: (细胞数, 重复次数)
            if hasattr(gene_data, 'toarray'): # 如果是稀疏矩阵，转换为稠密矩阵以便计算
                gene_data = gene_data.toarray()
            aggregated_values = agg_func(gene_data, axis=1)  # 使用指定的聚合函数（mean或median）计算聚合值
            # axis=1 表示沿着细胞方向聚合（对每个细胞聚合多个重复基因的表达值）

            new_X.append(aggregated_values) # 将聚合后的表达值添加到新表达矩阵
            new_gene_names.append(gene) # 将该基因名添加到新基因名列表中（不添加后缀）
            new_var_data.append(adata.var.iloc[indices[0]])  # 保留第一个出现位置的基因元数据
            duplicate_genes_processed += 1  # 更新重复基因计数器

    # 修改anndata对象
    adata.X = np.array(new_X).T  # 转置为 (细胞数, 基因数)
    adata.var_names = new_gene_names
    adata.var = pd.DataFrame(new_var_data, index=new_gene_names)

    print(f"{mode_name}模式完成: 从 {len(gene_names)} 个基因减少到 {adata.n_vars} 个基因")
    print(f"处理了 {duplicate_genes_processed} 组重复基因，保留了 {single_genes_processed} 个唯一基因")

    return adata


#### 基因名清洗（2）：先按照不清洗基因名的方式导入到数据库中，再进行清洗，修改X表，var表   ######################################
# 示例
# var 表
# id | gene_id
# ---|--------
# 1  | TP53
# 2  | EGFR
# 3  | TP53    -- 重复基因
# 4  | BRAF
# 5  | TP53    -- 重复基因

# X 表
# id | cell_id | gene_1 | gene_2 | gene_3 | gene_4 | gene_5
# ---|---------|--------|--------|--------|--------|--------
# 1  | cell_1  | 10     | 0      | 5      | 2      | 8
# 2  | cell_2  | 0      | 15     | 3      | 7      | 1
# 3  | cell_3  | 12     | 2      | 0      | 4      | 6

def clean_genes_in_database(atlas: Atlas, mode="add_suffix", gene_id_column="gene_id"):
    """
    在数据库层面清洗基因名，直接操作数据库表
    参数:
    ----------
    atlas : Atlas
        Atlas数据库对象
    mode : str
        清洗模式: "add_suffix", "keep_first", "mean", "median"
    gene_id_column : str
        基因ID列名，默认为'gene_id'
    """
    logger.info(f"开始在数据库层面清洗基因名，模式: {mode}")

    # 检查表是否存在
    tables = atlas.connection.execute("SHOW TABLES").df()
    if 'var' not in tables['name'].values or 'X' not in tables['name'].values:
        logger.error("var表或X表不存在，请先导入数据")
        return False

    try:
        if mode == "add_suffix":
            return _clean_genes_add_suffix(atlas, gene_id_column)
        elif mode == "keep_first":
            return _clean_genes_keep_first(atlas, gene_id_column)
        elif mode == "mean":
            return _clean_genes_aggregate(atlas, gene_id_column, "AVG")
        elif mode == "median":
            return _clean_genes_aggregate(atlas, gene_id_column, "MEDIAN")
        else:
            logger.error(f"不支持的清洗模式: {mode}")
            return False
    except Exception as e:
        logger.error(f"基因名清洗失败: {str(e)}")
        return False


def _clean_genes_add_suffix(atlas: Atlas, gene_id_column: str):
    """
    在数据库中添加后缀模式：为重复基因添加 _1, _2, _3 等后缀
    处理var表和X表列名

    参数:
        atlas: 数据库连接对象
        gene_id_column: 基因ID列名
    """
    logger.info("开始添加后缀模式...")

    # 1. 创建带后缀的基因名临时表
    atlas.connection.execute("""
        CREATE OR REPLACE TEMPORARY TABLE var_with_suffix AS
        WITH ranked_genes AS (
            SELECT *, 
                   ROW_NUMBER() OVER (PARTITION BY gene_id ORDER BY id) as rn
            FROM var
        )
        SELECT 
            id,
            CASE 
                WHEN rn = 1 THEN gene_id  -- 第一个重复基因保持原名
                ELSE gene_id || '_' || (rn - 1)::VARCHAR  -- 其他添加 _1, _2 等后缀
            END as gene_id
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

    # 2. 获取基因名映射关系用于日志记录
    gene_mapping = atlas.connection.execute("""
        SELECT v.gene_id as original_gene_id, vs.gene_id as new_gene_id 
        FROM var v
        JOIN var_with_suffix vs ON v.id = vs.id
        WHERE v.gene_id != vs.gene_id
    """).df()
    # 获取基因名映射关系。gene_mapping数据框
    #    original_gene_id  new_gene_id
    # 0              TP53       TP53_1
    # 1              TP53       TP53_2


    logger.info(f"发现 {len(gene_mapping)} 个需要添加后缀的重复基因")

    # 3. 更新X表的列名（将"gene_{var_id}"映射为真实的gene_id）
    logger.info("更新X表列名...")

    # 获取var表的id到gene_id映射
    var_mapping = atlas.connection.execute("""
        SELECT id, gene_id FROM var_with_suffix ORDER BY id
    """).df()

    # 创建id到gene_name的映射字典
    id_to_gene = dict(zip(var_mapping['id'], var_mapping['gene_id']))
    # id_to_gene = {
    #     1: "TP53",
    #     2: "EGFR",
    #     3: "TP53_1",
    #     4: "BRAF",
    #     5: "TP53_2"
    # }

    # 获取X表的所有列名
    x_columns = atlas.connection.execute("PRAGMA table_info(X)").df()
    all_columns = x_columns['name'].tolist() # all_columns = ['id', 'cell_id', 'gene_1', 'gene_2', 'gene_3', 'gene_4', 'gene_5']

    # 构建新的列定义
    new_columns = []
    for col in all_columns:
        if col in ['id', 'cell_id']:
            # 保持id和cell_id列不变
            new_columns.append(f'"{col}"')
        elif col.startswith('gene_'):
            try:
                # 提取gene_后面的数字部分
                var_id = int(col[5:])  # 去掉"gene_"前缀
                if var_id in id_to_gene:
                    new_gene_name = id_to_gene[var_id]
                    # 转义列名中的特殊字符
                    new_columns.append(f'"{col}" as "{new_gene_name}"')
                else:
                    # 如果找不到映射，保持原列名
                    new_columns.append(f'"{col}"')
                    logger.warning(f"未找到var_id {var_id} 对应的基因名")
            except ValueError:
                # 如果不是有效的数字，保持原列名
                new_columns.append(f'"{col}"')
        else:
            # 其他列保持原样
            new_columns.append(f'"{col}"')

    # 创建新的X表
    new_columns_sql = ", ".join(new_columns)
    # [
    #     '"id"',
    #     '"cell_id"',
    #     '"gene_1" as "TP53"',
    #     '"gene_2" as "EGFR"',
    #     '"gene_3" as "TP53_1"',
    #     '"gene_4" as "BRAF"',
    #     '"gene_5" as "TP53_2"'
    # ]

    atlas.connection.execute(f"""
        CREATE TABLE X_new AS
        SELECT {new_columns_sql}
        FROM X
    """)

    # 原子操作：替换原表
    atlas.connection.execute("DROP TABLE X")
    atlas.connection.execute("ALTER TABLE X_new RENAME TO X")
    logger.info("X表已更新")

    # 更新var表
    if(len(gene_mapping)>0):
        atlas.connection.execute("DROP TABLE var")
        atlas.connection.execute("ALTER TABLE var_with_suffix RENAME TO var")
        atlas.connection.execute("ALTER TABLE var ADD PRIMARY KEY (id)")
        logger.info("var表已更新")

    logger.info("添加后缀模式完成!")
    return True
# 更新后的var表：
# id | gene_id
# ---|---------
# 1  | TP53
# 2  | EGFR
# 3  | TP53_1
# 4  | BRAF
# 5  | TP53_2
# 更新后的X表：
# id | cell_id | TP53 | EGFR | TP53_1 | BRAF | TP53_2
# ---|---------|------|------|--------|------|-------
# 1  | cell_1  | 10   | 0    | 5      | 2    | 8
# 2  | cell_2  | 0    | 15   | 3      | 7    | 1
# 3  | cell_3  | 12   | 2    | 0      | 4    | 6


def _clean_genes_keep_first(atlas: Atlas, gene_id_column: str):
    """对重复的基因ID，只保留第一个，其余删除"""
    logger.info("开始保留第一个模式...")

    # 1. 创建只保留第一个重复基因的临时表
    atlas.connection.execute("""
        CREATE OR REPLACE TEMPORARY TABLE var_keep_first AS
        WITH ranked_genes AS (
            SELECT *, 
                   ROW_NUMBER() OVER (PARTITION BY gene_id ORDER BY id) as rn
            FROM var
        )
        SELECT 
            id,
            gene_id
        FROM ranked_genes
        WHERE rn = 1  -- 只保留每个基因的第一个出现
        ORDER BY id
    """)
    # var_keep_first表的内容
    # id | gene_id
    # ---|--------
    # 1  | TP53    -- 第一个TP53保留
    # 2  | EGFR    -- EGFR只有一个，保留
    # 4  | BRAF    -- BRAF只有一个，保留

    # 2. 获取被删除的基因信息用于日志记录
    deleted_genes = atlas.connection.execute("""
        SELECT v.id, v.gene_id
        FROM var v
        LEFT JOIN var_keep_first vf ON v.id = vf.id
        WHERE vf.id IS NULL
    """).df()
    # deleted_genes数据框
    #    id gene_id
    # 0   3    TP53
    # 1   5    TP53

    logger.info(f"将删除 {len(deleted_genes)} 个重复基因")

    # 3. 更新X表的列名（只保留第一个出现的基因）
    logger.info("更新X表列名...")

    var_mapping = atlas.connection.execute("""
        SELECT id, gene_id FROM var_keep_first ORDER BY id
    """).df()
    # 从var_keep_first临时表中获取所有保留的基因记录，按id排序
    #    id gene_id
    # 0   1    TP53
    # 1   2    EGFR
    # 2   4    BRAF


    # 创建id到gene_name的映射字典
    id_to_gene = dict(zip(var_mapping['id'], var_mapping['gene_id']))
    # zip(var_mapping['id'], var_mapping['gene_id']) 将两个列表配对  id列 和 gene_id列
    # dict(...) → 将配对转换为字典
    # id_to_gene = {
    #     1: "TP53",
    #     2: "EGFR",
    #     4: "BRAF"
    # }

    # 获取X表的所有列名
    x_columns = atlas.connection.execute("PRAGMA table_info(X)").df()
    all_columns = x_columns['name'].tolist() # all_columns = ['id', 'cell_id', 'gene_1', 'gene_2', 'gene_3', 'gene_4', 'gene_5']

    # 构建新的列定义
    new_columns = []
    retained_columns = []  # 记录保留的列

    for col in all_columns:
        if col in ['id', 'cell_id']:
            # 保持id和cell_id列不变
            new_columns.append(f'"{col}"')
            retained_columns.append(col)
        elif col.startswith('gene_'):
            try:
                # 提取gene_后面的数字部分
                var_id = int(col[5:])  # 去掉"gene_"前缀
                if var_id in id_to_gene:
                    new_gene_name = id_to_gene[var_id]
                    # 转义列名中的特殊字符
                    new_columns.append(f'"{col}" as "{new_gene_name}"')
                    retained_columns.append(new_gene_name)
                else:
                    # 如果不在保留列表中，跳过此列（即删除）
                    logger.debug(f"删除列 {col}，对应基因ID {var_id}")
            except ValueError:
                # 如果不是有效的数字，保持原列名
                new_columns.append(f'"{col}"')
                retained_columns.append(col)
        else:
            # 其他列保持原样
            new_columns.append(f'"{col}"')
            retained_columns.append(col)

    # 最终的new_columns列表
    # [
    #     '"id"',
    #     '"cell_id"',
    #     '"gene_1" as "TP53"',
    #     '"gene_2" as "EGFR"',
    #     '"gene_4" as "BRAF"'
    # ]
    # retained_columns列表
    # ['id', 'cell_id', 'TP53', 'EGFR', 'BRAF']

    # 创建新的X表
    new_columns_sql = ", ".join(new_columns)
    # "id", "cell_id", "gene_1" as "TP53", "gene_2" as "EGFR", "gene_4" as "BRAF"

    atlas.connection.execute(f"""
        CREATE TABLE X_new AS
        SELECT {new_columns_sql}
        FROM X
    """)

    # 原子操作：替换原表
    atlas.connection.execute("DROP TABLE X")
    atlas.connection.execute("ALTER TABLE X_new RENAME TO X")

    # 更新var表
    atlas.connection.execute("DROP TABLE var")
    atlas.connection.execute("ALTER TABLE var_keep_first RENAME TO var")
    atlas.connection.execute("ALTER TABLE var ADD PRIMARY KEY (id)")

    logger.info(f"保留第一个模式完成：删除了 {len(deleted_genes)} 个重复基因，保留了 {len(id_to_gene)} 个基因")
    return True
# 更新后的var表
# id | gene_id
# ---|--------
# 1  | TP53
# 2  | EGFR
# 4  | BRAF

# 更新后的X表
# id | cell_id | TP53 | EGFR | BRAF
# ---|---------|------|------|------
# 1  | cell_1  | 10   | 0    | 2
# 2  | cell_2  | 0    | 15   | 7
# 3  | cell_3  | 12   | 2    | 4


def _clean_genes_aggregate(atlas: Atlas, gene_id_column: str, agg_func: str):
    """在数据库中，对重复的基因ID进行聚合操作（平均值或中位数），作为表达值保留。其余删除"""
    logger.info(f"开始聚合模式，使用聚合函数: {agg_func}...")

    # 1. 确定聚合函数
    if agg_func.lower() == 'median':
        agg_sql = "MEDIAN"
    elif agg_func.lower() in ['mean', 'avg']:
        agg_sql = "AVG"
    else:
        logger.warning(f"不支持的聚合函数: {agg_func}，默认使用AVG")
        agg_sql = "AVG"

    # 2. 创建聚合后的var表（每个基因只保留一行）
    atlas.connection.execute(f"""
        CREATE OR REPLACE TEMPORARY TABLE var_aggregated AS
        SELECT 
            MIN(id) as id,  -- 使用最小的id作为新id
            gene_id
        FROM var
        GROUP BY gene_id
        ORDER BY id
    """)

    # 3. 获取被聚合的基因信息用于日志记录
    aggregated_genes = atlas.connection.execute("""
        SELECT gene_id, COUNT(*) as count
        FROM var
        GROUP BY gene_id
        HAVING COUNT(*) > 1
    """).df()

    logger.info(f"将聚合 {len(aggregated_genes)} 组重复基因")

    # 4. 创建聚合后的X表
    logger.info("创建聚合后的X表...")

    # 获取var表的基因分组信息
    gene_groups = atlas.connection.execute("""
        SELECT v1.gene_id, GROUP_CONCAT(v1.id) as id_list
        FROM var v1   -- v1是表的别名，方便后续引用
        GROUP BY v1.gene_id
    """).df()
    # 得到如下结果
    # gene_id | id_list
    # -------- | ---------
    # TP53 | 1, 3, 5
    # EGFR | 2, 6
    # BRAF | 4

    # 创建基因到id列表的映射字典
    gene_to_ids = {}
    for _, row in gene_groups.iterrows(): # iterrows(): 遍历数据框的每一行
        gene_id = row['gene_id']
        id_list = [int(x) for x in row['id_list'].split(',')]  # [int(x) for x in ...]: 将每个字符串转换为整数，得到 [1, 3, 5]
        gene_to_ids[gene_id] = id_list
    # 最终生成的 gene_to_ids 字典
    # {
    #     "TP53": [1, 3, 5],
    #     "EGFR": [2],
    #     "BRAF": [4]
    # }

    # 获取X表的所有列名
    x_columns = atlas.connection.execute("PRAGMA table_info(X)").df()
    all_columns = x_columns['name'].tolist()

    # 构建新的列定义
    new_columns = ['"id"', '"cell_id"']  # 保留id和cell_id列

    # 为每个唯一的基因创建聚合表达式
    for gene_id, id_list in gene_to_ids.items():
        if len(id_list) == 1:
            # 单个基因，直接使用原列 # gene_id = 'EGFR', id_list = [2]
            col_name = f"gene_{id_list[0]}"  # col_name = "gene_2"
            new_columns.append(f'"{col_name}" as "{gene_id}"')  # new_columns.append('"gene_2" as "EGFR"')
        else:
            # 多个重复基因，需要聚合
            agg_expressions = []
            for var_id in id_list:
                col_name = f"gene_{var_id}"
                agg_expressions.append(f'"{col_name}"') # agg_expressions = ['"gene_1"', '"gene_3"', '"gene_5"']

            # 构建聚合表达式
            if agg_sql == "AVG":
                # 平均值：(gene_1 + gene_2 + ...) / 计数
                expr = f"({' + '.join(agg_expressions)}) / {len(agg_expressions)}" # expr = '("gene_1" + "gene_3" + "gene_5") / 3'
            elif agg_sql == "MEDIAN":
                # 中位数：使用SQLite的百分位函数近似计算
                # 注意：SQLite没有内置MEDIAN函数，这里使用近似方法
                values_expr = ", ".join(agg_expressions)
                expr = f"""
                    (SELECT CASE 
                     WHEN COUNT(*) % 2 = 1 THEN 
                         (SELECT val FROM (
                             SELECT val FROM (VALUES {values_expr}) AS values(val) 
                             ORDER BY val LIMIT 1 OFFSET COUNT(*)/2
                         ))
                     ELSE 
                         (SELECT AVG(val) FROM (
                             SELECT val FROM (VALUES {values_expr}) AS values(val) 
                             ORDER BY val LIMIT 2 OFFSET (COUNT(*)-1)/2
                         ))
                     END)
                """
            else:
                # 默认使用平均值
                expr = f"({' + '.join(agg_expressions)}) / {len(agg_expressions)}"

            new_columns.append(f"{expr} as \"{gene_id}\"")

    # 创建新的X表
    new_columns_sql = ", ".join(new_columns)
    #  '"id"',
    #     '"cell_id"',
    #     '("gene_1" + "gene_3" + "gene_5") / 3 as "TP53"',
    #     '"gene_2" as "EGFR"',
    #     '"gene_4" as "BRAF"'

    atlas.connection.execute(f"""
        CREATE TABLE X_new AS
        SELECT {new_columns_sql}
        FROM X
    """)

    # 原子操作：替换原表
    atlas.connection.execute("DROP TABLE X")
    atlas.connection.execute("ALTER TABLE X_new RENAME TO X")

    # 更新var表
    atlas.connection.execute("DROP TABLE var")
    atlas.connection.execute("ALTER TABLE var_aggregated RENAME TO var")
    atlas.connection.execute("ALTER TABLE var ADD PRIMARY KEY (id)")

    logger.info(f"聚合模式完成：聚合了 {len(aggregated_genes)} 组重复基因，使用函数 {agg_func}")
    return True

# 处理后
# 聚合后的X表：
# id | cell_id | TP53  | EGFR | BRAF
# ---|---------|-------|------|-----
# 1  | cell_1  | 7.67  | 0    | 2    -- TP53 = (10 + 5 + 8) / 3 = 7.67
# 2  | cell_2  | 1.33  | 15   | 7    -- TP53 = (0 + 3 + 1) / 3 = 1.33
# 3  | cell_3  | 6.00  | 2    | 4    -- TP53 = (12 + 0 + 6) / 3 = 6.00
#
#聚合后的var表：
# id | gene_id
# ---|--------
# 1  | TP53    -- 使用最小id
# 2  | EGFR
# 4  | BRAF