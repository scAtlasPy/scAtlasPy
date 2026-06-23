from __future__ import annotations
import h5py
from . import progress
import numpy as np
import pandas as pd
from scipy import sparse
from anndata import AnnData
from datetime import datetime
from os import PathLike, fspath
from typing import TYPE_CHECKING
if TYPE_CHECKING:  # TYPE_CHECKING = 给 IDE / 类型检查器看的导入;  正常运行时 = 不执行这个导入，避免循环导入
    from ..data import Atlas
import logging
logger = logging.getLogger("Atlas")
logger.addHandler(logging.NullHandler())

''' 数据导出: 把数据直接变成文件'''
def write_h5ad(
    atlas: Atlas,
    out_h5ad_path: PathLike[str] | str,
    *,
    batch_cells: int = 1_000_000,
    use_data: str = "data_count",
):
    """将 Atlas 数据库导出为 h5ad 文件。

    该函数从 Atlas 数据库读取 ``obs``、``var``、表达矩阵和可用的 embedding 结果，并写出为标准 AnnData ``.h5ad`` 文件，方便继续在 Scanpy 或其他工具中分析。

    Parameters
    ----------
    atlas
        Atlas 对象。通常需要已经连接到 DuckDB 数据库，并包含该函数读取或写入所需的 ``obs``、``var``、表达矩阵或结果表。
    out_h5ad_path
        输出 ``.h5ad`` 文件路径。
    batch_cells
        导出表达矩阵时每批处理的细胞数。
    use_data
        从 ``X_HyS_data`` 表中导出的表达值字段名，默认使用 ``"data_count"``。

    Returns
    -------
    None
        结果直接写入 Atlas 数据库或当前图形窗口。

    Examples
    --------
    导出当前数据库::

        atlas.write_h5ad(r"F:\\data\\pbmc_export.h5ad")

    使用对象式 API 并降低单批内存占用::

        atlas.write_h5ad(r"F:\\data\\pbmc_export.h5ad", batch_cells=200000)

    导出 log1p 表达矩阵::

        atlas.write_h5ad(r"F:\\data\\pbmc_log1p.h5ad", use_data="data_log1p")"""

    start_time = datetime.now()

    out_h5ad_path = fspath(out_h5ad_path)

    conn = atlas.connection

    if conn is None:
        raise ValueError("atlas.connection 为空，请先连接数据库")

    if not isinstance(use_data, str):
        raise TypeError("use_data 必须是 str")

    if use_data == "":
        raise ValueError("use_data 不能为空字符串")

    # 安全引用 SQL 标识符
    def _q(name: str) -> str:
        return '"' + str(name).replace('"', '""') + '"'

    x_field_exists = conn.execute(
        """
        SELECT COUNT(*)
        FROM information_schema.columns
        WHERE table_name = 'X_HyS_data'
          AND column_name = ?
        """,
        [use_data],
    ).fetchone()[0]

    if x_field_exists == 0:
        raise ValueError(f"X_HyS_data 中不存在字段: {use_data}")

    use_data_sql = _q(use_data)

    # 读取 obs / var

    obs = conn.execute("SELECT * FROM obs ORDER BY atlas_cell_id").df()
    var = conn.execute("SELECT * FROM var ORDER BY atlas_gene_id").df()

    obs = obs.set_index("atlas_cell_id")
    var = var.set_index("atlas_gene_id")

    n_cells = obs.shape[0]
    n_genes = var.shape[0]

    # 读取 CSR indptr
    indptr_df = conn.execute("""
        SELECT indptr
        FROM X_HyS_indptr
        ORDER BY atlas_cell_id
    """).df()

    indptr = np.empty(n_cells + 1, dtype=np.int64)
    indptr[0] = 0
    indptr[1:] = indptr_df["indptr"].to_numpy(dtype=np.int64)

    nnz = int(indptr[-1])

    # 创建 h5ad 文件
    with h5py.File(out_h5ad_path, "w") as f:

        # ---------- 根节点属性 ----------
        f.attrs["encoding-type"] = "anndata"
        f.attrs["encoding-version"] = "0.1.0"

        gX = f.create_group("X")

        # AnnData CSR 必须写在 attrs，而不是 dataset
        gX.attrs["encoding-type"] = "csr_matrix"
        gX.attrs["encoding-version"] = "0.1.0"
        gX.attrs["shape"] = (n_cells, n_genes)

        # ---------- datasets ----------
        d_data = gX.create_dataset(
            "data",
            shape=(nnz,),
            dtype="float32",
            chunks=(min(batch_cells, nnz),),
        )

        d_indices = gX.create_dataset(
            "indices",
            shape=(nnz,),
            dtype="uint16",
            chunks=(min(batch_cells, nnz),),
        )

        gX.create_dataset("indptr", data=indptr, dtype="int64")

        offset = 0

        for start in progress(
            range(0, nnz, batch_cells),
            desc="write_h5ad"
        ):
            end = min(start + batch_cells, nnz)

            rows = conn.execute(
                f"""
                SELECT atlas_gene_id, {use_data_sql}
                FROM X_HyS_data
                WHERE id >= ? AND id < ?
                ORDER BY id
                """,
                [int(start), int(end)],
            ).fetchall()

            if not rows:
                continue

            idx, val = zip(*rows)

            d_indices[offset:offset + len(idx)] = idx
            d_data[offset:offset + len(val)] = val

            offset += len(idx)

        assert offset == nnz, f"nnz mismatch: {offset} != {nnz}"

        _write_dataframe(f, "obs", obs)
        _write_dataframe(f, "var", var)

        g_obsm = f.create_group("obsm")

        for (table_name,) in conn.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_name LIKE 'obsm_%'
        """).fetchall():

            key = table_name.replace("obsm_", "")

            value_cols = [
                row[0]
                for row in conn.execute("""
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_name = ?
                      AND column_name <> 'atlas_cell_id'
                    ORDER BY ordinal_position
                """, [table_name]).fetchall()
            ]

            if len(value_cols) == 0:
                continue

            select_values = ", ".join([
                f"t.{_q(c)} AS {_q(c)}"
                for c in value_cols
            ])

            df = conn.execute(f"""
                SELECT
                    {select_values}
                FROM obs AS o
                LEFT JOIN {_q(table_name)} AS t
                  ON o.atlas_cell_id = t.atlas_cell_id
                ORDER BY o.atlas_cell_id
            """).df()

            arr = df.to_numpy(dtype=np.float32)

            if arr.shape[0] != n_cells:
                raise ValueError(
                    f"obsm[{key}] 行数错误: {arr.shape[0]} != n_cells {n_cells}"
                )

            g_obsm.create_dataset(key, data=arr)

        g_varm = f.create_group("varm")

        def _q(name: str) -> str:
            return '"' + str(name).replace('"', '""') + '"'

        for (table_name,) in conn.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_name LIKE 'varm_%'
        """).fetchall():

            key = table_name.replace("varm_", "")

            # AnnData 要求 varm[key] 的第一维必须等于 var 的行数。
            # 读取 varm 表中除 atlas_gene_id 外的数值列
            value_cols = [
                row[0]
                for row in conn.execute("""
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_name = ?
                      AND column_name <> 'atlas_gene_id'
                    ORDER BY ordinal_position
                """, [table_name]).fetchall()
            ]

            if len(value_cols) == 0:
                continue

            select_values = ", ".join([
                f"t.{_q(c)} AS {_q(c)}"
                for c in value_cols
            ])

            df = conn.execute(f"""
                SELECT
                    {select_values}
                FROM var AS v
                LEFT JOIN {_q(table_name)} AS t
                  ON v.atlas_gene_id = t.atlas_gene_id
                ORDER BY v.atlas_gene_id
            """).df()

            # ============================================================
            #   HVG 基因     : 原始 PCA loading
            #   非 HVG 基因  : NaN
            # ============================================================
            arr = df.to_numpy(dtype=np.float32)

            if arr.shape[0] != n_genes:
                raise ValueError(
                    f"varm[{key}] 行数错误: {arr.shape[0]} != n_genes {n_genes}"
                )

            g_varm.create_dataset(key, data=arr)

    logger.info(f" write_h5ad Done, 耗时: {(datetime.now() - start_time).total_seconds():.2f} 秒")


# 写 AnnData 到 h5ad
def _write_dataframe(f: h5py.File, key: str, df: pd.DataFrame):

    """将计算结果写入数据库表。

    该内部函数属于数据导出模块，用于支撑同一模块中的公共 API。

    把 Atlas 数据库重新组装为 h5ad、AnnData 或 pandas DataFrame。

    它通常不会作为用户入口直接调用；直接调用时需要保证输入对象、数据库连接和相关临时表已经由上游步骤准备好。

    Parameters
    ----------
    f
        打开的 HDF5 文件句柄。

    key
        结果键名、表名前缀或 HDF5 group 名称。

    df
        包含中间统计量或绘图数据的 pandas DataFrame。

    Notes
    -----
    这是内部 helper；除非需要扩展 scAtlasPy 内部流程，一般不建议在用户代码中直接调用。
    """
    g = f.create_group(key)

    # ---- AnnData dataframe metadata ----
    g.attrs["encoding-type"] = "dataframe"
    g.attrs["encoding-version"] = "0.2.0"

    # index
    index_name = df.index.name or "_index"
    index_data = np.array(df.index.astype(str).tolist(), dtype=object)

    g.create_dataset(
        index_name,
        data=index_data,
        dtype=h5py.string_dtype(encoding="utf-8"),
    )

    # columns
    colnames = []

    for col in df.columns:
        colnames.append(col)
        series = df[col]

        # pandas categorical → string
        if pd.api.types.is_categorical_dtype(series):
            series = series.astype(str)

        arr = series.to_numpy()

        # ===== 字符串列 =====
        if arr.dtype.kind in {"U", "O"}:
            data = np.array(series.astype(str).tolist(), dtype=object)

            g.create_dataset(
                col,
                data=data,
                dtype=h5py.string_dtype(encoding="utf-8"),
            )

        # ===== 数值列 =====
        else:
            g.create_dataset(col, data=arr)

    # AnnData spec attrs（关键）
    g.attrs["column-order"] = np.array(colnames, dtype="S")
    g.attrs["_index"] = index_name


# 将 DuckDB 中的 obs 表导出为 pandas DataFrame
def get_obs_df(
    atlas: Atlas,
    columns: list[str] | str | None = None,
):
    """读取 Atlas 数据库中的 obs 表。

    该函数把 ``obs`` 表中的全部列或指定列读取为 DataFrame，适合快速检查细胞元数据、导出统计结果或与外部分析结果合并。

    Parameters
    ----------
    atlas
        Atlas 对象。通常需要已经连接到 DuckDB 数据库，并包含该函数读取或写入所需的 ``obs``、``var``、表达矩阵或结果表。
    columns
        需要从 ``obs`` 中读取的列名。可以是单个字符串、字符串列表或 ``None``。

    Returns
    -------
    pandas.DataFrame
        包含查询、统计或绘图所需数据的表格。

    Examples
    --------
    读取全部 obs 信息::

        obs = atlas.get_obs_df()

    只读取聚类和自动注释列::

        obs = atlas.get_obs_df(columns=["kmeans", "cell_type_auto"])"""

    start_time = datetime.now()

    conn = atlas.connection

    if conn is None:
        raise ValueError("atlas.connection 为空，请先连接数据库")

    # 1. 检查 obs 表是否存在
    obs_exists = conn.execute("""
        SELECT COUNT(*)
        FROM information_schema.tables
        WHERE table_name = 'obs'
    """).fetchone()[0]

    if obs_exists == 0:
        raise ValueError("数据库中不存在 obs 表")

    # 2. 获取 obs 所有字段
    obs_columns = [
        row[0]
        for row in conn.execute("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'obs'
            ORDER BY ordinal_position
        """).fetchall()
    ]

    if "atlas_cell_id" not in obs_columns:
        raise ValueError("obs 表中不存在 atlas_cell_id 字段，无法设置 pandas index")

    # 3. 处理 columns
    if columns is None:
        select_columns = obs_columns
    else:
        if isinstance(columns, str):
            columns = [columns]

        # 检查字段是否存在
        missing = [c for c in columns if c not in obs_columns]
        if missing:
            raise ValueError(f"obs 表中不存在这些字段: {missing}")

        # atlas_cell_id 一定要有，并且放在第一列
        select_columns = ["atlas_cell_id"] + [
            c for c in columns
            if c != "atlas_cell_id"
        ]

    # 4. 查询 obs
    select_sql = ", ".join([f'"{c}"' for c in select_columns])

    sql = f"""
        SELECT {select_sql}
        FROM obs
    """

    df = conn.execute(sql).df()

    # 5. 默认 atlas_cell_id 作为 pandas index
    df = df.set_index("atlas_cell_id", drop=False)

    logger.info(f" get_obs_df Done, 耗时: {(datetime.now() - start_time).total_seconds():.2f} 秒")

    return df


# 根据 atlas_cell_id list，从 DuckDB 中导出子集 AnnData 到内存
def get_anndata(
    atlas: Atlas,
    atlas_cell_ids: list[int] | np.ndarray | None,
    use_data: str = "data_count",
    include_obsm: bool = True,
    include_varm: bool = True,
):
    """从 Atlas 数据库构建 AnnData 对象。

    该函数按指定细胞 ID、表达矩阵表和可选 embedding 结果，从 Atlas 数据库中构建内存 AnnData。适合抽样导出、局部分析或与 Scanpy 工作流衔接。

    Parameters
    ----------
    atlas
        Atlas 对象。通常需要已经连接到 DuckDB 数据库，并包含该函数读取或写入所需的 ``obs``、``var``、表达矩阵或结果表。
    atlas_cell_ids
        需要导出的 Atlas 细胞 ID 列表；为 ``None`` 时通常导出当前索引对应的全部细胞。
    use_data
        读取的表达矩阵或结果表名称。常用值包括 ``"data_count"``、``"data_normalize"``、``"data_log1p"`` 和
        ``"data_scale"``。
    include_obsm
        是否把 ``obsm_*`` 结果表写入返回的 AnnData。
    include_varm
        是否把 ``varm_*`` 结果表写入返回的 AnnData。

    Returns
    -------
    AnnData
        从 Atlas 数据库构建的 AnnData 对象。

    Examples
    --------
    导出指定细胞::

        cell_ids = [0, 1, 2, 3]
        adata = atlas.get_anndata(cell_ids, use_data="data_log1p")

    导出过滤后的前 5000 个细胞并包含 UMAP/PCA::

        cell_ids = atlas.query(
            "SELECT atlas_cell_id FROM obs WHERE filter_cells = TRUE LIMIT 5000"
        )["atlas_cell_id"].tolist()
        adata = atlas.get_anndata(cell_ids, include_obsm=True, include_varm=True)"""



    conn = atlas.connection

    if conn is None:
        raise ValueError("atlas.connection 为空，请先连接数据库")


    # 0. 基本检查
    if atlas_cell_ids is None or len(atlas_cell_ids) == 0:
        raise ValueError("atlas_cell_ids 不能为空")

    atlas_cell_ids = [int(x) for x in atlas_cell_ids]

    if len(atlas_cell_ids) != len(set(atlas_cell_ids)):
        raise ValueError("atlas_cell_ids 中存在重复值，请先去重")

    # DuckDB 标识符安全引用
    def _q(name: str) -> str:
        """为 SQL 标识符添加安全引用。

        该内部函数属于数据导出模块，用于支撑同一模块中的公共 API。

        把 Atlas 数据库重新组装为 h5ad、AnnData 或 pandas DataFrame。

        它通常不会作为用户入口直接调用；直接调用时需要保证输入对象、数据库连接和相关临时表已经由上游步骤准备好。

        Parameters
        ----------
        name
            对象名称、列名或 SQL 标识符，具体含义由调用位置决定。

        Returns
        -------
        quoted_name
            加双引号后的 SQL 标识符。

        Notes
        -----
        这是内部 helper；除非需要扩展 scAtlasPy 内部流程，一般不建议在用户代码中直接调用。
        """
        return '"' + name.replace('"', '""') + '"'

    # 检查 use_data 是否存在
    x_field_exists = conn.execute(
        """
        SELECT COUNT(*)
        FROM information_schema.columns
        WHERE table_name = 'X_HyS_data'
          AND column_name = ?
        """,
        [use_data],
    ).fetchone()[0]

    if x_field_exists == 0:
        raise ValueError(f"X_HyS_data 中不存在字段: {use_data}")

    # 1. 创建临时 selected cell 表，保留用户输入顺序
    selected_df = pd.DataFrame({
        "atlas_cell_id": atlas_cell_ids,
        "_cell_order": np.arange(len(atlas_cell_ids), dtype=np.int64),
    })

    conn.execute("DROP TABLE IF EXISTS _selected_cells")

    conn.register("_selected_cells_df", selected_df)
    conn.execute("""
        CREATE TEMP TABLE _selected_cells AS
        SELECT
            CAST(atlas_cell_id AS BIGINT) AS atlas_cell_id,
            CAST(_cell_order AS BIGINT) AS _cell_order
        FROM _selected_cells_df
    """)
    conn.unregister("_selected_cells_df")

    # 2. 读取 obs 子集
    obs = conn.execute("""
        SELECT o.*
        FROM obs AS o
        JOIN _selected_cells AS s
          ON o.atlas_cell_id = s.atlas_cell_id
        ORDER BY s._cell_order
    """).df()

    if obs.shape[0] != len(atlas_cell_ids):
        found = set(obs["atlas_cell_id"].astype(int).tolist())
        missing = [x for x in atlas_cell_ids if x not in found]
        raise ValueError(
            f"有 {len(missing)} 个 atlas_cell_id 在 obs 中不存在，"
            f"例如: {missing[:10]}"
        )

    if "atlas_cell_name" not in obs.columns:
        raise ValueError("obs 表中不存在 atlas_cell_name 字段，无法作为 AnnData obs index")

    obs = obs.set_index("atlas_cell_name", drop=False)
    obs.index = obs.index.astype(str)

    var = conn.execute("""
        SELECT *
        FROM var
        ORDER BY atlas_gene_id
    """).df()

    if "atlas_gene_id" not in var.columns:
        raise ValueError("var 表中不存在 atlas_gene_id 字段")

    if "atlas_gene_name" not in var.columns:
        raise ValueError("var 表中不存在 atlas_gene_name 字段，无法作为 AnnData var index")

    var = var.set_index("atlas_gene_name", drop=False)
    var.index = var.index.astype(str)

    n_cells = obs.shape[0]
    n_genes = var.shape[0]

    # 4. 读取 X 子集，并组装 CSR
    x_sql = f"""
        SELECT
            s._cell_order AS row_id,
            x.atlas_gene_id AS col_id,
            x.{_q(use_data)} AS value
        FROM X_HyS_data AS x
        JOIN _selected_cells AS s
          ON x.atlas_cell_id = s.atlas_cell_id
        ORDER BY s._cell_order, x.atlas_gene_id
    """

    x_df = conn.execute(x_sql).df()

    if x_df.shape[0] == 0:
        X = sparse.csr_matrix((n_cells, n_genes), dtype=np.float32)
        nnz = 0
    else:
        rows = x_df["row_id"].to_numpy(dtype=np.int64)
        cols = x_df["col_id"].to_numpy(dtype=np.int64)
        vals = x_df["value"].to_numpy(dtype=np.float32)

        X = sparse.csr_matrix(
            (vals, (rows, cols)),
            shape=(n_cells, n_genes),
            dtype=np.float32,
        )

        X.sort_indices()
        nnz = X.nnz

    # 5. 创建 AnnData
    adata = AnnData(
        X=X,
        obs=obs,
        var=var,
    )

    # 6. 读取 obsm 子集
    if include_obsm:

        obsm_tables = conn.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_name LIKE 'obsm_%'
            ORDER BY table_name
        """).fetchall()

        for (table_name,) in obsm_tables:
            key = table_name.replace("obsm_", "")

            # ============================================================
            #  以 _selected_cells 为基准 LEFT JOIN obsm 表：
            #   有 embedding 的 cell     -> 保留原值
            #   没有 embedding 的 cell   -> NaN
            # ============================================================

            value_cols = [
                row[0]
                for row in conn.execute("""
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_name = ?
                      AND column_name <> 'atlas_cell_id'
                    ORDER BY ordinal_position
                """, [table_name]).fetchall()
            ]

            if len(value_cols) == 0:
                continue

            select_values = ", ".join([
                f"t.{_q(c)} AS {_q(c)}"
                for c in value_cols
            ])

            df = conn.execute(f"""
                SELECT
                    {select_values}
                FROM _selected_cells AS s
                LEFT JOIN {_q(table_name)} AS t
                  ON s.atlas_cell_id = t.atlas_cell_id
                ORDER BY s._cell_order
            """).df()

            arr = df.to_numpy(dtype=np.float32)

            if arr.shape[0] != n_cells:
                raise ValueError(
                    f"obsm[{key}] 行数错误: {arr.shape[0]} != selected cells {n_cells}"
                )

            adata.obsm[key] = arr

    # 7. 读取 varm 全集
    if include_varm:

        varm_tables = conn.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_name LIKE 'varm_%'
            ORDER BY table_name
        """).fetchall()

        for (table_name,) in varm_tables:
            key = table_name.replace("varm_", "")

            value_cols = [
                row[0]
                for row in conn.execute("""
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_name = ?
                      AND column_name <> 'atlas_gene_id'
                    ORDER BY ordinal_position
                """, [table_name]).fetchall()
            ]

            if len(value_cols) == 0:
                continue

            select_values = ", ".join([
                f"t.{_q(c)} AS {_q(c)}"
                for c in value_cols
            ])

            df = conn.execute(f"""
                SELECT
                    {select_values}
                FROM var AS v
                LEFT JOIN {_q(table_name)} AS t
                  ON v.atlas_gene_id = t.atlas_gene_id
                ORDER BY v.atlas_gene_id
            """).df()

            arr = df.to_numpy(dtype=np.float32)

            if arr.shape[0] != n_genes:
                raise ValueError(
                    f"varm[{key}] 行数错误: {arr.shape[0]} != genes {n_genes}"
                )

            adata.varm[key] = arr

    # 8. 清理临时表
    conn.execute("DROP TABLE IF EXISTS _selected_cells")

    logger.info(" AnnData 导出完成")
    logger.info(f"  - cells: {adata.n_obs:,}")
    logger.info(f"  - genes: {adata.n_vars:,}")

    return adata
