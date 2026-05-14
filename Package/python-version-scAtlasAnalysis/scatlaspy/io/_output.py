import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm

''' 数据导出: 把数据直接变成文件'''
# 819200 耗时 11:09
def export_duckdb_to_h5ad(
    atlas,
    out_h5ad_path: str,
    *,
    batch_size: int = 1_000_000,  # nnz batch
):
    """
    从 DuckDB 流式导出 h5ad（不经过 AnnData）
    特点：
      - CSR-only
      - nnz streaming
      - 内存 O(batch_size)
      - scanpy / anndata 完全兼容
    """

    conn = atlas.connection

    # 1️⃣ 读取 obs / var
    print("[EXPORT] 读取 obs / var")

    obs = conn.execute("SELECT * FROM obs ORDER BY atlas_cell_id").df()
    var = conn.execute("SELECT * FROM var ORDER BY atlas_gene_id").df()

    obs = obs.set_index("atlas_cell_id")
    var = var.set_index("atlas_gene_id")

    n_cells = obs.shape[0]
    n_genes = var.shape[0]

    print(f"[EXPORT] cells={n_cells:,}, genes={n_genes:,}")

    # 2️⃣ 读取 CSR indptr
    print("[EXPORT] 读取 CSR indptr")

    indptr_df = conn.execute("""
        SELECT indptr
        FROM X_CSRO_indptr
        ORDER BY atlas_cell_id
    """).df()

    indptr = np.empty(n_cells + 1, dtype=np.int64)
    indptr[0] = 0
    indptr[1:] = indptr_df["indptr"].to_numpy(dtype=np.int64)

    nnz = int(indptr[-1])
    print(f"[EXPORT] nnz={nnz:,}")

    # 3️⃣ 创建 h5ad 文件
    print(f"[EXPORT] 创建 h5ad → {out_h5ad_path}")

    with h5py.File(out_h5ad_path, "w") as f:

        # ---------- 根节点属性 ----------
        f.attrs["encoding-type"] = "anndata"
        f.attrs["encoding-version"] = "0.1.0"

        # 4️⃣ 写 X (CSR, streaming)
        print("[EXPORT] 写 X (CSR streaming)")

        gX = f.create_group("X")

        # 🔴 FIX 1：AnnData CSR 必须写在 attrs，而不是 dataset
        gX.attrs["encoding-type"] = "csr_matrix"
        gX.attrs["encoding-version"] = "0.1.0"
        gX.attrs["shape"] = (n_cells, n_genes)

        # ---------- datasets ----------
        d_data = gX.create_dataset(
            "data",
            shape=(nnz,),
            dtype="float32",
            chunks=(min(batch_size, nnz),),
        )

        d_indices = gX.create_dataset(
            "indices",
            shape=(nnz,),
            dtype="uint16",
            chunks=(min(batch_size, nnz),),
        )

        gX.create_dataset("indptr", data=indptr, dtype="int64")

        offset = 0

        for start in tqdm(
            range(0, nnz, batch_size),
            desc="CSR data"
        ):
            end = min(start + batch_size, nnz)

            rows = conn.execute(
                """
                SELECT atlas_gene_id, data
                FROM X_CSRO_data
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

        # 5️⃣ 写 obs / var
        print("[EXPORT] 写 obs / var")

        _write_dataframe(f, "obs", obs)
        _write_dataframe(f, "var", var)

        # 6️⃣ 写 obsm
        print("[EXPORT] 写 obsm")

        g_obsm = f.create_group("obsm")

        for (table_name,) in conn.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_name LIKE 'obsm_%'
        """).fetchall():

            key = table_name.replace("obsm_", "")
            df = conn.execute(f"""
                SELECT *
                FROM {table_name}
                ORDER BY atlas_cell_id
            """).df()

            df = df.drop(columns=["atlas_cell_id"])
            g_obsm.create_dataset(key, data=df.to_numpy())

            print(f"  - obsm[{key}] {df.shape}")

        # 7️⃣ 写 varm
        print("[EXPORT] 写 varm")

        g_varm = f.create_group("varm")

        for (table_name,) in conn.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_name LIKE 'varm_%'
        """).fetchall():

            key = table_name.replace("varm_", "")
            df = conn.execute(f"""
                SELECT *
                FROM {table_name}
                ORDER BY atlas_gene_id
            """).df()

            df = df.drop(columns=["atlas_gene_id"])
            g_varm.create_dataset(key, data=df.to_numpy())

            print(f"  - varm[{key}] {df.shape}")

    print("✅ 导出完成")


''' 写 AnnData-compatible DataFrame 到 h5ad '''
def _write_dataframe(f, key, df):
    """
    写 AnnData-compatible DataFrame 到 h5ad
    """
    g = f.create_group(key)

    # ---- AnnData dataframe metadata ----
    g.attrs["encoding-type"] = "dataframe"
    g.attrs["encoding-version"] = "0.2.0"

    # 1️⃣ index
    index_name = df.index.name or "_index"
    index_data = np.array(df.index.astype(str).tolist(), dtype=object)

    g.create_dataset(
        index_name,
        data=index_data,
        dtype=h5py.string_dtype(encoding="utf-8"),
    )

    # 2️⃣ columns
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

    # 3️⃣ AnnData spec attrs（关键）
    g.attrs["column-order"] = np.array(colnames, dtype="S")
    g.attrs["_index"] = index_name


# 将 DuckDB 中的 obs 表导出为 pandas DataFrame
def export_obs_to_pandas(
    atlas,
    columns: list[str] | str | None = None,
):
    """
    将 DuckDB 中的 obs 表导出为 pandas DataFrame。

    Parameters
    ----------
    atlas : Atlas
        scAtlasPy 的 Atlas 对象，要求 atlas.connection 已连接 DuckDB。

    columns : list[str] | str | None
        需要导出的 obs 字段。

        - None：
            导出 obs 的所有字段。

        - list[str] 或 str：
            导出 atlas_cell_id + 指定字段。
            atlas_cell_id 一定会自动加入，用作 pandas index。

    Returns
    -------
    pandas.DataFrame
        obs 表对应的 DataFrame。
        默认使用 atlas_cell_id 作为 index。
    """

    conn = atlas.connection

    if conn is None:
        raise ValueError("atlas.connection 为空，请先连接数据库")

    # -------------------------------------------------
    # 1. 检查 obs 表是否存在
    # -------------------------------------------------
    obs_exists = conn.execute("""
        SELECT COUNT(*)
        FROM information_schema.tables
        WHERE table_name = 'obs'
    """).fetchone()[0]

    if obs_exists == 0:
        raise ValueError("数据库中不存在 obs 表")

    # -------------------------------------------------
    # 2. 获取 obs 所有字段
    # -------------------------------------------------
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

    # -------------------------------------------------
    # 3. 处理 columns
    # -------------------------------------------------
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

    # -------------------------------------------------
    # 4. 查询 obs
    # -------------------------------------------------
    select_sql = ", ".join([f'"{c}"' for c in select_columns])

    sql = f"""
        SELECT {select_sql}
        FROM obs
    """

    df = conn.execute(sql).df()

    # -------------------------------------------------
    # 5. 默认 atlas_cell_id 作为 pandas index
    # -------------------------------------------------
    df = df.set_index("atlas_cell_id")

    return df


# 示例函数； 从 obs_df 中筛选 filter_col == True 的细胞，
def get_filtered_cell_ids(obs_df, filter_col: str = "filter_cells"):
    """
    示例函数：
    从 obs_df 中筛选 filter_col == True 的细胞，
    返回 atlas_cell_id list。

    Parameters
    ----------
    obs_df : pandas.DataFrame
        export_obs_to_pandas() 返回的 obs DataFrame。

    filter_col : str
        用于筛选的布尔列名。
        默认是 filter_cells。

    Returns
    -------
    list[int]
        满足 filter_col == True 的 atlas_cell_id 列表。
    """

    if obs_df.index.name != "atlas_cell_id":
        raise ValueError(
            "obs_df 的 index 不是 atlas_cell_id。"
            "请确认 obs_df 是否来自 export_obs_to_pandas()。"
        )

    if filter_col not in obs_df.columns:
        raise ValueError(f"obs_df 中不存在字段: {filter_col}")

    filtered_df = obs_df[obs_df[filter_col] == True]

    return filtered_df.index.astype(int).tolist()


# 根据 atlas_cell_id list，从 DuckDB 中导出子集 AnnData 到内存
def export_cells_to_anndata(
    atlas,
    atlas_cell_ids,
    x_field: str = "data",
    include_obsm: bool = True,
    include_varm: bool = True,
):
    """
    根据 atlas_cell_id list，从 DuckDB 中导出子集 AnnData 到内存。

    导出内容：
    - obs  : 子集 obs
    - var  : 全量 var
    - X    : 子集 cell × 全量 gene
    - obsm : 子集 obsm
    - varm : 全量 varm

    Parameters
    ----------
    atlas : Atlas
        scAtlasPy 的 Atlas 对象，要求 atlas.connection 已连接 DuckDB。

    atlas_cell_ids : list[int]
        需要导出的 atlas_cell_id 列表。
        顺序会被保留。

    x_field : str
        X_CSRO_data 中作为表达矩阵值的字段。
        默认 "data"。
        也可以是：
            "data_log1p"
            "data_scale"
            "data_normalize"

    include_obsm : bool
        是否导出 obsm_* 表。

    include_varm : bool
        是否导出 varm_* 表。

    Returns
    -------
    AnnData
        内存中的 AnnData 对象。

    Notes
    -----
    ✅ 修改点：
    1. obs index 使用 atlas_cell_name，而不是 atlas_cell_id
    2. var index 使用 atlas_gene_name，而不是 atlas_gene_id
    3. obs.index / var.index 显式 astype(str)，消除 AnnData 的 ImplicitModificationWarning
    """

    import numpy as np
    import pandas as pd
    from scipy import sparse
    from anndata import AnnData

    conn = atlas.connection

    if conn is None:
        raise ValueError("atlas.connection 为空，请先连接数据库")

    # -------------------------------------------------
    # 0. 基本检查
    # -------------------------------------------------
    if atlas_cell_ids is None or len(atlas_cell_ids) == 0:
        raise ValueError("atlas_cell_ids 不能为空")

    atlas_cell_ids = [int(x) for x in atlas_cell_ids]

    if len(atlas_cell_ids) != len(set(atlas_cell_ids)):
        raise ValueError("atlas_cell_ids 中存在重复值，请先去重")

    # DuckDB 标识符安全引用
    def _q(name: str) -> str:
        return '"' + name.replace('"', '""') + '"'

    # 检查 x_field 是否存在
    x_field_exists = conn.execute(
        """
        SELECT COUNT(*)
        FROM information_schema.columns
        WHERE table_name = 'X_CSRO_data'
          AND column_name = ?
        """,
        [x_field],
    ).fetchone()[0]

    if x_field_exists == 0:
        raise ValueError(f"X_CSRO_data 中不存在字段: {x_field}")

    print("==== export_cells_to_anndata ====")
    print(f"[INFO] selected cells = {len(atlas_cell_ids):,}")
    print(f"[INFO] x_field = {x_field}")

    # -------------------------------------------------
    # 1. 创建临时 selected cell 表，保留用户输入顺序
    # -------------------------------------------------
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

    # -------------------------------------------------
    # 2. 读取 obs 子集
    # -------------------------------------------------
    print("[EXPORT] 读取 obs 子集")

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

    # -------------------------------------------------
    # ✅ 修改 1：AnnData obs index 改为 atlas_cell_name
    # ✅ 修改 2：显式转成 str，消除 ImplicitModificationWarning
    # -------------------------------------------------
    if "atlas_cell_name" not in obs.columns:
        raise ValueError("obs 表中不存在 atlas_cell_name 字段，无法作为 AnnData obs index")

    obs = obs.set_index("atlas_cell_name", drop=False)
    obs.index = obs.index.astype(str)

    # -------------------------------------------------
    # 3. 读取 var 全集
    # -------------------------------------------------
    print("[EXPORT] 读取 var 全集")

    var = conn.execute("""
        SELECT *
        FROM var
        ORDER BY atlas_gene_id
    """).df()

    if "atlas_gene_id" not in var.columns:
        raise ValueError("var 表中不存在 atlas_gene_id 字段")

    # -------------------------------------------------
    # ✅ 修改 3：AnnData var index 改为 atlas_gene_name
    # ✅ 修改 4：显式转成 str，消除 ImplicitModificationWarning
    # -------------------------------------------------
    if "atlas_gene_name" not in var.columns:
        raise ValueError("var 表中不存在 atlas_gene_name 字段，无法作为 AnnData var index")

    var = var.set_index("atlas_gene_name", drop=False)
    var.index = var.index.astype(str)

    n_cells = obs.shape[0]
    n_genes = var.shape[0]

    print(f"[EXPORT] AnnData shape = {n_cells:,} × {n_genes:,}")

    # -------------------------------------------------
    # 4. 读取 X 子集，并组装 CSR
    # -------------------------------------------------
    print("[EXPORT] 读取 X 子集并构建 CSR")

    x_sql = f"""
        SELECT
            s._cell_order AS row_id,
            x.atlas_gene_id AS col_id,
            x.{_q(x_field)} AS value
        FROM X_CSRO_data AS x
        JOIN _selected_cells AS s
          ON x.atlas_cell_id = s.atlas_cell_id
        WHERE x.{_q(x_field)} IS NOT NULL
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

    print(f"[EXPORT] X nnz = {nnz:,}")

    # -------------------------------------------------
    # 5. 创建 AnnData
    # -------------------------------------------------
    adata = AnnData(
        X=X,
        obs=obs,
        var=var,
    )

    # -------------------------------------------------
    # 6. 读取 obsm 子集
    # -------------------------------------------------
    if include_obsm:
        print("[EXPORT] 读取 obsm 子集")

        obsm_tables = conn.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_name LIKE 'obsm_%'
            ORDER BY table_name
        """).fetchall()

        for (table_name,) in obsm_tables:
            key = table_name.replace("obsm_", "")

            df = conn.execute(f"""
                SELECT t.*
                FROM {_q(table_name)} AS t
                JOIN _selected_cells AS s
                  ON t.atlas_cell_id = s.atlas_cell_id
                ORDER BY s._cell_order
            """).df()

            if df.shape[0] != n_cells:
                print(
                    f"  ⚠️ 跳过 obsm[{key}]："
                    f"行数 {df.shape[0]:,} != selected cells {n_cells:,}"
                )
                continue

            if "atlas_cell_id" in df.columns:
                df = df.drop(columns=["atlas_cell_id"])

            adata.obsm[key] = df.to_numpy()

            print(f"  - obsm[{key}] {adata.obsm[key].shape}")

    # -------------------------------------------------
    # 7. 读取 varm 全集
    # -------------------------------------------------
    if include_varm:
        print("[EXPORT] 读取 varm 全集")

        varm_tables = conn.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_name LIKE 'varm_%'
            ORDER BY table_name
        """).fetchall()

        for (table_name,) in varm_tables:
            key = table_name.replace("varm_", "")

            df = conn.execute(f"""
                SELECT *
                FROM {_q(table_name)}
                ORDER BY atlas_gene_id
            """).df()

            if df.shape[0] != n_genes:
                print(
                    f"  ⚠️ 跳过 varm[{key}]："
                    f"行数 {df.shape[0]:,} != genes {n_genes:,}"
                )
                continue

            if "atlas_gene_id" in df.columns:
                df = df.drop(columns=["atlas_gene_id"])

            adata.varm[key] = df.to_numpy()

            print(f"  - varm[{key}] {adata.varm[key].shape}")

    # -------------------------------------------------
    # 8. 清理临时表
    # -------------------------------------------------
    conn.execute("DROP TABLE IF EXISTS _selected_cells")

    print("✅ AnnData 导出完成")
    print(f"  - cells: {adata.n_obs:,}")
    print(f"  - genes: {adata.n_vars:,}")
    print(f"  - nnz:   {adata.X.nnz:,}")

    return adata