# scatlaspy（Python）重要函数源码



## 目录

1. [核心类 Atlas](#1-核心类-atlas)
2. [质控函数（Quality Control）](#2-质控函数-quality-control)
3. [转换函数（Transformation）](#3-转换函数-transformation)
4. [输入函数（Input/IO）](#4-输入函数-inputio)
5. [输出函数（Output/IO）](#5-输出函数-outputio)

---

## 1. 核心类 Atlas

### 1.1 类定义（_atlas.py:29-82）

```python
class Atlas:
    """
    这是一个Atlas类，用来管理和待分析数据集的数据库的交互
    """
    def __init__(self, name: str, path: str):
        """
        初始化类实例
        Args:
            name: 名称
            path: 路径
        """
        # 检查数据库文件是否存在，如果不存在则创建
        db_file = os.path.join(path, f"{name}.sasql")

        if not os.path.exists(db_file):
            self.__connection = self._create(name, path)
        else:
            self.__connection = self.connect("r+")
```

### 1.2 数据库连接（_atlas.py:313-368）

```python
def connect(self, mode: Literal["r+", "r"] = "r+") -> duckdb.DuckDBPyConnection:
    """
    和self.name命名的数据库进行连接。
    如果<name.sasql>不存在，则创建并连接；
    如果<name.sasql>存在，则直接连接

    :param mode: 指定模式，只读or读写，
    :return: 数据库连接对象
    """
    if mode == "r":  # 只读模式
        self.__connection = duckdb.connect(database=db_file, read_only=True)
    elif mode == "r+":  # 读写模式
        self.__connection = duckdb.connect(database=db_file, read_only=False)
```

### 1.3 视图/子集化（_atlas.py:127-174）

```python
def __getitem__(self, item):
    """
    使用 obs_cell_id 列表管理视图，不使用数据库视图
    """
    # 执行SQL查询获取所有细胞ID
    query = "SELECT cell_id FROM obs"
    result = self.execute_sql(query).fetchall()
    all_cell_ids = [row[0] for row in result] if result else []

    # 根据索引类型筛选
    if isinstance(item, int):
        __obs_cell_id = [all_cell_ids[item]]
    elif isinstance(item, slice):
        selected_ids = all_cell_ids[item]
        __obs_cell_id = selected_ids
    elif isinstance(item, str):
        __obs_cell_id = [item] if item in all_cell_ids else []

    # 创建新的atlas对象视图
    new_atlas = Atlas(self.__name, self.__path)
    new_atlas.__obs_cell_id = __obs_cell_id
    new_atlas.__isView = True
    return new_atlas
```

### 1.4 分批查询（_atlas.py:476-583）

```python
def query_minibatch(self, mode="order", batch_size=2048, drop_last=True):
    """
    minibatch查询
    以minibatch方式遍历整个数据集，返回的数据用anndata封装

    :param mode: order/random_no_replace/random_replace
    :param batch_size: 每次读取的大小
    :param drop_last: 是否丢弃最后不足batch_size的数据
    :return: 生成器，每次返回AnnData对象
    """
    # 调用具体的分批读取方法
    return self.minibatch_scan_order_cursor_csr_df_arrow_onlylie(
        total_num, batch_size, drop_last
    )
```

### 1.5 SQL执行（_atlas.py:392-418）

```python
def execute_sql(self, sql: str) -> DuckDBPyConnection | None:
    """
    提交执行sql语句
    :param sql: 要执行的SQL语句
    :return: 如果查询有结果则返回结果集，否则返回None
    """
    result = self.__connection.execute(sql)

    if sql_upper.startswith(('SELECT', 'SHOW', 'DESCRIBE', 'EXPLAIN')):
        return result  # 查询语句返回结果
    else:
        self.__connection.commit()  # 非查询语句提交事务
        return None
```

---

## 2. 质控函数（Quality Control）

### 2.1 filter_cells_CSR_ultrafast（_quality_control.py:22-104）

```python
def filter_cells_CSR_ultrafast(atlas: 'Atlas',
                               min_counts=None, min_genes=None,
                               max_counts=None, max_genes=None,
                               add_key="filter_cells_1") -> None:
    """
    使用 DuckDB 原生并行过滤细胞

    步骤：
    1. ALTER TABLE obs ADD COLUMN {add_key} BOOLEAN DEFAULT FALSE
    2. CREATE TEMP TABLE keep_cells AS SELECT cell_index FROM (...) WHERE condition
    3. UPDATE obs SET {add_key}=FALSE
    4. UPDATE obs SET {add_key}=TRUE WHERE id IN (SELECT cell_index FROM keep_cells)
    """
    conn = atlas.connect("r+")
    conn.execute(f"PRAGMA threads={os.cpu_count()}")

    # 预先添加列
    conn.execute(f"ALTER TABLE obs ADD COLUMN IF NOT EXISTS {add_key} BOOLEAN DEFAULT FALSE")

    # 构建 SQL 过滤条件
    conds = []
    if min_counts is not None: conds.append(f"sum_expr >= {min_counts}")
    if max_counts is not None: conds.append(f"sum_expr <= {max_counts}")
    if min_genes  is not None: conds.append(f"nonzero_genes >= {min_genes}")
    if max_genes  is not None: conds.append(f"nonzero_genes <= {max_genes}")
    condition = " AND ".join(conds) if conds else "TRUE"

    # 计算需要保留的 cell_index
    conn.execute(f"""
        CREATE TEMP TABLE keep_cells AS
        SELECT cell_index
        FROM (
            SELECT cell_index, SUM(data) AS sum_expr, COUNT(*) AS nonzero_genes
            FROM X_CSR_data
            GROUP BY cell_index
        ) WHERE {condition}
    """)

    # 更新 obs 表
    conn.execute(f"UPDATE obs SET {add_key}=FALSE")
    conn.execute(f"""
        UPDATE obs SET {add_key}=TRUE
        WHERE id IN (SELECT cell_index FROM keep_cells)
    """)
```

### 2.2 filter_genes_CSR（_quality_control.py:130-240）

```python
def filter_genes_CSR(atlas: 'Atlas',
                     min_counts: Optional[int] = None,
                     min_cells: Optional[int] = None,
                     max_counts: Optional[int] = None,
                     max_cells: Optional[int] = None,
                     add_key: str = "filter_genes_1") -> None:
    """
    使用 CSR 数据计算每个基因的 sum_expr 和 nonzero_expr，并写入 var 表
    """
    conn = atlas.connect("r+")

    # 添加过滤字段
    conn.execute(f"ALTER TABLE var ADD COLUMN IF NOT EXISTS {add_key} BOOLEAN DEFAULT FALSE")

    # CSR 聚合统计
    rows = conn.execute("""
        SELECT indices AS gene_index, SUM(data) AS sum_expr, COUNT(*) AS nonzero_expr
        FROM X_CSR_data
        GROUP BY indices
        ORDER BY indices
    """).fetchall()

    # 写入临时表并更新 var
    conn.execute("CREATE TEMP TABLE tmp_stats (gene_id TEXT, flag BOOLEAN)")
    keep_count = 0
    insert_rows = []

    # 收集已经出现的基因
    appeared_gene_ids = set()

    for gene_index, sum_expr, nonzero_expr in rows:

        # id → gene_id
        gene_id = gene_map.get(gene_index)
        if gene_id is None:
            continue

        appeared_gene_ids.add(gene_id)

        ok = True
        if min_counts is not None and sum_expr < min_counts:
            ok = False
        if max_counts is not None and sum_expr > max_counts:
            ok = False
        if min_cells is not None and nonzero_expr < min_cells:
            ok = False
        if max_cells is not None and nonzero_expr > max_cells:
            ok = False

        insert_rows.append((gene_id, ok))
        if ok:
            keep_count += 1
    all_gene_ids = set(gene_map.values())
    zero_genes = all_gene_ids - appeared_gene_ids
    for g in zero_genes:
        insert_rows.append((g, False))

    conn.executemany("INSERT INTO tmp_stats VALUES (?,?)", insert_rows)
    conn.execute(f"""
        UPDATE var SET {add_key} = tmp.flag
        FROM tmp_stats AS tmp
        WHERE var.gene_id = tmp.gene_id
    """)
```

### 2.3 calculate_qc_metrics（_quality_control.py:518-720）

```python
def calculate_qc_metrics(atlas: Atlas,
                         qc_prefix: str = "MT-",
                         qc_key: str = "mt") -> None:
    """
    使用 DuckDB + CSR 稀疏存储，实现 Scanpy 的 calculate_qc_metrics
    """
    conn = atlas.connection

    # 1. 标记线粒体基因
    conn.execute(f"""
        ALTER TABLE var ADD COLUMN IF NOT EXISTS {qc_key} BOOLEAN
    """)
    conn.execute(f"""
        UPDATE var SET {qc_key} = CASE WHEN gene_id LIKE '{qc_prefix}%' THEN TRUE ELSE FALSE END
    """)

    # 2. Cell-wise QC：total_counts, n_genes_by_counts
    conn.execute("""
        CREATE OR REPLACE TEMP TABLE _cell_basic AS
        SELECT cell_index AS id, SUM(data) AS total_counts, COUNT(*) AS n_genes_by_counts
        FROM X_CSR_data WHERE data IS NOT NULL GROUP BY cell_index
    """)
    conn.execute("ALTER TABLE obs ADD COLUMN IF NOT EXISTS total_counts REAL")
    conn.execute("ALTER TABLE obs ADD COLUMN IF NOT EXISTS n_genes_by_counts INTEGER")
    conn.execute("""
        UPDATE obs SET total_counts = c.total_counts, n_genes_by_counts = c.n_genes_by_counts
        FROM _cell_basic c WHERE obs.id = c.id
    """)

    # 3. Mitochondrial QC：total_counts_mt, pct_counts_mt
    conn.execute(f"""
        CREATE OR REPLACE TEMP TABLE _cell_qc AS
        SELECT x.cell_index AS id, SUM(x.data) AS total_counts_qc
        FROM X_CSR_data x JOIN var v ON x.indices = v.id
        WHERE v.{qc_key} = TRUE GROUP BY x.cell_index
    """)
    conn.execute("ALTER TABLE obs ADD COLUMN IF NOT EXISTS total_counts_mt REAL")
    conn.execute("ALTER TABLE obs ADD COLUMN IF NOT EXISTS pct_counts_mt REAL")
    conn.execute("""
        UPDATE obs SET total_counts_mt = COALESCE(q.total_counts_qc, 0),
                       pct_counts_mt = CASE WHEN obs.total_counts > 0
                       THEN 100.0 * COALESCE(q.total_counts_qc, 0) / obs.total_counts ELSE 0 END
        FROM _cell_qc q WHERE obs.id = q.id
    """)

    # 4. Gene-wise QC：total_counts, n_cells_by_counts
    conn.execute("""
        CREATE OR REPLACE TEMP TABLE _gene_qc AS
        SELECT indices AS id, SUM(data) AS total_counts, COUNT(DISTINCT cell_index) AS n_cells_by_counts
        FROM X_CSR_data WHERE data IS NOT NULL GROUP BY indices
    """)
    conn.execute("ALTER TABLE var ADD COLUMN IF NOT EXISTS total_counts REAL")
    conn.execute("ALTER TABLE var ADD COLUMN IF NOT EXISTS n_cells_by_counts INTEGER")
    conn.execute("""
        UPDATE var SET total_counts = g.total_counts, n_cells_by_counts = g.n_cells_by_counts
        FROM _gene_qc g WHERE var.id = g.id
    """)
```

### 2.4 calculate_cell_total_counts（_quality_control.py:247-355）

```python
def calculate_cell_total_counts(atlas: 'Atlas',
                 batch_size: Optional[int] = 2048,
                 add_key: str = "cell_total_counts") -> None:
    """
    计算每个细胞的总UMI计数
    """
    atlas.connection = atlas.connect("r+")
    conn.execute(f"ALTER TABLE obs ADD COLUMN {add_key} REAL")

    for adata_minibatch in atlas.query_minibatch(batch_size=batch_size):
        total_counts = np.array(adata_minibatch.X.sum(axis=1)).flatten()
        cell_ids = adata_minibatch.obs['cell_id'].tolist()
        update_data = list(zip(total_counts, cell_ids))
        conn.executemany(f"UPDATE obs SET {add_key} = ? WHERE cell_id = ?", update_data)
        conn.commit()
```

### 2.5 calculate_gene_total_counts（_quality_control.py:358-503）

```python
def calculate_gene_total_counts(atlas: 'Atlas',
                                batch_size: Optional[int] = 2048,
                                add_key1: str = "gene_total_counts",
                                add_key2: str = "gene_means_counts") -> None:
    """
    计算每个基因的总表达值，平均表达值
    """
    conn.execute(f"ALTER TABLE var ADD COLUMN {add_key1} FLOAT DEFAULT 0.0")
    conn.execute(f"ALTER TABLE var ADD COLUMN {add_key2} FLOAT DEFAULT 0.0")

    # 分批处理基因
    for i in range(0, len(gene_columns), batch_size):
        batch_genes = gene_columns[i:i + batch_size]
        # 构建 UNION ALL 查询
        union_query = " UNION ALL ".join(batch_queries)
        batch_results = conn.execute(union_query).fetchall()
        conn.executemany(f"UPDATE var SET {add_key1} = ?, {add_key2} = ? WHERE gene_id = ?", update_data)
        conn.commit()
```

---

## 3. 转换函数（Transformation）

### 3.1 normalize_total_scale_factor（_transformation.py:137-220）

```python
def normalize_total_scale_factor(atlas: Atlas,
                                 target_sum: Optional[float] = 10000,
                                 add_key: str = "scale_factor",
                                 select_data: str = "data") -> None:
    """
    高性能 normalize_total（Scanpy 等价）：
    - 不修改 X_CSR_data
    - 只计算每个 cell 的 scale_factor
    """
    conn = atlas.connection

    # 1. 计算每个 cell 的 total
    conn.execute(f"""
        CREATE OR REPLACE TEMP TABLE _cell_sum AS
        SELECT cell_index, SUM({select_data}) AS total
        FROM X_CSR_data GROUP BY cell_index
    """)

    # 2. target_sum = median(total)
    if target_sum is None:
        target_sum = conn.execute("SELECT median(total) FROM _cell_sum").fetchone()[0]

    # 3. 在 obs 中写 scale_factor
    conn.execute(f"ALTER TABLE obs ADD COLUMN IF NOT EXISTS {add_key} REAL")
    conn.execute(f"""
        UPDATE obs SET {add_key} = CASE WHEN s.total > 0 THEN {float(target_sum)} / s.total ELSE 0 END
        FROM _cell_sum AS s WHERE obs.id = s.cell_index
    """)
```

### 3.2 log1p_chunked（_transformation.py:226-315）

```python
def log1p_chunked(atlas: 'Atlas',
                  base: Optional[Number] = None,
                  add_field: str = "log1p_factor",
                  select_data: str = "data",
                  chunk_size: int = 100_000_000) -> None:
    """
    1e8 级 CSR 安全的 log1p 实现
    - 按 id 分块 UPDATE
    """
    conn = atlas.connection
    col_exists = conn.execute(f"""
        SELECT COUNT(*)
        FROM information_schema.columns
        WHERE table_name = 'X_CSR_data'
          AND column_name = '{select_data}'
    """).fetchone()[0]

    if col_exists == 0:
        raise ValueError(f"X_CSR_data 中不存在字段: {select_data}")
    # 构造 log 表达式
    if base is None:
        log_expr = f"ln(1.0 + {select_data})"
    else:
        log_expr = f"log({float(base)}, 1.0 + {select_data})"

    # 确保输出字段存在
    conn.execute(f"ALTER TABLE X_CSR_data ADD COLUMN IF NOT EXISTS {add_field} REAL")
    min_id, max_id = conn.execute("""
        SELECT MIN(id), MAX(id)
        FROM X_CSR_data
    """).fetchone()

    if min_id is None:
        print("X_CSR_data 为空，跳过")
        return

    n_chunks = math.ceil((max_id - min_id + 1) / chunk_size)
    print(f"共 {n_chunks} 个 chunk")
    # 分块 UPDATE
    for i in range(n_chunks):
        start_id = min_id + i * chunk_size
        end_id = start_id + chunk_size - 1
        conn.execute(f"""
            UPDATE X_CSR_data SET {add_field} = {log_expr}
            WHERE id BETWEEN {start_id} AND {end_id} AND {select_data} IS NOT NULL
        """)
```

### 3.3 scale_gene_chunked（_transformation.py:648-801）

```python
def scale_gene_chunked(atlas, select_data: str = "data",
                       add_field: str = "X_scale",
                       max_value: float = 10.0,
                       gene_chunk_size: int = 512,
                       use_hvg: bool = False,
                       hvg_key: str = "highly_variable") -> None:
    """
    Gene-wise z-score scale（DuckDB OLAP 优化版，1e8 cell 安全）
    """
    conn = atlas.connection

    # 添加输出字段
    conn.execute(f"ALTER TABLE X_CSR_data ADD COLUMN IF NOT EXISTS {add_field} REAL")

    # 获取 gene 列表
    if use_hvg:
        gene_ids = conn.execute(f"SELECT id FROM var WHERE {hvg_key} = TRUE ORDER BY id").fetchall()
    else:
        gene_ids = conn.execute("SELECT DISTINCT indices FROM X_CSR_data ORDER BY indices").fetchall()

    # 分批处理基因
    for i in range(n_chunks):
        chunk_genes = gene_ids[chunk_start: chunk_start + gene_chunk_size]
        gene_list_sql = ",".join(map(str, chunk_genes))

        # 5.1 gene-wise stat
        conn.execute(f"""
            CREATE OR REPLACE TEMP TABLE _gene_stat AS
            SELECT indices, AVG({select_data}) AS mean, STDDEV_POP({select_data}) AS std
            FROM X_CSR_data WHERE indices IN ({gene_list_sql}) AND {select_data} IS NOT NULL
            GROUP BY indices
        """)

        # 5.2 scale 并插入临时表
        conn.execute(f"""
            INSERT INTO _X_scale_tmp
            SELECT x.id, x.indices,
                CASE WHEN g.std > 0 THEN LEAST({float(max_value)}, GREATEST(-{float(max_value)},
                (x.{select_data} - g.mean) / g.std)) ELSE 0 END
            FROM X_CSR_data x JOIN _gene_stat g ON x.indices = g.indices
            WHERE x.indices IN ({gene_list_sql}) AND x.{select_data} IS NOT NULL
        """)

    # 6. merge 回 X_CSR_data
    conn.execute(f"""
        UPDATE X_CSR_data x SET {add_field} = t.val
        FROM _X_scale_tmp t WHERE x.id = t.id AND x.indices = t.indices
    """)
```

### 3.4 highly_variable_genes（_transformation.py:518-644）

```python
def highly_variable_genes(atlas: Atlas,
                         flavor: Literal["var", "cv"],
                         n_top_genes: int | None,
                         add_key: str = "highly_variable_genes",
                         select_data: str = "data") -> None:
    """
    类似 sc.pp.highly_variable_genes（简化版）
    - 在 var 表中新建布尔字段 add_key
    """
    conn = atlas.connection

    # 1. 计算每个 gene 的 mean / var / std
    conn.execute(f"""
        CREATE OR REPLACE TEMP TABLE _gene_stats AS
        SELECT indices AS gene_index, COUNT(*) AS n, AVG({select_data}) AS mean,
               VAR_POP({select_data}) AS var, STDDEV_POP({select_data}) AS std
        FROM X_CSR_data WHERE {select_data} IS NOT NULL GROUP BY indices
    """)

    # 2. 计算排序指标
    if flavor == "var":
        score_expr = "var"
    elif flavor == "cv":
        score_expr = "CASE WHEN mean > 0 THEN std / mean ELSE 0 END"

    conn.execute(f"CREATE OR REPLACE TEMP TABLE _gene_score AS SELECT gene_index, {score_expr} AS score FROM _gene_stats")

    # 3. 选 top genes
    if n_top_genes is not None:
        conn.execute(f"CREATE OR REPLACE TEMP TABLE _hvg AS SELECT gene_index FROM _gene_score ORDER BY score DESC LIMIT {int(n_top_genes)}")
    else:
        conn.execute("CREATE OR REPLACE TEMP TABLE _hvg AS SELECT gene_index FROM _gene_score")

    # 4. 在 var 表中写入布尔结果
    conn.execute(f"ALTER TABLE var ADD COLUMN IF NOT EXISTS {add_key} BOOLEAN")
    conn.execute(f"UPDATE var SET {add_key} = FALSE")
    conn.execute(f"UPDATE var SET {add_key} = TRUE FROM _hvg WHERE var.id = _hvg.gene_index")
```

### 3.5 normalize_and_log1p（_transformation.py:413-512）

```python
def normalize_and_log1p(atlas: Atlas,
                        target_sum: Optional[float] = 10000,
                        scale_key: str = "scale_factor",
                        add_field: str = "X_log1p",
                        select_data: str = "data",
                        chunk_size: int = 100_000_000) -> None:
    """
    Scanpy 等价的 normalize_total + log1p
    """
    # Step 1: 计算 scale_factor
    normalize_total_scale_factor(atlas=atlas, target_sum=target_sum, add_key=scale_key)

    # Step 2: log(1 + data * scale_factor)
    if base is None:
        log_expr = f"ln(1.0 + x.{select_data} * o.{scale_key})"
    else:
        log_expr = f"log({float(base)}, 1.0 + x.{select_data} * o.{scale_key})"

    conn.execute(f"ALTER TABLE X_CSR_data ADD COLUMN IF NOT EXISTS {add_field} REAL")
    min_id, max_id = conn.execute("""
        SELECT MIN(id), MAX(id)
        FROM X_CSR_data
    """).fetchone()

    if min_id is None:
        print("X_CSR_data 为空，跳过")
        return

    n_chunks = math.ceil((max_id - min_id + 1) / chunk_size)
    print(f"log1p 分 {n_chunks} 个 id chunk")
    for i in range(n_chunks):
        conn.execute(f"""
            UPDATE X_CSR_data AS x SET {add_field} = {log_expr}
            FROM obs AS o WHERE x.cell_index = o.id
              AND x.id BETWEEN {start_id} AND {end_id}
        """)
```

---

## 4. 输入函数（Input/IO）

### 4.1 load_AnnData（input.py:1169-1252）

```python
def load_AnnData(adata: AnnData, atlas: Atlas) -> None:
    """
    将anndata数据存入数据库中
    """
    atlas.connection = atlas.connect("r+")

    if hasattr(adata, 'obs'):
        add_obs(adata, atlas)      # 细胞表数据（对应obs）

    if hasattr(adata, 'var'):
        add_var(adata, atlas)      # 基因表数据（对应var）

    if hasattr(adata, 'X'):
        add_X_as_chunk_CSR_only(adata, atlas, chunk_size=4096)  # 分块导入CSR数据

    if hasattr(adata, 'obsm'):
        add_obsm(adata, atlas)

    if hasattr(adata, 'varm'):
        add_varm(adata, atlas)

    if hasattr(adata, 'obsp'):
        add_obsp(adata, atlas)

    if hasattr(adata, 'uns'):
        add_uns(adata, atlas)
```

### 4.2 add_obs（input.py:1286-1303）

```python
def add_obs(adata: AnnData, atlas: Atlas) -> None:
    """
    导入 obs 数据
    """
    obs_df = adata.obs.copy()
    obs_df['cell_id'] = adata.obs.index
    obs_df['id'] = range(len(obs_df))
    obs_df = obs_df[['id', 'cell_id'] + [col for col in obs_df.columns if col not in ['id', 'cell_id']]]

    atlas.connection.register('obs_df', obs_df)
    atlas.connection.execute("CREATE OR REPLACE TABLE obs AS SELECT * FROM obs_df")
    atlas.connection.execute("ALTER TABLE obs ADD PRIMARY KEY (id)")
    atlas.connection.unregister('obs_df')
```

### 4.3 add_var（input.py:1305-1318）

```python
def add_var(adata: AnnData, atlas: Atlas) -> None:
    """
    导入 var 数据
    """
    var_df = adata.var.reset_index().rename(columns={'index': 'gene_id'})
    var_df['id'] = range(len(var_df))
    var_df = var_df[['id', 'gene_id'] + [col for col in var_df.columns if col not in ['id', 'gene_id']]]

    atlas.connection.register('var_df', var_df)
    atlas.connection.execute("CREATE OR REPLACE TABLE var AS SELECT * FROM var_df")
    atlas.connection.execute("ALTER TABLE var ADD PRIMARY KEY (id)")
    atlas.connection.unregister('var_df')
```

### 4.4 add_X_as_chunk_CSR_only（input.py:1707-1827）

```python
def add_X_as_chunk_CSR_only(adata: AnnData, atlas: Atlas, chunk_size: int = 2048) -> None:
    """
    仅导入 CSR 结构：
      - X_CSR_indptr(id, cell_id, indptr)
      - X_CSR_data(id, indices, data, cell_index)
    """
    conn.execute("""
        CREATE OR REPLACE TABLE X_CSR_indptr (
            id BIGINT PRIMARY KEY, cell_id VARCHAR, indptr BIGINT
        )
    """)
    conn.execute("""
        CREATE OR REPLACE TABLE X_CSR_data (
            id BIGINT PRIMARY KEY, indices USMALLINT, data REAL, cell_index BIGINT
        )
    """)

    for chunk_idx in tqdm(range(total_chunks), desc="导入 CSR chunks"):
        # 获取数据块
        X_chunk = adata.X[start:end]
        csr = X_chunk.tocsr() if not sparse.issparse(X_chunk) else X_chunk

        # indptr 表
        adj_indptr = csr.indptr[1:] + global_indptr_offset
        indptr_df = pd.DataFrame({"id": np.arange(start, end), "cell_id": cell_ids[start:end], "indptr": adj_indptr})
        conn.execute("INSERT INTO X_CSR_indptr SELECT * FROM indptr_df")

        # data 表
        if len(data) > 0:
            row_lengths = np.diff(csr.indptr)
            cell_index = np.repeat(np.arange(start, end), row_lengths)
            data_df = pd.DataFrame({"id": data_ids, "indices": csr.indices, "data": csr.data, "cell_index": cell_index})
            conn.execute("INSERT INTO X_CSR_data SELECT * FROM data_df")
```

### 4.5 load_AnnData_chunk（input.py:829-924）

```python
def load_AnnData_chunk(h5ad_path: str, atlas: Atlas, batch_size: int = 4096) -> None:
    """
    超大 h5ad → DuckDB（chunk 模式）
    """
    adata_backed = sc.read_h5ad(h5ad_path, backed="r")  # lazy 加载
    atlas.connection = atlas.connect("r+")

    # 1. 一次性导入 var / varm / uns
    add_var(adata_backed, atlas)
    if hasattr(adata_backed, 'varm'):
        add_varm(adata_backed, atlas)
    if hasattr(adata_backed, 'uns'):
        add_uns_safe(adata_backed, atlas)

    # 2. 初始化 CSR 表
    conn.execute("""
        CREATE OR REPLACE TABLE X_CSR_indptr (id BIGINT PRIMARY KEY, cell_id VARCHAR, indptr BIGINT)
    """)
    conn.execute("""
        CREATE OR REPLACE TABLE X_CSR_data (id BIGINT PRIMARY KEY, indices USMALLINT, data REAL, cell_index BIGINT)
    """)

    # 3. batch 读取
    for i, adata_chunk in enumerate(h5ad_reader(h5ad_path, batch_size)):
        add_obs_chunk(adata_chunk, atlas, cell_offset=global_cell_offset)
        add_X_CSR_chunk_append(adata_chunk, atlas, cell_offset=global_cell_offset, ...)
        add_obsm_chunk(adata_chunk, atlas, cell_offset=global_cell_offset)
        global_cell_offset += adata_chunk.n_obs
```

---

## 5. 输出函数（Output/IO）

### 5.1 save_h5ad（output.py:10-13）

```python
def save_h5ad(file: str, atlas: Atlas) -> None:
    """
    保存数据为 h5ad 格式文件
    """
    atlas.save_h5ad()
```

### 5.2 create_anndata_from_tables（input.py:32-95）

```python
def create_anndata_from_tables(con, output_path="output_data.h5ad") -> AnnData:
    """
    从三张表创建 AnnData 对象并保存为 h5ad 文件
    """
    # 1. 读取 cells 表 (obs数据)
    cells_df = con.execute("SELECT * FROM cells").fetchdf()
    cells_df = cells_df.set_index('cell_id')

    # 2. 读取 genes 表 (var数据)
    genes_df = con.execute("SELECT * FROM genes").fetchdf()
    genes_df = genes_df.set_index('gene_id')

    # 3. 读取 expression 表 (X矩阵)
    expression_df = con.execute("SELECT * FROM expression").fetchdf()

    # 4. 创建稀疏矩阵
    cell_to_idx = {cell_id: idx for idx, cell_id in enumerate(cells_df.index)}
    gene_to_idx = {gene_id: idx for idx, gene_id in enumerate(genes_df.index)}
    row_indices = [cell_to_idx[cell_id] for cell_id in expression_df['cell_id']]
    col_indices = [gene_to_idx[gene_id] for gene_id in expression_df['gene_id']]
    X_sparse = sparse.csr_matrix((data_values, (row_indices, col_indices)), shape=(len(cells_df), len(genes_df)))

    # 5. 创建 AnnData 对象
    adata = anndata.AnnData(X=X_sparse, obs=cells_df, var=genes_df)

    # 6. 保存为 h5ad 文件
    adata.write_h5ad(output_path)
    return adata
```

---

## 函数返回类型汇总

| 函数类别 | 函数名 | 返回类型 | 说明 |
|---------|--------|---------|------|
| **核心类** | `Atlas.__init__` | None | 初始化，创建/连接数据库 |
| **核心类** | `Atlas.connect` | Connection | 返回数据库连接 |
| **核心类** | `Atlas.__getitem__` | Atlas | 返回新的视图对象 |
| **核心类** | `Atlas.query_minibatch` | Generator | 返回 AnnData 生成器 |
| **质控** | `filter_cells_CSR_ultrafast` | None | 原地修改数据库 |
| **质控** | `filter_genes_CSR` | None | 原地修改数据库 |
| **质控** | `calculate_qc_metrics` | None | 原地修改数据库 |
| **质控** | `calculate_cell_total_counts` | None | 原地修改数据库 |
| **质控** | `calculate_gene_total_counts` | None | 原地修改数据库 |
| **转换** | `normalize_total_scale_factor` | None | 原地修改数据库 |
| **转换** | `log1p_chunked` | None | 原地修改数据库 |
| **转换** | `scale_gene_chunked` | None | 原地修改数据库 |
| **转换** | `highly_variable_genes` | None | 原地修改数据库 |
| **转换** | `normalize_and_log1p` | None | 原地修改数据库 |
| **输入** | `load_AnnData` | None | 原地修改数据库 |
| **输入** | `add_obs` | None | 原地修改数据库 |
| **输入** | `add_var` | None | 原地修改数据库 |
| **输入** | `add_X_as_chunk_CSR_only` | None | 原地修改数据库 |
| **输入** | `load_AnnData_chunk` | None | 原地修改数据库 |

---

*文档版本: 1.0*
*最后更新: 2026-01-14*
