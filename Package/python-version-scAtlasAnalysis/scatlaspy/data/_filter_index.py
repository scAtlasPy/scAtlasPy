import duckdb
from tqdm import tqdm
from datetime import datetime

class FilterBuildIndex:

    def __init__(
        self,
        file_path: str,
        *, # file_path 可以按位置传， * 后面的参数必须写参数名
        cell_condition: str | None = None,
        gene_condition: str | None = None,
        use_hvg: bool = True,
        select_data: str = "data_scale",
    ):
        self.file_path = file_path       # sasql 文件绝对路径
        self.producer_num = 10           # minibatch 流式读取的线程数
        self.fetch_size = 1_0000_0000    # minibatch 流式读取的size
        self.chunk_size = 2_0000_0000    # 每次处理的数据量

        self.cell_condition = cell_condition # cell 过滤条件 filter_cells 表示只选 filter_cells = True 的cell
        self.gene_condition = gene_condition # gene 过滤条件 filter_genes 表示只选 filter_genes = True 的gene
        self.use_hvg = use_hvg               # 是否 使用hvg基因
        self.select_data = select_data       # 选择什么数据进行处理

        self.conn = duckdb.connect(file_path)
        self.conn.execute("PRAGMA preserve_insertion_order=true")
        #  false 不强制保留插入时的输入顺序，允许 DuckDB 为了性能重新组织执行和写入顺序。
        #  true  尽量保留 INSERT / COPY / SELECT 写入时的输入顺序


    # 外部入口
    def run(self):

        print("开始流程")

        start = datetime.now()

        self._rebuild_obs_filter_id()   # 重排 obs： 过滤细胞 + 生成 filter_cell_id
        self._rebuild_var_filter_id()   # 重排 var： 过滤基因 + 选择HVG基因 + 生成 filter_gene_id

        self._rebuild_X_HyS_data_filtered()

        self._rebuild_X_HyS_indptr_filtered()

        self.conn.close()

        print(" filter_build_index ，耗时: {:.2f} 秒".format(
            (datetime.now() - start).total_seconds()
        ))

        print("全流程完成")


    # 1.重排 obs： 过滤细胞 + 生成 filter_cell_id
    def _rebuild_obs_filter_id(self):

        print("Step 1. 重排 obs： 过滤细胞 + 生成 filter_cell_id ")

        # 删除旧列
        self.conn.execute(""" ALTER TABLE obs DROP COLUMN IF EXISTS filter_cell_id """)

        # 新增列
        self.conn.execute(""" ALTER TABLE obs ADD COLUMN filter_cell_id INTEGER """)

        # 如果 self.cell_condition is None，则不过滤 cell
        if self.cell_condition is None:
            where_sql = "TRUE"
            print("  -> 不使用 cell 过滤，保留全部 cells")
        else:
            where_sql = f"{self.cell_condition}=TRUE"
            print(f"  -> 使用 cell 条件: {where_sql} 过滤")

        # 只对满足条件的 cell 重新编号
        self.conn.execute(f"""
        UPDATE obs
        SET filter_cell_id = sub.new_id
        FROM (
            SELECT
                atlas_cell_id,
                ROW_NUMBER() OVER (ORDER BY atlas_cell_id) - 1 AS new_id
            FROM obs
            WHERE {where_sql}
        ) AS sub
        WHERE obs.atlas_cell_id = sub.atlas_cell_id
        """)

        print(" obs 重排完成")


    # 2.重排 var： 过滤基因 + 选择HVG基因 + 生成 filter_gene_id
    def _rebuild_var_filter_id(self):

        print("Step 2. 重排 var： 过滤基因 + 选择HVG基因 + 生成 filter_gene_id")

        # 删除旧列 + 新增列
        self.conn.execute(""" ALTER TABLE var DROP COLUMN IF EXISTS filter_gene_id """)

        self.conn.execute(""" ALTER TABLE var ADD COLUMN filter_gene_id USMALLINT """)

        # 如果 self.gene_condition is None，则不过滤 gene
        conditions = []

        if self.gene_condition is not None:
            conditions.append(f"({self.gene_condition})=TRUE")
            print(f"  -> 使用 gene 条件: {self.gene_condition}=TRUE")
        else:
            print("  -> 不使用 gene 过滤条件")

        # 如果启用 HVG，则叠加 highly_variable_genes
        if self.use_hvg:
            conditions.append("highly_variable_genes=TRUE")
            print("  -> 使用 HVG gene 子集")
        else:
            print("  -> 不使用 HVG 过滤，保留全部 genes")

        condition = " AND ".join(conditions) if conditions else "TRUE"

        # 重排 gene_id
        self.conn.execute(f"""
        UPDATE var
        SET filter_gene_id = sub.new_id
        FROM (
            SELECT
                atlas_gene_id,
                ROW_NUMBER() OVER (ORDER BY atlas_gene_id) - 1 AS new_id
            FROM var
            WHERE {condition}
        ) AS sub
        WHERE var.atlas_gene_id = sub.atlas_gene_id
        """)

        print(f" var 重排完成（use_hvg={self.use_hvg}）")


    # 3.重建新表：X_HyS_data_filtered
    def _rebuild_X_HyS_data_filtered(self):

        print("Step 3. 重建新表：X_HyS_data_filtered")

        conn = self.conn

        conn.execute("PRAGMA preserve_insertion_order = true")
        conn.execute("PRAGMA threads=10")

        # 提前建 轻量映射表
        conn.execute("""
        CREATE OR REPLACE TEMP TABLE _obs_keep AS
        SELECT
            atlas_cell_id,
            CAST(filter_cell_id AS INTEGER) AS filter_cell_id
        FROM obs
        WHERE filter_cell_id IS NOT NULL
        """)

        conn.execute("""
        CREATE OR REPLACE TEMP TABLE _var_keep AS
        SELECT
            atlas_gene_id,
            CAST(filter_gene_id AS USMALLINT) AS filter_gene_id
        FROM var
        WHERE filter_gene_id IS NOT NULL
        """)

        # ANALYZE 是给 DuckDB 优化器看的统计信息，不改变数据、不建索引，只是可能让后面的 JOIN 更快更稳
        try:
            conn.execute("ANALYZE _obs_keep")
            conn.execute("ANALYZE _var_keep")
        except Exception:
            pass

        # 创建目标表
        conn.execute("DROP TABLE IF EXISTS X_HyS_data_filtered")

        conn.execute("""
        CREATE TABLE X_HyS_data_filtered (
            filter_cell_id INTEGER,
            filter_gene_id USMALLINT,
            data REAL,
            tid TINYINT
        )
        """)

        # 获取 rowid 范围
        min_id, max_id = conn.execute("""
            SELECT MIN(rowid), MAX(rowid)
            FROM X_HyS_data
        """).fetchone()

        if min_id is None:
            print(" X_HyS_data 是空表，跳过")
            return

        total_rows = max_id - min_id + 1
        current = min_id

        print(f"rowid 范围: {min_id:,} ~ {max_id:,}")
        print(f"总扫描行数: {total_rows:,}")
        print(f"chunk_size = {self.chunk_size:,}")

        pbar = tqdm(
            total=total_rows,
            unit="rows",
            desc="Processing",
            ncols=130
        )

        # 分块顺序插入
        while current <= max_id:
            end = min(current + self.chunk_size, max_id + 1)

            conn.execute(f"""
            INSERT INTO X_HyS_data_filtered
            SELECT
                obs.filter_cell_id,
                var.filter_gene_id,
                CAST(X.{self.select_data} AS REAL) AS data,
                CAST(0 AS TINYINT) AS tid
            FROM X_HyS_data AS X
            JOIN _obs_keep AS obs
              ON X.atlas_cell_id = obs.atlas_cell_id
            JOIN _var_keep AS var
              ON X.atlas_gene_id = var.atlas_gene_id
            WHERE X.rowid >= {current}
              AND X.rowid < {end}
            ORDER BY X.rowid
            """)

            pbar.update(end - current)
            current = end

        pbar.close()

        print(" 计算 tid ...")

        # 计算 tid
        conn.execute(f"""
            UPDATE X_HyS_data_filtered
            SET tid = CAST((rowid // {self.fetch_size}) % {self.producer_num} AS TINYINT)
        """)

        final_nnz = conn.execute("""
            SELECT COUNT(*)
            FROM X_HyS_data_filtered
        """).fetchone()[0]

        print(f" X_HyS_data_filtered 构建完成！nnz = {final_nnz:,}")

        # 清理临时表
        conn.execute("DROP TABLE IF EXISTS _obs_keep")
        conn.execute("DROP TABLE IF EXISTS _var_keep")


    # 4.重建新表：X_HyS_indptr_filtered
    def _rebuild_X_HyS_indptr_filtered(self):
        """
        重建 CSR indptr

        关键点：
        1. X_HyS_data_filtered 只存非零值，不写入 0
        2. indptr 从 obs.filter_cell_id 出发补齐所有保留下来的 cell
        3. 没有非零值的 cell，cnt = 0
        4. indptr 仍然表示每个 cell 的结束位置 end_ptr

        修复bug X_HyS_indptr_filtered 不能只记录 X 中出现过的 cell，

          而要以 obs.filter_cell_id 为准，把所有保留下来的 cell 都记录进去；
          没有非零值的 cell，cnt = 0，所以它的 indptr 和上一个 cell 一样。
          filter_cell_id    indptr
           0                 2
           1                 2
           2                 4
          cell1: start == end , 说明这个 cell 没有任何非零值。 但依旧要保留

        """

        print(" 4.重建新表：X_HyS_indptr_filtered ")

        conn = self.conn

        # 1. 先统计 X 表中每个 cell 的非零元素数量
        conn.execute("""
        CREATE OR REPLACE TEMP TABLE cell_nnz_raw AS
        SELECT
            CAST(filter_cell_id AS INTEGER) AS filter_cell_id,
            COUNT(*) AS cnt
        FROM X_HyS_data_filtered
        GROUP BY filter_cell_id
        """)

        # 2. 从 obs 出发，补齐所有保留下来的 cell
        #    如果某个 cell 在 X 里不存在，说明它没有非零值，cnt 补 0
        conn.execute("""
        CREATE OR REPLACE TEMP TABLE cell_nnz AS
        SELECT
            CAST(obs.filter_cell_id AS INTEGER) AS filter_cell_id,
            CAST(COALESCE(cell_nnz_raw.cnt, 0) AS BIGINT) AS cnt
        FROM obs
        LEFT JOIN cell_nnz_raw
          ON obs.filter_cell_id = cell_nnz_raw.filter_cell_id
        WHERE obs.filter_cell_id IS NOT NULL
        ORDER BY obs.filter_cell_id
        """)

        print(" cell_nnz 补齐完成")

        # 3. 生成 prefix sum ；这里 indptr 存的是每个 cell 的结束位置 end_ptr
        conn.execute("""
        CREATE OR REPLACE TABLE X_HyS_indptr_filtered AS
        SELECT
            CAST(filter_cell_id AS INTEGER) AS filter_cell_id,
            CAST(SUM(cnt) OVER (ORDER BY filter_cell_id) AS BIGINT) AS indptr
        FROM cell_nnz
        ORDER BY filter_cell_id
        """)

        conn.execute("DROP TABLE IF EXISTS cell_nnz_raw")
        conn.execute("DROP TABLE IF EXISTS cell_nnz")

        print(" X_HyS_indptr_filtered 构建完成（已从 obs 补齐空 cell）")
