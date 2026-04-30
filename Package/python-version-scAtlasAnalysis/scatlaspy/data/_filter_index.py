import duckdb
from tqdm import tqdm
from datetime import datetime

# todo 不重排 id 带来的问题：
#  原始 shape = 6 * 5
#  不重排 ，shape 还是 6 * 5, 但是已经过滤掉了一些 cell和 gene， 于是就产生了以下问题；
#  会有空的 cell 和 gene ；
#  [1, 0, 2, 0]
#  [NaN, NaN, NaN, NaN]   ← 存在这种“空cell”
#  [4, 0, 5, 6]

# todo 重排
#  原始 shape = 6 * 5
#  新的实际的 shape = 4 * 3 ， 不会产生 空 cell 和 gene

# todo obs（细胞表）   ---->  过滤 ， 在原表上新建 filter_cell_id 列 ,重排一下cell_id
# atlas_cell_id	 atlas_cell_name  filter_cells  filter_cell_id（ 🔥新增 ）
#   0	           c0               TRUE          0
#   1        	   c1               FALSE
#   2	           c2               TRUE          1
#   3	           c3               FALSE
#   4	           c4               TRUE          2
#   5	           c5               TRUE          3

# todo var（基因表）     ---->  过滤 ， 在原表上新建 filter_gene_id 列 ,重排一下gene_id
# atlas_gene_id	 atlas_gene_name  filter_genes  filter_gene_id（ 🔥新增 ）
#   0	           g0               TRUE          0
#   1        	   g1               FALSE
#   2	           g2               TRUE          1
#   3	           g3               FALSE
#   4	           g4               TRUE          2


# todo X_CSRO_data 表  ---->  筛选符合要求的cell 和 gene ，根据obs 和 var表中的信息，重新排序id;
#  (1)重排 filter_cell_id，filter_gene_id，和 new_id ；
#  (2)添加 tid =  (id // fetch_size) % producer_num  (取余)

#  id	 atlas_cell_id   atlas_gene_id  data  是否过滤   new_id filter_cell_id filter_gene_id  tid
#   0	        0             0         1.0         ✅     0      0             0             0
#   1        	0             2         2.0         ✅     1      0             1             0
#   2	        1             0         2.0     X
#   3	        1             1         2.0     X
#   4	        2             1         2.0     X
#   5	        2             2         2.0         ✅     2      1             1             1
#   6	        3             3         2.0     X
#   7	        4             4         2.0         ✅     3      2             2             1
#   8	        5             0         2.0         ✅     4      3             0             0
#   9	        5             0         2.0         ✅     5      3             0             0

# 若
# fetch_size = 2
# producer_num = 2

# id // fetch_size ( 整除 )
# 👉 把数据按块分组（chunk）
# 每 fetch_size 条数据算一组
# 原值 id                             = 0,1,2,3,4,5  //  2
#     id // fetch_size               = 0,0,1,1,2,2

# tid =  (id // fetch_size) % producer_num  (取余)
# tid ： 把这些块轮流分配给对应的 producer
# 原值 id                             = 0,1,2,3,4,5  //  2
#     id // fetch_size               = 0,0,1,1,2,2
# (id // fetch_size) % producer_num  = 0,0,1,1,0,0

'''过滤 obs / var + 建新表 X_CSRO_data_filtered  + 建立索引   '''
class FilterBuildIndex:
    """
    DuckDB → 过滤 → 重排 → CSR 重建 pipeline
    功能：
        1. 过滤 obs / var
        2. 生成 filter_cell_id / filter_gene_id
        3. 重建 X_CSRO_data_filtered（带 tid 分片）
        4. 重建 X_CSRO_indptr_filtered
    """

    def __init__(
        self,
        file_path: str,
        *,
        fetch_size: int = 1_0000_0000,
        producer_num: int = 10,
        chunk_size: int = 5_0000_0000,
        cell_condition: str = "filter_cells",
        gene_condition: str = "filter_genes",
        use_hvg: bool = True,
        select_data: str = "data_scale",  # 🔥 新增
    ):
        self.file_path = file_path       # sasql 文件绝对路径
        self.fetch_size = fetch_size     # minibatch 流式读取的size
        self.producer_num = producer_num # minibatch 流式读取的线程数
        self.chunk_size = chunk_size     # 每次处理的数据量

        self.cell_condition = cell_condition
        self.gene_condition = gene_condition
        self.use_hvg = use_hvg    # 使用hvg基因
        self.select_data = select_data  # 🔥 将 X_CSRO_data.data ， 变成 输入 条件 { select_data = “ data ” }

        self.conn = duckdb.connect(file_path)
        self.conn.execute("PRAGMA preserve_insertion_order=false")


    # 外部入口
    def run(self):
        print("🚀 开始 CSR 过滤构建流程")

        start = datetime.now()

        self._rebuild_obs_filter_id() # todo 补充，不需要过滤的情况 以及 其他过滤条件的情况
        self._rebuild_var_filter_id()
        self._rebuild_X_chunked()
        self._rebuild_indptr()

        self.conn.close()

        print(" filter_build_index ，耗时: {:.2f} 秒".format(
            (datetime.now() - start).total_seconds()
        ))

        print("🎉 全流程完成（已得到真正 CSR 结构）")


    # Step 1：重排 obs（生成 filter_cell_id）
    def _rebuild_obs_filter_id(self):
        print("Step 1：重排 obs（生成 filter_cell_id）")

        # 删除旧列
        self.conn.execute("""
        ALTER TABLE obs DROP COLUMN IF EXISTS filter_cell_id
        """)

        # 新增列
        self.conn.execute("""
        ALTER TABLE obs ADD COLUMN filter_cell_id INTEGER
        """)

        # 只对满足过滤条件的 cell 重新编号（从0开始）
        self.conn.execute(f"""
        UPDATE obs
        SET filter_cell_id = sub.new_id
        FROM (
            SELECT
                atlas_cell_id,
                ROW_NUMBER() OVER (ORDER BY atlas_cell_id) - 1 AS new_id  -- 从0开始,重排cell_id
            FROM obs
            WHERE {self.cell_condition}=TRUE
        ) AS sub
        WHERE obs.atlas_cell_id = sub.atlas_cell_id
        """)

        print("✅ obs 重排完成")


    # Step 2：重排 var（生成 filter_gene_id） + HVG基因过滤
    def _rebuild_var_filter_id(self):
        print("Step 2：重排 var（生成 filter_gene_id）")

        # -----------------------------
        # 1️⃣ 删除旧列 + 新增列
        # -----------------------------
        self.conn.execute("""
        ALTER TABLE var DROP COLUMN IF EXISTS filter_gene_id
        """)

        self.conn.execute("""
        ALTER TABLE var ADD COLUMN filter_gene_id INTEGER
        """)

        # -----------------------------
        # 2️⃣ 构建过滤条件（🔥核心）
        # -----------------------------
        # 基础过滤条件（比如 filter_genes）
        condition = f"({self.gene_condition})=TRUE"

        # 如果启用 HVG，则叠加条件
        if self.use_hvg:
            condition += " AND highly_variable_genes=TRUE"

        # -----------------------------
        # 3️⃣ 重排 gene_id（只对符合条件的基因）
        # -----------------------------
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

        print(f"✅ var 重排完成（use_hvg={self.use_hvg}）")

    # Step 3：重建 X_CSRO_data（核心逻辑）--> 直接生成新表：X_CSRO_data_filtered
    def _rebuild_X_chunked(self):

        print("Step 3：重建 X_CSRO_data_filtered（去掉 new_id，tid 均匀分片）")

        # 删除旧表
        self.conn.execute("DROP TABLE IF EXISTS X_CSRO_data_filtered")

        # 创建新表（不存 new_id）
        self.conn.execute("""
        CREATE TABLE X_CSRO_data_filtered (
            filter_cell_id INTEGER,    -- 新 cell id
            filter_gene_id USMALLINT,  -- 新 gene id
            data REAL,                 -- float32
            tid TINYINT                -- 分片 id   可改 TINYINT	INT1	有符号的一字节整数 -128 ~ 127
        )
        """)

        # 获取 rowid 范围
        min_id, max_id = self.conn.execute("SELECT MIN(rowid), MAX(rowid) FROM X_CSRO_data").fetchone()
        total_rows = max_id - min_id + 1
        print(f"rowid 范围: {min_id} ~ {max_id}, 总行数: {total_rows}")

        current = min_id
        # global_offset = 0  # 🔥 保证全局连续行号 # todo 删掉
        pbar = tqdm(total=total_rows, unit="rows", desc="Processing X_CSRO_data", ncols=120)

        while current <= max_id:
            end = min(current + self.chunk_size, max_id + 1)

            # 使用 ROW_NUMBER() + global_offset 临时计算 tid，但不写入表
            # 删掉 ((ROW_NUMBER() OVER () - 1 + {global_offset}) // {self.fetch_size}) % {self.producer_num} AS tid
            self.conn.execute(f"""
            INSERT INTO X_CSRO_data_filtered
            SELECT
                obs.filter_cell_id,
                var.filter_gene_id,
                X_CSRO_data.{self.select_data},   -- 只取需要的列 
                ((X_CSRO_data.rowid - {min_id}) // {self.fetch_size}) % {self.producer_num} AS tid
            FROM X_CSRO_data
            JOIN obs
              ON X_CSRO_data.atlas_cell_id = obs.atlas_cell_id
            JOIN var
              ON X_CSRO_data.atlas_gene_id = var.atlas_gene_id
            WHERE X_CSRO_data.rowid >= {current} AND X_CSRO_data.rowid < {end}
              AND obs.filter_cell_id IS NOT NULL
              AND var.filter_gene_id IS NOT NULL
            """)

            # todo 删掉
            # # 更新 global_offset（累计已插入行数）
            # inserted = self.conn.execute("SELECT COUNT(*) FROM X_CSRO_data_filtered").fetchone()[0]
            # global_offset = inserted

            # 更新进度条
            pbar.update(end - current)

            current = end

        pbar.close()
        print("✅ X_CSRO_data_filtered 构建完成（tid 均匀分片，去掉 new_id，带 tqdm 进度条）")


    # Step 4：重建 CSR indptr
    def _rebuild_indptr(self):
        """
        重建 CSR 矩阵的 indptr（prefix sum）表 X_CSRO_indptr_filtered
        -----------------------------------------------
        输入：
            self.conn: DuckDB 连接对象
        输出：
            创建 / 替换 X_CSRO_indptr_filtered 表
        数据类型：
            - filter_cell_id: INTEGER (4 字节)
            - indptr: BIGINT (8 字节)
        """
        print("Step 4：重建 CSR indptr")

        # -----------------------------
        # 1️⃣ 统计每个 cell 的非零元素个数
        # -----------------------------
        # X_CSRO_data_filtered 每行是 (filter_cell_id, filter_gene_id, data)
        # COUNT(*) → 统计每个 cell 的非零元素数量
        self.conn.execute("""
        CREATE OR REPLACE TEMP TABLE cell_nnz AS
        SELECT
            CAST(filter_cell_id AS INTEGER) AS filter_cell_id,
            COUNT(*) AS cnt
        FROM X_CSRO_data_filtered
        GROUP BY filter_cell_id
        """)
        print("✅ cell_nnz 统计完成")

        # -----------------------------
        # 2️⃣ 生成 prefix sum (CSR indptr)
        # -----------------------------
        # SUM(cnt) OVER (ORDER BY filter_cell_id) → 累加和
        # CAST 为 BIGINT，保证大量非零元素不会溢出
        self.conn.execute("""
        CREATE OR REPLACE TABLE X_CSRO_indptr_filtered AS
        SELECT
            CAST(filter_cell_id AS INTEGER) AS filter_cell_id,
            CAST(SUM(cnt) OVER (ORDER BY filter_cell_id) AS BIGINT) AS indptr
        FROM cell_nnz
        ORDER BY filter_cell_id
        """)
        print("✅ X_CSRO_indptr_filtered 构建完成")

