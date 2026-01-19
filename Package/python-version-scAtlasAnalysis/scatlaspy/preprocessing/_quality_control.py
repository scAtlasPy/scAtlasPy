import math
import os
from datetime import datetime
import duckdb
import numpy as np
from h5py.h5a import delete
from ..data import Atlas
from typing import Optional
import logging
# 获取日志记录器
logger = logging.getLogger('Atlas')


#
# cell_id	gene_A	gene_B	gene_C	总表达值计算
# cell_1	10	     5	       3	10 + 5 + 3 = 18
# cell_2	NULL	 8	       2	0 + 8 + 2 = 10
# cell_3	15	    NULL	  NULL	15 + 0 + 0 = 15

#========== todo 过滤基因 ==========

# 全量过滤 SQL
def filter_cells(atlas: 'Atlas',
                 min_counts: int,
                 min_genes: Optional[int] = None,
                 max_counts: Optional[int] = None,
                 max_genes: Optional[int] = None,
                 add_key: str = "filter_cells_c1") -> None:
    """
    根据参数过滤细胞的离群点，并将过滤条件的布尔向量写入obs表
 该细胞所有基因的表达值之和小于min_counts或大于max_counts

    Args:
        atlas: Atlas对象
            min_counts: 最小总表达值 该细胞所有基因的表达值之和 sum_expr <  min_counts
            min_genes: 最小非零基因数 该细胞的非零表达的基因个数 nonzero_expr < min_genes
            max_counts: 最大总表达值 该细胞所有基因的表达值之和 sum_expr > max_counts
            max_genes: 最大非零基因数 该细胞的非零表达的基因个数 nonzero_expr > max_genes
        add_key: 写入obs表的字段名
    """
    logger.info("开始过滤细胞...")
    start_time = datetime.now()
    # 获取数据库连接
    atlas.connection = atlas.connect("r+")

    # 检查obs表是否已有add_key列，如果没有则添加
    table_info = atlas.connection.execute("PRAGMA table_info(obs)").fetchall()
    column_names = [col[1] for col in table_info]
    if add_key not in column_names:
        atlas.connection.execute(f"ALTER TABLE obs ADD COLUMN {add_key} BOOLEAN")
        logger.info(f"在obs表中添加了新列: {add_key}")

    # 获取所有基因列名 gene_ids
    gene_result = atlas.connection.execute("SELECT gene_id FROM var")
    gene_ids = [row[0] for row in gene_result.fetchall()]  # gene_ids为 符合条件的cell id列表 ， 可替换
    # gene_ids = atlas.var_gene_id
    # cell_ids = atlas.obs_cell_id # cene_ids为 符合条件的cell id列表 ， 可替换

    if not gene_ids:
        logger.warning("未找到任何基因")
        return

    # 方案一：使用for语句 效率很低
    # for cell_id in 所有的细胞:
    #     计算 cell_1 的表达值之和 sum_expr
    #     if sum_expr > min_counts:
    #         在obs表的 add_key 字段标记 cell_1 的值为True

    # 方案二：不使用for语句 效率高
    # 构建计算总表达值的SQL表达式, 例如 COALESCE("gene_A", 0) + COALESCE("gene_B", 0) + COALESCE("gene_C", 0)
    sum_expr = " + ".join([f'COALESCE("{gene}", 0)' for gene in gene_ids])
    # 创建临时视图，计算每个细胞的总表达值
    atlas.connection.execute(f"""
            CREATE OR REPLACE TEMPORARY VIEW cell_temp_view AS
            SELECT 
                cell_id,
                ({sum_expr}) as total_counts
            FROM X
        """)
    # 示例 VIEW cell_temp_view
    # cell_id	total_counts
    # cell_1	18	# 10 + 5 + 3 = 18
    # cell_2	10	# 0 + 8 + 2 = 10 (gene_A 为 NULL，转为 0)
    # cell_3	15	# 15 + 0 + 0 = 15 (gene_B 和 gene_C 为 NULL，转为 0)


    # 计算过滤条件并更新obs表 (total_counts >= min_counts 的细胞保留)
    atlas.connection.execute(f"""
            UPDATE obs
            SET {add_key} = (
                SELECT total_counts >= {min_counts}
                FROM cell_temp_view
                WHERE cell_temp_view.cell_id = obs.cell_id
            )
        """)
    # 更新后的 obs 表:
    # cell_id	other_info	filter_cells
    # cell_1	...	True	# 18 >= 12
    # cell_2	...	False	# 10 < 12
    # cell_3	...	True	# 15 >= 12


    # # 统计过滤结果
    # result = atlas.connection.execute(f"""
    #         SELECT
    #             COUNT(*) as total_cells,
    #             SUM(CASE WHEN {add_key} THEN 1 ELSE 0 END) as kept_cells,
    #             SUM(CASE WHEN NOT {add_key} THEN 1 ELSE 0 END) as filtered_cells
    #         FROM obs
    #     """).fetchone()
    #
    # total_cells, kept_cells, filtered_cells = result
    # # 统计结果:
    # # total_cells = 3
    # # kept_cells = 2 (cell_1 和 cell_3)
    # # filtered_cells = 1 (cell_2)
    #
    # logger.info(f"过滤完成: 总共 {total_cells} 个细胞，保留 {kept_cells} 个，过滤 {filtered_cells} 个")

    # 清理临时视图
    atlas.connection.execute("DROP VIEW IF EXISTS cell_temp_view")

    end_time = datetime.now()
    time_diff = end_time - start_time # 计算耗时

    print(f"全量过滤耗时: {time_diff.total_seconds():.2f} 秒")

# minibatch 过滤 方式一： SQL语句实现
def filter_cells_minibatch(atlas: 'Atlas',
                 min_counts: int,
                 min_genes: Optional[int] = None,
                 max_counts: Optional[int] = None,
                 max_genes: Optional[int] = None,
                 batch_size: Optional[int] = 2048,
                 add_key: str = "filter_cells_c1") -> None:
    """
    根据最小总表达值过滤细胞，并将过滤条件的布尔向量写入obs表

    Args:
        atlas: Atlas对象
        min_counts: 最小总表达值
        add_key: 写入obs表的字段名
        batch_size: 批次大小，默认1000个细胞一批
    """
    logger.info(f"开始过滤细胞，最小总表达值: {min_counts}，批次大小: {batch_size}")

    start_time = datetime.now()
    atlas.connection = atlas.connect("r+")

    # 检查obs表是否已有add_key列，如果没有则添加
    table_info = atlas.connection.execute("PRAGMA table_info(obs)").fetchall()
    column_names = [col[1] for col in table_info]
    if add_key not in column_names:
        atlas.connection.execute(f"ALTER TABLE obs ADD COLUMN {add_key} BOOLEAN")
        logger.info(f"在obs表中添加了新列: {add_key}")

    # 获取所有基因列名
    gene_result = atlas.connection.execute("SELECT gene_id FROM var")
    gene_ids = [row[0] for row in gene_result.fetchall()]

    if not gene_ids:
        logger.warning("未找到任何基因")
        return

    # 构建计算总表达值的SQL表达式
    sum_expr = " + ".join([f'COALESCE("{gene}", 0)' for gene in gene_ids])

    # 获取细胞总数
    count_result = atlas.connection.execute("SELECT COUNT(*) FROM obs")
    total_cells = count_result.fetchone()[0]

    # 分批处理细胞
    kept_cells = 0
    filtered_cells = 0
    processed_cells = 0

    # 计算总批次数
    total_batches = (total_cells + batch_size - 1) // batch_size

    # for adata_minibatch in atlas.query_minibatch():
    #     pass

    for batch_num in range(total_batches):
        # 计算当前批次的偏移量
        offset = batch_num * batch_size

        # 创建临时视图，计算当前批次细胞的总表达值
        atlas.connection.execute(f"""
            CREATE OR REPLACE TEMPORARY VIEW batch_cell_stats AS
            SELECT 
                cell_id,
                ({sum_expr}) as total_counts
            FROM X
            LIMIT {batch_size} OFFSET {offset}
        """)

        # 更新当前批次细胞的过滤标记
        atlas.connection.execute(f"""
            UPDATE obs
            SET {add_key} = (
                SELECT total_counts >= {min_counts}
                FROM batch_cell_stats
                WHERE batch_cell_stats.cell_id = obs.cell_id
            )
            WHERE obs.cell_id IN (
                SELECT cell_id FROM batch_cell_stats
            )
        """)

        # 统计当前批次的结果
        result = atlas.connection.execute(f"""
            SELECT
                SUM(CASE WHEN {add_key} THEN 1 ELSE 0 END) as kept_cells_batch,
                SUM(CASE WHEN NOT {add_key} THEN 1 ELSE 0 END) as filtered_cells_batch
            FROM obs
            WHERE obs.cell_id IN (
                SELECT cell_id FROM batch_cell_stats
            )
        """).fetchone()

        kept_cells_batch, filtered_cells_batch = result
        kept_cells += kept_cells_batch
        filtered_cells += filtered_cells_batch
        processed_cells += batch_size if (offset + batch_size) <= total_cells else (total_cells - offset)

        # 清理临时视图
        atlas.connection.execute("DROP VIEW IF EXISTS batch_cell_stats")

        # 记录进度
        if (batch_num + 1) % 10 == 0 or (batch_num + 1) == total_batches:
            logger.info(
                f"处理进度: {processed_cells}/{total_cells} 细胞 ({((batch_num + 1) / total_batches) * 100:.1f}%)")

    logger.info(f"过滤完成: 总共 {total_cells} 个细胞，保留 {kept_cells} 个，过滤 {filtered_cells} 个")

    end_time = datetime.now()
    time_diff = end_time - start_time  # 计算耗时

    print(f"minibatch 过滤 方式一： SQL语句实现 过滤耗时: {time_diff.total_seconds():.2f} 秒")


#================= todo minibatch 过滤 方式二：yield + anndata + SQL ； 目前最快
def filter_cells_minibatch_yield(atlas: 'Atlas',
                 min_counts: int,
                 min_genes: Optional[int] = None,
                 max_counts: Optional[int] = None,
                 max_genes: Optional[int] = None,
                 batch_size: Optional[int] = 2048,
                 add_key: str = "filter_cells_c1") -> None:
    """
    根据最小总表达值过滤细胞，并将过滤条件的布尔向量写入obs表

    Args:
        atlas: Atlas对象
        min_counts: 最小总表达值 该细胞所有基因的表达值之和 sum_expr >  min_counts
        min_genes: 最小非零基因数 该细胞的非零表达的基因个数 nonzero_expr > min_genes
        max_counts: 最大总表达值 该细胞所有基因的表达值之和 sum_expr < max_counts
        max_genes: 最大非零基因数 该细胞的非零表达的基因个数 nonzero_expr < max_genes
        add_key: 写入obs表的字段名
        batch_size: 批次大小，默认2048个细胞一批
        空细胞：sum_expr <  min_counts的某个值
        双细胞：sum_expr > max_counts的某个值
        死细胞：该细胞的 某些基因（MT,RPS）的 sum_expr > max_counts的某个值
    """
    logger.info(f"开始过滤细胞，最小总表达值: {min_counts}，批次大小: {batch_size}")

    start_time = datetime.now()
    atlas.connection = atlas.connect("r+")

    # 检查obs表是否已有add_key列，如果没有则添加
    table_info = atlas.connection.execute("PRAGMA table_info(obs)").fetchall()
    column_names = [col[1] for col in table_info]
    if add_key not in column_names:
        atlas.connection.execute(f"ALTER TABLE obs ADD COLUMN {add_key} BOOLEAN")
        logger.info(f"在obs表中添加了新列: {add_key}")

    # 获取所有基因列名
    gene_result = atlas.connection.execute("SELECT gene_id FROM var")
    gene_ids = [row[0] for row in gene_result.fetchall()]

    if not gene_ids:
        logger.warning("未找到任何基因")
        return
    logger.info("进入for循环... ")

    batch_num = 0 # 第几批
    true_num = 0 # 符合要求的细胞
    false_num = 0 # 不 符合要求的细胞

    time_1 = 0 # minibatch 读取用时
    time_2 = 0 # 计算用时
    time_3 = 0 # 写表用时

    start_time1 = datetime.now()

    for adata_minibatch in atlas.query_minibatch(batch_size=batch_size):

        end_time1 = datetime.now()
        time_diff1 = end_time1 - start_time1
        time_1 = time_1 + time_diff1.total_seconds()

        print(f"正在计算第 {batch_num} 批")

        start_time2 = datetime.now()

        # # 计算每个细胞（行）的总表达值 , 通过 .flatten() 将结果转换为一维数组
        # sum_expr = np.array(adata_minibatch.X.sum(axis=1)).flatten()  # axis=0 沿纵轴（按列）求和 ;  axis=1 沿横轴（按行）求和

        # todo 直接从 .sum() 方法得到的一维数组，更简洁高效
        sum_expr = adata_minibatch.X.sum(axis=1).A1  # .A1 等价于 .toarray().flatten()

        # todo 2. 计算每个细胞的非零基因数 - 核心优化
        # CSR格式下，getnnz(axis=1) 是最高效的方法，直接读取内部结构
        nonzero_expr = adata_minibatch.X.getnnz(axis=1)

        # # 计算每个细胞的非零基因数
        # nonzero_expr = np.array((adata_minibatch.X > 0).sum(axis=1)).flatten()

        # 初始化过滤掩码为全False
        filtered_cells = np.zeros(len(sum_expr), dtype=bool)

        # # 应用过滤条件
        # # sum_expr < min_counts 生成布尔掩码，标记所有低质量细胞
        # if min_counts is not None:
        #     filtered_cells = filtered_cells | (sum_expr < min_counts)
        #
        # # min_genes: 最小非零基因数 该细胞的非零表达的基因个数 nonzero_expr < min_genes
        # if min_genes is not None:
        #     filtered_cells = filtered_cells | (nonzero_expr < min_genes)
        #
        # # max_counts: 最大总表达值 该细胞所有基因的表达值之和 sum_expr > max_counts
        # if max_counts is not None:
        #     filtered_cells = filtered_cells | (sum_expr > max_counts)
        #
        # # max_genes: 最大非零基因数 该细胞的非零表达的基因个数 nonzero_expr > max_genes
        # if max_genes is not None:
        #     filtered_cells = filtered_cells | (nonzero_expr > max_genes)

        # 4. 应用过滤条件 - 保持原有逻辑，但使用更高效的数组操作
        if min_counts is not None:
            filtered_cells |= (sum_expr < min_counts)  # 使用 |= 原位操作符

        if min_genes is not None:
            filtered_cells |= (nonzero_expr < min_genes)

        if max_counts is not None:
            filtered_cells |= (sum_expr > max_counts)

        if max_genes is not None:
            filtered_cells |= (nonzero_expr > max_genes)

        # 通过 adata.obs['id'] 获取细胞标识列,使用布尔索引筛选满足条件的细胞 ID
        id_list = adata_minibatch.obs['cell_id'][filtered_cells].tolist()

        # 统计过滤细胞数量
        true_num = true_num + len(id_list)
        false_num = false_num + (len(adata_minibatch) - len(id_list))

        end_time2 = datetime.now()
        time_diff2 = end_time2 - start_time2
        time_2 = time_2 + time_diff2.total_seconds()

        start_time3 = datetime.now()
        # 更新数据库标记
        if id_list:
            print(f"正在标记符合要求的 {len(id_list)} 个细胞...")

            # 直接使用数组 语法简单
            atlas.connection.execute(f"""
                    UPDATE obs 
                    SET {add_key} = 1 
                    WHERE cell_id = ANY({id_list})
                """)

            # 使用 VALUES 语法批量更新 更快
            # values_str = ",".join([f"('{id}')" for id in id_list])
            # atlas.connection.execute(f"""
            #         UPDATE obs
            #         SET {add_key} = 1
            #         WHERE cell_id IN (SELECT column1 FROM (VALUES {values_str}) AS t)
            #     """)

            atlas.connection.commit()
            print("过滤完成！")
        else:
            print("没有需要过滤的细胞")

        end_time3 = datetime.now()
        time_diff3 = end_time3 - start_time3
        time_3 = time_3 + time_diff3.total_seconds()

        batch_num += 1
        start_time1 = datetime.now()


    end_time = datetime.now()
    time_diff = end_time - start_time  # 计算耗时
    print(f"总共过滤: {false_num} 个细胞，保留{true_num}个细胞")
    print(f"minibatch 过滤 方式二：yield + anndata + SQL 过滤耗时: {time_diff.total_seconds():.2f} 秒")
    print(f"minibatch 读取用时： {time_1:.2f} 秒")
    print(f"计算用时： {time_2:.2f} 秒")
    print(f"写表用时： {time_3:.2f} 秒")


# todo SQL 过滤 效率很低
def filter_cells_sql(atlas: 'Atlas',
                 min_counts: int,
                 min_genes: Optional[int] = None,
                 max_counts: Optional[int] = None,
                 max_genes: Optional[int] = None,
                 batch_size: Optional[int] = 2048,
                 add_key: str = "filter_cells_c1") -> None:
    """
    根据最小总表达值过滤细胞，并将过滤条件的布尔向量写入obs表

    Args:
        atlas: Atlas对象
        min_counts: 最小总表达值 该细胞所有基因的表达值之和 sum_expr <  min_counts
        min_genes: 最小非零基因数 该细胞的非零表达的基因个数 nonzero_expr < min_genes
        max_counts: 最大总表达值 该细胞所有基因的表达值之和 sum_expr > max_counts
        max_genes: 最大非零基因数 该细胞的非零表达的基因个数 nonzero_expr > max_genes
        add_key: 写入obs表的字段名
        batch_size: 批次大小，默认2048个细胞一批
    """
    logger.info(f"开始过滤细胞，最小总表达值: {min_counts}，批次大小: {batch_size}")

    start_time = datetime.now()
    atlas.connection = atlas.connect("r+")

    # 检查obs表是否已有add_key列，如果没有则添加
    table_info = atlas.connection.execute("PRAGMA table_info(obs)").fetchall()
    column_names = [col[1] for col in table_info]
    if add_key not in column_names:
        atlas.connection.execute(f"ALTER TABLE obs ADD COLUMN {add_key} BOOLEAN")
        logger.info(f"在obs表中添加了新列: {add_key}")

    # 获取所有基因列名
    gene_result = atlas.connection.execute("SELECT gene_id FROM var")
    gene_ids = [row[0] for row in gene_result.fetchall()]

    # MODIFIED: 先设置默认值为False
    atlas.connection.execute(f"UPDATE obs SET {add_key} = 0 WHERE {add_key} IS NULL")

    print("开始计算... ")

    batch_num = 0 # 第几批
    true_num = 0 # 符合要求的细胞
    false_num = 0 # 不 符合要求的细胞

    time_1 = 0 # 读取用时
    time_2 = 0 # 计算用时
    time_3 = 0 # 写表用时

    start_time1 = datetime.now()

    # MODIFIED: 去掉ORDER BY提高性能
    getdata = """
            WITH indptr_with_next AS (
                SELECT 
                    cell_id,
                    indptr,
                    COALESCE(
                        LEAD(indptr) OVER (ORDER BY indptr),
                        (SELECT MAX(id) + 1 FROM X_CSR_data)
                    ) AS next_indptr
                FROM X_CSR_indptr
            )
            SELECT 
                i.cell_id,
                SUM(d.data) AS sum_expr,
                COUNT(d.data) AS nonzero_expr
            FROM indptr_with_next i
            JOIN X_CSR_data d 
                ON d.id >= i.indptr 
                AND d.id < i.next_indptr
            GROUP BY i.cell_id
        """

    result =  atlas.connection.execute(getdata).df()
    # MODIFIED: 直接使用result DataFrame进行计算，避免索引问题
    sum_expr = result["sum_expr"].values
    nonzero_expr = result["nonzero_expr"].values

    print("获取result成功")
    print(sum_expr)
    print(nonzero_expr)

    end_time1 = datetime.now()
    time_1 = end_time1 - start_time1

    start_time2 = datetime.now()
    # 初始化过滤掩码为全False
    filtered_cells = np.zeros(len(result), dtype=bool)

    # 4. 应用过滤条件 - 保持原有逻辑，但使用更高效的数组操作
    if min_counts is not None:
        filtered_cells |= (sum_expr < min_counts)  # 使用 |= 原位操作符

    if min_genes is not None:
        filtered_cells |= (nonzero_expr < min_genes)

    if max_counts is not None:
        filtered_cells |= (sum_expr > max_counts)

    if max_genes is not None:
        filtered_cells |= (nonzero_expr > max_genes)

    # MODIFIED: 从result中获取细胞ID，而不是从obs_df
    # 符合过滤条件的细胞ID（应该被标记为True）
    filtered_cell_ids = result.loc[filtered_cells, 'cell_id'].tolist()

    true_num = len(filtered_cell_ids)  # 需要过滤的细胞数
    false_num = len(result) - true_num  # 保留的细胞数

    end_time2 = datetime.now()
    time_2 = (end_time2 - start_time2).total_seconds()

    start_time3 = datetime.now()

    print(f"正在标记需要过滤的 {len(filtered_cell_ids)} 个细胞...")

    # 直接使用数组 语法简单
    atlas.connection.execute(f"""
                        UPDATE obs 
                        SET {add_key} = 1 
                        WHERE cell_id = ANY({filtered_cell_ids})
                    """)

    atlas.connection.commit()
    print("过滤完成！")
    end_time3 = datetime.now()
    time_3 = end_time3 - start_time3


    end_time = datetime.now()
    time_diff = end_time - start_time  # 计算耗时
    print(f"总共过滤: {false_num} 个细胞，保留{true_num}个细胞")
    print(f"minibatch 过滤 方式二：yield + anndata + SQL 过滤耗时: {time_diff.total_seconds():.2f} 秒")
    print(f"minibatch 读取用时： {time_1.total_seconds():.2f} 秒")
    print(f"计算用时： {time_2:.2f} 秒")
    print(f"写表用时： {time_3.total_seconds():.2f} 秒")




# ========== todo 方法1： 通过CSR表过滤细胞，不用 minibatch=======
def filter_cells_CSR(atlas: 'Atlas',
                          min_counts=None, min_genes=None,
                          max_counts=None, max_genes=None,
                          add_key="filter_cells_1"):

    import os
    from datetime import datetime

    print("==== 基于 CSR + cell_index（一次 SQL）进行细胞过滤 ====")
    start = datetime.now()

    conn = atlas.connect("r+")
    atlas.connection = conn
    th = os.cpu_count()
    conn.execute(f"PRAGMA threads={th}")

    print("开始聚合 X_CSR_data（sum_expr, nonzero_genes）...")

    # ---- 一次性聚合 ----
    stats = conn.execute("""
        SELECT
            cell_index,
            SUM(data) AS sum_expr,
            COUNT(*) AS nonzero_genes
        FROM X_CSR_data
        GROUP BY cell_index
        ORDER BY cell_index
    """).fetchall()

    print(f"聚合完成，共 {len(stats)} 个细胞")

    # ---- 准备写入 obs ----
    conn.execute(f"""
        ALTER TABLE obs 
        ADD COLUMN IF NOT EXISTS {add_key} BOOLEAN DEFAULT FALSE
    """)

    results = []
    keep = 0

    for cell_index, sum_expr, nonzero_genes in stats:

        ok = True
        if min_counts is not None and sum_expr < min_counts: ok = False
        if max_counts is not None and sum_expr > max_counts: ok = False
        if min_genes  is not None and nonzero_genes < min_genes: ok = False
        if max_genes  is not None and nonzero_genes > max_genes: ok = False

        results.append((cell_index, ok))
        if ok: keep += 1

    conn.execute("CREATE TEMP TABLE tmp_flag (cell_index INTEGER, flag BOOLEAN)")
    conn.executemany("INSERT INTO tmp_flag VALUES (?,?)", results)

    conn.execute(f"""
        UPDATE obs
        SET {add_key} = t.flag
        FROM tmp_flag t
        WHERE obs.id = t.cell_index
    """)

    print(f"过滤完成: 保留 {keep} / {len(stats)}")
    print("耗时: {:.2f} 秒".format((datetime.now() - start).total_seconds()))


# ========== todo 方法2：方法1 + 使用 mmap 扫描 CSR 过滤细胞（安全支持 1 亿细胞） ==========

def filter_cells_CSR_fast(atlas: 'Atlas',
                          min_counts=None, min_genes=None,
                          max_counts=None, max_genes=None,
                          add_key="filter_cells_1"):

    import os
    from datetime import datetime
    start = datetime.now()

    print("==== DuckDB 原生并行过滤（无 Python 循环）====")

    conn = atlas.connect("r+")
    th = os.cpu_count()
    conn.execute(f"PRAGMA threads={th}")
    conn.execute("PRAGMA memory_limit='50GB'")
    conn.execute("PRAGMA temp_directory='.tmp_duckdb'")
    print(f"DuckDB 多线程: {th}")

    print("创建 obs 标记列 ...")
    conn.execute(f"""
        ALTER TABLE obs 
        ADD COLUMN IF NOT EXISTS {add_key} BOOLEAN DEFAULT FALSE
    """)

    # ------------------------------------------------
    # Step 1: 统计总细胞数
    # ------------------------------------------------
    total_cells = conn.execute("SELECT COUNT(*) FROM obs").fetchone()[0]
    print(f"总细胞数: {total_cells:,}")

    print("开始原生并行过滤 ...")

    # ------- 构造动态 SQL 条件 -------
    conds = []
    if min_counts is not None: conds.append(f"sum_expr >= {min_counts}")
    if max_counts is not None: conds.append(f"sum_expr <= {max_counts}")
    if min_genes  is not None: conds.append(f"nonzero_genes >= {min_genes}")
    if max_genes  is not None: conds.append(f"nonzero_genes <= {max_genes}")

    condition = " AND ".join(conds) if conds else "TRUE"

    # ------------------------------------------------
    # Step 2: 执行过滤 + 更新 obs
    # ------------------------------------------------
    conn.execute(f"""
        UPDATE obs SET {add_key} = sub.keep
        FROM (
            SELECT 
                cell_index,
                ({condition}) AS keep
            FROM (
                SELECT
                    cell_index,
                    SUM(data) AS sum_expr,
                    COUNT(*)  AS nonzero_genes
                FROM X_CSR_data
                GROUP BY cell_index
            )
        ) AS sub
        WHERE obs.id = sub.cell_index
    """)

    # ------------------------------------------------
    # Step 3: 统计结果
    # ------------------------------------------------
    keep_cells = conn.execute(f"""
        SELECT COUNT(*) FROM obs WHERE {add_key} = TRUE
    """).fetchone()[0]

    removed_cells = total_cells - keep_cells

    print(f"保留细胞: {keep_cells:,}")
    print(f"过滤细胞: {removed_cells:,}  (占 {removed_cells / total_cells * 100:.2f}%)")

    print("过滤完成，耗时 {:.2f} 秒".format(
        (datetime.now() - start).total_seconds()
    ))


# ========== todo 方法2：方法1 + 使用 mmap 扫描 CSR 过滤细胞（安全支持 1 亿细胞） ==========
def filter_cells_CSR_ultrafast(atlas: 'Atlas',
                               min_counts=None, min_genes=None,
                               max_counts=None, max_genes=None,
                               add_key="filter_cells_1"):

    from datetime import datetime
    import os
    start = datetime.now()

    conn = atlas.connect("r+")
    th = os.cpu_count()
    conn.execute(f"PRAGMA threads={th}")
    conn.execute(f"PRAGMA memory_limit='55GB'")
    conn.execute(f"PRAGMA temp_directory='.tmp_duckdb'")
    print(f"DuckDB threads = {th}")

    # 预先添加列
    conn.execute(f"""
        ALTER TABLE obs 
        ADD COLUMN IF NOT EXISTS {add_key} BOOLEAN DEFAULT FALSE
    """)

    # 统计细胞数
    total_cells = conn.execute("SELECT COUNT(*) FROM obs").fetchone()[0]
    print(f"总细胞数 = {total_cells:,}")

    # -------------------------------
    # 关键：如果未排序，对性能影响 2~5x
    # -------------------------------
    # print("排序 X_CSR_data（若已排序则很快）...")
    # print("开始建立索引.")
    # conn.execute("CREATE INDEX IF NOT EXISTS idx_csr_cell ON X_CSR_data(cell_index)")
    # print("索引建立完毕.")

    # -------------------------------
    # 构建 SQL 过滤条件
    # -------------------------------
    conds = []
    if min_counts is not None: conds.append(f"sum_expr >= {min_counts}")
    if max_counts is not None: conds.append(f"sum_expr <= {max_counts}")
    if min_genes  is not None: conds.append(f"nonzero_genes >= {min_genes}")
    if max_genes  is not None: conds.append(f"nonzero_genes <= {max_genes}")
    condition = " AND ".join(conds) if conds else "TRUE"

    # -------------------------------
    # Step1：先把需要保留的 cell_index 算好
    # -------------------------------
    print("统计基因数量与表达量（流式聚合）...")

    conn.execute("DROP TABLE IF EXISTS keep_cells")
    conn.execute(f"""
        CREATE TEMP TABLE keep_cells AS
        SELECT cell_index
        FROM (
            SELECT 
                cell_index,
                SUM(data) AS sum_expr,
            COUNT(*) AS nonzero_genes
            FROM X_CSR_data
            GROUP BY cell_index
        ) WHERE {condition}
    """)

    # -------------------------------
    # Step2：只更新 TRUE（避免笨重 UPDATE JOIN）
    # -------------------------------
    print("更新 obs（仅 TRUE，加速 3~10x）...")

    conn.execute(f"UPDATE obs SET {add_key}=FALSE")   # 全部设为 FALSE
    conn.execute(f"""
        UPDATE obs SET {add_key}=TRUE
        WHERE id IN (SELECT cell_index FROM keep_cells)
    """)

    # -------------------------------
    # Step3: 统计结果
    # -------------------------------
    keep_cells = conn.execute(f"SELECT COUNT(*) FROM keep_cells").fetchone()[0]
    removed = total_cells - keep_cells

    print(f"保留细胞 = {keep_cells:,}")
    print(f"过滤细胞 = {removed:,} ({removed/total_cells*100:.2f}%)")
    print("总耗时 {:.2f} 秒".format((datetime.now() - start).total_seconds()))


# X表（表达矩阵）
# id | cell_id | TP53 | EGFR | BRAF | KRAS | MYC
# ---|---------|------|------|------|------|----
# 1  | cell_1  | 10   | 0    | 2    | 5    | 0
# 2  | cell_2  | 0    | 15   | 7    | 0    | 8
# 3  | cell_3  | 12   | 2    | 4    | 3    | 0
# 4  | cell_4  | 8    | 0    | 1    | 6    | 4
# 5  | cell_5  | 0    | 10   | 0    | 2    | 0

# var表（基因信息）：
# id | gene_id
# ---|--------
# 1  | TP53
# 2  | EGFR
# 3  | BRAF
# 4  | KRAS
# 5  | MYC



#========== todo 过滤基因 ==========

# ========== todo 方法 1：通过X表 过滤基因 ， 每次只处理一个基因 ==========
def filter_genes_X_one(atlas: 'Atlas',
                 min_counts: int,
                 min_cells: Optional[int] = None,
                 max_counts: Optional[int] = None,
                 max_cells: Optional[int] = None,
                 add_key: str = "filter_genes_1") -> None:
    """
        根据条件过滤基因，并将过滤条件的布尔向量写入var表

    Args:
        atlas: Atlas对象
        min_counts: 最小总表达值 该基因的所有表达值之和 sum_expr >  min_counts
        min_cells: 最小非零细胞数 表达该基因的细胞数量 nonzero_expr > min_genes
        max_counts: 最大总表达值 该基因的所有表达值之和 sum_expr < max_counts
        max_cells: 最大非零细胞数 表达该基因的细胞数量 nonzero_expr < max_genes
        add_key: 写入obs表的字段名
    """
    print(f"开始过滤基因...")
    start_time = datetime.now()

    atlas.connection = atlas.connect("r+")

    # 获取所有基因列名（从X表中排除非基因列）
    gene_columns = [row[0] for row in atlas.connection.execute("""
           SELECT column_name 
           FROM information_schema.columns 
           WHERE table_name = 'X' 
           AND column_name NOT IN ('id', 'cell_id')
       """).fetchall()]
    print(f"找到 {len(gene_columns)} 个基因列")
    # gene_columns = ['TP53', 'EGFR', 'BRAF', 'KRAS', 'MYC']

    # 准备var表,添加标记列（如果不存在则添加）
    atlas.connection.execute(f"""
       ALTER TABLE var 
       ADD COLUMN IF NOT EXISTS {add_key} BOOLEAN DEFAULT FALSE
       """)

    # 初始化统计变量
    batch_num = 0  # 第几批
    true_num = 0  # 符合过滤条件的基因数量
    false_num = 0  # 不符合过滤条件的基因数量
    flag = 0
    time_1 = 0
    time_2 = 0
    time_3 = 0

    # ========== 处理基因 ==========
    for gene in gene_columns:
        print(f"开始处理 第 {batch_num} 个 基因： {gene}  ")
        batch_num = batch_num + 1

        # COUNT(NULLIF(expr, 0)) 统计非零值数量
        query = f"""
            SELECT
                SUM("{gene}") AS sum_expr,
                COUNT(NULLIF("{gene}", 0)) AS nonzero_expr
            FROM X
        """
        start_time1 = datetime.now()

        row = atlas.connection.execute(query).fetchone()
        sum_expr, nonzero_expr = row[0], row[1]

        end_time1 = datetime.now()
        time_diff1 = end_time1 - start_time1
        time_1 = time_1 + time_diff1.total_seconds()

        start_time2 = datetime.now()
        # --- 判断是否满足过滤条件 ---
        if min_counts is not None and sum_expr >= min_counts :
            flag = 1

        end_time2 = datetime.now()
        time_diff2 = end_time2 - start_time2
        time_2 = time_2 + time_diff2.total_seconds()

        start_time3 = datetime.now()
        if flag:
            print(f"正在标记符合要求的该基因...")
            true_num = true_num + 1

            # 直接使用数组 语法简单
            atlas.connection.execute(f"""
                            UPDATE var 
                            SET {add_key} = TRUE 
                            WHERE gene_id = ?
                             """, (gene,))
            flag = 0 # 重置标识符

        end_time3 = datetime.now()
        time_diff3 = end_time3 - start_time3
        time_3 = time_3 + time_diff3.total_seconds()

    print(f"基因过滤完成:")
    print(f"  总基因数: {len(gene_columns)}")
    print(f"  保留基因: {true_num}")
    print(f"  过滤基因: {len(gene_columns) - true_num}")
    print(f"  过滤结果已保存到var表的 '{add_key}' 列")
    end_time = datetime.now()
    time_diff = end_time - start_time
    print("##### 基因过滤用时 :", time_diff.total_seconds())
    print("##### 查询用时 :", time_1)
    print("##### 计算用时 :", time_2)
    print("##### 写表用时 :", time_3)

# ========== todo 方法 2：每次过滤多个基因 ==========
def filter_genes_X_batch(atlas: 'Atlas',
                 min_counts: int,
                 min_cells: Optional[int] = None,
                 max_counts: Optional[int] = None,
                 max_cells: Optional[int] = None,
                 batch_size: Optional[int] = 800,
                 add_key: str = "filter_genes_1") -> None:
    """
        根据条件过滤基因，并将过滤条件的布尔向量写入var表

    Args:
        atlas: Atlas对象
        min_counts: 最小总表达值 该基因的所有表达值之和 sum_expr >=  min_counts
        min_cells: 最小非零细胞数 表达该基因的细胞数量 nonzero_expr >= min_genes
        max_counts: 最大总表达值 该基因的所有表达值之和 sum_expr =< max_counts
        max_cells: 最大非零细胞数 表达该基因的细胞数量 nonzero_expr =< max_genes
        add_key: 写入obs表的字段名

            高性能基因过滤：
    —— 一次性扫描整个 X 表（DuckDB 列式聚合）
    —— 内存占用极小（仅读取 1 行结果）
    —— 批量标记 var（避免逐行 UPDATE）

    """
    print(f"开始过滤基因...")
    time_1 = 0
    time_2 = 0
    time_3 = 0

    start_time = datetime.now()

    atlas.connection = atlas.connect("r+")

    # 1. 获取基因列
    gene_columns = [row[0] for row in atlas.connection.execute("""
           SELECT column_name
           FROM information_schema.columns
           WHERE table_name = 'X'
           AND column_name NOT IN ('id', 'cell_id')
       """).fetchall()]
    G = len(gene_columns)
    print(f"共 {G} 个基因列")
    # gene_columns = ['TP53', 'EGFR', 'BRAF', 'KRAS', 'MYC']

    # 2. 添加过滤结果列到 var
    atlas.connection.execute(f"""
       ALTER TABLE var
       ADD COLUMN IF NOT EXISTS {add_key} BOOLEAN DEFAULT FALSE
       """)

    keep = {}

    batches = math.ceil(G / batch_size)

    for b in range(batches):

        cols = gene_columns[b*batch_size:(b+1)*batch_size]
        print(f"[{b+1}/{batches}] 处理基因列 batch 大小 = {len(cols)} ...")


        time_start1 = datetime.now()

        # 构造聚合 SQL
        agg_parts = []
        for g in cols:
            agg_parts.append(f'SUM("{g}") AS "{g}_sum"')
            agg_parts.append(f'COUNT(NULLIF("{g}",0)) AS "{g}_nz"')

        sql = "SELECT " + ", ".join(agg_parts) + " FROM X;"

        row = atlas.connection.execute(sql).fetchone()

        time_end1 = datetime.now()
        time_diff1 = time_end1 - time_start1
        time_1 = time_1 + time_diff1.total_seconds()

        time_start2 = datetime.now()
        # 解析结果
        idx = 0
        for g in cols:
            sum_expr = row[idx] or 0
            nonzero = row[idx+1] or 0
            idx += 2

            ok = True
            if min_counts is not None and sum_expr < min_counts:
                ok = False
            if max_counts is not None and sum_expr > max_counts:
                ok = False
            if min_cells is not None and nonzero < min_cells:
                ok = False
            if max_cells is not None and nonzero > max_cells:
                ok = False

            keep[g] = ok

            time_end2 = datetime.now()
            time_diff2 = time_end2 - time_start2
            time_2 = time_2 + time_diff2.total_seconds()

    time_start3 = datetime.now()
    # 写回 var（使用临时表 + JOIN）
    atlas.connection.execute("CREATE TEMP TABLE tmp_flag (gene_id TEXT, flag BOOLEAN);")
    atlas.connection.executemany(
        "INSERT INTO tmp_flag VALUES (?,?)",
        list(keep.items())
    )
    atlas.connection.execute(f"""
        UPDATE var
        SET {add_key} = tmp.flag
        FROM tmp_flag AS tmp
        WHERE var.gene_id = tmp.gene_id;
    """)

    time_end3 = datetime.now()
    time_diff3 = time_end3 - time_start3
    time_3 = time_3 + time_diff3.total_seconds()

    kept = sum(keep.values())



    print("过滤完成：")
    print(f"  总基因数：{G}")
    print(f"  保留基因：{kept}")
    print(f"  过滤基因：{G - kept}")
    print("过滤 用时：", (datetime.now() - start_time).total_seconds())



    print("##### 查询用时 :", time_1)
    print("##### 计算用时 :", time_2)
    print("##### 写表用时 :", time_3)


# ========== todo 方法 3：每次过滤多个基因 ， 加上计算改进  =========
def filter_genes_X_batch_1(atlas: 'Atlas',
                 min_counts: int,
                 min_cells: Optional[int] = None,
                 max_counts: Optional[int] = None,
                 max_cells: Optional[int] = None,
                 batch_size: Optional[int] = 800,  # ==== MODIFIED ==== 可调：每次聚合处理多少列（基因）
                 add_key: str = "filter_genes_1",
                 ) -> None:
    """
        根据条件过滤基因，并将过滤条件的布尔向量写入var表

        ==== 修改点 ====
        1) 启用 DuckDB 多线程（PRAGMA threads）
        2) 用 SUM((col <> 0)::BIGINT) 替代 COUNT(NULLIF(col,0))
        3) 列分批（column-batching），避免一次生成超长 SQL 导致解析/执行慢
        4) 保持原有打印/时间统计结构不变
    """

    import os
    import math
    from datetime import datetime

    print(f"开始过滤基因...")
    time_1 = 0     # SQL 查询用时（聚合）
    time_2 = 0     # Python 计算用时
    time_3 = 0     # 写表用时

    start_time = datetime.now()

    atlas.connection = atlas.connect("r+")
    conn = atlas.connection

    # ==== MODIFIED ==== 开启多线程，尽量使用所有 CPU 核心
    try:
        threads = os.cpu_count() or 1
        conn.execute("PRAGMA threads = " + str(threads))
        print(f"DuckDB 多线程已启用：PRAGMA threads={threads}")
    except Exception as e:
        print("设置 PRAGMA threads 失败，继续使用默认线程。错误：", e)

    # -------------------------
    # 1. 获取基因列
    # -------------------------
    gene_columns = [row[0] for row in conn.execute("""
           SELECT column_name 
           FROM information_schema.columns 
           WHERE table_name = 'X' 
           AND column_name NOT IN ('id', 'cell_id')
       """).fetchall()]
    G = len(gene_columns)
    print(f"共 {G} 个基因列")

    # -------------------------
    # 2. 添加过滤列到 var
    # -------------------------
    conn.execute(f"""
       ALTER TABLE var 
       ADD COLUMN IF NOT EXISTS {add_key} BOOLEAN DEFAULT FALSE
    """)

    keep = {}

    # ==== MODIFIED ==== 使用列分批（column-batching）来避免单条 SQL 过大


    col_batches = math.ceil(G / batch_size)
    print(f"使用列分批：col_batch_size={batch_size}，总批次={col_batches}")

    # 总索引游标（用于在每个批次中解析 row）
    total_idx = 0

    for cb in range(col_batches):
        cols = gene_columns[cb*batch_size:(cb+1)*batch_size]
        print(f"[列批 {cb+1}/{col_batches}] 处理 {len(cols)} 列 ...")

        # ==== MODIFIED ==== 构造每个批次的聚合 SQL，使用 SUM(col) 和 SUM((col <> 0)::BIGINT)
        # 使用 ( "{g}" <> 0 )::BIGINT 可以让 DuckDB 做向量化的布尔到整数转换
        agg_parts = []
        for g in cols:
            # 注意：保留列名用双引号避免特殊字符问题
            agg_parts.append(f'SUM("{g}") AS "{g}_sum"')
            # COUNT(NULLIF(...)) -> SUM((col <> 0)::BIGINT)
            agg_parts.append(f'SUM( ("{g}" <> 0)::BIGINT ) AS "{g}_nz"')

        sql = "SELECT " + ", ".join(agg_parts) + " FROM X;"

        # 执行该列批次的聚合
        t1_start = datetime.now()
        row = conn.execute(sql).fetchone()
        t1_end = datetime.now()
        time_1 += (t1_end - t1_start).total_seconds()

        # 解析该批次结果
        t2_start = datetime.now()

        idx = 0
        for g in cols:
            sum_expr = row[idx] or 0
            nz_expr  = row[idx+1] or 0
            idx += 2

            ok = True
            if min_counts is not None and sum_expr < min_counts:
                ok = False
            if max_counts is not None and sum_expr > max_counts:
                ok = False
            if min_cells is not None and nz_expr < min_cells:
                ok = False
            if max_cells is not None and nz_expr > max_cells:
                ok = False

            keep[g] = ok

        t2_end = datetime.now()
        time_2 += (t2_end - t2_start).total_seconds()

    # -------------------------
    # 3. 写回 var
    # -------------------------
    print("写回 var 表...")

    t3_start = datetime.now()

    conn.execute("CREATE TEMP TABLE tmp_flag (gene_id TEXT, flag BOOLEAN);")
    # 批量插入 tmp_flag（可能很大），可分片插入以防超大事务（下面简单一次性插入）
    conn.executemany(
        "INSERT INTO tmp_flag VALUES (?,?)",
        list(keep.items())
    )
    conn.execute(f"""
        UPDATE var
        SET {add_key} = tmp.flag
        FROM tmp_flag AS tmp
        WHERE var.gene_id = tmp.gene_id;
    """)

    t3_end = datetime.now()
    time_3 = (t3_end - t3_start).total_seconds()

    kept = sum(keep.values())

    print("过滤完成：")
    print(f"  总基因数：{G}")
    print(f"  保留基因：{kept}")
    print(f"  过滤基因：{G - kept}")
    print(f"过滤 用时： { (datetime.now() - start_time).total_seconds():.2f}")

    print(f"##### 查询用时 : {time_1:.2f}")
    print(f"##### 计算用时 : {time_2:.2f}")
    print(f"##### 写表用时 : {time_3:.2f}")

# ========== todo 方法 4：行-批 + NumPy 聚合 + 批写回 var ，当前最快 ==========
def filter_genes_CSR(atlas: 'Atlas',
                     min_counts: int,
                     min_cells: Optional[int] = None,
                     max_counts: Optional[int] = None,
                     max_cells: Optional[int] = None,
                     add_key: str = "filter_genes_1") -> None:
    """
    使用 CSR 数据（X_CSR_indptr + X_CSR_data）计算每个基因的：
        - sum_expr
        - nonzero_expr
    并写入 var 表。
    """

    import os
    from datetime import datetime

    print("==== 基于 CSR 数据进行基因过滤 ====")
    start_time = datetime.now()

    conn = atlas.connect("r+")
    atlas.connection = conn

    # 允许 DuckDB 多线程
    try:
        th = os.cpu_count() or 1
        conn.execute(f"PRAGMA threads={th}")
        print(f"DuckDB 多线程: {th}")
    except:
        pass

    # =======================================================
    # 👇 修改点：CSR indices 对应的是 var.id
    # =======================================================
    gene_map = dict(conn.execute("SELECT id, gene_id FROM var").fetchall())
    n_genes = len(gene_map)
    print(f"检测到基因数量: {n_genes}")

    # 添加过滤字段
    conn.execute(f"""
        ALTER TABLE var 
        ADD COLUMN IF NOT EXISTS {add_key} BOOLEAN DEFAULT FALSE;
    """)

    print("开始聚合 X_CSR_data ...")

    # -------- 核心：CSR 聚合统计 --------
    rows = conn.execute("""
        SELECT 
            indices AS gene_index,
            SUM(data) AS sum_expr,
            COUNT(*) AS nonzero_expr
        FROM X_CSR_data
        GROUP BY indices
        ORDER BY indices
    """).fetchall()

    print(f"聚合完成，共统计到 {len(rows)} 个非零基因")

    # -------- 写入临时表 --------
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

    # =======================================================
    # 👇 完全零表达的基因（不在 CSR 中出现）
    # =======================================================
    all_gene_ids = set(gene_map.values())
    zero_genes = all_gene_ids - appeared_gene_ids
    for g in zero_genes:
        insert_rows.append((g, False))

    # 写入临时表
    conn.executemany("INSERT INTO tmp_stats VALUES (?,?)", insert_rows)

    # 更新 var 表
    conn.execute(f"""
        UPDATE var
        SET {add_key} = tmp.flag
        FROM tmp_stats AS tmp
        WHERE var.gene_id = tmp.gene_id
    """)

    print(f"过滤完成: 保留基因 {keep_count} / 总 {n_genes}")
    print("总耗时: {:.2f} 秒".format((datetime.now() - start_time).total_seconds()))





#========== todo  计算每个细胞的总 UMI（Unique Molecular Identifier）计数 : 通过indptr可以很快计算==========
def calculate_cell_total_counts(atlas: 'Atlas',
                 batch_size: Optional[int] = 2048,
                 add_key: str = "cell_total_counts") -> None:
    """
    计算每个细胞的总UMI计数

    :param atlas:
    :param batch_size:
    :param add_key:
    :return:
    """

    logger.info(f"计算每个细胞的总UMI计数")

    start_time = datetime.now()
    atlas.connection = atlas.connect("r+")

    # 检查obs表是否已有add_key列，如果没有则添加
    table_info = atlas.connection.execute("PRAGMA table_info(obs)").fetchall()
    column_names = [col[1] for col in table_info]
    if add_key not in column_names:
        atlas.connection.execute(f"ALTER TABLE obs ADD COLUMN {add_key} BOOLEAN")
        logger.info(f"在obs表中添加了新列: {add_key}")

    # 获取所有基因列名
    gene_result = atlas.connection.execute("SELECT gene_id FROM var")
    gene_ids = [row[0] for row in gene_result.fetchall()]

    if not gene_ids:
        logger.warning("未找到任何基因")
        return
    logger.info("进入for循环... ")

    batch_num = 0  # 第几批
    processed_cells = 0  # 已处理的细胞数量

    # 时间统计
    time_loading = 0    # 数据加载用时
    time_calculation = 0  # 计算用时
    time_writing = 0     # 写入数据库用时

    start_time1 = datetime.now()

    for adata_minibatch in atlas.query_minibatch(batch_size=batch_size):

        end_time1 = datetime.now()
        time_diff1 = end_time1 - start_time1
        time_loading = time_loading + time_diff1.total_seconds()

        print(f"正在计算第 {batch_num} 批")

        start_time2 = datetime.now()

        # ========== 核心计算部分 ==========
        # 计算每个细胞（行）的总表达值（总UMI计数）
        # 通过 .flatten() 将结果转换为一维数组
        total_counts = np.array(adata_minibatch.X.sum(axis=1)).flatten()

        # 获取当前批次的细胞ID列表
        cell_ids = adata_minibatch.obs['cell_id'].tolist()

        # 统计处理的细胞数量
        processed_cells += len(cell_ids)

        end_time2 = datetime.now()
        time_diff2 = end_time2 - start_time2
        time_calculation = time_calculation + time_diff2.total_seconds()

        # ========== 写入数据库部分 ==========
        start_time3 = datetime.now()

        # 批量更新数据库中的总UMI计数
        if cell_ids:
            logger.info(f"正在更新 {len(cell_ids)} 个细胞的总UMI计数...")

            # 准备更新数据：每个细胞的(cell_id, total_count)对
            update_data = list(zip(total_counts, cell_ids))

            # 使用executemany进行批量更新
            atlas.connection.executemany(f"""
                       UPDATE obs 
                       SET {add_key} = ? 
                       WHERE cell_id = ?
                   """, update_data)

            atlas.connection.commit()
            logger.info(f"第 {batch_num} 批次更新完成")
        else:
            logger.warning("当前批次没有细胞数据")

        # ========== 批次统计信息 ==========
        batch_num += 1


        end_time3 = datetime.now()
        time_diff3 = end_time3 - start_time3
        time_writing = time_writing + time_diff3.total_seconds()

        batch_num += 1
        start_time1 = datetime.now()

    end_time = datetime.now()
    time_diff = end_time - start_time  # 计算耗时
    print(f"总共计算了: {processed_cells} 个细胞")
    print(f"计算每个细胞的总UMI计数 耗时 : {time_diff.total_seconds():.2f} 秒")
    print(f"minibatch 读取用时： {time_loading:.2f} 秒")
    print(f"计算用时： {time_calculation:.2f} 秒")
    print(f"写表用时： {time_writing:.2f} 秒")


# ========== todo  计算每个基因的表达值 ==========
def calculate_gene_total_counts(atlas: 'Atlas',
                                batch_size: Optional[int] = 2048,
                                add_key1: str = "gene_total_counts",
                                add_key2: str = "gene_means_counts") -> None:
    """
    计算每个基因的总表达值，平均表达值，并将结果写入 var表的add_key1列，和add_key2列

    :param atlas: Atlas对象
    :param batch_size: 批次大小
    :param add_key1: 总表达值列名
    :param add_key2: 平均表达值列名
    :return: None
    """

    logger.info(f"开始计算每个基因的总表达值和平均表达值")

    start_time = datetime.now()
    atlas.connection = atlas.connect("r+")

    # 检查var表是否已有目标列，如果没有则添加
    table_info = atlas.connection.execute("PRAGMA table_info(var)").fetchall()
    column_names = [col[1] for col in table_info]

    if add_key1 not in column_names:
        atlas.connection.execute(f"ALTER TABLE var ADD COLUMN {add_key1} FLOAT DEFAULT 0.0")
        logger.info(f"在var表中添加了新列: {add_key1}")

    if add_key2 not in column_names:
        atlas.connection.execute(f"ALTER TABLE var ADD COLUMN {add_key2} FLOAT DEFAULT 0.0")
        logger.info(f"在var表中添加了新列: {add_key2}")

    # 获取所有基因列名（从X表）
    gene_columns_result = atlas.connection.execute("""
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_name = 'X' 
        AND column_name NOT IN ('id', 'cell_id')
    """).fetchall()

    gene_columns = [row[0] for row in gene_columns_result]

    if not gene_columns:
        logger.warning("未找到任何基因列")
        return

    logger.info(f"找到 {len(gene_columns)} 个基因，开始分批处理...")

    # 获取细胞总数，用于计算平均表达值
    total_cells_result = atlas.connection.execute("SELECT COUNT(*) FROM obs").fetchone()
    total_cells = total_cells_result[0] if total_cells_result else 0

    logger.info(f"总细胞数: {total_cells}")

    # 统计变量
    batch_num = 0
    processed_genes = 0

    # 时间统计
    time_calculation = 0
    time_writing = 0

    # ========== 分批处理基因 ==========
    for i in range(0, len(gene_columns), batch_size):
        batch_start_time = datetime.now()

        # 获取当前批次的基因
        batch_genes = gene_columns[i:i + batch_size]
        batch_num += 1

        logger.info(f"正在处理第 {batch_num} 批次，包含 {len(batch_genes)} 个基因")

        # ========== 构建当前批次的查询 ==========
        batch_queries = []
        for gene in batch_genes:
            query = f"""
                SELECT 
                    '{gene}' as gene_id,
                    SUM({gene}) as total_counts,
                    AVG({gene}) as mean_counts
                FROM X
            """
            batch_queries.append(query)

        # 合并所有查询
        union_query = " UNION ALL ".join(batch_queries)

        # 执行查询
        batch_results = atlas.connection.execute(union_query).fetchall()

        calculation_end_time = datetime.now()
        time_calculation += (calculation_end_time - batch_start_time).total_seconds()

        # ========== 更新var表 ==========
        writing_start_time = datetime.now()

        # 准备更新数据
        update_data = []
        for row in batch_results:
            gene_id, total_count, mean_count = row
            update_data.append((total_count, mean_count, gene_id))

        # 批量更新var表
        if update_data:
            atlas.connection.executemany(f"""
                UPDATE var 
                SET {add_key1} = ?, {add_key2} = ?
                WHERE gene_id = ?
            """, update_data)

            atlas.connection.commit()

        writing_end_time = datetime.now()
        time_writing += (writing_end_time - writing_start_time).total_seconds()

        #======= 辅助信息 ===== #
        # 更新进度
        processed_genes += len(batch_genes)
        progress = processed_genes / len(gene_columns) * 100
        # 输出当前批次的统计信息
        if batch_results:
            total_counts = [row[1] for row in batch_results]
            mean_counts = [row[2] for row in batch_results]

            avg_total = np.mean(total_counts)
            avg_mean = np.mean(mean_counts)

            logger.info(f"批次 {batch_num} 完成: "
                        f"进度 {progress:.1f}% ({processed_genes}/{len(gene_columns)}) | "
                        f"平均总表达值: {avg_total:.2f} | "
                        f"平均表达水平: {avg_mean:.4f}")

    # ========== 最终统计和输出 ==========
    end_time = datetime.now()
    total_time = (end_time - start_time).total_seconds()

    logger.info("=" * 60)
    logger.info("基因表达值计算完成!")
    logger.info(f"处理总耗时: {total_time:.2f} 秒")
    logger.info(f"处理的基因总数: {processed_genes}")
    logger.info(f"时间分布:")
    logger.info(f"  - 计算用时: {time_calculation:.2f} 秒 ({time_calculation / total_time * 100:.1f}%)")
    logger.info(f"  - 数据库写入用时: {time_writing:.2f} 秒 ({time_writing / total_time * 100:.1f}%)")
    logger.info(f"结果已保存到var表:")
    logger.info(f"  - 总表达值: '{add_key1}' 列")
    logger.info(f"  - 平均表达值: '{add_key2}' 列")
    logger.info("=" * 60)


# ========== todo  线粒体基因比例计算 ==========
# def calculate_qc_metrics
# 过滤掉某些基因表达比例过高的细胞
# 不需要实际删除细胞记录，直接把计算出的过滤条件的布尔向量写入obs表，其字段名由参数add_key指定
# 使用sc.pp.calculate_qc_metrics函数计算质量控制指标，并通过可视化工具（如小提琴图、散点图）分析数据集，确定截断值以过滤异常细胞。
# sc.pp.calculate_qc_metrics(
#     adata,
#     qc_vars=['mt'],
#     percent_top=None,  是否计算 “top N 高表达基因占比” 大部分 QC 不计算这类指标
#     log1p=False, QC 计算时不要对表达矩阵做 log1p 变换; False在 原始 counts 上计算 QC
#     inplace=True 直接把结果写进 adata.obs / adata.var ;
# )
def calculate_qc_metrics(
    atlas: Atlas,
    qc_prefix: str = "MT-",   # 线粒体基因名前缀，如 MT-CO1
    qc_key: str = "mt"        # Scanpy 中 qc_vars=['mt']
) -> None:
    """
    使用 DuckDB + CSR 稀疏存储，实现 Scanpy 的 calculate_qc_metrics

    数据约定：
    --------------------------------------------------
    X_CSR_data:
        - cell_index : 细胞 ID（obs.id）
        - indices    : 基因索引（var.id）
        - data       : 表达值（非零）

    var:
        - id         : gene_index（与 indices 对齐）
        - gene_id    : 真实基因名（如 MT-CO1）

    obs:
        - id         : cell_id
    """

    print("==== calculate_qc_metrics (CSR + DuckDB) ====")
    start = datetime.now()

    # DuckDB 连接
    conn = atlas.connection

    # =================================================
    # 0️⃣ 并行执行设置
    # =================================================
    # DuckDB 支持多线程 pipeline aggregation
    # 对 GROUP BY / JOIN 非常关键
    try:
        conn.execute(f"PRAGMA threads={os.cpu_count()}")
    except:
        pass

    # =================================================
    # 1️⃣ 在 var 表中生成 qc 标记列（如 mt）
    # =================================================
    # 等价 Scanpy：
    # adata.var['mt'] = adata.var_names.str.startswith("MT-")

    print(f"-> 标记 qc gene: {qc_key} (prefix='{qc_prefix}')")

    # 若不存在 qc_key（如 mt），先添加一列 BOOLEAN
    conn.execute(f"""
        ALTER TABLE var
        ADD COLUMN IF NOT EXISTS {qc_key} BOOLEAN
    """)

    # 根据 gene_id 前缀判断是否为 qc gene
    # SQL LIKE 'MT-%' 等价 Python startswith("MT-")
    conn.execute(f"""
        UPDATE var
        SET {qc_key} =
            CASE
                WHEN gene_id LIKE '{qc_prefix}%'
                THEN TRUE
                ELSE FALSE
            END
    """)

    # =================================================
    # 2️⃣ Cell-wise QC（每个细胞）
    # =================================================
    print("-> 计算 cell-wise QC")

    # -------------------------------------------------
    # 2.1 total_counts / n_genes_by_counts
    # -------------------------------------------------
    # Scanpy 对应：
    # adata.obs['total_counts'] = X.sum(axis=1)
    # adata.obs['n_genes_by_counts'] = (X > 0).sum(axis=1)

    # CSR 中：
    # - SUM(data)        → total_counts  每个 cell 的总 counts
    # - COUNT(*)         → 非零基因数
    conn.execute("""
        CREATE OR REPLACE TEMP TABLE _cell_basic AS
        SELECT
            cell_index AS id,
            SUM(data)  AS total_counts,
            COUNT(*)   AS n_genes_by_counts
        FROM X_CSR_data
        WHERE data IS NOT NULL
        GROUP BY cell_index
    """)

    # 给 obs 表准备字段
    conn.execute("""
        ALTER TABLE obs
        ADD COLUMN IF NOT EXISTS total_counts REAL
    """)
    conn.execute("""
        ALTER TABLE obs
        ADD COLUMN IF NOT EXISTS n_genes_by_counts INTEGER
    """)

    # 把聚合结果写回 obs
    conn.execute("""
        UPDATE obs
        SET
            total_counts = c.total_counts,  -- 每个 cell 的总 counts
            n_genes_by_counts = c.n_genes_by_counts  -- 每个 cell 中 非零基因数
        FROM _cell_basic c
        WHERE obs.id = c.id
    """)

    # -------------------------------------------------
    # 2.2 total_counts_mt / pct_counts_mt
    # -------------------------------------------------
    # Scanpy 对应：
    # adata.obs['total_counts_mt']
    # adata.obs['pct_counts_mt']

    # 做法：
    # X_CSR_data → JOIN var → 只保留 mt 基因 → cell-wise SUM
    conn.execute(f"""
        CREATE OR REPLACE TEMP TABLE _cell_qc AS
        SELECT
            x.cell_index AS id,
            SUM(x.data)  AS total_counts_qc
        FROM X_CSR_data x
        JOIN var v
          ON x.indices = v.id
        WHERE v.{qc_key} = TRUE
        GROUP BY x.cell_index
    """)

    # obs 中增加字段
    conn.execute("""
        ALTER TABLE obs
        ADD COLUMN IF NOT EXISTS total_counts_mt REAL
    """)
    conn.execute("""
        ALTER TABLE obs
        ADD COLUMN IF NOT EXISTS pct_counts_mt REAL
    """)

    # 写回 obs
    # pct = total_counts_mt / total_counts * 100 , 线粒体基因的百分比
    conn.execute("""
        UPDATE obs
        SET
            total_counts_mt = COALESCE(q.total_counts_qc, 0),  -- 线粒体基因 counts 之和
            pct_counts_mt =
                CASE
                    WHEN obs.total_counts > 0
                    THEN 100.0 * COALESCE(q.total_counts_qc, 0) / obs.total_counts
                    ELSE 0
                END
        FROM _cell_qc q
        WHERE obs.id = q.id
    """)

    # =================================================
    # 3️⃣ Gene-wise QC（每个基因）
    # =================================================
    print("-> 计算 gene-wise QC")

    # Scanpy 对应：
    # adata.var['total_counts'] = X.sum(axis=0)
    # adata.var['n_cells_by_counts'] = (X > 0).sum(axis=0)

    conn.execute("""
        CREATE OR REPLACE TEMP TABLE _gene_qc AS
        SELECT
            indices AS id,
            SUM(data) AS total_counts,
            COUNT(DISTINCT cell_index) AS n_cells_by_counts
        FROM X_CSR_data
        WHERE data IS NOT NULL
        GROUP BY indices
    """)

    # var 表中增加字段
    conn.execute("""
        ALTER TABLE var
        ADD COLUMN IF NOT EXISTS total_counts REAL
    """)
    conn.execute("""
        ALTER TABLE var
        ADD COLUMN IF NOT EXISTS n_cells_by_counts INTEGER
    """)

    # 写回 var
    conn.execute("""
        UPDATE var
        SET
            total_counts = g.total_counts,  -- 该 gene 在所有 cells 的 counts 之和
            n_cells_by_counts = g.n_cells_by_counts  -- 有多少 cell 表达该 gene（非零）
        FROM _gene_qc g
        WHERE var.id = g.id
    """)

    # =================================================
    # 结束
    # =================================================
    print("calculate_qc_metrics 完成")
    print("耗时: {:.2f} 秒".format((datetime.now() - start).total_seconds()))

# 运行结果
# var 表  新增字段
#    字段	                    含义
# mt	              gene_id 是否以 MT- 开头
# total_counts	       该 gene 在所有 cells 的 counts 之和
# n_cells_by_counts	     非零 cell 数 ：有多少 cell 表达该 gene（非零）

# obs 表  新增字段
#    字段	                   含义
# total_counts	            每个 cell 的总 counts
# n_genes_by_counts       	每个 cell 的 非零基因数
# total_counts_mt	        线粒体基因 counts 之和
# pct_counts_mt           	线粒体基因的百分比
