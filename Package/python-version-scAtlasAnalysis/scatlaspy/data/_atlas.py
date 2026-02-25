from datetime import datetime
from typing import *
import numpy as np
from _duckdb import DuckDBPyConnection
from anndata import AnnData
import duckdb
import os
import logging # 管理各种类型的日志
import pandas as pd
import random
import tempfile
import matplotlib.pyplot as plt
import scipy.sparse as sp
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import multiprocessing as mp

from pyexpat.errors import messages

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # 输出到控制台
        logging.FileHandler('atlas.log')  # 输出到文件
    ]
)
logger = logging.getLogger('Atlas')

class Atlas:
    """
    这是一个Atlas类，用来管理和待分析数据集的数据库的交互

    """
    __reserved_words__={ # 这是一个SQL保留关键字列表。这些是在SQL语言中具有特殊含义的单词，不能直接用作表名、列名等标识符，除非使用引号转义。
        'add', 'all', 'alter', 'and', 'any', 'as', 'asc', 'between', 'by', 'case', 'cast', 'check',
		'column', 'create', 'cross', 'current_date', 'current_time', 'default', 'delete', 'desc',
		'distinct', 'drop', 'else', 'exists', 'false', 'for', 'foreign', 'from', 'full', 'group',
		'having', 'in', 'inner', 'insert', 'interval', 'into', 'is', 'join', 'left', 'like', 'limit',
		'not', 'null', 'on', 'or', 'order', 'outer', 'primary', 'references', 'right', 'select',
		'set', 'table', 'then', 'to', 'true', 'union', 'unique', 'update', 'values', 'when', 'where'
    }

    def __init__(self, name: str, path:str):
        """
        初始化类实例
        Args:
            name: 名称
            path: 路径
        """
        logger.info(f"开始初始化 Atlas 实例，名称: {name}, 路径: {path}")

        self.__name = name  # 该数据库的名称（无后缀）
        self.__path = path
        self.__connection = None  # 存储当前的数据库连接

        logger.debug(f"实例变量已设置 - name: {self.__name}, path: {self.__path}")

        # 检查数据库文件是否存在，如果不存在则创建
        db_file = os.path.join(self.__path, f"{self.__name}.sasql")
        logger.debug(f"数据库文件路径: {db_file}")

        self.__db_file = db_file # 数据库文件的绝对路径

        if not os.path.exists(db_file):
            logger.info(f"数据库文件不存在，开始创建新数据库: {db_file}")
            try:
                self.__connection=self._create(name, path)
                logger.info(f"数据库创建成功: {db_file}")
            except Exception as e:
                logger.error(f"数据库创建失败: {str(e)}")
                raise
        else:
            self.__connection = self.connect("r+")
            logger.info(f"数据库文件已存在: {db_file}，已创建连接")

        logger.info("Atlas 实例初始化完成")

    @property
    def db_file(self) -> str:
        """获取名称属性"""
        return self.__db_file

    @db_file.setter
    def db_file(self, value: str) -> None:
        """设置名称属性"""
        self.__db_file = value

    @property
    def name(self) -> str:
        """获取名称属性"""
        return self.__name

    @name.setter
    def name(self, value: str) -> None:
        """设置名称属性"""
        self.__name = value

    @property
    def path(self) -> str:
        """获取路径属性"""
        return self.__path

    @path.setter
    def path(self, value: str) -> None:
        """设置路径属性"""
        self.__path = value

    @property
    def connection(self) -> Optional[duckdb.DuckDBPyConnection]:
        """获取当前数据库连接"""
        return self.__connection

    @connection.setter
    def connection(self, value: str)-> None:
        self.__connection = value

    def _create(self, name: str, path: str) -> duckdb.DuckDBPyConnection:
        """
        在path路径下创建一个名称为<name.sasql>的数据库文件

        :param name: 数据库名称
        :param path: 数据库文件存储路径
        :return: 数据库连接对象
        """
        logger.debug(f"开始创建数据库，名称: {name}, 路径: {path}")

        # 构建完整的数据库文件路径
        db_file = os.path.join(path, f"{name}.sasql")
        logger.debug(f"数据库文件完整路径: {db_file}")

        # 检查数据库是否已存在
        if os.path.exists(db_file):
            logger.error(f"尝试创建已存在的数据库: {db_file}")
            raise RuntimeError(f"数据库已存在: {db_file}")

        try:
            # 确保目录存在
            logger.debug(f"创建目录: {path}")
            os.makedirs(path, exist_ok=True)

            # 连接到持久化数据库文件，如果文件不存在会自动创建
            logger.debug("连接 DuckDB 数据库")
            con = duckdb.connect(database=db_file)

            # 可选：进行一些基本的初始化操作
            # 例如设置一些默认配置或创建初始表结构

            logger.info(f"数据库已成功创建：{db_file}")
            return con

        except Exception as e:
            logger.error(f"创建数据库失败：{str(e)}")
            logger.exception("创建数据库异常详情:")
            raise RuntimeError(f"创建数据库失败：{str(e)}")

    def connect(self, mode: Literal["r+", "r"] = "r+") -> duckdb.DuckDBPyConnection:
        """
        和self.name命名的数据库进行连接。
        如果<name.sasql>不存在，则创建并连接；
        如果<name.sasql>存在，则直接连接

        :param mode: 指定模式，只读or读写，
        :return: 数据库连接对象
        """
        logger.info(f"请求数据库连接，模式: {mode}")

        # 如果已有连接，先关闭
        if self.__connection is not None:
            logger.debug("已有数据库连接，先关闭现有连接")
            self.close()

        # 构建数据库文件路径
        # db_file = os.path.join(self.__path, f"{self.__name}.sasql")
        db_file = self.db_file
        logger.debug(f"数据库文件路径: {db_file}")

        try:
            if mode == "r":  # 只读模式
                logger.debug("只读模式连接")
                # 检查文件是否存在
                if not os.path.exists(db_file):
                    logger.error(f"数据库文件不存在，无法以只读模式连接: {db_file}")
                    raise FileNotFoundError(f"数据库文件不存在: {db_file}")

                # 以只读模式连接
                self.__connection = duckdb.connect(database=db_file, read_only=True)
                logger.info(f"以只读模式连接数据库: {db_file}")

            elif mode == "r+":  # 读写模式
                logger.debug("读写模式连接")
                # 确保目录存在
                os.makedirs(self.__path, exist_ok=True)

                # 无论文件是否存在，都会创建或连接
                self.__connection = duckdb.connect(database=db_file, read_only=False)

                if os.path.exists(db_file):
                    logger.info(f"以读写模式连接现有数据库: {db_file}")
                else:
                    logger.info(f"创建并连接新数据库: {db_file}")

            else:
                logger.error(f"不支持的连接模式: {mode}")
                raise ValueError(f"不支持的连接模式: {mode}")

            logger.debug("数据库连接成功")
            return self.__connection

        except Exception as e:
            logger.error(f"连接数据库失败: {str(e)}")
            logger.exception("连接数据库异常详情:")
            raise RuntimeError(f"连接数据库失败: {str(e)}")

    def close(self):
        """
        关闭数据库连接
        :return: None
        """
        logger.info("关闭数据库连接")
        try:
            # 检查是否存在数据库连接
            if self.__connection is not None:
                # 关闭数据库连接
                self.__connection.close()
                # 将连接对象设为None，避免重复关闭
                self.__connection = None
                logger.info("数据库连接已关闭")
            else:
                logger.debug("没有活动的数据库连接需要关闭")

        except Exception as e:
            logger.error(f"关闭数据库连接时出错: {str(e)}")
            logger.exception("关闭数据库连接异常详情:")
            raise RuntimeError(f"关闭数据库连接时出错: {str(e)}")

    def execute_sql(self, sql: str) -> DuckDBPyConnection | None:
        """
        提交执行sql语句
        :param sql: 要执行的SQL语句
        :return: 如果查询有结果则返回结果集，否则返回None
        """
        logger.info(f"执行SQL语句: {sql}")

        # 检查是否有活动的数据库连接
        if self.__connection is None:
            logger.debug("没有活动的数据库连接，自动创建读写连接")
            # 如果没有连接，自动以读写模式连接
            self.connect("r+")
        # 执行SQL语句
        logger.debug("执行SQL语句")
        result = self.__connection.execute(sql)

        # 如果是查询语句，返回结果
        sql_upper = sql.strip().upper()
        if sql_upper.startswith(('SELECT', 'SHOW', 'DESCRIBE', 'EXPLAIN')):
            logger.debug("SQL语句为查询类型，返回结果")
            return result
        else:
            # 对于非查询语句，提交事务
            logger.debug("SQL语句为非查询类型，提交事务")
            self.__connection.commit()
            return None

    def exists(self) -> bool:
        """
        检查数据库文件是否存在

        :return: 如果数据库文件存在返回True，否则返回False
        """
        db_file = os.path.join(self.__path, f"{self.__name}.sasql")
        exists = os.path.exists(db_file)
        logger.debug(f"检查数据库文件是否存在: {db_file} -> {exists}")
        return exists

    def query(self, query, return_type='pandas'):
        """
        全量查询
        用sql语句进行查询，返回pandas DataFrame格式的结果
        :param query: SQL查询语句
        :return: pandas DataFrame
        """
        logger.info("查询数据库，返回值类型为pandas")
        if self.__connection is None:
            self.connect("r+")
        result = self.connection.execute(query)
        df = result.df() # 将查询结果转换为pandas DataFrame
        # self.close()
        return df

    def query_raw(self, query):
        """
        全量查询
        用sql语句进行查询，返回sql结果
        :param query:
        :return:
        """
        logger.info("查询数据库，返回值类型为duckDB")
        self.connect("r")
        result = self.__connection.execute(query)
        # self.close()
        return result

    #     # order模式：
    #     # 按数据在数据库中的存储顺序读取
    #     # 适合需要保持数据顺序的场景
    #     #
    #     # random_no_replace模式：
    #     # 随机打乱数据顺序，但每个样本只出现一次
    #     # 适合训练集划分，确保每个epoch看到所有数据
    #     #
    #     # random_replace模式：
    #     # 随机抽样，可能重复选择相同样本
    #     适合需要无限数据流的场景
    #     需要手动控制迭代次数
    #     #
    #     # drop_last参数：
    #     # 控制是否丢弃最后不足一个批次的数据
    #     # 在训练神经网络时通常设为True
    # 


# == todo==== minibatch 读取  =========
    def query_minibatch(self, mode="order", batch_size=2048, drop_last=True):
        """
        minibatch查询
        以minibatch方式遍历整个数据集，返回的数据用anndata封装
        :param mode: order - 按顺序读取, random_replace - 随机读取有放回, random_no_replace - 随机读取不放回
        :param batch_size: 每次读取的大小
        :param drop_last: 是否丢弃最后不足batch_size的数据
        :return: 生成器，每次返回AnnData对象
        """
        if mode == "order":
            # 按顺序读取模式
            return self.minibatch_scan_order_cursor_csr_df_arrow_onlylie(batch_size, drop_last) # CSR 读取

        elif mode == "random_no_replace":
            # 随机读取不放回模式
            return self.minibatch_scan_random_no_replace( batch_size, drop_last)

        elif mode == "random_replace":
            # 随机读取有放回模式
            return self.minibatch_scan_random_replace(batch_size)

        else:
            raise ValueError(f"不支持的扫描模式: {mode}")

    # todo CSR格式 游标优化，df优化方法2 arrow table + 只选择需要的列df格式；
    #   Arrow -> pandas ->NumPy  当前速度 15 batch/s ， 适合任意规模的数据
    def minibatch_scan_order_cursor_csr_df_arrow_onlylie(self, batch_size, drop_last):
        """按顺序读取模式"""
        # 计算批次数量
        time_2 = 0  # obs 表读取用时
        time_3 = 0  # var 表读取用时
        time_4 = 0  # 生成anndata数据用时
        time_5 = 0  # X_CSR_indptr 表 读取时间
        time_6 = 0  # X_CSR_data 表 读取时间
        time_7 = 0  # CSR 生成anndata数据用时
        time_8 = 0  # X_CSR_data 转 DF 时间

        data_read_start = 0  # X_CSR_data表 每次的读取的 起始点
        data_read_count = 0  # X_CSR_data表 每次的读取的 数据量

        # 新增：用于记录上次读取的最后一个id，用于游标分页
        start_data_id = 0  # 初始化为0，假设id从1开始，或根据实际情况调整

        # 新增：用于记录每批次时间的列表
        batch_times = {
            'batch_num': [],
            'time_X_CSR_indptr': [],
            'time_X_CSR_data': [],
            'time_X_CSR_data_df': [],
            'time_obs': [],
            'time_var': [],
            'time_anndata': [],
            'total_time': []
        }

        total_num = self.query("SELECT COUNT(*) AS count FROM obs").iloc[0, 0]

        if drop_last:
            num_batches = total_num // batch_size
        else:
            num_batches = (total_num + batch_size - 1) // batch_size

        for i in range(num_batches):
            offset = i * batch_size

            # 如果是最后一批且不丢弃剩余数据，调整limit
            if not drop_last and i == num_batches - 1 and total_num % batch_size != 0:
                current_batch_size = total_num % batch_size
            else:
                current_batch_size = batch_size

            print(f"\n--- 批次 {i} 详细时间分析 ---")
            print(f"\n--- 批次大小 {batch_size} ---")

            query_CSR_indptr = f"SELECT indptr FROM X_CSR_indptr LIMIT {current_batch_size} OFFSET {offset}" # 获取CSR_indptr

            CSR_indptr_start = datetime.now()

            result_CSR_indptr = self.connection.execute(query_CSR_indptr) # 获取CSR_indptr
            table = result_CSR_indptr.fetch_arrow_table()
            CSR_indptr_df = table.to_pandas()
            # todo: 第 0 批
            #  CSR_indptr_df =
            #  {  id:  0, 1 ;
            #    cell_id :  cell_0,cell_1 ;
            #    indptr : 2, 3
            #  }

            # todo: 第 1 批
            #  CSR_indptr_df =
            #  {  id:  2, 3 ;
            #    cell_id :  cell_2,cell_3 ;
            #    indptr : 3, 3
            #  }

            # todo: 第 2 批
            #  CSR_indptr_df =
            #  {  id:  4, 5 ;
            #    cell_id :  cell_4,cell_5 ;
            #    indptr : 6, 6
            #  }


            CSR_indptr_end = datetime.now()
            CSR_indptr_time = (CSR_indptr_end - CSR_indptr_start).total_seconds()
            print(f"#### CSR_indptr 转化为 查询时间 + df格式时间 : {CSR_indptr_time}")

            CSR_indptr_array = CSR_indptr_df['indptr'].to_numpy()
            # todo: 第 0 批
            #  CSR_indptr_array = {  2 , 3  }
            # todo: 第 1 批
            #  CSR_indptr_array = { 3 , 3  }
            # todo: 第 2 批
            #  CSR_indptr_array = { 6 , 6  }


            # todo 获取 CSR_data 表的读取终点
            end_data_id = CSR_indptr_array[-1]  # 获取值数量： 获取 3 个值， 即data_read_count = CSR_indptr_array[-1]=3,最后一个值
            # todo : 第 0 批
            #  终点： end_id  = 3 ，
            # todo : 第 1 批
            #  终点： end_id = 3 ，
            # todo : 第 2 批
            #  终点： end_id = 6 ，

            # todo 获取 CSR_indptr值； 在开头补上 0 ， 并减去上一批的末尾值
            CSR_indptr_array = np.concatenate([[0], CSR_indptr_df['indptr'].values - start_data_id])
            # print(f"indptr[-1]   {CSR_indptr_array[-1]}")
            # todo: 第 0 批
            #  CSR_indptr_array = { 0 , 2 , 3  } - 0 = { 0 , 2 , 3  }
            # todo: 第 1 批
            #  CSR_indptr_array = { 0 , 3 , 3  } - 3 = { 0 , 0 , 0  }
            # todo: 第 2 批
            #  CSR_indptr_array = { 0 , 6 , 6  } - 3 = { 0 , 3 , 3  }

            if (end_data_id - start_data_id > 0):

                # print(f"start_data_id 是 {start_data_id}" )
                # print(f"end_data_id 是 {end_data_id}" )
                # print(f" 读取个数 是 {end_data_id - start_data_id}")


                # todo 只提取需要的内容 indices, data
                query_CSR_data = f"SELECT indices, data FROM X_CSR_data WHERE id >= {start_data_id} AND id < {end_data_id} ORDER BY id"
                # todo : 第 0 批
                #  起点 ：start_data_id = 0
                #  终点： end_data_id = 3 ，
                #  CSR_data_df =
                #    {    id:  0, 1 ,2 ;
                #        indices :  0, 2 , 2 ;
                #        data : 8, 2, 5 ;
                #    }
                # todo : 第 1 批
                #  起点 ：start_data_id = 3
                #  终点： end_data_id = 3 ，
                #   不读取
                # todo : 第 2 批
                #  起点 ：start_data_id = 3
                #  读取量： end_data_id = 6 ，
                #  CSR_data_df =
                #    {    id:  3, 4 , 5 ;
                #        indices :  2, 3 , 4 ;
                #        data : 7, 1, 2 ;
                #    }


                CSR_data_df_start = datetime.now()

                CSR_data_df = self.connection.execute(query_CSR_data).fetch_arrow_table().to_pandas()

                CSR_data_df_end = datetime.now()
                CSR_data_df_time = (CSR_data_df_end - CSR_data_df_start).total_seconds()
                print(f"#### CSR_data 转化为 df  时间 : {CSR_data_df_time}")


                CSR_indices_array = CSR_data_df['indices'].to_numpy()  # todo:  CSR_indices_array = [ 0,2,2]
                CSR_data_array = CSR_data_df['data'].to_numpy()  # todo:  CSR_data_array = [ 8,2,5]

                start_data_id = end_data_id # 更新起点
                # todo : 第 0 批
                #  更新起点 ：start_data_id = end_data_id = 3
                # todo : 第 1 批
                #  更新起点 ：start_data_id = end_data_id = 3

                time_8 += CSR_data_df_time

            else:
                print("获取 0 个值，不查询 ")  # todo 空值的处理

            start_time2 = datetime.now()  # obs表读取用时

            sub_obs = self.query(f"SELECT * FROM obs LIMIT {current_batch_size} OFFSET {offset}")

            end_time2 = datetime.now()  # obs 表读取用时
            time_diff2 = end_time2 - start_time2  # obs 表读取用时
            time_2 = time_2 + time_diff2.total_seconds()  # obs 表读取用时

            start_time3 = datetime.now()  # var 表读取用时

            sub_var = self.query("SELECT * FROM var")

            end_time3 = datetime.now()  # var 表读取用时
            time_diff3 = end_time3 - start_time3  # var 表读取用时
            time_3 = time_3 + time_diff3.total_seconds()  # var 表读取用时

            start_time4 = datetime.now()  # 生成anndata数据用时

            sub_obs = sub_obs.iloc[:, 1:]  # 去掉数据表中的前1列 id
            sub_var = sub_var.iloc[:, 1:]  # 去掉数据表中的前1列
            cell_id = sub_obs['cell_id']  # 获取cell_id
            gene_id = sub_var['gene_id']  # 获取gene_id

            # print(f"indptr[-1]   {CSR_indptr_array[-1]}")
            # print(f"len(CSR_data_array)  {len(CSR_data_array)}")
            # print(f"len(CSR_indices_array)  {len(CSR_indices_array)}")
            # print(f"len(CSR_indptr_array)  {len(CSR_indptr_array)}")

            sub_obs.index = cell_id.astype(str)
            sub_var.index = gene_id.astype(str)

            # 创建CSR格式的稀疏矩阵
            X = sp.csr_matrix((CSR_data_array, CSR_indices_array, CSR_indptr_array),
                              shape=(len(sub_obs), len(sub_var)))
            # 创建AnnData对象
            adata = AnnData(X=X, obs=sub_obs, var=sub_var)

            end_time4 = datetime.now()  # 生成anndata数据用时
            time_diff4 = end_time4 - start_time4  # 生成anndata数据用时
            time_4 = time_4 + time_diff4.total_seconds()  # 生成anndata数据用时

            # 新增：记录当前批次的时间
            batch_total_time = time_diff2.total_seconds() + time_diff3.total_seconds() + time_diff4.total_seconds()

            # batch_times['time_X_CSR_indptr'].append(CSR_indptr_result_time + CSR_indptr_df_time)
            # batch_times['time_X_CSR_data'].append(CSR_data_result_time)
            # batch_times['time_X_CSR_data_df'].append(CSR_data_df_time)
            # batch_times['batch_num'].append(i + 1)
            # batch_times['time_obs'].append(time_diff2.total_seconds())
            # batch_times['time_var'].append(time_diff3.total_seconds())
            # batch_times['time_anndata'].append(time_diff4.total_seconds())
            # batch_times['total_time'].append(batch_total_time)

            # 返回数据
            yield adata

            print(f"obs 表读取用时： {time_2:.2f} 秒")
            print(f"var 表读取用时： {time_3:.2f} 秒")
            print(f"生成anndata数据用时： {time_4:.2f} 秒")
            print(f"X_CSR_indptr 表 读取时间： {time_5:.2f} 秒")
            print(f"X_CSR_data 表 读取时间： {time_6:.2f} 秒")
            print(f"X_CSR_data 表 转df时间 ： {time_8:.2f} 秒")

        # 新增：在所有批次处理完成后，绘制时间图表
        # self._plot_batch_times(batch_times, batch_size)

    def minibatch_scan_random_no_replace_coo( self, batch_size: int, drop_last: bool):
        """
        随机、不放回地按 cell 读取 AnnData（基于 COO 重建）

        核心思想：
        - 随机的是 cell（obs 行）
        - X 直接从 (cell_index, gene_index, data) 三元组重建
        - 完全不依赖 CSR indptr
        """
        # =========================================================
        # 1️⃣ 生成一个「随机但不重复」的 cell 顺序（只做一次）
        # =========================================================
        # 这里只读 obs.id，一列 int，内存占用极小
        rng = np.random.default_rng()

        cell_ids =  self.connection.execute("SELECT id FROM obs").fetch_arrow_table().to_pandas()['id'].to_numpy()

        var_count = self.connection.execute("SELECT COUNT(*) AS n FROM var").fetch_arrow_table().to_pandas().iloc[0, 0]

        # 打乱顺序 = random_no_replace
        rng.shuffle(cell_ids)

        total_cells = len(cell_ids)

        if drop_last:
            num_batches = total_cells // batch_size
        else:
            num_batches = (total_cells + batch_size - 1) // batch_size

        # =========================================================
        # 2️⃣ minibatch 循环
        # =========================================================
        for batch_idx in range(num_batches):

            # 当前 batch 的 cell id（这是“随机”的唯一来源）
            batch_cell_ids = cell_ids[
                batch_idx * batch_size: (batch_idx + 1) * batch_size
            ]

            if len(batch_cell_ids) == 0:
                break

            # 用于 SQL IN (...)
            ids_str = ",".join(map(str, batch_cell_ids))

            # =====================================================
            # 3️⃣ 直接读取 COO 三元组
            # =====================================================
            # 这里不需要 indptr，不需要 data 区间计算
            # 每一行就是一个非零值
            query_X = f"""
                SELECT
                    cell_index,   -- 行号（cell）
                    indices,      -- 列号（gene）
                    data          -- 非零值
                FROM X_CSR_data
                WHERE cell_index IN ({ids_str})
            """
            x1 = self.connection.execute(query_X)
            x2 = x1.fetch_arrow_table()
            X_df = x2.to_pandas()

            # 如果这个 batch 全是空 cell（极端情况）
            if X_df.empty:
                continue

            # =====================================================
            # 4️⃣ 把全局 cell_id → batch 内 row index
            # =====================================================
            # COO 要求 row ∈ [0, n_rows)
            # 所以需要一个映射
            #
            # 例：
            #   batch_cell_ids = [42, 7, 19]
            #   映射为
            #   42 → 0
            #    7 → 1
            #   19 → 2
            cell_id_to_row = { cid: i for i, cid in enumerate(batch_cell_ids) }

            rows = X_df['cell_index'].map(cell_id_to_row).to_numpy()
            cols = X_df['indices'].to_numpy()
            vals = X_df['data'].to_numpy()

            # =====================================================
            # 5️⃣ 构建稀疏矩阵（COO → CSR）
            # =====================================================
            # COO 构建最快
            # CSR 是 AnnData / 下游算法更友好的格式
            X = sp.coo_matrix(
                (vals, (rows, cols)),
                shape=(len(batch_cell_ids), var_count)
            ).tocsr()

            # =====================================================
            # 6️⃣ obs / var
            # =====================================================
            sub_obs = (
                self.connection.execute(f"""
                    SELECT * FROM obs
                    WHERE id IN ({ids_str})
                    ORDER BY id
                """)
                .fetch_arrow_table()
                .to_pandas()
                .iloc[:, 1:]  # 去掉内部 id
            )

            sub_var = (
                self.connection.execute("SELECT * FROM var")
                .fetch_arrow_table()
                .to_pandas()
                .iloc[:, 1:]
            )

            sub_obs.index = sub_obs['cell_id'].astype(str)
            sub_var.index = sub_var['gene_id'].astype(str)

            # =====================================================
            # 7️⃣ 生成 AnnData
            # =====================================================
            adata = AnnData(
                X=X,
                obs=sub_obs,
                var=sub_var
            )

            yield adata

# todo 加上时间分析
    def minibatch_scan_random_no_replace_coo_time(self, batch_size: int, drop_last: bool):
        """
        随机、不放回地按 cell 读取 AnnData（基于 COO 重建）
        + 各阶段耗时统计
        """

        # =========================================================
        # 1️⃣ 生成随机 cell 顺序（只做一次）
        # =========================================================
        t_init = time.perf_counter()

        rng = np.random.default_rng()

        cell_ids = (
            self.connection
            .execute("SELECT id FROM obs")
            .fetch_arrow_table()
            .to_pandas()['id']
            .to_numpy()
        )

        var_count = (
            self.connection
            .execute("SELECT COUNT(*) FROM var")
            .fetch_arrow_table()
            .to_pandas()
            .iloc[0, 0]
        )

        rng.shuffle(cell_ids)

        total_cells = len(cell_ids)

        if drop_last:
            num_batches = total_cells // batch_size
        else:
            num_batches = (total_cells + batch_size - 1) // batch_size

        print(f"[INIT] cell_ids={total_cells}, var_count={var_count}, "
              f"init_time={time.perf_counter() - t_init:.3f}s")

        # =========================================================
        # 2️⃣ minibatch 循环
        # =========================================================
        for batch_idx in range(num_batches):

            t_batch_start = time.perf_counter()

            batch_cell_ids = cell_ids[
                batch_idx * batch_size: (batch_idx + 1) * batch_size
            ]

            if len(batch_cell_ids) == 0:
                break

            ids_str = ",".join(map(str, batch_cell_ids))

            # =====================================================
            # 3️⃣ 读取 COO 三元组
            # =====================================================
            t_x_start = time.perf_counter()

            query_X = f"""
                SELECT
                    cell_index,
                    indices,
                    data
                FROM X_CSR_data
                WHERE cell_index IN ({ids_str})
            """

            X_df = (
                self.connection
                .execute(query_X)
                .fetch_arrow_table()
                .to_pandas()
            )

            t_x = time.perf_counter() - t_x_start

            if X_df.empty:
                print(f"[BATCH {batch_idx}] empty X, skip")
                continue

            # =====================================================
            # 4️⃣ cell_id → batch row 映射
            # =====================================================
            t_map_start = time.perf_counter()

            cell_id_to_row = {cid: i for i, cid in enumerate(batch_cell_ids)}

            rows = X_df['cell_index'].map(cell_id_to_row).to_numpy()
            cols = X_df['indices'].to_numpy()
            vals = X_df['data'].to_numpy()

            t_map = time.perf_counter() - t_map_start

            # =====================================================
            # 5️⃣ COO → CSR
            # =====================================================
            t_sparse_start = time.perf_counter()

            X = sp.coo_matrix(
                (vals, (rows, cols)),
                shape=(len(batch_cell_ids), var_count)
            ).tocsr()

            t_sparse = time.perf_counter() - t_sparse_start

            # =====================================================
            # 6️⃣ obs / var
            # =====================================================
            t_obs_start = time.perf_counter()

            sub_obs = (
                self.connection.execute(f"""
                    SELECT * FROM obs
                    WHERE id IN ({ids_str})
                    ORDER BY id
                """)
                .fetch_arrow_table()
                .to_pandas()
                .iloc[:, 1:]
            )

            t_obs = time.perf_counter() - t_obs_start

            t_var_start = time.perf_counter()

            sub_var = (
                self.connection.execute("SELECT * FROM var")
                .fetch_arrow_table()
                .to_pandas()
                .iloc[:, 1:]
            )

            t_var = time.perf_counter() - t_var_start

            sub_obs.index = sub_obs['cell_id'].astype(str)
            sub_var.index = sub_var['gene_id'].astype(str)

            # =====================================================
            # 7️⃣ AnnData 构建
            # =====================================================
            t_adata_start = time.perf_counter()

            adata = AnnData(X=X, obs=sub_obs, var=sub_var)

            t_adata = time.perf_counter() - t_adata_start
            t_batch = time.perf_counter() - t_batch_start

            # =====================================================
            # 8️⃣ 打印统计
            # =====================================================
            print(
                f"[BATCH {batch_idx:04d}] "
                f"cells={len(batch_cell_ids):5d} | "
                f"X_read={t_x:.3f}s | "
                f"map={t_map:.3f}s | "
                f"sparse={t_sparse:.3f}s | "
                f"obs={t_obs:.3f}s | "
                f"var={t_var:.3f}s | "
                f"adata={t_adata:.3f}s | "
                f"total={t_batch:.3f}s"
            )

            yield adata

# todo JOIN 版 + 时间统计
    def minibatch_scan_random_no_replace_coo_join(self, batch_size: int, drop_last: bool):
        """
        随机、不放回按 cell 读取 AnnData
        - 使用 DuckDB TEMP TABLE + JOIN
        - 带完整时间统计
        """

        # =========================================================
        # 0️⃣ 全局初始化
        # =========================================================
        t_init = time.perf_counter()

        rng = np.random.default_rng()

        cell_ids = (
            self.connection
            .execute("SELECT id FROM obs")
            .fetch_arrow_table()
            .to_pandas()['id']
            .to_numpy()
        )

        var_df = (
            self.connection
            .execute("SELECT * FROM var")
            .fetch_arrow_table()
            .to_pandas()
            .iloc[:, 1:]
        )
        var_count = var_df.shape[0]
        var_df.index = var_df['gene_id'].astype(str)

        rng.shuffle(cell_ids)
        total_cells = len(cell_ids)

        if drop_last:
            num_batches = total_cells // batch_size
        else:
            num_batches = (total_cells + batch_size - 1) // batch_size

        print(
            f"[INIT] cells={total_cells}, vars={var_count}, "
            f"batches={num_batches}, init_time={time.perf_counter() - t_init:.3f}s"
        )

        # =========================================================
        # 1️⃣ batch loop
        # =========================================================
        for batch_idx in range(num_batches):

            t_batch_start = time.perf_counter()

            batch_cell_ids = cell_ids[
                batch_idx * batch_size:(batch_idx + 1) * batch_size
            ]
            if len(batch_cell_ids) == 0:
                break

            # =====================================================
            # 2️⃣ 创建 TEMP TABLE
            # =====================================================
            t_temp_start = time.perf_counter()

            self.connection.execute("DROP TABLE IF EXISTS batch_cells")
            self.connection.execute("CREATE TEMP TABLE batch_cells(id INTEGER)")

            # DuckDB 可以直接从 Python list 插入
            self.connection.execute(
                "INSERT INTO batch_cells SELECT * FROM UNNEST(?)",
                [batch_cell_ids.tolist()]
            )

            t_temp = time.perf_counter() - t_temp_start

            # =====================================================
            # 3️⃣ JOIN 读取 X（COO）
            # =====================================================
            t_x_start = time.perf_counter()

            X_df = (
                self.connection.execute("""
                    SELECT
                        x.cell_index,
                        x.indices,
                        x.data
                    FROM X_CSR_data x
                    JOIN batch_cells b
                    ON x.cell_index = b.id
                """)
                .fetch_arrow_table()
                .to_pandas()
            )

            t_x = time.perf_counter() - t_x_start

            if X_df.empty:
                print(f"[BATCH {batch_idx}] empty X")
                continue

            # =====================================================
            # 4️⃣ cell_id → row 映射
            # =====================================================
            t_map_start = time.perf_counter()

            cell_id_to_row = {cid: i for i, cid in enumerate(batch_cell_ids)}

            rows = X_df['cell_index'].map(cell_id_to_row).to_numpy()
            cols = X_df['indices'].to_numpy()
            vals = X_df['data'].to_numpy()

            t_map = time.perf_counter() - t_map_start

            # =====================================================
            # 5️⃣ COO → CSR
            # =====================================================
            t_sparse_start = time.perf_counter()

            X = sp.coo_matrix(
                (vals, (rows, cols)),
                shape=(len(batch_cell_ids), var_count)
            ).tocsr()

            t_sparse = time.perf_counter() - t_sparse_start

            # =====================================================
            # 6️⃣ JOIN 读取 obs
            # =====================================================
            t_obs_start = time.perf_counter()

            sub_obs = (
                self.connection.execute("""
                    SELECT o.*
                    FROM obs o
                    JOIN batch_cells b
                    ON o.id = b.id
                    ORDER BY o.id
                """)
                .fetch_arrow_table()
                .to_pandas()
                .iloc[:, 1:]
            )

            t_obs = time.perf_counter() - t_obs_start
            sub_obs.index = sub_obs['cell_id'].astype(str)

            # =====================================================
            # 7️⃣ AnnData
            # =====================================================
            t_adata_start = time.perf_counter()

            adata = AnnData(X=X, obs=sub_obs, var=var_df)

            t_adata = time.perf_counter() - t_adata_start
            t_batch = time.perf_counter() - t_batch_start

            # =====================================================
            # 8️⃣ 打印统计
            # =====================================================
            print(
                f"[BATCH {batch_idx:04d}] "
                f"cells={len(batch_cell_ids):5d} | "
                f"temp={t_temp:.3f}s | "
                f"X_join={t_x:.3f}s | "
                f"map={t_map:.3f}s | "
                f"sparse={t_sparse:.3f}s | "
                f"obs_join={t_obs:.3f}s | "
                f"adata={t_adata:.3f}s | "
                f"total={t_batch:.3f}s"
            )

            yield adata

    def minibatch_scan_random_replace(self,batch_size):
        """随机读取有放回模式"""

        total_num = self.query("SELECT COUNT(*) AS count FROM obs").iloc[0, 0]

        # 无限循环，直到外部中断
        i = 0
        while True:
            # 随机选择current_batch_size个索引（有放回），确保批次内索引不重复
            batch_indices = random.sample(range(total_num), batch_size)

            # 构建IN查询
            indices_str = ",".join(map(str, batch_indices))

            # 使用rowid或主键进行随机查询
            if (self.isView):  # 数据库视图
                sub_X = self.query(f"SELECT * FROM {self.viewID} WHERE rowid IN ({indices_str})")
            else:
                sub_X = self.query(f"SELECT * FROM X WHERE rowid IN ({indices_str})")
            sub_obs = self.query(f"SELECT * FROM obs WHERE rowid IN ({indices_str})")
            sub_var = self.query("SELECT * FROM var")

            sub_X = sub_X.iloc[:, 2:]  # 去掉数据表中的前2列 id cell_id
            sub_obs = sub_obs.iloc[:, 1:]  # 去掉数据表中的前1列 id
            sub_var = sub_var.iloc[:, 1:]  # 去掉数据表中的前1列
            cell_id = sub_obs['cell_id']  # 获取cell_id
            gene_id = sub_var['gene_id']  # 获取gene_id

            # 创建AnnData对象，正确设置obs_names和var_names
            adata = AnnData(
                X=sub_X.values,
                obs=sub_obs,
                var=sub_var
            )
            adata.obs_names = cell_id  # 设置观测名为cell_id
            adata.var_names = gene_id  # 设置变量名为gene_id

            # 返回数据
            yield adata
            i += 1

    # == todo==== 数据库 配置优化 和 建立索引 ， 待修改 =========
    def comprehensive_database_optimization(self):
        """综合数据库优化方案 - 用户手动调用"""
        print("开始执行数据库综合优化...")

        # 1. 数据库配置优化
        self.optimize_database_settings()

        # 2. 创建核心索引
        self.create_all_indexes()
        print("数据库综合优化完成!")

    def optimize_database_settings(self):
        """优化数据库设置以提高查询性能"""
        try:
            # DuckDB特定优化
            self.connection.execute("PRAGMA enable_progress_bar=true")
            self.connection.execute("PRAGMA threads=8")  # 使用多线程
            self.connection.execute("PRAGMA memory_limit='48GB'")  # 增加内存限制

            print("数据库配置优化完成")
        except Exception as e:
            print(f"数据库配置优化时出错: {e}")

    def create_all_indexes(self):
        """创建所有表的索引"""

        """ 待添加的索引
            CREATE INDEX IF NOT EXISTS idx_x_id ON X_CSR_data(id);
            CREATE INDEX IF NOT EXISTS idx_x_cell ON X_CSR_data(cell_index);
            CREATE INDEX IF NOT EXISTS idx_obs_id ON obs(id);
        """

        # 所有索引按阶段分组
        all_indexes = {
            "第一阶段：核心索引": [
                # 原表索引
                "CREATE INDEX IF NOT EXISTS idx_x_id ON X(id)",
                "CREATE INDEX IF NOT EXISTS idx_obs_cell_id ON obs(cell_id)",
                "CREATE INDEX IF NOT EXISTS idx_var_gene_id ON var(gene_id)",

                # CSR表核心索引
                "CREATE INDEX IF NOT EXISTS idx_csr_indptr_cell ON X_CSR_indptr(cell_id)",
                "CREATE INDEX IF NOT EXISTS idx_csr_data_indices ON X_CSR_data(indices)"
            ],

            "第二阶段：复合索引": [
                # 原表复合索引
                "CREATE INDEX IF NOT EXISTS idx_x_id_cell_id ON X(id, cell_id)",
                "CREATE INDEX IF NOT EXISTS idx_obs_cell_id_id ON obs(cell_id, id)",

                # CSR表复合索引
                "CREATE INDEX IF NOT EXISTS idx_csr_indptr_combo ON X_CSR_indptr(cell_id, indptr)",
                "CREATE INDEX IF NOT EXISTS idx_csr_data_combo ON X_CSR_data(indices, data)"
            ],

            "第三阶段：辅助索引": [
                # 原表辅助索引
                "CREATE INDEX IF NOT EXISTS idx_x_cell_id ON X(cell_id)",
                "CREATE INDEX IF NOT EXISTS idx_obs_id ON obs(id)",
                "CREATE INDEX IF NOT EXISTS idx_var_id ON var(id)",

                # CSR表辅助索引
                "CREATE INDEX IF NOT EXISTS idx_csr_indptr_id ON X_CSR_indptr(id)",
                "CREATE INDEX IF NOT EXISTS idx_csr_data_id ON X_CSR_data(id)",
                "CREATE INDEX IF NOT EXISTS idx_csr_indptr_ptr ON X_CSR_indptr(indptr)"
            ],

            "第四阶段：CSR专用索引": [
                # CSR表专用优化索引
                "CREATE INDEX IF NOT EXISTS idx_csr_data_covering ON X_CSR_data(indices, id, data)",
                "CREATE INDEX IF NOT EXISTS idx_csr_data_nonzero ON X_CSR_data(indices) WHERE data > 0",
                "CREATE INDEX IF NOT EXISTS idx_csr_indptr_range ON X_CSR_indptr(indptr, cell_id)",
                "CREATE INDEX IF NOT EXISTS idx_csr_data_data_idx ON X_CSR_data(data, indices)"
            ]
        }

        # 按阶段执行
        for phase_name, indexes in all_indexes.items():
            print(f"{phase_name}...")
            for sql in indexes:
                try:
                    self.connection.execute(sql)
                except Exception as e:
                    print(f"执行失败: {sql[:60]}..., 错误: {e}")

        print("所有索引创建完成！")

# == todo==== atlas 的拼接  =========

    def combine(atlases: List['Atlas'], how: Literal["inner", "outer"] = "inner") -> 'Atlas':
        """
        将多个数据集拼接成一个大数据集

        Args:
            atlases: Atlas对象列表
            how: 合并方式
                - "inner": 只保留所有atlas都存在的基因
                - "outer": 保留所有基因的并集，不存在的值补NaN

        Returns:
            新的Atlas对象，包含合并后的数据
        """
        if not atlases:
            raise ValueError("至少需要提供一个Atlas对象")

        if len(atlases) == 1:
            logger.warning("只有一个Atlas对象，无需合并")
            return atlases[0]

        logger.info(f"开始合并 {len(atlases)} 个Atlas对象，合并方式: {how}")

        # 获取所有数据集的基因列表
        all_genes = []
        for i, atlas in enumerate(atlases):
            try:
                # 使用DuckDB查询获取基因列表
                result = atlas.connection.execute("SELECT gene_name FROM var")
                genes = [row[0] for row in result.fetchall()]
                all_genes.append(set(genes))
                logger.info(f"数据集 {i} 有 {len(genes)} 个基因")
            except Exception as e:
                logger.error(f"获取数据集 {i} 的基因列表时出错: {e}")
                raise

        # 确定合并后的基因列表
        if how == "inner":
            # 取所有数据集的基因交集
            common_genes = set.intersection(*all_genes)
            logger.info(f"取基因交集，共有 {len(common_genes)} 个共同基因")
        elif how == "outer":
            # 取所有数据集的基因并集
            common_genes = set.union(*all_genes)
            logger.info(f"取基因并集，共有 {len(common_genes)} 个基因")
        else:
            raise ValueError(f"不支持的合并方式: {how}")

        # 创建新的数据库文件
        temp_dir = tempfile.gettempdir()
        new_db_path = os.path.join(temp_dir, f"combined_atlas_{os.getpid()}.db")

        try:
            # 创建新的DuckDB连接
            new_conn = duckdb.connect(new_db_path)

            # 创建表结构
            new_conn.execute("""
                CREATE TABLE obs (
                    cell_id VARCHAR PRIMARY KEY,
                    atlas_source VARCHAR
                )
            """)

            new_conn.execute("""
                CREATE TABLE var (
                    gene_name VARCHAR PRIMARY KEY
                )
            """)

            # 创建表达矩阵表
            genes_list = sorted(common_genes)
            columns_def = ", ".join([f'"{gene}" DOUBLE' for gene in genes_list])
            new_conn.execute(f"""
                CREATE TABLE X (
                    cell_id VARCHAR PRIMARY KEY,
                    {columns_def}
                )
            """)

            # 插入基因信息
            for gene in genes_list:
                new_conn.execute("INSERT INTO var (gene_name) VALUES (?)", (gene,))

            # 合并数据
            total_cells = 0

            for i, atlas in enumerate(atlases):
                logger.info(f"处理数据集 {i}...")

                # 获取当前数据集的细胞ID
                result = atlas.connection.execute("SELECT cell_id FROM obs")
                cell_ids = [row[0] for row in result.fetchall()]

                # 为每个细胞添加来源标识
                source_cell_ids = [f"atlas_{i}_{cell_id}" for cell_id in cell_ids]

                # 插入细胞信息
                for cell_id in source_cell_ids:
                    new_conn.execute(
                        "INSERT INTO obs (cell_id, atlas_source) VALUES (?, ?)",
                        (cell_id, f"atlas_{i}")
                    )

                # 处理表达矩阵
                if how == "inner":
                    # 对于inner合并，只选择共同基因
                    genes_str = ", ".join([f'"{gene}"' for gene in genes_list])
                    query = f"SELECT cell_id, {genes_str} FROM X"
                else:
                    # 对于outer合并，选择所有基因，不存在的用NULL
                    genes_select = []
                    for gene in genes_list:
                        if gene in all_genes[i]:
                            genes_select.append(f'"{gene}"')
                        else:
                            genes_select.append("NULL")  # 不存在的基因用NULL填充

                    genes_str = ", ".join(genes_select)
                    query = f"SELECT cell_id, {genes_str} FROM X"

                try:
                    result = atlas.connection.execute(query)
                    rows = result.fetchall()

                    # 插入表达数据
                    for j, row in enumerate(rows):
                        cell_id = source_cell_ids[j]
                        values = [cell_id] + list(row[1:])  # 跳过原始cell_id，使用新的cell_id

                        # 构建INSERT语句
                        placeholders = ", ".join(["?"] * len(values))
                        new_conn.execute(f"INSERT INTO X VALUES ({placeholders})", values)

                    total_cells += len(rows)
                    logger.info(f"数据集 {i} 处理完成，添加了 {len(rows)} 个细胞")

                except Exception as e:
                    logger.error(f"处理数据集 {i} 的表达矩阵时出错: {e}")
                    raise

            logger.info(f"合并完成，总共 {total_cells} 个细胞，{len(genes_list)} 个基因")

            new_atlas = Atlas("combined_atlas", new_db_path)
            new_atlas.connection = new_conn

            return new_atlas

        except Exception as e:
            logger.error(f"合并过程中出错: {e}")
            # 清理临时文件
            if os.path.exists(new_db_path):
                os.remove(new_db_path)
            raise

# == todo ==== minibatch的新方案  =========
#       多进程（子进程 读取数据 ， 主进程消费数据） +
#       共享内存 +
#       （原来的自己写的 多线程 + 普通 queue 读取，每个线程对应一个 conn , 读取minibatch大小的数据）


# todo 第一步： 可以先完成 多线程 + 普通 queue 读取，每个线程对应一个 conn , 读取minibatch大小的数据；
    # todo 多线程 + 普通 queue 读取，每个线程对应一个 conn , 读取minibatch大小的数据；
# =========================
# minibatch_scan_order_mthread 保持原结构
# =========================
    def minibatch_scan_order_mthread_o(self, batch_size, drop_last):
        """按顺序读取模式（X_CSR 多线程版本）"""
#         time_2 = time_3 = time_4 = 0
#
#         self.connect("r")
#
#         total_num = self.connection.execute("SELECT COUNT(*) AS count FROM obs").fetchdf().iloc[0, 0] # 获取数量
#
#         if drop_last:
#             num_batches = total_num // batch_size
#         else:
#             num_batches = (total_num + batch_size - 1) // batch_size
#
#         # ========= ✅ 使用多线程 X_CSR Reader =========
#         csr_reader = MultiThreadCSRReader(
#             db_path=self.db_file,  # 使用 db_path 让线程每个都新建只读连接
#             total_num_cells=total_num,
#             batch_size=batch_size,
#             n_threads=15,
#             queue_size=50,
#             drop_last=drop_last,
#         )
#
#         for i, cur_bs, CSR_indptr_array, CSR_indices_array, CSR_data_array in (
#                 (item["batch_id"], item["batch_size"], item["indptr"], item["indices"], item["data"])
#                 for item in csr_reader
#         ):
#             offset = i * batch_size
#
#             print(f"\n--- 批次 {i} 详细时间分析 ---")
#             print(f"--- 批次大小 {cur_bs} ---")
#
#             # ========= obs =========
#             start_time2 = datetime.now()
#             sub_obs = self.connection.execute(f"SELECT * FROM obs LIMIT {cur_bs} OFFSET {offset}").fetchdf()
#             time_2 += (datetime.now() - start_time2).total_seconds()
#
#             # ========= var =========
#             start_time3 = datetime.now()
#             sub_var = self.connection.execute("SELECT * FROM var").fetchdf()
#             time_3 += (datetime.now() - start_time3).total_seconds()
#
#             # ========= AnnData =========
#             start_time4 = datetime.now()
#
#             # ========= 重置索引 =========
#             sub_obs = sub_obs.reset_index(drop=True)
#             sub_var = sub_var.reset_index(drop=True)
#
#             cell_id = sub_obs["cell_id"]
#             gene_id = sub_var["gene_id"]
#
#             print(f"indptr[-1] {CSR_indptr_array[-1]}")
#             print(f"len(CSR_data_array) {len(CSR_data_array)}")
#             print(f"len(CSR_indices_array) {len(CSR_indices_array)}")
#             print(f"len(CSR_indptr_array) {len(CSR_indptr_array)}")
#
#             X = sp.coo_matrix(
#                 (CSR_data_array, (np.repeat(np.arange(cur_bs), np.diff(CSR_indptr_array)), CSR_indices_array)),
#                 shape=(cur_bs, len(sub_var))
#             ).tocsr()
#
#             adata = AnnData(X=X, obs=sub_obs, var=sub_var)
#             adata.obs_names = cell_id.astype(str)
#             adata.var_names = gene_id.astype(str)
#
#             time_4 += (datetime.now() - start_time4).total_seconds()
#
#             yield adata
#
#             print(f"批次 {i}: nnz = {len(CSR_data_array)}, shape = ({cur_bs}, {len(sub_var)})")
#
#             print(f"obs 表读取用时： {time_2:.2f} 秒")
#             print(f"var 表读取用时： {time_3:.2f} 秒")
#             print(f"生成 anndata 用时： {time_4:.2f} 秒")

    def minibatch_scan_order_mthread(self, batch_size, drop_last=False, n_threads=10, queue_size=50):
        """
        按顺序读取模式（X_CSR 多线程版本），直接返回 AnnData

        参数:
        - batch_size: 每个 batch 的细胞数
        - drop_last: 是否丢弃最后不足 batch_size 的 batch
        - n_threads: worker 线程数
        - queue_size: 队列大小，控制背压

        返回:
        - yield AnnData 对象
        """

        from datetime import datetime
        import scipy.sparse as sp
        import numpy as np
        from anndata import AnnData

        self.connect("r")

        # 1️⃣ 总细胞数
        total_num = self.connection.execute("SELECT COUNT(*) AS count FROM obs").fetchdf().iloc[0, 0]

        # 2️⃣ 初始化多线程 CSR 读取器
        csr_reader = MultiThreadCSRReader(
            db_path=self.db_file,
            total_num_cells=total_num,
            batch_size=batch_size,
            n_threads=24,
            queue_size=50,
            drop_last=drop_last
        )

        # 3️⃣ 计时
        time_obs = 0
        time_var = 0
        time_adata = 0

        for item in csr_reader:
            batch_id = item["batch_id"]
            cur_bs = item["batch_size"]
            CSR_indptr_array = item["indptr"]
            CSR_indices_array = item["indices"]
            CSR_data_array = item["data"]

            offset = batch_id * batch_size

            print(f"\n--- 批次 {batch_id} 详细时间分析 ---")
            print(f"--- 批次大小 {cur_bs} ---")
            print(f"indptr[-1] {CSR_indptr_array[-1]}")
            print(f"len(CSR_data_array) {len(CSR_data_array)}")
            print(f"len(CSR_indices_array) {len(CSR_indices_array)}")
            print(f"len(CSR_indptr_array) {len(CSR_indptr_array)}")

            # ========= obs =========
            t0 = datetime.now()
            sub_obs = self.connection.execute(f"SELECT * FROM obs LIMIT {cur_bs} OFFSET {offset}").fetchdf()
            time_obs += (datetime.now() - t0).total_seconds()

            # ========= var =========
            t1 = datetime.now()
            sub_var = self.connection.execute("SELECT * FROM var").fetchdf()
            time_var += (datetime.now() - t1).total_seconds()

            # ========= AnnData =========
            t2 = datetime.now()

            # sub_obs = sub_obs.reset_index(drop=True)
            # sub_var = sub_var.reset_index(drop=True)

            cell_id = sub_obs["cell_id"]
            gene_id = sub_var["gene_id"]

            sub_obs.index = cell_id.astype(str)
            sub_var.index = gene_id.astype(str)

            # 构建 COO 矩阵
            X = sp.coo_matrix(
                (CSR_data_array, (np.repeat(np.arange(cur_bs), np.diff(CSR_indptr_array)), CSR_indices_array)),
                shape=(cur_bs, len(sub_var))
            )

            adata = AnnData(X=X, obs=sub_obs, var=sub_var)
            adata.obs_names = cell_id.astype(str)
            adata.var_names = gene_id.astype(str)

            time_adata += (datetime.now() - t2).total_seconds()

            yield adata

            print(f"批次 {batch_id}: nnz={len(CSR_data_array)}, shape=({cur_bs}, {len(sub_var)})")
            print(f"obs 表读取累计用时: {time_obs:.2f} 秒")
            print(f"var 表读取累计用时: {time_var:.2f} 秒")
            print(f"生成 AnnData 累计用时: {time_adata:.2f} 秒")

        # 等待 worker 线程结束
        csr_reader.join()
        print("所有批次处理完成，worker 已退出")


# import threading
# from queue import Queue
# import numpy as np
# import scipy.sparse as sp
# from datetime import datetime
# from anndata import AnnData
# import duckdb
#
#
# class MultiThreadCSRReader:
#     """
#     多线程 CSR 读取器（线程安全，每个线程独立只读连接）
#     """
#
#     def __init__(self, db_path, total_num_cells, batch_size, n_threads=10, queue_size=50, drop_last=False):
#         self.db_path = db_path
#         self.total_num_cells = total_num_cells
#         self.batch_size = batch_size
#         self.n_threads = n_threads
#         self.drop_last = drop_last
#
#         self.queue = Queue(maxsize=queue_size)
#
#         if drop_last:
#             self.num_batches = total_num_cells // batch_size
#         else:
#             self.num_batches = (total_num_cells + batch_size - 1) // batch_size
#
#         # 全局批次游标锁
#         self._batch_cursor = 0
#         self._cursor_lock = threading.Lock()
#
#         self.threads = [] # 通常这是一个 线程列表
#         self._start_threads()
#
#     # -----------------------------
#     # 启动线程
#     # -----------------------------
#     def _start_threads(self):
#         for tid in range(self.n_threads):
#             t = threading.Thread(
#                 target=self._worker,
#                 name=f"CSR-Worker-{tid}",
#                 daemon=True
#             ) # todo 此时：NEW 状态
#             self.threads.append(t)
#             t.start() #  todo start 之后，线程进入 READY
#
#     # -----------------------------
#     # 每个线程工作函数
#     # -----------------------------
#     def _worker(self):
#         """
#         每个线程工作函数：独立只读连接，多线程安全
#         """
#         import duckdb
#         import numpy as np
#         import scipy.sparse as sp
#         import threading
#         import time
#
#         thread_name = threading.current_thread().name
#
#         # 每个线程独立只读连接
#         conn = duckdb.connect(self.db_path, read_only=True)
#
#         while True:
#             # -----------------------------
#             # 1️⃣ 抢批次
#             # -----------------------------
#             with self._cursor_lock:
#                 if self._batch_cursor >= self.num_batches:
#                     print(f"[{thread_name}] 所有批次抢完，线程退出")
#                     # 所有批次抢完，向队列推送终止信号
#                     self.queue.put(None)
#                     return
#                 batch_id = self._batch_cursor
#                 self._batch_cursor += 1
#                 print(f"[{thread_name}] 抢到批次 {batch_id}, 当前全局批次游标 {self._batch_cursor}")
#
#             offset = batch_id * self.batch_size
#             if (not self.drop_last and batch_id == self.num_batches - 1
#                     and self.total_num_cells % self.batch_size != 0):
#                 current_batch_size = self.total_num_cells % self.batch_size
#             else:
#                 current_batch_size = self.batch_size
#
#             # -----------------------------
#             # 2️⃣ 读取 indptr
#             # -----------------------------
#             query_indptr = f"""
#                 SELECT indptr
#                 FROM X_CSR_indptr
#                 LIMIT {current_batch_size}
#                 OFFSET {offset}
#             """
#             indptr_df = conn.execute(query_indptr).fetch_arrow_table().to_pandas()
#             raw_indptr = indptr_df["indptr"].to_numpy()
#
#             # -----------------------------
#             # 3️⃣ 读取 CSR_data (indices + data + cell_index)
#             # -----------------------------
#             if current_batch_size > 0:
#                 query_data = f"""
#                     SELECT indices, data, cell_index
#                     FROM X_CSR_data
#                     WHERE cell_index >= {offset} AND cell_index < {offset + current_batch_size}
#                     ORDER BY id
#                 """
#                 data_df = conn.execute(query_data).fetch_arrow_table().to_pandas()
#
#                 rows = data_df["cell_index"].to_numpy()
#                 cols = data_df["indices"].to_numpy()
#                 vals = data_df["data"].to_numpy()
#
#                 # ✅ 计算 batch 内偏移
#                 rows -= offset
#             else:
#                 rows = np.array([], dtype=np.int32)
#                 cols = np.array([], dtype=np.int32)
#                 vals = np.array([], dtype=np.float32)
#
#             # -----------------------------
#             # 4️⃣ 构建 COO → CSR
#             # -----------------------------
#             var_count = conn.execute("SELECT COUNT(*) FROM var").fetchone()[0]
#             X = sp.coo_matrix(
#                 (vals, (rows, cols)),
#                 shape=(current_batch_size, var_count)
#             ).tocsr()
#
#             # -----------------------------
#             # 5️⃣ 推送到队列
#             # -----------------------------
#             self.queue.put({
#                 "batch_id": batch_id,
#                 "indptr": X.indptr,
#                 "indices": X.indices,
#                 "data": X.data,
#                 "batch_size": current_batch_size,
#             })
#             print(f"[{thread_name}] 批次 {batch_id} 放入队列, 当前队列大小 {self.queue.qsize()}")
#
#     # -----------------------------
#     # 对外消费接口
#     # -----------------------------
#     def __iter__(self):
#         return self
#
#     def __next__(self):
#         """
#         从队列获取下一个批次，如果队列中取到 None，则迭代结束
#         """
#         while True:
#             item = self.queue.get()  # 阻塞直到有数据
#             if item is None:
#                 # 将 None 再放回队列，保证其他线程也能停止迭代
#                 self.queue.put(None)
#                 print("[ITER] 队列收到终止信号，StopIteration")
#                 raise StopIteration
#             print(f"[ITER] 取到批次 {item['batch_id']}, 当前队列大小 {self.queue.qsize()}")
#             return item
#
#     def join(self):
#         # 阻塞当前线程，直到 self.threads 里的所有线程都执行完毕。
#         for t in self.threads: # 依次取出每一个线程对象 t
#             t.join() # 线程会进入阻塞状态，直到目标线程完成执行。
#             # 当前线程（通常是主线程



import threading
from queue import Queue
import numpy as np
import scipy.sparse as sp
from anndata import AnnData
from datetime import datetime
import duckdb

class MultiThreadCSRReader:
    """
    多线程 CSR 读取器（线程安全、顺序输出、不阻塞）

    特点：
    1. worker 线程可以乱序生成 batch
    2. 主线程顺序输出 batch，保证 batch_id 0,1,2,... 顺序
    3. 使用 Queue 做线程间通信
    4. 支持 drop_last 选项
    """

    def __init__(self, db_path, total_num_cells, batch_size,
                 n_threads=10, queue_size=50, drop_last=False):
        self.db_path = db_path
        self.total_num_cells = total_num_cells
        self.batch_size = batch_size
        self.n_threads = n_threads # 线程数
        self.drop_last = drop_last

        # 内部队列：线程间通信，存放 batch 数据
        self.queue = Queue(maxsize=queue_size)

        # 计算总批次数
        if drop_last:
            self.num_batches = total_num_cells // batch_size
        else:
            self.num_batches = (total_num_cells + batch_size - 1) // batch_size

        # 全局批次游标及锁，确保每个线程抢到不同 batch
        self._batch_cursor = 0
        self._cursor_lock = threading.Lock()

        # 保存线程对象
        self.threads = [] # 线程队列
        self._start_threads()

    # -----------------------------
    # 启动 worker 线程
    # -----------------------------
    def _start_threads(self):
        for tid in range(self.n_threads):
            t = threading.Thread(
                target=self._worker,
                name=f"CSR-Worker-{tid}",
                daemon=True  # 主线程退出，worker 自动结束
            )
            self.threads.append(t) # 将t加入到 线程队列
            t.start()

    # -----------------------------
    # worker 函数：负责生成 batch，放入队列
    # -----------------------------
    def _worker(self):
        """
        每个线程独立只读连接，多线程安全
        """
        name = threading.current_thread().name
        print(f"name 是什么？ [{name}]  线程编号 ？")
        conn = duckdb.connect(self.db_path, read_only=True)

        while True:
            # -----------------------------
            # 1️⃣ 抢批次（全局锁）
            # -----------------------------
            with self._cursor_lock:
                if self._batch_cursor >= self.num_batches:
                    # 没有更多 batch 了
                    print(f"[{name}] no more batch, exit")
                    return
                batch_id = self._batch_cursor
                self._batch_cursor += 1

            # -----------------------------
            # 2️⃣ 计算当前 batch 偏移量和大小
            # -----------------------------
            offset = batch_id * self.batch_size
            if (not self.drop_last and batch_id == self.num_batches - 1
                    and self.total_num_cells % self.batch_size != 0):
                cur_bs = self.total_num_cells % self.batch_size
            else:
                cur_bs = self.batch_size

            # -----------------------------
            # 3️⃣ 读取 CSR 数据 (indices, data, cell_index)
            # -----------------------------
            if cur_bs > 0:
                query = f"""
                    SELECT indices, data, cell_index
                    FROM X_CSR_data
                    WHERE cell_index >= {offset} AND cell_index < {offset + cur_bs}
                    ORDER BY id
                """
                df = conn.execute(query).fetch_arrow_table().to_pandas()

                rows = df["cell_index"].to_numpy() - offset  # batch 内行号， cell_id
                cols = df["indices"].to_numpy() # gene_id
                vals = df["data"].to_numpy() # data
            else:
                rows = cols = vals = np.array([])

            # -----------------------------
            # 4️⃣ 生成 CSR 矩阵
            # -----------------------------
            var_count = conn.execute("SELECT COUNT(*) FROM var").fetchone()[0]


            X = sp.coo_matrix(
                (vals, (rows, cols)),
                shape=(cur_bs, var_count)
            ).tocsr()

            # -----------------------------
            # 5️⃣ 放入队列
            # -----------------------------
            self.queue.put({
                "batch_id": batch_id,
                "indptr": X.indptr,
                "indices": X.indices,
                "data": X.data,
                "batch_size": cur_bs,
            })

            print(f"[{name}] produced batch {batch_id}, queue size={self.queue.qsize()}")

    # -----------------------------
    # 顺序迭代器：保证按 batch_id 输出
    # -----------------------------
    def __iter__(self):
        expected = 0  # 当前期望输出的批次ID（从0开始）
        buffer = {}  # 缓存乱序收到的批次 {batch_id: 数据}

        while expected < self.num_batches:
            # 1. 从队列取数据（阻塞，直到有数据）
            item = self.queue.get()
            bid = item["batch_id"]
            buffer[bid] = item  # 存入缓存

            # 2. 连续输出所有“已就绪”的批次
            while expected in buffer:
                out = buffer.pop(expected)  # 取出期望的批次
                yield out  # 输出
                expected += 1  # 期望下一个批次
# todo 假设队列里先入队 batch3，再入队 batch1，再入队 batch2：
#     第一次取到 batch3 → 存入 buffer {3: 数据} → expected=0 不在 buffer 中，继续等；
#     第二次取到 batch1 → 存入 buffer {3: 数据，1: 数据} → expected=0 仍不在 buffer 中，继续等；
#     第三次取到 batch2 → 存入 buffer {3: 数据，1: 数据，2: 数据} → expected=0 仍不在 buffer 中，继续等；
#     第四次取到 batch0 → 存入 buffer {0: 数据，1: 数据，2: 数据，3: 数据}；
#     此时检查 expected=0 在 buffer 中 → 输出 batch0，expected 变为 1；
#     继续检查 expected=1 在 buffer 中 → 输出 batch1，expected 变为 2；
#     继续检查 expected=2 在 buffer 中 → 输出 batch2，expected 变为 3；
#     继续检查 expected=3 在 buffer 中 → 输出 batch3，expected 变为 4；
#     直到 expected 达到总批次数，迭代结束。

    # -----------------------------
    # 等待所有线程结束
    # -----------------------------
    def join(self):
        for t in self.threads:
            t.join()













































