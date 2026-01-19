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
import uuid
import tempfile
import matplotlib.pyplot as plt
import scipy.sparse as sp

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

#
# import threading
# import queue
#
#
# class PrefetchGenerator:
#     """
#     通用异步 prefetch 生成器
#     后台线程提前生成 batch，主线程只消费
#     """
#
#     def __init__(self, generator, prefetch=8):
#         self.generator = generator
#         self.queue = queue.Queue(maxsize=prefetch)
#         self._stop_token = object()
#
#         self.thread = threading.Thread(
#             target=self._worker,
#             daemon=True
#         )
#         self.thread.start()
#
#     def _worker(self):
#         try:
#             for item in self.generator:
#                 self.queue.put(item)
#         finally:
#             self.queue.put(self._stop_token)
#
#     def __iter__(self):
#         return self
#
#     def __next__(self):
#         item = self.queue.get()
#         if item is self._stop_token:
#             raise StopIteration
#         return item
#


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
        self.__isView = False # False，不是视图；True,是视图。

        self.__viewID = None  # 数据库创建视图时使用

        self.__obs_cell_id = []  # 存储当前atlas对象符合条件的cell id列表
        self.__var_gene_id = []  # 存储当前atlas对象符合条件的gene id列表

        logger.debug(f"实例变量已设置 - name: {self.__name}, path: {self.__path}")

        # 检查数据库文件是否存在，如果不存在则创建
        db_file = os.path.join(self.__path, f"{self.__name}.sasql")
        logger.debug(f"数据库文件路径: {db_file}")

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

        # 初始化时加载X表的结构信息，不加载内容

        logger.info("Atlas 实例初始化完成")

    # todo 用使用数据库视图
    # def __getitem__(self, item):
    #
    #     logger.info(f"正在创建视图...")
    #     # 创建视图ID 当前时间并格式化为年月日_时分秒微秒精度格式,
    #     current_datetime = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    #     # view_ID = f"X_view_{current_datetime}"
    #     # view_ID = self.get_class_id()
    #     view_ID = "X_view_" + str(uuid.uuid4()).replace('-', '_')# 类定义时生成唯一 UUID
    #
    #     base_table = self.viewID if self.__isView else "X"  # 基于视图还是原表
    #
    #     if isinstance(item, int):
    #         create_new_view = f"CREATE OR REPLACE VIEW {view_ID} AS SELECT * FROM {base_table} LIMIT 1 OFFSET {item}"
    #
    #     elif isinstance(item, slice):
    #         start = item.start or 0
    #         limit = "ALL" if item.stop is None else item.stop - start
    #         create_new_view = f"CREATE OR REPLACE VIEW {view_ID} AS SELECT * FROM {base_table} LIMIT {limit} OFFSET {start}"
    #
    #     elif isinstance(item, str):
    #         create_new_view = f"CREATE OR REPLACE VIEW {view_ID} AS SELECT * FROM {base_table} WHERE cell_id = '{item}'"
    #
    #     else:
    #         raise TypeError('支持的类型: int, slice, str')
    #
    #     # 执行查询
    #     self.execute_sql(create_new_view)
    #
    #     # 调试信息 - 只显示手动创建的视图
    #     result = self.execute_sql(
    #         "SELECT table_name FROM information_schema.views WHERE table_name LIKE 'X_view_%'")
    #     df = result.fetchdf()
    #     print("手动创建的视图的数量:", len(df))
    #     print("手动创建的视图列表:", df['table_name'].tolist())
    #
    #     #创建新的Atlas对象，使用相同的数据库但不同的表结构
    #     new_atlas = Atlas(self.__name, self.__path)
    #     new_atlas.__isView = True
    #     new_atlas.__viewID = view_ID
    #     return new_atlas

    # # todo 用管理obs_cell_id = [] 的 方式管理视图，不使用数据库视图
    def __getitem__(self, item):
        logger.info(f"正在创建atlas实例视图...")

        # 执行SQL查询获取所有细胞ID
        query = "SELECT cell_id FROM obs"
        result = self.execute_sql(query).fetchall()
        all_cell_ids = [row[0] for row in result] if result else []

        __obs_cell_id = []  # 存储当前atlas对象符合条件的id列表

        if isinstance(item, int):
            if 0 <= item < len(all_cell_ids):
                __obs_cell_id = [all_cell_ids[item]]
            else:
                logger.warning(f"索引 {item} 超出范围")
                __obs_cell_id = []

        elif isinstance(item, slice):
            # 处理切片
            selected_ids = all_cell_ids[item]
            __obs_cell_id = selected_ids if isinstance(selected_ids, list) else [selected_ids]

        elif isinstance(item, str):
            # 检查细胞ID是否存在
            if item in all_cell_ids:
                __obs_cell_id = [item]
                logger.info(f"找到细胞ID: {item}")
            else:
                logger.warning(f"不存在该细胞ID: {item}")
                __obs_cell_id = []

        elif isinstance(item, list):
            # 处理列表，筛选存在的细胞ID
            __obs_cell_id = [cell_id for cell_id in item if cell_id in all_cell_ids]
            if len(__obs_cell_id) < len(item):
                missing_ids = set(item) - set(__obs_cell_id)
                logger.warning(f"以下细胞ID不存在: {missing_ids}")

        else:
            raise TypeError('支持的类型: int, slice, str, list')

        # 创建新的atlas对象视图
        new_atlas = Atlas(self.__name, self.__path)
        new_atlas.__obs_cell_id = __obs_cell_id
        new_atlas.__isView = True

        logger.info(f"创建视图完成，包含 {len(__obs_cell_id)} 个细胞")
        return new_atlas

    # def __enter__(self):
    #     """上下文管理器入口"""
    #     logger.debug("进入上下文管理器")
    #     return self

    # def __exit__(self, exc_type, exc_val, exc_tb):
    #     """上下文管理器出口，确保连接关闭"""
    #     logger.debug("退出上下文管理器")
    #     if exc_type is not None:
    #         logger.warning(f"上下文管理器退出时发生异常: {exc_type.__name__}: {exc_val}")
    #     self.close()

    def __del__(self):
        """
        析构函数，当对象被销毁时自动删除对应的数据库视图
        """
        try:
            # 只有当该对象是基于视图时才需要清理
            if self.__isView and self.__viewID:
                logger.info(f"正在清理视图: {self.__viewID}")

                # 连接数据库执行删除操作
                if self.__connection is None:
                    self.connect("r+")

                # 删除视图
                drop_query = f"DROP VIEW IF EXISTS {self.__viewID}"
                self.__connection.execute(drop_query)
                self.__connection.commit()

                logger.info(f"视图删除成功: {self.__viewID}")

                # 关闭连接
                # self.close()

        except Exception as e:
            # 在析构函数中避免抛出异常，只记录日志
            logger.error(f"清理视图时发生错误: {str(e)}")

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

    @property
    def isView(self) -> bool:
        return self.__isView

    @isView.setter
    def isView(self, value: bool) -> None:
        self.__isView = value

    @property
    def viewID(self) -> str:
        return self.__viewID

    @viewID.setter
    def viewID(self, value: str) -> None:
        self.__viewID = value

    @property
    def obs_cell_id(self) -> list:
        return self.__obs_cell_id
    @obs_cell_id.setter
    def obs_cell_id(self, value: list) -> None:
        self.__obs_cell_id = value

    @property
    def var_gene_id(self) -> list:
        return self.__var_gene_id
    @var_gene_id.setter
    def var_gene_id(self, value: list) -> None:
        self.__var_gene_id = value

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
        db_file = os.path.join(self.__path, f"{self.__name}.sasql")
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
    def query_minibatch(self, mode="order", batch_size=2048, drop_last=True,table_name = "X"):
        """
        minibatch查询
        以minibatch方式遍历整个数据集，返回的数据用anndata封装
        :param mode: order - 按顺序读取, random_replace - 随机读取有放回, random_no_replace - 随机读取不放回
        :param batch_size: 每次读取的大小
        :param drop_last: 是否丢弃最后不足batch_size的数据
        :return: 生成器，每次返回AnnData对象
        """
        if(self.isView):# 数据库视图
            # 获取总数据量
            total_num = self.query(f"SELECT COUNT(*) as count FROM {self.viewID}").iloc[0, 0]
        else:
            total_num = self.query("SELECT COUNT(*) as count FROM X").iloc[0, 0]

        if mode == "order":
            # 按顺序读取模式
            # return self.minibatch_scan_order(total_num, batch_size, drop_last)
            return self.minibatch_scan_order_cursor_csr_df_arrow_mega_obs_view(total_num, batch_size, drop_last) # CSR 读取
            # return self.minibatch_scan_order_cursor_csr(total_num, batch_size, drop_last)  # CSR 读取 self
            # return self.minibatch_scan_order_cursor_csr_df_arrow_onlylie(total_num, batch_size, drop_last)
            # return self.minibatch_scan_order_cursor(total_num, batch_size = batch_size,drop_last=drop_last) # 综合优化：使用游标分页替代OFFSET分页
            # return self.minibatch_scan_parallel(total_num, batch_size, drop_last) # 优化2：并行查询优化
            # return self.minibatch_scan_db_optimized(total_num, batch_size, drop_last) # 优化3：使用 DuckDB 数据库特定优化



        elif mode == "random_no_replace":
            # 随机读取不放回模式
            return self.minibatch_scan_random_no_replace(total_num, batch_size, drop_last)

        elif mode == "random_replace":
            # 随机读取有放回模式
            return self.minibatch_scan_random_replace(total_num, batch_size)

        else:
            raise ValueError(f"不支持的扫描模式: {mode}")

    def minibatch_scan_order(self, total_num, batch_size, drop_last):
        """按顺序读取模式"""
        # 计算批次数量
        time_1 = 0  # X 表读取用时
        time_2 = 0  #  obs 表读取用时
        time_3 = 0  #  var 表读取用时
        time_4 = 0  # 生成anndata数据用时

        # 新增：用于记录每批次时间的列表
        batch_times = {
            'batch_num': [],
            'time_X': [],
            'time_obs': [],
            'time_var': [],
            'time_anndata': [],
            'total_time': []
        }

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

            start_time1 = datetime.now()  # X 表读取用时

            # # 按顺序加载数据
            # if(self.isView): # 数据库视图
            #     sub_X = self.query(f"SELECT * FROM {self.viewID} LIMIT {current_batch_size} OFFSET {offset}")
            # else:
            #     sub_X = self.query(f"SELECT * FROM X LIMIT {current_batch_size} OFFSET {offset}")

            sub_X = self.query(f"SELECT * FROM X LIMIT {current_batch_size} OFFSET {offset}")

            end_time1 = datetime.now() # X 表读取用时
            time_diff1 = end_time1 - start_time1 # X 表读取用时
            time_1 = time_1 + time_diff1.total_seconds() # X 表读取用时

            start_time2 = datetime.now()  # obs表读取用时

            sub_obs = self.query(f"SELECT * FROM obs LIMIT {current_batch_size} OFFSET {offset}")

            end_time2 = datetime.now() # obs 表读取用时
            time_diff2 = end_time2 - start_time2  # obs 表读取用时
            time_2 = time_2 + time_diff2.total_seconds()  # obs 表读取用时

            start_time3= datetime.now()  # var 表读取用时

            sub_var = self.query("SELECT * FROM var")

            end_time3 = datetime.now()  # var 表读取用时
            time_diff3 = end_time3 - start_time3  # var 表读取用时
            time_3 = time_3 + time_diff3.total_seconds()  # var 表读取用时

            start_time4 = datetime.now()  # 生成anndata数据用时

            sub_X = sub_X.iloc[:, 2:]  # 去掉数据表中的前2列 id cell_id
            sub_obs = sub_obs.iloc[:, 1:]  # 去掉数据表中的前1列 id
            sub_var = sub_var.iloc[:, 1:]  # 去掉数据表中的前1列
            cell_id = sub_obs['cell_id'] # 获取cell_id
            gene_id = sub_var['gene_id']  # 获取gene_id

            # 创建AnnData对象，正确设置obs_names和var_names
            adata = AnnData(
                X=sub_X.values,
                obs=sub_obs,
                var=sub_var
            )

            from scipy.sparse import csr_matrix
            adata.X = csr_matrix(adata.X)

            # adata.obs.index = adata.obs.index.astype(str)
            # adata.var.index = adata.var.index.astype(str)
            adata.obs_names = cell_id.astype(str)  # 设置观测名为cell_id
            adata.var_names = gene_id.astype(str)  # 设置变量名为gene_id

            end_time4 = datetime.now()  # 生成anndata数据用时
            time_diff4 = end_time4 - start_time4  # 生成anndata数据用时
            time_4 = time_4 + time_diff4.total_seconds()  # 生成anndata数据用时

            # 新增：记录当前批次的时间
            batch_total_time = time_diff1.total_seconds() + time_diff2.total_seconds() + \
                               time_diff3.total_seconds() + time_diff4.total_seconds()

            batch_times['batch_num'].append(i + 1)
            batch_times['time_X'].append(time_diff1.total_seconds())
            batch_times['time_obs'].append(time_diff2.total_seconds())
            batch_times['time_var'].append(time_diff3.total_seconds())
            batch_times['time_anndata'].append(time_diff4.total_seconds())
            batch_times['total_time'].append(batch_total_time)

            # 返回数据
            yield adata

            print(f"X 表读取用时： {time_1:.2f} 秒")
            print(f"obs 表读取用时： {time_2:.2f} 秒")
            print(f"var 表读取用时： {time_3:.2f} 秒")
            print(f"生成anndata数据用时： {time_4:.2f} 秒")

        # 新增：在所有批次处理完成后，绘制时间图表
        self._plot_batch_times(batch_times, batch_size)

    # todo 优化1：使用游标分页替代OFFSET分页
    def minibatch_scan_order_cursor(self, total_num, batch_size, drop_last):
        """使用游标分页替代OFFSET分页 - 修复版本"""
        # 计算批次数量
        time_1 = 0  # X 表读取用时
        time_2 = 0  # obs 表读取用时
        time_3 = 0  # var 表读取用时
        time_4 = 0  # 生成anndata数据用时

        # 用于记录每批次时间的列表
        batch_times = {
            'batch_num': [],
            'time_X': [],
            'time_obs': [],
            'time_var': [],
            'time_anndata': [],
            'total_time': [],
            'time_X_query_execution': [],  # 查询执行时间
            'time_X_df_conversion': [],  # DataFrame转换时间
            'time_X_total': [],  # X表总时间
            'rows_retrieved': [],  # 返回行数
            'columns_retrieved': [],  # 返回列数
            'data_size_mb': []  # 数据大小(MB)
        }

        # 使用游标分页替代OFFSET分页
        last_id = 0  # 假设id是连续自增的主键,起始ID
        batch_num = 0
        processed_count = 0

        # 预先获取var表数据（只需一次）
        start_time3 = datetime.now()
        sub_var = self.query("SELECT * FROM var")
        end_time3 = datetime.now()
        time_diff3 = end_time3 - start_time3
        time_3 = time_3 + time_diff3.total_seconds()

        # 处理var表数据（只需一次）
        try:
            if sub_var.shape[1] > 1:
                sub_var_processed = sub_var.iloc[:, 1:]  # 去掉数据表中的前1列
            else:
                sub_var_processed = sub_var.copy()

            if 'gene_id' in sub_var_processed.columns:
                gene_id = sub_var_processed['gene_id']
            elif 'gene_id' in sub_var.columns:
                gene_id = sub_var['gene_id']
            else:
                gene_id = sub_var_processed.index.astype(str)
        except Exception as e:
            print(f"处理var表时出错: {e}")
            sub_var_processed = sub_var
            gene_id = sub_var.index.astype(str)

        while processed_count < total_num:
            batch_num += 1

            # ========== 详细时间监控开始 ==========
            print(f"\n--- 批次 {batch_num} 详细时间分析 ---")
            print(f"\n--- 批次大小 {batch_size} ---")

            # 关键语句1: 查询执行时间 (找到N行)
            query_start = datetime.now()

            if self.isView:
                query = f"""
                        SELECT * FROM {self.viewID} 
                        WHERE id > {last_id} 
                        ORDER BY id 
                        LIMIT {batch_size}
                    """
            else:
                query = f"""
                        SELECT * FROM X 
                        WHERE id > {last_id} 
                        ORDER BY id 
                        LIMIT {batch_size}
                    """

            # 关键语句2: 执行查询
            result = self.connection.execute(query)
            query_end = datetime.now()
            query_time = (query_end - query_start).total_seconds()

            # 关键语句4: 完整的DataFrame转换
            df_start = datetime.now()
            sub_X = result.df()  # sub_X = result.pl()
            df_end = datetime.now()
            df_time = (df_end - df_start).total_seconds()

            # 计算总X表时间和数据量信息
            total_x_time = query_time + df_time

            if not sub_X.empty:
                rows = len(sub_X)
                columns = len(sub_X.columns)
                data_size_mb = (rows * columns * 8) / (1024 * 1024)
            else:
                rows = 0
                columns = 0
                data_size_mb = 0

            # 输出详细时间分析
            print(f"🔍 查询执行时间: {query_time:.4f}秒 (编译和执行SQL，找到{rows}行)")
            print(f"🔄 DataFrame转换时间: {df_time:.4f}秒 (将原始数据转换为Pandas DataFrame)")
            print(f"⏱️  X表总时间: {total_x_time:.4f}秒")
            print(f"📊 数据量: {rows}行 × {columns}列 = {data_size_mb:.2f}MB")  # 数据量是自己计算的，不准确

            if total_x_time > 0:
                print(f"📈 时间占比 - 查询: {query_time / total_x_time * 100:.1f}%, "
                      f"转换: {df_time / total_x_time * 100:.1f}%")

            if df_time > 0 and data_size_mb > 0:
                conversion_speed = data_size_mb / df_time
                print(f"🚀 DataFrame转换速度: {conversion_speed:.2f} MB/秒")

            # 性能诊断建议
            if query_time > df_time * 2:
                print("💡 建议: 查询执行是主要瓶颈，考虑优化索引或SQL语句")
            elif df_time > query_time * 2:
                print("💡 建议: DataFrame转换是主要瓶颈，考虑减少返回列数或优化数据类型")

            # 更新X表累计时间（使用分解时间的总和）
            time_1 = time_1 + total_x_time

            # 如果没有数据了，退出循环
            if sub_X.empty:
                print("❌ 查询返回空结果，退出循环")
                break

            # 更新last_id为当前批次的最后一个id
            last_id = sub_X['id'].iloc[-1]
            current_batch_size = len(sub_X)
            processed_count += current_batch_size

            print(
                f"✅ 更新last_id: {last_id}, 当前批次大小: {current_batch_size}, 已处理: {processed_count}/{total_num}")

            # 获取obs表数据
            start_time2 = datetime.now()

            # 使用cell_id来关联obs表（更可靠的方法）
            cell_ids = sub_X['cell_id'].tolist()
            cell_ids_str = ",".join([f"'{cid}'" for cid in cell_ids])
            sub_obs = self.query(f"SELECT * FROM obs WHERE cell_id IN ({cell_ids_str})")

            # 检查obs表查询结果
            if sub_obs.empty:
                print(f"⚠️ 警告: 未找到对应的obs数据，cell_ids: {cell_ids[:5]}...")  # 只显示前5个
            else:
                print(f"✅ 找到obs数据: {len(sub_obs)} 行")

            end_time2 = datetime.now()
            time_diff2 = end_time2 - start_time2
            time_2 = time_2 + time_diff2.total_seconds()

            # 创建AnnData对象
            start_time4 = datetime.now()

            try:
                # 处理数据
                if sub_X.shape[1] >= 2:
                    sub_X_processed = sub_X.iloc[:, 2:]  # 去掉数据表中的前2列 id cell_id
                else:
                    sub_X_processed = sub_X.copy()
                    print("⚠️ 警告: X表列数不足，使用原始数据")

                if sub_obs.shape[1] >= 1:
                    sub_obs_processed = sub_obs.iloc[:, 1:]  # 去掉数据表中的前1列 id
                else:
                    sub_obs_processed = sub_obs.copy()
                    print("⚠️ 警告: obs表列数不足，使用原始数据")

                # 确保obs数据顺序与X数据一致
                if not sub_obs.empty and 'cell_id' in sub_obs_processed.columns:
                    sub_obs_processed = sub_obs_processed.set_index('cell_id')
                    # 确保所有cell_id都存在
                    missing_cells = set(cell_ids) - set(sub_obs_processed.index)
                    if missing_cells:
                        print(f"⚠️ 警告: 缺失{len(missing_cells)}个细胞的obs数据")

                    sub_obs_processed = sub_obs_processed.loc[cell_ids].reset_index()
                    cell_id_col = sub_obs_processed['cell_id']
                else:
                    cell_id_col = sub_obs_processed.index.astype(str) if not sub_obs.empty else []

                # 创建AnnData对象
                adata = AnnData(
                    X=sub_X_processed.values,
                    obs=sub_obs_processed,
                    var=sub_var_processed
                )
                adata.obs_names = cell_id_col.astype(str)
                adata.var_names = gene_id.astype(str)

                print(f"✅ 成功创建AnnData对象: {adata.n_obs}细胞 × {adata.n_vars}基因")

            except Exception as e:
                print(f"❌ 创建AnnData对象时出错: {e}")
                # 创建备用AnnData对象
                adata = AnnData(
                    X=sub_X.iloc[:, 2:].values if sub_X.shape[1] > 2 else sub_X.values,
                    obs=sub_obs.iloc[:, 1:] if sub_obs.shape[1] > 1 else sub_obs,
                    var=sub_var_processed
                )
                # 尝试设置名称
                try:
                    if 'cell_id' in sub_obs.columns:
                        adata.obs_names = sub_obs['cell_id'].astype(str)
                    if 'gene_id' in sub_var.columns:
                        adata.var_names = sub_var['gene_id'].astype(str)
                except:
                    print("⚠️ 设置观察和变量名称失败")

            end_time4 = datetime.now()
            time_diff4 = end_time4 - start_time4
            time_4 = time_4 + time_diff4.total_seconds()

            # 记录当前批次的时间
            batch_total_time = total_x_time + time_diff2.total_seconds() + time_diff3.total_seconds() + time_diff4.total_seconds()

            batch_times['batch_num'].append(batch_num)
            batch_times['time_X'].append(total_x_time)  # 使用分解时间的总和
            batch_times['time_obs'].append(time_diff2.total_seconds())
            batch_times['time_var'].append(time_diff3.total_seconds())
            batch_times['time_anndata'].append(time_diff4.total_seconds())
            batch_times['total_time'].append(batch_total_time)
            batch_times['time_X_query_execution'].append(query_time)
            batch_times['time_X_df_conversion'].append(df_time)
            batch_times['time_X_total'].append(total_x_time)
            batch_times['rows_retrieved'].append(rows)
            batch_times['columns_retrieved'].append(columns)
            batch_times['data_size_mb'].append(data_size_mb)

            # 返回数据
            yield adata

            # 输出累计时间
            print(f"\n📊 累计时间 - X表: {time_1:.2f}秒, obs表: {time_2:.2f}秒, "
                  f"var表: {time_3:.2f}秒, anndata: {time_4:.2f}秒")

            # 如果启用了drop_last且当前批次不是完整批次，则丢弃
            if drop_last and current_batch_size < batch_size:
                print(f"🔚 启用drop_last，当前批次不完整({current_batch_size}<{batch_size})，退出循环")
                break

            # 检查是否已处理所有数据
            if processed_count >= total_num:
                print(f"🎉 已完成所有数据处理: {processed_count}/{total_num}")
                break

        # 在所有批次处理完成后，绘制时间图表
        print(f"\n📈 处理完成，共处理 {batch_num} 批次")
        self._plot_batch_times(batch_times, batch_size)

    # todo CSR格式 代码 1
    def minibatch_scan_order_CSR(self, total_num, batch_size, drop_last):
        """按顺序读取模式"""
        # 计算批次数量
        time_2 = 0  #  obs 表读取用时
        time_3 = 0  #  var 表读取用时
        time_4 = 0  # 生成anndata数据用时
        time_5 = 0  # X_CSR_indptr 表 读取时间
        time_6 = 0  # X_CSR_data 表 读取时间
        time_7 = 0  # CSR 生成anndata数据用时
        time_8 = 0  # X_CSR_data 转 DF 时间

        data_read_start = 0 #  X_CSR_data表 每次的读取的 起始点
        data_read_count = 0  # X_CSR_data表 每次的读取的 数据量

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

            query_CSR_indptr = f"SELECT * FROM X_CSR_indptr LIMIT {current_batch_size} OFFSET {offset}" # todo 获取CSR_indptr

            CSR_indptr_result_start = datetime.now()
            result = self.connection.execute(query_CSR_indptr)  # todo CSR_indptr 查询时间
            CSR_indptr_result_end = datetime.now()
            CSR_indptr_result_time = (CSR_indptr_result_end - CSR_indptr_result_start).total_seconds()
            print(f"#### CSR_indptr 查询时间 : {CSR_indptr_result_time}")

            CSR_indptr_df_start = datetime.now()
            CSR_indptr_df = result.df()  # todo  CSR_indptr 转化为 df格式时间
            CSR_indptr_df_end = datetime.now()
            CSR_indptr_df_time = (CSR_indptr_df_end - CSR_indptr_df_start).total_seconds()
            print(f"#### CSR_indptr 转化为 df格式时间 : {CSR_indptr_df_time}")

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
            #    indptr : 3, 6
            #  }

            CSR_indptr_np_start = datetime.now()

            # todo 在开头补上 0 ， 并减去上一批的末尾值
            CSR_indptr_array = np.insert(CSR_indptr_df['indptr'].values - data_read_start, 0, 0)
            # todo: 第 0 批
            #  CSR_indptr_array = { 0 , 2 , 3  } - 0 = { 0 , 2 , 3  }
            # todo: 第 1 批
            #  CSR_indptr_array = { 0 , 3 , 3  } - 3 = { 0 , 0 , 0  }
            # todo: 第 2 批
            #  CSR_indptr_array = { 0 , 3 , 6  } - 3 = { 0 , 0 , 3  }


            CSR_indptr_np_end = datetime.now()
            CSR_indptr_np_time = (CSR_indptr_np_end - CSR_indptr_np_start).total_seconds()
            print(f"#### CSR_indptr 的 df 提取 np 时间 : {CSR_indptr_np_time}")

            time_5 = time_5 + CSR_indptr_result_time + CSR_indptr_df_time + CSR_indptr_np_time  # X_CSR_indptr 表 读取时间

            # 只保留indptr，并在最前面补上0值，并减去 data_read_start = 0
            # CSR_indptr_array = 0 +  ( CSR_indptr_df['indptr'] - data_read_count )

            # 起点： 初始 data_read_start = 0
            data_read_count = CSR_indptr_array[-1]  # 读取数量： 3 个值， 即 data_read_count = CSR_indptr_array[-1]=3,最后一个值
            # todo : 第 0 批
            #  data_read_count = CSR_indptr_array[-1] = 3, 最后一个值； 读取数量： 3 个值，
            # todo : 第 1 批
            #  data_read_count = CSR_indptr_array[-1] = 0, 最后一个值； 读取数量： 0 个值， 不读取 query_CSR_data 表
            # todo : 第 2 批
            #  data_read_count = CSR_indptr_array[-1] = 3, 最后一个值； 读取数量： 3 个值

            if (data_read_count > 0):

                query_CSR_data = f"SELECT * FROM X_CSR_data LIMIT {data_read_count} OFFSET {data_read_start}"
                # todo : 第 0 批
                #  起点 ：data_read_start = 0
                #  读取量： data_read_count = 3 ，
                #  CSR_data_df =
                #    {    id:  0, 1 ,2 ;
                #        indices :  0, 2 , 2 ;
                #        data : 8, 2, 5 ;
                #    }
                # todo : 第 1 批 不读取
                # todo : 第 2 批
                #  起点 ：data_read_start = 3
                #  读取量： data_read_count = 3 ，
                #  CSR_data_df =
                #    {    id:  3, 4 , 5 ;
                #        indices :  2, 3 , 4 ;
                #        data : 7, 1, 2 ;
                #    }

                CSR_data_result_start = datetime.now()
                rusult = self.connection.execute(query_CSR_data)  # todo 获取 CSR_data
                CSR_data_result_end = datetime.now()
                CSR_data_result_time = (CSR_data_result_end - CSR_data_result_start).total_seconds()
                print(f"#### CSR_data 查询时间 : {CSR_data_result_time}")

                CSR_data_df_start = datetime.now()
                CSR_data_df = rusult.df()  # todo  CSR_data 转化为 df
                CSR_data_df_end = datetime.now()
                CSR_data_df_time = (CSR_data_df_end - CSR_data_df_start).total_seconds()
                print(f"#### CSR_data 转化为 df  时间 : {CSR_data_df_time}")

                CSR_indices_array = CSR_data_df['indices'].to_numpy()  # todo:  CSR_indices_array = [ 0,2,2]
                CSR_data_array = CSR_data_df['data'].to_numpy()  # todo:  CSR_data_array = [ 8,2,5]

                data_read_start = data_read_start + data_read_count  # 更新起点
                # todo : 第 0 批
                #  更新起点 ：data_read_start = 0 + 3 = 3
                # todo : 第 1 批 值不变
                # todo : 第 2 批
                #  更新起点 ：data_read_start = 3 + 3 = 6


                time_6 = time_6 + CSR_data_result_time# X_CSR_data 表 读取时间
                time_8 += CSR_data_df_time

            else:
                print("获取 0 个值，不查询 ")  # todo 空值的处理

            start_time2 = datetime.now()  # obs表读取用时

            sub_obs = self.query(f"SELECT * FROM obs LIMIT {current_batch_size} OFFSET {offset}")

            end_time2 = datetime.now() # obs 表读取用时
            time_diff2 = end_time2 - start_time2  # obs 表读取用时
            time_2 = time_2 + time_diff2.total_seconds()  # obs 表读取用时

            start_time3= datetime.now()  # var 表读取用时

            sub_var = self.query("SELECT * FROM var")

            end_time3 = datetime.now()  # var 表读取用时
            time_diff3 = end_time3 - start_time3  # var 表读取用时
            time_3 = time_3 + time_diff3.total_seconds()  # var 表读取用时

            start_time4 = datetime.now()  # 生成anndata数据用时

            sub_obs = sub_obs.iloc[:, 1:]  # 去掉数据表中的前1列 id
            sub_var = sub_var.iloc[:, 1:]  # 去掉数据表中的前1列
            cell_id = sub_obs['cell_id'] # 获取cell_id
            gene_id = sub_var['gene_id']  # 获取gene_id

            # 创建CSR格式的稀疏矩阵
            X = sp.csr_matrix((CSR_data_array, CSR_indices_array, CSR_indptr_array),
                              shape=(len(sub_obs), len(sub_var)))
            # 创建AnnData对象
            adata = AnnData(X=X, obs=sub_obs, var=sub_var)


            adata.obs_names = cell_id.astype(str)  # 设置观测名为cell_id
            adata.obs.index = cell_id.astype(str)
            adata.var_names = gene_id.astype(str)  # 设置变量名为gene_id
            adata.var.index = gene_id.astype(str)

            end_time4 = datetime.now()  # 生成anndata数据用时
            time_diff4 = end_time4 - start_time4  # 生成anndata数据用时
            time_4 = time_4 + time_diff4.total_seconds()  # 生成anndata数据用时

            # 新增：记录当前批次的时间
            batch_total_time = time_diff2.total_seconds() + time_diff3.total_seconds() + time_diff4.total_seconds()

            batch_times['time_X_CSR_indptr'].append(CSR_indptr_result_time + CSR_indptr_df_time)
            batch_times['time_X_CSR_data'].append(CSR_data_result_time)
            batch_times['time_X_CSR_data_df'].append(CSR_data_df_time)
            batch_times['batch_num'].append(i + 1)
            batch_times['time_obs'].append(time_diff2.total_seconds())
            batch_times['time_var'].append(time_diff3.total_seconds())
            batch_times['time_anndata'].append(time_diff4.total_seconds())
            batch_times['total_time'].append(batch_total_time)

            # 返回数据
            yield adata

            print(f"obs 表读取用时： {time_2:.2f} 秒")
            print(f"var 表读取用时： {time_3:.2f} 秒")
            print(f"生成anndata数据用时： {time_4:.2f} 秒")
            print(f"X_CSR_indptr 表 读取时间： {time_5:.2f} 秒")
            print(f"X_CSR_data 表 读取时间： {time_6:.2f} 秒")
            print(f"X_CSR_data 表 转df时间 ： {time_8:.2f} 秒")

        # 新增：在所有批次处理完成后，绘制时间图表
        self._plot_batch_times(batch_times, batch_size)

    # # todo CSR格式 游标优化 ,deepseek
    # def minibatch_scan_order_cursor_CSR(self, total_num, batch_size, drop_last):
    #     """使用游标分页替代OFFSET分页 - data表也使用游标分页"""
    #     # 计算批次数量
    #     time_CSR_indptr = 0  # X_CSR_indptr 表 读取时间
    #     time_CSR_data_read = 0  # X_CSR_data 表 读取时间
    #     time_CSR_data_df = 0  # X_CSR_data 表 读取时间
    #
    #     # 使用游标分页替代OFFSET分页
    #     last_indptr_id = 0  # X_CSR_indptr表的游标
    #     last_data_id = 0  # X_CSR_data表的游标
    #     batch_num = 0
    #     processed_count = 0
    #
    #     # 新增：记录全局起始点
    #     global_data_start = 0  # 记录第一个细胞的indptr值，用于计算相对偏移
    #
    #     # 预先获取var表数据（只需一次）
    #     sub_var = self.query("SELECT * FROM var")
    #     sub_var = sub_var.iloc[:, 1:]  # 去掉数据表中的前1列
    #     gene_id = sub_var['gene_id']  # 获取gene_id
    #
    #     # 初始化data_id游标：获取X_CSR_data表的最小id
    #     init_query = "SELECT MIN(id) as min_id FROM X_CSR_data"
    #     init_result = self.connection.execute(init_query)
    #     init_df = init_result.df()
    #     last_data_id = init_df['min_id'].iloc[0] - 1 if not init_df.empty else 0
    #     print(f"#### 初始化data_id游标: {last_data_id}")
    #
    #     while processed_count < total_num:
    #         batch_num += 1
    #
    #         # ========== 详细时间监控开始 ==========
    #         print(f"\n--- 批次 {batch_num} 详细时间分析 ---")
    #         print(f"--- 批次大小 {batch_size} ---")
    #
    #         # 1. 查询X_CSR_indptr表（游标分页）
    #         CSR_indptr_start = datetime.now()
    #         CSR_indptr_df = self.connection.execute(
    #             f"SELECT * FROM X_CSR_indptr WHERE id > {last_indptr_id} ORDER BY id LIMIT {batch_size}"
    #         ).df()
    #         CSR_indptr_end = datetime.now()
    #         CSR_indptr_time = (CSR_indptr_end - CSR_indptr_start).total_seconds()
    #         print(f"#### CSR_indptr 查询时间: {CSR_indptr_time}")
    #
    #         # 检查是否获取到数据
    #         if len(CSR_indptr_df) == 0:
    #             print(f"#### 批次 {batch_num}: 没有获取到indptr数据，结束循环")
    #             break
    #
    #         # 更新indptr游标
    #         last_indptr_id = CSR_indptr_df['id'].iloc[-1]
    #
    #         # 获取当前批次的indptr值（全局值）
    #         current_indptr_values = CSR_indptr_df['indptr'].values
    #
    #         # 确定全局起始点（只在第一次循环）
    #         if batch_num == 1:
    #             global_data_start = current_indptr_values[0] if len(current_indptr_values) > 0 else 0
    #             print(f"#### 全局indptr起始点: {global_data_start}")
    #
    #         # 计算indptr数组：将全局indptr值转换为相对于当前批次起始点的偏移
    #         # 当前批次在data表中的起始点 = 上一个批次的结束点
    #         relative_indptr_values = current_indptr_values - global_data_start
    #         CSR_indptr_array = np.insert(relative_indptr_values, 0, 0)
    #
    #         print(f"#### indptr数组信息: 长度={len(CSR_indptr_array)}, 最后值={CSR_indptr_array[-1]}")
    #
    #         # 2. 查询obs表（游标分页）
    #         sub_obs = self.query(
    #             f"SELECT * FROM obs WHERE id > {last_indptr_id - batch_size} ORDER BY id LIMIT {batch_size}")
    #
    #         # 注意：这里我们假设obs表的id与X_CSR_indptr表的id是对齐的
    #         current_batch_size = len(sub_obs)
    #         processed_count += current_batch_size
    #         print(f"✅ 当前批次大小: {current_batch_size}, 已处理: {processed_count}/{total_num}")
    #
    #         # 3. 计算需要读取的data数据量
    #         data_read_count = CSR_indptr_array[-1]  # indptr最后一个值就是需要读取的数据量
    #
    #         # 4. 查询X_CSR_data表（游标分页）
    #         if data_read_count > 0:
    #             print(f"#### 需要读取 {data_read_count} 条CSR_data记录")
    #             print(f"#### CSR_data 查询起点 (data_id > {last_data_id})")
    #
    #             # 使用游标分页查询X_CSR_data表
    #             query_CSR_data = f"""
    #                 SELECT * FROM X_CSR_data
    #                 WHERE id > {last_data_id}
    #                 ORDER BY id
    #                 LIMIT {data_read_count}
    #             """
    #
    #             CSR_data_result_start = datetime.now()
    #             CSR_data_result = self.connection.execute(query_CSR_data)
    #             CSR_data_result_end = datetime.now()
    #             CSR_data_result_time = (CSR_data_result_end - CSR_data_result_start).total_seconds()
    #             print(f"#### CSR_data 查询时间: {CSR_data_result_time}")
    #
    #             CSR_data_df_start = datetime.now()
    #             CSR_data_df = CSR_data_result.df()
    #             CSR_data_df_end = datetime.now()
    #             CSR_data_df_time = (CSR_data_df_end - CSR_data_df_start).total_seconds()
    #             print(f"#### CSR_data 转化为 df 时间: {CSR_data_df_time}")
    #
    #             # 检查获取的数据量
    #             actual_data_count = len(CSR_data_df)
    #             print(f"#### 实际获取到 {actual_data_count} 条CSR_data记录")
    #
    #             if actual_data_count > 0:
    #                 # 更新data游标
    #                 last_data_id = CSR_data_df['id'].iloc[-1]
    #                 print(f"#### 更新data_id游标: {last_data_id}")
    #
    #                 # 提取数据
    #                 CSR_indices_array = CSR_data_df['indices'].to_numpy()
    #                 CSR_data_array = CSR_data_df['data'].to_numpy()
    #
    #                 # 验证数据一致性
    #                 if actual_data_count != data_read_count:
    #                     print(f"#### 警告: 期望 {data_read_count} 条数据，实际获取 {actual_data_count} 条")
    #                     print(f"#### 修正indptr最后一个值")
    #                     CSR_indptr_array[-1] = actual_data_count
    #             else:
    #                 print(f"#### 警告: 未获取到CSR_data数据")
    #                 CSR_indices_array = np.array([])
    #                 CSR_data_array = np.array([])
    #                 # 如果没有数据，修正indptr最后一个值为0
    #                 CSR_indptr_array[-1] = 0
    #         else:
    #             print("获取 0 个值，不查询")
    #             CSR_indices_array = np.array([])
    #             CSR_data_array = np.array([])
    #
    #         time_CSR_indptr += CSR_indptr_time
    #         if data_read_count > 0:
    #             time_CSR_data_read += CSR_data_result_time
    #             time_CSR_data_df += CSR_data_df_time
    #
    #         # 5. 处理obs数据
    #         sub_obs = sub_obs.iloc[:, 1:]  # 去掉数据表中的前1列 id
    #         cell_id = sub_obs['cell_id']  # 获取cell_id
    #
    #         # 6. 验证CSR数据格式
    #         print(f"#### 详细检查CSR数据:")
    #         print(f"  - indptr长度: {len(CSR_indptr_array)}")
    #         print(f"  - indptr最后值: {CSR_indptr_array[-1]}")
    #         print(f"  - indices长度: {len(CSR_indices_array)}")
    #         print(f"  - data长度: {len(CSR_data_array)}")
    #
    #         # 确保indptr最后一个值等于data数组长度
    #         if CSR_indptr_array[-1] != len(CSR_data_array):
    #             print(f"#### 修正: indptr[-1]={CSR_indptr_array[-1]}, data长度={len(CSR_data_array)}")
    #             CSR_indptr_array[-1] = len(CSR_data_array)
    #
    #         # 7. 创建CSR稀疏矩阵
    #         try:
    #             X = sp.csr_matrix(
    #                 (CSR_data_array, CSR_indices_array, CSR_indptr_array),
    #                 shape=(len(sub_obs), len(sub_var))
    #             )
    #             print(f"#### CSR矩阵创建成功")
    #         except Exception as e:
    #             print(f"#### 错误创建CSR矩阵: {e}")
    #             # 创建空矩阵作为后备
    #             X = sp.csr_matrix((len(sub_obs), len(sub_var)))
    #
    #         # 8. 创建AnnData对象
    #         adata = AnnData(X=X, obs=sub_obs, var=sub_var)
    #
    #         adata.obs_names = cell_id.astype(str)  # 设置观测名为cell_id
    #         adata.obs.index = cell_id.astype(str)
    #         adata.var_names = gene_id.astype(str)  # 设置变量名为gene_id
    #         adata.var.index = gene_id.astype(str)
    #
    #         # 9. 返回数据
    #         yield adata
    #
    #         # 10. 输出累计时间
    #         print(f"\n📊 累计时间 - CSR_indptr 表: {time_CSR_indptr:.2f}秒, "
    #               f"CSR_data_read: {time_CSR_data_read:.2f}秒, "
    #               f"CSR_data表 转化为df格式: {time_CSR_data_df:.2f}秒")
    #
    #         # 11. 检查终止条件
    #         if drop_last and current_batch_size < batch_size:
    #             print(f"🔚 启用drop_last，当前批次不完整({current_batch_size}<{batch_size})，退出循环")
    #             break
    #
    #         if processed_count >= total_num:
    #             print(f"🎉 已完成所有数据处理: {processed_count}/{total_num}")
    #             break


    # todo CSR格式 游标优化，代码2
    def minibatch_scan_order_cursor_csr(self, total_num, batch_size, drop_last):
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
        last_data_id = 0  # 初始化为0，假设id从1开始，或根据实际情况调整

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

            query_CSR_indptr = f"SELECT * FROM X_CSR_indptr LIMIT {current_batch_size} OFFSET {offset}"  # todo 获取CSR_indptr

            CSR_indptr_result_start = datetime.now()
            result = self.connection.execute(query_CSR_indptr)  # todo CSR_indptr 查询时间
            CSR_indptr_result_end = datetime.now()
            CSR_indptr_result_time = (CSR_indptr_result_end - CSR_indptr_result_start).total_seconds()
            print(f"#### CSR_indptr 查询时间 : {CSR_indptr_result_time}")

            CSR_indptr_df_start = datetime.now()
            CSR_indptr_df = result.df()  # todo  CSR_indptr 转化为 df格式时间
            CSR_indptr_df_end = datetime.now()
            CSR_indptr_df_time = (CSR_indptr_df_end - CSR_indptr_df_start).total_seconds()
            print(f"#### CSR_indptr 转化为 df格式时间 : {CSR_indptr_df_time}")

            # todo  可以读取出 df 再提取 需要的 array， 也可以直接提取需要的 array ；
            # 一次取2个,同X表的读取量
            # todo: 第 0 批
            # CSR_indptr_df =
            # {  id:  0, 1 ;
            #    cell_id :  cell_0,cell_1 ;
            #    indptr : 2, 3
            # }

            CSR_indptr_np_start = datetime.now()
            CSR_indptr_array = np.insert(CSR_indptr_df['indptr'].values - data_read_start, 0, 0)  # todo  df 提取 np 时间
            CSR_indptr_np_end = datetime.now()
            CSR_indptr_np_time = (CSR_indptr_np_end - CSR_indptr_np_start).total_seconds()
            print(f"#### CSR_indptr 的 df 提取 np 时间 : {CSR_indptr_np_time}")

            time_5 = time_5 + CSR_indptr_result_time + CSR_indptr_df_time + CSR_indptr_np_time  # X_CSR_indptr 表 读取时间

            # 只保留indptr，并在最前面补上0值，并减去 data_read_start = 0
            # CSR_indptr_array = 0 +  ( CSR_indptr_df['indptr'] - data_read_count )
            # todo 得到 CSR_indptr_array = [ 0,2,3]

            # 起点： 初始 data_read_start = 0
            data_read_count = CSR_indptr_array[-1]  # 获取值数量： 获取 3 个值， 即data_read_count = CSR_indptr_array[-1]=3,最后一个值
            if (data_read_count > 0):
                # ========== 修改点：将OFFSET/LIMIT改为游标分页（基于id的范围查询） ==========
                # 使用基于id的范围查询替代OFFSET，避免深翻页问题
                # 注意：这里假设X_CSR_data表的id是连续递增的
                end_id = last_data_id + data_read_count
                print(f"data_read_count (CSR_indptr_array[-1]) 是 {data_read_count}" )
                print(f"last_data_id 是 {last_data_id}" )
                print(f"end_id 是 {end_id}" )
                query_CSR_data = f"SELECT * FROM X_CSR_data WHERE id >= {last_data_id} AND id < {end_id} ORDER BY id"
                # ========== 修改结束 ==========

                CSR_data_result_start = datetime.now()
                rusult = self.connection.execute(query_CSR_data)  # todo 获取 CSR_data
                CSR_data_result_end = datetime.now()
                CSR_data_result_time = (CSR_data_result_end - CSR_data_result_start).total_seconds()
                print(f"#### CSR_data 查询时间 : {CSR_data_result_time}")

                CSR_data_df_start = datetime.now()
                CSR_data_df = rusult.df()  # todo  CSR_data 转化为 df
                CSR_data_df_end = datetime.now()
                CSR_data_df_time = (CSR_data_df_end - CSR_data_df_start).total_seconds()
                print(f"#### CSR_data 转化为 df  时间 : {CSR_data_df_time}")

                # CSR_data_df =
                # {  id:  0, 1 ,2 ;
                #    indices :  0, 2 , 2 ;
                #    data : 8, 2, 5 ;
                # }
                print(f"len(CSR_data_df)  {len(CSR_data_df)}")

                CSR_indices_array = CSR_data_df['indices'].to_numpy()  # todo:  CSR_indices_array = [ 0,2,2]
                CSR_data_array = CSR_data_df['data'].to_numpy()  # todo:  CSR_data_array = [ 8,2,5]

                # ========== 修改点：更新游标位置 ==========
                # 更新last_data_id为本次读取的最后一个id
                if not CSR_data_df.empty:
                    last_data_id = CSR_data_df['id'].iloc[-1]  # 获取本次读取的最后一个id
                # ========== 修改结束 ==========

                data_read_start = data_read_start + data_read_count  # 更新起点

                time_6 = time_6 + CSR_data_result_time  # X_CSR_data 表 读取时间
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

            print(f"indptr_array[-1]: {CSR_indptr_array[-1]}")
            print(f"indices长度: {len(CSR_indices_array)}")
            print(f"data长度: {len(CSR_data_array)}")

            # 创建CSR格式的稀疏矩阵
            X = sp.csr_matrix((CSR_data_array, CSR_indices_array, CSR_indptr_array),
                              shape=(len(sub_obs), len(sub_var)))
            # 创建AnnData对象
            adata = AnnData(X=X, obs=sub_obs, var=sub_var)

            adata.obs_names = cell_id.astype(str)  # 设置观测名为cell_id
            adata.obs.index = cell_id.astype(str)
            adata.var_names = gene_id.astype(str)  # 设置变量名为gene_id
            adata.var.index = gene_id.astype(str)

            end_time4 = datetime.now()  # 生成anndata数据用时
            time_diff4 = end_time4 - start_time4  # 生成anndata数据用时
            time_4 = time_4 + time_diff4.total_seconds()  # 生成anndata数据用时

            # 新增：记录当前批次的时间
            batch_total_time = time_diff2.total_seconds() + time_diff3.total_seconds() + time_diff4.total_seconds()

            batch_times['time_X_CSR_indptr'].append(CSR_indptr_result_time + CSR_indptr_df_time)
            batch_times['time_X_CSR_data'].append(CSR_data_result_time)
            batch_times['time_X_CSR_data_df'].append(CSR_data_df_time)
            batch_times['batch_num'].append(i + 1)
            batch_times['time_obs'].append(time_diff2.total_seconds())
            batch_times['time_var'].append(time_diff3.total_seconds())
            batch_times['time_anndata'].append(time_diff4.total_seconds())
            batch_times['total_time'].append(batch_total_time)

            # 返回数据
            yield adata

            print(f"obs 表读取用时： {time_2:.2f} 秒")
            print(f"var 表读取用时： {time_3:.2f} 秒")
            print(f"生成anndata数据用时： {time_4:.2f} 秒")
            print(f"X_CSR_indptr 表 读取时间： {time_5:.2f} 秒")
            print(f"X_CSR_data 表 读取时间： {time_6:.2f} 秒")
            print(f"X_CSR_data 表 转df时间 ： {time_8:.2f} 秒")

        # 新增：在所有批次处理完成后，绘制时间图表
        self._plot_batch_times(batch_times, batch_size)

    # todo CSR格式 游标优化，df优化方法1 arrow table
    def minibatch_scan_order_cursor_csr_df_arrow(self, total_num, batch_size, drop_last):
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
        last_data_id = 0  # 初始化为0，假设id从1开始，或根据实际情况调整

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

            query_CSR_indptr = f"SELECT * FROM X_CSR_indptr LIMIT {current_batch_size} OFFSET {offset}"  # todo 获取CSR_indptr

            CSR_indptr_result_start = datetime.now()
            result = self.connection.execute(query_CSR_indptr)  # todo CSR_indptr 查询时间
            CSR_indptr_result_end = datetime.now()
            CSR_indptr_result_time = (CSR_indptr_result_end - CSR_indptr_result_start).total_seconds()
            print(f"#### CSR_indptr 查询时间 : {CSR_indptr_result_time}")

            CSR_indptr_df_start = datetime.now()
            CSR_indptr_df = result.df()  # todo  CSR_indptr 转化为 df格式时间
            CSR_indptr_df_end = datetime.now()
            CSR_indptr_df_time = (CSR_indptr_df_end - CSR_indptr_df_start).total_seconds()
            print(f"#### CSR_indptr 转化为 df格式时间 : {CSR_indptr_df_time}")

            # todo  可以读取出 df 再提取 需要的 array， 也可以直接提取需要的 array ；
            # 一次取2个,同X表的读取量
            # todo: 第 0 批
            # CSR_indptr_df =
            # {  id:  0, 1 ;
            #    cell_id :  cell_0,cell_1 ;
            #    indptr : 2, 3
            # }

            CSR_indptr_np_start = datetime.now()
            CSR_indptr_array = np.insert(CSR_indptr_df['indptr'].values - data_read_start, 0, 0)  # todo  df 提取 np 时间
            CSR_indptr_np_end = datetime.now()
            CSR_indptr_np_time = (CSR_indptr_np_end - CSR_indptr_np_start).total_seconds()
            print(f"#### CSR_indptr 的 df 提取 np 时间 : {CSR_indptr_np_time}")

            time_5 = time_5 + CSR_indptr_result_time + CSR_indptr_df_time + CSR_indptr_np_time  # X_CSR_indptr 表 读取时间

            # 只保留indptr，并在最前面补上0值，并减去 data_read_start = 0
            # CSR_indptr_array = 0 +  ( CSR_indptr_df['indptr'] - data_read_count )
            # todo 得到 CSR_indptr_array = [ 0,2,3]

            # 起点： 初始 data_read_start = 0
            data_read_count = CSR_indptr_array[-1]  # 获取值数量： 获取 3 个值， 即data_read_count = CSR_indptr_array[-1]=3,最后一个值
            if (data_read_count > 0):


                end_id = last_data_id + data_read_count
                print(f"data_read_count (CSR_indptr_array[-1]) 是 {data_read_count}" )
                print(f"last_data_id 是 {last_data_id}" )
                print(f"end_id 是 {end_id}" )
                # 使用基于id的范围查询替代OFFSET，避免深翻页问题
                query_CSR_data = f"SELECT * FROM X_CSR_data WHERE id >= {last_data_id} AND id < {end_id} ORDER BY id"


                CSR_data_result_start = datetime.now()
                rusult = self.connection.execute(query_CSR_data)  # todo 获取 CSR_data
                CSR_data_result_end = datetime.now()
                CSR_data_result_time = (CSR_data_result_end - CSR_data_result_start).total_seconds()
                print(f"#### CSR_data 查询时间 : {CSR_data_result_time}")

                CSR_data_df_start = datetime.now()
                # CSR_data_df = rusult.df()  # todo  CSR_data 转化为 df

                # todo 方法1：使用fetch_arrow_table
                table = result.fetch_arrow_table()
                CSR_data_df = table.to_pandas()

                CSR_data_df_end = datetime.now()
                CSR_data_df_time = (CSR_data_df_end - CSR_data_df_start).total_seconds()
                print(f"#### CSR_data 转化为 df  时间 : {CSR_data_df_time}")

                # CSR_data_df =
                # {  id:  0, 1 ,2 ;
                #    indices :  0, 2 , 2 ;
                #    data : 8, 2, 5 ;
                # }
                print(f"len(CSR_data_df)  {len(CSR_data_df)}")

                CSR_indices_array = CSR_data_df['indices'].to_numpy()  # todo:  CSR_indices_array = [ 0,2,2]
                CSR_data_array = CSR_data_df['data'].to_numpy()  # todo:  CSR_data_array = [ 8,2,5]

                # ========== 修改点：更新游标位置 ==========
                # 更新last_data_id为本次读取的最后一个id
                if not CSR_data_df.empty:
                    last_data_id = CSR_data_df['id'].iloc[-1]  # 获取本次读取的最后一个id
                # ========== 修改结束 ==========

                data_read_start = data_read_start + data_read_count  # 更新起点

                time_6 = time_6 + CSR_data_result_time  # X_CSR_data 表 读取时间
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

            print(f"indptr_array[-1]: {CSR_indptr_array[-1]}")
            print(f"indices长度: {len(CSR_indices_array)}")
            print(f"data长度: {len(CSR_data_array)}")

            # 创建CSR格式的稀疏矩阵
            X = sp.csr_matrix((CSR_data_array, CSR_indices_array, CSR_indptr_array),
                              shape=(len(sub_obs), len(sub_var)))
            # 创建AnnData对象
            adata = AnnData(X=X, obs=sub_obs, var=sub_var)

            adata.obs_names = cell_id.astype(str)  # 设置观测名为cell_id
            adata.obs.index = cell_id.astype(str)
            adata.var_names = gene_id.astype(str)  # 设置变量名为gene_id
            adata.var.index = gene_id.astype(str)

            end_time4 = datetime.now()  # 生成anndata数据用时
            time_diff4 = end_time4 - start_time4  # 生成anndata数据用时
            time_4 = time_4 + time_diff4.total_seconds()  # 生成anndata数据用时

            # 新增：记录当前批次的时间
            batch_total_time = time_diff2.total_seconds() + time_diff3.total_seconds() + time_diff4.total_seconds()

            batch_times['time_X_CSR_indptr'].append(CSR_indptr_result_time + CSR_indptr_df_time)
            batch_times['time_X_CSR_data'].append(CSR_data_result_time)
            batch_times['time_X_CSR_data_df'].append(CSR_data_df_time)
            batch_times['batch_num'].append(i + 1)
            batch_times['time_obs'].append(time_diff2.total_seconds())
            batch_times['time_var'].append(time_diff3.total_seconds())
            batch_times['time_anndata'].append(time_diff4.total_seconds())
            batch_times['total_time'].append(batch_total_time)

            # 返回数据
            yield adata

            print(f"obs 表读取用时： {time_2:.2f} 秒")
            print(f"var 表读取用时： {time_3:.2f} 秒")
            print(f"生成anndata数据用时： {time_4:.2f} 秒")
            print(f"X_CSR_indptr 表 读取时间： {time_5:.2f} 秒")
            print(f"X_CSR_data 表 读取时间： {time_6:.2f} 秒")
            print(f"X_CSR_data 表 转df时间 ： {time_8:.2f} 秒")

        # 新增：在所有批次处理完成后，绘制时间图表
        self._plot_batch_times(batch_times, batch_size)

    # todo CSR格式 游标优化，df优化方法2 arrow table + 只选择需要的列df格式；
    #   Arrow -> pandas ->NumPy  当前速度 15 batch/s ， 适合任意规模的数据
    def minibatch_scan_order_cursor_csr_df_arrow_onlylie(self, total_num, batch_size, drop_last):
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

    # todo  对上面的代码去掉 orderby
    def minibatch_scan_order_cursor_csr_df_arrow_onlylie_no_orderby(self, total_num, batch_size, drop_last):
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
                query_CSR_data = f"SELECT indices, data FROM X_CSR_data WHERE id >= {start_data_id} AND id < {end_data_id}"
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

    # todo 对上面的优化 1 :    当前最快 25 batch/s . 适合少量的细胞102400。
    #                  Arrow -> NumPy（零拷贝）
    #                  data_table = self.connection.execute(query_data).fetch_arrow_table()
    #                 indices_np = data_table.column("indices").to_numpy()
    #                 data_np = data_table.column("data").to_numpy()
    def minibatch_scan_order_cursor_csr_df_arrow_onlylie_1(self, total_num, batch_size, drop_last):
        """
        最终稳定版本：
        - CSR_indptr / CSR_data：Arrow -> NumPy（高性能）
        - obs 表：退回 self.query（避免 Arrow dictionary 坑）
        - var 表：只读取一次
        """

        # ============================================================
        # 【步骤 0】var 表只读取一次（gene 维度全局不变）
        # ============================================================
        var_start = datetime.now()

        # todo: var 表结构
        #   id | gene_id | ...
        var_df = self.query("SELECT * FROM var")
        var_df = var_df.iloc[:, 1:]  # 去掉 id 列
        gene_id = var_df['gene_id'].astype(str)

        var_time = (datetime.now() - var_start).total_seconds()
        print(f"[Init] var 表读取一次耗时: {var_time:.4f}s")

        # ============================================================
        # 【步骤 1】计算 batch 数量
        # ============================================================
        if drop_last:
            num_batches = total_num // batch_size
        else:
            num_batches = (total_num + batch_size - 1) // batch_size

        # ============================================================
        # CSR_data 表游标（跨 batch 累积）
        # ============================================================
        start_data_id = 0

        total_time = 0.0

        # ============================================================
        # 【步骤 2】batch 主循环
        # ============================================================
        for i in range(num_batches):

            batch_start = datetime.now()
            offset = i * batch_size

            # todo: 最后一批 size 处理
            if (not drop_last) and (i == num_batches - 1) and (total_num % batch_size != 0):
                current_batch_size = total_num % batch_size
            else:
                current_batch_size = batch_size

            print(f"\n--- Batch {i} | size = {current_batch_size} ---")

            # ========================================================
            # 【步骤 2.1】读取 X_CSR_indptr
            # ========================================================
            # todo:
            #   Batch 0: indptr = [2, 3]
            #   Batch 1: indptr = [3, 3]
            #   Batch 2: indptr = [6, 6]

            t0 = datetime.now()

            query_indptr = (
                f"SELECT indptr FROM X_CSR_indptr "
                f"LIMIT {current_batch_size} OFFSET {offset}"
            )

            indptr_table = self.connection.execute(query_indptr).fetch_arrow_table()
            indptr_np_raw = indptr_table.column("indptr").to_numpy()

            indptr_time = (datetime.now() - t0).total_seconds()
            print(f"#### CSR_indptr Arrow->NumPy 时间: {indptr_time:.4f}s")

            # todo: 当前 batch 在 CSR_data 表中的读取终点
            end_data_id = indptr_np_raw[-1]

            # todo 【关键】
            #   在开头补 0，并减去上一批末尾 start_data_id
            #
            #   第 0 批: [0, 2, 3]
            #   第 1 批: [0, 0, 0]
            #   第 2 批: [0, 3, 3]
            indptr_np = np.concatenate([[0], indptr_np_raw - start_data_id])

            # ========================================================
            # 【步骤 2.2】读取 X_CSR_data（Arrow-only）
            # ========================================================
            data_time = 0.0

            if end_data_id > start_data_id:

                t1 = datetime.now()

                # todo:
                #   只读取当前 batch 范围内的 indices / data
                #   不使用 ORDER BY（id 已按顺序）
                query_data = (
                    f"SELECT indices, data FROM X_CSR_data "
                    f"WHERE id >= {start_data_id} AND id < {end_data_id}"
                )

                data_table = self.connection.execute(query_data).fetch_arrow_table()

                # todo: Arrow -> NumPy（零拷贝）
                indices_np = data_table.column("indices").to_numpy()
                data_np = data_table.column("data").to_numpy()


                data_time = (datetime.now() - t1).total_seconds()
                start_data_id = end_data_id  # 更新游标

            else:
                # todo: 本 batch 没有非零元素
                indices_np = np.array([], dtype=np.int32)
                data_np = np.array([], dtype=np.float32)

            print(f"#### CSR_data Arrow->NumPy 时间: {data_time:.4f}s")

            # ========================================================
            # 【步骤 2.3】读取 obs 表（回退到原始稳定方案）
            # ========================================================
            # todo:
            #   obs 表包含 dictionary-encoded 列
            #   Arrow -> Pandas 会报错，因此使用 self.query
            t2 = datetime.now()

            obs_df = self.query(
                f"SELECT * FROM obs LIMIT {current_batch_size} OFFSET {offset}"
            )

            obs_df = obs_df.iloc[:, 1:]  # 去掉 id 列
            cell_id = obs_df['cell_id'].astype(str)

            obs_time = (datetime.now() - t2).total_seconds()
            print(f"#### obs 表读取时间 (self.query): {obs_time:.4f}s")

            # ========================================================
            # 【步骤 2.4】构造 CSR 矩阵 + AnnData
            # ========================================================
            t3 = datetime.now()

            X = sp.csr_matrix(
                (data_np, indices_np, indptr_np),
                shape=(len(obs_df), len(var_df))
            )

            obs_df.index = cell_id
            var_df.index = gene_id

            adata = AnnData(X=X, obs=obs_df, var=var_df)
            # adata.obs_names = cell_id
            # adata.var_names = gene_id

            anndata_time = (datetime.now() - t3).total_seconds()
            print(f"#### AnnData 构造时间: {anndata_time:.4f}s")

            # ========================================================
            # 【步骤 2.5】batch 总耗时
            # ========================================================
            batch_time = (datetime.now() - batch_start).total_seconds()
            total_time += batch_time

            print(
                f"[Batch {i}] total={batch_time:.4f}s | "
                f"indptr={indptr_time:.4f}s | "
                f"data={data_time:.4f}s | "
                f"obs={obs_time:.4f}s | "
                f"anndata={anndata_time:.4f}s"
            )

            yield adata

        # ============================================================
        # 【步骤 3】整体统计
        # ============================================================
        print("\n================ Summary ================")
        print(f"Total batches        : {num_batches}")
        print(f"Total time (s)       : {total_time:.3f}")
        print(f"Time per batch (s)   : {total_time / num_batches:.4f}")
        print(f"Batches per second   : {num_batches / total_time:.2f}")

    # todo CSR格式 游标优化，df优化方法2 arrow table + 只选择需要的列df格式
    def minibatch_scan_order_cursor_csr_df_arrow_onlylie_2(self, total_num, batch_size, drop_last):
        """按顺序读取模式"""
        # 计算批次数量
        time_2 = 0  # obs 表读取用时
        time_3 = 0  # var 表读取用时
        time_4 = 0  # 生成anndata数据用时
        time_5 = 0  # X_CSR_indptr 表 读取时间
        time_6 = 0  # X_CSR_data 表 读取时间
        time_8 = 0  # X_CSR_data 转 DF 时间

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

            # todo: Arrow -> NumPy（零拷贝）
            data_table = self.connection.execute(query_CSR_indptr).fetch_arrow_table()
            CSR_indptr_array = data_table.column("indptr").to_numpy()

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
            CSR_indptr_array = np.concatenate([[0], CSR_indptr_array['indptr'].values - start_data_id])
            # print(f"indptr[-1]   {CSR_indptr_array[-1]}")
            # todo: 第 0 批
            #  CSR_indptr_array = { 0 , 2 , 3  } - 0 = { 0 , 2 , 3  }
            # todo: 第 1 批
            #  CSR_indptr_array = { 0 , 3 , 3  } - 3 = { 0 , 0 , 0  }
            # todo: 第 2 批
            #  CSR_indptr_array = { 0 , 6 , 6  } - 3 = { 0 , 3 , 3  }

            if (end_data_id - start_data_id > 0):

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

            # 创建CSR格式的稀疏矩阵
            X = sp.csr_matrix((CSR_data_array, CSR_indices_array, CSR_indptr_array),
                              shape=(len(sub_obs), len(sub_var)))
            # 创建AnnData对象
            adata = AnnData(X=X, obs=sub_obs, var=sub_var)

            adata.obs_names = cell_id.astype(str)  # 设置观测名为cell_id
            adata.obs.index = cell_id.astype(str)
            adata.var_names = gene_id.astype(str)  # 设置变量名为gene_id
            adata.var.index = gene_id.astype(str)

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

    # todo 对上面 onlylie 的优化 :  Arrow -> NumPy（零拷贝） +  彻底移除 OFFSET， 使用游标
    def minibatch_scan_order_cursor_csr_df_arrow_onlylie_3(
            self,
            total_num,
            batch_size,
            drop_last
    ):
        """
        最终稳定版本（无 zero_copy_only 参数）：
        - obs：回退 self.query（避免 dictionary / pandas 转换雷区）
        - CSR_indptr / CSR_data：Arrow → NumPy（combine_chunks 后自动拷贝）
        - 全 cursor 扫描（无 OFFSET）
        """

        # ============================================================
        # 0. var（一次性读取）
        # ============================================================
        var_df = self.query("SELECT * FROM var").iloc[:, 1:]
        gene_id = var_df["gene_id"].astype(str)
        var_df.index = gene_id

        # ============================================================
        # 1. 游标初始化
        # ============================================================
        obs_cursor_id = 0
        indptr_cursor_id = 0
        data_cursor_id = 0

        total_batches = 0
        start_time = datetime.now()

        # ============================================================
        # 2. 主循环
        # ============================================================
        while True:

            # --------------------------------------------------------
            # 2.1 obs（稳定路径）
            # --------------------------------------------------------
            obs_df = self.query(
                f"""
                SELECT *
                FROM obs
                WHERE id >= {obs_cursor_id}
                ORDER BY id
                LIMIT {batch_size}
                """
            )

            if len(obs_df) == 0:
                break

            obs_df = obs_df.iloc[:, 1:]
            obs_df.index = obs_df["cell_id"].astype(str)

            current_batch_size = len(obs_df)

            if drop_last and current_batch_size < batch_size:
                break

            obs_cursor_id += current_batch_size

            # --------------------------------------------------------
            # 2.2 indptr（Arrow → NumPy）
            # --------------------------------------------------------
            indptr_table = self.connection.execute(
                f"""
                SELECT indptr
                FROM X_CSR_indptr
                WHERE id >= {indptr_cursor_id}
                ORDER BY id
                LIMIT {current_batch_size}
                """
            ).fetch_arrow_table()

            indptr_raw = indptr_table.column("indptr").combine_chunks().to_numpy()

            end_data_id = indptr_raw[-1]

            indptr_np = np.concatenate(([0], indptr_raw - data_cursor_id))

            indptr_cursor_id += current_batch_size

            # --------------------------------------------------------
            # 2.3 data / indices
            # --------------------------------------------------------
            if end_data_id > data_cursor_id:

                data_table = self.connection.execute(
                    f"""
                    SELECT indices, data
                    FROM X_CSR_data
                    WHERE id >= {data_cursor_id}
                      AND id < {end_data_id}
                    """
                ).fetch_arrow_table().combine_chunks()

                indices_np = data_table.column("indices").to_numpy()
                data_np = data_table.column("data").to_numpy()

                data_cursor_id = end_data_id

            else:
                indices_np = np.empty(0, dtype=np.int32)
                data_np = np.empty(0, dtype=np.float32)

            # --------------------------------------------------------
            # 2.4 AnnData 构造
            # --------------------------------------------------------
            X = sp.csr_matrix(
                (data_np, indices_np, indptr_np),
                shape=(current_batch_size, len(var_df))
            )

            adata = AnnData(X=X, obs=obs_df, var=var_df)

            yield adata
            total_batches += 1

            if total_batches * batch_size >= total_num:
                break

        # ============================================================
        # 3. 统计
        # ============================================================
        total_time = (datetime.now() - start_time).total_seconds()

        print("\n================ Summary (Final) ================")
        print(f"Total batches        : {total_batches}")
        print(f"Total time (s)       : {total_time:.3f}")
        print(f"Time per batch (s)   : {total_time / max(total_batches, 1):.4f}")
        print(f"Batches per second   : {total_batches / total_time:.2f}")

    # todo 对上面 onlylie 的优化 :  Arrow -> NumPy（零拷贝） +  彻底移除 OFFSET， 使用游标
    def minibatch_scan_order_cursor_csr_df_arrow_onlylie_4(self, total_num, batch_size, drop_last):
        """按顺序读取模式（CSR 表 Arrow → NumPy 零拷贝优化版）"""

        time_2 = 0  # obs 表读取用时
        time_3 = 0  # var 表读取用时
        time_4 = 0  # 生成anndata数据用时
        time_5 = 0  # X_CSR_indptr 表 读取时间（保留但不细分）
        time_6 = 0  # X_CSR_data 表 读取时间（保留但不细分）
        time_8 = 0  # X_CSR_data Arrow->NumPy 时间

        start_data_id = 0  # CSR_data 游标起点

        if drop_last:
            num_batches = total_num // batch_size
        else:
            num_batches = (total_num + batch_size - 1) // batch_size

        for i in range(num_batches):
            offset = i * batch_size

            if not drop_last and i == num_batches - 1 and total_num % batch_size != 0:
                current_batch_size = total_num % batch_size
            else:
                current_batch_size = batch_size

            print(f"\n--- 批次 {i} ---")
            print(f"--- 批次大小 {current_batch_size} ---")

            # ======================================================
            # 1️⃣ X_CSR_indptr（Arrow → NumPy 零拷贝）
            # ======================================================
            query_CSR_indptr = (
                f"SELECT indptr FROM X_CSR_indptr "
                f"LIMIT {current_batch_size} OFFSET {offset}"
            )

            CSR_indptr_start = datetime.now()

            indptr_table = self.connection.execute(
                query_CSR_indptr
            ).fetch_arrow_table()

            # ✅ Arrow → NumPy（零拷贝）
            CSR_indptr_raw = indptr_table.column("indptr").to_numpy()

            CSR_indptr_end = datetime.now()
            print(
                f"#### CSR_indptr Arrow->NumPy 时间 : "
                f"{(CSR_indptr_end - CSR_indptr_start).total_seconds():.6f}s"
            )

            # CSR_data 读取终点（全局 id）
            end_data_id = CSR_indptr_raw[-1]

            # 构造 batch 内 indptr
            CSR_indptr_array = np.concatenate(
                [[0], CSR_indptr_raw - start_data_id]
            )

            # ======================================================
            # 2️⃣ X_CSR_data（Arrow → NumPy 零拷贝）
            # ======================================================
            if end_data_id - start_data_id > 0:
                query_CSR_data = f"""
                    SELECT indices, data
                    FROM X_CSR_data
                    WHERE id >= {start_data_id} AND id < {end_data_id}
                    ORDER BY id
                """

                CSR_data_start = datetime.now()

                data_table = self.connection.execute(
                    query_CSR_data
                ).fetch_arrow_table()

                # ✅ Arrow → NumPy（零拷贝）
                CSR_indices_array = data_table.column("indices").to_numpy()
                CSR_data_array = data_table.column("data").to_numpy()

                CSR_data_end = datetime.now()
                csr_data_time = (CSR_data_end - CSR_data_start).total_seconds()
                print(f"#### CSR_data Arrow->NumPy 时间 : {csr_data_time:.6f}s")

                start_data_id = end_data_id
                time_8 += csr_data_time

            else:
                print("获取 0 个值，不查询")

                CSR_indices_array = np.array([], dtype=np.int64)
                CSR_data_array = np.array([], dtype=np.float32)

            # ======================================================
            # 3️⃣ obs 表（原样不动）
            # ======================================================
            start_time2 = datetime.now()
            sub_obs = self.query(
                f"SELECT * FROM obs LIMIT {current_batch_size} OFFSET {offset}"
            )
            end_time2 = datetime.now()
            time_2 += (end_time2 - start_time2).total_seconds()

            # ======================================================
            # 4️⃣ var 表（原样不动）
            # ======================================================
            start_time3 = datetime.now()
            sub_var = self.query("SELECT * FROM var")
            end_time3 = datetime.now()
            time_3 += (end_time3 - start_time3).total_seconds()

            # ======================================================
            # 5️⃣ 构建 AnnData（原样不动）
            # ======================================================
            start_time4 = datetime.now()

            sub_obs = sub_obs.iloc[:, 1:]
            sub_var = sub_var.iloc[:, 1:]

            cell_id = sub_obs["cell_id"].astype(str)
            gene_id = sub_var["gene_id"].astype(str)

            sub_obs.index = cell_id
            sub_var.index = gene_id

            X = sp.csr_matrix(
                (CSR_data_array, CSR_indices_array, CSR_indptr_array),
                shape=(len(sub_obs), len(sub_var))
            )

            adata = AnnData(X=X, obs=sub_obs, var=sub_var)

            end_time4 = datetime.now()
            time_4 += (end_time4 - start_time4).total_seconds()

            # ======================================================
            # 6️⃣ 输出 & 返回
            # ======================================================
            print(f"obs 表累计用时： {time_2:.2f} 秒")
            print(f"var 表累计用时： {time_3:.2f} 秒")
            print(f"生成 AnnData 累计用时： {time_4:.2f} 秒")
            print(f"X_CSR_data Arrow->NumPy 累计时间： {time_8:.2f} 秒")

            yield adata

    # todo 对上面的代码 onlylie 进行优化:
    #   一次查 8192 cell（Mega-batch）
    #   → Arrow -> pandas ->NumPy
    #   → 在 NumPy 层面切 4 个 2048 子 batch
    #   → 每个子 batch 构造 CSR
    def minibatch_scan_order_cursor_csr_df_arrow_onlylie_mega1(
            self,
            total_num,
            batch_size=2048,
            mega_batch_size=4096,
            drop_last=False
    ):
        """
        Mega-batch 优化版本：
        - DB 层一次查 mega_batch_size（如 8192）
        - Arrow → pandas → NumPy 只发生一次
        - NumPy 层切成多个 2048 子 batch
        """

        assert mega_batch_size % batch_size == 0
        sub_batches = mega_batch_size // batch_size

        # =========================
        # 1️⃣ var 表只查一次（重大优化）
        # =========================
        sub_var = self.query("SELECT * FROM var")
        sub_var = sub_var.iloc[:, 1:]
        gene_id = sub_var["gene_id"].astype(str)

        start_data_id = 0  # CSR_data 游标

        for mega_start in range(0, total_num, mega_batch_size):

            mega_size = min(mega_batch_size, total_num - mega_start)
            if drop_last and mega_size < mega_batch_size:
                break

            print(f"\n🚀 Mega-batch [{mega_start}:{mega_start + mega_size}]")

            # =========================
            # 2️⃣ 一次读取 indptr
            # =========================
            indptr_df = (
                self.connection
                .execute(
                    f"""
                    SELECT indptr
                    FROM X_CSR_indptr
                    LIMIT {mega_size}
                    OFFSET {mega_start}
                    """
                )
                .fetch_arrow_table()
                .to_pandas()
            )

            raw_indptr = indptr_df["indptr"].to_numpy()
            end_data_id = raw_indptr[-1]

            # 归一化 indptr（以当前 mega-batch 起点为 0）
            mega_indptr = np.concatenate([[0], raw_indptr - start_data_id])

            # =========================
            # 3️⃣ 一次读取 CSR_data
            # =========================
            if end_data_id > start_data_id:
                csr_df = (
                    self.connection
                    .execute(
                        f"""
                        SELECT indices, data
                        FROM X_CSR_data
                        WHERE id >= {start_data_id}
                          AND id < {end_data_id}
                        ORDER BY id
                        """
                    )
                    .fetch_arrow_table()
                    .to_pandas()
                )

                csr_indices = csr_df["indices"].to_numpy()
                csr_data = csr_df["data"].to_numpy()
            else:
                csr_indices = np.empty(0, dtype=np.int32)
                csr_data = np.empty(0, dtype=np.float32)

            start_data_id = end_data_id

            # =========================
            # 4️⃣ obs 一次读取
            # =========================
            obs_df = self.query(
                f"""
                SELECT * FROM obs
                LIMIT {mega_size}
                OFFSET {mega_start}
                """
            )
            obs_df = obs_df.iloc[:, 1:]
            cell_ids = obs_df["cell_id"].astype(str)

            # =========================
            # 5️⃣ NumPy 层切子 batch
            # =========================
            for b in range(sub_batches):
                sub_start = b * batch_size
                sub_end = min((b + 1) * batch_size, mega_size)
                if sub_start >= sub_end:
                    break

                # 子 indptr
                sub_indptr = mega_indptr[sub_start: sub_end + 1]
                data_start = sub_indptr[0]
                data_end = sub_indptr[-1]

                sub_indptr = sub_indptr - data_start

                sub_indices = csr_indices[data_start:data_end]
                sub_data = csr_data[data_start:data_end]

                sub_obs = obs_df.iloc[sub_start:sub_end]
                sub_cell_ids = cell_ids.iloc[sub_start:sub_end]

                sub_obs.index = sub_cell_ids
                sub_var.index = gene_id

                X = sp.csr_matrix(
                    (sub_data, sub_indices, sub_indptr),
                    shape=(len(sub_obs), len(sub_var))
                )

                adata = AnnData(X=X, obs=sub_obs, var=sub_var)

                yield adata

    # todo 对上面的代码 onlylie_mega1 加上时间统计分析
    def minibatch_scan_order_cursor_csr_df_arrow_onlylie_mega2(
            self,
            total_num,
            batch_size=2048,
            mega_batch_size=8192,
            drop_last=False
    ):
        """
        Mega-batch 优化版本（带详细时间统计）：
        - DB 层一次查 mega_batch_size
        - Arrow → pandas → NumPy 只发生一次
        - NumPy 层切子 batch
        """

        import time
        import numpy as np
        import scipy.sparse as sp
        from anndata import AnnData

        assert mega_batch_size % batch_size == 0
        sub_batches = mega_batch_size // batch_size

        # =========================
        # ⏱ 全局时间累计
        # =========================
        T = {
            "indptr_query": 0.0,
            "csr_data_query": 0.0,
            "obs_query": 0.0,
            "arrow_to_pandas": 0.0,
            "numpy_slice": 0.0,
            "csr_build": 0.0,
            "anndata_build": 0.0,
        }

        # =========================
        # 1️⃣ var 表只查一次
        # =========================
        t0 = time.perf_counter()
        sub_var = self.query("SELECT * FROM var")
        sub_var = sub_var.iloc[:, 1:]
        gene_id = sub_var["gene_id"].astype(str)
        T["obs_query"] += time.perf_counter() - t0

        start_data_id = 0  # CSR_data 游标

        for mega_start in range(0, total_num, mega_batch_size):

            mega_size = min(mega_batch_size, total_num - mega_start)
            if drop_last and mega_size < mega_batch_size:
                break

            print(f"\n🚀 Mega-batch [{mega_start}:{mega_start + mega_size}]")

            # =========================
            # 2️⃣ indptr 查询 + Arrow → pandas
            # =========================
            t0 = time.perf_counter()
            indptr_arrow = (
                self.connection
                .execute(
                    f"""
                    SELECT indptr
                    FROM X_CSR_indptr
                    LIMIT {mega_size}
                    OFFSET {mega_start}
                    """
                )
                .fetch_arrow_table()
            )
            T["indptr_query"] += time.perf_counter() - t0

            t0 = time.perf_counter()
            indptr_df = indptr_arrow.to_pandas()
            raw_indptr = indptr_df["indptr"].to_numpy()
            T["arrow_to_pandas"] += time.perf_counter() - t0

            end_data_id = raw_indptr[-1]
            mega_indptr = np.concatenate([[0], raw_indptr - start_data_id])

            # =========================
            # 3️⃣ CSR_data 查询 + Arrow → pandas
            # =========================
            if end_data_id > start_data_id:
                t0 = time.perf_counter()
                csr_arrow = (
                    self.connection
                    .execute(
                        f"""
                        SELECT indices, data
                        FROM X_CSR_data
                        WHERE id >= {start_data_id}
                          AND id < {end_data_id}
                        ORDER BY id
                        """
                    )
                    .fetch_arrow_table()
                )
                T["csr_data_query"] += time.perf_counter() - t0

                t0 = time.perf_counter()
                csr_df = csr_arrow.to_pandas()
                csr_indices = csr_df["indices"].to_numpy()
                csr_data = csr_df["data"].to_numpy()
                T["arrow_to_pandas"] += time.perf_counter() - t0
            else:
                csr_indices = np.empty(0, dtype=np.int32)
                csr_data = np.empty(0, dtype=np.float32)

            start_data_id = end_data_id

            # =========================
            # 4️⃣ obs 查询
            # =========================
            t0 = time.perf_counter()
            obs_df = self.query(
                f"""
                SELECT * FROM obs
                LIMIT {mega_size}
                OFFSET {mega_start}
                """
            )
            obs_df = obs_df.iloc[:, 1:]
            cell_ids = obs_df["cell_id"].astype(str)
            T["obs_query"] += time.perf_counter() - t0

            # =========================
            # 5️⃣ NumPy 切子 batch + CSR / AnnData
            # =========================
            for b in range(sub_batches):
                sub_start = b * batch_size
                sub_end = min((b + 1) * batch_size, mega_size)
                if sub_start >= sub_end:
                    break

                # ---- NumPy slice
                t0 = time.perf_counter()
                sub_indptr = mega_indptr[sub_start: sub_end + 1]
                data_start = sub_indptr[0]
                data_end = sub_indptr[-1]
                sub_indptr = sub_indptr - data_start

                sub_indices = csr_indices[data_start:data_end]
                sub_data = csr_data[data_start:data_end]
                T["numpy_slice"] += time.perf_counter() - t0

                sub_obs = obs_df.iloc[sub_start:sub_end]
                sub_cell_ids = cell_ids.iloc[sub_start:sub_end]
                sub_obs.index = sub_cell_ids
                sub_var.index = gene_id

                # ---- CSR build
                t0 = time.perf_counter()
                X = sp.csr_matrix(
                    (sub_data, sub_indices, sub_indptr),
                    shape=(len(sub_obs), len(sub_var))
                )
                T["csr_build"] += time.perf_counter() - t0

                # ---- AnnData build
                t0 = time.perf_counter()
                adata = AnnData(X=X, obs=sub_obs, var=sub_var)
                T["anndata_build"] += time.perf_counter() - t0

                yield adata

            # =========================
            # 📊 每个 Mega-batch 打印一次统计
            # =========================
            print(
                f"⏱ Mega-batch time breakdown:\n"
                f"  indptr_query     : {T['indptr_query']:.4f}s\n"
                f"  csr_data_query   : {T['csr_data_query']:.4f}s\n"
                f"  obs_query        : {T['obs_query']:.4f}s\n"
                f"  arrow→pandas     : {T['arrow_to_pandas']:.4f}s\n"
                f"  numpy_slice      : {T['numpy_slice']:.4f}s\n"
                f"  csr_build        : {T['csr_build']:.4f}s\n"
                f"  anndata_build    : {T['anndata_build']:.4f}s"
            )


    # todo 对上面的代码 onlylie_mega2：  ORDER BY
    def minibatch_scan_order_cursor_csr_df_arrow_onlylie_mega1_no_orderby(
            self,
            total_num,
            batch_size=2048,
            mega_batch_size=2048*4,
            drop_last=False
    ):
        """
        方案 A：
        - 完全去掉 X_CSR_data 的 ORDER BY id
        - 其余逻辑 100% 保持一致
        """

        import time
        import numpy as np
        import scipy.sparse as sp
        from anndata import AnnData

        assert mega_batch_size % batch_size == 0
        sub_batches = mega_batch_size // batch_size

        # =========================
        # ⏱ 时间统计
        # =========================
        T = {
            "indptr_query": 0.0,
            "csr_data_query": 0.0,
            "obs_query": 0.0,
            "arrow_to_pandas": 0.0,
            "numpy_slice": 0.0,
            "csr_build": 0.0,
            "anndata_build": 0.0,
        }

        # =========================
        # 1️⃣ var 表只查一次
        # =========================
        t0 = time.perf_counter()
        sub_var = self.query("SELECT * FROM var")
        sub_var = sub_var.iloc[:, 1:]
        gene_id = sub_var["gene_id"].astype(str)
        T["obs_query"] += time.perf_counter() - t0

        start_data_id = 0  # CSR_data 游标

        for mega_start in range(0, total_num, mega_batch_size):

            mega_size = min(mega_batch_size, total_num - mega_start)
            if drop_last and mega_size < mega_batch_size:
                break

            print(f"\n🚀 Mega-batch [{mega_start}:{mega_start + mega_size}]")

            # =========================
            # 2️⃣ indptr 查询
            # =========================
            t0 = time.perf_counter()
            indptr_arrow = (
                self.connection
                .execute(
                    f"""
                    SELECT indptr
                    FROM X_CSR_indptr
                    LIMIT {mega_size}
                    OFFSET {mega_start}
                    """
                )
                .fetch_arrow_table()
            )
            T["indptr_query"] += time.perf_counter() - t0

            t0 = time.perf_counter()
            indptr_df = indptr_arrow.to_pandas()
            raw_indptr = indptr_df["indptr"].to_numpy()
            T["arrow_to_pandas"] += time.perf_counter() - t0

            end_data_id = raw_indptr[-1]
            mega_indptr = np.concatenate([[0], raw_indptr - start_data_id])

            # =========================
            # 3️⃣ CSR_data 查询（🔥去掉 ORDER BY🔥）
            # =========================
            if end_data_id > start_data_id:
                t0 = time.perf_counter()
                csr_arrow = (
                    self.connection
                    .execute(
                        f"""
                        SELECT indices, data
                        FROM X_CSR_data
                        WHERE id >= {start_data_id}
                          AND id < {end_data_id}
                        """
                    )
                    .fetch_arrow_table()
                )
                T["csr_data_query"] += time.perf_counter() - t0

                t0 = time.perf_counter()
                csr_df = csr_arrow.to_pandas()
                csr_indices = csr_df["indices"].to_numpy()
                csr_data = csr_df["data"].to_numpy()
                T["arrow_to_pandas"] += time.perf_counter() - t0
            else:
                csr_indices = np.empty(0, dtype=np.int32)
                csr_data = np.empty(0, dtype=np.float32)

            start_data_id = end_data_id

            # =========================
            # 4️⃣ obs 查询
            # =========================
            t0 = time.perf_counter()
            obs_df = self.query(
                f"""
                SELECT * FROM obs
                LIMIT {mega_size}
                OFFSET {mega_start}
                """
            )
            obs_df = obs_df.iloc[:, 1:]
            cell_ids = obs_df["cell_id"].astype(str)
            T["obs_query"] += time.perf_counter() - t0

            # =========================
            # 5️⃣ 子 batch CSR 构造
            # =========================
            for b in range(sub_batches):
                sub_start = b * batch_size
                sub_end = min((b + 1) * batch_size, mega_size)
                if sub_start >= sub_end:
                    break

                # NumPy slice
                t0 = time.perf_counter()
                sub_indptr = mega_indptr[sub_start: sub_end + 1]
                data_start = sub_indptr[0]
                data_end = sub_indptr[-1]
                sub_indptr = sub_indptr - data_start

                sub_indices = csr_indices[data_start:data_end]
                sub_data = csr_data[data_start:data_end]
                T["numpy_slice"] += time.perf_counter() - t0

                sub_obs = obs_df.iloc[sub_start:sub_end]
                sub_cell_ids = cell_ids.iloc[sub_start:sub_end]
                sub_obs.index = sub_cell_ids
                sub_var.index = gene_id

                # CSR
                t0 = time.perf_counter()
                X = sp.csr_matrix(
                    (sub_data, sub_indices, sub_indptr),
                    shape=(len(sub_obs), len(sub_var))
                )
                T["csr_build"] += time.perf_counter() - t0

                # AnnData
                t0 = time.perf_counter()
                adata = AnnData(X=X, obs=sub_obs, var=sub_var)
                T["anndata_build"] += time.perf_counter() - t0

                yield adata

            # =========================
            # 📊 Mega-batch 时间打印
            # =========================
            print(
                f"⏱ Mega-batch time breakdown (NO ORDER BY):\n"
                f"  indptr_query     : {T['indptr_query']:.4f}s\n"
                f"  csr_data_query   : {T['csr_data_query']:.4f}s\n"
                f"  obs_query        : {T['obs_query']:.4f}s\n"
                f"  arrow→pandas     : {T['arrow_to_pandas']:.4f}s\n"
                f"  numpy_slice      : {T['numpy_slice']:.4f}s\n"
                f"  csr_build        : {T['csr_build']:.4f}s\n"
                f"  anndata_build    : {T['anndata_build']:.4f}s"
            )

    # todo 对上面的代码 mega1_no_orderby ：  只构造 1 个 CSR  + 子 batch 只是 在这个 CSR 上做“行切片视图”
    def minibatch_scan_order_cursor_csr_df_arrow_onlylie_megaCSR(
            self,
            total_num,
            batch_size=2048,
            mega_batch_size=4096,
            drop_last=False
    ):
        """
        终极结构版：
        - Mega-batch 只构造 1 个 CSR
        - 子 batch 通过 CSR 行切片
        """

        import time
        import numpy as np
        import scipy.sparse as sp
        from anndata import AnnData

        assert mega_batch_size % batch_size == 0
        sub_batches = mega_batch_size // batch_size

        # =========================
        # 时间统计
        # =========================
        T = {
            "indptr_query": 0.0,
            "csr_data_query": 0.0,
            "obs_query": 0.0,
            "arrow_to_pandas": 0.0,
            "csr_build": 0.0,
            "csr_slice": 0.0,
            "anndata_build": 0.0,
        }

        # =========================
        # var 表只查一次
        # =========================
        t0 = time.perf_counter()
        sub_var = self.query("SELECT * FROM var")
        sub_var = sub_var.iloc[:, 1:]
        gene_id = sub_var["gene_id"].astype(str)
        T["obs_query"] += time.perf_counter() - t0

        start_data_id = 0

        for mega_start in range(0, total_num, mega_batch_size):

            mega_size = min(mega_batch_size, total_num - mega_start)
            if drop_last and mega_size < mega_batch_size:
                break

            print(f"\n🚀 Mega-batch [{mega_start}:{mega_start + mega_size}]")

            # =========================
            # indptr
            # =========================
            t0 = time.perf_counter()
            indptr_arrow = (
                self.connection
                .execute(
                    f"""
                    SELECT indptr
                    FROM X_CSR_indptr
                    LIMIT {mega_size}
                    OFFSET {mega_start}
                    """
                )
                .fetch_arrow_table()
            )
            T["indptr_query"] += time.perf_counter() - t0

            t0 = time.perf_counter()
            raw_indptr = indptr_arrow.to_pandas()["indptr"].to_numpy()
            T["arrow_to_pandas"] += time.perf_counter() - t0

            end_data_id = raw_indptr[-1]
            mega_indptr = np.concatenate([[0], raw_indptr - start_data_id])

            # =========================
            # csr_data（NO ORDER BY）
            # =========================
            if end_data_id > start_data_id:
                t0 = time.perf_counter()
                csr_arrow = (
                    self.connection
                    .execute(
                        f"""
                        SELECT indices, data
                        FROM X_CSR_data
                        WHERE id >= {start_data_id}
                          AND id < {end_data_id}
                        """
                    )
                    .fetch_arrow_table()
                )
                T["csr_data_query"] += time.perf_counter() - t0

                t0 = time.perf_counter()
                csr_df = csr_arrow.to_pandas()
                csr_indices = csr_df["indices"].to_numpy()
                csr_data = csr_df["data"].to_numpy()
                T["arrow_to_pandas"] += time.perf_counter() - t0
            else:
                csr_indices = np.empty(0, dtype=np.int32)
                csr_data = np.empty(0, dtype=np.float32)

            start_data_id = end_data_id

            # =========================
            # obs
            # =========================
            t0 = time.perf_counter()
            obs_df = self.query(
                f"""
                SELECT * FROM obs
                LIMIT {mega_size}
                OFFSET {mega_start}
                """
            )
            obs_df = obs_df.iloc[:, 1:]
            cell_ids = obs_df["cell_id"].astype(str)
            T["obs_query"] += time.perf_counter() - t0

            # =========================
            # 🔥 Mega CSR 一次构造
            # =========================
            t0 = time.perf_counter()
            X_mega = sp.csr_matrix(
                (csr_data, csr_indices, mega_indptr),
                shape=(mega_size, len(sub_var))
            )
            T["csr_build"] += time.perf_counter() - t0

            # =========================
            # 子 batch：CSR 行切片
            # =========================
            for b in range(sub_batches):
                sub_start = b * batch_size
                sub_end = min((b + 1) * batch_size, mega_size)
                if sub_start >= sub_end:
                    break

                t0 = time.perf_counter()
                X_sub = X_mega[sub_start:sub_end]
                T["csr_slice"] += time.perf_counter() - t0

                sub_obs = obs_df.iloc[sub_start:sub_end]
                sub_cell_ids = cell_ids.iloc[sub_start:sub_end]
                sub_obs.index = sub_cell_ids
                sub_var.index = gene_id

                t0 = time.perf_counter()
                adata = AnnData(X=X_sub, obs=sub_obs, var=sub_var)
                T["anndata_build"] += time.perf_counter() - t0

                yield adata

            # =========================
            # 打印时间
            # =========================
            print(
                f"⏱ Mega-CSR time breakdown:\n"
                f"  indptr_query   : {T['indptr_query']:.4f}s\n"
                f"  csr_data_query : {T['csr_data_query']:.4f}s\n"
                f"  arrow→pandas   : {T['arrow_to_pandas']:.4f}s\n"
                f"  csr_build      : {T['csr_build']:.4f}s\n"
                f"  csr_slice      : {T['csr_slice']:.4f}s\n"
                f"  anndata_build  : {T['anndata_build']:.4f}s"
            )

    # todo 对上面 onlylie_1 的优化:   无收益，负优化
    #   一次查 8192 cell（Mega-batch）
    #   → Arrow → NumPy（只 1 次）  Arrow -> pandas ->NumPy
    #   → 在 NumPy 层面切 4 个 2048 子 batch
    #   → 每个子 batch 构造 CSR
    def minibatch_scan_order_cursor_csr_df_arrow_onlylie_mega_final(
            self,
            total_num: int,
            batch_size: int = 2048,
            mega_batch_size: int = 4096,
            drop_last: bool = False,
    ):
        """
        最终交付版本（计时语义完全正确）：

        ✔ Arrow → NumPy 使用 Mega-batch（显著减少次数）
        ✔ Time per batch / batch/s = wall-clock（唯一可信）
        ✔ 内部 indptr / CSR_data / obs / AnnData 时间仅用于 debug
        """

        from datetime import datetime
        import numpy as np
        import scipy.sparse as sp
        from anndata import AnnData

        # ============================================================
        # 【全局 wall-clock 开始时间】—— 唯一用于性能统计
        # ============================================================
        func_start_time = datetime.now()

        # ============================================================
        # 【Step 0】var 表：只读取一次
        # ============================================================
        var_df = self.query("SELECT * FROM var")
        var_df = var_df.iloc[:, 1:]
        var_df.index = var_df["gene_id"].astype(str)

        # ============================================================
        # 【Step 1】CSR_data 全局游标
        # ============================================================
        start_data_id = 0

        # ============================================================
        # 【Step 2】Mega-batch 数量
        # ============================================================
        num_mega_batches = (total_num + mega_batch_size - 1) // mega_batch_size

        total_batches = 0

        # ============================================================
        # 【Step 3】Mega-batch 主循环
        # ============================================================
        for mega_i in range(num_mega_batches):

            mega_offset = mega_i * mega_batch_size
            mega_batch = min(mega_batch_size, total_num - mega_offset)

            print(f"\n================ Mega Batch {mega_i} | size={mega_batch} ================")

            # --------------------------------------------------------
            # 3.1 读取 Mega indptr（Arrow → NumPy）
            # --------------------------------------------------------
            t_indptr = datetime.now()

            indptr_table = self.connection.execute(
                f"SELECT indptr FROM X_CSR_indptr "
                f"LIMIT {mega_batch} OFFSET {mega_offset}"
            ).fetch_arrow_table()

            indptr_raw = indptr_table.column("indptr").to_numpy()

            indptr_time = (datetime.now() - t_indptr).total_seconds()

            # --------------------------------------------------------
            # 3.2 读取 Mega CSR_data（Arrow → NumPy）
            # --------------------------------------------------------
            t_data = datetime.now()

            mega_end_data_id = indptr_raw[-1]

            if mega_end_data_id > start_data_id:
                data_table = self.connection.execute(
                    f"SELECT indices, data FROM X_CSR_data "
                    f"WHERE id >= {start_data_id} AND id < {mega_end_data_id}"
                ).fetch_arrow_table()

                mega_indices = data_table.column("indices").to_pandas().to_numpy()



                mega_data = data_table.column("data").to_pandas().to_numpy()


            else:
                mega_indices = np.array([], dtype=np.int32)
                mega_data = np.array([], dtype=np.float32)

            data_time = (datetime.now() - t_data).total_seconds()

            mega_data_offset = start_data_id
            start_data_id = mega_end_data_id

            print(
                f"[Mega {mega_i}] indptr={indptr_time:.4f}s | "
                f"CSR_data={data_time:.4f}s"
            )

            # ========================================================
            # 3.3 NumPy 层切 sub-batch
            # ========================================================
            num_sub_batches = (mega_batch + batch_size - 1) // batch_size

            for sub_i in range(num_sub_batches):

                row_start = sub_i * batch_size
                row_end = min(row_start + batch_size, mega_batch)
                current_batch_size = row_end - row_start

                if drop_last and current_batch_size < batch_size:
                    continue

                # ---------------- 子 batch indptr ----------------
                sub_indptr_raw = indptr_raw[row_start:row_end]

                data_start = sub_indptr_raw[0] - mega_data_offset
                data_end = sub_indptr_raw[-1] - mega_data_offset

                sub_indices = mega_indices[data_start:data_end]
                sub_data = mega_data[data_start:data_end]

                sub_indptr = np.concatenate(
                    [[0], sub_indptr_raw - sub_indptr_raw[0]]
                )

                # ---------------- obs 表 ----------------
                t_obs = datetime.now()

                obs_df = self.query(
                    f"SELECT * FROM obs "
                    f"LIMIT {current_batch_size} OFFSET {mega_offset + row_start}"
                ).iloc[:, 1:]

                obs_df.index = obs_df["cell_id"].astype(str)

                obs_time = (datetime.now() - t_obs).total_seconds()

                # ---------------- CSR + AnnData ----------------
                t_build = datetime.now()

                X = sp.csr_matrix(
                    (sub_data, sub_indices, sub_indptr),
                    shape=(current_batch_size, len(var_df)),
                )

                adata = AnnData(X=X, obs=obs_df, var=var_df)

                build_time = (datetime.now() - t_build).total_seconds()

                print(
                    f"[Batch {total_batches}] "
                    f"obs={obs_time:.4f}s | anndata={build_time:.4f}s"
                )

                total_batches += 1
                yield adata

        # ============================================================
        # 【Summary】—— 只使用 wall-clock
        # ============================================================
        func_end_time = datetime.now()
        wall_time = (func_end_time - func_start_time).total_seconds()

        print("\n================ Summary ================")
        print(f"Total batches        : {total_batches}")
        print(f"Total time (s)       : {wall_time:.3f}")
        print(f"Time per batch (s)   : {wall_time / max(total_batches, 1):.4f}")
        print(f"Batches per second   : {total_batches / max(wall_time, 1e-9):.2f}")


    # todo 对上面 onlylie_1 的优化2
    #  当前 Batch i:
    #       ③ 读 obs
    #       ④ 构造 AnnData
    #   同时后台线程：
    #       ② 预读 Batch i+1 的 CSR_data
    def minibatch_scan_order_cursor_csr_df_arrow_prefetch(
            self, total_num, batch_size, drop_last
    ):
        """
        Prefetch 终局版本：
        - batch_size = 2048（cache sweet spot）
        - CSR_data 在后台线程预读
        - 主线程构造 AnnData
        - 计时为完整 wall time
        """

        import threading
        from datetime import datetime
        import numpy as np
        import scipy.sparse as sp
        from anndata import AnnData

        func_start_time = datetime.now()

        # ============================================================
        # 【0】var 表只读一次
        # ============================================================
        var_df = self.query("SELECT * FROM var").iloc[:, 1:]
        gene_id = var_df["gene_id"].astype(str)
        var_df.index = gene_id

        # ============================================================
        # 【1】batch 数量
        # ============================================================
        if drop_last:
            num_batches = total_num // batch_size
        else:
            num_batches = (total_num + batch_size - 1) // batch_size

        # ============================================================
        # CSR_data 游标
        # ============================================================
        start_data_id = 0

        # ============================================================
        # Prefetch buffer（单 slot）
        # ============================================================
        prefetch_buffer = {}
        prefetch_ready = threading.Event()

        def prefetch_csr_data(start_id, end_id):
            """后台线程：预读 CSR_data"""
            t0 = datetime.now()

            query = (
                f"SELECT indices, data FROM X_CSR_data "
                f"WHERE id >= {start_id} AND id < {end_id}"
            )
            table = self.connection.execute(query).fetch_arrow_table()

            indices_np = table.column("indices").to_numpy()
            data_np = table.column("data").to_numpy()

            prefetch_buffer["indices"] = indices_np
            prefetch_buffer["data"] = data_np
            prefetch_buffer["time"] = (datetime.now() - t0).total_seconds()

            prefetch_ready.set()

        # ============================================================
        # 主循环
        # ============================================================
        total_batches = 0

        for i in range(num_batches):

            batch_start_time = datetime.now()
            offset = i * batch_size

            if (not drop_last) and (i == num_batches - 1) and (total_num % batch_size != 0):
                current_batch_size = total_num % batch_size
            else:
                current_batch_size = batch_size

            # --------------------------------------------------------
            # 2.1 读取 indptr（主线程）
            # --------------------------------------------------------
            t0 = datetime.now()
            indptr_query = (
                f"SELECT indptr FROM X_CSR_indptr "
                f"LIMIT {current_batch_size} OFFSET {offset}"
            )
            indptr_table = self.connection.execute(indptr_query).fetch_arrow_table()
            indptr_raw = indptr_table.column("indptr").to_numpy()
            indptr_time = (datetime.now() - t0).total_seconds()

            end_data_id = indptr_raw[-1]
            indptr_np = np.concatenate([[0], indptr_raw - start_data_id])

            # --------------------------------------------------------
            # 2.2 启动下一批 CSR_data prefetch
            # --------------------------------------------------------
            if i < num_batches - 1:
                prefetch_ready.clear()
                t = threading.Thread(
                    target=prefetch_csr_data,
                    args=(start_data_id, end_data_id),
                )
                t.start()

            # --------------------------------------------------------
            # 2.3 当前 batch 等待 CSR_data（首批除外）
            # --------------------------------------------------------
            if i == 0:
                # 第 0 批：同步读取
                t1 = datetime.now()
                query = (
                    f"SELECT indices, data FROM X_CSR_data "
                    f"WHERE id >= {start_data_id} AND id < {end_data_id}"
                )
                table = self.connection.execute(query).fetch_arrow_table()
                indices_np = table.column("indices").to_numpy()
                data_np = table.column("data").to_numpy()
                data_time = (datetime.now() - t1).total_seconds()
            else:
                prefetch_ready.wait()
                indices_np = prefetch_buffer["indices"]
                data_np = prefetch_buffer["data"]
                data_time = prefetch_buffer["time"]

            start_data_id = end_data_id

            # --------------------------------------------------------
            # 2.4 obs + AnnData（主线程）
            # --------------------------------------------------------
            t2 = datetime.now()
            obs_df = self.query(
                f"SELECT * FROM obs LIMIT {current_batch_size} OFFSET {offset}"
            ).iloc[:, 1:]
            obs_df.index = obs_df["cell_id"].astype(str)
            obs_time = (datetime.now() - t2).total_seconds()

            t3 = datetime.now()
            X = sp.csr_matrix(
                (data_np, indices_np, indptr_np),
                shape=(len(obs_df), len(var_df)),
            )
            adata = AnnData(X=X, obs=obs_df, var=var_df)
            anndata_time = (datetime.now() - t3).total_seconds()

            batch_time = (datetime.now() - batch_start_time).total_seconds()

            print(
                f"[Batch {i}] total={batch_time:.4f}s | "
                f"indptr={indptr_time:.4f}s | "
                f"data={data_time:.4f}s | "
                f"obs={obs_time:.4f}s | "
                f"anndata={anndata_time:.4f}s"
            )

            total_batches += 1
            yield adata

        # ============================================================
        # Summary（真实 wall time）
        # ============================================================
        total_time = (datetime.now() - func_start_time).total_seconds()

        print("\n================ Summary (Prefetch) ================")
        print(f"Total batches        : {total_batches}")
        print(f"Total time (s)       : {total_time:.3f}")
        print(f"Time per batch (s)   : {total_time / max(total_batches, 1):.4f}")
        print(f"Batches per second   : {total_batches / total_time:.2f}")

    # # todo 对上面的优化 2  每次读取 4 × minibatch ， 再拆分，减少IO
    # def minibatch_scan_order_cursor_csr_df_arrow_mega4_obs(
    #         self,
    #         total_num,
    #         batch_size,
    #         drop_last,
    #         mega_factor=4,  # 👈 4× mega-batch
    # ):
    #     """
    #     【最终形态】
    #     CSR mega-batch + obs mega-batch + CSR 行切片
    #
    #     对外行为：
    #         - 每次 yield 1 个 minibatch (AnnData)
    #     内部策略：
    #         - 每 mega_factor 个 minibatch，只访问 1 次 obs / CSR 表
    #     """
    #
    #     import numpy as np
    #     import scipy.sparse as sp
    #     from anndata import AnnData
    #     from datetime import datetime
    #
    #     # ============================================================
    #     # 【Step 0】var 表只读取一次（全局不变）
    #     # ============================================================
    #     # todo:
    #     #   var 表规模远小于 obs / CSR
    #     #   没必要 Arrow-only，Pandas 即可
    #     var_df = self.query("SELECT * FROM var").iloc[:, 1:]
    #     gene_id = var_df["gene_id"].astype(str)
    #
    #     # ============================================================
    #     # 【Step 1】计算 batch / mega-batch 数量
    #     # ============================================================
    #     if drop_last:
    #         num_batches = total_num // batch_size
    #     else:
    #         num_batches = (total_num + batch_size - 1) // batch_size
    #
    #     mega_batch_size = batch_size * mega_factor
    #     num_mega_batches = (num_batches + mega_factor - 1) // mega_factor
    #
    #     # ============================================================
    #     # 【Step 2】CSR_data 全局游标
    #     # ============================================================
    #     # todo:
    #     #   CSR_data 是一条“连续扁平数组”
    #     #   start_data_id 必须全程单调递增
    #     start_data_id = 0
    #
    #     total_time = 0.0
    #     batch_counter = 0
    #
    #     # ============================================================
    #     # 【Step 3】mega-batch 主循环
    #     # ============================================================
    #     for m in range(num_mega_batches):
    #
    #         print(f"\n=== Mega-Batch {m} ===")
    #
    #         # --------------------------------------------------------
    #         # 3.1 当前 mega-batch 覆盖的 minibatch 范围
    #         # --------------------------------------------------------
    #         batch_start_idx = m * mega_factor
    #         batch_end_idx = min((m + 1) * mega_factor, num_batches)
    #         current_mega_batches = batch_end_idx - batch_start_idx
    #
    #         # --------------------------------------------------------
    #         # 3.2 当前 mega-batch 覆盖的 obs 行范围
    #         # --------------------------------------------------------
    #         offset = batch_start_idx * batch_size
    #         current_mega_cells = current_mega_batches * batch_size
    #
    #         # todo:
    #         #   最后一组 mega-batch 的 cell 数量修正
    #         if not drop_last and batch_end_idx == num_batches:
    #             current_mega_cells = min(
    #                 current_mega_cells,
    #                 total_num - offset
    #             )
    #
    #         # ========================================================
    #         # 【Step 4】obs mega-batch：只查 1 次
    #         # ========================================================
    #         t_obs = datetime.now()
    #
    #         obs_mega_df = self.query(
    #             f"SELECT * FROM obs "
    #             f"LIMIT {current_mega_cells} OFFSET {offset}"
    #         ).iloc[:, 1:]
    #
    #         # todo:
    #         #   obs 行顺序 == CSR_indptr 行顺序
    #         obs_cell_id = obs_mega_df["cell_id"].astype(str)
    #
    #         obs_time = (datetime.now() - t_obs).total_seconds()
    #
    #         # ========================================================
    #         # 【Step 5】CSR_indptr mega-batch
    #         # ========================================================
    #         t0 = datetime.now()
    #
    #         query_indptr = (
    #             f"SELECT indptr FROM X_CSR_indptr "
    #             f"LIMIT {current_mega_cells} OFFSET {offset}"
    #         )
    #
    #         indptr_table = (
    #             self.connection.execute(query_indptr)
    #             .fetch_arrow_table()
    #         )
    #
    #         # todo:
    #         #   Arrow → NumPy（零拷贝）
    #         indptr_raw = indptr_table.column("indptr").to_numpy()
    #
    #         indptr_time = (datetime.now() - t0).total_seconds()
    #
    #         # --------------------------------------------------------
    #         # todo:
    #         #   mega-batch CSR_data 的终点
    #         # --------------------------------------------------------
    #         end_data_id = indptr_raw[-1]
    #
    #         # --------------------------------------------------------
    #         # todo:
    #         #   CSR_indptr 处理规则（非常关键）：
    #         #   1) 在开头补 0
    #         #   2) 减去上一 mega-batch 的 start_data_id
    #         # --------------------------------------------------------
    #         indptr_mega = np.concatenate(
    #             [[0], indptr_raw - start_data_id]
    #         )
    #
    #         # ========================================================
    #         # 【Step 6】CSR_data mega-batch
    #         # ========================================================
    #         t1 = datetime.now()
    #
    #         if end_data_id > start_data_id:
    #             query_data = (
    #                 f"SELECT indices, data FROM X_CSR_data "
    #                 f"WHERE id >= {start_data_id} "
    #                 f"AND id < {end_data_id}"
    #             )
    #
    #             data_table = (
    #                 self.connection.execute(query_data)
    #                 .fetch_arrow_table()
    #             )
    #
    #             # todo:
    #             #   Arrow → NumPy（零拷贝）
    #             indices_mega = data_table.column("indices").to_numpy()
    #             data_mega = data_table.column("data").to_numpy()
    #         else:
    #             indices_mega = np.array([], dtype=np.int32)
    #             data_mega = np.array([], dtype=np.float32)
    #
    #         data_time = (datetime.now() - t1).total_seconds()
    #
    #         # todo:
    #         #   更新 CSR_data 游标
    #         start_data_id = end_data_id
    #
    #         # ========================================================
    #         # 【Step 7】构造 mega CSR（1 次）
    #         # ========================================================
    #         t2 = datetime.now()
    #
    #         X_mega = sp.csr_matrix(
    #             (data_mega, indices_mega, indptr_mega),
    #             shape=(current_mega_cells, len(var_df)),
    #         )
    #
    #         csr_time = (datetime.now() - t2).total_seconds()
    #
    #         print(
    #             f"[Mega {m}] obs={obs_time:.4f}s | "
    #             f"indptr={indptr_time:.4f}s | "
    #             f"data={data_time:.4f}s | "
    #             f"csr={csr_time:.4f}s"
    #         )
    #
    #         # ========================================================
    #         # 【Step 8】mega → minibatch 切片 & yield
    #         # ========================================================
    #         for b in range(current_mega_batches):
    #             t_batch = datetime.now()  # todo 开始时间不对
    #
    #             row_start = b * batch_size
    #             row_end = min(
    #                 row_start + batch_size,
    #                 current_mega_cells
    #             )
    #
    #             # ----------------------------------------------------
    #             # todo:
    #             #   obs / X 只做视图切片（O(1)）
    #             # ----------------------------------------------------
    #             obs_df = obs_mega_df.iloc[row_start:row_end]
    #             X_batch = X_mega[row_start:row_end]
    #
    #             obs_df.index = obs_cell_id.iloc[row_start:row_end].values
    #             var_df.index = gene_id.values
    #
    #             adata = AnnData(
    #                 X=X_batch,
    #                 obs=obs_df,
    #                 var=var_df,
    #             )
    #
    #             # adata.obs_names = obs_cell_id.iloc[row_start:row_end]
    #             # adata.var_names = gene_id
    #
    #             batch_time = (datetime.now() - t_batch).total_seconds()
    #             total_time += batch_time
    #             batch_counter += 1
    #
    #             print(
    #                 f"  -> Batch {batch_counter - 1} | "
    #                 f"time={batch_time:.4f}s"
    #             )
    #
    #             yield adata
    #
    #     # ============================================================
    #     # 【Step 9】最终统计
    #     # ============================================================
    #     print("\n================ Summary ================")
    #     print(f"Total batches        : {batch_counter}")
    #     print(f"Total time (s)       : {total_time:.3f}")
    #     print(f"Time per batch (s)   : {total_time / batch_counter:.4f}")
    #     print(f"Batches per second   : {batch_counter / total_time:.2f}")
    #
    # # todo 对上面的优化 3  一次 mega-batch → 构造 1 个 AnnData → 切 4 个 view → yield
    # def minibatch_scan_order_cursor_csr_df_arrow_mega_obs_view(
    #         self,
    #         total_num: int,
    #         batch_size: int,
    #         drop_last: bool,
    #         mega_factor: int = 4,  # 👈 一个 mega-batch = 4 个 minibatch
    # ):
    #     """
    #     【结构级最终优化版】
    #     - 每个 mega-batch 只构造 1 个 AnnData
    #     - minibatch 通过 AnnData view 切片得到
    #     """
    #
    #     import numpy as np
    #     import scipy.sparse as sp
    #     from anndata import AnnData
    #     from datetime import datetime
    #
    #     # ------------------------------
    #     # 0️⃣ 基础参数计算
    #     # ------------------------------
    #     mega_batch_size = batch_size * mega_factor
    #
    #     if drop_last:
    #         num_mega = total_num // mega_batch_size
    #     else:
    #         num_mega = (total_num + mega_batch_size - 1) // mega_batch_size
    #
    #     start_data_id = 0  # 👈 CSR_data 表游标（非常关键）
    #
    #     # ------------------------------
    #     # 1️⃣ mega-batch 主循环
    #     # ------------------------------
    #     for mega_idx in range(num_mega):
    #
    #         mega_offset = mega_idx * mega_batch_size
    #
    #         # todo: 处理最后一个 mega-batch 的大小
    #         if not drop_last and mega_idx == num_mega - 1:
    #             current_mega_batch = min(
    #                 mega_batch_size,
    #                 total_num - mega_offset
    #             )
    #         else:
    #             current_mega_batch = mega_batch_size
    #
    #         t0 = datetime.now()
    #
    #         # ==========================================================
    #         # 2️⃣ 读取 obs（一次性 mega-batch）
    #         # ==========================================================
    #         # ⚠️ obs 继续使用你验证过的 self.query（稳定 & 无 Arrow 坑）
    #         obs_df = self.query(
    #             f"SELECT * FROM obs LIMIT {current_mega_batch} OFFSET {mega_offset}"
    #         )
    #         obs_df = obs_df.iloc[:, 1:]  # 去掉 id 列
    #         cell_ids = obs_df["cell_id"].astype(str).values
    #
    #         # ==========================================================
    #         # 3️⃣ 读取 CSR_indptr（mega-batch）
    #         # ==========================================================
    #         # todo: CSR_indptr 记录的是“到目前 cell 为止的 data 累积量”
    #         query_indptr = (
    #             f"SELECT indptr FROM X_CSR_indptr "
    #             f"LIMIT {current_mega_batch} OFFSET {mega_offset}"
    #         )
    #
    #         indptr_df = (
    #             self.connection.execute(query_indptr)
    #             .fetch_arrow_table()
    #             .to_pandas()
    #         )
    #
    #         indptr_raw = indptr_df["indptr"].to_numpy()
    #
    #         # todo 获取 CSR_data 表的读取终点
    #         end_data_id = int(indptr_raw[-1])
    #
    #         # todo 获取 CSR_indptr 值：
    #         # - 在开头补 0
    #         # - 减去上一 mega-batch 的末尾值
    #         csr_indptr = np.concatenate(
    #             [[0], indptr_raw - start_data_id]
    #         )
    #
    #         # ==========================================================
    #         # 4️⃣ 读取 CSR_data（只读本 mega 需要的）
    #         # ==========================================================
    #         if end_data_id > start_data_id:
    #             query_data = (
    #                 f"SELECT indices, data FROM X_CSR_data "
    #                 f"WHERE id >= {start_data_id} AND id < {end_data_id} "
    #                 f"ORDER BY id"
    #             )
    #
    #             csr_data_df = (
    #                 self.connection.execute(query_data)
    #                 .fetch_arrow_table()
    #                 .to_pandas()
    #             )
    #
    #             csr_indices = csr_data_df["indices"].to_numpy()
    #             csr_data = csr_data_df["data"].to_numpy()
    #         else:
    #             # todo: 极端空 batch 兜底
    #             csr_indices = np.array([], dtype=np.int32)
    #             csr_data = np.array([], dtype=np.float32)
    #
    #         start_data_id = end_data_id  # 👈 更新游标
    #
    #         # ==========================================================
    #         # 5️⃣ 构造 mega CSR 矩阵
    #         # ==========================================================
    #         var_df = self.query("SELECT * FROM var").iloc[:, 1:]
    #         gene_ids = var_df["gene_id"].astype(str).values
    #
    #         mega_X = sp.csr_matrix(
    #             (csr_data, csr_indices, csr_indptr),
    #             shape=(current_mega_batch, len(var_df))
    #         )
    #
    #         # ==========================================================
    #         # 6️⃣ ⭐ 只构造 1 次 AnnData（核心）
    #         # ==========================================================
    #         mega_adata = AnnData(
    #             X=mega_X,
    #             obs=obs_df,
    #             var=var_df,
    #         )
    #
    #         mega_adata.obs_names = cell_ids
    #         mega_adata.var_names = gene_ids
    #
    #         t1 = datetime.now()
    #
    #         print(
    #             f"[Mega {mega_idx}] "
    #             f"obs+csr+AnnData = {(t1 - t0).total_seconds():.4f}s"
    #         )
    #
    #         # ==========================================================
    #         # 7️⃣ 切 minibatch view（零拷贝）
    #         # ==========================================================
    #         num_sub = (current_mega_batch + batch_size - 1) // batch_size
    #
    #         for sub_idx in range(num_sub):
    #             s = sub_idx * batch_size
    #             e = min((sub_idx + 1) * batch_size, current_mega_batch)
    #
    #             # ⚠️ 这是 view，不复制任何数据
    #             yield mega_adata[s:e]
    #
    #
    # # todo 完全绕开 AnnData
    # #         - minibatch 只返回 CSR + ids;
    # #         调用的时候再
    # #         def csr_batch_to_anndata(batch):
    # #     from anndata import AnnData
    # #     return AnnData(
    # #         X=batch["X"],
    # #         obs={"cell_id": batch["cell_ids"]},
    # #         var={"gene_id": batch["gene_ids"]},
    # #     )
    # def iter_csr_minibatch_fast(
    #         self,
    #         total_num: int,
    #         batch_size: int,
    #         drop_last: bool = False,
    # ):
    #     """
    #     【终极高吞吐版本】
    #     - 完全绕开 AnnData
    #     - minibatch 只返回 CSR + ids
    #     """
    #
    #     import numpy as np
    #     import scipy.sparse as sp
    #
    #     # ------------------------------------------------
    #     # 0️⃣ 基础参数
    #     # ------------------------------------------------
    #     if drop_last:
    #         num_batches = total_num // batch_size
    #     else:
    #         num_batches = (total_num + batch_size - 1) // batch_size
    #
    #     start_data_id = 0  # CSR_data 游标
    #
    #     # ------------------------------------------------
    #     # 1️⃣ 预取 gene_ids（一次即可）
    #     # ------------------------------------------------
    #     var_df = self.query("SELECT gene_id FROM var")
    #     gene_ids = var_df["gene_id"].astype(str).values
    #     n_vars = len(gene_ids)
    #
    #     # ------------------------------------------------
    #     # 2️⃣ minibatch 主循环
    #     # ------------------------------------------------
    #     for batch_idx in range(num_batches):
    #
    #         offset = batch_idx * batch_size
    #
    #         # todo: 最后一批大小修正
    #         if not drop_last and batch_idx == num_batches - 1:
    #             cur_bs = min(batch_size, total_num - offset)
    #         else:
    #             cur_bs = batch_size
    #
    #         # =============================================
    #         # 2.1 读取 obs（只取 cell_id）
    #         # =============================================
    #         obs_df = self.query(
    #             f"SELECT cell_id FROM obs LIMIT {cur_bs} OFFSET {offset}"
    #         )
    #         cell_ids = obs_df["cell_id"].astype(str).values
    #
    #         # =============================================
    #         # 2.2 读取 CSR_indptr
    #         # =============================================
    #         indptr_df = (
    #             self.connection.execute(
    #                 f"SELECT indptr FROM X_CSR_indptr "
    #                 f"LIMIT {cur_bs} OFFSET {offset}"
    #             )
    #             .fetch_arrow_table()
    #             .to_pandas()
    #         )
    #
    #         indptr_raw = indptr_df["indptr"].to_numpy()
    #         end_data_id = int(indptr_raw[-1])
    #
    #         # todo: CSR_indptr 归一化（减去上一批末尾）
    #         csr_indptr = np.concatenate(
    #             [[0], indptr_raw - start_data_id]
    #         )
    #
    #         # =============================================
    #         # 2.3 读取 CSR_data
    #         # =============================================
    #         if end_data_id > start_data_id:
    #             csr_data_df = (
    #                 self.connection.execute(
    #                     f"SELECT indices, data FROM X_CSR_data "
    #                     f"WHERE id >= {start_data_id} AND id < {end_data_id} "
    #                     f"ORDER BY id"
    #                 )
    #                 .fetch_arrow_table()
    #                 .to_pandas()
    #             )
    #
    #             csr_indices = csr_data_df["indices"].to_numpy()
    #             csr_data = csr_data_df["data"].to_numpy()
    #         else:
    #             csr_indices = np.array([], dtype=np.int32)
    #             csr_data = np.array([], dtype=np.float32)
    #
    #         start_data_id = end_data_id
    #
    #         # =============================================
    #         # 2.4 构造 CSR（极快）
    #         # =============================================
    #         X = sp.csr_matrix(
    #             (csr_data, csr_indices, csr_indptr),
    #             shape=(cur_bs, n_vars),
    #         )
    #
    #         # =============================================
    #         # 2.5 yield 纯数据 batch
    #         # =============================================
    #         yield {
    #             "X": X,
    #             "cell_ids": cell_ids,
    #             "gene_ids": gene_ids,
    #         }

    # # todo CSR格式 游标优化，df优化方法3 只选择需要的列  , 不使用df格式，直接提取列为np格式： 负优化？？
    # def minibatch_scan_order_cursor_csr_df_onlylie(self, total_num, batch_size, drop_last):
    #     """按顺序读取模式"""
    #     # 计算批次数量
    #     time_2 = 0  # obs 表读取用时
    #     time_3 = 0  # var 表读取用时
    #     time_4 = 0  # 生成anndata数据用时
    #     time_5 = 0  # X_CSR_indptr 表 读取时间
    #     time_6 = 0  # X_CSR_data 表 读取时间
    #     time_7 = 0  # CSR 生成anndata数据用时
    #     time_8 = 0  # X_CSR_data 转 DF 时间
    #
    #     data_read_start = 0  # X_CSR_data表 每次的读取的 起始点
    #     data_read_count = 0  # X_CSR_data表 每次的读取的 数据量
    #
    #     # 新增：用于记录上次读取的最后一个id，用于游标分页
    #     last_data_id = 0  # 初始化为0，假设id从1开始，或根据实际情况调整
    #
    #     # 新增：用于记录每批次时间的列表
    #     batch_times = {
    #         'batch_num': [],
    #         'time_X_CSR_indptr': [],
    #         'time_X_CSR_data': [],
    #         'time_X_CSR_data_df': [],
    #         'time_obs': [],
    #         'time_var': [],
    #         'time_anndata': [],
    #         'total_time': []
    #     }
    #
    #     if drop_last:
    #         num_batches = total_num // batch_size
    #     else:
    #         num_batches = (total_num + batch_size - 1) // batch_size
    #
    #     for i in range(num_batches):
    #         offset = i * batch_size
    #
    #         # 如果是最后一批且不丢弃剩余数据，调整limit
    #         if not drop_last and i == num_batches - 1 and total_num % batch_size != 0:
    #             current_batch_size = total_num % batch_size
    #         else:
    #             current_batch_size = batch_size
    #
    #         print(f"\n--- 批次 {i} 详细时间分析 ---")
    #         print(f"\n--- 批次大小 {batch_size} ---")
    #
    #         query_CSR_indptr = f"SELECT * FROM X_CSR_indptr LIMIT {current_batch_size} OFFSET {offset}"  # todo 获取CSR_indptr
    #
    #         CSR_indptr_result_start = datetime.now()
    #         result = self.connection.execute(query_CSR_indptr)  # todo CSR_indptr 查询时间
    #         CSR_indptr_result_end = datetime.now()
    #         CSR_indptr_result_time = (CSR_indptr_result_end - CSR_indptr_result_start).total_seconds()
    #         print(f"#### CSR_indptr 查询时间 : {CSR_indptr_result_time}")
    #
    #         CSR_indptr_df_start = datetime.now()
    #         CSR_indptr_df = result.df()  # todo  CSR_indptr 转化为 df格式时间
    #         CSR_indptr_df_end = datetime.now()
    #         CSR_indptr_df_time = (CSR_indptr_df_end - CSR_indptr_df_start).total_seconds()
    #         print(f"#### CSR_indptr 转化为 df格式时间 : {CSR_indptr_df_time}")
    #
    #         # todo  可以读取出 df 再提取 需要的 array， 也可以直接提取需要的 array ；
    #         # 一次取2个,同X表的读取量
    #         # todo: 第 0 批
    #         # CSR_indptr_df =
    #         # {  id:  0, 1 ;
    #         #    cell_id :  cell_0,cell_1 ;
    #         #    indptr : 2, 3
    #         # }
    #
    #         CSR_indptr_np_start = datetime.now()
    #         CSR_indptr_array = np.insert(CSR_indptr_df['indptr'].values - data_read_start, 0, 0)  # todo  df 提取 np 时间
    #         CSR_indptr_np_end = datetime.now()
    #         CSR_indptr_np_time = (CSR_indptr_np_end - CSR_indptr_np_start).total_seconds()
    #         print(f"#### CSR_indptr 的 df 提取 np 时间 : {CSR_indptr_np_time}")
    #
    #         time_5 = time_5 + CSR_indptr_result_time + CSR_indptr_df_time + CSR_indptr_np_time  # X_CSR_indptr 表 读取时间
    #
    #         # 只保留indptr，并在最前面补上0值，并减去 data_read_start = 0
    #         # CSR_indptr_array = 0 +  ( CSR_indptr_df['indptr'] - data_read_count )
    #         # todo 得到 CSR_indptr_array = [ 0,2,3]
    #
    #         # 起点： 初始 data_read_start = 0
    #         data_read_count = CSR_indptr_array[-1]  # 获取值数量： 获取 3 个值， 即data_read_count = CSR_indptr_array[-1]=3,最后一个值
    #         if (data_read_count > 0):
    #
    #
    #             end_id = last_data_id + data_read_count
    #             print(f"data_read_count (CSR_indptr_array[-1]) 是 {data_read_count}" )
    #             print(f"last_data_id 是 {last_data_id}" )
    #             print(f"end_id 是 {end_id}" )
    #             # todo 只提取需要的内容 indices, data
    #             query_CSR_data_indices = f"SELECT indices FROM X_CSR_data WHERE id >= {last_data_id} AND id < {end_id} ORDER BY id"
    #             query_CSR_data_data = f"SELECT data FROM X_CSR_data WHERE id >= {last_data_id} AND id < {end_id} ORDER BY id"
    #
    #             CSR_data_result_start = datetime.now()
    #
    #             rusult_indices = self.connection.execute(query_CSR_data_indices).fetchnumpy()  # todo 获取 CSR_data
    #             rusult_data = self.connection.execute(query_CSR_data_data).fetchnumpy()  # todo 获取 CSR_data
    #
    #             CSR_data_result_end = datetime.now()
    #             CSR_data_result_time = (CSR_data_result_end - CSR_data_result_start).total_seconds()
    #             print(f"#### CSR_data 查询时间 : {CSR_data_result_time}")
    #
    #             CSR_data_df_start = datetime.now()
    #             # CSR_data_df = rusult.df()  # todo  CSR_data 转化为 df
    #             # # todo 方法1：使用fetch_arrow_table
    #             # table = result.fetch_arrow_table()
    #             # CSR_data_df = table.to_pandas()
    #             # r1 = rusult_indices.fetchnumpy()
    #             # r2 = rusult_data.fetchnumpy()
    #
    #             CSR_indices_array = rusult_indices['indices']
    #             CSR_data_array = rusult_data['data']
    #
    #
    #             CSR_data_df_end = datetime.now()
    #             CSR_data_df_time = (CSR_data_df_end - CSR_data_df_start).total_seconds()
    #             print(f"#### CSR_data 转化为 df  时间 : {CSR_data_df_time}")
    #
    #             # CSR_data_df =
    #             # {  id:  0, 1 ,2 ;
    #             #    indices :  0, 2 , 2 ;
    #             #    data : 8, 2, 5 ;
    #             # }
    #             # print(f"len(CSR_data_df)  {len(CSR_data_df)}")
    #
    #             # CSR_indices_array = CSR_data_df['indices'].to_numpy()  # todo:  CSR_indices_array = [ 0,2,2]
    #             # CSR_data_array = CSR_data_df['data'].to_numpy()  # todo:  CSR_data_array = [ 8,2,5]
    #
    #             # ========== 修改点：更新游标位置 ==========
    #             # 更新last_data_id为本次读取的最后一个id
    #             # if not CSR_data_df.empty:
    #             last_data_id = end_id - 1  # 获取本次读取的最后一个id
    #             # ========== 修改结束 ==========
    #
    #             data_read_start = data_read_start + data_read_count  # 更新起点
    #
    #             time_6 = time_6 + CSR_data_result_time  # X_CSR_data 表 读取时间
    #             time_8 += CSR_data_df_time
    #
    #         else:
    #             print("获取 0 个值，不查询 ")  # todo 空值的处理
    #
    #         start_time2 = datetime.now()  # obs表读取用时
    #
    #         sub_obs = self.query(f"SELECT * FROM obs LIMIT {current_batch_size} OFFSET {offset}")
    #
    #         end_time2 = datetime.now()  # obs 表读取用时
    #         time_diff2 = end_time2 - start_time2  # obs 表读取用时
    #         time_2 = time_2 + time_diff2.total_seconds()  # obs 表读取用时
    #
    #         start_time3 = datetime.now()  # var 表读取用时
    #
    #         sub_var = self.query("SELECT * FROM var")
    #
    #         end_time3 = datetime.now()  # var 表读取用时
    #         time_diff3 = end_time3 - start_time3  # var 表读取用时
    #         time_3 = time_3 + time_diff3.total_seconds()  # var 表读取用时
    #
    #         start_time4 = datetime.now()  # 生成anndata数据用时
    #
    #         sub_obs = sub_obs.iloc[:, 1:]  # 去掉数据表中的前1列 id
    #         sub_var = sub_var.iloc[:, 1:]  # 去掉数据表中的前1列
    #         cell_id = sub_obs['cell_id']  # 获取cell_id
    #         gene_id = sub_var['gene_id']  # 获取gene_id
    #
    #         # print(f"indptr_array[-1]: {CSR_indptr_array[-1]}")
    #         # print(f"indices长度: {len(CSR_indices_array)}")
    #         # print(f"data长度: {len(CSR_data_array)}")
    #
    #         # 创建CSR格式的稀疏矩阵
    #         X = sp.csr_matrix((CSR_data_array, CSR_indices_array, CSR_indptr_array),
    #                           shape=(len(sub_obs), len(sub_var)))
    #         # 创建AnnData对象
    #         adata = AnnData(X=X, obs=sub_obs, var=sub_var)
    #
    #         adata.obs_names = cell_id.astype(str)  # 设置观测名为cell_id
    #         adata.obs.index = cell_id.astype(str)
    #         adata.var_names = gene_id.astype(str)  # 设置变量名为gene_id
    #         adata.var.index = gene_id.astype(str)
    #
    #         end_time4 = datetime.now()  # 生成anndata数据用时
    #         time_diff4 = end_time4 - start_time4  # 生成anndata数据用时
    #         time_4 = time_4 + time_diff4.total_seconds()  # 生成anndata数据用时
    #
    #         # 新增：记录当前批次的时间
    #         batch_total_time = time_diff2.total_seconds() + time_diff3.total_seconds() + time_diff4.total_seconds()
    #
    #         batch_times['time_X_CSR_indptr'].append(CSR_indptr_result_time + CSR_indptr_df_time)
    #         batch_times['time_X_CSR_data'].append(CSR_data_result_time)
    #         batch_times['time_X_CSR_data_df'].append(CSR_data_df_time)
    #         batch_times['batch_num'].append(i + 1)
    #         batch_times['time_obs'].append(time_diff2.total_seconds())
    #         batch_times['time_var'].append(time_diff3.total_seconds())
    #         batch_times['time_anndata'].append(time_diff4.total_seconds())
    #         batch_times['total_time'].append(batch_total_time)
    #
    #         # 返回数据
    #         yield adata
    #
    #         print(f"obs 表读取用时： {time_2:.2f} 秒")
    #         print(f"var 表读取用时： {time_3:.2f} 秒")
    #         print(f"生成anndata数据用时： {time_4:.2f} 秒")
    #         print(f"X_CSR_indptr 表 读取时间： {time_5:.2f} 秒")
    #         print(f"X_CSR_data 表 读取时间： {time_6:.2f} 秒")
    #         print(f"X_CSR_data 表 转df时间 ： {time_8:.2f} 秒")
    #
    #     # 新增：在所有批次处理完成后，绘制时间图表
    #     self._plot_batch_times(batch_times, batch_size)
    #
    # # todo CSR格式 游标优化，df优化方法4  arrow table + 只选择需要的列 + 并行  ; 负优化？？
    # def minibatch_scan_order_cursor_csr_df_arrow_onlylie_bingxing(self, total_num, batch_size, drop_last):
    #     """按顺序读取模式"""
    #     # 计算批次数量
    #     time_2 = 0  # obs 表读取用时
    #     time_3 = 0  # var 表读取用时
    #     time_4 = 0  # 生成anndata数据用时
    #     time_5 = 0  # X_CSR_indptr 表 读取时间
    #     time_6 = 0  # X_CSR_data 表 读取时间
    #     time_7 = 0  # CSR 生成anndata数据用时
    #     time_8 = 0  # X_CSR_data 转 DF 时间
    #
    #     data_read_start = 0  # X_CSR_data表 每次的读取的 起始点
    #     data_read_count = 0  # X_CSR_data表 每次的读取的 数据量
    #
    #     # 新增：用于记录上次读取的最后一个id，用于游标分页
    #     last_data_id = 0  # 初始化为0，假设id从1开始，或根据实际情况调整
    #
    #     # 新增：用于记录每批次时间的列表
    #     batch_times = {
    #         'batch_num': [],
    #         'time_X_CSR_indptr': [],
    #         'time_X_CSR_data': [],
    #         'time_X_CSR_data_df': [],
    #         'time_obs': [],
    #         'time_var': [],
    #         'time_anndata': [],
    #         'total_time': []
    #     }
    #
    #     if drop_last:
    #         num_batches = total_num // batch_size
    #     else:
    #         num_batches = (total_num + batch_size - 1) // batch_size
    #
    #     for i in range(num_batches):
    #         offset = i * batch_size
    #
    #         # 如果是最后一批且不丢弃剩余数据，调整limit
    #         if not drop_last and i == num_batches - 1 and total_num % batch_size != 0:
    #             current_batch_size = total_num % batch_size
    #         else:
    #             current_batch_size = batch_size
    #
    #         print(f"\n--- 批次 {i} 详细时间分析 ---")
    #         print(f"\n--- 批次大小 {batch_size} ---")
    #
    #         query_CSR_indptr = f"SELECT * FROM X_CSR_indptr LIMIT {current_batch_size} OFFSET {offset}"  # todo 获取CSR_indptr
    #
    #         CSR_indptr_result_start = datetime.now()
    #         result = self.connection.execute(query_CSR_indptr)  # todo CSR_indptr 查询时间
    #         CSR_indptr_result_end = datetime.now()
    #         CSR_indptr_result_time = (CSR_indptr_result_end - CSR_indptr_result_start).total_seconds()
    #         print(f"#### CSR_indptr 查询时间 : {CSR_indptr_result_time}")
    #
    #         CSR_indptr_df_start = datetime.now()
    #         CSR_indptr_df = result.df()  # todo  CSR_indptr 转化为 df格式时间
    #         CSR_indptr_df_end = datetime.now()
    #         CSR_indptr_df_time = (CSR_indptr_df_end - CSR_indptr_df_start).total_seconds()
    #         print(f"#### CSR_indptr 转化为 df格式时间 : {CSR_indptr_df_time}")
    #
    #         # todo  可以读取出 df 再提取 需要的 array， 也可以直接提取需要的 array ；
    #         # 一次取2个,同X表的读取量
    #         # todo: 第 0 批
    #         # CSR_indptr_df =
    #         # {  id:  0, 1 ;
    #         #    cell_id :  cell_0,cell_1 ;
    #         #    indptr : 2, 3
    #         # }
    #
    #         CSR_indptr_np_start = datetime.now()
    #         CSR_indptr_array = np.insert(CSR_indptr_df['indptr'].values - data_read_start, 0, 0)  # todo  df 提取 np 时间
    #         CSR_indptr_np_end = datetime.now()
    #         CSR_indptr_np_time = (CSR_indptr_np_end - CSR_indptr_np_start).total_seconds()
    #         print(f"#### CSR_indptr 的 df 提取 np 时间 : {CSR_indptr_np_time}")
    #
    #         time_5 = time_5 + CSR_indptr_result_time + CSR_indptr_df_time + CSR_indptr_np_time  # X_CSR_indptr 表 读取时间
    #
    #         # 只保留indptr，并在最前面补上0值，并减去 data_read_start = 0
    #         # CSR_indptr_array = 0 +  ( CSR_indptr_df['indptr'] - data_read_count )
    #         # todo 得到 CSR_indptr_array = [ 0,2,3]
    #
    #         # 起点： 初始 data_read_start = 0
    #         data_read_count = CSR_indptr_array[-1]  # 获取值数量： 获取 3 个值， 即data_read_count = CSR_indptr_array[-1]=3,最后一个值
    #         if (data_read_count > 0):
    #
    #             end_id = last_data_id + data_read_count
    #
    #             # todo 设置并行度
    #             self.connection.execute("PRAGMA threads=4")
    #             query_CSR_data = f"SELECT /*+ PARALLEL(4) */ indices, data FROM X_CSR_data WHERE id >= {last_data_id} AND id < {end_id} ORDER BY id"
    #
    #
    #             CSR_data_result_start = datetime.now()
    #             rusult = self.connection.execute(query_CSR_data)  # todo 获取 CSR_data
    #             CSR_data_result_end = datetime.now()
    #             CSR_data_result_time = (CSR_data_result_end - CSR_data_result_start).total_seconds()
    #             print(f"#### CSR_data 查询时间 : {CSR_data_result_time}")
    #
    #             CSR_data_df_start = datetime.now()
    #             # CSR_data_df = rusult.df()  # todo  CSR_data 转化为 df
    #
    #             # todo 方法1：使用fetch_arrow_table
    #             table = result.fetch_arrow_table()
    #             CSR_data_df = table.to_pandas()
    #
    #             CSR_data_df_end = datetime.now()
    #             CSR_data_df_time = (CSR_data_df_end - CSR_data_df_start).total_seconds()
    #             print(f"#### CSR_data 转化为 df  时间 : {CSR_data_df_time}")
    #
    #             # CSR_data_df =
    #             # {  id:  0, 1 ,2 ;
    #             #    indices :  0, 2 , 2 ;
    #             #    data : 8, 2, 5 ;
    #             # }
    #             print(f"len(CSR_data_df)  {len(CSR_data_df)}")
    #
    #             CSR_indices_array = CSR_data_df['indices'].to_numpy()  # todo:  CSR_indices_array = [ 0,2,2]
    #             CSR_data_array = CSR_data_df['data'].to_numpy()  # todo:  CSR_data_array = [ 8,2,5]
    #
    #             # ========== 修改点：更新游标位置 ==========
    #             # 更新last_data_id为本次读取的最后一个id
    #             if not CSR_data_df.empty:
    #                 last_data_id = end_id - 1  # 获取本次读取的最后一个id
    #             # ========== 修改结束 ==========
    #
    #             data_read_start = data_read_start + data_read_count  # 更新起点
    #
    #             time_6 = time_6 + CSR_data_result_time  # X_CSR_data 表 读取时间
    #             time_8 += CSR_data_df_time
    #
    #         else:
    #             print("获取 0 个值，不查询 ")  # todo 空值的处理
    #
    #         start_time2 = datetime.now()  # obs表读取用时
    #
    #         sub_obs = self.query(f"SELECT * FROM obs LIMIT {current_batch_size} OFFSET {offset}")
    #
    #         end_time2 = datetime.now()  # obs 表读取用时
    #         time_diff2 = end_time2 - start_time2  # obs 表读取用时
    #         time_2 = time_2 + time_diff2.total_seconds()  # obs 表读取用时
    #
    #         start_time3 = datetime.now()  # var 表读取用时
    #
    #         sub_var = self.query("SELECT * FROM var")
    #
    #         end_time3 = datetime.now()  # var 表读取用时
    #         time_diff3 = end_time3 - start_time3  # var 表读取用时
    #         time_3 = time_3 + time_diff3.total_seconds()  # var 表读取用时
    #
    #         start_time4 = datetime.now()  # 生成anndata数据用时
    #
    #         sub_obs = sub_obs.iloc[:, 1:]  # 去掉数据表中的前1列 id
    #         sub_var = sub_var.iloc[:, 1:]  # 去掉数据表中的前1列
    #         cell_id = sub_obs['cell_id']  # 获取cell_id
    #         gene_id = sub_var['gene_id']  # 获取gene_id
    #
    #         print(f"indptr_array[-1]: {CSR_indptr_array[-1]}")
    #         print(f"indices长度: {len(CSR_indices_array)}")
    #         print(f"data长度: {len(CSR_data_array)}")
    #
    #         # 创建CSR格式的稀疏矩阵
    #         X = sp.csr_matrix((CSR_data_array, CSR_indices_array, CSR_indptr_array),
    #                           shape=(len(sub_obs), len(sub_var)))
    #         # 创建AnnData对象
    #         adata = AnnData(X=X, obs=sub_obs, var=sub_var)
    #
    #         adata.obs_names = cell_id.astype(str)  # 设置观测名为cell_id
    #         adata.obs.index = cell_id.astype(str)
    #         adata.var_names = gene_id.astype(str)  # 设置变量名为gene_id
    #         adata.var.index = gene_id.astype(str)
    #
    #         end_time4 = datetime.now()  # 生成anndata数据用时
    #         time_diff4 = end_time4 - start_time4  # 生成anndata数据用时
    #         time_4 = time_4 + time_diff4.total_seconds()  # 生成anndata数据用时
    #
    #         # 新增：记录当前批次的时间
    #         batch_total_time = time_diff2.total_seconds() + time_diff3.total_seconds() + time_diff4.total_seconds()
    #
    #         batch_times['time_X_CSR_indptr'].append(CSR_indptr_result_time + CSR_indptr_df_time)
    #         batch_times['time_X_CSR_data'].append(CSR_data_result_time)
    #         batch_times['time_X_CSR_data_df'].append(CSR_data_df_time)
    #         batch_times['batch_num'].append(i + 1)
    #         batch_times['time_obs'].append(time_diff2.total_seconds())
    #         batch_times['time_var'].append(time_diff3.total_seconds())
    #         batch_times['time_anndata'].append(time_diff4.total_seconds())
    #         batch_times['total_time'].append(batch_total_time)
    #
    #         # 返回数据
    #         yield adata
    #
    #         print(f"obs 表读取用时： {time_2:.2f} 秒")
    #         print(f"var 表读取用时： {time_3:.2f} 秒")
    #         print(f"生成anndata数据用时： {time_4:.2f} 秒")
    #         print(f"X_CSR_indptr 表 读取时间： {time_5:.2f} 秒")
    #         print(f"X_CSR_data 表 读取时间： {time_6:.2f} 秒")
    #         print(f"X_CSR_data 表 转df时间 ： {time_8:.2f} 秒")
    #
    #     # 新增：在所有批次处理完成后，绘制时间图表
    #     self._plot_batch_times(batch_times, batch_size)

    # todo 优化4：综合数据库优化方案
    def comprehensive_database_optimization(self):
        """综合数据库优化方案 - 用户手动调用"""
        print("开始执行数据库综合优化...")

        # 1. 数据库配置优化
        self.optimize_database_settings()

        # 2. 创建核心索引
        self.create_all_indexes()

        # 3. 表维护
        self.maintain_tables()

        # 4. 查询计划分析
        self.analyze_query_plans()

        # 5. 性能监控
        self.monitor_performance()

        self._optimized = True
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

    def maintain_tables(self):
        """表维护操作"""
        try:
            # 更新统计信息
            self.connection.execute("ANALYZE X") # 收集表 X 的统计信息
            self.connection.execute("ANALYZE obs") # 收集表 obs 的统计信息
            self.connection.execute("ANALYZE var") # 收集表 var 的统计信息
            print("统计信息更新完成")
            # ANALYZE 收集的统计信息类型
            # 基数估算：表中行数的统计
            # 列数据分布：最小值、最大值、唯一值数量
            # 数据相关性：列之间的关联信息
            # 数据倾斜：数据分布的均匀程度

        except Exception as e:
            print(f"表维护时出错: {e}")

    def analyze_query_plans(self):
        """分析关键查询的执行计划"""
        test_queries = [
            "EXPLAIN SELECT * FROM X WHERE id > 1000 ORDER BY id LIMIT 1000",
            "EXPLAIN SELECT * FROM obs WHERE cell_id IN ('1', '2', '3')"
        ]

        for query in test_queries:
            try:
                result = self.connection.execute(query).fetchall()
                print(f"查询计划分析: {query}")
                for row in result:
                    print(f"  {row[0]}")
            except Exception as e:
                print(f"分析查询计划失败: {e}")

    def monitor_performance(self):
        """监控数据库性能"""
        try:
            # 查看内存使用
            memory = self.connection.execute("PRAGMA memory_usage").fetchone()
            print(f"当前内存使用: {memory[0]}")

            # 查看表大小
            tables = ['X', 'obs', 'var']
            for table in tables:
                size = self.connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
                print(f"{table}表行数: {size[0]:,}")

        except Exception as e:
            print(f"性能监控时出错: {e}")

    def _ensure_indexes(self):
        """确保必要的索引存在 - 在minibatch_scan_order中自动调用"""
        if not self._optimized:
            print("检测到未进行数据库优化，正在创建基础索引...")
            try:
                # 只创建最关键的索引
                critical_indexes = [
                    "CREATE INDEX IF NOT EXISTS idx_x_id ON X(id)",
                    "CREATE INDEX IF NOT EXISTS idx_obs_cell_id ON obs(cell_id)"
                ]

                for index_sql in critical_indexes:
                    self.connection.execute(index_sql)

                print("基础索引创建完成")
            except Exception as e:
                print(f"创建基础索引时出错: {e}")




    #========== 其他 ====================
    # def minibatch_scan_random_no_replace_CSR(self, total_num, batch_size, drop_last):
    #     """随机读取不放回模式"""
    #
    #     # 生成所有可能的索引
    #     all_indices = list(range(total_num))
    #     random.shuffle(all_indices)  # 随机打乱
    #
    #     # 计算批次数量
    #     if drop_last:
    #         num_batches = total_num // batch_size
    #     else:
    #         num_batches = (total_num + batch_size - 1) // batch_size
    #
    #     for i in range(num_batches):
    #         start_idx = i * batch_size
    #         end_idx = start_idx + batch_size
    #
    #         # 如果是最后一批且不丢弃剩余数据，调整结束位置
    #         if not drop_last and i == num_batches - 1 and total_num % batch_size != 0:
    #             end_idx = start_idx + (total_num % batch_size)
    #
    #         # 获取当前批次的索引
    #         batch_indices = all_indices[start_idx:end_idx]
    #
    #         print(f"\n--- 批次 {num_batches} 详细时间分析 ---")
    #         print(f"\n--- 批次大小 {batch_size} ---")
    #
    #         query_CSR_indptr = f"SELECT * FROM X_CSR_indptr LIMIT {batch_size} OFFSET {start_idx}" # todo 获取CSR_indptr
    #
    #         CSR_indptr_df = self.connection.execute(query_CSR_indptr).df()  # todo  CSR_indptr 转化为 df格式
    #
    #         # todo  可以读取出 df 再提取 需要的 array， 也可以直接提取需要的 array ；
    #         CSR_indptr_array = np.insert(CSR_indptr_df['indptr'].values - data_read_start, 0, 0)  # todo  df 提取 np 时间
    #
    #         # 起点： 初始 data_read_start = 0
    #         data_read_count = CSR_indptr_array[-1]  # 获取值数量： 获取 3 个值， 即data_read_count = CSR_indptr_array[-1]=3,最后一个值
    #         if (data_read_count > 0):
    #
    #             query_CSR_data = f"SELECT * FROM X_CSR_data LIMIT {data_read_count} OFFSET {data_read_start}"
    #
    #             CSR_data_df =  self.connection.execute(query_CSR_data).df()  # todo  CSR_data 转化为 df
    #
    #             CSR_indices_array = CSR_data_df['indices'].to_numpy()  # todo:  CSR_indices_array = [ 0,2,2]
    #             CSR_data_array = CSR_data_df['data'].to_numpy()  # todo:  CSR_data_array = [ 8,2,5]
    #             data_read_start = data_read_start + data_read_count  # 更新起点
    #
    #
    #         else:
    #             print("获取 0 个值，不查询 ")  # todo 空值的处理
    #
    #
    #         sub_obs = self.query(f"SELECT * FROM obs LIMIT {current_batch_size} OFFSET {offset}")
    #
    #         sub_var = self.query("SELECT * FROM var")
    #
    #         sub_obs = sub_obs.iloc[:, 1:]  # 去掉数据表中的前1列 id
    #         sub_var = sub_var.iloc[:, 1:]  # 去掉数据表中的前1列
    #         cell_id = sub_obs['cell_id'] # 获取cell_id
    #         gene_id = sub_var['gene_id']  # 获取gene_id
    #
    #         # 创建CSR格式的稀疏矩阵
    #         X = sp.csr_matrix((CSR_data_array, CSR_indices_array, CSR_indptr_array),
    #                           shape=(len(sub_obs), len(sub_var)))
    #         # 创建AnnData对象
    #         adata = AnnData(X=X, obs=sub_obs, var=sub_var)
    #
    #         adata.obs_names = cell_id.astype(str)  # 设置观测名为cell_id
    #         adata.obs.index = cell_id.astype(str)
    #         adata.var_names = gene_id.astype(str)  # 设置变量名为gene_id
    #         adata.var.index = gene_id.astype(str)
    #
    #         # 返回数据
    #         yield adata

    def minibatch_scan_random_replace_CSR(self, total_num, batch_size):
        """随机读取有放回模式"""
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


    def minibatch_scan_random_no_replace(self, total_num, batch_size, drop_last):
        """随机读取不放回模式"""

        # 生成所有可能的索引
        all_indices = list(range(total_num))
        random.shuffle(all_indices)  # 随机打乱

        # 计算批次数量
        if drop_last:
            num_batches = total_num // batch_size
        else:
            num_batches = (total_num + batch_size - 1) // batch_size

        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size

            # 如果是最后一批且不丢弃剩余数据，调整结束位置
            if not drop_last and i == num_batches - 1 and total_num % batch_size != 0:
                end_idx = start_idx + (total_num % batch_size)

            # 获取当前批次的索引
            batch_indices = all_indices[start_idx:end_idx]

            # 构建IN查询,将整数索引转换为字符串,用逗号连接所有字符串 ; 如 batch_indices = [0, 1, 2] ，indices_str = "0,1,2"
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

    def minibatch_scan_random_replace(self, total_num, batch_size):
        """随机读取有放回模式"""
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

    # todo 待修改和验证  将多个数据集拼接成一个大数据集
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

            # 创建新的Atlas对象
            from ._atlas import Atlas  # 根据你的实际导入路径调整
            new_atlas = Atlas("combined_atlas", new_db_path)
            new_atlas.connection = new_conn

            return new_atlas

        except Exception as e:
            logger.error(f"合并过程中出错: {e}")
            # 清理临时文件
            if os.path.exists(new_db_path):
                os.remove(new_db_path)
            raise

    def _plot_batch_times(self, batch_times, batch_size):
        """
        绘制批次与时间的图像

        Args:
            batch_times: 包含批次时间数据的字典
            batch_size: 批次大小
        """
        try:
            # 创建DataFrame以便绘图
            df = pd.DataFrame(batch_times)

            # 创建图表
            plt.figure(figsize=(12, 8))

            # 绘制各阶段时间
            plt.subplot(2, 1, 1)

            plt.plot(df['batch_num'], df['time_X_CSR_indptr'], marker='o', label='time_X_CSR_indptr')
            plt.plot(df['batch_num'], df['time_X_CSR_data'], marker='s', label='time_X_CSR_data')
            plt.plot(df['batch_num'], df['time_X_CSR_data_df'], marker='^', label='time_X_CSR_data_df')
            plt.plot(df['batch_num'], df['time_anndata'], marker='d', label='time_anndata')

            plt.xlabel('batch_num')
            plt.ylabel('time (s)')
            plt.title(f'Processing Time per Batch (batch_size: {batch_size})')
            plt.legend()
            plt.grid(True, alpha=0.3)

            # 绘制总时间
            plt.subplot(2, 1, 2)
            plt.plot(df['batch_num'], df['total_time'], marker='o', color='red', linewidth=2)

            plt.xlabel('batch_num')
            plt.ylabel('total_time (s)')
            plt.title(f'Total Processing Time per Batch (batch_size: {batch_size})')
            plt.grid(True, alpha=0.3)

            # 添加统计信息
            avg_time = df['total_time'].mean()
            max_time = df['total_time'].max()
            min_time = df['total_time'].min()

            plt.text(0.02, 0.98, f'avg_time: {avg_time:.2f}s\nmax_time: {max_time:.2f}s\nmin_time: {min_time:.2f}s',
                     transform=plt.gca().transAxes, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            plt.tight_layout()

            # 保存图表
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"batch_times_plot_{batch_size}_{timestamp}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"时间图表已保存为: {filename}")

            # 显示图表（如果环境支持）
            plt.show()

            # 打印详细统计信息
            # self._print_time_statistics(df, batch_size)

        except Exception as e:
            print(f"绘制时间图表时出错: {e}")

    def _print_time_statistics(self, df, batch_size):
        """
        打印详细的时间统计信息

        Args:
            df: 包含时间数据的DataFrame
            batch_size: 批次大小
        """
        print("\n" + "=" * 60)
        print(f"批次大小 {batch_size} 的时间统计")
        print("=" * 60)

        # 计算各阶段平均时间
        avg_X = df['time_X'].mean()
        avg_obs = df['time_obs'].mean()
        avg_var = df['time_var'].mean()
        avg_anndata = df['time_anndata'].mean()
        avg_total = df['total_time'].mean()

        # 计算各阶段占比
        total_sum = df['time_X'].sum() + df['time_obs'].sum() + df['time_var'].sum() + df['time_anndata'].sum()
        pct_X = (df['time_X'].sum() / total_sum) * 100
        pct_obs = (df['time_obs'].sum() / total_sum) * 100
        pct_var = (df['time_var'].sum() / total_sum) * 100
        pct_anndata = (df['time_anndata'].sum() / total_sum) * 100

        print(f"总批次数: {len(df)}")
        print(f"平均每批次总时间: {avg_total:.2f} 秒")
        print("\n各阶段平均时间:")
        print(f"  - X表读取: {avg_X:.2f} 秒 ({pct_X:.1f}%)")
        print(f"  - obs表读取: {avg_obs:.2f} 秒 ({pct_obs:.1f}%)")
        print(f"  - var表读取: {avg_var:.2f} 秒 ({pct_var:.1f}%)")
        print(f"  - anndata生成: {avg_anndata:.2f} 秒 ({pct_anndata:.1f}%)")

        # 检测性能异常
        max_batch = df.loc[df['total_time'].idxmax()]
        min_batch = df.loc[df['total_time'].idxmin()]

        print(f"\n性能分析:")
        print(f"  - 最慢批次: #{int(max_batch['batch_num'])} ({max_batch['total_time']:.2f}秒)")
        print(f"  - 最快批次: #{int(min_batch['batch_num'])} ({min_batch['total_time']:.2f}秒)")
        print(f"  - 性能波动: {(max_batch['total_time'] - min_batch['total_time']):.2f}秒")

        # 检测是否有深翻页问题（时间随批次增加而增加）
        if len(df) > 5:  # 至少需要5个批次才能检测趋势
            first_half_avg = df.iloc[:len(df) // 2]['total_time'].mean()
            second_half_avg = df.iloc[len(df) // 2:]['total_time'].mean()

            if second_half_avg > first_half_avg * 1.2:  # 如果后半段平均时间比前半段多20%
                print(
                    f"  - ⚠️ 检测到潜在深翻页问题: 后半段批次比前半段慢 {(second_half_avg / first_half_avg - 1) * 100:.1f}%")
            else:
                print(f"  - ✅ 未检测到明显深翻页问题")

        print("=" * 60)

    def _plot_detailed_batch_times(self, batch_times, batch_size):
        """绘制详细的批次时间分析图表"""
        try:

            df = pd.DataFrame(batch_times)

            # 创建详细的性能分析图表
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

            # 图表1: X表各阶段时间分解
            x = np.arange(len(df))
            width = 0.2

            ax1.bar(x - width, df['time_X_query_execution'], width, label='查询执行', alpha=0.8)
            ax1.bar(x, df['time_X_data_extraction'], width, label='数据提取', alpha=0.8)
            ax1.bar(x + width, df['time_X_df_conversion'], width, label='DF转换', alpha=0.8)

            ax1.set_xlabel('批次编号')
            ax1.set_ylabel('时间 (秒)')
            ax1.set_title(f'X表各阶段时间分解 (批次大小: {batch_size})')
            ax1.set_xticks(x)
            ax1.set_xticklabels(df['batch_num'])
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # 图表2: 时间占比趋势
            query_ratio = df['time_X_query_execution'] / df['time_X_total'] * 100
            extract_ratio = df['time_X_data_extraction'] / df['time_X_total'] * 100
            df_ratio = df['time_X_df_conversion'] / df['time_X_total'] * 100

            ax2.stackplot(df['batch_num'], query_ratio, extract_ratio, df_ratio,
                          labels=['查询执行', '数据提取', 'DF转换'], alpha=0.7)

            ax2.set_xlabel('批次编号')
            ax2.set_ylabel('时间占比 (%)')
            ax2.set_title('X表各阶段时间占比趋势')
            ax2.legend(loc='upper left')
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim(0, 100)

            # 图表3: 数据处理效率
            ax3.plot(df['batch_num'], df['data_size_mb'], 'o-',
                     label='数据量 (MB)', color='green', linewidth=2)
            ax3_twin = ax3.twinx()

            # 计算转换速度
            conversion_speed = df['data_size_mb'] / df['time_X_df_conversion']
            ax3_twin.plot(df['batch_num'], conversion_speed, 's-',
                          label='DF转换速度 (MB/s)', color='red', linewidth=2)

            ax3.set_xlabel('批次编号')
            ax3.set_ylabel('数据量 (MB)', color='green')
            ax3_twin.set_ylabel('转换速度 (MB/s)', color='red')
            ax3.set_title('数据量与转换速度')
            ax3.legend(loc='upper left')
            ax3_twin.legend(loc='upper right')

            # 图表4: 性能相关性分析
            scatter = ax4.scatter(df['data_size_mb'], df['time_X_df_conversion'],
                                  c=df['batch_num'], cmap='viridis', s=100, alpha=0.7)

            ax4.set_xlabel('数据量 (MB)')
            ax4.set_ylabel('DF转换时间 (秒)')
            ax4.set_title('数据量 vs DF转换时间')

            # 添加颜色条
            cbar = plt.colorbar(scatter, ax=ax4)
            cbar.set_label('批次编号')

            # 添加趋势线
            if len(df) > 1:
                z = np.polyfit(df['data_size_mb'], df['time_X_df_conversion'], 1)
                p = np.poly1d(z)
                ax4.plot(df['data_size_mb'], p(df['data_size_mb']), "r--", alpha=0.8,
                         label=f'趋势线: y = {z[0]:.4f}x + {z[1]:.4f}')
                ax4.legend()

            # 统计信息
            avg_query_time = df['time_X_query_execution'].mean()
            avg_extract_time = df['time_X_data_extraction'].mean()
            avg_df_time = df['time_X_df_conversion'].mean()
            avg_total_x = df['time_X_total'].mean()

            avg_query_ratio = avg_query_time / avg_total_x * 100
            avg_extract_ratio = avg_extract_time / avg_total_x * 100
            avg_df_ratio = avg_df_time / avg_total_x * 100

            avg_speed = df['data_size_mb'].sum() / df['time_X_df_conversion'].sum()

            stats_text = f"""X表时间统计:
    平均查询时间: {avg_query_time:.3f}s ({avg_query_ratio:.1f}%)
    平均提取时间: {avg_extract_time:.3f}s ({avg_extract_ratio:.1f}%)
    平均转换时间: {avg_df_time:.3f}s ({avg_df_ratio:.1f}%)
    平均转换速度: {avg_speed:.2f} MB/s
    总数据量: {df['data_size_mb'].sum():.2f} MB"""

            fig.text(0.02, 0.02, stats_text, fontsize=10,
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            plt.tight_layout()

            # 保存图表
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"detailed_x_table_analysis_{batch_size}_{timestamp}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"详细X表分析图表已保存为: {filename}")

            plt.show()

        except Exception as e:
            print(f"绘制详细分析图表时出错: {e}")

    # def query_minibatch_prefetch(
    #         self,
    #         batch_size,
    #         drop_last=False,
    #         prefetch=8,
    # ):
    #     """
    #     带 prefetch 的 minibatch 读取（路 1）
    #     """
    #
    #     # === 关键修复点：安全获取 obs 总数 ===
    #     if not hasattr(self, "_cached_n_obs"):
    #         # 只查一次，缓存下来
    #         result = self.connection.execute(
    #             "SELECT COUNT(*) AS cnt FROM obs"
    #         ).fetchone()
    #         self._cached_n_obs = result[0]
    #
    #     total_num = self._cached_n_obs
    #
    #     base_gen = self.minibatch_scan_order_cursor_csr_df_arrow_mega4_obs(
    #         total_num=total_num,
    #         batch_size=batch_size,
    #         drop_last=drop_last,
    #     )
    #
    #     return PrefetchGenerator(base_gen, prefetch=prefetch)

