import time
from typing import *
from _duckdb import DuckDBPyConnection
import duckdb
import os
import logging # 管理各种类型的日志
from ._minibatch import MinibatchFetchMultiThreads
from ._filter_index import FilterBuildIndex
import scatlaspy as sap
import numpy as np

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
            path: 文件夹路径
            file_path：数据库文件的绝对路径
        """

        self.ipca = None # todo PCA降维 调试用

        logger.info(f"开始初始化 Atlas 实例，名称: {name}, 路径: {path}")

        self.__name = name  # 该数据库的名称（无后缀）
        self.__path = path  # 该数据库所在文件夹的路径
        self.__connection = None  # 存储当前的数据库连接
        self.__file_path = os.path.join(self.__path, f"{self.__name}.sasql") # 数据库文件的绝对路径

        if not os.path.exists(self.file_path):
            logger.info(f"数据库文件不存在，开始创建新数据库: {self.file_path}")
            try:
                self.__connection=self._create(name, path)
                logger.info(f"数据库创建成功: {self.file_path}")
            except Exception as e:
                logger.error(f"数据库创建失败: {str(e)}")
                raise
        else:
            self.__connection = self.connect("r+")
            logger.info(f"数据库文件已存在: {self.file_path}，已创建连接")

        logger.info("Atlas 实例初始化完成")

    @property
    def file_path(self) -> str:
        return self.__file_path

    @file_path.setter
    def file_path(self, value: str) -> None:
        self.__file_path = value

    @property
    def name(self) -> str:
        return self.__name

    @name.setter
    def name(self, value: str) -> None:
        self.__name = value

    @property
    def path(self) -> str:
        return self.__path

    @path.setter
    def path(self, value: str) -> None:
        self.__path = value

    @property
    def connection(self) -> Optional[duckdb.DuckDBPyConnection]:
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

        # 检查数据库是否已存在
        if os.path.exists(self.file_path):
            raise RuntimeError(f"数据库已存在: {self.file_path}")
        try:
            # 确保目录存在
            logger.debug(f"创建目录: {path}")
            os.makedirs(path, exist_ok=True)

            # 连接到持久化数据库文件，如果文件不存在会自动创建
            logger.debug("连接 DuckDB 数据库")
            con = duckdb.connect(database=self.file_path)

            logger.debug(f"数据库已成功创建：{self.file_path}")
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

        if self.__connection is not None: # 如果已有连接，先关闭
            logger.debug("已有数据库连接，先关闭现有连接")
            self.close()

        try:
            if mode == "r":  # 只读模式
                logger.debug("只读模式连接")
                # 检查文件是否存在
                if not os.path.exists(self.file_path):
                    logger.error(f"数据库文件不存在，无法以只读模式连接: {self.file_path}")
                    raise FileNotFoundError(f"数据库文件不存在: {self.file_path}")

                # 以只读模式连接
                self.__connection = duckdb.connect(database=self.file_path, read_only=True)
                logger.info(f"以只读模式连接数据库: {self.file_path}")

            elif mode == "r+":  # 读写模式
                logger.debug("读写模式连接")
                os.makedirs(self.__path, exist_ok=True)  # 确保目录存在

                # 无论文件是否存在，都会创建或连接
                self.__connection = duckdb.connect(database=self.file_path, read_only=False)

                if os.path.exists(self.file_path):
                    logger.info(f"以读写模式连接现有数据库: {self.file_path}")
                else:
                    logger.info(f"创建并连接新数据库: {self.file_path}")

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
        exists = os.path.exists(self.file_path)
        logger.debug(f"检查数据库文件是否存在: {self.file_path} -> {exists}")
        return exists

    def query(self, query):
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
        result = self.connection.execute(query)
        return result



    ''' 过滤 + 建新表 + 建tid分块索引 '''
    #  833206 * 17745   耗时 1:12
    #  2840130 x 24552  耗时 03:48
    def filter_build_index(
            self,
            fetch_size: int = 1_0000_0000,
            producer_num: int = 10,
            chunk_size: int = 2_0000_0000,
            cell_condition: str | None = None,
            gene_condition: str | None = None,
            use_hvg: bool = True,
            select_data: str = "data_scale",
    ):
        builder = FilterBuildIndex(
            self.file_path,
            fetch_size=fetch_size,
            producer_num=producer_num,
            chunk_size=chunk_size,
            cell_condition=cell_condition,
            gene_condition=gene_condition,
            use_hvg=use_hvg,
            select_data=select_data,
        )
        builder.run()


    ''' minibatch_CSR 格式读取 '''
    # 833206 * 17745   203 batch/s
    def minibatch_CSR(self , X_type = "CSR" ):

        fetcher = MinibatchFetchMultiThreads( file_path = self.file_path , X_type =  X_type )
        for X_batch in fetcher.run():
            pass
            # yield X_batch

    ''' minibatch_CSR 格式读取 '''
    # 833206 * 17745   single-pass 单次遍历 38 batch/s
    #                  multi-pass  多次遍历（加入缓存区，保证多次的随机性） 但会变慢
    # 缓冲区batch数量    buffer_batch_num = 2   17.93 batch/s
    #                  buffer_batch_num = 3   18.86 batch/s
    #                  buffer_batch_num = 5   19.30 batch/s
    #                  buffer_batch_num = 10  19.93 batch/s
    #                  buffer_batch_num = 15  20.21 batch/s
    #                  buffer_batch_num = 20  19.94 batch/s
    def minibatch_dense(
            self,
            pass_mode: str = "single-pass",
            buffer_batch_num: int = 5,
            max_batches: int | None = None,  # ✅ 新增：最多输出多少个 batch
            max_passes: int | None = None,  # ✅ 新增：最多遍历多少遍
            batch_size: int = 2048,  # ✅ 新增：传给 fetcher
            producer_num: int = 10,  # ✅ 新增：传给 fetcher
    ):
        """
        输出 dense minibatch。

        pass_mode
        ---------
        single-pass:
            只遍历数据库一遍。

        multi-pass:
            遍历完一遍后，自动重新创建 fetcher，
            继续第二遍、第三遍，直到满足 max_batches 或 max_passes。

        max_batches
        -----------
        最多输出多少个 batch。
        例如 PCA 训练需要 1000 个 batch：
            atlas.minibatch_dense(pass_mode="multi-pass", max_batches=1000)

        max_passes
        ----------
        最多遍历多少遍数据库。
        如果 max_batches=None 且 max_passes=None，
        multi-pass 会无限遍历，外部 break 后停止。
        """

        if pass_mode not in ("single-pass", "multi-pass"):
            raise ValueError("pass_mode 只支持 'single-pass' 或 'multi-pass'")

        # =====================================================
        # 1. single-pass：只跑一遍
        # =====================================================
        if pass_mode == "single-pass":

            fetcher = MinibatchFetchMultiThreads(
                file_path=self.file_path,
                batch_size=batch_size,
                producer_num=producer_num,
                X_type="dense",
                pass_mode="single-pass",
                buffer_batch_num=buffer_batch_num,
                max_batches=max_batches,
            )

            for X_batch in fetcher.run():
                yield X_batch

            return

        # =====================================================
        # 2. multi-pass：自动循环多遍
        # =====================================================
        produced_batches = 0
        pass_id = 0

        while True:

            # 如果达到 max_passes，停止
            if max_passes is not None and pass_id >= max_passes:
                print(f"[minibatch_dense] reach max_passes={max_passes}, stop")
                break

            # 如果达到 max_batches，停止
            if max_batches is not None and produced_batches >= max_batches:
                print(f"[minibatch_dense] reach max_batches={max_batches}, stop")
                break

            # 当前 pass 还需要输出多少 batch
            if max_batches is None:
                remain_batches = None
            else:
                remain_batches = max_batches - produced_batches

            print(
                f"[minibatch_dense] multi-pass start pass={pass_id + 1}, "
                f"produced={produced_batches}, "
                f"remain={remain_batches}"
            )

            fetcher = MinibatchFetchMultiThreads(
                file_path=self.file_path,
                batch_size=batch_size,
                producer_num=producer_num,
                X_type="dense",
                pass_mode="multi-pass",
                buffer_batch_num=buffer_batch_num,
                max_batches=remain_batches,
            )

            pass_batches = 0

            for X_batch in fetcher.run():
                produced_batches += 1
                pass_batches += 1

                yield X_batch

                if max_batches is not None and produced_batches >= max_batches:
                    break

            print(
                f"[minibatch_dense] pass={pass_id + 1} done, "
                f"pass_batches={pass_batches}, "
                f"total_produced={produced_batches}"
            )

            pass_id += 1

            # 防止异常情况下空 pass 无限循环
            if pass_batches == 0:
                print("[minibatch_dense] pass produced 0 batch, stop")
                break


