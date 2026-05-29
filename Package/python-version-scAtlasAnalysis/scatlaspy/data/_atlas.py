from typing import *
from _duckdb import DuckDBPyConnection
import duckdb
import os
import logging
from ._minibatch import MinibatchFetchMultiThreads
from ._filter_index import FilterBuildIndex

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
        """

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
    def filter_build_index(
            self,
            cell_condition: str | None = None,
            gene_condition: str | None = None,
            use_hvg: bool = True,
            select_data: str = "data_scale",
    ):
        builder = FilterBuildIndex(
            self.file_path,
            cell_condition=cell_condition,
            gene_condition=gene_condition,
            use_hvg=use_hvg,
            select_data=select_data,
        )
        builder.run()


    ''' minibatch_CSR 格式读取 '''
    def minibatch_CSR(self , X_type = "CSR" ):

        fetcher = MinibatchFetchMultiThreads( file_path = self.file_path , X_type =  X_type )
        for X_batch in fetcher.run():
            pass
            # yield X_batch

    ''' minibatch_CSR 格式读取 '''
    def minibatch_dense(
            self,
            pass_mode: str = "single-pass",
            buffer_batch_num: int = 5,
            max_batches: int | None = None,  # 最多输出多少个 batch
            batch_size: int = 2048,
    ):

        if pass_mode not in ("single-pass", "multi-pass"):
            raise ValueError("pass_mode 只支持 'single-pass' 或 'multi-pass'")


        # 1. single-pass：只跑一遍
        if pass_mode == "single-pass":

            fetcher = MinibatchFetchMultiThreads(
                file_path=self.file_path,
                batch_size=batch_size,
                X_type="dense",
                pass_mode="single-pass",
                buffer_batch_num=buffer_batch_num,
                max_batches=max_batches,
            )

            for X_batch in fetcher.run():
                yield X_batch
                # pass

            return

        # 2. multi-pass：自动循环多遍
        produced_batches = 0
        pass_id = 0

        while True:

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
                # pass

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


