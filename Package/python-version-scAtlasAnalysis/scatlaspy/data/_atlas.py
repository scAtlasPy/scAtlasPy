from typing import *
from _duckdb import DuckDBPyConnection
import duckdb
import os
import logging
from ._minibatch import MultiThreadedMinibatchFetcher
from ._filter_index import FilterIndexBuilder

# 配置日志
logger = logging.getLogger("Atlas")
logger.addHandler(logging.NullHandler())


def set_verbosity(level: str = "warning"):
    """设置 Atlas 包的日志输出级别。

    该函数调整名为 ``Atlas`` 的 logger，用于控制导入导出、预处理、工具函数和绘图流程中的日志详细程度。

    它只影响 scAtlasPy 自己的 logger，不会修改 Python root logger 或第三方库的日志设置。

    Parameters
    ----------
    level
        日志级别字符串，例如 ``"debug"``、``"info"``、``"warning"`` 或 ``"error"``。

    Examples
    --------
    调用该函数：::

        sap.set_verbosity(...)
    """

    level_map = {
        "error": logging.ERROR,
        "warning": logging.WARNING,
        "info": logging.INFO,
        "debug": logging.DEBUG,
    }

    if level not in level_map:
        raise ValueError("level 只支持: error, warning, info, debug")

    logging.basicConfig(
        level=level_map[level],
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logging.getLogger("Atlas").setLevel(level_map[level])

class Atlas:
    """Atlas 数据库对象。

    ``Atlas`` 封装一个持久化 DuckDB-backed ``.sasql`` 数据库文件，保存数据库名称、文件路径和当前连接。

    该对象提供创建、打开、查询、查看表结构、构建过滤索引和按 minibatch 读取表达矩阵等入口，是 scAtlasPy 中多数
    ``sap.pp``、``sap.tl`` 和 ``sap.pl`` 函数共同依赖的核心对象。

    Parameters
    ----------
    name
        对象名称、列名或 SQL 标识符，具体含义由调用位置决定。

    path
        目录路径或文件路径。
    """

    def __init__(self, name: str, path:str):

        """初始化对象。

        该内部函数属于Atlas 数据库核心模块，用于支撑同一模块中的公共 API。

        负责 ``.sasql`` 数据库对象、DuckDB 连接、SQL 查询、表结构查看、过滤索引和 minibatch 入口。

        它通常不会作为用户入口直接调用；直接调用时需要保证输入对象、数据库连接和相关临时表已经由上游步骤准备好。

        Parameters
        ----------
        name
            对象名称、列名或 SQL 标识符，具体含义由调用位置决定。

        path
            目录路径或文件路径。

        Notes
        -----
        这是内部 helper；除非需要扩展 scAtlasPy 内部流程，一般不建议在用户代码中直接调用。
        """
        logger.info(f"开始初始化 Atlas 实例，名称: {name}, 路径: {path}")

        self.__name = name  # 该数据库的名称（无后缀）
        self.__path = path  # 该数据库所在文件夹的路径
        self.__connection = None  # 存储当前的数据库连接
        self.__mode: Literal["r+", "r"] = "r+"  # 当前连接模式，默认读写
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

    @classmethod
    def open(
            cls,
            file_path: str,
            mode: Literal["r+", "r"] = "r+",
    ) -> "Atlas":
        """打开已经存在的 Atlas 数据库。

        该类方法接收一个 ``.sasql`` 文件路径，解析数据库名称和目录，创建 ``Atlas`` 对象，并按指定模式建立 DuckDB 连接。

        适合在已有 Atlas 数据库上继续执行过滤、预处理、降维、聚类、绘图或导出。

        Parameters
        ----------
        file_path
            输入文件路径或 Atlas ``.sasql`` 数据库文件路径。

        mode
            数据库打开模式，通常为 ``"r+"`` 或 ``"r"``。

        Returns
        -------
        result
            函数返回结果。具体类型取决于参数设置和内部执行路径。

        Examples
        --------
        调用该函数：::

            sap.open(...)
        """

        file_path = os.path.abspath(file_path)

        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"数据库文件不存在，无法打开: {file_path}\n"
                f"如果你想创建新数据库，请使用 Atlas.create(name, path)"
            )

        if os.path.isdir(file_path):
            raise IsADirectoryError(
                f"传入的是文件夹，不是数据库文件: {file_path}\n"
                f"请传入完整数据库文件路径，例如：Atlas.open(r'F:\\data\\xxx.sasql')"
            )
        path = os.path.dirname(file_path)
        filename = os.path.basename(file_path)

        if filename.endswith(".sasql"):
            name = filename[:-len(".sasql")]
        else:
            name = os.path.splitext(filename)[0]

        atlas = cls.__new__(cls)

        atlas._Atlas__name = name
        atlas._Atlas__path = path
        atlas._Atlas__file_path = file_path
        atlas._Atlas__connection = None
        atlas._Atlas__mode = mode

        atlas.connect(mode)

        logger.info(f"已打开 Atlas 数据库: {file_path}, mode={mode}")

        return atlas

    @classmethod
    def create(
            cls,
            name: str,
            path: str,
    ) -> "Atlas":
        """创建新的 Atlas 数据库。

        该类方法根据数据库名称和目录创建新的 ``.sasql`` 文件，并初始化 DuckDB 连接。

        如果目标文件已经存在，函数会按当前实现的检查逻辑避免无意覆盖已有数据库。

        Parameters
        ----------
        name
            对象名称、列名或 SQL 标识符，具体含义由调用位置决定。

        path
            目录路径或文件路径。

        Returns
        -------
        result
            函数返回结果。具体类型取决于参数设置和内部执行路径。

        Examples
        --------
        调用该函数：::

            sap.create(...)
        """

        path = os.path.abspath(path)
        file_path = os.path.join(path, f"{name}.sasql")

        if os.path.exists(file_path):
            raise FileExistsError(
                f"数据库已存在: {file_path}\n"
                f"如果你只是想重新连接已有数据库，请使用：\n"
                f"Atlas.open(r'{file_path}')"
            )

        return cls(name=name, path=path)

    @property
    def file_path(self) -> str:
        """执行 ``file_path`` 的核心功能。

        负责 ``.sasql`` 数据库对象、DuckDB 连接、SQL 查询、表结构查看、过滤索引和 minibatch 入口。

        函数会直接读取或写入 Atlas 数据库中的相关表，并尽量通过 SQL、分块读取或流式计算减少内存占用。

        整体用法和 Scanpy 中相近的 ``sap.file_path`` 风格 API 类似，但结果保存在 Atlas 数据库表中，便于后续步骤复用。

        Returns
        -------
        result
            函数返回结果。具体类型取决于参数设置和内部执行路径。

        Examples
        --------
        调用该函数：::

            sap.file_path(...)
        """
        return self.__file_path

    @file_path.setter
    def file_path(self, value: str) -> None:
        """执行 ``file_path`` 的核心功能。

        负责 ``.sasql`` 数据库对象、DuckDB 连接、SQL 查询、表结构查看、过滤索引和 minibatch 入口。

        函数会直接读取或写入 Atlas 数据库中的相关表，并尽量通过 SQL、分块读取或流式计算减少内存占用。

        整体用法和 Scanpy 中相近的 ``sap.file_path`` 风格 API 类似，但结果保存在 Atlas 数据库表中，便于后续步骤复用。

        Parameters
        ----------
        value
            属性的新值。

        Examples
        --------
        调用该函数：::

            sap.file_path(...)
        """
        self.__file_path = value

    @property
    def name(self) -> str:
        """执行 ``name`` 的核心功能。

        负责 ``.sasql`` 数据库对象、DuckDB 连接、SQL 查询、表结构查看、过滤索引和 minibatch 入口。

        函数会直接读取或写入 Atlas 数据库中的相关表，并尽量通过 SQL、分块读取或流式计算减少内存占用。

        整体用法和 Scanpy 中相近的 ``sap.name`` 风格 API 类似，但结果保存在 Atlas 数据库表中，便于后续步骤复用。

        Returns
        -------
        result
            函数返回结果。具体类型取决于参数设置和内部执行路径。

        Examples
        --------
        调用该函数：::

            sap.name(...)
        """
        return self.__name

    @name.setter
    def name(self, value: str) -> None:
        """执行 ``name`` 的核心功能。

        负责 ``.sasql`` 数据库对象、DuckDB 连接、SQL 查询、表结构查看、过滤索引和 minibatch 入口。

        函数会直接读取或写入 Atlas 数据库中的相关表，并尽量通过 SQL、分块读取或流式计算减少内存占用。

        整体用法和 Scanpy 中相近的 ``sap.name`` 风格 API 类似，但结果保存在 Atlas 数据库表中，便于后续步骤复用。

        Parameters
        ----------
        value
            属性的新值。

        Examples
        --------
        调用该函数：::

            sap.name(...)
        """
        self.__name = value

    @property
    def path(self) -> str:
        """执行 ``path`` 的核心功能。

        负责 ``.sasql`` 数据库对象、DuckDB 连接、SQL 查询、表结构查看、过滤索引和 minibatch 入口。

        函数会直接读取或写入 Atlas 数据库中的相关表，并尽量通过 SQL、分块读取或流式计算减少内存占用。

        整体用法和 Scanpy 中相近的 ``sap.path`` 风格 API 类似，但结果保存在 Atlas 数据库表中，便于后续步骤复用。

        Returns
        -------
        result
            函数返回结果。具体类型取决于参数设置和内部执行路径。

        Examples
        --------
        调用该函数：::

            sap.path(...)
        """
        return self.__path

    @path.setter
    def path(self, value: str) -> None:
        """执行 ``path`` 的核心功能。

        负责 ``.sasql`` 数据库对象、DuckDB 连接、SQL 查询、表结构查看、过滤索引和 minibatch 入口。

        函数会直接读取或写入 Atlas 数据库中的相关表，并尽量通过 SQL、分块读取或流式计算减少内存占用。

        整体用法和 Scanpy 中相近的 ``sap.path`` 风格 API 类似，但结果保存在 Atlas 数据库表中，便于后续步骤复用。

        Parameters
        ----------
        value
            属性的新值。

        Examples
        --------
        调用该函数：::

            sap.path(...)
        """
        self.__path = value

    @property
    def connection(self) -> Optional[duckdb.DuckDBPyConnection]:
        """执行 ``connection`` 的核心功能。

        负责 ``.sasql`` 数据库对象、DuckDB 连接、SQL 查询、表结构查看、过滤索引和 minibatch 入口。

        函数会直接读取或写入 Atlas 数据库中的相关表，并尽量通过 SQL、分块读取或流式计算减少内存占用。

        整体用法和 Scanpy 中相近的 ``sap.connection`` 风格 API 类似，但结果保存在 Atlas 数据库表中，便于后续步骤复用。

        Returns
        -------
        result
            函数返回结果。具体类型取决于参数设置和内部执行路径。

        Examples
        --------
        调用该函数：::

            sap.connection(...)
        """
        return self.__connection

    @connection.setter
    def connection(self, value: str)-> None:
        """执行 ``connection`` 的核心功能。

        负责 ``.sasql`` 数据库对象、DuckDB 连接、SQL 查询、表结构查看、过滤索引和 minibatch 入口。

        函数会直接读取或写入 Atlas 数据库中的相关表，并尽量通过 SQL、分块读取或流式计算减少内存占用。

        整体用法和 Scanpy 中相近的 ``sap.connection`` 风格 API 类似，但结果保存在 Atlas 数据库表中，便于后续步骤复用。

        Parameters
        ----------
        value
            属性的新值。

        Examples
        --------
        调用该函数：::

            sap.connection(...)
        """
        self.__connection = value

    def _create(self, name: str, path: str) -> duckdb.DuckDBPyConnection:
        """创建 Atlas 工作流所需的数据库表。

        该内部函数属于Atlas 数据库核心模块，用于支撑同一模块中的公共 API。

        负责 ``.sasql`` 数据库对象、DuckDB 连接、SQL 查询、表结构查看、过滤索引和 minibatch 入口。

        它通常不会作为用户入口直接调用；直接调用时需要保证输入对象、数据库连接和相关临时表已经由上游步骤准备好。

        Parameters
        ----------
        name
            对象名称、列名或 SQL 标识符，具体含义由调用位置决定。

        path
            目录路径或文件路径。

        Returns
        -------
        result
            函数返回结果。具体类型取决于参数设置和内部执行路径。

        Notes
        -----
        这是内部 helper；除非需要扩展 scAtlasPy 内部流程，一般不建议在用户代码中直接调用。
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
        """建立 DuckDB 数据库连接。

        该方法打开当前 Atlas 对象指向的 ``.sasql`` 文件，并把连接保存到 ``atlas.connection``。

        多数读写函数要求连接已存在；如果连接已关闭或对象刚被创建，可以调用该方法重新连接。

        Parameters
        ----------
        mode
            数据库打开模式，通常为 ``"r+"`` 或 ``"r"``。

        Returns
        -------
        result
            函数返回结果。具体类型取决于参数设置和内部执行路径。

        Examples
        --------
        调用该函数：::

            sap.connect(...)
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
        """执行 ``close`` 的核心功能。

        负责 ``.sasql`` 数据库对象、DuckDB 连接、SQL 查询、表结构查看、过滤索引和 minibatch 入口。

        函数会直接读取或写入 Atlas 数据库中的相关表，并尽量通过 SQL、分块读取或流式计算减少内存占用。

        整体用法和 Scanpy 中相近的 ``sap.close`` 风格 API 类似，但结果保存在 Atlas 数据库表中，便于后续步骤复用。

        Examples
        --------
        调用该函数：::

            sap.close(...)
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
        """执行 ``execute_sql`` 的核心功能。

        负责 ``.sasql`` 数据库对象、DuckDB 连接、SQL 查询、表结构查看、过滤索引和 minibatch 入口。

        函数会直接读取或写入 Atlas 数据库中的相关表，并尽量通过 SQL、分块读取或流式计算减少内存占用。

        整体用法和 Scanpy 中相近的 ``sap.execute_sql`` 风格 API 类似，但结果保存在 Atlas
        数据库表中，便于后续步骤复用。

        Parameters
        ----------
        sql
            需要执行的 SQL 语句。

        Returns
        -------
        result
            函数返回结果。具体类型取决于参数设置和内部执行路径。

        Examples
        --------
        调用该函数：::

            sap.execute_sql(...)
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
        """执行 ``exists`` 的核心功能。

        负责 ``.sasql`` 数据库对象、DuckDB 连接、SQL 查询、表结构查看、过滤索引和 minibatch 入口。

        函数会直接读取或写入 Atlas 数据库中的相关表，并尽量通过 SQL、分块读取或流式计算减少内存占用。

        整体用法和 Scanpy 中相近的 ``sap.exists`` 风格 API 类似，但结果保存在 Atlas 数据库表中，便于后续步骤复用。

        Returns
        -------
        result
            函数返回结果。具体类型取决于参数设置和内部执行路径。

        Examples
        --------
        调用该函数：::

            sap.exists(...)
        """
        exists = os.path.exists(self.file_path)
        logger.debug(f"检查数据库文件是否存在: {self.file_path} -> {exists}")
        return exists

    def query(self, query: str):
        """执行 ``query`` 的核心功能。

        负责 ``.sasql`` 数据库对象、DuckDB 连接、SQL 查询、表结构查看、过滤索引和 minibatch 入口。

        函数会直接读取或写入 Atlas 数据库中的相关表，并尽量通过 SQL、分块读取或流式计算减少内存占用。

        整体用法和 Scanpy 中相近的 ``sap.query`` 风格 API 类似，但结果保存在 Atlas 数据库表中，便于后续步骤复用。

        Parameters
        ----------
        query
            需要执行的 SQL 查询语句。

        Returns
        -------
        result
            函数返回结果。具体类型取决于参数设置和内部执行路径。

        Examples
        --------
        调用该函数：::

            sap.query(...)
        """
        logger.info("查询数据库，返回值类型为pandas")
        if self.__connection is None:
            self.connect("r+")
        result = self.connection.execute(query)
        df = result.df() # 将查询结果转换为pandas DataFrame
        return df

    def query_raw(self, query: str):
        """执行 ``query_raw`` 的核心功能。

        负责 ``.sasql`` 数据库对象、DuckDB 连接、SQL 查询、表结构查看、过滤索引和 minibatch 入口。

        函数会直接读取或写入 Atlas 数据库中的相关表，并尽量通过 SQL、分块读取或流式计算减少内存占用。

        整体用法和 Scanpy 中相近的 ``sap.query_raw`` 风格 API 类似，但结果保存在 Atlas 数据库表中，便于后续步骤复用。

        Parameters
        ----------
        query
            需要执行的 SQL 查询语句。

        Returns
        -------
        result
            函数返回结果。具体类型取决于参数设置和内部执行路径。

        Examples
        --------
        调用该函数：::

            sap.query_raw(...)
        """
        logger.info("查询数据库，返回值类型为duckDB")

        if self.__connection is None:
            self.connect("r+")

        result = self.connection.execute(query)
        return result

    def describe(self) -> str:
        """查看 Atlas 数据库概要。

        该方法读取数据库中的表列表、列信息和行数，并生成适合打印查看的文本摘要。

        它用于快速确认数据库是否包含 ``obs``、``var``、``X_HyS_data``、embedding 和分析结果表。

        Returns
        -------
        result
            函数返回结果。具体类型取决于参数设置和内部执行路径。

        Examples
        --------
        调用该函数：::

            sap.describe(...)
        """

        if self.__connection is None:
            self.connect("r+")

        conn = self.__connection

        # 1. 数据库路径
        database = self.file_path

        # 2. 查询所有表
        try:
            tables = [r[0] for r in conn.execute("SHOW TABLES").fetchall()]
        except Exception:
            tables = []

        table_names = ", ".join(tables) if len(tables) > 0 else "None"

        # 3. 查询 obs 细胞数
        if "obs" in tables:
            try:
                n_cells = conn.execute("SELECT COUNT(*) FROM obs").fetchone()[0]
            except Exception:
                n_cells = None
        else:
            n_cells = None

        # 4. 查询 var 基因数
        if "var" in tables:
            try:
                n_genes = conn.execute("SELECT COUNT(*) FROM var").fetchone()[0]
            except Exception:
                n_genes = None
        else:
            n_genes = None

        # 5. 格式化输出
        def fmt(x: Any):
            """执行 ``fmt`` 的核心功能。

            负责 ``.sasql`` 数据库对象、DuckDB 连接、SQL 查询、表结构查看、过滤索引和 minibatch 入口。

            函数会直接读取或写入 Atlas 数据库中的相关表，并尽量通过 SQL、分块读取或流式计算减少内存占用。

            整体用法和 Scanpy 中相近的 ``sap.fmt`` 风格 API 类似，但结果保存在 Atlas 数据库表中，便于后续步骤复用。

            Parameters
            ----------
            x
                需要排序、格式化或转换的单个输入值。

            Returns
            -------
            result
                函数返回结果。具体类型取决于参数设置和内部执行路径。

            Examples
            --------
            调用该函数：::

                sap.fmt(...)
            """
            return "NA" if x is None else f"{int(x):,}"

        text = (
            f"database    : {database}\n"
            f"tables      : {len(tables)}\n"
            f"table names : {table_names}\n"
            f"n_cells     : {fmt(n_cells)}\n"
            f"n_genes     : {fmt(n_genes)}"
        )

        return text

    def show(self, table_name: str, n: int = 5):
        """查看指定数据库表的前几行。

        该方法会检查目标表是否存在，读取表结构，并返回指定行数的数据。

        适合调试导入结果、查看 ``obs``/``var`` 新增列，或检查分析结果表是否写入成功。

        Parameters
        ----------
        table_name
            数据库表名。

        n
            数量参数，例如返回行数、抽样数量或参与计算的元素个数。

        Returns
        -------
        result
            函数返回结果。具体类型取决于参数设置和内部执行路径。

        Examples
        --------
        调用该函数：::

            sap.show(...)
        """

        if self.__connection is None:
            self.connect("r+")

        conn = self.__connection

        # 1. 检查表是否存在
        tables = [r[0] for r in conn.execute("SHOW TABLES").fetchall()]

        if table_name not in tables:
            raise ValueError(
                f"数据库中不存在表: {table_name}\n"
                f"当前可用表: {', '.join(tables) if len(tables) > 0 else 'None'}"
            )

        # 2. 安全引用表名
        table_sql = '"' + table_name.replace('"', '""') + '"'

        # 3. 获取字段名
        columns = [
            r[0]
            for r in conn.execute("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = ?
                ORDER BY ordinal_position
            """, [table_name]).fetchall()
        ]

        # 4. 查询前 n 行
        df = conn.execute(f"""
            SELECT *
            FROM {table_sql}
            LIMIT {int(n)}
        """).df()

        # 5. 打印结果
        print(f"table   : {table_name}")
        print(f"columns : {', '.join(columns)}")
        print(f"rows    : first {int(n)}")
        print(df)

        return df

    def __repr__(self) -> str:
        """执行 ``__repr__`` 的核心功能。

        该内部函数属于Atlas 数据库核心模块，用于支撑同一模块中的公共 API。

        负责 ``.sasql`` 数据库对象、DuckDB 连接、SQL 查询、表结构查看、过滤索引和 minibatch 入口。

        它通常不会作为用户入口直接调用；直接调用时需要保证输入对象、数据库连接和相关临时表已经由上游步骤准备好。

        Returns
        -------
        result
            函数返回结果。具体类型取决于参数设置和内部执行路径。

        Notes
        -----
        这是内部 helper；除非需要扩展 scAtlasPy 内部流程，一般不建议在用户代码中直接调用。
        """
        return self.describe()


    def __str__(self) -> str:
        """执行 ``__str__`` 的核心功能。

        该内部函数属于Atlas 数据库核心模块，用于支撑同一模块中的公共 API。

        负责 ``.sasql`` 数据库对象、DuckDB 连接、SQL 查询、表结构查看、过滤索引和 minibatch 入口。

        它通常不会作为用户入口直接调用；直接调用时需要保证输入对象、数据库连接和相关临时表已经由上游步骤准备好。

        Returns
        -------
        result
            函数返回结果。具体类型取决于参数设置和内部执行路径。

        Notes
        -----
        这是内部 helper；除非需要扩展 scAtlasPy 内部流程，一般不建议在用户代码中直接调用。
        """
        return self.describe()

    def build_read_index(
            self,
            cell_condition: str | None = None,
            gene_condition: str | None = None,
            use_hvg: bool = True,
            select_data: str = "data_scale",
    ):
        """根据过滤条件重建 Atlas 过滤索引。

        该方法调用 ``FilterIndexBuilder``，根据 ``obs`` 和 ``var`` 中的过滤列生成连续
        ``filter_cell_id`` 与 ``filter_gene_id``。

        随后会重建 ``X_HyS_data_filtered`` 和 ``X_HyS_indptr_filtered``，供 PCA、KMeans 和
        dense/CSR minibatch 读取使用。

        Parameters
        ----------
        cell_condition
            ``obs`` 中用于筛选细胞的布尔列名或条件。

        gene_condition
            ``var`` 中用于筛选基因的布尔列名或条件。

        use_hvg
            是否只处理高变基因。

        select_data
            从 ``X_HyS_data`` 中读取的表达字段。

        Examples
        --------
        调用该函数：::

            sap.build_read_index(...)
        """
        builder = FilterIndexBuilder(
            self.file_path,
            cell_condition=cell_condition,
            gene_condition=gene_condition,
            use_hvg=use_hvg,
            select_data=select_data,
        )
        builder.run()


    def get_minibatch_csr(self, x_type: str = "CSR"):
        """按 minibatch 读取 CSR 表达矩阵。

        该方法构造 ``MultiThreadedMinibatchFetcher``，从过滤后的 HyS 表中逐批恢复 CSR 矩阵。

        它适合需要稀疏矩阵输入的训练、调试或和 scipy sparse 工作流对接的场景。

        Parameters
        ----------
        x_type
            输出矩阵类型，通常为 ``"CSR"`` 或 ``"dense"``。

        Examples
        --------
        调用该函数：::

            sap.get_minibatch_csr(...)
        """

        fetcher = MultiThreadedMinibatchFetcher(file_path = self.file_path, x_type=  x_type)
        for X_batch in fetcher.run():
            pass
            # yield X_batch


    def get_minibatch_dense(
            self,
            pass_mode: str = "single-pass",
            buffer_batch_num: int = 5,
            max_batches: int | None = None,
            batch_size: int = 2048,
    ):
        """按 minibatch 读取 dense 表达矩阵。

        该方法从过滤后的 HyS 表中逐批恢复 dense 矩阵，并支持 ``single-pass`` 和 ``multi-pass`` 两种遍历模式。

        ``multi-pass`` 会使用 shuffle buffer 提高训练数据随机性，常用于流式 PCA 和 MiniBatchKMeans。

        Parameters
        ----------
        pass_mode
            minibatch 遍历模式，通常为 ``"single-pass"`` 或 ``"multi-pass"``。

        buffer_batch_num
            shuffle buffer 中缓存的 minibatch 数量。

        max_batches
            最多输出的 minibatch 数量；为 ``None`` 时不限制。

        batch_size
            每批读取、写入或处理的细胞数量；较大值通常更快但占用更多内存。

        Yields
        -------
        batch
            逐批生成的数据。具体类型取决于函数参数，例如 CSR 矩阵、dense 矩阵、DataFrame 或绘图数据。

        Examples
        --------
        调用该函数：::

            sap.get_minibatch_dense(...)
        """
        if pass_mode not in ("single-pass", "multi-pass"):
            raise ValueError("pass_mode 只支持 'single-pass' 或 'multi-pass'")


        # 1. single-pass：只跑一遍
        if pass_mode == "single-pass":

            fetcher = MultiThreadedMinibatchFetcher(
                file_path=self.file_path,
                batch_size=batch_size,
                x_type="dense",
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
                print(f"[get_minibatch_dense] reach max_batches={max_batches}, stop")
                break

            # 当前 pass 还需要输出多少 batch
            if max_batches is None:
                remain_batches = None
            else:
                remain_batches = max_batches - produced_batches

            print(
                f"[get_minibatch_dense] multi-pass start pass={pass_id + 1}, "
                f"produced={produced_batches}, "
                f"remain={remain_batches}"
            )

            fetcher = MultiThreadedMinibatchFetcher(
                file_path=self.file_path,
                batch_size=batch_size,
                x_type="dense",
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
                f"[get_minibatch_dense] pass={pass_id + 1} done, "
                f"pass_batches={pass_batches}, "
                f"total_produced={produced_batches}"
            )

            pass_id += 1

            # 防止异常情况下空 pass 无限循环
            if pass_batches == 0:
                print("[get_minibatch_dense] pass produced 0 batch, stop")
                break


