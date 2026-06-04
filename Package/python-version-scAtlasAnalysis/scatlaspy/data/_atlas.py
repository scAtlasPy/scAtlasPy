from typing import *
from _duckdb import DuckDBPyConnection
import duckdb
import os
import logging
from ._minibatch import MinibatchFetchMultiThreads
from ._filter_index import FilterBuildIndex

# 配置日志
logger = logging.getLogger("Atlas")
logger.addHandler(logging.NullHandler())


def set_verbosity(level: str = "warning"):
    """Set the verbosity level for Atlas logging.

    Parameters
    ----------
    level
        Verbosity level. Available options are `"error"`, `"warning"`,
        `"info"`, and `"debug"`.

        - `"error"` only shows error messages.
        - `"warning"` shows warnings and errors.
        - `"info"` shows general running information.
        - `"debug"` shows detailed debugging information.

    Returns
    -------
    None
        The logging level is updated in place.

    Examples
    --------
    Show general running information::

        set_verbosity("info")

    Show detailed debugging information::

        set_verbosity("debug")

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
    """
    这是一个Atlas类，用来管理和待分析数据集的数据库的交互

    """

    def __init__(self, name: str, path:str):

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
        """
        打开一个已经存在的 Atlas 数据库。

        推荐用法：
            atlas = Atlas.open(r"F:\\data\\xxx.sasql")
            atlas = Atlas.open(r"F:\\data\\xxx.sasql", mode="r")
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

        # 绕过 __init__，避免触发“自动创建数据库”的逻辑
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
        """
        创建一个新的 Atlas 数据库。

        推荐用法：
            atlas = Atlas.create("my_atlas", r"F:\\data")
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
        用 SQL 查询，返回 DuckDB 原始结果对象。
        不改变当前数据库连接模式。
        """
        logger.info("查询数据库，返回值类型为duckDB")

        if self.__connection is None:
            self.connect("r+")

        result = self.connection.execute(query)
        return result

    def describe(self) -> str:
        """
        显示 Atlas 数据库的基本信息：
        database / tables / table names / n_cells / n_genes
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
        def fmt(x):
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
        """
        显示指定表的字段名和前 n 行数据。

        用法：
            atlas.show("obs")
            atlas.show("var")
            atlas.show("X_HyS_data", n=5)
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
        """
        在交互环境中直接显示 Atlas 基本信息。
        """
        return self.describe()


    def __str__(self) -> str:
        """
        print(atlas) 时显示 Atlas 基本信息。
        """
        return self.describe()

    def filter_build_index(
            self,
            cell_condition: str | None = None,
            gene_condition: str | None = None,
            use_hvg: bool = True,
            select_data: str = "data_scale",
    ):
        ''' 过滤 + 建新表 + 建tid分块索引 '''
        builder = FilterBuildIndex(
            self.file_path,
            cell_condition=cell_condition,
            gene_condition=gene_condition,
            use_hvg=use_hvg,
            select_data=select_data,
        )
        builder.run()


    def minibatch_CSR(self , X_type = "CSR" ):
        ''' minibatch_CSR 格式读取 '''

        fetcher = MinibatchFetchMultiThreads( file_path = self.file_path , X_type =  X_type )
        for X_batch in fetcher.run():
            pass
            # yield X_batch


    def minibatch_dense(
            self,
            pass_mode: str = "single-pass",
            buffer_batch_num: int = 5,
            max_batches: int | None = None,
            batch_size: int = 2048,
    ):
        ''' minibatch_dense 格式读取 '''
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


