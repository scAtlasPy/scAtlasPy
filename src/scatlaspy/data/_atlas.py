from typing import *
from _duckdb import DuckDBPyConnection
import duckdb
import os
import logging
import numpy as np
import pandas as pd
from os import PathLike
from anndata import AnnData
from ._minibatch import MultiThreadedMinibatchFetcher
from ._filter_index import FilterIndexBuilder
from ..io import (
    rename_duplicated_genes as _io_rename_duplicated_genes,
    get_anndata as _io_get_anndata,
    get_obs_df as _io_get_obs_df,
    load_anndata as _io_load_anndata,
    load_h5ad as _io_load_h5ad,
    load_multi_format as _io_load_multi_format,
    write_h5ad as _io_write_h5ad,
)

# 配置日志
logger = logging.getLogger("Atlas")
logger.addHandler(logging.NullHandler())


def set_verbosity(
        level: Literal["silence", "error", "warning", "info", "debug"] | None = "silence",
) -> None:
    """设置 scAtlasPy 的日志输出级别。

    该函数调整包内使用的 ``Atlas`` logger，统一控制导入、预处理、
    工具函数和绘图流程中的日志详细程度。它不会主动修改第三方库的日志级别。

    Parameters
    ----------
    level
        日志级别字符串。可选值包括 ``"silence"``、``"error"``、``"warning"``、``"info"`` 和 ``"debug"``。

        默认值为 ``"silence"``，表示关闭 ``Atlas`` logger，不输出 scAtlasPy自身日志。

        传入 ``None`` 时也会关闭 ``Atlas`` logger，用于兼容旧版本写法。

    Returns
    -------
    None
        该函数直接修改 ``Atlas`` logger 的输出级别，不返回对象。

    Examples
    --------
    默认关闭 scAtlasPy 自身日志::

        sap.set_verbosity()

    显式关闭 scAtlasPy 自身日志::

        sap.set_verbosity("silence")

    只显示警告和错误信息::

        sap.set_verbosity("warning")

    调试导入或预处理流程::

        sap.set_verbosity("debug")
        atlas = sap.Atlas(r"F:\\data\\pbmc")
    """

    atlas_logger = logging.getLogger("Atlas")

    if level is None:
        level = "silence"

    level = str(level).lower()

    level_map = {
        "error": logging.ERROR,
        "warning": logging.WARNING,
        "info": logging.INFO,
        "debug": logging.DEBUG,
    }

    if level == "silence":
        atlas_logger.disabled = True
        return

    if level not in level_map:
        raise ValueError("level 只支持: silence, error, warning, info, debug, None")

    atlas_logger.disabled = False
    atlas_logger.setLevel(level_map[level])

    atlas_logger.handlers = [
        handler
        for handler in atlas_logger.handlers
        if not isinstance(handler, logging.NullHandler)
    ]

    if not atlas_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        atlas_logger.addHandler(handler)

    for handler in atlas_logger.handlers:
        handler.setLevel(level_map[level])

    atlas_logger.propagate = False

class Atlas:
    """Atlas 数据库对象。

    Atlas 对象管理一个持久化的 DuckDB-backed ``.sasql`` 数据库，保存数据库路径、活动连接，
    并提供创建、打开、查询、检查、读取和调用 IO 函数的便捷方法。

    Attributes
    ----------
    file_path
        当前 ``.sasql`` 数据库文件路径。
    connection
        当前 DuckDB 连接对象。

    Examples
    --------
    创建或连接一个 Atlas 数据库::

        atlas = sap.Atlas(r"F:\\data\\pbmc")

    使用对象式 API 导入 h5ad 文件::

        atlas = sap.Atlas(r"F:\\data\\pbmc")
        atlas.load_h5ad(r"F:\\data\\pbmc.h5ad")

    查看数据库内容并关闭连接::

        atlas.describe()
        atlas.head("obs", n=5)
        atlas.close()"""

    def __init__(
            self,
            file_name: PathLike[str] | str,
            db_memory_limit: str | int | None = None,
    ):
        """初始化 Atlas 数据库对象。

        构造函数会根据 ``file_name`` 推断 ``.sasql`` 数据库路径，
        创建父目录，并建立 DuckDB 连接。如果数据库文件尚不存在，
        会自动创建空数据库。

        Parameters
        ----------
        file_name
            Atlas 数据库文件路径或数据库名称。可以传入完整 ``.sasql`` 路径，
            也可以传入不带后缀的路径；函数会自动补全 ``.sasql`` 后缀。

        db_memory_limit
            DuckDB 可使用的内存上限。可以传入 DuckDB 支持的字符串，例如``"4GB"``；
            也可以传入整数，整数会按 GB 解释，例如 ``4`` 等价于 ``"4GB"``。

            默认值为 ``None``。当为 ``None`` 时，会自动获取当前系统物理内存总量，
            并向下取整为整数 GB 后设置为 DuckDB 的内存上限。
            这意味着内存管理由DuckDB自身的引擎负责，而非施加明确的限制。
            例如当前系统内存约为 31.8GB 时，会自动设置为 ``"31GB"``。

            该参数只限制 DuckDB 查询和中间计算使用的内存，
            不限制Python、NumPy 或 pandas 本身占用的内存。

        Returns
        -------
        None
            结果直接写入 Atlas 数据库。

        Examples
        --------
        传入不带后缀的数据库路径::

            atlas = sap.Atlas(r"F:\\data\\test_10W")

        传入完整 ``.sasql`` 文件路径::

            atlas = sap.Atlas(r"F:\\data\\test_10W.sasql")

        全局打开更详细日志::

            sap.set_verbosity("info")
            atlas = sap.Atlas(r"F:\\data\\test_10W.sasql")

        限制 DuckDB 查询和中间计算最多使用 4GB 内存::

            atlas = sap.Atlas(r"F:\\data\\test_10W", db_memory_limit="4GB")
        """

        self.__file_path = self._resolve_file_path(file_name)
        self.__connection = None
        self.__mode: Literal["r+", "r"] = "r+"
        self.__db_memory_limit = self._resolve_db_memory_limit(db_memory_limit)

        logger.info(f"开始初始化 Atlas 实例，file_name: {self.file_path}")

        if not os.path.exists(self.file_path):
            logger.info(f"数据库文件不存在，开始创建新数据库: {self.file_path}")
            try:
                self.__connection = self._create()
                logger.info(f"数据库创建成功: {self.file_path}")
            except Exception as e:
                logger.error(f"数据库创建失败: {str(e)}")
                raise
        else:
            self.__connection = self.connect("r+")
            logger.info(f"数据库文件已存在: {self.file_path}，已创建连接")

        logger.info("Atlas 实例初始化完成")


    @staticmethod
    def _resolve_file_path(file_name: PathLike[str] | str) -> str:
        """
        解析 Atlas 数据库路径。

        支持：
            Atlas(r"F:\\data\\file_name\\sql_obs.sasql")
            Atlas(r"F:\\data\\file_name\\sql_obs")
            Atlas(Path(r"F:\\data\\file_name\\sql_obs.sasql"))
            Atlas(Path(r"F:\\data\\file_name\\sql_obs"))

        如果没有 .sasql 后缀，会自动补上。
        """

        file_name = os.fspath(file_name)

        if not isinstance(file_name, str):
            raise TypeError(
                "file_name 必须是 str 或 PathLike[str] 类型，"
                f"但收到的是: {type(file_name)}"
            )

        file_name = file_name.strip()

        if file_name == "":
            raise ValueError("file_name 不能为空")

        file_name = os.path.expanduser(file_name)

        if file_name.lower().endswith(".sasql"):
            return os.path.abspath(file_name)

        return os.path.abspath(file_name + ".sasql")


    @staticmethod
    def _get_system_memory_gb_floor() -> int:
        """获取当前系统物理内存总量，并向下取整为整数 GB。

        该方法优先使用标准库实现，不额外依赖 psutil。
        Windows 下使用 GlobalMemoryStatusEx；
        Linux/macOS 下使用 os.sysconf。
        """

        total_bytes: int | None = None

        # Windows
        if os.name == "nt":
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            memory_status = MEMORYSTATUSEX()
            memory_status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)

            success = ctypes.windll.kernel32.GlobalMemoryStatusEx(
                ctypes.byref(memory_status)
            )

            if not success:
                raise RuntimeError("无法获取当前 Windows 系统物理内存大小")

            total_bytes = int(memory_status.ullTotalPhys)

        # Linux / macOS
        else:
            try:
                page_size = os.sysconf("SC_PAGE_SIZE")
                physical_pages = os.sysconf("SC_PHYS_PAGES")
                total_bytes = int(page_size * physical_pages)
            except Exception as e:
                raise RuntimeError(
                    "无法获取当前系统物理内存大小，请显式传入 db_memory_limit，例如 '32GB'"
                ) from e

        memory_gb = int(total_bytes // (1024 ** 3))

        if memory_gb <= 0:
            raise RuntimeError(
                "获取到的系统物理内存小于 1GB，请显式传入 db_memory_limit"
            )

        return memory_gb


    @staticmethod
    def _resolve_db_memory_limit(
            db_memory_limit: str | int | None,
    ) -> str | int:
        """解析 DuckDB 内存限制参数。

        当 ``db_memory_limit`` 为 ``None`` 时，自动获取当前系统物理内存总量，
        并向下取整为整数 GB，例如 31.8GB 会解析为 ``"31GB"``。
        """

        if db_memory_limit is None:
            memory_gb = Atlas._get_system_memory_gb_floor()
            return f"{memory_gb}GB"

        return db_memory_limit


    @property
    def file_path(self) -> str:
        """返回 Atlas 数据库文件路径。

        该属性返回当前 Atlas 对象指向的 ``.sasql`` 文件路径，可用于确认数据库实际保存位置。

        Returns
        -------
        str
            当前 Atlas 对象对应的 ``.sasql`` 数据库绝对路径。

        Examples
        --------
        查看当前数据库路径::

            atlas = sap.Atlas(r"F:\\data\\pbmc")
            atlas.file_path"""
        return self.__file_path


    @property
    def connection(self) -> Optional[duckdb.DuckDBPyConnection]:
        """返回当前 DuckDB 连接。

        该属性保存 Atlas 当前使用的 DuckDB 连接对象。通常不需要直接操作它，除非需要调用 DuckDB 的底层 API。

        Returns
        -------
        duckdb.DuckDBPyConnection
            当前 Atlas 数据库连接。

        Examples
        --------
        使用底层 DuckDB 连接执行查询::

            con = atlas.connection
            con.sql("SELECT COUNT(*) FROM obs").fetchone()"""
        return self.__connection


    @connection.setter
    def connection(self, value: Optional[duckdb.DuckDBPyConnection]) -> None:
        """设置当前 DuckDB 连接对象。

        该 setter 主要用于内部流程或高级用户手动替换 Atlas 当前连接。
        一般情况下不建议直接修改 ``atlas.connection``，应优先使用
        ``atlas.connect(...)`` 和 ``atlas.close()`` 管理连接。

        Parameters
        ----------
        value
            需要保存到当前 Atlas 对象中的 DuckDB 连接对象；也可以为 ``None``，
            表示清空当前连接。

        Returns
        -------
        None
            该属性 setter 只更新内部连接引用，不返回对象。
        """
        self.__connection = value

    @property
    def db_memory_limit(self) -> str | int | None:
        """返回当前 Atlas 对象设置的 DuckDB 内存上限。"""

        return self.__db_memory_limit


    def _apply_memory_limit(self) -> None:
        """应用 DuckDB 内存限制。

        ``db_memory_limit`` 只限制 DuckDB 查询和中间计算可使用的内存，
        不限制 Python、NumPy 或 pandas 本身占用的内存。
        如果 ``db_memory_limit`` 是整数，则按 GB 解释，
        例如 ``4`` 会被转换为``"4GB"``。

        Returns
        -------
        None
            该方法直接作用于当前 DuckDB 连接。

        Examples
        --------
        初始化 Atlas 时限制 DuckDB 查询内存::

            atlas = sap.Atlas(r"F:\\data\\pbmc", db_memory_limit="4GB")

        使用整数设置 GB 单位的内存限制::

            atlas = sap.Atlas(r"F:\\data\\pbmc", db_memory_limit=4)

        连接已经存在的数据库时也会自动应用该限制::

            atlas = sap.Atlas(r"F:\\data\\pbmc.sasql", db_memory_limit="1024MB")
        """

        if self.__connection is None:
            return

        if self.db_memory_limit is None:
            return

        if isinstance(self.db_memory_limit, int):
            db_memory_limit = f"{self.db_memory_limit}GB"
        else:
            db_memory_limit = str(self.db_memory_limit).strip()

        if db_memory_limit == "":
            return

        memory_limit_sql = db_memory_limit.replace("'", "''")

        self.__connection.execute(
            f"SET memory_limit = '{memory_limit_sql}'"
        )

        logger.info(f"DuckDB db_memory_limit 设置为: {db_memory_limit}")


    def _create(self) -> duckdb.DuckDBPyConnection:
        """创建 Atlas 数据库文件并返回连接。

        该方法在 ``self.file_path`` 指向的位置创建新的 ``.sasql`` 数据库文件。
        连接创建完成后，会立即调用 ``self._apply_memory_limit()``，确保
        初始化时传入的 ``db_memory_limit`` 对新连接生效。

        Returns
        -------
        duckdb.DuckDBPyConnection
            新创建的 DuckDB 连接对象。
        """

        db_dir = os.path.dirname(self.file_path)

        logger.debug(f"开始创建数据库: {self.file_path}")

        if os.path.exists(self.file_path):
            raise RuntimeError(f"数据库已存在: {self.file_path}")

        try:
            logger.debug(f"创建目录: {db_dir}")
            os.makedirs(db_dir, exist_ok=True)

            logger.debug("连接 DuckDB 数据库")
            con = duckdb.connect(database=self.file_path)

            self.__connection = con
            self._apply_memory_limit()

            logger.debug(f"数据库已成功创建: {self.file_path}")
            return con

        except Exception as e:
            logger.exception("创建数据库异常详情:")
            raise RuntimeError(f"创建数据库失败: {str(e)}")


    def connect(self, mode: Literal["r+", "r"] = "r+") -> duckdb.DuckDBPyConnection:
        """连接 Atlas 数据库。

        根据当前 ``file_path`` 建立 DuckDB 连接，并把连接对象保存到 ``atlas.connection``。已有连接会被复用或替换为新的连接。
        如果初始化 Atlas 时设置了 ``db_memory_limit``，每次重新建立连接后都会自动应用该限制。

        Parameters
        ----------
        mode
            数据库连接模式。``"r+"`` 表示可读写连接，``"r"`` 表示只读连接。

        Returns
        -------
        duckdb.DuckDBPyConnection
            当前 Atlas 数据库连接。

        Examples
        --------
        以默认可读写模式连接数据库::

            atlas.connect()

        以只读模式打开数据库，适合检查已有结果::

            atlas.connect(mode="r")
            atlas.head("obs")"""

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
                db_dir = os.path.dirname(self.file_path)
                os.makedirs(db_dir, exist_ok=True)

                # 无论文件是否存在，都会创建或连接
                self.__connection = duckdb.connect(database=self.file_path, read_only=False)

                if os.path.exists(self.file_path):
                    logger.info(f"以读写模式连接现有数据库: {self.file_path}")
                else:
                    logger.info(f"创建并连接新数据库: {self.file_path}")

            else:
                logger.error(f"不支持的连接模式: {mode}")
                raise ValueError(f"不支持的连接模式: {mode}")

            self.__mode = mode
            self._apply_memory_limit()
            logger.debug("数据库连接成功")
            return self.__connection

        except Exception as e:
            logger.exception("连接数据库异常详情:")
            raise RuntimeError(f"连接数据库失败: {str(e)}")


    def close(self):
        """关闭当前数据库连接。

        关闭 ``atlas.connection`` 并释放 DuckDB 连接资源。关闭后如需继续使用数据库，可再次调用 ``atlas.connect()``。

        Returns
        -------
        None
            结果直接写入 Atlas 数据库或当前图形窗口。

        Examples
        --------
        完成分析后关闭连接::

            atlas = sap.Atlas(r"F:\\data\\pbmc")
            atlas.describe()
            atlas.close()"""

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
            logger.exception("关闭数据库连接异常详情:")
            raise RuntimeError(f"关闭数据库连接时出错: {str(e)}")


    def execute_sql(self, sql: str) -> DuckDBPyConnection | None:
        """执行一条 SQL 语句。

        该方法适合执行建表、更新、删除临时表等 SQL 操作。若需要把查询结果直接转为 DataFrame，优先使用 ``atlas.query``。

        Parameters
        ----------
        sql
            需要执行的 SQL 语句。适合用于建表、更新字段、删除临时表等不一定需要返回结果的操作。

        Returns
        -------
        duckdb.DuckDBPyConnection 或 None
            DuckDB 执行结果对象；执行失败时会抛出异常。

        Examples
        --------
        新增一个布尔过滤列::

            atlas.execute_sql(
                "ALTER TABLE obs ADD COLUMN IF NOT EXISTS filter_custom BOOLEAN"
            )

        将已有列的空值填为 ``False``::

            atlas.execute_sql(
                "UPDATE obs SET filter_custom = FALSE WHERE filter_custom IS NULL"
            )"""

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
        """检查 Atlas 数据库文件是否存在。

        该方法只检查 ``atlas.file_path`` 指向的文件是否存在，不验证数据库内部表结构是否完整。

        -------
        bool
            如果 ``atlas.file_path`` 指向的数据库文件存在，则返回 ``True``；
            否则返回 ``False``。

        Examples
        --------
        判断数据库文件是否已经创建::

            atlas = sap.Atlas(r"F:\\data\\pbmc")
            atlas.exists()"""

        exists = os.path.exists(self.file_path)
        logger.debug(f"检查数据库文件是否存在: {self.file_path} -> {exists}")
        return exists


    def query(self, query: str):
        """执行 SQL 查询并返回 DataFrame。

        该方法通过当前 DuckDB 连接执行查询，并将结果转为 ``pandas.DataFrame``，适合交互式检查 ``obs``、``var`` 和结果表。

        Parameters
        ----------
        query
            需要执行并返回结果的 SQL 查询语句。

        Returns
        -------
        pandas.DataFrame
            包含查询、统计或绘图所需数据的表格。

        Examples
        --------
        查看细胞数量::

            atlas.query("SELECT COUNT(*) AS n_cells FROM obs")

        按聚类统计细胞数::

            atlas.query(
                "SELECT kmeans, COUNT(*) AS n_cells FROM obs GROUP BY kmeans ORDER BY kmeans"
            )"""

        logger.info("查询数据库，返回值类型为pandas")

        if self.__connection is None:
            self.connect("r+")
        result = self.connection.execute(query)
        df = result.df() # 将查询结果转换为pandas DataFrame
        return df


    def query_raw(self, query: str):
        """执行 SQL 查询并返回 DuckDB 原始结果。

        与 ``atlas.query`` 不同，该方法保留 DuckDB 的原始返回对象，适合继续调用 ``fetchone``、``fetchall`` 或 DuckDB 原生方法。

        Parameters
        ----------
        query
            需要执行并返回结果的 SQL 查询语句。

        Returns
        -------
        duckdb.DuckDBPyConnection 或 None
            DuckDB 执行结果对象；执行失败时会抛出异常。

        Examples
        --------
        读取单个统计值::

            result = atlas.query_raw("SELECT COUNT(*) FROM obs")
            n_cells = result.fetchone()[0]"""

        logger.info("查询数据库，返回值类型为duckDB")

        if self.__connection is None:
            self.connect("r+")

        result = self.connection.execute(query)
        return result


    def describe(self) -> str:
        """汇总 Atlas 数据库中的表结构。

        该方法扫描数据库表和部分关键字段，生成可读的数据库摘要，适合在导入数据或完成分析后检查当前 Atlas 对象包含哪些内容。

        Returns
        -------
        str
            数据库或对象的文本摘要。

        Examples
        --------
        打印数据库摘要::

            print(atlas.describe())

        在 notebook 中直接查看摘要::

            atlas.describe()"""

        if self.__connection is None:
            self.connect("r+")

        conn = self.__connection

        # 1. 数据库路径
        file_name = self.file_path

        # 1.1 DuckDB 内存限制
        db_memory_limit = self.db_memory_limit

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
            """将计数值格式化为带千位分隔符的字符串。"""
            return "NA" if x is None else f"{int(x):,}"

        text = (
            f"file_name       : {file_name}\n"
            f"db_memory_limit : {db_memory_limit}\n"
            f"tables          : {len(tables)}\n"
            f"table names     : {table_names}\n"
            f"n_cells         : {fmt(n_cells)}\n"
            f"n_genes         : {fmt(n_genes)}"
        )

        return text


    def __repr__(self) -> str:
        """返回 Atlas 对象的数据库摘要字符串。

        该方法调用 ``self.describe()``，用于在交互式环境中显示
        当前数据库路径、表数量、细胞数和基因数等信息。
        """
        return self.describe()


    def __str__(self) -> str:
        """返回 Atlas 对象的可读字符串摘要。

        该方法调用 ``self.describe()``，使 ``print(atlas)`` 可以直接显示
        当前数据库的基本信息。
        """
        return self.describe()


    def head(self, table_name: str, n: int = 5):
        """打印数据库表的前几行。

        该方法查询指定表的前 ``n`` 行，并在控制台打印表名、列名和数据内容，
        适合快速检查导入结果或分析结果。

        Parameters
        ----------
        table_name
            数据库表名。
        n
            打印的记录数量。

        Returns
        -------
        None
            该方法只打印结果，不返回 DataFrame。

        Examples
        --------
        查看 ``obs`` 前 5 行::

            atlas.head("obs")

        查看差异基因结果前 10 行::

            atlas.head("rank_genes_groups", n=10)
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

        with pd.option_context(
                "display.max_columns", None,
                "display.max_rows", int(n),
                "display.width", 0,
                "display.max_colwidth", None,
        ):
            print(df.to_string(index=True))


    def _save_read_index_meta(
            self,
            *,
            cell_condition: str | None,
            gene_condition: str | None,
            use_hvg: bool,
            use_data: str,
    ) -> None:
        """保存当前 read index 构建参数。

        该方法把本次 ``build_read_index`` 使用的细胞过滤条件、基因过滤条件、
        是否使用 HVG 以及表达值列名保存到 ``atlas_read_index_meta`` 表中，
        用于后续检查当前读取索引对应的数据范围和表达层。

        如果 ``atlas_read_index_meta`` 表不存在，则自动创建该表；
        如果该表已经存在，则不会重复创建，并打印提示信息。

        Parameters
        ----------
        cell_condition
            本次构建读取索引时使用的细胞过滤列名。
        gene_condition
            本次构建读取索引时使用的基因过滤列名。
        use_hvg
            是否叠加使用 ``highly_variable_genes`` 过滤。
        use_data
            本次构建过滤表达矩阵时读取的表达值列名。

        Returns
        -------
        None
            结果直接写入 ``atlas_read_index_meta`` 表。
        """

        if self.__connection is None:
            self.connect("r+")

        conn = self.__connection

        # 先检查 atlas_read_index_meta 表是否已经存在
        table_exists = conn.execute("""
            SELECT COUNT(*)
            FROM information_schema.tables
            WHERE table_name = 'atlas_read_index_meta'
        """).fetchone()[0]

        # 不存在才新建；
        # 存在则不重复建表，并打印提示
        if table_exists == 0:
            conn.execute("""
                CREATE TABLE atlas_read_index_meta (
                    key VARCHAR PRIMARY KEY,
                    value VARCHAR
                )
            """)
        else:
            print("minibatch读取 表达矩阵所需的索引表 已经重新构建，依赖该表的PCA、K-means等操作，请重新运行！")

        rows = [
            ("cell_condition", str(cell_condition)),
            ("gene_condition", str(gene_condition)),
            ("use_hvg", str(bool(use_hvg))),
            ("use_data", str(use_data)),
        ]

        for key, value in rows:
            conn.execute("""
                INSERT OR REPLACE INTO atlas_read_index_meta(key, value)
                VALUES (?, ?)
            """, [key, value])

        conn.commit()


    def build_read_index(
            self,
            cell_condition: str  = "filter_cells",
            gene_condition: str  = "filter_genes",
            use_hvg: bool = True,
            use_data: str = "data_log1p",
    ):
        """构建表达矩阵读取索引。

        该方法根据细胞过滤条件、基因过滤条件和高变基因标记，构建后续小批量读取表达矩阵所需的索引表。PCA、K-means 和部分绘图函数通常依赖该索引。

        Parameters
        ----------
        cell_condition
            ``obs`` 表中用于筛选细胞的布尔列名。默认 ``"filter_cells"``。
            为 ``None`` 时不进行细胞过滤。
        gene_condition
            ``var`` 表中用于筛选基因的布尔列名。默认 ``"filter_genes"``。
            为 ``None`` 时不进行基因过滤。
        use_hvg
            是否在 ``gene_condition`` 之外继续叠加 ``highly_variable_genes=TRUE``。
            为 ``True`` 时，最终基因集合需要同时满足基因过滤条件和 HVG 条件。
        use_data
            从 ``X_HyS_data`` 表中读取的表达值列名。常用值包括
            ``"data_count"``、``"data_normalize"``、``"data_log1p"`` 和 ``"data_scale"``。

        Returns
        -------
        None
            结果直接写入 Atlas 数据库或当前图形窗口。

        Examples
        --------
        使用默认过滤列构建索引::

            sap.pp.filter_cells(atlas, min_genes=200)
            sap.pp.filter_genes(atlas, min_cells=3)
            atlas.build_read_index(cell_condition="filter_cells", gene_condition="filter_genes")

        只使用高变基因构建 PCA 输入索引::

            sap.pp.highly_variable_genes(atlas, n_top_genes=3000)
            atlas.build_read_index(use_hvg=True)"""

        builder = FilterIndexBuilder(
            self.file_path,
            cell_condition=cell_condition,
            gene_condition=gene_condition,
            use_hvg=use_hvg,
            use_data=use_data,
        )
        builder.run()

        # 把本次读取索引用到的参数持久化到数据库
        self._save_read_index_meta(
            cell_condition=cell_condition,
            gene_condition=gene_condition,
            use_hvg=use_hvg,
            use_data=use_data,
        )


    def get_minibatch_csr(self, x_type: str = "CSR"):
        """以稀疏 CSR 小批量读取表达矩阵。

        该方法基于已经构建的读取索引，从数据库中按批返回稀疏矩阵，适合需要流式处理表达矩阵的算法。

        Parameters
        ----------
        x_type
            返回的小批量矩阵格式。常用值为 ``"CSR"`` 或其他函数支持的稀疏矩阵格式。

        Yields
        -------
        ``X_batch`` 为 ``float32`` csr 格式的稀疏矩阵格式；

        Examples
        --------
        遍历 CSR 小批量::

            for X_batch in atlas.get_minibatch_csr():
                print(X_batch.shape)
                break"""

        fetcher = MultiThreadedMinibatchFetcher(file_path = self.file_path, x_type = x_type)
        for X_batch in fetcher.run():
            yield X_batch


    def get_minibatch_dense(
            self,
            pass_mode: str = "single-pass",
            batch_size: int = 2048,
            max_batches: int | None = None,
            buffer_batch_num: int = 5,
            get_obs_col: str | None = None,
    ):
        """以 dense 小批量读取表达矩阵。

        该方法基于 ``atlas.build_read_index(...)`` 生成的过滤后读取索引，
        从 ``X_HyS_data_filtered`` / ``X_HyS_indptr_filtered`` 中按批恢复表达矩阵，
        并把稀疏表达记录转换为 ``float32`` dense array。
        它主要服务于``IncrementalPCA``、``MiniBatchKMeans``、流式训练和大数据分批推理等
        需要 dense 输入的算法。

        返回值是一个生成器，会逐批 ``yield`` minibatch，而不是一次性把全部数据读入内存。
        默认情况下，每次只返回一个 ``X_batch`` 矩阵；
        当传入 ``get_obs_col`` 时，会额外返回当前 batch 每一行对应的 ``filter_cell_ids``，
        并根据这些``filter_cell_ids`` 从 ``obs`` 表中取出指定列，例如 ``kmeans`` 标签。

        Parameters
        ----------
        pass_mode
            小批量读取模式。可选值为 ``"single-pass"`` 或 ``"multi-pass"``。

            - ``"single-pass"``：按当前读取索引顺序遍历一次数据库，适合评估、
              导出、预测或需要确定性顺序的流程。
            - ``"multi-pass"``：每一轮重新创建读取器，并在 dense batch 层使用
              ``ShuffleBuffer`` 做随机化输出，适合需要多轮随机小批量训练的算法。
              该模式通常应配合 ``max_batches`` 使用，用于控制总训练批次数。
        batch_size
            每个 minibatch 包含的细胞数量。较大值通常可以减少 Python 层循环次数、
            提高吞吐，但会增加单批 dense 矩阵的内存占用。输出矩阵形状通常为
            ``(当前批细胞数, 过滤后基因数)``；最后一个 batch 的细胞数可能小于``batch_size``。
        max_batches
            最多读取或输出的 minibatch 数量。为 ``None`` 时：
            - 在 ``single-pass`` 模式下遍历当前读取索引中的全部 batch；
            - 在 ``multi-pass`` 模式下不主动限制轮数，通常建议显式传入该参数。
        buffer_batch_num
            ``multi-pass`` 模式下 ``ShuffleBuffer`` 缓存的 batch 数量。
            实际 shuffle buffer 的细胞容量约为 ``batch_size * buffer_batch_num``。
            值越大，随机化范围越大，但 dense 缓冲区占用的内存也越高。
            在 ``single-pass`` 模式下该参数会传到底层读取器，但不会改变顺序输出语义。
        get_obs_col
            需要随 minibatch 一起返回的 ``obs`` 表字段名，例如 ``"kmeans"``。
            默认为 ``None``，不查询 ``obs`` 表字段，且每次只 ``yield X_batch``。

            当传入字段名时，例如 ``get_obs_col="kmeans"``，该方法会自动让底层
            minibatch 读取器返回 ``filter_cell_ids``，再根据这些 ID 查询``obs.kmeans``，并返回字典：

            ``{"X": X_batch, "filter_cell_ids": filter_cell_ids, "kmeans": values}``

            其中 ``X_batch[i, :]``、``filter_cell_ids[i]`` 和 ``values[i]``三者一一对应。

        Yields
        -------
        numpy.ndarray 或 dict
            当 ``get_obs_col is None`` 时，每次生成一个 dense ``numpy.ndarray``：

            ``X_batch``

            当 ``get_obs_col`` 不为 ``None`` 时，每次生成一个字典：

            ``{"X": X_batch, "filter_cell_ids": filter_cell_ids, get_obs_col: values}``

            ``X_batch`` 为 ``float32`` dense 矩阵；
            ``filter_cell_ids`` 为当前 batch每行对应的过滤后细胞 ID；
            ``values`` 为 ``obs`` 表中指定列的值。

        Examples
        --------
        顺序读取，用于模型拟合::

            for X_batch in atlas.get_minibatch_dense(batch_size=4096):
                print(X_batch.shape)
                break

        同时返回 ``obs.kmeans`` 标签::

            for batch in atlas.get_minibatch_dense(
                batch_size=4096,
                get_obs_col="kmeans",
            ):
                X_batch = batch["X"]
                labels = batch["kmeans"]
                filter_cell_ids = batch["filter_cell_ids"]
                break

        多轮随机 batch 训练，并限制总批次数::

            for X_batch in atlas.get_minibatch_dense(
                pass_mode="multi-pass",
                batch_size=2048,
                buffer_batch_num=5,
                max_batches=100,
            ):
                model.partial_fit(X_batch)
        """

        if pass_mode not in ("single-pass", "multi-pass"):
            raise ValueError("pass_mode 只支持 'single-pass' 或 'multi-pass'")

        if get_obs_col is not None and not isinstance(get_obs_col, str):
            raise TypeError("get_obs_col 必须是 str 或 None")

        if get_obs_col is not None and get_obs_col == "":
            raise ValueError("get_obs_col 不能为空字符串")

        # 内部参数：get_obs_col 依赖 filter_cell_ids 做映射，因此传入 obs 字段时自动打开。
        return_cell_ids = get_obs_col is not None

        obs_conn = None
        obs_col_sql = None


        def _quote_identifier(name: str) -> str:
            return '"' + name.replace('"', '""') + '"'


        def _attach_obs_col(batch):

            if get_obs_col is None:
                return batch

            filter_cell_ids = np.asarray(batch["filter_cell_ids"], dtype=np.int64)

            ids_df = pd.DataFrame({
                "row_order": np.arange(len(filter_cell_ids), dtype=np.int64),
                "filter_cell_id": filter_cell_ids,
            })

            obs_conn.register("_minibatch_obs_ids", ids_df)
            try:
                obs_values = obs_conn.execute(f"""
                    SELECT ids.row_order, obs.{obs_col_sql} AS obs_value
                    FROM _minibatch_obs_ids AS ids
                    LEFT JOIN obs
                      ON obs.filter_cell_id = ids.filter_cell_id
                    ORDER BY ids.row_order
                """).fetchnumpy()["obs_value"]
            finally:
                obs_conn.unregister("_minibatch_obs_ids")

            batch[get_obs_col] = obs_values
            return batch


        if get_obs_col is not None:
            obs_conn = duckdb.connect(self.file_path)
            obs_columns = obs_conn.execute("PRAGMA table_info('obs')").fetchdf()["name"].tolist()

            if "filter_cell_id" not in obs_columns:
                obs_conn.close()
                raise ValueError("obs 表缺少 filter_cell_id 字段，请先运行 atlas.build_read_index(...)")

            if get_obs_col not in obs_columns:
                obs_conn.close()
                raise ValueError(f"obs 表不存在字段: {get_obs_col}")

            if get_obs_col in {"X", "filter_cell_ids"}:
                obs_conn.close()
                raise ValueError("get_obs_col 不能是 X 或 filter_cell_ids，避免覆盖 minibatch 保留字段")

            obs_col_sql = _quote_identifier(get_obs_col)

        # 1. single-pass：只跑一遍
        try:
            if pass_mode == "single-pass":

                fetcher = MultiThreadedMinibatchFetcher(
                    file_path=self.file_path,
                    batch_size=batch_size,
                    x_type="dense",
                    pass_mode="single-pass",
                    buffer_batch_num=buffer_batch_num,
                    max_batches=max_batches,
                    return_cell_ids=return_cell_ids,
                )

                for batch in fetcher.run():

                    yield _attach_obs_col(batch)

                return

            # 2. multi-pass：自动循环多遍
            produced_batches = 0
            pass_id = 0

            while True:

                # 如果达到 max_batches，停止
                if max_batches is not None and produced_batches >= max_batches:
                    logger.info(f"[get_minibatch_dense] reach max_batches={max_batches}, stop")
                    break

                # 当前 pass 还需要输出多少 batch
                if max_batches is None:
                    remain_batches = None
                else:
                    remain_batches = max_batches - produced_batches

                logger.info(
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
                    return_cell_ids=return_cell_ids,
                )

                pass_batches = 0

                for batch in fetcher.run():
                    produced_batches += 1
                    pass_batches += 1

                    yield _attach_obs_col(batch)

                    if max_batches is not None and produced_batches >= max_batches:
                        break

                pass_id += 1

                # 防止异常情况下空 pass 无限循环
                if pass_batches == 0:
                    logger.debug("[get_minibatch_dense] pass produced 0 batch, stop")
                    break
        finally:
            if obs_conn is not None:
                obs_conn.close()


    # =====================================================
    # io 方法包装
    # -----------------------------------------------------
    # 保留原函数位置不动：
    #   load_h5ad(file_path, atlas, ...)
    #
    # 同时支持对象式调用：
    #   atlas.load_h5ad(file_path, ...)
    # =====================================================

    def load_h5ad(
        self,
        h5ad_path: PathLike[str] | str | list[PathLike[str] | str],
        *,
        load_type: Literal["order", "random"] = "random",
        cells_per_block: int | None = None,
    ) -> Any:
        """将 h5ad 文件导入 Atlas 数据库。

        该函数是 h5ad 导入 Atlas 的统一入口。它可以读取单个 ``.h5ad`` 文件，也可以
        读取多个 ``.h5ad`` 文件组成的列表，并把细胞元数据、基因元数据和表达矩阵
        写入 Atlas 的 DuckDB 数据库。

        导入时会根据 ``load_type`` 和 ``h5ad_path`` 的类型自动分派到对应实现：

        - 单文件 + ``"order"``：按原始细胞顺序导入；
        - 单文件 + ``"random"``：按 shuffle-window 方式随机导入；
        - 多文件 + ``"order"``：按文件列表顺序和文件内细胞顺序导入；
        - 多文件 + ``"random"``：把多个文件切成 block 后全局随机导入。

        表达矩阵会统一保存为 count 尺度，写入 ``X_HyS_data.data_count`` 字段。
        如果输入 ``X`` 被检测为 log 尺度，会在写入前转换回 count。

        Parameters
        ----------
        h5ad_path
            输入 ``.h5ad`` 文件路径，或多个 ``.h5ad`` 文件路径组成的列表。
        atlas
            Atlas 对象。函数会通过 ``atlas.connect("r+")`` 获取 DuckDB 连接，并把
            数据写入该 Atlas 数据库。
        load_type
            导入方式，只支持 ``"order"`` 和 ``"random"``。
            当 ``h5ad_path`` 是单个路径时，分别执行单文件顺序或随机导入；
            当 ``h5ad_path`` 是列表时，分别执行多文件顺序或随机导入。
        cells_per_block
            读取和写入表达矩阵时每个连续 cell block 包含的细胞数量。
            为 ``None`` 时会根据细胞总数自动估算一个默认值。

        Returns
        -------
        Any
            返回所调用底层导入函数的结果。当前主要用于执行导入副作用，通常不依赖
            返回值。

        Notes
        -----
        ``random`` 导入会重排细胞顺序，因此单文件随机导入默认不会导入 ``obsm``，
        以避免 embedding 与重排后的 ``obs`` 错位；
        ``order`` 导入会保留原始顺序，因此可以安全导入 ``obsm`` 和 ``varm``。

        Examples
        --------
        顺序导入单个 h5ad 文件::

            atlas = sap.Atlas(r"F:\\data\\pbmc")
            atlas.load_h5ad(r"F:\\data\\pbmc.h5ad", load_type="order")

        随机分块导入多个文件::

            atlas.load_h5ad(
                [r"F:\\data\\batch1.h5ad", r"F:\\data\\batch2.h5ad"],
                load_type="random",
                cells_per_block=1000,
            )
        """

        return _io_load_h5ad(
            h5ad_path,
            self,
            load_type=load_type,
            cells_per_block=cells_per_block,
        )


    def load_anndata(self, adata: AnnData) -> None:

        """将 AnnData 对象写入 Atlas 数据库。

        该函数直接接收内存中的 AnnData 对象，并把其中的 ``obs``、``var``、
        ``X``、``obsm`` 和 ``varm`` 写入 Atlas 数据库。适合已经用 Scanpy 或其他
        工具完成读取、筛选或预处理后，再转入 Atlas 管理的场景。

        与 ``load_h5ad`` 的 backed 分块导入不同，该函数要求 AnnData 已经在内存中，
        因此更适合中小型数据或已经抽样后的数据。

        Parameters
        ----------
        adata
            AnnData 对象。函数会把其中的 ``obs``、``var``、表达矩阵和可支持的结果写入 Atlas 数据库。
        atlas
            Atlas 对象。要求对象已经连接或可连接到 DuckDB 数据库。

        Returns
        -------
        None
            结果直接写入 Atlas 数据库，不返回对象。

        Notes
        -----
        该路径会重建 ``obs``、``var``、``X_HyS_indptr`` 和 ``X_HyS_data`` 表，并把
        ``obsm``、``varm`` 中的二维数组写成 ``obsm_*``、``varm_*`` 表。

        Examples
        --------
        从 Scanpy 读取并导入::

            adata = sc.read_h5ad(r"F:\\data\\pbmc.h5ad")
            atlas = sap.Atlas(r"F:\\data\\pbmc")
            atlas.load_anndata(adata)
        """

        return _io_load_anndata(adata, self)


    def load_multi_format(self, file_path: PathLike[str] | str) -> None:

        """根据文件格式导入数据到 Atlas。

        该函数是小型或通用格式数据的导入入口。它会先根据 ``file_path`` 的后缀
        调用 ``_read_smart`` 读取为内存 AnnData，然后调用 ``load_anndata`` 写入
        Atlas 数据库。

        与 ``load_h5ad`` 的 backed 分块导入不同，该函数会先把数据完整读入内存，
        因此更适合小文件、临时转换或非 h5ad 格式数据。

        Parameters
        ----------
        file_path
            输入文件路径。函数会根据文件格式选择合适的读取方式。
        atlas
            Atlas 对象。要求对象已经连接或可连接到 DuckDB 数据库。

        Returns
        -------
        None
            结果直接写入 Atlas 数据库，不返回对象。

        Notes
        -----
        支持的读取格式由 ``_read_smart`` 决定，包括 h5ad、loom、Matrix Market、
        csv、txt/tsv、Excel、10x h5 和 UMI-tools 等常见格式。

        Examples
        --------
        自动识别并导入文件::

            atlas = sap.Atlas(r"F:\\data\\pbmc")
            atlas.load_multi_format(r"F:\\data\\pbmc.h5ad")
        """

        return _io_load_multi_format(file_path, self)


    def rename_duplicated_genes(
        self,
        gene_name_column: str = "atlas_gene_name",
    ) -> bool | None:

        """检查 Atlas 数据库中的基因名是否重复。

        该函数读取 ``var`` 表中的基因名称列，判断是否存在重复基因名，并为重复项
        添加 ``_1``、``_2`` 等后缀。重复基因名可能影响按名称绘图、差异基因展示
        和 AnnData 导出，因此导入后可以运行该函数进行清洗。

        对每个重复基因名，第一次出现的名称保持不变，后续重复项按
        ``原名_1``、``原名_2`` 的形式重命名。

        Parameters
        ----------
        atlas
            Atlas 对象。要求对象已经连接到 DuckDB 数据库，并且数据库中存在``var`` 表。
        gene_name_column
            保存基因名称的 ``var`` 列名。通常为 ``"atlas_gene_name"``。

        Returns
        -------
        bool
            成功检查或更新时返回 ``True``；如果 ``var`` 表不存在，返回 ``False``。

        Notes
        -----
        只有检测到重复基因名时才会更新 ``var`` 表；没有重复时保持原表不变。

        Examples
        --------
        检查默认基因名称列::

            atlas.rename_duplicated_genes()
        """

        return _io_rename_duplicated_genes(self, gene_name_column=gene_name_column)


    def write_h5ad(
        self,
        out_h5ad_path: PathLike[str] | str,
        *,
        batch_cells: int = 1_000_000,
        use_data: str = "data_count",
    ) -> None:

        """将 Atlas 数据库导出为 h5ad 文件。

        该函数从 Atlas 的 DuckDB 数据库读取 ``obs``、``var``、稀疏表达矩阵、
        ``obsm_*`` 和 ``varm_*`` 结果表，并写出为标准 AnnData ``.h5ad`` 文件，
        方便继续在 Scanpy 或其他支持 AnnData 的工具中分析。

        表达矩阵会按照 Atlas 内部的 HyS 稀疏结构重新组装为 h5ad 中的 CSR
        ``X``。其中 ``X.data`` 来自 ``X_HyS_data`` 表中由 ``use_data`` 指定的
        字段，``X.indices`` 来自 ``atlas_gene_id``，``X.indptr`` 来自
        ``X_HyS_indptr``。

        Parameters
        ----------
        atlas
            Atlas 对象。要求对象已经连接到 DuckDB 数据库，并且数据库中至少包含
            ``obs``、``var``、``X_HyS_indptr`` 和 ``X_HyS_data`` 表。
        out_h5ad_path
            输出 ``.h5ad`` 文件路径。
        batch_cells
            分批写出表达矩阵 ``data`` 和 ``indices`` 时每批处理的非零表达记录数量。
            较大的值通常更快，但会增加单批内存占用。
        use_data
            从 ``X_HyS_data`` 表中导出的表达值字段名，默认使用 ``"data_count"``。
            也可以传入 ``"data_log1p"``、``"data_normalize"`` 等已经存在的字段。

        Returns
        -------
        None
            结果直接写入 ``out_h5ad_path`` 指定的 h5ad 文件，不返回对象。

        Notes
        -----
        ``obsm_*`` 表会导出到 h5ad 的 ``obsm``，表名中的 ``obsm_`` 前缀会被去掉；
        ``varm_*`` 表会导出到 h5ad 的 ``varm``，表名中的 ``varm_`` 前缀会被去掉。

        导出前会检查 ``use_data`` 是否存在于 ``X_HyS_data`` 表中；不存在时会直接
        抛出中文错误，避免导出空矩阵或错误字段。

        Examples
        --------
        导出当前数据库::

            atlas.write_h5ad(r"F:\\data\\pbmc_export.h5ad")

        使用对象式 API 并降低单批内存占用::

            atlas.write_h5ad(r"F:\\data\\pbmc_export.h5ad", batch_cells=200000)

        导出 log1p 表达矩阵::

            atlas.write_h5ad(r"F:\\data\\pbmc_log1p.h5ad", use_data="data_log1p")"""

        return _io_write_h5ad(
            self,
            out_h5ad_path,
            batch_cells=batch_cells,
            use_data=use_data,
        )


    def get_obs_df(
        self,
        columns: list[str] | str | None = None,
    ) -> pd.DataFrame:

        """读取 Atlas 数据库中的 obs 表。

        该函数把 ``obs`` 表中的全部列或指定列读取为 pandas DataFrame，适合快速
        检查细胞元数据、导出统计结果或与外部分析结果合并。
        返回结果会以``atlas_cell_id`` 作为 pandas index，同时保留 ``atlas_cell_id`` 列本身。

        Parameters
        ----------
        atlas
            Atlas 对象。要求对象已经连接到 DuckDB 数据库，并且数据库中存在
            ``obs`` 表。
        columns
            需要从 ``obs`` 中读取的列名。可以是单个字符串、字符串列表或 ``None``。
            为 ``None`` 时读取全部列。

        Returns
        -------
        pandas.DataFrame
            ``obs`` 的查询结果。默认 index 为 ``atlas_cell_id``。

        Notes
        -----
        即使 ``columns`` 中没有显式包含 ``atlas_cell_id``，函数也会自动把
        ``atlas_cell_id`` 放在第一列，用于设置 DataFrame index。

        Examples
        --------
        读取全部 obs 信息::

            obs = atlas.get_obs_df()

        只读取聚类和自动注释列::

            obs = atlas.get_obs_df(columns=["kmeans", "cell_type_auto"])"""

        return _io_get_obs_df(self, columns=columns)


    def get_anndata(
        self,
        atlas_cell_ids: list[int] | np.ndarray | None,
        use_data: str = "data_count",
        include_obsm: bool = True,
        include_varm: bool = True,
    ) -> AnnData:

        """从 Atlas 数据库构建 AnnData 对象。

        该函数根据用户提供的 ``atlas_cell_ids`` 从 Atlas 数据库中导出一个内存
        AnnData 对象。函数会保留输入细胞 ID 的顺序，读取对应的 ``obs`` 子集、
        全量 ``var``、指定表达字段组成的稀疏 CSR ``X``，并可选读取 ``obsm_*`` 和
        ``varm_*`` 结果表。

        该函数适合小规模抽样导出、局部 Scanpy 分析、模型检查或把 Atlas 中的一组
        细胞临时转换回 AnnData。

        Parameters
        ----------
        atlas
            Atlas 对象。要求对象已经连接到 DuckDB 数据库，并且数据库中至少包含
            ``obs``、``var`` 和 ``X_HyS_data`` 表。
        atlas_cell_ids
            需要导出的 Atlas 细胞 ID 列表。不能为空，且不能包含重复值。

            返回 AnnData 中细胞的顺序会与该列表顺序一致。
        use_data
            从 ``X_HyS_data`` 表读取的表达字段名。常用值包括 ``"data_count"``、
            ``"data_normalize"``、``"data_log1p"`` 和 ``"data_scale"``。
        include_obsm
            是否把 ``obsm_*`` 结果表写入返回的 AnnData。
        include_varm
            是否把 ``varm_*`` 结果表写入返回的 AnnData。

        Returns
        -------
        AnnData
            从 Atlas 数据库构建的 AnnData 对象。

        Notes
        -----
        ``obsm_*`` 表会按所选细胞顺序左连接导出；某些细胞没有 embedding 时，对应
        位置会写入 ``NaN``。``varm_*`` 表按全量基因顺序导出。

        该函数会创建临时表 ``_selected_cells``，用于保留用户传入的细胞顺序。

        Examples
        --------
        导出指定细胞::

            cell_ids = [0, 1, 2, 3]
            adata = atlas.get_anndata(cell_ids, use_data="data_log1p")

        导出过滤后的前 5000 个细胞并包含 UMAP/PCA::

            cell_ids = atlas.query(
                "SELECT atlas_cell_id FROM obs WHERE filter_cells = TRUE LIMIT 5000"
            )["atlas_cell_id"].tolist()
            adata = atlas.get_anndata(cell_ids, include_obsm=True, include_varm=True)
        """

        return _io_get_anndata(
            self,
            atlas_cell_ids=atlas_cell_ids,
            use_data=use_data,
            include_obsm=include_obsm,
            include_varm=include_varm,
        )

