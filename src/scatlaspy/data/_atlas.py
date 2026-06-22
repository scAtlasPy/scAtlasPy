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
    gene_names_duplicated as _io_gene_names_duplicated,
    get_anndata as _io_get_anndata,
    get_obs_df as _io_get_obs_df,
    load_anndata as _io_load_anndata,
    load_h5ad as _io_load_h5ad,
    load_multi_format as _io_load_multi_format,
    write_h5ad as _io_write_h5ad,
)
from ..io._input import StoreType

# 配置日志
logger = logging.getLogger("Atlas")
logger.addHandler(logging.NullHandler())


def set_verbosity(
        level: Literal["error", "warning", "info", "debug"] | None = None,
) -> None:
    """设置 scAtlasPy 的日志输出级别。

    该函数调整包内使用的 ``Atlas`` logger，统一控制导入、预处理、工具函数和绘图流程中的日志详细程度。它不会主动修改第三方库的日志级别。

    Parameters
    ----------
    level
        日志级别字符串。可选值包括 ``"debug"``、``"info"``、``"warning"`` 和 ``"error"``。
        传入 ``None`` 时关闭 ``Atlas`` logger，不输出 scAtlasPy 自身日志。

    Returns
    -------
    None
        该函数直接修改 ``Atlas`` logger 的输出级别，不返回对象。

    Examples
    --------
    只显示警告和错误信息::

        sap.set_verbosity("warning")

    调试导入或预处理流程::

        sap.set_verbosity("debug")
        atlas = sap.Atlas(r"F:\\data\\pbmc")"""

    atlas_logger = logging.getLogger("Atlas")

    if level is None:
        atlas_logger.disabled = True
        return

    atlas_logger.disabled = False
    level = str(level).lower()
    level_map = {
        "error": logging.ERROR,
        "warning": logging.WARNING,
        "info": logging.INFO,
        "debug": logging.DEBUG,
    }

    if level not in level_map:
        raise ValueError("level 只支持: error, warning, info, debug, None")

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

    Atlas 对象管理一个持久化的 DuckDB-backed ``.sasql`` 数据库，保存数据库路径、活动连接，并提供创建、打开、查询、检查、读取和调用 IO 函数的便捷方法。

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
            db_memory_limit: str | int | None = "32GB",
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
            DuckDB 可使用的内存上限。可以传入 DuckDB 支持的字符串，例如
            ``"4GB"``；也可以传入整数，整数会按 GB 解释，
            例如 ``4`` 等价于 ``"4GB"``。
            该参数只限制 DuckDB 查询和中间计算使用的内存，不限制
            Python、NumPy 或 pandas 本身占用的内存。为 ``None`` 时不设置限制。

        Returns
        -------
        None
            结果直接写入 Atlas 数据库或当前图形窗口。

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
        self.__db_memory_limit = db_memory_limit

        logger.info(f"开始初始化 Atlas 实例，file_name: {self.file_path}")

        # 修改：保留原来的数据库创建 / 连接逻辑
        # 注意：这里不再设置 verbosity，日志级别完全由 sap.set_verbosity(...) 全局控制
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
        如果 ``db_memory_limit`` 是整数，则按 GB 解释，例如 ``4`` 会被转换为
        ``"4GB"``。

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


    @staticmethod
    def _quote_identifier(name: str) -> str:
        """Return a safely quoted DuckDB identifier."""

        return '"' + str(name).replace('"', '""') + '"'


    def table_names(self) -> list[str]:
        """Return the names of tables stored in the Atlas database.

        This is a convenience wrapper around ``SHOW TABLES`` for tutorials,
        notebooks, and interactive inspection.
        """

        if self.__connection is None:
            self.connect("r+")

        return [r[0] for r in self.connection.execute("SHOW TABLES").fetchall()]


    def has_table(self, table_name: str) -> bool:
        """Return whether a table exists in the Atlas database."""

        return str(table_name) in set(self.table_names())


    def table_info(self, table_name: str) -> pd.DataFrame:
        """Return column metadata for an Atlas database table.

        Parameters
        ----------
        table_name
            Name of the table to inspect.

        Returns
        -------
        pandas.DataFrame
            DuckDB ``PRAGMA table_info`` output for the table.
        """

        if not self.has_table(table_name):
            available = ", ".join(self.table_names()) or "None"
            raise ValueError(
                f"Table does not exist: {table_name}. "
                f"Available tables: {available}"
            )

        table_sql = self._quote_identifier(table_name)
        return self.connection.execute(f"PRAGMA table_info({table_sql})").df()


    def has_column(self, table_name: str, column_name: str) -> bool:
        """Return whether ``table_name`` contains ``column_name``."""

        if not self.has_table(table_name):
            return False

        info = self.table_info(table_name)
        return str(column_name) in set(info["name"].astype(str))


    def read_index_info(self) -> pd.DataFrame:
        """Return the persisted ``build_read_index`` configuration.

        Returns an empty DataFrame with columns ``key`` and ``value`` when the
        read index metadata table is not present.
        """

        columns = ["key", "value"]
        if not self.has_table("atlas_read_index_meta"):
            return pd.DataFrame(columns=columns)

        return self.query("""
            SELECT key, value
            FROM atlas_read_index_meta
            ORDER BY key
        """)


    def workflow_state(self) -> pd.DataFrame:
        """Summarize common persisted workflow artifacts.

        This method provides a compact checklist for reopening an existing Atlas:
        it reports whether common tables or metadata columns produced by import,
        preprocessing, dimensionality reduction, clustering, and annotation are
        present. Presence is a useful resume hint, but it does not prove that a
        result matches the current read index or the intended parameters.
        """

        rows = [
            {
                "artifact": "imported_data",
                "present": self.has_table("obs") and self.has_table("var"),
                "evidence": "obs and var tables",
                "meaning": "Data have been imported.",
            },
            {
                "artifact": "cell_filter",
                "present": self.has_column("obs", "filter_cells"),
                "evidence": "obs.filter_cells",
                "meaning": "Cell filtering has been computed.",
            },
            {
                "artifact": "gene_filter",
                "present": self.has_column("var", "filter_genes"),
                "evidence": "var.filter_genes",
                "meaning": "Gene filtering has been computed.",
            },
            {
                "artifact": "highly_variable_genes",
                "present": self.has_column("var", "highly_variable_genes"),
                "evidence": "var.highly_variable_genes",
                "meaning": "HVG selection has been computed.",
            },
            {
                "artifact": "read_index",
                "present": self.has_table("atlas_read_index_meta"),
                "evidence": "atlas_read_index_meta table",
                "meaning": "A read index has been constructed.",
            },
            {
                "artifact": "pca",
                "present": self.has_table("obsm_X_pca") and self.has_table("varm_PCs"),
                "evidence": "obsm_X_pca and varm_PCs tables",
                "meaning": "PCA coordinates and loadings have been stored.",
            },
            {
                "artifact": "kmeans",
                "present": self.has_column("obs", "kmeans"),
                "evidence": "obs.kmeans",
                "meaning": "KMeans cluster labels have been stored.",
            },
            {
                "artifact": "umap",
                "present": self.has_table("obsm_X_umap"),
                "evidence": "obsm_X_umap table",
                "meaning": "UMAP coordinates have been stored.",
            },
            {
                "artifact": "rank_genes_groups",
                "present": self.has_table("rank_genes_groups"),
                "evidence": "rank_genes_groups table",
                "meaning": "Marker-gene ranking results have been stored.",
            },
            {
                "artifact": "manual_annotation",
                "present": (
                    self.has_table("manual_cluster_annotation")
                    or self.has_column("obs", "cell_type_manual")
                ),
                "evidence": "manual_cluster_annotation table or obs.cell_type_manual",
                "meaning": "Manual annotation results have been stored.",
            },
        ]

        return pd.DataFrame(rows)


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
            tables = self.table_names()
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


    def head(self, table_name: str, n: int = 5) -> pd.DataFrame:
        """Return the first rows of an Atlas database table.

        This method queries the first ``n`` rows of a table and returns them as
        a :class:`pandas.DataFrame`, which is convenient for interactive
        notebooks and quick inspection.

        Parameters
        ----------
        table_name
            Name of the database table to inspect.
        n
            Number of rows to return.

        Returns
        -------
        pandas.DataFrame
            The first ``n`` rows from ``table_name``.

        Examples
        --------
        Inspect the first 5 rows of ``obs``::

            atlas.head("obs")

        Inspect the first 10 rows of marker-gene results::

            atlas.head("rank_genes_groups", n=10)
        """

        if self.__connection is None:
            self.connect("r+")

        if not self.has_table(table_name):
            tables = self.table_names()
            raise ValueError(
                f"Table does not exist: {table_name}. "
                f"Available tables: {', '.join(tables) if len(tables) > 0 else 'None'}"
            )

        table_sql = self._quote_identifier(table_name)
        return self.connection.execute(f"""
            SELECT *
            FROM {table_sql}
            LIMIT {int(n)}
        """).df()

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

        conn.execute("""
            CREATE TABLE IF NOT EXISTS atlas_read_index_meta (
                key VARCHAR PRIMARY KEY,
                value VARCHAR
            )
        """)

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
            cell_condition: str | None  = "filter_cells",
            gene_condition: str | None = "filter_genes",
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
            ``"data"``、``"data_normalize"``、``"data_log1p"`` 和 ``"data_scale"``。

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

        Returns
        -------
        Any
            函数返回底层实现产生的结果。

        Examples
        --------
        遍历 CSR 小批量::

            for X_batch in atlas.get_minibatch_csr():
                print(X_batch.shape)
                break"""

        fetcher = MultiThreadedMinibatchFetcher(file_path = self.file_path, x_type = x_type)
        for X_batch in fetcher.run():
            pass
            # yield X_batch


    def get_minibatch_dense(
            self,
            pass_mode: str = "single-pass",
            batch_size: int = 2048,
            max_batches: int | None = None,
            buffer_batch_num: int = 5,
    ):
        """以 dense 小批量读取表达矩阵。

        该方法把数据库中的稀疏表达记录按批转换为 dense array，适合 IncrementalPCA、MiniBatchKMeans 等需要 dense 输入的算法。

        Parameters
        ----------
        pass_mode
            小批量读取模式。用于控制迭代器如何遍历数据库中的表达矩阵。
        batch_size
            每个小批量包含的细胞数量。较大值通常更快，但会增加内存占用。
        max_batches
            最多读取的小批量数量。为 ``None`` 时遍历全部可用批次。
        buffer_batch_num
            预取缓冲区中的批次数量。较大值可提高吞吐，但会占用更多内存。

        Returns
        -------
        Any
            函数返回底层实现产生的结果。

        Examples
        --------
        读取用于模型拟合的小批量::

            for X_batch in atlas.get_minibatch_dense(batch_size=4096):
                print(X_batch.shape)
                break

        限制只读取前 100 个批次做快速测试::

            batches = atlas.get_minibatch_dense(batch_size=2048, max_batches=100)"""

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
            )

            pass_batches = 0

            for X_batch in fetcher.run():
                produced_batches += 1
                pass_batches += 1

                yield X_batch

                if max_batches is not None and produced_batches >= max_batches:
                    break

            pass_id += 1

            # 防止异常情况下空 pass 无限循环
            if pass_batches == 0:
                logger.debug("[get_minibatch_dense] pass produced 0 batch, stop")
                break


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
        load_type: Literal["order", "random", "list_random"] = "random",
        store_type: StoreType = "count",
        cells_per_block: int | None = None,
    ) -> Any:
        """通过 Atlas 对象导入 h5ad 文件。

        这是底层 h5ad 导入函数的对象式入口，可以用 ``atlas.load_h5ad(...)`` 直接调用。

        Parameters
        ----------
        h5ad_path
            输入 ``.h5ad`` 文件路径，或多个 ``.h5ad`` 文件路径组成的列表。
        load_type
            导入方式。``"order"`` 表示按原始顺序导入，``"random"`` 表示按随机/分块策略导入。
        store_type
            表达值写入类型。当前约定支持 ``"count"`` 和 ``"log"``。
        cells_per_block
            写入稀疏表达矩阵时每个细胞块包含的细胞数。

        Returns
        -------
        Any
            函数返回底层实现产生的结果。

        Examples
        --------
        顺序导入单个 h5ad 文件::

            atlas = sap.Atlas(r"F:\\data\\pbmc")
            atlas.load_h5ad(r"F:\\data\\pbmc.h5ad", load_type="order")

        导入多个 h5ad 文件::

            atlas.load_h5ad([
                r"F:\\data\\batch1.h5ad",
                r"F:\\data\\batch2.h5ad",
            ])"""
        return _io_load_h5ad(
            h5ad_path,
            self,
            load_type=load_type,
            store_type=store_type,
            cells_per_block=cells_per_block,
        )


    def load_anndata(self, adata: AnnData) -> None:
        """通过 Atlas 对象导入 AnnData。

        这是底层 AnnData 导入函数的对象式入口，用于把内存中的 AnnData 写入当前 Atlas 数据库。

        Parameters
        ----------
        adata
            AnnData 对象。函数会把其中的 ``obs``、``var``、表达矩阵和可支持的结果写入 Atlas 数据库。

        Returns
        -------
        None
            结果直接写入 Atlas 数据库或当前图形窗口。

        Examples
        --------
        从 Scanpy 读取 h5ad 后导入 Atlas::

            adata = sc.read_h5ad(r"F:\\data\\pbmc.h5ad")
            atlas = sap.Atlas(r"F:\\data\\pbmc")
            atlas.load_anndata(adata)"""

        return _io_load_anndata(adata, self)


    def load_multi_format(self, file_path: PathLike[str] | str) -> None:
        """通过 Atlas 对象导入多格式输入文件。

        这是底层多格式导入函数的对象式入口，用于根据文件后缀选择对应导入流程。

        Parameters
        ----------
        file_path
            输入文件路径。函数会根据文件格式选择合适的读取方式。

        Returns
        -------
        None
            结果直接写入 Atlas 数据库或当前图形窗口。

        Examples
        --------
        导入一个支持的输入文件::

            atlas = sap.Atlas(r"F:\\data\\pbmc")
            atlas.load_multi_format(r"F:\\data\\pbmc.h5ad")"""

        return _io_load_multi_format(file_path, self)


    def gene_names_duplicated(
        self,
        gene_name_column: str = "atlas_gene_name",
    ) -> bool | None:
        """检查基因名称是否重复。

        这是底层基因名重复检查函数的对象式入口，用于检查 ``var`` 中指定基因名称列是否存在重复值。

        Parameters
        ----------
        gene_name_column
            保存基因名称的 ``var`` 列名。通常为 ``"atlas_gene_name"``。

        Returns
        -------
        bool 或 None
            检查结果。无法完成检查时可能返回 ``None``。

        Examples
        --------
        检查默认基因名列::

            atlas.gene_names_duplicated()

        检查自定义基因名列::

            atlas.gene_names_duplicated(gene_name_column="gene_symbol")"""

        return _io_gene_names_duplicated(self, gene_name_column=gene_name_column)


    def write_h5ad(
        self,
        out_h5ad_path: PathLike[str] | str,
        *,
        batch_cells: int = 1_000_000,
    ) -> None:
        """通过 Atlas 对象导出 h5ad 文件。

        这是底层 h5ad 导出函数的对象式入口，会把当前 Atlas 数据库中的表达矩阵和元数据导出为 AnnData/h5ad。

        Parameters
        ----------
        out_h5ad_path
            输出 ``.h5ad`` 文件路径。
        batch_cells
            导出表达矩阵时每批处理的细胞数。

        Returns
        -------
        None
            结果直接写入 Atlas 数据库或当前图形窗口。

        Examples
        --------
        导出为 h5ad 文件::

            atlas.write_h5ad(r"F:\\data\\pbmc_export.h5ad")

        使用较小批次降低内存占用::

            atlas.write_h5ad(r"F:\\data\\pbmc_export.h5ad", batch_cells=200000)"""

        return _io_write_h5ad(
            self,
            out_h5ad_path,
            batch_cells=batch_cells,
        )


    def get_obs_df(
        self,
        columns: list[str] | str | None = None,
    ) -> pd.DataFrame:
        """通过 Atlas 对象读取 obs 表。

        这是底层 ``obs`` 读取函数的对象式入口，用于把 ``obs`` 表中的指定列读取为 DataFrame。

        Parameters
        ----------
        columns
            需要从 ``obs`` 中读取的列名。可以是单个字符串、字符串列表或 ``None``。

        Returns
        -------
        pandas.DataFrame
            包含查询、统计或绘图所需数据的表格。

        Examples
        --------
        读取全部 ``obs`` 列::

            obs = atlas.get_obs_df()

        只读取聚类和 QC 列::

            obs = atlas.get_obs_df(columns=["kmeans", "n_genes_by_counts", "pct_counts_mt"])"""

        return _io_get_obs_df(self, columns=columns)


    def get_anndata(
        self,
        atlas_cell_ids: list[int] | np.ndarray | None,
        use_data: str = "data",
        include_obsm: bool = True,
        include_varm: bool = True,
    ) -> AnnData:
        """通过 Atlas 对象构建 AnnData。

        该函数不是遍历函数，仅用于取用较小的数据子集；

        这是底层 AnnData 构建函数的对象式入口，用于从 Atlas 数据库读取指定细胞、表达矩阵和 embedding，构建内存中的 AnnData 对象。

        Parameters
        ----------
        atlas_cell_ids
            需要导出的 Atlas 细胞 ID 列表；为 ``None`` 时通常导出当前索引对应的全部细胞。
        use_data
            读取的表达矩阵或结果表名称。常用值包括 ``"data"``、``"data_normalize"``、``"data_log1p"`` 和
            ``"data_scale"``。
        include_obsm
            是否把 ``obsm_*`` 结果表写入返回的 AnnData。
        include_varm
            是否把 ``varm_*`` 结果表写入返回的 AnnData。

        Returns
        -------
        AnnData
            从 Atlas 数据库构建的 AnnData 对象。

        Examples
        --------
        导出前 1000 个细胞为 AnnData::

            cell_ids = atlas.query("SELECT atlas_cell_id FROM obs LIMIT 1000")["atlas_cell_id"]
            adata = atlas.get_anndata(cell_ids.tolist(), use_data="data_log1p")

        导出当前过滤索引中的全部细胞::

            adata = atlas.get_anndata(None, include_obsm=True)"""

        return _io_get_anndata(
            self,
            atlas_cell_ids=atlas_cell_ids,
            use_data=use_data,
            include_obsm=include_obsm,
            include_varm=include_varm,
        )
