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


def set_verbosity(level: str = "warning") -> None:
    """设置 scAtlasPy 的日志输出级别。

    该函数调整包内使用的 ``Atlas`` logger，统一控制导入、预处理、工具函数和绘图流程中的日志详细程度。它不会主动修改第三方库的日志级别。

    Parameters
    ----------
    level
        日志级别字符串。可选值包括 ``"debug"``、``"info"``、``"warning"`` 和 ``"error"``。

    Returns
    -------
    None
        结果直接写入 Atlas 数据库或当前图形窗口。

    Examples
    --------
    只显示警告和错误信息::

        sap.set_verbosity("warning")

    调试导入或预处理流程::

        sap.set_verbosity("debug")
        atlas = sap.Atlas(r"F:\\data\\pbmc")"""

    level = str(level).lower()
    level_map = {
        "error": logging.ERROR,
        "warning": logging.WARNING,
        "info": logging.INFO,
        "debug": logging.DEBUG,
    }

    if level not in level_map:
        raise ValueError("level 只支持: error, warning, info, debug")

    atlas_logger = logging.getLogger("Atlas")
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

        atlas = sap.Atlas(r"F:\\data\\pbmc", verbosity="info")
        atlas.load_h5ad(r"F:\\data\\pbmc.h5ad", load_type="order")

    查看数据库内容并关闭连接::

        atlas.describe()
        atlas.head("obs", n=5)
        atlas.close()"""

    def __init__(
            self,
            file_name: PathLike[str] | str,
            verbosity: Literal["error", "warning", "info", "debug"] | None = "warning",
    ):
        """初始化 Atlas 数据库对象。

        构造函数会根据 ``file_name`` 推断 ``.sasql`` 数据库路径，创建父目录，并建立 DuckDB 连接。如果数据库文件尚不存在，会自动创建空数据库。

        Parameters
        ----------
        file_name
            Atlas 数据库文件路径或数据库名称。可以传入完整 ``.sasql`` 路径，也可以传入不带后缀的路径；函数会自动补全 ``.sasql``
            后缀。
        verbosity
            初始化 Atlas 时设置的日志级别。设为 ``None`` 时不修改当前日志配置。

        Returns
        -------
        None
            结果直接写入 Atlas 数据库或当前图形窗口。

        Examples
        --------
        传入不带后缀的数据库路径::

            atlas = sap.Atlas(r"F:\\data\\test_10W")

        传入完整 ``.sasql`` 文件路径，并打开更详细日志::

            atlas = sap.Atlas(r"F:\\data\\test_10W.sasql", verbosity="info")"""

        if verbosity is not None:
            set_verbosity(verbosity)

        self.__file_path = self._resolve_file_path(file_name)
        self.__connection = None
        self.__mode: Literal["r+", "r"] = "r+"

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


    @property
    def file_path(self) -> str:
        """返回 Atlas 数据库文件路径。

        该属性返回当前 Atlas 对象指向的 ``.sasql`` 文件路径，可用于确认数据库实际保存位置。

        Returns
        -------
        str
            数据库或对象的文本摘要。

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
        """返回当前 DuckDB 连接。

        该属性保存 Atlas 当前使用的 DuckDB 连接对象。通常不需要直接操作它，除非需要调用 DuckDB 的底层 API。

        Parameters
        ----------
        value
            参数。用于控制该函数的输入、输出或计算细节；默认值适合常规 Atlas 工作流。

        Returns
        -------
        duckdb.DuckDBPyConnection
            当前 Atlas 数据库连接。

        Examples
        --------
        使用底层 DuckDB 连接执行查询::

            con = atlas.connection
            con.sql("SELECT COUNT(*) FROM obs").fetchone()"""
        self.__connection = value


    def _create(self) -> duckdb.DuckDBPyConnection:
        """
        创建 Atlas 数据库文件。
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

            logger.debug(f"数据库已成功创建: {self.file_path}")
            return con

        except Exception as e:
            logger.exception("创建数据库异常详情:")
            raise RuntimeError(f"创建数据库失败: {str(e)}")


    def connect(self, mode: Literal["r+", "r"] = "r+") -> duckdb.DuckDBPyConnection:
        """连接 Atlas 数据库。

        根据当前 ``file_path`` 建立 DuckDB 连接，并把连接对象保存到 ``atlas.connection``。已有连接会被复用或替换为新的连接。

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

        Returns
        -------
        bool 或 None
            检查结果。无法完成检查时可能返回 ``None``。

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
            f"file_name    : {file_name}\n"
            f"tables      : {len(tables)}\n"
            f"table names : {table_names}\n"
            f"n_cells     : {fmt(n_cells)}\n"
            f"n_genes     : {fmt(n_genes)}"
        )

        return text


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


    def head(self, table_name: str, n: int = 5):
        """查看数据库表的前几行。

        该方法打印并返回指定表的前 ``n`` 行，同时展示表名、列名和行数信息，适合快速检查导入结果或分析结果。

        Parameters
        ----------
        table_name
            数据库表名。
        n
            返回或展示的记录数量。

        Returns
        -------
        pandas.DataFrame
            包含查询、统计或绘图所需数据的表格。

        Examples
        --------
        查看 ``obs`` 前 5 行::

            atlas.head("obs")

        查看差异基因结果前 10 行::

            atlas.head("rank_genes_groups", n=10)"""

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


    def build_read_index(
            self,
            cell_condition: str | None = None,
            gene_condition: str | None = None,
            use_hvg: bool = True,
            use_data: str = "data_log1p",
    ):
        """构建表达矩阵读取索引。

        该方法根据细胞过滤条件、基因过滤条件和高变基因标记，构建后续小批量读取表达矩阵所需的索引表。PCA、K-means 和部分绘图函数通常依赖该索引。

        Parameters
        ----------
        cell_condition
            用于筛选细胞的 ``obs`` 条件列名或 SQL 条件；为 ``None`` 时不按细胞过滤。
        gene_condition
            用于筛选基因的 ``var`` 条件列名或 SQL 条件；为 ``None`` 时不按基因过滤。
        use_hvg
            是否优先使用高变基因列构建读取索引或是否只使用高变基因。
        use_data
            读取的表达矩阵或结果表名称。常用值包括 ``"data"``、``"data_normalize"``、``"data_log1p"`` 和
            ``"data_scale"``。

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

            for X_batch, cell_ids in atlas.get_minibatch_csr():
                print(X_batch.shape, len(cell_ids))
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

            for X_batch, cell_ids in atlas.get_minibatch_dense(batch_size=4096):
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
    #   sap.io.load_h5ad(file_path, atlas, ...)
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
        cells_per_block: int = 500,
        blocks_per_pool: int = 10,
    ) -> Any:
        """通过 Atlas 对象导入 h5ad 文件。

        这是 ``sap.io.load_h5ad`` 的对象式包装。函数位置仍在 ``scatlaspy.io``，但可以用 ``atlas.load_h5ad(...)`` 直接调用。

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
        blocks_per_pool
            批量写入时每个处理池包含的块数量。

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
            blocks_per_pool=blocks_per_pool,
        )


    def load_anndata(self, adata: AnnData) -> None:
        """通过 Atlas 对象导入 AnnData。

        这是 ``sap.io.load_anndata`` 的对象式包装，用于把内存中的 AnnData 写入当前 Atlas 数据库。

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

        这是 ``sap.io.load_multi_format`` 的对象式包装，用于根据文件后缀选择对应导入流程。

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

        这是 ``sap.io.gene_names_duplicated`` 的对象式包装，用于检查 ``var`` 中指定基因名称列是否存在重复值。

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

        这是 ``sap.io.write_h5ad`` 的对象式包装，会把当前 Atlas 数据库中的表达矩阵和元数据导出为 AnnData/h5ad。

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

        这是 ``sap.io.get_obs_df`` 的对象式包装，用于把 ``obs`` 表中的指定列读取为 DataFrame。

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

        这是 ``sap.io.get_anndata`` 的对象式包装，用于从 Atlas 数据库读取指定细胞、表达矩阵和 embedding，构建内存中的 AnnData 对象。

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

