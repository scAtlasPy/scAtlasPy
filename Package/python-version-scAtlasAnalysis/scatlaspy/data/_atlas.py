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

    def __init__(
            self,
            file_name: PathLike[str] | str,
            verbosity: Literal["error", "warning", "info", "debug"] | None = "warning",
    ):
        """
        初始化 Atlas 数据库对象。

        支持：
            Atlas(r"F:\\data\\file_name\\sql_obs.sasql")
            Atlas(r"F:\\data\\file_name\\sql_obs")
            Atlas(Path(r"F:\\data\\file_name\\sql_obs.sasql"))
            Atlas(Path(r"F:\\data\\file_name\\sql_obs"))
        """

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
        """
        Atlas 数据库文件的绝对路径。
        """
        return self.__file_path


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
    def connection(self, value: Optional[duckdb.DuckDBPyConnection]) -> None:
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

        use_data
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
            use_data=use_data,
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
            batch_size: int = 2048,
            max_batches: int | None = None,
            buffer_batch_num: int = 5,
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
        return _io_load_h5ad(
            h5ad_path,
            self,
            load_type=load_type,
            store_type=store_type,
            cells_per_block=cells_per_block,
            blocks_per_pool=blocks_per_pool,
        )


    def load_anndata(self, adata: AnnData) -> None:
        return _io_load_anndata(adata, self)


    def load_multi_format(self, file_path: PathLike[str] | str) -> None:
        return _io_load_multi_format(file_path, self)


    def gene_names_duplicated(
        self,
        gene_name_column: str = "atlas_gene_name",
    ) -> bool | None:
        return _io_gene_names_duplicated(self, gene_name_column=gene_name_column)


    def write_h5ad(
        self,
        out_h5ad_path: PathLike[str] | str,
        *,
        batch_cells: int = 1_000_000,
    ) -> None:
        return _io_write_h5ad(
            self,
            out_h5ad_path,
            batch_cells=batch_cells,
        )


    def get_obs_df(
        self,
        columns: list[str] | str | None = None,
    ) -> pd.DataFrame:
        return _io_get_obs_df(self, columns=columns)


    def get_anndata(
        self,
        atlas_cell_ids: list[int] | np.ndarray | None,
        use_data: str = "data",
        include_obsm: bool = True,
        include_varm: bool = True,
    ) -> AnnData:
        return _io_get_anndata(
            self,
            atlas_cell_ids=atlas_cell_ids,
            use_data=use_data,
            include_obsm=include_obsm,
            include_varm=include_varm,
        )

