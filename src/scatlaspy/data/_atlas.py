from typing import *
from _duckdb import DuckDBPyConnection
import duckdb
import gc
import os
import logging
import numpy as np
import pandas as pd
from functools import wraps
from os import PathLike
from anndata import AnnData
from ._minibatch import MultiThreadedMinibatchFetcher
from ._filter_index import FilterIndexBuilder
from ..io import (
    rename_duplicated_genes as _io_rename_duplicated_genes,
    get_anndata as _io_get_anndata,
    get_obs_df as _io_get_obs_df,
    get_obsm_df as _io_get_obsm_df,
    get_var_df as _io_get_var_df,
    get_varm_df as _io_get_varm_df,
    get_uns_df as _io_get_uns_df,
    load_anndata as _io_load_anndata,
    load_h5ad as _io_load_h5ad,
    load_multi_format as _io_load_multi_format,
    write_h5ad as _io_write_h5ad,
)

# Configure logging.
logger = logging.getLogger("Atlas")
logger.addHandler(logging.NullHandler())

_GB = 1024 ** 3
_MB = 1024 ** 2


class _AtlasLike(Protocol):
    """Minimal Atlas interface needed by the memory-limit decorator."""

    @property
    def connection(self) -> Optional[duckdb.DuckDBPyConnection]:
        ...

    @property
    def db_memory_limit(self) -> str | int | None:
        ...

    def _resolve_step_memory_limit(self, memory_limit: str | int) -> str:
        ...

    def _set_db_memory_limit(self, db_memory_limit: str | int | None) -> None:
        ...


def _parse_memory_limit_to_bytes(memory_limit: str | int | None) -> int | None:
    """Convert a DuckDB memory-limit value into bytes."""

    if memory_limit is None:
        return None

    if isinstance(memory_limit, int):
        if memory_limit <= 0:
            raise ValueError("db_memory_limit must be > 0")
        return int(memory_limit * _GB)

    if not isinstance(memory_limit, str):
        raise TypeError(
            "db_memory_limit must be str, int, or None, "
            f"but received: {type(memory_limit)}"
        )

    text = memory_limit.strip().upper().replace(" ", "")
    if text in {"", "NONE"}:
        return None

    units = [
        ("TB", 1024 ** 4),
        ("T", 1024 ** 4),
        ("GB", _GB),
        ("G", _GB),
        ("MB", _MB),
        ("M", _MB),
        ("KB", 1024),
        ("K", 1024),
        ("B", 1),
    ]

    for suffix, scale in units:
        if text.endswith(suffix):
            value_text = text[: -len(suffix)]
            break
    else:
        value_text = text
        scale = _GB

    value = float(value_text)
    if value <= 0:
        raise ValueError("db_memory_limit must be > 0")
    return int(value * scale)


def _format_memory_limit_bytes(memory_bytes: int) -> str:
    """Format bytes as a DuckDB memory-limit string."""

    memory_bytes = max(int(memory_bytes), _MB)
    if memory_bytes % _GB == 0:
        return f"{memory_bytes // _GB}GB"
    if memory_bytes % _MB == 0:
        return f"{memory_bytes // _MB}MB"
    return f"{memory_bytes}B"


def _get_atlas_from_call(args: tuple[Any, ...], kwargs: dict[str, Any]) -> _AtlasLike:
    """Find the Atlas object in a function or bound-method call."""

    if args:
        candidate = args[0]
        if hasattr(candidate, "connection") and hasattr(candidate, "db_memory_limit"):
            return candidate
    if "atlas" in kwargs:
        candidate = kwargs["atlas"]
        if hasattr(candidate, "connection") and hasattr(candidate, "db_memory_limit"):
            return candidate
    raise TypeError("duckdb_memory_limit requires an Atlas object as the first argument")


def _get_duckdb_memory_setting(atlas: _AtlasLike) -> str:
    """Return DuckDB's active memory_limit setting for logging."""

    if atlas.connection is None:
        return "connection-unavailable"

    try:
        return str(
            atlas.connection.execute(
                "SELECT current_setting('memory_limit')"
            ).fetchone()[0]
        )
    except Exception as exc:
        return f"unavailable: {type(exc).__name__}: {exc}"


def duckdb_memory_limit(memory_limit: str | int):
    """Temporarily use a smaller DuckDB memory limit for one Atlas operation.

    The requested limit is capped by the current Atlas ``db_memory_limit``. This
    means a function-level limit can lower the memory limit for memory-sensitive
    operations, but it cannot silently raise a user-provided global limit.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            atlas = _get_atlas_from_call(args, kwargs)
            old_limit = atlas.db_memory_limit
            step_limit = atlas._resolve_step_memory_limit(memory_limit)
            atlas._set_db_memory_limit(step_limit)
            logger.info(
                "[duckdb_memory_limit] enter %s: requested=%s previous=%s effective=%s duckdb=%s",
                func.__name__,
                memory_limit,
                old_limit,
                step_limit,
                _get_duckdb_memory_setting(atlas),
            )
            try:
                return func(*args, **kwargs)
            finally:
                atlas._set_db_memory_limit(old_limit)
                logger.info(
                    "[duckdb_memory_limit] exit %s: restored=%s duckdb=%s",
                    func.__name__,
                    old_limit,
                    _get_duckdb_memory_setting(atlas),
                )

        return wrapper

    return decorator


def set_verbosity(
    level: Literal["silence", "error", "warning", "info", "debug"] | None = "silence",
) -> None:
    """Set the scAtlasPy logging verbosity.
    This function configures the package-level ``Atlas`` logger used by import, preprocessing, tools, and plotting workflows.
    It only controls scAtlasPy logging and does not change log levels for third-party libraries.

    Parameters
    ----------
    level
        Logging verbosity. Supported values are ``"silence"``, ``"error"``, ``"warning"``, ``"info"``, ``"debug"``, and ``None``.
        The default value, ``"silence"``, disables the ``Atlas`` logger so scAtlasPy does not emit its own log messages.
        Passing ``None`` also disables the logger for compatibility with older calling code.

    Returns
    -------
    None
        The logger configuration is updated in place and no object is returned.

    Examples
    --------
    Disable scAtlasPy logging::

        sap.set_verbosity()
        sap.set_verbosity("silence")

    Show warnings and errors::

        sap.set_verbosity("warning")

    Enable detailed logs while debugging an import or preprocessing workflow::

        sap.set_verbosity("debug")
        atlas = sap.Atlas("./data/pbmc")
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
        raise ValueError("level must be one of: silence, error, warning, info, debug, None")

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
    """Atlas database object.

    An ``Atlas`` instance manages a persistent DuckDB-backed ``.sasql`` database.
    It stores the database path and active connection, and provides convenient methods for creating,
    opening, querying, inspecting, reading, and delegating IO operations.

    Attributes
    ----------
    file_path
        Path to the current ``.sasql`` database file.
    connection
        Active DuckDB connection object.

    Examples
    --------
    Create or connect to an Atlas database::

        atlas = sap.Atlas("./data/pbmc")

    Import an h5ad file through the object API::

        atlas = sap.Atlas("./data/pbmc")
        atlas.load_h5ad("./data/pbmc.h5ad")

    Inspect the database and close the connection::

        atlas.describe()
        atlas.head("obs", n=5)
        atlas.close()
    """

    def __init__(
        self,
        file_name: PathLike[str] | str,
        db_memory_limit: str | int | None = "20G",
    ):
        """Initialize an Atlas database object.

        The constructor resolves the ``.sasql`` database path from ``file_name``,
        creates the parent directory when needed, and opens a DuckDB connection.
        If the database file does not exist yet, an empty database is created automatically.

        Parameters
        ----------
        file_name
            Atlas database file path or database name.
            A full ``.sasql`` path can be provided directly;
            if the suffix is omitted, ``.sasql`` is appended automatically.

        db_memory_limit
            Memory limit used by DuckDB.
            This can be a DuckDB-compatible string such as ``"4GB"`` or an integer interpreted as GB,
            for example ``4`` is equivalent to ``"4GB"``.

            The default requested limit is ``"20G"``. Atlas applies the smaller
            value between the requested limit and 60% of detected system physical
            memory, rounded down to whole GB. For example, on a 32 GB machine,
            the default effective DuckDB limit is ``"19GB"``.

            If ``None`` is passed explicitly, Atlas uses only the 60% system
            memory cap.

            This parameter only limits memory used by DuckDB queries and intermediate computation.
            It does not limit memory allocated directly by Python, NumPy, or pandas.

        Returns
        -------
        None
            The object is initialized in place.

        Examples
        --------
        Pass a database path without a suffix::
            atlas = sap.Atlas("./data/test_10W")

        Pass a complete ``.sasql`` file path::
            atlas = sap.Atlas("./data/test_10W.sasql")

        Enable more detailed package logs::
            sap.set_verbosity("info")
            atlas = sap.Atlas("./data/test_10W.sasql")

        Limit DuckDB query and intermediate-computation memory to 4 GB::
            atlas = sap.Atlas("./data/test_10W", db_memory_limit="4GB")
        """

        self.__file_path = self._resolve_file_path(file_name)
        self.__connection = None
        self.__mode: Literal["r+", "r"] = "r+"
        self.__db_memory_limit = self._resolve_db_memory_limit(db_memory_limit)

        logger.info(f"Initializing Atlas instance, file_name:{self.file_path}")

        if not os.path.exists(self.file_path):
            logger.info(f"Database file does not exist; creating a new database: {self.file_path}")
            try:
                self._create()
                logger.info(f"Database created successfully: {self.file_path}")
            except Exception as e:
                logger.error(f"Database creation failed: {str(e)}")
                raise
        else:
            self.connect("r+")
            logger.info(f"Database file already exists: {self.file_path}; connection created")

        logger.info("Atlas instance initialized")

    def __enter__(self) -> "Atlas":
        """Return the Atlas object when used as a context manager.

        This enables ``with``-style usage while keeping explicit ``close()``
        calls available for users who prefer manual connection management.

        Examples
        --------
        Use Atlas in a context manager::

            with sap.Atlas("./data/pbmc") as atlas:
                atlas.describe()
        """

        return self


    def __exit__(self, exc_type, exc, tb) -> None:
        """Close the Atlas connection when leaving a context manager."""

        self.close()


    def __del__(self) -> None:
        """Best-effort cleanup for an Atlas object that was not closed explicitly."""

        try:
            self.close()
        except Exception:
            # Destructors may run during interpreter shutdown, when modules,
            # loggers, or DuckDB internals are already partially finalized.
            pass


    @staticmethod
    def _resolve_file_path(file_name: PathLike[str] | str) -> str:
        """Resolve the Atlas database file path.

        The method accepts paths with or without the ``.sasql`` suffix and returns an absolute path.
        The suffix is appended automatically when it is missing.

        Supported examples::

            Atlas("./data/file_name/sql_obs.sasql")
            Atlas("./data/file_name/sql_obs")
            Atlas(Path("./data/file_name/sql_obs.sasql"))
            Atlas(Path("./data/file_name/sql_obs"))
        """

        file_name = os.fspath(file_name)

        if not isinstance(file_name, str):
            raise TypeError(
                "file_name must be str or PathLike[str], "
                f"but received: {type(file_name)}"
            )

        file_name = file_name.strip()

        if file_name == "":
            raise ValueError("file_name cannot be empty")

        file_name = os.path.expanduser(file_name)

        if file_name.lower().endswith(".sasql"):
            return os.path.abspath(file_name)

        return os.path.abspath(file_name + ".sasql")


    @staticmethod
    def _get_system_memory_gb_floor() -> int:
        """Return the system physical memory rounded down to integer GB.

        The implementation prefers standard-library mechanisms and does not require ``psutil``.
         It uses ``GlobalMemoryStatusEx`` on Windows and ``os.sysconf`` on Linux or macOS.
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
                raise RuntimeError("Unable to get current Windows system physical memory size")

            total_bytes = int(memory_status.ullTotalPhys)

        # Linux / macOS
        else:
            try:
                page_size = os.sysconf("SC_PAGE_SIZE")
                physical_pages = os.sysconf("SC_PHYS_PAGES")
                total_bytes = int(page_size * physical_pages)
            except Exception as e:
                raise RuntimeError(
                    "Unable to get current system physical memory size. Please pass db_memory_limit explicitly, for example '32GB'"
                ) from e

        memory_gb = int(total_bytes // (1024 ** 3))

        if memory_gb <= 0:
            raise RuntimeError(
                "Detected system physical memory is less than 1GB. Please pass db_memory_limit explicitly"
            )

        return memory_gb


    @staticmethod
    def _resolve_db_memory_limit(
        db_memory_limit: str | int | None,
    ) -> str:
        """Resolve the DuckDB memory-limit argument.

        The resolved value is always capped at 60% of system physical memory.
        Passing ``None`` uses this 60% cap directly.
        """

        system_memory_gb = Atlas._get_system_memory_gb_floor()
        system_cap_bytes = max(int(system_memory_gb * 0.6), 1) * _GB

        if db_memory_limit is None:
            return _format_memory_limit_bytes(system_cap_bytes)

        user_limit_bytes = _parse_memory_limit_to_bytes(db_memory_limit)
        if user_limit_bytes is None:
            return _format_memory_limit_bytes(system_cap_bytes)

        return _format_memory_limit_bytes(min(user_limit_bytes, system_cap_bytes))


    def _resolve_step_memory_limit(self, memory_limit: str | int) -> str:
        """Cap a function-level memory limit by the current Atlas limit."""

        current_limit_bytes = _parse_memory_limit_to_bytes(self.db_memory_limit)
        step_limit_bytes = _parse_memory_limit_to_bytes(memory_limit)

        if step_limit_bytes is None:
            return self.db_memory_limit
        if current_limit_bytes is None:
            return _format_memory_limit_bytes(step_limit_bytes)

        return _format_memory_limit_bytes(min(step_limit_bytes, current_limit_bytes))


    @property
    def file_path(self) -> str:
        """Return the Atlas database file path.

        This property returns the absolute path of the ``.sasql`` file associated with the current Atlas object.

        Returns
        -------
        str
            Absolute path to the current ``.sasql`` database file.

        Examples
        --------
        Inspect the current database path::

            atlas = sap.Atlas("./data/pbmc")
            atlas.file_path
        """
        return self.__file_path


    @property
    def connection(self) -> Optional[duckdb.DuckDBPyConnection]:
        """Return the current DuckDB connection.

        This property stores the active DuckDB connection used by the Atlas object.
        Direct access is usually unnecessary unless low-level DuckDB APIs are needed.

        Returns
        -------
        duckdb.DuckDBPyConnection
            Active Atlas database connection.

        Examples
        --------
        Run a query through the underlying DuckDB connection::

            con = atlas.connection
            con.sql("SELECT COUNT(*) FROM obs").fetchone()
        """
        return self.__connection


    @connection.setter
    def connection(self, value: Optional[duckdb.DuckDBPyConnection]) -> None:
        """Set the current DuckDB connection object.

        This setter is mainly intended for internal workflows or advanced callers that need to replace the active Atlas connection manually.
        In ordinary use, prefer ``atlas.connect(...)`` and ``atlas.close()``.

        Parameters
        ----------
        value
            DuckDB connection object to store on the current Atlas instance, or ``None`` to clear the active connection.

        Returns
        -------
        None
            The internal connection reference is updated in place.
        """
        self.__connection = value

    @property
    def db_memory_limit(self) -> str | int | None:
        """Return the DuckDB memory limit configured for the current Atlas object."""

        return self.__db_memory_limit


    def _set_db_memory_limit(self, db_memory_limit: str | int | None) -> None:
        """Update the active Atlas memory limit and apply it to the connection."""

        self.__db_memory_limit = db_memory_limit
        self._apply_memory_limit()


    def _apply_memory_limit(self) -> None:
        """Apply the DuckDB memory limit to the active connection.

        ``db_memory_limit`` only limits memory available to DuckDB queries and intermediate computation.
        It does not limit memory allocated directly by Python, NumPy, or pandas.
        If ``db_memory_limit`` is an integer, it is interpreted as GB, so ``4`` becomes ``"4GB"``.

        Returns
        -------
        None
            The setting is applied directly to the active DuckDB connection.

        Examples
        --------
        Limit DuckDB query memory when initializing Atlas::
            atlas = sap.Atlas("./data/pbmc", db_memory_limit="4GB")

        Use an integer value interpreted as GB::
            atlas = sap.Atlas("./data/pbmc", db_memory_limit=4)

        The limit is also applied when connecting to an existing database::
            atlas = sap.Atlas("./data/pbmc.sasql", db_memory_limit="1024MB")
        """

        if self.__connection is None:
            return

        if self.db_memory_limit is None:
            return

        db_memory_limit = str(self.db_memory_limit).strip()

        if db_memory_limit == "":
            return

        memory_limit_sql = db_memory_limit.replace("'", "''")

        self.__connection.execute(
            f"SET memory_limit = '{memory_limit_sql}'"
        )

        logger.info(f"DuckDB db_memory_limit set to: {db_memory_limit}")


    def _create(self) -> duckdb.DuckDBPyConnection:
        """Create an Atlas database file and return its connection.

        The method creates a new ``.sasql`` database file at ``self.file_path``.
        After the connection is created, ``self._apply_memory_limit()`` is called so the configured ``db_memory_limit``
        takes effect on the new connection.

        Returns
        -------
        duckdb.DuckDBPyConnection
            Newly created DuckDB connection object.
        """

        db_dir = os.path.dirname(self.file_path)

        logger.debug(f"Start creating database: {self.file_path}")

        if os.path.exists(self.file_path):
            raise RuntimeError(f"Database already exists: {self.file_path}")

        try:
            logger.debug(f"Creating directory: {db_dir}")
            os.makedirs(db_dir, exist_ok=True)

            logger.debug("Connecting to DuckDB database")
            con = duckdb.connect(database=self.file_path)

            self.__connection = con
            self._apply_memory_limit()

            logger.debug(f"Database created successfully: {self.file_path}")
            return con

        except Exception as e:
            logger.exception("Database creation exception details:")
            raise RuntimeError(f"Failed to create database: {str(e)}")


    def connect(self, mode: Literal["r+", "r"] = "r+") -> duckdb.DuckDBPyConnection:
        """Connect to the Atlas database.

        A DuckDB connection is created from the current ``file_path`` and stored on ``atlas.connection``.
        Existing connections are closed before a new connection is opened.
        If ``db_memory_limit`` was configured during initialization, it is applied after every new connection.

        Parameters
        ----------
        mode
            Database connection mode. ``"r+"`` opens a read-write connection, while ``"r"`` opens a read-only connection.

        Returns
        -------
        duckdb.DuckDBPyConnection
            Active Atlas database connection.

        Examples
        --------
        Connect with the default read-write mode::

            atlas.connect()

        Open the database in read-only mode to inspect existing results::

            atlas.connect(mode="r")
            atlas.head("obs")
        """

        logger.info(f"Database connection requested, mode: {mode}")

        if self.__connection is not None: # Close any existing connection first.
            logger.debug("Existing database connection found; closing it first")
            self.close()
            gc.collect()

        try:
            if mode == "r":  # Read-only mode.
                logger.debug("Read-only connection mode")
                # Check whether the file exists.
                if not os.path.exists(self.file_path):
                    logger.error(f"Database file does not exist; cannot connect in read-only mode: {self.file_path}")
                    raise FileNotFoundError(f"Database file does not exist: {self.file_path}")

                # Connect in read-only mode.
                self.__connection = duckdb.connect(database=self.file_path, read_only=True)
                logger.info(f"Connected to database in read-only mode: {self.file_path}")

            elif mode == "r+":  # Read-write mode.
                logger.debug("Read-write connection mode")
                db_dir = os.path.dirname(self.file_path)
                os.makedirs(db_dir, exist_ok=True)

                # Create or connect regardless of whether the file already exists.
                self.__connection = duckdb.connect(database=self.file_path, read_only=False)

                if os.path.exists(self.file_path):
                    logger.info(f"Connected to existing database in read-write mode: {self.file_path}")
                else:
                    logger.info(f"Created and connected to new database: {self.file_path}")

            else:
                logger.error(f"Unsupported connection mode: {mode}")
                raise ValueError(f"Unsupported connection mode: {mode}")

            self.__mode = mode
            self._apply_memory_limit()
            logger.debug("Database connection succeeded")
            return self.__connection

        except Exception as e:
            logger.exception("Database connection exception details:")
            raise RuntimeError(f"Failed to connect to database: {str(e)}")


    def close(self) -> None:
        """Close the current database connection.

        The method closes ``atlas.connection`` and releases DuckDB connection resources.
        Call ``atlas.connect()`` again if the database needs to be used after closing.

        Returns
        -------
        None
            The active connection is closed and cleared in place.

        Examples
        --------
        Close the connection after analysis::

            atlas = sap.Atlas("./data/pbmc")
            atlas.describe()
            atlas.close()
        """

        logger.info("Closing database connection")
        try:
            # Check whether a database connection exists.
            if self.__connection is not None:
                # Close the database connection.
                self.__connection.close()
                # Clear the connection object to avoid closing it twice.
                self.__connection = None
                logger.info("Database connection closed")
            else:
                logger.debug("No active database connection to close")

        except Exception as e:
            logger.exception("Database close exception details:")
            raise RuntimeError(f"Error while closing database connection: {str(e)}")


    def execute_sql(self, sql: str) -> DuckDBPyConnection | None:
        """Execute one SQL statement.
        This method is suitable for SQL operations such as creating tables,
        updating columns, deleting temporary tables, or other statements that may not need to return a DataFrame.
        Use ``atlas.query`` when a query result should be converted directly to a DataFrame.

        Parameters
        ----------
        sql
            SQL statement to execute. It can be used for table creation, field updates, temporary-table cleanup, and similar operations.

        Returns
        -------
        duckdb.DuckDBPyConnection or None
            DuckDB execution result for query-like statements. Non-query statements are committed and return ``None``.

        Examples
        --------
        Add a boolean filter column::

            atlas.execute_sql(
                "ALTER TABLE obs ADD COLUMN IF NOT EXISTS filter_custom BOOLEAN"
            )

        Fill missing values in an existing column::

            atlas.execute_sql(
                "UPDATE obs SET filter_custom = FALSE WHERE filter_custom IS NULL"
            )
        """

        # Check whether an active database connection exists.
        if self.__connection is None:
            logger.debug("No active database connection; creating a read-write connection automatically")
            # Automatically connect in read-write mode when no connection exists.
            self.connect("r+")
        # Execute the SQL statement.
        logger.debug("Executing SQL statement")
        result = self.__connection.execute(sql)

        # Return the result for query statements.
        sql_upper = sql.strip().upper()
        if sql_upper.startswith(('SELECT', 'SHOW', 'DESCRIBE', 'EXPLAIN')):
            logger.debug("SQL statement is query-like; returning result")
            return result
        else:
            # Commit the transaction for non-query statements.
            logger.debug("SQL statement is non-query; committing transaction")
            self.__connection.commit()
            return None


    def exists(self) -> bool:
        """Check whether the Atlas database file exists.
        This method only checks whether the file pointed to by ``atlas.file_path`` exists.
        It does not validate the internal database schema.

        Returns
        -------
        bool
            ``True`` if the database file exists, otherwise ``False``.

        Examples
        --------
        Check whether the database file has been created::

            atlas = sap.Atlas("./data/pbmc")
            atlas.exists()
        """

        exists = os.path.exists(self.file_path)
        logger.debug(f"Checking database file existence: {self.file_path} -> {exists}")
        return exists


    def query(self, query: str) -> pd.DataFrame:
        """Execute a SQL query and return a DataFrame.
        The query is executed on the active DuckDB connection and converted to ``pandas.DataFrame``.
        This is convenient for interactive inspection of ``obs``, ``var``, and result tables.

        Parameters
        ----------
        query
            SQL query to execute and return.

        Returns
        -------
        pandas.DataFrame
            Table containing the query, summary, or plotting data.

        Examples
        --------
        Inspect the number of cells::

            atlas.query("SELECT COUNT(*) AS n_cells FROM obs")

        Count cells by cluster::

            atlas.query(
                "SELECT kmeans, COUNT(*) AS n_cells FROM obs GROUP BY kmeans ORDER BY kmeans"
            )
        """

        logger.info("Querying database; return type is pandas")

        if self.__connection is None:
            self.connect("r+")
        result = self.connection.execute(query)
        df = result.df() # Convert the query result to a pandas DataFrame.
        return df


    def query_raw(self, query: str) -> DuckDBPyConnection:
        """Execute a SQL query and return the raw DuckDB result.
        Unlike ``atlas.query``, this method keeps the original DuckDB result object,
        making it suitable for subsequent calls such as ``fetchone``, ``fetchall``, or other DuckDB-native methods.

        Parameters
        ----------
        query
            SQL query to execute and return.

        Returns
        -------
        duckdb.DuckDBPyConnection or None
            DuckDB execution result object.

        Examples
        --------
        Read a single summary value::

            result = atlas.query_raw("SELECT COUNT(*) FROM obs")
            n_cells = result.fetchone()[0]
        """

        logger.info("Querying database; return type is DuckDB")

        if self.__connection is None:
            self.connect("r+")

        result = self.connection.execute(query)
        return result

    def _describe_text(self) -> str:
        """Build and return a summary string for the Atlas database."""

        if self.__connection is None:
            self.connect("r+")

        conn = self.__connection

        # 1. Database path.
        file_name = self.file_path

        # 1.1 DuckDB memory limit.
        db_memory_limit = self.db_memory_limit

        # 2. Query all tables.
        try:
            tables = [r[0] for r in conn.execute("SHOW TABLES").fetchall()]
        except Exception:
            tables = []

        table_names = ", ".join(tables) if len(tables) > 0 else "None"

        # 3. Query the number of cells in obs.
        if "obs" in tables:
            try:
                n_cells = conn.execute("SELECT COUNT(*) FROM obs").fetchone()[0]
            except Exception:
                n_cells = None
        else:
            n_cells = None

        # 4. Query the number of genes in var.
        if "var" in tables:
            try:
                n_genes = conn.execute("SELECT COUNT(*) FROM var").fetchone()[0]
            except Exception:
                n_genes = None
        else:
            n_genes = None

        # 5. Format output.
        def fmt(x: Any) -> str:
            """Format a count value with thousands separators."""
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

    def describe(self) -> None:
        """Print a summary of the table structure in the Atlas database.

        This method scans database tables and selected key fields to generate and
        print a readable database summary. It is useful for checking what an Atlas
        object contains after importing data or completing analysis.

        Returns
        -------
        None
            The summary is printed directly and no object is returned.

        Examples
        --------
        Print a database summary::

            atlas.describe()
        """

        print(self._describe_text())
        return None

    def __repr__(self) -> str:
        """Return a database summary string for the Atlas object."""
        return self._describe_text()

    def __str__(self) -> str:
        """Return a readable summary string for the Atlas object."""
        return self._describe_text()


    def head(self, table_name: str, n: int = 5) -> pd.DataFrame:
        """Return the first rows of a database table.

        This method checks whether the table exists, reads its columns,
        and returns the first ``n`` rows as a DataFrame.

        Parameters
        ----------
        table_name
            Name of the table to preview.
        n
            Number of rows to return. The default is ``5``.

        Returns
        -------
        pandas.DataFrame
            The first ``n`` rows of the requested table.

        Examples
        --------
        Preview the ``obs`` table::
            atlas.head("obs", n=5)

        Preview the ``var`` table::
            atlas.head("var")
        """

        if self.__connection is None:
            self.connect("r+")

        conn = self.__connection

        # 1. Check whether the table exists.
        tables = [r[0] for r in conn.execute("SHOW TABLES").fetchall()]

        if table_name not in tables:
            raise ValueError(
                f"Table does not exist in database: {table_name}\n"
                f"Available tables: {', '.join(tables) if len(tables) > 0 else 'None'}"
            )

        # 2. Safely quote the table name.
        table_sql = '"' + table_name.replace('"', '""') + '"'

        # 3. Query the first n rows.
        df = conn.execute(f"""
            SELECT *
            FROM {table_sql}
            LIMIT {int(n)}
        """).df()

        return df


    def _save_read_index_meta(
        self,
        *,
        cell_condition: str | None,
        gene_condition: str | None,
        use_hvg: bool,
        use_data: str,
    ) -> None:
        """Save the current read index construction parameters.

        This method saves the cell filtering condition, gene filtering condition,
        whether HVG is used, and the expression value column name used by the current
        ``build_read_index`` call into the ``atlas_read_index_meta`` table.
        It is used later to check the data range and expression layer corresponding
        to the current read index.

        If the ``atlas_read_index_meta`` table does not exist, it will be created automatically.
        If the table already exists, it will not be created again, and a message will be printed.

        Parameters
        ----------
        cell_condition
            The cell filtering column name used when constructing the current read index.
        gene_condition
            The gene filtering column name used when constructing the current read index.
        use_hvg
            Whether to additionally apply ``highly_variable_genes`` filtering.
        use_data
            The expression value column name read when constructing the filtered expression matrix.

        Returns
        -------
        None
            The result is written directly into the ``atlas_read_index_meta`` table.
        """

        if self.__connection is None:
            self.connect("r+")

        conn = self.__connection

        # First check whether the atlas_read_index_meta table already exists
        table_exists = conn.execute("""
                SELECT COUNT(*)
                FROM information_schema.tables
                WHERE table_name = 'atlas_read_index_meta'
            """).fetchone()[0]

        # Create the table only if it does not exist;
        # if it already exists, do not recreate it and print a message
        if table_exists == 0:
            conn.execute("""
                    CREATE TABLE atlas_read_index_meta (
                        key VARCHAR PRIMARY KEY,
                        value VARCHAR
                    )
                """)
        else:
            print(
                "The index table required for minibatch expression matrix reading has been rebuilt. "
                "Please rerun PCA, K-means, and other operations that depend on this table!")

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

    @duckdb_memory_limit("3G")
    def build_read_index(
        self,
        cell_condition: str | None = "filter_cells",
        gene_condition: str | None = "filter_genes",
        use_hvg: bool = True,
        use_data: str = "data_log1p",
    ) -> None:
        """Build the expression matrix read index.

        This method constructs the index tables required for subsequent minibatch
        expression matrix reading based on the cell filtering condition, gene filtering
        condition, and highly variable gene marker. PCA, K-means, and some plotting
        functions usually depend on this index.

        Parameters
        ----------
        cell_condition
            Boolean column name in the ``obs`` table used for cell filtering.
            Defaults to ``"filter_cells"``.
            If ``None``, cell filtering is not applied.
        gene_condition
            Boolean column name in the ``var`` table used for gene filtering.
            Defaults to ``"filter_genes"``.
            If ``None``, gene filtering is not applied.
        use_hvg
            Whether to additionally apply ``highly_variable_genes=TRUE`` on top of
            ``gene_condition``.
            If ``True``, the final gene set must satisfy both the gene filtering
            condition and the HVG condition.
        use_data
            Expression value column name read from the resolved expression source. Common
            values include ``"data_count"``, ``"data_normalize"``, ``"data_log1p"``,
            and ``"data_scale"``.

        Returns
        -------
        None
            The result is written directly into the Atlas database.

        Examples
        --------
        Build an index using the default filtering columns::

            sap.pp.filter_cells(atlas, min_genes=200)
            sap.pp.filter_genes(atlas, min_cells=3)
            atlas.build_read_index(cell_condition="filter_cells", gene_condition="filter_genes")

        Build a PCA input index using only highly variable genes::

            sap.pp.highly_variable_genes(atlas, n_top_genes=3000)
            atlas.build_read_index(use_hvg=True)"""

        if self.connection is None:
            self.connect("r+")

        builder = FilterIndexBuilder(
            self,
            cell_condition=cell_condition,
            gene_condition=gene_condition,
            use_hvg=use_hvg,
            use_data=use_data,
        )
        builder.run()

        # Persist the parameters used by this read index into the database
        self._save_read_index_meta(
            cell_condition=cell_condition,
            gene_condition=gene_condition,
            use_hvg=use_hvg,
            use_data=use_data,
        )

    def get_minibatch_csr(self, x_type: str = "CSR") -> Iterator[Any]:
        """Read the expression matrix in sparse CSR minibatches.

        This method returns sparse matrices batch by batch from the database based on
        the already constructed read index. It is suitable for algorithms that need
        streaming expression matrix processing.

        Parameters
        ----------
        x_type
            The returned minibatch matrix format. Common values include ``"CSR"`` or
            other sparse matrix formats supported by the underlying function.

        Yields
        -------
        ``X_batch`` is a sparse matrix in ``float32`` CSR format.

        Examples
        --------
        Iterate over CSR minibatches::

            for X_batch in atlas.get_minibatch_csr():
                print(X_batch.shape)
                break"""

        fetcher = MultiThreadedMinibatchFetcher(file_path=self.file_path, x_type=x_type)
        for X_batch in fetcher.run():
            yield X_batch

    def get_minibatch_dense(
        self,
        pass_mode: str = "single-pass",
        batch_size: int = 2048,
        max_batches: int | None = None,
        buffer_batch_num: int = 5,
        get_obs_col: str | None = None,
    ) -> Iterator[np.ndarray | dict[str, Any]]:
        """Read the expression matrix in dense minibatches.

        This method restores expression matrices batch by batch from
        ``X_HyS_data_filtered`` / ``X_HyS_indptr_filtered`` based on the filtered
        read index generated by ``atlas.build_read_index(...)``, and converts sparse
        expression records into ``float32`` dense arrays.
        It mainly supports algorithms that require dense input, such as
        streaming randomized PCA, ``MiniBatchKMeans``, streaming training, and
        large-scale batched inference.

        The return value is a generator that yields minibatches one by one,
        instead of loading all data into memory at once.
        By default, each iteration returns only one ``X_batch`` matrix.
        When ``get_obs_col`` is provided, the current batch's row-wise
        ``filter_cell_ids`` are also returned, and the specified column from the
        ``obs`` table, such as ``kmeans`` labels, is retrieved based on these
        ``filter_cell_ids``.

        Parameters
        ----------
        pass_mode
            Minibatch reading mode. Supported values are ``"single-pass"`` and
            ``"multi-pass"``.

            - ``"single-pass"``: Traverse the database once in the order of the
              current read index. This is suitable for evaluation, export, prediction,
              or workflows that require deterministic ordering.
            - ``"multi-pass"``: Recreate the reader for each pass and use
              ``ShuffleBuffer`` at the dense batch layer to produce randomized output.
              This is suitable for algorithms that require multiple rounds of random
              minibatch training. This mode should usually be used together with
              ``max_batches`` to control the total number of training batches.

        batch_size
            Number of cells contained in each minibatch. A larger value usually reduces
            the number of Python-level loops and improves throughput, but increases
            the memory usage of each dense batch. The output matrix shape is usually
            ``(current batch cell count, filtered gene count)``; the last batch may
            contain fewer cells than ``batch_size``.

        max_batches
            Maximum number of minibatches to read or output. If ``None``:

            - in ``single-pass`` mode, all batches in the current read index are traversed;
            - in ``multi-pass`` mode, the number of passes is not actively limited,
              so it is usually recommended to pass this parameter explicitly.

        buffer_batch_num
            Number of batches cached by ``ShuffleBuffer`` in ``multi-pass`` mode.
            The actual cell capacity of the shuffle buffer is approximately
            ``batch_size * buffer_batch_num``.
            A larger value increases the randomization range, but also increases
            the memory usage of the dense buffer.
            In ``single-pass`` mode, this parameter is still passed to the underlying
            reader but does not change the ordered output semantics.

        get_obs_col
            Column name from the ``obs`` table to return together with each minibatch,
            for example ``"kmeans"``.
            Defaults to ``None``, meaning no ``obs`` column is queried and each
            iteration only yields ``X_batch``.

            When a column name is provided, such as ``get_obs_col="kmeans"``, this
            method automatically asks the underlying minibatch reader to return
            ``filter_cell_ids``, then queries ``obs.kmeans`` according to these IDs,
            and returns a dictionary:

            ``{"X": X_batch, "filter_cell_ids": filter_cell_ids, "kmeans": values}``

            Here ``X_batch[i, :]``, ``filter_cell_ids[i]``, and ``values[i]`` have
            a one-to-one correspondence.

        Yields
        -------
        numpy.ndarray or dict
            When ``get_obs_col is None``, each iteration yields a dense ``numpy.ndarray``:

            ``X_batch``

            When ``get_obs_col`` is not ``None``, each iteration yields a dictionary:

            ``{"X": X_batch, "filter_cell_ids": filter_cell_ids, get_obs_col: values}``

            ``X_batch`` is a ``float32`` dense matrix;
            ``filter_cell_ids`` are the filtered cell IDs corresponding to each row
            in the current batch;
            ``values`` are the values of the specified column in the ``obs`` table.

        Examples
        --------
        Sequential reading for model fitting::

            for X_batch in atlas.get_minibatch_dense(batch_size=4096):
                print(X_batch.shape)
                break

        Return ``obs.kmeans`` labels together::

            for batch in atlas.get_minibatch_dense(
                batch_size=4096,
                get_obs_col="kmeans",
            ):
                X_batch = batch["X"]
                labels = batch["kmeans"]
                filter_cell_ids = batch["filter_cell_ids"]
                break

        Multi-pass random batch training with a limited number of total batches::

            for X_batch in atlas.get_minibatch_dense(
                pass_mode="multi-pass",
                batch_size=2048,
                buffer_batch_num=5,
                max_batches=100,
            ):
                model.partial_fit(X_batch)
        """

        if pass_mode not in ("single-pass", "multi-pass"):
            raise ValueError("pass_mode only supports 'single-pass' or 'multi-pass'")

        if get_obs_col is not None and not isinstance(get_obs_col, str):
            raise TypeError("get_obs_col must be str or None")

        if get_obs_col is not None and get_obs_col == "":
            raise ValueError("get_obs_col cannot be an empty string")

        # Internal parameter: get_obs_col depends on filter_cell_ids for mapping,
        # so it is automatically enabled when an obs column is provided.
        return_cell_ids = get_obs_col is not None

        obs_conn = None
        obs_col_sql = None

        def _quote_identifier(name: str) -> str:
            return '"' + name.replace('"', '""') + '"'

        def _attach_obs_col(batch: dict[str, Any]) -> dict[str, Any]:

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
                raise ValueError(
                    "The obs table is missing the filter_cell_id column. Please run atlas.build_read_index(...) first.")

            if get_obs_col not in obs_columns:
                obs_conn.close()
                raise ValueError(f"The obs table does not contain column: {get_obs_col}")

            if get_obs_col in {"X", "filter_cell_ids"}:
                obs_conn.close()
                raise ValueError(
                    "get_obs_col cannot be X or filter_cell_ids, to avoid overwriting reserved minibatch fields")

            obs_col_sql = _quote_identifier(get_obs_col)

        # 1. single-pass: run only once
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

            # 2. multi-pass: automatically loop over multiple passes
            produced_batches = 0
            pass_id = 0

            while True:

                # Stop if max_batches has been reached
                if max_batches is not None and produced_batches >= max_batches:
                    logger.info(f"[get_minibatch_dense] reach max_batches={max_batches}, stop")
                    break

                # Number of batches still needed in the current pass
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

                # Prevent an infinite loop if an abnormal empty pass occurs
                if pass_batches == 0:
                    logger.debug("[get_minibatch_dense] pass produced 0 batch, stop")
                    break
        finally:
            if obs_conn is not None:
                obs_conn.close()

    # =====================================================
    # io method wrappers
    # -----------------------------------------------------
    # Keep the original function position unchanged:
    #   load_h5ad(file_path, atlas, ...)
    #
    # Also support object-style calls:
    #   atlas.load_h5ad(file_path, ...)
    # =====================================================

    @duckdb_memory_limit("3G")
    def load_h5ad(
        self,
        h5ad_path: PathLike[str] | str | list[PathLike[str] | str],
        *,
        load_type: Literal["order", "random"] = "random",
        cells_per_block: int | None = None,
        import_window_memory_factor: float = 1.0,
        cell_name_col: str | None = None,
        gene_name_col: str | None = None,
    ) -> Any:
        """Import h5ad files into an Atlas database.

        This function is the unified entry point for importing h5ad data into Atlas.
        It can read a single ``.h5ad`` file or a list of multiple ``.h5ad`` files,
        and writes cell metadata, gene metadata, and the expression matrix into the
        Atlas DuckDB database.

        During import, it automatically dispatches to the corresponding implementation
        based on ``load_type`` and the type of ``h5ad_path``:

        - single file + ``"order"``: import in the original cell order;
        - single file + ``"random"``: import randomly using a shuffle-window strategy;
        - multiple files + ``"order"``: import according to the file list order and the
          original cell order within each file;
        - multiple files + ``"random"``: split multiple files into blocks and import them
          with global randomization.

        The expression matrix is stored uniformly on the count scale and written to
        the ``X_HyS_data.data_count`` field.
        If the input ``X`` is detected to be on the log scale, it is converted back
        to counts before writing.

        Parameters
        ----------
        h5ad_path
            Input ``.h5ad`` file path, or a list of multiple ``.h5ad`` file paths.

        load_type
            Import mode. Only ``"order"`` and ``"random"`` are supported.
            When ``h5ad_path`` is a single path, single-file ordered or random import
            is performed respectively.
            When ``h5ad_path`` is a list, multi-file ordered or random import is
            performed respectively.

        cells_per_block
            Number of cells contained in each contiguous cell block when reading and
            writing the expression matrix.
            If ``None``, a default value is automatically estimated based on the total
            number of cells.
        import_window_memory_factor
            Empirical scaling factor used to estimate the Python-side h5ad import
            window size. This does not change DuckDB's own memory limit.
        cell_name_col
            Column in ``adata.obs`` to use as ``atlas_cell_name``. If ``None``,
            the AnnData obs index is used.
        gene_name_col
            Column in ``adata.var`` to use as ``atlas_gene_name``. If ``None``,
            the AnnData var index is used.

        Returns
        -------
        Any
            Returns the result of the called underlying import function. Currently,
            this is mainly used to execute import side effects, and the return value
            is usually not relied upon.

        Notes
        -----
        ``random`` import reorders cells, so single-file random import does not import
        ``obsm`` by default, to avoid misalignment between embeddings and reordered ``obs``.
        ``order`` import preserves the original order, so ``obsm`` and ``varm`` can
        be safely imported.

        Examples
        --------
        Import a single h5ad file in order::

            atlas = sap.Atlas("./data/pbmc")
            atlas.load_h5ad("./data/pbmc.h5ad", load_type="order")

        Import multiple files with random blocks::

            atlas.load_h5ad(
                ["./data/batch1.h5ad", "./data/batch2.h5ad"],
                load_type="random",
                cells_per_block=1000,
            )
        """

        return _io_load_h5ad(
            h5ad_path,
            self,
            load_type=load_type,
            cells_per_block=cells_per_block,
            import_window_memory_factor=import_window_memory_factor,
            cell_name_col=cell_name_col,
            gene_name_col=gene_name_col,
        )

    def load_anndata(self, adata: AnnData) -> None:

        """Write an AnnData object into an Atlas database.

        This function directly accepts an in-memory AnnData object and writes its
        ``obs``, ``var``, ``X``, ``obsm``, and ``varm`` into the Atlas database.
        It is suitable for scenarios where data has already been read, filtered,
        or preprocessed using Scanpy or other tools and then needs to be managed
        by Atlas.

        Unlike the backed chunked import path of ``load_h5ad``, this function requires
        the AnnData object to already be in memory. Therefore, it is more suitable for
        small to medium-sized datasets or already sampled datasets.

        Parameters
        ----------
        adata
            AnnData object. The function writes its ``obs``, ``var``, expression matrix,
            and supported results into the Atlas database.
        Returns
        -------
        None
            The result is written directly into the Atlas database and no object is returned.

        Notes
        -----
        This path rebuilds the ``obs``, ``var``, ``X_HyS_indptr``, and ``X_HyS_data``
        tables, and writes two-dimensional arrays in ``obsm`` and ``varm`` into
        ``obsm_*`` and ``varm_*`` tables.

        Examples
        --------
        Read with Scanpy and import::

            adata = sc.read_h5ad("./data/pbmc.h5ad")
            atlas = sap.Atlas("./data/pbmc")
            atlas.load_anndata(adata)
        """

        _io_load_anndata(adata, self)

    def load_multi_format(self, file_path: PathLike[str] | str) -> None:

        """Import data into Atlas according to the file format.

        This function is the import entry point for small or general-format data.
        It first calls ``_read_smart`` according to the suffix of ``file_path`` to read
        the data into an in-memory AnnData object, and then calls ``load_anndata`` to
        write it into the Atlas database.

        Unlike the backed chunked import path of ``load_h5ad``, this function first
        loads the full data into memory. Therefore, it is more suitable for small files,
        temporary conversion, or non-h5ad format data.

        Parameters
        ----------
        file_path
            Input file path. The function selects an appropriate reading method based
            on the file format.
        Returns
        -------
        None
            The result is written directly into the Atlas database and no object is returned.

        Notes
        -----
        Supported reading formats are determined by ``_read_smart``, including common
        formats such as h5ad, loom, Matrix Market, csv, txt/tsv, Excel, 10x h5,
        and UMI-tools.

        Examples
        --------
        Automatically detect and import a file::

            atlas = sap.Atlas("./data/pbmc")
            atlas.load_multi_format("./data/pbmc.h5ad")
        """

        _io_load_multi_format(file_path, self)

    def rename_duplicated_genes(
        self,
        gene_name_column: str = "atlas_gene_name",
    ) -> bool:

        """Check whether gene names in the Atlas database are duplicated.

        This function reads the gene name column in the ``var`` table, checks whether
        duplicate gene names exist, and appends suffixes such as ``_1`` and ``_2`` to
        duplicated entries. Duplicate gene names may affect plotting by gene name,
        differential gene display, and AnnData export, so this function can be run
        after import for cleanup.

        For each duplicated gene name, the first occurrence remains unchanged, and
        subsequent duplicated entries are renamed in the form ``original_name_1`` and
        ``original_name_2``.

        Parameters
        ----------
        gene_name_column
            The ``var`` column name that stores gene names. Usually
            ``"atlas_gene_name"``.

        Returns
        -------
        bool
            Returns ``True`` when checking or updating succeeds; returns ``False`` if
            the ``var`` table does not exist.

        Notes
        -----
        The ``var`` table is updated only when duplicated gene names are detected.
        If no duplicates are found, the original table remains unchanged.

        Examples
        --------
        Check the default gene name column::

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

        """Export an Atlas database to an h5ad file.

        This function reads ``obs``, ``var``, the sparse expression matrix,
        ``obsm_*`` result tables, and ``varm_*`` result tables from the Atlas DuckDB
        database, and writes them out as a standard AnnData ``.h5ad`` file for
        continued analysis in Scanpy or other tools that support AnnData.

        The expression matrix is reassembled into the h5ad CSR ``X`` according to
        the internal Atlas HyS sparse structure. ``X.data`` comes from the
        expression source resolved by ``use_data``; this can be ``data_count``
        in ``X_HyS_data`` or a derived expression table. ``X.indices`` comes
        from ``atlas_gene_id``.

        Parameters
        ----------
        out_h5ad_path
            Output ``.h5ad`` file path.

        batch_cells
            Number of nonzero expression records processed per batch when writing
            expression matrix ``data`` and ``indices``.
            A larger value is usually faster, but increases per-batch memory usage.

        use_data
            Expression value field exported from the resolved expression source.
            The default is ``"data_count"``. Existing fields such as
            ``"data_log1p"`` and ``"data_normalize"`` can also be used after the
            corresponding preprocessing steps.

        Returns
        -------
        None
            The result is written directly to the h5ad file specified by
            ``out_h5ad_path`` and no object is returned.

        Notes
        -----
        ``obsm_*`` tables are exported to h5ad ``obsm``. The ``obsm_`` prefix in the
        table name is removed.
        ``varm_*`` tables are exported to h5ad ``varm``. The ``varm_`` prefix in the
        table name is removed.

        Before export, the function resolves ``use_data`` from the base expression
        table or a derived expression table. If it cannot be resolved, an error is
        raised directly to avoid exporting an empty matrix or an incorrect field.

        Examples
        --------
        Export the current database::

            atlas.write_h5ad("./data/pbmc_export.h5ad")

        Use the object-style API and reduce per-batch memory usage::

            atlas.write_h5ad("./data/pbmc_export.h5ad", batch_cells=200_000)

        Export the log1p expression matrix::

            atlas.write_h5ad("./data/pbmc_log1p.h5ad", use_data="data_log1p")"""

        _io_write_h5ad(
            self,
            out_h5ad_path,
            batch_cells=batch_cells,
            use_data=use_data,
        )

    def get_obs_df(
        self,
        columns: list[str] | str | None = None,
    ) -> pd.DataFrame:

        """Read the obs table from the Atlas database.

        This function reads all columns or selected columns from the ``obs`` table
        into a pandas DataFrame. It is suitable for quickly checking cell metadata,
        exporting statistical results, or merging with external analysis results.
        The returned result uses ``atlas_cell_id`` as the pandas index while also
        preserving the ``atlas_cell_id`` column itself.

        Parameters
        ----------
        columns
            Column names to read from ``obs``. This can be a single string, a list of
            strings, or ``None``.
            If ``None``, all columns are read.

        Returns
        -------
        pandas.DataFrame
            Query result from ``obs``. The default index is ``atlas_cell_id``.

        Notes
        -----
        Even if ``atlas_cell_id`` is not explicitly included in ``columns``, the
        function automatically places ``atlas_cell_id`` as the first column to set
        the DataFrame index.

        Examples
        --------
        Read all obs information::

            obs = atlas.get_obs_df()

        Read only clustering and automatic annotation columns::

            obs = atlas.get_obs_df(columns=["kmeans", "cell_type_auto"])"""

        return _io_get_obs_df(self, columns=columns)

    def get_var_df(
        self,
        columns: list[str] | str | None = None,
    ) -> pd.DataFrame:

        """Read the var table from the Atlas database.

        This function reads all columns or selected columns from the ``var`` table
        into a pandas DataFrame. It is suitable for checking gene metadata,
        exporting gene-level statistics, or aligning external gene-level results.
        The returned result uses ``atlas_gene_id`` as the pandas index while also
        preserving the ``atlas_gene_id`` column itself.

        Parameters
        ----------
        columns
            Column names to read from ``var``. This can be a single string, a list of
            strings, or ``None``.
            If ``None``, all columns are read.

        Returns
        -------
        pandas.DataFrame
            Query result from ``var``. The default index is ``atlas_gene_id``.

        Notes
        -----
        Even if ``atlas_gene_id`` is not explicitly included in ``columns``, the
        function automatically places ``atlas_gene_id`` as the first column to set
        the DataFrame index.

        Examples
        --------
        Read all var information::

            var = atlas.get_var_df()

        Read only selected gene-level columns::

            var = atlas.get_var_df(columns=["atlas_gene_name", "highly_variable_genes"])
        """

        return _io_get_var_df(self, columns=columns)

    def get_obsm_df(
        self,
        table_name: str,
        atlas_cell_id: list[int] | np.ndarray | pd.Series | None = None,
        columns: list[str] | str | None = None,
    ) -> pd.DataFrame:

        """Read an ``obsm_*`` table from the Atlas database.

        This function reads cell-level multidimensional results, such as
        ``obsm_X_pca`` or ``obsm_X_umap``, into a pandas DataFrame. The returned
        result uses ``atlas_cell_id`` as the pandas index while also preserving
        the ``atlas_cell_id`` column itself.

        Parameters
        ----------
        table_name
            Name of the ``obsm_*`` table to read, for example ``"obsm_X_pca"``
            or ``"obsm_X_umap"``.
        atlas_cell_id
            Optional Atlas cell IDs to select. If ``None``, all rows in the
            table are returned ordered by ``atlas_cell_id``. If a list or array
            is passed, the output follows the order of the provided IDs.
        columns
            Value columns to read from the ``obsm_*`` table. This can be a
            single string, a list of strings, or ``None``. If ``None``, all
            columns are read.

        Returns
        -------
        pandas.DataFrame
            Query result indexed by ``atlas_cell_id``.

        Examples
        --------
        Read all PCA coordinates::

            pca = atlas.get_obsm_df("obsm_X_pca")

        Read selected UMAP coordinates in a specific cell order::

            umap = atlas.get_obsm_df(
                "obsm_X_umap",
                atlas_cell_id=[10, 2, 7],
                columns=["umap1", "umap2"],
            )
        """

        return _io_get_obsm_df(
            self,
            table_name=table_name,
            atlas_cell_id=atlas_cell_id,
            columns=columns,
        )

    def get_varm_df(
        self,
        table_name: str,
        atlas_gene_id: list[int] | np.ndarray | pd.Series | None = None,
        columns: list[str] | str | None = None,
    ) -> pd.DataFrame:

        """Read a ``varm_*`` table from the Atlas database.

        This function reads gene-level multidimensional results, such as
        ``varm_PCs``, into a pandas DataFrame. The returned result uses
        ``atlas_gene_id`` as the pandas index while also preserving the
        ``atlas_gene_id`` column itself.

        Parameters
        ----------
        table_name
            Name of the ``varm_*`` table to read, for example ``"varm_PCs"``.
        atlas_gene_id
            Optional Atlas gene IDs to select. If ``None``, all rows in the
            table are returned ordered by ``atlas_gene_id``. If a list or array
            is passed, the output follows the order of the provided IDs.
        columns
            Value columns to read from the ``varm_*`` table. This can be a
            single string, a list of strings, or ``None``. If ``None``, all
            columns are read.

        Returns
        -------
        pandas.DataFrame
            Query result indexed by ``atlas_gene_id``.

        Examples
        --------
        Read all PCA loadings::

            pcs = atlas.get_varm_df("varm_PCs")

        Read selected PCs in a specific gene order::

            pcs = atlas.get_varm_df(
                "varm_PCs",
                atlas_gene_id=[10, 2, 7],
                columns=["pc0", "pc1"],
            )
        """

        return _io_get_varm_df(
            self,
            table_name=table_name,
            atlas_gene_id=atlas_gene_id,
            columns=columns,
        )

    def get_uns_df(
        self,
        table_name: str,
        columns: list[str] | str | None = None,
    ) -> pd.DataFrame:

        """Read an ``uns_*`` table from the Atlas database.

        This function reads unstructured analysis result tables, such as
        ``uns_pca_stats`` or ``uns_umap_params``, into a pandas DataFrame.

        Parameters
        ----------
        table_name
            Name of the ``uns_*`` table to read, for example
            ``"uns_pca_stats"``.
        columns
            Columns to read from the ``uns_*`` table. This can be a single
            string, a list of strings, or ``None``. If ``None``, all columns
            are read.

        Returns
        -------
        pandas.DataFrame
            Query result from the requested ``uns_*`` table.

        Examples
        --------
        Read PCA explained-variance statistics::

            pca_stats = atlas.get_uns_df("uns_pca_stats")

        Read stored UMAP parameters::

            umap_params = atlas.get_uns_df(
                "uns_umap_params",
                columns=["param_name", "param_value"],
            )
        """

        return _io_get_uns_df(
            self,
            table_name=table_name,
            columns=columns,
        )

    def get_anndata(
        self,
        atlas_cell_ids: list[int] | np.ndarray | None,
        use_data: str = "data_count",
        include_obsm: bool = True,
        include_varm: bool = True,
    ) -> AnnData:

        """Construct an AnnData object from the Atlas database.

        This function exports an in-memory AnnData object from the Atlas database
        according to the user-provided ``atlas_cell_ids``. It preserves the order of
        the input cell IDs, reads the corresponding ``obs`` subset, the full ``var``,
        a sparse CSR ``X`` composed from the specified expression representation,
        and optionally reads ``obsm_*`` and ``varm_*`` result tables.

        This function is suitable for small-scale sampling export, local Scanpy analysis,
        model checking, or temporarily converting a group of cells in Atlas back to AnnData.

        Parameters
        ----------
        atlas_cell_ids
            List of Atlas cell IDs to export. It cannot be empty and cannot contain
            duplicate values.

            The order of cells in the returned AnnData object will be the same as
            the order of this list.

        use_data
            Expression field read from the resolved expression source. Common
            values include ``"data_count"``, ``"data_normalize"``,
            ``"data_log1p"``, and ``"data_scale"``.

        include_obsm
            Whether to write ``obsm_*`` result tables into the returned AnnData object.

        include_varm
            Whether to write ``varm_*`` result tables into the returned AnnData object.

        Returns
        -------
        AnnData
            AnnData object constructed from the Atlas database.

        Notes
        -----
        ``obsm_*`` tables are exported by left-joining according to the selected cell
        order. If some cells do not have embeddings, the corresponding positions are
        written as ``NaN``. ``varm_*`` tables are exported in full gene order.

        This function creates a temporary table ``_selected_cells`` to preserve the
        order of user-provided cells.

        Examples
        --------
        Export specified cells::

            cell_ids = [0, 1, 2, 3]
            adata = atlas.get_anndata(cell_ids, use_data="data_log1p")

        Export the first 5000 filtered cells and include UMAP/PCA::

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
