from __future__ import annotations

from dataclasses import dataclass
from _duckdb import DuckDBPyConnection


def quote_identifier(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def has_table(conn: DuckDBPyConnection, table_name: str) -> bool:
    return conn.execute(
        """
        SELECT 1
        FROM information_schema.tables
        WHERE table_name = ?
        LIMIT 1
        """,
        [table_name],
    ).fetchone() is not None


def has_column(conn: DuckDBPyConnection, table_name: str, column_name: str) -> bool:
    return conn.execute(
        """
        SELECT 1
        FROM information_schema.columns
        WHERE table_name = ?
          AND column_name = ?
        LIMIT 1
        """,
        [table_name, column_name],
    ).fetchone() is not None


def derived_data_table_name(data_name: str) -> str:
    return f"X_HyS_data_{data_name}"


@dataclass(frozen=True)
class ExpressionSource:
    from_sql: str
    value_sql: str
    id_sql: str
    cell_sql: str
    gene_sql: str
    source_table: str


def resolve_expression_source(
    conn: DuckDBPyConnection,
    data_name: str,
    *,
    x_alias: str = "x",
    d_alias: str = "d",
) -> ExpressionSource:
    """Resolve an expression representation from X_HyS_data or a derived table."""

    data_sql = quote_identifier(data_name)
    if has_column(conn, "X_HyS_data", data_name):
        return ExpressionSource(
            from_sql=f"X_HyS_data AS {x_alias}",
            value_sql=f"{x_alias}.{data_sql}",
            id_sql=f"{x_alias}.id",
            cell_sql=f"{x_alias}.atlas_cell_id",
            gene_sql=f"{x_alias}.atlas_gene_id",
            source_table="X_HyS_data",
        )

    table_name = derived_data_table_name(data_name)
    table_sql = quote_identifier(table_name)
    if not (has_table(conn, table_name) and has_column(conn, table_name, data_name)):
        raise ValueError(
            f"Expression field does not exist: {data_name}. "
            f"Expected X_HyS_data.{data_name} or {table_name}.{data_name}."
        )

    if has_column(conn, table_name, "atlas_cell_id") and has_column(conn, table_name, "atlas_gene_id"):
        return ExpressionSource(
            from_sql=f"{table_sql} AS {x_alias}",
            value_sql=f"{x_alias}.{data_sql}",
            id_sql=f"{x_alias}.id",
            cell_sql=f"{x_alias}.atlas_cell_id",
            gene_sql=f"{x_alias}.atlas_gene_id",
            source_table=table_name,
        )

    if not has_column(conn, table_name, "id"):
        raise ValueError(
            f"Derived expression table {table_name} must contain id, or both "
            "atlas_cell_id and atlas_gene_id."
        )

    return ExpressionSource(
        from_sql=f"X_HyS_data AS {x_alias} JOIN {table_sql} AS {d_alias} ON {x_alias}.id = {d_alias}.id",
        value_sql=f"{d_alias}.{data_sql}",
        id_sql=f"{x_alias}.id",
        cell_sql=f"{x_alias}.atlas_cell_id",
        gene_sql=f"{x_alias}.atlas_gene_id",
        source_table=table_name,
    )
