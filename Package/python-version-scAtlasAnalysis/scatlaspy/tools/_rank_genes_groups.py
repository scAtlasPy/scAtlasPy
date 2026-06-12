from datetime import datetime
import os
import gc
import numpy as np
import pandas as pd
from scipy import stats
from typing import Any
from ..data import Atlas


def _q(name: str) -> str:
    """为 SQL 标识符添加安全引用。

    该内部函数属于差异表达模块，用于支撑同一模块中的公共 API。

    按分组聚合表达统计量，计算 t-test、log fold change、校正 p 值和 marker 排名。

    它通常不会作为用户入口直接调用；直接调用时需要保证输入对象、数据库连接和相关临时表已经由上游步骤准备好。

    Parameters
    ----------
    name
        对象名称、列名或 SQL 标识符，具体含义由调用位置决定。

    Returns
    -------
    quoted_name
        加双引号后的 SQL 标识符。

    Notes
    -----
    这是内部 helper；除非需要扩展 scAtlasPy 内部流程，一般不建议在用户代码中直接调用。
    """
    return '"' + str(name).replace('"', '""') + '"'


def _p_adjust_bh(pvals: np.ndarray) -> np.ndarray:
    """校正多重检验 p 值。

    该内部函数属于差异表达模块，用于支撑同一模块中的公共 API。

    按分组聚合表达统计量，计算 t-test、log fold change、校正 p 值和 marker 排名。

    它通常不会作为用户入口直接调用；直接调用时需要保证输入对象、数据库连接和相关临时表已经由上游步骤准备好。

    Parameters
    ----------
    pvals
        待校正或处理的 p 值数组。

    Returns
    -------
    result
        函数返回结果。具体类型取决于参数设置和内部执行路径。

    Notes
    -----
    这是内部 helper；除非需要扩展 scAtlasPy 内部流程，一般不建议在用户代码中直接调用。
    """
    pvals = np.asarray(pvals, dtype=np.float64)

    out = np.full(len(pvals), np.nan, dtype=np.float64)
    valid = np.isfinite(pvals)

    if valid.sum() == 0:
        return out

    p = pvals[valid]
    m = len(p)

    order = np.argsort(p)
    ranked = p[order]

    adj = ranked * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)

    tmp = np.empty_like(adj)
    tmp[order] = adj

    out[valid] = tmp
    return out


def _p_adjust_bonferroni(pvals: np.ndarray) -> np.ndarray:
    """校正多重检验 p 值。

    该内部函数属于差异表达模块，用于支撑同一模块中的公共 API。

    按分组聚合表达统计量，计算 t-test、log fold change、校正 p 值和 marker 排名。

    它通常不会作为用户入口直接调用；直接调用时需要保证输入对象、数据库连接和相关临时表已经由上游步骤准备好。

    Parameters
    ----------
    pvals
        待校正或处理的 p 值数组。

    Returns
    -------
    result
        函数返回结果。具体类型取决于参数设置和内部执行路径。

    Notes
    -----
    这是内部 helper；除非需要扩展 scAtlasPy 内部流程，一般不建议在用户代码中直接调用。
    """
    pvals = np.asarray(pvals, dtype=np.float64)

    out = np.full(len(pvals), np.nan, dtype=np.float64)
    valid = np.isfinite(pvals)

    if valid.sum() == 0:
        return out

    out[valid] = np.clip(pvals[valid] * valid.sum(), 0, 1)
    return out


def _compute_ttest_from_summary(df: pd.DataFrame) -> pd.DataFrame:
    """根据汇总统计量计算检验结果。

    该内部函数属于差异表达模块，用于支撑同一模块中的公共 API。

    按分组聚合表达统计量，计算 t-test、log fold change、校正 p 值和 marker 排名。

    它通常不会作为用户入口直接调用；直接调用时需要保证输入对象、数据库连接和相关临时表已经由上游步骤准备好。

    Parameters
    ----------
    df
        包含中间统计量或绘图数据的 pandas DataFrame。

    Returns
    -------
    result
        函数返回结果。具体类型取决于参数设置和内部执行路径。

    Notes
    -----
    这是内部 helper；除非需要扩展 scAtlasPy 内部流程，一般不建议在用户代码中直接调用。
    """

    n_in = df["n_in"].to_numpy(dtype=np.float64)
    n_ref = df["n_ref"].to_numpy(dtype=np.float64)

    sum_in = df["sum_in"].to_numpy(dtype=np.float64)
    sum_ref = df["sum_ref"].to_numpy(dtype=np.float64)

    sumsq_in = df["sumsq_in"].to_numpy(dtype=np.float64)
    sumsq_ref = df["sumsq_ref"].to_numpy(dtype=np.float64)

    # -------------------------------------------------
    # 1. mean
    # -------------------------------------------------
    mean_in = np.divide(
        sum_in,
        n_in,
        out=np.zeros_like(sum_in, dtype=np.float64),
        where=n_in > 0,
    )

    mean_ref = np.divide(
        sum_ref,
        n_ref,
        out=np.zeros_like(sum_ref, dtype=np.float64),
        where=n_ref > 0,
    )

    # -------------------------------------------------
    # 2. sample variance
    #    CSR 中未出现的值按 0 处理；
    #    sum / sumsq 是非零项聚合，但 n_in / n_ref 是总细胞数。
    # -------------------------------------------------
    var_in = np.zeros_like(mean_in, dtype=np.float64)
    var_ref = np.zeros_like(mean_ref, dtype=np.float64)

    mask_in = n_in > 1
    mask_ref = n_ref > 1

    var_in[mask_in] = (
        sumsq_in[mask_in] - (sum_in[mask_in] ** 2) / n_in[mask_in]
    ) / (n_in[mask_in] - 1)

    var_ref[mask_ref] = (
        sumsq_ref[mask_ref] - (sum_ref[mask_ref] ** 2) / n_ref[mask_ref]
    ) / (n_ref[mask_ref] - 1)

    var_in = np.maximum(var_in, 0.0)
    var_ref = np.maximum(var_ref, 0.0)

    # -------------------------------------------------
    # 3. Welch t-test score
    # -------------------------------------------------
    a = np.divide(
        var_in,
        n_in,
        out=np.zeros_like(var_in, dtype=np.float64),
        where=n_in > 0,
    )

    b = np.divide(
        var_ref,
        n_ref,
        out=np.zeros_like(var_ref, dtype=np.float64),
        where=n_ref > 0,
    )

    se2 = a + b
    se = np.sqrt(se2)

    scores = np.divide(
        mean_in - mean_ref,
        se,
        out=np.zeros_like(mean_in, dtype=np.float64),
        where=se > 0,
    )

    # -------------------------------------------------
    # 4. Welch-Satterthwaite df
    # -------------------------------------------------
    numerator = se2 ** 2

    denominator = (
        np.divide(
            a ** 2,
            n_in - 1,
            out=np.zeros_like(a, dtype=np.float64),
            where=n_in > 1,
        )
        +
        np.divide(
            b ** 2,
            n_ref - 1,
            out=np.zeros_like(b, dtype=np.float64),
            where=n_ref > 1,
        )
    )

    dof = np.divide(
        numerator,
        denominator,
        out=np.ones_like(numerator, dtype=np.float64),
        where=denominator > 0,
    )

    pvals = 2.0 * stats.t.sf(np.abs(scores), dof)

    # se=0 或 dof 异常时，设为不显著
    pvals[~np.isfinite(pvals)] = 1.0

    df["mean_in"] = mean_in
    df["mean_ref"] = mean_ref
    df["var_in"] = var_in
    df["var_ref"] = var_ref
    df["scores"] = scores
    df["pvals"] = pvals

    return df


def rank_genes_groups(
        atlas: Atlas,
        groupby: str = "kmeans",
        use_data: str = "data_log1p",
        groups: list | None = None,
        reference: str | int = "rest",
        n_genes: int | None = None,
        mask_var: str | None = None, # = "highly_variable_genes"
        corr_method: str = "benjamini-hochberg",
        rankby_abs: bool = False,
        add_table: str = "rank_genes_groups",
        input_is_log: bool = True,
        lfc_eps: float = 1e-9,  # 对齐 Scanpy 的 logFC 伪计数
        inplace: bool = True,
        return_df: bool = True,
):
    """按细胞分组计算 marker gene 排名。

    该函数按 ``groupby`` 指定的 ``obs`` 分组，对每个 cluster 和参考组执行差异表达统计，并把结果写入数据库表。它类似 Scanpy 的 ``sc.tl.rank_genes_groups``，但返回结果和持久化结果都面向 Atlas 数据库。

    Parameters
    ----------
    atlas
        Atlas 对象。通常需要已经连接到 DuckDB 数据库，并包含该函数读取或写入所需的 ``obs``、``var``、表达矩阵或结果表。
    groupby
        ``obs`` 中的分组列名，例如 ``"kmeans"``、``"leiden"`` 或 ``"cell_type"``。
    use_data
        读取的表达矩阵或结果表名称。常用值包括 ``"data"``、``"data_normalize"``、``"data_log1p"`` 和
        ``"data_scale"``。
    groups
        需要计算、展示或保留的分组列表。为 ``None`` 时使用全部分组。
    reference
        差异分析参考组。``"rest"`` 表示与其他所有细胞比较。
    n_genes
        每个分组保留或展示的基因数量。为 ``None`` 时保留全部可用基因。
    mask_var
        用于限制参与分析基因范围的 ``var`` 列名、布尔数组或条件。
    corr_method
        多重检验校正方法。常用值为 ``"benjamini-hochberg"``。
    rankby_abs
        是否按统计量绝对值排序。
    add_table
        写入数据库的结果表名。
    input_is_log
        输入表达矩阵是否已经做过对数变换。
    lfc_eps
        计算 log fold change 时加入的极小值，用于避免除零。
    inplace
        是否把结果写回 Atlas 数据库。
    return_df
        是否返回结果 DataFrame。

    Returns
    -------
    pandas.DataFrame 或 None
        当 ``return_df=True`` 时返回差异基因结果；否则结果仅写入数据库。

    Examples
    --------
    基于 K-means cluster 计算 marker genes::

        result = sap.tl.rank_genes_groups(atlas, groupby="kmeans")

    只计算指定 cluster，并保留每组前 100 个基因::

        result = sap.tl.rank_genes_groups(
            atlas,
            groupby="kmeans",
            groups=["0", "1", "2"],
            n_genes=100,
            add_table="rank_genes_groups_kmeans_top100",
        )

    计算后直接绘图和自动注释::

        sap.pl.rank_genes_groups(atlas, use_table="rank_genes_groups")
        summary_df, score_df = sap.tl.annotate_clusters(atlas, groupby="kmeans")"""

    method = "t-test"

    start = datetime.now()

    conn = atlas.connection
    if conn is None:
        atlas.connect("r+")
        conn = atlas.connection

    try:
        conn.execute(f"PRAGMA threads={os.cpu_count() or 1}")
    except Exception:
        pass

    temp_tables = [
        "_rgg_gene_set",
        "_rgg_group_stats",
        "_rgg_group_n",
        "_rgg_total_stats",
    ]

    try:
        # -------------------------------------------------
        # 0. 基础检查
        # -------------------------------------------------
        obs_cols = [r[1] for r in conn.execute("PRAGMA table_info(obs)").fetchall()]
        var_cols = [r[1] for r in conn.execute("PRAGMA table_info(var)").fetchall()]
        x_cols = [r[1] for r in conn.execute("PRAGMA table_info(X_HyS_data)").fetchall()]

        if groupby not in obs_cols:
            raise ValueError(f"obs 中不存在列: {groupby}")

        if use_data not in x_cols:
            raise ValueError(f"X_HyS_data 中不存在字段: {use_data}")

        if mask_var is not None and mask_var not in var_cols:
            raise ValueError(f"var 中不存在列: {mask_var}")

        if corr_method not in {"benjamini-hochberg", "bonferroni"}:
            raise ValueError("corr_method 只支持 'benjamini-hochberg' 或 'bonferroni'")

        # -------------------------------------------------
        # 1. 候选基因集合
        # -------------------------------------------------
        for t in temp_tables:
            conn.execute(f"DROP TABLE IF EXISTS {t}")

        if mask_var is None:
            var_where = "1=1"
        else:
            var_where = f"COALESCE({_q(mask_var)}, FALSE)=TRUE"

        # 保留 atlas_gene_name
        # 如果你的 var 里有 gene_name，并且想优先显示 gene symbol，
        # 可以把 atlas_gene_name 换成 COALESCE(gene_name, atlas_gene_name)
        if "gene_name" in var_cols:
            gene_name_expr = "COALESCE(gene_name, atlas_gene_name)"
        else:
            gene_name_expr = "atlas_gene_name"

        conn.execute(f"""
            CREATE TEMP TABLE _rgg_gene_set AS
            SELECT
                atlas_gene_id,
                {gene_name_expr} AS atlas_gene_name
            FROM var
            WHERE {var_where}
        """)

        gene_count = conn.execute("""
            SELECT COUNT(*)
            FROM _rgg_gene_set
        """).fetchone()[0]

        if gene_count == 0:
            raise ValueError("候选基因集合为空，请检查 mask_var 设置")

        # -------------------------------------------------
        # 2. group 信息
        # -------------------------------------------------
        group_df = conn.execute(f"""
            SELECT
                CAST({_q(groupby)} AS TEXT) AS group_name,
                COUNT(*) AS n_cells
            FROM obs
            WHERE {_q(groupby)} IS NOT NULL
            GROUP BY 1
            ORDER BY 1
        """).fetchdf()

        if len(group_df) == 0:
            raise ValueError(f"obs.{groupby} 中没有可用分组")

        all_groups = group_df["group_name"].astype(str).tolist()

        def _group_sort_key(x: Any):
            """生成分组或标签的自然排序键。

            该内部函数属于差异表达模块，用于支撑同一模块中的公共 API。

            按分组聚合表达统计量，计算 t-test、log fold change、校正 p 值和 marker 排名。

            它通常不会作为用户入口直接调用；直接调用时需要保证输入对象、数据库连接和相关临时表已经由上游步骤准备好。

            Parameters
            ----------
            x
                需要排序、格式化或转换的单个输入值。

            Returns
            -------
            sort_key
                可用于自然排序的键。

            Notes
            -----
            这是内部 helper；除非需要扩展 scAtlasPy 内部流程，一般不建议在用户代码中直接调用。
            """
            try:
                return (0, int(x))
            except Exception:
                try:
                    return (0, float(x))
                except Exception:
                    return (1, str(x))

        all_groups = sorted(all_groups, key=_group_sort_key)

        if groups is None:
            group_list = all_groups
        else:
            wanted = {str(g) for g in groups}
            group_list = [g for g in all_groups if str(g) in wanted]

        if len(group_list) == 0:
            raise ValueError("groups 过滤后没有可用 group")

        if reference != "rest":
            reference = str(reference)

            if reference not in set(all_groups):
                raise ValueError(
                    f"reference={reference!r} 不在 obs.{groupby} 中，"
                    f"可用 groups={all_groups}"
                )

            group_list = [g for g in group_list if str(g) != str(reference)]

            if len(group_list) == 0:
                raise ValueError("去掉 reference 后 groups 为空")

        # -------------------------------------------------
        # 3. 一次性聚合 group × gene 统计
        # -------------------------------------------------
        conn.execute(f"""
            CREATE TEMP TABLE _rgg_group_stats AS
            SELECT
                CAST(o.{_q(groupby)} AS TEXT) AS group_name,
                x.atlas_gene_id,
                SUM(x.{_q(use_data)}) AS sum_expr,
                SUM(x.{_q(use_data)} * x.{_q(use_data)}) AS sumsq_expr,
                COUNT(*) AS nnz
            FROM X_HyS_data x
            JOIN obs o
              ON x.atlas_cell_id = o.atlas_cell_id
            JOIN _rgg_gene_set gs
              ON x.atlas_gene_id = gs.atlas_gene_id
            WHERE o.{_q(groupby)} IS NOT NULL
              AND x.{_q(use_data)} IS NOT NULL
            GROUP BY 1, 2
        """)

        conn.execute(f"""
            CREATE TEMP TABLE _rgg_group_n AS
            SELECT
                CAST({_q(groupby)} AS TEXT) AS group_name,
                COUNT(*) AS n_cells
            FROM obs
            WHERE {_q(groupby)} IS NOT NULL
            GROUP BY 1
        """)

        # -------------------------------------------------
        # 4. reference='rest' 时，预先计算所有 group 总和
        # -------------------------------------------------
        if reference == "rest":
            total_cells = int(group_df["n_cells"].sum())

            conn.execute("""
                CREATE TEMP TABLE _rgg_total_stats AS
                SELECT
                    gs.atlas_gene_id,
                    gs.atlas_gene_name,
                    COALESCE(SUM(g.sum_expr), 0.0) AS sum_total,
                    COALESCE(SUM(g.sumsq_expr), 0.0) AS sumsq_total,
                    COALESCE(SUM(g.nnz), 0) AS nnz_total
                FROM _rgg_gene_set gs
                LEFT JOIN _rgg_group_stats g
                  ON gs.atlas_gene_id = g.atlas_gene_id
                GROUP BY 1, 2
            """)
        else:
            total_cells = None

        # -------------------------------------------------
        # 5. 逐 group 计算 rank_genes_groups 结果
        # -------------------------------------------------
        result_list = []

        for grp in group_list:

            if reference == "rest":
                df = conn.execute("""
                    WITH n_in AS (
                        SELECT n_cells AS n_in
                        FROM _rgg_group_n
                        WHERE group_name = ?
                    ),
                    base AS (
                        SELECT
                            t.atlas_gene_id,
                            t.atlas_gene_name AS names,

                            COALESCE(g.sum_expr, 0.0) AS sum_in,
                            COALESCE(g.sumsq_expr, 0.0) AS sumsq_in,
                            COALESCE(g.nnz, 0) AS nnz_in,

                            t.sum_total - COALESCE(g.sum_expr, 0.0) AS sum_ref,
                            t.sumsq_total - COALESCE(g.sumsq_expr, 0.0) AS sumsq_ref,
                            t.nnz_total - COALESCE(g.nnz, 0) AS nnz_ref,

                            n_in.n_in AS n_in,
                            (? - n_in.n_in) AS n_ref
                        FROM _rgg_total_stats t
                        CROSS JOIN n_in
                        LEFT JOIN _rgg_group_stats g
                          ON t.atlas_gene_id = g.atlas_gene_id
                         AND g.group_name = ?
                    )
                    SELECT *
                    FROM base
                """, [grp, total_cells, grp]).fetchdf()

                ref_name = "rest"

            else:
                df = conn.execute("""
                    WITH n_in AS (
                        SELECT n_cells AS n_in
                        FROM _rgg_group_n
                        WHERE group_name = ?
                    ),
                    n_ref AS (
                        SELECT n_cells AS n_ref
                        FROM _rgg_group_n
                        WHERE group_name = ?
                    ),
                    g_in AS (
                        SELECT
                            gs.atlas_gene_id,
                            gs.atlas_gene_name AS names,
                            COALESCE(g.sum_expr, 0.0) AS sum_in,
                            COALESCE(g.sumsq_expr, 0.0) AS sumsq_in,
                            COALESCE(g.nnz, 0) AS nnz_in
                        FROM _rgg_gene_set gs
                        LEFT JOIN _rgg_group_stats g
                          ON gs.atlas_gene_id = g.atlas_gene_id
                         AND g.group_name = ?
                    ),
                    g_ref AS (
                        SELECT
                            gs.atlas_gene_id,
                            COALESCE(g.sum_expr, 0.0) AS sum_ref,
                            COALESCE(g.sumsq_expr, 0.0) AS sumsq_ref,
                            COALESCE(g.nnz, 0) AS nnz_ref
                        FROM _rgg_gene_set gs
                        LEFT JOIN _rgg_group_stats g
                          ON gs.atlas_gene_id = g.atlas_gene_id
                         AND g.group_name = ?
                    )
                    SELECT
                        i.atlas_gene_id,
                        i.names,
                        i.sum_in,
                        i.sumsq_in,
                        i.nnz_in,
                        r.sum_ref,
                        r.sumsq_ref,
                        r.nnz_ref,
                        n_in.n_in,
                        n_ref.n_ref
                    FROM g_in i
                    JOIN g_ref r
                      ON i.atlas_gene_id = r.atlas_gene_id
                    CROSS JOIN n_in
                    CROSS JOIN n_ref
                """, [grp, reference, grp, reference]).fetchdf()

                ref_name = str(reference)

            if len(df) == 0:
                continue

            # -------------------------------------------------
            # t-test
            # -------------------------------------------------
            df = _compute_ttest_from_summary(df)

            # -------------------------------------------------
            # pct expressing
            # -------------------------------------------------
            df["pct_nz_in"] = df["nnz_in"] / df["n_in"].replace(0, np.nan)
            df["pct_nz_ref"] = df["nnz_ref"] / df["n_ref"].replace(0, np.nan)

            # -------------------------------------------------
            # logfoldchanges
            # -------------------------------------------------
            if input_is_log:
                mean_for_fc_in = np.expm1(df["mean_in"].to_numpy(dtype=np.float64))
                mean_for_fc_ref = np.expm1(df["mean_ref"].to_numpy(dtype=np.float64))
            else:
                mean_for_fc_in = df["mean_in"].to_numpy(dtype=np.float64)
                mean_for_fc_ref = df["mean_ref"].to_numpy(dtype=np.float64)

            eps = float(lfc_eps)

            df["logfoldchanges"] = np.log2(
                (mean_for_fc_in + eps) / (mean_for_fc_ref + eps)
            )

            # -------------------------------------------------
            # p-value correction
            # -------------------------------------------------
            if corr_method == "benjamini-hochberg":
                df["pvals_adj"] = _p_adjust_bh(df["pvals"].to_numpy())
            elif corr_method == "bonferroni":
                df["pvals_adj"] = _p_adjust_bonferroni(df["pvals"].to_numpy())

            # -------------------------------------------------
            # 排名
            # -------------------------------------------------
            if rankby_abs:
                df = df.sort_values(
                    by=["scores", "atlas_gene_id"],
                    key=lambda s: s.abs() if s.name == "scores" else s,
                    ascending=[False, True],
                )
            else:
                df = df.sort_values(
                    by=["scores", "atlas_gene_id"],
                    ascending=[False, True],
                )

            if n_genes is not None:
                df = df.head(int(n_genes))

            df = df.reset_index(drop=True)
            df["rank"] = np.arange(len(df), dtype=np.int64)
            df["group"] = str(grp)
            df["reference"] = ref_name
            df["method"] = method

            keep_cols = [
                "group",
                "reference",
                "rank",
                "atlas_gene_id",
                "names",
                "scores",
                "logfoldchanges",
                "pvals",
                "pvals_adj",
                "mean_in",
                "mean_ref",
                "pct_nz_in",
                "pct_nz_ref",
                "n_in",
                "n_ref",
                "method",
            ]

            result_list.append(df[keep_cols])

        if len(result_list) == 0:
            result_df = pd.DataFrame(
                columns=[
                    "group",
                    "reference",
                    "rank",
                    "atlas_gene_id",
                    "names",
                    "scores",
                    "logfoldchanges",
                    "pvals",
                    "pvals_adj",
                    "mean_in",
                    "mean_ref",
                    "pct_nz_in",
                    "pct_nz_ref",
                    "n_in",
                    "n_ref",
                    "method",
                ]
            )
        else:
            result_df = pd.concat(result_list, axis=0, ignore_index=True)

        # -------------------------------------------------
        # 6. 写入数据库表
        # -------------------------------------------------
        if inplace:
            conn.execute(f"DROP TABLE IF EXISTS {_q(add_table)}")
            conn.register("_rgg_result_py", result_df)

            conn.execute(f"""
                CREATE TABLE {_q(add_table)} AS
                SELECT *
                FROM _rgg_result_py
            """)

            conn.unregister("_rgg_result_py")


        print("rank_genes_groups 完成, 耗时: {:.2f} 秒".format((datetime.now() - start).total_seconds()))

        if return_df:
            return result_df

        return None

    finally:
        # -------------------------------------------------
        # 7. 轻量清理：只清临时表，不关闭连接
        # -------------------------------------------------
        for t in temp_tables:
            try:
                conn.execute(f"DROP TABLE IF EXISTS {t}")
            except Exception:
                pass

        try:
            conn.unregister("_rgg_result_py")
        except Exception:
            pass

        gc.collect()