from datetime import datetime
import os
import gc
import numpy as np
import pandas as pd
from scipy import stats


def _q(name: str) -> str:
    """
    DuckDB 字段 / 表名安全引用。
    """
    return '"' + str(name).replace('"', '""') + '"'


def _p_adjust_bh(pvals: np.ndarray) -> np.ndarray:
    """
    Benjamini-Hochberg FDR 校正。
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
    """
    Bonferroni 校正。
    """
    pvals = np.asarray(pvals, dtype=np.float64)

    out = np.full(len(pvals), np.nan, dtype=np.float64)
    valid = np.isfinite(pvals)

    if valid.sum() == 0:
        return out

    out[valid] = np.clip(pvals[valid] * valid.sum(), 0, 1)
    return out


def _compute_ttest_from_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    根据 group/ref 的 summary statistics 计算 Welch t-test。

    输入 df 需要包含：
        sum_in, sumsq_in, n_in
        sum_ref, sumsq_ref, n_ref

    输出新增：
        mean_in
        mean_ref
        var_in
        var_ref
        scores
        pvals
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
        atlas,
        groupby: str = "kmeans",
        use_expr_field: str = "data_log1p",
        groups: list | None = None,
        reference: str | int = "rest",
        n_genes: int | None = None,
        mask_var: str | None = None, # = "highly_variable_genes"
        corr_method: str = "benjamini-hochberg",
        rankby_abs: bool = False,
        key_added: str = "rank_genes_groups",
        input_is_log: bool = True,
        lfc_eps: float = 1e-9,  # 对齐 Scanpy 的 logFC 伪计数
        inplace: bool = True,
        return_df: bool = True,
) -> pd.DataFrame | None:
    """
    DuckDB / scAtlasPy 版 rank_genes_groups 计算函数。

    只实现 t-test，对齐：
        sc.tl.rank_genes_groups(
            adata,
            groupby=groupby,
            groups=groups,
            reference=reference,
            method="t-test",
        )

    输出 long-format 结果：
        group
        reference
        rank
        atlas_gene_id
        names
        scores
        logfoldchanges
        pvals
        pvals_adj
        mean_in
        mean_ref
        pct_nz_in
        pct_nz_ref
        n_in
        n_ref
        method

    Parameters
    ----------
    input_is_log
        如果 use_expr_field 是 data_log1p，设为 True。
        如果 use_expr_field 是 data 或 data_normalize，设为 False。

    lfc_eps
        计算 logfoldchanges 时的伪计数。
        不建议使用 1e-9，否则 mean_in=0 时容易出现 ±20 以上极端 logFC。
        推荐 1e-3 或 1e-2。
    """

    method = "t-test"

    print(f"\n==== rank_genes_groups (method={method}, reference={reference}) ====")
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

        if use_expr_field not in x_cols:
            raise ValueError(f"X_HyS_data 中不存在字段: {use_expr_field}")

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

        print(f"-> candidate genes = {gene_count:,}")

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

        def _group_sort_key(x):
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

        print(f"-> groupby = {groupby}")
        print(f"-> groups = {group_list}")
        print(f"-> reference = {reference}")
        print(f"-> input_is_log = {input_is_log}")
        print(f"-> lfc_eps = {lfc_eps}")

        # -------------------------------------------------
        # 3. 一次性聚合 group × gene 统计
        # -------------------------------------------------
        conn.execute(f"""
            CREATE TEMP TABLE _rgg_group_stats AS
            SELECT
                CAST(o.{_q(groupby)} AS TEXT) AS group_name,
                x.atlas_gene_id,
                SUM(x.{_q(use_expr_field)}) AS sum_expr,
                SUM(x.{_q(use_expr_field)} * x.{_q(use_expr_field)}) AS sumsq_expr,
                COUNT(*) AS nnz
            FROM X_HyS_data x
            JOIN obs o
              ON x.atlas_cell_id = o.atlas_cell_id
            JOIN _rgg_gene_set gs
              ON x.atlas_gene_id = gs.atlas_gene_id
            WHERE o.{_q(groupby)} IS NOT NULL
              AND x.{_q(use_expr_field)} IS NOT NULL
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
            print(f"-> computing group {grp} ...")

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
            conn.execute(f"DROP TABLE IF EXISTS {_q(key_added)}")
            conn.register("_rgg_result_py", result_df)

            conn.execute(f"""
                CREATE TABLE {_q(key_added)} AS
                SELECT *
                FROM _rgg_result_py
            """)

            conn.unregister("_rgg_result_py")

            print(f"-> result written to table: {key_added}")

        print("rank_genes_groups 完成")
        print("耗时: {:.2f} 秒".format((datetime.now() - start).total_seconds()))

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