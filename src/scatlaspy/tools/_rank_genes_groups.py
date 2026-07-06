from datetime import datetime
import os
import gc
import numpy as np
import pandas as pd
from scipy import stats
from typing import Any
from ..data import Atlas
import logging
logger = logging.getLogger('Atlas')


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
        lfc_eps: float = 1e-9,
        inplace: bool = True,
        return_df: bool = True,
) -> pd.DataFrame | None:
    """Rank marker genes by cell groups.

    This function groups cells according to the column specified by ``groupby``
    in the ``obs`` table, performs differential expression statistics between
    each target group and the reference group, and generates marker gene ranking
    results.

    The function first aggregates the expression sum, squared sum, and nonzero
    expression record count for each ``group x gene`` in SQL. It then calculates
    Welch's t-test, log fold change, expression percentages, and multiple-testing
    adjusted p-values in pandas/numpy. The results can be written to a database
    table or returned as a ``pandas.DataFrame``.

    Parameters
    ----------
    atlas
        Atlas object. The object must already be connected to a DuckDB database,
        and the database must contain at least the ``obs``, ``var``, and
        ``X_HyS_data`` tables.

        The ``obs`` table must contain ``atlas_cell_id`` and the grouping column
        specified by ``groupby``; the ``var`` table must contain
        ``atlas_gene_id`` and ``atlas_gene_name``; the ``X_HyS_data`` table must
        contain ``atlas_cell_id``, ``atlas_gene_id``, and the expression field
        specified by ``use_data``.

    groupby
        Grouping column name in ``obs``, such as ``"kmeans"``, ``"leiden"``, or
        ``"cell_type"``.

    use_data
        Expression field name read from the ``X_HyS_data`` table. The default
        value is ``"data_log1p"``. Common values include ``"data_count"``,
        ``"data_normalize"``, ``"data_log1p"``, and ``"data_scale"``.

    groups
        List of groups to calculate, display, or retain. When set to ``None``,
        all groups are used.

    reference
        Reference group for differential analysis. The default value is
        ``"rest"``, meaning that each target group is compared against all cells
        outside that group.

        A specific group name or group ID can also be passed. In that case, each
        target group is compared with the specified reference group, and the
        reference group itself is not output as a target group.

    n_genes
        Number of genes to retain or display for each group. When set to
        ``None``, all available genes are retained.

    mask_var
        Boolean column name in ``var`` used to restrict the gene set included in
        the analysis. For example, passing ``"highly_variable_genes"`` analyzes
        only genes where this column is ``TRUE``. When set to ``None``, all genes
        in the ``var`` table are used.

    corr_method
        Multiple-testing correction method. Supports ``"benjamini-hochberg"``
        and ``"bonferroni"``.

    rankby_abs
        Whether to sort by the absolute value of the t statistic. The default is
        ``False``, meaning that genes relatively more highly expressed in the
        target group than in the reference group are prioritized.

    add_table
        Name of the result table written to the database.

    input_is_log
        Whether the input expression field has already been log-transformed. The
        default is ``True``. When ``True``, the mean is approximately restored
        using ``expm1`` before calculating log fold change; when ``False``, the
        raw mean is used directly.

    lfc_eps
        Small value added when calculating log fold change to avoid division by zero.

    inplace
        Whether to write the results back to the Atlas database.

    return_df
        Whether to return the result DataFrame.

    Returns
    -------
    pandas.DataFrame or None
        When ``return_df=True``, returns the differential gene result DataFrame.
        When ``return_df=False``, returns ``None``. If ``inplace=True``, the
        results are also written to the database table specified by
        ``add_table``.

    Notes
    -----
    The output results contain fields such as ``group``, ``reference``, ``rank``,
    ``atlas_gene_id``, ``names``, ``scores``, ``logfoldchanges``, ``pvals``,
    ``pvals_adj``, ``mean_in``, ``mean_ref``, ``pct_nz_in``, ``pct_nz_ref``,
    ``n_in``, ``n_ref``, and ``method``.

    This implementation is based on sparse-table aggregated statistics and does
    not construct a full dense matrix. Expression values that are not explicitly
    stored are treated as 0 when calculating means and variances.

    Examples
    --------
    Calculate marker genes based on K-means clusters::

        result = sap.tl.rank_genes_groups(atlas, groupby="kmeans")

    Calculate only specified clusters and retain the top 100 genes per group::

        result = sap.tl.rank_genes_groups(
            atlas,
            groupby="kmeans",
            groups=["0", "1", "2"],
            n_genes=100,
        )
    """

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
        # 0. Basic checks
        # -------------------------------------------------
        obs_cols = [r[1] for r in conn.execute("PRAGMA table_info(obs)").fetchall()]
        var_cols = [r[1] for r in conn.execute("PRAGMA table_info(var)").fetchall()]
        x_cols = [r[1] for r in conn.execute("PRAGMA table_info(X_HyS_data)").fetchall()]

        if groupby not in obs_cols:
            raise ValueError(f"Column does not exist in obs: {groupby}")

        if use_data not in x_cols:
            raise ValueError(f"Field does not exist in X_HyS_data: {use_data}")

        if mask_var is not None and mask_var not in var_cols:
            raise ValueError(f"Column does not exist in var: {mask_var}")

        if corr_method not in {"benjamini-hochberg", "bonferroni"}:
            raise ValueError("corr_method only supports 'benjamini-hochberg' or 'bonferroni'")

        # -------------------------------------------------
        # 1. Candidate gene set
        # -------------------------------------------------
        for t in temp_tables:
            conn.execute(f"DROP TABLE IF EXISTS {t}")

        if mask_var is None:
            var_where = "1=1"
        else:
            var_where = f"COALESCE({_q(mask_var)}, FALSE)=TRUE"

        # Keep atlas_gene_name
        # If your var contains gene_name and you want to prioritize displaying gene symbols,
        # you can replace atlas_gene_name with COALESCE(gene_name, atlas_gene_name)
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
            raise ValueError("The candidate gene set is empty. Please check the mask_var setting")

        # -------------------------------------------------
        # 2. Group information
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
            raise ValueError(f"No available groups in obs.{groupby}")

        all_groups = group_df["group_name"].astype(str).tolist()

        def _group_sort_key(x: Any):
            """Generate a natural sorting key for groups or labels.

            This internal helper is used to make group-name sorting closer to the
            natural order commonly used in single-cell analysis. If a group name
            can be converted to an integer or a float, it is sorted numerically;
            otherwise, it is sorted as a string.

            Parameters
            ----------
            x
                Group name or label to sort.

            Returns
            -------
            sort_key
                Sorting key that can be used by ``sorted(..., key=...)``.

            Notes
            -----
            This function only affects the processing order of groups in the
            results. It does not change the original group values in the database.
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
            raise ValueError("No available groups remain after filtering by groups")

        if reference != "rest":
            reference = str(reference)

            if reference not in set(all_groups):
                raise ValueError(
                    f"reference={reference!r} is not in obs.{groupby}; "
                    f"available groups={all_groups}"
                )

            group_list = [g for g in group_list if str(g) != str(reference)]

            if len(group_list) == 0:
                raise ValueError("groups is empty after removing the reference group")

        # -------------------------------------------------
        # 3. Aggregate group x gene statistics in one pass
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
        # 4. When reference='rest', precompute the total sums across all groups
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
        # 5. Calculate rank_genes_groups results group by group
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
            # Ranking
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
        # 6. Write to database table
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


        ("rank_genes_groups completed, elapsed time: {:.2f} seconds".format((datetime.now() - start).total_seconds()))

        if return_df:
            return result_df

        return None

    finally:
        # -------------------------------------------------
        # 7. Lightweight cleanup: only clean temporary tables, do not close the connection
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


def _q(name: str) -> str:
    """Add safe quoting to a SQL identifier.

    This internal helper is used in differential gene analysis to safely quote
    dynamically passed DuckDB table names or field names, such as ``groupby``,
    ``use_data``, ``mask_var``, and ``add_table``. The function escapes existing
    double quotes in the name and adds double quotes around it, avoiding SQL
    parsing failures when field names contain special characters or keywords.

    Parameters
    ----------
    name
        SQL identifier to quote.

    Returns
    -------
    quoted_name
        SQL identifier enclosed in double quotes.

    Notes
    -----
    This function is only used for SQL identifiers, not for ordinary string
    values. String values should be passed through DuckDB parameter binding.
    """
    return '"' + str(name).replace('"', '""') + '"'


def _p_adjust_bh(pvals: np.ndarray) -> np.ndarray:
    """Adjust multiple-testing p-values using the Benjamini-Hochberg method.

    This internal function is used for the ``corr_method="benjamini-hochberg"``
    branch of ``rank_genes_groups``. It performs FDR correction on the raw
    p-values of all genes within each group and preserves the original order of
    the input array.

    ``NaN`` or infinite values are treated as invalid p-values, and the
    corresponding positions in the output remain ``NaN``. Only finite p-values
    participate in sorting and correction.

    Parameters
    ----------
    pvals
        Array of raw p-values to adjust.

    Returns
    -------
    numpy.ndarray
        Adjusted p-value array with the same length as ``pvals``. Finite inputs
        are clipped to the ``[0, 1]`` range, and invalid inputs correspond to
        ``NaN``.

    Notes
    -----
    This is an internal helper; users usually use it indirectly through
    ``rank_genes_groups``.
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
    """Adjust multiple-testing p-values using the Bonferroni method.

    This internal function is used for the ``corr_method="bonferroni"`` branch
    of ``rank_genes_groups``. It multiplies each finite p-value by the number of
    finite p-values participating in the correction and clips the result to the
    ``[0, 1]`` range.

    Parameters
    ----------
    pvals
        Array of raw p-values to adjust.

    Returns
    -------
    numpy.ndarray
        Adjusted p-value array with the same length as ``pvals``. Invalid inputs
        correspond to ``NaN``.

    Notes
    -----
    This is an internal helper; users usually use it indirectly through
    ``rank_genes_groups``.
    """
    pvals = np.asarray(pvals, dtype=np.float64)

    out = np.full(len(pvals), np.nan, dtype=np.float64)
    valid = np.isfinite(pvals)

    if valid.sum() == 0:
        return out

    out[valid] = np.clip(pvals[valid] * valid.sum(), 0, 1)
    return out


def _compute_ttest_from_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate test results from summary statistics.

    This internal function receives the within-group and reference-group summary
    statistics aggregated from DuckDB by ``rank_genes_groups``, and calculates
    the mean, sample variance, t statistic, and two-sided p-value required for
    Welch's t-test in pandas/numpy.

    The input ``sum_in``, ``sumsq_in``, ``sum_ref``, and ``sumsq_ref`` come from
    aggregating explicitly expressed records in the sparse table; ``n_in`` and
    ``n_ref`` are the total numbers of cells in the corresponding groups.
    Therefore, sparse 0 values that do not appear in ``X_HyS_data`` implicitly
    participate in mean and variance calculations through the total cell count.

    Parameters
    ----------
    df
        DataFrame containing summary statistics for each gene. It must contain at
        least: ``n_in``, ``n_ref``, ``sum_in``, ``sum_ref``, ``sumsq_in``, and
        ``sumsq_ref``.

    Returns
    -------
    pandas.DataFrame
        Returns the original DataFrame with the following columns added:
        ``mean_in``, ``mean_ref``, ``var_in``, ``var_ref``, ``scores``, and
        ``pvals``.

    Notes
    -----
    When the standard error is 0 or the degrees of freedom are abnormal, the
    p-value is set to ``1.0``, indicating that the gene is not significant in the
    current comparison.
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
    #    Values that do not appear in CSR are treated as 0;
    #    sum / sumsq are aggregated from nonzero entries, but n_in / n_ref are the total cell counts.
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

    # When se=0 or dof is abnormal, set it to not significant
    pvals[~np.isfinite(pvals)] = 1.0

    df["mean_in"] = mean_in
    df["mean_ref"] = mean_ref
    df["var_in"] = var_in
    df["var_ref"] = var_ref
    df["scores"] = scores
    df["pvals"] = pvals

    return df
