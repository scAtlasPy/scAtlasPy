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
):
    """按细胞分组计算 marker gene 排名。

    该函数按 ``obs`` 表中 ``groupby`` 指定的列对细胞分组，对每个目标分组与
    参考组执行差异表达统计，并生成 marker gene 排名结果。

    函数会先在 SQL 中聚合每个 ``group × gene`` 的表达总和、平方和和非零
    表达记录数量，再在 pandas/numpy 中计算 Welch t-test、log fold change、
    表达比例和多重检验校正后的 p 值。结果可写入数据库表，也可以作为
    ``pandas.DataFrame`` 返回。

    Parameters
    ----------
    atlas
        Atlas 对象。要求对象已经连接到 DuckDB 数据库，并且数据库中至少包含
        ``obs``、``var`` 和 ``X_HyS_data`` 表。

        ``obs`` 表需要包含 ``atlas_cell_id`` 和 ``groupby`` 指定的分组列；
        ``var`` 表需要包含 ``atlas_gene_id`` 和 ``atlas_gene_name``；
        ``X_HyS_data`` 表需要包含 ``atlas_cell_id``、``atlas_gene_id`` 以及
        由 ``use_data`` 指定的表达字段。
    groupby
        ``obs`` 中的分组列名，例如 ``"kmeans"``、``"leiden"`` 或 ``"cell_type"``。
    use_data
        从 ``X_HyS_data`` 表中读取的表达字段名。默认值为 ``"data_log1p"``。
        常用值包括 ``"data_count"``、``"data_normalize"``、``"data_log1p"``
        和 ``"data_scale"``。
    groups
        需要计算、展示或保留的分组列表。为 ``None`` 时使用全部分组。
    reference
        差异分析参考组。默认值为 ``"rest"``，表示每个目标分组与其他所有
        非该分组细胞比较。

        也可以传入某个具体分组名或分组 ID，此时每个目标分组会与该参考分组
        比较，且参考分组本身不会作为目标分组输出。
    n_genes
        每个分组保留或展示的基因数量。为 ``None`` 时保留全部可用基因。
    mask_var
        用于限制参与分析基因范围的 ``var`` 布尔列名。例如传入
        ``"highly_variable_genes"`` 时，只分析该列为 ``TRUE`` 的基因。
        为 ``None`` 时使用 ``var`` 表中的全部基因。
    corr_method
        多重检验校正方法。支持 ``"benjamini-hochberg"`` 和 ``"bonferroni"``。
    rankby_abs
        是否按 t 统计量绝对值排序。默认 ``False``，表示优先保留在目标分组中
        相对参考组更高表达的基因。
    add_table
        写入数据库的结果表名。
    input_is_log
        输入表达字段是否已经做过对数变换。默认 ``True``。
        当为 ``True`` 时，计算 log fold change 前会对均值执行 ``expm1`` 近似还原；
        当为 ``False`` 时直接基于原始均值计算。
    lfc_eps
        计算 log fold change 时加入的极小值，用于避免除零。
    inplace
        是否把结果写回 Atlas 数据库。
    return_df
        是否返回结果 DataFrame。

    Returns
    -------
    pandas.DataFrame 或 None
        当 ``return_df=True`` 时返回差异基因结果 DataFrame。
        当 ``return_df=False`` 时返回 ``None``。
        如果 ``inplace=True``，结果还会写入 ``add_table`` 指定的数据库表。

    Notes
    -----
    输出结果包含 ``group``、``reference``、``rank``、``atlas_gene_id``、
    ``names``、``scores``、``logfoldchanges``、``pvals``、``pvals_adj``、
    ``mean_in``、``mean_ref``、``pct_nz_in``、``pct_nz_ref``、``n_in``、
    ``n_ref`` 和 ``method`` 等字段。

    该实现基于稀疏表聚合统计量，不会构造完整 dense 矩阵；未显式存储的表达值
    按 0 参与均值和方差计算。

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

            该内部 helper 用于让分组名排序更接近单细胞分析中的自然顺序。
            如果分组名可以转为整数或浮点数，则按数值排序；否则按字符串排序。

            Parameters
            ----------
            x
                需要排序的分组名或标签。

            Returns
            -------
            sort_key
                可用于 ``sorted(..., key=...)`` 的排序键。

            Notes
            -----
            该函数只影响结果中分组的处理顺序，不改变数据库中的原始分组值。
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


        ("rank_genes_groups 完成, 耗时: {:.2f} 秒".format((datetime.now() - start).total_seconds()))

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


def _q(name: str) -> str:
    """为 SQL 标识符添加安全引用。

    该内部 helper 用于在差异基因分析中安全引用动态传入的 DuckDB 表名或
    字段名，例如 ``groupby``、``use_data``、``mask_var`` 和 ``add_table``。
    函数会转义名称中已有的双引号，并在外层添加双引号，避免字段名包含
    特殊字符或关键字时 SQL 解析失败。

    Parameters
    ----------
    name
        需要引用的 SQL 标识符。

    Returns
    -------
    quoted_name
        加双引号后的 SQL 标识符。

    Notes
    -----
    该函数只用于 SQL 标识符，不用于普通字符串值。字符串值应通过 DuckDB
    参数绑定传入。
    """
    return '"' + str(name).replace('"', '""') + '"'


def _p_adjust_bh(pvals: np.ndarray) -> np.ndarray:
    """使用 Benjamini-Hochberg 方法校正多重检验 p 值。

    该内部函数用于 ``rank_genes_groups`` 的 ``corr_method="benjamini-hochberg"``
    分支。它对每个分组内所有基因的原始 p 值执行 FDR 校正，并保持输入数组
    的原始顺序。

    ``NaN`` 或无穷值会被视为无效 p 值，输出中对应位置保持为 ``NaN``；
    只有有限 p 值会参与排序和校正。

    Parameters
    ----------
    pvals
        待校正的原始 p 值数组。

    Returns
    -------
    numpy.ndarray
        与 ``pvals`` 等长的校正后 p 值数组。有限输入会被限制在 ``[0, 1]``
        范围内，无效输入对应 ``NaN``。

    Notes
    -----
    这是内部 helper；用户通常通过 ``rank_genes_groups`` 间接使用。
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
    """使用 Bonferroni 方法校正多重检验 p 值。

    该内部函数用于 ``rank_genes_groups`` 的 ``corr_method="bonferroni"``
    分支。它将每个有限 p 值乘以参与校正的有限 p 值数量，并把结果限制在
    ``[0, 1]`` 范围内。

    Parameters
    ----------
    pvals
        待校正的原始 p 值数组。

    Returns
    -------
    numpy.ndarray
        与 ``pvals`` 等长的校正后 p 值数组。无效输入对应 ``NaN``。

    Notes
    -----
    这是内部 helper；用户通常通过 ``rank_genes_groups`` 间接使用。
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

    该内部函数接收 ``rank_genes_groups`` 从 DuckDB 聚合得到的 group 内外统计量，
    并在 pandas/numpy 中计算 Welch t-test 所需的均值、样本方差、t 统计量和
    双侧 p 值。

    输入中的 ``sum_in``、``sumsq_in``、``sum_ref`` 和 ``sumsq_ref`` 来自稀疏
    表中显式表达记录的聚合；``n_in`` 和 ``n_ref`` 则是对应分组的总细胞数。
    因此未出现在 ``X_HyS_data`` 中的稀疏 0 值会通过总细胞数隐式参与均值和
    方差计算。

    Parameters
    ----------
    df
        包含每个基因的汇总统计量 DataFrame。至少需要包含：
        ``n_in``、``n_ref``、``sum_in``、``sum_ref``、``sumsq_in`` 和
        ``sumsq_ref``。

    Returns
    -------
    pandas.DataFrame
        在原 DataFrame 上新增并返回以下列：
        ``mean_in``、``mean_ref``、``var_in``、``var_ref``、``scores`` 和
        ``pvals``。

    Notes
    -----
    当标准误为 0 或自由度异常时，p 值会被设置为 ``1.0``，表示该基因在当前
    比较中不显著。
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