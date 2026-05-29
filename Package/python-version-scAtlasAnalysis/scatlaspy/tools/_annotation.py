import numpy as np
import pandas as pd
from datetime import datetime

# 示例： 内置 PBMC marker reference（Phase 1）
def _get_builtin_pbmc_marker_reference():

    return {
        "CD4 T cells": ["IL7R"],
        "CD14+ Monocytes": ["CD14", "LYZ"],
        "B cells": ["MS4A1"],
        "CD8 T cells": ["CD8A"],
        "NK cells": ["GNLY", "NKG7"],
        "FCGR3A+ Monocytes": ["FCGR3A", "MS4A7"],
        "Dendritic Cells": ["FCER1A", "CST3"],
        "Megakaryocytes": ["PPBP"]
    }


# 工具函数：0~1 归一化
def _minmax_scale(series: pd.Series) -> pd.Series:

    x = series.astype(float).copy()
    if len(x) == 0:
        return x
    xmin = float(x.min())
    xmax = float(x.max())
    if xmax - xmin < 1e-12:
        return pd.Series(np.ones(len(x)), index=x.index)
    return (x - xmin) / (xmax - xmin)


# 单个 cluster × 单个 cell_type 打分
def _score_one_celltype_v2(
        marker_df: pd.DataFrame,
        marker_genes: list[str],
        top_n: int = 50,
        single_marker_penalty: float = 0.6
) -> tuple[float, list[str], int, float, float]:

    if marker_df is None or len(marker_df) == 0:
        return 0.0, [], 0, 0.0, 0.0

    df = marker_df.copy().head(top_n).reset_index(drop=True)

    if "atlas_gene_name" not in df.columns:
        raise ValueError("rank_genes_groups 结果中缺少 atlas_gene_name")

    df["atlas_gene_name"] = df["atlas_gene_name"].astype(str)
    df["_gene_upper"] = df["atlas_gene_name"].str.upper()

    marker_upper = [g.upper() for g in marker_genes]
    n_ref = len(marker_upper)

    # 选择主支持列
    if "final_score" in df.columns:
        base_col = "final_score"
    elif "t_like_score" in df.columns:
        base_col = "t_like_score"
    elif "score" in df.columns:
        base_col = "score"
    else:
        raise ValueError("marker_df 中缺少可用主分数字段（final_score / t_like_score / score）")

    # 构造融合支持度（cluster 内部）
    df["_base_norm"] = _minmax_scale(df[base_col])

    if "log2fc" in df.columns:
        df["_fc_pos"] = df["log2fc"].clip(lower=0)
        df["_fc_norm"] = _minmax_scale(df["_fc_pos"])
    else:
        df["_fc_norm"] = 0.0

    if "pct_diff" in df.columns:
        df["_pct_pos"] = df["pct_diff"].clip(lower=0)
        df["_pct_norm"] = _minmax_scale(df["_pct_pos"])
    elif "pct_in" in df.columns and "pct_rest" in df.columns:
        df["_pct_pos"] = (df["pct_in"] - df["pct_rest"]).clip(lower=0)
        df["_pct_norm"] = _minmax_scale(df["_pct_pos"])
    else:
        df["_pct_norm"] = 0.0

    # cluster 内 gene 支持度（0~1 量级）
    df["_gene_support_raw"] = (
        0.50 * df["_base_norm"] +
        0.30 * df["_fc_norm"] +
        0.20 * df["_pct_norm"]
    )

    # 排名权重：越靠前越重要
    df["_rank"] = np.arange(len(df))
    df["_rank_weight"] = 1.0 / (df["_rank"] + 1.0)

    df["_gene_support"] = df["_gene_support_raw"] * df["_rank_weight"]

    # 命中 reference marker
    hit_df = df[df["_gene_upper"].isin(marker_upper)].copy()

    if len(hit_df) == 0:
        return 0.0, [], 0, 0.0, 0.0

    matched_markers = len(hit_df)
    support_markers = hit_df["atlas_gene_name"].astype(str).tolist()

    mean_support = float(hit_df["_gene_support"].mean())
    match_fraction = matched_markers / max(n_ref, 1)

    # 最终 annotation score ：  核心：平均支持度 × 命中比例
    score = mean_support * match_fraction

    # 单-marker cell type 惩罚： 避免 PPBP 这种只靠 1 个 marker 劫持
    if n_ref == 1:
        score *= single_marker_penalty

    return score, support_markers, matched_markers, mean_support, match_fraction


# 主函数：自动 cluster 注释
def annotate_clusters(
        atlas,
        rank_result: dict,
        groupby: str = "kmeans",
        reference_name: str = "builtin_pbmc",
        write_to_obs: bool = True,
        obs_col: str = "cell_type_auto",
        obs_conf_col: str = "cell_type_auto_confidence",
        top_n: int = 50,
        unknown_label: str = "Unknown",
        ambiguity_threshold_high: float = 0.08,
        ambiguity_threshold_medium: float = 0.03
):

    print("\n==== annotate_clusters (Phase 1, revised) ====")
    start = datetime.now()
    conn = atlas.connection

    # 检查输入
    if not isinstance(rank_result, dict) or len(rank_result) == 0:
        raise ValueError("rank_result 不能为空，需传入 plot_rank_genes_groups() 的返回结果")

    obs_cols = [r[1] for r in conn.execute("PRAGMA table_info(obs)").fetchall()]
    if groupby not in obs_cols:
        raise ValueError(f"obs 中不存在列: {groupby}")

    # 读取内置 reference
    if reference_name != "builtin_pbmc":
        raise ValueError("Phase 1 当前只支持 reference_name='builtin_pbmc'")

    ref_dict = _get_builtin_pbmc_marker_reference()

    # 计算 cluster × cell_type score
    score_rows = []

    for grp, marker_df in rank_result.items():
        for cell_type, marker_genes in ref_dict.items():
            score, support_markers, matched_markers, mean_support, match_fraction = _score_one_celltype_v2(
                marker_df=marker_df,
                marker_genes=marker_genes,
                top_n=top_n,
                single_marker_penalty=0.6
            )

            score_rows.append({
                "groupby": groupby,
                "cluster_id": str(grp),
                "cell_type": cell_type,
                "score": float(score),
                "matched_markers": int(matched_markers),
                "match_fraction": float(match_fraction),
                "mean_support": float(mean_support),
                "support_markers": ", ".join(support_markers)
            })

    score_df = pd.DataFrame(score_rows)

    if len(score_df) == 0:
        raise ValueError("score_df 为空，无法注释")

    # 生成 summary
    summary_rows = []

    cluster_ids = list(score_df["cluster_id"].unique())

    # 尽量按数字排序
    def _sort_key(x):
        try:
            return (0, int(x))
        except:
            return (1, str(x))

    cluster_ids = sorted(cluster_ids, key=_sort_key)

    for grp in cluster_ids:
        sub = score_df[score_df["cluster_id"] == grp].copy()
        sub = sub.sort_values(
            ["score", "matched_markers", "match_fraction", "mean_support"],
            ascending=False
        ).reset_index(drop=True)

        best = sub.iloc[0]
        runner_up = sub.iloc[1] if len(sub) > 1 else None

        best_type = best["cell_type"]
        best_score = float(best["score"])
        best_support = best["support_markers"]
        best_matched = int(best["matched_markers"])

        runner_type = runner_up["cell_type"] if runner_up is not None else None
        runner_score = float(runner_up["score"]) if runner_up is not None else 0.0

        delta = best_score - runner_score

        # 置信度规则
        if best_score <= 0 or best_matched == 0:
            predicted = unknown_label
            confidence = "low"

        elif best_matched == 1 and delta < ambiguity_threshold_high:
            # 单 marker 命中且领先不明显 → 保守
            predicted = best_type
            confidence = "low"

        elif delta >= ambiguity_threshold_high:
            predicted = best_type
            confidence = "high"

        elif delta >= ambiguity_threshold_medium:
            predicted = best_type
            confidence = "medium"

        else:
            predicted = best_type
            confidence = "low"

        summary_rows.append({
            "groupby": groupby,
            "cluster_id": grp,
            "best_cell_type": predicted,
            "confidence": confidence,
            "best_score": best_score,
            "runner_up": runner_type,
            "runner_up_score": runner_score,
            "delta_score": delta,
            "matched_markers": best_matched,
            "support_markers": best_support
        })

    summary_df = pd.DataFrame(summary_rows)

    # 写数据库：cluster_annotation_scores
    conn.execute("DROP TABLE IF EXISTS cluster_annotation_scores")
    conn.execute("""
        CREATE TABLE cluster_annotation_scores (
            groupby TEXT,
            cluster_id TEXT,
            cell_type TEXT,
            score DOUBLE,
            matched_markers INTEGER,
            match_fraction DOUBLE,
            mean_support DOUBLE,
            support_markers TEXT
        )
    """)
    conn.append("cluster_annotation_scores", score_df)

    # 写数据库：cluster_annotation_summary
    conn.execute("DROP TABLE IF EXISTS cluster_annotation_summary")
    conn.execute("""
        CREATE TABLE cluster_annotation_summary (
            groupby TEXT,
            cluster_id TEXT,
            best_cell_type TEXT,
            confidence TEXT,
            best_score DOUBLE,
            runner_up TEXT,
            runner_up_score DOUBLE,
            delta_score DOUBLE,
            matched_markers INTEGER,
            support_markers TEXT
        )
    """)
    conn.append("cluster_annotation_summary", summary_df)

    # 写入 obs
    if write_to_obs:
        obs_cols = [r[1] for r in conn.execute("PRAGMA table_info(obs)").fetchall()]

        if obs_col not in obs_cols:
            conn.execute(f"ALTER TABLE obs ADD COLUMN {obs_col} TEXT")
        if obs_conf_col not in obs_cols:
            conn.execute(f"ALTER TABLE obs ADD COLUMN {obs_conf_col} TEXT")

        conn.execute(f"UPDATE obs SET {obs_col} = NULL")
        conn.execute(f"UPDATE obs SET {obs_conf_col} = NULL")

        tmp_df = summary_df[["cluster_id", "best_cell_type", "confidence"]].copy()
        conn.register("_cluster_annotation_tmp", tmp_df)

        conn.execute(f"""
            UPDATE obs
            SET
                {obs_col} = t.best_cell_type,
                {obs_conf_col} = t.confidence
            FROM _cluster_annotation_tmp t
            WHERE CAST(obs.{groupby} AS TEXT) = CAST(t.cluster_id AS TEXT)
        """)

        conn.unregister("_cluster_annotation_tmp")

    print(f"-> done in {(datetime.now() - start).total_seconds():.2f}s ✅")

    return summary_df, score_df

# summary_df 是什么？
# ✅ 每个 cluster 的最终注释结果总表
# 它通常是一行一个 cluster，比如：

# cluster_id	best_cell_type	confidence	best_score	runner_up	    delta_score 	support_markers
# 0	            CD4 T cells	         high	 8.2	    CD8 T cells	        3.1	        IL7R
# 1	            CD14+ Monocytes 	 high	 10.5	    FCGR3A+ Monocytes	4.3	        CD14, LYZ
# 2	            B cells	             high	 7.6	    Dendritic Cells	    2.0	        MS4A1
# 3	            NK cells	         high	 9.3	    CD8 T cells	        2.7	        GNLY, NKG7

# 也就是说：
# 👉 summary_df = 最终“cluster 叫什么名字”的表

# score_df 是什么？
# ✅ cluster × cell_type 的打分明细表
# 它不是只给你最终答案，而是把“候选答案都列出来”。

# cluster_id	cell_type	    score	matched_markers	support_markers
# 0	           CD4 T cells 	    8.2	    1	                IL7R
# 0	           CD8 T cells	    2.4	    0
# 0	           NK cells	        1.1	    0
# 1	          CD14+ Monocytes	10.5	2	             CD14, LYZ
# 1	         FCGR3A+ Monocytes	4.9	    1	              FCGR3A

# 👉 score_df = “每个 cluster 对每个 cell type 的评分明细”



