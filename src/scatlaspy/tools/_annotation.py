from collections.abc import Mapping, Sequence
import pandas as pd
from ..data import Atlas


def manual_annotate_clusters(
        atlas: Atlas,
        cluster_to_cell_type: Mapping[str, str] | Sequence[str],
        groupby: str = "kmeans",
        obs_col: str = "cell_type_manual",
        table_name: str | None = "manual_cluster_annotation",
        unknown_label: str | None = None,
        return_df: bool = True,
) -> pd.DataFrame | None:
    """手动为 cluster 添加细胞类型注释。

    该函数对应 Scanpy PBMC3k 流程中的手动注释步骤：
    在查看 marker genes、rank genes 图和 UMAP 图之后，由用户手动指定
    每个 cluster 对应的细胞类型名称，并将这些标签写回 ``obs`` 表。

    函数支持两种输入方式：一种是显式字典，直接指定每个 cluster ID 对应的
    细胞类型；另一种是 Scanpy 风格的有序列表，按 ``obs[groupby]`` 中真实
    cluster 的排序顺序依次写入标签。运行时会检查 cluster 是否存在、是否
    重复，以及列表长度是否与 cluster 数量一致。

    Parameters
    ----------
    atlas
        Atlas 对象。需要已经连接到包含 ``obs`` 表的 DuckDB 数据库。
        如果尚未连接，函数会自动以 ``r+`` 模式连接。

    cluster_to_cell_type
        用户提供的手动注释结果。

        如果传入字典，字典的 key 是 cluster ID，value 是细胞类型名称。
        例如::

            {
                "0": "CD4 T cells",
                "1": "CD14+ Monocytes",
                "2": "B cells",
            }

        如果传入 list 或 tuple，则会按照 ``obs[groupby]`` 中 cluster ID
        的排序顺序依次赋值，类似 Scanpy 的 ``new_cluster_names`` 用法。
        例如::

            [
                "CD4 T cells",
                "CD14+ Monocytes",
                "B cells",
            ]

    groupby
        ``obs`` 中保存聚类结果的列名。

        如果使用 kmeans 聚类，通常是 ``"kmeans"``。
        如果当前数据库中使用的是 KMeans 聚类，则应设置为 ``"kmeans"``。

    obs_col
        写入手动细胞类型注释的 ``obs`` 列名。
        默认写入 ``obs.cell_type_manual``。

    table_name
        可选的数据库表名，用于保存 cluster 到 cell type 的映射关系。
        如果设置为 ``None``，则不额外保存映射表。

    unknown_label
        对未提供手动注释的 cluster 使用的标签。

        如果为 ``None``，未注释的 cluster 会被写为 ``NULL``。
        如果设置为 ``"Unknown"``，未注释的 cluster 会被写为 ``"Unknown"``。

    return_df
        是否返回注释结果汇总表。

    Returns
    -------
    pandas.DataFrame or None
        如果 ``return_df=True``，返回每个 cluster 的手动注释结果和细胞数量。
        如果 ``return_df=False``，返回 ``None``。

    Notes
    -----
    该函数会修改 ``obs`` 表：如果 ``obs_col`` 不存在，会自动新增文本列；
    如果已经存在，会先清空旧值，再写入本次提供的手动注释。

    当 ``table_name`` 不为 ``None`` 时，函数还会保存一张 cluster 到 cell type
    的映射表，便于追踪本次手动注释使用的标签。

    未提供标签的 cluster 会根据 ``unknown_label`` 写成 ``NULL`` 或指定的未知
    标签；提供了但在 ``obs[groupby]`` 中不存在的 cluster 会直接报错。

    Examples
    --------
    使用显式字典进行手动注释::

        mapping = {
            "0": "CD4 T cells",
            "1": "CD14+ Monocytes",
            "2": "B cells",
            "3": "CD8 T cells",
            "4": "NK cells",
            "5": "FCGR3A+ Monocytes",
            "6": "Dendritic Cells",
            "7": "Megakaryocytes",
        }

        summary = sap.tl.manual_annotate_clusters(
            atlas,
            mapping,
            groupby="kmeans",
            obs_col="cell_type_manual",
        )

    使用 Scanpy 风格的有序列表进行手动注释::

        new_cluster_names = [
            "CD4 T cells",
            "CD14+ Monocytes",
            "B cells",
            "CD8 T cells",
            "NK cells",
            "FCGR3A+ Monocytes",
            "Dendritic Cells",
            "Megakaryocytes",
        ]

        summary = sap.tl.manual_annotate_clusters(
            atlas,
            new_cluster_names,
            groupby="kmeans",
        )

    绘制手动注释后的 UMAP 图::
        sap.pl.umap(atlas, color="cell_type_manual")
    """

    # 获取数据库连接
    conn = atlas.connection
    if conn is None:
        atlas.connect("r+")
        conn = atlas.connection

    # 检查 obs 表中是否存在 groupby 列
    obs_cols = [r[1] for r in conn.execute("PRAGMA table_info(obs)").fetchall()]

    if groupby not in obs_cols:
        raise ValueError(f"obs 中不存在聚类列: {groupby!r}")

    group_col = _quote_identifier(groupby)
    obs_label_col = _quote_identifier(obs_col)

    # 读取 obs 中真实存在的 cluster ID：数字 cluster 按数字顺序排列
    cluster_df = conn.execute(
        f"""
        SELECT CAST({group_col} AS TEXT) AS cluster_id
        FROM obs
        WHERE {group_col} IS NOT NULL
        GROUP BY 1
        ORDER BY TRY_CAST(cluster_id AS INTEGER), cluster_id
        """
    ).df()

    if len(cluster_df) == 0:
        raise ValueError(f"obs.{groupby} 中没有找到任何 cluster")

    cluster_ids = cluster_df["cluster_id"].astype(str).tolist()

    # ============================================================
    # 整理用户传入的手动注释
    #    支持两种形式：
    #    1）字典：{"0": "CD4 T cells", "1": "B cells"}
    #    2）列表：["CD4 T cells", "B cells", ...]
    # ============================================================

    if isinstance(cluster_to_cell_type, Mapping):
        mapping_df = pd.DataFrame({
            "cluster_id": [str(k) for k in cluster_to_cell_type.keys()],
            "cell_type": [str(v) for v in cluster_to_cell_type.values()],
        })

    else:
        if isinstance(cluster_to_cell_type, (str, bytes)):
            raise TypeError(
                "cluster_to_cell_type 必须是字典，或者是细胞类型名称组成的列表 / 元组，"
                "不能是单个字符串。"
            )

        labels = [str(v) for v in cluster_to_cell_type]

        if len(labels) != len(cluster_ids):
            raise ValueError(
                "手动注释标签数量与 cluster 数量不一致："
                f"你提供了 {len(labels)} 个标签，"
                f"但 obs.{groupby} 中有 {len(cluster_ids)} 个 cluster。"
                f"当前 cluster 顺序为: {cluster_ids}"
            )

        mapping_df = pd.DataFrame({
            "cluster_id": cluster_ids,
            "cell_type": labels,
        })

    # 检查是否有重复的 cluster ID
    duplicated = mapping_df["cluster_id"][mapping_df["cluster_id"].duplicated()].tolist()

    if duplicated:
        raise ValueError(f"手动注释中存在重复的 cluster ID: {duplicated}")

    # 检查用户提供的 cluster 是否真的存在于 obs[groupby]
    unknown_clusters = sorted(set(mapping_df["cluster_id"]) - set(cluster_ids))

    if unknown_clusters:
        raise ValueError(
            f"手动注释中包含 obs.{groupby} 中不存在的 cluster: {unknown_clusters}"
        )

    # 提示哪些 cluster 没有被手动注释
    #    这些 cluster 会被写成 NULL 或 unknown_label
    missing_clusters = sorted(set(cluster_ids) - set(mapping_df["cluster_id"]))

    if missing_clusters:
        if unknown_label is None:
            print(
                f"[manual_annotate_clusters] 提示：以下 cluster 没有提供手动注释，"
                f"将被写为 NULL: {missing_clusters}"
            )
        else:
            print(
                f"[manual_annotate_clusters] 提示：以下 cluster 没有提供手动注释，"
                f"将被写为 {unknown_label!r}: {missing_clusters}"
            )

    # 如果 obs_col 不存在，则创建该列
    if obs_col not in obs_cols:
        conn.execute(f"ALTER TABLE obs ADD COLUMN {obs_label_col} TEXT")

    # 每次运行都先清空旧注释
    #    这样下一次运行会覆盖上一次，不会保留旧标签
    if unknown_label is None:
        conn.execute(f"UPDATE obs SET {obs_label_col} = NULL")
    else:
        conn.execute(f"UPDATE obs SET {obs_label_col} = ?", [str(unknown_label)])

    # 注册临时映射表，并把手动注释写回 obs
    conn.register("_manual_annotation_tmp", mapping_df)

    try:
        conn.execute(
            f"""
            UPDATE obs AS o
            SET {obs_label_col} = m.cell_type
            FROM _manual_annotation_tmp AS m
            WHERE CAST(o.{group_col} AS TEXT) = CAST(m.cluster_id AS TEXT)
            """
        )

        # 保存 cluster -> cell type 的映射表
        if table_name is not None:
            table = _quote_identifier(table_name)

            saved_mapping_df = mapping_df.copy()
            saved_mapping_df.insert(0, "groupby", groupby)
            saved_mapping_df["obs_col"] = obs_col

            conn.register("_manual_annotation_save_tmp", saved_mapping_df)

            try:
                conn.execute(f"DROP TABLE IF EXISTS {table}")
                conn.execute(f"CREATE TABLE {table} AS SELECT * FROM _manual_annotation_save_tmp")
            finally:
                conn.unregister("_manual_annotation_save_tmp")

    finally:
        conn.unregister("_manual_annotation_tmp")

    # 保存数据库状态
    conn.execute("CHECKPOINT")

    # 是否返回汇总结果
    if not return_df:
        return None

    return conn.execute(
        f"""
        SELECT
            CAST({group_col} AS TEXT) AS cluster_id,
            {obs_label_col} AS cell_type,
            COUNT(*) AS n_cells
        FROM obs
        GROUP BY 1, 2
        ORDER BY TRY_CAST(cluster_id AS INTEGER), cluster_id, cell_type
        """
    ).df()


def _quote_identifier(name: str) -> str:
    """为 DuckDB SQL 标识符添加安全引用。

    该内部 helper 用于引用动态传入的表名或字段名，例如 ``groupby``、
    ``obs_col`` 和 ``table_name``。函数会转义名称中已有的双引号，并在外层
    添加双引号，避免字段名包含特殊字符、空格或 DuckDB 关键字时 SQL 解析失败。

    Parameters
    ----------
    name
        需要引用的 SQL 表名、列名或其他标识符。

    Returns
    -------
    str
        加双引号后的 SQL 标识符。

    Notes
    -----
    该函数只用于 SQL 标识符，不用于引用普通字符串值。字符串值应使用 DuckDB
    参数绑定传入。
    """
    return '"' + str(name).replace('"', '""') + '"'
