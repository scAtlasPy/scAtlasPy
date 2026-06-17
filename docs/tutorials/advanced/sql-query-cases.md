# SQL 查询案例

本教程通过完整案例演示如何用 SQL 直接查询 Atlas 数据库中的分析结果。`Atlas.query()` 返回 pandas DataFrame，适合继续保存为 CSV、画图或交给自己的算法。不需要深入理解数据库原理。

## 数据库核心表速查

每次查询前先了解数据库结构：

| 表名 | 存储内容 | 关键列 |
|---|---|---|
| `obs` | 细胞元数据 | `atlas_cell_id`, `atlas_cell_name`, `kmeans`, `cell_type_auto`, `filter_cells`, `cell_total_counts`, `pct_counts_mt` 等 |
| `var` | 基因元数据 | `atlas_gene_id`, `atlas_gene_name`, `filter_genes`, `highly_variable_genes`, `zero_scale_transform` |
| `X_HyS_data` | 稀疏表达值（长表） | `atlas_cell_id`, `atlas_gene_id`, `data`, `data_log1p`, `data_scale`, `data_normalize`, `data_sqrt` |
| `X_HyS_indptr` | CSR 行指针 | `atlas_cell_id`, `indptr` |
| `X_HyS_data_filtered` | 过滤后的表达值 | `filter_cell_id`, `filter_gene_id`, `data`, `tid` |
| `X_HyS_indptr_filtered` | 过滤后的 CSR 行指针 | `filter_cell_id`, `indptr` |
| `obsm_X_pca` | PCA 坐标 | `atlas_cell_id`, `pc0`, `pc1`, ... |
| `obsm_X_umap` | UMAP 坐标 | `atlas_cell_id`, `umap1`, `umap2` |
| `varm_PCs` | 基因 PC loading | `atlas_gene_id`, `pc0`, `pc1`, ... |
| `uns_pca_stats` | PCA 统计 | `pc_index`, `variance`, `variance_ratio` |
| `uns_umap_params` | UMAP 参数 | 参数记录的 JSON/键值对 |
| `uns_umap_eval` | UMAP 质量评估 | trustworthiness, knn_overlap 等指标 |

---

## 案例 1：基础检查与探索

### 1.1 查看表结构和字段

```python
# 列出所有表
print(atlas.query("SHOW TABLES"))

# 查看各表有哪些列
print(atlas.query("PRAGMA table_info(obs)"))
print(atlas.query("PRAGMA table_info(var)"))
print(atlas.query("PRAGMA table_info(X_HyS_data)"))
print(atlas.query("PRAGMA table_info(obsm_X_pca)"))
print(atlas.query("PRAGMA table_info(obsm_X_umap)"))
```

### 1.2 获取各表的行数概况

```python
summary = atlas.query("""
    SELECT
        (SELECT COUNT(*) FROM obs) AS n_cells,
        (SELECT COUNT(*) FROM var) AS n_genes,
        (SELECT COUNT(*) FROM X_HyS_data) AS n_nonzero,
        (SELECT COUNT(*) FROM obsm_X_pca) AS n_pca_cells,
        (SELECT COUNT(*) FROM obsm_X_umap) AS n_umap_cells
""")
print(summary)
```

### 1.3 查看几个细胞的完整 obs 信息

```python
print(atlas.query("SELECT * FROM obs LIMIT 5"))
```

---

## 案例 2：按 cluster 汇总分析

### 2.1 统计每个 cluster 的细胞数

```python
cluster_size = atlas.query("""
    SELECT kmeans, COUNT(*) AS n_cells
    FROM obs
    GROUP BY kmeans
    ORDER BY kmeans
""")
print(cluster_size)
```

### 2.2 按 cluster 统计 QC 指标的中位数

```python
cluster_qc = atlas.query("""
    SELECT
        kmeans,
        COUNT(*) AS n_cells,
        MEDIAN(cell_total_counts) AS median_counts,
        MEDIAN(n_genes_by_counts) AS median_genes,
        MEDIAN(pct_counts_mt) AS median_pct_mt
    FROM obs
    WHERE kmeans IS NOT NULL
    GROUP BY kmeans
    ORDER BY kmeans
""")
print(cluster_qc)
```

### 2.3 找出每个 cluster 中细胞数占比

```python
cluster_pct = atlas.query("""
    SELECT
        kmeans,
        COUNT(*) AS n_cells,
        ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS pct
    FROM obs
    WHERE kmeans IS NOT NULL
    GROUP BY kmeans
    ORDER BY kmeans
""")
print(cluster_pct)
```

---

## 案例 3：查询 marker gene 表达

### 3.1 查询基因在部分细胞中的表达

```python
cst3_expr = atlas.query("""
    SELECT x.atlas_cell_id, v.atlas_gene_name, x.data_log1p
    FROM X_HyS_data AS x
    JOIN var AS v
      ON x.atlas_gene_id = v.atlas_gene_id
    WHERE v.atlas_gene_name = 'CST3'
    LIMIT 20
""")
print(cst3_expr)
```

把 `'CST3'` 换成你关心的基因名，`x.data_log1p` 换成需要的表达字段。

### 3.2 同时查多个基因

```python
multi_genes = atlas.query("""
    SELECT x.atlas_cell_id, v.atlas_gene_name, x.data_log1p
    FROM X_HyS_data AS x
    JOIN var AS v
      ON x.atlas_gene_id = v.atlas_gene_id
    WHERE v.atlas_gene_name IN ('CST3', 'NKG7', 'PPBP', 'MS4A1')
    LIMIT 100
""")
print(multi_genes)
```

### 3.3 按 cluster 计算基因平均表达

```python
gene_by_cluster = atlas.query("""
    SELECT
        o.kmeans,
        v.atlas_gene_name,
        AVG(x.data_log1p) AS mean_log1p,
        COUNT(*) AS n_nonzero_cells
    FROM X_HyS_data AS x
    JOIN var AS v ON x.atlas_gene_id = v.atlas_gene_id
    JOIN obs AS o ON x.atlas_cell_id = o.atlas_cell_id
    WHERE v.atlas_gene_name IN ('CST3', 'NKG7', 'PPBP', 'MS4A1')
    GROUP BY o.kmeans, v.atlas_gene_name
    ORDER BY v.atlas_gene_name, o.kmeans
""")
print(gene_by_cluster)
```

这个查询连接了三张核心表：
- `X_HyS_data`：表达值（长表，每行一个 cell-gene 对的非零值）
- `var`：基因名映射（`atlas_gene_id` → `atlas_gene_name`）
- `obs`：细胞分组信息（`atlas_cell_id` → `kmeans`）

### 3.4 计算表达该基因的细胞比例（含隐式零值）

```python
gene_pct = atlas.query("""
    WITH total_per_cluster AS (
        SELECT kmeans, COUNT(*) AS n_total
        FROM obs
        WHERE kmeans IS NOT NULL
        GROUP BY kmeans
    )
    SELECT
        o.kmeans,
        v.atlas_gene_name,
        COUNT(x.data_log1p) * 100.0 / MAX(t.n_total) AS pct_expressing
    FROM obs AS o
    CROSS JOIN var AS v
    LEFT JOIN X_HyS_data AS x
      ON o.atlas_cell_id = x.atlas_cell_id
     AND v.atlas_gene_id = x.atlas_gene_id
    JOIN total_per_cluster AS t ON o.kmeans = t.kmeans
    WHERE v.atlas_gene_name IN ('CST3', 'NKG7', 'PPBP')
      AND o.kmeans IS NOT NULL
    GROUP BY o.kmeans, v.atlas_gene_name
    ORDER BY v.atlas_gene_name, o.kmeans
""")
print(gene_pct)
```

这里用 `CROSS JOIN` 补全隐式零值：即使某细胞没有表达某基因（X_HyS_data 中没有对应行），也会在结果中留下一行，`COUNT(x.data_log1p)` 会正确计数非零值。

---

## 案例 4：筛选特定细胞子集

### 4.1 只看通过 QC 过滤的细胞

```python
filtered_stats = atlas.query("""
    SELECT
        COUNT(*) AS n_filtered_cells,
        AVG(cell_total_counts) AS avg_counts,
        AVG(n_genes_by_counts) AS avg_genes,
        AVG(pct_counts_mt) AS avg_pct_mt
    FROM obs
    WHERE filter_cells IS NOT NULL
""")
print(filtered_stats)
```

### 4.2 只看特定 cluster

```python
cluster0_genes = atlas.query("""
    SELECT COUNT(*) AS n_nonzero
    FROM X_HyS_data AS x
    JOIN obs AS o ON x.atlas_cell_id = o.atlas_cell_id
    WHERE o.kmeans = 0
""")
print(cluster0_genes)
```

### 4.3 抽取部分细胞用于小规模调试

```python
sample_cells = atlas.query("""
    SELECT atlas_cell_id, kmeans, cell_total_counts
    FROM obs
    WHERE kmeans IS NOT NULL
    USING SAMPLE 1000 ROWS
""")
print(sample_cells.head())
```

`USING SAMPLE N ROWS` 是 DuckDB 的语法，在 SQL 层高效抽样，不需要先把全量数据加载到 Python。

---

## 案例 5：提取绘图数据

### 5.1 提取 UMAP 坐标 + 聚类标签（用于自定义 ggplot/plotly）

```python
umap_data = atlas.query("""
    SELECT
        u.umap1,
        u.umap2,
        CAST(o.kmeans AS VARCHAR) AS cluster,
        o.cell_total_counts,
        o.pct_counts_mt
    FROM obsm_X_umap AS u
    JOIN obs AS o ON u.atlas_cell_id = o.atlas_cell_id
    USING SAMPLE 50000 ROWS
""")
print(umap_data.head())

# 可以直接用 matplotlib/seaborn/plotly 画图
import matplotlib.pyplot as plt
for cluster in sorted(umap_data["cluster"].unique()):
    sub = umap_data[umap_data["cluster"] == cluster]
    plt.scatter(sub["umap1"], sub["umap2"], s=1, label=cluster)
plt.legend(markerscale=5)
plt.show()
```

### 5.2 提取基因表达 + UMAP 坐标（画 gene expression UMAP）

```python
gene_umap = atlas.query("""
    SELECT
        u.umap1,
        u.umap2,
        COALESCE(x.data_log1p, 0.0) AS expression
    FROM obsm_X_umap AS u
    JOIN var AS v ON v.atlas_gene_name = 'CST3'
    LEFT JOIN X_HyS_data AS x
      ON u.atlas_cell_id = x.atlas_cell_id
     AND v.atlas_gene_id = x.atlas_gene_id
    USING SAMPLE 50000 ROWS
""")

plt.scatter(
    gene_umap["umap1"], gene_umap["umap2"],
    c=gene_umap["expression"], cmap="viridis", s=1
)
plt.colorbar(label="CST3 log1p")
plt.show()
```

### 5.3 提取 dotplot 所需统计量（自主实现自定义 dotplot）

```python
dotplot_data = atlas.query("""
    SELECT
        o.kmeans,
        v.atlas_gene_name,
        AVG(COALESCE(x.data_log1p, 0.0)) AS mean_expr,
        (COUNT(x.data_log1p) * 100.0 / COUNT(*)) AS pct_expr
    FROM obs AS o
    CROSS JOIN var AS v
    LEFT JOIN X_HyS_data AS x
      ON o.atlas_cell_id = x.atlas_cell_id
     AND v.atlas_gene_id = x.atlas_gene_id
    WHERE o.kmeans IS NOT NULL
      AND v.atlas_gene_name IN ('IL7R', 'CD79A', 'MS4A1', 'CD8A', 'LYZ', 'NKG7', 'PPBP')
    GROUP BY o.kmeans, v.atlas_gene_name
    ORDER BY v.atlas_gene_name, o.kmeans
""")
print(dotplot_data)
```

---

## 案例 6：查询 PCA 和 UMAP 质量指标

### 6.1 PCA 方差解释率

```python
pca_stats = atlas.query("""
    SELECT pc_index, variance, variance_ratio
    FROM uns_pca_stats
    ORDER BY pc_index
    LIMIT 10
""")
print(pca_stats)
```

### 6.2 查看 UMAP 评估指标

```python
# 查看 uns_umap_eval 中有哪些指标
umap_eval_table = atlas.query("SELECT * FROM uns_umap_eval")
print(umap_eval_table)
```

---

## 案例 7：导出查询结果

### 保存为 CSV

```python
cluster_qc = atlas.query("""
    SELECT kmeans, COUNT(*) AS n_cells, MEDIAN(pct_counts_mt) AS median_pct_mt
    FROM obs WHERE kmeans IS NOT NULL
    GROUP BY kmeans ORDER BY kmeans
""")
cluster_qc.to_csv("cluster_qc_summary.csv", index=False)
```

### 导出全部 obs 为 pandas 再筛选

```python
# 方式 1：用内置函数
obs_df = atlas.get_obs_df()
# 然后可以用 pandas 做任何分析

# 方式 2：用 SQL 直接查
obs_df = atlas.query("SELECT * FROM obs WHERE filter_cells IS NOT NULL")
```

---

## 常用表达字段速查

`X_HyS_data` 表中的表达字段：

| 字段 | 含义 | 典型值范围 |
|---|---|---|
| `data` | 原始表达值（counts） | 0 到几万 |
| `data_normalize` | 标准化后的值 | 0 到几百 |
| `data_log1p` | 标准化 + log1p | 0 到 ~10 |
| `data_scale` | z-score 标准化 | -3 到 +3 |
| `data_sqrt` | sqrt 变换 | 0 到几百 |

画图和 marker gene 分析一般用 `data_log1p`，PCA 用 `data_scale`。

---

## SQL 查询性能提示

| 场景 | 建议 |
|---|---|
| 查询全表概况 | 用 `COUNT(*)` 不加 WHERE，很快 |
| 查询少量细胞 | 加 `LIMIT`，避免返回百万行 |
| 多表 JOIN | 先在小表上做 WHERE 过滤，减少 JOIN 的数据量 |
| 大数据抽样 | 用 `USING SAMPLE N ROWS` 在 SQL 层抽样 |
| 只要几列 | 明确写 `SELECT col1, col2` 而不是 `SELECT *` |
| 重复查询 | 把中间结果存为 TEMP TABLE 或 Python 变量 |
