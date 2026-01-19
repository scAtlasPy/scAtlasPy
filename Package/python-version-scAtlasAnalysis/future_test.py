import scatlaspy as sap

# 创建一个atlas对象，该对象管理与duckdb的各种交互，包括提交sql
atlas=sap.Atlas("mouse_brain_atlas")

# 从h5ad文件直接读取数据，写到atlas所管理的数据库中
sap.io.load_h5ad("mouse_brain_atlas.h5ad",atlas)

# 过滤低表达的细胞，
sap.pp.filter_cells(atlas, min_genes=3, min_total_counts=500)
