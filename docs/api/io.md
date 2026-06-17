# 数据导入和导出：`sap.io`

`sap.io` 负责 h5ad、AnnData 和 Atlas 数据库之间的数据转换。所有 IO 函数同时提供 `sap.io.xxx(atlas, ...)` 和 `atlas.xxx(...)` 两种调用方式，推荐使用更简洁的 `atlas.xxx()` 写法。

```{eval-rst}
.. currentmodule:: scatlaspy

.. autosummary::
   :toctree: generated/
   :nosignatures:

   io.load_h5ad
   io.load_multi_format
   io.load_anndata
   io.gene_names_duplicated
   io.write_h5ad
   io.get_obs_df
   io.get_anndata
```

## 选择哪个导入函数

| 数据情况 | 推荐函数 |
|---|---|
| 单个 h5ad，任意规模 | `atlas.load_h5ad(path, load_type="order")` |
| 单个 h5ad，需要随机化 | `atlas.load_h5ad(path)` （默认 `load_type="random"`） |
| 多个 h5ad 合并 | `atlas.load_h5ad([paths], load_type="list_random")` |
| 已有内存中的 AnnData | `atlas.load_anndata(adata)` |
| 非 h5ad 小文件（.loom/.mtx/.csv 等） | `atlas.load_multi_format(path)` |

## load_h5ad 的 load_type

| `load_type` | 说明 |
|---|---|
| `"order"` | 顺序导入单个 h5ad，保留细胞顺序，导入 obsm/varm |
| `"random"`（默认） | 单个 h5ad 随机窗口导入 |
| `"list_random"` | 多个 h5ad 文件全局随机混合导入 |

## 两种调用方式

推荐使用 `atlas.xxx()` 实例方法，更短、更直观：

| `atlas.xxx()` 写法（推荐） | `sap.io.xxx(atlas, ...)` 写法 |
|---|---|
| `atlas.load_h5ad("file.h5ad", load_type="order")` | `sap.io.load_h5ad("file.h5ad", atlas, load_type="order")` |
| `atlas.load_anndata(adata)` | `sap.io.load_anndata(adata, atlas)` |
| `atlas.load_multi_format("file.loom")` | `sap.io.load_multi_format("file.loom", atlas)` |
| `atlas.write_h5ad("out.h5ad")` | `sap.io.write_h5ad(atlas, "out.h5ad")` |
| `atlas.get_obs_df()` | `sap.io.get_obs_df(atlas)` |
| `atlas.get_anndata(ids, use_data="data_log1p")` | `sap.io.get_anndata(atlas, ids, use_data="data_log1p")` |
| `atlas.gene_names_duplicated()` | `sap.io.gene_names_duplicated(atlas)` |

## 导出说明

`write_h5ad()` 当前主路径默认把 `X_HyS_data.data` 写入 h5ad 的 `X`。如需指定其他表达字段（如 `data_log1p` 或 `data_scale`），使用 `get_anndata(use_data=...)`。

`get_obs_df()` 导出 obs 细胞信息表为 pandas DataFrame，支持 `columns` 参数指定只导出某些列。

`get_anndata()` 导出细胞子集为 AnnData 对象，支持 `use_data` 选择表达字段，`include_obsm`/`include_varm` 控制是否附带降维结果。
