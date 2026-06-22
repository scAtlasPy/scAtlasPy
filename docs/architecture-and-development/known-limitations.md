# Known Limitations

This page records limitations of the current scAtlasPy release that users and
developers should consider when designing workflows or interpreting results.

The goal is to distinguish currently supported behavior from capabilities that
are not yet implemented or stabilized.

# 开发完成前要解决的问题

两个文档未检查，对应的底层支撑未实现。
Train Logistic Regression with Minibatches
Train a PyTorch Classifier with Minibatches

对仓库，issue等页面的链接都是占位的，未替换真实页面
github的readme里对本docs的也是占位的链接

docs未做系统性的审查


## KMeans Clustering

`sap.tl.kmeans()` uses a minibatch implementation of KMeans.

It is designed to support clustering on large datasets through bounded-memory
computation, but it is not equivalent to graph-based clustering methods such as
Leiden or Louvain.

In particular, `sap.tl.kmeans()`:

- assigns cells to a predefined number of clusters;
- operates on a numerical representation such as PCA coordinates;
- does not construct a cell-neighbor graph;
- does not optimize a graph-community objective;
- may produce different cluster structures from Leiden or Louvain.

```{important}
Results from `sap.tl.kmeans()` should be interpreted as KMeans clusters rather
than as graph communities.

Parameters such as `n_clusters` are therefore not directly comparable to the
resolution parameters used by graph-based clustering methods.
```

When graph-based clustering is required, a focused cell population can be
retrieved as an AnnData object and analyzed with a compatible external method:

```python
obs = atlas.get_obs_df(
    columns=[
        "atlas_cell_id",
        "cell_type_manual",
    ]
)

selected_ids = obs.loc[
    obs["cell_type_manual"].eq("T cell"),
    "atlas_cell_id",
].tolist()

adata = atlas.get_anndata(
    selected_ids,
    use_data="data_log1p",
)
```

The selected data can then be analyzed with a graph-based workflow in another
single-cell package.

See
{doc}`../cross-platform-workflows/use-external-methods-on-atlas-subsets`
for the complete workflow.

## Reporting Additional Limitations

This page will be updated as unsupported, experimental, or version-dependent
behavior is identified.

Users and contributors are encouraged to report cases where:

- documented behavior differs from the implementation;
- an interface works only for specific data layouts or scales;
- a workflow requires access to undocumented internal tables;
- an analysis method has assumptions that are not clearly stated.

When reporting a limitation, include the scAtlasPy version, a minimal example,
and the expected and observed behavior.