---
orphan: true
---

# Should I use raw counts, log1p-transformed data, or scaled data?

There is no universally best expression representation for every single-cell
analysis. Raw counts, log1p-transformed values, and scaled values preserve
different properties of the matrix and give genes different effective weights.

The right choice depends on the downstream method and on the biological
question. In particular, decide whether genes with larger expression variance
should contribute more strongly, or whether selected genes should have more
comparable influence.

## Raw Counts

Raw counts preserve the original count distribution, the mean-variance
relationship, and information related to sequencing depth.

Use raw counts for methods that explicitly model count data, such as:

- Poisson or negative binomial models;
- count-based differential expression methods;
- models that explicitly account for library size;
- generative models based on count likelihoods;
- any method that specifically requires integer counts.

Raw counts are generally not recommended as direct input for standard PCA,
Euclidean distances, neighborhood graph construction, or clustering methods that
do not model count distributions. Highly expressed genes usually have much
larger absolute variance in raw count data, and high-depth cells may appear more
distant simply because they contain more counts.

Unless the downstream method is designed for raw counts, normalized and
log1p-transformed values are usually more appropriate.

## Log1p-Transformed HVGs Without Scaling

Log1p-transformed highly variable genes, without scaling, preserve differences
in variability among selected genes. Genes with larger expression variation
contribute more strongly to PCA and cell-to-cell distances. This can be
desirable when larger expression changes are considered more informative and
should naturally receive greater weight.

This representation has several useful properties:

- relative differences in gene-level variability are preserved;
- strongly varying genes naturally contribute more to the analysis;
- small fluctuations in low-variance genes are not artificially amplified;
- information about expression magnitude is partially retained.

The log1p transformation compresses the dynamic range:

```text
x' = log(1 + x)
```

When scaling is not used, scAtlasPy generally recommends normalized
log1p-transformed HVGs rather than raw counts:

```text
raw counts
-> library-size normalization
-> log1p transformation
-> highly variable gene selection
-> PCA
```

In scAtlasPy, this means building the analysis view from `data_log1p` and
restricting it to highly variable genes:

```python
atlas.build_read_index(
    cell_condition="filter_cells",
    gene_condition="filter_genes",
    use_hvg=True,
    use_data="data_log1p",
)
```

## Scaled Data

Scaling usually means centering and standardizing each gene:

```text
z_ij = (x_ij - mean_j) / sd_j
```

After scaling, each selected gene has approximately zero mean and unit variance.
PCA therefore focuses more on coordinated expression patterns among genes and
less on differences in their absolute variance.

Scaling can be useful when you want to:

- prevent a small number of highly expressed or highly variable genes from
  dominating PCA;
- give selected genes more comparable influence;
- emphasize correlation and co-expression patterns;
- allow lower-expression but biologically informative genes to contribute more
  strongly.

Scaling is not inherently better than using unscaled data. It changes the
weighting of the analysis by making selected genes contribute approximately
equal variance.

## Why Scale Highly Variable Genes Rather Than All Genes?

Scaling can strongly amplify low-variance genes. A gene with little meaningful
biological variation may contain weak technical fluctuations; dividing by a
small standard deviation can raise those fluctuations to a variance comparable
to that of an informative gene.

If all genes are scaled, many low-information genes may receive weights similar
to genuinely informative genes. This can dilute biological structure, amplify
technical noise, distribute variance across many uninformative dimensions, and
reduce the stability of PCA, neighborhood graphs, and clustering.

For this reason, when scaling is used, select highly variable genes first and
build the PCA analysis view from scaled HVGs:

```text
raw counts
-> library-size normalization
-> log1p transformation
-> highly variable gene selection
-> scaling
-> PCA
```

In scAtlasPy, this corresponds to using `data_scale` together with `use_hvg=True`:

```python
atlas.build_read_index(
    cell_condition="filter_cells",
    gene_condition="filter_genes",
    use_hvg=True,
    use_data="data_scale",
)
```

## Why Can Scaling Reduce PCA Explained Variance?

It is common for the explained variance ratio of the first few PCs to decrease
after scaling.

Without scaling, variance may be concentrated in a small number of highly
expressed or highly variable genes. These genes can create a few dominant
directions, producing large explained variance ratios for the first PCs.

After scaling, each selected gene contributes approximately the same total
variance. The variance is distributed more evenly across genes and principal
components. A lower explained variance ratio after scaling does not necessarily
indicate a worse PCA representation; it may simply mean that the representation
is no longer dominated by a small number of high-variance genes.

For PBMC3K, the same pattern appears in both Scanpy and scAtlasPy. The values
below are cumulative explained variance ratios recomputed on PBMC3K after the
same cell and gene filters and HVG selection:

| Implementation and PCA input | Cumulative explained variance, first 10 PCs | Cumulative explained variance, first 30 PCs |
| --- | ---: | ---: |
| scAtlasPy, log-normalized HVGs using `data_log1p` | 0.2045 | 0.2615 |
| scAtlasPy, scaled HVGs using `data_scale` | 0.0713 | 0.1049 |
| Scanpy, log-normalized HVGs, no scaling | 0.2045 | 0.2621 |
| Scanpy, scaled HVGs | 0.0752 | 0.1132 |

The comparison shows that explained variance changes primarily with the input
representation: log-normalized HVGs concentrate more variance in the leading
PCs, while scaled HVGs distribute variance more evenly across components.

For single-cell analysis, PCA quality should not be judged only by the variance
explained by the first few PCs. Also check whether known cell types and states
are preserved, whether technical factors dominate leading PCs, whether clusters
are stable, whether marker genes remain interpretable, and whether rare
populations are retained.

## Which Representation Should I Choose?

| Analysis goal | Recommended representation |
| --- | --- |
| Model the original sequencing count distribution | Raw counts |
| Use a Poisson, negative binomial, or count-likelihood model | Raw counts |
| Preserve differences in variability among selected genes | Normalized log1p-transformed highly variable genes |
| Reduce the dominance of highly expressed genes without equalizing all gene variances | Normalized log1p-transformed highly variable genes |
| Give selected genes more comparable influence in PCA | Scaled highly variable genes |
| Emphasize coordinated expression patterns rather than absolute variance | Scaled highly variable genes |
| Perform routine PCA and clustering without scaling | Normalized log1p-transformed highly variable genes |
| Perform routine PCA and clustering with scaling | Scaled highly variable genes |
| Perform differential expression analysis | Follow the input requirements of the selected statistical method |

## Practical Recommendation

For most routine analyses:

- use **normalized log1p-transformed highly variable genes** when you want to
  preserve meaningful differences in gene-level variance;
- use **scaled highly variable genes** when you want selected genes to have more
  comparable influence;
- use **raw counts only when the downstream method explicitly models count data**.

Always preserve the original count matrix. Do not overwrite it with
log1p-transformed or scaled values, because later analyses such as count-based
differential expression or generative modeling may still require raw counts.

scAtlasPy stores multiple expression representations side by side, so you can
choose the input field for each downstream task:

```python
# Preserve expression-magnitude structure
atlas.build_read_index(use_hvg=True, use_data="data_log1p")

# Balance the contribution of selected HVGs
atlas.build_read_index(use_hvg=True, use_data="data_scale")
```

When interpreting results, remember that `sap.tl.pca()` uses an incremental,
minibatch-based implementation so that PCA can run on atlas-scale data without
materializing the full matrix in memory. Small differences from exact in-memory
PCA implementations such as Scanpy or `sklearn.decomposition.PCA` are expected.
