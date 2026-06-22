# Citation

If you use scAtlasPy in academic work, software, teaching material, or
benchmark reports, please cite the project so that the development work can be
properly credited and the exact software version can be traced.

## Recommended Citation

A formal manuscript citation or archived-release DOI will be added when it
becomes available. Until then, please cite the project repository and specify
the version or commit used in your analysis.

```text
Han Xu, Yangzhan Ye, and the scAtlasPy contributors.
scAtlasPy: A scalable Python platform for atlas-scale single-cell omics
analysis beyond in-memory limits.
Version 0.1.0. https://github.com/scAtlasAnalysis/scAtlaspy
```

When reporting results, include:

- the scAtlasPy version or commit hash;
- the date the analysis was run;
- any major workflow settings that affect reproducibility, such as filtering
  rules, expression representation, minibatch parameters, or external methods.

## BibTeX

The following BibTeX entry can be used as a temporary software citation before
a formal DOI or paper citation is available:

```bibtex
@software{scatlaspy,
  title = {scAtlasPy: A scalable Python platform for atlas-scale single-cell omics analysis beyond in-memory limits},
  author = {Xu, Han and Ye, Yangzhan and the scAtlasPy contributors},
  version = {0.1.0},
  url = {https://github.com/scAtlasAnalysis/scAtlaspy},
  note = {Please include the version or commit hash used in the analysis}
}
```

## Citing Dependencies and Methods

scAtlasPy builds on the broader scientific Python and single-cell analysis
ecosystem. Depending on the workflow, users should also cite relevant
underlying methods or libraries, such as AnnData, Scanpy, DuckDB,
scikit-learn, UMAP, NumPy, SciPy, pandas, and any external method used together
with scAtlasPy.

For analyses that rely on external packages after exporting an Atlas subset,
please cite both scAtlasPy and the external package or method used for the
downstream analysis.

## Citation Updates

This page will be updated when an official manuscript, release archive, DOI, or
preferred citation format becomes available. For citation questions, contact
the maintainers or open an issue in the project repository.
