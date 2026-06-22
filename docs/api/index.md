# API Reference

Use the API reference to look up the public classes, methods, functions, and
configuration options provided by scAtlasPy.

If you are new to scAtlasPy, begin with {doc}`../installation` and the
{doc}`../tutorials/index` before consulting individual API pages.

scAtlasPy is conventionally imported as:

```python
import scatlaspy as sap
```

## API Sections

- {doc}`Global settings <settings>`  
  Configure progress bars, logging, and other package-wide behavior.

- {doc}`Atlas object <atlas>`  
  Create, open, inspect, query, and manage an Atlas through `sap.Atlas`.

- {doc}`Input and output <io>`  
  Import data into an Atlas and export selected data or analysis results through
  `sap.io`.

- {doc}`Preprocessing <preprocessing>`  
  Perform quality control, filtering, normalization, transformation, highly
  variable gene selection, and scaling through `sap.pp`.

- {doc}`Analysis tools <tools>`  
  Run dimensionality reduction, clustering, marker analysis, annotation, and
  other downstream computations through `sap.tl`.

- {doc}`Plotting <plotting>`  
  Visualize quality-control metrics, dimensionality reductions, clusters,
  annotations, and marker expression through `sap.pl`.

```{note}
Many scAtlasPy operations update the current Atlas and persist their outputs in
the `.sasql` database rather than returning a new `AnnData` object.

Each API entry should be consulted for its Python return value, results written
to the Atlas, required preprocessing state, and whether existing results may be
replaced.
```

```{important}
This reference documents the supported public API. Internal tables, helper
functions, and undocumented storage details should not be treated as stable
interfaces.
```

```{toctree}
:hidden:
:maxdepth: 1

Global Settings <settings>
Atlas <atlas>
I/O <io>
Preprocessing <preprocessing>
Tools <tools>
Plotting <plotting>
```