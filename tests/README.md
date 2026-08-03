# scAtlasPy Tests

This directory defines the current public behavior contract for scAtlasPy.

The default pytest suite must be fast, deterministic, and independent of large
external datasets. It uses small synthetic AnnData objects to test the same
storage schema, preprocessing transforms, read-index construction, minibatch
reading, and workflow APIs used by atlas-scale runs.

Use these layers when adding tests:

- `test_storage_io.py`: Atlas schema, AnnData import/export, and accessor behavior.
- `test_preprocessing.py`: filtering, QC metrics, derived expression tables, HVG, and scale.
- `test_read_index_minibatch.py`: read-index correctness and dense minibatch semantics.
- `test_tools_workflow.py`: PCA, graph clustering, ranking, annotation, and public API smoke tests.
- Optional real-data or benchmark tests should be marked `realdata` or `slow` and
  should not run in the default quick suite.

Run the quick suite with:

```bash
python -m pytest -q
```
