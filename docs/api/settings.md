# Global Settings

## Overview

These functions control package-level behavior that affects interactive
workflows, notebooks, scripts, and documentation examples.

Use them through the top-level `scatlaspy` namespace:

```python
import scatlaspy as sap

sap.set_progress(False)
sap.set_verbosity("info")
```

```{eval-rst}
.. currentmodule:: scatlaspy

.. autosummary::
   :toctree: generated/
   :nosignatures:

   set_progress
   set_verbosity
```

## Common Uses

- `set_progress(...)`: enable, disable, or restore automatic progress-bar
  behavior.
- `set_verbosity(...)`: control scAtlasPy logging output.
