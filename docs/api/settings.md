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
   io.progress
```

## Common Uses

- `set_progress(...)`: enable, disable, or restore automatic progress-bar
  behavior.
- `set_verbosity(...)`: control scAtlasPy logging output.
- `sap.io.progress(...)`: create a progress bar using scAtlasPy's default
  progress-display policy.

### Verbosity Levels

`set_verbosity()` accepts the following levels:

| Level | Behavior |
|---|---|
| `"silence"` (default) | No scAtlasPy log output |
| `"error"` | Only errors |
| `"warning"` | Warnings and errors |
| `"info"` | Workflow progress messages |
| `"debug"` | Detailed per-step diagnostics |

Pass `None` to restore the default level (`"silence"`).
