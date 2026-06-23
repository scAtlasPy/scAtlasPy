"""Progress-bar helpers used across scAtlasPy."""

from __future__ import annotations

import os
import sys
from typing import Any

from tqdm.auto import tqdm as _tqdm


_PROGRESS_ENABLED: bool | None = None


def set_progress(enabled: bool | None = True) -> None:
    """Set global progress-bar visibility.

    Parameters
    ----------
    enabled
        ``True`` always enables progress bars, ``False`` disables them, and
        ``None`` restores automatic detection.
    """

    global _PROGRESS_ENABLED
    if enabled is not None and not isinstance(enabled, bool):
        raise TypeError("enabled must be True, False, or None")
    _PROGRESS_ENABLED = enabled


def _env_progress_setting() -> bool | None:
    value = os.environ.get("SCATLASPY_PROGRESS")
    if value is None:
        return None

    value = value.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    if value == "auto":
        return None

    return None


def _in_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__  # type: ignore[name-defined]
    except Exception:
        return False
    return shell == "ZMQInteractiveShell"


def _auto_disable() -> bool:
    env_setting = _env_progress_setting()
    if env_setting is not None:
        return not env_setting

    if _PROGRESS_ENABLED is not None:
        return not _PROGRESS_ENABLED

    if _in_notebook():
        return False

    stream = getattr(sys, "stderr", None)
    if stream is None:
        return True

    return not stream.isatty()


def progress(*args: Any, **kwargs: Any) -> Any:
    """Return a tqdm progress bar with scAtlasPy defaults.

    The wrapper keeps progress bars visible in interactive terminals and
    notebooks, but disables them when stderr is captured by non-interactive
    runners such as documentation builds or CI jobs.
    """

    if "disable" not in kwargs or kwargs["disable"] is None:
        kwargs["disable"] = _auto_disable()
    kwargs.setdefault("dynamic_ncols", True)
    kwargs.setdefault("leave", False)
    kwargs.setdefault("mininterval", 0.5)
    return _tqdm(*args, **kwargs)
