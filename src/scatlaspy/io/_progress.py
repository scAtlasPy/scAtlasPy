from __future__ import annotations
import os
import sys
from typing import Any
from tqdm.auto import tqdm as _tqdm
_PROGRESS_ENABLED: bool | None = None


def set_progress(enabled: bool | None = True) -> None:

    """Set the global progress bar display policy.

    This function controls whether the ``scatlaspy.io.progress`` wrapper displays
    ``tqdm`` progress bars. Once set, it affects progress bars created later through
    ``progress(...)`` in the current Python process.

    Parameters
    ----------
    enabled
        Progress bar display policy.

        - ``True``: always show progress bars;
        - ``False``: always disable progress bars;
        - ``None``: restore automatic detection and decide whether to show progress
          bars according to the runtime environment.

    Returns
    -------
    None
        Only updates the module-level global setting and does not return an object.

    Examples
    --------
    Disable all scAtlasPy progress bars in a script::

        sap.io.set_progress(False)

    Restore automatic detection::

        sap.io.set_progress(None)
    """

    global _PROGRESS_ENABLED
    if enabled is not None and not isinstance(enabled, bool):
        raise TypeError("enabled must be True, False, or None")
    _PROGRESS_ENABLED = enabled


def _env_progress_setting() -> bool | None:
    """Read the progress bar setting from environment variables.

    This internal function reads the ``SCATLASPY_PROGRESS`` environment variable
    and converts common strings into boolean values or automatic mode.

    Returns
    -------
    bool or None
        ``True`` means progress bars are forced to be shown, ``False`` means
        progress bars are forced to be disabled, and ``None`` means no setting
        is provided or automatic detection is used.

    Notes
    -----
    Supported enabled values include ``1``, ``true``, ``yes``, and ``on``;
    supported disabled values include ``0``, ``false``, ``no``, and ``off``;
    ``auto`` is interpreted as ``None``.
    """

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
    """Determine whether the current code is running in a Jupyter Notebook environment.

    Returns
    -------
    bool
        Returns ``True`` if the detected IPython shell type is
        ``ZMQInteractiveShell``; otherwise returns ``False``.
    """
    try:
        shell = get_ipython().__class__.__name__  # type: ignore[name-defined]
    except Exception:
        return False
    return shell == "ZMQInteractiveShell"


def _auto_disable() -> bool:
    """Automatically determine whether to disable progress bars according to the environment.

    The priority order is: the ``SCATLASPY_PROGRESS`` environment variable, the global
    ``set_progress`` setting, Notebook environment detection, and whether stderr is
    an interactive terminal.

    Returns
    -------
    bool
        Boolean value used by ``tqdm(disable=...)``. ``True`` means disabling the
        progress bar, and ``False`` means showing the progress bar.
    """
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
    """Create a tqdm progress bar with scAtlasPy default settings.

    This function is the unified ``tqdm`` wrapper used internally by the project.
    By default, it shows progress bars in interactive terminals and Notebooks, and
    automatically disables progress bars in non-interactive environments such as CI,
    documentation builds, or cases where stderr is captured, reducing log noise.

    Unless the user explicitly passes ``disable``, this function calls
    ``_auto_disable`` to automatically decide whether to show the progress bar.
    It also sets ``dynamic_ncols=True``, ``leave=False``, and ``mininterval=0.5``
    as default values.

    Parameters
    ----------
    args
        Positional arguments passed through to ``tqdm.auto.tqdm``.
    kwargs
        Keyword arguments passed through to ``tqdm.auto.tqdm``.

    Returns
    -------
    Any
        The progress bar object or iterator wrapper returned by ``tqdm``.

    Examples
    --------
    Use the unified progress bar in an internal loop::

        for i in progress(range(100), desc="load"):
            ...
    """

    if "disable" not in kwargs or kwargs["disable"] is None:
        kwargs["disable"] = _auto_disable()
    kwargs.setdefault("dynamic_ncols", True)
    kwargs.setdefault("leave", False)
    kwargs.setdefault("mininterval", 0.5)
    return _tqdm(*args, **kwargs)
