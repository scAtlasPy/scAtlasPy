from __future__ import annotations
import os
import sys
from typing import Any
from tqdm.auto import tqdm as _tqdm
_PROGRESS_ENABLED: bool | None = None


def set_progress(enabled: bool | None = True) -> None:

    """设置全局进度条显示策略。

    该函数用于控制 ``scatlaspy.io.progress`` 包装器是否显示 ``tqdm`` 进度条。
    设置后会影响当前 Python 进程中后续通过 ``progress(...)`` 创建的进度条。

    Parameters
    ----------
    enabled
        进度条显示策略。

        - ``True``：始终显示进度条；
        - ``False``：始终关闭进度条；
        - ``None``：恢复自动检测，根据运行环境决定是否显示。

    Returns
    -------
    None
        只更新模块级全局设置，不返回对象。

    Examples
    --------
    在脚本中关闭所有 scAtlasPy 进度条::

        sap.io.set_progress(False)

    恢复自动检测::

        sap.io.set_progress(None)
    """

    global _PROGRESS_ENABLED
    if enabled is not None and not isinstance(enabled, bool):
        raise TypeError("enabled must be True, False, or None")
    _PROGRESS_ENABLED = enabled


def _env_progress_setting() -> bool | None:
    """读取环境变量中的进度条设置。

    该内部函数读取 ``SCATLASPY_PROGRESS`` 环境变量，并将常见字符串转换为
    布尔值或自动模式。

    Returns
    -------
    bool or None
        ``True`` 表示强制显示，``False`` 表示强制关闭，``None`` 表示没有设置
        或使用自动检测。

    Notes
    -----
    支持的开启值包括 ``1``、``true``、``yes``、``on``；
    支持的关闭值包括 ``0``、``false``、``no``、``off``；
    ``auto`` 会被解释为 ``None``。
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
    """判断当前代码是否运行在 Jupyter Notebook 环境中。

    Returns
    -------
    bool
        如果检测到 IPython shell 类型为 ``ZMQInteractiveShell``，返回 ``True``；
        否则返回 ``False``。
    """
    try:
        shell = get_ipython().__class__.__name__  # type: ignore[name-defined]
    except Exception:
        return False
    return shell == "ZMQInteractiveShell"


def _auto_disable() -> bool:
    """根据环境自动判断是否关闭进度条。

    判断优先级为：环境变量 ``SCATLASPY_PROGRESS``、全局
    ``set_progress`` 设置、Notebook 环境检测、stderr 是否为交互式终端。

    Returns
    -------
    bool
        返回给 ``tqdm(disable=...)`` 使用的布尔值。``True`` 表示关闭进度条，
        ``False`` 表示显示进度条。
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
    """创建带 scAtlasPy 默认设置的 tqdm 进度条。

    该函数是项目内部统一使用的 ``tqdm`` 包装器。它会在交互式终端和 Notebook
    中默认显示进度条，在 CI、文档构建或 stderr 被捕获的非交互环境中自动关闭
    进度条，减少日志噪声。

    除非用户显式传入 ``disable``，否则函数会调用 ``_auto_disable`` 自动决定
    是否显示。同时会设置 ``dynamic_ncols=True``、``leave=False`` 和
    ``mininterval=0.5`` 作为默认值。

    Parameters
    ----------
    *args
        透传给 ``tqdm.auto.tqdm`` 的位置参数。
    **kwargs
        透传给 ``tqdm.auto.tqdm`` 的关键字参数。

    Returns
    -------
    Any
        ``tqdm`` 返回的进度条对象或迭代器包装对象。

    Examples
    --------
    在内部循环中使用统一进度条::

        for i in progress(range(100), desc="load"):
            ...
    """

    if "disable" not in kwargs or kwargs["disable"] is None:
        kwargs["disable"] = _auto_disable()
    kwargs.setdefault("dynamic_ncols", True)
    kwargs.setdefault("leave", False)
    kwargs.setdefault("mininterval", 0.5)
    return _tqdm(*args, **kwargs)
