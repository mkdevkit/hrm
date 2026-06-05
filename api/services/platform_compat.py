"""Windows / 跨平台兼容：在导入 LHM++ 前注入缺失的标准库模块。"""
from __future__ import annotations

import sys
import types


def apply_lhm_import_compat() -> None:
    """LHM++ 依赖 megfile，其在 Windows 上会 import fcntl（仅 Unix）。"""
    if sys.platform == "win32" and "fcntl" not in sys.modules:
        fcntl = types.ModuleType("fcntl")
        fcntl.LOCK_EX = 2
        fcntl.LOCK_UN = 8
        fcntl.LOCK_SH = 1
        fcntl.LOCK_NB = 4

        def _noop_flock(_fd: int, _op: int) -> None:
            return None

        fcntl.flock = _noop_flock
        sys.modules["fcntl"] = fcntl
