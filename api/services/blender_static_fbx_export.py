"""Blender headless：SuGaR 等导出的静态 OBJ → FBX。"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from services.blender_fbx_export import (
    _blender_subprocess_env,
    _log_blender_output,
    resolve_blender_executable,
)

logger = logging.getLogger(__name__)

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "blender_export_static_fbx.py"


def export_static_fbx(obj_path: Path, fbx_path: Path, *, timeout_sec: int = 600) -> bool:
    blender = resolve_blender_executable()
    if not blender:
        logger.warning("未找到 Blender，跳过 SuGaR 静态 FBX 转换")
        return False
    if not obj_path.is_file():
        logger.error("缺少 OBJ: %s", obj_path)
        return False
    if not _SCRIPT_PATH.is_file():
        logger.error("脚本不存在: %s", _SCRIPT_PATH)
        return False

    fbx_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        blender,
        "--background",
        "--python-use-system-env",
        "--python",
        str(_SCRIPT_PATH),
        "--",
        str(obj_path.resolve()),
        str(fbx_path.resolve()),
    ]
    logger.info("Blender 静态 FBX: %s", " ".join(cmd))
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
            env=_blender_subprocess_env(),
        )
    except subprocess.TimeoutExpired:
        logger.error("Blender 静态 FBX 超时 (%ds)", timeout_sec)
        return False
    except OSError as exc:
        logger.error("无法启动 Blender: %s", exc)
        return False

    if proc.returncode != 0 or not fbx_path.is_file():
        _log_blender_output(proc, "静态 FBX 失败")
        return False
    logger.info("SuGaR 静态 FBX 完成: %s (%.1f KB)", fbx_path.name, fbx_path.stat().st_size / 1024)
    return True
