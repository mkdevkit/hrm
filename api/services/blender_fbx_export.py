"""通过 Blender headless 将 SMPL-X 蒙皮网格导出为 FBX。"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

from config import settings

logger = logging.getLogger(__name__)

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "blender_export_fbx.py"


def _log_blender_output(proc: subprocess.CompletedProcess[str], reason: str) -> None:
    out = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()
    combined = "\n".join(x for x in (out, err) if x)
    tail = combined[-4000:] if combined else "（无 stdout/stderr）"
    logger.error(
        "Blender FBX 导出%s (code=%s, target=%s):\n%s",
        reason,
        proc.returncode,
        proc.args[-1] if proc.args else "?",
        tail,
    )


def resolve_blender_executable() -> str | None:
    if settings.blender_executable:
        p = Path(settings.blender_executable)
        if p.is_file():
            return str(p.resolve())
    return shutil.which("blender")


def blender_available() -> bool:
    return resolve_blender_executable() is not None


def export_skinned_fbx(
    obj_path: Path,
    skel_path: Path,
    weights_path: Path,
    fbx_path: Path,
    *,
    timeout_sec: int = 180,
) -> bool:
    """调用 Blender 将 OBJ + 骨骼 JSON + 权重 NPZ 合并为蒙皮 FBX。"""
    blender = resolve_blender_executable()
    if not blender:
        logger.warning(
            "未找到 Blender 可执行文件，跳过 FBX 导出。"
            " 请在 api/.env 设置 BLENDER_EXECUTABLE=/usr/bin/blender"
        )
        return False

    if not _SCRIPT_PATH.is_file():
        logger.error("Blender 脚本不存在: %s", _SCRIPT_PATH)
        return False

    for path in (obj_path, skel_path, weights_path):
        if not path.is_file():
            logger.error("FBX 导出缺少输入文件: %s", path)
            return False

    fbx_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        blender,
        "--background",
        "--python",
        str(_SCRIPT_PATH),
        "--",
        str(obj_path.resolve()),
        str(skel_path.resolve()),
        str(weights_path.resolve()),
        str(fbx_path.resolve()),
    ]
    logger.info("Blender FBX 导出: %s", " ".join(cmd))

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
    except subprocess.TimeoutExpired:
        logger.error("Blender FBX 导出超时 (%ds)", timeout_sec)
        return False
    except OSError as exc:
        logger.error("无法启动 Blender: %s", exc)
        return False

    if proc.returncode != 0:
        _log_blender_output(proc, "失败")
        return False

    if not fbx_path.is_file():
        _log_blender_output(proc, "未生成 FBX")
        return False

    logger.info("FBX 导出完成: %s (%.1f KB)", fbx_path.name, fbx_path.stat().st_size / 1024)
    return True
