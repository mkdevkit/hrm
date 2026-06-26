"""通过 Blender headless 将 SMPL-X 蒙皮网格导出为 FBX。"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path

from config import settings

logger = logging.getLogger(__name__)

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "blender_export_fbx.py"


def _system_numpy_paths() -> list[str]:
    candidates = [
        "/usr/lib/python3/dist-packages",
        "/usr/local/lib/python3.10/dist-packages",
        "/usr/local/lib/python3.11/dist-packages",
        "/usr/local/lib/python3.12/dist-packages",
    ]
    return [p for p in candidates if Path(p).is_dir()]


def _blender_subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    extra = _system_numpy_paths()
    if extra:
        current = env.get("PYTHONPATH", "")
        merged = os.pathsep.join(extra + ([current] if current else []))
        env["PYTHONPATH"] = merged
    return env


def _log_blender_output(proc: subprocess.CompletedProcess[str], reason: str) -> None:
    out = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()
    combined = "\n".join(x for x in (out, err) if x)
    tail = combined[-4000:] if combined else "（无 stdout/stderr）"
    logger.error(
        "Blender FBX 导出%s (code=%s, target=%s):\n%s",
        reason,
        proc.returncode,
        proc.args[-3] if len(proc.args) >= 3 else "?",
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


def blender_fbx_ready() -> bool:
    """Blender 可用且 FBX 插件能 import numpy（apt 版 Blender 常缺此项）。"""
    blender = resolve_blender_executable()
    if not blender:
        return False
    try:
        proc = subprocess.run(
            [
                blender,
                "--background",
                "--python-use-system-env",
                "--python-expr",
                "import numpy; import bpy; print('ok')",
            ],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
            env=_blender_subprocess_env(),
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return proc.returncode == 0 and "ok" in (proc.stdout or "")


def export_skinned_fbx(
    obj_path: Path,
    skel_path: Path,
    weights_path: Path,
    fbx_path: Path,
    *,
    texture_path: Path | None = None,
    subdivision_levels: int = 0,
    subdivision_type: str = "simple",
    timeout_sec: int = 600,
) -> bool:
    """调用 Blender 将 OBJ + 骨骼 JSON + 权重 NPZ + 可选贴图 合并为蒙皮 FBX。"""
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

    try:
        from services.mesh_export_service import ensure_skeleton_rest_positions

        ensure_skeleton_rest_positions(obj_path, skel_path, weights_path)
    except Exception as exc:
        logger.warning("补全 skeleton.json 失败，将依赖 Blender 脚本回退: %s", exc)

    fbx_path.parent.mkdir(parents=True, exist_ok=True)
    tex_arg = str(texture_path.resolve()) if texture_path and texture_path.is_file() else "-"
    subdiv = str(max(0, subdivision_levels))
    subdiv_type = (subdivision_type or "simple").strip().lower()
    if subdiv_type not in ("simple", "catmull", "catmull_clark"):
        subdiv_type = "simple"
    cmd = [
        blender,
        "--background",
        "--python-use-system-env",
        "--python",
        str(_SCRIPT_PATH),
        "--",
        str(obj_path.resolve()),
        str(skel_path.resolve()),
        str(weights_path.resolve()),
        str(fbx_path.resolve()),
        tex_arg,
        subdiv,
        subdiv_type,
    ]
    logger.info("Blender FBX 导出: %s", " ".join(cmd))

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
