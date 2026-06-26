"""SuGaR 实验管线：PLY → 抽网格 + 贴图 OBJ → Blender 静态 FBX。

与 SMPL+Blender 蒙皮 FBX 并行；SuGaR FBX 为静态网格（无 SMPL 骨骼），几何通常更接近 3DGS。
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from config import settings

logger = logging.getLogger(__name__)


def resolve_sugar_root() -> Path | None:
    root = (settings.sugar_root or os.environ.get("SUGAR_ROOT", "")).strip()
    if not root:
        return None
    p = Path(root).resolve()
    return p if p.is_dir() else None


def sugar_available() -> bool:
    root = resolve_sugar_root()
    if not root:
        return False
    return _find_train_script(root) is not None


def _find_train_script(sugar_root: Path) -> Path | None:
    for candidate in (
        sugar_root / "train_full_pipeline.py",
        sugar_root / "SuGaR" / "train_full_pipeline.py",
    ):
        if candidate.is_file():
            return candidate
    return None


def _sugar_python() -> str:
    exe = (settings.sugar_python or os.environ.get("SUGAR_PYTHON", "")).strip()
    if exe:
        return exe
    import sys

    return sys.executable


def prepare_sugar_work(ply_path: Path, output_dir: Path, *, ref_image_dir: Path | None = None) -> tuple[Path, dict]:
    """准备 SuGaR 工作目录；不删除已有 sugar_work 时由调用方控制。"""
    output_dir = output_dir.resolve()
    work_dir = output_dir / "sugar_work"
    work_dir.mkdir(parents=True, exist_ok=True)

    gs_dir = work_dir / "vanilla_3dgs"
    iter_dir = gs_dir / "point_cloud" / "iteration_7000"
    iter_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ply_path, iter_dir / "point_cloud.ply")

    images_out = work_dir / "source_images"
    images_out.mkdir(exist_ok=True)
    copied: list[str] = []
    search_dirs = []
    if ref_image_dir and ref_image_dir.is_dir():
        search_dirs.append(ref_image_dir)
    for sub in ("ref_images", "../inputs"):
        p = (output_dir / sub).resolve()
        if p.is_dir() and p not in search_dirs:
            search_dirs.append(p)

    for folder in search_dirs:
        for src in sorted(folder.iterdir()):
            if src.suffix.lower() not in (".png", ".jpg", ".jpeg", ".webp"):
                continue
            dst = images_out / f"ref_{len(copied):03d}{src.suffix.lower()}"
            if not dst.exists():
                shutil.copy2(src, dst)
            copied.append(dst.name)

    meta = {
        "ply_source": str(ply_path.resolve()),
        "vanilla_gs_dir": str(gs_dir),
        "ref_image_count": len(copied),
        "work_dir": str(work_dir),
    }
    (work_dir / "hrm_sugar_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return work_dir, meta


def _run_sugar_train(sugar_root: Path, work_dir: Path, meta: dict) -> Path:
    train = _find_train_script(sugar_root)
    if train is None:
        raise RuntimeError(f"未在 {sugar_root} 找到 train_full_pipeline.py")

    images_dir = work_dir / "source_images"
    sugar_out = work_dir / "sugar_output"
    sugar_out.mkdir(parents=True, exist_ok=True)
    scene = str(images_dir if meta.get("ref_image_count") else work_dir)

    cmd = [
        _sugar_python(),
        str(train),
        "-s",
        scene,
        "-c",
        str(sugar_out),
        "--gs_output_dir",
        meta["vanilla_gs_dir"],
        "--export_obj",
        "true",
        "--export_ply",
        "true",
    ]
    logger.info("SuGaR 全链路: %s", " ".join(cmd))
    proc = subprocess.run(
        cmd,
        cwd=str(sugar_root),
        capture_output=True,
        text=True,
        timeout=settings.sugar_timeout_sec,
        check=False,
    )
    tail = "\n".join(x for x in ((proc.stdout or ""), (proc.stderr or "")) if x).strip()[-4000:]
    if proc.returncode != 0:
        raise RuntimeError(f"SuGaR 退出码 {proc.returncode}\n{tail}")
    logger.info("SuGaR 完成 (code=0)")
    return sugar_out


def _find_sugar_obj(sugar_out: Path) -> Path | None:
    if not sugar_out.is_dir():
        return None
    preferred: list[Path] = []
    others: list[Path] = []
    for obj in sugar_out.rglob("*.obj"):
        name = obj.name.lower()
        if "textured" in name or "mesh" in name:
            preferred.append(obj)
        else:
            others.append(obj)
    pool = preferred or others
    if not pool:
        return None
    return max(pool, key=lambda p: p.stat().st_mtime)


def _find_sugar_ply(sugar_out: Path) -> Path | None:
    if not sugar_out.is_dir():
        return None
    candidates = [p for p in sugar_out.rglob("*.ply") if "point_cloud" not in p.as_posix()]
    if not candidates:
        candidates = list(sugar_out.rglob("*.ply"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def export_sugar_fbx_from_gaussian(
    ply_path: Path,
    output_dir: Path,
    *,
    ref_image_dir: Path | None = None,
) -> dict[str, Any]:
    """SuGaR 抽网格 → 静态 FBX（需 SUGAR_ROOT + Blender）。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    obj_path = output_dir / "avatar.obj"
    fbx_path = output_dir / "avatar.fbx"
    ply_out = output_dir / "avatar_refined.ply"

    if settings.mock_mode:
        obj_path.write_text("# mock sugar obj\n", encoding="utf-8")
        fbx_path.write_text("", encoding="utf-8")
        return {
            "mesh_fbx_path": str(fbx_path),
            "mesh_obj_path": str(obj_path),
            "mesh_sugar_fbx_path": str(fbx_path),
            "mesh_sugar_obj_path": str(obj_path),
            "mesh_sugar_ply_path": None,
            "sugar_export_ok": True,
            "mesh_export_backend": "sugar",
            "mock": True,
        }

    sugar_root = resolve_sugar_root()
    if not sugar_root:
        logger.warning("未配置 SUGAR_ROOT，跳过 SuGaR FBX 导出")
        return {
            "mesh_fbx_path": None,
            "mesh_obj_path": None,
            "mesh_sugar_fbx_path": None,
            "mesh_sugar_obj_path": None,
            "mesh_sugar_ply_path": None,
            "sugar_export_ok": False,
            "mesh_export_backend": "sugar",
        }

    work_dir, meta = prepare_sugar_work(ply_path, output_dir, ref_image_dir=ref_image_dir)
    try:
        sugar_out = _run_sugar_train(sugar_root, work_dir, meta)
    except Exception as exc:
        logger.warning("SuGaR 管线失败: %s", exc, exc_info=True)
        return {
            "mesh_fbx_path": None,
            "mesh_obj_path": None,
            "mesh_sugar_fbx_path": None,
            "mesh_sugar_obj_path": None,
            "mesh_sugar_ply_path": None,
            "sugar_export_ok": False,
            "mesh_export_backend": "sugar",
            "sugar_error": str(exc),
        }

    found_obj = _find_sugar_obj(sugar_out)
    if not found_obj:
        logger.warning("SuGaR 输出中未找到 OBJ: %s", sugar_out)
        return {
            "mesh_fbx_path": None,
            "mesh_obj_path": None,
            "mesh_sugar_fbx_path": None,
            "mesh_sugar_obj_path": None,
            "mesh_sugar_ply_path": None,
            "sugar_export_ok": False,
            "mesh_export_backend": "sugar",
            "sugar_error": "no_obj_in_sugar_output",
        }

    shutil.copy2(found_obj, obj_path)
    mtl = found_obj.with_suffix(".mtl")
    if mtl.is_file():
        shutil.copy2(mtl, obj_path.with_suffix(".mtl"))
    for tex in found_obj.parent.glob("*"):
        if tex.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"):
            dst = output_dir / tex.name
            if not dst.exists():
                shutil.copy2(tex, dst)

    found_ply = _find_sugar_ply(sugar_out)
    mesh_sugar_ply: str | None = None
    if found_ply:
        shutil.copy2(found_ply, ply_out)
        mesh_sugar_ply = str(ply_out)

    from services.blender_static_fbx_export import export_static_fbx

    fbx_ok = export_static_fbx(obj_path, fbx_path, timeout_sec=settings.sugar_blender_timeout_sec)

    # 与 Web / 下载 API 兼容的主字段
    mesh_fbx = str(fbx_path) if fbx_ok else None
    mesh_obj = str(obj_path)
    texture_path: str | None = None
    for tex in output_dir.glob("*"):
        if tex.suffix.lower() in (".png", ".jpg", ".jpeg") and "diffuse" in tex.name.lower():
            texture_path = str(tex)
            break
    if texture_path is None:
        for tex in output_dir.glob("*.png"):
            texture_path = str(tex)
            break

    return {
        "mesh_fbx_path": mesh_fbx,
        "mesh_obj_path": mesh_obj,
        "mesh_texture_path": texture_path,
        "mesh_sugar_fbx_path": mesh_fbx,
        "mesh_sugar_obj_path": mesh_obj,
        "mesh_sugar_ply_path": mesh_sugar_ply,
        "sugar_export_ok": bool(fbx_ok),
        "mesh_export_backend": "sugar",
        "sugar_work_dir": str(work_dir),
    }
