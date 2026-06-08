from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from config import settings
from services.blender_fbx_export import blender_available
from services.job_manager import JobManager, JobStatus
from services.lhmpp_service import _cap_ref_view, lhmpp_service
from services.motion_service import motion_service

router = APIRouter(prefix="/api/v1", tags=["avatars"])

job_manager = JobManager(settings.jobs_dir)
settings.avatars_dir.mkdir(parents=True, exist_ok=True)


def _avatar_dir(avatar_id: str) -> Path:
    return settings.avatars_dir / avatar_id


def _load_meta(avatar_id: str) -> dict[str, Any]:
    meta_path = _avatar_dir(avatar_id) / "meta.json"
    if not meta_path.exists():
        raise HTTPException(404, "Avatar 不存在")
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _save_meta(avatar_id: str, meta: dict[str, Any]) -> None:
    meta_path = _avatar_dir(avatar_id) / "meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


async def _save_uploads(files: list[UploadFile], dest: Path) -> list[Path]:
    dest.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for i, f in enumerate(files):
        suffix = Path(f.filename or "image.png").suffix or ".png"
        out = dest / f"input_{i:03d}{suffix}"
        content = await f.read()
        out.write_bytes(content)
        paths.append(out)
    return paths


@router.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "lhmpp_available": lhmpp_service.available,
        "mock_mode": settings.mock_mode,
        "infer_low_memory": settings.infer_low_memory,
        "infer_max_image_size": settings.effective_infer_max_image_size,
        "infer_ref_view_max": settings.effective_infer_ref_view_max,
        "infer_anim_batch_size": settings.effective_infer_anim_batch_size,
        "infer_dense_sample_pts": settings.effective_infer_dense_sample_pts,
        "blender_available": blender_available(),
    }


@router.post("/avatars")
async def create_avatar(
    images: list[UploadFile] = File(...),
    ref_view: int = Form(8),
    export_skinned_mesh: bool = Form(False),
) -> dict[str, Any]:
    if not images:
        raise HTTPException(400, "请至少上传一张人物图片")
    if len(images) > settings.max_ref_images:
        raise HTTPException(400, f"最多上传 {settings.max_ref_images} 张图片")

    ref_view = _cap_ref_view(ref_view)

    avatar_id = str(uuid.uuid4())
    avatar_path = _avatar_dir(avatar_id)
    input_dir = avatar_path / "inputs"
    output_dir = avatar_path / "output"

    image_paths = await _save_uploads(images, input_dir)
    meta = {
        "id": avatar_id,
        "status": "pending",
        "ref_view": ref_view,
        "image_count": len(image_paths),
        "export_skinned_mesh": export_skinned_mesh,
    }
    _save_meta(avatar_id, meta)

    job = job_manager.create("reconstruct")

    def _run(_job) -> None:
        job_manager.update(job.id, progress=10, message="正在重建 3D 模型...")
        result = lhmpp_service.reconstruct_avatar(
            image_paths,
            output_dir,
            ref_view=ref_view,
            export_skinned_mesh=export_skinned_mesh,
        )
        if export_skinned_mesh:
            job_manager.update(job.id, progress=70, message="正在导出 SMPL-X 蒙皮 FBX...")
        job_manager.update(job.id, progress=90, result=result)
        meta.update({"status": "ready", "job_id": job.id, **result})
        _save_meta(avatar_id, meta)

    job_manager.run_async(job.id, _run)

    return {"avatar_id": avatar_id, "job_id": job.id, "status": "pending"}


@router.get("/avatars/{avatar_id}")
def get_avatar(avatar_id: str) -> dict[str, Any]:
    return _load_meta(avatar_id)


@router.get("/avatars/{avatar_id}/model")
def download_model(avatar_id: str):
    meta = _load_meta(avatar_id)
    ply = meta.get("ply_path")
    if not ply or not Path(ply).exists():
        raise HTTPException(404, "3D 模型尚未生成")
    return FileResponse(ply, filename="avatar.ply", media_type="application/octet-stream")


@router.get("/avatars/{avatar_id}/mesh")
def download_skinned_mesh(avatar_id: str, format: str = "fbx"):
    meta = _load_meta(avatar_id)
    fmt = format.lower()
    if fmt == "fbx":
        path = meta.get("mesh_fbx_path")
        filename = "avatar_skinned.fbx"
        media = "application/octet-stream"
    elif fmt == "obj":
        path = meta.get("mesh_obj_path")
        filename = "avatar_skinned.obj"
        media = "text/plain"
    else:
        raise HTTPException(400, "format 仅支持 fbx 或 obj")
    if not path or not Path(path).exists():
        if fmt == "fbx":
            raise HTTPException(
                404,
                "蒙皮 FBX 尚未生成。请开启「导出蒙皮网格」并确认服务器已安装 Blender（BLENDER_EXECUTABLE）",
            )
        raise HTTPException(404, "蒙皮网格尚未生成，请开启「导出蒙皮网格」开关")
    return FileResponse(path, filename=filename, media_type=media)


@router.get("/avatars/{avatar_id}/skeleton")
def download_skeleton(avatar_id: str):
    meta = _load_meta(avatar_id)
    path = meta.get("skeleton_json_path")
    if not path or not Path(path).exists():
        raise HTTPException(404, "骨骼数据尚未生成")
    return FileResponse(path, filename="avatar_skeleton.json", media_type="application/json")


@router.get("/avatars/{avatar_id}/preview")
def get_preview(avatar_id: str):
    meta = _load_meta(avatar_id)
    preview = meta.get("preview_path")
    if not preview or not Path(preview).exists():
        raise HTTPException(404, "预览图不存在")
    return FileResponse(preview)


@router.post("/avatars/{avatar_id}/animate")
async def animate_avatar(
    avatar_id: str,
    motion_video: UploadFile = File(...),
    motion_frames: int = Form(settings.default_motion_frames),
    render_backend: str = Form("neural"),
) -> dict[str, Any]:
    meta = _load_meta(avatar_id)
    if meta.get("status") != "ready":
        raise HTTPException(400, "Avatar 尚未就绪")

    avatar_path = _avatar_dir(avatar_id)
    input_dir = avatar_path / "inputs"
    image_paths = sorted(input_dir.glob("input_*"))
    if not image_paths:
        raise HTTPException(400, "找不到参考图片")

    motion_path = avatar_path / "motion" / (motion_video.filename or "motion.mp4")
    motion_path.parent.mkdir(parents=True, exist_ok=True)
    motion_path.write_bytes(await motion_video.read())

    motion_frames = min(max(30, motion_frames), settings.max_motion_frames)
    job = job_manager.create("animate")

    def _run(_job) -> None:
        job_manager.update(job.id, progress=5, message="提取动作参数...")
        smplx_dir = motion_service.extract_from_video(motion_path, max_frames=motion_frames)

        job_manager.update(job.id, progress=40, message="渲染动画...")
        out_dir = avatar_path / "animations" / job.id
        result = lhmpp_service.animate_avatar(
            image_paths,
            smplx_dir,
            out_dir,
            ref_view=meta.get("ref_view", 8),
            motion_frames=motion_frames,
            render_backend=render_backend,
        )
        job_manager.update(job.id, progress=95, result=result)
        meta["last_animation"] = {"job_id": job.id, **result}
        _save_meta(avatar_id, meta)

    job_manager.run_async(job.id, _run)
    return {"avatar_id": avatar_id, "job_id": job.id, "status": "pending"}


@router.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    job = job_manager.get(job_id)
    if job is None:
        raise HTTPException(404, "任务不存在")
    return job.to_dict()


@router.get("/jobs/{job_id}/video")
def get_job_video(job_id: str):
    job = job_manager.get(job_id)
    if job is None:
        raise HTTPException(404, "任务不存在")
    video = job.result.get("video_path")
    if not video or not Path(video).exists():
        raise HTTPException(404, "视频尚未生成")
    return FileResponse(video, filename="animation.mp4", media_type="video/mp4")
