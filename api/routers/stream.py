from __future__ import annotations

import base64
import json
from typing import Any

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from config import settings
from routers.avatars import _avatar_dir, _load_meta, _save_meta, job_manager
from services.lhmpp_service import lhmpp_service
from services.motion_service import motion_service

router = APIRouter(prefix="/api/v1", tags=["stream"])


@router.websocket("/avatars/{avatar_id}/motion-stream")
async def motion_stream(websocket: WebSocket, avatar_id: str) -> None:
    """WebSocket 实时动作捕获：客户端发送 base64 JPEG 帧，服务端缓冲后批量推理。"""
    await websocket.accept()

    try:
        meta = _load_meta(avatar_id)
    except Exception:
        await websocket.send_json({"type": "error", "message": "Avatar 不存在"})
        await websocket.close()
        return

    if meta.get("status") != "ready":
        await websocket.send_json({"type": "error", "message": "请先完成 3D 重建"})
        await websocket.close()
        return

    avatar_path = _avatar_dir(avatar_id)
    input_dir = avatar_path / "inputs"
    image_paths = sorted(input_dir.glob("input_*"))

    frame_buffer: list[np.ndarray] = []
    min_frames = 30
    fps = settings.render_fps

    await websocket.send_json({"type": "ready", "message": "开始发送视频帧 (base64 JPEG)"})

    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)
            msg_type = msg.get("type")

            if msg_type == "frame":
                data = msg.get("data", "")
                if "," in data:
                    data = data.split(",", 1)[1]
                img_bytes = base64.b64decode(data)
                arr = np.frombuffer(img_bytes, dtype=np.uint8)
                import cv2

                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame is None:
                    continue
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_buffer.append(frame_rgb)
                await websocket.send_json(
                    {"type": "buffer", "count": len(frame_buffer), "required": min_frames}
                )

            elif msg_type == "flush":
                if len(frame_buffer) < min_frames:
                    await websocket.send_json(
                        {
                            "type": "error",
                            "message": f"帧数不足，至少需要 {min_frames} 帧，当前 {len(frame_buffer)}",
                        }
                    )
                    continue

                job = job_manager.create("stream_animate")
                frames = frame_buffer.copy()
                frame_buffer.clear()

                await websocket.send_json(
                    {"type": "processing", "job_id": job.id, "frames": len(frames)}
                )

                def _run(_job) -> None:
                    smplx_dir = motion_service.extract_from_frame_buffer(
                        frames, fps=fps, max_frames=settings.max_motion_frames
                    )
                    out_dir = avatar_path / "animations" / job.id
                    result = lhmpp_service.animate_avatar(
                        image_paths,
                        smplx_dir,
                        out_dir,
                        ref_view=meta.get("ref_view", 8),
                        motion_frames=min(len(frames), settings.default_motion_frames),
                    )
                    job_manager.update(job.id, result=result)

                job_manager.run_async(job.id, _run)
                await websocket.send_json({"type": "job_started", "job_id": job.id})

            elif msg_type == "poll":
                job_id = msg.get("job_id")
                if not job_id:
                    continue
                job = job_manager.get(job_id)
                if job is None:
                    await websocket.send_json({"type": "error", "message": "任务不存在"})
                    continue
                payload: dict[str, Any] = {
                    "type": "job_status",
                    "job_id": job_id,
                    "status": job.status.value,
                    "progress": job.progress,
                    "message": job.message,
                }
                if job.status.value == "completed":
                    payload["video_url"] = f"/api/v1/jobs/{job_id}/video"
                if job.error:
                    payload["error"] = job.error
                await websocket.send_json(payload)

            elif msg_type == "close":
                break

    except WebSocketDisconnect:
        pass
