from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from config import settings


def video_hash(video_path: Path) -> str:
    h = hashlib.sha256()
    with video_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


class MotionService:
    """从动作视频或摄像头帧序列提取 SMPL-X 参数。"""

    def __init__(self) -> None:
        self.cache_dir = settings.motion_cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _lhm_root(self) -> Path:
        root = Path(settings.lhm_root).resolve()
        if not root.exists():
            raise RuntimeError("LHM_ROOT 未配置或不存在")
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        return root

    def extract_from_video(self, video_path: Path, max_frames: int = 1000) -> Path:
        """返回 smplx_params 目录路径。"""
        if settings.mock_mode:
            out = self.cache_dir / f"mock_{video_hash(video_path)}"
            smplx_dir = out / "smplx_params"
            smplx_dir.mkdir(parents=True, exist_ok=True)
            for i in range(min(30, max_frames)):
                frame = {
                    "betas": [0.0] * 10,
                    "root_pose": [0.0, 0.0, 0.0],
                    "body_pose": [[0.0, 0.0, 0.0]] * 21,
                    "jaw_pose": [0.0, 0.0, 0.0],
                    "leye_pose": [0.0, 0.0, 0.0],
                    "reye_pose": [0.0, 0.0, 0.0],
                    "lhand_pose": [[0.0, 0.0, 0.0]] * 15,
                    "rhand_pose": [[0.0, 0.0, 0.0]] * 15,
                    "trans": [0.0, 0.0, 0.0],
                    "focal": [1000.0, 1000.0],
                    "princpt": [256.0, 256.0],
                    "img_size_wh": [512, 512],
                    "pad_ratio": 0.2,
                }
                (smplx_dir / f"{i + 1:05d}.json").write_text(
                    json.dumps(frame), encoding="utf-8"
                )
            return smplx_dir

        cache_key = video_hash(video_path)
        cached = self.cache_dir / cache_key
        smplx_dir = cached / "smplx_params"
        if smplx_dir.exists() and any(smplx_dir.glob("*.json")):
            return smplx_dir

        root = self._lhm_root()
        human_model = root / "pretrained_models" / "human_model_files"

        # 优先使用 LHM++ / LHM 的 Video2MotionPipeline
        try:
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError("动作提取需要 CUDA GPU")

            from engine.pose_estimation.video2motion import Video2MotionPipeline

            device = torch.device("cuda:0")
            pipeline = Video2MotionPipeline(
                str(human_model),
                fitting_steps=[30, 50],
                device=device,
                kp_mode="vitpose",
                visualize=False,
                pad_ratio=0.2,
                fov=60,
            )
            cached.mkdir(parents=True, exist_ok=True)
            smplx_dir = pipeline(str(video_path), str(cached), is_file_only=True)
            del pipeline
            torch.cuda.empty_cache()
            return Path(smplx_dir)
        except ImportError:
            pass

        # 回退：检查 LHM++ 预置 motion_video
        motion_name = video_path.stem
        preset = root / "motion_video" / motion_name / "smplx_params"
        if preset.exists():
            return preset

        raise RuntimeError(
            "无法从视频提取动作。请确保 LHM-plusplus 已安装 engine/pose_estimation/video2motion，"
            "或上传已预处理的 motion 数据。"
        )

    def save_frames_to_video(self, frames: list[np.ndarray], output_path: Path, fps: int = 30) -> Path:
        if not frames:
            raise ValueError("没有可用帧")
        h, w = frames[0].shape[:2]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (w, h),
        )
        for frame in frames:
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) if frame.shape[2] == 3 else frame
            writer.write(bgr)
        writer.release()
        return output_path

    def extract_from_frame_buffer(
        self,
        frames: list[np.ndarray],
        fps: int = 30,
        max_frames: int = 1000,
    ) -> Path:
        if len(frames) > max_frames:
            frames = frames[:max_frames]
        tmp_video = self.cache_dir / f"stream_{hash(tuple(f.tobytes()[:100] for f in frames[:3])) & 0xFFFFFFFF:x}.mp4"
        self.save_frames_to_video(frames, tmp_video, fps=fps)
        return self.extract_from_video(tmp_video, max_frames=max_frames)


motion_service = MotionService()
