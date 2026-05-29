from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

from config import settings


class LHMPPService:
    """封装 LHM++ 推理：多图重建 3D Gaussian + 动作驱动渲染。"""

    def __init__(self) -> None:
        self._initialized = False
        self._lhmpp = None
        self._pose_estimator = None
        self._dataset_pipeline = None
        self._cfg = None

    @property
    def available(self) -> bool:
        if settings.mock_mode:
            return True
        return bool(settings.lhm_root) and Path(settings.lhm_root).exists()

    def _ensure_lhm_path(self) -> Path:
        root = Path(settings.lhm_root).resolve()
        if not root.exists():
            raise RuntimeError(
                f"LHM_ROOT 不存在: {root}。请克隆 https://github.com/aigc3d/LHM-plusplus 并设置环境变量。"
            )
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        os.chdir(root)
        return root

    def initialize(self) -> None:
        if self._initialized or settings.mock_mode:
            self._initialized = True
            return

        root = self._ensure_lhm_path()
        os.environ.update(
            {
                "APP_ENABLED": "1",
                "APP_MODEL_NAME": settings.model_name,
                "APP_TYPE": "infer.human_lrm_a4o",
                "NUMBA_THREADING_LAYER": "omp",
            }
        )

        import torch

        torch._dynamo.config.disable = True

        from accelerate import Accelerator
        from app import prior_model_check
        from core.datasets.data_utils import SrcImagePipeline
        from core.utils.model_card import MODEL_CONFIG
        from core.utils.model_download_utils import AutoModelQuery
        from engine.pose_estimation.pose_estimator import PoseEstimator
        from scripts.download_motion_video import motion_video_check
        from scripts.inference.app_inference import (
            build_app_model,
            parse_app_configs,
        )

        prior_model_check(save_dir=str(root / "pretrained_models"))
        motion_video_check(save_dir=str(root))

        model_config = MODEL_CONFIG[settings.model_name]
        if settings.model_path:
            model_path = settings.model_path
        else:
            auto_query = AutoModelQuery(save_dir=str(root / "pretrained_models"))
            model_path = auto_query.query(settings.model_name)

        model_cards = {
            settings.model_name: {
                "model_path": model_path,
                "model_config": model_config,
            }
        }

        processing_list = [
            dict(
                name="PadRatioWithScale",
                target_ratio=5 / 3,
                tgt_max_size_list=[840],
                val=True,
            ),
        ]
        self._dataset_pipeline = SrcImagePipeline(*processing_list)
        Accelerator()
        self._cfg, _ = parse_app_configs(model_cards)

        self._lhmpp = build_app_model(self._cfg)
        self._lhmpp.to("cuda")

        if self._cfg.get("use_smplx_shape_estimator", True):
            self._pose_estimator = PoseEstimator(
                str(root / "pretrained_models" / "human_model_files"),
                device="cpu",
            )
            self._pose_estimator.device = "cuda"
        else:
            self._pose_estimator = None

        self._initialized = True

    def reconstruct_avatar(
        self,
        image_paths: list[Path],
        output_dir: Path,
        ref_view: int = 8,
        export_skinned_mesh: bool = False,
    ) -> dict[str, Any]:
        output_dir.mkdir(parents=True, exist_ok=True)

        if settings.mock_mode:
            ply_path = output_dir / "avatar.ply"
            ply_path.write_text("# mock gaussian splat ply\n", encoding="utf-8")
            preview = output_dir / "preview.png"
            if image_paths:
                from shutil import copyfile

                copyfile(image_paths[0], preview)
            result: dict[str, Any] = {
                "ply_path": str(ply_path),
                "preview_path": str(preview) if preview.exists() else None,
                "ref_count": len(image_paths),
                "export_skinned_mesh": export_skinned_mesh,
                "mock": True,
            }
            if export_skinned_mesh:
                from services.mesh_export_service import export_skinned_mesh_from_gaussian

                mesh_result = export_skinned_mesh_from_gaussian(ply_path, output_dir)
                result.update(mesh_result)
            return result

        self.initialize()
        root = self._ensure_lhm_path()

        import numpy as np
        import torch
        from PIL import Image

        from core.utils.app_utils import obtain_ref_imgs

        imgs_pil = [Image.open(p).convert("RGBA") for p in image_paths]
        imgs_arr = obtain_ref_imgs([(np.array(img),) for img in imgs_pil], ref_view)

        betas_list: list[float] | None = None
        if self._pose_estimator is not None:
            from scripts.inference.utils import easy_memory_manager

            with torch.no_grad():
                with easy_memory_manager(self._pose_estimator, device="cuda"):
                    shape_pose = self._pose_estimator(imgs_arr[0])
            if not shape_pose.is_full_body:
                raise ValueError(f"输入图片不符合要求: {shape_pose.msg}")
            betas_list = shape_pose.beta.tolist() if hasattr(shape_pose.beta, "tolist") else list(shape_pose.beta)
            (output_dir / "betas.json").write_text(
                json.dumps(betas_list), encoding="utf-8"
            )

        image_glob_dir = output_dir / "ref_images"
        image_glob_dir.mkdir(parents=True, exist_ok=True)
        for i, img in enumerate(imgs_arr):
            Image.fromarray(img.astype(np.uint8)).save(image_glob_dir / f"ref_{i:03d}.png")

        ply_out = output_dir / "avatar.ply"
        model_path = settings.model_path
        if not model_path:
            from core.utils.model_download_utils import AutoModelQuery

            auto_query = AutoModelQuery(save_dir=str(root / "pretrained_models"))
            model_path = auto_query.query(settings.model_name)

        import subprocess

        cmd = [
            sys.executable,
            str(root / "scripts" / "inference" / "to_gs_ply.py"),
            "--model_path",
            model_path,
            "--image_glob",
            str(image_glob_dir / "ref_*.png"),
            "--output",
            str(ply_out),
        ]
        proc = subprocess.run(cmd, cwd=str(root), capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"3D 重建失败: {proc.stderr or proc.stdout}")

        preview_path = image_glob_dir / "ref_000.png"
        result = {
            "ply_path": str(ply_out),
            "preview_path": str(preview_path),
            "ref_count": len(imgs_arr),
            "export_skinned_mesh": export_skinned_mesh,
            "mock": False,
        }

        if export_skinned_mesh:
            from services.mesh_export_service import export_skinned_mesh_from_gaussian

            mesh_result = export_skinned_mesh_from_gaussian(
                ply_out,
                output_dir,
                betas=betas_list,
                lhm_root=str(root),
            )
            result.update(mesh_result)

        return result

    def animate_avatar(
        self,
        image_paths: list[Path],
        motion_smplx_dir: Path,
        output_dir: Path,
        *,
        ref_view: int = 8,
        motion_frames: int = 120,
        render_backend: str = "neural",
    ) -> dict[str, Any]:
        output_dir.mkdir(parents=True, exist_ok=True)
        video_path = output_dir / "animation.mp4"

        if settings.mock_mode:
            video_path.write_bytes(b"")
            return {"video_path": str(video_path), "frame_count": motion_frames, "mock": True}

        self.initialize()

        import imageio.v3 as iio
        import numpy as np
        import torch

        from core.utils.app_utils import get_motion_information, obtain_ref_imgs
        from PIL import Image
        from scripts.inference.app_inference import inference_results
        from scripts.inference.utils import easy_memory_manager

        imgs_pil = [Image.open(p).convert("RGBA") for p in image_paths]
        imgs_arr = obtain_ref_imgs([(np.array(img),) for img in imgs_pil], ref_view)

        device = "cuda"
        dtype = torch.float32

        if self._pose_estimator is not None:
            with torch.no_grad():
                with easy_memory_manager(self._pose_estimator, device="cuda"):
                    shape_pose = self._pose_estimator(imgs_arr[0])
            if not shape_pose.is_full_body:
                raise ValueError(f"输入图片不符合要求: {shape_pose.msg}")

        motion_name, motion_seqs = get_motion_information(
            str(motion_smplx_dir), self._cfg, motion_size=motion_frames
        )
        video_size = len(motion_seqs["motion_seqs"])

        img_np = np.stack(imgs_arr) / 255.0
        ref_imgs_tensor = torch.from_numpy(img_np).permute(0, 3, 1, 2).float().to(device)
        smplx_params = motion_seqs["smplx_params"]
        if self._pose_estimator is not None:
            smplx_params["betas"] = torch.tensor(
                shape_pose.beta, dtype=dtype, device=device
            ).unsqueeze(0)

        rgbs = inference_results(
            self._lhmpp,
            ref_imgs_tensor,
            smplx_params,
            motion_seqs,
            video_size=video_size,
            device=device,
            infer_output_renderer=render_backend,
        )

        iio.imwrite(
            str(video_path),
            rgbs,
            fps=settings.render_fps,
            codec="libx264",
            pixelformat="yuv420p",
            bitrate="10M",
            macro_block_size=16,
        )

        return {
            "video_path": str(video_path),
            "frame_count": video_size,
            "motion_name": motion_name,
            "mock": False,
        }


lhmpp_service = LHMPPService()
