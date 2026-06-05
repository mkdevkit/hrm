from __future__ import annotations

import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional

from config import settings
from services.platform_compat import apply_lhm_import_compat


def _prior_model_check(save_dir: Path) -> None:
    """检查/下载 LHM++ 先验模型（与 app.prior_model_check 相同，避免 import Gradio app.py）。"""
    human_model_path = save_dir / "human_model_files"
    if human_model_path.exists():
        return
    if human_model_path.is_symlink():
        try:
            human_model_path.unlink()
            print("Removed broken symlink: human_model_files")
        except OSError as exc:
            print(f"Failed to remove broken symlink: {exc}")
    print("Prior models not found or invalid. Downloading...")
    from core.utils.model_download_utils import AutoModelQuery

    AutoModelQuery(save_dir=str(save_dir)).download_all_prior_models()
    print("Prior models ready.")


def _cap_ref_view(ref_view: int) -> int:
    return max(1, min(ref_view, settings.effective_infer_ref_view_max))


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
        apply_lhm_import_compat()
        # LHM++ 内部大量使用相对导入（如 engine/pose_estimation/model.py 的 `from blocks import ...`）
        for rel in ("", "engine", "engine/pose_estimation"):
            entry = str(root / rel) if rel else str(root)
            if entry not in sys.path:
                sys.path.insert(0, entry)
        return root

    @contextmanager
    def _lhm_runtime(self, root: Path) -> Iterator[Path]:
        """临时切换到 LHM++ 根目录，避免其相对路径/相对 import 失败；结束后恢复 cwd。"""
        prev_cwd = os.getcwd()
        os.chdir(root)
        try:
            yield root
        finally:
            os.chdir(prev_cwd)

    def initialize(self) -> None:
        if self._initialized or settings.mock_mode:
            self._initialized = True
            return

        root = self._ensure_lhm_path()
        with self._lhm_runtime(root):
            self._initialize_models(root)

        self._initialized = True

    def _initialize_models(self, root: Path) -> None:
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
        from core.datasets.data_utils import SrcImagePipeline
        from core.utils.model_card import MODEL_CONFIG
        from core.utils.model_download_utils import AutoModelQuery
        from scripts.download_motion_video import motion_video_check
        from scripts.inference.app_inference import (
            build_app_model,
            parse_app_configs,
        )

        _prior_model_check(root / "pretrained_models")
        motion_video_check(save_dir=str(root))

        model_config = MODEL_CONFIG[settings.model_name]
        if settings.model_path:
            model_path = settings.model_path
        else:
            auto_query = AutoModelQuery(save_dir=str(root / "pretrained_models"))
            model_path = auto_query.query(settings.model_name)

        load_path = model_path
        target_dense = settings.effective_infer_dense_sample_pts
        if target_dense > 0:
            from services.lhm_infer_utils import prepare_patched_model_path

            infer_work = root / "infer_work"
            load_path, _ = prepare_patched_model_path(
                model_path, root, infer_work, target_dense
            )

        model_cards = {
            settings.model_name: {
                "model_path": load_path,
                "model_config": model_config,
            }
        }

        processing_list = [
            dict(
                name="PadRatioWithScale",
                target_ratio=5 / 3,
                tgt_max_size_list=[settings.effective_infer_max_image_size],
                val=True,
            ),
        ]
        self._dataset_pipeline = SrcImagePipeline(*processing_list)
        Accelerator()
        self._cfg, _ = parse_app_configs(model_cards)

        self._lhmpp = build_app_model(self._cfg)
        self._lhmpp.to("cuda")

        from services.lhm_infer_utils import apply_infer_memory_overrides

        apply_infer_memory_overrides(self._lhmpp, root)

        if self._cfg.get("use_smplx_shape_estimator", True):
            from engine.pose_estimation.pose_estimator import PoseEstimator

            self._pose_estimator = PoseEstimator(
                str(root / "pretrained_models" / "human_model_files"),
                device="cpu",
            )
            self._pose_estimator.device = "cuda"
        else:
            self._pose_estimator = None

    def reconstruct_avatar(
        self,
        image_paths: list[Path],
        output_dir: Path,
        ref_view: int = 8,
        export_skinned_mesh: bool = False,
    ) -> dict[str, Any]:
        output_dir.mkdir(parents=True, exist_ok=True)
        ref_view = _cap_ref_view(ref_view)

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

        # 重建走 to_gs_ply 单进程加载；勿先 initialize()，否则与子进程/二次加载争用显存导致 OOM。
        root = self._ensure_lhm_path()

        with self._lhm_runtime(root):
            return self._reconstruct_avatar_real(
                image_paths,
                output_dir,
                root,
                ref_view=ref_view,
                export_skinned_mesh=export_skinned_mesh,
            )

    def _reconstruct_avatar_real(
        self,
        image_paths: list[Path],
        output_dir: Path,
        root: Path,
        *,
        ref_view: int,
        export_skinned_mesh: bool,
    ) -> dict[str, Any]:
        import torch
        from shutil import copy2
        from types import SimpleNamespace

        ref_view = _cap_ref_view(ref_view)

        os.environ.update(
            {
                "APP_ENABLED": "1",
                "APP_MODEL_NAME": settings.model_name,
                "APP_TYPE": "infer.human_lrm_a4o",
                "NUMBA_THREADING_LAYER": "omp",
            }
        )
        torch._dynamo.config.disable = True

        image_glob_dir = output_dir / "ref_images"
        image_glob_dir.mkdir(parents=True, exist_ok=True)
        for i, src in enumerate(image_paths):
            ext = src.suffix or ".png"
            copy2(src, image_glob_dir / f"ref_{i:03d}{ext}")

        ply_out = output_dir / "avatar.ply"
        model_path = settings.model_path
        if not model_path:
            from core.utils.model_download_utils import AutoModelQuery

            auto_query = AutoModelQuery(save_dir=str(root / "pretrained_models"))
            model_path = auto_query.query(settings.model_name)

        from scripts.inference.to_gs_ply import run_tpose_export
        from services.lhm_infer_utils import setup_loaders_for_hrm

        args = SimpleNamespace(
            model_name=settings.model_name,
            model_path=model_path,
            image_glob=str(image_glob_dir / "ref_*.png"),
            images_dir=None,
            pose_dir="",
            ref_view=ref_view,
            output=str(ply_out),
            work_dir=str(output_dir / "tpose_gs_work"),
            device="cuda",
            max_image_size=settings.effective_infer_max_image_size,
            lhm_root=str(root),
        )

        betas_list: list[float] | None = None
        model = None
        ref_count = len(image_paths)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        try:
            (
                model,
                _cfg,
                ref_imgs_tensor,
                smplx_params,
                motion_seqs,
                _pose_estimator,
                device,
            ) = setup_loaders_for_hrm(args)
            ref_count = int(ref_imgs_tensor.shape[0])
            import gc

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            run_tpose_export(
                model,
                ref_imgs_tensor,
                motion_seqs,
                device=device,
                output_ply=str(ply_out),
                export_animation_pose=False,
            )
            betas = smplx_params.get("betas")
            if betas is not None:
                betas_list = betas.detach().cpu().reshape(-1).tolist()
                (output_dir / "betas.json").write_text(
                    json.dumps(betas_list), encoding="utf-8"
                )
        except Exception as exc:
            raise RuntimeError(f"3D 重建失败: {exc}") from exc
        finally:
            if model is not None:
                del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        preview_path = image_glob_dir / "ref_000.png"
        if not preview_path.exists() and image_paths:
            preview_path = image_paths[0]

        result = {
            "ply_path": str(ply_out),
            "preview_path": str(preview_path),
            "ref_count": ref_count,
            "export_skinned_mesh": export_skinned_mesh,
            "mock": False,
            "infer_low_memory": settings.infer_low_memory,
            "infer_max_image_size": settings.effective_infer_max_image_size,
            "infer_dense_sample_pts": settings.effective_infer_dense_sample_pts,
            "ref_view_used": ref_view,
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
        ref_view = _cap_ref_view(ref_view)
        video_path = output_dir / "animation.mp4"

        if settings.mock_mode:
            video_path.write_bytes(b"")
            return {"video_path": str(video_path), "frame_count": motion_frames, "mock": True}

        self.initialize()
        root = self._ensure_lhm_path()

        with self._lhm_runtime(root):
            return self._animate_avatar_real(
                image_paths,
                motion_smplx_dir,
                output_dir,
                video_path=video_path,
                ref_view=ref_view,
                motion_frames=motion_frames,
                render_backend=render_backend,
            )

    def _animate_avatar_real(
        self,
        image_paths: list[Path],
        motion_smplx_dir: Path,
        output_dir: Path,
        *,
        video_path: Path,
        ref_view: int,
        motion_frames: int,
        render_backend: str,
    ) -> dict[str, Any]:
        import imageio.v3 as iio
        import numpy as np
        import torch

        from core.utils.app_utils import get_motion_information, obtain_ref_imgs
        from PIL import Image
        from scripts.inference.app_inference import inference_results
        from services.lhm_infer_utils import normalize_ref_imgs
        from scripts.inference.utils import easy_memory_manager

        imgs_pil = [Image.open(p).convert("RGBA") for p in image_paths]
        imgs_arr = normalize_ref_imgs(
            obtain_ref_imgs([(np.array(img),) for img in imgs_pil], ref_view),
        )

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
            batch_size=settings.effective_infer_anim_batch_size,
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
