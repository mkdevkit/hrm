"""HRM 侧 LHM++ 推理辅助：不修改 LHM-plusplus 仓库即可用的补丁逻辑。"""

from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from accelerate import Accelerator
from PIL import Image

from config import settings
from core.utils.app_utils import obtain_ref_imgs
from core.utils.model_card import MODEL_CONFIG
from core.utils.model_download_utils import AutoModelQuery

logger = logging.getLogger(__name__)

_DENSE_SAMPLE_CANDIDATES = (40000, 80000, 160000)


def list_dense_sample_cache(lhm_root: Path) -> list[str]:
    cache_dir = lhm_root / "pretrained_models" / "dense_sample_points"
    try:
        if not cache_dir.exists():
            return []
        return sorted(
            p.name
            for p in cache_dir.iterdir()
            if p.suffix == ".ply" and p.is_file()
        )
    except OSError:
        try:
            return sorted(
                x
                for x in os.listdir(str(cache_dir))
                if x.endswith(".ply") and os.path.isfile(os.path.join(str(cache_dir), x))
            )
        except OSError:
            return []


def resolve_dense_sample_pts(lhm_root: Path, target: int, cano_pose_type: int) -> int:
    """Pick the largest cached dense_sample PLY count not exceeding ``target``."""
    cache_dir = lhm_root / "pretrained_models" / "dense_sample_points"

    def _has(cano: int, pts: int) -> bool:
        path = cache_dir / f"{cano}_{pts}.ply"
        try:
            return path.is_file()
        except OSError:
            return os.path.isfile(str(path))

    if _has(cano_pose_type, target):
        return target
    for pts in sorted(_DENSE_SAMPLE_CANDIDATES, reverse=True):
        if pts <= target and _has(cano_pose_type, pts):
            return pts
    for cano in (0, 1):
        for pts in sorted(_DENSE_SAMPLE_CANDIDATES, reverse=True):
            if pts <= target and _has(cano, pts):
                return pts
    for cano in (0, 1):
        for pts in sorted(_DENSE_SAMPLE_CANDIDATES):
            if _has(cano, pts):
                return pts
    return 0


def require_dense_sample_pts(lhm_root: Path, target: int, cano_pose_type: int) -> int:
    resolved = resolve_dense_sample_pts(lhm_root, target, cano_pose_type)
    if resolved > 0:
        return resolved
    cache_dir = lhm_root / "pretrained_models" / "dense_sample_points"
    available = list_dense_sample_cache(lhm_root)
    raise RuntimeError(
        "低显存推理需要 dense_sample_points prior 缓存，但未找到可用 PLY。"
        f" 目标≤{target}（cano_pose_type={cano_pose_type}）。"
        f" 目录: {cache_dir}；当前可见文件: {available or '（空或不可读）'}。"
        " 请在 LHM++ 根目录执行 prior 模型下载（AutoModelQuery.download_all_prior_models）。"
    )


def prepare_patched_model_path(
    model_path: str,
    lhm_root: Path,
    work_dir: Path,
    dense_sample_pts: int,
) -> tuple[str, int]:
    """Symlink 权重目录 + 写入降采样 config.json，使模型从构建起就用更少的体素点。"""
    mp = Path(model_path).resolve()
    config = json.loads((mp / "config.json").read_text(encoding="utf-8"))
    cano = int(config.get("cano_pose_type", 1))
    resolved = require_dense_sample_pts(lhm_root, dense_sample_pts, cano)
    orig = int(config.get("dense_sample_pts", 160000))
    if resolved >= orig:
        logger.info(
            "dense_sample_pts 无需 patch：checkpoint=%s，目标=%s，已满足",
            orig,
            resolved,
        )
        return str(mp), resolved

    patch_root = work_dir / "model_load_patch"
    if patch_root.exists():
        shutil.rmtree(patch_root, ignore_errors=True)
    patch_root.mkdir(parents=True, exist_ok=True)

    patched = dict(config)
    patched["dense_sample_pts"] = resolved
    (patch_root / "config.json").write_text(
        json.dumps(patched, indent=2), encoding="utf-8"
    )

    for item in mp.iterdir():
        if item.name == "config.json":
            continue
        dst = patch_root / item.name
        if dst.exists():
            continue
        try:
            os.symlink(item, dst)
        except OSError:
            if item.is_dir():
                shutil.copytree(item, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(item, dst)

    logger.info(
        "已创建低显存 model 加载目录 dense_sample_pts=%s -> %s",
        resolved,
        patch_root,
    )
    return str(patch_root.resolve()), resolved


def apply_infer_memory_overrides(model: torch.nn.Module, lhm_root: Path) -> int | None:
    """Reload skinning dense_pts if build path still used the default count."""
    target = settings.effective_infer_dense_sample_pts
    if target <= 0:
        return None

    renderer = getattr(model, "renderer", None)
    skinning = getattr(renderer, "smplx_model", None) if renderer is not None else None
    if skinning is None or not hasattr(skinning, "dense_sample"):
        return None

    params = getattr(skinning, "input_params", None) or {}
    current = int(params.get("dense_sample_points", 0) or 0)
    cano_pose_type = int(getattr(skinning, "cano_pose_type", params.get("cano_pose_type", 1)))
    resolved = resolve_dense_sample_pts(lhm_root, target, cano_pose_type)
    if resolved <= 0 or current == resolved:
        return resolved if resolved > 0 else None

    body_face_ratio = int(params.get("body_face_ratio", 3))
    prev_cwd = os.getcwd()
    os.chdir(lhm_root)
    try:
        skinning.dense_sample(body_face_ratio, resolved)
    finally:
        os.chdir(prev_cwd)
    params["dense_sample_points"] = resolved
    skinning.input_params = params
    return resolved


def normalize_ref_imgs(
    imgs: list[np.ndarray],
    *,
    tgt_max_size: int | None = None,
) -> list[np.ndarray]:
    """多视角参考图 PadRatioWithScale 对齐（与 Gradio app 一致）。"""
    from core.datasets.data_utils import SrcImagePipeline

    if tgt_max_size is None:
        tgt_max_size = settings.effective_infer_max_image_size

    rgb_float: list[np.ndarray] = []
    for img in imgs:
        arr = np.asarray(img)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        elif arr.shape[-1] >= 4:
            arr = arr[..., :3]
        if arr.dtype == np.uint8 or float(arr.max()) > 1.0:
            arr = arr.astype(np.float32) / 255.0
        else:
            arr = arr.astype(np.float32)
        rgb_float.append(arr)

    pipeline = SrcImagePipeline(
        dict(
            name="PadRatioWithScale",
            target_ratio=5 / 3,
            tgt_max_size_list=[tgt_max_size],
            val=True,
        )
    )
    normalized = pipeline(rgb_float)
    return [(x * 255.0).clip(0, 255).astype(np.uint8) for x in normalized]


def setup_loaders_for_hrm(args: SimpleNamespace) -> tuple[Any, ...]:
    """``to_gs_ply.setup_loaders_and_inputs`` 的 HRM 封装：lazy PoseEstimator + 图像归一化。"""
    from scripts.inference.to_gs_ply import (
        _build_motion_seq_from_pose_json,
        _build_synthetic_motion_seq,
        _easy_memory_manager,
        _effective_pose_dir,
        _require_gs_output_model,
        _resolve_image_paths,
        build_app_model,
        parse_app_configs,
        prior_model_check,
    )

    _require_gs_output_model(args.model_name)

    device = args.device
    dtype = torch.float32
    lhm_root = Path(getattr(args, "lhm_root", ".")).resolve()
    work_dir = Path(getattr(args, "work_dir", ".")).resolve()

    prior_model_check(save_dir="./pretrained_models")
    model_config = MODEL_CONFIG[args.model_name]
    if args.model_path:
        model_path = args.model_path
    else:
        auto_query = AutoModelQuery(save_dir=str(lhm_root / "pretrained_models"))
        model_path = auto_query.query(args.model_name)

    load_path = model_path
    dense_applied: int | None = None
    target_dense = settings.effective_infer_dense_sample_pts
    logger.info(
        "推理显存配置: low_memory=%s max_image=%s ref_view_max=%s target_dense=%s",
        settings.infer_low_memory,
        settings.effective_infer_max_image_size,
        settings.effective_infer_ref_view_max,
        target_dense,
    )
    if target_dense > 0:
        load_path, dense_applied = prepare_patched_model_path(
            model_path, lhm_root, work_dir, target_dense
        )
    else:
        logger.info("未启用 dense_sample_pts 覆盖（INFER_DENSE_SAMPLE_PTS=0 且 INFER_LOW_MEMORY=false）")

    model_cards = {
        args.model_name: {
            "model_path": load_path,
            "model_config": model_config,
        }
    }

    _ = Accelerator()
    cfg, _ = parse_app_configs(model_cards)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model = build_app_model(cfg)
    model.to(device)

    if dense_applied is None:
        dense_applied = apply_infer_memory_overrides(model, lhm_root)
    logger.info("[HRM] infer dense_sample_points=%s", dense_applied or "default(checkpoint)")

    pose_estimator = None
    if cfg.get("use_smplx_shape_estimator", True):
        from engine.pose_estimation.pose_estimator import PoseEstimator

        pose_estimator = PoseEstimator(
            "./pretrained_models/human_model_files/", device="cpu"
        )
        pose_estimator.device = device

    image_paths = _resolve_image_paths(args)
    imgs_pil = [Image.open(p) for p in image_paths]
    image_for_prepare = [(np.asarray(img),) for img in imgs_pil]

    out_parent = os.path.dirname(os.path.abspath(args.output))
    if not out_parent:
        out_parent = os.getcwd()
    work_dir_path = args.work_dir or os.path.join(out_parent, "debug", "tpose_gs_work")
    os.makedirs(work_dir_path, exist_ok=True)
    working_dir = SimpleNamespace()
    working_dir.name = os.path.abspath(work_dir_path)

    imgs = obtain_ref_imgs(image_for_prepare, ref_view=args.ref_view)
    max_image_size = int(getattr(args, "max_image_size", 0) or settings.effective_infer_max_image_size)
    imgs = normalize_ref_imgs(imgs, tgt_max_size=max_image_size)
    sample_imgs = np.concatenate(imgs, axis=1)
    save_sample_imgs = os.path.join(working_dir.name, "raw.png")
    with Image.fromarray(sample_imgs) as img:
        img.save(save_sample_imgs)

    pose_json = _effective_pose_dir(args)
    if pose_json is not None:
        motion_seqs = _build_motion_seq_from_pose_json(pose_json, cfg)
    else:
        motion_seqs = _build_synthetic_motion_seq(cfg)

    if pose_estimator is not None:
        with torch.no_grad():
            with _easy_memory_manager(pose_estimator, device=device):
                shape_pose = pose_estimator(imgs[0])
        if not shape_pose.is_full_body:
            raise ValueError(f"Input image invalid for shape estimator: {shape_pose.msg}")

    img_np = np.stack(imgs) / 255.0
    ref_imgs_tensor = torch.from_numpy(img_np).permute(0, 3, 1, 2).float().to(device)
    smplx_params = motion_seqs["smplx_params"].copy()
    if pose_estimator is not None:
        smplx_params["betas"] = torch.tensor(
            shape_pose.beta, dtype=dtype, device=device
        ).unsqueeze(0)
    motion_seqs["smplx_params"] = smplx_params

    return model, cfg, ref_imgs_tensor, smplx_params, motion_seqs, pose_estimator, device
