"""LHM++ 3DGS 导出并写入 mesh 对齐 sidecar（gs_anchors / smplx_canonical_verts）。"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)


def _try_save_smplx_canonical_verts(
    model: torch.nn.Module,
    gs_smplx: dict[str, torch.Tensor],
    sidecar_dir: Path,
) -> bool:
    """从 LHM renderer 导出与 3DGS 同坐标系的 canonical SMPL-X 顶点。"""
    renderer = getattr(model, "renderer", None)
    if renderer is None:
        return False
    try:
        smplx_view = renderer.get_single_view_smpl_data(gs_smplx, 0)
        smplx_one = renderer._get_single_batch_data(smplx_view, 0)
        skinning = renderer.smplx_model
        with torch.no_grad():
            out = skinning(smplx_one)
            if isinstance(out, dict):
                verts = out.get("vertices") or out.get("verts")
            else:
                verts = getattr(out, "vertices", None)
            if verts is None:
                return False
            arr = verts.detach().cpu().numpy().astype(np.float32).reshape(-1, 3)
        if arr.shape[0] < 1000:
            return False
        np.save(sidecar_dir / "smplx_canonical_verts.npy", arr)
        logger.info("已保存 smplx_canonical_verts.npy (%d verts)", len(arr))
        return True
    except Exception as exc:
        logger.warning("smplx_canonical_verts 导出失败（将回退 SMPL_Layer）: %s", exc)
        return False


@torch.no_grad()
def run_tpose_export_with_sidecars(
    model: torch.nn.Module,
    ref_imgs_tensor: torch.Tensor,
    motion_seq: dict[str, Any],
    device: str,
    output_ply: str,
    sidecar_dir: Path,
    *,
    export_animation_pose: bool = False,
) -> None:
    """与 LHM++ ``run_tpose_export`` 相同，并保存推理侧锚点供蒙皮网格对齐。"""
    from scripts.inference.to_gs_ply import (
        build_animation_frame_smplx_params,
        build_tpose_smplx_params,
        slice_motion_seq_to_single_frame,
    )

    sidecar_dir = Path(sidecar_dir)
    sidecar_dir.mkdir(parents=True, exist_ok=True)

    dev = torch.device(device)
    use_pred_render = getattr(model, "use_pred_shape_for_render", False)
    motion_one = slice_motion_seq_to_single_frame(motion_seq, frame_idx=0)

    render_c2ws = motion_one["render_c2ws"].to(dev)
    render_intrs = motion_one["render_intrs"].to(dev)
    render_bg_colors = motion_one["render_bg_colors"].to(dev)
    smplx_dev = {k: v.to(dev) for k, v in motion_one["smplx_params"].items()}

    ref_batch = ref_imgs_tensor.unsqueeze(0)
    ref_mask = torch.ones(ref_imgs_tensor.shape[0], dtype=torch.bool, device=dev).unsqueeze(0)

    model_outputs = model.infer_single_view(
        ref_batch,
        None,
        None,
        render_c2ws=render_c2ws,
        render_intrs=render_intrs,
        render_bg_colors=render_bg_colors,
        smplx_params=smplx_dev,
        ref_imgs_bool=ref_mask,
        return_pred_shape=use_pred_render,
    )

    pred_shape = None
    if len(model_outputs) == 8:
        (
            gs_model_list,
            query_points,
            transform_mat_neutral_pose,
            gs_hidden_features,
            _image_feats,
            _motion_emb,
            _pos_emb,
            pred_shape,
        ) = model_outputs
    elif len(model_outputs) == 7:
        (
            gs_model_list,
            query_points,
            transform_mat_neutral_pose,
            gs_hidden_features,
            _image_feats,
            _motion_emb,
            _pos_emb,
        ) = model_outputs
    else:
        raise RuntimeError(f"Unexpected infer_single_view outputs: {len(model_outputs)}")

    anchors = query_points["neutral_coords"][0].detach().cpu().numpy().astype(np.float32)
    np.save(sidecar_dir / "gs_anchors.npy", anchors)
    np.save(
        sidecar_dir / "gs_transform_neutral.npy",
        transform_mat_neutral_pose.detach().cpu().numpy().astype(np.float32),
    )
    meta = {
        "anchor_count": int(len(anchors)),
        "export_pose": "animation" if export_animation_pose else "tpose",
        "use_pred_shape_for_render": bool(use_pred_render),
    }
    (sidecar_dir / "gs_export_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("已保存 gs_anchors.npy (%d 点，与 PLY 高斯索引对齐)", len(anchors))

    merged = type(model).smplx_params_with_pred_shape_betas(smplx_dev, pred_shape)
    merged_betas = merged["betas"]
    dtype = merged_betas.dtype

    if export_animation_pose:
        gs_smplx = build_animation_frame_smplx_params(
            motion_one,
            transform_mat_neutral_pose,
            merged_betas,
            dev,
            dtype,
        )
    else:
        gs_smplx = build_tpose_smplx_params(
            motion_one,
            transform_mat_neutral_pose,
            merged_betas,
            dev,
            dtype,
        )

    _try_save_smplx_canonical_verts(model, gs_smplx, sidecar_dir)

    view_idx = 0
    renderer = model.renderer
    if export_animation_pose:
        smplx_view = renderer.get_single_view_smpl_data(gs_smplx, view_idx)
        smplx_one = renderer._get_single_batch_data(smplx_view, 0)
        anim_models, _ = renderer.animate_gs_model(
            gs_model_list[0],
            query_points["neutral_coords"][0],
            smplx_one,
            debug=False,
            mesh_meta=query_points["mesh_meta"],
        )
        if not anim_models:
            cano_gs = model.inference_gs(
                gs_model_list,
                query_points,
                gs_smplx,
                render_c2ws,
                render_intrs,
                render_bg_colors,
                gs_hidden_features,
                pad_forward=False,
            )
        else:
            cano_gs = anim_models[0]
    else:
        cano_gs = model.inference_gs(
            gs_model_list,
            query_points,
            gs_smplx,
            render_c2ws,
            render_intrs,
            render_bg_colors,
            gs_hidden_features,
            pad_forward=False,
        )

    out_abs = Path(output_ply).resolve()
    out_abs.parent.mkdir(parents=True, exist_ok=True)
    cano_gs.save_ply(str(out_abs))
    logger.info("3DGS PLY 已保存: %s", out_abs.name)
