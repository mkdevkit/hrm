"""将 LHM++ 3D Gaussian Splat 转为带 SMPL-X 骨骼的蒙皮网格（HRM 自研后处理）。"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np

from config import settings
from services.blender_fbx_export import export_skinned_fbx
from services.lhm_infer_utils import list_dense_sample_cache, resolve_dense_sample_pts

logger = logging.getLogger(__name__)

# SMPL-X 标准 55 关节（root + body21 + jaw + eyes2 + hands30）
SMPLX_JOINT_NAMES: list[str] = [
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "jaw",
    "left_eye_smplhf",
    "right_eye_smplhf",
    "left_index1",
    "left_index2",
    "left_index3",
    "left_middle1",
    "left_middle2",
    "left_middle3",
    "left_pinky1",
    "left_pinky2",
    "left_pinky3",
    "left_ring1",
    "left_ring2",
    "left_ring3",
    "left_thumb1",
    "left_thumb2",
    "left_thumb3",
    "right_index1",
    "right_index2",
    "right_index3",
    "right_middle1",
    "right_middle2",
    "right_middle3",
    "right_pinky1",
    "right_pinky2",
    "right_pinky3",
    "right_ring1",
    "right_ring2",
    "right_ring3",
    "right_thumb1",
    "right_thumb2",
    "right_thumb3",
]

SMPLX_PARENTS: list[int] = [
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19,
    15, 15, 15, 20, 25, 26, 20, 28, 29, 20, 31, 32, 20, 34, 35, 20, 37, 38,
    21, 40, 41, 21, 43, 44, 21, 46, 47, 21, 49, 50, 21, 52, 53,
]


def read_gaussian_ply_xyz(ply_path: Path) -> np.ndarray:
    """读取 3DGS PLY 中的高斯中心坐标。"""
    try:
        from plyfile import PlyData

        ply = PlyData.read(str(ply_path))
        vertex = ply["vertex"]
        return np.stack(
            [np.asarray(vertex["x"]), np.asarray(vertex["y"]), np.asarray(vertex["z"])],
            axis=1,
        ).astype(np.float32)
    except Exception:
        return _read_ply_xyz_fallback(ply_path)


def _read_ply_xyz_fallback(ply_path: Path) -> np.ndarray:
    """简易 PLY 解析（无 plyfile 时）。"""
    lines = ply_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    vertex_count = 0
    header_end = 0
    for i, line in enumerate(lines):
        if line.startswith("element vertex"):
            vertex_count = int(line.split()[-1])
        if line.strip() == "end_header":
            header_end = i + 1
            break
    points = []
    for line in lines[header_end : header_end + vertex_count]:
        parts = line.split()
        if len(parts) >= 3:
            points.append([float(parts[0]), float(parts[1]), float(parts[2])])
    if not points:
        raise ValueError(f"无法解析 PLY: {ply_path}")
    return np.asarray(points, dtype=np.float32)


def _load_smplx_mesh(
    lhm_root: Path,
    betas: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], np.ndarray]:
    """从 LHM++ human_model_files 加载 SMPL-X T-pose 网格、LBS 权重与关节位置。"""
    import sys
    import torch

    if str(lhm_root) not in sys.path:
        sys.path.insert(0, str(lhm_root))

    human_model = lhm_root / "pretrained_models" / "human_model_files"
    from engine.pose_estimation.blocks import SMPL_Layer

    layer = SMPL_Layer(
        str(human_model),
        type="smplx",
        gender="neutral",
        num_betas=10,
        kid=False,
        person_center="head",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    layer = layer.to(device)

    if betas is None:
        betas = np.zeros(10, dtype=np.float32)
    betas_t = torch.tensor(betas[:10], dtype=torch.float32, device=device).unsqueeze(0)

    batch_size = 1
    # T-pose：53 关节 axis-angle（与 SMPL_Layer.forward_local 一致，无需相机 K）
    poses = torch.zeros(batch_size, 53, 3, device=device)

    with torch.no_grad():
        output = layer.forward_local(poses, betas_t)
        if output is None:
            raise RuntimeError("SMPL_Layer.forward_local 返回空结果")

    verts = output.vertices[0].detach().cpu().numpy().astype(np.float32)
    faces = layer.bm_x.faces_tensor.detach().cpu().numpy().astype(np.int32)
    weights = layer.bm_x.lbs_weights.detach().cpu().numpy().astype(np.float32)
    joint_names = SMPLX_JOINT_NAMES[: weights.shape[1]]

    if hasattr(output, "joints") and output.joints is not None:
        joints = output.joints[0].detach().cpu().numpy().astype(np.float32)
        if joints.ndim == 3:
            joints = joints[0]
    else:
        j_reg = getattr(layer.bm_x, "J_regressor", None)
        if j_reg is not None:
            import torch

            jv = torch.tensor(verts, dtype=torch.float32, device=j_reg.device)
            joints = torch.matmul(j_reg, jv).detach().cpu().numpy().astype(np.float32)
        else:
            joints = _estimate_joint_positions(verts, weights, len(joint_names))

    return verts, faces, weights, joint_names, joints


def _estimate_joint_positions(
    verts: np.ndarray,
    weights: np.ndarray,
    joint_count: int,
) -> np.ndarray:
    """无 joints 输出时，用 LBS 权重对顶点加权估计关节位置。"""
    joints = np.zeros((joint_count, 3), dtype=np.float32)
    for ji in range(joint_count):
        w = weights[:, ji]
        total = float(w.sum())
        if total > 1e-6:
            joints[ji] = (verts * w[:, None]).sum(axis=0) / total
    return joints


def _read_obj_vertices(obj_path: Path) -> np.ndarray:
    """读取 Wavefront OBJ 顶点坐标（v x y z）。"""
    verts: list[list[float]] = []
    for line in obj_path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("v "):
            continue
        parts = line.split()
        if len(parts) >= 4:
            verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if not verts:
        raise ValueError(f"OBJ 无顶点: {obj_path}")
    return np.asarray(verts, dtype=np.float32)


def ensure_skeleton_rest_positions(
    obj_path: Path,
    skel_path: Path,
    weights_path: Path,
) -> bool:
    """旧版 skeleton.json 缺 joint_rest_positions 时，从 OBJ + LBS 权重补全并写回。"""
    skel = json.loads(skel_path.read_text(encoding="utf-8"))
    joint_names: list[str] = skel.get("joint_names") or []
    joints = skel.get("joint_rest_positions") or []
    if joint_names and len(joints) == len(joint_names):
        return False

    if not joint_names:
        raise ValueError(f"skeleton.json 缺少 joint_names: {skel_path}")

    verts = _read_obj_vertices(obj_path)
    weights = np.load(weights_path)["weights"]
    joint_positions = _estimate_joint_positions(verts, weights, len(joint_names))
    skel["joint_rest_positions"] = joint_positions.tolist()
    skel["version"] = max(int(skel.get("version") or 1), 2)
    skel_path.write_text(json.dumps(skel, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(
        "已补全 skeleton.json 的 joint_rest_positions（%d 关节，来源 OBJ+LBS 权重）",
        len(joint_names),
    )
    return True


def _read_ply_xyz(ply_path: Path) -> np.ndarray:
    """读取 PLY 顶点坐标（dense_sample prior 与 3DGS 导出共用）。"""
    return read_gaussian_ply_xyz(ply_path)


def _load_anchor_points(
    lhm_root: Path,
    *,
    gaussian_count: Optional[int] = None,
    cano_pose_type: int = 1,
) -> np.ndarray:
    """加载 LHM++ 在 SMPL-X 表面的 dense_sample 锚点（与推理时 `{cano}_{pts}.ply` 一致）。"""
    cache_dir = lhm_root / "pretrained_models" / "dense_sample_points"

    if gaussian_count and gaussian_count > 0:
        for cano in (cano_pose_type, 0, 1):
            path = cache_dir / f"{cano}_{gaussian_count}.ply"
            if path.is_file():
                pts = _read_ply_xyz(path)
                logger.info("锚点 PLY: %s (%d points)", path.name, len(pts))
                return pts

    target = gaussian_count or settings.effective_infer_dense_sample_pts or 160000
    if target <= 0:
        target = 160000
    resolved = resolve_dense_sample_pts(lhm_root, target, cano_pose_type)
    if resolved > 0:
        for cano in (cano_pose_type, 0, 1):
            path = cache_dir / f"{cano}_{resolved}.ply"
            if path.is_file():
                pts = _read_ply_xyz(path)
                if gaussian_count and len(pts) != gaussian_count:
                    logger.warning(
                        "锚点数量 (%d) 与高斯数量 (%d) 不一致；可用 prior: %s",
                        len(pts),
                        gaussian_count,
                        list_dense_sample_cache(lhm_root),
                    )
                else:
                    logger.info("锚点 PLY: %s (%d points)", path.name, len(pts))
                return pts

    available = list_dense_sample_cache(lhm_root)
    logger.warning(
        "未找到 dense_sample_points prior（目录 %s，可见: %s），"
        "回退到 SMPL-X 网格顶点，蒙皮网格会严重失真",
        cache_dir,
        available or "（空）",
    )
    verts, _, _, _, _ = _load_smplx_mesh(lhm_root)
    return verts


def apply_gaussian_displacement(
    mesh_verts: np.ndarray,
    gaussian_xyz: np.ndarray,
    anchor_xyz: np.ndarray,
    *,
    k_neighbors: int = 8,
    max_disp: float = 0.12,
    blend_strength: float = 0.85,
) -> np.ndarray:
    """将高斯相对锚点的位移场映射到 SMPL-X 网格顶点。"""
    from scipy.spatial import cKDTree

    if len(gaussian_xyz) != len(anchor_xyz):
        raise RuntimeError(
            f"高斯点数 ({len(gaussian_xyz)}) 与锚点数 ({len(anchor_xyz)}) 不一致，"
            "无法按索引对齐位移场。请确认推理体素采样与 "
            "LHM_ROOT/pretrained_models/dense_sample_points 中的 prior PLY 一致（如 1_40000.ply），"
            "并重新导出蒙皮网格。"
        )

    displacement = gaussian_xyz - anchor_xyz
    norms = np.linalg.norm(displacement, axis=1, keepdims=True)
    displacement = np.where(norms > max_disp, displacement * (max_disp / (norms + 1e-8)), displacement)

    tree = cKDTree(anchor_xyz)
    k = min(k_neighbors, len(anchor_xyz))
    dists, indices = tree.query(mesh_verts, k=k)
    if k == 1:
        dists = dists[:, None]
        indices = indices[:, None]

    weights = 1.0 / (dists + 1e-6)
    weights /= weights.sum(axis=1, keepdims=True)
    vert_disp = (displacement[indices] * weights[..., None]).sum(axis=1)
    return mesh_verts + vert_disp * blend_strength


def _write_obj(
    path: Path,
    verts: np.ndarray,
    faces: np.ndarray,
    uvs: np.ndarray | None = None,
    uv_faces: np.ndarray | None = None,
) -> None:
    lines: list[str] = []
    for v in verts:
        lines.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}")
    if uvs is not None:
        for uv in uvs:
            lines.append(f"vt {uv[0]:.6f} {uv[1]:.6f}")
        use_faces = uv_faces if uv_faces is not None else faces
        for f, uf in zip(faces, use_faces):
            if len(uf) == 3 and np.array_equal(f, uf):
                lines.append(f"f {f[0]+1}/{f[0]+1} {f[1]+1}/{f[1]+1} {f[2]+1}/{f[2]+1}")
            else:
                lines.append(
                    f"f {f[0]+1}/{uf[0]+1} {f[1]+1}/{uf[1]+1} {f[2]+1}/{uf[2]+1}"
                )
    else:
        for f in faces:
            lines.append(f"f {f[0]+1} {f[1]+1} {f[2]+1}")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_skeleton_json(
    path: Path,
    joint_names: list[str],
    parents: list[int],
    betas: list[float],
    weights: np.ndarray,
    joint_rest_positions: np.ndarray,
) -> None:
    sparse_weights: list[dict[str, Any]] = []
    for vi, row in enumerate(weights):
        nz = np.where(row > 1e-4)[0]
        if len(nz):
            sparse_weights.append(
                {"vertex": vi, "joints": nz.tolist(), "values": row[nz].tolist()}
            )

    data = {
        "format": "SMPL-X-skinned-mesh",
        "version": 2,
        "joint_count": len(joint_names),
        "joint_names": joint_names,
        "parents": parents[: len(joint_names)],
        "betas": betas,
        "joint_rest_positions": joint_rest_positions.tolist(),
        "weights_sparse": sparse_weights,
        "note": "SMPL-X 蒙皮数据；FBX 含骨骼绑定，JSON 含完整权重与关节 T-pose 位置",
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _mock_skinned_export(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    verts = np.array(
        [
            [-0.2, 0.0, -0.1], [0.2, 0.0, -0.1], [0.2, 1.2, -0.1], [-0.2, 1.2, -0.1],
            [-0.2, 0.0, 0.1], [0.2, 0.0, 0.1], [0.2, 1.2, 0.1], [-0.2, 1.2, 0.1],
            [-0.15, 1.25, -0.08], [0.15, 1.25, -0.08], [0.15, 1.55, -0.08],
            [-0.15, 1.55, -0.08], [-0.15, 1.25, 0.08], [0.15, 1.25, 0.08],
            [0.15, 1.55, 0.08], [-0.15, 1.55, 0.08],
        ],
        dtype=np.float32,
    )
    faces = np.array(
        [
            [0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6],
            [0, 4, 5], [0, 5, 1], [2, 6, 7], [2, 7, 3],
            [8, 9, 10], [8, 10, 11], [12, 14, 13], [12, 15, 14],
        ],
        dtype=np.int32,
    )
    weights = np.zeros((len(verts), 55), dtype=np.float32)
    weights[:8, 0] = 1.0
    weights[8:, 15] = 1.0
    joint_positions = _estimate_joint_positions(verts, weights, 55)

    obj_path = output_dir / "avatar_skinned.obj"
    skel_path = output_dir / "avatar_skeleton.json"
    fbx_path = output_dir / "avatar_skinned.fbx"
    weights_path = output_dir / "avatar_lbs_weights.npz"

    texture_path = output_dir / "avatar_diffuse.png"
    try:
        from PIL import Image as PILImage

        grad = np.linspace(0.55, 0.85, 256, dtype=np.uint8)
        tile = np.stack(np.meshgrid(grad, grad, indexing="xy"), axis=-1)
        rgb = np.zeros((256, 256, 3), dtype=np.uint8)
        rgb[..., 0] = tile[..., 0]
        rgb[..., 1] = tile[..., 1] * 0.85
        rgb[..., 2] = tile[..., 1] * 0.7
        PILImage.fromarray(rgb, mode="RGB").save(texture_path)
    except Exception:
        texture_path = None

    _write_obj(obj_path, verts, faces)
    _write_skeleton_json(
        skel_path, SMPLX_JOINT_NAMES, SMPLX_PARENTS, [0.0] * 10, weights, joint_positions
    )
    np.savez_compressed(weights_path, weights=weights, faces=faces)
    fbx_ok = export_skinned_fbx(
        obj_path,
        skel_path,
        weights_path,
        fbx_path,
        texture_path=texture_path if texture_path and texture_path.is_file() else None,
        subdivision_levels=1,
    )

    return {
        "mesh_obj_path": str(obj_path),
        "mesh_fbx_path": str(fbx_path) if fbx_ok else None,
        "mesh_texture_path": str(texture_path) if texture_path and texture_path.is_file() else None,
        "skeleton_json_path": str(skel_path),
        "lbs_weights_path": str(weights_path),
        "joint_count": 55,
        "mock": True,
    }


def export_skinned_mesh_from_gaussian(
    ply_path: Path,
    output_dir: Path,
    *,
    betas: Optional[list[float] | np.ndarray] = None,
    lhm_root: Optional[str] = None,
) -> dict[str, Any]:
    """高斯 PLY → SMPL-X 蒙皮网格（OBJ 中间产物 + FBX + 骨骼 JSON）。"""
    output_dir.mkdir(parents=True, exist_ok=True)

    if settings.mock_mode:
        return _mock_skinned_export(output_dir)

    if lhm_root:
        root = Path(lhm_root).resolve()
    elif settings.lhm_root:
        root = Path(settings.lhm_root).resolve()
    else:
        root = (Path(__file__).resolve().parents[2] / "LHM-plusplus").resolve()

    if not root.exists():
        raise RuntimeError("LHM_ROOT 未配置，无法导出蒙皮网格")

    betas_arr = np.asarray(betas if betas is not None else np.zeros(10), dtype=np.float32)

    gaussian_xyz = read_gaussian_ply_xyz(ply_path)
    logger.info("高斯 PLY: %s (%d points)", ply_path.name, len(gaussian_xyz))
    anchor_xyz = _load_anchor_points(root, gaussian_count=len(gaussian_xyz))
    mesh_verts, faces, lbs_weights, joint_names, joint_positions = _load_smplx_mesh(root, betas_arr)
    displaced_verts = apply_gaussian_displacement(mesh_verts, gaussian_xyz, anchor_xyz)

    obj_path = output_dir / "avatar_skinned.obj"
    skel_path = output_dir / "avatar_skeleton.json"
    fbx_path = output_dir / "avatar_skinned.fbx"
    weights_path = output_dir / "avatar_lbs_weights.npz"
    texture_path: Path | None = output_dir / "avatar_diffuse.png"
    uvs: np.ndarray | None = None
    uv_faces: np.ndarray | None = None
    if settings.fbx_bake_texture:
        try:
            from services.texture_bake_service import bake_diffuse_from_gaussian_ply, load_smplx_uv

            uvs, uv_faces = load_smplx_uv(root, faces)
            bake_diffuse_from_gaussian_ply(
                ply_path,
                displaced_verts,
                faces,
                root,
                texture_path,
                texture_size=settings.fbx_texture_size,
            )
        except Exception as exc:
            logger.warning("3DGS UV 贴图烘焙失败，FBX 将无贴图: %s", exc)
            texture_path = None
            uvs = None
            uv_faces = None
    else:
        texture_path = None

    _write_obj(obj_path, displaced_verts, faces, uvs=uvs, uv_faces=uv_faces)
    _write_skeleton_json(
        skel_path,
        joint_names,
        SMPLX_PARENTS,
        betas_arr.tolist(),
        lbs_weights,
        joint_positions,
    )
    np.savez_compressed(weights_path, weights=lbs_weights, faces=faces)
    tex_for_fbx = texture_path if texture_path and texture_path.is_file() else None
    fbx_ok = export_skinned_fbx(
        obj_path,
        skel_path,
        weights_path,
        fbx_path,
        texture_path=tex_for_fbx,
        subdivision_levels=settings.fbx_subdivision_levels,
    )

    return {
        "mesh_obj_path": str(obj_path),
        "mesh_fbx_path": str(fbx_path) if fbx_ok else None,
        "mesh_texture_path": str(tex_for_fbx) if tex_for_fbx else None,
        "skeleton_json_path": str(skel_path),
        "lbs_weights_path": str(weights_path),
        "joint_count": len(joint_names),
        "vertex_count": int(len(displaced_verts)),
        "fbx_subdivision_levels": settings.fbx_subdivision_levels,
        "fbx_texture_baked": bool(tex_for_fbx),
        "mock": False,
    }
