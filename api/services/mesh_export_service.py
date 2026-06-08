"""将 LHM++ 3D Gaussian Splat 转为带 SMPL-X 骨骼的蒙皮网格（HRM 自研后处理）。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np

from config import settings

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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """从 LHM++ human_model_files 加载 SMPL-X T-pose 网格与 LBS 权重。"""
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
    return verts, faces, weights, joint_names


def _load_anchor_points(lhm_root: Path, count: Optional[int] = None) -> np.ndarray:
    """加载 LHM++ 在 SMPL-X 表面的采样锚点。"""
    candidates = [
        lhm_root / "pretrained_models" / "dense_sample_points" / "smplx_dense_points.npy",
        lhm_root / "pretrained_models" / "dense_sample_points" / "dense_points.npy",
        lhm_root / "pretrained_models" / "dense_sample_points" / "points.npy",
    ]
    for path in candidates:
        if path.exists():
            pts = np.load(path).astype(np.float32)
            if pts.ndim == 3:
                pts = pts[0]
            if count and len(pts) > count:
                idx = np.linspace(0, len(pts) - 1, count, dtype=int)
                pts = pts[idx]
            return pts

    verts, _, _, _ = _load_smplx_mesh(lhm_root)
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

    n_gs = min(len(gaussian_xyz), len(anchor_xyz))
    gaussian_xyz = gaussian_xyz[:n_gs]
    anchor_xyz = anchor_xyz[:n_gs]

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


def _write_obj(path: Path, verts: np.ndarray, faces: np.ndarray) -> None:
    lines = []
    for v in verts:
        lines.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}")
    for f in faces:
        lines.append(f"f {f[0]+1} {f[1]+1} {f[2]+1}")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_skeleton_json(
    path: Path,
    joint_names: list[str],
    parents: list[int],
    betas: list[float],
    weights: np.ndarray,
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
        "version": 1,
        "joint_count": len(joint_names),
        "joint_names": joint_names,
        "parents": parents[: len(joint_names)],
        "betas": betas,
        "weights_sparse": sparse_weights,
        "note": "传统蒙皮网格，可用 Blender/Maya 导入 OBJ 并绑定此骨骼权重",
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_glb_skinned(
    path: Path,
    verts: np.ndarray,
    faces: np.ndarray,
) -> bool:
    """尝试导出 GLB（依赖 trimesh，仅几何体，无骨骼绑定）。"""
    try:
        import trimesh

        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        mesh.export(str(path))
        return path.exists()
    except Exception:
        return False


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

    obj_path = output_dir / "avatar_skinned.obj"
    skel_path = output_dir / "avatar_skeleton.json"
    glb_path = output_dir / "avatar_skinned.glb"

    _write_obj(obj_path, verts, faces)
    _write_skeleton_json(skel_path, SMPLX_JOINT_NAMES, SMPLX_PARENTS, [0.0] * 10, weights)
    _write_glb_skinned(glb_path, verts, faces)

    return {
        "mesh_obj_path": str(obj_path),
        "skeleton_json_path": str(skel_path),
        "mesh_glb_path": str(glb_path) if glb_path.exists() else None,
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
    """高斯 PLY → SMPL-X 蒙皮网格（OBJ + 骨骼 JSON + 可选 GLB）。"""
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
    anchor_xyz = _load_anchor_points(root, count=len(gaussian_xyz))
    mesh_verts, faces, lbs_weights, joint_names = _load_smplx_mesh(root, betas_arr)
    displaced_verts = apply_gaussian_displacement(mesh_verts, gaussian_xyz, anchor_xyz)

    obj_path = output_dir / "avatar_skinned.obj"
    skel_path = output_dir / "avatar_skeleton.json"
    glb_path = output_dir / "avatar_skinned.glb"
    weights_path = output_dir / "avatar_lbs_weights.npz"

    _write_obj(obj_path, displaced_verts, faces)
    _write_skeleton_json(skel_path, joint_names, SMPLX_PARENTS, betas_arr.tolist(), lbs_weights)
    np.savez_compressed(weights_path, weights=lbs_weights, faces=faces)
    glb_ok = _write_glb_skinned(glb_path, displaced_verts, faces)

    return {
        "mesh_obj_path": str(obj_path),
        "skeleton_json_path": str(skel_path),
        "mesh_glb_path": str(glb_path) if glb_ok else None,
        "lbs_weights_path": str(weights_path),
        "joint_count": len(joint_names),
        "vertex_count": int(len(displaced_verts)),
        "mock": False,
    }
