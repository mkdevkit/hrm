"""将 3DGS 颜色烘焙到 SMPL-X UV 贴图（供 FBX 导出使用）。"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

_SH_C0 = 0.28209479177387814


def read_gaussian_ply_rgb(ply_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """读取 3DGS PLY 坐标与 RGB（由 SH0 / 顶点色字段转换）。"""
    from plyfile import PlyData

    ply = PlyData.read(str(ply_path))
    vertex = ply["vertex"]
    names = vertex.data.dtype.names or ()
    xyz = np.stack(
        [np.asarray(vertex["x"]), np.asarray(vertex["y"]), np.asarray(vertex["z"])],
        axis=1,
    ).astype(np.float32)

    if all(k in names for k in ("f_dc_0", "f_dc_1", "f_dc_2")):
        sh = np.stack(
            [np.asarray(vertex["f_dc_0"]), np.asarray(vertex["f_dc_1"]), np.asarray(vertex["f_dc_2"])],
            axis=1,
        ).astype(np.float32)
        rgb = np.clip(0.5 + _SH_C0 * sh, 0.0, 1.0)
    elif all(k in names for k in ("red", "green", "blue")):
        rgb = np.stack(
            [np.asarray(vertex["red"]), np.asarray(vertex["green"]), np.asarray(vertex["blue"])],
            axis=1,
        ).astype(np.float32)
        if rgb.max() > 1.5:
            rgb /= 255.0
    else:
        logger.warning("PLY 无 f_dc_* / RGB 字段，使用默认灰色")
        rgb = np.full((len(xyz), 3), 0.72, dtype=np.float32)

    return xyz, rgb.astype(np.float32)


def sample_colors_at_vertices(
    mesh_verts: np.ndarray,
    gaussian_xyz: np.ndarray,
    gaussian_rgb: np.ndarray,
    *,
    k_neighbors: int = 8,
) -> np.ndarray:
    """用最近邻高斯颜色插值得到网格顶点色。"""
    from scipy.spatial import cKDTree

    k = min(k_neighbors, len(gaussian_xyz))
    dists, indices = cKDTree(gaussian_xyz).query(mesh_verts, k=k)
    if k == 1:
        dists = dists[:, None]
        indices = indices[:, None]
    weights = 1.0 / (dists + 1e-6)
    weights /= weights.sum(axis=1, keepdims=True)
    return (gaussian_rgb[indices] * weights[..., None]).sum(axis=1).astype(np.float32)


def parse_obj_uv(obj_path: Path) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """从 OBJ 解析 vt 与 f v/vt 面。"""
    verts_uv: list[list[float]] = []
    faces_uv: list[list[int]] = []
    for line in obj_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("vt "):
            parts = line.split()
            if len(parts) >= 3:
                verts_uv.append([float(parts[1]), float(parts[2])])
        elif line.startswith("f "):
            uv_idx: list[int] = []
            for token in line.split()[1:]:
                chunks = token.split("/")
                if len(chunks) >= 2 and chunks[1]:
                    uv_idx.append(int(chunks[1]) - 1)
            if len(uv_idx) == 3:
                faces_uv.append(uv_idx)
    if not verts_uv or not faces_uv:
        return None, None
    return np.asarray(verts_uv, dtype=np.float32), np.asarray(faces_uv, dtype=np.int32)


def _load_uv_from_trimesh(obj_path: Path, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    try:
        import trimesh
    except ImportError:
        return None

    try:
        loaded = trimesh.load(str(obj_path), process=False, force="mesh")
    except Exception:
        return None

    if isinstance(loaded, trimesh.Scene):
        meshes = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not meshes:
            return None
        loaded = trimesh.util.concatenate(meshes)

    visual = getattr(loaded, "visual", None)
    uvs = getattr(visual, "uv", None) if visual is not None else None
    if uvs is None or len(uvs) == 0:
        return None

    uvs = np.asarray(uvs, dtype=np.float32)
    if uvs.ndim != 2 or uvs.shape[1] < 2:
        return None
    uvs = uvs[:, :2]

    if len(uvs) == len(loaded.vertices):
        if len(loaded.faces) == len(faces):
            return uvs, np.asarray(loaded.faces, dtype=np.int32)
        return uvs, faces.copy()
    return None


def _load_uv_from_pkl_npz(path: Path, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    data: dict | None = None
    if path.suffix.lower() == ".npz":
        archive = np.load(path, allow_pickle=True)
        data = {k: archive[k] for k in archive.files}
    elif path.suffix.lower() in (".pkl", ".pickle"):
        import pickle

        with path.open("rb") as f:
            try:
                raw = pickle.load(f, encoding="latin1")
            except TypeError:
                raw = pickle.load(f)
        data = raw if isinstance(raw, dict) else None

    if not data:
        return None

    for uk, fk in (
        ("vt", "ft"),
        ("uv", "ft"),
        ("texcoords", "texfaces"),
        ("vtx_uv", "faces_uv"),
    ):
        if uk not in data or fk not in data:
            continue
        uvs = np.asarray(data[uk], dtype=np.float32)
        uv_faces = np.asarray(data[fk], dtype=np.int32)
        if uvs.ndim == 2 and uvs.shape[1] >= 2 and len(uv_faces) == len(faces):
            return uvs[:, :2], uv_faces
    return None


def _load_uv_from_bm_x(bm: object, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    import torch

    for attr in (
        "vtx_uv",
        "verts_uv",
        "texcoords",
        "uv",
        "uvs",
        "tex_uv",
        "texture_uv",
    ):
        if not hasattr(bm, attr):
            continue
        raw = getattr(bm, attr)
        if isinstance(raw, torch.Tensor):
            uvs = raw.detach().cpu().numpy().astype(np.float32)
        else:
            uvs = np.asarray(raw, dtype=np.float32)
        if uvs.ndim == 2 and uvs.shape[1] >= 2:
            logger.info("SMPL-X UV 来自 bm_x.%s (%d)", attr, len(uvs))
            uv_faces = faces.copy()
            for fk in ("faces_uv", "ft", "texfaces"):
                if hasattr(bm, fk):
                    ff = getattr(bm, fk)
                    if isinstance(ff, torch.Tensor):
                        uv_faces = ff.detach().cpu().numpy().astype(np.int32)
                    else:
                        uv_faces = np.asarray(ff, dtype=np.int32)
                    break
            return uvs[:, :2], uv_faces
    return None


def _generate_cylindrical_uv(verts: np.ndarray) -> np.ndarray:
    """无官方 UV 时的柱面展开回退（可烘焙 3DGS 颜色，质量低于 SMPL-X UV）。"""
    x, y, z = verts[:, 0], verts[:, 1], verts[:, 2]
    theta = np.arctan2(x, z)
    u = (theta / (2.0 * np.pi)) + 0.5
    y_min, y_max = float(y.min()), float(y.max())
    v = (y - y_min) / (y_max - y_min + 1e-8)
    return np.stack([u, v], axis=1).astype(np.float32)


def _candidate_uv_paths(human_model: Path) -> list[Path]:
    from config import settings

    candidates: list[Path] = []
    if settings.smplx_uv_obj:
        candidates.append(Path(settings.smplx_uv_obj))
    bundled = Path(__file__).resolve().parents[1] / "assets" / "smplx_uv.obj"
    candidates.append(bundled)

    if human_model.is_dir():
        patterns = (
            "*uv*.obj",
            "smplx_uv.obj",
            "smplx*.obj",
            "*.obj",
            "*uv*.npz",
            "*.npz",
            "*.pkl",
        )
        for pattern in patterns:
            candidates.extend(sorted(human_model.rglob(pattern)))

    seen: set[Path] = set()
    unique: list[Path] = []
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen or not path.is_file():
            continue
        seen.add(resolved)
        unique.append(path)
    return unique


def load_smplx_uv(
    lhm_root: Path,
    faces: np.ndarray,
    *,
    mesh_verts: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """加载 SMPL-X UV；多源回退，最后使用柱面 UV。"""
    import sys

    if str(lhm_root) not in sys.path:
        sys.path.insert(0, str(lhm_root))

    human_model = lhm_root / "pretrained_models" / "human_model_files"

    try:
        from engine.pose_estimation.blocks import SMPL_Layer

        layer = SMPL_Layer(
            str(human_model),
            type="smplx",
            gender="neutral",
            num_betas=10,
            kid=False,
            person_center="head",
        )
        result = _load_uv_from_bm_x(layer.bm_x, faces)
        if result is not None:
            return result
    except Exception as exc:
        logger.warning("从 SMPL_Layer 读取 UV 失败: %s", exc)

    for path in _candidate_uv_paths(human_model):
        if path.suffix.lower() in (".pkl", ".pickle", ".npz"):
            result = _load_uv_from_pkl_npz(path, faces)
            if result is not None:
                logger.info("SMPL-X UV 来自 %s", path)
                return result
            continue

        uvs, uv_faces = parse_obj_uv(path)
        if uvs is not None and uv_faces is not None and len(uv_faces) == len(faces):
            logger.info("SMPL-X UV 来自 OBJ vt: %s", path.name)
            return uvs, uv_faces

        result = _load_uv_from_trimesh(path, faces)
        if result is not None:
            logger.info("SMPL-X UV 来自 trimesh: %s", path.name)
            return result

    if mesh_verts is not None and len(mesh_verts) > 0:
        logger.warning(
            "未找到 SMPL-X 官方 UV，使用柱面 UV 回退。"
            " 建议将 smplx_uv.obj 放到 %s 或设置 SMPLX_UV_OBJ",
            human_model,
        )
        return _generate_cylindrical_uv(mesh_verts), faces.copy()

    raise RuntimeError(
        "未找到 SMPL-X UV。请将官方 smplx_uv.obj 复制到 "
        f"{human_model} 或 api/assets/smplx_uv.obj，"
        "或在 api/.env 设置 SMPLX_UV_OBJ=/path/to/smplx_uv.obj"
    )


def _barycentric(px: float, py: float, tri: np.ndarray) -> tuple[float, float, float] | None:
    x0, y0 = tri[0]
    x1, y1 = tri[1]
    x2, y2 = tri[2]
    denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    if abs(denom) < 1e-12:
        return None
    w0 = ((y1 - y2) * (px - x2) + (x2 - x1) * (py - y2)) / denom
    w1 = ((y2 - y0) * (px - x2) + (x0 - x2) * (py - y2)) / denom
    w2 = 1.0 - w0 - w1
    if w0 < -1e-4 or w1 < -1e-4 or w2 < -1e-4:
        return None
    return float(w0), float(w1), float(w2)


def bake_vertex_colors_to_texture(
    uvs: np.ndarray,
    uv_faces: np.ndarray,
    vert_colors: np.ndarray,
    *,
    geom_faces: np.ndarray | None = None,
    size: int = 2048,
) -> np.ndarray:
    """将顶点色光栅化到 UV 纹理（float32 RGB 0–1）。"""
    tex = np.zeros((size, size, 3), dtype=np.float32)
    weight = np.zeros((size, size), dtype=np.float32)
    geom_faces = geom_faces if geom_faces is not None else uv_faces

    uv_px = np.zeros_like(uvs, dtype=np.float32)
    uv_px[:, 0] = np.clip(uvs[:, 0], 0.0, 1.0) * (size - 1)
    uv_px[:, 1] = (1.0 - np.clip(uvs[:, 1], 0.0, 1.0)) * (size - 1)

    for face_i in range(len(uv_faces)):
        vi_uv = uv_faces[face_i]
        vi_geom = geom_faces[face_i]
        tri = uv_px[vi_uv]
        cols = vert_colors[vi_geom]
        min_x = max(0, int(np.floor(tri[:, 0].min())))
        max_x = min(size - 1, int(np.ceil(tri[:, 0].max())))
        min_y = max(0, int(np.floor(tri[:, 1].min())))
        max_y = min(size - 1, int(np.ceil(tri[:, 1].max())))
        for y in range(min_y, max_y + 1):
            py = float(y) + 0.5
            for x in range(min_x, max_x + 1):
                px = float(x) + 0.5
                bc = _barycentric(px, py, tri)
                if bc is None:
                    continue
                w0, w1, w2 = bc
                color = cols[0] * w0 + cols[1] * w1 + cols[2] * w2
                w = w0 + w1 + w2
                tex[y, x] += color * w
                weight[y, x] += w

    mask = weight > 1e-6
    tex[mask] /= weight[mask, None]

    holes = (~mask).astype(np.uint8) * 255
    if holes.any() and mask.any():
        fill = (np.clip(tex, 0, 1) * 255).astype(np.uint8)
        fill = cv2.inpaint(fill, holes, 3, cv2.INPAINT_TELEA)
        tex = fill.astype(np.float32) / 255.0

    return np.clip(tex, 0.0, 1.0)


def save_texture_png(texture: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.clip(texture, 0, 1) * 255).astype(np.uint8), mode="RGB").save(path)


def bake_diffuse_from_gaussian_ply(
    ply_path: Path,
    mesh_verts: np.ndarray,
    faces: np.ndarray,
    lhm_root: Path,
    output_png: Path,
    *,
    texture_size: int = 2048,
) -> Path:
    """3DGS PLY → SMPL-X UV diffuse 贴图。"""
    gaussian_xyz, gaussian_rgb = read_gaussian_ply_rgb(ply_path)
    vert_colors = sample_colors_at_vertices(mesh_verts, gaussian_xyz, gaussian_rgb)
    uvs, uv_faces = load_smplx_uv(lhm_root, faces, mesh_verts=mesh_verts)
    if len(uvs) != len(mesh_verts):
        logger.warning(
            "UV 顶点数 (%d) 与网格 (%d) 不一致，尝试按面索引映射",
            len(uvs),
            len(mesh_verts),
        )
    texture = bake_vertex_colors_to_texture(
        uvs, uv_faces, vert_colors, geom_faces=faces, size=texture_size
    )
    save_texture_png(texture, output_png)
    logger.info("UV 贴图已烘焙: %s (%dpx)", output_png.name, texture_size)
    return output_png
