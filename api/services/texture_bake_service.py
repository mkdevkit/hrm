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
    k_neighbors: int = 24,
) -> np.ndarray:
    """用最近邻高斯颜色插值得到网格顶点色（无锚点时的回退）。"""
    from scipy.spatial import cKDTree

    k = min(k_neighbors, len(gaussian_xyz))
    dists, indices = cKDTree(gaussian_xyz).query(mesh_verts, k=k)
    if k == 1:
        dists = dists[:, None]
        indices = indices[:, None]
    weights = 1.0 / (dists + 1e-6)
    weights /= weights.sum(axis=1, keepdims=True)
    return (gaussian_rgb[indices] * weights[..., None]).sum(axis=1).astype(np.float32)


def sample_colors_at_vertices_via_anchors(
    mesh_verts: np.ndarray,
    gaussian_rgb: np.ndarray,
    anchor_xyz: np.ndarray,
    *,
    k_neighbors: int = 24,
) -> np.ndarray:
    """与位移场相同的锚点 kNN 插值，保证颜色与高斯索引一一对应。"""
    from scipy.spatial import cKDTree

    if len(gaussian_rgb) != len(anchor_xyz):
        raise ValueError(
            f"高斯颜色数 ({len(gaussian_rgb)}) 与锚点数 ({len(anchor_xyz)}) 不一致"
        )
    k = min(k_neighbors, len(anchor_xyz))
    dists, indices = cKDTree(anchor_xyz).query(mesh_verts, k=k)
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


def bundled_smplx_uv_path() -> Path:
    return Path(__file__).resolve().parents[1] / "assets" / "smplx_uv.obj"


def _priority_uv_paths(human_model: Path) -> list[Path]:
    """用户显式提供的 UV 路径（优先于 LHM++ 内置 bm_x）。"""
    from config import settings

    paths: list[Path] = []
    if settings.smplx_uv_obj:
        paths.append(Path(settings.smplx_uv_obj))
    paths.append(bundled_smplx_uv_path())
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in paths:
        resolved = path.resolve()
        if resolved in seen or not path.is_file():
            continue
        seen.add(resolved)
        unique.append(path)
    return unique


def _candidate_uv_paths(human_model: Path) -> list[Path]:
    from config import settings

    candidates: list[Path] = []
    if settings.smplx_uv_obj:
        candidates.append(Path(settings.smplx_uv_obj))
    candidates.append(bundled_smplx_uv_path())

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


def _uv_indices_valid(
    uvs: np.ndarray,
    uv_faces: np.ndarray,
    faces: np.ndarray,
    mesh_verts: np.ndarray | None,
) -> bool:
    if len(uv_faces) != len(faces):
        return False
    if len(uvs) == 0:
        return False
    max_uv = int(uv_faces.max()) if len(uv_faces) else 0
    max_geom = int(faces.max()) if len(faces) else 0
    if max_uv >= len(uvs):
        return False
    if mesh_verts is not None and max_geom >= len(mesh_verts):
        return False
    return True


def _try_load_uv_from_path(
    path: Path,
    faces: np.ndarray,
    *,
    mesh_verts: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    if path.suffix.lower() in (".pkl", ".pickle", ".npz"):
        result = _load_uv_from_pkl_npz(path, faces)
        if result is not None and _uv_indices_valid(result[0], result[1], faces, mesh_verts):
            return result
        return None

    uvs, uv_faces = parse_obj_uv(path)
    if uvs is not None and uv_faces is not None:
        if _uv_indices_valid(uvs, uv_faces, faces, mesh_verts):
            return uvs, uv_faces
        if mesh_verts is not None and len(uvs) == len(mesh_verts):
            per_vertex = uvs, faces.copy()
            if _uv_indices_valid(*per_vertex, faces, mesh_verts):
                return per_vertex

    result = _load_uv_from_trimesh(path, faces)
    if result is not None and _uv_indices_valid(result[0], result[1], faces, mesh_verts):
        return result
    return None


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

    for path in _priority_uv_paths(human_model):
        result = _try_load_uv_from_path(path, faces, mesh_verts=mesh_verts)
        if result is not None:
            logger.info("SMPL-X UV 来自 %s", path)
            return result

    bundled = bundled_smplx_uv_path()
    if not bundled.is_file():
        logger.info(
            "未找到 api/assets/smplx_uv.obj（可将官方 smplx_uv.obj 放到 %s）",
            bundled,
        )

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
        if result is not None and _uv_indices_valid(result[0], result[1], faces, mesh_verts):
            return result
        if result is not None:
            logger.warning("SMPL_Layer bm_x UV 索引无效，继续尝试其他来源")
    except Exception as exc:
        logger.warning("从 SMPL_Layer 读取 UV 失败: %s", exc)

    for path in _candidate_uv_paths(human_model):
        if path.resolve() in {p.resolve() for p in _priority_uv_paths(human_model)}:
            continue
        result = _try_load_uv_from_path(path, faces, mesh_verts=mesh_verts)
        if result is not None:
            logger.info("SMPL-X UV 来自 %s", path)
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


def sample_uv_at_mesh_points(
    mesh_verts: np.ndarray,
    faces: np.ndarray,
    uvs: np.ndarray,
    uv_faces: np.ndarray,
    points: np.ndarray,
) -> np.ndarray:
    """将 3D 点投影到网格表面，再插值得到 UV（用于逐高斯烘焙）。"""
    import trimesh
    from trimesh.triangles import points_to_barycentric

    mesh = trimesh.Trimesh(vertices=mesh_verts, faces=faces, process=False)
    closest, _dist, tri_ids = trimesh.proximity.closest_point(mesh, points)
    tri_verts = mesh.vertices[mesh.faces[tri_ids]]
    bary = points_to_barycentric(tri_verts, closest)
    geom_idx = faces[tri_ids]
    uv_idx = uv_faces[tri_ids]
    uv_corners = uvs[uv_idx]
    return (uv_corners * bary[:, :, None]).sum(axis=1).astype(np.float32)


def gaussian_splat_colors_to_texture(
    sampled_uv: np.ndarray,
    colors: np.ndarray,
    *,
    size: int = 2048,
    splat_radius: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """将逐点颜色 splat 到 UV 贴图（高斯核混合，覆盖比顶点 splat 更密）。"""
    tex = np.zeros((size, size, 3), dtype=np.float32)
    weight = np.zeros((size, size), dtype=np.float32)
    cx = np.clip(sampled_uv[:, 0], 0.0, 1.0) * (size - 1)
    cy = (1.0 - np.clip(sampled_uv[:, 1], 0.0, 1.0)) * (size - 1)

    r = max(1, int(np.ceil(splat_radius)))
    yy, xx = np.mgrid[-r : r + 1, -r : r + 1]
    disk = (xx.astype(np.float32) ** 2 + yy.astype(np.float32) ** 2) <= splat_radius**2
    offs_y = yy[disk].astype(np.float32)
    offs_x = xx[disk].astype(np.float32)
    sigma = max(0.6, splat_radius * 0.45)
    kern = np.exp(-(offs_x**2 + offs_y**2) / (2.0 * sigma**2)).astype(np.float32)

    for dy, dx, kw in zip(offs_y, offs_x, kern, strict=True):
        xi = np.clip((cx + dx).astype(np.int32), 0, size - 1)
        yi = np.clip((cy + dy).astype(np.int32), 0, size - 1)
        np.add.at(tex[..., 0], (yi, xi), colors[:, 0] * kw)
        np.add.at(tex[..., 1], (yi, xi), colors[:, 1] * kw)
        np.add.at(tex[..., 2], (yi, xi), colors[:, 2] * kw)
        np.add.at(weight, (yi, xi), kw)

    return tex, weight


def _refine_baked_texture(tex: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """归一化、补洞并轻微平滑，减轻斑块感。"""
    mask = weight > 1e-6
    if mask.any():
        tex[mask] /= weight[mask, None]

    holes = (~mask).astype(np.uint8) * 255
    if holes.any() and mask.any():
        fill = (np.clip(tex, 0, 1) * 255).astype(np.uint8)
        fill = cv2.inpaint(fill, holes, 5, cv2.INPAINT_NS)
        tex = fill.astype(np.float32) / 255.0
        mask = np.any(tex > 1e-4, axis=2)

    if mask.any():
        src = (np.clip(tex, 0, 1) * 255).astype(np.uint8)
        smooth = cv2.bilateralFilter(src, d=5, sigmaColor=18, sigmaSpace=5)
        out = tex.copy()
        out[mask] = smooth[mask].astype(np.float32) / 255.0
        return np.clip(out, 0.0, 1.0)

    return np.clip(tex, 0.0, 1.0)


def bake_gaussian_colors_via_anchors(
    gaussian_rgb: np.ndarray,
    anchor_xyz: np.ndarray,
    mesh_verts: np.ndarray,
    faces: np.ndarray,
    uvs: np.ndarray,
    uv_faces: np.ndarray,
    *,
    size: int = 2048,
    splat_radius: float = 2.0,
) -> np.ndarray:
    """每个 3DGS 高斯按锚点 UV splat，细节优于仅 1 万顶点插值。"""
    logger.info(
        "逐高斯 UV splat: %d 点, %dpx, 半径 %.1fpx",
        len(gaussian_rgb),
        size,
        splat_radius,
    )
    sampled_uv = sample_uv_at_mesh_points(mesh_verts, faces, uvs, uv_faces, anchor_xyz)
    tex, weight = gaussian_splat_colors_to_texture(
        sampled_uv,
        gaussian_rgb,
        size=size,
        splat_radius=splat_radius,
    )
    return _refine_baked_texture(tex, weight)


def _barycentric_batch(
    px: np.ndarray, py: np.ndarray, tri: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    x0, y0 = tri[0]
    x1, y1 = tri[1]
    x2, y2 = tri[2]
    denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    if abs(denom) < 1e-12:
        return None
    w0 = ((y1 - y2) * (px - x2) + (x2 - x1) * (py - y2)) / denom
    w1 = ((y2 - y0) * (px - x2) + (x0 - x2) * (py - y2)) / denom
    w2 = 1.0 - w0 - w1
    return w0, w1, w2


def bake_vertex_colors_to_texture(
    uvs: np.ndarray,
    uv_faces: np.ndarray,
    vert_colors: np.ndarray,
    *,
    geom_faces: np.ndarray | None = None,
    size: int = 2048,
) -> np.ndarray:
    """将顶点色光栅化到 UV 纹理（float32 RGB 0–1）。"""
    n_faces = len(uv_faces)
    logger.info("开始 UV 光栅化: %d 三角面, %d×%d", n_faces, size, size)
    tex = np.zeros((size, size, 3), dtype=np.float32)
    weight = np.zeros((size, size), dtype=np.float32)
    geom_faces = geom_faces if geom_faces is not None else uv_faces

    uv_px = np.zeros_like(uvs, dtype=np.float32)
    uv_px[:, 0] = np.clip(uvs[:, 0], 0.0, 1.0) * (size - 1)
    uv_px[:, 1] = (1.0 - np.clip(uvs[:, 1], 0.0, 1.0)) * (size - 1)

    log_step = max(1, n_faces // 10)
    for face_i in range(n_faces):
        if face_i > 0 and face_i % log_step == 0:
            logger.info("UV 光栅化进度: %d/%d (%.0f%%)", face_i, n_faces, 100.0 * face_i / n_faces)

        vi_uv = uv_faces[face_i]
        vi_geom = geom_faces[face_i]
        tri = uv_px[vi_uv]
        cols = vert_colors[vi_geom]

        min_x = max(0, int(np.floor(tri[:, 0].min())))
        max_x = min(size - 1, int(np.ceil(tri[:, 0].max())))
        min_y = max(0, int(np.floor(tri[:, 1].min())))
        max_y = min(size - 1, int(np.ceil(tri[:, 1].max())))
        if min_x > max_x or min_y > max_y:
            continue

        xs = np.arange(min_x, max_x + 1, dtype=np.float32) + 0.5
        ys = np.arange(min_y, max_y + 1, dtype=np.float32) + 0.5
        gx, gy = np.meshgrid(xs, ys, indexing="xy")
        px_flat = gx.ravel()
        py_flat = gy.ravel()

        bc = _barycentric_batch(px_flat, py_flat, tri)
        if bc is None:
            continue
        w0, w1, w2 = bc
        inside = (w0 >= -1e-4) & (w1 >= -1e-4) & (w2 >= -1e-4)
        if not inside.any():
            continue

        w_sum = w0 + w1 + w2
        color = (
            cols[0] * w0[inside, None]
            + cols[1] * w1[inside, None]
            + cols[2] * w2[inside, None]
        )
        xi = px_flat[inside].astype(np.int32)
        yi = py_flat[inside].astype(np.int32)
        ws = w_sum[inside]

        np.add.at(tex, (yi, xi, 0), color[:, 0] * ws)
        np.add.at(tex, (yi, xi, 1), color[:, 1] * ws)
        np.add.at(tex, (yi, xi, 2), color[:, 2] * ws)
        np.add.at(weight, (yi, xi), ws)

    logger.info("UV 光栅化完成，填充空洞…")
    return _refine_baked_texture(tex, weight)


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
    uvs: np.ndarray | None = None,
    uv_faces: np.ndarray | None = None,
    anchor_xyz: np.ndarray | None = None,
    splat_radius: float | None = None,
) -> Path:
    """3DGS PLY → SMPL-X UV diffuse 贴图。"""
    from config import settings

    if splat_radius is None:
        splat_radius = settings.fbx_texture_splat_radius

    logger.info("读取 3DGS PLY: %s", ply_path.name)
    gaussian_xyz, gaussian_rgb = read_gaussian_ply_rgb(ply_path)
    logger.info("高斯点 %d", len(gaussian_xyz))
    if uvs is None or uv_faces is None:
        uvs, uv_faces = load_smplx_uv(lhm_root, faces, mesh_verts=mesh_verts)
    elif not _uv_indices_valid(uvs, uv_faces, faces, mesh_verts):
        raise ValueError(
            f"预加载 UV 索引无效: uvs={len(uvs)}, uv_faces={len(uv_faces)}, faces={len(faces)}"
        )

    if anchor_xyz is not None and len(anchor_xyz) == len(gaussian_rgb):
        texture = bake_gaussian_colors_via_anchors(
            gaussian_rgb,
            anchor_xyz,
            mesh_verts,
            faces,
            uvs,
            uv_faces,
            size=texture_size,
            splat_radius=splat_radius,
        )
    else:
        logger.warning("锚点不可用，回退顶点色 + 三角面光栅化")
        vert_colors = sample_colors_at_vertices(mesh_verts, gaussian_xyz, gaussian_rgb)
        texture = bake_vertex_colors_to_texture(
            uvs, uv_faces, vert_colors, geom_faces=faces, size=texture_size
        )
    save_texture_png(texture, output_png)
    logger.info("UV 贴图已烘焙: %s (%dpx)", output_png.name, texture_size)
    return output_png
