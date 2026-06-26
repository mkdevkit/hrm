"""3DGS PLY → 泊松重建网格 → UV 烘焙 → 静态 FBX（默认 FBX 导出路径）。"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from config import settings

logger = logging.getLogger(__name__)


def _subsample_points(xyz: np.ndarray, max_points: int) -> np.ndarray:
    if len(xyz) <= max_points:
        return xyz
    rng = np.random.default_rng(42)
    idx = rng.choice(len(xyz), max_points, replace=False)
    return xyz[idx]


def poisson_mesh_from_gaussians(
    xyz: np.ndarray,
    *,
    depth: int | None = None,
    density_quantile: float | None = None,
    normal_radius: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """对高斯中心点云做 Poisson 表面重建（与 SuGaR 同类思路，轻量 Open3D 实现）。"""
    import open3d as o3d

    depth = depth if depth is not None else settings.poisson_depth
    density_quantile = (
        density_quantile
        if density_quantile is not None
        else settings.poisson_density_quantile
    )
    normal_radius = (
        normal_radius if normal_radius is not None else settings.poisson_normal_radius
    )

    pts = _subsample_points(xyz.astype(np.float64), settings.poisson_max_points)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=float(normal_radius),
            max_nn=30,
        )
    )
    pcd.orient_normals_consistent_tangent_plane(30)

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd,
        depth=int(depth),
        scale=1.1,
    )
    if len(densities) == 0:
        raise RuntimeError("Poisson 重建未产生有效网格")

    densities = np.asarray(densities)
    keep = densities >= np.quantile(densities, density_quantile)
    mesh = mesh.remove_vertices_by_mask(~keep)
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()

    verts = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.triangles, dtype=np.int32)
    if len(verts) < 100 or len(faces) < 100:
        raise RuntimeError(f"Poisson 网格过小: {len(verts)} verts, {len(faces)} faces")
    logger.info("Poisson 网格: %d verts, %d faces (depth=%d)", len(verts), len(faces), depth)
    return verts, faces


def cylindrical_uv(verts: np.ndarray) -> np.ndarray:
    """人体常用柱面 UV（烘焙用，非专业 unwrap）。"""
    c = verts.mean(axis=0)
    x = verts[:, 0] - c[0]
    z = verts[:, 2] - c[2]
    y = verts[:, 1]
    theta = np.arctan2(x, z)
    u = (theta / (2.0 * np.pi)) + 0.5
    ymin, ymax = float(y.min()), float(y.max())
    v = (y - ymin) / (ymax - ymin + 1e-8)
    return np.stack([np.clip(u, 0.0, 1.0), np.clip(v, 0.0, 1.0)], axis=1).astype(np.float32)


def _write_obj_with_mtl(
    obj_path: Path,
    verts: np.ndarray,
    faces: np.ndarray,
    uvs: np.ndarray,
    mtl_filename: str,
) -> None:
    lines = [f"mtllib {mtl_filename}", "usemtl avatar_material"]
    for v in verts:
        lines.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}")
    for uv in uvs:
        lines.append(f"vt {uv[0]:.6f} {uv[1]:.6f}")
    for f in faces:
        a, b, c = f[0] + 1, f[1] + 1, f[2] + 1
        lines.append(f"f {a}/{a} {b}/{b} {c}/{c}")
    obj_path.write_text("\n".join(lines), encoding="utf-8")


def _write_mtl(mtl_path: Path, texture_filename: str) -> None:
    mtl_path.write_text(
        "\n".join(
            [
                "newmtl avatar_material",
                "Ka 1.0 1.0 1.0",
                "Kd 1.0 1.0 1.0",
                "Ks 0.0 0.0 0.0",
                "d 1.0",
                f"map_Kd {texture_filename}",
            ]
        ),
        encoding="utf-8",
    )


def _load_anchors(output_dir: Path, gaussian_count: int) -> np.ndarray | None:
    sidecar = output_dir / "gs_anchors.npy"
    if not sidecar.is_file():
        return None
    anchors = np.load(sidecar).astype(np.float32)
    if gaussian_count and len(anchors) != gaussian_count:
        logger.warning(
            "gs_anchors (%d) 与 PLY 高斯数 (%d) 不一致，烘焙将回退 kNN",
            len(anchors),
            gaussian_count,
        )
        return None
    return anchors


def export_poisson_fbx_from_gaussian(
    ply_path: Path,
    output_dir: Path,
    *,
    lhm_root: Path | None = None,
) -> dict[str, Any]:
    """泊松重建 + 3DGS 烘焙 + Blender 静态 FBX。"""
    from services.mesh_export_service import read_gaussian_ply_xyz
    from services.texture_bake_service import bake_diffuse_from_gaussian_ply

    output_dir.mkdir(parents=True, exist_ok=True)
    obj_path = output_dir / "avatar.obj"
    mtl_path = output_dir / "avatar.mtl"
    tex_path = output_dir / "avatar_diffuse.png"
    fbx_path = output_dir / "avatar.fbx"

    if settings.mock_mode:
        obj_path.write_text("# mock poisson obj\n", encoding="utf-8")
        fbx_path.write_text("", encoding="utf-8")
        return {
            "mesh_fbx_path": str(fbx_path),
            "mesh_obj_path": str(obj_path),
            "mesh_export_backend": "poisson",
            "mock": True,
        }

    gaussian_xyz = read_gaussian_ply_xyz(ply_path)
    verts, faces = poisson_mesh_from_gaussians(gaussian_xyz)
    uvs = cylindrical_uv(verts)
    uv_faces = faces.copy()

    if settings.fbx_bake_texture:
        root = lhm_root
        if root is None and settings.lhm_root:
            root = Path(settings.lhm_root).resolve()
        if root is None:
            root = (Path(__file__).resolve().parents[2] / "LHM-plusplus").resolve()
        anchor_xyz = _load_anchors(output_dir, len(gaussian_xyz))
        try:
            bake_diffuse_from_gaussian_ply(
                ply_path,
                verts,
                faces,
                root,
                tex_path,
                texture_size=settings.fbx_texture_size,
                uvs=uvs,
                uv_faces=uv_faces,
                anchor_xyz=anchor_xyz,
            )
        except Exception as exc:
            logger.warning("泊松网格 UV 烘焙失败: %s", exc, exc_info=True)
            tex_path = None

    if tex_path and tex_path.is_file():
        _write_mtl(mtl_path, tex_path.name)
        _write_obj_with_mtl(obj_path, verts, faces, uvs, mtl_path.name)
    else:
        from services.mesh_export_service import _write_obj

        _write_obj(obj_path, verts, faces, uvs=uvs, uv_faces=uv_faces)

    from services.blender_static_fbx_export import export_static_fbx

    fbx_ok = export_static_fbx(obj_path, fbx_path, timeout_sec=settings.sugar_blender_timeout_sec)

    return {
        "mesh_fbx_path": str(fbx_path) if fbx_ok else None,
        "mesh_obj_path": str(obj_path),
        "mesh_texture_path": str(tex_path) if tex_path and tex_path.is_file() else None,
        "mesh_export_backend": "poisson",
        "poisson_vertex_count": int(len(verts)),
        "poisson_face_count": int(len(faces)),
        "export_ok": bool(fbx_ok),
    }
