"""Blender headless：带 MTL/贴图的 OBJ → 静态 FBX（SuGaR 网格导出用，无骨骼）。

用法:
  blender --background --python blender_export_static_fbx.py -- \\
    <mesh.obj> <output.fbx>
"""

from __future__ import annotations

import subprocess
import sys
import traceback
from pathlib import Path

import bpy


def _ensure_numpy() -> None:
    try:
        import numpy as np  # noqa: F401

        return
    except ModuleNotFoundError:
        pass
    ver = f"{sys.version_info.major}.{sys.version_info.minor}"
    for path in (
        "/usr/lib/python3/dist-packages",
        f"/usr/local/lib/python{ver}/dist-packages",
    ):
        if Path(path).is_dir() and path not in sys.path:
            sys.path.insert(0, path)
    try:
        import numpy as np  # noqa: F401

        return
    except ModuleNotFoundError:
        raise RuntimeError("Blender FBX 插件需要 numpy，请 apt install python3-numpy") from None


def _ensure_fbx_exporter() -> None:
    if hasattr(bpy.ops.export_scene, "fbx"):
        return
    import addon_utils

    for mod in ("io_scene_fbx", "blender_io_scene_fbx"):
        try:
            addon_utils.enable(mod, default_set=True, persistent=True)
        except Exception:
            pass
        if hasattr(bpy.ops.export_scene, "fbx"):
            return
    raise RuntimeError("FBX 导出插件不可用")


def _parse_args() -> tuple[Path, Path]:
    argv = sys.argv
    if "--" not in argv:
        raise SystemExit("用法: blender --background --python blender_export_static_fbx.py -- <obj> <fbx>")
    args = argv[argv.index("--") + 1 :]
    if len(args) < 2:
        raise SystemExit("需要 2 个参数: input.obj output.fbx")
    return Path(args[0]), Path(args[1])


def _import_obj(obj_path: Path) -> bpy.types.Object:
    before = {o.name for o in bpy.data.objects}
    if hasattr(bpy.ops.wm, "obj_import"):
        bpy.ops.wm.obj_import(filepath=str(obj_path))
    else:
        bpy.ops.import_scene.obj(filepath=str(obj_path))
    meshes = [o for o in bpy.data.objects if o.name not in before and o.type == "MESH"]
    if not meshes:
        meshes = [o for o in bpy.data.objects if o.type == "MESH"]
    if not meshes:
        raise RuntimeError(f"OBJ 未包含网格: {obj_path}")
    return meshes[0]


def _export_fbx(fbx_path: Path) -> None:
    fbx_path = fbx_path.resolve()
    fbx_path.parent.mkdir(parents=True, exist_ok=True)
    result = bpy.ops.export_scene.fbx(
        filepath=str(fbx_path),
        use_selection=False,
        object_types={"MESH"},
        add_leaf_bones=False,
        bake_anim=False,
        path_mode="COPY",
        embed_textures=True,
    )
    if result != {"FINISHED"}:
        raise RuntimeError(f"export_scene.fbx 返回 {result!r}")
    if not fbx_path.is_file():
        raise RuntimeError(f"FBX 未写入: {fbx_path}")


def main() -> None:
    obj_path, fbx_path = _parse_args()
    _ensure_numpy()
    _ensure_fbx_exporter()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    mesh_obj = _import_obj(obj_path)
    for poly in mesh_obj.data.polygons:
        poly.use_smooth = True
    _export_fbx(fbx_path)
    print(f"[HRM] 静态 FBX 已导出: {fbx_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[HRM] 静态 FBX 导出失败: {exc}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
