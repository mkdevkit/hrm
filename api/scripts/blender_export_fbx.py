"""Blender headless：OBJ + SMPL-X 骨骼权重 → 蒙皮 FBX。

用法:
  blender --background --python blender_export_fbx.py -- \\
    <mesh.obj> <skeleton.json> <lbs_weights.npz> <output.fbx> [diffuse.png|-] [subdiv_levels]
"""

from __future__ import annotations

import ast
import array
import json
import struct
import subprocess
import sys
import traceback
import zipfile
from pathlib import Path

import bpy


def _ensure_numpy() -> None:
    """Blender 自带 Python 与 io_scene_fbx 插件均依赖 numpy（apt 包通常未内置）。"""
    try:
        import numpy as np  # noqa: F401

        return
    except ModuleNotFoundError:
        pass

    ver = f"{sys.version_info.major}.{sys.version_info.minor}"
    for path in (
        "/usr/lib/python3/dist-packages",
        f"/usr/local/lib/python{ver}/dist-packages",
        f"/usr/lib/python{ver}/site-packages",
    ):
        if Path(path).is_dir() and path not in sys.path:
            sys.path.insert(0, path)

    try:
        import numpy as np  # noqa: F401

        print("[HRM] 已从系统路径加载 numpy", file=sys.stderr)
        return
    except ModuleNotFoundError:
        pass

    print("[HRM] Blender 缺少 numpy，正在用 pip 安装...", file=sys.stderr)
    try:
        subprocess.run(
            [sys.executable, "-m", "ensurepip", "--user"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "numpy"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        import numpy as np  # noqa: F401

        print("[HRM] numpy 安装完成", file=sys.stderr)
        return
    except Exception as exc:
        raise RuntimeError(
            "Blender FBX 插件需要 numpy。请执行: "
            "sudo apt install -y python3-numpy "
            "或 blender --background --python-use-system-env --python-expr "
            "\"import subprocess,sys; subprocess.check_call([sys.executable,'-m','pip','install','numpy'])\""
        ) from exc


def _parse_args() -> tuple[Path, Path, Path, Path, Path | None, int]:
    argv = sys.argv
    if "--" not in argv:
        raise SystemExit("用法: blender --background --python blender_export_fbx.py -- <args>")
    args = argv[argv.index("--") + 1 :]
    if len(args) < 4:
        raise SystemExit(
            "至少需要 4 个参数: obj skeleton.json weights.npz output.fbx "
            "[diffuse.png|-] [subdiv_levels]"
        )
    texture_path: Path | None = None
    if len(args) > 4 and args[4] not in ("", "-"):
        texture_path = Path(args[4])
    subdiv = int(args[5]) if len(args) > 5 else 0
    return Path(args[0]), Path(args[1]), Path(args[2]), Path(args[3]), texture_path, subdiv


def _ensure_fbx_exporter() -> None:
    """headless 工厂默认场景可能未启用 FBX 插件。"""
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
    raise RuntimeError(
        "FBX 导出插件不可用。请安装带 io_scene_fbx 的 Blender，"
        "或在 GUI 中启用 Edit → Preferences → Add-ons → FBX。"
    )


def _import_obj(obj_path: Path) -> bpy.types.Object:
    before = {o.name for o in bpy.data.objects}
    if hasattr(bpy.ops.wm, "obj_import"):
        bpy.ops.wm.obj_import(filepath=str(obj_path))
    else:
        bpy.ops.import_scene.obj(filepath=str(obj_path))

    meshes = [o for o in bpy.data.objects if o.name not in before and o.type == "MESH"]
    if not meshes:
        meshes = [o for o in bpy.context.selected_objects if o.type == "MESH"]
    if not meshes:
        meshes = [o for o in bpy.data.objects if o.type == "MESH"]
    if not meshes:
        raise RuntimeError(f"OBJ 未包含网格: {obj_path}")
    return meshes[0]


def _create_armature(joint_names: list[str], parents: list[int], joints: list[list[float]]) -> bpy.types.Object:
    arm_data = bpy.data.armatures.new("SMPLX_Armature")
    arm_obj = bpy.data.objects.new("SMPLX_Armature", arm_data)
    bpy.context.collection.objects.link(arm_obj)
    bpy.context.view_layer.objects.active = arm_obj
    arm_obj.select_set(True)

    bpy.ops.object.mode_set(mode="EDIT")
    edit_bones = arm_data.edit_bones
    bone_objs: dict[str, bpy.types.EditBone] = {}

    for i, name in enumerate(joint_names):
        head = joints[i]
        bone = edit_bones.new(name)
        bone.head = (head[0], head[1], head[2])
        bone_objs[name] = bone

    for i, name in enumerate(joint_names):
        bone = bone_objs[name]
        parent_idx = parents[i] if i < len(parents) else -1
        if parent_idx >= 0 and parent_idx < len(joint_names):
            bone.parent = bone_objs[joint_names[parent_idx]]

        children = [c for c, p in enumerate(parents) if p == i and c < len(joints)]
        head = joints[i]
        if children:
            cx = sum(joints[c][0] for c in children) / len(children)
            cy = sum(joints[c][1] for c in children) / len(children)
            cz = sum(joints[c][2] for c in children) / len(children)
            bone.tail = (cx, cy, cz)
        elif parent_idx >= 0:
            ph = joints[parent_idx]
            direction = (head[0] - ph[0], head[1] - ph[1], head[2] - ph[2])
            length = (direction[0] ** 2 + direction[1] ** 2 + direction[2] ** 2) ** 0.5
            if length > 1e-6:
                scale = 0.05 / length
                bone.tail = (head[0] + direction[0] * scale, head[1] + direction[1] * scale, head[2] + direction[2] * scale)
            else:
                bone.tail = (head[0], head[1] + 0.05, head[2])
        else:
            bone.tail = (head[0], head[1] + 0.05, head[2])

        # Blender 拒绝零长度骨骼
        dx = bone.tail[0] - bone.head[0]
        dy = bone.tail[1] - bone.head[1]
        dz = bone.tail[2] - bone.head[2]
        if (dx * dx + dy * dy + dz * dz) < 1e-10:
            bone.tail = (head[0], head[1] + 0.01, head[2])

    bpy.ops.object.mode_set(mode="OBJECT")
    arm_obj.select_set(False)
    return arm_obj


def _load_npy_array(data: bytes) -> tuple[tuple[int, ...], array.array]:
    """解析 .npy（仅支持 float32/float64，无 numpy 依赖）。"""
    if len(data) < 10 or data[:6] != b"\x93NUMPY":
        raise RuntimeError("无效的 .npy 文件")

    major = data[6]
    if major == 1:
        header_len = struct.unpack("<H", data[8:10])[0]
        header_start = 10
    elif major == 2:
        header_len = struct.unpack("<I", data[8:12])[0]
        header_start = 12
    else:
        raise RuntimeError(f"不支持的 .npy 版本: {major}")

    header = ast.literal_eval(data[header_start : header_start + header_len].decode("latin1").strip())
    shape = tuple(int(x) for x in header["shape"])
    descr = header["descr"]
    offset = header_start + header_len
    if major == 1:
        rem = offset % 16
        if rem:
            offset += 16 - rem

    if descr == "<f4":
        arr = array.array("f")
        arr.frombytes(data[offset:])
    elif descr == "<f8":
        arr = array.array("d")
        arr.frombytes(data[offset:])
    else:
        raise RuntimeError(f"不支持的 .npy dtype: {descr!r}")

    expected = 1
    for dim in shape:
        expected *= dim
    if len(arr) != expected:
        raise RuntimeError(f".npy 元素数 ({len(arr)}) 与 shape {shape} 不一致")
    return shape, arr


def _load_weights_npz(weights_path: Path) -> tuple[int, int, array.array]:
    """从 .npz 读取 weights 数组（Blender 自带 Python 通常无 numpy）。"""
    with zipfile.ZipFile(weights_path) as zf:
        npy_name = next((n for n in zf.namelist() if n.endswith("weights.npy")), None)
        if npy_name is None:
            raise RuntimeError(f"npz 中未找到 weights.npy: {weights_path}")
        shape, arr = _load_npy_array(zf.read(npy_name))

    if len(shape) != 2:
        raise RuntimeError(f"weights 应为二维数组，实际 shape={shape}")
    rows, cols = shape
    return rows, cols, arr


def _assign_vertex_groups(
    mesh_obj: bpy.types.Object,
    joint_names: list[str],
    rows: int,
    cols: int,
    flat_weights: array.array,
) -> None:
    vert_count = len(mesh_obj.data.vertices)
    if rows != vert_count:
        raise RuntimeError(f"权重顶点数 ({rows}) 与 OBJ 网格 ({vert_count}) 不一致")

    for name in joint_names:
        if name not in mesh_obj.vertex_groups:
            mesh_obj.vertex_groups.new(name=name)

    for vi in range(rows):
        base = vi * cols
        nz: list[tuple[int, float]] = []
        for ji in range(cols):
            val = float(flat_weights[base + ji])
            if val > 1e-4:
                nz.append((ji, val))
        if not nz:
            continue
        total = sum(v for _, v in nz)
        if total <= 0:
            continue
        for ji, val in nz:
            mesh_obj.vertex_groups[joint_names[ji]].add([vi], val / total, "ADD")


def _mesh_vertex_coords(mesh_obj: bpy.types.Object) -> list[tuple[float, float, float]]:
    return [(v.co.x, v.co.y, v.co.z) for v in mesh_obj.data.vertices]


def _estimate_joint_rest_positions(
    verts: list[tuple[float, float, float]],
    joint_count: int,
    rows: int,
    cols: int,
    flat_weights: array.array,
) -> list[list[float]]:
    """旧 skeleton.json 无 joint_rest_positions 时，用 LBS 权重对顶点加权估计。"""
    joints = [[0.0, 0.0, 0.0] for _ in range(joint_count)]
    for ji in range(min(joint_count, cols)):
        sx = sy = sz = 0.0
        total = 0.0
        for vi in range(rows):
            w = float(flat_weights[vi * cols + ji])
            if w <= 1e-6:
                continue
            vx, vy, vz = verts[vi]
            sx += vx * w
            sy += vy * w
            sz += vz * w
            total += w
        if total > 1e-6:
            joints[ji] = [sx / total, sy / total, sz / total]
    return joints


def _resolve_joint_rest_positions(
    skel: dict,
    joint_names: list[str],
    mesh_obj: bpy.types.Object,
    rows: int,
    cols: int,
    flat_weights: array.array,
) -> list[list[float]]:
    joints: list[list[float]] = skel.get("joint_rest_positions") or []
    if len(joints) == len(joint_names):
        return joints

    print(
        "[HRM] skeleton.json 无 joint_rest_positions，"
        "从 OBJ 顶点 + LBS 权重估算关节位置",
        file=sys.stderr,
    )
    verts = _mesh_vertex_coords(mesh_obj)
    return _estimate_joint_rest_positions(verts, len(joint_names), rows, cols, flat_weights)


def _parent_mesh(mesh_obj: bpy.types.Object, arm_obj: bpy.types.Object) -> None:
    for mod in list(mesh_obj.modifiers):
        if mod.type == "ARMATURE":
            mesh_obj.modifiers.remove(mod)
    mod = mesh_obj.modifiers.new(name="Armature", type="ARMATURE")
    mod.object = arm_obj
    mesh_obj.parent = arm_obj
    mesh_obj.parent_type = "OBJECT"


def _apply_subdivision(mesh_obj: bpy.types.Object, levels: int) -> None:
    """在绑 Armature 之前细分，避免 modifier 顺序警告。"""
    if levels <= 0:
        return
    for mod in list(mesh_obj.modifiers):
        mesh_obj.modifiers.remove(mod)
    bpy.context.view_layer.objects.active = mesh_obj
    mesh_obj.select_set(True)
    mod = mesh_obj.modifiers.new(name="Subsurf", type="SUBSURF")
    mod.levels = levels
    mod.render_levels = levels
    bpy.ops.object.modifier_apply(modifier=mod.name)
    mesh_obj.select_set(False)
    print(f"[HRM] 细分曲面 Level {levels} 已应用", file=sys.stderr)


def _prepare_mesh_for_export(mesh_obj: bpy.types.Object, texture_path: Path | None) -> None:
    """平滑着色 + 贴图或默认肤色材质。"""
    import math

    mesh = mesh_obj.data
    for poly in mesh.polygons:
        poly.use_smooth = True
    if hasattr(mesh, "use_auto_smooth"):
        mesh.use_auto_smooth = True
    if hasattr(mesh, "auto_smooth_angle"):
        mesh.auto_smooth_angle = math.radians(180)
    mesh.update()

    mat = bpy.data.materials.new(name="HRM_Body")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    bsdf = nodes.get("Principled BSDF")
    if bsdf is None:
        mesh.materials.clear()
        mesh.materials.append(mat)
        return

    if texture_path and texture_path.is_file():
        img = bpy.data.images.load(str(texture_path.resolve()))
        tex_node = nodes.new("ShaderNodeTexImage")
        tex_node.image = img
        tex_node.location = (-300, 300)
        links.new(tex_node.outputs["Color"], bsdf.inputs["Base Color"])
        bsdf.inputs["Roughness"].default_value = 0.55
        print(f"[HRM] 已绑定 diffuse 贴图: {texture_path.name}", file=sys.stderr)
    else:
        bsdf.inputs["Base Color"].default_value = (0.82, 0.68, 0.58, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.52
        for key in ("Subsurface Weight", "Subsurface"):
            if key in bsdf.inputs:
                bsdf.inputs[key].default_value = 0.12
                break

    mesh.materials.clear()
    mesh.materials.append(mat)


def _export_fbx(fbx_path: Path, *, embed_textures: bool) -> None:
    _ensure_numpy()
    _ensure_fbx_exporter()
    fbx_path = fbx_path.resolve()
    fbx_path.parent.mkdir(parents=True, exist_ok=True)
    result = bpy.ops.export_scene.fbx(
        filepath=str(fbx_path),
        use_selection=False,
        object_types={"ARMATURE", "MESH"},
        add_leaf_bones=False,
        bake_anim=False,
        armature_nodetype="NULL",
        mesh_smooth_type="EDGE",
        path_mode="COPY",
        embed_textures=embed_textures,
    )
    if result != {"FINISHED"}:
        raise RuntimeError(f"export_scene.fbx 返回 {result!r}")
    if not fbx_path.is_file():
        raise RuntimeError(f"FBX 文件未写入: {fbx_path}")


def main() -> None:
    obj_path, skel_path, weights_path, fbx_path, texture_path, subdiv = _parse_args()

    skel = json.loads(skel_path.read_text(encoding="utf-8"))
    joint_names: list[str] = skel["joint_names"]
    parents: list[int] = skel["parents"]
    rows, cols, flat_weights = _load_weights_npz(weights_path)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    _ensure_fbx_exporter()

    mesh_obj = _import_obj(obj_path)
    _assign_vertex_groups(mesh_obj, joint_names, rows, cols, flat_weights)
    _apply_subdivision(mesh_obj, subdiv)
    joints = _resolve_joint_rest_positions(
        skel, joint_names, mesh_obj, rows, cols, flat_weights
    )
    arm_obj = _create_armature(joint_names, parents, joints)
    _parent_mesh(mesh_obj, arm_obj)
    _prepare_mesh_for_export(mesh_obj, texture_path)
    _export_fbx(fbx_path, embed_textures=bool(texture_path and texture_path.is_file()))
    print(f"[HRM] FBX exported: {fbx_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[HRM] FBX export failed: {exc}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
