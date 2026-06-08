"""Blender headless：OBJ + SMPL-X 骨骼权重 → 蒙皮 FBX。

用法:
  blender --background --python blender_export_fbx.py -- \\
    <mesh.obj> <skeleton.json> <lbs_weights.npz> <output.fbx>
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import bpy


def _parse_args() -> tuple[Path, Path, Path, Path]:
    argv = sys.argv
    if "--" not in argv:
        raise SystemExit("用法: blender --background --python blender_export_fbx.py -- <args>")
    args = argv[argv.index("--") + 1 :]
    if len(args) != 4:
        raise SystemExit("需要 4 个参数: obj skeleton.json weights.npz output.fbx")
    return Path(args[0]), Path(args[1]), Path(args[2]), Path(args[3])


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


def _assign_vertex_groups(mesh_obj: bpy.types.Object, joint_names: list[str], weights) -> None:
    import numpy as np

    w = np.asarray(weights, dtype=np.float64)
    vert_count = len(mesh_obj.data.vertices)
    if w.shape[0] != vert_count:
        raise RuntimeError(
            f"权重顶点数 ({w.shape[0]}) 与 OBJ 网格 ({vert_count}) 不一致"
        )

    for name in joint_names:
        if name not in mesh_obj.vertex_groups:
            mesh_obj.vertex_groups.new(name=name)

    for vi in range(w.shape[0]):
        row = w[vi]
        nz = np.where(row > 1e-4)[0]
        if len(nz) == 0:
            continue
        vals = row[nz]
        total = float(vals.sum())
        if total <= 0:
            continue
        for ji, val in zip(nz, vals):
            mesh_obj.vertex_groups[joint_names[int(ji)]].add([vi], float(val / total), "ADD")


def _parent_mesh(mesh_obj: bpy.types.Object, arm_obj: bpy.types.Object) -> None:
    for mod in list(mesh_obj.modifiers):
        if mod.type == "ARMATURE":
            mesh_obj.modifiers.remove(mod)
    mod = mesh_obj.modifiers.new(name="Armature", type="ARMATURE")
    mod.object = arm_obj
    mesh_obj.parent = arm_obj
    mesh_obj.parent_type = "OBJECT"


def _export_fbx(fbx_path: Path) -> None:
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
        mesh_smooth_type="FACE",
    )
    if result != {"FINISHED"}:
        raise RuntimeError(f"export_scene.fbx 返回 {result!r}")
    if not fbx_path.is_file():
        raise RuntimeError(f"FBX 文件未写入: {fbx_path}")


def main() -> None:
    obj_path, skel_path, weights_path, fbx_path = _parse_args()

    import numpy as np

    skel = json.loads(skel_path.read_text(encoding="utf-8"))
    joint_names: list[str] = skel["joint_names"]
    parents: list[int] = skel["parents"]
    joints: list[list[float]] = skel.get("joint_rest_positions") or []
    if len(joints) != len(joint_names):
        raise RuntimeError(
            "skeleton.json 缺少 joint_rest_positions（version<2 的旧任务需重新导出蒙皮网格）"
        )

    weights = np.load(weights_path)["weights"]

    bpy.ops.wm.read_factory_settings(use_empty=True)
    _ensure_fbx_exporter()

    mesh_obj = _import_obj(obj_path)
    arm_obj = _create_armature(joint_names, parents, joints)
    _assign_vertex_groups(mesh_obj, joint_names, weights)
    _parent_mesh(mesh_obj, arm_obj)
    _export_fbx(fbx_path)
    print(f"[HRM] FBX exported: {fbx_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[HRM] FBX export failed: {exc}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
