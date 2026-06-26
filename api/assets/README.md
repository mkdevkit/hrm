# HRM 静态资源

## SMPL-X UV（可选，提升贴图烘焙质量）

LHM++ 的 `human_model_files` 有时不含 UV。若 3DGS 贴图烘焙失败，可从 **SMPL-X 官方下载包** 复制：

```
smplx/smplx_uv.obj  →  api/assets/smplx_uv.obj
```

或在 `api/.env` 中设置：

```env
SMPLX_UV_OBJ=/path/to/smplx_uv.obj
```

未提供官方 UV 时，HRM 会自动使用**柱面 UV 回退**（能烘焙颜色，但接缝与精度不如官方 UV）。

## 已有 avatar 仅重烘焙贴图 / 重导出 FBX

无需重跑 3D 重建，在 `api` 目录执行：

```bash
python scripts/rebake_avatar_fbx.py data/avatars/<avatar_id>/output
```

成功时日志应出现 `SMPL-X UV 来自 .../api/assets/smplx_uv.obj` 与 `UV 贴图已烘焙`，Blender 命令中贴图参数为 `avatar_diffuse.png` 而非 `-`。
