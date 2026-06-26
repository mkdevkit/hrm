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

## 蒙皮网格与 3DGS 对齐（重要）

3D 重建时会写入 `gs_anchors.npy`（与 PLY 高斯**逐点索引对齐**的 `neutral_coords`）。
**仅 rebake 不能生成该文件**；若 FBX 与 3DGS 严重错位，请用 Web/API **重新跑一次 3D 重建**（勾选导出蒙皮网格），再 rebake。

```bash
python scripts/rebake_avatar_fbx.py data/avatars/<avatar_id>/output
```
