# HRM 3DGS 查看器

本地 Web 工具，用于预览 LHM++ / HRM 导出的 **3D Gaussian Splatting** `.ply` 文件（普通 MeshLab 会显示为黑点，需 splat 渲染器）。

## 快速开始

```bash
cd 3dgs
npm install
npm run dev
```

浏览器打开：**http://localhost:5174**

## 使用方式

| 方式 | 说明 |
|------|------|
| **拖放** | 将 `avatar.ply` 拖进页面 |
| **打开文件** | 点击工具栏「打开 PLY」 |
| **URL 参数** | `http://localhost:5174/?ply=/path/to/avatar.ply` |
| **HRM API** | `?ply=http://localhost:8000/api/v1/avatars/{id}/model` |

从 HRM API 跨域加载时，请在 `api/.env` 的 `CORS_ORIGINS` 中加入查看器地址，例如：

```env
CORS_ORIGINS=http://localhost:3000,http://localhost:5174
```

## 操作

- **左键拖动**：旋转
- **右键拖动**：平移
- **滚轮**：缩放

界面支持 **中文 / English**，默认跟随系统语言（非中文则显示英文）；右上角按钮可切换，选择会保存在浏览器本地。

## 构建静态版

```bash
npm run build
npm run preview   # 预览 dist/
```

`dist/` 可部署到任意静态文件服务器。

## 技术栈

- [Vite](https://vitejs.dev/)
- [@mkkellogg/gaussian-splats-3d](https://github.com/mkkellogg/GaussianSplats3D) — 浏览器端 3DGS 光栅化

## 说明

- 支持标准 3DGS PLY（含 `scale_*`、`opacity`、`f_dc_*` 等字段），与 LHM++ `to_gs_ply.py` 输出兼容。
- Mock 模式生成的占位 PLY **无法**正常预览，请使用真实重建结果。
