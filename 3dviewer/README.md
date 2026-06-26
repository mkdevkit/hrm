# 3D 模型查看器

本地 Web 工具，用 **Three.js** 预览 HRM 导出的 **FBX** 蒙皮网格，以及 GLB/GLTF、OBJ 等常见格式。若模型含动作片段，可在工具栏 **动作** 下拉框中切换。

## 快速开始

```bash
cd 3dviewer
npm install
npm run dev
```

浏览器打开：**http://localhost:5175**

## 使用方式

| 方式 | 说明 |
|------|------|
| **拖放** | 将 `.fbx` / `.glb` / `.gltf` / `.obj` 拖进页面 |
| **打开文件** | 点击工具栏「打开模型」 |
| **URL 参数** | `?model=https://example.com/avatar.fbx` |
| **HRM API** | `?avatar={avatar_id}&api=http://localhost:8000&format=fbx` |

HRM 蒙皮 FBX 示例：

```
http://localhost:5175/?avatar=YOUR_AVATAR_ID&api=http://localhost:8000&format=fbx
```

从 HRM API 跨域加载时，请在 `api/.env` 的 `CORS_ORIGINS` 中加入查看器地址：

```env
CORS_ORIGINS=http://localhost:3000,http://localhost:5174,http://localhost:5175
```

## 动作

- FBX / GLB 若包含 **AnimationClip**，工具栏会出现 **动作** 下拉框与 **播放/暂停** 按钮。
- HRM 当前导出的 T-pose 蒙皮 FBX **通常不含动作**；带动作的 GLB/FBX 可直接加载预览。
- 切换动作时会交叉淡入淡出（0.25s）。

## 操作

- **左键拖动**：旋转
- **右键拖动**：平移
- **滚轮**：缩放

界面支持 **中文 / English**，默认跟随系统语言；右上角按钮可切换。

## 构建静态版

```bash
npm run build
npm run preview   # 预览 dist/
```

## 技术栈

- [Vite](https://vitejs.dev/)
- [Three.js](https://threejs.org/) — FBXLoader / GLTFLoader / OBJLoader / AnimationMixer

## 说明

- FBX 依赖 `fflate`（Three.js FBXLoader 解压用）。
- OBJ 仅加载几何体，不含骨骼动作。
- 与 `3dgs/`（高斯 PLY 查看器，端口 5174）互补：PLY 用 3dgs，FBX/GLB 用本工具。
