# HRM — LHM++ 人体 3D 重建与动作驱动

基于 [LHM++](https://lingtengqiu.github.io/LHM++/) 的全栈工具：用一组人物图片生成可动画 3D 模型，再通过动作视频或摄像头视频流驱动角色。

- **后端**：FastAPI（`api/`），封装 LHM++ GPU 推理
- **前端**：Next.js（`web/`），单页工作流 UI

---

## 目录

- [架构](#架构)
- [功能一览](#功能一览)
- [环境要求](#环境要求)
- [安装与启动](#安装与启动)
  - [1. 安装 LHM-plusplus](#1-安装-lhm-plusplus)
  - [2. 启动 API 后端](#2-启动-api-后端)
  - [3. 启动 Web 前端](#3-启动-web-前端)
- [Web 界面使用](#web-界面使用)
- [API 使用说明](#api-使用说明)
  - [健康检查](#健康检查)
  - [创建 Avatar（3D 重建）](#创建-avatar3d-重建)
  - [下载模型文件](#下载模型文件)
  - [动作视频驱动](#动作视频驱动)
  - [摄像头流驱动（WebSocket）](#摄像头流驱动websocket)
  - [任务查询](#任务查询)
- [输出文件说明](#输出文件说明)
- [蒙皮网格导出说明](#蒙皮网格导出说明)
- [Mock 模式（无 GPU 联调）](#mock-模式无-gpu-联调)
- [配置项参考](#配置项参考)
- [项目结构](#项目结构)
- [常见问题](#常见问题)
- [参考链接](#参考链接)
- [License](#license)

---

## 架构

```
┌─────────────┐     HTTP/WS      ┌──────────────────┐     Python API     ┌──────────────┐
│  Next.js    │ ◄──────────────► │  FastAPI (api/)  │ ◄────────────────► │  LHM-plusplus │
│  Web 前端   │                  │  任务队列/文件    │                    │  GPU 推理     │
└─────────────┘                  └──────────────────┘                    └──────────────┘
```

数据默认保存在 `api/data/` 下：

```
api/data/
├── avatars/{avatar_id}/   # 每个 Avatar 的输入图、输出模型、动画
├── jobs/                  # 异步任务状态
└── motion_cache/          # 动作提取缓存
```

---

## 功能一览

| 功能 | API | 说明 |
|------|-----|------|
| 健康检查 | `GET /api/v1/health` | 服务状态、LHM++ 可用性、是否 Mock 模式 |
| 多图 3D 重建 | `POST /api/v1/avatars` | 上传 1–8 张全身人物图，导出 Gaussian Splat `.ply` |
| 蒙皮网格导出 | `POST /api/v1/avatars` + `export_skinned_mesh=true` | 同时导出 SMPL-X 蒙皮网格 OBJ/GLB + 骨骼 JSON |
| 查询 Avatar | `GET /api/v1/avatars/{id}` | 获取元数据与生成结果路径 |
| 下载高斯模型 | `GET /api/v1/avatars/{id}/model` | 下载 `.ply` |
| 下载蒙皮网格 | `GET /api/v1/avatars/{id}/mesh?format=obj\|glb` | 下载 OBJ 或 GLB |
| 下载骨骼 | `GET /api/v1/avatars/{id}/skeleton` | SMPL-X 55 关节 + LBS 权重 JSON |
| 预览图 | `GET /api/v1/avatars/{id}/preview` | 参考图预览 |
| 动作视频驱动 | `POST /api/v1/avatars/{id}/animate` | 上传动作视频，渲染动画 MP4 |
| 摄像头流驱动 | `WS /api/v1/avatars/{id}/motion-stream` | 实时发送 JPEG 帧，批量推理动画 |
| 任务查询 | `GET /api/v1/jobs/{id}` | 轮询重建/动画进度 |
| 下载动画视频 | `GET /api/v1/jobs/{id}/video` | 下载渲染结果 MP4 |

---

## 环境要求

| 组件 | 要求 |
|------|------|
| GPU | 推荐 NVIDIA ≥8GB 显存（LHMPP-700M-PixelShuffle） |
| Python | 3.10+（与 LHM++ 官方一致） |
| Node.js | 18+ |
| LHM-plusplus | 需单独克隆安装，见下方 |
| CUDA | 12.1（与 LHM++ 官方推荐一致） |

---

## 安装与启动

### 1. 安装 LHM-plusplus

在 GPU 服务器上克隆并安装 [LHM-plusplus](https://github.com/aigc3d/LHM-plusplus)：

```bash
git clone https://github.com/aigc3d/LHM-plusplus
cd LHM-plusplus

# 安装 PyTorch 2.3 + CUDA 12.1（见官方 README）
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121
pip install -U xformers==0.0.26.post1 --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
pip install rembg[cpu]

# 安装 pointops、spconv、pytorch3d、gsplat 等（见官方 INSTALL.md）

# 一键下载模型与动作数据
python scripts/download_all.py
```

**动作视频提取（可选）**：若需从用户上传视频提取 SMPL-X 动作，需确保存在 `engine/pose_estimation/video2motion.py`（可从 [LHM 仓库](https://github.com/aigc3d/LHM) 复制到 LHM++ 同路径）。

**官方 Gradio 演示**（验证 LHM++ 安装）：

```bash
python app.py --model_name LHMPP-700M-PixelShuffle
```

### 2. 启动 API 后端

```bash
cd api
pip install -r requirements.txt
cp .env.example .env
```

编辑 `api/.env`：

```env
# LHM-plusplus 仓库绝对路径（真实推理必填）
LHM_ROOT=D:/path/to/LHM-plusplus

# 模型名称（与 LHM++ 一致）
MODEL_NAME=LHMPP-700M-PixelShuffle

# 可选：本地权重路径，不填则自动从 HuggingFace/ModelScope 下载
MODEL_PATH=

# true = 无 GPU 联调模式；false = 真实推理
MOCK_MODE=false

# 前端地址（CORS）
CORS_ORIGINS=http://localhost:3000

# API 监听
API_PORT=8000
```

启动：

```bash
# 方式一
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# 方式二
python run.py
```

- API 根路径：http://localhost:8000
- 交互文档：http://localhost:8000/docs
- OpenAPI JSON：http://localhost:8000/openapi.json

### 3. 启动 Web 前端

```bash
cd web
npm install
cp .env.local.example .env.local
```

编辑 `web/.env.local`：

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000
```

启动：

```bash
npm run dev      # 开发：http://localhost:3000
npm run build    # 生产构建
npm run start    # 生产运行
```

前端通过 `next.config.js` 将 `/api/*` 代理到 `NEXT_PUBLIC_API_URL`，也可直接请求后端地址。

---

## Web 界面使用

打开 http://localhost:3000 ，按以下步骤操作：

### 步骤 1：上传参考图片

- 拖拽或选择 **1–8 张**全身人物照片（不同角度更佳，无需相机位姿）
- 调整 **参考视角数量**（1–8，默认 8）
- 可选：勾选 **「同时导出 SMPL-X 蒙皮网格」**（见 [蒙皮网格导出说明](#蒙皮网格导出说明)）
- 点击 **「开始 3D 重建」**

### 步骤 2：等待重建完成

- 页面显示任务进度条与状态（`pending` → `running` → `completed`）
- 完成后可预览参考图，并下载：
  - **Gaussian Splat PLY**（高斯点云模型）
  - **蒙皮网格 OBJ / GLB**（若已开启导出开关）
  - **SMPL-X 骨骼 JSON**（若已开启导出开关）

### 步骤 3：动作驱动

重建完成后，任选一种方式：

**方式 A：上传动作视频**

1. 选择一段人体动作视频（最多 1000 帧）
2. 调整渲染帧数（30–300，默认 120）
3. 点击 **「生成动画」**
4. 完成后页面播放 MP4 结果

**方式 B：摄像头实时捕获**

1. 点击 **「开启摄像头」**
2. 对着摄像头做动作（至少采集 **30 帧**）
3. 点击 **「提交动作」** 开始推理
4. 等待完成后播放动画视频

---

## API 使用说明

所有接口前缀为 `/api/v1`。异步任务通过 `job_id` 轮询进度。

### 健康检查

```bash
curl http://localhost:8000/api/v1/health
```

响应示例：

```json
{
  "status": "ok",
  "lhmpp_available": true,
  "mock_mode": false
}
```

### 创建 Avatar（3D 重建）

```bash
curl -X POST http://localhost:8000/api/v1/avatars \
  -F "images=@photo1.png" \
  -F "images=@photo2.png" \
  -F "ref_view=8" \
  -F "export_skinned_mesh=false"
```

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `images` | file[] | 必填 | 1–8 张人物图片 |
| `ref_view` | int | 8 | 实际用于推理的参考视角数 |
| `export_skinned_mesh` | bool | false | 是否同时导出 SMPL-X 蒙皮网格 |

响应：

```json
{
  "avatar_id": "uuid",
  "job_id": "uuid",
  "status": "pending"
}
```

轮询任务：

```bash
curl http://localhost:8000/api/v1/jobs/{job_id}
```

任务 `status` 为 `completed` 时，`result` 中包含 `ply_path`、`preview_path` 等路径；若开启蒙皮导出，还包含 `mesh_obj_path`、`skeleton_json_path` 等。

### 下载模型文件

```bash
# 高斯 PLY
curl -O http://localhost:8000/api/v1/avatars/{avatar_id}/model

# 蒙皮网格 OBJ
curl -O "http://localhost:8000/api/v1/avatars/{avatar_id}/mesh?format=obj"

# 蒙皮网格 GLB
curl -O "http://localhost:8000/api/v1/avatars/{avatar_id}/mesh?format=glb"

# SMPL-X 骨骼 JSON
curl -O http://localhost:8000/api/v1/avatars/{avatar_id}/skeleton

# 预览图
curl -O http://localhost:8000/api/v1/avatars/{avatar_id}/preview
```

### 动作视频驱动

```bash
curl -X POST http://localhost:8000/api/v1/avatars/{avatar_id}/animate \
  -F "motion_video=@dance.mp4" \
  -F "motion_frames=120" \
  -F "render_backend=neural"
```

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `motion_video` | file | 必填 | 动作参考视频 |
| `motion_frames` | int | 120 | 渲染帧数（30–1000） |
| `render_backend` | string | neural | `neural`（神经渲染）或 `gs`（纯高斯光栅，需模型支持） |

完成后下载视频：

```bash
curl -O http://localhost:8000/api/v1/jobs/{job_id}/video
```

### 摄像头流驱动（WebSocket）

连接：

```
ws://localhost:8000/api/v1/avatars/{avatar_id}/motion-stream
```

**客户端 → 服务端消息**（JSON 文本）：

| type | 字段 | 说明 |
|------|------|------|
| `frame` | `data` | base64 JPEG 帧（可带 `data:image/jpeg;base64,` 前缀） |
| `flush` | — | 提交缓冲帧，开始推理（至少 30 帧） |
| `poll` | `job_id` | 查询推理任务状态 |
| `close` | — | 关闭连接 |

**服务端 → 客户端消息**：

| type | 说明 |
|------|------|
| `ready` | 连接就绪，可开始发帧 |
| `buffer` | 当前缓冲帧数 `{ count, required }` |
| `processing` | 开始处理 `{ job_id, frames }` |
| `job_started` | 任务已创建 `{ job_id }` |
| `job_status` | 任务状态；完成时含 `video_url` |
| `error` | 错误信息 |

示例（JavaScript）：

```javascript
const ws = new WebSocket(`ws://localhost:8000/api/v1/avatars/${avatarId}/motion-stream`);

ws.onmessage = (e) => {
  const msg = JSON.parse(e.data);
  if (msg.type === "job_status" && msg.status === "completed") {
    window.open(`http://localhost:8000${msg.video_url}`);
  }
};

// 发送帧
ws.send(JSON.stringify({ type: "frame", data: canvas.toDataURL("image/jpeg", 0.7) }));

// 提交推理
ws.send(JSON.stringify({ type: "flush" }));

// 轮询
ws.send(JSON.stringify({ type: "poll", job_id: "..." }));
```

### 任务查询

```bash
curl http://localhost:8000/api/v1/jobs/{job_id}
```

响应字段：

| 字段 | 说明 |
|------|------|
| `status` | `pending` / `running` / `completed` / `failed` |
| `progress` | 0–100 |
| `message` | 当前阶段描述 |
| `result` | 完成时的输出路径等信息 |
| `error` | 失败时的错误信息 |

---

## 输出文件说明

每个 Avatar 在 `api/data/avatars/{avatar_id}/` 下生成：

```
{avatar_id}/
├── inputs/              # 上传的原始图片
│   └── input_000.png
├── output/              # 重建结果
│   ├── ref_images/      # 预处理后的参考图
│   ├── avatar.ply       # 3D Gaussian Splat（LHM++ 官方 to_gs_ply 导出）
│   ├── betas.json       # SMPL-X 体型参数（真实推理时）
│   ├── avatar_skinned.obj          # 蒙皮网格（可选）
│   ├── avatar_skinned.glb            # 蒙皮网格 GLB（可选）
│   ├── avatar_skeleton.json        # SMPL-X 骨骼与权重（可选）
│   └── avatar_lbs_weights.npz      # 完整 LBS 权重矩阵（可选）
├── motion/              # 上传的动作视频
└── animations/{job_id}/ # 动画渲染结果
    └── animation.mp4
```

---

## 蒙皮网格导出说明

LHM++ **官方**仅提供 3D Gaussian Splatting PLY 导出（`scripts/inference/to_gs_ply.py`），**不提供**传统蒙皮网格 OBJ/GLB 一键导出。

HRM 在官方高斯重建之上，增加了**可选的自研后处理层**（`api/services/mesh_export_service.py`），将高斯点云转换为带 SMPL-X 骨骼的传统网格，便于导入 Blender / Maya / Unity。

### LHM++ 官方 vs HRM 自研

| 输出 | 来源 | 说明 |
|------|------|------|
| Gaussian PLY | LHM++ `to_gs_ply.py` | 官方 3DGS 点云 |
| 蒙皮网格 OBJ/GLB/骨骼 JSON | HRM `mesh_export_service.py` | 自研后处理，非官方能力 |

### 自研实现用了什么

| 用途 | 库 / 来源 |
|------|-----------|
| 读取高斯 PLY 顶点坐标 | **plyfile** |
| 高斯位移 → SMPL-X 网格顶点映射 | **scipy** `cKDTree` |
| SMPL-X 模板网格、面片、55 关节 LBS 权重 | LHM++ **`SMPL_Layer`**（PyTorch） |
| 写 OBJ | 手写 `v` / `f` 文本（不用 trimesh） |
| 写 GLB | **trimesh**（`Trimesh.export()`） |
| 骨骼与权重元数据 | **json**（稀疏权重）+ **numpy** `npz`（完整权重矩阵） |

依赖见 `api/requirements.txt`：`plyfile`、`scipy`、`trimesh`。

### 流程简述

1. 调用 LHM++ `to_gs_ply.py` 生成高斯 PLY（`avatar.ply`）
2. 用 **plyfile** 读取高斯中心坐标 `(x, y, z)`
3. 加载 LHM++ `pretrained_models/dense_sample_points` 作为表面锚点
4. 通过 **`SMPL_Layer`** 获取 T-pose 网格顶点、三角面与 LBS 权重
5. 计算高斯相对锚点的位移，用 **cKDTree** 最近邻加权插值到网格顶点
6. 导出：
   - `avatar_skinned.obj` — 位移后的传统网格
   - `avatar_skeleton.json` — 55 关节名、父子关系、稀疏 LBS 权重、betas
   - `avatar_skinned.glb` — 可选（见下方局限）
   - `avatar_lbs_weights.npz` — 完整权重矩阵与面片索引

### 关于 trimesh 的局限

**trimesh 只用于 GLB 几何体导出**，不是整条管线的核心。

它对**蒙皮 GLB**（带骨骼/armature 绑定）支持很弱，因此：

- **OBJ + `avatar_skeleton.json`** 才是完整蒙皮信息，适合 DCC 工具绑骨
- **GLB** 仅为方便预览的**纯网格**，不含完整骨骼绑定

若需要标准蒙皮 GLB，建议在 Blender 中导入 OBJ 与骨骼 JSON 后重新导出。

### 使用方式

勾选 Web 界面的「同时导出 SMPL-X 蒙皮网格」，或 API 传 `export_skinned_mesh=true` 即可启用。

需配置 `LHM_ROOT` 指向已安装依赖并下载 prior 模型的 `LHM-plusplus` 目录（示例：`../LHM-plusplus`）。

---

## Mock 模式（无 GPU 联调）

在 `api/.env` 中设置：

```env
MOCK_MODE=true
```

无需配置 `LHM_ROOT`，也无需 GPU。此时：

- 重建返回占位 PLY 与预览图
- 蒙皮导出返回简易人形占位网格 + 完整 55 关节骨骼结构
- 动画返回空视频文件

用于前端 UI 与 API 联调。真实推理请设 `MOCK_MODE=false` 并配置 `LHM_ROOT`。

---

## 配置项参考

### API（`api/.env`）

| 变量 | 默认 | 说明 |
|------|------|------|
| `LHM_ROOT` | 空 | LHM-plusplus 仓库绝对路径 |
| `MODEL_NAME` | `LHMPP-700M-PixelShuffle` | 模型名 |
| `MODEL_PATH` | 空 | 本地权重路径（可选） |
| `MOCK_MODE` | `false` | 是否 Mock 模式 |
| `CORS_ORIGINS` | `http://localhost:3000` | 允许的前端源，逗号分隔 |
| `API_PORT` | `8000` | 监听端口 |

代码内固定参数（`api/config.py`）：

| 参数 | 值 | 说明 |
|------|-----|------|
| `max_ref_images` | 8 | 最大参考图数量 |
| `max_motion_frames` | 1000 | 最大动作帧数 |
| `default_motion_frames` | 120 | 默认动画帧数 |
| `render_fps` | 30 | 输出视频帧率 |

### Web（`web/.env.local`）

| 变量 | 默认 | 说明 |
|------|------|------|
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | 后端 HTTP 地址 |
| `NEXT_PUBLIC_WS_URL` | `ws://localhost:8000` | 后端 WebSocket 地址 |

---

## 项目结构

```
hrm/
├── README.md
├── api/                          # FastAPI 后端
│   ├── main.py                   # 应用入口
│   ├── run.py                    # uvicorn 启动脚本
│   ├── config.py                 # 配置
│   ├── requirements.txt
│   ├── .env.example
│   ├── routers/
│   │   ├── avatars.py            # REST：重建、下载、动画
│   │   └── stream.py             # WebSocket：摄像头流
│   └── services/
│       ├── lhmpp_service.py      # LHM++ 推理封装
│       ├── motion_service.py     # 动作视频 → SMPL-X 提取
│       ├── mesh_export_service.py # 高斯 PLY → SMPL-X 蒙皮网格（自研后处理）
│       └── job_manager.py        # 异步任务管理
└── web/                          # Next.js 前端
    ├── package.json
    ├── next.config.js
    ├── .env.local.example
    └── src/
        ├── app/
        │   ├── page.tsx          # 主工作流页面
        │   ├── layout.tsx
        │   └── globals.css
        └── lib/
            └── api.ts            # API 客户端
```

---

## 常见问题

**Q: 重建报错「LHM_ROOT 不存在」**  
A: 在 `api/.env` 中设置正确的 `LHM_ROOT` 路径，或临时用 `MOCK_MODE=true` 联调。

**Q: 动作视频无法提取 SMPL-X**  
A: 确认 LHM++ 已安装 `video2motion.py`（可从 LHM 仓库复制），且 GPU 可用。

**Q: 蒙皮网格下载 404**  
A: 创建 Avatar 时需传 `export_skinned_mesh=true`，或在 Web 界面勾选对应开关。

**Q: 摄像头流提交后无结果**  
A: 至少采集 30 帧再点「提交动作」；通过 WebSocket 发送 `poll` 消息查询 `job_id` 状态。

**Q: 显存不足**  
A: LHM++ 推荐 ≥8GB；可减少 `ref_view` 或 `motion_frames`。

**Q: 前端跨域错误**  
A: 检查 `CORS_ORIGINS` 是否包含前端地址（如 `http://localhost:3000`）。

---

## 参考链接

- [LHM++ 项目页](https://lingtengqiu.github.io/LHM++/)
- [LHM++ GitHub](https://github.com/aigc3d/LHM-plusplus)
- [HuggingFace Demo](https://huggingface.co/spaces/Lingteng/LHMPP)
- [LHM GitHub](https://github.com/aigc3d/LHM)（含 video2motion）
- 论文：[arXiv:2506.13766](https://arxiv.org/pdf/2506.13766v2)

---

## License

本工具封装层代码可自由使用；LHM++ 模型与权重遵循其 [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) 许可。
