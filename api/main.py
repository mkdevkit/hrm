from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from config import settings
from routers import avatars, stream
from services.lhmpp_service import lhmpp_service
from services.blender_fbx_export import blender_available, blender_fbx_ready

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    log = logging.getLogger("hrm.config")
    log.info(
        "已加载 api/.env | LHM_ROOT=%s MOCK=%s PRELOAD_MODEL=%s INFER_LOW_MEMORY=%s "
        "max_image=%s ref_view_max=%s dense_sample=%s",
        settings.lhm_root or "(未设置)",
        settings.mock_mode,
        settings.preload_model,
        settings.infer_low_memory,
        settings.effective_infer_max_image_size,
        settings.effective_infer_ref_view_max,
        settings.effective_infer_dense_sample_pts,
    )
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.avatars_dir.mkdir(parents=True, exist_ok=True)
    settings.jobs_dir.mkdir(parents=True, exist_ok=True)
    settings.motion_cache_dir.mkdir(parents=True, exist_ok=True)

    if settings.preload_model:
        if settings.mock_mode:
            log.info("PRELOAD_MODEL=true 但 MOCK_MODE=true，跳过模型预加载")
        elif not lhmpp_service.available:
            log.warning("PRELOAD_MODEL=true 但 LHM_ROOT 不可用，跳过模型预加载")
        else:
            await asyncio.to_thread(lhmpp_service.initialize)

    if blender_available() and not blender_fbx_ready():
        log.warning(
            "Blender 已安装但 FBX 导出缺少 numpy。"
            " 请执行: sudo apt install -y python3-numpy"
        )

    yield


app = FastAPI(
    title="HRM - LHM++ Avatar API",
    description="多图重建可动画 3D 人体 + 视频流动作驱动",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(avatars.router)
app.include_router(stream.router)

# 静态文件：动画输出
if settings.data_dir.exists():
    app.mount("/files", StaticFiles(directory=str(settings.data_dir)), name="files")


@app.get("/")
def root():
    return {
        "service": "HRM LHM++ API",
        "docs": "/docs",
        "health": "/api/v1/health",
    }
