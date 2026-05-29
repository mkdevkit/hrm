from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from config import settings
from routers import avatars, stream


@asynccontextmanager
async def lifespan(_app: FastAPI):
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.avatars_dir.mkdir(parents=True, exist_ok=True)
    settings.jobs_dir.mkdir(parents=True, exist_ok=True)
    settings.motion_cache_dir.mkdir(parents=True, exist_ok=True)
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
