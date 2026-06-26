from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

_API_ROOT = Path(__file__).resolve().parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_API_ROOT / ".env"),
        extra="ignore",
    )

    # LHM-plusplus 仓库根目录（需已安装依赖并下载模型）
    lhm_root: str = ""
    model_name: str = "LHMPP-700M-PixelShuffle"
    model_path: str = ""

    # 数据目录（固定为 api/data，避免 LHM++ 推理时 chdir 导致相对路径错乱）
    data_dir: Path = _API_ROOT / "data"
    max_ref_images: int = 8
    max_motion_frames: int = 1000
    default_motion_frames: int = 120
    render_fps: int = 30

    # 无 GPU 时启用 mock 模式（仅用于前端联调）
    mock_mode: bool = False

    # 启动时预加载 LHM++ 模型（动作驱动）；false 则首次调用 animate 时再加载
    preload_model: bool = True

    # 推理显存控制（测试 / 低显存 GPU）
    # INFER_LOW_MEMORY=true 时默认：ref_view≤4、输入高 560px、动画 batch=8
    infer_low_memory: bool = False
    infer_max_image_size: int = 0  # 0=自动（低显存 560，正常 840）
    infer_ref_view_max: int = 0  # 0=自动（低显存 4，正常 max_ref_images）
    infer_anim_batch_size: int = 0  # 0=自动（低显存 8，正常 40）
    infer_dense_sample_pts: int = 0  # 0=自动（低显存 40000，正常用 checkpoint 160000）

    # Blender headless 路径（蒙皮 FBX 导出）；空则自动在 PATH 中查找 blender
    blender_executable: str = ""

    # FBX 导出：3DGS→UV 烘焙 + 细分高模
    fbx_texture_size: int = 2048
    fbx_subdivision_levels: int = 1  # 0=低模，1≈4×面，2≈16×面
    fbx_bake_texture: bool = True

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: str = "http://localhost:3000"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def avatars_dir(self) -> Path:
        return self.data_dir / "avatars"

    @property
    def jobs_dir(self) -> Path:
        return self.data_dir / "jobs"

    @property
    def motion_cache_dir(self) -> Path:
        return self.data_dir / "motion_cache"

    @property
    def effective_infer_max_image_size(self) -> int:
        if self.infer_max_image_size > 0:
            return self.infer_max_image_size
        return 560 if self.infer_low_memory else 840

    @property
    def effective_infer_ref_view_max(self) -> int:
        if self.infer_ref_view_max > 0:
            return self.infer_ref_view_max
        return 4 if self.infer_low_memory else self.max_ref_images

    @property
    def effective_infer_anim_batch_size(self) -> int:
        if self.infer_anim_batch_size > 0:
            return self.infer_anim_batch_size
        return 8 if self.infer_low_memory else 40

    @property
    def effective_infer_dense_sample_pts(self) -> int:
        """体素/query 采样点数上限；0 表示不覆盖 checkpoint 默认值。"""
        if self.infer_dense_sample_pts > 0:
            return self.infer_dense_sample_pts
        return 40000 if self.infer_low_memory else 0


settings = Settings()
