from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LHM-plusplus 仓库根目录（需已安装依赖并下载模型）
    lhm_root: str = ""
    model_name: str = "LHMPP-700M-PixelShuffle"
    model_path: str = ""

    # 数据目录
    data_dir: Path = Path("./data")
    max_ref_images: int = 8
    max_motion_frames: int = 1000
    default_motion_frames: int = 120
    render_fps: int = 30

    # 无 GPU 时启用 mock 模式（仅用于前端联调）
    mock_mode: bool = False

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


settings = Settings()
