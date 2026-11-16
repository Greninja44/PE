"""Application settings, loaded from environment variables (prefix ``PE_``) or ``backend/.env``."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[2]
REPO_DIR = BACKEND_DIR.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PE_", env_file=str(BACKEND_DIR / ".env"), extra="ignore")

    app_name: str = "Image-to-Prompt Reconstruction Lab"
    database_url: str = f"sqlite:///{BACKEND_DIR / 'data' / 'pe.db'}"
    experiments_dir: Path = REPO_DIR / "experiments"
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    # Upload limits
    max_upload_bytes: int = 20 * 1024 * 1024
    allowed_image_types: tuple[str, ...] = ("image/png", "image/jpeg", "image/webp")

    # Worker / job queue
    admin_token: str = Field(default="", description="Required to register GPU workers. Empty disables registration.")
    job_lease_seconds: int = 300
    job_max_attempts: int = 3

    # Remote providers (all optional)
    hf_token: str = ""
    hf_base_url: str = "https://router.huggingface.co"
    openai_compat_base_url: str = ""
    openai_compat_api_key: str = ""
    openai_compat_model: str = "qwen3-vl-2b"

    # Default model choices
    default_vlm_model: str = "Qwen/Qwen3-VL-2B-Instruct"

    @property
    def originals_dir(self) -> Path:
        return self.experiments_dir / "originals"

    @property
    def generated_dir(self) -> Path:
        return self.experiments_dir / "generated"

    @property
    def reports_dir(self) -> Path:
        return self.experiments_dir / "reports"

    @property
    def prompts_dir(self) -> Path:
        return self.experiments_dir / "prompts"

    @property
    def metrics_dir(self) -> Path:
        return self.experiments_dir / "metrics"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    def ensure_dirs(self) -> None:
        for d in (self.originals_dir, self.generated_dir, self.reports_dir, self.prompts_dir, self.metrics_dir):
            d.mkdir(parents=True, exist_ok=True)
        if self.database_url.startswith("sqlite:///"):
            Path(self.database_url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()
