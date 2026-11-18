from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = PROJECT_ROOT / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AGENT_",
        env_file=ENV_PATH,
        env_file_encoding="utf-8-sig",
        extra="ignore",
    )

    classifier_base_url: str = "http://127.0.0.1:8000"
    classifier_timeout_seconds: float = Field(default=10.0, gt=0)

    groq_api_key: SecretStr = Field(
        validation_alias="GROQ_API_KEY",
    )
    llm_model: str = "openai/gpt-oss-20b"

    prompt_path: Path =  PROJECT_ROOT / "configs" / "prompt" / "finalize.json"