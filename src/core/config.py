"""Application configuration loader."""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

load_dotenv()


def _load_secrets_from_aws() -> None:
    """Fetch secrets from AWS Secrets Manager and inject them as env vars.

    Only runs when the AWS_SECRETS_NAME environment variable is set,
    so local development (using .env) is unaffected.
    """
    secret_name = os.environ.get("AWS_SECRETS_NAME")
    if not secret_name:
        return

    try:
        import boto3
        from botocore.exceptions import ClientError

        region = os.environ.get("AWS_REGION", "us-east-1")
        client = boto3.client("secretsmanager", region_name=region)
        response = client.get_secret_value(SecretId=secret_name)
        secrets = json.loads(response["SecretString"])
        for key, value in secrets.items():
            os.environ.setdefault(key, value)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to load secrets from AWS Secrets Manager: {exc}") from exc


_load_secrets_from_aws()

_ROOT = Path(__file__).parent.parent.parent


class LLMConfig(BaseModel):
    provider: str = "anthropic"
    model: str = "claude-sonnet-4-6"
    temperature: float = 0.1
    max_tokens: int = 4096


class EmbeddingsConfig(BaseModel):
    model: str = "all-MiniLM-L6-v2"
    dimension: int = 384


class RAGConfig(BaseModel):
    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k: int = 5
    index_path: str = "data/faiss_index"
    knowledge_base_path: str = "src/data/knowledge_base"


class RedisConfig(BaseModel):
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    ttl_seconds: int = 3600


class APIConfig(BaseModel):
    alpha_vantage_base_url: str = "https://www.alphavantage.co/query"
    fred_base_url: str = "https://api.stlouisfed.org/fred"
    news_api_base_url: str = "https://newsapi.org/v2"
    sec_edgar_base_url: str = "https://efts.sec.gov/LATEST/search-index"


class WorkflowConfig(BaseModel):
    max_iterations: int = 10
    timeout_seconds: int = 60


class AppConfig(BaseModel):
    name: str = "Finnie - AI Finance Assistant"
    version: str = "1.0.0"
    debug: bool = False
    log_level: str = "INFO"


class Settings(BaseSettings):
    # Loaded from .env
    anthropic_api_key: str = Field(default="", env="ANTHROPIC_API_KEY")
    alpha_vantage_api_key: str = Field(default="", env="ALPHA_VANTAGE_API_KEY")
    fred_api_key: str = Field(default="", env="FRED_API_KEY")
    news_api_key: str = Field(default="", env="NEWS_API_KEY")
    redis_host: str = Field(default="localhost", env="REDIS_HOST")
    redis_port: int = Field(default=6379, env="REDIS_PORT")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    debug: bool = Field(default=False, env="DEBUG")

    # Nested config (from YAML)
    app: AppConfig = AppConfig()
    llm: LLMConfig = LLMConfig()
    embeddings: EmbeddingsConfig = EmbeddingsConfig()
    rag: RAGConfig = RAGConfig()
    redis: RedisConfig = RedisConfig()
    apis: APIConfig = APIConfig()
    workflow: WorkflowConfig = WorkflowConfig()

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    @property
    def root_dir(self) -> Path:
        return _ROOT

    @property
    def rag_index_path(self) -> Path:
        return _ROOT / self.rag.index_path

    @property
    def knowledge_base_path(self) -> Path:
        return _ROOT / self.rag.knowledge_base_path


def _load_yaml(path: Path) -> dict[str, Any]:
    if path.exists():
        with open(path) as f:
            return yaml.safe_load(f) or {}
    return {}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    yaml_data = _load_yaml(_ROOT / "config.yaml")

    # Override Settings fields from yaml where env vars aren't set
    overrides: dict[str, Any] = {}

    if "llm" in yaml_data:
        overrides["llm"] = LLMConfig(**yaml_data["llm"])
    if "embeddings" in yaml_data:
        overrides["embeddings"] = EmbeddingsConfig(**yaml_data["embeddings"])
    if "rag" in yaml_data:
        overrides["rag"] = RAGConfig(**yaml_data["rag"])
    if "redis" in yaml_data:
        redis_yaml = yaml_data["redis"]
        overrides["redis"] = RedisConfig(**redis_yaml)
    if "workflow" in yaml_data:
        overrides["workflow"] = WorkflowConfig(**yaml_data["workflow"])
    if "app" in yaml_data:
        overrides["app"] = AppConfig(**yaml_data["app"])

    settings = Settings(**overrides)

    # Redis host/port from env take precedence
    settings.redis.host = settings.redis_host
    settings.redis.port = settings.redis_port

    return settings
