"""Application configuration loader."""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

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


class PromptCachingConfig(BaseModel):
    """Anthropic prompt caching on agent system prompts (see src/agents/prompts/)."""
    enabled: bool = True
    # "5m" writes at 1.25x input price, "1h" at 2x; reads are ~0.1x on both.
    # 5m is strictly cheaper while requests sharing the prefix start < 5 min
    # apart (every read refreshes the timer); 1h only pays off for gaps of 5-60 min.
    ttl: Literal["5m", "1h"] = "5m"


class LLMConfig(BaseModel):
    provider: str = "anthropic"
    model: str = "claude-sonnet-5"
    temperature: float = 0.1  # only applied to models that accept sampling params (see llm.py)
    max_tokens: int = 4096
    # "adaptive" sends no thinking parameter: the model's default, which on
    # Claude Sonnet 5 means it may think silently before its first text token
    # (seconds on planning questions). "disabled" turns that off. The voice
    # worker sets FINNIE_LLM_THINKING=disabled; see src/core/llm.py.
    thinking: Literal["adaptive", "disabled"] = "adaptive"
    # Model for the merged classifier (scope gate + routing). None = `model`.
    # A smaller model answers this short structured call faster; check routing
    # accuracy before switching (spike/voice/compare_classifier.py).
    classifier_model: str | None = None
    # The classifier also writes a one-sentence reason (shown with chat answers,
    # logged with refusals). It comes after the decision, so it never changes
    # it, but it is about a quarter of the call's output: ~0.35 s on Haiku 5.5.
    # The voice worker turns it off (voice.classifier_reason).
    classifier_reason: bool = True
    prompt_caching: PromptCachingConfig = PromptCachingConfig()


class EmbeddingsConfig(BaseModel):
    model: str = "all-MiniLM-L6-v2"
    dimension: int = 384


class RAGConfig(BaseModel):
    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k: int = 5
    index_path: str = "data/faiss_index"
    knowledge_base_path: str = "src/data/knowledge_base"
    # "auto": pgvector when DATABASE_URL is set, else the FAISS file index.
    backend: Literal["auto", "faiss", "pgvector"] = "auto"
    # Hybrid search (pgvector): each ranker contributes this many candidates,
    # fused by reciprocal rank fusion with constant rrf_k.
    hybrid_candidates: int = 20
    rrf_k: int = 60


class RedisConfig(BaseModel):
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    ttl_seconds: int = 3600
    stale_ttl_seconds: int = 604800  # 7 days; retention for stale cache mirror


class APIConfig(BaseModel):
    alpha_vantage_base_url: str = "https://www.alphavantage.co/query"
    fred_base_url: str = "https://api.stlouisfed.org/fred"
    news_api_base_url: str = "https://newsapi.org/v2"
    sec_edgar_base_url: str = "https://efts.sec.gov/LATEST/search-index"


class WorkflowConfig(BaseModel):
    max_iterations: int = 10
    timeout_seconds: int = 60
    # Conversation messages sent to an agent per call. Persistent threads keep
    # the full history in the checkpoint; this bounds what each LLM call pays for.
    max_history_messages: int = 40


class PersistenceConfig(BaseModel):
    """Per-user profile, holdings, saved plan and chat history (src/persistence/)."""
    # "auto": Postgres when DATABASE_URL is set, otherwise in-process memory
    # (data lost on restart — the pre-phase-3 behaviour, kept so a deploy without
    # a database still runs). "postgres" requires DATABASE_URL.
    backend: Literal["auto", "postgres", "memory"] = "auto"
    pool_max_size: int = 10
    connect_timeout_seconds: float = 10.0


class CircuitBreakerConfig(BaseModel):
    failure_threshold: int = 5
    recovery_timeout_seconds: int = 60
    success_threshold: int = 1


class GuardrailConfig(BaseModel):
    """Pre-router safety/topic gate configuration."""
    enabled: bool = True
    refusal_message: str = (
        "I'm Finnie, a financial education assistant, so I can only help with "
        "finance-related questions — things like investing, budgeting, markets, "
        "taxes, and financial planning. Try asking me something in that area!"
    )


class FAQCacheConfig(BaseModel):
    """Semantic answer cache served before the classifier."""
    enabled: bool = True
    # "auto": Postgres when DATABASE_URL is set (shared by every instance),
    # else Redis with an in-process fallback.
    backend: Literal["auto", "postgres", "redis"] = "auto"
    similarity_threshold: float = 0.92  # conservative: a wrong figure beats a miss
    ttl_seconds: int = 604800  # 7 days
    max_entries: int = 500


class FastPathConfig(BaseModel):
    """Phase-1 latency features. Each flag is independently reversible."""
    merged_classifier: bool = True  # one guardrail+router call instead of two
    streaming: bool = True  # stream agent tokens to the UI
    faq_cache: FAQCacheConfig = FAQCacheConfig()


class TracingConfig(BaseModel):
    """Arize Phoenix / OpenInference tracing configuration."""
    enabled: bool = False
    project_name: str = "finnie"
    endpoint: str = ""  # e.g. http://localhost:4317 (local, gRPC) or
                         # https://app.phoenix.arize.com/s/<space> (Phoenix Cloud, HTTP)
    protocol: str = ""  # "grpc" | "http/protobuf" ; empty -> inferred from endpoint
    api_key: str = ""   # Phoenix Cloud API key (from app.phoenix.arize.com, not Arize AX)


class PlanningConfig(BaseModel):
    """Life-event timeline projection defaults and caps."""
    default_inflation: float = 0.03
    max_horizon_years: int = 60
    max_events: int = 25


class MemoryConfig(BaseModel):
    """Long-term memory: durable facts users share in chat (src/memory/)."""
    enabled: bool = True
    # Only facts the extractor is at least this sure the user stated as true.
    min_confidence: float = 0.8
    max_facts_per_turn: int = 5
    # Memories injected into a turn, ranked by relevance to the question.
    top_k: int = 5
    # A new fact this similar to an existing one of the same category replaces
    # it (so "Is 35" supersedes "Is 34") instead of accumulating.
    dedup_similarity: float = 0.85
    max_memories_per_user: int = 100


class TokenRates(BaseModel):
    """US dollars per million tokens for one Claude model."""
    input: float
    output: float
    cache_read: float
    cache_write: float  # 5-minute cache


class VoiceCostConfig(BaseModel):
    """List prices for the estimated cost on each call's `voice_call_summary` log
    line (src/voice/usage.py). Checked 2026-11-15; replace with your plan's rates.
    A model missing here is listed as unpriced on the line, not guessed."""
    livekit_per_minute: float = 0.01        # agent session minutes, billed by the started minute
    stt_per_minute: float = 0.0077          # Deepgram Nova-3 streaming (a $0.0048 promotion runs)
    tts_per_1k_chars: dict[str, float] = {
        "eleven_flash_v2_5": 0.05,
        "aura-2-asteria-en": 0.03,
    }
    llm_per_mtok: dict[str, TokenRates] = {
        "claude-sonnet-5": TokenRates(input=2.0, output=10.0, cache_read=0.20, cache_write=2.50),
        "claude-haiku-5-5": TokenRates(input=0.10, output=0.50, cache_read=0.01, cache_write=0.125),
    }


class VoiceConfig(BaseModel):
    """Voice calls (src/voice/): who may call, and what the worker runs.

    Off by default. The worker re-checks `enabled` and the allowlist when each
    call joins, so a token minted before access was withdrawn does not work.
    """
    enabled: bool = False
    # Lower-cased Google account emails allowed to call. Empty lets nobody in
    # (unlike the app's ALLOWED_EMAILS); a single "*" lets every signed-in user in.
    allowed_emails: list[str] = []
    stt_model: str = "nova-3"
    tts_model: str = "eleven_flash_v2_5"
    tts_voice_id: str = ""  # empty: the ElevenLabs plugin's default voice
    # Latency settings the worker applies to its own process only (see llm.*).
    classifier_model: str = "claude-haiku-5-5"
    classifier_reason: bool = False
    llm_thinking: Literal["adaptive", "disabled"] = "disabled"
    # The worker holds few concurrent turns per process; chat's pool size would
    # multiply Cloud SQL connections by the number of worker processes.
    db_pool_max_size: int = 2
    # Each call runs in its own process, which loads the embedding model and the
    # graph (~850 MB, 15-40 s). LiveKit keeps this many loaded and waiting; its
    # production default, 14, would need ~12 GB per worker instance.
    idle_processes: int = 2
    job_memory_warn_mb: int = 1500  # LiveKit's 1000 MB default sits below one idle process
    # Starting a call (src/voice/tokens.py).
    tokens_per_hour: int = 10
    token_ttl_minutes: int = 10  # to join; a joined call outlives its token
    # Limits during a call (src/voice/limits.py). Starting points from the
    # design doc, to revisit with measured cost per minute.
    max_call_minutes: float = 20
    call_warning_minutes: float = 18
    idle_prompt_seconds: float = 90
    idle_hangup_seconds: float = 120
    max_turns_per_call: int = 60
    daily_minutes_per_user: float = 30
    # Turn taking. LiveKit's defaults to start; tune with simulated calls.
    endpointing_min_delay: float = 0.5   # silence before a turn may end
    endpointing_max_delay: float = 3.0   # longest wait for a mid-question pause
    interruption_min_duration: float = 0.5  # shorter speech ("uh-huh") doesn't interrupt
    interruption_min_words: int = 0
    false_interruption_timeout: float = 2.0  # resume if the "interruption" had no words
    # Failures (src/voice/turns.py, health.py).
    stt_stall_seconds: float = 3.0     # speech with no transcript this long after it ends
    stt_stalls_to_end: int = 2         # stalls in a row that end the call
    # Failed vendor reconnects in a row before LiveKit closes the session on
    # its own; high enough that the worker ends a broken call after saying why.
    vendor_error_tolerance: int = 15
    slow_turn_notice_seconds: float = 8
    slow_turn_abandon_seconds: float = 20
    # Backup voice from another vendor, used for the rest of a call once the
    # main one fails, and alone while the ElevenLabs breaker is open.
    tts_fallback_model: str = "aura-2-asteria-en"
    tts_connect_timeout_seconds: float = 2.0
    # Memory from voice turns (src/voice/memory.py). A misheard sentence can
    # become a wrong fact, so voice is stricter than chat: turns Deepgram was
    # unsure of are skipped, and facts need more confidence than
    # memory.min_confidence. Saves still running when a call ends get this long.
    memory_min_transcript_confidence: float = 0.85
    memory_min_confidence: float = 0.9
    memory_wait_seconds: float = 10.0
    cost: VoiceCostConfig = VoiceCostConfig()


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
    # NOTE: pydantic-settings v2 ignores Field(env=...) (silently a no-op — it's a
    # pydantic v1 kwarg). The fields above this comment only work by coincidence
    # because their auto-derived env var name (uppercased field name) already
    # matches. These two use validation_alias since PHOENIX_COLLECTOR_ENDPOINT
    # doesn't match the field name.
    phoenix_api_key: str = Field(default="", validation_alias="PHOENIX_API_KEY")
    phoenix_endpoint: str = Field(default="", validation_alias="PHOENIX_COLLECTOR_ENDPOINT")
    database_url: str = Field(default="", validation_alias="DATABASE_URL")

    # Nested config (from YAML)
    app: AppConfig = AppConfig()
    llm: LLMConfig = LLMConfig()
    embeddings: EmbeddingsConfig = EmbeddingsConfig()
    rag: RAGConfig = RAGConfig()
    redis: RedisConfig = RedisConfig()
    apis: APIConfig = APIConfig()
    workflow: WorkflowConfig = WorkflowConfig()
    persistence: PersistenceConfig = PersistenceConfig()
    circuit_breaker: CircuitBreakerConfig = CircuitBreakerConfig()
    guardrail: GuardrailConfig = GuardrailConfig()
    fast_path: FastPathConfig = FastPathConfig()
    tracing: TracingConfig = TracingConfig()
    planning: PlanningConfig = PlanningConfig()
    memory: MemoryConfig = MemoryConfig()
    voice: VoiceConfig = VoiceConfig()

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
    if "persistence" in yaml_data:
        overrides["persistence"] = PersistenceConfig(**yaml_data["persistence"])
    if "circuit_breaker" in yaml_data:
        overrides["circuit_breaker"] = CircuitBreakerConfig(**yaml_data["circuit_breaker"])
    if "guardrail" in yaml_data:
        overrides["guardrail"] = GuardrailConfig(**yaml_data["guardrail"])
    if "fast_path" in yaml_data:
        overrides["fast_path"] = FastPathConfig(**yaml_data["fast_path"])
    if "tracing" in yaml_data:
        overrides["tracing"] = TracingConfig(**yaml_data["tracing"])
    if "planning" in yaml_data:
        overrides["planning"] = PlanningConfig(**yaml_data["planning"])
    if "memory" in yaml_data:
        overrides["memory"] = MemoryConfig(**yaml_data["memory"])
    if "voice" in yaml_data:
        overrides["voice"] = VoiceConfig(**yaml_data["voice"])
    if "app" in yaml_data:
        overrides["app"] = AppConfig(**yaml_data["app"])

    settings = Settings(**overrides)

    # Redis host/port from env take precedence
    settings.redis.host = settings.redis_host
    settings.redis.port = settings.redis_port

    # Phoenix API key / endpoint from env take precedence (keeps secrets out of config.yaml)
    if settings.phoenix_api_key:
        settings.tracing.api_key = settings.phoenix_api_key
    if settings.phoenix_endpoint:
        settings.tracing.endpoint = settings.phoenix_endpoint

    # Optional env override to flip tracing on/off without editing config.yaml or
    # rebuilding the image (handy for toggling per Cloud Run deployment/revision).
    tracing_enabled_raw = os.environ.get("TRACING_ENABLED")
    if tracing_enabled_raw is not None:
        settings.tracing.enabled = tracing_enabled_raw.strip().lower() in ("1", "true", "yes")

    classifier_raw = os.environ.get("FINNIE_CLASSIFIER_MODEL")
    if classifier_raw:
        settings.llm.classifier_model = classifier_raw.strip()

    # Per-process override for latency-sensitive callers (the voice worker).
    thinking_raw = os.environ.get("FINNIE_LLM_THINKING")
    if thinking_raw:
        settings.llm.thinking = LLMConfig.model_validate({"thinking": thinking_raw.strip().lower()}).thinking

    # Any voice.* setting as JSON, for local test runs of the worker (simulated
    # calls, the turn-detection sweep): '{"endpointing_min_delay": 0.8}'.
    voice_settings_raw = os.environ.get("FINNIE_VOICE_SETTINGS")
    if voice_settings_raw:
        overrides = json.loads(voice_settings_raw)
        unknown = set(overrides) - set(VoiceConfig.model_fields)
        if unknown:  # a typo would otherwise be ignored and a sweep would measure nothing
            raise ValueError(f"FINNIE_VOICE_SETTINGS: unknown voice settings {sorted(unknown)}")
        settings.voice = VoiceConfig.model_validate({**settings.voice.model_dump(), **overrides})

    # Voice access without editing config.yaml: per deployment, or for local dev.
    voice_enabled_raw = os.environ.get("FINNIE_VOICE_ENABLED")
    if voice_enabled_raw is not None:
        settings.voice.enabled = voice_enabled_raw.strip().lower() in ("1", "true", "yes")
    voice_emails_raw = os.environ.get("FINNIE_VOICE_ALLOWED_EMAILS")
    if voice_emails_raw is not None:
        settings.voice.allowed_emails = [e for e in voice_emails_raw.split(",") if e.strip()]
    settings.voice.allowed_emails = [e.strip().lower() for e in settings.voice.allowed_emails if e.strip()]
    for env_name, field in (("VOICE_TTS_MODEL", "tts_model"), ("VOICE_ELEVEN_VOICE_ID", "tts_voice_id")):
        if os.environ.get(env_name):
            setattr(settings.voice, field, os.environ[env_name].strip())

    return settings
