"""
Central configuration for the REI Platform.
All values are read from environment variables with sensible defaults.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DB_", extra="ignore")

    host: str = Field(default="localhost", description="PostgreSQL host")
    port: int = Field(default=5432, description="PostgreSQL port")
    name: str = Field(default="rei_platform", description="Database name")
    user: str = Field(default="rei_user", description="Database user")
    password: str = Field(default="rei_password", description="Database password")
    pool_size: int = Field(default=20, description="Connection pool size")
    max_overflow: int = Field(default=10, description="Max pool overflow")
    pool_timeout: int = Field(default=30, description="Pool timeout seconds")
    pool_recycle: int = Field(default=3600, description="Connection recycle seconds")
    echo_sql: bool = Field(default=False, description="Echo SQL statements")

    @property
    def async_dsn(self) -> str:
        return (
            f"postgresql+asyncpg://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.name}"
        )

    @property
    def sync_dsn(self) -> str:
        return (
            f"postgresql+psycopg2://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.name}"
        )


class RedisConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="REDIS_", extra="ignore")

    host: str = Field(default="localhost", description="Redis host")
    port: int = Field(default=6379, description="Redis port")
    password: str | None = Field(default=None, description="Redis password")
    db: int = Field(default=0, description="Redis database index")
    pool_min_size: int = Field(default=5, description="Min connection pool size")
    pool_max_size: int = Field(default=50, description="Max connection pool size")
    socket_timeout: float = Field(default=5.0, description="Socket timeout seconds")
    socket_connect_timeout: float = Field(default=5.0, description="Connect timeout")
    retry_on_timeout: bool = Field(default=True)
    health_check_interval: int = Field(default=30, description="Health check interval")

    @property
    def url(self) -> str:
        auth = f":{self.password}@" if self.password else ""
        return f"redis://{auth}{self.host}:{self.port}/{self.db}"

    @property
    def celery_broker_url(self) -> str:
        return self.url


class KafkaConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="KAFKA_", extra="ignore")

    bootstrap_servers: str = Field(
        default="localhost:9092", description="Kafka bootstrap servers"
    )
    security_protocol: str = Field(default="PLAINTEXT")
    sasl_mechanism: str | None = Field(default=None)
    sasl_username: str | None = Field(default=None)
    sasl_password: str | None = Field(default=None)
    ssl_cafile: str | None = Field(default=None)

    # Producer settings
    producer_acks: str = Field(default="all", description="Producer acks setting")
    producer_retries: int = Field(default=5)
    producer_batch_size: int = Field(default=16384)
    producer_linger_ms: int = Field(default=10)
    producer_compression: str = Field(default="snappy")

    # Consumer settings
    consumer_group_id: str = Field(default="rei-platform-ingestion")
    consumer_auto_offset_reset: str = Field(default="earliest")
    consumer_enable_auto_commit: bool = Field(default=False)
    consumer_max_poll_records: int = Field(default=500)
    consumer_session_timeout_ms: int = Field(default=30000)
    consumer_heartbeat_interval_ms: int = Field(default=10000)

    # Topics
    topics: dict[str, str] = Field(
        default={
            "distress_events": "rei.distress.events",
            "documents_ingested": "rei.documents.ingested",
            "properties_updated": "rei.properties.updated",
            "leads_scored": "rei.leads.scored",
            "ingestion_commands": "rei.ingestion.commands",
            "dead_letter": "rei.dead_letter",
        }
    )

    @property
    def bootstrap_servers_list(self) -> list[str]:
        return [s.strip() for s in self.bootstrap_servers.split(",")]


class Neo4jConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="NEO4J_", extra="ignore")

    uri: str = Field(default="bolt://localhost:7687", description="Neo4j URI")
    user: str = Field(default="neo4j", description="Neo4j user")
    password: str = Field(default="neo4j_password", description="Neo4j password")
    database: str = Field(default="neo4j", description="Neo4j database")
    max_connection_pool_size: int = Field(default=50)
    connection_timeout: float = Field(default=30.0)
    max_transaction_retry_time: float = Field(default=30.0)
    encrypted: bool = Field(default=False)


class ElasticsearchConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ES_", extra="ignore")

    hosts: str = Field(
        default="http://localhost:9200", description="Elasticsearch hosts (comma-sep)"
    )
    username: str | None = Field(default=None)
    password: str | None = Field(default=None)
    api_key: str | None = Field(default=None)
    verify_certs: bool = Field(default=True)
    ca_certs: str | None = Field(default=None)
    timeout: int = Field(default=30)
    max_retries: int = Field(default=3)
    retry_on_timeout: bool = Field(default=True)

    # Index names
    properties_index: str = Field(default="rei.properties")
    leads_index: str = Field(default="rei.leads")
    documents_index: str = Field(default="rei.documents")
    distress_events_index: str = Field(default="rei.distress_events")

    @property
    def hosts_list(self) -> list[str]:
        return [h.strip() for h in self.hosts.split(",")]


class ScraperConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SCRAPER_", extra="ignore")

    proxy_pool: str = Field(
        default="",
        description="Comma-separated proxy URLs (user:pass@host:port)",
    )
    user_agent_rotation: bool = Field(default=True)
    headless: bool = Field(default=True)
    playwright_timeout_ms: int = Field(default=30000)
    page_load_timeout_ms: int = Field(default=60000)
    rate_limit_rps: float = Field(default=1.0, description="Requests per second")
    rate_limit_burst: int = Field(default=5)
    max_retries: int = Field(default=3)
    retry_base_delay: float = Field(default=2.0)
    retry_max_delay: float = Field(default=60.0)

    @property
    def proxy_list(self) -> list[str]:
        if not self.proxy_pool:
            return []
        return [p.strip() for p in self.proxy_pool.split(",") if p.strip()]


class OcrConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="OCR_", extra="ignore")

    tesseract_path: str = Field(default="/usr/bin/tesseract")
    tesseract_lang: str = Field(default="eng")
    dpi: int = Field(default=300)
    enable_gpu: bool = Field(default=False)
    batch_size: int = Field(default=10)
    confidence_threshold: float = Field(default=0.6)
    pdf_storage_path: str = Field(default="/data/pdfs")
    processed_storage_path: str = Field(default="/data/processed")


class AppConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="APP_", extra="ignore")

    env: str = Field(default="development", description="Environment name")
    debug: bool = Field(default=False)
    secret_key: str = Field(
        default="change-this-in-production-use-a-long-random-string"
    )
    api_title: str = Field(default="REI Platform Ingestion API")
    api_version: str = Field(default="1.0.0")
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    api_workers: int = Field(default=4)
    log_level: str = Field(default="INFO")
    log_format: str = Field(default="json")
    sentry_dsn: str | None = Field(default=None)
    allowed_origins: list[str] = Field(
        default=["http://localhost:3000", "http://localhost:8080"]
    )

    @field_validator("env")
    @classmethod
    def validate_env(cls, v: str) -> str:
        allowed = {"development", "staging", "production", "test"}
        if v not in allowed:
            raise ValueError(f"env must be one of {allowed}")
        return v

    @property
    def is_production(self) -> bool:
        return self.env == "production"

    @property
    def is_development(self) -> bool:
        return self.env == "development"


class Settings(BaseSettings):
    """Top-level settings container — loads all sub-configs."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app: AppConfig = Field(default_factory=AppConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    kafka: KafkaConfig = Field(default_factory=KafkaConfig)
    neo4j: Neo4jConfig = Field(default_factory=Neo4jConfig)
    elasticsearch: ElasticsearchConfig = Field(default_factory=ElasticsearchConfig)
    scraper: ScraperConfig = Field(default_factory=ScraperConfig)
    ocr: OcrConfig = Field(default_factory=OcrConfig)

    def as_dict(self) -> dict[str, Any]:
        return self.model_dump()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    return Settings()


# Convenience alias
settings = get_settings()
