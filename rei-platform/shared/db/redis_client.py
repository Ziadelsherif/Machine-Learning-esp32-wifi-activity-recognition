"""
Async Redis client with connection pooling.
"""
from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

import structlog
from redis.asyncio import Redis, ConnectionPool
from redis.asyncio.retry import Retry
from redis.backoff import ExponentialBackoff

from shared.config import settings

logger = structlog.get_logger(__name__)

_pool: ConnectionPool | None = None
_client: Redis | None = None


def _build_pool() -> ConnectionPool:
    retry = Retry(ExponentialBackoff(cap=10, base=1), retries=3)
    return ConnectionPool.from_url(
        settings.redis.url,
        max_connections=settings.redis.pool_max_size,
        socket_timeout=settings.redis.socket_timeout,
        socket_connect_timeout=settings.redis.socket_connect_timeout,
        retry=retry,
        retry_on_timeout=settings.redis.retry_on_timeout,
        health_check_interval=settings.redis.health_check_interval,
        decode_responses=True,
    )


def get_redis_pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        _pool = _build_pool()
    return _pool


def get_redis_client() -> Redis:
    global _client
    if _client is None:
        _client = Redis(connection_pool=get_redis_pool())
    return _client


async def get_redis() -> AsyncGenerator[Redis, None]:
    """FastAPI dependency — yields a Redis client."""
    client = get_redis_client()
    try:
        yield client
    finally:
        pass  # Pool is shared; don't close on each request


async def close_redis() -> None:
    global _client, _pool
    if _client is not None:
        await _client.aclose()
        _client = None
    if _pool is not None:
        await _pool.aclose()
        _pool = None
    logger.info("redis_closed")
