"""
Async Neo4j driver wrapper.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

import structlog
from neo4j import AsyncGraphDatabase, AsyncDriver, AsyncSession

from shared.config import settings

logger = structlog.get_logger(__name__)


class Neo4jClient:
    """Thin async wrapper around the official Neo4j async driver."""

    def __init__(self) -> None:
        self._driver: AsyncDriver | None = None

    async def connect(self) -> None:
        self._driver = AsyncGraphDatabase.driver(
            settings.neo4j.uri,
            auth=(settings.neo4j.user, settings.neo4j.password),
            encrypted=settings.neo4j.encrypted,
            max_connection_pool_size=settings.neo4j.max_connection_pool_size,
            connection_timeout=settings.neo4j.connection_timeout,
            max_transaction_retry_time=settings.neo4j.max_transaction_retry_time,
        )
        await self._driver.verify_connectivity()
        logger.info("neo4j_connected", uri=settings.neo4j.uri)

    async def close(self) -> None:
        if self._driver:
            await self._driver.close()
            self._driver = None
            logger.info("neo4j_closed")

    @asynccontextmanager
    async def session(self, **kwargs: Any) -> AsyncGenerator[AsyncSession, None]:
        if not self._driver:
            raise RuntimeError("Neo4jClient not connected — call connect() first")
        async with self._driver.session(
            database=settings.neo4j.database, **kwargs
        ) as session:
            yield session

    async def run(
        self, query: str, parameters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Execute a Cypher query and return all records as dicts."""
        async with self.session() as s:
            result = await s.run(query, parameters or {})
            records = await result.data()
            return records

    async def run_write(
        self, query: str, parameters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Execute a write Cypher query inside a write transaction."""

        async def _tx(tx: Any) -> list[dict[str, Any]]:
            result = await tx.run(query, parameters or {})
            return await result.data()

        async with self.session() as s:
            return await s.execute_write(_tx)

    async def run_batch(
        self, queries: list[tuple[str, dict[str, Any]]]
    ) -> None:
        """Execute multiple write queries in a single transaction."""

        async def _tx(tx: Any) -> None:
            for query, params in queries:
                await tx.run(query, params)

        async with self.session() as s:
            await s.execute_write(_tx)


# Module-level singleton
_neo4j_client: Neo4jClient | None = None


def get_neo4j_client() -> Neo4jClient:
    global _neo4j_client
    if _neo4j_client is None:
        _neo4j_client = Neo4jClient()
    return _neo4j_client
