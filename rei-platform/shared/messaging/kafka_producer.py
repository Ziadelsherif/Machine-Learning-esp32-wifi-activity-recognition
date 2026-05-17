"""
Async Kafka producer with retry, serialization, and topic routing.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any

import structlog
from aiokafka import AIOKafkaProducer
from aiokafka.errors import KafkaError

from shared.config import settings

logger = structlog.get_logger(__name__)


def _json_serializer(value: Any) -> bytes:
    def _default(obj: Any) -> Any:
        if isinstance(obj, datetime):
            return obj.isoformat()
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    return json.dumps(value, default=_default).encode("utf-8")


class KafkaProducer:
    """Async Kafka producer wrapping aiokafka."""

    def __init__(self) -> None:
        self._producer: AIOKafkaProducer | None = None

    async def start(self) -> None:
        kafka_cfg = settings.kafka
        self._producer = AIOKafkaProducer(
            bootstrap_servers=kafka_cfg.bootstrap_servers_list,
            value_serializer=_json_serializer,
            key_serializer=lambda k: k.encode("utf-8") if k else None,
            acks=kafka_cfg.producer_acks,
            compression_type=kafka_cfg.producer_compression,
            max_batch_size=kafka_cfg.producer_batch_size,
            linger_ms=kafka_cfg.producer_linger_ms,
            retry_backoff_ms=200,
            request_timeout_ms=30000,
        )
        await self._producer.start()
        logger.info(
            "kafka_producer_started",
            servers=kafka_cfg.bootstrap_servers,
        )

    async def stop(self) -> None:
        if self._producer:
            await self._producer.stop()
            self._producer = None
            logger.info("kafka_producer_stopped")

    async def publish(
        self,
        topic_key: str,
        payload: dict[str, Any],
        key: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        """Publish a message to a named topic (resolved via settings)."""
        if not self._producer:
            raise RuntimeError("KafkaProducer not started")

        topic = settings.kafka.topics.get(topic_key, topic_key)
        envelope = {
            "event_id": str(uuid.uuid4()),
            "event_time": datetime.now(timezone.utc).isoformat(),
            "topic": topic,
            "payload": payload,
        }
        kafka_headers = []
        if headers:
            kafka_headers = [
                (k, v.encode("utf-8")) for k, v in headers.items()
            ]

        try:
            await self._producer.send_and_wait(
                topic,
                value=envelope,
                key=key,
                headers=kafka_headers,
            )
            logger.debug("kafka_message_published", topic=topic, key=key)
        except KafkaError as exc:
            logger.error(
                "kafka_publish_failed",
                topic=topic,
                error=str(exc),
                exc_info=True,
            )
            raise

    async def publish_distress_event(
        self, event_data: dict[str, Any], property_id: str
    ) -> None:
        await self.publish(
            "distress_events",
            event_data,
            key=property_id,
            headers={"event_type": "distress_event"},
        )

    async def publish_document_processed(
        self, doc_data: dict[str, Any], document_id: str
    ) -> None:
        await self.publish(
            "documents_ingested",
            doc_data,
            key=document_id,
            headers={"event_type": "document_processed"},
        )


# Module-level singleton
_kafka_producer: KafkaProducer | None = None


def get_kafka_producer() -> KafkaProducer:
    global _kafka_producer
    if _kafka_producer is None:
        _kafka_producer = KafkaProducer()
    return _kafka_producer
