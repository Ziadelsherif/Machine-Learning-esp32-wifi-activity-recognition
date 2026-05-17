"""
Raw document storage models — represents any ingested source document
before or after processing / OCR / AI extraction.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import DateTime, Float, Index, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from shared.db.postgres import Base


class ProcessingStatus(str, Enum):
    PENDING = "PENDING"
    OCR_PROCESSING = "OCR_PROCESSING"
    AI_EXTRACTING = "AI_EXTRACTING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class DocumentType(str, Enum):
    PROBATE = "PROBATE"
    FORECLOSURE = "FORECLOSURE"
    TAX_DELINQUENT = "TAX_DELINQUENT"
    BANKRUPTCY = "BANKRUPTCY"
    EVICTION = "EVICTION"
    DIVORCE = "DIVORCE"
    UTILITY_SHUTOFF = "UTILITY_SHUTOFF"
    CODE_VIOLATION = "CODE_VIOLATION"
    VACANT_REGISTRATION = "VACANT_REGISTRATION"
    CONTRACTOR_LIEN = "CONTRACTOR_LIEN"
    AUCTION = "AUCTION"
    EXPIRED_LISTING = "EXPIRED_LISTING"
    ESTATE_SALE = "ESTATE_SALE"
    OBITUARY = "OBITUARY"
    UNKNOWN = "UNKNOWN"


class RawDocument(Base):
    """Stores every raw ingested document for audit trail and reprocessing."""

    __tablename__ = "raw_documents"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True
    )

    # Source information
    source_url: Mapped[str] = mapped_column(Text, nullable=False)
    source_type: Mapped[DocumentType] = mapped_column(
        String(32), nullable=False, index=True
    )
    source_name: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    county: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    state: Mapped[str | None] = mapped_column(String(2), nullable=True, index=True)
    fips_code: Mapped[str | None] = mapped_column(String(10), nullable=True, index=True)

    # Raw content storage
    raw_html: Mapped[str | None] = mapped_column(Text, nullable=True)
    raw_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    raw_pdf_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    raw_pdf_size_bytes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    raw_image_paths: Mapped[list[str] | None] = mapped_column(JSONB, nullable=True)

    # OCR output
    ocr_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    ocr_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    ocr_page_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    ocr_structured: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB, nullable=True,
        comment="Structured OCR output with bounding boxes and regions"
    )

    # Processing status
    processing_status: Mapped[ProcessingStatus] = mapped_column(
        String(32), nullable=False, default=ProcessingStatus.PENDING, index=True
    )
    processing_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    processing_attempts: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    last_processing_attempt: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # AI extraction results
    extracted_events: Mapped[list[dict[str, Any]] | None] = mapped_column(
        JSONB, nullable=True,
        comment="Distress events extracted by AI from this document"
    )
    extraction_model: Mapped[str | None] = mapped_column(String(64), nullable=True)
    extraction_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Content metadata
    content_hash: Mapped[str | None] = mapped_column(
        String(64), nullable=True, index=True,
        comment="SHA-256 of raw content for deduplication"
    )
    content_length: Mapped[int | None] = mapped_column(Integer, nullable=True)
    mime_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    language: Mapped[str | None] = mapped_column(String(16), nullable=True)
    page_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    total_pages: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # HTTP metadata
    http_status_code: Mapped[int | None] = mapped_column(Integer, nullable=True)
    request_headers: Mapped[dict[str, str] | None] = mapped_column(JSONB, nullable=True)
    response_headers: Mapped[dict[str, str] | None] = mapped_column(JSONB, nullable=True)

    # Scrape job reference
    scrape_job_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    processed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    __table_args__ = (
        Index("ix_rawdoc_source_type_status", "source_type", "processing_status"),
        Index("ix_rawdoc_county_type", "county", "state", "source_type"),
        Index("ix_rawdoc_content_hash", "content_hash", postgresql_where="content_hash IS NOT NULL"),
        Index("ix_rawdoc_created_pending", "created_at", postgresql_where="processing_status = 'PENDING'"),
    )

    def __repr__(self) -> str:
        return (
            f"<RawDocument id={self.id} type={self.source_type} "
            f"status={self.processing_status} url={self.source_url[:60]!r}>"
        )


# ─── Pydantic Schemas ─────────────────────────────────────────────────────────

class OcrRegion(BaseModel):
    text: str
    confidence: float
    bbox: tuple[int, int, int, int]  # x, y, width, height
    region_type: str  # PARAGRAPH, TABLE, HEADER, FOOTER


class OcrPage(BaseModel):
    page_number: int
    width: int
    height: int
    regions: list[OcrRegion]
    full_text: str
    confidence: float


class OcrResult(BaseModel):
    pages: list[OcrPage]
    full_text: str
    average_confidence: float
    processing_time_ms: int


class RawDocumentCreate(BaseModel):
    source_url: str
    source_type: DocumentType
    source_name: str
    county: str | None = None
    state: str | None = None
    fips_code: str | None = None
    raw_html: str | None = None
    raw_text: str | None = None
    raw_pdf_path: str | None = None
    content_hash: str | None = None
    mime_type: str | None = None
    scrape_job_id: str | None = None
    http_status_code: int | None = None


class RawDocumentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    source_url: str
    source_type: DocumentType
    source_name: str
    county: str | None
    state: str | None
    processing_status: ProcessingStatus
    ocr_confidence: float | None
    extraction_confidence: float | None
    processing_attempts: int
    content_hash: str | None
    created_at: datetime
    processed_at: datetime | None


class DocumentSubmitRequest(BaseModel):
    """Request body for manual document submission via API."""

    source_url: str = Field(..., description="URL of the document")
    source_type: DocumentType
    source_name: str = Field(..., description="Human-readable source name")
    county: str | None = None
    state: str | None = None
    fips_code: str | None = None
    raw_html: str | None = Field(None, description="Pre-fetched HTML content")
    raw_text: str | None = Field(None, description="Pre-extracted text content")
    priority: int = Field(default=5, ge=1, le=10, description="Processing priority 1-10")
    force_reprocess: bool = Field(
        default=False, description="Reprocess even if already completed"
    )
