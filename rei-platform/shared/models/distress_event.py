"""
SQLAlchemy + Pydantic models for distress events.
Each event represents a single public-record distress signal tied to a property.
"""
from __future__ import annotations

import uuid
from datetime import date, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import DateTime, Float, ForeignKey, Index, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from shared.db.postgres import Base


class DistressEventType(str, Enum):
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


class DistressEventStatus(str, Enum):
    ACTIVE = "ACTIVE"
    RESOLVED = "RESOLVED"
    PENDING = "PENDING"
    DISMISSED = "DISMISSED"
    UNKNOWN = "UNKNOWN"


class ForeclosureStage(str, Enum):
    PRE_FORECLOSURE = "PRE_FORECLOSURE"
    NOD_FILED = "NOD_FILED"
    LIS_PENDENS = "LIS_PENDENS"
    AUCTION_SCHEDULED = "AUCTION_SCHEDULED"
    REO = "REO"


# Weight map for distress score contribution
EVENT_TYPE_WEIGHTS: dict[DistressEventType, float] = {
    DistressEventType.FORECLOSURE: 35.0,
    DistressEventType.TAX_DELINQUENT: 25.0,
    DistressEventType.BANKRUPTCY: 20.0,
    DistressEventType.PROBATE: 18.0,
    DistressEventType.AUCTION: 30.0,
    DistressEventType.DIVORCE: 12.0,
    DistressEventType.EVICTION: 15.0,
    DistressEventType.CODE_VIOLATION: 10.0,
    DistressEventType.CONTRACTOR_LIEN: 8.0,
    DistressEventType.VACANT_REGISTRATION: 20.0,
    DistressEventType.UTILITY_SHUTOFF: 12.0,
    DistressEventType.ESTATE_SALE: 18.0,
    DistressEventType.EXPIRED_LISTING: 10.0,
    DistressEventType.OBITUARY: 15.0,
}


class DistressEvent(Base):
    """A single distress signal event linked to a property."""

    __tablename__ = "distress_events"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True
    )
    property_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("property_records.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    # Event classification
    event_type: Mapped[DistressEventType] = mapped_column(
        String(32), nullable=False, index=True
    )
    event_subtype: Mapped[str | None] = mapped_column(String(64), nullable=True)
    status: Mapped[DistressEventStatus] = mapped_column(
        String(32), nullable=False, default=DistressEventStatus.ACTIVE, index=True
    )

    # Source tracking
    source: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    source_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_document_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("raw_documents.id", ondelete="SET NULL"),
        nullable=True,
    )
    county: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    state: Mapped[str | None] = mapped_column(String(2), nullable=True, index=True)
    fips_code: Mapped[str | None] = mapped_column(String(10), nullable=True, index=True)

    # Raw and extracted data
    raw_data: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    extracted_data: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)

    # Scoring
    confidence_score: Mapped[float | None] = mapped_column(
        Float, nullable=True,
        comment="0-1 confidence in extraction accuracy"
    )
    distress_weight: Mapped[float | None] = mapped_column(
        Float, nullable=True,
        comment="Contribution to property distress score"
    )

    # Key dates
    filing_date: Mapped[date | None] = mapped_column(nullable=True, index=True)
    recorded_date: Mapped[date | None] = mapped_column(nullable=True)
    hearing_date: Mapped[date | None] = mapped_column(nullable=True)
    sale_date: Mapped[date | None] = mapped_column(nullable=True, index=True)
    expiration_date: Mapped[date | None] = mapped_column(nullable=True)

    # Legal identifiers
    case_number: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    document_number: Mapped[str | None] = mapped_column(String(128), nullable=True)
    book_page: Mapped[str | None] = mapped_column(String(64), nullable=True)
    instrument_number: Mapped[str | None] = mapped_column(String(128), nullable=True)

    # Parties (plaintiff, defendant, attorney, trustee, etc.)
    parties: Mapped[list[dict[str, Any]] | None] = mapped_column(JSONB, nullable=True)

    # Financial data
    amount: Mapped[float | None] = mapped_column(Float, nullable=True)
    amount_currency: Mapped[str] = mapped_column(String(3), default="USD", nullable=False)
    opening_bid: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Property address from filing (may differ from canonical)
    property_address_raw: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Metadata
    is_duplicate: Mapped[bool] = mapped_column(default=False, nullable=False)
    duplicate_of: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), nullable=True
    )
    manually_verified: Mapped[bool] = mapped_column(default=False, nullable=False)
    verified_by: Mapped[str | None] = mapped_column(String(128), nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # Relationships
    property: Mapped["PropertyRecord | None"] = relationship(  # type: ignore[name-defined]
        "PropertyRecord",
        back_populates=None,
        foreign_keys=[property_id],
    )

    __table_args__ = (
        Index("ix_distress_event_type_status", "event_type", "status"),
        Index("ix_distress_event_property_type", "property_id", "event_type"),
        Index("ix_distress_event_filing_date", "filing_date"),
        Index("ix_distress_event_sale_date", "sale_date"),
        Index("ix_distress_event_county_type", "county", "state", "event_type"),
        Index("ix_distress_event_case_number", "case_number", postgresql_where="case_number IS NOT NULL"),
    )

    def __repr__(self) -> str:
        return (
            f"<DistressEvent id={self.id} type={self.event_type} "
            f"property_id={self.property_id} case={self.case_number!r}>"
        )

    @property
    def computed_distress_weight(self) -> float:
        base = EVENT_TYPE_WEIGHTS.get(self.event_type, 5.0)
        confidence = self.confidence_score or 1.0
        return round(base * confidence, 2)


# ─── Pydantic Schemas ─────────────────────────────────────────────────────────

class PartySchema(BaseModel):
    role: str  # PLAINTIFF, DEFENDANT, ATTORNEY, TRUSTEE, DEBTOR, CREDITOR, etc.
    name: str
    address: str | None = None
    phone: str | None = None
    email: str | None = None
    bar_number: str | None = None
    entity_type: str | None = None


class DistressEventBase(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    event_type: DistressEventType
    event_subtype: str | None = None
    status: DistressEventStatus = DistressEventStatus.ACTIVE
    source: str
    source_url: str | None = None
    county: str | None = None
    state: str | None = None
    fips_code: str | None = None
    raw_data: dict[str, Any] | None = None
    extracted_data: dict[str, Any] | None = None
    confidence_score: float | None = Field(None, ge=0.0, le=1.0)
    filing_date: date | None = None
    recorded_date: date | None = None
    hearing_date: date | None = None
    sale_date: date | None = None
    case_number: str | None = None
    document_number: str | None = None
    parties: list[PartySchema] | None = None
    amount: float | None = None
    opening_bid: float | None = None
    property_address_raw: str | None = None
    notes: str | None = None


class DistressEventCreate(DistressEventBase):
    property_id: uuid.UUID | None = None
    source_document_id: uuid.UUID | None = None


class DistressEventResponse(DistressEventBase):
    id: uuid.UUID
    property_id: uuid.UUID | None = None
    distress_weight: float | None = None
    is_duplicate: bool = False
    manually_verified: bool = False
    created_at: datetime
    updated_at: datetime


class DistressEventSummary(BaseModel):
    """Lightweight summary for API list responses."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    property_id: uuid.UUID | None
    event_type: DistressEventType
    status: DistressEventStatus
    source: str
    filing_date: date | None
    case_number: str | None
    confidence_score: float | None
    distress_weight: float | None
    created_at: datetime
