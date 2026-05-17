"""
SQLAlchemy ORM + Pydantic models for PropertyRecord.
Includes PostGIS geometry column for spatial queries.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any

from geoalchemy2 import Geometry
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import (
    BigInteger,
    Column,
    DateTime,
    Float,
    Index,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from shared.db.postgres import Base


class OwnerType(str, Enum):
    INDIVIDUAL = "INDIVIDUAL"
    LLC = "LLC"
    TRUST = "TRUST"
    CORPORATION = "CORPORATION"
    ESTATE = "ESTATE"
    UNKNOWN = "UNKNOWN"


class PropertyStatus(str, Enum):
    ACTIVE = "ACTIVE"
    DISTRESSED = "DISTRESSED"
    IN_FORECLOSURE = "IN_FORECLOSURE"
    BANK_OWNED = "BANK_OWNED"
    VACANT = "VACANT"
    SOLD = "SOLD"


class PropertyRecord(Base):
    """Canonical property record with enrichment scores."""

    __tablename__ = "property_records"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True
    )
    parcel_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    fips_code: Mapped[str | None] = mapped_column(String(10), nullable=True, index=True)

    # Address
    address_raw: Mapped[str] = mapped_column(Text, nullable=False)
    address_normalized: Mapped[str | None] = mapped_column(Text, nullable=True)
    street_number: Mapped[str | None] = mapped_column(String(20), nullable=True)
    street_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    street_suffix: Mapped[str | None] = mapped_column(String(32), nullable=True)
    unit: Mapped[str | None] = mapped_column(String(32), nullable=True)
    city: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    state: Mapped[str] = mapped_column(String(2), nullable=False, index=True)
    zip_code: Mapped[str] = mapped_column(String(10), nullable=False, index=True)
    county: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)

    # Geospatial
    lat: Mapped[float | None] = mapped_column(Float, nullable=True)
    lng: Mapped[float | None] = mapped_column(Float, nullable=True)
    geom = Column(
        Geometry(geometry_type="POINT", srid=4326),
        nullable=True,
        index=True,  # GiST index created by GeoAlchemy2
    )

    # Ownership
    owner_name: Mapped[str | None] = mapped_column(String(512), nullable=True, index=True)
    owner_name_normalized: Mapped[str | None] = mapped_column(String(512), nullable=True)
    owner_type: Mapped[OwnerType] = mapped_column(
        String(32), nullable=False, default=OwnerType.UNKNOWN
    )
    owner_mailing_address: Mapped[str | None] = mapped_column(Text, nullable=True)
    owner_phone: Mapped[str | None] = mapped_column(String(20), nullable=True)
    owner_email: Mapped[str | None] = mapped_column(String(255), nullable=True)

    # Valuation
    assessed_value: Mapped[float | None] = mapped_column(Numeric(14, 2), nullable=True)
    market_value: Mapped[float | None] = mapped_column(Numeric(14, 2), nullable=True)
    last_sale_price: Mapped[float | None] = mapped_column(Numeric(14, 2), nullable=True)
    last_sale_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    mortgage_balance: Mapped[float | None] = mapped_column(Numeric(14, 2), nullable=True)
    equity_estimate: Mapped[float | None] = mapped_column(Numeric(14, 2), nullable=True)
    equity_percentage: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Property characteristics
    property_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    year_built: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    square_footage: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    lot_size_sqft: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    bedrooms: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    bathrooms: Mapped[float | None] = mapped_column(Float, nullable=True)
    garage_spaces: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    pool: Mapped[bool | None] = mapped_column(nullable=True)

    # Status and scores
    status: Mapped[PropertyStatus] = mapped_column(
        String(32), nullable=False, default=PropertyStatus.ACTIVE, index=True
    )
    distress_score: Mapped[float | None] = mapped_column(
        Float, nullable=True, index=True,
        comment="0-100 composite distress score"
    )
    vacancy_probability: Mapped[float | None] = mapped_column(
        Float, nullable=True,
        comment="0-1 probability property is vacant"
    )
    motivation_score: Mapped[float | None] = mapped_column(
        Float, nullable=True,
        comment="0-100 seller motivation score"
    )
    equity_score: Mapped[float | None] = mapped_column(
        Float, nullable=True,
        comment="0-100 equity-based opportunity score"
    )

    # Active distress signals (denormalized for fast querying)
    active_foreclosure: Mapped[bool] = mapped_column(default=False, nullable=False)
    active_tax_lien: Mapped[bool] = mapped_column(default=False, nullable=False)
    active_probate: Mapped[bool] = mapped_column(default=False, nullable=False)
    active_bankruptcy: Mapped[bool] = mapped_column(default=False, nullable=False)
    active_code_violation: Mapped[bool] = mapped_column(default=False, nullable=False)
    distress_event_count: Mapped[int] = mapped_column(default=0, nullable=False)

    # Metadata
    data_sources: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    extra_data: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    data_quality_score: Mapped[float | None] = mapped_column(Float, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
    last_scraped_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    __table_args__ = (
        UniqueConstraint("parcel_id", "fips_code", name="uq_property_parcel_fips"),
        Index("ix_property_distress_score", "distress_score", postgresql_where="distress_score IS NOT NULL"),
        Index("ix_property_state_city", "state", "city"),
        Index("ix_property_owner_name", "owner_name_normalized"),
        Index("ix_property_status_distress", "status", "distress_score"),
        Index("ix_property_active_signals", "active_foreclosure", "active_tax_lien", "active_probate"),
    )

    def __repr__(self) -> str:
        return f"<PropertyRecord id={self.id} parcel={self.parcel_id} address={self.address_raw!r}>"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": str(self.id),
            "parcel_id": self.parcel_id,
            "fips_code": self.fips_code,
            "address_raw": self.address_raw,
            "address_normalized": self.address_normalized,
            "city": self.city,
            "state": self.state,
            "zip_code": self.zip_code,
            "county": self.county,
            "lat": self.lat,
            "lng": self.lng,
            "owner_name": self.owner_name,
            "owner_type": self.owner_type,
            "assessed_value": float(self.assessed_value) if self.assessed_value else None,
            "market_value": float(self.market_value) if self.market_value else None,
            "equity_estimate": float(self.equity_estimate) if self.equity_estimate else None,
            "distress_score": self.distress_score,
            "vacancy_probability": self.vacancy_probability,
            "motivation_score": self.motivation_score,
            "status": self.status,
            "active_foreclosure": self.active_foreclosure,
            "active_tax_lien": self.active_tax_lien,
            "active_probate": self.active_probate,
            "active_bankruptcy": self.active_bankruptcy,
            "distress_event_count": self.distress_event_count,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


# ─── Pydantic Schemas ─────────────────────────────────────────────────────────

class PropertyRecordBase(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    parcel_id: str
    fips_code: str | None = None
    address_raw: str
    city: str
    state: str = Field(..., min_length=2, max_length=2)
    zip_code: str
    county: str | None = None
    lat: float | None = None
    lng: float | None = None
    owner_name: str | None = None
    owner_type: OwnerType = OwnerType.UNKNOWN
    owner_mailing_address: str | None = None
    assessed_value: float | None = None
    market_value: float | None = None
    last_sale_price: float | None = None
    last_sale_date: datetime | None = None
    mortgage_balance: float | None = None
    equity_estimate: float | None = None
    equity_percentage: float | None = None
    property_type: str | None = None
    year_built: int | None = None
    square_footage: int | None = None
    bedrooms: int | None = None
    bathrooms: float | None = None
    status: PropertyStatus = PropertyStatus.ACTIVE
    extra_data: dict[str, Any] | None = None

    @field_validator("state")
    @classmethod
    def uppercase_state(cls, v: str) -> str:
        return v.upper()

    @field_validator("zip_code")
    @classmethod
    def validate_zip(cls, v: str) -> str:
        clean = v.strip().replace("-", "")
        if not (len(clean) == 5 or len(clean) == 9) or not clean.isdigit():
            raise ValueError(f"Invalid ZIP code: {v}")
        if len(clean) == 9:
            return f"{clean[:5]}-{clean[5:]}"
        return clean


class PropertyRecordCreate(PropertyRecordBase):
    pass


class PropertyRecordUpdate(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    address_raw: str | None = None
    city: str | None = None
    state: str | None = None
    zip_code: str | None = None
    owner_name: str | None = None
    owner_type: OwnerType | None = None
    assessed_value: float | None = None
    market_value: float | None = None
    equity_estimate: float | None = None
    distress_score: float | None = None
    vacancy_probability: float | None = None
    motivation_score: float | None = None
    status: PropertyStatus | None = None
    active_foreclosure: bool | None = None
    active_tax_lien: bool | None = None
    active_probate: bool | None = None
    active_bankruptcy: bool | None = None
    extra_data: dict[str, Any] | None = None


class PropertyRecordResponse(PropertyRecordBase):
    id: uuid.UUID
    address_normalized: str | None = None
    distress_score: float | None = None
    vacancy_probability: float | None = None
    motivation_score: float | None = None
    equity_score: float | None = None
    active_foreclosure: bool = False
    active_tax_lien: bool = False
    active_probate: bool = False
    active_bankruptcy: bool = False
    active_code_violation: bool = False
    distress_event_count: int = 0
    data_quality_score: float | None = None
    created_at: datetime
    updated_at: datetime
    last_scraped_at: datetime | None = None
