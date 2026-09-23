from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import MetaData
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.types import String, TypeDecorator

from trailforge.domain.time import (
    TimestampParseError,
    canonical_text,
    parse_stored_timestamp,
)

NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)


class UTCDateTime(TypeDecorator[datetime]):
    """Aware datetime stored as canonical UTC text (``...Z``).

    Binding converts any timezone-aware value to its equivalent UTC instant,
    which is reversible: reading it back yields the same absolute moment. Naive
    values are rejected. Decoding accepts canonical text and legacy fixed-offset
    text but refuses naive/garbage values with :class:`TimestampParseError`.
    """

    impl = String(32)
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect: Any) -> str | None:
        if value is None:
            return None
        return canonical_text(value)

    def process_result_value(self, value: str | None, dialect: Any) -> datetime | None:
        if value is None:
            return None
        try:
            return parse_stored_timestamp(value)
        except TimestampParseError:
            raise


def utc_now() -> datetime:
    return datetime.now(UTC)
