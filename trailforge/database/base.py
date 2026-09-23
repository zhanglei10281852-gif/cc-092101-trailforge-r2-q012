from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import MetaData
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.types import String, TypeDecorator

from trailforge.timekeeping import parse_timestamp_text, to_utc_text, utc_now

__all__ = ["Base", "UTCDateTime", "utc_now"]

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
    """可逆的带时区时间列：写入归一化为 UTC 文本，读取必须带显式偏移。

    时间语义约定见 trailforge.timekeeping；无时区的存量文本会在读取时
    抛出 UnparseableTimestampError，而不是按服务器本地时区猜测。
    """

    impl = String(32)
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect: Any) -> str | None:
        if value is None:
            return None
        return to_utc_text(value)

    def process_result_value(self, value: str | None, dialect: Any) -> datetime | None:
        if value is None:
            return None
        return parse_timestamp_text(value)
