"""Timezone-safe timestamp semantics shared by the whole request-to-disk chain.

All persisted instants are stored as canonical UTC text (``...Z``).  Inputs may
carry any fixed offset (or ``Z``); they are converted to the equivalent UTC
instant, which is a reversible transformation because the absolute point in time
is preserved.  Naive datetimes are rejected everywhere rather than guessed at.
"""

from __future__ import annotations

from datetime import UTC, datetime

from trailforge.domain.enums import RiskLevel

SECONDS_PER_MINUTE = 60

# Inclusive lower bounds (in whole minutes) for each overdue risk band.
# Being overdue by exactly the boundary minute belongs to the higher band.
OVERDUE_MODERATE_MINUTES = 30
OVERDUE_HIGH_MINUTES = 120
OVERDUE_CRITICAL_MINUTES = 360


class TimestampParseError(ValueError):
    """Raised when persisted text cannot be interpreted as a timezone-aware instant."""


def parse_stored_timestamp(value: str | datetime) -> datetime:
    """Parse text read from the database into a timezone-aware UTC datetime.

    Accepts canonical ``...Z`` text and legacy fixed-offset ISO text.  Naive or
    otherwise unparseable values raise :class:`TimestampParseError` so callers
    can surface an actionable error instead of silently assuming a zone.
    """
    if isinstance(value, datetime):
        parsed = value
    else:
        if not isinstance(value, str) or not value.strip():
            raise TimestampParseError("timestamp must be a non-empty ISO 8601 string")
        text = value.strip()
        try:
            parsed = datetime.fromisoformat(text[:-1] + "+00:00" if text.endswith("Z") else text)
        except ValueError as exc:
            raise TimestampParseError(f"unparseable timestamp {value!r}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise TimestampParseError(
            f"stored timestamp {value!r} carries no timezone offset; fix the row manually"
        )
    return parsed.astimezone(UTC)


def require_utc(value: datetime, *, field: str = "datetime") -> datetime:
    """Return ``value`` as a UTC datetime, rejecting naive input."""
    if not isinstance(value, datetime):
        raise TypeError(f"{field} must be a datetime instance")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must include timezone information")
    return value.astimezone(UTC)


def canonical_text(value: datetime) -> str:
    """Serialize an aware datetime to the canonical UTC text stored in SQLite."""
    normalized = require_utc(value)
    return normalized.isoformat(timespec="microseconds").replace("+00:00", "Z")


def elapsed_minutes(later: datetime, earlier: datetime) -> int:
    """Elapsed whole minutes from ``earlier`` to ``later`` as a floor, clamped at 0.

    Both arguments must be timezone-aware; subtraction is then offset-proof
    (including DST overlap/gap instants).  Flooring positive seconds means
    59.9 seconds is still 0 minutes while 60 seconds is 1; clamping means an
    early punch can never produce a negative minute count.
    """
    delta_seconds = (require_utc(later) - require_utc(earlier)).total_seconds()
    return max(delta_seconds // SECONDS_PER_MINUTE, 0)


def late_minutes(due_at: datetime, checked_in_at: datetime) -> int:
    """Whole minutes a check-in is late (0 when early or on time)."""
    return int(elapsed_minutes(checked_in_at, due_at))


def overdue_risk_level(minutes: int) -> RiskLevel:
    """Map an overdue duration to a risk band with inclusive thresholds.

    ``minutes >= 360`` critical, ``>= 120`` high, ``>= 30`` moderate, otherwise
    low.  The boundary minute itself belongs to the higher band.
    """
    if minutes >= OVERDUE_CRITICAL_MINUTES:
        return RiskLevel.CRITICAL
    if minutes >= OVERDUE_HIGH_MINUTES:
        return RiskLevel.HIGH
    if minutes >= OVERDUE_MODERATE_MINUTES:
        return RiskLevel.MODERATE
    return RiskLevel.LOW
