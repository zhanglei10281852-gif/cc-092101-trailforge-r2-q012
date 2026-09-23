"""End-to-end timezone semantics for safety check-ins.

Covers request validation, SQLite encoding, repository filtering and service
computation with positive/negative offsets, DST overlap (fall-back) and gap
(spring-forward) instants, one-second threshold boundaries, restart
persistence, legacy data migration and consistency across the overdue query,
summary, risk statistics and audit log entry points.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest
from sqlalchemy import func, select, text

from tests.conftest import create_route, create_user
from trailforge.config import Settings
from trailforge.database.migrations import initialize_database
from trailforge.database.session import Database
from trailforge.domain.enums import RiskLevel
from trailforge.domain.time import (
    elapsed_minutes,
    late_minutes,
    overdue_risk_level,
)
from trailforge.errors import (
    IdempotencyConflictError,
    TimestampMigrationError,
    ValidationError,
)
from trailforge.models.audit import AuditLog, SchemaMigration
from trailforge.models.safety import ItineraryCheckIn
from trailforge.schemas.activities import ExpeditionCreate
from trailforge.schemas.safety import CheckInScheduleCreate, CheckInSubmit
from trailforge.services.activities import ExpeditionService
from trailforge.services.safety import SafetyService
from trailforge.services.statistics import StatisticsService

NY = ZoneInfo("America/New_York")


# --------------------------------------------------------------------------- #
# Pure time semantics
# --------------------------------------------------------------------------- #


def test_elapsed_minutes_floors_positive_and_clamps_negative() -> None:
    base = datetime(2026, 9, 15, 12, 0, 0, tzinfo=UTC)
    assert elapsed_minutes(base + timedelta(seconds=59), base) == 0
    assert elapsed_minutes(base + timedelta(seconds=60), base) == 1
    assert elapsed_minutes(base + timedelta(seconds=119, microseconds=999999), base) == 1
    assert elapsed_minutes(base + timedelta(seconds=120), base) == 2
    # early punches must never become zero-by-floor or negative
    assert elapsed_minutes(base - timedelta(seconds=1), base) == 0
    assert elapsed_minutes(base - timedelta(seconds=30), base) == 0
    assert late_minutes(base, base - timedelta(seconds=59)) == 0


def test_overdue_risk_thresholds_are_inclusive_on_the_upper_band() -> None:
    assert overdue_risk_level(0) is RiskLevel.LOW
    assert overdue_risk_level(29) is RiskLevel.LOW
    assert overdue_risk_level(30) is RiskLevel.MODERATE
    assert overdue_risk_level(119) is RiskLevel.MODERATE
    assert overdue_risk_level(120) is RiskLevel.HIGH
    assert overdue_risk_level(359) is RiskLevel.HIGH
    assert overdue_risk_level(360) is RiskLevel.CRITICAL


# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #


def _windowed_expedition(session, *, center: datetime | None = None) -> int:
    """Create an expedition whose active window straddles ``center`` (default now)."""
    organizer = create_user(session, email="safety@example.com", name="Safety Lead")
    route = create_route(session, actor_id=organizer)
    center = center or datetime.now(UTC)
    expedition = ExpeditionService(session).create(
        ExpeditionCreate(
            organizer_id=organizer,
            route_id=route,
            name="DST Drill",
            meeting_location="Trailhead",
            meeting_at=center - timedelta(hours=2),
            start_at=center - timedelta(hours=1),
            end_at=center + timedelta(hours=6),
            registration_deadline=center - timedelta(days=1),
            capacity=4,
            minimum_fitness_level=1,
            risk_level="moderate",
        )
    )
    session.info["organizer"] = organizer
    return expedition.id


def _schedule(session, expedition_id: int, user_id: int, due_at: datetime, key: str = "routine"):
    return SafetyService(session).schedule_check_in(
        expedition_id,
        CheckInScheduleCreate(user_id=user_id, check_in_type=key, due_at=due_at),
        actor_id=user_id,
    )


def _raw_column(session, row_id: int, column: str) -> str:
    return session.execute(
        text(f"SELECT {column} FROM itinerary_check_ins WHERE id = :id"),
        {"id": row_id},
    ).scalar_one()


# --------------------------------------------------------------------------- #
# Offset and DST encoding
# --------------------------------------------------------------------------- #


def test_positive_and_negative_offsets_encode_to_same_canonical_utc(session) -> None:
    instant = datetime(2026, 9, 15, 6, 30, tzinfo=UTC)
    expedition_id = _windowed_expedition(session, center=instant)
    user_id = session.info["organizer"]
    east = datetime.fromisoformat("2026-09-15T12:00:00+05:30")
    utc_same = datetime.fromisoformat("2026-09-15T06:30:00+00:00")
    west = datetime.fromisoformat("2026-09-14T22:30:00-08:00")
    a = _schedule(session, expedition_id, user_id, east, key="assembly")
    b = _schedule(session, expedition_id, user_id, utc_same, key="departure")
    c = _schedule(session, expedition_id, user_id, west, key="routine")
    raw_a = _raw_column(session, a.id, "due_at")
    raw_b = _raw_column(session, b.id, "due_at")
    raw_c = _raw_column(session, c.id, "due_at")
    assert raw_a == raw_b == raw_c == "2026-09-15T06:30:00.000000Z"
    # reading back gives the identical instant regardless of source offset
    reloaded = session.get(ItineraryCheckIn, a.id)
    assert reloaded.due_at == east
    assert reloaded.due_at.utcoffset() == timedelta(0)


def test_dst_fall_back_overlap_keeps_the_two_instants_distinct(session) -> None:
    center = datetime(2026, 11, 1, 6, 0, tzinfo=UTC)
    expedition_id = _windowed_expedition(session, center=center)
    user_id = session.info["organizer"]
    first = datetime(2026, 11, 1, 1, 30, fold=0, tzinfo=NY)  # 05:30Z (EDT)
    second = datetime(2026, 11, 1, 1, 30, fold=1, tzinfo=NY)  # 06:30Z (EST)
    a = _schedule(session, expedition_id, user_id, first, key="assembly")
    b = _schedule(session, expedition_id, user_id, second, key="departure")
    assert _raw_column(session, a.id, "due_at") == "2026-11-01T05:30:00.000000Z"
    assert _raw_column(session, b.id, "due_at") == "2026-11-01T06:30:00.000000Z"


def test_late_minutes_are_correct_across_fall_back_and_spring_gap(session) -> None:
    center = datetime(2026, 11, 1, 6, 0, tzinfo=UTC)
    expedition_id = _windowed_expedition(session, center=center)
    user_id = session.info["organizer"]
    # fall-back: due 01:00 EDT (05:00Z), punch at 01:45 *EST* clock time (06:45Z)
    due_fall = datetime(2026, 11, 1, 1, 0, fold=0, tzinfo=NY)
    punch_fall = datetime(2026, 11, 1, 1, 45, fold=1, tzinfo=NY)
    assert late_minutes(due_fall, punch_fall) == 105
    # spring gap: due 01:45 EST (06:45Z), punch 03:15 EDT (07:15Z): 30 real minutes
    due_spring = datetime(2026, 3, 8, 1, 45, tzinfo=NY)
    punch_spring = datetime(2026, 3, 8, 3, 15, tzinfo=NY)
    assert late_minutes(due_spring, punch_spring) == 30
    # and the whole persistence + service path preserves it
    check = _schedule(session, expedition_id, user_id, due_fall, key="waypoint")
    submitted = SafetyService(session).submit_check_in(
        check.id,
        CheckInSubmit(
            checked_in_at=punch_fall,
            is_safe=True,
            idempotency_key="fallback-punch",
        ),
        now=punch_fall,
    )
    assert submitted.late_minutes == 105
    assert _raw_column(session, check.id, "checked_in_at") == "2026-11-01T06:45:00.000000Z"


# --------------------------------------------------------------------------- #
# One-second threshold boundaries through the service + repository
# --------------------------------------------------------------------------- #


def test_overdue_boundary_one_second_before_and_after(session) -> None:
    due = datetime(2026, 9, 15, 10, 0, 0, tzinfo=UTC)
    expedition_id = _windowed_expedition(session, center=due)
    user_id = session.info["organizer"]
    check = _schedule(session, expedition_id, user_id, due, key="routine")
    service = SafetyService(session)
    one_second_before = due + timedelta(minutes=30, seconds=-1)
    exactly = due + timedelta(minutes=30)
    before = service.overdue(now=one_second_before)
    after = service.overdue(now=exactly)
    assert before[0].check_in_id == check.id
    assert before[0].overdue_minutes == 29
    assert before[0].risk_level is RiskLevel.LOW
    assert after[0].overdue_minutes == 30
    assert after[0].risk_level is RiskLevel.MODERATE
    # 120 and 360 boundaries likewise
    assert service.overdue(now=due + timedelta(minutes=120))[0].risk_level is RiskLevel.HIGH
    assert service.overdue(now=due + timedelta(minutes=360))[0].risk_level is RiskLevel.CRITICAL


def test_submit_under_one_minute_early_is_zero_never_negative(session) -> None:
    expedition_id = _windowed_expedition(session)
    user_id = session.info["organizer"]
    due = datetime.now(UTC) - timedelta(minutes=10)
    check_a = _schedule(session, expedition_id, user_id, due, key="assembly")
    submitted = SafetyService(session).submit_check_in(
        check_a.id,
        CheckInSubmit(
            checked_in_at=due - timedelta(seconds=45),
            is_safe=True,
            idempotency_key="early-45",
        ),
    )
    assert submitted.late_minutes == 0
    assert int(_raw_column(session, check_a.id, "late_minutes")) == 0


def test_late_submit_seconds_rounding_boundaries(session) -> None:
    expedition_id = _windowed_expedition(session)
    user_id = session.info["organizer"]
    due = datetime.now(UTC) - timedelta(hours=2)
    check = _schedule(session, expedition_id, user_id, due, key="departure")
    submitted = SafetyService(session).submit_check_in(
        check.id,
        CheckInSubmit(
            checked_in_at=due + timedelta(seconds=1799),
            is_safe=True,
            idempotency_key="sec-1799",
        ),
        now=due + timedelta(seconds=1800),
    )
    assert submitted.late_minutes == 29
    check_two = _schedule(session, expedition_id, user_id, due, key="waypoint")
    submitted_two = SafetyService(session).submit_check_in(
        check_two.id,
        CheckInSubmit(
            checked_in_at=due + timedelta(seconds=1800),
            is_safe=True,
            idempotency_key="sec-1800",
        ),
        now=due + timedelta(seconds=1801),
    )
    assert submitted_two.late_minutes == 30


# --------------------------------------------------------------------------- #
# Future / window / duplicate behaviour
# --------------------------------------------------------------------------- #


def test_future_check_in_is_rejected_with_actionable_error(session) -> None:
    expedition_id = _windowed_expedition(session)
    user_id = session.info["organizer"]
    due = datetime.now(UTC) + timedelta(minutes=30)
    check = _schedule(session, expedition_id, user_id, due)
    with pytest.raises(ValidationError, match="future") as exc:
        SafetyService(session).submit_check_in(
            check.id,
            CheckInSubmit(
                checked_in_at=due + timedelta(minutes=1),
                is_safe=True,
                idempotency_key="future-punch",
            ),
        )
    assert "now" in exc.value.context


def test_check_in_outside_expedition_window_is_rejected(session) -> None:
    expedition_id = _windowed_expedition(session)
    user_id = session.info["organizer"]
    check = _schedule(
        session,
        expedition_id,
        user_id,
        datetime.now(UTC) - timedelta(minutes=5),
    )
    expedition = ExpeditionService(session).expeditions.get(expedition_id)
    service = SafetyService(session)
    with pytest.raises(ValidationError, match="window"):
        service.submit_check_in(
            check.id,
            CheckInSubmit(
                checked_in_at=expedition.meeting_at - timedelta(seconds=1),
                is_safe=True,
                idempotency_key="before-window",
            ),
        )
    check_two = _schedule(
        session,
        expedition_id,
        user_id,
        datetime.now(UTC) - timedelta(minutes=4),
        key="departure",
    )
    with pytest.raises(ValidationError, match="window"):
        service.submit_check_in(
            check_two.id,
            CheckInSubmit(
                checked_in_at=expedition.end_at + timedelta(seconds=1),
                is_safe=True,
                idempotency_key="after-window",
            ),
            now=expedition.end_at + timedelta(minutes=1),
        )


def test_duplicate_submit_conflicts_but_idempotent_replay_returns_same(session) -> None:
    expedition_id = _windowed_expedition(session)
    user_id = session.info["organizer"]
    due = datetime.now(UTC) - timedelta(minutes=30)
    check = _schedule(session, expedition_id, user_id, due)
    payload = CheckInSubmit(
        checked_in_at=due + timedelta(minutes=10),
        is_safe=True,
        idempotency_key="replay-key",
    )
    service = SafetyService(session)
    first = service.submit_check_in(check.id, payload)
    replayed = service.submit_check_in(check.id, payload)
    assert first.id == replayed.id
    assert replayed.late_minutes == 10
    # same idempotency key, different body -> conflict
    with pytest.raises(IdempotencyConflictError):
        service.submit_check_in(
            check.id,
            payload.model_copy(update={"is_safe": False}),
        )
    # a fresh key against an already completed check-in -> conflict
    with pytest.raises(Exception, match="already been submitted"):
        service.submit_check_in(
            check.id,
            payload.model_copy(update={"idempotency_key": "brand-new-key"}),
        )
    # exactly one audit row for the submission
    count = session.scalar(
        select(func.count())
        .select_from(AuditLog)
        .where(
            AuditLog.entity_type == "itinerary_check_in",
            AuditLog.entity_id == check.id,
            AuditLog.action == "checked_in",
        )
    )
    assert count == 1


def test_naive_now_override_is_rejected(session) -> None:
    _windowed_expedition(session)
    with pytest.raises(ValidationError, match="timezone"):
        SafetyService(session).overdue(now=datetime(2026, 9, 15, 12, 0, 0))


# --------------------------------------------------------------------------- #
# Consistency across overdue / summary / statistics entry points
# --------------------------------------------------------------------------- #


def test_overdue_summary_and_statistics_agree(session) -> None:
    due = datetime(2026, 9, 15, 10, 0, 0, tzinfo=UTC)
    expedition_id = _windowed_expedition(session, center=due)
    user_id = session.info["organizer"]
    _schedule(session, expedition_id, user_id, due, key="assembly")
    other_due = due + timedelta(hours=2)
    _schedule(session, expedition_id, user_id, other_due, key="departure")
    current = due + timedelta(minutes=30)
    service = SafetyService(session)
    overdue = service.overdue(now=current)
    summary = service.summary(expedition_id, now=current)
    risks = StatisticsService(session).risks(now=current)
    dashboard = StatisticsService(session).dashboard(now=current)
    assert len(overdue) == 1
    assert overdue[0].overdue_minutes == 30
    assert overdue[0].risk_level is RiskLevel.MODERATE
    assert summary.overdue_check_ins == 1
    assert risks.overdue_check_ins == 1
    assert dashboard.overdue_check_ins == 1
    # once the boundary minute is not yet reached, every entry point says zero
    early = due + timedelta(seconds=-1)
    assert service.overdue(now=early) == []
    assert service.summary(expedition_id, now=early).overdue_check_ins == 0
    assert StatisticsService(session).risks(now=early).overdue_check_ins == 0


def test_submit_writes_canonical_audit_state(session) -> None:
    expedition_id = _windowed_expedition(session)
    user_id = session.info["organizer"]
    due = datetime.now(UTC) - timedelta(minutes=20)
    check = _schedule(session, expedition_id, user_id, due)
    SafetyService(session).submit_check_in(
        check.id,
        CheckInSubmit(
            checked_in_at=due + timedelta(minutes=5),
            is_safe=False,
            note="Off route",
            idempotency_key="audit-check",
        ),
    )
    log = session.scalar(
        select(AuditLog).where(
            AuditLog.entity_type == "itinerary_check_in",
            AuditLog.entity_id == check.id,
            AuditLog.action == "checked_in",
        )
    )
    assert log is not None
    assert log.after_state["late_minutes"] == 5
    assert log.after_state["is_safe"] is False
    assert isinstance(log.after_state["checked_in_at"], str)
    assert log.after_state["checked_in_at"].endswith("Z")
    assert log.correlation_id == "audit-check"


# --------------------------------------------------------------------------- #
# Restart persistence
# --------------------------------------------------------------------------- #


def test_values_stay_canonical_and_consistent_across_restart(tmp_path) -> None:
    settings = Settings(database_url=f"sqlite:///{tmp_path / 'restart.db'}")
    db = Database(settings)
    initialize_database(db)
    with db.session() as session:
        center = datetime(2026, 11, 1, 6, 0, tzinfo=UTC)
        eid = _windowed_expedition(session, center=center)
        uid = session.info["organizer"]
        due = datetime(2026, 11, 1, 1, 0, fold=0, tzinfo=NY)
        check = _schedule(session, eid, uid, due)
        SafetyService(session).submit_check_in(
            check.id,
            CheckInSubmit(
                checked_in_at=datetime(2026, 11, 1, 1, 45, fold=1, tzinfo=NY),
                is_safe=True,
                idempotency_key="restart-punch",
            ),
            now=datetime(2026, 11, 1, 1, 45, fold=1, tzinfo=NY),
        )
        check_id = check.id
    db.engine.dispose()

    second = Database(settings)
    assert initialize_database(second) == []  # 0002 already recorded
    with second.session() as session:
        rows = session.execute(
            text(
                "SELECT due_at, checked_in_at FROM itinerary_check_ins WHERE id = :id"
            ),
            {"id": check_id},
        ).one()
        assert rows.due_at == "2026-11-01T05:00:00.000000Z"
        assert rows.checked_in_at == "2026-11-01T06:45:00.000000Z"
        overdue = SafetyService(session).overdue(
            now=datetime(2026, 11, 1, 1, 45, fold=1, tzinfo=NY)
        )
        assert overdue == []  # already checked in
        record = session.get(ItineraryCheckIn, check_id)
        assert record.late_minutes == 105
    second.engine.dispose()


# --------------------------------------------------------------------------- #
# Legacy data migration (0002)
# --------------------------------------------------------------------------- #


def _seed_legacy_database(tmp_path, *, raw_due: str) -> tuple[Settings, int]:
    settings = Settings(database_url=f"sqlite:///{tmp_path / 'legacy.db'}")
    db = Database(settings)
    db.create_schema()
    # mark only the initial migration as applied, so 0002 is pending
    with db.session() as session:
        session.add(SchemaMigration(version="0001", description="Initial TrailForge schema"))
        due = datetime(2026, 9, 15, 10, 0, tzinfo=UTC)
        eid = _windowed_expedition(session, center=due)
        uid = session.info["organizer"]
        check = _schedule(session, eid, uid, due)
        check_id = check.id
    db.engine.dispose()

    db = Database(settings)
    with db.session() as session:
        session.execute(
            text("UPDATE itinerary_check_ins SET due_at = :value WHERE id = :id"),
            {"value": raw_due, "id": check_id},
        )
    db.engine.dispose()
    return settings, check_id


def test_migration_normalizes_legacy_offset_text(tmp_path) -> None:
    settings, check_id = _seed_legacy_database(
        tmp_path, raw_due="2026-09-15T05:00:00-05:00"
    )
    db = Database(settings)
    applied = initialize_database(db)
    assert "0002" in applied
    with db.session() as session:
        raw = session.execute(
            text("SELECT due_at FROM itinerary_check_ins WHERE id = :id"),
            {"id": check_id},
        ).scalar_one()
        assert raw == "2026-09-15T10:00:00.000000Z"
    # second startup is a no-op
    assert initialize_database(db) == []
    db.engine.dispose()


def test_migration_naive_text_raises_actionable_error_and_changes_nothing(tmp_path) -> None:
    settings, check_id = _seed_legacy_database(
        tmp_path, raw_due="2026-09-15T10:00:00"  # no offset
    )
    db = Database(settings)
    with pytest.raises(TimestampMigrationError) as exc:
        initialize_database(db)
    detail = exc.value.context
    assert detail["failure_count"] == 1
    invalid = detail["invalid_rows"][0]
    assert invalid["table"] == "itinerary_check_ins"
    assert invalid["column"] == "due_at"
    assert invalid["row_id"] == check_id
    assert invalid["value"] == "2026-09-15T10:00:00"
    db.engine.dispose()

    # the bad row is untouched and 0002 was not recorded, so re-running still fails
    db = Database(settings)
    with db.session() as session:
        raw = session.execute(
            text("SELECT due_at FROM itinerary_check_ins WHERE id = :id"),
            {"id": check_id},
        ).scalar_one()
        assert raw == "2026-09-15T10:00:00"
        versions = {r[0] for r in session.execute(text("SELECT version FROM schema_migrations"))}
    assert "0002" not in versions
    with pytest.raises(TimestampMigrationError):
        initialize_database(db)
    db.engine.dispose()


# --------------------------------------------------------------------------- #
# Full API chain: request body offsets, database value, summary and audit
# --------------------------------------------------------------------------- #


def test_api_offset_submit_roundtrip_and_db_value(client) -> None:
    user = client.post(
        "/api/v1/users",
        json={"email": "api-tz@example.com", "display_name": "TZ Hiker"},
    ).json()
    route = client.post(
        "/api/v1/routes",
        params={"actor_id": user["id"]},
        json={
            "name": "TZ Route",
            "region": "Mountains",
            "distance_km": 5,
            "elevation_gain_m": 100,
            "estimated_duration_minutes": 120,
            "difficulty": "easy",
            "is_published": True,
            "segments": [
                {
                    "sequence": 1,
                    "name": "S",
                    "distance_km": 5,
                    "estimated_duration_minutes": 120,
                    "difficulty": "easy",
                    "start_latitude": 0,
                    "start_longitude": 0,
                    "end_latitude": 0,
                    "end_longitude": 0,
                }
            ],
        },
    ).json()
    now = datetime.now(UTC)
    expedition = client.post(
        "/api/v1/expeditions",
        json={
            "organizer_id": user["id"],
            "route_id": route["id"],
            "name": "TZ Expedition",
            "meeting_location": "Trailhead",
            "meeting_at": (now - timedelta(hours=2)).isoformat(),
            "start_at": (now - timedelta(hours=1)).isoformat(),
            "end_at": (now + timedelta(hours=6)).isoformat(),
            "registration_deadline": (now - timedelta(days=1)).isoformat(),
            "capacity": 4,
            "minimum_fitness_level": 1,
            "risk_level": "moderate",
        },
    ).json()
    # due 31 minutes ago expressed with a positive offset; punch exactly 30
    # minutes after it => boundary minute MODERATE, one minute of clock slack
    due_utc = now - timedelta(minutes=31)
    due_local = due_utc.astimezone(ZoneInfo("Asia/Kolkata"))
    scheduled = client.post(
        f"/api/v1/safety/expeditions/{expedition['id']}/check-ins",
        params={"actor_id": user["id"]},
        json={
            "user_id": user["id"],
            "check_in_type": "routine",
            "due_at": due_local.isoformat(),
        },
    )
    assert scheduled.status_code == 201, scheduled.text
    check_id = scheduled.json()["id"]

    # punch exactly 30 minutes late (boundary), expressed with negative offset
    punch_utc = due_utc + timedelta(minutes=30)
    punch_local = punch_utc.astimezone(ZoneInfo("America/Los_Angeles"))
    submit = client.post(
        f"/api/v1/safety/check-ins/{check_id}/submit",
        json={
            "checked_in_at": punch_local.isoformat(),
            "is_safe": True,
            "idempotency_key": "api-tz-submit",
        },
    )
    assert submit.status_code == 200, submit.text
    body = submit.json()
    assert body["late_minutes"] == 30
    assert body["due_at"].endswith("Z")
    assert body["checked_in_at"].endswith("Z")

    # database holds canonical Z text
    database = client.app.state.database
    with database.session() as session:
        raw_due, raw_punch = session.execute(
            text("SELECT due_at, checked_in_at FROM itinerary_check_ins WHERE id = :id"),
            {"id": check_id},
        ).one()
        assert raw_due.endswith("Z")
        assert raw_punch.endswith("Z")

    # idempotent replay via API yields the same value
    replay = client.post(
        f"/api/v1/safety/check-ins/{check_id}/submit",
        json={
            "checked_in_at": punch_local.isoformat(),
            "is_safe": True,
            "idempotency_key": "api-tz-submit",
        },
    )
    assert replay.status_code == 200
    assert replay.json()["late_minutes"] == 30

    # audit recorded through the API
    logs = client.get(
        "/api/v1/audit-logs",
        params={"entity_type": "itinerary_check_in", "entity_id": check_id},
    ).json()
    audit_actions = [item["action"] for item in logs["items"]]
    assert "checked_in" in audit_actions


def test_api_rejects_naive_future_and_out_of_window_submits(client) -> None:
    user = client.post(
        "/api/v1/users",
        json={"email": "api-bad@example.com", "display_name": "Bad Hiker"},
    ).json()

    # naive timestamp rejected at request validation
    naive = client.post(
        "/api/v1/safety/check-ins/1/submit",
        json={
            "checked_in_at": "2026-09-15T10:00:00",
            "is_safe": True,
            "idempotency_key": "naive-timestamp",
        },
    )
    assert naive.status_code == 422
    assert naive.json()["detail"]["code"] == "request_validation_error"

    route = client.post(
        "/api/v1/routes",
        params={"actor_id": user["id"]},
        json={
            "name": "Bad Route",
            "region": "Mountains",
            "distance_km": 5,
            "elevation_gain_m": 100,
            "estimated_duration_minutes": 120,
            "difficulty": "easy",
            "is_published": True,
            "segments": [
                {
                    "sequence": 1,
                    "name": "S",
                    "distance_km": 5,
                    "estimated_duration_minutes": 120,
                    "difficulty": "easy",
                    "start_latitude": 0,
                    "start_longitude": 0,
                    "end_latitude": 0,
                    "end_longitude": 0,
                }
            ],
        },
    ).json()
    now = datetime.now(UTC)
    expedition = client.post(
        "/api/v1/expeditions",
        json={
            "organizer_id": user["id"],
            "route_id": route["id"],
            "name": "Bad Expedition",
            "meeting_location": "Trailhead",
            "meeting_at": (now - timedelta(hours=2)).isoformat(),
            "start_at": (now - timedelta(hours=1)).isoformat(),
            "end_at": (now + timedelta(hours=6)).isoformat(),
            "registration_deadline": (now - timedelta(days=1)).isoformat(),
            "capacity": 4,
            "minimum_fitness_level": 1,
            "risk_level": "moderate",
        },
    ).json()

    def schedule(due_at: datetime) -> int:
        response = client.post(
            f"/api/v1/safety/expeditions/{expedition['id']}/check-ins",
            params={"actor_id": user["id"]},
            json={
                "user_id": user["id"],
                "check_in_type": "routine",
                "due_at": due_at.isoformat(),
            },
        )
        assert response.status_code == 201, response.text
        return response.json()["id"]

    # future punch -> 422 domain validation
    future_due = now + timedelta(minutes=30)
    future_check = schedule(future_due)
    future = client.post(
        f"/api/v1/safety/check-ins/{future_check}/submit",
        json={
            "checked_in_at": (future_due + timedelta(minutes=1)).isoformat(),
            "is_safe": True,
            "idempotency_key": "future-punch-api",
        },
    )
    assert future.status_code == 422
    assert future.json()["detail"]["code"] == "domain_validation_error"
    assert "future" in future.json()["detail"]["message"]

    # outside-window punch -> 422 domain validation
    past_due = now - timedelta(minutes=5)
    window_check = schedule(past_due)
    outside = client.post(
        f"/api/v1/safety/check-ins/{window_check}/submit",
        json={
            "checked_in_at": (now - timedelta(hours=3)).isoformat(),
            "is_safe": True,
            "idempotency_key": "outside-window-api",
        },
    )
    assert outside.status_code == 422
    assert "window" in outside.json()["detail"]["message"]

