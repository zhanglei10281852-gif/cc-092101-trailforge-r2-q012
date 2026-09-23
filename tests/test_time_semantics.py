"""时间语义端到端测试：偏移归一化、DST、阈值边界、迁移、幂等与入口一致性。

覆盖链路：API 请求校验 → 服务计算 → 数据库编码 → 仓储筛选 → 汇总/审计。
"""

from __future__ import annotations

import re
import time
from datetime import UTC, datetime, timedelta, timezone
from itertools import count
from zoneinfo import ZoneInfo

import pytest
from sqlalchemy import select, text

from tests.conftest import create_route, create_user
from trailforge.database.base import UTCDateTime
from trailforge.database.migrations import initialize_database, migration_status
from trailforge.database.session import Database
from trailforge.domain.enums import RiskLevel
from trailforge.domain.risk import overdue_risk_level, score_risk_level
from trailforge.errors import (
    ConflictError,
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
from trailforge.timekeeping import (
    UnparseableTimestampError,
    parse_timestamp_text,
    to_utc_text,
    whole_minutes_late,
)

NEW_YORK = ZoneInfo("America/New_York")
CANONICAL_TEXT = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z$")

_counter = count(1)


def _windowed_expedition(
    session,
    *,
    meeting: datetime,
    start: datetime,
    end: datetime,
) -> tuple[int, int]:
    """建立固定时间窗口的活动（不依赖运行当天日期），返回 (组织者, 活动 ID)。"""
    suffix = next(_counter)
    organizer = create_user(session, email=f"organizer-{suffix}@example.com")
    route = create_route(session, actor_id=organizer, name=f"Route {suffix}")
    expedition = ExpeditionService(session).create(
        ExpeditionCreate(
            organizer_id=organizer,
            route_id=route,
            name=f"Expedition {suffix}",
            meeting_location="Trailhead",
            meeting_at=meeting,
            start_at=start,
            end_at=end,
            registration_deadline=meeting - timedelta(days=1),
            capacity=5,
            minimum_fitness_level=1,
            risk_level="moderate",
        )
    )
    return organizer, expedition.id


def _past_expedition(session) -> tuple[int, int]:
    """窗口固定在 2026-01-10 08:00–17:00 UTC，便于构造超时签到。"""
    return _windowed_expedition(
        session,
        meeting=datetime(2026, 1, 10, 8, 0, tzinfo=UTC),
        start=datetime(2026, 1, 10, 9, 0, tzinfo=UTC),
        end=datetime(2026, 1, 10, 17, 0, tzinfo=UTC),
    )


def _schedule(
    session,
    expedition_id: int,
    user_id: int,
    *,
    due_at: datetime,
    check_in_type: str = "routine",
):
    return SafetyService(session).schedule_check_in(
        expedition_id,
        CheckInScheduleCreate(user_id=user_id, check_in_type=check_in_type, due_at=due_at),
        actor_id=user_id,
    )


def _submit(session, check_in_id: int, *, at: datetime, key: str, is_safe: bool = True):
    return SafetyService(session).submit_check_in(
        check_in_id,
        CheckInSubmit(checked_in_at=at, is_safe=is_safe, idempotency_key=key),
    )


def _stored_text(session, column: str, check_in_id: int) -> str:
    return session.execute(
        text(f"SELECT {column} FROM itinerary_check_ins WHERE id = :id"),
        {"id": check_in_id},
    ).scalar_one()


# ---------------------------------------------------------------------------
# A. 编解码单元语义
# ---------------------------------------------------------------------------


def test_same_instant_in_many_offsets_encodes_to_one_canonical_text() -> None:
    instant = datetime(2026, 3, 29, 2, 30, tzinfo=UTC)
    representations = [
        "2026-03-29T02:30:00+00:00",
        "2026-03-29T10:30:00+08:00",  # 正偏移
        "2026-03-28T21:30:00-05:00",  # 负偏移
        "2026-03-29T08:00:00+05:30",  # 半时区正偏移
        "2026-03-28T22:00:00-04:30",  # 半时区负偏移
    ]
    canonical = to_utc_text(instant)
    assert canonical == "2026-03-29T02:30:00.000000Z"
    for raw in representations:
        parsed = parse_timestamp_text(raw)
        assert parsed == instant
        assert to_utc_text(parsed) == canonical


def test_round_trip_is_reversible_under_any_server_timezone(monkeypatch) -> None:
    instant = datetime(2026, 3, 29, 2, 30, 15, 123456, tzinfo=UTC)
    stored = to_utc_text(instant)
    try:
        for tz in ("America/New_York", "Asia/Shanghai", "UTC"):
            monkeypatch.setenv("TZ", tz)
            time.tzset()
            assert parse_timestamp_text(stored) == instant
            assert to_utc_text(parse_timestamp_text(stored)) == stored
    finally:
        monkeypatch.undo()
        time.tzset()


def test_naive_text_is_rejected_not_guessed_under_any_server_timezone(monkeypatch) -> None:
    try:
        for tz in ("America/New_York", "Asia/Shanghai", "UTC"):
            monkeypatch.setenv("TZ", tz)
            time.tzset()
            with pytest.raises(UnparseableTimestampError, match="missing UTC offset"):
                parse_timestamp_text("2026-03-29 02:30:00")
    finally:
        monkeypatch.undo()
        time.tzset()


def test_unparseable_text_error_is_actionable() -> None:
    with pytest.raises(UnparseableTimestampError) as excinfo:
        parse_timestamp_text(
            "not-a-time",
            context={"table": "itinerary_check_ins", "column": "due_at", "rowid": 7},
        )
    message = str(excinfo.value)
    assert "not-a-time" in message
    assert "itinerary_check_ins" in message and "due_at" in message and "rowid=7" in message
    assert "init-db" in message  # 可操作的修复指引


def test_utc_datetime_type_bind_requires_timezone() -> None:
    column_type = UTCDateTime()
    with pytest.raises(ValueError, match="timezone"):
        column_type.process_bind_param(datetime(2026, 3, 29, 2, 30), None)
    aware = datetime(2026, 3, 29, 10, 30, tzinfo=timezone(timedelta(hours=8)))
    assert column_type.process_bind_param(aware, None) == "2026-03-29T02:30:00.000000Z"


def test_utc_datetime_type_read_rejects_naive_stored_text() -> None:
    column_type = UTCDateTime()
    with pytest.raises(UnparseableTimestampError):
        column_type.process_result_value("2026-03-29 02:30:00", None)


# ---------------------------------------------------------------------------
# B. 秒→分钟换算与风险阈值包含关系
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (-3600, 0),  # 提前一小时 → 0，绝不为负
        (-61, 0),
        (-1, 0),  # 提前不到一分钟 → 0
        (0, 0),  # 恰好准点 → 0
        (59, 0),  # 迟到 59 秒 → 0 个已完成整分钟
        (60, 1),
        (89, 1),
        (1799, 29),
        (1800, 30),
    ],
)
def test_whole_minutes_late_conversion(seconds: int, expected: int) -> None:
    assert whole_minutes_late(timedelta(seconds=seconds)) == expected


@pytest.mark.parametrize(
    ("minutes", "expected"),
    [
        (0, RiskLevel.LOW),
        (29, RiskLevel.LOW),
        (30, RiskLevel.MODERATE),  # 边界值进入更高一档
        (119, RiskLevel.MODERATE),
        (120, RiskLevel.HIGH),
        (359, RiskLevel.HIGH),
        (360, RiskLevel.CRITICAL),
        (10000, RiskLevel.CRITICAL),
    ],
)
def test_overdue_risk_threshold_inclusivity(minutes: int, expected: RiskLevel) -> None:
    assert overdue_risk_level(minutes) == expected


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (1, RiskLevel.LOW),
        (4, RiskLevel.LOW),
        (5, RiskLevel.MODERATE),
        (9, RiskLevel.MODERATE),
        (10, RiskLevel.HIGH),
        (16, RiskLevel.HIGH),
        (17, RiskLevel.CRITICAL),
        (25, RiskLevel.CRITICAL),
    ],
)
def test_score_risk_threshold_inclusivity(score: int, expected: RiskLevel) -> None:
    assert score_risk_level(score) == expected


# ---------------------------------------------------------------------------
# C. 服务层：偏移归一化、DST、阈值前后一秒、幂等、不回归、审计
# ---------------------------------------------------------------------------


def test_submit_with_offset_normalizes_to_utc_in_response_and_database(session) -> None:
    organizer, expedition_id = _past_expedition(session)
    due = datetime(2026, 1, 10, 10, 0, tzinfo=UTC)
    check = _schedule(session, expedition_id, organizer, due_at=due)
    # 同一时刻的 +08:00 表示：UTC 12:00，迟到恰好 120 分钟
    submitted = _submit(
        session,
        check.id,
        at=datetime(2026, 1, 10, 20, 0, tzinfo=timezone(timedelta(hours=8))),
        key="offset-submit-1",
    )
    assert submitted.checked_in_at == datetime(2026, 1, 10, 12, 0, tzinfo=UTC)
    assert submitted.late_minutes == 120
    stored = _stored_text(session, "checked_in_at", check.id)
    assert CANONICAL_TEXT.match(stored), stored
    assert stored == "2026-01-10T12:00:00.000000Z"
    assert CANONICAL_TEXT.match(_stored_text(session, "due_at", check.id))


def test_dst_spring_forward_gap_instant_round_trips_unambiguously(session) -> None:
    """春季跳跃：纽约 2026-03-08 02:30 本地不存在，但 -05:00 表示的瞬间合法。"""
    organizer, expedition_id = _windowed_expedition(
        session,
        meeting=datetime(2026, 3, 8, 5, 0, tzinfo=UTC),
        start=datetime(2026, 3, 8, 6, 0, tzinfo=UTC),
        end=datetime(2026, 3, 8, 14, 0, tzinfo=UTC),
    )
    gap_instant = datetime(2026, 3, 8, 2, 30, tzinfo=timezone(timedelta(hours=-5)))
    check = _schedule(session, expedition_id, organizer, due_at=gap_instant)
    expected_utc = datetime(2026, 3, 8, 7, 30, tzinfo=UTC)
    assert check.due_at == expected_utc
    session.expire_all()
    reloaded = session.get(ItineraryCheckIn, check.id)
    assert reloaded.due_at == expected_utc
    assert _stored_text(session, "due_at", check.id) == "2026-03-08T07:30:00.000000Z"


def test_dst_fall_back_overlap_instants_stay_one_hour_apart(session) -> None:
    """秋季重叠：纽约 2026-11-01 01:30 发生两次（EDT 与 EST），是两个不同瞬间。"""
    organizer, expedition_id = _windowed_expedition(
        session,
        meeting=datetime(2026, 10, 31, 22, 0, tzinfo=UTC),
        start=datetime(2026, 11, 1, 0, 0, tzinfo=UTC),
        end=datetime(2026, 11, 1, 12, 0, tzinfo=UTC),
    )
    first_occurrence = datetime(2026, 11, 1, 1, 30, tzinfo=NEW_YORK, fold=0)  # EDT -04:00
    second_occurrence = datetime(2026, 11, 1, 1, 30, tzinfo=NEW_YORK, fold=1)  # EST -05:00
    first = _schedule(
        session, expedition_id, organizer, due_at=first_occurrence, check_in_type="routine"
    )
    second = _schedule(
        session, expedition_id, organizer, due_at=second_occurrence, check_in_type="waypoint"
    )
    session.expire_all()
    reloaded_first = session.get(ItineraryCheckIn, first.id)
    reloaded_second = session.get(ItineraryCheckIn, second.id)
    assert reloaded_first.due_at == datetime(2026, 11, 1, 5, 30, tzinfo=UTC)
    assert reloaded_second.due_at == datetime(2026, 11, 1, 6, 30, tzinfo=UTC)
    assert (reloaded_second.due_at - reloaded_first.due_at) == timedelta(hours=1)
    # 同一物理瞬间的不同表示必须归一到同一存储文本
    assert _stored_text(session, "due_at", first.id) == "2026-11-01T05:30:00.000000Z"
    assert _stored_text(session, "due_at", second.id) == "2026-11-01T06:30:00.000000Z"


def test_late_minutes_one_second_around_minute_boundary(session) -> None:
    organizer, expedition_id = _past_expedition(session)
    due = datetime(2026, 1, 10, 10, 0, tzinfo=UTC)
    cases = [
        ("routine", timedelta(seconds=59), 0),  # 迟到 59 秒 → 0
        ("waypoint", timedelta(seconds=60), 1),  # 迟到 60 秒 → 1
        ("departure", timedelta(seconds=-1), 0),  # 提前 1 秒 → 0（不为负）
        ("assembly", timedelta(seconds=-59), 0),  # 提前不到一分钟 → 0
    ]
    for check_in_type, offset, expected_minutes in cases:
        check = _schedule(
            session, expedition_id, organizer, due_at=due, check_in_type=check_in_type
        )
        submitted = _submit(
            session, check.id, at=due + offset, key=f"boundary-{check_in_type}"
        )
        assert submitted.late_minutes == expected_minutes


def test_overdue_risk_one_second_around_each_threshold(session) -> None:
    organizer, expedition_id = _past_expedition(session)
    due = datetime(2026, 1, 10, 10, 0, tzinfo=UTC)
    check = _schedule(session, expedition_id, organizer, due_at=due)
    service = SafetyService(session)
    cases = [
        (1799, 29, RiskLevel.LOW),  # 29 分 59 秒
        (1800, 30, RiskLevel.MODERATE),  # 恰好 30 分钟
        (1801, 30, RiskLevel.MODERATE),
        (7199, 119, RiskLevel.MODERATE),
        (7200, 120, RiskLevel.HIGH),  # 恰好 120 分钟
        (21599, 359, RiskLevel.HIGH),
        (21600, 360, RiskLevel.CRITICAL),  # 恰好 360 分钟
    ]
    for seconds, expected_minutes, expected_risk in cases:
        overdue = service.overdue(now=due + timedelta(seconds=seconds))
        assert len(overdue) == 1
        assert overdue[0].check_in_id == check.id
        assert overdue[0].overdue_minutes == expected_minutes
        assert overdue[0].risk_level == expected_risk


def test_duplicate_submit_is_idempotent_and_conflicting_key_rejected(session) -> None:
    organizer, expedition_id = _past_expedition(session)
    due = datetime(2026, 1, 10, 10, 0, tzinfo=UTC)
    check = _schedule(session, expedition_id, organizer, due_at=due)
    payload = CheckInSubmit(
        checked_in_at=due + timedelta(minutes=5),
        is_safe=True,
        note="All good",
        idempotency_key="dup-submit-key",
    )
    service = SafetyService(session)
    first = service.submit_check_in(check.id, payload)
    second = service.submit_check_in(check.id, payload)
    assert second.id == first.id
    assert second.late_minutes == first.late_minutes == 5
    changed = CheckInSubmit(
        checked_in_at=due + timedelta(minutes=6),
        is_safe=True,
        idempotency_key="dup-submit-key",
    )
    with pytest.raises(IdempotencyConflictError):
        service.submit_check_in(check.id, changed)


def test_completed_check_in_cannot_be_submitted_again(session) -> None:
    organizer, expedition_id = _past_expedition(session)
    due = datetime(2026, 1, 10, 10, 0, tzinfo=UTC)
    check = _schedule(session, expedition_id, organizer, due_at=due)
    _submit(session, check.id, at=due, key="first-submit")
    with pytest.raises(ConflictError, match="already been submitted"):
        _submit(session, check.id, at=due + timedelta(minutes=1), key="second-submit")


def test_future_check_in_is_not_overdue(session) -> None:
    organizer, expedition_id = _past_expedition(session)
    due = datetime(2026, 1, 10, 16, 0, tzinfo=UTC)
    _schedule(session, expedition_id, organizer, due_at=due)
    service = SafetyService(session)
    assert service.overdue(now=due - timedelta(seconds=1)) == []
    summary = service.summary(expedition_id, now=due - timedelta(seconds=1))
    assert summary.overdue_check_ins == 0
    assert not any("overdue" in warning for warning in summary.warnings)


def test_schedule_outside_expedition_window_is_rejected(session) -> None:
    organizer, expedition_id = _past_expedition(session)
    with pytest.raises(ValidationError, match="within the expedition window"):
        _schedule(
            session,
            expedition_id,
            organizer,
            due_at=datetime(2026, 1, 10, 17, 0, 1, tzinfo=UTC),
        )
    with pytest.raises(ValidationError, match="within the expedition window"):
        _schedule(
            session,
            expedition_id,
            organizer,
            due_at=datetime(2026, 1, 10, 7, 59, 59, tzinfo=UTC),
        )


def test_audit_trail_records_reversible_utc_check_in(session) -> None:
    organizer, expedition_id = _past_expedition(session)
    due = datetime(2026, 1, 10, 10, 0, tzinfo=UTC)
    check = _schedule(session, expedition_id, organizer, due_at=due)
    instant = datetime(2026, 1, 10, 19, 45, tzinfo=timezone(timedelta(hours=8)))  # 11:45 UTC
    _submit(session, check.id, at=instant, key="audit-submit")
    log = session.scalar(
        select(AuditLog).where(
            AuditLog.entity_type == "itinerary_check_in",
            AuditLog.entity_id == check.id,
            AuditLog.action == "checked_in",
        )
    )
    assert log is not None
    recorded = parse_timestamp_text(log.after_state["checked_in_at"])
    assert recorded == datetime(2026, 1, 10, 11, 45, tzinfo=UTC)
    assert log.after_state["late_minutes"] == 105
    assert log.correlation_id == "audit-submit"


def test_service_rejects_naive_now_with_domain_error(session) -> None:
    organizer, expedition_id = _past_expedition(session)
    service = SafetyService(session)
    naive = datetime(2026, 1, 10, 12, 0)
    with pytest.raises(ValidationError, match="timezone"):
        service.overdue(now=naive)
    with pytest.raises(ValidationError, match="timezone"):
        service.summary(expedition_id, now=naive)
    with pytest.raises(ValidationError, match="timezone"):
        StatisticsService(session).risks(now=naive)


# ---------------------------------------------------------------------------
# D. API：多偏移入口一致性、422、汇总一致
# ---------------------------------------------------------------------------


def _api_expedition(client) -> tuple[int, int]:
    suffix = next(_counter)
    user = client.post(
        "/api/v1/users",
        json={"email": f"api-user-{suffix}@example.com", "display_name": "API User"},
    ).json()
    route = client.post(
        "/api/v1/routes",
        params={"actor_id": user["id"]},
        json={
            "name": f"API Route {suffix}",
            "region": "API Mountains",
            "distance_km": 8,
            "elevation_gain_m": 400,
            "elevation_loss_m": 400,
            "min_altitude_m": 100,
            "max_altitude_m": 500,
            "estimated_duration_minutes": 180,
            "difficulty": "moderate",
            "is_loop": True,
            "is_published": True,
            "segments": [
                {
                    "sequence": 1,
                    "name": "Segment",
                    "distance_km": 8,
                    "elevation_gain_m": 400,
                    "estimated_duration_minutes": 180,
                    "difficulty": "moderate",
                    "start_latitude": 30,
                    "start_longitude": 120,
                    "end_latitude": 30,
                    "end_longitude": 120,
                }
            ],
            "points": [],
            "risk_tag_ids": [],
        },
    ).json()
    expedition = client.post(
        "/api/v1/expeditions",
        json={
            "organizer_id": user["id"],
            "route_id": route["id"],
            "name": f"API Expedition {suffix}",
            "meeting_location": "Trailhead",
            "meeting_at": "2026-01-10T08:00:00Z",
            "start_at": "2026-01-10T09:00:00Z",
            "end_at": "2026-01-10T17:00:00Z",
            "registration_deadline": "2026-01-09T09:00:00Z",
            "capacity": 5,
            "minimum_fitness_level": 1,
            "risk_level": "moderate",
        },
    ).json()
    return user["id"], expedition["id"]


def test_api_check_in_chain_with_offsets_and_canonical_storage(client) -> None:
    user_id, expedition_id = _api_expedition(client)
    scheduled = client.post(
        f"/api/v1/safety/expeditions/{expedition_id}/check-ins",
        params={"actor_id": user_id},
        json={
            "user_id": user_id,
            "check_in_type": "routine",
            "due_at": "2026-01-10T18:00:00+08:00",  # = 10:00 UTC
        },
    )
    assert scheduled.status_code == 201, scheduled.text
    check = scheduled.json()
    assert check["due_at"] == "2026-01-10T10:00:00Z"
    submitted = client.post(
        f"/api/v1/safety/check-ins/{check['id']}/submit",
        json={
            "checked_in_at": "2026-01-10T04:45:00-05:30",  # = 10:15 UTC
            "is_safe": True,
            "idempotency_key": "api-offset-submit",
        },
    )
    assert submitted.status_code == 200, submitted.text
    body = submitted.json()
    assert body["checked_in_at"] == "2026-01-10T10:15:00Z"
    assert body["late_minutes"] == 15
    # 数据库中保存为规范 UTC 文本
    with client.app.state.database.session() as session:
        stored_due, stored_submit = session.execute(
            text("SELECT due_at, checked_in_at FROM itinerary_check_ins WHERE id = :id"),
            {"id": check["id"]},
        ).one()
    assert stored_due == "2026-01-10T10:00:00.000000Z"
    assert stored_submit == "2026-01-10T10:15:00.000000Z"
    # 审计日志中的时间可逆解析为同一瞬间
    with client.app.state.database.session() as session:
        log = session.scalar(
            select(AuditLog).where(
                AuditLog.entity_type == "itinerary_check_in",
                AuditLog.entity_id == check["id"],
                AuditLog.action == "checked_in",
            )
        )
    assert parse_timestamp_text(log.after_state["checked_in_at"]) == datetime(
        2026, 1, 10, 10, 15, tzinfo=UTC
    )


def test_api_equivalent_now_representations_give_identical_results(client) -> None:
    user_id, expedition_id = _api_expedition(client)
    created = client.post(
        f"/api/v1/safety/expeditions/{expedition_id}/check-ins",
        params={"actor_id": user_id},
        json={
            "user_id": user_id,
            "check_in_type": "routine",
            "due_at": "2026-01-10T10:00:00Z",
        },
    )
    assert created.status_code == 201
    representations = [
        "2026-01-10T12:00:00Z",
        "2026-01-10T20:00:00+08:00",
        "2026-01-10T07:00:00-05:00",
    ]
    overdue_results = []
    for raw in representations:
        response = client.get("/api/v1/safety/check-ins/overdue", params={"now": raw})
        assert response.status_code == 200
        overdue_results.append(response.json())
    assert overdue_results[0] == overdue_results[1] == overdue_results[2]
    assert len(overdue_results[0]) == 1
    assert overdue_results[0][0]["overdue_minutes"] == 120
    assert overdue_results[0][0]["risk_level"] == "high"
    # 汇总与各统计入口在同一时刻给出一致计数
    for raw in representations:
        summary = client.get(
            f"/api/v1/safety/expeditions/{expedition_id}/summary", params={"now": raw}
        ).json()
        risks = client.get("/api/v1/statistics/risks", params={"now": raw}).json()
        dashboard = client.get("/api/v1/statistics/dashboard", params={"now": raw}).json()
        assert summary["overdue_check_ins"] == 1
        assert risks["overdue_check_ins"] == 1
        assert dashboard["overdue_check_ins"] == 1


def test_api_naive_now_is_422_on_every_time_entry_point(client) -> None:
    user_id, expedition_id = _api_expedition(client)
    naive = "2026-01-10T12:00:00"
    paths = [
        "/api/v1/safety/check-ins/overdue",
        f"/api/v1/safety/expeditions/{expedition_id}/summary",
        "/api/v1/statistics/dashboard",
        "/api/v1/statistics/risks",
        "/api/v1/statistics/gear",
    ]
    for path in paths:
        response = client.get(path, params={"now": naive})
        assert response.status_code == 422, path
        assert response.json()["detail"]["code"] == "request_validation_error"
    audit = client.get("/api/v1/audit-logs", params={"occurred_after": naive})
    assert audit.status_code == 422


def test_api_duplicate_submit_returns_original_and_conflict_on_change(client) -> None:
    user_id, expedition_id = _api_expedition(client)
    check = client.post(
        f"/api/v1/safety/expeditions/{expedition_id}/check-ins",
        params={"actor_id": user_id},
        json={
            "user_id": user_id,
            "check_in_type": "routine",
            "due_at": "2026-01-10T10:00:00Z",
        },
    ).json()
    payload = {
        "checked_in_at": "2026-01-10T10:30:00Z",
        "is_safe": True,
        "idempotency_key": "api-dup-submit",
    }
    first = client.post(f"/api/v1/safety/check-ins/{check['id']}/submit", json=payload)
    second = client.post(f"/api/v1/safety/check-ins/{check['id']}/submit", json=payload)
    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json() == second.json()
    changed = client.post(
        f"/api/v1/safety/check-ins/{check['id']}/submit",
        json={**payload, "checked_in_at": "2026-01-10T10:31:00Z"},
    )
    assert changed.status_code == 409
    fresh_key = client.post(
        f"/api/v1/safety/check-ins/{check['id']}/submit",
        json={**payload, "idempotency_key": "another-key-123"},
    )
    assert fresh_key.status_code == 409  # 已完成签到不能重复提交


def test_api_schedule_outside_window_and_naive_due_are_422(client) -> None:
    user_id, expedition_id = _api_expedition(client)
    outside = client.post(
        f"/api/v1/safety/expeditions/{expedition_id}/check-ins",
        params={"actor_id": user_id},
        json={
            "user_id": user_id,
            "check_in_type": "routine",
            "due_at": "2026-01-11T10:00:00Z",
        },
    )
    assert outside.status_code == 422
    assert outside.json()["detail"]["code"] == "domain_validation_error"
    naive = client.post(
        f"/api/v1/safety/expeditions/{expedition_id}/check-ins",
        params={"actor_id": user_id},
        json={
            "user_id": user_id,
            "check_in_type": "routine",
            "due_at": "2026-01-10T10:00:00",
        },
    )
    assert naive.status_code == 422
    assert naive.json()["detail"]["code"] == "request_validation_error"


# ---------------------------------------------------------------------------
# E. 重启持久化
# ---------------------------------------------------------------------------


def test_check_in_instants_survive_engine_restart(settings) -> None:
    first = Database(settings)
    initialize_database(first)
    with first.session() as session:
        organizer, expedition_id = _past_expedition(session)
        due = datetime(2026, 1, 10, 18, 0, tzinfo=timezone(timedelta(hours=8)))  # 10:00 UTC
        check = _schedule(session, expedition_id, organizer, due_at=due)
        _submit(
            session,
            check.id,
            at=datetime(2026, 1, 10, 11, 45, tzinfo=UTC),
            key="restart-submit",
        )
        check_id = check.id
    first.engine.dispose()

    second = Database(settings)
    assert initialize_database(second) == []  # 迁移已应用，重启不重复执行
    with second.session() as session:
        reloaded = session.get(ItineraryCheckIn, check_id)
        assert reloaded.due_at == datetime(2026, 1, 10, 10, 0, tzinfo=UTC)
        assert reloaded.checked_in_at == datetime(2026, 1, 10, 11, 45, tzinfo=UTC)
        assert reloaded.late_minutes == 105
        overdue = SafetyService(session).overdue(now=datetime(2026, 1, 10, 12, 0, tzinfo=UTC))
        assert overdue == []  # 已签到，重启后不出现在超时列表
        other = _schedule(
            session,
            expedition_id,
            organizer,
            due_at=datetime(2026, 1, 10, 10, 0, tzinfo=UTC),
            check_in_type="waypoint",
        )
        overdue = SafetyService(session).overdue(now=datetime(2026, 1, 10, 12, 0, tzinfo=UTC))
        assert [item.check_in_id for item in overdue] == [other.id]
        assert overdue[0].overdue_minutes == 120
        assert overdue[0].risk_level == RiskLevel.HIGH
    second.engine.dispose()


# ---------------------------------------------------------------------------
# F. 迁移：混合 UTC 文本与旧偏移文本的安全规范化
# ---------------------------------------------------------------------------


def _legacy_database(settings) -> Database:
    """构造只应用过 0001 的“旧库”。"""
    database = Database(settings)
    database.create_schema()
    with database.session() as session:
        session.add(
            SchemaMigration(version="0001", description="Initial TrailForge schema")
        )
    return database


def _raw_value(database: Database, sql: str, params: dict) -> object:
    with database.engine.connect() as connection:
        return connection.execute(text(sql), params).scalar_one()


def test_migration_normalizes_mixed_utc_and_legacy_offset_text(settings) -> None:
    database = _legacy_database(settings)
    with database.session() as session:
        organizer, expedition_id = _past_expedition(session)
        check = _schedule(
            session,
            expedition_id,
            organizer,
            due_at=datetime(2026, 1, 10, 10, 0, tzinfo=UTC),
        )
        check_id = check.id
    # 模拟旧库：同一瞬间的偏移文本（T 分隔）与空格分隔文本
    with database.engine.begin() as connection:
        connection.execute(
            text("UPDATE itinerary_check_ins SET due_at = :v WHERE id = :id"),
            {"v": "2026-01-10T18:00:00+08:00", "id": check_id},
        )
        connection.execute(
            text("UPDATE itinerary_check_ins SET checked_in_at = :v WHERE id = :id"),
            {"v": "2026-01-10 04:15:00-05:30", "id": check_id},
        )
        connection.execute(
            text("UPDATE expeditions SET start_at = :v WHERE id = :id"),
            {"v": "2026-01-10T09:00:00+00:00", "id": expedition_id},
        )
    applied = initialize_database(database)
    assert applied == ["0002"]
    assert migration_status(database) == {
        "initialized": True,
        "applied": ["0001", "0002"],
        "pending": [],
    }
    # 瞬间不变，文本归一为规范 UTC
    assert (
        _raw_value(
            database, "SELECT due_at FROM itinerary_check_ins WHERE id = :id", {"id": check_id}
        )
        == "2026-01-10T10:00:00.000000Z"
    )
    assert (
        _raw_value(
            database,
            "SELECT checked_in_at FROM itinerary_check_ins WHERE id = :id",
            {"id": check_id},
        )
        == "2026-01-10T09:45:00.000000Z"
    )
    assert (
        _raw_value(
            database, "SELECT start_at FROM expeditions WHERE id = :id", {"id": expedition_id}
        )
        == "2026-01-10T09:00:00.000000Z"
    )
    # 迁移后 ORM 读取与超时查询正常
    with database.session() as session:
        reloaded = session.get(ItineraryCheckIn, check_id)
        assert reloaded.due_at == datetime(2026, 1, 10, 10, 0, tzinfo=UTC)
        assert reloaded.checked_in_at == datetime(2026, 1, 10, 9, 45, tzinfo=UTC)
    # 迁移幂等：重复执行不再产生变更
    assert initialize_database(database) == []
    database.engine.dispose()


def test_migration_reports_unparseable_rows_and_is_retryable(settings) -> None:
    database = _legacy_database(settings)
    with database.session() as session:
        organizer, expedition_id = _past_expedition(session)
        first = _schedule(
            session,
            expedition_id,
            organizer,
            due_at=datetime(2026, 1, 10, 10, 0, tzinfo=UTC),
        )
        second = _schedule(
            session,
            expedition_id,
            organizer,
            due_at=datetime(2026, 1, 10, 11, 0, tzinfo=UTC),
            check_in_type="waypoint",
        )
        first_id, second_id = first.id, second.id
    with database.engine.begin() as connection:
        connection.execute(
            text("UPDATE itinerary_check_ins SET due_at = :v WHERE id = :id"),
            {"v": "2026-01-10 10:00:00", "id": first_id},  # 无时区文本
        )
        connection.execute(
            text("UPDATE itinerary_check_ins SET checked_in_at = :v WHERE id = :id"),
            {"v": "not-a-time", "id": second_id},  # 无法解析
        )
        connection.execute(
            text("UPDATE itinerary_check_ins SET due_at = :v WHERE id = :id"),
            {"v": "2026-01-10T19:00:00+08:00", "id": second_id},  # 合法旧偏移文本
        )
    with pytest.raises(TimestampMigrationError) as excinfo:
        initialize_database(database)
    failures = excinfo.value.context["failures"]
    assert len(failures) == 2
    by_column = {item["column"]: item for item in failures}
    assert by_column["due_at"]["table"] == "itinerary_check_ins"
    assert by_column["due_at"]["rowid"] == first_id
    assert by_column["due_at"]["value"] == "2026-01-10 10:00:00"
    assert "missing UTC offset" in by_column["due_at"]["reason"]
    assert by_column["checked_in_at"]["value"] == "not-a-time"
    assert "init-db" in str(excinfo.value)
    # 迁移整体回滚：合法偏移行也未被改写，版本未推进
    assert (
        _raw_value(
            database, "SELECT due_at FROM itinerary_check_ins WHERE id = :id", {"id": second_id}
        )
        == "2026-01-10T19:00:00+08:00"
    )
    assert migration_status(database) == {
        "initialized": True,
        "applied": ["0001"],
        "pending": ["0002"],
    }
    # 修复数据后重跑成功
    with database.engine.begin() as connection:
        connection.execute(
            text("UPDATE itinerary_check_ins SET due_at = :v WHERE id = :id"),
            {"v": "2026-01-10T10:00:00Z", "id": first_id},
        )
        connection.execute(
            text("UPDATE itinerary_check_ins SET checked_in_at = :v WHERE id = :id"),
            {"v": "2026-01-10T11:30:00Z", "id": second_id},
        )
    assert initialize_database(database) == ["0002"]
    assert (
        _raw_value(
            database, "SELECT due_at FROM itinerary_check_ins WHERE id = :id", {"id": second_id}
        )
        == "2026-01-10T11:00:00.000000Z"
    )
    database.engine.dispose()


def test_orm_read_of_unmigrated_naive_text_fails_loudly(settings) -> None:
    """迁移前直接读取无时区文本：报错而不是按本地时区猜测。"""
    database = _legacy_database(settings)
    with database.session() as session:
        organizer, expedition_id = _past_expedition(session)
        check = _schedule(
            session,
            expedition_id,
            organizer,
            due_at=datetime(2026, 1, 10, 10, 0, tzinfo=UTC),
        )
        check_id = check.id
    with database.engine.begin() as connection:
        connection.execute(
            text("UPDATE itinerary_check_ins SET due_at = :v WHERE id = :id"),
            {"v": "2026-01-10 10:00:00", "id": check_id},
        )
    with database.session() as session, pytest.raises(
        UnparseableTimestampError, match="missing UTC offset"
    ):
        session.get(ItineraryCheckIn, check_id)
    database.engine.dispose()
