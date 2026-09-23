"""TrailForge 时间语义的唯一事实来源。

整条链路（请求校验 → 服务计算 → 数据库编码 → 仓储筛选 → 汇总统计）
共用本模块的四条约定：

1. 存储：所有业务时间在 SQLite 中保存为可逆的 UTC ISO-8601 文本
   ``YYYY-MM-DDTHH:MM:SS.ffffffZ``（见 :func:`to_utc_text`）。
2. 读取：只接受带显式偏移的 ISO-8601 文本（``Z`` 或 ``±HH:MM``），
   统一换算为 UTC；无时区文本绝不按服务器本地时区猜测，
   而是抛出 :class:`UnparseableTimestampError`（见 :func:`parse_timestamp_text`）。
3. 换算：秒级差值换算分钟只有一种规则 —— 已完成的整分钟数
   （非负秒数向下取整），提前一律计 0（见 :func:`whole_minutes_late`）。
4. 边界：任何进入服务层的“当前时刻”都必须带时区并归一化为 UTC
   （见 :func:`ensure_aware` / :func:`resolve_now`）。
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from trailforge.errors import ValidationError

#: 存储文本的规范后缀（UTC 的 ISO-8601 表示）。
UTC_SUFFIX = "Z"


class UnparseableTimestampError(ValueError):
    """存储的时间文本无法安全解释时抛出。

    宁可报错也不按服务器本地时区静默猜测 —— 后者正是夏令时切换周
    迟到分钟数与超时列表相差一小时的根源。错误消息包含原始值、
    可选的定位上下文（表/列/行）以及可操作的修复指引。
    """

    def __init__(
        self,
        value: object,
        *,
        reason: str,
        context: dict[str, Any] | None = None,
    ) -> None:
        self.value = value
        self.reason = reason
        self.context = dict(context or {})
        location = ""
        if self.context:
            parts = ", ".join(f"{key}={item}" for key, item in self.context.items())
            location = f" ({parts})"
        super().__init__(
            f"cannot interpret stored timestamp {value!r}{location}: {reason}. "
            "Store timestamps as ISO-8601 with an explicit UTC offset "
            "(e.g. '2026-03-29T02:30:00Z' or '2026-03-29T10:30:00+08:00'); "
            "fix the value and re-run 'python -m trailforge.cli init-db'."
        )


def utc_now() -> datetime:
    return datetime.now(UTC)


def ensure_aware(value: datetime, *, field: str = "datetime") -> datetime:
    """服务层守卫：要求带时区的时刻并归一化为 UTC。

    与请求校验层（schemas.common.require_aware）语义一致；
    缺少时区时抛出 422 域错误而不是在更深层变成 500。
    """
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValidationError(f"{field} must include timezone information")
    return value.astimezone(UTC)


def resolve_now(value: datetime | None) -> datetime:
    """把可选的“当前时刻”参数解析为带时区的 UTC 时刻。"""
    if value is None:
        return utc_now()
    return ensure_aware(value, field="now")


def to_utc_text(value: datetime) -> str:
    """把带时区的时刻编码为规范 UTC 文本（可逆存储格式）。"""
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"datetime must include timezone information: {value!r}")
    normalized = value.astimezone(UTC)
    return normalized.isoformat(timespec="microseconds").replace("+00:00", UTC_SUFFIX)


def parse_timestamp_text(
    value: str,
    *,
    context: dict[str, Any] | None = None,
) -> datetime:
    """把存储文本解码为 UTC 时刻。

    接受 ``Z`` 后缀与任何显式 ``±HH:MM`` 偏移（含旧库遗留的偏移文本，
    以及空格分隔的日期时间）；拒绝无时区文本与无法解析的文本。
    """
    if not isinstance(value, str):
        raise UnparseableTimestampError(value, reason="expected ISO-8601 text", context=context)
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        raise UnparseableTimestampError(
            value, reason="not a valid ISO-8601 timestamp", context=context
        ) from None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise UnparseableTimestampError(
            value,
            reason="missing UTC offset; refusing to guess the server local timezone",
            context=context,
        )
    return parsed.astimezone(UTC)


def normalize_timestamp_text(value: str) -> str:
    """把任何可安全解释的存储文本改写为规范 UTC 文本（迁移用）。"""
    return to_utc_text(parse_timestamp_text(value))


def whole_minutes_late(delta: timedelta) -> int:
    """把秒级差值换算为迟到整分钟数（全系统唯一约定）。

    - 只计已完成的整分钟：非负秒数向下取整，``floor(秒 / 60)``。
      59.999 秒 → 0；60 秒 → 1；89 秒 → 1。
    - 提前或恰好准点（差值 ≤ 0）一律计 0，绝不产生负值。
    """
    seconds = delta.total_seconds()
    if seconds <= 0:
        return 0
    return int(seconds // 60)
