from __future__ import annotations

from typing import Any


class TrailForgeError(Exception):
    status_code = 400
    code = "trailforge_error"

    def __init__(self, message: str, *, context: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.context = context or {}

    def as_detail(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "context": self.context}


class NotFoundError(TrailForgeError):
    status_code = 404
    code = "not_found"


class ConflictError(TrailForgeError):
    status_code = 409
    code = "conflict"


class InvalidStateError(ConflictError):
    code = "invalid_state_transition"


class CapacityError(ConflictError):
    code = "capacity_exceeded"


class InventoryError(ConflictError):
    code = "inventory_error"


class ValidationError(TrailForgeError):
    status_code = 422
    code = "domain_validation_error"


class IdempotencyConflictError(ConflictError):
    code = "idempotency_conflict"


class DatabaseBusyError(TrailForgeError):
    status_code = 503
    code = "database_busy"


class TimestampMigrationError(TrailForgeError):
    """存量时间文本无法安全规范化时由迁移抛出。

    context["failures"] 逐行列出 table/column/rowid/value/reason，
    迁移整体回滚，修复数据后可安全重跑。
    """

    status_code = 500
    code = "timestamp_migration_failed"

    def __init__(self, failures: list[dict[str, Any]]) -> None:
        self.failures = failures
        preview = "; ".join(
            f"{item['table']}.{item['column']} rowid={item['rowid']} "
            f"value={item['value']!r} ({item['reason']})"
            for item in failures[:5]
        )
        super().__init__(
            f"{len(failures)} stored timestamp value(s) cannot be interpreted "
            f"safely: {preview}. Add an explicit UTC offset to each value "
            "(e.g. suffix 'Z') or correct it, then re-run "
            "'python -m trailforge.cli init-db'.",
            context={"failures": failures},
        )


class UnauthorizedOperationError(TrailForgeError):
    status_code = 403
    code = "operation_not_allowed"
