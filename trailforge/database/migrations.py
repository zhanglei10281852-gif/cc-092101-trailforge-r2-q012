from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import Engine, inspect, text

from trailforge.database.base import Base, UTCDateTime
from trailforge.database.session import Database
from trailforge.errors import TimestampMigrationError
from trailforge.models.audit import SchemaMigration
from trailforge.timekeeping import UnparseableTimestampError, normalize_timestamp_text


@dataclass(frozen=True)
class Migration:
    version: str
    description: str
    #: 可选的数据迁移；在记录版本号之前执行，失败则整体回滚、可安全重跑。
    upgrade: Callable[[Engine], dict[str, int]] | None = field(default=None, compare=False)


def _utc_datetime_columns() -> list[tuple[str, str]]:
    """列出模型元数据中所有 UTCDateTime 列（表名, 列名）。"""
    from trailforge.models import load_all_models

    load_all_models()
    targets: list[tuple[str, str]] = []
    for table in Base.metadata.sorted_tables:
        for column in table.columns:
            if isinstance(column.type, UTCDateTime):
                targets.append((table.name, column.name))
    return sorted(targets)


def normalize_stored_timestamps(engine: Engine) -> dict[str, int]:
    """把库中所有 UTCDateTime 列规范化为可逆 UTC 文本（迁移 0002）。

    已是规范 ``Z`` 文本的行保持不变；带显式偏移的旧文本（如 ``+08:00``、
    空格分隔形式）换算为同一瞬间的 UTC 文本；无偏移或无法解析的文本
    不会被静默猜测 —— 收集全部问题行后抛出 TimestampMigrationError，
    整个迁移回滚，修复数据后重跑即可。
    """
    failures: list[dict[str, Any]] = []
    stats = {"columns": 0, "normalized": 0, "unchanged": 0}
    with engine.begin() as connection:
        for table, column in _utc_datetime_columns():
            stats["columns"] += 1
            rows = connection.execute(
                text(f'SELECT rowid AS row_id, "{column}" AS stored_value FROM "{table}"')
            ).all()
            for row in rows:
                row_id, raw = row[0], row[1]
                if raw is None:
                    continue
                try:
                    canonical = normalize_timestamp_text(raw)
                except UnparseableTimestampError as exc:
                    failures.append(
                        {
                            "table": table,
                            "column": column,
                            "rowid": row_id,
                            "value": raw,
                            "reason": exc.reason,
                        }
                    )
                    continue
                if canonical != raw:
                    connection.execute(
                        text(f'UPDATE "{table}" SET "{column}" = :value WHERE rowid = :rowid'),
                        {"value": canonical, "rowid": row_id},
                    )
                    stats["normalized"] += 1
                else:
                    stats["unchanged"] += 1
        if failures:
            raise TimestampMigrationError(failures)
    return stats


MIGRATIONS = [
    Migration(version="0001", description="Initial TrailForge schema"),
    Migration(
        version="0002",
        description="Normalize stored timestamps to reversible UTC text",
        upgrade=normalize_stored_timestamps,
    ),
]


def initialize_database(database: Database) -> list[str]:
    database.create_schema()
    with database.session() as session:
        known = {
            row.version
            for row in session.query(SchemaMigration).order_by(SchemaMigration.version).all()
        }
    applied: list[str] = []
    for migration in MIGRATIONS:
        if migration.version in known:
            continue
        if migration.upgrade is not None:
            migration.upgrade(database.engine)
        with database.session() as session:
            session.add(
                SchemaMigration(
                    version=migration.version,
                    description=migration.description,
                )
            )
        applied.append(migration.version)
    return applied


def migration_status(database: Database) -> dict[str, object]:
    inspector = inspect(database.engine)
    if "schema_migrations" not in inspector.get_table_names():
        return {
            "initialized": False,
            "applied": [],
            "pending": [item.version for item in MIGRATIONS],
        }
    with database.session() as session:
        applied = [
            row.version
            for row in session.query(SchemaMigration).order_by(SchemaMigration.version).all()
        ]
    pending = [item.version for item in MIGRATIONS if item.version not in set(applied)]
    return {"initialized": True, "applied": applied, "pending": pending}


def assert_database_integrity(database: Database) -> dict[str, object]:
    with database.engine.connect() as connection:
        integrity = connection.exec_driver_sql("PRAGMA integrity_check").scalar_one()
        foreign_key_rows = connection.exec_driver_sql("PRAGMA foreign_key_check").all()
    return {
        "integrity_check": str(integrity),
        "foreign_key_violations": [list(row) for row in foreign_key_rows],
        "healthy": integrity == "ok" and not foreign_key_rows,
    }
