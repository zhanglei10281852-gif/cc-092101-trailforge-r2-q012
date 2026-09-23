from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import inspect, text

from trailforge.database.base import UTCDateTime
from trailforge.database.session import Database
from trailforge.domain.time import TimestampParseError, canonical_text, parse_stored_timestamp
from trailforge.errors import TimestampMigrationError
from trailforge.models.audit import SchemaMigration


@dataclass(frozen=True)
class Migration:
    version: str
    description: str


MIGRATIONS = [
    Migration(version="0001", description="Initial TrailForge schema"),
    Migration(
        version="0002",
        description="Normalize all timestamp columns to canonical UTC (Z) text",
    ),
]


def _timestamp_columns() -> list[tuple[str, str]]:
    """Return ``(table, column)`` pairs for every UTCDateTime column in the metadata."""
    from trailforge.database.base import Base
    from trailforge.models import load_all_models

    load_all_models()
    pairs: list[tuple[str, str]] = []
    for table in Base.metadata.sorted_tables:
        for column in table.columns:
            if isinstance(column.type, UTCDateTime):
                pairs.append((table.name, column.name))
    return sorted(pairs)


def normalize_utc_text(database: Database) -> dict[str, int]:
    """Rewrite legacy offset timestamp text to canonical UTC text in place.

    Existing ``...Z`` rows are untouched. Rows whose text cannot be parsed as a
    timezone-aware instant abort the migration with a
    :class:`TimestampMigrationError` listing the offending table, column, row id
    and raw value, so an operator can repair them instead of having a zone
    guessed silently.
    """
    inspector = inspect(database.engine)
    existing = set(inspector.get_table_names())
    columns = [pair for pair in _timestamp_columns() if pair[0] in existing]

    # First pass: detect every unparseable value before changing anything.
    failures: list[dict[str, str | int]] = []
    with database.engine.connect() as connection:
        for table_name, column_name in columns:
            rows = connection.execute(
                text(f"SELECT id, {column_name} FROM {table_name} WHERE {column_name} IS NOT NULL")
            ).all()
            for row_id, raw in rows:
                try:
                    parse_stored_timestamp(raw)
                except TimestampParseError as exc:
                    failures.append(
                        {
                            "table": table_name,
                            "column": column_name,
                            "row_id": int(row_id),
                            "value": str(raw),
                            "reason": str(exc),
                        }
                    )

    if failures:
        raise TimestampMigrationError(
            "found timestamp values that cannot be parsed as timezone-aware instants; "
            "repair or remove the listed rows and re-run init-db",
            context={"invalid_rows": failures[:50], "failure_count": len(failures)},
        )

    # Second pass: apply every conversion inside one transaction.
    rewritten_total = 0
    with database.session() as session:
        for table_name, column_name in columns:
            rows = session.execute(
                text(f"SELECT id, {column_name} FROM {table_name} WHERE {column_name} IS NOT NULL")
            ).all()
            for row_id, raw in rows:
                canonical = canonical_text(parse_stored_timestamp(raw))
                if canonical != raw:
                    session.execute(
                        text(f"UPDATE {table_name} SET {column_name} = :value WHERE id = :row_id"),
                        {"value": canonical, "row_id": row_id},
                    )
                    rewritten_total += 1

    return {"rewritten_values": rewritten_total}


def initialize_database(database: Database) -> list[str]:
    database.create_schema()
    applied: list[str] = []
    with database.session() as session:
        known = {
            row.version
            for row in session.query(SchemaMigration).order_by(SchemaMigration.version).all()
        }
        pending = [migration for migration in MIGRATIONS if migration.version not in known]

    for migration in pending:
        if migration.version == "0002":
            normalize_utc_text(database)
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
