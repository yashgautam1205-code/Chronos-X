"""SQLite persistence for received industrial sensor measurements."""

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import timezone
from pathlib import Path
from typing import Any, Iterator
from uuid import UUID

from models import SensorReading


class DuplicateSensorReadingError(ValueError):
    """Raised when a sensor-reading ID already exists in durable storage."""


def _database_path() -> Path:
    configured = os.environ.get("CHRONOS_SENSOR_DB_PATH")
    if configured:
        return Path(configured)
    return Path(__file__).resolve().parents[1] / "data" / "chronos_x.sqlite3"


def _connect() -> sqlite3.Connection:
    path = _database_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=10.0)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA busy_timeout = 10000")
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS sensor_readings (
            id TEXT PRIMARY KEY,
            equipment_id TEXT NOT NULL,
            sensor_id TEXT NOT NULL,
            metric TEXT NOT NULL,
            value REAL NOT NULL,
            unit TEXT NOT NULL,
            event_timestamp TEXT NOT NULL,
            received_at TEXT NOT NULL,
            scenario_id TEXT,
            provenance_json TEXT NOT NULL,
            metadata_json TEXT NOT NULL,
            trust_score REAL NOT NULL DEFAULT 1.0,
            trust_flags_json TEXT NOT NULL DEFAULT '[]',
            payload_json TEXT NOT NULL
        )
        """
    )
    columns = {
        row["name"]
        for row in connection.execute("PRAGMA table_info(sensor_readings)")
    }
    if "trust_score" not in columns:
        connection.execute(
            "ALTER TABLE sensor_readings ADD COLUMN trust_score REAL NOT NULL DEFAULT 1.0"
        )
    if "trust_flags_json" not in columns:
        connection.execute(
            "ALTER TABLE sensor_readings "
            "ADD COLUMN trust_flags_json TEXT NOT NULL DEFAULT '[]'"
        )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_sensor_readings_equipment_time "
        "ON sensor_readings (equipment_id, event_timestamp, id)"
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_sensor_readings_scenario_time "
        "ON sensor_readings (scenario_id, event_timestamp, id)"
    )
    return connection


@contextmanager
def _connection() -> Iterator[sqlite3.Connection]:
    connection = _connect()
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def store_sensor_reading(
    reading: SensorReading,
    *,
    trust_score: float = 1.0,
    trust_flags: tuple[str, ...] = (),
) -> None:
    if reading.equipment_id is None:
        raise ValueError("equipment_id is required for persistent sensor readings")
    if reading.received_at is None:
        raise ValueError("received_at must be set before sensor persistence")
    payload = reading.model_dump(mode="json")
    metadata = payload.get("metadata") or {}
    event_time = reading.timestamp
    if event_time.tzinfo is None:
        event_time = event_time.replace(tzinfo=timezone.utc)
    received_time = reading.received_at
    if received_time.tzinfo is None:
        received_time = received_time.replace(tzinfo=timezone.utc)
    try:
        with _connection() as connection:
            connection.execute(
                """
                INSERT INTO sensor_readings (
                    id, equipment_id, sensor_id, metric, value, unit,
                    event_timestamp, received_at, scenario_id, provenance_json,
                    metadata_json, trust_score, trust_flags_json, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(reading.id),
                    reading.equipment_id,
                    reading.sensor_id,
                    reading.metric,
                    reading.value,
                    reading.unit,
                    event_time.astimezone(timezone.utc).isoformat(),
                    received_time.astimezone(timezone.utc).isoformat(),
                    str(metadata["scenario_id"]) if metadata.get("scenario_id") else None,
                    json.dumps(payload["provenance"], sort_keys=True),
                    json.dumps(metadata, sort_keys=True),
                    trust_score,
                    json.dumps(trust_flags),
                    json.dumps(payload, sort_keys=True),
                ),
            )
    except sqlite3.IntegrityError as error:
        raise DuplicateSensorReadingError(
            f"Sensor reading {reading.id} has already been ingested"
        ) from error


def load_sensor_readings(
    *,
    offset: int = 0,
    limit: int | None = 1000,
    equipment_id: str | None = None,
    scenario_id: UUID | str | None = None,
) -> list[SensorReading]:
    clauses: list[str] = []
    parameters: list[Any] = []
    if equipment_id is not None:
        clauses.append("equipment_id = ?")
        parameters.append(equipment_id)
    if scenario_id is not None:
        clauses.append("scenario_id = ?")
        parameters.append(str(scenario_id))
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    if limit is not None:
        parameters.extend((limit, offset))
    else:
        parameters.append(offset)
    with _connection() as connection:
        rows = connection.execute(
            f"""
            SELECT payload_json FROM sensor_readings
            {where}
            ORDER BY event_timestamp, id
            {"LIMIT ? OFFSET ?" if limit is not None else "LIMIT -1 OFFSET ?"}
            """,
            parameters,
        ).fetchall()
    return [
        SensorReading.model_validate_json(row["payload_json"])
        for row in rows
    ]
