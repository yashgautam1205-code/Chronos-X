from datetime import datetime, timedelta, timezone
import json
import sqlite3
from uuid import uuid4

from fastapi.testclient import TestClient

from backend import main
from backend.sensor_storage import load_sensor_readings
from models import SensorReading


def test_sensor_reading_survives_backend_memory_restart(tmp_path, monkeypatch) -> None:
    database_path = tmp_path / "sensors.sqlite3"
    monkeypatch.setenv("CHRONOS_SENSOR_DB_PATH", str(database_path))
    scenario_id = uuid4()
    event_time = datetime.now(timezone.utc)
    payload = {
        "id": str(uuid4()),
        "sensor_id": "EQ-9-FLOW",
        "metric": "coolant_flow_lpm",
        "value": 18.5,
        "unit": "L/min",
        "timestamp": event_time.isoformat(),
        "equipment_id": "EQ-9",
        "provenance": {"origin": "meter-9", "source_verified": True},
        "metadata": {"scenario_id": str(scenario_id), "batch": "sample-1"},
    }

    client = TestClient(main.app)
    ingested = client.post("/sensors", json=payload)
    assert ingested.status_code == 201, ingested.text

    restarted_client = TestClient(main.app)
    listed = restarted_client.get(
        "/sensors",
        params={"scenario_id": str(scenario_id)},
    )
    assert listed.status_code == 200
    persisted = listed.json()[0]
    assert persisted["id"] == payload["id"]
    assert datetime.fromisoformat(persisted["timestamp"]) == event_time
    assert persisted["received_at"] == ingested.json()["received_at"]
    assert persisted["equipment_id"] == "EQ-9"
    assert persisted["provenance"] == payload["provenance"]
    assert persisted["metadata"] == payload["metadata"]
    assert load_sensor_readings(scenario_id=scenario_id)[0].id == SensorReading.model_validate(
        payload
    ).id
    with sqlite3.connect(database_path) as connection:
        trust_score, trust_flags = connection.execute(
            "SELECT trust_score, trust_flags_json FROM sensor_readings WHERE id = ?",
            (payload["id"],),
        ).fetchone()
    assert trust_score == 1.0
    assert json.loads(trust_flags) == []


def test_sensor_storage_rejects_duplicate_ids_atomically(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv(
        "CHRONOS_SENSOR_DB_PATH",
        str(tmp_path / "sensors.sqlite3"),
    )
    payload = {
        "id": str(uuid4()),
        "sensor_id": "EQ-9-CURRENT",
        "metric": "current_a",
        "value": 2.4,
        "unit": "A",
        "timestamp": datetime(2026, 2, 1, tzinfo=timezone.utc).isoformat(),
        "equipment_id": "EQ-9",
    }
    client = TestClient(main.app)
    assert client.post("/sensors", json=payload).status_code == 201
    assert client.post("/sensors", json=payload).status_code == 409


def test_synthetic_telemetry_ingests_persists_and_is_idempotent(
    tmp_path, monkeypatch
) -> None:
    database_path = tmp_path / "synthetic-telemetry.sqlite3"
    monkeypatch.setenv("CHRONOS_SENSOR_DB_PATH", str(database_path))
    scenario_id = uuid4()
    start_time = datetime.now(timezone.utc) - timedelta(seconds=40)
    request_payload = {
        "sample_count": 5,
        "seed": 23,
        "start_time": start_time.isoformat(),
        "equipment_id": " DEMO-STORE-23 ",
        "scenario_id": str(scenario_id),
    }

    client = TestClient(main.app)
    generated = client.post("/sensors/synthetic/generate", json=request_payload)
    assert generated.status_code == 201, generated.text
    result = generated.json()
    assert result["data_classification"] == "SYNTHETIC"
    assert result["generated_readings"] == 30
    assert result["inserted_readings"] == 30
    assert result["already_persisted_readings"] == 0
    assert result["equipment_id"] == "DEMO-STORE-23"

    repeated = client.post("/sensors/synthetic/generate", json=request_payload)
    assert repeated.status_code == 201, repeated.text
    assert repeated.json()["inserted_readings"] == 0
    assert repeated.json()["already_persisted_readings"] == 30

    restarted_client = TestClient(main.app)
    persisted = restarted_client.get(
        "/sensors",
        params={"scenario_id": str(scenario_id), "limit": 100},
    )
    assert persisted.status_code == 200
    readings = persisted.json()
    assert len(readings) == 30
    assert {reading["metric"] for reading in readings} == {
        "temperature_c",
        "pressure_kpa",
        "vibration_mm_s",
        "coolant_flow_lpm",
        "voltage_v",
        "current_a",
    }
    assert all(reading["equipment_id"] == "DEMO-STORE-23" for reading in readings)
    assert all(reading["received_at"] for reading in readings)
    assert all(reading["provenance"]["synthetic"] for reading in readings)
    assert all(
        reading["metadata"]["data_classification"] == "SYNTHETIC"
        for reading in readings
    )
    assert len(load_sensor_readings(scenario_id=scenario_id, limit=None)) == 30
    with sqlite3.connect(database_path) as connection:
        stored_count = connection.execute(
            "SELECT COUNT(*) FROM sensor_readings"
        ).fetchone()[0]
    assert stored_count == 30
