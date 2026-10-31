import csv
import itertools
import json
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from backend.main import app
from models import IndustrialEvent
from core.synthetic_data import (
    SCENARIOS,
    generate_synthetic_data,
    generate_synthetic_telemetry,
    iter_sensor_records,
    save_synthetic_data,
)


def test_generation_is_reproducible_and_labels_are_separate() -> None:
    readings, labels, events = generate_synthetic_data(record_count=160, seed=7)
    repeated = generate_synthetic_data(record_count=160, seed=7)

    assert (readings, labels, events) == repeated
    assert len(readings) == len(labels) == 160
    assert len({reading["reading_id"] for reading in readings}) == 160
    assert set(SCENARIOS) == {label["scenario_name"] for label in labels}
    assert all("ground_truth_fault" not in reading for reading in readings)
    assert all("fault_type" not in reading for reading in readings)
    assert all(reading["equipment_id"] for reading in readings)
    assert all(reading["temperature_sensor_id"] for reading in readings)
    assert all(
        datetime.fromisoformat(reading["timestamp"]).tzinfo is not None
        for reading in readings
    )
    assert [reading["timestamp"] for reading in readings] == sorted(
        reading["timestamp"] for reading in readings
    )
    assert any(label["ground_truth_fault"] for label in labels)
    assert events


def test_scenario_faults_are_simulated() -> None:
    readings, labels, _ = generate_synthetic_data(record_count=800, seed=3)
    by_scenario = {
        name: [
            reading
            for reading, label in zip(readings, labels)
            if label["scenario_name"] == name and label["ground_truth_fault"]
        ]
        for name in SCENARIOS
    }

    assert not by_scenario["normal"]
    assert max(row["temperature_c"] for row in by_scenario["gradual_temperature_increase"]) > 72
    assert max(row["pressure_kpa"] for row in by_scenario["sudden_pressure_increase"]) > 120
    assert {row["cooling_pump_status"] for row in by_scenario["cooling_pump_failure"]} == {0}
    assert {row["power_supply_status"] for row in by_scenario["power_interruption"]} == {0}
    assert max(row["vibration_mm_s"] for row in by_scenario["equipment_malfunction"]) > 8
    assert {row["smoke_detected"] for row in by_scenario["multiple_simultaneous_faults"]} == {1}
    assert {row["pressure_kpa"] for row in by_scenario["sensor_malfunction"]} == {100.0}


def test_dataset_export_writes_csv_and_json_with_separate_labels(tmp_path) -> None:
    readings_csv, readings_json, labels_csv, labels_json, events_json = (
        save_synthetic_data(tmp_path, record_count=80, seed=11)
    )

    with readings_csv.open(newline="", encoding="utf-8") as file:
        csv_readings = list(csv.DictReader(file))
    json_readings = json.loads(readings_json.read_text(encoding="utf-8"))
    with labels_csv.open(newline="", encoding="utf-8") as file:
        csv_labels = list(csv.DictReader(file))
    json_labels = json.loads(labels_json.read_text(encoding="utf-8"))
    events = json.loads(events_json.read_text(encoding="utf-8"))

    assert len(csv_readings) == len(json_readings) == 80
    assert len(csv_labels) == len(json_labels) == 80
    assert "ground_truth_fault" not in csv_readings[0]
    assert "ground_truth_fault" not in json_readings[0]
    assert events
    parsed_events = [IndustrialEvent.model_validate(event) for event in events]
    assert all(event.timestamp.tzinfo is not None for event in parsed_events)


def test_continuous_generator_yields_timestamped_readings() -> None:
    first, second = list(
        itertools.islice(
            iter_sensor_records(
                "cooling_pump_failure",
                seed=19,
                start_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
            ),
            2,
        )
    )

    assert first["timestamp"] == "2026-01-01T00:00:00+00:00"
    assert second["timestamp"] == "2026-01-01T00:00:01+00:00"


def test_synthetic_telemetry_is_seeded_classified_and_has_stable_ids() -> None:
    start_time = datetime(2026, 1, 1, tzinfo=timezone.utc)
    generated = generate_synthetic_telemetry(
        sample_count=5,
        seed=17,
        start_time=start_time,
        equipment_id="DEMO-EQ-17",
    )
    repeated = generate_synthetic_telemetry(
        sample_count=5,
        seed=17,
        start_time=start_time,
        equipment_id="DEMO-EQ-17",
    )
    shifted = generate_synthetic_telemetry(
        sample_count=5,
        seed=17,
        start_time=start_time.replace(day=2),
        equipment_id="DEMO-EQ-17",
    )

    assert generated == repeated
    assert [item["id"] for item in generated] == [item["id"] for item in shifted]
    assert {item["metric"] for item in generated} == {
        "temperature_c",
        "pressure_kpa",
        "vibration_mm_s",
        "coolant_flow_lpm",
        "voltage_v",
        "current_a",
    }
    assert len(generated) == 30
    assert all(item["equipment_id"] == "DEMO-EQ-17" for item in generated)
    assert all(item["sensor_id"] for item in generated)
    assert all(item["provenance"]["synthetic"] is True for item in generated)
    assert all(
        item["metadata"]["data_classification"] == "SYNTHETIC"
        for item in generated
    )


def test_invalid_generation_inputs_are_rejected() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        generate_synthetic_data(record_count=0)
    with pytest.raises(ValueError, match="Unknown scenario"):
        next(iter_sensor_records("unknown"))


def test_event_ingestion_validation_and_listing() -> None:
    client = TestClient(app)
    event_payload = {
        "source": "EQ-001",
        "event_type": "temperature_threshold",
        "description": "Temperature exceeded the safe range.",
        "severity": "high",
        "metadata": {"equipment_id": "EQ-001"},
    }
    response = client.post("/events", json=event_payload)

    assert response.status_code == 201
    assert response.json()["source"] == "EQ-001"
    assert response.json()["id"]
    assert any(event["id"] == response.json()["id"] for event in client.get("/events").json())

    invalid_response = client.post(
        "/events",
        json={**event_payload, "severity": "unknown"},
    )
    assert invalid_response.status_code == 422
