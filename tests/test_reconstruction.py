from datetime import datetime, timedelta, timezone
from uuid import uuid4

from fastapi.testclient import TestClient

from backend.main import app
from core.reconstruction import reconstruct_timeline
from models import IndustrialEvent

BASE_TIME = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _event(
    scenario_id,
    *,
    timestamp_offset: int,
    received_offset: int,
    event_type: str,
    description: str,
    provenance: dict[str, object] | None = None,
) -> IndustrialEvent:
    return IndustrialEvent(
        timestamp=BASE_TIME + timedelta(seconds=timestamp_offset),
        received_at=BASE_TIME + timedelta(seconds=received_offset),
        source="EQ-TEST",
        event_type=event_type,
        description=description,
        provenance=provenance or {},
        metadata={"scenario_id": str(scenario_id), "equipment_id": "EQ-TEST"},
    )


def test_reconstruction_flags_temporal_and_evidence_issues_without_dropping_data() -> None:
    scenario_id = uuid4()
    delayed = _event(
        scenario_id,
        timestamp_offset=0,
        received_offset=10,
        event_type="pressure_alarm",
        description="Pressure alarm",
    )
    duplicate = _event(
        scenario_id,
        timestamp_offset=0,
        received_offset=10,
        event_type="pressure_alarm",
        description="Pressure alarm",
    )
    later = _event(
        scenario_id,
        timestamp_offset=5,
        received_offset=5,
        event_type="inspection",
        description="Inspection started",
        provenance={"origin": "control_system", "source_verified": True},
    )
    sensor_rows = [
        {
            "reading_id": "sensor-a",
            "timestamp": BASE_TIME.isoformat(),
            "received_at": BASE_TIME.isoformat(),
            "equipment_id": "EQ-TEST",
            "sensor_id": "P-1",
            "metric": "pressure_kpa",
            "value": 120,
            "unit": "kPa",
            "provenance": "sensor_gateway",
        },
        {
            "reading_id": "sensor-b",
            "timestamp": BASE_TIME.isoformat(),
            "received_at": BASE_TIME.isoformat(),
            "equipment_id": "EQ-TEST",
            "sensor_id": "P-1",
            "metric": "pressure_kpa",
            "value": 145,
            "unit": "kPa",
            "provenance": "sensor_gateway",
        },
        {
            "reading_id": "sensor-c",
            "timestamp": (BASE_TIME + timedelta(seconds=2)).isoformat(),
            "received_at": (BASE_TIME + timedelta(seconds=2)).isoformat(),
            "equipment_id": "EQ-TEST",
            "sensor_id": "P-1",
            "metric": "pressure_kpa",
            "value": 146,
            "unit": "kPa",
            "provenance": "sensor_gateway",
        },
    ]

    report = reconstruct_timeline(
        scenario_id,
        [delayed, duplicate, later],
        sensor_rows,
        expected_interval_seconds=1,
    )
    repeated = reconstruct_timeline(
        scenario_id,
        [delayed, duplicate, later],
        sensor_rows,
        expected_interval_seconds=1,
    )
    reordered = reconstruct_timeline(
        scenario_id,
        [later, duplicate, delayed],
        sensor_rows,
        expected_interval_seconds=1,
    )

    assert report == repeated
    assert report["replay_timeline"] == reordered["replay_timeline"]
    assert len(report["reconstructed_timeline"]) == 6
    assert len(report["replay_timeline"]) == 5
    assert sum(
        len(item["suppressed_duplicate_ids"]) for item in report["replay_timeline"]
    ) == 1
    assert len(report["delayed_events"]) == 2
    assert len(report["duplicate_events"]) == 1
    assert report["missing_events"][0]["missing_count"] == 1
    assert len(report["out_of_order_events"]) == 2
    assert len(report["conflicting_evidence"]) == 1
    assert report["conflicting_evidence"][0]["observations"][0]["original_evidence"]
    assert report["alternative_histories"][0]["status"] == "UNRESOLVED_CONFLICT"
    assert report["alternative_histories"][0]["selection"] is None
    assert len(report["alternative_histories"][0]["alternatives"]) == 2
    assert report["suspicious_sources"][0]["source"] == "EQ-TEST"
    assert any(
        "source_verification_unattested" in item["reasons"]
        for item in report["suspicious_sources"]
    )
    assert len(report["evidence_reliability"]) == 6
    assert all(item["original_evidence"] for item in report["reconstructed_timeline"])
    assert [item["timestamp"] for item in report["reconstructed_timeline"]] == sorted(
        item["timestamp"] for item in report["reconstructed_timeline"]
    )


def test_wide_synthetic_sensor_rows_are_reconstructed_chronologically() -> None:
    scenario_id = uuid4()
    rows = [
        {
            "reading_id": "wide-1",
            "timestamp": BASE_TIME.isoformat(),
            "received_at": BASE_TIME.isoformat(),
            "equipment_id": "EQ-1",
            "temperature_sensor_id": "EQ-1-TEMP",
            "temperature_c": 72.0,
            "provenance": "synthetic_generator",
        },
        {
            "reading_id": "wide-2",
            "timestamp": (BASE_TIME + timedelta(seconds=1)).isoformat(),
            "received_at": (BASE_TIME + timedelta(seconds=1)).isoformat(),
            "equipment_id": "EQ-1",
            "temperature_sensor_id": "EQ-1-TEMP",
            "temperature_c": 73.0,
            "provenance": "synthetic_generator",
        },
    ]

    report = reconstruct_timeline(scenario_id, [], rows)

    assert len(report["reconstructed_timeline"]) == 2
    assert report["reconstructed_timeline"][0]["metric"] == "temperature_c"
    assert report["reconstructed_timeline"][0]["reliability_score"] == 1.0
    assert not report["missing_events"]
    assert not report["suspicious_sources"]


def test_sensor_source_verified_claim_is_unattested_and_original_is_retained() -> None:
    scenario_id = uuid4()
    row = {
        "reading_id": "unattested-sensor",
        "timestamp": BASE_TIME.isoformat(),
        "received_at": BASE_TIME.isoformat(),
        "equipment_id": "EQ-1",
        "sensor_id": "EQ-1-TEMP",
        "metric": "temperature_c",
        "value": 93.0,
        "unit": "degC",
        "provenance": {
            "origin": "claimed_sensor_gateway",
            "source_verified": True,
        },
    }

    report = reconstruct_timeline(scenario_id, [], [row])

    item = report["reconstructed_timeline"][0]
    assert item["original_evidence"]["reading_id"] == "unattested-sensor"
    assert "source_verification_unattested" in item["flags"]
    assert report["suspicious_sources"][0]["source"] == "EQ-1-TEMP"
    assert report["evidence_reliability"][0]["score"] < 1.0


def test_reconstruction_rejects_naive_timestamps_and_invalid_intervals() -> None:
    scenario_id = uuid4()
    naive_row = {
        "timestamp": "2026-01-01T00:00:00",
        "metric": "temperature",
        "value": 70,
    }

    try:
        reconstruct_timeline(scenario_id, [], [naive_row])
    except ValueError as error:
        assert "timezone" in str(error)
    else:
        raise AssertionError("Expected naive timestamp to be rejected")

    try:
        reconstruct_timeline(scenario_id, [], [], expected_interval_seconds=0)
    except ValueError as error:
        assert "greater than zero" in str(error)
    else:
        raise AssertionError("Expected non-positive interval to be rejected")


def test_missing_received_time_falls_back_to_event_time_reproducibly() -> None:
    scenario_id = uuid4()
    event = {
        "timestamp": BASE_TIME.isoformat(),
        "source": "EQ-REPRO",
        "event_type": "status",
        "description": "Status event",
        "metadata": {"scenario_id": str(scenario_id)},
    }

    report = reconstruct_timeline(scenario_id, [event], [])
    repeated = reconstruct_timeline(scenario_id, [event], [])

    assert report == repeated
    assert report["reconstructed_timeline"][0]["received_at"] == BASE_TIME


def test_conflicting_cooling_status_preserves_provenance_and_alternative_values() -> None:
    scenario_id = uuid4()
    rows = [
        {
            "reading_id": "cooling-report-a",
            "timestamp": BASE_TIME.isoformat(),
            "received_at": BASE_TIME.isoformat(),
            "equipment_id": "EQ-COOLING",
            "cooling_pump_sensor_id": "PUMP-STATE",
            "cooling_pump_status": 1,
            "provenance": "controller-A",
        },
        {
            "reading_id": "cooling-report-b",
            "timestamp": BASE_TIME.isoformat(),
            "received_at": (BASE_TIME + timedelta(seconds=1)).isoformat(),
            "equipment_id": "EQ-COOLING",
            "cooling_pump_sensor_id": "PUMP-STATE",
            "cooling_pump_status": 0,
            "provenance": "controller-B",
        },
    ]

    report = reconstruct_timeline(scenario_id, [], rows)

    assert len(report["conflicting_evidence"]) == 1
    alternatives = report["alternative_histories"][0]
    assert alternatives["selection"] is None
    assert {item["value"] for item in alternatives["alternatives"]} == {0, 1}
    assert {
        item["original_evidence"]["provenance"]
        for item in alternatives["alternatives"]
    } == {"controller-A", "controller-B"}


def test_reconstruction_api_returns_scenario_report_and_404_for_unknown() -> None:
    client = TestClient(app)
    scenario_id = uuid4()
    response = client.post(
        "/events",
        json={
            "timestamp": BASE_TIME.isoformat(),
            "received_at": (BASE_TIME + timedelta(seconds=8)).isoformat(),
            "source": "EQ-API",
            "event_type": "temperature_alarm",
            "description": "Temperature limit exceeded",
            "severity": "high",
            "provenance": {"origin": "test", "source_verified": True},
            "metadata": {
                "scenario_id": str(scenario_id),
                "equipment_id": "EQ-API",
            },
        },
    )
    assert response.status_code == 201

    reconstruction = client.get(f"/scenarios/{scenario_id}/reconstruct")
    assert reconstruction.status_code == 200
    report = reconstruction.json()
    assert report["scenario_id"] == str(scenario_id)
    assert len(report["reconstructed_timeline"]) == 1
    assert len(report["delayed_events"]) == 1

    post_reconstruction = client.post(f"/scenarios/{scenario_id}/reconstruct")
    assert post_reconstruction.status_code == 200
    assert post_reconstruction.json()["reconstructed_timeline"] == report[
        "reconstructed_timeline"
    ]

    missing = client.get(f"/scenarios/{uuid4()}/reconstruct")
    assert missing.status_code == 404
