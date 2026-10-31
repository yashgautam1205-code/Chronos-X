from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from backend import main
from backend.agents.predictor import run_predictor_agent
from backend.agents.optimizer import run_optimizer_agent
from backend.agents.red_team import run_red_team_agent
from backend.agents.safety import run_safety_agent
from backend.main import app
from core.sensor_fusion import analyze_sensor_readings
from core.causal_investigation import investigate_causes
from core.synthetic_data import generate_synthetic_telemetry
from backend.ml.preprocessing import load_sensor_dataset
from core.sensor_fusion import sensor_records_from_reconstruction

BASE_TIME = datetime(2026, 10, 31, 0, 0, tzinfo=timezone.utc)


def _record(
    index: int,
    metric: str,
    value: float,
    *,
    equipment_id: str = "PUMP-1",
    sensor_id: str | None = None,
    seconds: int | None = None,
    received_delay: int = 0,
    unit: str | None = None,
    metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    units = {
        "coolant_flow_lpm": "L/min",
        "temperature_c": "degC",
        "pressure_kpa": "kPa",
        "vibration_mm_s": "mm/s",
        "voltage_v": "V",
        "current_a": "A",
        "power_kw": "kW",
        "flow_failure_status": "status",
        "current_overload": "status",
        "electrical_fault_status": "status",
    }
    timestamp = BASE_TIME + timedelta(seconds=index if seconds is None else seconds)
    return {
        "reading_id": f"reading-{index}-{metric}-{sensor_id or 'sensor'}",
        "sensor_id": sensor_id or f"{equipment_id}-{metric}",
        "equipment_id": equipment_id,
        "metric": metric,
        "value": value,
        "unit": unit or units[metric],
        "timestamp": timestamp.isoformat(),
        "received_at": (
            timestamp + timedelta(seconds=received_delay)
        ).isoformat(),
        "provenance": {"origin": "sensor-fusion-test", "source_verified": True},
        "metadata": metadata or {},
    }


def test_sensor_readings_validate_and_standardize_units() -> None:
    report = analyze_sensor_readings(
        [
            _record(1, "coolant_flow_rate", 0.5, unit="L/s"),
            _record(2, "voltage", 230, unit="V"),
            _record(3, "current", 2000, unit="mA"),
            _record(4, "power_kw", 1200, unit="W"),
        ]
    )

    assert [item["metric"] for item in report["records"]] == [
        "coolant_flow_lpm",
        "voltage_v",
        "current_a",
        "power_kw",
    ]
    assert report["records"][0]["value"] == 30
    assert report["records"][2]["value"] == 2
    assert report["records"][3]["value"] == pytest.approx(1.2)
    assert report["rejected_records"] == []


def test_synthetic_stream_aligns_all_six_metrics_without_health_validation() -> None:
    generated = generate_synthetic_telemetry(
        sample_count=5,
        seed=42,
        start_time=datetime.now(timezone.utc) - timedelta(seconds=40),
        equipment_id="DEMO-FUSION-1",
    )

    report = analyze_sensor_readings(generated)

    assert report["rejected_records"] == []
    assert len(report["aligned_observations"]) == 5
    assert len(report["cross_sensor_correlations"]) == 15
    assert {
        item["matched_observations"]
        for item in report["cross_sensor_correlations"]
    } == {5}
    assert {
        item["synthetic_observations"]
        for item in report["cross_sensor_correlations"]
    } == {5}
    assert report["equipment_overview"][0]["health_evidence_status"] == (
        "SYNTHETIC_ONLY"
    )
    assert report["equipment_overview"][0]["health_assessment"] == "UNKNOWN"
    assert report["data_quality_summary"]["classification"]["SYNTHETIC"] == 30


def test_missing_invalid_and_non_finite_readings_are_reported() -> None:
    missing = _record(1, "coolant_flow_lpm", 10)
    del missing["equipment_id"]
    invalid = _record(2, "voltage_v", -1)
    not_finite = _record(3, "current_a", float("nan"))

    report = analyze_sensor_readings([missing, invalid, not_finite])

    assert len(report["rejected_records"]) == 3
    assert [item["record_index"] for item in report["rejected_records"]] == [0, 1, 2]
    assert "equipment_id" in report["rejected_records"][0]["reason"]
    assert "cannot be negative" in report["rejected_records"][1]["reason"]
    assert "finite" in report["rejected_records"][2]["reason"]


def test_missing_intervals_use_sensor_cadence_without_resampling() -> None:
    records = [
        _record(1, "coolant_flow_lpm", 20, seconds=0),
        _record(2, "coolant_flow_lpm", 19, seconds=1),
        _record(3, "coolant_flow_lpm", 17, seconds=5),
        _record(4, "coolant_flow_lpm", 16, seconds=6),
    ]

    report = analyze_sensor_readings(records)

    assert len(report["missing_intervals"]) == 1
    missing = report["missing_intervals"][0]
    assert missing["equipment_id"] == "PUMP-1"
    assert missing["expected_interval_seconds"] == 1
    assert missing["missing_count"] == 3
    assert len(report["aligned_observations"]) == 4


def test_duplicate_conflicting_out_of_order_and_delayed_records_preserved() -> None:
    first = _record(1, "voltage_v", 230, seconds=0)
    duplicate = {**first, "reading_id": "duplicate"}
    delayed = _record(2, "voltage_v", 231, seconds=2, received_delay=20)
    out_of_order = _record(3, "voltage_v", 229, seconds=1)
    conflicting = _record(
        4,
        "voltage_v",
        232,
        sensor_id=first["sensor_id"],
        seconds=2,
    )

    report = analyze_sensor_readings(
        [first, duplicate, delayed, out_of_order, conflicting]
    )

    assert len(report["duplicate_records"]) == 1
    assert any("out_of_order" in item["flags"] for item in report["records"])
    assert any("delayed" in item["flags"] for item in report["records"])
    assert report["conflicting_observations"]
    assert report["conflicting_observations"][0]["resolution"] is None


def test_asynchronous_alignment_respects_equipment_and_never_uses_future_values() -> None:
    temperature = _record(1, "temperature_c", 70, seconds=0)
    coolant = _record(2, "coolant_flow_lpm", 12, seconds=3)
    future_current = _record(3, "current_a", 10, seconds=8)
    other_equipment = _record(
        4,
        "current_a",
        90,
        equipment_id="PUMP-2",
        seconds=0,
    )

    report = analyze_sensor_readings(
        [temperature, coolant, future_current, other_equipment],
        sync_tolerance_seconds=5,
    )
    at_temperature_time = next(
        item
        for item in report["aligned_observations"]
        if item["equipment_id"] == "PUMP-1"
        and item["timestamp"] == temperature["timestamp"]
    )
    at_coolant_time = next(
        item
        for item in report["aligned_observations"]
        if item["equipment_id"] == "PUMP-1"
        and item["timestamp"] == coolant["timestamp"]
    )

    assert "coolant_flow_lpm" not in at_temperature_time["measurements"]
    assert at_coolant_time["measurements"]["temperature_c"]["value"] == 70
    assert "current_a" not in at_coolant_time["measurements"]
    assert all(
        item["equipment_id"] != "PUMP-2"
        or item["measurements"].get("current_a", {}).get("value") == 90
        for item in report["aligned_observations"]
    )


def test_flow_reduction_uses_equipment_history_and_operating_context() -> None:
    records = [
        _record(
            index,
            "coolant_flow_lpm",
            20 if index < 3 else 10,
            metadata={"operating": True},
        )
        for index in range(4)
    ]
    records.extend(
        [
            _record(index + 10, "coolant_flow_lpm", 20, equipment_id="PUMP-2")
            for index in range(3)
        ]
    )

    report = analyze_sensor_readings(records)
    reductions = [
        item
        for item in report["anomaly_indicators"]
        if item["indicator"] == "coolant_flow_reduction"
    ]

    assert len(reductions) == 1
    assert reductions[0]["equipment_id"] == "PUMP-1"
    assert reductions[0]["confidence"] == "medium"
    assert reductions[0]["reduction_fraction"] == 0.5


def test_zero_flow_and_explicit_failure_are_evidence_not_confirmed_cause() -> None:
    records = [
        _record(index, "coolant_flow_lpm", 20 if index < 3 else 0)
        for index in range(4)
    ]
    records.append(_record(5, "flow_failure_status", 1))

    indicators = analyze_sensor_readings(records)["anomaly_indicators"]

    assert {item["indicator"] for item in indicators} == {
        "observed_zero_coolant_flow",
        "reported_flow_failure",
    }
    assert all("evidence_ids" in item for item in indicators)
    assert all("does not establish" in item["interpretation"] for item in indicators)


def test_electrical_deviation_overload_and_direct_power_only() -> None:
    records = [
        _record(index, "voltage_v", 230, seconds=index)
        for index in range(6)
    ]
    records.extend(
        [
            _record(7, "voltage_v", 210, seconds=7),
            _record(8, "current_a", 12, seconds=8),
            _record(9, "power_kw", 2.8, seconds=9),
        ]
    )
    report = analyze_sensor_readings(
        records,
        rated_current_a={"PUMP-1": 10},
    )

    indicators = report["anomaly_indicators"]
    assert any(item["indicator"] == "voltage_deviation_from_history" for item in indicators)
    assert any(item["indicator"] == "current_above_configured_rating" for item in indicators)
    assert "power_kw" in report["aligned_observations"][-1]["measurements"]
    assert all("calculated_power_kw" not in item for item in report["records"])


def test_correlation_is_equipment_scoped_and_descriptive() -> None:
    records = []
    for index in range(4):
        records.append(_record(index, "temperature_c", 70 + index))
        records.append(
            _record(index, "coolant_flow_lpm", 20 - index, seconds=index + 1)
        )
    report = analyze_sensor_readings(records)

    correlation = next(
        item
        for item in report["cross_sensor_correlations"]
        if item["left_metric"] == "temperature_c"
        and item["right_metric"] == "coolant_flow_lpm"
    )
    assert correlation["equipment_id"] == "PUMP-1"
    assert correlation["matched_observations"] == 3
    assert correlation["pearson_r"] == -1
    assert "does not establish causality" in correlation["interpretation"]


def test_predictor_agent_exposes_additional_sensor_evidence(tmp_path) -> None:
    scenario_id = str(uuid4())
    records = [
        _record(index, "coolant_flow_lpm", 20 if index < 3 else 5)
        for index in range(4)
    ]
    timeline = [
        {
            "kind": "sensor_reading",
            "evidence_id": record["reading_id"],
            "equipment_id": record["equipment_id"],
            "timestamp": record["timestamp"],
            "received_at": record["received_at"],
            "sensor_id": record["sensor_id"],
            "metric": record["metric"],
            "value": record["value"],
            "original_evidence": record,
            "flags": [],
        }
        for record in records
    ]

    reconstruction = {
        "scenario_id": scenario_id,
        "reconstructed_timeline": timeline,
    }
    result = run_predictor_agent(
        reconstruction,
        model_dir=tmp_path,
    )

    assert result.outputs["additional_sensor_analysis"]["anomaly_indicators"]
    assert "complete six-feature history" in result.errors[0]
    assert result.evidence_ids
    red_team = run_red_team_agent(reconstruction, result.outputs)
    assert any(
        item["indicator"] == "additional_sensor_condition_indicator"
        for item in red_team.findings
    )


def test_optimizer_and_safety_receive_sensor_evidence_without_relabeling_it() -> None:
    sensor_context = {
        "measurement_count": 3,
        "anomaly_indicators": [{"indicator": "coolant_flow_reduction"}],
    }

    optimizer = run_optimizer_agent(None, sensor_evidence_context=sensor_context)
    safety = run_safety_agent(None, sensor_evidence_context=sensor_context)

    assert optimizer.status == "UNKNOWN"
    assert optimizer.outputs["sensor_evidence_context"] == sensor_context
    assert optimizer.outputs["sensor_evidence_used_for_risk_scoring"] is False
    assert safety.status == "UNKNOWN"
    assert safety.outputs["sensor_evidence_context"] == sensor_context
    assert safety.outputs["sensor_evidence_used_as_safety_approval"] is False


def test_existing_cooling_failure_data_does_not_invent_new_modalities() -> None:
    from pathlib import Path

    data_dir = Path(__file__).resolve().parents[1] / "data"
    readings, labels = load_sensor_dataset(data_dir)
    cooling_ids = set(
        labels.loc[
            labels["scenario_name"] == "cooling_pump_failure",
            "reading_id",
        ].astype(str)
    )
    cooling_readings = readings[
        readings["reading_id"].astype(str).isin(cooling_ids)
    ]
    reconstruction = {
        "reconstructed_timeline": [
            {
                "kind": "sensor_reading",
                "evidence_id": str(row["reading_id"]),
                "equipment_id": row["equipment_id"],
                "timestamp": row["timestamp"].isoformat(),
                "received_at": row["received_at"],
                "original_evidence": row.to_dict(),
                "flags": [],
            }
            for _, row in cooling_readings.iterrows()
        ]
    }

    report = analyze_sensor_readings(
        sensor_records_from_reconstruction(reconstruction)
    )

    assert cooling_readings["cooling_pump_status"].eq(0).any()
    assert not {
        "coolant_flow_lpm",
        "voltage_v",
        "current_a",
    }.intersection({item["metric"] for item in report["records"]})
    assert not report["anomaly_indicators"]


def test_investigator_receives_new_sensor_evidence_without_reweighting_causes() -> None:
    scenario_id = uuid4()
    records = [
        _record(index, "coolant_flow_lpm", 20 if index < 3 else 5)
        for index in range(4)
    ]
    timeline = [
        {
            "kind": "sensor_reading",
            "evidence_id": record["reading_id"],
            "equipment_id": record["equipment_id"],
            "timestamp": record["timestamp"],
            "received_at": record["received_at"],
            "sensor_id": record["sensor_id"],
            "metric": record["metric"],
            "value": record["value"],
            "original_evidence": record,
            "flags": [],
        }
        for record in records
    ]
    report = investigate_causes(
        scenario_id,
        {
            "scenario_id": str(scenario_id),
            "reconstructed_timeline": timeline,
            "evidence_reliability": [
                {"evidence_id": record["reading_id"], "score": 1.0}
                for record in records
            ],
        },
    )
    baseline_report = investigate_causes(
        scenario_id,
        {
            "scenario_id": str(scenario_id),
            "reconstructed_timeline": [],
            "evidence_reliability": [],
        },
    )

    new_sensor_evidence = report["evidence_summary"]["additional_sensor_evidence"]
    assert any(
        item["indicator"] == "coolant_flow_reduction"
        for item in new_sensor_evidence["anomaly_indicators"]
    )
    cooling_hypothesis = next(
        item
        for item in new_sensor_evidence["screening_hypotheses"]
        if item["hypothesis_id"] == "possible_cooling_system_malfunction"
    )
    assert cooling_hypothesis["status"] == "POSSIBLE_UNCONFIRMED"
    assert cooling_hypothesis["supporting_evidence"]
    assert "do not establish cause" in cooling_hypothesis[
        "interpretation"
    ]
    assert {
        item["hypothesis_id"]: item["posterior_probability"]
        for item in report["ranked_hypotheses"]
    } == {
        item["hypothesis_id"]: item["posterior_probability"]
        for item in baseline_report["ranked_hypotheses"]
    }
    assert "do not alter configured hypothesis probabilities" in new_sensor_evidence[
        "interpretation"
    ]


def test_sensor_api_validates_persists_and_analyzes_actual_readings(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(main, "_DECISION_AUDIT_PATH", tmp_path / "audit.jsonl")
    monkeypatch.setenv(
        "CHRONOS_SENSOR_DB_PATH",
        str(tmp_path / "sensor-readings.sqlite3"),
    )
    client = TestClient(app)
    scenario_id = str(uuid4())
    payload = _record(1, "coolant_flow_lpm", 15)
    payload["id"] = str(uuid4())
    payload["metadata"] = {"scenario_id": scenario_id}

    missing_timestamp = dict(payload)
    del missing_timestamp["timestamp"]
    assert client.post("/sensors", json=missing_timestamp).status_code == 422

    accepted = client.post("/sensors", json=payload)
    assert accepted.status_code == 201, accepted.text
    assert accepted.json()["received_at"] is not None
    assert client.get(
        "/sensors",
        params={"scenario_id": scenario_id},
    ).json()[0]["metric"] == "coolant_flow_lpm"
    stored_analysis = client.get(
        "/sensors/analysis",
        params={"scenario_id": scenario_id},
    )
    assert stored_analysis.status_code == 200
    assert stored_analysis.json()["records"][0]["metric"] == "coolant_flow_lpm"
    analysis = client.post(
        "/sensors/analyze",
        json={"sensor_readings": [payload]},
    )
    assert analysis.status_code == 200, analysis.text
    assert analysis.json()["records"][0]["value"] == 15
    assert client.post("/sensors", json=payload).status_code == 409


def test_scenario_reconstruction_and_sensor_analysis_route_include_new_readings(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(main, "_DECISION_AUDIT_PATH", tmp_path / "audit.jsonl")
    monkeypatch.setenv(
        "CHRONOS_SENSOR_DB_PATH",
        str(tmp_path / "sensor-readings.sqlite3"),
    )
    scenario_id = str(uuid4())
    client = TestClient(app)
    payload = _record(1, "voltage_v", 230)
    payload["metadata"] = {"scenario_id": scenario_id}
    assert client.post("/sensors", json=payload).status_code == 201

    reconstruction = client.get(
        f"/scenarios/{scenario_id}/reconstruct"
    )
    analysis = client.get(
        f"/scenarios/{scenario_id}/sensor-analysis"
    )

    assert reconstruction.status_code == 200
    assert any(
        item.get("metric") == "voltage_v"
        for item in reconstruction.json()["reconstructed_timeline"]
    )
    assert analysis.status_code == 200, analysis.text
    assert analysis.json()["records"][0]["metric"] == "voltage_v"


def test_all_eight_bundled_scenarios_return_sensor_and_causal_reports() -> None:
    from pathlib import Path

    data_dir = Path(__file__).resolve().parents[1] / "data"
    _, labels = load_sensor_dataset(data_dir)
    scenario_ids = (
        labels[["scenario_name", "scenario_id"]]
        .drop_duplicates()
        .sort_values("scenario_name")
    )
    expected_names = {
        "normal",
        "gradual_temperature_increase",
        "sudden_pressure_increase",
        "cooling_pump_failure",
        "power_interruption",
        "equipment_malfunction",
        "multiple_simultaneous_faults",
        "sensor_malfunction",
    }
    assert set(scenario_ids["scenario_name"]) == expected_names

    client = TestClient(app)
    for row in scenario_ids.itertuples(index=False):
        scenario_id = row.scenario_id
        sensor_response = client.get(
            f"/scenarios/{scenario_id}/sensor-analysis"
        )
        cause_response = client.get(f"/scenarios/{scenario_id}/causes")
        assert sensor_response.status_code == 200, sensor_response.text
        assert cause_response.status_code == 200, cause_response.text
        additional = cause_response.json()["evidence_summary"][
            "additional_sensor_evidence"
        ]
        assert len(additional["screening_hypotheses"]) == 3
