from datetime import datetime, timedelta, timezone
from uuid import uuid4

from fastapi.testclient import TestClient

from backend.agents import orchestrator as orchestrator_module
from backend import main
from backend.main import app
from models import AgentResult


def _base_payload():
    scenario_id = str(uuid4())
    timestamp = datetime(2026, 10, 30, 12, 0, tzinfo=timezone.utc)
    return {
        "scenario_id": scenario_id,
        "name": "Unseen compressor incident",
        "description": "Runtime fixture that does not use a bundled scenario.",
        "infrastructure": [
            {
                "asset_id": "compressor-7",
                "asset_type": "thermal_pressure_equipment",
                "state": {"mode": "running", "operator_confirmed": False},
            }
        ],
        "events": [
            {
                "timestamp": timestamp.isoformat(),
                "received_at": (timestamp + timedelta(seconds=2)).isoformat(),
                "source": "compressor-7-controller",
                "event_type": "bearing_fault_reported",
                "description": "Controller reports a bearing fault",
                "severity": "high",
                "provenance": {
                    "origin": "runtime-test",
                    "source_verified": True,
                },
                "metadata": {
                    "scenario_id": scenario_id,
                    "equipment_id": "compressor-7",
                },
            }
        ],
        "sensor_readings": [
            {
                "sensor_id": "compressor-7-temp",
                "metric": "temperature_c",
                "value": 90.0,
                "unit": "degC",
                "timestamp": timestamp.isoformat(),
                "equipment_id": "compressor-7",
            },
            {
                "sensor_id": "compressor-7-pressure",
                "metric": "pressure_kpa",
                "value": 130.0,
                "unit": "kPa",
                "timestamp": timestamp.isoformat(),
                "equipment_id": "compressor-7",
            },
        ],
        "causal_hypotheses": [
            {
                "hypothesis_id": "bearing_degradation",
                "title": "Bearing degradation",
                "expected_event_types": ["bearing_fault_reported"],
                "sensor_conditions": [
                    {
                        "metric": "temperature_c",
                        "operator": "gte",
                        "value": 85,
                        "description": "Bearing temperature is elevated",
                    }
                ],
                "asset_conditions": [
                    {
                        "asset_id": "compressor-7",
                        "state_key": "mode",
                        "operator": "eq",
                        "value": "running",
                    }
                ],
            }
        ],
        "causal_relationships": [
            {
                "cause": "bearing_degradation",
                "effect": "temperature_c",
                "mechanism": "Friction may increase local temperature.",
            }
        ],
    }


def test_runtime_scenario_uses_submitted_evidence_and_returns_full_pipeline(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(
        main,
        "_DECISION_AUDIT_PATH",
        tmp_path / "runtime-audit.jsonl",
    )
    client = TestClient(app)
    payload = _base_payload()
    payload["simulation"] = {
        "rollouts": 100,
        "seed": 2026,
        "thermal_incident_threshold_c": 105.0,
        "pressure_incident_threshold_kpa": 170.0,
    }
    payload["available_interventions"] = [
        "no_intervention",
        "activate_backup_cooling",
        "isolate_affected_system",
        "controlled_shutdown",
    ]

    response = client.post("/scenarios/analyze", json=payload)
    repeated = client.post("/scenarios/analyze", json=payload)

    assert response.status_code == 200, response.text
    assert repeated.status_code == 200, repeated.text
    result = response.json()
    assert [
        item["evidence_id"]
        for item in result["reconstructed_timeline"]["reconstructed_timeline"]
    ] == [
        item["evidence_id"]
        for item in repeated.json()["reconstructed_timeline"]["reconstructed_timeline"]
    ]
    assert result["scenario_id"] == payload["scenario_id"]
    assert result["reconstructed_timeline"]["reconstructed_timeline"]
    assert result["causal_investigation"]["ranked_hypotheses"][0][
        "hypothesis_id"
    ] == "bearing_degradation"
    assert result["causal_investigation"]["ranked_hypotheses"][0]["confidence"] is None
    assert len(
        result["causal_investigation"]["ranked_hypotheses"][0][
            "supporting_evidence"
        ]
    ) == 3
    assert result["simulation"]["number_of_rollouts_per_intervention"] == 100
    assert len(result["simulation"]["interventions"]) == 4
    assert len(result["agent_council"]["agent_results"]) == 5
    assert result["decision"]["status"] == "ABSTAINED"
    assert result["decision"]["human_review_required"] is True
    assert (tmp_path / "runtime-audit.jsonl").is_file()
    assert all(
        item["original_evidence"].get("metadata", {}).get("scenario_id")
        == payload["scenario_id"]
        for item in result["reconstructed_timeline"]["reconstructed_timeline"]
        if item["kind"] == "event"
    )


def test_unknown_equipment_type_is_not_simulated_or_recommended(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(
        main,
        "_DECISION_AUDIT_PATH",
        tmp_path / "runtime-audit.jsonl",
    )
    payload = _base_payload()
    payload["infrastructure"][0]["asset_type"] = "unregistered_turbine"
    payload["simulation"] = {
        "rollouts": 100,
        "seed": 2026,
        "thermal_incident_threshold_c": 105.0,
        "pressure_incident_threshold_kpa": 170.0,
    }

    response = TestClient(app).post("/scenarios/analyze", json=payload)

    assert response.status_code == 200, response.text
    result = response.json()
    assert result["simulation"] is None
    assert result["decision"]["status"] == "ABSTAINED"
    assert result["decision"]["selected_feasible_intervention"] is None
    assert result["decision"]["human_review_required"] is True
    assert any(
        "no compatible physics model is registered" in capability
        for capability in result["unsupported_capabilities"]
    )


def test_runtime_scenario_without_simulation_does_not_substitute_demo_data(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(
        main,
        "_DECISION_AUDIT_PATH",
        tmp_path / "runtime-audit.jsonl",
    )
    client = TestClient(app)
    payload = _base_payload()

    response = client.post("/scenarios/analyze", json=payload)

    assert response.status_code == 200, response.text
    result = response.json()
    assert result["simulation"] is None
    assert result["decision"]["status"] == "ABSTAINED"
    assert any("Simulation was not run" in item for item in result["unsupported_capabilities"])
    assert result["agent_council"]["status"] == "PARTIAL"
    assert len(result["reconstructed_timeline"]["reconstructed_timeline"]) == 3


def test_runtime_investigation_exposes_competing_uncertainty_alternatives(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(
        main,
        "_DECISION_AUDIT_PATH",
        tmp_path / "runtime-audit.jsonl",
    )
    payload = _base_payload()
    payload["causal_hypotheses"] = []

    response = TestClient(app).post("/scenarios/analyze", json=payload)

    assert response.status_code == 200, response.text
    hypotheses = response.json()["causal_investigation"]["ranked_hypotheses"]
    assert [item["hypothesis_id"] for item in hypotheses] == [
        "hidden_confounder",
        "unmodeled_cause",
    ]
    assert all(item["confidence"] is None for item in hypotheses)
    assert all(not item["supporting_evidence"] for item in hypotheses)


def test_unrecognized_deadline_and_asset_dependency_fields_are_not_ignored(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(
        main,
        "_DECISION_AUDIT_PATH",
        tmp_path / "runtime-audit.jsonl",
    )
    payload = _base_payload()
    payload["virtual_deadline_seconds"] = 1.0
    payload["infrastructure"][0]["dependencies"] = ["cooling-loop-2"]

    response = TestClient(app).post("/scenarios/analyze", json=payload)

    assert response.status_code == 200, response.text
    result = response.json()
    assert result["decision"]["status"] == "ABSTAINED"
    assert result["decision"]["human_review_required"] is True
    assert any(
        "virtual_deadline_seconds" in capability
        for capability in result["unsupported_capabilities"]
    )
    assert any(
        "dependencies" in capability
        for capability in result["unsupported_capabilities"]
    )
    assert result["infrastructure"][0]["dependencies"] == ["cooling-loop-2"]


def test_runtime_scenario_rejects_unmodeled_intervention_and_abstains(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(
        main,
        "_DECISION_AUDIT_PATH",
        tmp_path / "runtime-audit.jsonl",
    )
    client = TestClient(app)
    payload = _base_payload()
    payload["available_interventions"] = ["judge_specific_catalyst_dump"]

    response = client.post("/scenarios/analyze", json=payload)

    assert response.status_code == 200, response.text
    result = response.json()
    assert any(
        "judge_specific_catalyst_dump" in item
        for item in result["unsupported_capabilities"]
    )
    assert result["decision"]["status"] == "ABSTAINED"
    assert result["decision"]["human_review_required"] is True


def test_latest_conflicting_sensor_values_never_seed_simulation(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(
        main,
        "_DECISION_AUDIT_PATH",
        tmp_path / "runtime-audit.jsonl",
    )
    client = TestClient(app)
    payload = _base_payload()
    payload["simulation"] = {
        "rollouts": 100,
        "seed": 2026,
        "thermal_incident_threshold_c": 120.0,
        "pressure_incident_threshold_kpa": 170.0,
    }
    payload["available_interventions"] = [
        "no_intervention",
        "activate_backup_cooling",
        "isolate_affected_system",
        "controlled_shutdown",
    ]
    conflicting_reading = dict(payload["sensor_readings"][0])
    conflicting_reading["value"] = 115.0
    payload["sensor_readings"].append(conflicting_reading)
    reversed_payload = {
        **payload,
        "sensor_readings": list(reversed(payload["sensor_readings"])),
    }

    responses = [
        client.post("/scenarios/analyze", json=variant)
        for variant in (payload, reversed_payload)
    ]

    assert all(response.status_code == 200 for response in responses)
    reports = [response.json() for response in responses]
    for report in reports:
        assert report["simulation"] is None
        assert report["decision"]["status"] == "ABSTAINED"
        assert report["decision"]["human_review_required"] is True
        assert any(
            "latest sensor values conflict" in capability
            for capability in report["unsupported_capabilities"]
        )
        safety = next(
            item
            for item in report["agent_council"]["agent_results"]
            if item["agent_name"] == "safety"
        )
        assert safety["status"] == "UNKNOWN"
        assert safety["outputs"]["overall_result"] == "UNKNOWN"
    assert reports[0]["decision"]["selected_feasible_intervention"] == reports[1][
        "decision"
    ]["selected_feasible_intervention"]


def test_paradox_protocol_subset_fails_closed_and_audits(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(
        main,
        "_DECISION_AUDIT_PATH",
        tmp_path / "paradox-audit.jsonl",
    )
    compromised_safety = AgentResult(
        agent_name="safety",
        status="COMPLETED",
        summary="Compromised test agent recommends isolation.",
        findings=[
            {
                "intervention_name": "isolate_affected_system",
                "result": "PASS",
            }
        ],
        outputs={
            "overall_result": "PASS",
            "recommended_intervention": "isolate_affected_system",
        },
    )
    monkeypatch.setattr(
        orchestrator_module,
        "run_safety_agent",
        lambda _: compromised_safety.model_copy(deep=True),
    )

    payload = _base_payload()
    scenario_id = payload["scenario_id"]
    start = datetime.fromisoformat(payload["sensor_readings"][0]["timestamp"])
    payload["events"].append(
        {
            "timestamp": (start - timedelta(hours=1)).isoformat(),
            "received_at": start.isoformat(),
            "source": "innocent-subsystem",
            "event_type": "historical_subsystem_failure",
            "description": "Unverified history implicates an innocent subsystem.",
            "severity": "high",
            "provenance": {
                "origin": "untrusted_archive",
                "source_verified": True,
            },
            "metadata": {
                "scenario_id": scenario_id,
                "equipment_id": "cooling-pump-1",
            },
        }
    )
    payload["sensor_readings"] = []
    for step in range(3):
        timestamp = start + timedelta(minutes=5 * step)
        values = {
            "temperature_c": 88.0 + step * 1.5,
            "pressure_kpa": 145.0 + step * 2.5,
            "vibration_mm_s": 2.0,
            "cooling_pump_status": 0.0,
            "power_supply_status": 1.0,
            "smoke_detected": 0.0,
        }
        for metric, value in values.items():
            payload["sensor_readings"].append(
                {
                    "sensor_id": f"cooling-pump-1-{metric}",
                    "metric": metric,
                    "value": value,
                    "unit": "status" if metric.endswith("_status") else "unit",
                    "timestamp": timestamp.isoformat(),
                    "received_at": timestamp.isoformat(),
                    "equipment_id": "compressor-7",
                    "metadata": {"scenario_id": scenario_id},
                }
            )
    older_pressure = payload["sensor_readings"][7].copy()
    older_pressure["id"] = str(uuid4())
    older_pressure["value"] = 151.0
    payload["sensor_readings"].append(older_pressure)
    payload["simulation"] = {
        "rollouts": 100,
        "seed": 2026,
        "simulation_hours": 2.0,
        "time_step_minutes": 5.0,
        "pump_failure_rate_per_hour": 0.0,
        "equipment_failure_rate_per_hour": 0.0,
        "backup_available_probability": 1.0,
        "isolation_secondary_failure_probability": 1.0,
        "isolation_secondary_failure_delay_minutes": 15.0,
        "isolation_secondary_pressure_rise_kpa_per_hour": 300.0,
        "thermal_incident_threshold_c": 105.0,
        "pressure_incident_threshold_kpa": 160.0,
    }
    payload["available_interventions"] = [
        "no_intervention",
        "activate_backup_cooling",
        "isolate_affected_system",
        "controlled_shutdown",
    ]
    payload["operational_constraints"] = {
        "backup_cooling_available": True,
        "system_isolation_available": True,
        "controlled_shutdown_authorized": True,
        "available_operators": 4,
        "model_outputs_reviewed": True,
    }
    payload["virtual_deadline_seconds"] = 0.001

    response = TestClient(app).post("/scenarios/analyze", json=payload)

    assert response.status_code == 200, response.text
    report = response.json()
    assert report["simulation"] is not None
    isolation = next(
        item
        for item in report["simulation"]["interventions"]
        if item["intervention_name"] == "isolate_affected_system"
    )
    assert (
        isolation["simulated_incident_outcomes"][
            "delayed_secondary_failure_count"
        ]
        == 100
    )
    assert report["reconstructed_timeline"]["conflicting_evidence"]
    assert len(report["causal_investigation"]["ranked_hypotheses"]) >= 2
    suspicious = report["reconstructed_timeline"]["suspicious_sources"]
    assert any(item["source"] == "innocent-subsystem" for item in suspicious)
    assert any(
        item["original_evidence"].get("description")
        == "Unverified history implicates an innocent subsystem."
        for item in report["reconstructed_timeline"]["reconstructed_timeline"]
        if item["kind"] == "event"
    )
    safety = next(
        item
        for item in report["agent_council"]["agent_results"]
        if item["agent_name"] == "safety"
    )
    assert safety["status"] == "UNKNOWN"
    assert any(
        conflict["type"] == "safety_agent_simulation_disagreement"
        for conflict in report["agent_council"]["conflicts"]
    )
    assert report["decision"]["status"] == "ABSTAINED"
    assert report["decision"]["human_review_required"] is True
    isolation_rejection = next(
        item
        for item in report["decision"]["rejected_alternatives"]
        if item["intervention_name"] == "isolate_affected_system"
    )
    assert isolation_rejection["reasons"]
    assert any(
        "virtual_deadline_seconds" in capability
        for capability in report["unsupported_capabilities"]
    )
    audit = TestClient(app).get(f"/scenarios/{scenario_id}/audit")
    assert audit.status_code == 200
    assert len(audit.json()) == 1
    assert (
        audit.json()[0]["request_payload"]["virtual_deadline_seconds"] == 0.001
    )
