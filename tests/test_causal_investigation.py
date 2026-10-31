from datetime import datetime, timedelta, timezone
from uuid import uuid4

import networkx as nx
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from core.causal_investigation import (
    HYPOTHESES,
    build_causal_graph,
    investigate_causes,
)
from core.reconstruction import reconstruct_timeline
from models import IndustrialEvent

BASE_TIME = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _reconstruction(
    scenario_id,
    *,
    pump_status: int = 1,
    power_status: int = 1,
    temperature: float = 72.0,
    pressure: float = 100.0,
    vibration: float = 2.0,
    smoke: int = 0,
    pressure_values: list[float] | None = None,
    event_type: str | None = None,
) -> dict:
    readings = []
    for index in range(10):
        readings.append(
            {
                "reading_id": f"reading-{index}",
                "timestamp": (BASE_TIME + timedelta(seconds=index)).isoformat(),
                "received_at": (BASE_TIME + timedelta(seconds=index)).isoformat(),
                "equipment_id": "EQ-1",
                "temperature_sensor_id": "EQ-1-TEMP",
                "pressure_sensor_id": "EQ-1-PRES",
                "vibration_sensor_id": "EQ-1-VIB",
                "cooling_pump_sensor_id": "EQ-1-PUMP",
                "power_supply_sensor_id": "EQ-1-POWER",
                "smoke_sensor_id": "EQ-1-SMOKE",
                "temperature_c": temperature,
                "pressure_kpa": (
                    pressure_values[index]
                    if pressure_values is not None
                    else pressure
                ),
                "vibration_mm_s": vibration,
                "cooling_pump_status": pump_status,
                "power_supply_status": power_status,
                "smoke_detected": smoke,
                "provenance": "synthetic_generator",
            }
        )
    events = []
    if event_type:
        events.append(
            IndustrialEvent(
                timestamp=BASE_TIME + timedelta(seconds=5),
                received_at=BASE_TIME + timedelta(seconds=5),
                source="EQ-1",
                event_type=event_type,
                description=f"Simulated {event_type.replace('_', ' ')}.",
                provenance={"origin": "synthetic_generator", "source_verified": True},
                metadata={"scenario_id": str(scenario_id), "equipment_id": "EQ-1"},
            )
        )
    return reconstruct_timeline(scenario_id, events, readings)


def test_causal_graph_is_directed_and_has_explainable_mechanisms() -> None:
    graph = build_causal_graph()

    assert isinstance(graph, nx.DiGraph)
    assert nx.is_directed_acyclic_graph(graph)
    assert graph.has_edge("cooling_pump_failure", "reduced_cooling_capacity")
    assert graph.has_edge("temperature_increase", "pressure_increase")
    assert graph.edges["cooling_pump_failure", "reduced_cooling_capacity"]["mechanism"]


def test_multiple_causes_rank_with_support_contradiction_and_missing_evidence() -> None:
    scenario_id = uuid4()
    reconstruction = _reconstruction(
        scenario_id,
        pump_status=0,
        temperature=86,
        pressure=130,
        vibration=2,
    )
    reconstruction["reconstructed_timeline"] = [
        item
        for item in reconstruction["reconstructed_timeline"]
        if item.get("metric") != "cooling_pump_status"
    ]
    reconstruction["evidence_reliability"] = [
        item
        for item in reconstruction["evidence_reliability"]
        if not item["evidence_id"].endswith(":cooling_pump_status")
    ]

    report = investigate_causes(scenario_id, reconstruction)

    assert len(report["ranked_hypotheses"]) == len(HYPOTHESES)
    assert report["ranked_hypotheses"][0]["hypothesis_id"] == "cooling_pump_failure"
    assert report["ranked_hypotheses"][0]["posterior_probability"] > (
        report["ranked_hypotheses"][-1]["posterior_probability"]
    )
    assert sum(
        cause["posterior_probability"] for cause in report["ranked_hypotheses"]
    ) == pytest.approx(1.0)
    pump_cause = next(
        cause
        for cause in report["possible_root_causes"]
        if cause["hypothesis_id"] == "cooling_pump_failure"
    )
    assert pump_cause["supporting_evidence"]
    assert pump_cause["contradicting_evidence"]
    assert pump_cause["missing_evidence"]
    assert pump_cause["supporting_evidence"][0]["evidence"][0]["evidence_id"]
    assert any(
        "does not establish causality" in assumption
        for assumption in report["assumptions"]
    )


def test_sensor_flatline_and_conflicting_evidence_support_sensor_fault() -> None:
    scenario_id = uuid4()
    original_rows = [
        {
            "reading_id": f"reading-{index}",
            "timestamp": (BASE_TIME + timedelta(seconds=index)).isoformat(),
            "received_at": (BASE_TIME + timedelta(seconds=index)).isoformat(),
            "equipment_id": "EQ-1",
            "pressure_sensor_id": "EQ-1-PRES",
            "pressure_kpa": 100.0,
            "provenance": "synthetic_generator",
        }
        for index in range(10)
    ]
    conflict = {
        **original_rows[0],
        "reading_id": "conflicting-reading",
        "pressure_kpa": 135.0,
    }
    conflicted = reconstruct_timeline(
        scenario_id,
        [],
        [*original_rows, conflict],
    )
    report = investigate_causes(scenario_id, conflicted)

    assert report["ranked_hypotheses"][0]["hypothesis_id"] == "sensor_malfunction"
    sensor_cause = report["ranked_hypotheses"][0]
    assert {item["signal"] for item in sensor_cause["supporting_evidence"]} == {
        "pressure_flatline",
        "conflicting_sensor_evidence",
    }
    assert sensor_cause["supporting_evidence"][0]["evidence"]


def test_investigation_is_reproducible_and_requires_matching_scenario() -> None:
    scenario_id = uuid4()
    reconstruction = _reconstruction(scenario_id, power_status=0, pump_status=0)

    assert investigate_causes(scenario_id, reconstruction) == investigate_causes(
        scenario_id, reconstruction
    )
    with pytest.raises(ValueError, match="does not match"):
        investigate_causes(uuid4(), reconstruction)


def test_causes_api_returns_ranked_report_and_unknown_scenario_not_found() -> None:
    client = TestClient(app)
    scenario_id = uuid4()
    response = client.post(
        "/events",
        json={
            "timestamp": BASE_TIME.isoformat(),
            "received_at": BASE_TIME.isoformat(),
            "source": "EQ-CAUSE-API",
            "event_type": "cooling_pump_failure",
            "description": "Pump reported stopped",
            "severity": "high",
            "provenance": {"origin": "test", "source_verified": True},
            "metadata": {
                "scenario_id": str(scenario_id),
                "equipment_id": "EQ-CAUSE-API",
            },
        },
    )
    assert response.status_code == 201

    investigation = client.get(f"/scenarios/{scenario_id}/causes")
    assert investigation.status_code == 200
    result = investigation.json()
    assert result["scenario_id"] == str(scenario_id)
    assert len(result["ranked_hypotheses"]) == len(HYPOTHESES)
    assert result["ranked_hypotheses"][0]["hypothesis_id"] == "cooling_pump_failure"
    assert result["causal_relationships"]
    assert result["assumptions"]

    missing = client.get(f"/scenarios/{uuid4()}/causes")
    assert missing.status_code == 404
