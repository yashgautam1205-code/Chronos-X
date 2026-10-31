import asyncio

from fastapi import Request
from fastapi.testclient import TestClient

from backend.main import app, handle_unexpected_exception
from models import (
    AgentReport,
    AgentResponse,
    Decision,
    FaultAlert,
    IndustrialEvent,
    Prediction,
    RiskAssessment,
    Scenario,
    SensorReading,
)


def test_health_returns_ok() -> None:
    response = TestClient(app).get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_shared_models_validate_payloads() -> None:
    event = IndustrialEvent(
        source="sensor-1",
        event_type="temperature",
        description="Temperature threshold exceeded",
    )
    scenario = Scenario(name="High temperature", events=[event])
    report = AgentReport(agent_name="baseline", scenario_id=scenario.id)
    risk = RiskAssessment(scenario_id=scenario.id, level="medium", score=0.5)
    decision = Decision(scenario_id=scenario.id, action="inspect", confidence=0.8)

    assert scenario.events == [event]
    assert report.scenario_id == scenario.id
    assert risk.score == 0.5
    assert decision.confidence == 0.8


def test_new_shared_schemas_validate_payloads() -> None:
    reading = SensorReading(
        sensor_id="sensor-1",
        metric="temperature",
        value=82.5,
        unit="celsius",
    )
    scenario = Scenario(name="High temperature", sensor_readings=[reading])
    alert = FaultAlert(
        scenario_id=scenario.id,
        title="Temperature warning",
        severity="high",
    )
    prediction = Prediction(
        scenario_id=scenario.id,
        predicted_event_type="overheating",
        probability=0.75,
    )
    response = AgentResponse(agent_name="inspector", scenario_id=scenario.id)

    assert scenario.sensor_readings == [reading]
    assert alert.scenario_id == scenario.id
    assert prediction.probability == 0.75
    assert isinstance(response, AgentReport)


def test_unexpected_exception_handler_returns_safe_error_response() -> None:
    request = Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/test",
            "headers": [],
            "query_string": b"",
            "server": ("testserver", 80),
            "scheme": "http",
            "http_version": "1.1",
        }
    )

    response = asyncio.run(
        handle_unexpected_exception(request, RuntimeError("internal detail"))
    )

    assert response.status_code == 500
    assert response.body == b'{"detail":"Internal server error"}'
