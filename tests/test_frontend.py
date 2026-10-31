from pathlib import Path

import httpx
import pytest

from frontend.api import APIError, ChronosAPI


def test_api_client_calls_backend_and_decodes_json(monkeypatch) -> None:
    captured = {}

    def fake_request(method, url, **kwargs):
        captured.update(method=method, url=url, **kwargs)
        return httpx.Response(200, json={"status": "ok"})

    monkeypatch.setattr(httpx, "request", fake_request)
    api = ChronosAPI("http://localhost:8000/")

    result = api.health()

    assert result == {"status": "ok"}
    assert captured["method"] == "GET"
    assert captured["url"] == "http://localhost:8000/health"


def test_api_client_ingests_synthetic_event(monkeypatch) -> None:
    captured = {}
    payload = {"source": "dashboard-test", "event_type": "test_alarm"}

    def fake_request(method, url, **kwargs):
        captured.update(method=method, url=url, **kwargs)
        return httpx.Response(201, json=payload)

    monkeypatch.setattr(httpx, "request", fake_request)
    api = ChronosAPI("http://localhost:8000")

    assert api.ingest_event(payload) == payload
    assert captured["method"] == "POST"
    assert captured["url"] == "http://localhost:8000/events"
    assert captured["json"] == payload


def test_api_client_submits_runtime_scenario(monkeypatch) -> None:
    captured = {}
    payload = {"name": "unseen", "events": [], "sensor_readings": []}

    def fake_request(method, url, **kwargs):
        captured.update(method=method, url=url, **kwargs)
        return httpx.Response(200, json={"scenario_id": "runtime-id"})

    monkeypatch.setattr(httpx, "request", fake_request)
    api = ChronosAPI("http://localhost:8000")

    result = api.analyze_runtime_scenario(payload)

    assert result == {"scenario_id": "runtime-id"}
    assert captured["method"] == "POST"
    assert captured["url"] == "http://localhost:8000/scenarios/analyze"
    assert captured["json"] == payload


def test_api_client_requests_backend_sensor_analysis(monkeypatch) -> None:
    captured = {}

    def fake_request(method, url, **kwargs):
        captured.update(method=method, url=url, **kwargs)
        return httpx.Response(200, json={"records": [], "anomaly_indicators": []})

    monkeypatch.setattr(httpx, "request", fake_request)
    scenario_id = "1b7c19a5-79b9-4c79-8f70-1140a7bced09"

    from uuid import UUID

    result = ChronosAPI("http://localhost:8000").scenario_sensor_analysis(
        UUID(scenario_id)
    )

    assert result["records"] == []
    assert captured["method"] == "GET"
    assert captured["url"] == (
        f"http://localhost:8000/scenarios/{scenario_id}/sensor-analysis"
    )


def test_api_client_requests_global_sensor_monitoring_data(monkeypatch) -> None:
    captured = {}

    def fake_request(method, url, **kwargs):
        captured.update(method=method, url=url, **kwargs)
        return httpx.Response(200, json={"records": []})

    monkeypatch.setattr(httpx, "request", fake_request)

    result = ChronosAPI("http://localhost:8000").sensor_analysis()

    assert result == {"records": []}
    assert captured["method"] == "GET"
    assert captured["url"] == "http://localhost:8000/sensors/analysis"


def test_api_client_generates_explicit_synthetic_telemetry(monkeypatch) -> None:
    from uuid import uuid4

    captured = {}

    def fake_request(method, url, **kwargs):
        captured.update(method=method, url=url, **kwargs)
        return httpx.Response(
            201,
            json={"data_classification": "SYNTHETIC", "inserted_readings": 18},
        )

    monkeypatch.setattr(httpx, "request", fake_request)
    scenario_id = uuid4()

    result = ChronosAPI("http://localhost:8000").generate_synthetic_telemetry(
        sample_count=3,
        seed=7,
        equipment_id="DEMO-FE",
        scenario_id=scenario_id,
    )

    assert result["data_classification"] == "SYNTHETIC"
    assert captured["method"] == "POST"
    assert captured["url"] == "http://localhost:8000/sensors/synthetic/generate"
    assert captured["json"] == {
        "sample_count": 3,
        "seed": 7,
        "equipment_id": "DEMO-FE",
        "scenario_id": str(scenario_id),
    }


def test_api_client_surfaces_backend_errors(monkeypatch) -> None:
    def fake_request(method, url, **kwargs):
        assert method == "GET"
        assert url == "http://localhost:8000/health"
        assert kwargs["timeout"] > 0
        return httpx.Response(
            404,
            json={"detail": "Scenario not found"},
        )

    monkeypatch.setattr(
        httpx,
        "request",
        fake_request,
    )

    with pytest.raises(APIError, match="404: Scenario not found"):
        ChronosAPI("http://localhost:8000").health()


def test_scenario_catalog_and_sensor_stream_load_actual_dataset() -> None:
    from frontend.app import load_scenario_catalog, load_sensor_stream

    catalog = load_scenario_catalog()
    normal = catalog.loc[catalog["scenario_name"] == "normal"].iloc[0]
    readings = load_sensor_stream(str(normal["scenario_id"]))

    assert len(catalog) == 8
    assert len(readings) > 0
    assert set(
        (
            "temperature_c",
            "pressure_kpa",
            "vibration_mm_s",
            "cooling_pump_status",
            "power_supply_status",
            "smoke_detected",
        )
    ).issubset(readings.columns)
    assert Path("data/sensor_readings.csv").is_file()


def test_sensor_trend_visualization_uses_actual_sensor_records() -> None:
    from frontend.app import _sensor_trend_figure

    assert (
        _sensor_trend_figure(
            [],
            metrics={"voltage_v"},
            title="Voltage",
            y_label="V",
        )
        is None
    )
    figure = _sensor_trend_figure(
        [
            {
                "metric": "voltage_v",
                "timestamp": "2026-10-31T00:00:00+00:00",
                "value": 230,
                "equipment_id": "PUMP-1",
            }
        ],
        metrics={"voltage_v"},
        title="Voltage",
        y_label="V",
    )

    assert figure is not None
    assert len(figure.data) == 1
    assert list(figure.data[0].y) == [230]

    synthetic_figure = _sensor_trend_figure(
        [
            {
                "metric": "voltage_v",
                "timestamp": "2026-10-31T00:00:00+00:00",
                "value": 230,
                "equipment_id": "DEMO-PUMP",
                "data_classification": "SYNTHETIC",
                "provenance": {
                    "origin": "SYNTHETIC_CHRONOS_DEMO_GENERATOR",
                    "synthetic": True,
                },
            }
        ],
        metrics={"voltage_v"},
        title="Voltage",
        y_label="V",
    )

    assert synthetic_figure is not None
    assert "SYNTHETIC" in synthetic_figure.data[0].name


def test_sensor_trend_accepts_mixed_iso8601_timestamp_precision() -> None:
    from frontend.app import _sensor_trend_figure

    figure = _sensor_trend_figure(
        [
            {
                "metric": "coolant_flow_lpm",
                "timestamp": "2026-10-31T00:00:00Z",
                "value": 20,
                "equipment_id": "PUMP-1",
            },
            {
                "metric": "coolant_flow_lpm",
                "timestamp": "2026-10-31T00:00:01.123456Z",
                "value": 19,
                "equipment_id": "PUMP-1",
            },
        ],
        metrics={"coolant_flow_lpm"},
        title="Coolant Flow Rate",
        y_label="Flow (L/min)",
    )

    assert figure is not None
    assert len(figure.data[0].x) == 2
    assert list(figure.data[0].y) == [20, 19]


def test_timeline_frame_serializes_mixed_sensor_and_event_values() -> None:
    import pyarrow as pa

    from frontend.app import _timeline_frame

    frame = _timeline_frame(
        [
            {"evidence_id": "event-1", "value": "1"},
            {"evidence_id": "sensor-1", "value": 72.174},
        ]
    )

    assert str(frame["value"].dtype) == "string"
    assert frame["value"].tolist() == ["1", "72.174"]
    assert pa.Table.from_pandas(frame, preserve_index=False).num_rows == 2


def test_all_new_sensor_trends_plot_readings_from_backend_analysis(
    tmp_path, monkeypatch
) -> None:
    from datetime import datetime, timedelta, timezone
    from uuid import uuid4

    from fastapi.testclient import TestClient

    from backend.main import app
    from frontend.app import _sensor_trend_figure

    monkeypatch.setenv(
        "CHRONOS_SENSOR_DB_PATH",
        str(tmp_path / "dashboard-sensors.sqlite3"),
    )
    client = TestClient(app)
    latest = datetime.now(timezone.utc)
    sensor_series = {
        "coolant_flow_lpm": ("L/min", {15.0, 14.0, 13.0}),
        "voltage_v": ("V", {230.0, 229.0, 228.0}),
        "current_a": ("A", {4.0, 4.5, 5.0}),
    }
    for metric, (unit, values) in sensor_series.items():
        for index, value in enumerate(sorted(values)):
            timestamp = latest - timedelta(minutes=2 - index)
            response = client.post(
                "/sensors",
                json={
                    "id": str(uuid4()),
                    "equipment_id": "EQ-DASHBOARD",
                    "sensor_id": f"EQ-DASHBOARD-{metric}",
                    "metric": metric,
                    "value": value,
                    "unit": unit,
                    "timestamp": timestamp.isoformat(),
                    "received_at": timestamp.isoformat(),
                    "provenance": {"origin": "dashboard integration test"},
                },
            )
            assert response.status_code == 201, response.text

    analysis_response = client.get("/sensors/analysis")
    assert analysis_response.status_code == 200
    records = analysis_response.json()["records"]
    expected_metrics = {
        "coolant_flow_lpm": {"15.0", "14.0", "13.0"},
        "voltage_v": {"230.0", "229.0", "228.0"},
        "current_a": {"4.0", "4.5", "5.0"},
    }
    for metric, values in expected_metrics.items():
        metric_records = [item for item in records if item["metric"] == metric]
        assert {str(item["value"]) for item in metric_records} == values
        figure = _sensor_trend_figure(
            metric_records,
            metrics={metric},
            title=metric,
            y_label=sensor_series[metric][0],
        )
        assert figure is not None
        assert len(figure.data) == 1
        assert len(figure.data[0].x) == 3
        assert "Timestamp" in figure.layout.xaxis.title.text
        assert sensor_series[metric][0] in figure.layout.yaxis.title.text
