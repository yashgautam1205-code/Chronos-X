from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from fastapi.testclient import TestClient

from backend.agents.orchestrator import orchestrate_agents
from backend.agents.predictor import run_predictor_agent
from backend.agents.red_team import run_red_team_agent
from backend.agents.safety import run_safety_agent
from backend.ml.anomaly_detector import (
    detect_anomalies,
    load_anomaly_detector,
    train_multisensor_anomaly_detector,
)
from backend.ml.preprocessing import (
    SENSOR_FEATURES,
    build_temporal_dataset,
    build_synchronized_multisensor_features,
    load_sensor_dataset,
)
from backend.ml.sensor_predictor import (
    create_prediction_features,
    load_sensor_predictors,
    predict_future_sensors,
    train_sensor_predictors,
)
from backend.ml.training import train_all_models
from core.counterfactual_simulation import simulate_interventions
from core.reconstruction import reconstruct_timeline
from backend.main import app

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
BASE_TIME = datetime(2026, 1, 1, tzinfo=timezone.utc)


@pytest.fixture(scope="module")
def trained_models(tmp_path_factory):
    model_dir = tmp_path_factory.mktemp("trained-models")
    report = train_all_models(DATA_DIR, model_dir, random_state=47)
    return model_dir, report


def _readings() -> pd.DataFrame:
    readings, _ = load_sensor_dataset(DATA_DIR)
    return readings[readings["equipment_id"] == "EQ-001"].head(40).reset_index(
        drop=True
    )


def test_training_pipeline_persists_both_models_and_evaluation_metrics(
    trained_models,
) -> None:
    model_dir, report = trained_models

    assert (model_dir / "isolation_forest.joblib").is_file()
    assert (model_dir / "random_forest_predictor.joblib").is_file()
    assert (model_dir / "training_metrics.json").is_file()
    assert report["isolation_forest"]["training_scenario"] == "normal"
    assert report["isolation_forest"]["training_records"] > 0
    assert report["isolation_forest"]["held_out_normal_records"] > 0
    assert report["isolation_forest_multisensor"]["status"] == (
        "not_evaluated_no_validated_sensor_data"
    )
    assert (
        report["isolation_forest"]["training_records"]
        + report["isolation_forest"]["held_out_normal_records"]
        == 1250
    )
    predictor_metrics = report["random_forest_regressor"]["target_metrics"]
    assert set(predictor_metrics) == {
        "future_temperature_c",
        "future_pressure_kpa",
    }
    assert all(
        {"mae", "rmse", "r_squared"} <= set(metrics)
        for metrics in predictor_metrics.values()
    )
    assert (
        report["random_forest_regressor"]["training_target_end"]
        < report["random_forest_regressor"]["testing_target_start"]
    )
    anomaly_model, feature_names = load_anomaly_detector(
        model_dir / "isolation_forest.joblib"
    )
    predictor_model, predictor_features, horizon = load_sensor_predictors(
        model_dir / "random_forest_predictor.joblib"
    )
    assert anomaly_model
    assert feature_names == SENSOR_FEATURES
    assert isinstance(predictor_model, RandomForestRegressor)
    assert predictor_features
    assert horizon == 5


def test_multisensor_features_align_by_equipment_and_prior_event_time() -> None:
    timestamps = pd.to_datetime(
        [
            "2026-01-01T00:00:00Z",
            "2026-01-01T00:00:07Z",
            "2026-01-01T00:00:00Z",
        ],
        utc=True,
    )
    readings = pd.DataFrame(
        {
            "reading_id": ["a-0", "a-10", "b-0"],
            "equipment_id": ["EQ-A", "EQ-A", "EQ-B"],
            "timestamp": timestamps,
            "received_at": timestamps,
        }
    )

    def measurement(reading_id, equipment_id, metric, value, timestamp):
        return {
            "reading_id": reading_id,
            "equipment_id": equipment_id,
            "sensor_id": f"{equipment_id}-{metric}",
            "metric": metric,
            "value": value,
            "unit": "L/min",
            "timestamp": timestamp.isoformat(),
            "received_at": timestamp.isoformat(),
            "provenance": {"origin": "test fixture", "source_verified": True},
        }

    time_zero = timestamps[0]
    time_ten = timestamps[1]
    time_twenty = time_ten + pd.Timedelta(seconds=10)
    records = [
        measurement("a-flow-20", "EQ-A", "coolant_flow_lpm", 2.0, time_twenty),
        measurement("b-flow-0", "EQ-B", "coolant_flow_lpm", 50.0, time_zero),
        measurement("a-flow-0", "EQ-A", "coolant_flow_lpm", 12.0, time_zero),
    ]

    fused, feature_names = build_synchronized_multisensor_features(
        readings,
        records,
        sync_tolerance_seconds=8,
    )

    assert feature_names == ("coolant_flow_lpm",)
    assert fused["coolant_flow_lpm"].iloc[0] == 12.0
    assert fused["coolant_flow_lpm"].iloc[1] == 12.0
    assert fused["coolant_flow_lpm"].iloc[2] == 50.0


def test_multisensor_features_exclude_conflicts_and_equipment_mismatches() -> None:
    timestamp = pd.Timestamp("2026-01-01T00:00:00Z")
    readings = pd.DataFrame(
        {
            "reading_id": ["a-0"],
            "equipment_id": ["EQ-A"],
            "timestamp": [timestamp],
            "received_at": [timestamp],
        }
    )
    records = [
        {
            "reading_id": reading_id,
            "equipment_id": equipment_id,
            "sensor_id": sensor_id,
            "metric": "voltage_v",
            "value": value,
            "unit": "V",
            "timestamp": timestamp.isoformat(),
            "received_at": timestamp.isoformat(),
        }
        for reading_id, equipment_id, sensor_id, value in (
            ("a-voltage-1", "EQ-A", "voltage-1", 230.0),
            ("a-voltage-2", "EQ-A", "voltage-2", 240.0),
            ("b-voltage", "EQ-B", "voltage-3", 220.0),
        )
    ]

    fused, feature_names = build_synchronized_multisensor_features(
        readings,
        records,
    )

    assert feature_names == ()
    assert pd.isna(fused.loc[0, "voltage_v"])


def test_optional_multisensor_training_compares_the_same_held_out_readings(
    tmp_path, monkeypatch
) -> None:
    from sklearn.ensemble import IsolationForest

    timestamps = pd.date_range("2026-01-01", periods=40, freq="8s", tz="UTC")
    readings = pd.DataFrame(
        {
            "reading_id": [f"reading-{index}" for index in range(40)],
            "equipment_id": ["EQ-TEST"] * 40,
            "timestamp": timestamps,
            "received_at": timestamps,
            "temperature_c": [70.0 + index / 10 for index in range(40)],
            "pressure_kpa": [100.0 + index / 20 for index in range(40)],
            "vibration_mm_s": [2.0 + index / 100 for index in range(40)],
            "cooling_pump_status": [1] * 40,
            "power_supply_status": [1] * 40,
            "smoke_detected": [0] * 40,
        }
    )
    labels = pd.DataFrame(
        {
            "reading_id": readings["reading_id"],
            "scenario_name": ["normal"] * 20 + ["fault_fixture"] * 20,
            "ground_truth_fault": [False] * 20 + [True] * 20,
        }
    )
    monkeypatch.setattr(
        "backend.ml.anomaly_detector.load_sensor_dataset",
        lambda _: (readings.copy(), labels.copy()),
    )
    sensor_file = tmp_path / "coolant_flow_readings.csv"
    pd.DataFrame(
        {
            "reading_id": [f"flow-{index}" for index in range(40)],
            "equipment_id": ["EQ-TEST"] * 40,
            "sensor_id": ["EQ-TEST-FLOW"] * 40,
            "metric": ["coolant_flow_lpm"] * 40,
            "value": [20.0] * 20 + [12.0] * 20,
            "unit": ["L/min"] * 40,
            "timestamp": timestamps,
            "received_at": timestamps,
        }
    ).to_csv(sensor_file, index=False)
    baseline = IsolationForest(
        n_estimators=20,
        contamination=0.1,
        random_state=42,
    ).fit(readings.iloc[:16][list(SENSOR_FEATURES)])

    report = train_multisensor_anomaly_detector(
        tmp_path,
        tmp_path / "models",
        baseline_model=baseline,
        contamination=0.1,
    )

    assert report["status"] == "evaluated"
    assert report["evaluation_records"] == 24
    assert "coolant_flow_lpm" in report["feature_names"]
    assert report["baseline"].keys() == report["multisensor"].keys()
    assert Path(report["artifact"]).is_file()
    assert "generated scenarios" in report["evaluation_data_limitation"]


def test_isolation_forest_scores_normal_and_injected_abnormal_reading(
    trained_models,
) -> None:
    model_dir, _ = trained_models
    readings = _readings()
    normal = readings.iloc[0].to_dict()
    abnormal = {
        **normal,
        "reading_id": "injected-extreme-reading",
        "temperature_c": 350.0,
        "pressure_kpa": 800.0,
        "vibration_mm_s": 80.0,
        "cooling_pump_status": 0,
        "smoke_detected": 1,
    }

    results = detect_anomalies(
        [normal, abnormal],
        model_path=model_dir / "isolation_forest.joblib",
    )

    assert len(results) == 2
    assert results[1]["anomaly"] is True
    assert results[1]["anomaly_score"] > results[0]["anomaly_score"]
    assert results[1]["evidence"]["temperature_c"] == 350.0
    assert results[1]["evidence_id"] == "injected-extreme-reading"


def test_random_forest_predicts_future_temperature_and_pressure(
    trained_models,
) -> None:
    model_dir, _ = trained_models
    history = _readings().tail(10).to_dict(orient="records")

    features = create_prediction_features(history[-3:])
    prediction = predict_future_sensors(
        history[-3:],
        model_path=model_dir / "random_forest_predictor.joblib",
    )

    assert set(features) >= {
        "temperature_c_lag_1",
        "pressure_kpa_lag_2",
        "vibration_mm_s_trend_3",
        "cooling_pump_status",
    }
    assert np.isfinite(prediction["predicted_temperature_c"])
    assert np.isfinite(prediction["predicted_pressure_kpa"])
    assert prediction["horizon_readings"] == 5
    assert len(prediction["input_evidence"]) == 3


def test_preprocessing_excludes_labels_and_rejects_bad_inference_inputs(
    trained_models,
) -> None:
    readings, _ = load_sensor_dataset(DATA_DIR)
    temporal, features = build_temporal_dataset(readings, horizon_readings=3)

    assert "ground_truth_fault" not in features
    assert "future_temperature_c" not in features
    assert "future_pressure_kpa" not in features
    assert temporal["future_temperature_c"].notna().all()
    assert temporal["future_pressure_kpa"].notna().all()

    model_dir, _ = trained_models
    with pytest.raises(ValueError, match="Missing required sensor features"):
        detect_anomalies(
            [{"temperature_c": 70}],
            model_path=model_dir / "isolation_forest.joblib",
        )
    with pytest.raises(ValueError, match="finite"):
        detect_anomalies(
            [
                {
                    **_readings().iloc[0].to_dict(),
                    "pressure_kpa": float("nan"),
                }
            ],
            model_path=model_dir / "isolation_forest.joblib",
        )
    with pytest.raises(ValueError, match="At least three"):
        create_prediction_features([])
    history = _readings().tail(3).to_dict(orient="records")
    history[0], history[-1] = history[-1], history[0]
    with pytest.raises(ValueError, match="strictly increasing"):
        create_prediction_features(history)


def test_isolation_forest_handles_extreme_missing_and_invalid_values(
    trained_models,
) -> None:
    model_dir, _ = trained_models
    normal = _readings().iloc[0].to_dict()
    extreme = {
        **normal,
        "reading_id": "phase9-extreme-reading",
        "temperature_c": 1000.0,
        "pressure_kpa": 100_000.0,
        "vibration_mm_s": 1000.0,
        "cooling_pump_status": 0,
        "power_supply_status": 0,
        "smoke_detected": 1,
    }

    results = detect_anomalies(
        [normal, extreme],
        model_path=model_dir / "isolation_forest.joblib",
    )

    assert results[0]["anomaly"] is False
    assert results[1]["anomaly"] is True
    assert results[1]["anomaly_score"] > results[0]["anomaly_score"]
    with pytest.raises(ValueError, match="Missing required sensor features"):
        detect_anomalies(
            [{"temperature_c": 72.0}],
            model_path=model_dir / "isolation_forest.joblib",
        )
    with pytest.raises(ValueError, match="numeric"):
        detect_anomalies(
            [{**normal, "pressure_kpa": "invalid"}],
            model_path=model_dir / "isolation_forest.joblib",
        )
    with pytest.raises(ValueError, match="finite"):
        detect_anomalies(
            [{**normal, "pressure_kpa": float("nan")}],
            model_path=model_dir / "isolation_forest.joblib",
        )


def test_random_forest_rejects_missing_non_numeric_and_non_finite_history(
    trained_models,
) -> None:
    model_dir, _ = trained_models
    history = _readings().tail(3).to_dict(orient="records")
    model_path = model_dir / "random_forest_predictor.joblib"

    missing_feature = [
        {key: value for key, value in reading.items() if key != "temperature_c"}
        for reading in history
    ]
    with pytest.raises(ValueError, match="Missing required sensor features"):
        predict_future_sensors(missing_feature, model_path=model_path)
    with pytest.raises(ValueError, match="numeric"):
        predict_future_sensors(
            [{**history[0], "pressure_kpa": "invalid"}, *history[1:]],
            model_path=model_path,
        )
    with pytest.raises(ValueError, match="finite"):
        predict_future_sensors(
            [{**history[0], "pressure_kpa": float("nan")}, *history[1:]],
            model_path=model_path,
        )


def test_random_forest_is_measured_on_a_scenario_excluded_from_training() -> None:
    readings, labels = load_sensor_dataset(DATA_DIR)
    held_out_ids = set(
        labels.loc[labels["scenario_name"] == "sensor_malfunction", "reading_id"]
    )
    training_readings = readings.loc[
        ~readings["reading_id"].isin(held_out_ids)
    ].copy()
    held_out_readings = readings.loc[
        readings["reading_id"].isin(held_out_ids)
    ].copy()

    model, _ = train_sensor_predictors(training_readings, random_state=42)
    held_out_data, feature_names = build_temporal_dataset(
        held_out_readings,
        horizon_readings=5,
    )
    predictions = model.predict(held_out_data[list(feature_names)])
    actual = held_out_data[
        ["future_temperature_c", "future_pressure_kpa"]
    ].to_numpy(dtype=float)

    assert len(training_readings) == 8750
    assert len(held_out_data) == 1243
    assert predictions.shape == (1243, 2)
    assert np.isfinite(predictions).all()
    assert np.isfinite(actual).all()


def test_all_agents_execute_with_actual_upstream_outputs(trained_models) -> None:
    model_dir, _ = trained_models
    scenario_id = uuid4()
    readings = []
    for index in range(6):
        readings.append(
            {
                "reading_id": f"agent-reading-{index}",
                "timestamp": (BASE_TIME + timedelta(seconds=index)).isoformat(),
                "received_at": (BASE_TIME + timedelta(seconds=index)).isoformat(),
                "equipment_id": "EQ-AGENT",
                "temperature_sensor_id": "EQ-AGENT-TEMP",
                "pressure_sensor_id": "EQ-AGENT-PRES",
                "vibration_sensor_id": "EQ-AGENT-VIB",
                "cooling_pump_sensor_id": "EQ-AGENT-PUMP",
                "power_supply_sensor_id": "EQ-AGENT-POWER",
                "smoke_sensor_id": "EQ-AGENT-SMOKE",
                "temperature_c": 73.0 + index * 0.1,
                "pressure_kpa": 100.0 + index * 0.1,
                "vibration_mm_s": 2.0,
                "cooling_pump_status": 1,
                "power_supply_status": 1,
                "smoke_detected": 0,
                "provenance": "test_sensor",
            }
        )
    reconstruction = reconstruct_timeline(scenario_id, [], readings)
    simulation = simulate_interventions(scenario_id, rollouts=100, seed=88)

    result = orchestrate_agents(
        scenario_id,
        reconstruction,
        simulation_report=simulation,
        model_dir=model_dir,
    )

    assert result.status == "COMPLETED"
    assert {agent.agent_name for agent in result.agent_results} == {
        "investigator",
        "predictor",
        "optimizer",
        "safety",
        "red_team",
    }
    assert all(agent.status == "COMPLETED" for agent in result.agent_results)
    predictor = next(
        agent for agent in result.agent_results if agent.agent_name == "predictor"
    )
    assert predictor.outputs["anomaly_assessments"]
    assert predictor.outputs["future_sensor_prediction"]
    optimizer = next(
        agent for agent in result.agent_results if agent.agent_name == "optimizer"
    )
    assert optimizer.outputs["decision_inputs"]
    assert len(optimizer.outputs["decision_inputs"]) == 6
    assert all(
        "risk_adjusted_loss_index" in item
        for item in optimizer.outputs["decision_inputs"]
    )
    safety = next(
        agent for agent in result.agent_results if agent.agent_name == "safety"
    )
    assert safety.findings
    red_team = next(
        agent for agent in result.agent_results if agent.agent_name == "red_team"
    )
    assert red_team.outputs["conclusion"]


def test_predictor_agent_reports_unknown_when_artifacts_are_missing(tmp_path) -> None:
    scenario_id = uuid4()
    readings = [
        {
            "reading_id": f"missing-model-{index}",
            "timestamp": (BASE_TIME + timedelta(seconds=index)).isoformat(),
            "received_at": (BASE_TIME + timedelta(seconds=index)).isoformat(),
            "equipment_id": "EQ-MISSING",
            "temperature_sensor_id": "EQ-MISSING-TEMP",
            "pressure_sensor_id": "EQ-MISSING-PRES",
            "vibration_sensor_id": "EQ-MISSING-VIB",
            "cooling_pump_sensor_id": "EQ-MISSING-PUMP",
            "power_supply_sensor_id": "EQ-MISSING-POWER",
            "smoke_sensor_id": "EQ-MISSING-SMOKE",
            "temperature_c": 72.0,
            "pressure_kpa": 100.0,
            "vibration_mm_s": 2.0,
            "cooling_pump_status": 1,
            "power_supply_status": 1,
            "smoke_detected": 0,
            "provenance": "test",
        }
        for index in range(3)
    ]
    reconstruction = reconstruct_timeline(scenario_id, [], readings)

    result = run_predictor_agent(reconstruction, model_dir=tmp_path)

    assert result.status == "UNKNOWN"
    assert len(result.errors) == 2
    assert not result.outputs


def test_safety_agent_returns_unknown_when_no_candidate_outcomes_exist() -> None:
    result = run_safety_agent(
        {
            "scenario_id": str(uuid4()),
            "interventions": [],
            "sequential_action_plans": [],
        }
    )

    assert result.status == "UNKNOWN"
    assert result.outputs["overall_result"] == "UNKNOWN"
    assert result.outputs["escalate"] is True
    assert result.findings == [{"check": "simulation_evidence", "result": "UNKNOWN"}]


def test_orchestrator_quarantines_safety_agent_that_contradicts_simulation(
    trained_models,
    monkeypatch,
) -> None:
    from models import AgentResult

    scenario_id = uuid4()
    readings = _readings().head(3).to_dict(orient="records")
    reconstruction = reconstruct_timeline(scenario_id, [], readings)
    simulation = simulate_interventions(
        scenario_id,
        rollouts=100,
        seed=61,
        initial_temperature_c=99.0,
        thermal_incident_threshold_c=100.0,
        pump_failure_rate_per_hour=0,
        equipment_failure_rate_per_hour=0,
        backup_available_probability=1,
    )

    def compromised_safety_agent(_simulation):
        names = [
            item["intervention_name"]
            for item in [
                *_simulation["interventions"],
                *_simulation["sequential_action_plans"],
            ]
        ]
        return AgentResult(
            agent_name="safety",
            status="COMPLETED",
            summary="Injected contradictory safety output.",
            findings=[
                {"intervention_name": name, "result": "PASS"}
                for name in names
            ],
        )

    monkeypatch.setattr(
        "backend.agents.orchestrator.run_safety_agent",
        compromised_safety_agent,
    )
    model_dir, _ = trained_models
    result = orchestrate_agents(
        scenario_id,
        reconstruction,
        simulation_report=simulation,
        model_dir=model_dir,
    )

    safety = next(
        agent for agent in result.agent_results if agent.agent_name == "safety"
    )
    assert safety.status == "UNKNOWN"
    assert result.status == "PARTIAL"
    assert any(
        item["type"] == "safety_agent_simulation_disagreement"
        for item in result.conflicts
    )
    assert result.integrity_assurance.startswith("Cross-agent consistency")


def test_red_team_surfaces_delayed_unverified_historical_event() -> None:
    from models import IndustrialEvent

    scenario_id = uuid4()
    event = IndustrialEvent(
        timestamp=BASE_TIME,
        received_at=BASE_TIME + timedelta(seconds=20),
        source="UNVERIFIED-REPORTER",
        event_type="cooling_failure",
        description="Unverified historical claim",
        provenance={"origin": "untrusted_archive", "source_verified": False},
        metadata={
            "scenario_id": str(scenario_id),
            "equipment_id": "EQ-DIFFERENT",
        },
    )
    reconstruction = reconstruct_timeline(scenario_id, [event], [])

    result = run_red_team_agent(reconstruction)

    indicators = {item["indicator"] for item in result.findings}
    assert "delayed_evidence" in indicators
    assert "suspicious_source_provenance" in indicators
    assert "low_reliability_evidence" in indicators
    assert str(event.id) in result.evidence_ids
    assert "do not prove deliberate tampering" in result.outputs["conclusion"]


def test_anomaly_training_rejects_invalid_contamination() -> None:
    from backend.ml.anomaly_detector import train_anomaly_detector

    with pytest.raises(ValueError, match="contamination"):
        train_anomaly_detector(DATA_DIR, contamination=0)


def test_agent_orchestration_api_runs_all_five_agents() -> None:
    client = TestClient(app)
    scenario_id = uuid4()
    ingestion = client.post(
        "/events",
        json={
            "timestamp": BASE_TIME.isoformat(),
            "received_at": BASE_TIME.isoformat(),
            "source": "EQ-AGENT-API",
            "event_type": "temperature_warning",
            "description": "Test event for agent orchestration",
            "severity": "medium",
            "provenance": {"origin": "test", "source_verified": True},
            "metadata": {
                "scenario_id": str(scenario_id),
                "equipment_id": "EQ-AGENT-API",
            },
        },
    )
    assert ingestion.status_code == 201

    response = client.post(
        f"/scenarios/{scenario_id}/agents/run",
        json={"simulation": {"rollouts": 100, "seed": 71}},
    )

    assert response.status_code == 200
    result = response.json()
    assert result["scenario_id"] == str(scenario_id)
    assert result["shared_evidence_count"] == 1
    assert {agent["agent_name"] for agent in result["agent_results"]} == {
        "investigator",
        "predictor",
        "optimizer",
        "safety",
        "red_team",
    }
