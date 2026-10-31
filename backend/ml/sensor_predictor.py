"""Random Forest future temperature and pressure forecasting."""

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from backend.ml.preprocessing import (
    TEMPORAL_FEATURES,
    build_temporal_dataset,
    validate_feature_record,
)

MODEL_FILENAME = "random_forest_predictor.joblib"


def train_sensor_predictors(
    readings: pd.DataFrame,
    *,
    horizon_readings: int = 5,
    test_fraction: float = 0.2,
    random_state: int = 42,
    n_estimators: int = 150,
) -> tuple[RandomForestRegressor, dict[str, Any]]:
    if n_estimators < 1:
        raise ValueError("n_estimators must be positive")
    dataset, feature_names = build_temporal_dataset(
        readings,
        horizon_readings=horizon_readings,
    )
    train, test = _split_by_time(dataset, test_fraction=test_fraction)
    metrics: dict[str, Any] = {
        "horizon_readings": horizon_readings,
        "training_records": int(len(train)),
        "testing_records": int(len(test)),
        "training_target_end": train["target_timestamp"].max().isoformat(),
        "testing_target_start": test["target_timestamp"].min().isoformat(),
        "feature_names": list(feature_names),
        "target_metrics": {},
        "leakage_controls": [
            "IDs, ground-truth labels, scenario names, receive timestamps, and future targets are excluded from model features.",
            "Target rows are shifted forward within each equipment stream.",
            "Train/test split is chronological by target timestamp with no target-time overlap.",
        ],
        "random_state": random_state,
    }
    target_names = ("future_temperature_c", "future_pressure_kpa")
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        min_samples_leaf=2,
        max_features=1.0,
        n_jobs=-1,
        random_state=random_state,
    )
    model.fit(train[list(feature_names)], train[list(target_names)])
    predictions = model.predict(test[list(feature_names)])
    for index, target in enumerate(target_names):
        actual = test[target].to_numpy(dtype=float)
        errors = predictions[:, index] - actual
        squared_correlation_denominator = float(np.sum((actual - np.mean(actual)) ** 2))
        r_squared = (
            1.0 - float(np.sum(errors**2)) / squared_correlation_denominator
            if squared_correlation_denominator > 0
            else 0.0
        )
        metrics["target_metrics"][target] = {
            "mae": round(float(np.mean(np.abs(errors))), 6),
            "rmse": round(float(np.sqrt(np.mean(errors**2))), 6),
            "r_squared": round(r_squared, 6),
            "samples": int(len(actual)),
        }
    return model, metrics


def _split_by_time(
    dataset: pd.DataFrame,
    *,
    test_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    from backend.ml.preprocessing import chronological_split

    return chronological_split(dataset, test_fraction=test_fraction)


def save_sensor_predictors(
    model: RandomForestRegressor,
    model_dir: Path,
    *,
    horizon_readings: int,
    random_state: int,
) -> Path:
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / MODEL_FILENAME
    joblib.dump(
        {
            "model": model,
            "feature_names": list(TEMPORAL_FEATURES),
            "target_names": ["future_temperature_c", "future_pressure_kpa"],
            "horizon_readings": horizon_readings,
            "random_state": random_state,
        },
        path,
    )
    return path


def load_sensor_predictors(
    model_path: Path,
) -> tuple[RandomForestRegressor, tuple[str, ...], int]:
    if not model_path.is_file():
        raise FileNotFoundError(f"Random Forest predictor artifact not found: {model_path}")
    artifact = joblib.load(model_path)
    model = artifact.get("model")
    feature_names = tuple(artifact.get("feature_names", ()))
    horizon = artifact.get("horizon_readings")
    if (
        not isinstance(model, RandomForestRegressor)
        or feature_names != TEMPORAL_FEATURES
        or artifact.get("target_names")
        != ["future_temperature_c", "future_pressure_kpa"]
        or not isinstance(horizon, int)
    ):
        raise ValueError(f"Invalid Random Forest predictor artifact: {model_path}")
    return model, feature_names, horizon


def create_prediction_features(
    history: list[dict[str, object]],
) -> dict[str, float]:
    """Build inference features from the current and previous two sensor rows."""
    if len(history) < 3:
        raise ValueError("At least three chronological sensor readings are required")
    timestamps = []
    equipment_ids = set()
    for record in history[-3:]:
        try:
            timestamp = pd.to_datetime(
                record["timestamp"],
                utc=True,
                errors="raise",
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Each historical sensor reading needs a valid timestamp") from error
        timestamps.append(timestamp)
        if record.get("equipment_id") is not None:
            equipment_ids.add(str(record["equipment_id"]))
    if timestamps != sorted(timestamps) or len(set(timestamps)) != len(timestamps):
        raise ValueError("Historical sensor readings must have strictly increasing timestamps")
    if len(equipment_ids) > 1:
        raise ValueError("Historical sensor readings must belong to one equipment stream")
    required_now = (
        "temperature_c",
        "pressure_kpa",
        "vibration_mm_s",
        "cooling_pump_status",
        "power_supply_status",
        "smoke_detected",
    )
    values = [
        validate_feature_record(row, required_now)
        for row in history[-3:]
    ]
    ordered = list(reversed(values))
    newest, previous, older = ordered
    row: dict[str, float] = {}
    for index, metric in enumerate(
        ("temperature_c", "pressure_kpa", "vibration_mm_s")
    ):
        current_value = float(newest[index])
        previous_value = float(previous[index])
        oldest_value = float(older[index])
        row[f"{metric}_lag_1"] = previous_value
        row[f"{metric}_lag_2"] = oldest_value
        row[f"{metric}_rolling_mean_3"] = float(
            np.mean([current_value, previous_value, oldest_value])
        )
        row[f"{metric}_trend_3"] = current_value - oldest_value
    row["cooling_pump_status"] = float(newest[3])
    row["power_supply_status"] = float(newest[4])
    row["smoke_detected"] = float(newest[5])
    return row


def predict_future_sensors(
    history: list[dict[str, object]],
    *,
    model_path: Path,
) -> dict[str, Any]:
    model, feature_names, horizon = load_sensor_predictors(model_path)
    features = create_prediction_features(history)
    vector = pd.DataFrame(
        [validate_feature_record(features, feature_names)],
        columns=feature_names,
    )
    last_record = history[-1]
    try:
        predicted_at = pd.to_datetime(last_record["timestamp"], utc=True, errors="raise")
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Latest history record must have a valid timestamp") from error
    predictions = model.predict(vector)[0]
    temperature = float(predictions[0])
    pressure = float(predictions[1])
    return {
        "predicted_temperature_c": temperature,
        "predicted_pressure_kpa": pressure,
        "horizon_readings": horizon,
        "prediction_origin_timestamp": predicted_at.isoformat(),
        "model": "RandomForestRegressor",
        "limitations": [
            "Forecasts reflect the synthetic training distribution and are not calibrated for plant control.",
            "The horizon is measured in equipment readings, not a guaranteed wall-clock duration.",
            "No prediction interval or causal guarantee is provided.",
        ],
        "input_evidence": [
            {
                "reading_id": record.get("reading_id", record.get("id")),
                "timestamp": record.get("timestamp"),
                "equipment_id": record.get("equipment_id"),
                "provenance": record.get("provenance"),
                "features": {
                    feature: record.get(feature)
                    for feature in (
                        "temperature_c",
                        "pressure_kpa",
                        "vibration_mm_s",
                        "cooling_pump_status",
                        "power_supply_status",
                        "smoke_detected",
                    )
                },
            }
            for record in history[-3:]
        ],
    }
