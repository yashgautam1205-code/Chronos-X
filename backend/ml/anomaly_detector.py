"""Isolation Forest training, persistence, and anomaly evidence reporting."""

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer

from backend.ml.preprocessing import (
    SENSOR_FEATURES,
    build_synchronized_multisensor_features,
    load_optional_multisensor_records,
    load_sensor_dataset,
    validate_feature_record,
)

MODEL_FILENAME = "isolation_forest.joblib"
MULTISENSOR_MODEL_FILENAME = "isolation_forest_multisensor.joblib"


def train_anomaly_detector(
    data_dir: Path,
    *,
    contamination: float = 0.01,
    random_state: int = 42,
) -> tuple[IsolationForest, dict[str, Any]]:
    """Train Isolation Forest on records from the normal synthetic scenario only."""
    if not 0 < contamination <= 0.5:
        raise ValueError("contamination must be in (0, 0.5]")
    readings, labels = load_sensor_dataset(data_dir)
    normal_ids = set(
        labels.loc[labels["scenario_name"] == "normal", "reading_id"].astype(str)
    )
    normal = readings[
        readings["reading_id"].astype(str).isin(normal_ids)
    ].sort_values(["timestamp", "reading_id"])
    split_index = int(len(normal) * 0.8)
    normal_train = normal.iloc[:split_index]
    if len(normal_train) < 10 or split_index == len(normal):
        raise ValueError("At least 10 valid normal sensor readings are required")

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(normal_train[list(SENSOR_FEATURES)])

    training_ids = set(normal_train["reading_id"].astype(str))
    evaluation = readings[
        ~readings["reading_id"].astype(str).isin(training_ids)
    ]
    labeled = evaluation.merge(
        labels[["reading_id", "ground_truth_fault"]],
        on="reading_id",
        validate="one_to_one",
    )
    scores = model.decision_function(labeled[list(SENSOR_FEATURES)])
    predicted_anomalies = scores < 0
    truth = labeled["ground_truth_fault"].to_numpy(dtype=bool)
    metrics = {
        "training_scenario": "normal",
        "training_records": int(len(normal_train)),
        "held_out_normal_records": int(len(normal) - len(normal_train)),
        "evaluation_records": int(len(labeled)),
        "ground_truth_fault_records": int(truth.sum()),
        "detected_anomaly_records": int(predicted_anomalies.sum()),
        "true_positive": int(np.logical_and(predicted_anomalies, truth).sum()),
        "false_positive": int(np.logical_and(predicted_anomalies, ~truth).sum()),
        "false_negative": int(np.logical_and(~predicted_anomalies, truth).sum()),
        "feature_names": list(SENSOR_FEATURES),
        "contamination": contamination,
        "random_state": random_state,
        "evaluation_note": (
            "Ground truth is used only for evaluation, never as an Isolation Forest feature. "
            "The newest 20% of normal readings and all non-training scenarios are held out."
        ),
    }
    return model, metrics


def train_multisensor_anomaly_detector(
    data_dir: Path,
    model_dir: Path,
    *,
    baseline_model: IsolationForest,
    contamination: float = 0.01,
    random_state: int = 42,
    sync_tolerance_seconds: float = 8.0,
) -> dict[str, Any]:
    """Compare a synchronized model with the baseline on the same held-out rows."""
    sensor_records = load_optional_multisensor_records(data_dir)
    if not sensor_records:
        return {
            "status": "not_evaluated_no_validated_sensor_data",
            "reason": "No coolant-flow or electrical measurement files were found.",
        }

    readings, labels = load_sensor_dataset(data_dir)
    fused, available_features = build_synchronized_multisensor_features(
        readings,
        sensor_records,
        sync_tolerance_seconds=sync_tolerance_seconds,
    )
    if not available_features:
        return {
            "status": "not_evaluated_no_synchronized_sensor_data",
            "sensor_records": len(sensor_records),
            "reason": (
                "The optional measurements did not yield valid, same-equipment, "
                "event-time-aligned features within the configured tolerance."
            ),
        }

    normal_ids = set(
        labels.loc[labels["scenario_name"] == "normal", "reading_id"].astype(str)
    )
    normal = readings[
        readings["reading_id"].astype(str).isin(normal_ids)
    ].sort_values(["timestamp", "reading_id"])
    split_index = int(len(normal) * 0.8)
    normal_train = normal.iloc[:split_index]
    if len(normal_train) < 10 or split_index == len(normal):
        raise ValueError("At least 10 valid normal sensor readings are required")
    training_ids = set(normal_train["reading_id"].astype(str))

    fused["reading_id"] = fused["reading_id"].astype(str)
    train_rows = fused[fused["reading_id"].isin(training_ids)]
    learned_features = tuple(
        feature
        for feature in available_features
        if train_rows[feature].notna().any()
    )
    if not learned_features:
        return {
            "status": "not_evaluated_no_training_sensor_values",
            "sensor_records": len(sensor_records),
            "reason": "No added sensor feature has values in the normal training partition.",
        }
    feature_names = (*SENSOR_FEATURES, *learned_features)
    imputer = SimpleImputer(strategy="median")
    training_matrix = imputer.fit_transform(
        train_rows[list(feature_names)].replace([np.inf, -np.inf], np.nan)
    )
    if training_matrix.shape[1] != len(feature_names):
        raise ValueError("Multisensor imputation unexpectedly dropped a feature")
    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(training_matrix)

    evaluation = readings[
        ~readings["reading_id"].astype(str).isin(training_ids)
    ][["reading_id", *SENSOR_FEATURES]].merge(
        labels[["reading_id", "ground_truth_fault"]],
        on="reading_id",
        validate="one_to_one",
    )
    evaluation["reading_id"] = evaluation["reading_id"].astype(str)
    evaluation = evaluation.merge(
        fused[["reading_id", *learned_features]],
        on="reading_id",
        how="left",
        validate="one_to_one",
    )
    truth = evaluation["ground_truth_fault"].to_numpy(dtype=bool)
    baseline_predictions = baseline_model.decision_function(
        evaluation[list(SENSOR_FEATURES)]
    ) < 0
    multisensor_matrix = imputer.transform(
        evaluation[list(feature_names)].replace([np.inf, -np.inf], np.nan)
    )
    multisensor_predictions = model.decision_function(multisensor_matrix) < 0

    def metrics(predictions: np.ndarray) -> dict[str, float | int]:
        true_positive = int(np.logical_and(predictions, truth).sum())
        false_positive = int(np.logical_and(predictions, ~truth).sum())
        false_negative = int(np.logical_and(~predictions, truth).sum())
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
        return {
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(
                2 * precision * recall / (precision + recall)
                if precision + recall
                else 0.0,
                6,
            ),
        }

    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / MULTISENSOR_MODEL_FILENAME
    joblib.dump(
        {
            "model": model,
            "imputer": imputer,
            "feature_names": list(feature_names),
            "sync_tolerance_seconds": sync_tolerance_seconds,
            "random_state": random_state,
        },
        model_path,
    )
    return {
        "status": "evaluated",
        "training_scenario": "normal",
        "training_records": int(len(train_rows)),
        "evaluation_records": int(len(evaluation)),
        "ground_truth_fault_records": int(truth.sum()),
        "feature_names": list(feature_names),
        "sync_tolerance_seconds": sync_tolerance_seconds,
        "baseline": metrics(baseline_predictions),
        "multisensor": metrics(multisensor_predictions),
        "evaluation_protocol": (
            "Both models use the same held-out reading IDs and ground-truth labels; "
            "the multivariate imputer is fitted on normal training rows only. "
            "The original Isolation Forest artifact is unchanged."
        ),
        "evaluation_data_limitation": (
            "The checked-in labels describe generated scenarios, not verified "
            "industrial fault ground truth; these metrics measure scenario "
            "discrimination and do not establish field accuracy."
        ),
        "artifact": str(model_path),
    }


def save_anomaly_detector(
    model: IsolationForest,
    model_dir: Path,
    *,
    random_state: int = 42,
) -> Path:
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / MODEL_FILENAME
    joblib.dump(
        {
            "model": model,
            "feature_names": list(SENSOR_FEATURES),
            "random_state": random_state,
        },
        model_path,
    )
    return model_path


def load_anomaly_detector(
    model_path: Path,
) -> tuple[IsolationForest, tuple[str, ...]]:
    if not model_path.is_file():
        raise FileNotFoundError(f"Isolation Forest artifact not found: {model_path}")
    artifact = joblib.load(model_path)
    model = artifact.get("model")
    feature_names = tuple(artifact.get("feature_names", ()))
    if not isinstance(model, IsolationForest) or feature_names != SENSOR_FEATURES:
        raise ValueError(f"Invalid Isolation Forest artifact: {model_path}")
    return model, feature_names


def detect_anomalies(
    readings: pd.DataFrame | list[dict[str, object]],
    *,
    model_path: Path,
) -> list[dict[str, Any]]:
    """Score readings and return observed sensor evidence without inventing causes."""
    records = readings.to_dict(orient="records") if isinstance(readings, pd.DataFrame) else readings
    model, feature_names = load_anomaly_detector(model_path)
    if not records:
        return []
    feature_vectors = [
        validate_feature_record(record, feature_names)
        for record in records
    ]
    matrix = pd.DataFrame(
        np.vstack(feature_vectors),
        columns=feature_names,
    )
    decisions = model.decision_function(matrix)
    results = []
    for record, decision in zip(records, decisions):
        evidence = {
            feature: float(record[feature])
            for feature in feature_names
        }
        results.append(
            {
                "anomaly": bool(decision < 0),
                "anomaly_score": round(float(-decision), 6),
                "decision_function": round(float(decision), 6),
                "evidence": evidence,
                "evidence_id": record.get("reading_id", record.get("id")),
                "equipment_id": record.get("equipment_id"),
                "timestamp": record.get("timestamp"),
                "provenance": record.get("provenance"),
                "model": "IsolationForest",
            }
        )
    return results
