"""Repeatable training command for the two approved sklearn model families."""

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from backend.ml.anomaly_detector import (
    save_anomaly_detector,
    train_anomaly_detector,
    train_multisensor_anomaly_detector,
)
from backend.ml.preprocessing import load_sensor_dataset
from backend.ml.sensor_predictor import (
    save_sensor_predictors,
    train_sensor_predictors,
)

logger = logging.getLogger(__name__)


def train_all_models(
    data_dir: Path,
    model_dir: Path,
    *,
    random_state: int = 42,
    horizon_readings: int = 5,
    test_fraction: float = 0.2,
) -> dict[str, Any]:
    readings, _ = load_sensor_dataset(data_dir)
    anomaly_model, anomaly_metrics = train_anomaly_detector(
        data_dir,
        random_state=random_state,
    )
    multisensor_metrics = train_multisensor_anomaly_detector(
        data_dir,
        model_dir,
        baseline_model=anomaly_model,
        random_state=random_state,
    )
    predictor_models, predictor_metrics = train_sensor_predictors(
        readings,
        horizon_readings=horizon_readings,
        test_fraction=test_fraction,
        random_state=random_state,
    )
    anomaly_path = save_anomaly_detector(
        anomaly_model,
        model_dir,
        random_state=random_state,
    )
    predictor_path = save_sensor_predictors(
        predictor_models,
        model_dir,
        horizon_readings=horizon_readings,
        random_state=random_state,
    )
    report: dict[str, Any] = {
        "random_state": random_state,
        "data_dir": str(data_dir),
        "model_dir": str(model_dir),
        "artifacts": {
            "isolation_forest": str(anomaly_path),
            "random_forest_regressor": str(predictor_path),
        },
        "isolation_forest": anomaly_metrics,
        "isolation_forest_multisensor": multisensor_metrics,
        "random_forest_regressor": predictor_metrics,
        "models_trained": ["IsolationForest", "RandomForestRegressor"],
    }
    model_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = model_dir / "training_metrics.json"
    metrics_path.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    logger.info("Saved model artifacts and training metrics to %s", model_dir)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("backend/ml/saved_models"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--horizon-readings", type=int, default=5)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    args = parser.parse_args()
    report = train_all_models(
        args.data_dir,
        args.model_dir,
        random_state=args.seed,
        horizon_readings=args.horizon_readings,
        test_fraction=args.test_fraction,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
