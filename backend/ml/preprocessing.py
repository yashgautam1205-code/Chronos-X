"""Validated feature preparation for the two CHRONOS-X sklearn models."""

from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from core.sensor_fusion import analyze_sensor_readings

SENSOR_FEATURES: Final[tuple[str, ...]] = (
    "temperature_c",
    "pressure_kpa",
    "vibration_mm_s",
    "cooling_pump_status",
    "power_supply_status",
    "smoke_detected",
)
CONTINUOUS_FEATURES: Final[tuple[str, ...]] = (
    "temperature_c",
    "pressure_kpa",
    "vibration_mm_s",
)
STATUS_FEATURES: Final[tuple[str, ...]] = (
    "cooling_pump_status",
    "power_supply_status",
    "smoke_detected",
)
TEMPORAL_FEATURES: Final[tuple[str, ...]] = tuple(
    feature
    for metric in CONTINUOUS_FEATURES
    for feature in (
        f"{metric}_lag_1",
        f"{metric}_lag_2",
        f"{metric}_rolling_mean_3",
        f"{metric}_trend_3",
    )
) + STATUS_FEATURES
TEMP_TARGET: Final[str] = "future_temperature_c"
PRESSURE_TARGET: Final[str] = "future_pressure_kpa"
MULTISENSOR_COLUMNS: Final[dict[str, str]] = {
    "coolant_flow_lpm": "coolant_flow_lpm",
    "coolant_inlet_flow_lpm": "coolant_inlet_flow_lpm",
    "coolant_outlet_flow_lpm": "coolant_outlet_flow_lpm",
    "flow_fluctuation_lpm": "flow_fluctuation_lpm",
    "voltage_v": "voltage_v",
    "voltage_fluctuation_v": "voltage_fluctuation_v",
    "current_a": "current_a",
    "current_fluctuation_a": "current_fluctuation_a",
    "power_kw": "power_kw",
}


def load_optional_multisensor_records(data_dir: Path) -> list[dict[str, object]]:
    """Load real optional telemetry; absent modality files produce no fabricated rows."""
    records: list[dict[str, object]] = []
    for filename in ("coolant_flow_readings.csv", "electrical_readings.csv"):
        path = data_dir / filename
        if not path.is_file():
            continue
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        required = {"equipment_id", "sensor_id", "metric", "value", "unit", "timestamp"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(
                f"Optional sensor dataset {path} is missing columns: {sorted(missing)}"
            )
        records.extend(frame.to_dict(orient="records"))
    return records


def build_synchronized_multisensor_features(
    readings: pd.DataFrame,
    sensor_records: list[dict[str, object]],
    *,
    sync_tolerance_seconds: float = 8.0,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Attach prior, same-equipment measurements without cross-stream leakage."""
    if not np.isfinite(sync_tolerance_seconds) or sync_tolerance_seconds < 0:
        raise ValueError("sync_tolerance_seconds must be finite and non-negative")
    required = {"reading_id", "equipment_id", "timestamp"}
    missing = required.difference(readings.columns)
    if missing:
        raise ValueError(f"Base sensor readings are missing columns: {sorted(missing)}")
    frame = readings.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    if "received_at" in frame:
        frame["received_at"] = pd.to_datetime(
            frame["received_at"], utc=True, errors="coerce"
        )
    else:
        frame["received_at"] = frame["timestamp"]
    if frame["timestamp"].isna().any() or frame["equipment_id"].isna().any():
        raise ValueError("Base sensor readings need valid timestamps and equipment IDs")
    for feature in MULTISENSOR_COLUMNS.values():
        frame[feature] = np.nan
    if not sensor_records:
        return frame, ()

    ordered_records = sorted(
        sensor_records,
        key=lambda record: (
            str(record.get("equipment_id", "")),
            str(record.get("metric", "")),
            str(record.get("received_at", "")),
            str(record.get("timestamp", "")),
            str(record.get("id", record.get("reading_id", ""))),
        ),
    )
    validation = analyze_sensor_readings(
        ordered_records,
        sync_tolerance_seconds=sync_tolerance_seconds,
    )
    conflicted_ids = {
        evidence_id
        for conflict in validation["conflicting_observations"]
        for evidence_id in conflict["evidence_ids"]
    }
    duplicated_ids = {
        duplicate["evidence_id"] for duplicate in validation["duplicate_records"]
    }
    observations = pd.DataFrame(
        [
            record
            for record in validation["records"]
            if record["metric"] in MULTISENSOR_COLUMNS
            and record["evidence_id"] not in conflicted_ids | duplicated_ids
            and not {
                "delayed",
                "out_of_order",
                "received_before_event_time",
            }.intersection(record["flags"])
        ]
    )
    if observations.empty:
        return frame, ()

    observations["timestamp"] = pd.to_datetime(
        observations["timestamp"], utc=True, errors="coerce"
    )
    observations["received_at"] = pd.to_datetime(
        observations["received_at"], utc=True, errors="coerce"
    )
    observations["value"] = pd.to_numeric(observations["value"], errors="coerce")
    observations = observations.dropna(
        subset=["equipment_id", "timestamp", "received_at", "value"]
    )
    observations = observations[
        observations["received_at"] <= observations["timestamp"]
    ]
    tolerance = pd.Timedelta(seconds=sync_tolerance_seconds)

    for feature, metric in MULTISENSOR_COLUMNS.items():
        metric_observations = observations[
            observations["metric"] == feature
        ][["equipment_id", "timestamp", "value"]]
        if metric_observations.empty:
            continue
        feature_rows: list[pd.DataFrame] = []
        for equipment_id, base_group in frame.groupby("equipment_id", sort=False):
            source_group = metric_observations[
                metric_observations["equipment_id"] == equipment_id
            ].drop(columns="equipment_id")
            if source_group.empty:
                continue
            base_sorted = base_group.sort_values("timestamp", kind="stable")
            source_sorted = source_group.sort_values("timestamp", kind="stable")
            merged = pd.merge_asof(
                base_sorted[["reading_id", "timestamp"]],
                source_sorted.rename(
                    columns={"timestamp": "sensor_timestamp", "value": feature}
                ).sort_values("sensor_timestamp", kind="stable"),
                left_on="timestamp",
                right_on="sensor_timestamp",
                direction="backward",
                tolerance=tolerance,
            )
            feature_rows.append(merged[["reading_id", feature]])
        if feature_rows:
            values = pd.concat(feature_rows, ignore_index=True)
            frame = frame.merge(values, on="reading_id", how="left", suffixes=("", "_new"))
            if f"{feature}_new" in frame:
                frame[feature] = frame[f"{feature}_new"]
                frame = frame.drop(columns=f"{feature}_new")
    available = tuple(
        feature
        for feature in MULTISENSOR_COLUMNS
        if frame[feature].notna().any()
    )
    return frame, available


def load_sensor_dataset(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load sensor features and labels separately, joining only by reading ID."""
    readings_path = data_dir / "sensor_readings.csv"
    labels_path = data_dir / "ground_truth_labels.csv"
    if not readings_path.is_file():
        raise FileNotFoundError(f"Sensor dataset not found: {readings_path}")
    if not labels_path.is_file():
        raise FileNotFoundError(f"Ground-truth dataset not found: {labels_path}")

    readings = pd.read_csv(readings_path)
    labels = pd.read_csv(labels_path)
    required_features = {"reading_id", "timestamp", "equipment_id", *SENSOR_FEATURES}
    required_labels = {"reading_id", "scenario_name", "ground_truth_fault", "fault_type"}
    missing_features = required_features.difference(readings.columns)
    missing_labels = required_labels.difference(labels.columns)
    if missing_features:
        raise ValueError(f"Sensor dataset is missing columns: {sorted(missing_features)}")
    if missing_labels:
        raise ValueError(f"Label dataset is missing columns: {sorted(missing_labels)}")
    if readings["reading_id"].duplicated().any():
        raise ValueError("Sensor dataset contains duplicate reading_id values")
    if labels["reading_id"].duplicated().any():
        raise ValueError("Label dataset contains duplicate reading_id values")

    readings["timestamp"] = pd.to_datetime(readings["timestamp"], utc=True, errors="coerce")
    for feature in SENSOR_FEATURES:
        readings[feature] = pd.to_numeric(readings[feature], errors="coerce")
    invalid_rows = readings["timestamp"].isna() | readings[list(SENSOR_FEATURES)].isna().any(axis=1)
    if invalid_rows.any():
        raise ValueError(
            f"Sensor dataset contains {int(invalid_rows.sum())} invalid or missing feature rows"
        )
    if not np.isfinite(readings[list(SENSOR_FEATURES)].to_numpy(dtype=float)).all():
        raise ValueError("Sensor dataset contains non-finite feature values")
    for status_feature in STATUS_FEATURES:
        if not readings[status_feature].isin((0, 1)).all():
            raise ValueError(f"Sensor dataset contains invalid {status_feature} values")
    for nonnegative_feature in ("pressure_kpa", "vibration_mm_s"):
        if (readings[nonnegative_feature] < 0).any():
            raise ValueError(f"Sensor dataset contains negative {nonnegative_feature} values")

    labels["ground_truth_fault"] = labels["ground_truth_fault"].map(
        {"True": True, "False": False, True: True, False: False}
    )
    if labels["ground_truth_fault"].isna().any():
        raise ValueError("Ground-truth labels contain invalid boolean values")
    merged_ids = readings[["reading_id"]].merge(
        labels[["reading_id"]],
        on="reading_id",
        how="left",
        indicator=True,
    )
    if (merged_ids["_merge"] != "both").any():
        raise ValueError("Sensor readings are missing matching ground-truth labels")
    return readings.sort_values(["equipment_id", "timestamp"]).reset_index(drop=True), labels


def build_temporal_dataset(
    readings: pd.DataFrame,
    *,
    horizon_readings: int = 5,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Create lagged features and future targets independently per equipment."""
    if horizon_readings < 1:
        raise ValueError("horizon_readings must be at least one")
    required = {"equipment_id", "timestamp", *SENSOR_FEATURES}
    missing = required.difference(readings.columns)
    if missing:
        raise ValueError(f"Sensor history is missing columns: {sorted(missing)}")
    frame = readings.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    for feature in SENSOR_FEATURES:
        frame[feature] = pd.to_numeric(frame[feature], errors="coerce")
    invalid_rows = frame["timestamp"].isna() | frame[list(SENSOR_FEATURES)].isna().any(axis=1)
    if invalid_rows.any():
        raise ValueError(f"Sensor history contains {int(invalid_rows.sum())} invalid rows")
    if not np.isfinite(frame[list(SENSOR_FEATURES)].to_numpy(dtype=float)).all():
        raise ValueError("Sensor history contains non-finite values")
    for status_feature in STATUS_FEATURES:
        if not frame[status_feature].isin((0, 1)).all():
            raise ValueError(f"Sensor history contains invalid {status_feature} values")
    for nonnegative_feature in ("pressure_kpa", "vibration_mm_s"):
        if (frame[nonnegative_feature] < 0).any():
            raise ValueError(f"Sensor history contains negative {nonnegative_feature} values")

    frame = frame.sort_values(["equipment_id", "timestamp"], kind="stable")
    groups = frame.groupby("equipment_id", sort=False, observed=True)
    for metric in CONTINUOUS_FEATURES:
        frame[f"{metric}_lag_1"] = groups[metric].shift(1)
        frame[f"{metric}_lag_2"] = groups[metric].shift(2)
        frame[f"{metric}_rolling_mean_3"] = groups[metric].transform(
            lambda series: series.rolling(window=3, min_periods=3).mean()
        )
        frame[f"{metric}_trend_3"] = frame[metric] - frame[f"{metric}_lag_2"]
    frame[TEMP_TARGET] = groups["temperature_c"].shift(-horizon_readings)
    frame[PRESSURE_TARGET] = groups["pressure_kpa"].shift(-horizon_readings)
    frame["target_timestamp"] = groups["timestamp"].shift(-horizon_readings)
    frame["target_timestamp"] = pd.to_datetime(frame["target_timestamp"], utc=True)
    columns = (
        "reading_id",
        "equipment_id",
        "timestamp",
        "target_timestamp",
        *TEMPORAL_FEATURES,
        TEMP_TARGET,
        PRESSURE_TARGET,
    )
    return frame[list(columns)].dropna().reset_index(drop=True), TEMPORAL_FEATURES


def chronological_split(
    dataset: pd.DataFrame,
    *,
    test_fraction: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by target time so future targets cannot enter the training set."""
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between zero and one")
    if dataset.empty or "target_timestamp" not in dataset:
        raise ValueError("Temporal dataset must have rows and target_timestamp")
    cutoff = dataset["target_timestamp"].quantile(1 - test_fraction)
    train = dataset[dataset["target_timestamp"] < cutoff].copy()
    test = dataset[dataset["target_timestamp"] >= cutoff].copy()
    if train.empty or test.empty:
        raise ValueError("Chronological split produced an empty train or test set")
    if train["target_timestamp"].max() >= test["target_timestamp"].min():
        raise ValueError("Chronological split overlaps the training and test targets")
    return train.reset_index(drop=True), test.reset_index(drop=True)


def validate_feature_record(
    record: dict[str, object],
    required_features: tuple[str, ...],
) -> np.ndarray:
    """Validate a single model input and return an ordered finite numeric vector."""
    missing = [feature for feature in required_features if feature not in record]
    if missing:
        raise ValueError(f"Missing required sensor features: {missing}")
    try:
        values = np.asarray([float(record[feature]) for feature in required_features])
    except (TypeError, ValueError) as error:
        raise ValueError("Sensor features must be numeric") from error
    if not np.isfinite(values).all():
        raise ValueError("Sensor features must be finite numeric values")
    feature_values = dict(zip(required_features, values))
    for status_feature in STATUS_FEATURES:
        if status_feature in feature_values and feature_values[status_feature] not in (0, 1):
            raise ValueError(f"{status_feature} must be 0 or 1")
    for nonnegative_feature in ("pressure_kpa", "vibration_mm_s"):
        if (
            nonnegative_feature in feature_values
            and feature_values[nonnegative_feature] < 0
        ):
            raise ValueError(f"{nonnegative_feature} must not be negative")
    return values
