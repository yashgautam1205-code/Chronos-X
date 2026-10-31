"""Predictor agent using saved Isolation Forest and Random Forest artifacts."""

from pathlib import Path
from typing import Any

from backend.ml.anomaly_detector import detect_anomalies
from backend.ml.sensor_predictor import predict_future_sensors
from core.sensor_fusion import (
    analyze_sensor_readings,
    sensor_records_from_reconstruction,
)
from models import AgentResult

SENSOR_FEATURES = (
    "temperature_c",
    "pressure_kpa",
    "vibration_mm_s",
    "cooling_pump_status",
    "power_supply_status",
    "smoke_detected",
)


def _wide_readings(reconstruction: dict[str, Any]) -> list[dict[str, Any]]:
    supplied_readings = reconstruction.get("predictor_readings")
    if isinstance(supplied_readings, list):
        complete_readings = [
            reading
            for reading in supplied_readings
            if isinstance(reading, dict)
            and all(feature in reading for feature in SENSOR_FEATURES)
        ]
        if complete_readings:
            return sorted(
                complete_readings,
                key=lambda reading: (
                    str(reading.get("equipment_id", "")),
                    str(reading.get("timestamp", "")),
                    str(reading.get("reading_id", "")),
                ),
            )
    readings: dict[str, dict[str, Any]] = {}
    for item in reconstruction.get("reconstructed_timeline", []):
        original = item.get("original_evidence")
        if not isinstance(original, dict):
            continue
        if not all(feature in original for feature in SENSOR_FEATURES):
            continue
        reading_id = str(original.get("reading_id") or item.get("evidence_id"))
        readings[reading_id] = original
    return sorted(
        readings.values(),
        key=lambda reading: (
            str(reading.get("equipment_id", "")),
            str(reading.get("timestamp", "")),
            str(reading.get("reading_id", "")),
        ),
    )


def _additional_sensor_records(
    reconstruction: dict[str, Any],
) -> list[dict[str, Any]]:
    return sensor_records_from_reconstruction(reconstruction)


def run_predictor_agent(
    reconstruction: dict[str, Any],
    *,
    model_dir: Path,
) -> AgentResult:
    readings = _wide_readings(reconstruction)
    all_sensor_records = _additional_sensor_records(reconstruction)
    additional_records = [
        record
        for record in all_sensor_records
        if record.get("metric")
        not in {
            "temperature_c",
            "pressure_kpa",
            "vibration_mm_s",
            "cooling_pump_status",
            "power_supply_status",
            "smoke_detected",
        }
    ]
    if not readings and not additional_records:
        return AgentResult(
            agent_name="predictor",
            status="UNKNOWN",
            summary="No complete model records or supported additional sensor readings are available.",
            findings=[{"reason": "sensor_history_unavailable"}],
        )

    outputs: dict[str, Any] = {}
    errors: list[str] = []
    has_model_artifact = any(
        (model_dir / name).is_file()
        for name in (
            "isolation_forest.joblib",
            "random_forest_predictor.joblib",
        )
    )
    if all_sensor_records and (additional_records or has_model_artifact):
        outputs["additional_sensor_analysis"] = analyze_sensor_readings(
            all_sensor_records
        )
    if readings:
        try:
            outputs["anomaly_assessments"] = detect_anomalies(
                readings[-100:],
                model_path=model_dir / "isolation_forest.joblib",
            )
        except FileNotFoundError as error:
            errors.append(str(error))
        try:
            outputs["future_sensor_prediction"] = predict_future_sensors(
                readings[-3:],
                model_path=model_dir / "random_forest_predictor.joblib",
            )
        except (FileNotFoundError, ValueError) as error:
            errors.append(str(error))
    else:
        errors.append(
            "Trained sklearn models require the existing complete six-feature history; "
            "additional sensor features are reported without model inference."
        )

    status = (
        "COMPLETED"
        if "anomaly_assessments" in outputs
        or "future_sensor_prediction" in outputs
        else "UNKNOWN"
    )
    anomaly_assessments = outputs.get("anomaly_assessments", [])
    findings = [
        {
            "anomaly_count": sum(
                bool(result["anomaly"]) for result in anomaly_assessments
            ),
            "scored_reading_count": len(anomaly_assessments),
            "note": "Anomaly flags indicate model novelty, not a confirmed fault.",
        }
    ]
    if "future_sensor_prediction" in outputs:
        findings.append(
            {
                "predicted_temperature_c": outputs["future_sensor_prediction"][
                    "predicted_temperature_c"
                ],
                "predicted_pressure_kpa": outputs["future_sensor_prediction"][
                    "predicted_pressure_kpa"
                ],
                "horizon_readings": outputs["future_sensor_prediction"][
                    "horizon_readings"
                ],
            }
        )
    additional_analysis = outputs.get("additional_sensor_analysis", {})
    additional_indicators = additional_analysis.get("anomaly_indicators", [])
    if additional_analysis:
        findings.append(
            {
                "additional_sensor_indicator_count": len(additional_indicators),
                "additional_sensor_measurement_count": len(
                    additional_analysis.get("records", [])
                ),
                "note": (
                    "Rule-based sensor indicators are evidence prompts, not confirmed faults "
                    "or trained-model predictions."
                ),
            }
        )
    for indicator in additional_indicators:
        findings.append(
            {
                "indicator": indicator.get("indicator"),
                "status": indicator.get("status"),
                "equipment_id": indicator.get("equipment_id"),
                "evidence_ids": indicator.get("evidence_ids", []),
            }
        )
    evidence_ids = sorted(
        {
            str(result["evidence_id"])
            for result in anomaly_assessments
            if result.get("evidence_id")
        }
        | {
            str(evidence_id)
            for indicator in additional_indicators
            for evidence_id in indicator.get("evidence_ids", [])
        }
    )
    return AgentResult(
        agent_name="predictor",
        status=status,
        summary=(
            "Trained sklearn models evaluated reconstructed sensor evidence."
            if not errors
            else "Available model outputs are reported; unavailable inference is explicitly listed."
        ),
        findings=findings,
        evidence_ids=evidence_ids,
        provenance=[
            {
                "evidence_id": record.get("reading_id"),
                "source": record.get("provenance"),
                "timestamp": record.get("timestamp"),
            }
            for record in readings[-3:]
        ] + [
            {
                "evidence_id": record.get("reading_id") or record.get("id"),
                "source": record.get("provenance"),
                "timestamp": record.get("timestamp"),
            }
            for record in all_sensor_records
        ],
        outputs=outputs,
        errors=errors,
    )
