"""Event-time normalization and evidence features for industrial sensors."""

from bisect import bisect_right
from datetime import datetime, timezone
from itertools import combinations
import json
from math import isfinite, sqrt
from statistics import median, pstdev
from typing import Any


_METRICS: dict[str, dict[str, tuple[str, float]]] = {
    "temperature_c": {"degc": ("degC", 1.0), "°c": ("degC", 1.0), "c": ("degC", 1.0)},
    "pressure_kpa": {"kpa": ("kPa", 1.0), "pa": ("kPa", 0.001)},
    "vibration_mm_s": {"mm/s": ("mm/s", 1.0)},
    "cooling_pump_status": {"status": ("status", 1.0), "boolean": ("status", 1.0)},
    "power_supply_status": {"status": ("status", 1.0), "boolean": ("status", 1.0)},
    "smoke_detected": {"status": ("status", 1.0), "boolean": ("status", 1.0)},
    "coolant_flow_lpm": {"l/min": ("L/min", 1.0), "lpm": ("L/min", 1.0), "l/s": ("L/min", 60.0)},
    "coolant_inlet_flow_lpm": {"l/min": ("L/min", 1.0), "lpm": ("L/min", 1.0), "l/s": ("L/min", 60.0)},
    "coolant_outlet_flow_lpm": {"l/min": ("L/min", 1.0), "lpm": ("L/min", 1.0), "l/s": ("L/min", 60.0)},
    "flow_fluctuation_lpm": {"l/min": ("L/min", 1.0), "lpm": ("L/min", 1.0)},
    "flow_failure_status": {"status": ("status", 1.0), "boolean": ("status", 1.0)},
    "voltage_v": {"v": ("V", 1.0), "mv": ("V", 0.001)},
    "voltage_fluctuation_v": {"v": ("V", 1.0), "mv": ("V", 0.001)},
    "current_a": {"a": ("A", 1.0), "ma": ("A", 0.001)},
    "current_fluctuation_a": {"a": ("A", 1.0), "ma": ("A", 0.001)},
    "power_kw": {"kw": ("kW", 1.0), "w": ("kW", 0.001)},
    "current_overload": {"status": ("status", 1.0), "boolean": ("status", 1.0)},
    "electrical_fault_status": {"status": ("status", 1.0), "boolean": ("status", 1.0)},
}
_ALIASES = {
    "coolant_flow_rate": "coolant_flow_lpm",
    "coolant_flow_rate_lpm": "coolant_flow_lpm",
    "coolant_inlet_flow_rate": "coolant_inlet_flow_lpm",
    "coolant_inlet_lpm": "coolant_inlet_flow_lpm",
    "coolant_outlet_flow_rate": "coolant_outlet_flow_lpm",
    "coolant_outlet_lpm": "coolant_outlet_flow_lpm",
    "flow_fluctuation": "flow_fluctuation_lpm",
    "coolant_flow_fluctuation_lpm": "flow_fluctuation_lpm",
    "flow_interruption_status": "flow_failure_status",
    "coolant_flow_failure_status": "flow_failure_status",
    "voltage": "voltage_v",
    "voltage_volts": "voltage_v",
    "current": "current_a",
    "current_amps": "current_a",
    "electrical_fault": "electrical_fault_status",
    "power_consumption_kw": "power_kw",
}
_STATUS_METRICS = {
    "cooling_pump_status",
    "power_supply_status",
    "smoke_detected",
    "flow_failure_status",
    "current_overload",
    "electrical_fault_status",
}
_NONNEGATIVE_METRICS = {
    "pressure_kpa",
    "vibration_mm_s",
    "coolant_flow_lpm",
    "coolant_inlet_flow_lpm",
    "coolant_outlet_flow_lpm",
    "flow_fluctuation_lpm",
    "voltage_v",
    "voltage_fluctuation_v",
    "current_a",
    "current_fluctuation_a",
    "power_kw",
}
_CORRELATION_METRICS = (
    "temperature_c",
    "pressure_kpa",
    "vibration_mm_s",
    "coolant_flow_lpm",
    "voltage_v",
    "current_a",
)
_CORRELATION_PAIRS = tuple(combinations(_CORRELATION_METRICS, 2))
_HEALTH_EVIDENCE_METRICS = _CORRELATION_METRICS
_WIDE_METRICS = {
    "temperature_c": "temperature_sensor_id",
    "pressure_kpa": "pressure_sensor_id",
    "vibration_mm_s": "vibration_sensor_id",
    "cooling_pump_status": "cooling_pump_sensor_id",
    "power_supply_status": "power_supply_sensor_id",
    "smoke_detected": "smoke_sensor_id",
}


def sensor_records_from_reconstruction(
    reconstruction: dict[str, Any],
) -> list[dict[str, Any]]:
    """Extract generic and legacy wide-row sensor measurements from evidence."""
    records: list[dict[str, Any]] = []
    seen_wide_readings: set[str] = set()
    for item in reconstruction.get("reconstructed_timeline", []):
        original = item.get("original_evidence")
        if not isinstance(original, dict):
            continue
        if "metric" in original and "value" in original:
            records.append(
                {
                    **original,
                    "reading_id": str(
                        original.get("reading_id")
                        or original.get("id")
                        or item.get("evidence_id")
                    ),
                    "equipment_id": original.get("equipment_id")
                    or item.get("equipment_id"),
                    "timestamp": original.get("timestamp") or item.get("timestamp"),
                    "received_at": original.get("received_at")
                    or item.get("received_at"),
                    "sensor_id": original.get("sensor_id") or item.get("sensor_id"),
                    "flags": item.get("flags", []),
                }
            )
            continue
        equipment_id = original.get("equipment_id") or item.get("equipment_id")
        wide_reading_id = str(
            original.get("reading_id")
            or original.get("id")
            or item.get("evidence_id")
        )
        wide_signature = json.dumps(original, sort_keys=True, default=str)
        if wide_signature in seen_wide_readings:
            continue
        seen_wide_readings.add(wide_signature)
        for metric, sensor_id_field in _WIDE_METRICS.items():
            if metric not in original:
                continue
            records.append(
                {
                    "reading_id": f"{wide_reading_id}:{metric}",
                    "equipment_id": equipment_id,
                    "sensor_id": original.get(sensor_id_field)
                    or f"{equipment_id}-{metric}",
                    "metric": metric,
                    "value": original[metric],
                    "unit": {
                        "temperature_c": "degC",
                        "pressure_kpa": "kPa",
                        "vibration_mm_s": "mm/s",
                        "cooling_pump_status": "status",
                        "power_supply_status": "status",
                        "smoke_detected": "status",
                    }[metric],
                    "timestamp": original.get("timestamp") or item.get("timestamp"),
                    "received_at": original.get("received_at")
                    or item.get("received_at"),
                    "provenance": original.get("provenance", {}),
                    "metadata": original.get("metadata", {}),
                    "flags": item.get("flags", []),
                }
            )
    return records


def _missing_intervals(
    streams: dict[tuple[str, str], list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    missing: list[dict[str, Any]] = []
    for (equipment_id, metric), stream in streams.items():
        timestamps = [item["_event_time"] for item in stream]
        intervals = [
            (later - earlier).total_seconds()
            for earlier, later in zip(timestamps, timestamps[1:])
            if later > earlier
        ]
        if len(intervals) < 2:
            continue
        cadence = median(intervals)
        if cadence <= 0:
            continue
        for earlier, later in zip(timestamps, timestamps[1:]):
            gap = (later - earlier).total_seconds()
            if gap <= cadence * 1.5:
                continue
            missing.append(
                {
                    "equipment_id": equipment_id,
                    "metric": metric,
                    "expected_interval_seconds": round(cadence, 6),
                    "missing_count": max(1, round(gap / cadence) - 1),
                    "after_timestamp": earlier.isoformat(),
                    "before_timestamp": later.isoformat(),
                }
            )
    return missing


def _parse_aware_timestamp(value: Any) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    else:
        raise ValueError("timestamp must be an ISO-8601 datetime")
    if parsed.tzinfo is None:
        raise ValueError("timestamp must include a timezone")
    return parsed.astimezone(timezone.utc)


def _canonical_metric(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("metric must be a string")
    metric = value.strip().lower()
    metric = _ALIASES.get(metric, metric)
    if metric not in _METRICS:
        raise ValueError(f"unsupported metric: {value}")
    return metric


def _normalize_record(record: dict[str, Any], index: int) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise ValueError("sensor reading must be an object")
    equipment_id = record.get("equipment_id")
    sensor_id = record.get("sensor_id")
    if not isinstance(equipment_id, str) or not equipment_id.strip():
        raise ValueError("equipment_id is required")
    if not isinstance(sensor_id, str) or not sensor_id.strip():
        raise ValueError("sensor_id is required")
    metadata = record.get("metadata") or {}
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be an object")
    declared_equipment = metadata.get("equipment_id")
    if declared_equipment is not None and str(declared_equipment) != equipment_id:
        raise ValueError("metadata equipment_id does not match equipment_id")
    metadata_equipment_id = equipment_id.strip()

    metric = _canonical_metric(record.get("metric"))
    declared_rating = metadata.get("rated_current_a")
    if declared_rating is not None:
        try:
            numeric_rating = float(declared_rating)
        except (TypeError, ValueError) as error:
            raise ValueError("metadata rated_current_a must be a positive number") from error
        if not isfinite(numeric_rating) or numeric_rating <= 0:
            raise ValueError("metadata rated_current_a must be a positive number")
    unit = record.get("unit")
    if not isinstance(unit, str):
        raise ValueError("unit is required")
    unit_info = _METRICS[metric].get(unit.strip().lower())
    if unit_info is None:
        raise ValueError(f"unsupported unit for {metric}: {unit}")
    try:
        value = float(record["value"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("value must be numeric") from error
    if not isfinite(value):
        raise ValueError("value must be finite")
    canonical_unit, factor = unit_info
    value *= factor
    if metric in _NONNEGATIVE_METRICS and value < 0:
        raise ValueError(f"{metric} cannot be negative")
    if metric in _STATUS_METRICS and value not in (0.0, 1.0):
        raise ValueError(f"{metric} must be 0 or 1")
    if metric == "temperature_c" and not -273.15 <= value <= 2000:
        raise ValueError("temperature_c is outside the supported physical range")

    event_time = _parse_aware_timestamp(record.get("timestamp"))
    received_at_supplied = record.get("received_at") is not None
    received_at = _parse_aware_timestamp(record.get("received_at") or event_time)
    provenance = record.get("provenance") or {}
    inherited_flags = record.get("flags", [])
    if not isinstance(inherited_flags, list) or not all(
        isinstance(flag, str) for flag in inherited_flags
    ):
        inherited_flags = []
    flags = list(inherited_flags)
    if not received_at_supplied:
        flags.append("received_at_inferred")
    trust_score = 1.0
    if "out_of_order" in flags:
        trust_score -= 0.15
    if "delayed" in flags:
        trust_score -= 0.15
    if "conflicting" in flags:
        trust_score -= 0.35
    if "duplicate" in flags:
        trust_score -= 0.25
    if received_at < event_time:
        flags.append("received_before_event_time")
        trust_score -= 0.1
    if not provenance:
        flags.append("provenance_not_provided")
        trust_score -= 0.15
    elif isinstance(provenance, dict):
        if provenance.get("source_verified") is not True:
            flags.append("source_verification_missing")
            trust_score -= 0.2
        if not provenance.get("origin"):
            flags.append("provenance_origin_missing")
            trust_score -= 0.1
    else:
        flags.append("unrecognized_provenance")
        trust_score = 0.75
    if "received_at_inferred" in flags:
        trust_score -= 0.05

    evidence_id = str(
        record.get("reading_id")
        or record.get("id")
        or f"sensor-reading-{index}"
    )
    return {
        "evidence_id": evidence_id,
        "equipment_id": metadata_equipment_id,
        "sensor_id": sensor_id.strip(),
        "metric": metric,
        "value": value,
        "unit": canonical_unit,
        "timestamp": event_time.isoformat(),
        "received_at": received_at.isoformat(),
        "provenance": provenance,
        "metadata": metadata,
        "trust_score": round(max(0.0, trust_score), 2),
        "flags": flags,
        "_event_time": event_time,
        "_received_at": received_at,
        "_input_index": index,
    }


def _pearson(values: list[tuple[float, float]]) -> float | None:
    if len(values) < 3:
        return None
    left = [pair[0] for pair in values]
    right = [pair[1] for pair in values]
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    left_variance = sum((value - left_mean) ** 2 for value in left)
    right_variance = sum((value - right_mean) ** 2 for value in right)
    denominator = sqrt(left_variance * right_variance)
    if denominator == 0:
        return None
    return round(
        sum(
            (left_value - left_mean) * (right_value - right_mean)
            for left_value, right_value in values
        )
        / denominator,
        4,
    )


def analyze_sensor_readings(
    records: list[dict[str, Any]],
    *,
    sync_tolerance_seconds: float = 5.0,
    flow_reduction_fraction: float = 0.3,
    electrical_deviation_threshold_sigma: float = 3.0,
    rated_current_a: dict[str, float] | None = None,
    stale_after_seconds: float = 300.0,
) -> dict[str, Any]:
    """Validate, align and summarize telemetry without inferring unobserved values."""
    if not isfinite(sync_tolerance_seconds) or sync_tolerance_seconds < 0:
        raise ValueError("sync_tolerance_seconds must be finite and non-negative")
    if not 0 < flow_reduction_fraction < 1:
        raise ValueError("flow_reduction_fraction must be between zero and one")
    if (
        not isfinite(electrical_deviation_threshold_sigma)
        or electrical_deviation_threshold_sigma <= 0
    ):
        raise ValueError(
            "electrical_deviation_threshold_sigma must be finite and positive"
        )
    if not isfinite(stale_after_seconds) or stale_after_seconds < 0:
        raise ValueError("stale_after_seconds must be finite and non-negative")
    analysis_time = datetime.now(timezone.utc)
    ratings = dict(rated_current_a or {})
    rating_candidates: dict[str, set[float]] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        try:
            metric = _canonical_metric(record.get("metric"))
        except ValueError:
            continue
        metadata = record.get("metadata")
        if (
            metric == "current_a"
            and isinstance(metadata, dict)
            and metadata.get("rated_current_a") is not None
        ):
            try:
                rating = float(metadata["rated_current_a"])
            except (TypeError, ValueError):
                continue
            equipment_id = record.get("equipment_id")
            if (
                isinstance(equipment_id, str)
                and equipment_id.strip()
                and isfinite(rating)
                and rating > 0
            ):
                rating_candidates.setdefault(equipment_id, set()).add(rating)
    conflicting_ratings = sorted(
        equipment_id
        for equipment_id, candidates in rating_candidates.items()
        if len(candidates) > 1 and equipment_id not in ratings
    )
    for equipment_id, candidates in rating_candidates.items():
        if len(candidates) == 1 and equipment_id not in ratings:
            ratings[equipment_id] = next(iter(candidates))
    for equipment_id, rating in ratings.items():
        try:
            numeric_rating = float(rating)
        except (TypeError, ValueError) as error:
            raise ValueError("rated_current_a entries need positive numeric values") from error
        if (
            not isinstance(equipment_id, str)
            or not equipment_id.strip()
            or isinstance(rating, bool)
            or not isfinite(numeric_rating)
            or numeric_rating <= 0
        ):
            raise ValueError("rated_current_a entries need equipment IDs and positive values")
        ratings[equipment_id] = numeric_rating

    normalized: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    previous_event_time: dict[tuple[str, str, str], datetime] = {}
    for index, record in enumerate(records):
        try:
            item = _normalize_record(record, index)
        except (KeyError, TypeError, ValueError) as error:
            rejected.append({"record_index": index, "reason": str(error)})
            continue
        staleness_seconds = (
            analysis_time - item["_event_time"]
        ).total_seconds()
        if staleness_seconds < 0:
            freshness_status = "FUTURE_TIMESTAMP"
        elif staleness_seconds > stale_after_seconds:
            freshness_status = "STALE"
        else:
            freshness_status = "CURRENT"
        provenance = item["provenance"]
        is_synthetic = (
            isinstance(provenance, dict)
            and (
                provenance.get("synthetic") is True
                or "synthetic" in str(provenance.get("origin", "")).lower()
            )
        ) or "synthetic" in str(provenance).lower() or (
            item["metadata"].get("data_classification") == "SYNTHETIC"
        )
        item["data_classification"] = "SYNTHETIC" if is_synthetic else "MEASURED"
        item["freshness_status"] = freshness_status
        item["staleness_seconds"] = round(max(0.0, staleness_seconds), 3)
        stream = (item["equipment_id"], item["sensor_id"], item["metric"])
        latest = previous_event_time.get(stream)
        if latest is not None and item["_event_time"] < latest:
            if "out_of_order" not in item["flags"]:
                item["flags"].append("out_of_order")
                item["trust_score"] = round(
                    max(0.0, item["trust_score"] - 0.15), 2
                )
        previous_event_time[stream] = max(latest or item["_event_time"], item["_event_time"])
        if (
            (item["_received_at"] - item["_event_time"]).total_seconds()
            > sync_tolerance_seconds
            and "delayed" not in item["flags"]
        ):
            item["flags"].append("delayed")
            item["trust_score"] = round(max(0.0, item["trust_score"] - 0.15), 2)
        normalized.append(item)

    duplicate_groups: dict[tuple[str, str, str, datetime], list[dict[str, Any]]] = {}
    for item in normalized:
        key = (
            item["equipment_id"],
            item["sensor_id"],
            item["metric"],
            item["_event_time"],
        )
        duplicate_groups.setdefault(key, []).append(item)

    duplicates: list[dict[str, Any]] = []
    conflicting: list[dict[str, Any]] = []
    usable: list[dict[str, Any]] = []
    for key, group in duplicate_groups.items():
        group.sort(key=lambda item: (item["_received_at"], item["evidence_id"]))
        values = {item["value"] for item in group}
        if len(values) > 1:
            for item in group:
                item["flags"].append("conflicting")
            conflicting.append(
                {
                    "equipment_id": key[0],
                    "sensor_id": key[1],
                    "metric": key[2],
                    "timestamp": key[3].isoformat(),
                    "evidence_ids": [item["evidence_id"] for item in group],
                    "values": sorted(values),
                    "resolution": None,
                }
            )
            continue
        representative = group[0]
        usable.append(representative)
        for duplicate in group[1:]:
            duplicate["flags"].append("duplicate")
            duplicates.append(
                {
                    "evidence_id": duplicate["evidence_id"],
                    "duplicate_of": representative["evidence_id"],
                }
            )

    # Conflicting same-time readings from different sensors are retained, but
    # deliberately excluded from fused values rather than arbitrarily selected.
    by_measurement_time: dict[tuple[str, str, datetime], list[dict[str, Any]]] = {}
    for item in usable:
        key = (item["equipment_id"], item["metric"], item["_event_time"])
        by_measurement_time.setdefault(key, []).append(item)
    conflict_keys: set[tuple[str, str, datetime]] = set()
    for key, group in by_measurement_time.items():
        if len({item["value"] for item in group}) > 1:
            conflict_keys.add(key)
            conflicting.append(
                {
                    "equipment_id": key[0],
                    "metric": key[1],
                    "timestamp": key[2].isoformat(),
                    "evidence_ids": [item["evidence_id"] for item in group],
                    "values": sorted({item["value"] for item in group}),
                    "resolution": None,
                }
            )
            for item in group:
                item["flags"].append("conflicting")
    usable = [
        item
        for item in usable
        if (item["equipment_id"], item["metric"], item["_event_time"]) not in conflict_keys
    ]

    streams: dict[tuple[str, str], list[dict[str, Any]]] = {}
    equipment_times: dict[str, set[datetime]] = {}
    for item in usable:
        streams.setdefault((item["equipment_id"], item["metric"]), []).append(item)
        equipment_times.setdefault(item["equipment_id"], set()).add(item["_event_time"])
    stream_times: dict[tuple[str, str], list[datetime]] = {}
    for key, stream in streams.items():
        stream.sort(key=lambda item: (item["_event_time"], item["evidence_id"]))
        stream_times[key] = [item["_event_time"] for item in stream]
    missing_intervals = _missing_intervals(streams)

    tolerance = sync_tolerance_seconds
    aligned: list[dict[str, Any]] = []
    anomalies: list[dict[str, Any]] = []
    correlations: list[dict[str, Any]] = []
    for equipment_id, times in sorted(equipment_times.items()):
        equipment_streams = {
            metric: stream
            for (asset, metric), stream in streams.items()
            if asset == equipment_id
        }
        source_by_id = {
            item["evidence_id"]: item
            for stream in equipment_streams.values()
            for item in stream
        }
        for event_time in sorted(times):
            row: dict[str, Any] = {
                "equipment_id": equipment_id,
                "timestamp": event_time.isoformat(),
                "measurements": {},
                "derived_features": {},
            }
            stream_positions: dict[str, int] = {}
            for metric, stream in equipment_streams.items():
                timestamps = stream_times[(equipment_id, metric)]
                position = bisect_right(timestamps, event_time) - 1
                if position < 0:
                    continue
                stream_positions[metric] = position
                candidate = stream[position]
                age = (event_time - candidate["_event_time"]).total_seconds()
                if age > tolerance:
                    continue
                row["measurements"][metric] = {
                    "value": candidate["value"],
                    "unit": candidate["unit"],
                    "event_timestamp": candidate["timestamp"],
                    "received_at": candidate["received_at"],
                    "evidence_id": candidate["evidence_id"],
                    "sensor_id": candidate["sensor_id"],
                    "age_seconds": round(age, 3),
                    "trust_score": candidate["trust_score"],
                    "data_classification": candidate["data_classification"],
                    "freshness_status": candidate["freshness_status"],
                    "staleness_seconds": candidate["staleness_seconds"],
                }
            aligned.append(row)

            flow = row["measurements"].get("coolant_flow_lpm")
            if flow and flow["event_timestamp"] == event_time.isoformat():
                flow_position = stream_positions["coolant_flow_lpm"]
                flow_history = equipment_streams["coolant_flow_lpm"][
                    max(0, flow_position - 20) : flow_position
                ]
                prior_flow = [item["value"] for item in flow_history]
                current_flow = float(flow["value"])
                baseline = median(prior_flow) if len(prior_flow) >= 3 else None
                row["derived_features"]["coolant_flow_fluctuation_lpm"] = (
                    round(pstdev(prior_flow[-5:]), 6)
                    if len(prior_flow) >= 2
                    else None
                )
                metadata = source_by_id[flow["evidence_id"]]["metadata"]
                operating = metadata.get("operating", metadata.get("equipment_operating"))
                if current_flow == 0:
                    anomalies.append(
                        {
                            "equipment_id": equipment_id,
                            "indicator": "observed_zero_coolant_flow",
                            "status": (
                                "ZERO_WHILE_NOT_OPERATING"
                                if operating is False
                                else "POSSIBLE_INTERRUPTION"
                            ),
                            "value": current_flow,
                            "baseline_lpm": baseline,
                            "operating_state": operating,
                            "evidence_ids": [flow["evidence_id"]],
                            "interpretation": (
                                "Zero flow was measured; this alone does not establish a cooling failure."
                            ),
                        }
                    )
                elif baseline is not None and baseline > 0:
                    reduction = (baseline - current_flow) / baseline
                    if reduction >= flow_reduction_fraction and operating is not False:
                        anomalies.append(
                            {
                                "equipment_id": equipment_id,
                                "indicator": "coolant_flow_reduction",
                                "status": "POSSIBLE_REDUCTION",
                                "value": current_flow,
                                "baseline_lpm": round(baseline, 6),
                                "reduction_fraction": round(reduction, 6),
                                "operating_state": operating,
                                "confidence": "medium" if operating is True else "low",
                                "evidence_ids": [flow["evidence_id"]],
                                "interpretation": (
                                    "Flow fell relative to this equipment's prior readings; verify operating "
                                    "state and setpoint before attributing it to a malfunction."
                                ),
                            }
                        )
            failure = row["measurements"].get("flow_failure_status")
            if (
                failure
                and failure["event_timestamp"] == event_time.isoformat()
                and failure["value"] == 1
            ):
                anomalies.append(
                    {
                        "equipment_id": equipment_id,
                        "indicator": "reported_flow_failure",
                        "status": "SENSOR_REPORTED",
                        "value": 1,
                        "evidence_ids": [failure["evidence_id"]],
                        "interpretation": (
                            "A sensor reported a flow-failure status; this report does not "
                            "establish a confirmed failure."
                        ),
                    }
                )

            for metric, feature_name, indicator_name in (
                ("voltage_v", "voltage_robust_z_score", "voltage_deviation_from_history"),
                ("current_a", "current_robust_z_score", "current_deviation_from_history"),
            ):
                measurement = row["measurements"].get(metric)
                if (
                    not measurement
                    or measurement["event_timestamp"] != event_time.isoformat()
                ):
                    continue
                position = stream_positions[metric]
                history = equipment_streams[metric][max(0, position - 20) : position]
                prior_values = [item["value"] for item in history]
                fluctuation_name = (
                    "voltage_fluctuation_v"
                    if metric == "voltage_v"
                    else "current_fluctuation_a"
                )
                row["derived_features"][fluctuation_name] = (
                    round(pstdev(prior_values[-5:]), 6)
                    if len(prior_values) >= 2
                    else None
                )
                robust_z_score = None
                deviation_from_baseline = False
                prior_median = None
                if len(prior_values) >= 5:
                    prior_median = median(prior_values)
                    mad = median([abs(value - prior_median) for value in prior_values])
                    if mad > 0:
                        robust_z_score = round(
                            0.6745 * (measurement["value"] - prior_median) / mad,
                            4,
                        )
                        deviation_from_baseline = (
                            abs(robust_z_score)
                            >= electrical_deviation_threshold_sigma
                        )
                    else:
                        deviation_from_baseline = (
                            measurement["value"] != prior_median
                        )
                row["derived_features"][feature_name] = robust_z_score
                if deviation_from_baseline:
                    anomalies.append(
                        {
                            "equipment_id": equipment_id,
                            "indicator": indicator_name,
                            "status": "POSSIBLE_DEVIATION",
                            "value": measurement["value"],
                            "baseline_value": prior_median,
                            "robust_z_score": robust_z_score,
                            "evidence_ids": [measurement["evidence_id"]],
                            "interpretation": (
                                "Measurement differs from its own historical baseline; confirm "
                                "operating mode, instrument quality and equipment rating."
                            ),
                        }
                    )

            current = row["measurements"].get("current_a")
            rating = ratings.get(equipment_id)
            if (
                current
                and current["event_timestamp"] == event_time.isoformat()
                and rating is not None
                and current["value"] > float(rating)
            ):
                anomalies.append(
                    {
                        "equipment_id": equipment_id,
                        "indicator": "current_above_configured_rating",
                        "status": "POSSIBLE_OVERLOAD",
                        "value": current["value"],
                        "rated_current_a": float(rating),
                        "evidence_ids": [current["evidence_id"]],
                        "interpretation": "Measured current exceeds the caller-supplied equipment rating.",
                    }
                )
            explicit_overload = row["measurements"].get("current_overload")
            if (
                explicit_overload
                and explicit_overload["event_timestamp"] == event_time.isoformat()
                and explicit_overload["value"] == 1
            ):
                anomalies.append(
                    {
                        "equipment_id": equipment_id,
                        "indicator": "reported_current_overload",
                        "status": "SENSOR_REPORTED",
                        "evidence_ids": [explicit_overload["evidence_id"]],
                        "interpretation": "An input sensor reported an overload; this is not independently verified.",
                    }
                )
            fault = row["measurements"].get("electrical_fault_status")
            if (
                fault
                and fault["event_timestamp"] == event_time.isoformat()
                and fault["value"] == 1
            ):
                anomalies.append(
                    {
                        "equipment_id": equipment_id,
                        "indicator": "reported_electrical_fault",
                        "status": "SENSOR_REPORTED",
                        "evidence_ids": [fault["evidence_id"]],
                        "interpretation": "A sensor reported an electrical fault; this is not independently verified.",
                    }
                )

            flow = row["measurements"].get("coolant_flow_lpm")
            temperature = row["measurements"].get("temperature_c")
            pressure = row["measurements"].get("pressure_kpa")
            if flow and (temperature or pressure):
                flow_findings = [
                    item
                    for item in anomalies
                    if item["equipment_id"] == equipment_id
                    and item["indicator"]
                    in {"coolant_flow_reduction", "observed_zero_coolant_flow"}
                    and flow["evidence_id"] in item["evidence_ids"]
                ]
                if flow_findings:
                    associated = [
                        measurement
                        for measurement in (flow, temperature, pressure)
                        if measurement is not None
                    ]
                    anomalies.append(
                        {
                            "equipment_id": equipment_id,
                            "indicator": "coolant_process_co_observation",
                            "status": "CORRELATED_EVIDENCE_FOR_REVIEW",
                            "evidence_ids": [
                                item["evidence_id"] for item in associated
                            ],
                            "freshness_status": (
                                "STALE"
                                if any(
                                    item["freshness_status"] == "STALE"
                                    for item in associated
                                )
                                else "CURRENT"
                            ),
                            "data_classification": (
                                "SYNTHETIC"
                                if any(
                                    item["data_classification"] == "SYNTHETIC"
                                    for item in associated
                                )
                                else "MEASURED"
                            ),
                            "interpretation": (
                                "A coolant-flow indicator co-occurs with temperature "
                                "or pressure evidence; this association does not establish cause."
                            ),
                        }
                    )

            electrical_findings = [
                item
                for item in anomalies
                if item["equipment_id"] == equipment_id
                and item["indicator"]
                in {
                    "voltage_deviation_from_history",
                    "current_deviation_from_history",
                    "current_above_configured_rating",
                }
                and any(
                    measurement is not None
                    and measurement["evidence_id"] in item["evidence_ids"]
                    for measurement in (
                        row["measurements"].get("voltage_v"),
                        row["measurements"].get("current_a"),
                    )
                )
            ]
            electrical_context = [
                row["measurements"].get(metric)
                for metric in ("temperature_c", "vibration_mm_s")
            ]
            if electrical_findings and any(electrical_context):
                associated = [
                    item
                    for item in (
                        row["measurements"].get("voltage_v"),
                        row["measurements"].get("current_a"),
                        *electrical_context,
                    )
                    if item is not None
                ]
                anomalies.append(
                    {
                        "equipment_id": equipment_id,
                        "indicator": "electrical_equipment_co_observation",
                        "status": "CORRELATED_EVIDENCE_FOR_REVIEW",
                        "evidence_ids": [
                            item["evidence_id"] for item in associated
                        ],
                        "freshness_status": (
                            "STALE"
                            if any(
                                item["freshness_status"] == "STALE"
                                for item in associated
                            )
                            else "CURRENT"
                        ),
                        "data_classification": (
                            "SYNTHETIC"
                            if any(
                                item["data_classification"] == "SYNTHETIC"
                                for item in associated
                            )
                            else "MEASURED"
                        ),
                        "interpretation": (
                            "Electrical and temperature/vibration evidence co-occurs; "
                            "this association does not establish a motor or equipment fault."
                        ),
                    }
                )

        equipment_rows = [item for item in aligned if item["equipment_id"] == equipment_id]
        for left_metric, right_metric in _CORRELATION_PAIRS:
            pairs = []
            fresh_pairs = 0
            stale_pairs = 0
            synthetic_pairs = 0
            used_left: set[str] = set()
            used_right: set[str] = set()
            for row in equipment_rows:
                left = row["measurements"].get(left_metric)
                right = row["measurements"].get(right_metric)
                if (
                    left
                    and right
                    and left["evidence_id"] not in used_left
                    and right["evidence_id"] not in used_right
                ):
                    pairs.append((float(left["value"]), float(right["value"])))
                    if (
                        left["freshness_status"] == "CURRENT"
                        and right["freshness_status"] == "CURRENT"
                    ):
                        fresh_pairs += 1
                    else:
                        stale_pairs += 1
                    if (
                        left["data_classification"] == "SYNTHETIC"
                        or right["data_classification"] == "SYNTHETIC"
                    ):
                        synthetic_pairs += 1
                    used_left.add(left["evidence_id"])
                    used_right.add(right["evidence_id"])
            correlation = _pearson(pairs)
            if correlation is not None:
                correlations.append(
                    {
                        "equipment_id": equipment_id,
                        "left_metric": left_metric,
                        "right_metric": right_metric,
                        "pearson_r": correlation,
                        "matched_observations": len(pairs),
                        "fresh_observations": fresh_pairs,
                        "stale_observations": stale_pairs,
                        "synthetic_observations": synthetic_pairs,
                        "interpretation": (
                            "Descriptive association only; sampling is backward-aligned and "
                            "does not establish causality."
                        ),
                    }
                )

    # Do not derive kW from V and A: power factor and phase configuration are
    # not supplied by this schema. Only directly measured power is reported.
    public_records = [
        {key: value for key, value in item.items() if not key.startswith("_")}
        for item in normalized
    ]
    summaries = []
    for equipment_id in sorted({item["equipment_id"] for item in normalized}):
        findings = [item for item in anomalies if item["equipment_id"] == equipment_id]
        equipment_records = [
            item for item in normalized if item["equipment_id"] == equipment_id
        ]
        fresh_measured = [
            item
            for item in equipment_records
            if item["data_classification"] == "MEASURED"
            and item["freshness_status"] == "CURRENT"
            and item["trust_score"] >= 0.75
        ]
        fresh_synthetic = [
            item
            for item in equipment_records
            if item["data_classification"] == "SYNTHETIC"
            and item["freshness_status"] == "CURRENT"
        ]
        measured_counts = {
            metric: sum(item["metric"] == metric for item in fresh_measured)
            for metric in _HEALTH_EVIDENCE_METRICS
        }
        sufficient_for_review = all(
            measured_counts[metric] >= 3 for metric in _HEALTH_EVIDENCE_METRICS
        )
        if sufficient_for_review:
            health_evidence_status = "SUFFICIENT_FOR_REVIEW"
            health_assessment = (
                "ATTENTION" if findings else "NO_CONFIGURED_INDICATOR"
            )
        elif fresh_synthetic and not fresh_measured:
            health_evidence_status = "SYNTHETIC_ONLY"
            health_assessment = "UNKNOWN"
        elif equipment_records and not fresh_measured:
            health_evidence_status = (
                "STALE_DATA_ONLY"
                if all(item["freshness_status"] == "STALE" for item in equipment_records)
                else "INSUFFICIENT_VALIDATED_DATA"
            )
            health_assessment = "UNKNOWN"
        else:
            health_evidence_status = "INSUFFICIENT_VALIDATED_DATA"
            health_assessment = "UNKNOWN"
        summaries.append(
            {
                "equipment_id": equipment_id,
                "measurement_count": len(equipment_records),
                "indicator_count": len(findings),
                "sensor_evidence_status": "ATTENTION" if findings else "NO_INDICATOR",
                "health_evidence_status": health_evidence_status,
                "health_assessment": health_assessment,
                "fresh_measured_record_count": len(fresh_measured),
                "fresh_synthetic_record_count": len(fresh_synthetic),
                "stale_record_count": sum(
                    item["freshness_status"] == "STALE"
                    for item in equipment_records
                ),
                "fresh_measured_counts_by_metric": measured_counts,
                "note": (
                    "Health is not certified. Evidence is sufficient for review only "
                    "when at least three fresh, non-synthetic readings are available "
                    "for every configured sensor metric; provenance is not source authentication."
                ),
            }
        )
    freshness_summary = {
        status_name: sum(
            item["freshness_status"] == status_name for item in normalized
        )
        for status_name in ("CURRENT", "STALE", "FUTURE_TIMESTAMP")
    }
    data_classification_summary = {
        classification: sum(
            item["data_classification"] == classification for item in normalized
        )
        for classification in ("MEASURED", "SYNTHETIC")
    }
    return {
        "records": public_records,
        "rejected_records": rejected,
        "duplicate_records": duplicates,
        "conflicting_observations": conflicting,
        "missing_intervals": missing_intervals,
        "aligned_observations": aligned,
        "anomaly_indicators": anomalies,
        "cross_sensor_correlations": correlations,
        "equipment_overview": summaries,
        "data_quality_summary": {
            "freshness": freshness_summary,
            "classification": data_classification_summary,
            "stale_after_seconds": stale_after_seconds,
        },
        "configuration": {
            "sync_tolerance_seconds": tolerance,
            "flow_reduction_fraction": flow_reduction_fraction,
            "electrical_deviation_threshold_sigma": electrical_deviation_threshold_sigma,
            "stale_after_seconds": stale_after_seconds,
            "rated_current_a": ratings,
            "conflicting_equipment_ratings": conflicting_ratings,
        },
        "limitations": [
            "Asynchronous streams are aligned only to prior or same-event-time measurements for the same equipment.",
            "No values are interpolated, cross-equipment matched, or inferred from absent sensors.",
            "Flow reductions are compared with at least three prior readings and are indicators, not confirmed failures.",
            "Current overload is evaluated only against a caller-supplied equipment rating or an explicit sensor status.",
            (
                "Conflicting rated-current metadata was withheld for equipment: "
                + ", ".join(conflicting_ratings)
                if conflicting_ratings
                else "Equipment ratings embedded in current-measurement metadata are caller supplied and are not independently verified."
            ),
            "Electrical deviation thresholds require a historical baseline; voltage/current kW is not calculated without power-factor and phase data.",
            "Correlations are descriptive and are not causal evidence.",
            "Synthetic measurements are explicitly classified and are never counted as validated health evidence.",
            "Health evidence sufficiency is not a health certification or safety approval.",
        ],
    }
