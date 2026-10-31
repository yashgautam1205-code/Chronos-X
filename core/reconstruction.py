"""Explainable temporal reconstruction for industrial event evidence."""

from collections import defaultdict
from datetime import datetime
import json
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid5

from models import IndustrialEvent

SENSOR_FIELDS = {
    "temperature_c": ("temperature_sensor_id", "degC"),
    "pressure_kpa": ("pressure_sensor_id", "kPa"),
    "vibration_mm_s": ("vibration_sensor_id", "mm/s"),
    "cooling_pump_status": ("cooling_pump_sensor_id", "status"),
    "power_supply_status": ("power_supply_sensor_id", "status"),
    "smoke_detected": ("smoke_sensor_id", "boolean"),
}


def _parse_timestamp(value: datetime | str) -> datetime:
    timestamp = value if isinstance(value, datetime) else datetime.fromisoformat(value.replace("Z", "+00:00"))
    if timestamp.tzinfo is None:
        raise ValueError("Evidence timestamps must include timezone information")
    return timestamp


def _event_dict(event: IndustrialEvent | dict[str, Any]) -> dict[str, Any]:
    if isinstance(event, IndustrialEvent):
        return event.model_dump(mode="json")
    event_data = dict(event)
    if "received_at" not in event_data and event_data.get("timestamp") is not None:
        event_data["received_at"] = event_data["timestamp"]
    if "id" not in event_data:
        canonical_event = json.dumps(event_data, sort_keys=True, default=str)
        event_data["id"] = str(uuid5(NAMESPACE_URL, f"chronos-x:event:{canonical_event}"))
    return IndustrialEvent.model_validate(event_data).model_dump(mode="json")


def _normalize_sensor_rows(
    sensor_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row_index, row in enumerate(sensor_rows):
        timestamp = _parse_timestamp(row["timestamp"])
        received_at = _parse_timestamp(row.get("received_at") or row["timestamp"])
        equipment_id = str(row.get("equipment_id") or "unknown")
        row_id = str(row.get("reading_id") or row.get("id") or f"sensor-row-{row_index}")
        provenance = row.get("provenance", {})
        for metric, (sensor_field, unit) in SENSOR_FIELDS.items():
            if metric not in row:
                continue
            sensor_id = str(row.get(sensor_field) or f"{equipment_id}-{metric}")
            normalized.append(
                {
                    "evidence_id": f"{row_id}:{metric}",
                    "kind": "sensor_reading",
                    "timestamp": timestamp,
                    "received_at": received_at,
                    "equipment_id": equipment_id,
                    "source": sensor_id,
                    "sensor_id": sensor_id,
                    "metric": metric,
                    "value": row[metric],
                    "unit": unit,
                    "provenance": provenance,
                    "original_evidence": row,
                    "flags": [],
                }
            )

        if "metric" in row and "value" in row:
            sensor_id = str(row.get("sensor_id") or row.get("source") or "unknown")
            metric = str(row["metric"])
            normalized.append(
                {
                    "evidence_id": row_id,
                    "kind": "sensor_reading",
                    "timestamp": timestamp,
                    "received_at": received_at,
                    "equipment_id": equipment_id,
                    "source": sensor_id,
                    "sensor_id": sensor_id,
                    "metric": metric,
                    "value": row["value"],
                    "unit": str(row.get("unit", "")),
                    "provenance": provenance,
                    "original_evidence": row,
                    "flags": [],
                }
            )
    return normalized


def _provenance_reasons(evidence: dict[str, Any]) -> list[str]:
    provenance = evidence.get("provenance")
    if not provenance:
        return ["provenance_not_provided"]
    if isinstance(provenance, dict):
        reasons = []
        if provenance.get("source_verified") is False:
            reasons.append("source_not_verified")
        elif provenance.get("source_verified") is True:
            reasons.append("source_verification_unattested")
        else:
            reasons.append("source_verification_missing")
        if not provenance.get("origin"):
            reasons.append("provenance_origin_missing")
        return reasons
    return [] if provenance == "synthetic_generator" else ["unrecognized_provenance"]


def reconstruct_timeline(
    scenario_id: UUID,
    events: list[IndustrialEvent | dict[str, Any]],
    sensor_rows: list[dict[str, Any]],
    *,
    delay_threshold_seconds: float = 5.0,
    expected_interval_seconds: float = 1.0,
) -> dict[str, Any]:
    """Reconstruct evidence chronologically without dropping questionable records."""
    if delay_threshold_seconds < 0:
        raise ValueError("delay_threshold_seconds must not be negative")
    if expected_interval_seconds <= 0:
        raise ValueError("expected_interval_seconds must be greater than zero")

    evidence: list[dict[str, Any]] = []
    for event_value in events:
        event = _event_dict(event_value)
        metadata = event.get("metadata") or {}
        event_scenario = metadata.get("scenario_id")
        if event_scenario and str(event_scenario) != str(scenario_id):
            continue
        timestamp = _parse_timestamp(event["timestamp"])
        received_at = _parse_timestamp(event.get("received_at") or event["timestamp"])
        evidence.append(
            {
                "evidence_id": str(event["id"]),
                "kind": "event",
                "timestamp": timestamp,
                "received_at": received_at,
                "equipment_id": str(
                    metadata.get("equipment_id")
                    or metadata.get("asset_id")
                    or event["source"]
                ),
                "source": str(event["source"]),
                "event_type": str(event["event_type"]),
                "provenance": event.get("provenance") or {},
                "original_evidence": event,
                "flags": [],
            }
        )
    evidence.extend(_normalize_sensor_rows(sensor_rows))

    evidence_by_fingerprint: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for item in evidence:
        if item["kind"] == "event":
            fingerprint = (
                "event",
                item["equipment_id"],
                item["source"],
                item["event_type"],
                item["timestamp"],
                item["original_evidence"]["description"],
            )
        else:
            fingerprint = (
                "sensor",
                item["equipment_id"],
                item["sensor_id"],
                item["metric"],
                item["timestamp"],
                str(item["value"]),
            )
        evidence_by_fingerprint[fingerprint].append(item)

    duplicates: list[dict[str, Any]] = []
    for group in evidence_by_fingerprint.values():
        group.sort(key=lambda item: (item["received_at"], item["evidence_id"]))
        representative = group[0]
        for item in group[1:]:
            item["flags"].append("duplicate")
            duplicates.append(
                {
                    "evidence_id": item["evidence_id"],
                    "duplicate_of": representative["evidence_id"],
                    "kind": item["kind"],
                    "original_evidence": item["original_evidence"],
                }
            )

    received_order = sorted(
        evidence,
        key=lambda item: (
            item["received_at"],
            item["evidence_id"],
        ),
    )
    latest_by_stream: dict[tuple[str, str], datetime] = {}
    for item in received_order:
        stream = (item["kind"], item["equipment_id"])
        latest_timestamp = latest_by_stream.get(stream)
        if latest_timestamp is not None and item["timestamp"] < latest_timestamp:
            item["flags"].append("out_of_order")
        latest_by_stream[stream] = max(
            latest_timestamp or item["timestamp"],
            item["timestamp"],
        )
        delay = (item["received_at"] - item["timestamp"]).total_seconds()
        if delay > delay_threshold_seconds:
            item["flags"].append("delayed")
        elif delay < 0:
            item["flags"].append("received_before_event_time")
        item["flags"].extend(_provenance_reasons(item))

    conflicting_groups: dict[tuple[str, str, str, datetime], list[dict[str, Any]]] = defaultdict(list)
    for item in evidence:
        if item["kind"] == "sensor_reading":
            key = (
                item["equipment_id"],
                item["sensor_id"],
                item["metric"],
                item["timestamp"],
            )
            conflicting_groups[key].append(item)

    conflicts: list[dict[str, Any]] = []
    for (equipment_id, sensor_id, metric, timestamp), group in conflicting_groups.items():
        values = {str(item["value"]) for item in group}
        if len(values) <= 1:
            continue
        conflict_items = []
        for item in group:
            item["flags"].append("conflicting")
            conflict_items.append(
                {
                    "evidence_id": item["evidence_id"],
                    "value": item["value"],
                    "source": item["source"],
                    "original_evidence": item["original_evidence"],
                }
            )
        conflicts.append(
            {
                "equipment_id": equipment_id,
                "sensor_id": sensor_id,
                "metric": metric,
                "timestamp": timestamp,
                "observations": conflict_items,
            }
        )

    missing: list[dict[str, Any]] = []
    sensor_streams: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in evidence:
        if item["kind"] == "sensor_reading":
            sensor_streams[
                (item["equipment_id"], item["sensor_id"], item["metric"])
            ].append(item)
    for (equipment_id, sensor_id, metric), stream in sensor_streams.items():
        timestamps = sorted({item["timestamp"] for item in stream})
        for previous, current in zip(timestamps, timestamps[1:]):
            gap = (current - previous).total_seconds()
            if gap <= expected_interval_seconds * 1.5:
                continue
            missing_count = max(
                1,
                round(gap / expected_interval_seconds) - 1,
            )
            missing.append(
                {
                    "equipment_id": equipment_id,
                    "sensor_id": sensor_id,
                    "metric": metric,
                    "missing_count": missing_count,
                    "expected_interval_seconds": expected_interval_seconds,
                    "after_timestamp": previous,
                    "before_timestamp": current,
                }
            )

    suspicious_by_source: dict[str, dict[str, Any]] = {}
    for item in evidence:
        reasons = [
            reason
            for reason in item["flags"]
            if reason in {
                "provenance_not_provided",
                "source_not_verified",
                "provenance_origin_missing",
                "unrecognized_provenance",
                "source_verification_unattested",
                "source_verification_missing",
            }
        ]
        event_metadata = item["original_evidence"].get("metadata") or {}
        if item["kind"] == "event" and event_metadata.get("equipment_id") and (
            str(event_metadata["equipment_id"]) != item["source"]
        ):
            reasons.append("source_equipment_mismatch")
        if not item["source"].strip() or item["source"].lower() in {"unknown", "none"}:
            reasons.append("source_identifier_missing")
        if reasons:
            source_report = suspicious_by_source.setdefault(
                item["source"],
                {"source": item["source"], "evidence_ids": [], "reasons": []},
            )
            source_report["evidence_ids"].append(item["evidence_id"])
            source_report["reasons"] = sorted(set(source_report["reasons"] + reasons))
            item["flags"].extend(reason for reason in reasons if reason not in item["flags"])

    reliability: list[dict[str, Any]] = []
    for item in evidence:
        reasons = sorted(set(item["flags"]))
        score = 1.0
        deductions = {
            "delayed": 0.2,
            "duplicate": 0.25,
            "conflicting": 0.35,
            "out_of_order": 0.15,
            "received_before_event_time": 0.2,
            "provenance_not_provided": 0.15,
            "source_not_verified": 0.35,
            "source_verification_unattested": 0.35,
            "source_verification_missing": 0.2,
            "provenance_origin_missing": 0.15,
            "unrecognized_provenance": 0.25,
            "source_equipment_mismatch": 0.35,
            "source_identifier_missing": 0.35,
        }
        for reason in reasons:
            score -= deductions.get(reason, 0)
        score = round(max(0.0, score), 2)
        level = "high" if score >= 0.8 else "medium" if score >= 0.5 else "low"
        reliability.append(
            {
                "evidence_id": item["evidence_id"],
                "kind": item["kind"],
                "score": score,
                "level": level,
                "reasons": reasons,
            }
        )
        item["reliability_score"] = score

    timeline = sorted(
        evidence,
        key=lambda item: (
            item["timestamp"],
            item["received_at"],
            item["evidence_id"],
        ),
    )
    timeline_output = [
        {
            **item,
            "timestamp": item["timestamp"],
            "received_at": item["received_at"],
        }
        for item in timeline
    ]
    duplicate_ids = {
        str(item["evidence_id"]): str(item["duplicate_of"])
        for item in duplicates
    }
    suppressed_by_representative: dict[str, list[str]] = defaultdict(list)
    for duplicate_id, representative_id in duplicate_ids.items():
        suppressed_by_representative[representative_id].append(duplicate_id)
    replay_timeline = []
    for item in timeline_output:
        evidence_id = str(item["evidence_id"])
        if evidence_id in duplicate_ids:
            continue
        replay_item = dict(item)
        replay_item["suppressed_duplicate_ids"] = sorted(
            suppressed_by_representative.get(evidence_id, [])
        )
        replay_timeline.append(replay_item)
    alternative_histories = [
        {
            "history_id": (
                f"conflict:{conflict['equipment_id']}:{conflict['sensor_id']}:"
                f"{conflict['metric']}:{conflict['timestamp'].isoformat()}"
            ),
            "status": "UNRESOLVED_CONFLICT",
            "selection": None,
            "reason": (
                "Conflicting observations have equal event time and source identity; "
                "the available evidence does not justify selecting one value."
            ),
            "alternatives": [
                {
                    "evidence_id": observation["evidence_id"],
                    "value": observation["value"],
                    "source": observation["source"],
                    "original_evidence": observation["original_evidence"],
                }
                for observation in conflict["observations"]
            ],
        }
        for conflict in conflicts
    ]
    delayed_events = [
        item
        for item in timeline_output
        if item["kind"] == "event" and "delayed" in item["flags"]
    ]
    out_of_order = [
        item
        for item in timeline_output
        if "out_of_order" in item["flags"]
    ]
    findings = [
        f"Reconstructed {len(timeline_output)} evidence records in event-time order.",
        f"Identified {len(delayed_events)} delayed events, {len(duplicates)} duplicate records, "
        f"{len(missing)} missing sensor intervals, and {len(conflicts)} conflicting sensor groups.",
        f"Flagged {len(suspicious_by_source)} suspicious event sources; all original evidence is retained in the timeline.",
        "Reliability scores are deterministic rule-based indicators, not model predictions.",
    ]
    return {
        "scenario_id": scenario_id,
        "reconstructed_timeline": timeline_output,
        "replay_timeline": replay_timeline,
        "alternative_histories": alternative_histories,
        "delayed_events": delayed_events,
        "duplicate_events": duplicates,
        "missing_events": missing,
        "out_of_order_events": out_of_order,
        "conflicting_evidence": conflicts,
        "suspicious_sources": list(suspicious_by_source.values()),
        "evidence_reliability": reliability,
        "summary": "Temporal reconstruction completed with questionable evidence preserved and flagged.",
        "findings": findings,
    }
