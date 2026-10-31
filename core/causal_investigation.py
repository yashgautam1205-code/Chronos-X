"""Explainable causal hypothesis ranking from reconstructed event evidence."""

import math
from typing import Any
from uuid import UUID

import networkx as nx
import pandas as pd

from core.sensor_fusion import analyze_sensor_readings, sensor_records_from_reconstruction

HYPOTHESES: dict[str, dict[str, Any]] = {
    "cooling_pump_failure": {
        "title": "Cooling pump failure",
        "prior": 0.14,
        "expected": {
            "pump_off": "Cooling pump reports an off or failed status.",
            "high_temperature": "Temperature rises above the configured 82 degC indicator.",
            "power_off": "Power supply remains available to distinguish a local pump fault.",
        },
        "likelihoods": {
            "pump_off": 0.90,
            "power_off": 0.10,
            "high_temperature": 0.80,
            "high_pressure": 0.55,
            "high_vibration": 0.10,
            "smoke_detected": 0.20,
            "pressure_flatline": 0.05,
            "conflicting_sensor_evidence": 0.15,
            "reported_cooling_pump_failure": 0.75,
        },
    },
    "power_interruption": {
        "title": "Power interruption",
        "prior": 0.14,
        "expected": {
            "power_off": "Power supply status drops.",
            "pump_off": "Power-dependent equipment, such as the cooling pump, stops.",
        },
        "likelihoods": {
            "pump_off": 0.70,
            "power_off": 0.95,
            "high_temperature": 0.50,
            "high_pressure": 0.25,
            "high_vibration": 0.15,
            "smoke_detected": 0.10,
            "pressure_flatline": 0.05,
            "conflicting_sensor_evidence": 0.15,
            "reported_power_interruption": 0.75,
        },
    },
    "pressure_system_failure": {
        "title": "Pressure-system failure",
        "prior": 0.14,
        "expected": {
            "high_pressure": "Pressure rises above the configured 120 kPa indicator.",
        },
        "likelihoods": {
            "pump_off": 0.15,
            "power_off": 0.10,
            "high_temperature": 0.30,
            "high_pressure": 0.95,
            "high_vibration": 0.25,
            "smoke_detected": 0.20,
            "pressure_flatline": 0.05,
            "conflicting_sensor_evidence": 0.20,
            "reported_sudden_pressure_increase": 0.75,
        },
    },
    "mechanical_equipment_malfunction": {
        "title": "Mechanical equipment malfunction",
        "prior": 0.14,
        "expected": {
            "high_vibration": "Vibration rises above the configured 6 mm/s indicator.",
        },
        "likelihoods": {
            "pump_off": 0.20,
            "power_off": 0.10,
            "high_temperature": 0.30,
            "high_pressure": 0.30,
            "high_vibration": 0.90,
            "smoke_detected": 0.15,
            "pressure_flatline": 0.05,
            "conflicting_sensor_evidence": 0.20,
            "reported_equipment_malfunction": 0.75,
        },
    },
    "sensor_malfunction": {
        "title": "Sensor malfunction",
        "prior": 0.14,
        "expected": {
            "pressure_flatline": "A sensor repeats an identical pressure value over at least five consecutive samples.",
            "conflicting_sensor_evidence": "The same sensor reports conflicting values at the same event time.",
        },
        "likelihoods": {
            "pump_off": 0.10,
            "power_off": 0.10,
            "high_temperature": 0.10,
            "high_pressure": 0.15,
            "high_vibration": 0.10,
            "smoke_detected": 0.10,
            "pressure_flatline": 0.90,
            "conflicting_sensor_evidence": 0.85,
            "reported_sensor_malfunction": 0.75,
        },
    },
    "thermal_runaway": {
        "title": "Thermal runaway or process overheating",
        "prior": 0.14,
        "expected": {
            "high_temperature": "Temperature rises above the configured 82 degC indicator.",
            "smoke_detected": "Smoke detection is triggered.",
        },
        "likelihoods": {
            "pump_off": 0.25,
            "power_off": 0.10,
            "high_temperature": 0.90,
            "high_pressure": 0.60,
            "high_vibration": 0.20,
            "smoke_detected": 0.75,
            "pressure_flatline": 0.05,
            "conflicting_sensor_evidence": 0.10,
            "reported_multiple_simultaneous_faults": 0.55,
        },
    },
    "other_or_unresolved": {
        "title": "Other or unresolved cause",
        "prior": 0.16,
        "expected": {},
        "likelihoods": {
            "pump_off": 0.5,
            "power_off": 0.5,
            "high_temperature": 0.5,
            "high_pressure": 0.5,
            "high_vibration": 0.5,
            "smoke_detected": 0.5,
            "pressure_flatline": 0.5,
            "conflicting_sensor_evidence": 0.5,
            "reported_cooling_pump_failure": 0.5,
            "reported_power_interruption": 0.5,
            "reported_sudden_pressure_increase": 0.5,
            "reported_equipment_malfunction": 0.5,
            "reported_sensor_malfunction": 0.5,
            "reported_multiple_simultaneous_faults": 0.5,
        },
    },
}

SENSOR_FEATURES = {
    "cooling_pump_status": ("pump_off", lambda values: values < 0.5, "Cooling pump reported stopped."),
    "power_supply_status": ("power_off", lambda values: values < 0.5, "Power supply reported unavailable."),
    "temperature_c": ("high_temperature", lambda values: values >= 82.0, "Temperature exceeded 82 degC."),
    "pressure_kpa": ("high_pressure", lambda values: values >= 120.0, "Pressure reached 120 kPa or higher."),
    "vibration_mm_s": ("high_vibration", lambda values: values >= 6.0, "Vibration reached 6 mm/s or higher."),
    "smoke_detected": ("smoke_detected", lambda values: values >= 0.5, "Smoke detector was active."),
}

ASSUMPTIONS = [
    "Prior probabilities are configured synthetic-scenario assumptions: 0.14 for each of six named causes and 0.16 for other/unresolved; they sum to one.",
    "Configured sensor indicators are screening thresholds (82 degC, 120 kPa, and 6 mm/s), not certified operating limits.",
    "Likelihoods are expert-configured illustrative conditional probabilities, not learned from validated plant failure data.",
    "Evidence signals are treated as conditionally independent for this naive Bayesian ranking; correlated sensors can make scores overconfident.",
    "Evidence contributions are discounted toward neutral when reconstruction assigns low reliability; duplicates are not counted as independent samples.",
    "Posterior probabilities are conditional on this listed hypothesis set and do not establish that any cause is true.",
    "A causal graph describes hypothesized mechanisms; temporal or statistical association alone does not establish causality.",
]


def build_causal_graph() -> nx.DiGraph:
    """Return the domain graph of candidate causes and hypothesized mechanisms."""
    graph = nx.DiGraph()
    relationships = (
        ("power_interruption", "cooling_pump_failure", "Power loss can stop an electrically powered pump."),
        ("cooling_pump_failure", "reduced_cooling_capacity", "Pump failure reduces coolant circulation."),
        ("reduced_cooling_capacity", "temperature_increase", "Reduced heat removal can raise process temperature."),
        ("temperature_increase", "pressure_increase", "Heating can increase pressure in a constrained process."),
        ("pressure_increase", "thermal_runaway", "Elevated pressure may accompany or worsen overheating."),
        ("pressure_system_failure", "pressure_increase", "A pressure-system fault can directly raise measured pressure."),
        ("mechanical_equipment_malfunction", "high_vibration", "Mechanical imbalance or damage can increase vibration."),
        ("sensor_malfunction", "unreliable_sensor_evidence", "A faulty sensor can produce flatlined or conflicting measurements."),
        ("thermal_runaway", "smoke_detection", "Severe overheating may produce smoke or trigger smoke detection."),
    )
    for source, target, mechanism in relationships:
        graph.add_edge(source, target, mechanism=mechanism)
    return graph


def _reliability_index(report: dict[str, Any]) -> dict[str, float]:
    return {
        str(item["evidence_id"]): float(item["score"])
        for item in report.get("evidence_reliability", [])
    }


def _evidence_reference(item: dict[str, Any], reliability: dict[str, float]) -> dict[str, Any]:
    return {
        "evidence_id": str(item["evidence_id"]),
        "kind": item["kind"],
        "timestamp": item.get("timestamp"),
        "description": item.get("description")
        or item.get("event_type")
        or f"{item.get('metric')} = {item.get('value')}",
        "metric": item.get("metric"),
        "value": item.get("value"),
        "source": item.get("source"),
        "reliability_score": reliability.get(str(item["evidence_id"]), 0.5),
        "reconstruction_flags": list(item.get("flags", [])),
    }


def _make_signal(
    name: str,
    positive: bool,
    description: str,
    evidence: list[dict[str, Any]],
    reliability: dict[str, float],
    *,
    available: bool = True,
) -> dict[str, Any]:
    return {
        "name": name,
        "positive": positive,
        "available": available,
        "description": description,
        "evidence": [
            _evidence_reference(item, reliability)
            for item in evidence[:10]
        ],
        "reliability": (
            sum(reliability.get(str(item["evidence_id"]), 0.5) for item in evidence)
            / len(evidence)
            if evidence
            else 0.5
        ),
    }


def _extract_signals(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    timeline = report.get("reconstructed_timeline", [])
    reliability = _reliability_index(report)
    sensor_items = [
        item
        for item in timeline
        if item.get("kind") == "sensor_reading"
        and item.get("metric") in SENSOR_FEATURES
    ]
    frame = pd.DataFrame(
        [
            {
                "evidence_id": str(item["evidence_id"]),
                "timestamp": item["timestamp"],
                "sensor_id": str(item.get("sensor_id") or item.get("source") or ""),
                "metric": str(item.get("metric") or ""),
                "value": pd.to_numeric(item.get("value"), errors="coerce"),
                "item": item,
            }
            for item in sensor_items
        ]
    )
    if not frame.empty:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    signals: dict[str, dict[str, Any]] = {}

    for metric, (feature_name, predicate, description) in SENSOR_FEATURES.items():
        metric_frame = (
            frame[frame["metric"] == metric].copy()
            if not frame.empty
            else pd.DataFrame()
        )
        if metric_frame.empty:
            signals[feature_name] = _make_signal(
                feature_name, False, description, [], reliability, available=False
            )
            continue
        metric_frame = metric_frame.dropna(subset=["value"])
        if metric_frame.empty:
            signals[feature_name] = _make_signal(
                feature_name, False, description, [], reliability, available=False
            )
            continue
        metric_frame["is_positive"] = predicate(metric_frame["value"])
        metric_frame["weight"] = metric_frame["evidence_id"].map(reliability).fillna(0.5)
        weighted_rate = float(
            (metric_frame["is_positive"] * metric_frame["weight"]).sum()
            / metric_frame["weight"].sum()
        )
        positive = weighted_rate >= 0.5
        selected = metric_frame[
            metric_frame["is_positive"] if positive else ~metric_frame["is_positive"]
        ]
        if selected.empty:
            selected = metric_frame
        signals[feature_name] = _make_signal(
            feature_name,
            positive,
            description,
            [row["item"] for _, row in selected.iterrows()],
            reliability,
        )

    pressure_frame = (
        frame[frame["metric"] == "pressure_kpa"].copy()
        if not frame.empty
        else pd.DataFrame()
    )
    flatline_rows: list[dict[str, Any]] = []
    if not pressure_frame.empty:
        for _, sensor_group in pressure_frame.groupby("sensor_id"):
            sensor_group = sensor_group.sort_values("timestamp")
            run_ids = sensor_group["value"].ne(sensor_group["value"].shift()).cumsum()
            run_lengths = sensor_group.groupby(run_ids)["value"].transform("size")
            flatline_rows.extend(
                row["item"]
                for (_, row), run_length in zip(sensor_group.iterrows(), run_lengths)
                if run_length >= 5
            )
    flatline_rows = list({item["evidence_id"]: item for item in flatline_rows}.values())
    signals["pressure_flatline"] = _make_signal(
        "pressure_flatline",
        bool(flatline_rows),
        "Pressure sensor repeated a value for at least five consecutive readings.",
        flatline_rows or (
            [item for item in sensor_items if item.get("metric") == "pressure_kpa"][:3]
        ),
        reliability,
        available=not pressure_frame.empty,
    )

    conflicts = [
        conflict
        for conflict in report.get("conflicting_evidence", [])
        if conflict.get("metric") in SENSOR_FEATURES
    ]
    conflict_refs = [
        {
            "evidence_id": observation["evidence_id"],
            "kind": "sensor_reading",
            "timestamp": conflict.get("timestamp"),
            "metric": conflict.get("metric"),
            "value": observation.get("value"),
            "source": observation.get("source"),
            "flags": ["conflicting"],
        }
        for conflict in conflicts
        for observation in conflict.get("observations", [])
    ]
    signals["conflicting_sensor_evidence"] = _make_signal(
        "conflicting_sensor_evidence",
        bool(conflicts),
        "Reconstruction found distinct values from one sensor at the same event time.",
        conflict_refs,
        reliability,
        available=bool(sensor_items),
    )

    event_items = [item for item in timeline if item.get("kind") == "event"]
    event_names = {
        "cooling_pump_failure",
        "power_interruption",
        "sudden_pressure_increase",
        "equipment_malfunction",
        "sensor_malfunction",
        "multiple_simultaneous_faults",
    }
    for event_name in event_names:
        matches = [
            item
            for item in event_items
            if event_name in str(item.get("event_type", "")).lower()
        ]
        signal_name = f"reported_{event_name}"
        signals[signal_name] = _make_signal(
            signal_name,
            bool(matches),
            f"An event record reported {event_name.replace('_', ' ')}; this is a report, not independent causal confirmation.",
            matches or event_items[:3],
            reliability,
            available=bool(event_items),
        )

    return signals


def _sensor_screening_hypotheses(
    sensor_analysis: dict[str, Any],
    signals: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Expose low-confidence sensor hypotheses without changing Bayesian scores."""
    indicators = sensor_analysis["anomaly_indicators"]
    records = sensor_analysis["records"]
    correlations = sensor_analysis["cross_sensor_correlations"]

    def evidence_for(indicator_names: set[str]) -> list[dict[str, Any]]:
        return [
            {
                "evidence_id": item.get("evidence_id"),
                "equipment_id": item.get("equipment_id"),
                "timestamp": item.get("timestamp"),
                "indicator": item.get("indicator"),
                "status": item.get("status"),
                "interpretation": item.get("interpretation"),
            }
            for item in indicators
            if item.get("indicator") in indicator_names
        ]

    def status_evidence(metric: str) -> list[dict[str, Any]]:
        return [
            {
                "evidence_id": item.get("evidence_id"),
                "equipment_id": item.get("equipment_id"),
                "timestamp": item.get("timestamp"),
                "metric": metric,
                "value": item.get("value"),
                "interpretation": (
                    "Sensor-reported status only; it is not an independent "
                    "verification of equipment condition."
                ),
            }
            for item in records
            if item.get("metric") == metric and item.get("value") == 0.0
        ]

    flow_support = evidence_for(
        {
            "coolant_flow_reduction",
            "observed_zero_coolant_flow",
            "reported_flow_failure",
        }
    )
    electrical_support = evidence_for(
        {
            "voltage_deviation_from_history",
            "current_deviation_from_history",
            "current_above_configured_rating",
            "reported_current_overload",
            "reported_electrical_fault",
        }
    )
    motor_support = list(electrical_support)
    vibration = signals.get("high_vibration", {})
    if electrical_support and vibration.get("positive"):
        motor_support.extend(
            {
                "evidence_id": item.get("evidence_id"),
                "equipment_id": item.get("equipment_id"),
                "timestamp": item.get("timestamp"),
                "indicator": "high_vibration",
                "interpretation": (
                    "Vibration indicator co-occurs with electrical evidence; "
                    "this association does not establish a motor fault."
                ),
            }
            for item in vibration.get("evidence", [])
        )

    observed_metrics = {
        item.get("metric")
        for item in records
        if isinstance(item.get("metric"), str)
    }

    def make_hypothesis(
        hypothesis_id: str,
        title: str,
        supporting: list[dict[str, Any]],
        contradicting: list[dict[str, Any]],
        modalities: tuple[str, ...],
        contextual_associations: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        missing = [metric for metric in modalities if metric not in observed_metrics]
        return {
            "hypothesis_id": hypothesis_id,
            "title": title,
            "status": (
                "POSSIBLE_UNCONFIRMED"
                if supporting
                else "INSUFFICIENT_EVIDENCE"
                if missing
                else "NOT_INDICATED_BY_AVAILABLE_THRESHOLDS"
            ),
            "supporting_evidence": supporting,
            "contradictory_evidence": contradicting,
            "missing_evidence": missing,
            "contextual_associations": contextual_associations or [],
            "interpretation": (
                "Screening hypothesis only. Sensor status and threshold indicators "
                "are not independently verified. Contextual associations are "
                "descriptive and do not establish cause."
            ),
        }

    cooling_associations = [
        item
        for item in correlations
        if "coolant_flow_lpm" in (item["left_metric"], item["right_metric"])
        and {item["left_metric"], item["right_metric"]}.intersection(
            {"temperature_c", "pressure_kpa"}
        )
    ]
    electrical_associations = [
        item
        for item in correlations
        if {item["left_metric"], item["right_metric"]}.intersection(
            {"voltage_v", "current_a"}
        )
        and {item["left_metric"], item["right_metric"]}.intersection(
            {"temperature_c", "vibration_mm_s"}
        )
    ]
    motor_associations = [
        item
        for item in correlations
        if {item["left_metric"], item["right_metric"]}
        == {"vibration_mm_s", "current_a"}
    ]

    return [
        make_hypothesis(
            "possible_cooling_system_malfunction",
            "Possible cooling-system malfunction",
            flow_support,
            status_evidence("flow_failure_status"),
            ("coolant_flow_lpm", "temperature_c"),
            cooling_associations,
        ),
        make_hypothesis(
            "possible_electrical_fault",
            "Possible electrical-system fault or abnormal load",
            electrical_support,
            status_evidence("current_overload")
            + status_evidence("electrical_fault_status"),
            ("voltage_v", "current_a"),
            electrical_associations,
        ),
        make_hypothesis(
            "possible_motor_or_equipment_fault",
            "Possible motor or equipment fault",
            motor_support,
            [],
            ("current_a", "vibration_mm_s"),
            motor_associations,
        ),
    ]


def _posterior(
    config: dict[str, Any],
    signals: dict[str, dict[str, Any]],
) -> tuple[float, list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    log_probability = math.log(config["prior"])
    supporting: list[dict[str, Any]] = []
    contradicting: list[dict[str, Any]] = []
    missing: list[str] = []
    for signal_name, probability_if_cause in config["likelihoods"].items():
        signal = signals.get(signal_name)
        if signal is None or not signal["available"]:
            if signal_name in config["expected"]:
                missing.append(config["expected"][signal_name])
            continue
        observed_probability = (
            probability_if_cause
            if signal["positive"]
            else 1.0 - probability_if_cause
        )
        adjusted_probability = (
            0.5 + (observed_probability - 0.5) * signal["reliability"]
        )
        log_probability += math.log(max(adjusted_probability, 1e-12))
        if signal_name in config["expected"]:
            evidence_entries = signal["evidence"]
            evidence_detail = {
                "signal": signal_name,
                "description": signal["description"],
                "evidence": evidence_entries,
            }
            (supporting if signal["positive"] else contradicting).append(evidence_detail)

    return log_probability, supporting, contradicting, missing


def investigate_causes(
    scenario_id: UUID,
    reconstruction: dict[str, Any],
) -> dict[str, Any]:
    """Rank competing causes using reliability-adjusted naive Bayesian evidence."""
    if str(reconstruction.get("scenario_id")) != str(scenario_id):
        raise ValueError("Reconstruction scenario_id does not match investigation scenario_id")

    signals = _extract_signals(reconstruction)
    graph = build_causal_graph()
    log_scores: dict[str, float] = {}
    hypothesis_details: dict[str, dict[str, Any]] = {}
    for hypothesis_id, config in HYPOTHESES.items():
        log_score, supporting, contradicting, missing = _posterior(
            config,
            signals,
        )
        log_scores[hypothesis_id] = log_score
        hypothesis_details[hypothesis_id] = {
            "hypothesis_id": hypothesis_id,
            "title": config["title"],
            "prior_probability": config["prior"],
            "supporting_evidence": supporting,
            "contradicting_evidence": contradicting,
            "missing_evidence": missing,
            "causal_relationships": [
                {
                    "cause": source,
                    "effect": target,
                    "mechanism": attributes["mechanism"],
                }
                for source, target, attributes in graph.edges(data=True)
                if source == hypothesis_id or target == hypothesis_id
            ],
        }

    maximum_log_score = max(log_scores.values())
    unnormalized = {
        key: math.exp(value - maximum_log_score)
        for key, value in log_scores.items()
    }
    denominator = sum(unnormalized.values())
    ranked = []
    for hypothesis_id, _ in sorted(
        unnormalized.items(),
        key=lambda item: (-item[1], item[0]),
    ):
        probability = unnormalized[hypothesis_id] / denominator
        detail = hypothesis_details[hypothesis_id]
        detail["posterior_probability"] = round(probability, 8)
        detail["confidence"] = round(probability, 8)
        detail["rank"] = len(ranked) + 1
        detail["interpretation"] = (
            "Conditional posterior within the configured hypothesis set; not proof of cause."
        )
        ranked.append(detail)

    causal_relationships = [
        {
            "cause": source,
            "effect": target,
            "mechanism": attributes["mechanism"],
        }
        for source, target, attributes in sorted(
            graph.edges(data=True),
            key=lambda edge: (edge[0], edge[1]),
        )
    ]
    sensor_analysis = analyze_sensor_readings(
        sensor_records_from_reconstruction(reconstruction)
    )
    screening_hypotheses = _sensor_screening_hypotheses(
        sensor_analysis,
        signals,
    )
    return {
        "scenario_id": scenario_id,
        "possible_root_causes": ranked,
        "ranked_hypotheses": ranked,
        "causal_relationships": causal_relationships,
        "evidence_summary": {
            "timeline_records": len(reconstruction.get("reconstructed_timeline", [])),
            "signals": {
                name: {
                    "observed": signal["positive"],
                    "available": signal["available"],
                    "description": signal["description"],
                    "evidence_count": len(signal["evidence"]),
                }
                for name, signal in signals.items()
            },
            "reconstruction_summary": reconstruction.get("summary", ""),
            "additional_sensor_evidence": {
                "anomaly_indicators": sensor_analysis["anomaly_indicators"],
                "conflicting_observations": sensor_analysis[
                    "conflicting_observations"
                ],
                "cross_sensor_correlations": sensor_analysis[
                    "cross_sensor_correlations"
                ],
                "screening_hypotheses": screening_hypotheses,
                "limitations": sensor_analysis["limitations"],
                "interpretation": (
                    "Additional sensor indicators and associations are supplied for "
                    "investigator review; they do not alter configured hypothesis "
                    "probabilities or establish causal relationships."
                ),
            },
        },
        "assumptions": ASSUMPTIONS,
        "summary": (
            f"Ranked {len(ranked)} candidate causes from "
            f"{len(reconstruction.get('reconstructed_timeline', []))} reconstructed evidence records. "
            "Scores are conditional, reliability-adjusted Bayesian estimates and do not prove causation."
        ),
    }
