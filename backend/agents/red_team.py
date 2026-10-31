"""Red-team agent for suspicious provenance and evidence-integrity indicators."""

from typing import Any

from models import AgentResult


def run_red_team_agent(
    reconstruction: dict[str, Any],
    predictor_outputs: dict[str, Any] | None = None,
) -> AgentResult:
    timeline = reconstruction.get("reconstructed_timeline", [])
    evidence_by_id = {
        str(item["evidence_id"]): item
        for item in timeline
        if item.get("evidence_id") is not None
    }
    findings: list[dict[str, Any]] = []
    evidence_ids: set[str] = set()

    evidence_lists = (
        ("delayed_evidence", reconstruction.get("delayed_events", [])),
        ("duplicate_evidence", reconstruction.get("duplicate_events", [])),
        ("out_of_order_evidence", reconstruction.get("out_of_order_events", [])),
    )
    for category, items in evidence_lists:
        for item in items:
            evidence_id = str(item.get("evidence_id", ""))
            evidence_ids.add(evidence_id)
            findings.append(
                {
                    "indicator": category,
                    "evidence_id": evidence_id,
                    "reference": item,
                    "interpretation": "Integrity or sequencing indicator; not proof of manipulation.",
                }
            )

    for conflict in reconstruction.get("conflicting_evidence", []):
        for observation in conflict.get("observations", []):
            evidence_id = str(observation.get("evidence_id", ""))
            evidence_ids.add(evidence_id)
            findings.append(
                {
                    "indicator": "conflicting_sensor_message",
                    "evidence_id": evidence_id,
                    "reference": observation,
                    "interpretation": "Conflicting values are preserved; source investigation is required.",
                }
            )

    for source in reconstruction.get("suspicious_sources", []):
        for evidence_id_value in source.get("evidence_ids", []):
            evidence_id = str(evidence_id_value)
            evidence_ids.add(evidence_id)
            findings.append(
                {
                    "indicator": "suspicious_source_provenance",
                    "source": source.get("source"),
                    "reasons": source.get("reasons", []),
                    "evidence_id": evidence_id,
                }
            )

    reliability_by_id = {
        str(item["evidence_id"]): float(item["score"])
        for item in reconstruction.get("evidence_reliability", [])
    }
    for evidence_id, score in reliability_by_id.items():
        if score < 0.5:
            evidence_ids.add(evidence_id)
            findings.append(
                {
                    "indicator": "low_reliability_evidence",
                    "evidence_id": evidence_id,
                    "reliability_score": score,
                    "reconstruction_flags": evidence_by_id.get(evidence_id, {}).get(
                        "flags", []
                    ),
                }
            )

    anomaly_assessments = (predictor_outputs or {}).get("anomaly_assessments", [])
    for assessment in anomaly_assessments:
        if assessment.get("anomaly"):
            evidence_id = str(assessment.get("evidence_id", ""))
            evidence_ids.add(evidence_id)
            findings.append(
                {
                    "indicator": "model_flagged_sensor_outlier",
                    "evidence_id": evidence_id,
                    "anomaly_score": assessment["anomaly_score"],
                    "sensor_evidence": assessment["evidence"],
                    "interpretation": (
                        "Statistical novelty only; does not establish a forged reading."
                    ),
                }
            )

    sensor_analysis = (predictor_outputs or {}).get("additional_sensor_analysis", {})
    for indicator in sensor_analysis.get("anomaly_indicators", []):
        for evidence_id_value in indicator.get("evidence_ids", []):
            evidence_id = str(evidence_id_value)
            evidence_ids.add(evidence_id)
            findings.append(
                {
                    "indicator": "additional_sensor_condition_indicator",
                    "condition": indicator.get("indicator"),
                    "status": indicator.get("status"),
                    "evidence_id": evidence_id,
                    "reference": indicator,
                    "interpretation": (
                        "A rule-based telemetry indicator warrants validation; it does not "
                        "prove a fault or deliberate manipulation."
                    ),
                }
            )

    suspicious_count = len(findings)
    return AgentResult(
        agent_name="red_team",
        status="COMPLETED",
        summary=(
            f"Reviewed reconstructed evidence and identified {suspicious_count} integrity indicators."
        ),
        findings=findings,
        evidence_ids=sorted(evidence_ids),
        provenance=[
            {
                "evidence_id": evidence_id,
                "source": evidence_by_id.get(evidence_id, {}).get("source"),
                "provenance": evidence_by_id.get(evidence_id, {}).get("provenance"),
            }
            for evidence_id in sorted(evidence_ids)
        ],
        outputs={
            "potential_manipulation_indicators": suspicious_count,
            "conclusion": (
                "Indicators warrant review; automated evidence flags do not prove deliberate tampering."
            ),
        },
    )
