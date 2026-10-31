"""Safety agent for explicit simulation-threshold gate checks."""

from typing import Any

from models import AgentResult


def run_safety_agent(
    simulation_report: dict[str, Any] | None,
    *,
    sensor_evidence_context: dict[str, Any] | None = None,
) -> AgentResult:
    if not simulation_report:
        return AgentResult(
            agent_name="safety",
            status="UNKNOWN",
            summary="Safety cannot be assessed without simulation outcomes and configured thresholds.",
            findings=[{"check": "simulation_evidence", "result": "UNKNOWN"}],
            outputs={
                "sensor_evidence_context": sensor_evidence_context or {},
                "sensor_evidence_used_as_safety_approval": False,
            },
        )

    interventions = [
        *simulation_report.get("interventions", []),
        *simulation_report.get("sequential_action_plans", []),
    ]
    if not interventions:
        return AgentResult(
            agent_name="safety",
            status="UNKNOWN",
            summary="Safety cannot be assessed because no intervention outcomes were provided.",
            findings=[{"check": "simulation_evidence", "result": "UNKNOWN"}],
            outputs={
                "overall_result": "UNKNOWN",
                "escalate": True,
                "sensor_evidence_context": sensor_evidence_context or {},
                "sensor_evidence_used_as_safety_approval": False,
                "limitations": [
                    "No candidate intervention has simulation evidence.",
                    "A PASS is not a facility safety approval or substitute for qualified review.",
                ],
            },
        )

    assessments = []
    for intervention in interventions:
        outcomes = intervention.get("simulated_incident_outcomes")
        if not outcomes or intervention.get("number_of_rollouts", 0) < 100:
            result = "UNKNOWN"
            reason = "Insufficient rollout evidence for the safety gate."
        elif (
            outcomes.get("incident_count", 0) > 0
            or outcomes.get("equipment_failure_count", 0) > 0
        ):
            result = "FAIL"
            reason = (
                "At least one simulated rollout crossed a configured incident threshold "
                "or reported equipment failure."
            )
        else:
            result = "PASS"
            reason = (
                "No configured incident threshold or equipment failure was reached "
                "in the provided rollouts; this is not a real-world safety certification."
            )
        assessments.append(
            {
                "intervention_name": intervention.get("intervention_name"),
                "result": result,
                "reason": reason,
                "rollouts": intervention.get("number_of_rollouts"),
            }
        )
    has_unknown = any(item["result"] == "UNKNOWN" for item in assessments)
    has_failure = any(item["result"] == "FAIL" for item in assessments)
    status = "UNKNOWN" if has_unknown else "COMPLETED"
    return AgentResult(
        agent_name="safety",
        status=status,
        summary=(
            "Safety checks were evaluated; FAIL and UNKNOWN require escalation."
        ),
        findings=assessments,
        outputs={
            "overall_result": "UNKNOWN" if has_unknown else "FAIL" if has_failure else "PASS",
            "escalate": has_unknown or has_failure,
            "sensor_evidence_context": sensor_evidence_context or {},
            "sensor_evidence_used_as_safety_approval": False,
            "limitations": [
                "Checks apply only to supplied simulated outcomes and model thresholds.",
                "A PASS is not a facility safety approval or substitute for qualified review.",
            ],
        },
    )
