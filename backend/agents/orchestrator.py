"""Sequential orchestration of the five evidence-grounded CHRONOS-X agents."""

import logging
from pathlib import Path
from typing import Any, Callable
from uuid import UUID

from backend.agents.investigator import run_investigator_agent
from backend.agents.optimizer import run_optimizer_agent
from backend.agents.predictor import run_predictor_agent
from backend.agents.red_team import run_red_team_agent
from backend.agents.safety import run_safety_agent
from core.counterfactual_simulation import simulate_interventions
from models import AgentOrchestrationReport, AgentResult

logger = logging.getLogger(__name__)
DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[1] / "ml" / "saved_models"
EXPECTED_AGENT_NAMES = {"investigator", "predictor", "optimizer", "safety", "red_team"}


def _validate_agent_consistency(
    agent_results: list[AgentResult],
    reconstruction: dict[str, Any],
    simulation_report: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    timeline = reconstruction.get("reconstructed_timeline", [])
    known_evidence_ids = {
        str(item["evidence_id"])
        for item in timeline
        if item.get("evidence_id") is not None
    }
    known_evidence_ids.update(
        str(original["reading_id"])
        for item in timeline
        if isinstance((original := item.get("original_evidence")), dict)
        and original.get("reading_id") is not None
    )
    known_evidence_ids.update(
        str(original["id"])
        for item in timeline
        if isinstance((original := item.get("original_evidence")), dict)
        and original.get("id") is not None
    )
    integrity_checks: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    names = [result.agent_name for result in agent_results]
    if set(names) != EXPECTED_AGENT_NAMES or len(names) != len(EXPECTED_AGENT_NAMES):
        conflicts.append(
            {
                "type": "agent_roster_conflict",
                "expected_agents": sorted(EXPECTED_AGENT_NAMES),
                "observed_agents": names,
            }
        )

    simulation_results = {
        str(item["intervention_name"]): item
        for item in [
            *simulation_report.get("interventions", []),
            *simulation_report.get("sequential_action_plans", []),
        ]
        if item.get("intervention_name")
    }
    for result in agent_results:
        invalid_evidence = sorted(
            {
                str(evidence_id)
                for evidence_id in result.evidence_ids
                if str(evidence_id) not in known_evidence_ids
            }
        )
        invalid_provenance = sorted(
            {
                str(item["evidence_id"])
                for item in result.provenance
                if item.get("evidence_id")
                and str(item["evidence_id"]) not in known_evidence_ids
            }
        )
        if invalid_evidence or invalid_provenance:
            conflict = {
                "type": "unresolved_evidence_reference",
                "agent_name": result.agent_name,
                "unknown_evidence_ids": sorted(set(invalid_evidence + invalid_provenance)),
            }
            conflicts.append(conflict)
            result.status = "UNKNOWN"
            result.errors.append(
                "Agent evidence references could not be matched to the reconstructed timeline."
            )
        integrity_checks.append(
            {
                "check": f"{result.agent_name}_evidence_references",
                "result": "FAIL" if invalid_evidence or invalid_provenance else "PASS",
                "unknown_evidence_ids": sorted(set(invalid_evidence + invalid_provenance)),
            }
        )

    safety = next(
        (result for result in agent_results if result.agent_name == "safety"),
        None,
    )
    if safety is None:
        integrity_checks.append(
            {"check": "safety_simulation_consistency", "result": "UNKNOWN"}
        )
    else:
        claimed = {
            str(item.get("intervention_name")): str(item.get("result"))
            for item in safety.findings
            if item.get("intervention_name")
        }
        safety_conflicts = []
        for name, simulation in simulation_results.items():
            outcomes = simulation.get("simulated_incident_outcomes", {})
            rollouts = simulation.get("number_of_rollouts")
            if (
                not isinstance(outcomes, dict)
                or not isinstance(rollouts, int)
                or rollouts < 100
            ):
                expected = "UNKNOWN"
            elif outcomes.get("incident_count", 0) > 0 or outcomes.get(
                "equipment_failure_count", 0
            ) > 0:
                expected = "FAIL"
            else:
                expected = "PASS"
            if claimed.get(name) != expected:
                safety_conflicts.append(
                    {
                        "intervention_name": name,
                        "simulation_result": expected,
                        "agent_result": claimed.get(name, "MISSING"),
                    }
                )
        if safety_conflicts:
            conflicts.append(
                {
                    "type": "safety_agent_simulation_disagreement",
                    "agent_name": "safety",
                    "details": safety_conflicts,
                }
            )
            safety.status = "UNKNOWN"
            safety.errors.append(
                "Safety claims disagreed with the supplied simulation outcomes; human review is required."
            )
        integrity_checks.append(
            {
                "check": "safety_simulation_consistency",
                "result": "FAIL" if safety_conflicts else "PASS",
                "conflicts": safety_conflicts,
            }
        )
    return integrity_checks, conflicts


def _run_agent(
    agent_name: str,
    operation: Callable[[], AgentResult],
) -> AgentResult:
    try:
        return operation()
    except Exception as error:
        logger.exception("Agent %s failed", agent_name)
        return AgentResult(
            agent_name=agent_name,
            status="FAILED",
            summary=f"{agent_name} agent failed; see server logs for details.",
            errors=[f"{type(error).__name__}: {error}"],
        )


def orchestrate_agents(
    scenario_id: UUID,
    reconstruction: dict[str, Any],
    *,
    simulation_report: dict[str, Any] | None = None,
    simulation_parameters: dict[str, Any] | None = None,
    investigation_report: dict[str, Any] | None = None,
    model_dir: Path = DEFAULT_MODEL_DIR,
    risk_aversion_weight: float = 1.0,
) -> AgentOrchestrationReport:
    """Run all five agents sequentially, passing actual upstream evidence outputs."""
    if str(reconstruction.get("scenario_id")) != str(scenario_id):
        raise ValueError("Reconstruction scenario_id does not match orchestrator scenario_id")

    if simulation_report is None:
        simulation = simulate_interventions(
            scenario_id,
            **(simulation_parameters or {}),
        )
        simulation_report = simulation

    if investigation_report is None:
        investigation = _run_agent(
            "investigator",
            lambda: run_investigator_agent(scenario_id, reconstruction),
        )
    else:
        referenced_evidence = sorted(
            {
                str(evidence["evidence_id"])
                for hypothesis in investigation_report.get("ranked_hypotheses", [])
                for category in ("supporting_evidence", "contradicting_evidence")
                for rule in hypothesis.get(category, [])
                for evidence in rule.get("evidence", [])
                if evidence.get("evidence_id")
            }
        )
        investigation = AgentResult(
            agent_name="investigator",
            status=(
                "COMPLETED"
                if investigation_report.get("ranked_hypotheses")
                else "UNKNOWN"
            ),
            summary=str(investigation_report.get("summary", "")),
            findings=[
                {
                    "hypotheses_evaluated": len(
                        investigation_report.get("ranked_hypotheses", [])
                    ),
                    "causal_probability_asserted": False,
                }
            ],
            evidence_ids=referenced_evidence,
            outputs=investigation_report,
        )
    predictor = _run_agent(
        "predictor",
        lambda: run_predictor_agent(reconstruction, model_dir=model_dir),
    )
    sensor_analysis = predictor.outputs.get("additional_sensor_analysis", {})
    sensor_evidence_context = {
        "measurement_count": len(sensor_analysis.get("records", [])),
        "anomaly_indicators": sensor_analysis.get("anomaly_indicators", []),
        "cross_sensor_correlations": sensor_analysis.get(
            "cross_sensor_correlations", []
        ),
        "limitations": sensor_analysis.get("limitations", []),
    }
    optimizer = _run_agent(
        "optimizer",
        lambda: run_optimizer_agent(
            simulation_report,
            risk_aversion_weight=risk_aversion_weight,
            sensor_evidence_context=sensor_evidence_context,
        ),
    )
    safety = _run_agent(
        "safety",
        lambda: run_safety_agent(
            simulation_report,
            sensor_evidence_context=sensor_evidence_context,
        ),
    )
    red_team = _run_agent(
        "red_team",
        lambda: run_red_team_agent(
            reconstruction,
            predictor.outputs,
        ),
    )
    agent_results = [investigation, predictor, optimizer, safety, red_team]
    integrity_checks, conflicts = _validate_agent_consistency(
        agent_results,
        reconstruction,
        simulation_report,
    )
    statuses = {result.status for result in agent_results}
    overall_status = (
        "FAILED"
        if statuses == {"FAILED"}
        else "PARTIAL"
        if statuses.intersection({"FAILED", "UNKNOWN"}) or conflicts
        else "COMPLETED"
    )
    evidence_ids = {
        str(item["evidence_id"])
        for item in reconstruction.get("reconstructed_timeline", [])
        if item.get("evidence_id") is not None
    }
    return AgentOrchestrationReport(
        scenario_id=scenario_id,
        status=overall_status,
        agent_results=agent_results,
        shared_evidence_count=len(evidence_ids),
        summary=(
            f"Executed five specialized agents with {len(evidence_ids)} distinct reconstructed "
            f"evidence records; recorded {len(conflicts)} cross-agent integrity conflicts; "
            f"overall status: {overall_status}."
        ),
        integrity_checks=integrity_checks,
        conflicts=conflicts,
    )
