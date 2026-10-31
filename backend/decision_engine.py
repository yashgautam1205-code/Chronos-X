"""Safety-gated OR-Tools intervention selection and explainable audit reports."""

from math import isfinite
from typing import Any
from uuid import UUID, uuid4

from ortools.sat.python import cp_model

from backend.safety_attestation import verify_safety_attestation
from core.counterfactual_simulation import INTERVENTIONS, SEQUENTIAL_PLAN_ACTIONS
from models import (
    AgentOrchestrationReport,
    DecisionReport,
    OperationalConstraints,
    RankedDecision,
    RejectedAlternative,
    SafetyCheckInput,
    SafetyVerification,
    SimulationReport,
)

OPTIMIZATION_CONSTRAINTS = [
    "Exactly one feasible intervention may be selected.",
    "Only the four supported interventions are candidates.",
    "A candidate is eligible only when every applicable safety verification is PASS.",
    "FAIL and UNKNOWN candidates are excluded; UNKNOWN is never treated as PASS.",
    "The objective first minimizes mean loss index + risk_aversion_weight * empirical 95% CVaR loss index, then incident count, then mean peak temperature and pressure.",
    "ML anomaly and forecast outputs are supporting evidence only and cannot override safety checks.",
    "Equipment availability, staffing, authorization, site checks, and agent completeness are enforced as explicit constraints.",
]

REQUIRED_AGENT_NAMES = {
    "investigator",
    "predictor",
    "optimizer",
    "safety",
    "red_team",
}


def _check(
    name: str,
    result: str,
    details: str,
    evidence_ids: list[str] | None = None,
    *,
    attestation_status: str = "NOT_APPLICABLE",
    reviewer_id: str | None = None,
) -> SafetyVerification:
    return SafetyVerification(
        check_name=name,
        result=result,
        details=details,
        evidence_ids=evidence_ids or [],
        attestation_status=attestation_status,
        reviewer_id=reviewer_id,
    )


def _simulation_safety_check(
    intervention: dict[str, Any] | None,
) -> SafetyVerification:
    if not intervention:
        return _check(
            "simulation_evidence",
            "UNKNOWN",
            "No simulation result exists for this intervention.",
        )
    outcomes = intervention.get("simulated_incident_outcomes")
    rollouts = intervention.get("number_of_rollouts")
    if (
        not isinstance(outcomes, dict)
        or not isinstance(rollouts, int)
        or rollouts < 100
        or not isinstance(outcomes.get("incident_count"), int)
        or not isinstance(outcomes.get("equipment_failure_count"), int)
    ):
        return _check(
            "simulation_evidence",
            "UNKNOWN",
            "Simulation results are missing, malformed, or contain fewer than 100 rollouts.",
        )
    if outcomes["incident_count"] > 0 or outcomes["equipment_failure_count"] > 0:
        return _check(
            "simulation_thresholds",
            "FAIL",
            (
                f"{outcomes['incident_count']} of {rollouts} rollouts crossed an incident "
                f"threshold and {outcomes['equipment_failure_count']} reported equipment failure."
            ),
        )
    return _check(
        "simulation_thresholds",
        "PASS",
        (
            f"No configured incident threshold or equipment failure occurred in {rollouts} "
            "rollouts; this is not facility safety certification."
        ),
    )


def _simulation_risk_check(
    intervention: dict[str, Any] | None,
    risk_aversion_weight: float,
) -> tuple[SafetyVerification, float | None, float | None, float | None]:
    risk = intervention.get("risk_estimates") if intervention else None
    mean_loss = risk.get("mean_loss_index") if isinstance(risk, dict) else None
    cvar_loss = risk.get("cvar_95_loss_index") if isinstance(risk, dict) else None
    if (
        isinstance(mean_loss, bool)
        or not isinstance(mean_loss, (int, float))
        or isinstance(cvar_loss, bool)
        or not isinstance(cvar_loss, (int, float))
        or not isfinite(mean_loss)
        or not isfinite(cvar_loss)
        or mean_loss < 0
        or cvar_loss < mean_loss
    ):
        return (
            _check(
                "simulation_risk_metrics",
                "UNKNOWN",
                "Mean loss and empirical CVaR95 are missing or invalid; risk-aware ranking is unavailable.",
            ),
            None,
            None,
            None,
        )
    adjusted_loss = float(mean_loss) + risk_aversion_weight * float(cvar_loss)
    return (
        _check(
            "simulation_risk_metrics",
            "PASS",
            (
                f"Mean loss index {mean_loss:.4f}; empirical CVaR95 {cvar_loss:.4f}; "
                f"risk-adjusted objective {adjusted_loss:.4f}."
            ),
        ),
        float(mean_loss),
        float(cvar_loss),
        adjusted_loss,
    )


def _rank_feasible_decisions(
    decisions: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], str | None]:
    if not decisions:
        return [], None

    metrics = (
        (
            "risk_adjusted_loss_milliunits",
            lambda item: round(float(item["risk_adjusted_loss_index"]) * 1000),
        ),
        (
            "simulated_incident_count",
            lambda item: int(item["simulated_incident_count"]),
        ),
        (
            "mean_maximum_temperature_millic",
            lambda item: round(float(item["mean_maximum_temperature_c"]) * 1000),
        ),
        (
            "mean_maximum_pressure_millikpa",
            lambda item: round(float(item["mean_maximum_pressure_kpa"]) * 1000),
        ),
    )
    model = cp_model.CpModel()
    choices = {
        item["intervention_name"]: model.new_bool_var(item["intervention_name"])
        for item in decisions
    }
    model.add_exactly_one(list(choices.values()))
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = 0
    for _, value_for in metrics:
        objective_value = sum(
            value_for(item) * choices[item["intervention_name"]]
            for item in decisions
        )
        model.minimize(objective_value)
        solver_status = solver.solve(model)
        if solver_status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            raise RuntimeError(
                f"OR-Tools could not optimize feasible interventions: {solver.status_name(solver_status)}"
            )
        optimum = sum(
            value_for(item)
            for item in decisions
            if solver.boolean_value(choices[item["intervention_name"]])
        )
        model.add(objective_value == optimum)

    selected = next(
        item["intervention_name"]
        for item in decisions
        if solver.boolean_value(choices[item["intervention_name"]])
    )
    ranked = sorted(
        decisions,
        key=lambda item: (
            round(float(item["risk_adjusted_loss_index"]) * 1000),
            int(item["simulated_incident_count"]),
            round(float(item["mean_maximum_temperature_c"]) * 1000),
            round(float(item["mean_maximum_pressure_kpa"]) * 1000),
            item["intervention_name"],
        ),
    )
    return ranked, selected


def optimize_decision(
    scenario_id: UUID,
    simulation_report: SimulationReport | dict[str, Any],
    agent_report: AgentOrchestrationReport,
    operational_constraints: OperationalConstraints,
    site_safety_checks: list[SafetyCheckInput],
    risk_aversion_weight: float = 1.0,
) -> DecisionReport:
    """Select a simulation alternative only after candidate-specific safety gates."""
    if isinstance(simulation_report, SimulationReport):
        simulation = simulation_report.model_dump(mode="json")
    else:
        simulation = simulation_report
    if str(simulation.get("scenario_id")) != str(scenario_id):
        raise ValueError("Simulation scenario_id does not match decision scenario_id")
    if agent_report.scenario_id != scenario_id:
        raise ValueError("Agent report scenario_id does not match decision scenario_id")
    if not 0 <= risk_aversion_weight <= 100:
        raise ValueError("risk_aversion_weight must be between 0 and 100")

    results_by_name = {
        result["intervention_name"]: result
        for result in [
            *simulation.get("interventions", []),
            *simulation.get("sequential_action_plans", []),
        ]
        if isinstance(result, dict)
        and (
            result.get("intervention_name") in INTERVENTIONS
            or result.get("intervention_name") in SEQUENTIAL_PLAN_ACTIONS
        )
    }
    agents_by_name = {agent.agent_name: agent for agent in agent_report.agent_results}
    agent_statuses = {
        name: (
            agents_by_name[name].status
            if name in agents_by_name
            else "UNKNOWN"
        )
        for name in sorted(REQUIRED_AGENT_NAMES)
    }
    agent_completeness = (
        _check(
            "five_agent_inputs",
            "PASS",
            "All five specialized agents returned structured results.",
        )
        if set(agents_by_name) == REQUIRED_AGENT_NAMES
        and all(agent_statuses[name] == "COMPLETED" for name in REQUIRED_AGENT_NAMES)
        else _check(
            "five_agent_inputs",
            "UNKNOWN",
            "One or more agents are missing, failed, or returned UNKNOWN; human review is required.",
        )
    )

    site_checks = []
    for item in site_safety_checks:
        result = item.result
        details = item.details
        attestation_status = "NOT_APPLICABLE"
        reviewer_id = None
        if result == "PASS":
            if verify_safety_attestation(item, scenario_id):
                attestation_status = "VERIFIED"
                reviewer_id = item.reviewer_id
                details = (
                    f"{details} Verified reviewer attestation for {item.reviewer_id}."
                ).strip()
            else:
                result = "UNKNOWN"
                attestation_status = "UNVERIFIED"
                details = (
                    f"{details} Unverified caller-supplied PASS was downgraded to "
                    "UNKNOWN; no trusted reviewer attestation was verified."
                ).strip()
        site_checks.append(
            _check(
                item.check_name,
                result,
                details,
                item.evidence_ids,
                attestation_status=attestation_status,
                reviewer_id=reviewer_id,
            )
        )
    if not site_checks:
        site_checks = [
            _check(
                "site_safety_review",
                "UNKNOWN",
                "No site-specific safety checks were supplied.",
            )
        ]

    predictor = agents_by_name.get("predictor")
    predictor_outputs = predictor.outputs if predictor else {}
    future_prediction = predictor_outputs.get("future_sensor_prediction")
    model_review_check: SafetyVerification
    if predictor is None or predictor.status != "COMPLETED" or not future_prediction:
        model_review_check = _check(
            "model_evidence_review",
            "UNKNOWN",
            "A completed model forecast is unavailable; ML outputs cannot substitute for safety verification.",
        )
    elif operational_constraints.model_outputs_reviewed:
        model_review_check = _check(
            "model_evidence_review",
            "PASS",
            "Model outputs were explicitly marked reviewed; they remain supporting evidence only.",
            [
                str(item["reading_id"])
                for item in future_prediction.get("input_evidence", [])
                if item.get("reading_id")
            ],
        )
    else:
        model_review_check = _check(
            "model_evidence_review",
            "UNKNOWN",
            "Model forecasts are available but have not been explicitly reviewed by a human.",
            [
                str(item["reading_id"])
                for item in future_prediction.get("input_evidence", [])
                if item.get("reading_id")
            ],
        )

    evidence_by_id: dict[str, dict[str, Any]] = {}
    investigator = agents_by_name.get("investigator")
    if investigator:
        ranked_hypotheses = investigator.outputs.get("ranked_hypotheses", [])
        if ranked_hypotheses:
            best = ranked_hypotheses[0]
            for category in ("supporting_evidence", "contradicting_evidence"):
                for signal in best.get(category, []):
                    for evidence in signal.get("evidence", []):
                        evidence_id = evidence.get("evidence_id")
                        if evidence_id:
                            evidence_by_id[str(evidence_id)] = evidence
    red_team = agents_by_name.get("red_team")
    if red_team:
        for provenance in red_team.provenance:
            evidence_id = provenance.get("evidence_id")
            if evidence_id:
                evidence_by_id.setdefault(
                    str(evidence_id),
                    {
                        "evidence_id": str(evidence_id),
                        "source": provenance.get("source"),
                        "provenance": provenance.get("provenance"),
                        "origin": "red_team_evidence_integrity_review",
                    },
                )
    anomaly_assessments = predictor_outputs.get("anomaly_assessments", [])
    for assessment in anomaly_assessments:
        if assessment.get("anomaly") and assessment.get("evidence_id"):
            evidence_id = str(assessment["evidence_id"])
            evidence_by_id[evidence_id] = {
                "evidence_id": evidence_id,
                "origin": "isolation_forest",
                "anomaly_score": assessment["anomaly_score"],
                "sensor_evidence": assessment["evidence"],
                "interpretation": "Statistical novelty only; not a safety clearance or confirmed fault.",
            }

    required_operators = operational_constraints.required_operators
    available_operators = operational_constraints.available_operators
    verification_by_intervention: dict[str, list[SafetyVerification]] = {}
    feasible_decisions: list[dict[str, Any]] = []
    rejected: list[RejectedAlternative] = []

    candidate_names = [
        *INTERVENTIONS,
        *(
            name
            for name in SEQUENTIAL_PLAN_ACTIONS
            if name in results_by_name
        ),
    ]
    for intervention_name in candidate_names:
        intervention = results_by_name.get(intervention_name)
        risk_check, mean_loss, cvar_loss, adjusted_loss = _simulation_risk_check(
            intervention,
            risk_aversion_weight,
        )
        checks = [
            _simulation_safety_check(intervention),
            risk_check,
            agent_completeness,
        ]
        checks.extend(site_checks)
        checks.append(model_review_check)

        equipment_field = {
            "activate_backup_cooling": (
                "backup_cooling_available",
                operational_constraints.backup_cooling_available,
            ),
            "isolate_affected_system": (
                "system_isolation_available",
                operational_constraints.system_isolation_available,
            ),
            "controlled_shutdown": (
                "controlled_shutdown_authorized",
                operational_constraints.controlled_shutdown_authorized,
            ),
        }
        actions = SEQUENTIAL_PLAN_ACTIONS.get(
            intervention_name,
            (intervention_name,) if intervention_name in equipment_field else (),
        )
        for action in actions:
            check_name, available = equipment_field[action]
            if available is None:
                checks.append(
                    _check(
                        f"{intervention_name}:{check_name}",
                        "UNKNOWN",
                        f"{check_name.replace('_', ' ')} was not confirmed.",
                    )
                )
            elif available:
                checks.append(
                    _check(
                        f"{intervention_name}:{check_name}",
                        "PASS",
                        f"{check_name.replace('_', ' ')} was confirmed available.",
                    )
                )
            else:
                checks.append(
                    _check(
                        f"{intervention_name}:{check_name}",
                        "FAIL",
                        f"{check_name.replace('_', ' ')} is unavailable.",
                    )
                )

        if not actions:
            checks.append(
                _check("staffing", "PASS", "This alternative has no modeled operator requirement.")
            )
        for action in actions:
            operator_requirement = required_operators[action]
            if operator_requirement == 0:
                checks.append(
                    _check(
                        f"{intervention_name}:staffing:{action}",
                        "PASS",
                        f"{action} has no modeled operator requirement.",
                    )
                )
            elif available_operators is None:
                checks.append(
                    _check(
                        f"{intervention_name}:staffing:{action}",
                        "UNKNOWN",
                        "Available operator staffing was not supplied.",
                    )
                )
            elif available_operators >= operator_requirement:
                checks.append(
                    _check(
                        f"{intervention_name}:staffing:{action}",
                        "PASS",
                        f"{available_operators} operators available; {operator_requirement} required.",
                    )
                )
            else:
                checks.append(
                    _check(
                        f"{intervention_name}:staffing:{action}",
                        "FAIL",
                        f"{available_operators} operators available; {operator_requirement} required.",
                    )
                )

        verification_by_intervention[intervention_name] = checks
        statuses = {check.result for check in checks}
        if statuses == {"PASS"} and intervention is not None:
            outcomes = intervention["simulated_incident_outcomes"]
            temperature = intervention["temperature_distribution"][
                "maximum_temperature_c"
            ]["mean"]
            pressure = intervention["pressure_distribution"][
                "maximum_pressure_kpa"
            ]["mean"]
            evidence = [
                {
                    **value,
                    "relevance": "supporting decision input; not proof of intervention efficacy",
                }
                for value in evidence_by_id.values()
            ]
            feasible_decisions.append(
                {
                    "intervention_name": intervention_name,
                    "simulated_incident_count": outcomes["incident_count"],
                    "mean_maximum_temperature_c": temperature,
                    "mean_maximum_pressure_kpa": pressure,
                    "mean_loss_index": mean_loss,
                    "cvar_95_loss_index": cvar_loss,
                    "risk_adjusted_loss_index": adjusted_loss,
                    "supporting_evidence": evidence,
                }
            )
        else:
            reasons = [
                f"{check.check_name}: {check.result} - {check.details}"
                for check in checks
                if check.result != "PASS"
            ]
            if intervention is None:
                reasons.append("simulation: UNKNOWN - intervention simulation is missing.")
            rejected.append(
                RejectedAlternative(
                    intervention_name=intervention_name,
                    reasons=reasons,
                    safety_verification=checks,
                )
            )

    ranked, selected_name = _rank_feasible_decisions(feasible_decisions)
    ranked_models = [
        RankedDecision(
            intervention_name=item["intervention_name"],
            rank=index,
            simulated_incident_count=item["simulated_incident_count"],
            mean_maximum_temperature_c=item["mean_maximum_temperature_c"],
            mean_maximum_pressure_kpa=item["mean_maximum_pressure_kpa"],
            mean_loss_index=item["mean_loss_index"],
            cvar_95_loss_index=item["cvar_95_loss_index"],
            risk_adjusted_loss_index=item["risk_adjusted_loss_index"],
            supporting_evidence=item["supporting_evidence"],
        )
        for index, item in enumerate(ranked, start=1)
    ]

    unknown_checks = [
        f"{intervention_name}: {check.check_name}"
        for intervention_name, checks in verification_by_intervention.items()
        for check in checks
        if check.result == "UNKNOWN"
    ]
    failed_checks = [
        f"{intervention_name}: {check.check_name}"
        for intervention_name, checks in verification_by_intervention.items()
        for check in checks
        if check.result == "FAIL"
    ]
    human_review_required = bool(unknown_checks) or selected_name is None
    review_reasons = []
    if unknown_checks:
        review_reasons.append(
            "One or more decision-critical safety or evidence checks are UNKNOWN; UNKNOWN is not treated as PASS."
        )
    if selected_name is None:
        review_reasons.append("No intervention passed every applicable safety and feasibility constraint.")
    if failed_checks and selected_name is None:
        review_reasons.append("Unsafe or infeasible alternatives were rejected; qualified review is required.")

    selected = next(
        (item for item in ranked if item["intervention_name"] == selected_name),
        None,
    )
    explanation = (
        (
            f"Selected {selected_name} as the highest-ranked intervention that passed every "
            "applicable safety and feasibility check."
        )
        if selected
        else "Abstained: no intervention passed all applicable safety and feasibility checks."
    )
    if selected:
        explanation += (
            f" Its risk-adjusted loss index was {selected['risk_adjusted_loss_index']:.4f} "
            f"(mean loss {selected['mean_loss_index']:.4f} + "
            f"{risk_aversion_weight:g} × CVaR95 {selected['cvar_95_loss_index']:.4f})."
        )
    explanation += (
        " OR-Tools ranked eligible alternatives by risk-adjusted loss, incident count, "
        "mean maximum temperature, then mean maximum pressure. ML outputs were supporting "
        "evidence and did not override safety verification."
    )

    return DecisionReport(
        decision_id=uuid4(),
        scenario_id=scenario_id,
        status="SELECTED" if selected_name else "ABSTAINED",
        selected_feasible_intervention=selected_name,
        ranked_feasible_decisions=ranked_models,
        rejected_alternatives=rejected,
        optimization_constraints=OPTIMIZATION_CONSTRAINTS,
        safety_verification_results=verification_by_intervention,
        supporting_evidence=list(evidence_by_id.values()),
        decision_explanation=explanation,
        human_review_required=human_review_required,
        human_review_reasons=review_reasons,
        agent_statuses=agent_statuses,
        simulation_seed=int(simulation.get("seed", 0)),
        simulation_rollouts_per_intervention=int(
            simulation.get("number_of_rollouts_per_intervention", 0)
        ),
        risk_aversion_weight=risk_aversion_weight,
    )
