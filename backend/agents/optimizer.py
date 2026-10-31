"""Optimizer agent preparing intervention choices for OR-Tools."""

from math import isfinite
from typing import Any

from ortools.sat.python import cp_model

from models import AgentResult


def run_optimizer_agent(
    simulation_report: dict[str, Any] | None,
    *,
    risk_aversion_weight: float = 1.0,
    sensor_evidence_context: dict[str, Any] | None = None,
) -> AgentResult:
    if not simulation_report:
        return AgentResult(
            agent_name="optimizer",
            status="UNKNOWN",
            summary="No simulation outcomes are available to compare.",
            findings=[{"reason": "simulation_outcomes_unavailable"}],
            outputs={
                "sensor_evidence_context": sensor_evidence_context or {},
                "sensor_evidence_used_for_risk_scoring": False,
            },
        )
    if not 0 <= risk_aversion_weight <= 100:
        raise ValueError("risk_aversion_weight must be between 0 and 100")
    interventions = [
        *simulation_report.get("interventions", []),
        *simulation_report.get("sequential_action_plans", []),
    ]
    if not interventions:
        return AgentResult(
            agent_name="optimizer",
            status="UNKNOWN",
            summary="Simulation report has no intervention alternatives.",
            findings=[{"reason": "intervention_alternatives_missing"}],
        )

    decision_inputs = []
    for intervention in interventions:
        outcomes = intervention.get("simulated_incident_outcomes", {})
        risk = intervention.get("risk_estimates", {})
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
            return AgentResult(
                agent_name="optimizer",
                status="UNKNOWN",
                summary="Risk metrics are missing; the optimizer cannot compare expected and tail loss.",
                findings=[{"reason": "simulation_risk_metrics_unavailable"}],
            )
        risk_adjusted_loss = float(risk["mean_loss_index"]) + (
            risk_aversion_weight * float(risk["cvar_95_loss_index"])
        )
        decision_inputs.append(
            {
                "intervention_name": intervention["intervention_name"],
                "simulated_incident_count": int(outcomes["incident_count"]),
                "rollouts": int(intervention["number_of_rollouts"]),
                "mean_maximum_temperature_c": intervention[
                    "temperature_distribution"
                ]["maximum_temperature_c"]["mean"],
                "mean_maximum_pressure_kpa": intervention[
                    "pressure_distribution"
                ]["maximum_pressure_kpa"]["mean"],
                "mean_loss_index": float(mean_loss),
                "cvar_95_loss_index": float(cvar_loss),
                "risk_adjusted_loss_index": risk_adjusted_loss,
            }
        )

    model = cp_model.CpModel()
    choices = [
        model.new_bool_var(item["intervention_name"])
        for item in decision_inputs
    ]
    model.add_exactly_one(choices)
    solver = cp_model.CpSolver()
    objective_metrics = (
        lambda item: round(item["risk_adjusted_loss_index"] * 1000),
        lambda item: item["simulated_incident_count"],
    )
    status_code = cp_model.UNKNOWN
    for value_for in objective_metrics:
        objective_value = sum(
            value_for(item) * choice
            for item, choice in zip(decision_inputs, choices)
        )
        model.minimize(objective_value)
        status_code = solver.solve(model)
        if status_code not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return AgentResult(
                agent_name="optimizer",
                status="FAILED",
                summary="OR-Tools could not solve the intervention comparison.",
                errors=[f"OR-Tools returned status {solver.status_name(status_code)}"],
                outputs={
                    "decision_inputs": decision_inputs,
                    "sensor_evidence_context": sensor_evidence_context or {},
                    "sensor_evidence_used_for_risk_scoring": False,
                },
            )
        optimum = sum(
            value_for(item)
            for item, choice in zip(decision_inputs, choices)
            if solver.boolean_value(choice)
        )
        model.add(objective_value == optimum)
    selected = next(
        item["intervention_name"]
        for item, choice in zip(decision_inputs, choices)
        if solver.boolean_value(choice)
    )
    return AgentResult(
        agent_name="optimizer",
        status="COMPLETED",
        summary="Compared standalone interventions and conditional sequences using mean loss plus weighted empirical CVaR95.",
        findings=[
            {
                "selected_under_objective": selected,
                "objective": "minimize mean loss index + risk_aversion_weight * CVaR95, then incident count",
                "risk_aversion_weight": risk_aversion_weight,
                "solver_status": solver.status_name(status_code),
                "decision_support_only": True,
            }
        ],
        outputs={
            "decision_inputs": decision_inputs,
            "risk_aversion_weight": risk_aversion_weight,
            "sensor_evidence_context": sensor_evidence_context or {},
            "sensor_evidence_used_for_risk_scoring": False,
            "optimization_limitations": [
                "This independent agent ranking does not verify site feasibility; final selection must pass the separate safety-gated decision engine.",
                "The selected alternative is conditional on the simulation and objective; it is not an operational instruction.",
            ],
        },
    )
