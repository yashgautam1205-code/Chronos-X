"""Seeded SimPy Monte Carlo simulation of industrial interventions."""

from collections import Counter
from dataclasses import dataclass, field
from math import ceil
from typing import Any
from uuid import UUID

import numpy as np
import simpy

INTERVENTIONS = (
    "no_intervention",
    "activate_backup_cooling",
    "isolate_affected_system",
    "controlled_shutdown",
)
SEQUENTIAL_PLAN_ACTIONS = {
    "backup_then_shutdown": ("activate_backup_cooling", "controlled_shutdown"),
    "isolation_then_shutdown": ("isolate_affected_system", "controlled_shutdown"),
}
SEQUENTIAL_PLAN_NAMES = tuple(SEQUENTIAL_PLAN_ACTIONS)

SIMULATION_ASSUMPTIONS = [
    "This is a simplified, illustrative single-equipment model, not an engineering safety case or calibrated plant digital twin.",
    "Each rollout uses one-minute virtual steps; the default horizon is four hours.",
    "Temperature and pressure start at the request's stated values and evolve according to the documented rates and seeded Gaussian process noise.",
    "Illustrative temperature rates are 4.5 degC/hour of active process heating, 4.2 degC/hour of normal cooling, 3.8 degC/hour of backup cooling, and 0.8 degC/hour of shutdown residual cooling.",
    "Pressure changes by 0.055 kPa per degree above 75 degC per hour, plus 0.65 kPa/hour relief when isolated or 1.1 kPa/hour during shutdown.",
    "Temperature noise has a 0.22 degC per square-root-hour scale; pressure noise has a 0.7 kPa per square-root-hour scale.",
    "A pre-existing cooling failure is controlled by initial_cooling_failure; new pump and equipment failures are sampled from configured per-hour hazard rates.",
    "Backup cooling takes effect after its configured activation delay and is available according to its configured probability.",
    "Isolation reduces process heat generation after its response delay; controlled shutdown stops process heat generation after its shutdown delay and allows modeled cooling and pressure relief.",
    "After isolation takes effect, a configurable toy-model probability of a delayed trapped-pressure secondary failure is sampled; this assumption is synthetic and is not calibrated to a real plant.",
    "Sequential plans escalate to controlled shutdown only when the simulated first action is unavailable or configured incident thresholds/equipment-failure indicators are reached; escalation uses the configured shutdown delay.",
    "The loss index is a synthetic comparison objective: temperature exceedance degree-hours, pressure exceedance kPa-hours scaled by 0.1, threshold penalties, and intervention burden; it is not monetary loss.",
    "Reported mean-loss uncertainty is a normal-approximation 95% interval over rollout samples; CVaR is the empirical mean of the worst 5% of the loss-index samples.",
    "Thermal and pressure incident thresholds are configurable model indicators, not certified operating limits.",
    "The simulated incident outcome is the highest-severity threshold reached in a rollout; it is not an empirically validated catastrophe probability.",
    "Interventions share matching rollout seeds to reduce Monte Carlo comparison noise; results remain stochastic and conditional on these assumptions.",
    "Reported incident rates are observed simulated frequencies with Wilson intervals; these quantify finite-rollout sampling uncertainty only, not model uncertainty.",
]


@dataclass
class _EquipmentState:
    temperature_c: float
    pressure_kpa: float
    pump_failed: bool
    equipment_failed: bool = False
    process_active: bool = True
    isolated: bool = False
    backup_active: bool = False
    max_temperature_c: float = 0.0
    max_pressure_kpa: float = 0.0
    catastrophic_incident: bool = False
    equipment_failure_incident: bool = False
    thermal_incident: bool = False
    pressure_incident: bool = False
    smoke_incident: bool = False
    secondary_failure: bool = False
    shutdown_requested_at: float | None = None
    actions_taken: list[str] = field(default_factory=list)
    temperature_exceedance_degree_hours: float = 0.0
    pressure_exceedance_kpa_hours: float = 0.0


def _wilson_interval(successes: int, trials: int) -> list[float]:
    if trials == 0:
        return [0.0, 0.0]
    z = 1.959963984540054
    observed = successes / trials
    denominator = 1 + z * z / trials
    center = (observed + z * z / (2 * trials)) / denominator
    margin = (
        z
        * np.sqrt(observed * (1 - observed) / trials + z * z / (4 * trials * trials))
        / denominator
    )
    return [round(float(max(0, center - margin)), 4), round(float(min(1, center + margin)), 4)]


def _one_rollout(
    *,
    intervention: str,
    rng: np.random.Generator,
    initial_temperature_c: float,
    initial_pressure_kpa: float,
    initial_cooling_failure: bool,
    simulation_hours: float,
    time_step_minutes: float,
    pump_failure_rate_per_hour: float,
    equipment_failure_rate_per_hour: float,
    backup_available_probability: float,
    backup_activation_delay_minutes: float,
    isolation_delay_minutes: float,
    shutdown_delay_minutes: float,
    isolation_secondary_failure_probability: float,
    isolation_secondary_failure_delay_minutes: float,
    isolation_secondary_pressure_rise_kpa_per_hour: float,
    thermal_incident_threshold_c: float,
    pressure_incident_threshold_kpa: float,
) -> dict[str, Any]:
    env = simpy.Environment()
    state = _EquipmentState(
        temperature_c=initial_temperature_c,
        pressure_kpa=initial_pressure_kpa,
        pump_failed=initial_cooling_failure,
        max_temperature_c=initial_temperature_c,
        max_pressure_kpa=initial_pressure_kpa,
    )
    backup_available = bool(rng.random() < backup_available_probability)
    secondary_rng = np.random.default_rng(int(rng.bit_generator.random_raw()))
    isolation_secondary_failure_sampled = bool(
        intervention in {"isolate_affected_system", "isolation_then_shutdown"}
        and secondary_rng.random() < isolation_secondary_failure_probability
    )
    step_hours = time_step_minutes / 60
    time_step_minutes_float = float(time_step_minutes)
    simulation_minutes = simulation_hours * 60
    thermal_severity_threshold = thermal_incident_threshold_c + 10.0
    pressure_severity_threshold = pressure_incident_threshold_kpa + 40.0

    def run_process():
        while env.now < simulation_minutes:
            elapsed_minutes = float(env.now)
            if intervention in {
                "activate_backup_cooling",
                "backup_then_shutdown",
            } and elapsed_minutes >= backup_activation_delay_minutes and backup_available:
                state.backup_active = True
                if "activate_backup_cooling" not in state.actions_taken:
                    state.actions_taken.append("activate_backup_cooling")
            if intervention in {
                "isolate_affected_system",
                "isolation_then_shutdown",
            } and elapsed_minutes >= isolation_delay_minutes:
                state.isolated = True
                if "isolate_affected_system" not in state.actions_taken:
                    state.actions_taken.append("isolate_affected_system")
            if (
                intervention == "controlled_shutdown"
                and elapsed_minutes >= shutdown_delay_minutes
            ):
                state.process_active = False
                state.isolated = True
                if "controlled_shutdown" not in state.actions_taken:
                    state.actions_taken.append("controlled_shutdown")
            if intervention == "backup_then_shutdown":
                first_action_failed = (
                    elapsed_minutes >= backup_activation_delay_minutes
                    and not backup_available
                )
                escalation_condition = (
                    first_action_failed
                    or state.thermal_incident
                    or state.temperature_c >= thermal_incident_threshold_c
                    or state.equipment_failed
                )
                if escalation_condition and state.shutdown_requested_at is None:
                    state.shutdown_requested_at = elapsed_minutes
                if (
                    state.shutdown_requested_at is not None
                    and elapsed_minutes
                    >= state.shutdown_requested_at + shutdown_delay_minutes
                ):
                    state.process_active = False
                    state.isolated = True
                    if "controlled_shutdown" not in state.actions_taken:
                        state.actions_taken.append("controlled_shutdown")
            if intervention == "isolation_then_shutdown":
                escalation_condition = (
                    state.thermal_incident
                    or state.pressure_incident
                    or state.temperature_c >= thermal_incident_threshold_c
                    or state.pressure_kpa >= pressure_incident_threshold_kpa
                    or state.equipment_failed
                )
                if state.isolated and escalation_condition and state.shutdown_requested_at is None:
                    state.shutdown_requested_at = elapsed_minutes
                if (
                    state.shutdown_requested_at is not None
                    and elapsed_minutes
                    >= state.shutdown_requested_at + shutdown_delay_minutes
                ):
                    state.process_active = False
                    if "controlled_shutdown" not in state.actions_taken:
                        state.actions_taken.append("controlled_shutdown")
            if (
                intervention in {"isolate_affected_system", "isolation_then_shutdown"}
                and isolation_secondary_failure_sampled
                and elapsed_minutes
                >= isolation_delay_minutes + isolation_secondary_failure_delay_minutes
            ):
                state.secondary_failure = True

            pump_failure_probability = 1 - np.exp(
                -pump_failure_rate_per_hour * step_hours
            )
            equipment_failure_probability = 1 - np.exp(
                -equipment_failure_rate_per_hour
                * np.exp(max(0.0, state.temperature_c - 80.0) / 20.0)
                * step_hours
            )
            if not state.pump_failed and rng.random() < pump_failure_probability:
                state.pump_failed = True
            if (
                not state.equipment_failed
                and rng.random() < equipment_failure_probability
            ):
                state.equipment_failed = True
                state.equipment_failure_incident = True

            process_heat_rate = 4.5 if state.process_active else 0.0
            if state.isolated and state.process_active:
                process_heat_rate *= 0.25
            cooling_rate = 4.2 if not state.pump_failed else 0.0
            if state.backup_active:
                cooling_rate += 3.8
            if not state.process_active:
                cooling_rate += 0.8

            temperature_noise = float(rng.normal(0.0, 0.22 * np.sqrt(step_hours)))
            state.temperature_c += (
                process_heat_rate - cooling_rate
            ) * step_hours + temperature_noise
            state.temperature_c = max(-20.0, state.temperature_c)

            pressure_noise = float(rng.normal(0.0, 0.7 * np.sqrt(step_hours)))
            temperature_pressure_effect = (
                (state.temperature_c - 75.0) * 0.055 * step_hours
            )
            if not state.process_active:
                pressure_relief = 1.1 * step_hours
            elif state.isolated:
                pressure_relief = 0.65 * step_hours
            else:
                pressure_relief = 0.0
            state.pressure_kpa = max(
                0.0,
                state.pressure_kpa
                + temperature_pressure_effect
                - pressure_relief
                + (
                    isolation_secondary_pressure_rise_kpa_per_hour * step_hours
                    if state.secondary_failure
                    else 0.0
                )
                + pressure_noise,
            )

            state.max_temperature_c = max(
                state.max_temperature_c,
                state.temperature_c,
            )
            state.max_pressure_kpa = max(state.max_pressure_kpa, state.pressure_kpa)
            state.temperature_exceedance_degree_hours += (
                max(0.0, state.temperature_c - thermal_incident_threshold_c)
                * step_hours
            )
            state.pressure_exceedance_kpa_hours += (
                max(0.0, state.pressure_kpa - pressure_incident_threshold_kpa)
                * step_hours
            )
            state.thermal_incident |= (
                state.temperature_c >= thermal_incident_threshold_c
            )
            state.pressure_incident |= (
                state.pressure_kpa >= pressure_incident_threshold_kpa
            )
            state.smoke_incident |= (
                state.temperature_c >= thermal_severity_threshold
            )
            state.catastrophic_incident |= (
                state.temperature_c >= thermal_severity_threshold
                or state.pressure_kpa >= pressure_severity_threshold
            )
            yield env.timeout(time_step_minutes_float)

    env.process(run_process())
    env.run(until=simulation_minutes + time_step_minutes_float)

    if state.catastrophic_incident:
        outcome = "catastrophic_threshold"
    elif state.thermal_incident or state.pressure_incident:
        outcome = "incident_threshold"
    elif state.equipment_failure_incident:
        outcome = "equipment_failure_without_threshold"
    else:
        outcome = "no_incident_threshold"
    return {
        "maximum_temperature_c": state.max_temperature_c,
        "final_temperature_c": state.temperature_c,
        "maximum_pressure_kpa": state.max_pressure_kpa,
        "final_pressure_kpa": state.pressure_kpa,
        "incident_outcome": outcome,
        "incident": outcome != "no_incident_threshold",
        "catastrophic_incident": state.catastrophic_incident,
        "equipment_failure": state.equipment_failure_incident,
        "delayed_secondary_failure": state.secondary_failure,
        "thermal_threshold_exceeded": state.thermal_incident,
        "pressure_threshold_exceeded": state.pressure_incident,
        "smoke_threshold_exceeded": state.smoke_incident,
        "backup_available": backup_available,
        "actions_taken": list(state.actions_taken),
        "temperature_exceedance_degree_hours": (
            state.temperature_exceedance_degree_hours
        ),
        "pressure_exceedance_kpa_hours": state.pressure_exceedance_kpa_hours,
    }


def _distribution(values: list[float]) -> dict[str, float]:
    samples = np.asarray(values, dtype=float)
    return {
        "mean": round(float(np.mean(samples)), 3),
        "maximum": round(float(np.max(samples)), 3),
        "minimum": round(float(np.min(samples)), 3),
        "p50": round(float(np.percentile(samples, 50)), 3),
        "p90": round(float(np.percentile(samples, 90)), 3),
        "p95": round(float(np.percentile(samples, 95)), 3),
    }


def _loss_index(
    sample: dict[str, Any],
    *,
    intervention: str,
) -> float:
    temperature_excess = sample["temperature_exceedance_degree_hours"]
    pressure_excess = sample["pressure_exceedance_kpa_hours"] * 0.1
    threshold_penalty = (
        100.0 if sample["catastrophic_incident"] else
        30.0 if sample["thermal_threshold_exceeded"] or sample["pressure_threshold_exceeded"] else 0.0
    )
    failure_penalty = 20.0 if sample["equipment_failure"] else 0.0
    intervention_burden = {
        "no_intervention": 0.0,
        "activate_backup_cooling": 2.0,
        "isolate_affected_system": 4.0,
        "controlled_shutdown": 8.0,
        "backup_then_shutdown": 10.0,
        "isolation_then_shutdown": 12.0,
    }[intervention]
    return round(
        temperature_excess + pressure_excess + threshold_penalty
        + failure_penalty + intervention_burden,
        4,
    )


def _loss_risk_summary(losses: list[float]) -> dict[str, Any]:
    samples = np.asarray(losses, dtype=float)
    tail_size = max(1, ceil(0.05 * len(samples)))
    mean = float(np.mean(samples))
    standard_error = float(np.std(samples, ddof=1) / np.sqrt(len(samples)))
    return {
        "loss_index_definition": (
            "integrated temperature exceedance degree-hours + 0.1 * integrated pressure "
            "exceedance kPa-hours + threshold/equipment-failure penalties + intervention burden"
        ),
        "mean_loss_index": round(mean, 4),
        "cvar_95_loss_index": round(float(np.mean(np.sort(samples)[-tail_size:])), 4),
        "loss_index_standard_deviation": round(float(np.std(samples, ddof=1)), 4),
        "mean_loss_index_95_interval": [
            round(max(0.0, mean - 1.96 * standard_error), 4),
            round(mean + 1.96 * standard_error, 4),
        ],
        "uncertainty_method": "Normal-approximation 95% interval for the sample mean across rollouts.",
    }


def simulate_interventions(
    scenario_id: UUID,
    *,
    rollouts: int = 100,
    seed: int = 42,
    initial_temperature_c: float = 84.0,
    initial_pressure_kpa: float = 112.0,
    initial_cooling_failure: bool = True,
    simulation_hours: float = 4.0,
    time_step_minutes: float = 1.0,
    pump_failure_rate_per_hour: float = 0.01,
    equipment_failure_rate_per_hour: float = 0.002,
    backup_available_probability: float = 0.9,
    backup_activation_delay_minutes: float = 10.0,
    isolation_delay_minutes: float = 10.0,
    shutdown_delay_minutes: float = 2.0,
    isolation_secondary_failure_probability: float = 0.05,
    isolation_secondary_failure_delay_minutes: float = 60.0,
    isolation_secondary_pressure_rise_kpa_per_hour: float = 8.0,
    thermal_incident_threshold_c: float = 100.0,
    pressure_incident_threshold_kpa: float = 160.0,
) -> dict[str, Any]:
    """Run reproducible, paired Monte Carlo rollouts for each intervention."""
    if rollouts < 100:
        raise ValueError("At least 100 rollouts per intervention are required")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    if simulation_hours <= 0 or time_step_minutes <= 0:
        raise ValueError("Simulation duration and time step must be greater than zero")
    if not 0 < time_step_minutes <= simulation_hours * 60:
        raise ValueError("time_step_minutes must not exceed the simulation duration")
    total_steps = (
        rollouts
        * (len(INTERVENTIONS) + len(SEQUENTIAL_PLAN_NAMES))
        * ceil(simulation_hours * 60 / time_step_minutes)
    )
    if total_steps > 2_000_000:
        raise ValueError(
            "Simulation exceeds the 2,000,000-step workload limit; reduce rollouts or duration"
        )
    if pump_failure_rate_per_hour < 0 or equipment_failure_rate_per_hour < 0:
        raise ValueError("Failure rates must not be negative")
    if not 0 <= backup_available_probability <= 1:
        raise ValueError("backup_available_probability must be between 0 and 1")
    if backup_activation_delay_minutes < 0 or isolation_delay_minutes < 0:
        raise ValueError("Intervention delays must not be negative")
    if shutdown_delay_minutes < 0:
        raise ValueError("shutdown_delay_minutes must not be negative")
    if not 0 <= isolation_secondary_failure_probability <= 1:
        raise ValueError("isolation_secondary_failure_probability must be between 0 and 1")
    if isolation_secondary_failure_delay_minutes < 0:
        raise ValueError("isolation_secondary_failure_delay_minutes must not be negative")
    if isolation_secondary_pressure_rise_kpa_per_hour < 0:
        raise ValueError("isolation_secondary_pressure_rise_kpa_per_hour must not be negative")
    if thermal_incident_threshold_c <= initial_temperature_c:
        raise ValueError("Thermal threshold must exceed initial temperature")
    if pressure_incident_threshold_kpa <= initial_pressure_kpa:
        raise ValueError("Pressure threshold must exceed initial pressure")

    input_parameters = {
        "initial_temperature_c": initial_temperature_c,
        "initial_pressure_kpa": initial_pressure_kpa,
        "initial_cooling_failure": initial_cooling_failure,
        "simulation_hours": simulation_hours,
        "time_step_minutes": time_step_minutes,
        "pump_failure_rate_per_hour": pump_failure_rate_per_hour,
        "equipment_failure_rate_per_hour": equipment_failure_rate_per_hour,
        "backup_available_probability": backup_available_probability,
        "backup_activation_delay_minutes": backup_activation_delay_minutes,
        "isolation_delay_minutes": isolation_delay_minutes,
        "shutdown_delay_minutes": shutdown_delay_minutes,
        "isolation_secondary_failure_probability": isolation_secondary_failure_probability,
        "isolation_secondary_failure_delay_minutes": isolation_secondary_failure_delay_minutes,
        "isolation_secondary_pressure_rise_kpa_per_hour": isolation_secondary_pressure_rise_kpa_per_hour,
        "thermal_incident_threshold_c": thermal_incident_threshold_c,
        "pressure_incident_threshold_kpa": pressure_incident_threshold_kpa,
    }
    results: list[dict[str, Any]] = []
    sequential_results: list[dict[str, Any]] = []
    for intervention in (*INTERVENTIONS, *SEQUENTIAL_PLAN_NAMES):
        samples = []
        for rollout_index in range(rollouts):
            rollout_seed = seed + rollout_index
            rng = np.random.default_rng(rollout_seed)
            sample = _one_rollout(
                intervention=intervention,
                rng=rng,
                **input_parameters,
            )
            sample["rollout_index"] = rollout_index
            sample["loss_index"] = _loss_index(
                sample,
                intervention=intervention,
            )
            samples.append(sample)

        incident_count = sum(sample["incident"] for sample in samples)
        catastrophic_count = sum(
            sample["catastrophic_incident"] for sample in samples
        )
        failure_count = sum(sample["equipment_failure"] for sample in samples)
        secondary_failure_count = sum(
            sample["delayed_secondary_failure"] for sample in samples
        )
        outcome_counts = Counter(sample["incident_outcome"] for sample in samples)
        summary = {
                "intervention_name": intervention,
                "number_of_rollouts": rollouts,
                "temperature_distribution": {
                    "maximum_temperature_c": _distribution(
                        [sample["maximum_temperature_c"] for sample in samples]
                    ),
                    "final_temperature_c": _distribution(
                        [sample["final_temperature_c"] for sample in samples]
                    ),
                },
                "pressure_distribution": {
                    "maximum_pressure_kpa": _distribution(
                        [sample["maximum_pressure_kpa"] for sample in samples]
                    ),
                    "final_pressure_kpa": _distribution(
                        [sample["final_pressure_kpa"] for sample in samples]
                    ),
                },
                "simulated_incident_outcomes": {
                    "counts": dict(sorted(outcome_counts.items())),
                    "incident_count": incident_count,
                    "catastrophic_incident_count": catastrophic_count,
                    "equipment_failure_count": failure_count,
                    "delayed_secondary_failure_count": secondary_failure_count,
                },
                "action_execution_counts": dict(
                    sorted(
                        Counter(
                            action
                            for sample in samples
                            for action in sample["actions_taken"]
                        ).items()
                    )
                ),
                "risk_estimates": {
                    "simulated_incident_rate": round(incident_count / rollouts, 4),
                    "simulated_incident_rate_wilson_95_interval": _wilson_interval(
                        incident_count,
                        rollouts,
                    ),
                    "simulated_catastrophic_rate": round(
                        catastrophic_count / rollouts,
                        4,
                    ),
                    "simulated_catastrophic_rate_wilson_95_interval": _wilson_interval(
                        catastrophic_count,
                        rollouts,
                    ),
                    "simulated_equipment_failure_rate": round(
                        failure_count / rollouts,
                        4,
                    ),
                    "simulated_delayed_secondary_failure_rate": round(
                        secondary_failure_count / rollouts,
                        4,
                    ),
                    **_loss_risk_summary(
                        [sample["loss_index"] for sample in samples]
                    ),
                    "interpretation": (
                        "Observed frequency in this configured simulation only; "
                        "not a calibrated real-world risk estimate."
                    ),
                },
                "rollout_samples": samples,
            }
        if intervention in INTERVENTIONS:
            results.append(summary)
        else:
            sequential_results.append(summary)

    baseline_rate = results[0]["risk_estimates"]["simulated_incident_rate"]
    comparison = []
    for result in results:
        incident_rate = result["risk_estimates"]["simulated_incident_rate"]
        comparison.append(
            {
                "intervention_name": result["intervention_name"],
                "simulated_incident_rate": incident_rate,
                "incident_rate_difference_vs_no_intervention": round(
                    incident_rate - baseline_rate,
                    4,
                ),
                "simulated_incidents_avoided_vs_no_intervention": (
                    results[0]["simulated_incident_outcomes"]["incident_count"]
                    - result["simulated_incident_outcomes"]["incident_count"]
                ),
                "mean_maximum_temperature_c": result["temperature_distribution"][
                    "maximum_temperature_c"
                ]["mean"],
                "mean_maximum_pressure_kpa": result["pressure_distribution"][
                    "maximum_pressure_kpa"
                ]["mean"],
            }
        )

    sequential_comparison = [
        {
            "plan_name": result["intervention_name"],
            "incident_rate": result["risk_estimates"]["simulated_incident_rate"],
            "incident_rate_difference_vs_no_intervention": round(
                result["risk_estimates"]["simulated_incident_rate"] - baseline_rate,
                4,
            ),
            "mean_loss_index": result["risk_estimates"]["mean_loss_index"],
            "cvar_95_loss_index": result["risk_estimates"]["cvar_95_loss_index"],
            "mean_maximum_temperature_c": result["temperature_distribution"][
                "maximum_temperature_c"
            ]["mean"],
            "mean_maximum_pressure_kpa": result["pressure_distribution"][
                "maximum_pressure_kpa"
            ]["mean"],
        }
        for result in sequential_results
    ]

    return {
        "scenario_id": scenario_id,
        "seed": seed,
        "number_of_rollouts_per_intervention": rollouts,
        "interventions": results,
        "sequential_action_plans": sequential_results,
        "sequential_plan_comparison": sequential_comparison,
        "comparison_of_interventions": comparison,
        "simulation_assumptions": SIMULATION_ASSUMPTIONS,
        "input_parameters": input_parameters,
        "simulation_type": "stochastic SimPy process simulation with NumPy Monte Carlo; not machine learning",
    }
