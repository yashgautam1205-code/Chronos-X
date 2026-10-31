"""Runs the evidence-driven pipeline for judge-supplied runtime scenarios."""

from datetime import datetime
from typing import Any
from backend.agents.orchestrator import orchestrate_agents
from backend.decision_engine import optimize_decision
from core.counterfactual_simulation import INTERVENTIONS, simulate_interventions
from core.reconstruction import reconstruct_timeline
from core.runtime_investigation import investigate_runtime_scenario
from models import (
    CausalInvestigationReport,
    ReconstructionReport,
    RuntimeScenarioReport,
    RuntimeScenarioRequest,
    SafetyCheckInput,
    SimulationReport,
)

METRIC_ALIASES = {
    "temperature": "temperature_c",
    "temp_c": "temperature_c",
    "temperature_c": "temperature_c",
    "pressure": "pressure_kpa",
    "pressure_kpa": "pressure_kpa",
    "vibration": "vibration_mm_s",
    "vibration_mm_s": "vibration_mm_s",
    "cooling_pump_status": "cooling_pump_status",
    "power_supply_status": "power_supply_status",
    "smoke_detected": "smoke_detected",
}
SIMULATOR_SUPPORTED_ASSET_TYPE = "thermal_pressure_equipment"


def _scenario_sensor_rows(request: RuntimeScenarioRequest) -> list[dict[str, Any]]:
    rows = []
    for reading in request.sensor_readings:
        rows.append(
            {
                "reading_id": str(reading.id),
                "timestamp": reading.timestamp,
                "received_at": reading.received_at or reading.timestamp,
                "equipment_id": reading.equipment_id or "unknown",
                "sensor_id": reading.sensor_id,
                "metric": reading.metric,
                "value": reading.value,
                "unit": reading.unit,
                "provenance": reading.provenance,
                "metadata": reading.metadata,
            }
        )
    return rows


def _latest_measurements(
    request: RuntimeScenarioRequest,
) -> tuple[dict[str, float], set[str], set[str]]:
    equipment_ids = {
        reading.equipment_id or "unknown"
        for reading in request.sensor_readings
    }
    if request.simulation_equipment_id:
        selected_equipment = request.simulation_equipment_id
    elif len(equipment_ids) == 1:
        selected_equipment = next(iter(equipment_ids))
    else:
        return {}, equipment_ids, set()

    latest: dict[str, tuple[datetime, set[float]]] = {}
    for reading in request.sensor_readings:
        if (reading.equipment_id or "unknown") != selected_equipment:
            continue
        metric = METRIC_ALIASES.get(reading.metric.strip().lower())
        if metric is None:
            continue
        current = latest.get(metric)
        if current is None or reading.timestamp > current[0]:
            latest[metric] = (reading.timestamp, {float(reading.value)})
        elif reading.timestamp == current[0]:
            current[1].add(float(reading.value))
    conflicting_metrics = {
        metric
        for metric, (_, values) in latest.items()
        if len(values) > 1
    }
    return (
        {
            metric: next(iter(values))
            for metric, (_, values) in latest.items()
            if metric not in conflicting_metrics
        },
        equipment_ids,
        conflicting_metrics,
    )


def _predictor_readings(request: RuntimeScenarioRequest) -> list[dict[str, Any]]:
    readings_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    conflicts: set[tuple[str, str]] = set()
    feature_names = {
        "temperature_c",
        "pressure_kpa",
        "vibration_mm_s",
        "cooling_pump_status",
        "power_supply_status",
        "smoke_detected",
    }
    for reading in request.sensor_readings:
        metric = METRIC_ALIASES.get(reading.metric.strip().lower())
        if metric not in feature_names:
            continue
        equipment_id = reading.equipment_id or "unknown"
        timestamp = reading.timestamp.isoformat()
        key = (equipment_id, timestamp)
        target = readings_by_key.setdefault(
            key,
            {
                "reading_id": str(reading.id),
                "equipment_id": equipment_id,
                "timestamp": timestamp,
                "provenance": reading.provenance,
            },
        )
        if metric in target and target[metric] != reading.value:
            conflicts.add(key)
        else:
            target[metric] = reading.value
    return [
        row
        for key, row in sorted(readings_by_key.items())
        if key not in conflicts and feature_names.issubset(row)
    ]


def analyze_runtime_scenario(request: RuntimeScenarioRequest) -> RuntimeScenarioReport:
    scenario_id = request.scenario_id
    sensor_rows = _scenario_sensor_rows(request)
    reconstruction_data = reconstruct_timeline(
        scenario_id,
        request.events,
        sensor_rows,
        expected_interval_seconds=8.0,
    )
    predictor_readings = _predictor_readings(request)
    reconstruction_data["predictor_readings"] = predictor_readings
    reconstruction = ReconstructionReport.model_validate(reconstruction_data)

    investigation_data = investigate_runtime_scenario(
        scenario_id,
        reconstruction.model_dump(mode="json"),
        request.causal_hypotheses,
        request.causal_relationships,
        [asset.model_dump(mode="json") for asset in request.infrastructure],
    )
    investigation = CausalInvestigationReport.model_validate(investigation_data)

    unsupported: list[str] = []
    simulation: SimulationReport | None = None
    simulation_data: dict[str, Any] = {
        "scenario_id": str(scenario_id),
        "seed": 0,
        "number_of_rollouts_per_intervention": 0,
        "interventions": [],
        "sequential_action_plans": [],
    }
    decision_constraints = request.operational_constraints
    safety_checks = list(request.safety_checks)
    additional_modalities = {
        reading.metric.strip().lower()
        for reading in request.sensor_readings
        if reading.metric.strip().lower()
        in {
            "coolant_flow_lpm",
            "coolant_flow_rate",
            "coolant_flow_rate_lpm",
            "coolant_inlet_flow_lpm",
            "coolant_inlet_flow_rate",
            "coolant_outlet_flow_lpm",
            "coolant_outlet_flow_rate",
            "flow_failure_status",
            "voltage_v",
            "voltage",
            "current_a",
            "current",
            "power_kw",
            "power_consumption_kw",
            "current_overload",
            "electrical_fault_status",
        }
    }
    if additional_modalities:
        unsupported.append(
            "Coolant-flow and electrical evidence is reconstructed and exposed to agents, "
            "but the current counterfactual simulator has no validated physics for these "
            "modalities: " + ", ".join(sorted(additional_modalities))
        )
        safety_checks.append(
            SafetyCheckInput(
                check_name="additional_sensor_simulator_support",
                result="UNKNOWN",
                details=(
                    "Measurements are available for investigation, but intervention outcomes "
                    "cannot be simulated for coolant-flow/electrical dynamics."
                ),
            )
        )
    unsupported_fields = sorted(request.model_extra or {})
    if unsupported_fields:
        unsupported.append(
            "Unsupported runtime scenario fields were preserved but not interpreted: "
            + ", ".join(unsupported_fields)
        )
        safety_checks.append(
            SafetyCheckInput(
                check_name="runtime_schema_support",
                result="UNKNOWN",
                details=(
                    "Unrecognized fields may contain decision-relevant constraints "
                    "or scenario semantics."
                ),
            )
        )
    unsupported_asset_fields = sorted(
        {
            field
            for asset in request.infrastructure
            for field in (asset.model_extra or {})
        }
    )
    if unsupported_asset_fields:
        unsupported.append(
            "Unsupported infrastructure fields were preserved but not interpreted: "
            + ", ".join(unsupported_asset_fields)
        )
        safety_checks.append(
            SafetyCheckInput(
                check_name="infrastructure_schema_support",
                result="UNKNOWN",
                details=(
                    "Unrecognized asset attributes may change equipment physics, "
                    "dependencies, or safety constraints."
                ),
            )
        )

    unknown_interventions = sorted(
        set(request.available_interventions) - set(INTERVENTIONS)
    )
    if unknown_interventions:
        unsupported.append(
            "No counterfactual physics model is registered for intervention(s): "
            + ", ".join(unknown_interventions)
        )
        safety_checks.append(
            SafetyCheckInput(
                check_name="scenario_intervention_model_support",
                result="UNKNOWN",
                details="One or more requested interventions are not modeled; they cannot be ranked or recommended.",
            )
        )

    if request.available_interventions:
        decision_constraints = decision_constraints.model_copy(
            update={
                "backup_cooling_available": (
                    "activate_backup_cooling" in request.available_interventions
                ),
                "system_isolation_available": (
                    "isolate_affected_system" in request.available_interventions
                ),
                "controlled_shutdown_authorized": (
                    "controlled_shutdown" in request.available_interventions
                ),
            }
        )

    if request.simulation is None:
        unsupported.append(
            "Simulation was not run: provide explicit scenario thresholds and a simulation configuration."
        )
        safety_checks.append(
            SafetyCheckInput(
                check_name="scenario_simulation_support",
                result="UNKNOWN",
                details="No scenario-specific counterfactual configuration was provided.",
            )
        )
    else:
        latest, equipment_ids, conflicting_metrics = _latest_measurements(request)
        selected_equipment_id = request.simulation_equipment_id
        if selected_equipment_id is None and len(equipment_ids) == 1:
            selected_equipment_id = next(iter(equipment_ids))
        selected_asset = next(
            (
                asset
                for asset in request.infrastructure
                if asset.asset_id == selected_equipment_id
            ),
            None,
        )
        if len(equipment_ids) > 1 and not request.simulation_equipment_id:
            unsupported.append(
                "The simulator models one equipment stream; set simulation_equipment_id to select it."
            )
        elif (
            selected_asset is None
            or selected_asset.asset_type != SIMULATOR_SUPPORTED_ASSET_TYPE
        ):
            unsupported.append(
                "Simulation was not run because the selected equipment is not declared "
                f"as {SIMULATOR_SUPPORTED_ASSET_TYPE}; no compatible physics model is registered."
            )
            safety_checks.append(
                SafetyCheckInput(
                    check_name="simulation_equipment_model_support",
                    result="UNKNOWN",
                    details=(
                        "The generic thermal/pressure simulator cannot establish "
                        "physics for this equipment type."
                    ),
                )
            )
        elif conflicting_metrics:
            metric_names = ", ".join(sorted(conflicting_metrics))
            unsupported.append(
                "Simulation was not run because latest sensor values conflict for: "
                f"{metric_names}."
            )
            safety_checks.append(
                SafetyCheckInput(
                    check_name="latest_sensor_conflicts",
                    result="UNKNOWN",
                    details=(
                        "Conflicting latest readings are preserved and cannot be "
                        "collapsed into a single simulation state."
                    ),
                )
            )
        elif not {"temperature_c", "pressure_kpa"}.issubset(latest):
            unsupported.append(
                "The simulator requires current temperature_c and pressure_kpa readings for the selected equipment."
            )
            safety_checks.append(
                SafetyCheckInput(
                    check_name="simulation_initial_state",
                    result="UNKNOWN",
                    details=(
                        "Current temperature and pressure readings are required "
                        "to establish the simulation initial state."
                    ),
                )
            )
        else:
            parameters = request.simulation.model_dump()
            parameters["initial_temperature_c"] = latest["temperature_c"]
            parameters["initial_pressure_kpa"] = latest["pressure_kpa"]
            if "cooling_pump_status" in latest:
                parameters["initial_cooling_failure"] = (
                    latest["cooling_pump_status"] < 0.5
                )
            if (
                parameters["thermal_incident_threshold_c"]
                <= parameters["initial_temperature_c"]
                or parameters["pressure_incident_threshold_kpa"]
                <= parameters["initial_pressure_kpa"]
            ):
                unsupported.append(
                    "Simulation thresholds must exceed the latest selected-equipment readings; no rollout was substituted."
                )
            else:
                simulation_data = simulate_interventions(
                    scenario_id,
                    **parameters,
                )
                simulation = SimulationReport.model_validate(simulation_data)

    reconstruction_payload = reconstruction.model_dump(mode="json")
    reconstruction_payload["predictor_readings"] = predictor_readings
    agent_report = orchestrate_agents(
        scenario_id,
        reconstruction_payload,
        simulation_report=simulation_data,
        investigation_report=investigation.model_dump(mode="json"),
        risk_aversion_weight=request.risk_aversion_weight,
    )
    decision = optimize_decision(
        scenario_id,
        simulation_data,
        agent_report,
        decision_constraints,
        safety_checks,
        request.risk_aversion_weight,
    )

    assumptions = [
        "The scenario was analyzed from only the submitted events, sensor readings, infrastructure description, and constraints; the predefined synthetic scenario datasets were not consulted.",
        "The counterfactual simulator models one thermal/pressure equipment stream and the four built-in interventions; it is not a general physical simulator.",
        "Unmodeled action physics, unsupported sensor modalities, missing thresholds, and unknown site-safety inputs cannot establish action safety.",
        "ML inference runs only when submitted evidence can be transformed into the trained model's complete six-feature sensor format; model outputs remain supporting evidence.",
        "No free-text semantic extraction or organizer simulator adapter is available in this runtime interface.",
    ]
    if simulation:
        assumptions.extend(simulation.simulation_assumptions)

    return RuntimeScenarioReport(
        scenario_id=scenario_id,
        name=request.name,
        description=request.description,
        infrastructure=request.infrastructure,
        reconstructed_timeline=reconstruction,
        causal_investigation=investigation,
        simulation=simulation,
        agent_council=agent_report,
        decision=decision,
        unsupported_capabilities=unsupported,
        assumptions_and_uncertainty=assumptions,
    )
