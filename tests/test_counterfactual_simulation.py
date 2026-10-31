from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from backend.main import app
from core.counterfactual_simulation import (
    INTERVENTIONS,
    SIMULATION_ASSUMPTIONS,
    simulate_interventions,
)


def test_all_interventions_run_reproducibly_with_at_least_100_rollouts() -> None:
    scenario_id = uuid4()
    report = simulate_interventions(scenario_id, rollouts=100, seed=121)
    repeated = simulate_interventions(scenario_id, rollouts=100, seed=121)

    assert report == repeated
    assert report["number_of_rollouts_per_intervention"] == 100
    assert [item["intervention_name"] for item in report["interventions"]] == list(
        INTERVENTIONS
    )
    assert len(report["comparison_of_interventions"]) == 4
    assert len(report["sequential_action_plans"]) == 2
    assert len(report["sequential_plan_comparison"]) == 2
    assert report["simulation_assumptions"] == SIMULATION_ASSUMPTIONS
    for intervention in report["interventions"]:
        assert intervention["number_of_rollouts"] == 100
        assert len(intervention["rollout_samples"]) == 100
        assert 0 <= intervention["risk_estimates"]["simulated_incident_rate"] <= 1
        assert len(
            intervention["risk_estimates"][
                "simulated_incident_rate_wilson_95_interval"
            ]
        ) == 2
        assert intervention["temperature_distribution"]["maximum_temperature_c"]["mean"]
        assert intervention["pressure_distribution"]["maximum_pressure_kpa"]["mean"]
        assert sum(
            intervention["simulated_incident_outcomes"]["counts"].values()
        ) == 100
        risk = intervention["risk_estimates"]
        assert risk["cvar_95_loss_index"] >= risk["mean_loss_index"]
        assert risk["loss_index_standard_deviation"] >= 0
        assert risk["mean_loss_index_95_interval"][0] <= risk["mean_loss_index"]
        assert risk["mean_loss_index"] <= risk["mean_loss_index_95_interval"][1]


def test_interventions_change_temperature_and_incident_outcomes() -> None:
    report = simulate_interventions(uuid4(), rollouts=100, seed=42)
    by_name = {
        item["intervention_name"]: item for item in report["interventions"]
    }

    baseline = by_name["no_intervention"]
    backup = by_name["activate_backup_cooling"]
    isolate = by_name["isolate_affected_system"]
    shutdown = by_name["controlled_shutdown"]

    assert (
        backup["temperature_distribution"]["maximum_temperature_c"]["mean"]
        < baseline["temperature_distribution"]["maximum_temperature_c"]["mean"]
    )
    assert (
        isolate["temperature_distribution"]["maximum_temperature_c"]["mean"]
        < baseline["temperature_distribution"]["maximum_temperature_c"]["mean"]
    )
    assert (
        shutdown["temperature_distribution"]["maximum_temperature_c"]["mean"]
        < baseline["temperature_distribution"]["maximum_temperature_c"]["mean"]
    )
    assert (
        shutdown["simulated_incident_outcomes"]["incident_count"]
        < baseline["simulated_incident_outcomes"]["incident_count"]
    )


def test_rollout_count_and_parameters_are_validated() -> None:
    with pytest.raises(ValueError, match="At least 100"):
        simulate_interventions(uuid4(), rollouts=99)
    with pytest.raises(ValueError, match="backup_available_probability"):
        simulate_interventions(uuid4(), backup_available_probability=1.1)
    with pytest.raises(ValueError, match="Thermal threshold"):
        simulate_interventions(uuid4(), initial_temperature_c=100)
    with pytest.raises(ValueError, match="workload limit"):
        simulate_interventions(uuid4(), rollouts=2_000, simulation_hours=168)
    with pytest.raises(ValueError, match="isolation_secondary_failure_probability"):
        simulate_interventions(uuid4(), isolation_secondary_failure_probability=1.1)


def test_isolation_models_configured_delayed_secondary_failure_reproducibly() -> None:
    scenario_id = uuid4()
    report = simulate_interventions(
        scenario_id,
        rollouts=100,
        seed=17,
        isolation_secondary_failure_probability=1.0,
        isolation_secondary_failure_delay_minutes=30,
        isolation_secondary_pressure_rise_kpa_per_hour=200,
    )
    repeated = simulate_interventions(
        scenario_id,
        rollouts=100,
        seed=17,
        isolation_secondary_failure_probability=1.0,
        isolation_secondary_failure_delay_minutes=30,
        isolation_secondary_pressure_rise_kpa_per_hour=200,
    )
    isolate = next(
        item
        for item in report["interventions"]
        if item["intervention_name"] == "isolate_affected_system"
    )
    assert report == repeated
    assert isolate["simulated_incident_outcomes"]["delayed_secondary_failure_count"] == 100
    assert isolate["risk_estimates"]["simulated_delayed_secondary_failure_rate"] == 1
    assert isolate["simulated_incident_outcomes"]["incident_count"] > 0
    backup = next(
        item
        for item in report["interventions"]
        if item["intervention_name"] == "activate_backup_cooling"
    )
    assert (
        isolate["risk_estimates"]["cvar_95_loss_index"]
        > backup["risk_estimates"]["cvar_95_loss_index"]
    )


def test_sequential_plans_escalate_only_when_trigger_conditions_are_met() -> None:
    report = simulate_interventions(
        uuid4(),
        rollouts=100,
        seed=3,
        backup_available_probability=0,
        equipment_failure_rate_per_hour=0,
    )
    plans = {
        item["intervention_name"]: item
        for item in report["sequential_action_plans"]
    }
    backup_plan = plans["backup_then_shutdown"]
    assert backup_plan["action_execution_counts"] == {"controlled_shutdown": 100}
    assert all(
        "controlled_shutdown" in sample["actions_taken"]
        for sample in backup_plan["rollout_samples"]
    )

    working_backup_report = simulate_interventions(
        uuid4(),
        rollouts=100,
        seed=3,
        backup_available_probability=1,
        equipment_failure_rate_per_hour=0,
        thermal_incident_threshold_c=120,
    )
    working_backup_plan = next(
        item
        for item in working_backup_report["sequential_action_plans"]
        if item["intervention_name"] == "backup_then_shutdown"
    )
    assert working_backup_plan["action_execution_counts"] == {
        "activate_backup_cooling": 100
    }


def test_simulation_api_returns_all_interventions_and_rejects_small_rollouts() -> None:
    client = TestClient(app)
    scenario_id = uuid4()
    response = client.post(
        f"/scenarios/{scenario_id}/simulate",
        json={"rollouts": 100, "seed": 34},
    )

    assert response.status_code == 200
    report = response.json()
    assert report["scenario_id"] == str(scenario_id)
    assert report["number_of_rollouts_per_intervention"] == 100
    assert len(report["interventions"]) == 4
    assert len(report["comparison_of_interventions"]) == 4
    assert "not machine learning" in report["simulation_type"]

    invalid = client.post(
        f"/scenarios/{uuid4()}/simulate",
        json={"rollouts": 10},
    )
    assert invalid.status_code == 422


def test_simulation_api_rejects_workload_over_limit_as_validation_error() -> None:
    client = TestClient(app)
    response = client.post(
        f"/scenarios/{uuid4()}/simulate",
        json={
            "rollouts": 2_000,
            "simulation_hours": 168,
            "time_step_minutes": 1,
        },
    )

    assert response.status_code == 422
    assert "workload limit" in response.text
