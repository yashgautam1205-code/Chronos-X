import hashlib
import hmac
import json
from pathlib import Path
from uuid import NAMESPACE_URL, uuid4, uuid5

import pytest
from fastapi.testclient import TestClient

import backend.main as main_module
from backend.decision_engine import optimize_decision
from models import (
    AgentOrchestrationReport,
    AgentResult,
    DecisionRequest,
    OperationalConstraints,
    SafetyCheckInput,
)

TEST_ATTESTATION_KEY = "local-test-review-key-32-bytes-minimum"


@pytest.fixture(autouse=True)
def _trusted_test_review_key(monkeypatch):
    monkeypatch.setenv("CHRONOS_SAFETY_ATTESTATION_KEY", TEST_ATTESTATION_KEY)


def _agents(scenario_id, *, predictor_status="COMPLETED"):
    return AgentOrchestrationReport(
        scenario_id=scenario_id,
        status="COMPLETED" if predictor_status == "COMPLETED" else "PARTIAL",
        shared_evidence_count=2,
        summary="test agent results",
        agent_results=[
            AgentResult(
                agent_name="investigator",
                status="COMPLETED",
                summary="causal evidence",
                outputs={
                    "ranked_hypotheses": [
                        {
                            "supporting_evidence": [
                                {
                                    "evidence": [
                                        {
                                            "evidence_id": "event-1",
                                            "reliability_score": 0.9,
                                        }
                                    ]
                                }
                            ],
                            "contradicting_evidence": [],
                        }
                    ]
                },
            ),
            AgentResult(
                agent_name="predictor",
                status=predictor_status,
                summary="sensor predictions",
                outputs={
                    "future_sensor_prediction": {
                        "predicted_temperature_c": 75.0,
                        "predicted_pressure_kpa": 105.0,
                        "input_evidence": [{"reading_id": "reading-1"}],
                    },
                    "anomaly_assessments": [],
                }
                if predictor_status == "COMPLETED"
                else {},
            ),
            AgentResult(
                agent_name="optimizer",
                status="COMPLETED",
                summary="simulation comparison",
                outputs={"decision_inputs": []},
            ),
            AgentResult(
                agent_name="safety",
                status="COMPLETED",
                summary="safety gate",
                findings=[
                    {"intervention_name": name, "result": "PASS"}
                    for name in (
                        "no_intervention",
                        "activate_backup_cooling",
                        "isolate_affected_system",
                        "controlled_shutdown",
                    )
                ],
            ),
            AgentResult(
                agent_name="red_team",
                status="COMPLETED",
                summary="integrity review",
                evidence_ids=["event-1"],
            ),
        ],
    )


def _simulation(scenario_id, incident_counts=None):
    incident_counts = incident_counts or {
        "no_intervention": 3,
        "activate_backup_cooling": 2,
        "isolate_affected_system": 1,
        "controlled_shutdown": 0,
    }
    return {
        "scenario_id": str(scenario_id),
        "seed": 12,
        "number_of_rollouts_per_intervention": 100,
        "interventions": [
            {
                "intervention_name": name,
                "number_of_rollouts": 100,
                "simulated_incident_outcomes": {
                    "incident_count": incident_counts[name],
                    "equipment_failure_count": 0,
                },
                "temperature_distribution": {
                    "maximum_temperature_c": {"mean": 80.0 + index}
                },
                "pressure_distribution": {
                    "maximum_pressure_kpa": {"mean": 110.0 + index}
                },
                "risk_estimates": {
                    "mean_loss_index": float(incident_counts[name] * 10 + index),
                    "cvar_95_loss_index": float(incident_counts[name] * 10 + index),
                },
            }
            for index, name in enumerate(
                (
                    "no_intervention",
                    "activate_backup_cooling",
                    "isolate_affected_system",
                    "controlled_shutdown",
                )
            )
        ],
    }


def _constraints() -> OperationalConstraints:
    return OperationalConstraints(
        backup_cooling_available=True,
        system_isolation_available=True,
        controlled_shutdown_authorized=True,
        available_operators=3,
        model_outputs_reviewed=True,
    )


def _site_checks(scenario_id, result="PASS"):
    check_data = {
        "check_name": "site_safety_review",
        "result": result,
        "details": "Qualified site review recorded for this test.",
        "evidence_ids": ["review-1"],
        "reviewer_id": "test-reviewer",
    }
    signed_data = json.dumps(
        {
            "check_name": check_data["check_name"],
            "result": check_data["result"],
            "details": check_data["details"],
            "evidence_ids": check_data["evidence_ids"],
            "reviewer_id": check_data["reviewer_id"],
            "scenario_id": str(scenario_id),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    check_data["attestation"] = hmac.new(
        TEST_ATTESTATION_KEY.encode("utf-8"),
        signed_data,
        hashlib.sha256,
    ).hexdigest()
    return [
        SafetyCheckInput(**check_data)
    ]


def test_optimizer_selects_best_candidate_after_all_safety_checks_pass() -> None:
    scenario_id = uuid4()

    report = optimize_decision(
        scenario_id,
        _simulation(
            scenario_id,
            {
                "no_intervention": 0,
                "activate_backup_cooling": 0,
                "isolate_affected_system": 0,
                "controlled_shutdown": 0,
            },
        ),
        _agents(scenario_id),
        _constraints(),
        _site_checks(scenario_id),
    )

    assert report.status == "SELECTED"
    assert report.selected_feasible_intervention == "no_intervention"
    assert report.ranked_feasible_decisions[0].rank == 1
    assert len(report.ranked_feasible_decisions) == 4
    assert report.human_review_required is False
    assert all(
        check.result == "PASS"
        for checks in report.safety_verification_results.values()
        for check in checks
    )
    assert report.supporting_evidence
    assert report.simulation_rollouts_per_intervention == 100


def test_unknown_safety_conditions_abstain_and_require_human_review() -> None:
    scenario_id = uuid4()

    report = optimize_decision(
        scenario_id,
        _simulation(
            scenario_id,
            {
                "no_intervention": 0,
                "activate_backup_cooling": 0,
                "isolate_affected_system": 0,
                "controlled_shutdown": 0,
            },
        ),
        _agents(scenario_id),
        OperationalConstraints(
            backup_cooling_available=True,
            system_isolation_available=True,
            controlled_shutdown_authorized=True,
            available_operators=3,
            model_outputs_reviewed=True,
        ),
        [],
    )

    assert report.status == "ABSTAINED"
    assert report.selected_feasible_intervention is None
    assert report.human_review_required is True
    assert report.human_review_reasons
    assert all(
        any(check.result == "UNKNOWN" for check in checks)
        for checks in report.safety_verification_results.values()
    )
    assert report.ranked_feasible_decisions == []


def test_unattested_caller_pass_is_downgraded_to_unknown() -> None:
    scenario_id = uuid4()
    report = optimize_decision(
        scenario_id,
        _simulation(
            scenario_id,
            {
                "no_intervention": 0,
                "activate_backup_cooling": 0,
                "isolate_affected_system": 0,
                "controlled_shutdown": 0,
            },
        ),
        _agents(scenario_id),
        _constraints(),
        [
            SafetyCheckInput(
                check_name="site_safety_review",
                result="PASS",
                details="Unverified caller declaration.",
                evidence_ids=["review-1"],
            )
        ],
    )

    assert report.status == "ABSTAINED"
    assert report.selected_feasible_intervention is None
    assert report.human_review_required is True
    assert all(
        any(
            check.check_name == "site_safety_review"
            and check.result == "UNKNOWN"
            and check.attestation_status == "UNVERIFIED"
            and "downgraded to UNKNOWN" in check.details
            for check in checks
        )
        for checks in report.safety_verification_results.values()
    )


def test_valid_attested_review_is_machine_readable_in_safety_output() -> None:
    scenario_id = uuid4()
    report = optimize_decision(
        scenario_id,
        _simulation(
            scenario_id,
            {
                "no_intervention": 0,
                "activate_backup_cooling": 0,
                "isolate_affected_system": 0,
                "controlled_shutdown": 0,
            },
        ),
        _agents(scenario_id),
        _constraints(),
        _site_checks(scenario_id),
    )

    site_check = next(
        check
        for check in report.safety_verification_results["no_intervention"]
        if check.check_name == "site_safety_review"
    )
    assert site_check.result == "PASS"
    assert site_check.attestation_status == "VERIFIED"
    assert site_check.reviewer_id == "test-reviewer"


def test_safety_attestation_cannot_be_replayed_for_another_scenario() -> None:
    signed_scenario_id = uuid4()
    report_scenario_id = uuid4()
    report = optimize_decision(
        report_scenario_id,
        _simulation(
            report_scenario_id,
            {
                "no_intervention": 0,
                "activate_backup_cooling": 0,
                "isolate_affected_system": 0,
                "controlled_shutdown": 0,
            },
        ),
        _agents(report_scenario_id),
        _constraints(),
        _site_checks(signed_scenario_id),
    )

    site_check = next(
        check
        for check in report.safety_verification_results["no_intervention"]
        if check.check_name == "site_safety_review"
    )
    assert site_check.result == "UNKNOWN"
    assert site_check.attestation_status == "UNVERIFIED"


def test_failed_constraints_reject_only_unsafe_alternative() -> None:
    scenario_id = uuid4()

    report = optimize_decision(
        scenario_id,
        _simulation(scenario_id),
        _agents(scenario_id),
        OperationalConstraints(
            backup_cooling_available=False,
            system_isolation_available=True,
            controlled_shutdown_authorized=True,
            available_operators=3,
            model_outputs_reviewed=True,
        ),
        _site_checks(scenario_id),
    )

    rejected_backup = next(
        item
        for item in report.rejected_alternatives
        if item.intervention_name == "activate_backup_cooling"
    )
    assert any("unavailable" in reason for reason in rejected_backup.reasons)
    assert report.selected_feasible_intervention == "controlled_shutdown"
    assert report.safety_verification_results["activate_backup_cooling"][
        -2
    ].result == "FAIL"


def test_optimizer_uses_cvar_weight_when_ranking_safe_candidates() -> None:
    scenario_id = uuid4()
    simulation = _simulation(
        scenario_id,
        {
            "no_intervention": 0,
            "activate_backup_cooling": 0,
            "isolate_affected_system": 0,
            "controlled_shutdown": 0,
        },
    )
    by_name = {item["intervention_name"]: item for item in simulation["interventions"]}
    for intervention in by_name.values():
        intervention["risk_estimates"]["mean_loss_index"] = 5.0
        intervention["risk_estimates"]["cvar_95_loss_index"] = 10.0
    by_name["no_intervention"]["risk_estimates"]["cvar_95_loss_index"] = 30.0
    by_name["controlled_shutdown"]["risk_estimates"]["cvar_95_loss_index"] = 6.0

    risk_neutral = optimize_decision(
        scenario_id,
        simulation,
        _agents(scenario_id),
        _constraints(),
        _site_checks(scenario_id),
        risk_aversion_weight=0,
    )
    risk_averse = optimize_decision(
        scenario_id,
        simulation,
        _agents(scenario_id),
        _constraints(),
        _site_checks(scenario_id),
        risk_aversion_weight=1,
    )

    assert risk_neutral.selected_feasible_intervention == "no_intervention"
    assert risk_averse.selected_feasible_intervention == "controlled_shutdown"
    assert (
        risk_averse.ranked_feasible_decisions[0].risk_adjusted_loss_index
        == 11.0
    )


def test_missing_loss_risk_metrics_are_unknown_and_not_ranked() -> None:
    scenario_id = uuid4()
    simulation = _simulation(
        scenario_id,
        {
            "no_intervention": 0,
            "activate_backup_cooling": 0,
            "isolate_affected_system": 0,
            "controlled_shutdown": 0,
        },
    )
    for intervention in simulation["interventions"]:
        intervention.pop("risk_estimates")

    report = optimize_decision(
        scenario_id,
        simulation,
        _agents(scenario_id),
        _constraints(),
        _site_checks(scenario_id),
    )

    assert report.status == "ABSTAINED"
    assert report.ranked_feasible_decisions == []
    assert all(
        checks[1].result == "UNKNOWN"
        for checks in report.safety_verification_results.values()
    )


def test_optimizer_includes_sequential_plan_and_checks_each_required_action() -> None:
    scenario_id = uuid4()
    simulation = _simulation(
        scenario_id,
        {
            "no_intervention": 1,
            "activate_backup_cooling": 1,
            "isolate_affected_system": 1,
            "controlled_shutdown": 1,
        },
    )
    simulation["sequential_action_plans"] = [
        {
            "intervention_name": "backup_then_shutdown",
            "number_of_rollouts": 100,
            "simulated_incident_outcomes": {
                "incident_count": 0,
                "equipment_failure_count": 0,
            },
            "temperature_distribution": {
                "maximum_temperature_c": {"mean": 78.0}
            },
            "pressure_distribution": {
                "maximum_pressure_kpa": {"mean": 105.0}
            },
            "risk_estimates": {
                "mean_loss_index": 1.0,
                "cvar_95_loss_index": 2.0,
            },
        }
    ]

    report = optimize_decision(
        scenario_id,
        simulation,
        _agents(scenario_id),
        _constraints(),
        _site_checks(scenario_id),
    )

    assert report.selected_feasible_intervention == "backup_then_shutdown"
    checks = report.safety_verification_results["backup_then_shutdown"]
    assert any("backup_cooling_available" in check.check_name for check in checks)
    assert any("controlled_shutdown_authorized" in check.check_name for check in checks)
    assert all(check.result == "PASS" for check in checks)


def test_optimizer_rejects_sequence_when_later_action_lacks_staffing() -> None:
    scenario_id = uuid4()
    simulation = _simulation(
        scenario_id,
        {
            "no_intervention": 1,
            "activate_backup_cooling": 1,
            "isolate_affected_system": 1,
            "controlled_shutdown": 1,
        },
    )
    simulation["sequential_action_plans"] = [
        {
            "intervention_name": "backup_then_shutdown",
            "number_of_rollouts": 100,
            "simulated_incident_outcomes": {
                "incident_count": 0,
                "equipment_failure_count": 0,
            },
            "temperature_distribution": {
                "maximum_temperature_c": {"mean": 78.0}
            },
            "pressure_distribution": {
                "maximum_pressure_kpa": {"mean": 105.0}
            },
            "risk_estimates": {
                "mean_loss_index": 1.0,
                "cvar_95_loss_index": 2.0,
            },
        }
    ]
    constraints = OperationalConstraints(
        backup_cooling_available=True,
        system_isolation_available=True,
        controlled_shutdown_authorized=True,
        available_operators=1,
        model_outputs_reviewed=True,
    )

    report = optimize_decision(
        scenario_id,
        simulation,
        _agents(scenario_id),
        constraints,
        _site_checks(scenario_id),
    )

    rejected = next(
        item
        for item in report.rejected_alternatives
        if item.intervention_name == "backup_then_shutdown"
    )
    assert any("controlled_shutdown" in reason and "required" in reason for reason in rejected.reasons)
    assert report.selected_feasible_intervention is None
    assert report.status == "ABSTAINED"


def test_failed_or_unknown_agent_inputs_block_automatic_selection() -> None:
    scenario_id = uuid4()
    agents = _agents(scenario_id, predictor_status="UNKNOWN")

    report = optimize_decision(
        scenario_id,
        _simulation(
            scenario_id,
            {
                "no_intervention": 0,
                "activate_backup_cooling": 0,
                "isolate_affected_system": 0,
                "controlled_shutdown": 0,
            },
        ),
        agents,
        _constraints(),
        _site_checks(scenario_id),
    )

    assert report.status == "ABSTAINED"
    assert report.human_review_required
    assert all(
        any(
            check.check_name == "five_agent_inputs" and check.result == "UNKNOWN"
            for check in checks
        )
        for checks in report.safety_verification_results.values()
    )


def test_decision_and_audit_endpoints_persist_explainable_record(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        main_module,
        "_DECISION_AUDIT_PATH",
        tmp_path / "decision_audit.jsonl",
    )
    scenario_id = uuid5(
        NAMESPACE_URL,
        "chronos-x:42:scenario:normal",
    )
    request = DecisionRequest(
        simulation={
            "rollouts": 100,
            "seed": 23,
            "initial_temperature_c": 72,
            "initial_pressure_kpa": 100,
            "initial_cooling_failure": False,
            "pump_failure_rate_per_hour": 0,
            "equipment_failure_rate_per_hour": 0,
            "thermal_incident_threshold_c": 1000,
            "pressure_incident_threshold_kpa": 100_000,
        },
        operational_constraints={
            "backup_cooling_available": True,
            "system_isolation_available": True,
            "controlled_shutdown_authorized": True,
            "available_operators": 3,
            "model_outputs_reviewed": True,
        },
        safety_checks=[
            {
                "check_name": "site_safety_review",
                "result": "PASS",
                "details": "Site review complete.",
                "evidence_ids": ["review-api"],
            }
        ],
    )
    client = TestClient(main_module.app)

    response = client.post(
        f"/scenarios/{scenario_id}/decide",
        json=request.model_dump(mode="json"),
    )

    assert response.status_code == 200, response.text
    decision = response.json()
    assert decision["status"] in {"SELECTED", "ABSTAINED"}
    assert Path(main_module._DECISION_AUDIT_PATH).is_file()
    audit = client.get(f"/scenarios/{scenario_id}/audit")
    assert audit.status_code == 200
    assert len(audit.json()) == 1
    entry = audit.json()[0]
    assert entry["decision"]["decision_id"] == decision["decision_id"]
    assert len(entry["agent_report"]["agent_results"]) == 5
    assert entry["simulation_summary"]["number_of_rollouts_per_intervention"] == 100
