import json
from math import ceil
from math import isfinite
from datetime import datetime, timezone
from typing import Literal
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SensorReading(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    sensor_id: str
    metric: str
    value: float = Field(allow_inf_nan=False)
    unit: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    received_at: datetime | None = None
    equipment_id: str | None = None
    provenance: dict[str, object] = Field(default_factory=dict)
    metadata: dict[str, object] = Field(default_factory=dict)


class SensorAnalysisRequest(BaseModel):
    sensor_readings: list[SensorReading] = Field(default_factory=list, max_length=10_000)
    sync_tolerance_seconds: float = Field(default=5.0, ge=0, le=86_400)
    stale_after_seconds: float = Field(default=300.0, ge=0, le=31_536_000)
    flow_reduction_fraction: float = Field(default=0.3, gt=0, lt=1)
    electrical_deviation_threshold_sigma: float = Field(default=3.0, gt=0, le=100)
    rated_current_a: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_current_ratings(self) -> "SensorAnalysisRequest":
        for reading in self.sensor_readings:
            if "timestamp" not in reading.model_fields_set:
                raise ValueError(
                    "Every sensor reading must provide an explicit event timestamp"
                )
            if not reading.equipment_id:
                raise ValueError("Every sensor reading must provide equipment_id")
        for equipment_id, rating in self.rated_current_a.items():
            if (
                not equipment_id.strip()
                or not isfinite(rating)
                or rating <= 0
            ):
                raise ValueError(
                    "rated_current_a entries need equipment IDs and positive values"
                )
        return self


class SyntheticTelemetryRequest(BaseModel):
    sample_count: int = Field(default=12, ge=3, le=500)
    seed: int = Field(default=42, ge=0)
    sample_interval_seconds: float = Field(default=10.0, gt=0, le=3600)
    start_time: datetime | None = None
    equipment_id: str = Field(default="DEMO-EQ-001", min_length=1, max_length=128)
    scenario_id: UUID | None = None

    @model_validator(mode="after")
    def validate_synthetic_request(self) -> "SyntheticTelemetryRequest":
        if self.start_time is not None and self.start_time.tzinfo is None:
            raise ValueError("start_time must include a timezone")
        if not self.equipment_id.strip():
            raise ValueError("equipment_id must not be blank")
        return self


class IndustrialEvent(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    received_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    source: str
    event_type: str
    description: str
    severity: Literal["low", "medium", "high", "critical"] = "low"
    provenance: dict[str, object] = Field(default_factory=dict)
    metadata: dict[str, object] = Field(default_factory=dict)


class ReconstructionReport(BaseModel):
    scenario_id: UUID
    reconstructed_timeline: list[dict[str, object]]
    replay_timeline: list[dict[str, object]] = Field(default_factory=list)
    alternative_histories: list[dict[str, object]] = Field(default_factory=list)
    delayed_events: list[dict[str, object]]
    duplicate_events: list[dict[str, object]]
    missing_events: list[dict[str, object]]
    out_of_order_events: list[dict[str, object]]
    conflicting_evidence: list[dict[str, object]]
    suspicious_sources: list[dict[str, object]]
    evidence_reliability: list[dict[str, object]]
    summary: str
    findings: list[str]


class CausalInvestigationReport(BaseModel):
    scenario_id: UUID
    possible_root_causes: list[dict[str, object]]
    ranked_hypotheses: list[dict[str, object]]
    causal_relationships: list[dict[str, object]]
    evidence_summary: dict[str, object]
    assumptions: list[str]
    summary: str


class SimulationRequest(BaseModel):
    rollouts: int = Field(default=100, ge=100, le=2_000)
    seed: int = Field(default=42, ge=0)
    initial_temperature_c: float = Field(default=84.0, ge=-20, le=500)
    initial_pressure_kpa: float = Field(default=112.0, ge=0, le=10_000)
    initial_cooling_failure: bool = True
    simulation_hours: float = Field(default=4.0, gt=0, le=168)
    time_step_minutes: float = Field(default=1.0, gt=0, le=60)
    pump_failure_rate_per_hour: float = Field(default=0.01, ge=0, le=100)
    equipment_failure_rate_per_hour: float = Field(default=0.002, ge=0, le=100)
    backup_available_probability: float = Field(default=0.9, ge=0, le=1)
    backup_activation_delay_minutes: float = Field(default=10.0, ge=0, le=1440)
    isolation_delay_minutes: float = Field(default=10.0, ge=0, le=1440)
    shutdown_delay_minutes: float = Field(default=2.0, ge=0, le=1440)
    isolation_secondary_failure_probability: float = Field(default=0.05, ge=0, le=1)
    isolation_secondary_failure_delay_minutes: float = Field(default=60.0, ge=0, le=1440)
    isolation_secondary_pressure_rise_kpa_per_hour: float = Field(default=8.0, ge=0, le=100_000)
    thermal_incident_threshold_c: float = Field(default=100.0, ge=-20, le=1000)
    pressure_incident_threshold_kpa: float = Field(default=160.0, ge=0, le=100_000)

    @model_validator(mode="after")
    def validate_simulation_thresholds(self) -> "SimulationRequest":
        if self.time_step_minutes > self.simulation_hours * 60:
            raise ValueError("time_step_minutes must not exceed the simulation duration")
        if self.thermal_incident_threshold_c <= self.initial_temperature_c:
            raise ValueError("thermal threshold must exceed initial temperature")
        if self.pressure_incident_threshold_kpa <= self.initial_pressure_kpa:
            raise ValueError("pressure threshold must exceed initial pressure")
        workload_steps = (
            self.rollouts
            * 6
            * ceil(self.simulation_hours * 60 / self.time_step_minutes)
        )
        if workload_steps > 2_000_000:
            raise ValueError(
                "Simulation exceeds the 2,000,000-step workload limit; "
                "reduce rollouts or duration"
            )
        return self


class SimulationReport(BaseModel):
    scenario_id: UUID
    seed: int
    number_of_rollouts_per_intervention: int
    interventions: list[dict[str, object]]
    sequential_action_plans: list[dict[str, object]] = Field(default_factory=list)
    sequential_plan_comparison: list[dict[str, object]] = Field(default_factory=list)
    comparison_of_interventions: list[dict[str, object]]
    simulation_assumptions: list[str]
    input_parameters: dict[str, object]
    simulation_type: str


class AgentResult(BaseModel):
    agent_name: str
    status: Literal["COMPLETED", "UNKNOWN", "FAILED"]
    summary: str
    findings: list[dict[str, object]] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    provenance: list[dict[str, object]] = Field(default_factory=list)
    outputs: dict[str, object] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)


class AgentOrchestrationReport(BaseModel):
    scenario_id: UUID
    status: Literal["COMPLETED", "PARTIAL", "FAILED"]
    agent_results: list[AgentResult]
    shared_evidence_count: int
    summary: str
    integrity_checks: list[dict[str, object]] = Field(default_factory=list)
    conflicts: list[dict[str, object]] = Field(default_factory=list)
    integrity_assurance: str = (
        "Cross-agent consistency checks only; agent identity is not cryptographically authenticated."
    )


class AgentOrchestrationRequest(BaseModel):
    simulation: SimulationRequest = Field(default_factory=SimulationRequest)


class SafetyCheckInput(BaseModel):
    check_name: str
    result: Literal["PASS", "FAIL", "UNKNOWN"] = "UNKNOWN"
    details: str = ""
    evidence_ids: list[str] = Field(default_factory=list)
    reviewer_id: str | None = Field(default=None, min_length=1, max_length=200)
    attestation: str | None = Field(default=None, min_length=64, max_length=64)


class OperationalConstraints(BaseModel):
    backup_cooling_available: bool | None = None
    system_isolation_available: bool | None = None
    controlled_shutdown_authorized: bool | None = None
    available_operators: int | None = Field(default=None, ge=0, le=100_000)
    model_outputs_reviewed: bool = False
    required_operators: dict[str, int] = Field(
        default_factory=lambda: {
            "no_intervention": 0,
            "activate_backup_cooling": 1,
            "isolate_affected_system": 1,
            "controlled_shutdown": 2,
        }
    )

    @model_validator(mode="after")
    def validate_operator_requirements(self) -> "OperationalConstraints":
        required_interventions = {
            "no_intervention",
            "activate_backup_cooling",
            "isolate_affected_system",
            "controlled_shutdown",
        }
        if set(self.required_operators) != required_interventions:
            raise ValueError(
                "required_operators must define exactly the four supported interventions"
            )
        if any(value < 0 for value in self.required_operators.values()):
            raise ValueError("required operator counts must not be negative")
        return self


class DecisionRequest(BaseModel):
    simulation: SimulationRequest = Field(default_factory=SimulationRequest)
    operational_constraints: OperationalConstraints = Field(
        default_factory=OperationalConstraints
    )
    safety_checks: list[SafetyCheckInput] = Field(default_factory=list)
    risk_aversion_weight: float = Field(default=1.0, ge=0, le=100)


class SafetyVerification(BaseModel):
    check_name: str
    result: Literal["PASS", "FAIL", "UNKNOWN"]
    details: str
    evidence_ids: list[str] = Field(default_factory=list)
    attestation_status: Literal["VERIFIED", "UNVERIFIED", "NOT_APPLICABLE"] = (
        "NOT_APPLICABLE"
    )
    reviewer_id: str | None = None


class RejectedAlternative(BaseModel):
    intervention_name: str
    reasons: list[str]
    safety_verification: list[SafetyVerification]


class RankedDecision(BaseModel):
    intervention_name: str
    rank: int
    simulated_incident_count: int
    mean_maximum_temperature_c: float
    mean_maximum_pressure_kpa: float
    mean_loss_index: float
    cvar_95_loss_index: float
    risk_adjusted_loss_index: float
    supporting_evidence: list[dict[str, object]] = Field(default_factory=list)


class DecisionReport(BaseModel):
    decision_id: UUID
    scenario_id: UUID
    status: Literal["SELECTED", "ABSTAINED"]
    selected_feasible_intervention: str | None
    ranked_feasible_decisions: list[RankedDecision]
    rejected_alternatives: list[RejectedAlternative]
    optimization_constraints: list[str]
    safety_verification_results: dict[str, list[SafetyVerification]]
    supporting_evidence: list[dict[str, object]]
    decision_explanation: str
    human_review_required: bool
    human_review_reasons: list[str]
    agent_statuses: dict[str, str]
    simulation_seed: int
    simulation_rollouts_per_intervention: int
    risk_aversion_weight: float = 1.0
    decided_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class DecisionAuditEntry(BaseModel):
    decision_id: UUID
    scenario_id: UUID
    recorded_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    decision: DecisionReport
    agent_report: AgentOrchestrationReport
    request_payload: dict[str, object]
    simulation_summary: dict[str, object]


class RuntimeSensorCondition(BaseModel):
    metric: str
    operator: Literal["eq", "ne", "gt", "gte", "lt", "lte"]
    value: float
    description: str = ""


class RuntimeAssetCondition(BaseModel):
    asset_id: str
    state_key: str
    operator: Literal["eq", "ne"]
    value: str | float | bool | None
    description: str = ""


class RuntimeCausalHypothesis(BaseModel):
    hypothesis_id: str
    title: str
    expected_event_types: list[str] = Field(default_factory=list)
    sensor_conditions: list[RuntimeSensorCondition] = Field(default_factory=list)
    asset_conditions: list[RuntimeAssetCondition] = Field(default_factory=list)


class RuntimeCausalRelationship(BaseModel):
    cause: str
    effect: str
    mechanism: str


class RuntimeAsset(BaseModel):
    model_config = ConfigDict(extra="allow")

    asset_id: str
    asset_type: str
    state: dict[str, object] = Field(default_factory=dict)


class RuntimeScenarioRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    scenario_id: UUID = Field(default_factory=uuid4)
    name: str
    description: str = ""
    events: list[IndustrialEvent] = Field(default_factory=list, max_length=10_000)
    sensor_readings: list[SensorReading] = Field(
        default_factory=list, max_length=10_000
    )
    infrastructure: list[RuntimeAsset] = Field(default_factory=list, max_length=1_000)
    causal_hypotheses: list[RuntimeCausalHypothesis] = Field(
        default_factory=list, max_length=100
    )
    causal_relationships: list[RuntimeCausalRelationship] = Field(
        default_factory=list, max_length=1_000
    )
    available_interventions: list[str] = Field(
        default_factory=list, max_length=100
    )
    simulation_equipment_id: str | None = None
    simulation: SimulationRequest | None = None
    operational_constraints: OperationalConstraints = Field(
        default_factory=OperationalConstraints
    )
    safety_checks: list[SafetyCheckInput] = Field(default_factory=list)
    risk_aversion_weight: float = Field(default=1.0, ge=0, le=100)

    @model_validator(mode="after")
    def validate_runtime_scenario(self) -> "RuntimeScenarioRequest":
        for event in self.events:
            if "timestamp" not in event.model_fields_set:
                raise ValueError("Every runtime event must provide its event timestamp")
            if "received_at" not in event.model_fields_set:
                event.received_at = event.timestamp
            if "id" not in event.model_fields_set:
                canonical_event = json.dumps(
                    event.model_dump(mode="json", exclude={"id"}),
                    sort_keys=True,
                )
                event.id = uuid5(NAMESPACE_URL, f"chronos-x:runtime-event:{canonical_event}")
            event_scenario = (event.metadata or {}).get("scenario_id")
            if event_scenario and str(event_scenario) != str(self.scenario_id):
                raise ValueError(
                    f"Event {event.id} belongs to a different scenario_id"
                )
        for reading in self.sensor_readings:
            if "timestamp" not in reading.model_fields_set:
                raise ValueError(
                    "Every runtime sensor reading must provide its event timestamp"
                )
            if "id" not in reading.model_fields_set:
                canonical_reading = json.dumps(
                    reading.model_dump(mode="json", exclude={"id"}),
                    sort_keys=True,
                )
                reading.id = uuid5(
                    NAMESPACE_URL,
                    f"chronos-x:runtime-reading:{canonical_reading}",
                )
        hypothesis_ids = [item.hypothesis_id for item in self.causal_hypotheses]
        if len(hypothesis_ids) != len(set(hypothesis_ids)):
            raise ValueError("causal_hypotheses must have unique hypothesis_id values")
        if self.simulation is not None:
            required_simulation_fields = {
                "thermal_incident_threshold_c",
                "pressure_incident_threshold_kpa",
            }
            if not required_simulation_fields.issubset(
                self.simulation.model_fields_set
            ):
                raise ValueError(
                    "Runtime scenarios must explicitly provide thermal and pressure "
                    "incident thresholds before simulation."
                )
        return self


class RuntimeScenarioReport(BaseModel):
    scenario_id: UUID
    name: str
    description: str
    infrastructure: list[RuntimeAsset]
    reconstructed_timeline: ReconstructionReport
    causal_investigation: CausalInvestigationReport
    simulation: SimulationReport | None
    agent_council: AgentOrchestrationReport
    decision: DecisionReport
    unsupported_capabilities: list[str] = Field(default_factory=list)
    assumptions_and_uncertainty: list[str] = Field(default_factory=list)


class Scenario(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    name: str
    description: str = ""
    events: list[IndustrialEvent] = Field(default_factory=list)
    sensor_readings: list[SensorReading] = Field(default_factory=list)


class FaultAlert(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    scenario_id: UUID
    title: str
    description: str = ""
    severity: Literal["low", "medium", "high", "critical"]
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Prediction(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    scenario_id: UUID
    predicted_event_type: str
    probability: float = Field(ge=0, le=1)
    rationale: str = ""
    predicted_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class AgentReport(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    agent_name: str
    scenario_id: UUID
    summary: str = ""
    findings: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class AgentResponse(AgentReport):
    pass


class RiskAssessment(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    scenario_id: UUID
    level: Literal["low", "medium", "high", "critical"]
    score: float = Field(ge=0, le=1)
    rationale: str = ""


class Decision(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    scenario_id: UUID
    action: str
    rationale: str = ""
    confidence: float = Field(ge=0, le=1)
