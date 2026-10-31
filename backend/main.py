import csv
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from uuid import UUID

from fastapi import FastAPI, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse

from backend.agents.orchestrator import orchestrate_agents
from backend.decision_engine import optimize_decision
from backend.scenario_analysis import analyze_runtime_scenario
from backend.sensor_storage import (
    DuplicateSensorReadingError,
    load_sensor_readings as load_persisted_sensor_readings,
    store_sensor_reading,
)
from core.synthetic_data import generate_synthetic_telemetry
from core.sensor_fusion import (
    analyze_sensor_readings,
    sensor_records_from_reconstruction,
)
from core.causal_investigation import investigate_causes
from core.counterfactual_simulation import simulate_interventions
from core.reconstruction import reconstruct_timeline
from models import (
    CausalInvestigationReport,
    AgentOrchestrationReport,
    AgentOrchestrationRequest,
    DecisionAuditEntry,
    DecisionReport,
    DecisionRequest,
    IndustrialEvent,
    ReconstructionReport,
    SimulationReport,
    SimulationRequest,
    RuntimeScenarioReport,
    RuntimeScenarioRequest,
    SensorAnalysisRequest,
    SensorReading,
    SyntheticTelemetryRequest,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="CHRONOS-X", version="0.1.0")
_events: dict[UUID, IndustrialEvent] = {}
_events_lock = Lock()
_audit_lock = Lock()
_DECISION_AUDIT_PATH = Path(__file__).resolve().parents[1] / "data" / "decision_audit.jsonl"


@app.exception_handler(Exception)
async def handle_unexpected_exception(
    request: Request,
    exception: Exception,
) -> JSONResponse:
    logger.error(
        "Unhandled exception while handling %s %s",
        request.method,
        request.url.path,
        exc_info=(type(exception), exception, exception.__traceback__),
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


@app.get("/health", tags=["health"])
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post(
    "/events",
    response_model=IndustrialEvent,
    status_code=status.HTTP_201_CREATED,
    tags=["events"],
)
async def ingest_event(event: IndustrialEvent) -> IndustrialEvent:
    with _events_lock:
        if event.id in _events:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Event {event.id} has already been ingested",
            )
        _events[event.id] = event
    logger.info("Ingested industrial event %s from %s", event.id, event.source)
    return event


@app.get("/events", response_model=list[IndustrialEvent], tags=["events"])
async def list_events(
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=1000),
) -> list[IndustrialEvent]:
    with _events_lock:
        events = list(_events.values())
    return events[offset : offset + limit]


@app.post(
    "/sensors",
    response_model=SensorReading,
    status_code=status.HTTP_201_CREATED,
    tags=["sensors"],
)
async def ingest_sensor_reading(reading: SensorReading) -> SensorReading:
    """Ingest one measured sensor value; arrival time is server receipt time."""
    if "timestamp" not in reading.model_fields_set:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Sensor readings must provide an explicit event timestamp",
        )
    received_reading = reading
    if reading.received_at is None:
        received_reading = reading.model_copy(
            update={"received_at": datetime.now(timezone.utc)}
        )
    validation = analyze_sensor_readings(
        [received_reading.model_dump(mode="json")],
    )
    if validation["rejected_records"]:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=validation["rejected_records"][0]["reason"],
        )
    try:
        normalized_record = validation["records"][0]
        store_sensor_reading(
            received_reading,
            trust_score=normalized_record["trust_score"],
            trust_flags=tuple(normalized_record["flags"]),
        )
    except DuplicateSensorReadingError as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(error),
        ) from error
    logger.info(
        "Ingested sensor reading %s (%s) for equipment %s",
        received_reading.id,
        received_reading.metric,
        received_reading.equipment_id,
    )
    return received_reading


@app.post(
    "/sensors/synthetic/generate",
    status_code=status.HTTP_201_CREATED,
    tags=["sensors"],
)
async def generate_and_ingest_synthetic_telemetry(
    request: SyntheticTelemetryRequest,
) -> dict[str, object]:
    """Generate an explicitly synthetic demo series and ingest via the normal path."""
    records = generate_synthetic_telemetry(
        sample_count=request.sample_count,
        seed=request.seed,
        sample_interval_seconds=request.sample_interval_seconds,
        start_time=request.start_time,
        equipment_id=request.equipment_id,
        scenario_id=request.scenario_id,
    )
    inserted_by_metric: dict[str, int] = {}
    existing_by_metric: dict[str, int] = {}
    for record in records:
        reading = SensorReading.model_validate(
            {key: value for key, value in record.items() if key != "scenario_id"}
        )
        try:
            await ingest_sensor_reading(reading)
        except HTTPException as error:
            if error.status_code != status.HTTP_409_CONFLICT:
                raise
            existing_by_metric[reading.metric] = (
                existing_by_metric.get(reading.metric, 0) + 1
            )
            continue
        inserted_by_metric[reading.metric] = (
            inserted_by_metric.get(reading.metric, 0) + 1
        )
    return {
        "data_classification": "SYNTHETIC",
        "seed": request.seed,
        "sample_count": request.sample_count,
        "generated_readings": len(records),
        "inserted_readings": sum(inserted_by_metric.values()),
        "already_persisted_readings": sum(existing_by_metric.values()),
        "inserted_by_metric": inserted_by_metric,
        "already_persisted_by_metric": existing_by_metric,
        "scenario_id": str(
            request.scenario_id
            or records[0]["metadata"]["scenario_id"]
        ),
        "equipment_id": records[0]["equipment_id"],
        "provenance": "SYNTHETIC_CHRONOS_DEMO_GENERATOR",
        "note": (
            "Synthetic development/demo evidence, not measured plant telemetry "
            "and not health validation."
        ),
    }


@app.get("/sensors", response_model=list[SensorReading], tags=["sensors"])
async def list_sensor_readings(
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=1000),
    equipment_id: str | None = Query(default=None, min_length=1),
    scenario_id: UUID | None = None,
) -> list[SensorReading]:
    return load_persisted_sensor_readings(
        offset=offset,
        limit=limit,
        equipment_id=equipment_id,
        scenario_id=scenario_id,
    )


@app.post(
    "/sensors/analyze",
    tags=["sensors"],
)
def analyze_submitted_sensor_readings(
    request: SensorAnalysisRequest,
) -> dict[str, object]:
    return analyze_sensor_readings(
        [reading.model_dump(mode="json") for reading in request.sensor_readings],
        sync_tolerance_seconds=request.sync_tolerance_seconds,
        flow_reduction_fraction=request.flow_reduction_fraction,
        electrical_deviation_threshold_sigma=(
            request.electrical_deviation_threshold_sigma
        ),
        rated_current_a=request.rated_current_a,
        stale_after_seconds=request.stale_after_seconds,
    )


@app.get("/sensors/analysis", tags=["sensors"])
async def analyze_ingested_sensor_readings(
    sync_tolerance_seconds: float = Query(default=5.0, ge=0, le=86_400),
    flow_reduction_fraction: float = Query(default=0.3, gt=0, lt=1),
    electrical_deviation_threshold_sigma: float = Query(default=3.0, gt=0, le=100),
    stale_after_seconds: float = Query(default=300.0, ge=0, le=31_536_000),
    scenario_id: UUID | None = None,
) -> dict[str, object]:
    readings = load_persisted_sensor_readings(
        limit=None,
        scenario_id=scenario_id,
    )
    return analyze_sensor_readings(
        [reading.model_dump(mode="json") for reading in readings],
        sync_tolerance_seconds=sync_tolerance_seconds,
        flow_reduction_fraction=flow_reduction_fraction,
        electrical_deviation_threshold_sigma=electrical_deviation_threshold_sigma,
        stale_after_seconds=stale_after_seconds,
    )


@app.post(
    "/scenarios/analyze",
    response_model=RuntimeScenarioReport,
    tags=["runtime scenarios"],
)
def analyze_submitted_scenario(
    request: RuntimeScenarioRequest,
) -> RuntimeScenarioReport:
    """Analyze submitted judge/runtime evidence without loading demo datasets."""
    report = analyze_runtime_scenario(request)
    simulation_summary: dict[str, object] = {}
    if report.simulation:
        simulation_summary = {
            "seed": report.simulation.seed,
            "number_of_rollouts_per_intervention": (
                report.simulation.number_of_rollouts_per_intervention
            ),
            "input_parameters": report.simulation.input_parameters,
            "interventions": report.simulation.interventions,
            "sequential_action_plans": report.simulation.sequential_action_plans,
        }
    audit_entry = DecisionAuditEntry(
        decision_id=report.decision.decision_id,
        scenario_id=report.scenario_id,
        decision=report.decision,
        agent_report=report.agent_council,
        request_payload=request.model_dump(mode="json"),
        simulation_summary=simulation_summary,
    )
    _append_decision_audit(audit_entry)
    logger.info(
        "Analyzed runtime scenario %s with decision status %s",
        report.scenario_id,
        report.decision.status,
    )
    return report


@app.get(
    "/scenarios/{scenario_id}/reconstruct",
    response_model=ReconstructionReport,
    tags=["reconstruction"],
)
@app.post(
    "/scenarios/{scenario_id}/reconstruct",
    response_model=ReconstructionReport,
    tags=["reconstruction"],
)
async def reconstruct_scenario(
    scenario_id: UUID,
    expected_interval_seconds: float = Query(default=8.0, gt=0),
) -> ReconstructionReport:
    data_dir = Path(__file__).resolve().parents[1] / "data"
    events: list[IndustrialEvent] = []
    with _events_lock:
        events.extend(_events.values())

    generated_events_path = data_dir / "industrial_events.json"
    if generated_events_path.is_file():
        generated_events = json.loads(generated_events_path.read_text(encoding="utf-8"))
        events.extend(
            IndustrialEvent.model_validate(event)
            for event in generated_events
            if str((event.get("metadata") or {}).get("scenario_id")) == str(scenario_id)
        )

    sensor_rows: list[dict[str, str]] = []
    labels_path = data_dir / "ground_truth_labels.csv"
    readings_path = data_dir / "sensor_readings.csv"
    if labels_path.is_file() and readings_path.is_file():
        matching_reading_ids: set[str] = set()
        with labels_path.open(newline="", encoding="utf-8") as labels_file:
            for label in csv.DictReader(labels_file):
                if label.get("scenario_id") == str(scenario_id):
                    matching_reading_ids.add(label["reading_id"])
        if matching_reading_ids:
            with readings_path.open(newline="", encoding="utf-8") as readings_file:
                sensor_rows.extend(
                    row
                    for row in csv.DictReader(readings_file)
                    if row.get("reading_id") in matching_reading_ids
                )

    stored_sensor_readings = load_persisted_sensor_readings(
        limit=None,
        scenario_id=scenario_id,
    )
    for reading in stored_sensor_readings:
        reading_scenario_id = str((reading.metadata or {}).get("scenario_id", ""))
        if reading_scenario_id == str(scenario_id):
            sensor_rows.append(
                {
                    **reading.model_dump(mode="json"),
                    "reading_id": str(reading.id),
                }
            )

    matching_events = [
        event
        for event in events
        if str((event.metadata or {}).get("scenario_id")) == str(scenario_id)
    ]
    if not matching_events and not sensor_rows:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No evidence found for scenario {scenario_id}",
        )

    report = reconstruct_timeline(
        scenario_id,
        matching_events,
        sensor_rows,
        expected_interval_seconds=expected_interval_seconds,
    )
    return ReconstructionReport.model_validate(report)


@app.get("/scenarios/{scenario_id}/sensor-analysis", tags=["sensors"])
async def analyze_scenario_sensor_readings(
    scenario_id: UUID,
    sync_tolerance_seconds: float = Query(default=5.0, ge=0, le=86_400),
    flow_reduction_fraction: float = Query(default=0.3, gt=0, lt=1),
    electrical_deviation_threshold_sigma: float = Query(default=3.0, gt=0, le=100),
    stale_after_seconds: float = Query(default=300.0, ge=0, le=31_536_000),
) -> dict[str, object]:
    reconstruction = await reconstruct_scenario(scenario_id, 8.0)
    records = sensor_records_from_reconstruction(reconstruction.model_dump(mode="json"))
    return analyze_sensor_readings(
        records,
        sync_tolerance_seconds=sync_tolerance_seconds,
        flow_reduction_fraction=flow_reduction_fraction,
        electrical_deviation_threshold_sigma=electrical_deviation_threshold_sigma,
        stale_after_seconds=stale_after_seconds,
    )


@app.get(
    "/scenarios/{scenario_id}/causes",
    response_model=CausalInvestigationReport,
    tags=["investigation"],
)
async def investigate_scenario_causes(
    scenario_id: UUID,
    expected_interval_seconds: float = Query(default=8.0, gt=0),
) -> CausalInvestigationReport:
    reconstruction = await reconstruct_scenario(
        scenario_id,
        expected_interval_seconds,
    )
    report = investigate_causes(
        scenario_id,
        reconstruction.model_dump(),
    )
    return CausalInvestigationReport.model_validate(report)


@app.post(
    "/scenarios/{scenario_id}/simulate",
    response_model=SimulationReport,
    tags=["simulation"],
)
def simulate_scenario(
    scenario_id: UUID,
    request: SimulationRequest,
) -> SimulationReport:
    report = simulate_interventions(
        scenario_id,
        **request.model_dump(),
    )
    return SimulationReport.model_validate(report)


@app.post(
    "/scenarios/{scenario_id}/agents/run",
    response_model=AgentOrchestrationReport,
    tags=["agents"],
)
async def run_scenario_agents(
    scenario_id: UUID,
    request: AgentOrchestrationRequest,
) -> AgentOrchestrationReport:
    reconstruction = await reconstruct_scenario(scenario_id, 8.0)
    simulation = simulate_interventions(
        scenario_id,
        **request.simulation.model_dump(),
    )
    return orchestrate_agents(
        scenario_id,
        reconstruction.model_dump(mode="json"),
        simulation_report=simulation,
    )


def _append_decision_audit(entry: DecisionAuditEntry) -> None:
    _DECISION_AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _audit_lock:
        with _DECISION_AUDIT_PATH.open("a", encoding="utf-8") as audit_file:
            audit_file.write(entry.model_dump_json() + "\n")
            audit_file.flush()
            os.fsync(audit_file.fileno())


@app.post(
    "/scenarios/{scenario_id}/decide",
    response_model=DecisionReport,
    tags=["decision"],
)
async def decide_scenario(
    scenario_id: UUID,
    request: DecisionRequest,
) -> DecisionReport:
    reconstruction = await reconstruct_scenario(scenario_id, 8.0)
    simulation = simulate_interventions(
        scenario_id,
        **request.simulation.model_dump(),
    )
    agent_report = orchestrate_agents(
        scenario_id,
        reconstruction.model_dump(mode="json"),
        simulation_report=simulation,
        risk_aversion_weight=request.risk_aversion_weight,
    )
    decision = optimize_decision(
        scenario_id,
        simulation,
        agent_report,
        request.operational_constraints,
        request.safety_checks,
        request.risk_aversion_weight,
    )
    simulation_summary = {
        "seed": simulation["seed"],
        "number_of_rollouts_per_intervention": simulation[
            "number_of_rollouts_per_intervention"
        ],
        "input_parameters": simulation["input_parameters"],
        "interventions": [
            {
                "intervention_name": item["intervention_name"],
                "simulated_incident_outcomes": item["simulated_incident_outcomes"],
                "risk_estimates": item["risk_estimates"],
                "temperature_distribution": item["temperature_distribution"],
                "pressure_distribution": item["pressure_distribution"],
            }
            for item in simulation["interventions"]
        ],
        "sequential_action_plans": [
            {
                "intervention_name": item["intervention_name"],
                "simulated_incident_outcomes": item["simulated_incident_outcomes"],
                "risk_estimates": item["risk_estimates"],
                "action_execution_counts": item["action_execution_counts"],
            }
            for item in simulation["sequential_action_plans"]
        ],
    }
    audit_entry = DecisionAuditEntry(
        decision_id=decision.decision_id,
        scenario_id=scenario_id,
        decision=decision,
        agent_report=agent_report,
        request_payload=request.model_dump(mode="json"),
        simulation_summary=simulation_summary,
    )
    _append_decision_audit(audit_entry)
    logger.info(
        "Recorded decision %s for scenario %s with status %s",
        decision.decision_id,
        scenario_id,
        decision.status,
    )
    return decision


@app.get(
    "/scenarios/{scenario_id}/audit",
    response_model=list[DecisionAuditEntry],
    tags=["decision"],
)
async def get_scenario_audit(scenario_id: UUID) -> list[DecisionAuditEntry]:
    if not _DECISION_AUDIT_PATH.is_file():
        return []
    entries = []
    with _audit_lock:
        with _DECISION_AUDIT_PATH.open(encoding="utf-8") as audit_file:
            for line_number, line in enumerate(audit_file, start=1):
                if not line.strip():
                    continue
                try:
                    entry = DecisionAuditEntry.model_validate_json(line)
                except ValueError as error:
                    logger.exception(
                        "Invalid decision audit record at line %d",
                        line_number,
                    )
                    raise RuntimeError(
                        f"Decision audit log is invalid at line {line_number}"
                    ) from error
                if entry.scenario_id == scenario_id:
                    entries.append(entry)
    return entries
