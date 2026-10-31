# CHRONOS-X

CHRONOS-X is a modular prototype for an AI-powered industrial catastrophe prediction, investigation, and prevention system. It provides shared data contracts, reproducible synthetic industrial data, event and scenario APIs, evidence-based analysis, counterfactual simulation, trained sensor models, and safety-gated intervention decisions. Its probabilistic and simulation outputs are illustrative, not certified operational guidance.

## Requirements

- Python 3.11 or newer
- Dependencies listed in `requirements.txt`

## Setup

Create and activate a virtual environment, then install the dependencies:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

`.env.example` documents optional environment settings; the application does not automatically load `.env`. `CHRONOS_SAFETY_ATTESTATION_KEY` must be set in the backend process environment if a trusted reviewer needs to submit signed site-safety `PASS` checks; without it, caller-supplied PASS values become UNKNOWN. In PowerShell, an ephemeral local key can be generated with `$env:CHRONOS_SAFETY_ATTESTATION_KEY = (& python -c "import secrets; print(secrets.token_hex(32))")`. Keep this secret out of source control and do not expose it to the dashboard or untrusted clients.

Generate 10,000 reproducible sequential sensor records, separate ground-truth labels, and fault events:

```powershell
python -m core.synthetic_data --records 10000 --seed 42 --output-dir data
```

The sensor feature CSV/JSON does not include ground-truth labels. Labels are written separately to `ground_truth_labels.csv` and `ground_truth_labels.json`, keyed by `reading_id`. Keep IDs, timestamps, and label files out of model feature matrices unless deliberately engineering appropriate time-window features. Generating data does not train models; use the training command below. `core.synthetic_data.iter_sensor_records()` provides an endless timestamped sensor stream for dashboard testing.

## Run

From the project root:

```powershell
uvicorn backend.main:app --reload
```

The service listens at `http://127.0.0.1:8000`. Check `http://127.0.0.1:8000/health` for the service status and `http://127.0.0.1:8000/docs` for the generated API documentation. The health endpoint returns `{"status": "ok"}` when the API is available. `POST /events` validates and ingests an industrial event, and `GET /events` lists ingested events (use `offset` and `limit` for pagination). Events are held in memory and are cleared when the API process restarts. Sensor measurements use SQLite persistence as described below.

The eight checked-in label scenarios are generated scenario fixtures, not a separately specified official adversarial test suite.

### Judge-provided runtime scenarios

Use `POST /scenarios/analyze` to submit a scenario-specific payload directly. Unlike the predefined `/scenarios/{scenario_id}/...` demo routes, this endpoint reconstructs only the submitted events and readings; it does not look up the synthetic scenario CSV/JSON files. The endpoint returns the evidence timeline, scenario-supplied causal hypotheses and relationships, applicable simulation results, all five agent outputs, a safety-gated decision, audit record, and any unsupported capabilities. A generated `scenario_id` is assigned if one is omitted.

The request accepts:

- `events`, `sensor_readings`, and descriptive `infrastructure` assets.
- Optional `causal_hypotheses` with exact `expected_event_types` and sensor comparison rules (`eq`, `ne`, `gt`, `gte`, `lt`, `lte`), plus directed `causal_relationships`.
- Optional `simulation` settings. Include explicit `thermal_incident_threshold_c` and `pressure_incident_threshold_kpa`; the service derives the initial temperature and pressure from the latest submitted readings for the selected equipment. The current simulator requires both metrics, one equipment stream, and an infrastructure asset declared as `thermal_pressure_equipment`. Conflicting latest readings or an unregistered asset type prevent simulation and force an uncertainty-gated decision.
- `available_interventions`, `operational_constraints`, and `safety_checks`. The only counterfactual action models currently implemented are `no_intervention`, `activate_backup_cooling`, `isolate_affected_system`, and `controlled_shutdown`. Other action names are reported as unsupported and cause the decision gate to retain UNKNOWN/abstain.

Unrecognized top-level scenario and infrastructure fields are retained in the response/audit but are not interpreted; they are reported under `unsupported_capabilities` and force an UNKNOWN safety gate. For example, virtual deadlines and component dependencies are not yet implemented as scheduling constraints.

Causal evidence-balance scores are not probabilities. Arbitrary natural-language descriptions are not interpreted into causal structure; provide structured evidence rules and graph relationships rather than expecting an unconfigured language model to infer them. The simulator is the existing single-stream temperature/pressure model, not an arbitrary process simulator; declaring the generic asset type does not establish real equipment compatibility. ML inference is attempted only when the submitted readings can be assembled into the trained six-feature sensor format. Missing model support, thresholds, relevant evidence, intervention physics, or site-safety confirmation is exposed and may force abstention. A selected action is never an instruction to operate real equipment.

Site safety `PASS` checks must include a reviewer ID and an HMAC-SHA256 attestation over the canonical JSON fields `check_name`, `result`, `details`, `evidence_ids`, `reviewer_id`, and `scenario_id`, with keys sorted and compact separators. The API verifies signatures using a `CHRONOS_SAFETY_ATTESTATION_KEY` of at least 32 bytes; without a configured key or valid scenario-bound signature, a caller-supplied `PASS` is downgraded to `UNKNOWN`. The response marks each site check as `VERIFIED`, `UNVERIFIED`, or `NOT_APPLICABLE`. This is a prototype trust boundary: protect the key and sign only after an authorized human review. The attestation is not a safety certification.

Minimal request shape:

```json
{
  "name": "Judge scenario",
  "description": "Scenario-specific context",
  "events": [],
  "sensor_readings": [],
  "infrastructure": [],
  "causal_hypotheses": [],
  "causal_relationships": [],
  "available_interventions": [],
  "simulation": {
    "thermal_incident_threshold_c": 110,
    "pressure_incident_threshold_kpa": 180
  }
}
```

In PowerShell, save the full judge-provided JSON as `scenario.json`, then submit it with:

```powershell
$scenario = Get-Content .\scenario.json -Raw | ConvertFrom-Json
Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8000/scenarios/analyze `
  -ContentType "application/json" `
  -Body ($scenario | ConvertTo-Json -Depth 100)
```

The response schema and interactive request are available through `http://127.0.0.1:8000/docs` and the dashboard's **Judge Scenario** tab.

`GET /scenarios/{scenario_id}/reconstruct` combines ingested events with matching generated sensor/event evidence when the datasets exist; `POST` to the same path is a compatibility alias with the same query parameters and no required body. The chronological `reconstructed_timeline` preserves every original record. `replay_timeline` collapses exact duplicates deterministically, choosing the earliest-received record (then evidence ID) as the replay representative and listing suppressed IDs; duplicates remain in the original timeline and duplicate report. Conflicting same-sensor/time readings produce explicit `UNRESOLVED_CONFLICT` alternatives with no selected value. Delays, out-of-order arrivals, gaps, provenance concerns, and reliability reasons remain visible. Missing sensor intervals use `expected_interval_seconds` (default: 8 seconds for the generated dataset). Reliability scores are explainable rule-based indicators, not model predictions. The endpoint returns `404` when it finds no evidence for the requested scenario.

### Coolant-flow and electrical sensor evidence

The checked-in `sensor_readings.csv`/`.json` remains the original generated wide-row dataset (temperature, pressure, vibration, and status signals). It contains no measured coolant-flow, voltage, or current columns. New observed sensor measurements can be submitted without rewriting those files:

- `POST /sensors`: ingest one measured value. Supply an explicit timezone-aware `timestamp`, `equipment_id`, `sensor_id`, `metric`, `value`, and `unit`. The service records `received_at` on arrival when it was not supplied; duplicate IDs return `409`. Sensor records, UTC-indexed event/arrival times, provenance, metadata, and the ingestion-time trust score/flags are persisted to `data/chronos_x.sqlite3` and remain available after backend restarts. Set `CHRONOS_SENSOR_DB_PATH` to use a different SQLite file. The SQLite database is local storage, not a multi-host database or a substitute for backups.
- `POST /sensors/synthetic/generate`: explicitly generate and ingest development/demo telemetry through the same validated sensor endpoint and SQLite store. Each sample has equipment/sensor IDs and separate temperature, pressure, vibration, coolant-flow, voltage, and current records; provenance and metadata mark every record `SYNTHETIC`. The response reports inserted and already-persisted counts. Repeating the same seed, equipment ID, scenario ID, and sample interval is idempotent for overlapping sample indices. Supply a fixed timezone-aware `start_time` as well when exact timestamps must be reproducible. This endpoint is opt-in; it does not seed data at startup and synthetic records are never counted as validated health evidence.
- `GET /sensors?scenario_id=...`: list scenario-tagged, ingested sensor measurements.
- `GET /sensors/analysis`: analyze all persisted readings for the live telemetry dashboard. Backward-aligned correlations are scoped to one equipment ID and report synthetic/stale pair counts.
- `POST /sensors/analyze`: analyze submitted `sensor_readings` and optionally supply caller-configured per-equipment `rated_current_a`.
- `GET /scenarios/{scenario_id}/sensor-analysis`: analyze the reconstructed legacy and newly ingested sensor evidence for that scenario.

Example deterministic development request (choose and retain a fixed timestamp and scenario UUID for an exactly reproducible series):

```json
{
  "sample_count": 12,
  "seed": 42,
  "sample_interval_seconds": 10,
  "start_time": "2026-01-01T00:00:00Z",
  "equipment_id": "DEMO-EQ-001",
  "scenario_id": "aeae7bdd-1e08-42d7-b787-279d30300e53"
}
```

The dashboard's **Multi-Sensor Telemetry** tab exposes the same generator behind an explicit **Generate and persist SYNTHETIC telemetry** action. It refreshes its charts from `GET /sensors/analysis`; synthetic series are labeled in chart legends. Repeated generation with the same seed/equipment/scenario/sample interval/sample indices returns already-persisted records instead of duplicating them. Use a new scenario ID to create a separate demo series. Synthetic values demonstrate plumbing and analysis behavior only; they do not represent measured plant behavior, validated fault patterns, or a safety/health assessment.

Supported measures include `coolant_flow_lpm`, `coolant_inlet_flow_lpm`, `coolant_outlet_flow_lpm`, `flow_fluctuation_lpm`, `flow_failure_status`, `voltage_v`, `voltage_fluctuation_v`, `current_a`, `current_fluctuation_a`, `power_kw`, `current_overload`, and `electrical_fault_status`, alongside existing temperature, pressure, vibration and status metrics. Flow accepts `L/min` or `L/s`; voltage accepts `V` or `mV`; current accepts `A` or `mA`; directly measured power accepts `kW` or `W`. Power is not calculated from voltage/current because power factor and phase configuration are not provided.

Analysis validates units, finite/physical values and equipment identity; preserves event time, arrival time, provenance and trust indicators; flags delayed/out-of-order/duplicate/conflicting/missing-interval observations; and aligns asynchronous streams only backward in time within one equipment and the configured tolerance. Conflicts are retained without selecting a winning value. Flow reduction is relative to at least three prior same-equipment measurements and remains a possible indicator; an equipment operating-state value can be supplied in measurement metadata as `{"operating": true|false}`. A current overload is reported only for an explicit overload status or against a caller-supplied equipment rating; these ratings are not independently verified. Historical voltage/current deviation scores require a sufficient baseline. Cross-sensor correlations are one-to-one, descriptive evidence, not causal findings.

Scenario reconstruction exposes these measurements to the Investigator, Predictor, Optimizer, Safety and Red-Team agent outputs. The Optimizer and Safety agents explicitly do not use additional sensor indicators as simulation risk scores or safety approval. The existing counterfactual simulator models only its configured thermal/pressure dynamics; scenarios containing coolant/electrical measures are marked unsupported for those dynamics, require an UNKNOWN safety check, and may abstain. No new sensor training labels or measured readings are bundled, so the existing Isolation Forest and Random Forest artifacts and their baseline metrics remain unchanged.

To optionally evaluate a combined Isolation Forest when paired measurements are available, place long-format `coolant_flow_readings.csv` and/or `electrical_readings.csv` in the training data directory. Each file requires `equipment_id,sensor_id,metric,value,unit,timestamp`; `received_at`, `reading_id`, `provenance`, and `metadata` may also be supplied. The feature pipeline validates/normalizes readings, rejects conflicting or delayed observations, and joins only prior or same-time values for the same equipment within an eight-second default tolerance. It imputes missing feature values using medians fitted on normal training rows only. The optional model is saved separately as `isolation_forest_multisensor.joblib`; the six-feature baseline artifact is not overwritten. Its report compares both models on identical held-out reading IDs and uses the existing scenario labels only for evaluation. Those labels describe generated scenarios, not verified industrial fault ground truth; resulting metrics are scenario-discrimination measurements, not field accuracy. With no valid synchronized measurement files, the report explicitly records that the comparison was not evaluated. Random Forest is not extended because no labeled target dataset for the additional modalities exists.

`GET /scenarios/{scenario_id}/causes` runs the causal investigation on that reconstructed evidence. It uses a NetworkX directed graph of hypothesized mechanisms, Pandas to summarize chronological sensor signals, and a reliability-adjusted naive Bayesian ranking of competing causes. Priors, likelihood assumptions, screening thresholds, uncertainty, evidence for and against each hypothesis, and missing evidence are included in the report. Additional screening hypotheses for possible cooling-system, electrical-system, or motor/equipment faults list supporting, contradictory, and missing sensor evidence without changing the configured Bayesian scores. These are unconfirmed screening findings; sensor status reports are not independent verification. Configurable synthetic priors and illustrative likelihoods are not calibrated plant probabilities; signals are assumed conditionally independent for the calculation, and neither graph links, correlations, nor posterior rankings establish causation.

`POST /scenarios/{scenario_id}/simulate` runs at least 100 seeded SimPy/NumPy Monte Carlo rollouts for no intervention, backup cooling, system isolation, and controlled shutdown. It also evaluates two conditional sequences: backup cooling escalates to shutdown if backup is unavailable or a configured thermal/equipment trigger occurs; isolation escalates to shutdown if a configured threshold/equipment trigger occurs. The request accepts explicit initial conditions, incident thresholds, failure rates, delays, isolation-secondary-failure probability/delay/pressure rise, duration, seed, and rollout count. Reports include primary and sequence outcomes, action execution counts, mean and maximum measures, observed simulated incident frequencies with Wilson intervals, an integrated synthetic loss index, empirical CVaR95, and a normal-approximation interval for mean loss. Isolation can trigger a delayed trapped-pressure secondary failure under a configurable, explicitly synthetic assumption. The loss index combines integrated temperature exceedance degree-hours, pressure exceedance kPa-hours scaled by 0.1, threshold/failure penalties, and intervention burden; it is not monetary loss or a calibrated safety metric. These are outputs of the documented illustrative model, not measured/calibrated real-world risks or machine-learning predictions.

The simulation assumes an illustrative one-minute time step, temperature heating/cooling rates of 4.5/4.2 degC per hour, backup cooling of 3.8 degC per hour, shutdown residual cooling of 0.8 degC per hour, Gaussian process noise, a default four-hour horizon, and defaults for intervention delays, failure rates, and thresholds. Full assumptions and request parameters are returned by the endpoint. Use facility-specific engineering validation before operational decisions.

Event and sensor provenance indicators are not cryptographic source authentication. A payload's `source_verified: true` claim is explicitly marked unattested and receives a reliability penalty; retain original evidence and use trusted external identity verification before treating a source as authenticated. No event/source signature registry is currently configured.

Train and persist the approved Isolation Forest and multi-output Random Forest from the Phase 2 datasets:

```powershell
python -m backend.ml.training --data-dir data --model-dir backend/ml/saved_models --seed 42 --horizon-readings 5
```

Isolation Forest is fitted only on the `normal` scenario; labels are used only for evaluation. The Random Forest temperature and pressure estimators use lagged/rolling sensor features and future targets shifted within each equipment stream, with a chronological target-time split. Training metrics and `.joblib` artifacts are written to the model directory. Inference rejects missing, non-numeric, or non-finite features. Do not load model artifacts from untrusted sources.

The Random Forest is one multi-output `RandomForestRegressor` for the two targets. With the checked-in synthetic data and seed 42, its chronological holdout (1,989 target rows) reported temperature MAE/RMSE/R-squared of 2.487/4.749/0.9735 and pressure MAE/RMSE/R-squared of 1.473/1.937/0.9815. Isolation Forest evaluation held out the newest 20% of normal readings plus non-training scenarios (9,000 rows): 1,911 true positives, 56 false positives, and 3,088 false negatives against synthetic fault labels. These metrics characterize this synthetic dataset only; the low fault recall is a material limitation, and a statistical outlier is not synonymous with a labeled fault.

`POST /scenarios/{scenario_id}/agents/run` runs the Investigator, Predictor, Optimizer, Safety, and Red-Team agents over reconstructed evidence and a seeded simulation. The predictor uses the trained artifacts; if artifacts or sensor history are unavailable, it reports `UNKNOWN` rather than making up a prediction. The orchestrator checks evidence references and compares the Safety agent's claims with simulation outcomes; contradictory agents are marked `UNKNOWN`, reported in `conflicts`, and force an incomplete council status. These checks are consistency checks only: agent identity is not cryptographically authenticated, so they do not prove resistance to a compromised runtime. Safety `PASS` only means the supplied simulation rollouts did not cross the configured indicators; it is not operational approval. The independent Optimizer agent ranks standalone and sequential options by mean loss plus weighted empirical CVaR95, then simulated incident count; it does not certify site feasibility. Red-Team findings are integrity indicators, not proof of deliberate forgery.

`POST /scenarios/{scenario_id}/decide` runs a seeded simulation and all five agents, then evaluates four supported standalone interventions and any simulated sequential plans against operational constraints and submitted site safety checks. Include `operational_constraints` to confirm equipment availability, shutdown authorization, operator staffing, and that model outputs were reviewed; include `safety_checks` with explicit `PASS`, `FAIL`, or `UNKNOWN` results and evidence IDs. Omitted confirmations and checks remain `UNKNOWN`. Every action in a sequence must pass its equipment authorization and staffing checks. Only candidates whose simulation evidence, all five agent outputs, site checks, model review, equipment constraints, and staffing checks are all `PASS` are eligible. Any `FAIL` or `UNKNOWN` excludes that alternative; if none qualify, the response abstains and requires human review. Model outputs are supporting evidence only and cannot override safety checks.

Eligible standalone interventions and sequential plans are ranked by OR-Tools using mean loss index + `risk_aversion_weight` × empirical CVaR95, followed by simulated incident count, mean maximum temperature, and mean maximum pressure. Sequential options must pass equipment authorization and operator checks for every action. The simulation gate rejects candidates with any observed incident or equipment failure in the rollouts; zero observed events do not establish that an intervention is safe. A candidate missing valid loss/CVaR metrics is `UNKNOWN` and cannot be ranked. The selected candidate is a synthetic decision-support comparison, not an instruction or facility approval. Every decision is appended to `data/decision_audit.jsonl`, including the request, simulation/sequence summary, agent report, decision, and evidence references. `GET /scenarios/{scenario_id}/audit` returns the persisted audit records for that scenario. Audit records are local JSONL files and should be protected and backed up according to the deployment's data-handling requirements.

## Command center dashboard

Start the FastAPI backend in one terminal and the Streamlit dashboard in another, from the project root:

```powershell
uvicorn backend.main:app --reload
streamlit run frontend/app.py
```

The dashboard defaults to `http://127.0.0.1:8000`; set `CHRONOS_API_URL` or change the sidebar field to point at another API. It loads scenario IDs and observed sensor streams from the checked-in generated CSV data, then uses the existing API for health, event history, reconstruction, causal investigation, simulations, five-agent runs, safety-gated decisions, and audit history. Use the timeline scrubber to inspect deterministic replay or the complete original evidence, and use the explicitly labeled synthetic-event form to exercise delayed/unverified evidence handling. The dashboard can compare conditional action plans, display council consistency conflicts, and show risk-adjusted rankings and decision IDs. Use the module buttons to explicitly launch simulations, agent runs, and decisions. Refresh re-queries backend data. A scenario change clears scenario-bound simulation, agent, and decision results from the current Streamlit session.

Sensor charts show generated observations; anomaly flags and Random Forest forecasts are displayed only from actual backend model responses; Monte Carlo charts are labeled simulated outcomes. Model, causal, and simulation limitations are shown alongside their outputs. The dashboard does not train models and is not certified for operational use. `streamlit` and `plotly` are included in `requirements.txt`.

For complete validation results, synthetic model limitations, and the reproducible risk sample, see [PHASE9_FINAL_REPORT.md](./PHASE9_FINAL_REPORT.md). The [provisional compliance matrix PDF](./CHRONOS_X_COMPLIANCE_MATRIX.pdf) records verified requirements and explicit gaps; it is not official PDF certification because the organizer's statement was unavailable. The five-minute demo and ten-minute technical-defense guide are in [DEMO.md](./DEMO.md).

## Test

```powershell
python -m pytest
```

## Project layout

- `backend/`: FastAPI application
- `backend/ml/`: preprocessing, training, Isolation Forest and Random Forest inference, and persisted model artifacts
- `backend/agents/`: five independently testable evidence-grounded agents and their orchestrator
- `frontend/`: Streamlit command-center dashboard and FastAPI client
- `agents/`: reserved for agent implementations
- `core/`: shared application logic, synthetic data generation, temporal reconstruction, causal investigation, and counterfactual simulation
- `models/`: shared Pydantic schemas for sensor readings, events, reconstruction, investigation, simulation, agent, and decision reports
- `data/`: local data assets
- `tests/`: automated tests
