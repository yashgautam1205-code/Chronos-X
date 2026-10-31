# CHRONOS-X Phase 9 Validation and Readiness Report

**Validation date:** 2026-10-31 (follow-up hardening validation)  
**Project status:** Synthetic decision-support prototype; not approved to control industrial equipment.

**Specification caveat:** The official hackathon problem-statement PDF and organizer test harness were not present in the workspace. This report verifies only the available source code and requirements supplied in the conversation; exact official compliance and a current official-rubric score cannot be established.

## Executive summary

The project passed its latest full automated suite: **68 passed, 0 failed**. One non-failing Starlette `TestClient` deprecation warning remains. Follow-up tests verify order-independent abstention on conflicting latest sensor values, rejection of unsupported simulator equipment and schema fields, UNKNOWN Safety results when no candidates exist, request-time simulation workload validation, scenario-bound HMAC review attestations, explicit hidden-confounder/unmodeled-cause alternatives, unattested sensor-provenance handling, and an integrated synthetic Paradox subset that fails closed and records an audit. `pip check`, Python compilation, and targeted Pylance diagnostics were clean. The checked-in synthetic datasets reproduce byte-for-byte from the documented 10,000-record, seed-42 generator command. Both persisted scikit-learn artifacts load and produce outputs; earlier fixed-seed training validation reproduced the saved metrics exactly. The backend and dashboard started locally; Streamlit AppTest rendered the dashboard, including the new runtime-scenario tab, with no exceptions (15 tabs including nested test tabs). Existing service and API checks are recorded below.

The implementation follow-up added duplicate-collapsed deterministic replay while retaining raw evidence, unresolved conflict alternatives, a configurable delayed isolation-secondary-failure mechanism, two conditional sequential plans, integrated synthetic loss/CVaR95 reporting, risk-aware OR-Tools selection, agent evidence/safety consistency checks, and dashboard replay/event-injection controls. This update adds `POST /scenarios/analyze` and a **Judge Scenario** dashboard tab to accept scenario-specific evidence, infrastructure state, causal rules/relationships, constraints, and (when compatible) simulator settings without loading the predefined demo dataset. It preserves abstention when the scenario lacks supported action physics, thresholds, or safety evidence. The endpoint and client are covered by tests. Remaining limits are substantial: this is a structured-input adapter, not arbitrary natural-language understanding or a general industrial simulator; the Isolation Forest missed 3,088 of 4,999 labeled fault records in its held-out evaluation; Random Forest R-squared was negative for a held-out `sensor_malfunction` scenario; hidden-confounder adjustment, cryptographic trust, deadline/time-dependent resource scheduling, scale validation and official Paradox Protocol behavior remain unavailable or incomplete. All metrics are synthetic-only.

## Completed features

- Modular Python packages for shared Pydantic schemas, synthetic data, temporal reconstruction, causal investigation, SimPy/Monte Carlo simulation, ML training and inference, five agents, decision optimization, FastAPI, and Streamlit.
- FastAPI routes for health, event ingestion/listing, reconstruction, causes, simulation, five-agent orchestration, safety-gated decisions, runtime scenario analysis (`POST /scenarios/analyze`), and decision audit retrieval.
- Runtime scenario support with stable evidence IDs when omitted, per-scenario reconstruction, user-supplied causal hypotheses/rules and NetworkX relationships, infrastructure-state checks, optional seedable simulation from submitted temperature/pressure readings, model-compatible feature assembly, agent orchestration, decision abstention, and audit persistence.
- Reproducible sensor and event datasets, with ground-truth labels stored separately from sensor features.
- Saved Isolation Forest and Random Forest Regressor artifacts in `backend/ml/saved_models/`.
- Five evidence-grounded agents: Investigator, Predictor, Optimizer, Safety, and Red-Team.
- OR-Tools intervention selection with PASS/FAIL/UNKNOWN gates; UNKNOWN safety evidence is not treated as PASS and requires abstention/human review.
- OR-Tools ranking for four standalone interventions and two simulated conditional sequences, with mean synthetic loss + configurable weighted empirical CVaR95, followed by incident count and peak measures.
- Agent evidence-reference and Safety-vs-simulation consistency checks. Contradictory Safety claims are reported, downgraded to UNKNOWN, and force a PARTIAL council status. Agent identity is not cryptographically authenticated.
- Streamlit command center with the existing dashboard sections plus a Judge Scenario tab, deterministic replay scrubbing, raw-vs-deduplicated evidence views, synthetic event injection, actual sensor charts, API integration, model outputs, sequential simulation comparisons, agent-conflict display, and risk-ranked decision/audit views.
- Final validation in this report; five-minute demo and ten-minute technical-defense guide in [DEMO.md](./DEMO.md). A provisional matrix is also available as [CHRONOS_X_COMPLIANCE_MATRIX.pdf](./CHRONOS_X_COMPLIANCE_MATRIX.pdf).

## Validation results

### Project, dependencies, imports, and services

| Check | Result |
|---|---|
| Automated tests | `68 passed, 0 failed` (`.venv\Scripts\python.exe -m pytest -q`) |
| Dependency consistency | `pip check`: no broken requirements |
| Pylance workspace diagnostics | No diagnostics after removing unused imports/variables |
| FastAPI startup and health | Started; `GET /health` returned `200 {"status":"ok"}` |
| Streamlit startup | Started; dashboard returned HTTP 200 and Streamlit health `ok` |
| Dashboard rendering | Streamlit AppTest rendered all dashboard tabs, including Judge Scenario, with no exceptions; nested tabs yield 15 total |
| Runtime judge-scenario endpoint | TestClient verified submitted-only evidence, deterministic IDs, dynamic hypothesis/evidence mapping, optional simulation, five agent outputs, unknown-action abstention, and audit persistence |
| Dashboard simulation action | Ran four-intervention simulation and rendered six charts with no exceptions |
| API integration | Event ingest `201`; reconstruction and causes `200`; simulation and agent run `200`; decision and audit `200` |
| Five-agent results | The agent test suite verified all five actual outputs; live event-only smoke correctly returned `PARTIAL` because sensor history for the Predictor was unavailable. A deliberately contradictory Safety agent was quarantined by test. |
| Safety attestation | Unattested caller `PASS` values are downgraded to `UNKNOWN`; a valid HMAC is bound to the check content and scenario ID. Missing/invalid attestation produces abstention. |
| Safety abstention | Omitted safety confirmations produced `ABSTAINED`, human review required, and one retrievable audit entry |
| Dataset reproduction | Regenerated all five 10,000-record CSV/JSON outputs with seed 42; each matched its checked-in counterpart byte-for-byte |
| Model training reproduction | Training with seed 42 reproduced both saved evaluation metric objects exactly |

The API integration used the generated `cooling_pump_failure` scenario. Reconstruction returned 7,502 timeline items after the test event was ingested; causal investigation returned seven ranked hypotheses. Simulation returned 100 rollouts for each of the four interventions.

Adversarial evidence checks cover delayed, duplicate, out-of-order, conflicting cooling-state, suspicious-source, and contradictory Safety-agent evidence. `reconstructed_timeline` retains originals; `replay_timeline` deterministically collapses duplicate records and references suppressed IDs. Conflicting observations have unresolved alternatives rather than a selected winner.

### Reproducible simulation risk sample

The following is a single fixed run using seed 42, 100 rollouts per candidate, and the checked-in default simulation parameters. These are outputs of the synthetic model, not calibrated industrial probabilities or losses. The loss index is a synthetic comparison unit.

| Candidate | Simulated incidents | Delayed secondary failures | Mean loss index | Empirical CVaR95 | Mean-loss 95% normal interval |
|---|---:|---:|---:|---:|---:|
| No intervention | 100 | 0 | 31.4754 | 50.4951 | 30.6153–32.3355 |
| Backup cooling | 18 | 0 | 7.1694 | 32.6009 | 4.9716–9.3671 |
| Isolation | 3 | 7 | 4.6000 | 16.0000 | 3.9279–5.2721 |
| Controlled shutdown | 3 | 0 | 8.6000 | 20.0000 | 7.9279–9.2721 |
| Backup then conditional shutdown | 3 | 0 | 10.6000 | 22.0000 | 9.9279–11.2721 |
| Isolation then conditional shutdown | 3 | 7 | 12.6000 | 24.0000 | 11.9279–13.2721 |

The default decision engine rejects candidates with any observed simulated threshold crossing or equipment failure, so lower mean/CVaR alone does not clear an action as safe. This sample demonstrates that risk metrics can distinguish candidates; it does not demonstrate plant safety.

### Local sequential API timing

Measured with FastAPI `TestClient` on this development environment. Requests were sequential, not concurrent; these are smoke measurements, not a capacity benchmark.

| Endpoint | Requests | Median | p95 |
|---|---:|---:|---:|
| `GET /health` | 30 | 3.39 ms | 4.05 ms |
| `GET /scenarios/{id}/reconstruct` | 5 | 134.29 ms | 198.96 ms |
| `GET /scenarios/{id}/causes` | 3 | 350.90 ms | 398.07 ms |
| `POST /scenarios/{id}/simulate` (100 rollouts per intervention) | 3 | 461.74 ms | 470.25 ms |

### Isolation Forest

The saved training report records training on 1,000 normal readings; the newest 250 normal readings were held out. Evaluation used 9,000 records, including 4,999 synthetic fault labels.

| Measure | Result |
|---|---:|
| True positives | 1,911 |
| False positives | 56 |
| False negatives | 3,088 |
| Precision | 97.15% |
| Fault recall | 38.23% |

The saved model loaded successfully, scored a normal record as non-anomalous, and flagged a synthetic multiple-fault record and an injected extreme record. Missing features, non-numeric inputs, and NaN values were rejected. In this probe, the abnormal and extreme readings had equal anomaly scores (`0.134301`); an anomaly score is not a calibrated severity scale and should not be assumed to increase monotonically with physical extremity.

### Random Forest Regressor

The persisted artifact is a multi-output `RandomForestRegressor` with a five-reading horizon. Its saved chronological evaluation is:

| Target | Samples | MAE | RMSE | R-squared |
|---|---:|---:|---:|---:|
| Future temperature | 1,989 | 2.487428 °C | 4.749450 °C | 0.973495 |
| Future pressure | 1,989 | 1.473422 kPa | 1.936725 kPa | 0.981540 |

The chronological split recorded a training target end of `2026-01-01T02:13:30+00:00` and test target start of `2026-01-01T02:13:31+00:00`. Feature construction excludes IDs, labels, and future targets and shifts targets within equipment streams.

A separate scenario holdout was measured by training a fresh model on the other seven scenarios and testing only on the 1,250 `sensor_malfunction` readings (1,243 future targets):

| Target | MAE | RMSE | R-squared |
|---|---:|---:|---:|
| Future temperature | 1.285445 °C | 1.599753 °C | -0.030707 |
| Future pressure | 0.965962 kPa | 1.465593 kPa | -0.104701 |

The negative R-squared results show weak generalization for this scenario against a mean baseline, despite favorable chronological metrics. The scenario contains a synthetic sensor-flatline pattern. Saved-model inference produced finite temperature/pressure forecasts. Missing, non-numeric, non-finite, too-short, or non-chronological histories are rejected.

## Failed tests and warnings

- **Repository test failures in final run:** none (68 passed).
- An intermediate full-suite run after adding the agent evidence check failed one legitimate-agent test because Predictor evidence IDs refer to source sensor rows while the reconstruction timeline stores row/metric IDs. The lineage check was corrected to recognize the original reading ID; the complete suite then passed.
- An initial one-off validation snippet stopped before held-out evaluation because the temporary script incorrectly unpacked a function object. The snippet was corrected and rerun; the held-out metrics above come from that successful run. This was not a repository test or application failure.
- Pytest emitted one `StarletteDeprecationWarning` because the installed Starlette `TestClient` integration uses `httpx`. It did not fail tests or affect endpoint behavior.
- Streamlit AppTest emitted its expected bare-mode `ScriptRunContext` warning; dashboard execution and assertions succeeded.

## Known limitations and remaining issues

1. **Synthetic-only evidence:** neither model, Bayesian assumptions, nor SimPy rates/outcomes are calibrated to any real facility or industrial operating envelope.
2. **Isolation Forest recall:** 38.23% on the recorded synthetic fault evaluation; a low anomaly score is not evidence that an event is safe.
3. **Scenario generalization:** the held-out `sensor_malfunction` Random Forest test had negative R-squared for both targets. The headline chronological scores are not held-out-scenario performance.
4. **No operational authorization:** even a decision status of `SELECTED` is not an instruction or facility approval. Simulation PASS is only a rule result for the specified simulated rollouts; zero observed failures is not proof of safety.
5. **Evidence and model uncertainty:** causal posterior values are conditional on configured hypotheses and likelihood assumptions; correlation and graph links do not prove causation. Red-Team signals do not prove malicious manipulation.
6. **Agent integrity:** the orchestrator detects certain evidence-reference and simulation-claim inconsistencies; it does not cryptographically authenticate agent code or detect every plausible forged claim.
7. **Risk/planning assumptions:** loss weights are illustrative; only two conditional plans exist. No action durations, deadlines, cooldowns, shared time-dependent resource scheduling, or exhaustive-search budget is implemented.
8. **Causal confounding:** no hidden-confounder adjustment or validated structural causal model is implemented.
9. **Persistence and deployment:** ingested events are in memory and disappear on API restart. Decision audit is a local JSONL file; no multi-process locking strategy, access control, database retention, or production backup policy was validated.
10. **Performance scope:** only sequential local smoke timings and a 100-rollout risk sample were measured. No concurrent-load, million-event/50,000-component, long-duration, deployment, or scale corruption benchmark was performed.
11. **Official evaluation unavailable:** the PDF, organizer simulator, exact five-disruption Paradox Protocol, and hidden tests were not available. The final official compliance score cannot be estimated from evidence here.
12. **Human review is mandatory:** the dashboard is a prototype for demonstration and must not control equipment, override site procedures, or replace qualified industrial safety personnel.
13. **Safety attestation is configuration-dependent:** a caller-supplied site `PASS` is accepted only with a valid scenario-bound HMAC and a configured secret. This does not independently validate the reviewer, evidence, or facility procedure; key custody and reviewer authorization remain external responsibilities.
14. **Unseen equipment is fail-closed:** runtime simulation is skipped unless the selected infrastructure asset is explicitly declared as `thermal_pressure_equipment`; this generic model declaration is not proof of compatibility with actual equipment physics.

No failing test or startup issue remains in the verified local paths. Official compliance, production readiness, real-world model validity, and operational safety certification remain unverified and explicitly out of scope.

## Reproducible setup and run commands

Run these commands from the repository root in PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip check
python -m core.synthetic_data --records 10000 --seed 42 --output-dir data
python -m backend.ml.training --data-dir data --model-dir backend/ml/saved_models --seed 42 --horizon-readings 5 --test-fraction 0.2
python -m pytest -q
```

In separate terminals, start the services:

```powershell
.\.venv\Scripts\Activate.ps1
uvicorn backend.main:app --reload
```

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run frontend/app.py
```

Open `http://127.0.0.1:8501` for the dashboard and `http://127.0.0.1:8000/docs` for the API documentation. The Streamlit sidebar accepts a different API URL if needed.
