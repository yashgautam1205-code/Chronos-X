# CHRONOS-X Five-Minute Demo

## Before the demo

Install dependencies, generate the reproducible dataset, and ensure the saved model artifacts exist. These setup steps can take longer than five minutes and should be done ahead of time:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m core.synthetic_data --records 10000 --seed 42 --output-dir data
python -m backend.ml.training --data-dir data --model-dir backend/ml/saved_models --seed 42 --horizon-readings 5 --test-fraction 0.2
```

Open two PowerShell terminals at the project root and start the services:

```powershell
.\.venv\Scripts\Activate.ps1
uvicorn backend.main:app --reload
```

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run frontend/app.py
```

Open `http://127.0.0.1:8501`. Select **Cooling Pump Failure** in the sidebar. The dashboard defaults to the local API at `http://127.0.0.1:8000`.

For a judge-provided scenario, open the **Judge Scenario** tab and submit its structured JSON, or save the payload as `scenario.json` and use the `POST /scenarios/analyze` PowerShell command in the README. This route does not use the predefined demo datasets. It supports the existing temperature/pressure simulator and four built-in interventions only; missing thresholds, unsupported action physics, and unknown safety checks are reported and can force abstention. It does not interpret arbitrary natural-language narratives.

## Demo runbook

| Time | Show |
|---|---|
| 0:00–0:35 | **System overview:** API health, selected scenario, evidence alert count, equipment streams, and observed equipment status. Point out that the displayed sensor data is generated observation data. |
| 0:35–1:10 | **Sensor monitoring:** show temperature, pressure, and vibration traces for the selected equipment. Explain that these are observed synthetic sensor values, not forecasts. |
| 1:10–1:45 | **Temporal reconstruction:** scrub the deterministic replay, switch to the full evidence timeline, and use **Inject a synthetic test event** with a 10-second arrival delay. Show the delayed/unverified evidence and the unresolved conflicting-value alternatives; emphasize that raw evidence is preserved. |
| 1:45–2:20 | **Causal investigation:** show ranked hypotheses, evidence for and against a cause, missing evidence, and the configured causal graph. State that posterior rankings do not prove causation and hidden-confounder adjustment is not implemented. |
| 2:20–3:10 | **Five-agent council:** click **Run five-agent council**. Show Investigator, Predictor, Optimizer, Safety, and Red-Team statuses plus evidence-reference/simulation consistency checks. In the Predictor panel, distinguish Isolation Forest flags and Random Forest forecast values from measured sensor readings; mention synthetic-data limitations. |
| 3:10–4:00 | **Counterfactual simulation:** click **Run four-intervention simulation**. Show 100 rollouts per standalone and sequential plan, seed, conditional shutdown triggers, loss/CVaR95, and assumptions. Explain that these are simulated futures and illustrative loss units, not actual incidents or calibrated risks. |
| 4:00–4:40 | **Decision center:** keep the site review and equipment confirmations at `UNKNOWN`, then click **Evaluate safety-gated decision**. Show abstention and human-review reasons; point out the risk-aversion control and that UNKNOWN must never be treated as PASS. |
| 4:40–5:00 | **Audit log:** show the persisted decision certificate, weighted loss/CVaR ranking, safety checks, agent conflicts, and event history. Close by noting local audit storage, unauthenticated runtime agent identity, and prototype-only status. |

## Ten-minute technical defense preparation

| Time | Technical point |
|---|---|
| 0:00–1:00 | State the boundary: synthetic decision-support prototype; no industrial connectivity or actuation. The official problem-statement PDF was not present, so exact organizer compliance cannot be claimed. |
| 1:00–2:00 | Walk through FastAPI, Pydantic contracts, synthetic data/label separation, and in-memory event ingestion. Explain that durable event sourcing/authentication are not implemented. |
| 2:00–3:00 | Explain replay semantics: stable event-time ordering, duplicate-collapsed replay view, full original evidence retention, and unresolved alternatives for conflicting observations. |
| 3:00–4:00 | Explain the configured NetworkX causal graph and Bayesian evidence ranking. Disclose conditional-independence assumptions and lack of hidden-confounder adjustment or causal discovery. |
| 4:00–5:00 | Explain SimPy/Monte Carlo standalone and conditional sequential plans. Show how delayed isolation pressure risk can be configured, and distinguish seed reproducibility from model validity. |
| 5:00–6:00 | Explain synthetic loss-index components, empirical CVaR95, and risk-aversion weight. State that weights/penalties are illustrative comparison units, not monetary or calibrated operational risk. |
| 6:00–7:00 | Explain the two scikit-learn models, chronological leakage control, and actual saved metrics. Call out Isolation Forest recall and negative held-out sensor-malfunction R-squared. |
| 7:00–8:00 | Explain five agent roles and consistency checks. Show the contradictory safety-agent test. Disclose checks do not cryptographically authenticate agent identity or guarantee runtime compromise resistance. |
| 8:00–9:00 | Explain OR-Tools selection, per-action sequence checks, UNKNOWN abstention, audit JSONL, and human review. Note that site-safety inputs are declarations and the plan set is limited to two predefined sequences. |
| 9:00–10:00 | State verified test count and known gaps: official PDF/T1–T8 exact alignment, hidden confounders, organizer simulator, scale benchmark, durable authenticated event ledger, and real-facility validation. |

### Likely judge questions

- **Is this safe to connect to a plant?** No. It has no industrial protocols or actuators and is not an engineering safety case.
- **Are the risks calibrated?** No. The simulator and loss index are synthetic assumptions; CVaR summarizes those simulated samples only.
- **What happens when agents disagree?** Evidence-reference and safety-vs-simulation conflicts are recorded; the affected agent is set to UNKNOWN and the council becomes partial, which blocks automatic selection. This is consistency checking, not cryptographic identity verification.
- **Can the system address all official adversarial cases?** Only cases concretely tested in the repository can be claimed. The official PDF and organizer test harness were unavailable, and hidden-confounder support is incomplete.
- **Why show machine learning?** To demonstrate an actual training/inference pipeline, not industrial accuracy. Present the synthetic validation limitations with the metrics.

## Optional API proof

Open `http://127.0.0.1:8000/docs` to show the FastAPI route contracts, or verify health in PowerShell:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

## Safety and honesty notes for presenters

- Do not describe this prototype as predicting real industrial catastrophes or certified plant risk.
- Do not treat anomaly scores as fault severity or as a safe/unsafe threshold.
- Do not use a model forecast to override safety constraints or site procedures.
- The Random Forest held-out `sensor_malfunction` scenario validation had negative R-squared; explain the limitation if presenting model metrics.
- No dashboard decision controls real equipment. Qualified human review and facility-approved procedures remain mandatory.
