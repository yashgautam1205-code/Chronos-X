"""CHRONOS-X — Scenario-Aware Industrial AI Command Center.

Premium futuristic redesign with full scenario investigation workflow.
Every data source is the existing backend; no mock data, no hardcoded values,
no bypassed endpoints. All agent, simulation, and decision results come from
the actual FastAPI service.
"""

import json
import os
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID

import networkx as nx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from frontend.api import APIError, ChronosAPI

# ─── Constants ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DEFAULT_SIMULATION: dict[str, object] = {
    "rollouts": 100,
    "seed": 42,
    "initial_temperature_c": 84.0,
    "initial_pressure_kpa": 112.0,
    "initial_cooling_failure": True,
    "simulation_hours": 4.0,
    "time_step_minutes": 1.0,
    "pump_failure_rate_per_hour": 0.01,
    "equipment_failure_rate_per_hour": 0.002,
    "backup_available_probability": 0.9,
    "backup_activation_delay_minutes": 10.0,
    "isolation_delay_minutes": 10.0,
    "shutdown_delay_minutes": 2.0,
    "thermal_incident_threshold_c": 100.0,
    "pressure_incident_threshold_kpa": 160.0,
}

INTERVENTION_LABELS = {
    "no_intervention": "No Intervention",
    "activate_backup_cooling": "Activate Backup Cooling",
    "isolate_affected_system": "Isolate Affected System",
    "controlled_shutdown": "Controlled Shutdown",
    "backup_then_shutdown": "Backup → Shutdown",
    "isolation_then_shutdown": "Isolation → Shutdown",
}
INTERVENTION_ICONS = {
    "no_intervention": "⊘",
    "activate_backup_cooling": "❄",
    "isolate_affected_system": "🔒",
    "controlled_shutdown": "⏻",
    "backup_then_shutdown": "❄→⏻",
    "isolation_then_shutdown": "🔒→⏻",
}
INTERVENTION_OPERATORS = {
    "no_intervention": 0,
    "activate_backup_cooling": 1,
    "isolate_affected_system": 1,
    "controlled_shutdown": 2,
}

# Theme palette
CY = "#00e5ff"
CY_DIM = "#0097a7"
BLUE = "#2979ff"
ORANGE = "#ff9100"
RED = "#ff1744"
GREEN = "#00e676"
SURFACE = "#0a1929"
BG = "#060e1a"
BORDER = "#12344d"
TEXT = "#e0f7fa"
TEXT_DIM = "#78909c"

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CHRONOS-X | Industrial AI Command Center",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Premium CSS ─────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    :root {{ color-scheme: dark; }}
    .stApp {{ background: {BG}; color: {TEXT}; font-family: 'Inter', sans-serif; }}
    [data-testid="stHeader"] {{ background: linear-gradient(180deg, rgba(6,14,26,.95) 0%, rgba(6,14,26,.8) 100%); backdrop-filter: blur(12px); border-bottom: 1px solid {BORDER}; }}
    [data-testid="stSidebar"] {{ background: linear-gradient(180deg, #0b1d30 0%, {SURFACE} 100%); border-right: 1px solid {BORDER}; }}
    [data-testid="stSidebar"] .stMarkdown h2 {{ color: {CY}; font-family: 'JetBrains Mono', monospace; font-size: .82rem; letter-spacing: .15em; text-transform: uppercase; border-bottom: 1px solid {BORDER}; padding-bottom: 10px; }}
    [data-testid="stMetric"] {{ background: linear-gradient(135deg, #0d2137 0%, #0a1726 100%); border: 1px solid {BORDER}; border-radius: 12px; padding: 18px 16px; transition: border-color .3s ease, box-shadow .3s ease; }}
    [data-testid="stMetric"]:hover {{ border-color: {CY_DIM}; box-shadow: 0 0 20px rgba(0,229,255,.08); }}
    [data-testid="stMetricLabel"] {{ color: {TEXT_DIM}; font-family: 'JetBrains Mono', monospace; font-size: .7rem; letter-spacing: .1em; text-transform: uppercase; }}
    [data-testid="stMetricValue"] {{ color: {CY}; font-family: 'JetBrains Mono', monospace; font-weight: 700; }}
    div[data-testid="stTabs"] button {{ color: {TEXT_DIM}; font-family: 'JetBrains Mono', monospace; font-size: .72rem; letter-spacing: .06em; text-transform: uppercase; padding: 10px 12px; border-bottom: 2px solid transparent; transition: all .3s ease; }}
    div[data-testid="stTabs"] button:hover {{ color: {CY}; background: rgba(0,229,255,.04); }}
    div[data-testid="stTabs"] button[aria-selected="true"] {{ color: {CY}; border-bottom-color: {CY}; background: rgba(0,229,255,.06); }}
    div.stButton > button[kind="primary"] {{ background: linear-gradient(135deg, #006064 0%, #00838f 100%); border: 1px solid {CY_DIM}; color: white; font-family: 'JetBrains Mono', monospace; font-size: .76rem; letter-spacing: .08em; text-transform: uppercase; border-radius: 8px; padding: 10px 20px; transition: all .3s ease; }}
    div.stButton > button[kind="primary"]:hover {{ background: linear-gradient(135deg, #00838f 0%, #00acc1 100%); border-color: {CY}; box-shadow: 0 0 24px rgba(0,229,255,.2); transform: translateY(-1px); }}
    div.stButton > button {{ border: 1px solid {BORDER}; color: {TEXT_DIM}; border-radius: 8px; font-family: 'JetBrains Mono', monospace; font-size: .72rem; transition: all .2s ease; }}
    div.stButton > button:hover {{ border-color: {CY_DIM}; color: {CY}; }}
    [data-testid="stExpander"] {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 10px; overflow: hidden; }}
    [data-testid="stExpander"] summary {{ font-family: 'JetBrains Mono', monospace; font-size: .8rem; color: {TEXT}; }}
    [data-testid="stDataFrame"] {{ border: 1px solid {BORDER}; border-radius: 10px; overflow: hidden; }}
    .cx-hero {{ padding: 24px 28px; border: 1px solid {BORDER}; border-radius: 16px; background: linear-gradient(120deg, #0d2137 0%, {SURFACE} 50%, #0f2a3d 100%); margin: 0 0 20px; position: relative; overflow: hidden; }}
    .cx-hero::before {{ content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px; background: linear-gradient(90deg, transparent, {CY}, transparent); opacity: .6; }}
    .cx-hero h1 {{ margin: 0; color: {TEXT}; font-family: 'JetBrains Mono', monospace; font-weight: 800; font-size: 1.5rem; letter-spacing: .12em; }}
    .cx-hero .cx-subtitle {{ color: {TEXT_DIM}; margin: 6px 0 0; font-size: .85rem; }}
    .cx-badge {{ display: inline-block; color: {CY}; border: 1px solid {CY_DIM}; padding: 4px 12px; border-radius: 20px; font-family: 'JetBrains Mono', monospace; font-size: .65rem; letter-spacing: .14em; text-transform: uppercase; margin-bottom: 12px; background: rgba(0,229,255,.06); }}
    .cx-card {{ background: linear-gradient(135deg, #0d2137, {SURFACE}); border: 1px solid {BORDER}; border-radius: 12px; padding: 18px; margin-bottom: 10px; }}
    .cx-card-title {{ color: {CY}; font-family: 'JetBrains Mono', monospace; font-size: .76rem; letter-spacing: .1em; text-transform: uppercase; margin-bottom: 10px; padding-bottom: 8px; border-bottom: 1px solid {BORDER}; }}
    .cx-section {{ color: {CY}; font-family: 'JetBrains Mono', monospace; font-size: .8rem; letter-spacing: .12em; text-transform: uppercase; margin: 16px 0 12px; padding-left: 12px; border-left: 3px solid {CY}; }}
    .cx-fact {{ color: {GREEN}; font-size: .75rem; font-family: 'JetBrains Mono', monospace; padding: 2px 8px; background: rgba(0,230,118,.08); border: 1px solid rgba(0,230,118,.25); border-radius: 4px; }}
    .cx-hypothesis {{ color: {ORANGE}; font-size: .75rem; font-family: 'JetBrains Mono', monospace; padding: 2px 8px; background: rgba(255,145,0,.08); border: 1px solid rgba(255,145,0,.25); border-radius: 4px; }}
    .cx-uncertain {{ color: {TEXT_DIM}; font-size: .75rem; font-family: 'JetBrains Mono', monospace; padding: 2px 8px; background: rgba(120,144,156,.08); border: 1px solid rgba(120,144,156,.25); border-radius: 4px; }}
    .cx-alert {{ color: {RED}; font-size: .75rem; font-family: 'JetBrains Mono', monospace; padding: 2px 8px; background: rgba(255,23,68,.08); border: 1px solid rgba(255,23,68,.25); border-radius: 4px; }}
    .cx-pass {{ color: {GREEN}; }} .cx-fail {{ color: {RED}; }} .cx-unknown {{ color: {ORANGE}; }}
    .cx-console {{ background: linear-gradient(135deg, #0a1f35, #081828); border: 1px solid {CY_DIM}; border-radius: 14px; padding: 20px 22px; margin: 16px 0; position: relative; overflow: hidden; }}
    .cx-console::before {{ content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px; background: linear-gradient(90deg, {CY}, {BLUE}, {CY}); opacity: .7; }}
    .cx-console-title {{ color: {CY}; font-family: 'JetBrains Mono', monospace; font-size: .82rem; letter-spacing: .15em; text-transform: uppercase; margin-bottom: 14px; }}
    hr {{ border-color: {BORDER}; opacity: .4; }}
    .stAlert {{ border-radius: 10px; font-size: .85rem; }}
    ::-webkit-scrollbar {{ width: 6px; height: 6px; }}
    ::-webkit-scrollbar-track {{ background: {BG}; }}
    ::-webkit-scrollbar-thumb {{ background: {BORDER}; border-radius: 3px; }}
    ::-webkit-scrollbar-thumb:hover {{ background: {CY_DIM}; }}
    .stDeployButton {{ display: none; }}
    footer {{ display: none; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADERS — Preserved from original, no changes to data sourcing
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_scenario_catalog() -> pd.DataFrame:
    path = DATA_DIR / "ground_truth_labels.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Scenario catalog is missing: {path}")
    labels = pd.read_csv(path, usecols=["scenario_id", "scenario_name"])
    return (
        labels.drop_duplicates()
        .sort_values("scenario_name")
        .reset_index(drop=True)
    )


@st.cache_data(show_spinner=False)
def load_sensor_stream(scenario_id: str) -> pd.DataFrame:
    readings_path = DATA_DIR / "sensor_readings.csv"
    labels_path = DATA_DIR / "ground_truth_labels.csv"
    if not readings_path.is_file() or not labels_path.is_file():
        raise FileNotFoundError("Synthetic sensor datasets are missing from the data directory.")
    labels = pd.read_csv(
        labels_path,
        usecols=["reading_id", "scenario_id"],
        dtype={"reading_id": str, "scenario_id": str},
    )
    matching_ids = set(labels.loc[labels["scenario_id"] == scenario_id, "reading_id"])
    if not matching_ids:
        raise ValueError(f"No generated sensor readings exist for scenario {scenario_id}.")
    readings = pd.read_csv(
        readings_path,
        dtype={"reading_id": str, "equipment_id": str},
    )
    readings = readings[readings["reading_id"].isin(matching_ids)].copy()
    readings["timestamp"] = pd.to_datetime(readings["timestamp"], utc=True, errors="raise")
    return readings.sort_values(["timestamp", "equipment_id"]).reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def _show_api_error(label: str, error: Exception) -> None:
    st.error(f"{label}: {error}")


def _include_equipment_in_sensor_selection(
    scenario_id: UUID,
    equipment_id: str,
) -> None:
    key = f"multi_sensor_equipment_{scenario_id}"
    selected = st.session_state.get(key, [])
    st.session_state[key] = sorted(
        {str(item) for item in selected} | {equipment_id.strip()}
    )


def _records_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.json_normalize(records, sep=".")


def _timeline_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    frame = _records_frame(records)
    if "value" in frame and frame["value"].dtype == object:
        frame["value"] = frame["value"].astype("string")
    return frame


def _sensor_trend_figure(
    records: list[dict[str, Any]],
    *,
    metrics: set[str],
    title: str,
    y_label: str,
) -> go.Figure | None:
    frame = pd.DataFrame(records)
    if frame.empty or not {"metric", "timestamp", "value", "equipment_id"}.issubset(frame):
        return None
    frame = frame[frame["metric"].isin(metrics)].copy()
    if frame.empty:
        return None
    frame["timestamp"] = pd.to_datetime(
        frame["timestamp"],
        format="ISO8601",
        utc=True,
        errors="coerce",
    )
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame.dropna(subset=["timestamp", "value"])
    if frame.empty:
        return None
    metric_labels = {
        "coolant_flow_lpm": "Coolant Flow Rate",
        "coolant_inlet_flow_lpm": "Coolant Inlet Flow",
        "coolant_outlet_flow_lpm": "Coolant Outlet Flow",
        "voltage_v": "Voltage",
        "current_a": "Current",
    }
    frame["signal"] = frame["metric"].map(metric_labels).fillna(frame["metric"])
    frame["data_classification"] = frame.apply(
        lambda item: (
            "SYNTHETIC"
            if item.get("data_classification") == "SYNTHETIC"
            or (
                isinstance(item.get("provenance"), dict)
                and (
                    item["provenance"].get("synthetic") is True
                    or "synthetic"
                    in str(item["provenance"].get("origin", "")).lower()
                )
            )
            or (
                isinstance(item.get("provenance"), str)
                and "synthetic" in item["provenance"].lower()
            )
            else "MEASURED"
        ),
        axis=1,
    )
    frame["series"] = (
        frame["signal"] + " · " + frame["data_classification"]
    )
    figure = px.line(
        frame.sort_values("timestamp"),
        x="timestamp",
        y="value",
        color="series",
        line_dash="equipment_id",
        markers=True,
        title=title,
        labels={
            "value": y_label,
            "timestamp": "Timestamp",
            "signal": "Sensor",
            "series": "Sensor data",
            "equipment_id": "Equipment",
            "data_classification": "Data classification",
        },
        hover_data={
            "timestamp": "|%Y-%m-%d %H:%M:%S %Z",
            "equipment_id": True,
            "data_classification": True,
        },
    )
    figure.update_traces(
        hovertemplate=(
            "%{fullData.name}<br>"
            "Timestamp: %{x|%Y-%m-%d %H:%M:%S %Z}<br>"
            f"{y_label}: %{{y}}<extra></extra>"
        )
    )
    figure.update_layout(
        xaxis_title="Timestamp",
        yaxis_title=y_label,
        legend_title_text="Sensor / equipment",
    )
    return _plotly_layout(figure, title)


def _plotly_layout(fig: go.Figure, title: str = "", height: int | None = None) -> go.Figure:
    layout_kwargs: dict[str, Any] = {
        "template": "plotly_dark",
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "margin": dict(l=16, r=16, t=52, b=16),
        "font": dict(family="JetBrains Mono, monospace", size=11, color=TEXT_DIM),
        "title": dict(text=title, font=dict(size=13, color=TEXT)),
        "legend": dict(bgcolor="rgba(0,0,0,0)", bordercolor=BORDER, borderwidth=1, font=dict(size=10)),
        "xaxis": dict(gridcolor="rgba(18,52,77,.4)", zerolinecolor=BORDER),
        "yaxis": dict(gridcolor="rgba(18,52,77,.4)", zerolinecolor=BORDER),
    }
    if height:
        layout_kwargs["height"] = height
    fig.update_layout(**layout_kwargs)
    return fig


def _distribution_chart(
    report: dict[str, Any], *, metric_key: str, title: str, y_label: str,
) -> go.Figure:
    data = []
    for intervention in report.get("interventions", []):
        distribution = intervention.get(metric_key, {})
        for statistic, value in distribution.items():
            data.append({"Intervention": INTERVENTION_LABELS.get(intervention.get("intervention_name"), intervention.get("intervention_name")), "Statistic": statistic, "Value": value})
    frame = pd.DataFrame(data)
    figure = px.bar(frame, x="Intervention", y="Value", color="Statistic", barmode="group", title=title, color_discrete_sequence=[CY, BLUE, ORANGE, RED])
    _plotly_layout(figure, title)
    figure.update_layout(yaxis_title=y_label)
    return figure


def _causal_graph(report: dict[str, Any]) -> go.Figure | None:
    relationships = report.get("causal_relationships", [])
    if not relationships:
        return None
    graph = nx.DiGraph()
    for edge in relationships:
        source = str(edge.get("cause", ""))
        target = str(edge.get("effect", ""))
        if source and target:
            graph.add_edge(source, target, mechanism=edge.get("mechanism", ""))
    if not graph.nodes:
        return None
    positions = nx.spring_layout(graph, seed=42, k=1.1)
    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    for source, target in graph.edges:
        edge_x.extend([positions[source][0], positions[target][0], None])
        edge_y.extend([positions[source][1], positions[target][1], None])
    node_x = [positions[node][0] for node in graph.nodes]
    node_y = [positions[node][1] for node in graph.nodes]
    figure = go.Figure(data=[
        go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(width=1.5, color=CY_DIM), hoverinfo="skip"),
        go.Scatter(x=node_x, y=node_y, mode="markers+text", text=[n.replace("_", " ").title() for n in graph.nodes], textposition="top center", textfont=dict(size=10, color=TEXT, family="JetBrains Mono"), marker=dict(size=18, color=CY, line=dict(width=2, color="rgba(0,229,255,.3)")),
            hovertext=["<br>".join([f"<b>{n.replace('_',' ').title()}</b>"] + [f"-> {t.replace('_',' ').title()}: {graph.edges[n,t]['mechanism']}" for t in graph.successors(n)]) for n in graph.nodes], hoverinfo="text"),
    ])
    _plotly_layout(figure, "Root-Cause Causal Graph", height=420)
    figure.update_layout(showlegend=False, xaxis=dict(visible=False), yaxis=dict(visible=False))
    return figure


def _intervention_comparison_table(simulation_report: dict[str, Any]) -> pd.DataFrame:
    """Build a side-by-side comparison dataframe from actual simulation data."""
    rows = []
    all_interventions = [
        *simulation_report.get("interventions", []),
        *simulation_report.get("sequential_action_plans", []),
    ]
    for iv in all_interventions:
        name = iv.get("intervention_name", "")
        outcomes = iv.get("simulated_incident_outcomes", {})
        risk = iv.get("risk_estimates", {})
        temp = iv.get("temperature_distribution", {}).get("maximum_temperature_c", {})
        pres = iv.get("pressure_distribution", {}).get("maximum_pressure_kpa", {})
        rows.append({
            "Intervention": INTERVENTION_LABELS.get(name, name),
            "Incident Count": outcomes.get("incident_count", "N/A"),
            "Catastrophic": outcomes.get("catastrophic_incident_count", "N/A"),
            "Equipment Failures": outcomes.get("equipment_failure_count", "N/A"),
            "Secondary Failures": outcomes.get("delayed_secondary_failure_count", "N/A"),
            "Incident Rate": risk.get("simulated_incident_rate", "N/A"),
            "Mean Loss": risk.get("mean_loss_index", "N/A"),
            "CVaR 95%": risk.get("cvar_95_loss_index", "N/A"),
            "Mean Max Temp (C)": temp.get("mean", "N/A"),
            "P95 Max Temp (C)": temp.get("p95", "N/A"),
            "Mean Max Press (kPa)": pres.get("mean", "N/A"),
            "P95 Max Press (kPa)": pres.get("p95", "N/A"),
            "Operators Required": INTERVENTION_OPERATORS.get(name, "N/A"),
            "Rollouts": iv.get("number_of_rollouts", "N/A"),
        })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# DECISION FORM — Preserved logic from original
# ═══════════════════════════════════════════════════════════════════════════════

def _decision_request() -> dict[str, object]:
    with st.form("decision_form"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<p class="cx-section">Equipment Status</p>', unsafe_allow_html=True)
            backup = st.selectbox("Backup cooling confirmed", ["UNKNOWN", "PASS", "FAIL"])
            isolation = st.selectbox("System isolation confirmed", ["UNKNOWN", "PASS", "FAIL"])
            shutdown = st.selectbox("Controlled shutdown authorized", ["UNKNOWN", "PASS", "FAIL"])
        with col2:
            st.markdown('<p class="cx-section">Operational Parameters</p>', unsafe_allow_html=True)
            operators = st.number_input("Operators available", min_value=0, max_value=100000, value=0)
            model_reviewed = st.checkbox("Model outputs reviewed by a qualified person")
            site_review = st.selectbox("Site safety review", ["UNKNOWN", "PASS", "FAIL"])
            risk_aversion_weight = st.slider("Tail-risk aversion weight", min_value=0.0, max_value=5.0, value=1.0, step=0.25)
        submitted = st.form_submit_button("Evaluate Safety-Gated Decision", type="primary")
    if not submitted:
        return {}
    return {
        "submitted": True,
        "risk_aversion_weight": risk_aversion_weight,
        "safety_checks": [{"check_name": "site_safety_review", "result": site_review, "details": "Entered in the command-center decision form.", "evidence_ids": []}],
        "operational_constraints": {
            "backup_cooling_available": None if backup == "UNKNOWN" else backup == "PASS",
            "system_isolation_available": None if isolation == "UNKNOWN" else isolation == "PASS",
            "controlled_shutdown_authorized": None if shutdown == "UNKNOWN" else shutdown == "PASS",
            "available_operators": int(operators),
            "model_outputs_reviewed": model_reviewed,
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    # ─── Hero ────────────────────────────────────────────────────────────
    st.markdown(
        '<div class="cx-hero"><div class="cx-badge">Industrial Intelligence · Live API · Scenario-Aware</div>'
        '<h1>CHRONOS-X</h1>'
        '<p class="cx-subtitle">AI-powered industrial catastrophe prediction, investigation, simulation & safety-gated intervention decisions</p></div>',
        unsafe_allow_html=True,
    )

    # ─── Sidebar ─────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## Command Panel")
        api_url = st.text_input("FastAPI base URL", value=os.getenv("CHRONOS_API_URL", "http://127.0.0.1:8000"))
        st.divider()
        catalog_error = None
        try:
            catalog = load_scenario_catalog()
        except (OSError, ValueError, pd.errors.ParserError) as error:
            catalog_error = error
            catalog = pd.DataFrame(columns=["scenario_id", "scenario_name"])
        if catalog.empty:
            scenario_id_text = st.text_input("Scenario UUID")
            scenario_name = "Custom scenario"
        else:
            scenario_options = catalog.to_dict(orient="records")
            scenario_by_name = {s["scenario_name"]: s for s in scenario_options}
            selected = st.selectbox("Industrial Scenario", list(scenario_by_name), format_func=lambda n: n.replace("_", " ").title())
            selected_scenario = scenario_by_name[selected]
            scenario_id_text = str(selected_scenario["scenario_id"])
            scenario_name = str(selected_scenario["scenario_name"])
        st.divider()
        refresh = st.button("Refresh Backend Data", use_container_width=True)
        if refresh:
            st.cache_data.clear()
            st.rerun()
        st.divider()
        st.markdown("## Investigation Status")
        has_recon = bool(st.session_state.get("_reconstruction_loaded"))
        has_sim = bool(st.session_state.get("simulation_report"))
        has_agents = bool(st.session_state.get("agent_report"))
        has_decision = bool(st.session_state.get("decision_report"))
        status_items = [
            ("Reconstruction", has_recon), ("Simulation", has_sim),
            ("Agent Council", has_agents), ("Decision", has_decision),
        ]
        for label, done in status_items:
            color = GREEN if done else TEXT_DIM
            icon = "●" if done else "○"
            st.markdown(f'<span style="color:{color}; font-family: JetBrains Mono, monospace; font-size:.78rem;">{icon} {label}</span>', unsafe_allow_html=True)
        st.caption(
            "Scenario charts use generated observations; multi-sensor telemetry uses "
            "persisted backend readings."
        )

    if catalog_error:
        st.error(f"Could not load the generated scenario catalog: {catalog_error}")
    try:
        scenario_id = UUID(scenario_id_text)
    except ValueError:
        st.error("Enter a valid scenario UUID to query the backend.")
        return
    if st.session_state.get("active_scenario_id") != str(scenario_id):
        for key in ("simulation_report", "agent_report", "decision_report", "_reconstruction_loaded"):
            st.session_state.pop(key, None)
        st.session_state["active_scenario_id"] = str(scenario_id)

    # ─── Backend data fetch ──────────────────────────────────────────────
    api = ChronosAPI(api_url)
    try:
        with st.spinner("Connecting to CHRONOS-X and loading scenario evidence…"):
            health = api.health()
            reconstruction = api.reconstruct(scenario_id)
            causes = api.causes(scenario_id)
            events = api.events()
            audit = api.audit(scenario_id)
        api_error: str | None = None
        st.session_state["_reconstruction_loaded"] = True
    except APIError as error:
        api_error = str(error)
        health, reconstruction, causes, events, audit = {}, {}, {}, [], []

    if api_error:
        st.error(f"Backend data unavailable: {api_error}")
        st.info("Start the API with `uvicorn backend.main:app --reload`, then use Refresh.")

    sensor_api_error = None
    sensor_analysis: dict[str, Any] = {}
    try:
        with st.spinner("Loading ingested multi-sensor telemetry…"):
            sensor_analysis = api.sensor_analysis()
    except APIError as error:
        sensor_api_error = str(error)

    try:
        sensor_frame = load_sensor_stream(str(scenario_id))
    except (OSError, ValueError, pd.errors.ParserError) as error:
        sensor_frame = pd.DataFrame()
        st.warning(f"Sensor stream unavailable: {error}")
    scenario_rows = sensor_frame.tail(500).copy() if not sensor_frame.empty else sensor_frame
    latest = scenario_rows.sort_values("timestamp").tail(1) if not scenario_rows.empty else pd.DataFrame()

    events_for_scenario = [e for e in events if str((e.get("metadata") or {}).get("scenario_id", "")) == str(scenario_id)]
    alerts_count = sum(len(reconstruction.get(k, [])) for k in ("delayed_events", "conflicting_evidence", "suspicious_sources"))

    # ═══════════════════════════════════════════════════════════════════════
    # SECTION 1: FACILITY STATUS BAR
    # ═══════════════════════════════════════════════════════════════════════
    status_health = health.get("status", "Unavailable").upper()
    status_color = GREEN if status_health == "OK" else RED

    s1, s2, s3, s4, s5 = st.columns(5)
    s1.metric("API Status", status_health)
    s2.metric("Evidence Alerts", alerts_count)
    s3.metric("Equipment Streams", sensor_frame["equipment_id"].nunique() if not sensor_frame.empty else 0)
    s4.metric("Decision Records", len(audit))
    sim_hours = DEFAULT_SIMULATION.get("simulation_hours", 4.0)
    s5.metric("Decision Window", f"{sim_hours}h")

    if not sensor_frame.empty:
        t1, t2, t3, t4, t5 = st.columns(5)
        t1.metric("Temperature", f"{latest.iloc[0]['temperature_c']:.1f} °C")
        t2.metric("Pressure", f"{latest.iloc[0]['pressure_kpa']:.1f} kPa")
        t3.metric("Vibration", f"{latest.iloc[0]['vibration_mm_s']:.2f} mm/s")
        cooling_val = latest.iloc[0].get("cooling_pump_status", None)
        t4.metric("Cooling Pump", "RUNNING" if cooling_val == 1 else "FAULT" if cooling_val == 0 else "N/A")
        smoke_val = latest.iloc[0].get("smoke_detected", None)
        t5.metric("Smoke Detector", "CLEAR" if smoke_val == 0 else "ALERT" if smoke_val == 1 else "N/A")

    # ═══════════════════════════════════════════════════════════════════════
    # SECTION 2: INCIDENT BRIEFING
    # ═══════════════════════════════════════════════════════════════════════
    st.markdown(
        f'<div class="cx-card"><div class="cx-card-title">Incident Briefing — {scenario_name.replace("_"," ").title()}</div>'
        f'<table style="width:100%;border-collapse:collapse;">'
        f'<tr><td style="color:{TEXT_DIM};padding:5px 0;width:160px;">Scenario ID</td><td style="color:{CY};font-family:JetBrains Mono,monospace;font-size:.78rem;">{scenario_id}</td></tr>'
        f'<tr><td style="color:{TEXT_DIM};padding:5px 0;">API Health</td><td style="color:{status_color};">{status_health}</td></tr>'
        f'<tr><td style="color:{TEXT_DIM};padding:5px 0;">Assessment</td><td style="color:{TEXT};">{reconstruction.get("summary", "Backend response not loaded.")}</td></tr>'
        f'<tr><td style="color:{TEXT_DIM};padding:5px 0;">Simulation Horizon</td><td style="color:{TEXT};">{sim_hours} hours · {DEFAULT_SIMULATION.get("time_step_minutes", 1.0)} min time step · {DEFAULT_SIMULATION.get("rollouts", 100)} rollouts</td></tr>'
        f'<tr><td style="color:{TEXT_DIM};padding:5px 0;">Initial Conditions</td><td style="color:{TEXT};">{DEFAULT_SIMULATION.get("initial_temperature_c", 84)} °C · {DEFAULT_SIMULATION.get("initial_pressure_kpa", 112)} kPa · Cooling failure: {"Yes" if DEFAULT_SIMULATION.get("initial_cooling_failure") else "No"}</td></tr>'
        f'<tr><td style="color:{TEXT_DIM};padding:5px 0;">Thresholds</td><td style="color:{ORANGE};">Thermal: {DEFAULT_SIMULATION.get("thermal_incident_threshold_c", 100)} °C · Pressure: {DEFAULT_SIMULATION.get("pressure_incident_threshold_kpa", 160)} kPa</td></tr>'
        f'</table></div>',
        unsafe_allow_html=True,
    )

    # ═══════════════════════════════════════════════════════════════════════
    # SECTION 3: AI INCIDENT RESPONSE CONSOLE
    # ═══════════════════════════════════════════════════════════════════════
    st.markdown(
        '<div class="cx-console"><div class="cx-console-title">AI Incident Response Console</div></div>',
        unsafe_allow_html=True,
    )

    rc1, rc2, rc3, rc4 = st.columns(4)
    with rc1:
        if st.button("Simulate Interventions", type="primary", use_container_width=True):
            try:
                with st.spinner("Running seeded Monte Carlo rollouts…"):
                    st.session_state["simulation_report"] = api.simulate(scenario_id, dict(DEFAULT_SIMULATION))
                st.rerun()
            except APIError as error:
                _show_api_error("Simulation failed", error)
    with rc2:
        if st.button("Run Agent Council", type="primary", use_container_width=True):
            try:
                with st.spinner("Running five-agent intelligence council…"):
                    st.session_state["agent_report"] = api.run_agents(scenario_id, DEFAULT_SIMULATION)
                st.rerun()
            except APIError as error:
                _show_api_error("Agent run failed", error)
    with rc3:
        if st.button("Full Analysis Pipeline", type="primary", use_container_width=True):
            try:
                with st.spinner("Running simulation + agents…"):
                    st.session_state["simulation_report"] = api.simulate(scenario_id, dict(DEFAULT_SIMULATION))
                    st.session_state["agent_report"] = api.run_agents(scenario_id, DEFAULT_SIMULATION)
                st.rerun()
            except APIError as error:
                _show_api_error("Pipeline failed", error)
    with rc4:
        if st.button("Refresh Evidence", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    # ═══════════════════════════════════════════════════════════════════════
    # MAIN INVESTIGATION TABS
    # ═══════════════════════════════════════════════════════════════════════
    agent_report = st.session_state.get("agent_report", {})
    simulation_report = st.session_state.get("simulation_report", {})
    decision_report_state = st.session_state.get("decision_report", {})

    predictor = next((a for a in agent_report.get("agent_results", []) if a.get("agent_name") == "predictor"), {})

    tabs = st.tabs([
        "Situation Assessment",
        "Root-Cause Investigation",
        "Predictive Analysis",
        "Counterfactual Comparison",
        "Agent Intelligence",
        "Decision & Safety",
        "Scenario Testing",
        "Audit & Replay",
        "Judge Scenario",
        "Multi-Sensor Telemetry",
    ])

    # ═══ TAB 0: SITUATION ASSESSMENT ═════════════════════════════════════
    with tabs[0]:
        st.markdown('<p class="cx-section">Incident Situation Assessment</p>', unsafe_allow_html=True)

        # Equipment status
        if not sensor_frame.empty:
            st.markdown('<p class="cx-section">Equipment Health Matrix</p>', unsafe_allow_html=True)
            equipment_status = (
                sensor_frame.sort_values("timestamp").groupby("equipment_id", as_index=False).tail(1)
                .loc[:, ["equipment_id", "timestamp", "temperature_c", "pressure_kpa", "vibration_mm_s", "cooling_pump_status", "power_supply_status", "smoke_detected"]].copy()
            )
            equipment_status["Cooling"] = equipment_status["cooling_pump_status"].map({1: "RUNNING", 0: "FAULT"})
            equipment_status["Power"] = equipment_status["power_supply_status"].map({1: "ONLINE", 0: "INTERRUPTED"})
            equipment_status["Smoke"] = equipment_status["smoke_detected"].map({1: "ALERT", 0: "CLEAR"})
            st.dataframe(equipment_status.drop(columns=["cooling_pump_status", "power_supply_status", "smoke_detected"]), use_container_width=True)
        else:
            st.info("No sensor observations available for this scenario.")

        # Live event timeline with trust analysis
        st.markdown('<p class="cx-section">Incoming Events with Source & Trust Analysis</p>', unsafe_allow_html=True)
        if events_for_scenario:
            events_df = _records_frame(events_for_scenario)
            st.dataframe(events_df, use_container_width=True)
        else:
            st.caption("No matching events returned by GET /events.")

        # Evidence integrity from reconstruction
        if reconstruction:
            st.markdown('<p class="cx-section">Evidence Integrity Report</p>', unsafe_allow_html=True)
            ic1, ic2, ic3, ic4 = st.columns(4)
            ic1.metric("Delayed", len(reconstruction.get("delayed_events", [])))
            ic2.metric("Duplicates", len(reconstruction.get("duplicate_events", [])))
            ic3.metric("Conflicts", len(reconstruction.get("conflicting_evidence", [])))
            ic4.metric("Suspicious", len(reconstruction.get("suspicious_sources", [])))

            # Highlight conflicting evidence
            conflicts = reconstruction.get("conflicting_evidence", [])
            if conflicts:
                st.markdown(f'<p class="cx-section" style="border-left-color:{RED};">Conflicting Reports Detected</p>', unsafe_allow_html=True)
                for conflict in conflicts:
                    with st.expander(f"Conflict: {conflict.get('metric', 'Unknown')} at {conflict.get('timestamp', 'Unknown time')}"):
                        st.markdown(f'<span class="cx-alert">UNRESOLVED CONFLICT</span> — Different sensors report different values for the same metric at the same time.', unsafe_allow_html=True)
                        for obs in conflict.get("observations", []):
                            st.markdown(f"- Source: `{obs.get('source', 'N/A')}` | Value: `{obs.get('value', 'N/A')}` | Evidence: `{obs.get('evidence_id', 'N/A')}`")

            # Highlight suspicious sources
            suspicious = reconstruction.get("suspicious_sources", [])
            if suspicious:
                st.markdown(f'<p class="cx-section" style="border-left-color:{ORANGE};">Suspicious Sources Identified</p>', unsafe_allow_html=True)
                for src in suspicious:
                    with st.expander(f"Source: {src.get('source', 'Unknown')}"):
                        for reason in src.get("reasons", []):
                            st.markdown(f"- {reason}")
                        st.markdown(f"Affected evidence IDs: {src.get('evidence_ids', [])}")

            # Delayed events
            delayed = reconstruction.get("delayed_events", [])
            if delayed:
                st.markdown(f'<p class="cx-section" style="border-left-color:{ORANGE};">Delayed Event Arrivals</p>', unsafe_allow_html=True)
                st.dataframe(_records_frame(delayed), use_container_width=True)

        # Sensor monitoring
        if not scenario_rows.empty:
            st.markdown('<p class="cx-section">Sensor Streams</p>', unsafe_allow_html=True)
            equipment_ids = sorted(scenario_rows["equipment_id"].dropna().unique())
            chosen_equipment = st.multiselect("Equipment", equipment_ids, default=equipment_ids[:min(4, len(equipment_ids))])
            visible = scenario_rows[scenario_rows["equipment_id"].isin(chosen_equipment)]
            for metric, label, unit, color in (("temperature_c", "Temperature", "°C", CY), ("pressure_kpa", "Pressure", "kPa", BLUE), ("vibration_mm_s", "Vibration", "mm/s", ORANGE)):
                fig = px.line(visible, x="timestamp", y=metric, color="equipment_id", title=f"{label} · Observed", labels={metric: f"{label} ({unit})", "timestamp": "Timestamp"}, color_discrete_sequence=[color, BLUE, ORANGE, "#b388ff"])
                _plotly_layout(fig, f"{label} · Observed Sensor Readings")
                st.plotly_chart(fig, use_container_width=True)
            st.caption("All plotted values are observed synthetic sensor data, not predictions.")

    # ═══ TAB 1: ROOT-CAUSE INVESTIGATION ═════════════════════════════════
    with tabs[1]:
        st.markdown('<p class="cx-section">Causal Investigation & Root-Cause Analysis</p>', unsafe_allow_html=True)

        if causes:
            st.info(causes.get("summary", ""))

            # Causal graph
            graph_fig = _causal_graph(causes)
            if graph_fig:
                st.plotly_chart(graph_fig, use_container_width=True)

            # Competing hypotheses
            hypotheses = causes.get("ranked_hypotheses", [])
            if len(hypotheses) >= 2:
                st.markdown(f'<p class="cx-section">Competing Causal Explanations ({len(hypotheses)} hypotheses)</p>', unsafe_allow_html=True)

                for idx, hyp in enumerate(hypotheses):
                    prob = hyp.get("posterior_probability", "N/A")
                    title = hyp.get("title", hyp.get("hypothesis_id", f"Hypothesis {idx+1}"))
                    interp = hyp.get("interpretation", "")

                    if idx == 0:
                        badge = f'<span class="cx-hypothesis">MOST LIKELY</span>'
                    else:
                        badge = f'<span class="cx-uncertain">ALTERNATIVE</span>'

                    with st.expander(f"{'#'+str(idx+1)} {title} — P={prob}", expanded=(idx < 2)):
                        st.markdown(f'{badge} <span style="color:{TEXT_DIM}; font-size:.82rem; margin-left:8px;">{interp}</span>', unsafe_allow_html=True)

                        # Distinguish facts from hypotheses
                        supporting = hyp.get("supporting_evidence", [])
                        contradicting = hyp.get("contradicting_evidence", [])
                        missing = hyp.get("missing_evidence", [])

                        ev_left, ev_right = st.columns(2)
                        with ev_left:
                            st.markdown(f'<span class="cx-fact">SUPPORTING EVIDENCE</span>', unsafe_allow_html=True)
                            if supporting:
                                st.dataframe(_records_frame(supporting), use_container_width=True)
                            else:
                                st.caption("No supporting evidence.")
                        with ev_right:
                            st.markdown(f'<span class="cx-alert">CONTRADICTING EVIDENCE</span>', unsafe_allow_html=True)
                            if contradicting:
                                st.dataframe(_records_frame(contradicting), use_container_width=True)
                            else:
                                st.caption("No contradicting evidence.")

                        if missing:
                            st.markdown(f'<span class="cx-uncertain">MISSING EVIDENCE</span>', unsafe_allow_html=True)
                            for item in missing:
                                st.markdown(f"- {item}")
            elif hypotheses:
                st.markdown('<p class="cx-section">Root-Cause Hypothesis</p>', unsafe_allow_html=True)
                st.dataframe(_records_frame(hypotheses), use_container_width=True)

            # Possible root causes
            root_causes = causes.get("possible_root_causes", [])
            if root_causes:
                st.markdown('<p class="cx-section">Identified Possible Root Causes</p>', unsafe_allow_html=True)
                st.dataframe(_records_frame(root_causes), use_container_width=True)

            st.caption("Conditional hypotheses are not proof of cause; causal links represent configured mechanisms with synthetic priors.")
        else:
            st.info("Causal investigation response is unavailable.")

    # ═══ TAB 2: PREDICTIVE ANALYSIS ══════════════════════════════════════
    with tabs[2]:
        st.markdown('<p class="cx-section">Predictive Analysis — ML Model Outputs</p>', unsafe_allow_html=True)

        assessments = predictor.get("outputs", {}).get("anomaly_assessments", [])
        prediction = predictor.get("outputs", {}).get("future_sensor_prediction")

        if not assessments and not prediction:
            st.info("Run the Agent Council to generate ML-based predictions from the trained Isolation Forest and Random Forest models.")

        # Fault detection
        if assessments:
            st.markdown('<p class="cx-section">Fault Detection — Isolation Forest</p>', unsafe_allow_html=True)
            anomaly_frame = _records_frame(assessments)
            flagged = anomaly_frame[anomaly_frame["anomaly"] == True]  # noqa: E712
            fc1, fc2 = st.columns(2)
            fc1.metric("Model-Flagged", len(flagged))
            fc2.metric("Total Assessed", len(anomaly_frame))
            st.dataframe(flagged.sort_values("anomaly_score", ascending=False), use_container_width=True)
            history = anomaly_frame.sort_values("timestamp") if "timestamp" in anomaly_frame else anomaly_frame
            if "anomaly_score" in history:
                score_chart = px.scatter(history, x="timestamp" if "timestamp" in history else history.index, y="anomaly_score", color="anomaly", title="Anomaly Scores", color_discrete_map={True: RED, False: CY})
                _plotly_layout(score_chart, "Anomaly Scores · Isolation Forest")
                st.plotly_chart(score_chart, use_container_width=True)
            st.caption("Anomaly scores are actual trained-model outputs; they do not prove a fault.")

        # Future prediction
        if prediction:
            st.markdown('<p class="cx-section">Future Prediction — Random Forest Regressor</p>', unsafe_allow_html=True)
            st.markdown(f'<span class="cx-hypothesis">ML FORECAST — Not an observed value</span>', unsafe_allow_html=True)
            p1, p2, p3 = st.columns(3)
            p1.metric("Predicted Temp", f"{prediction['predicted_temperature_c']:.2f} °C")
            p2.metric("Predicted Pressure", f"{prediction['predicted_pressure_kpa']:.2f} kPa")
            p3.metric("Horizon", f"{prediction['horizon_readings']} readings")
            st.caption(f"Model: {prediction.get('model', 'N/A')} · Origin: {prediction.get('prediction_origin_timestamp', 'N/A')}")

            # No-action prediction context
            no_action_sim = None
            if simulation_report:
                for iv in simulation_report.get("interventions", []):
                    if iv.get("intervention_name") == "no_intervention":
                        no_action_sim = iv
                        break
            if no_action_sim:
                st.markdown('<p class="cx-section">What Happens If No Action Is Taken (Simulation)</p>', unsafe_allow_html=True)
                outcomes = no_action_sim.get("simulated_incident_outcomes", {})
                risk = no_action_sim.get("risk_estimates", {})
                na1, na2, na3, na4 = st.columns(4)
                na1.metric("Incident Rate", f"{risk.get('simulated_incident_rate', 'N/A')}")
                na2.metric("Mean Loss", f"{risk.get('mean_loss_index', 'N/A')}")
                na3.metric("CVaR 95%", f"{risk.get('cvar_95_loss_index', 'N/A')}")
                na4.metric("Incidents / Rollouts", f"{outcomes.get('incident_count', 'N/A')} / {no_action_sim.get('number_of_rollouts', 'N/A')}")

            if not scenario_rows.empty:
                input_evidence = prediction.get("input_evidence", [])
                equipment_id = input_evidence[-1].get("equipment_id") if input_evidence else None
                pred_history = scenario_rows
                if equipment_id:
                    pred_history = pred_history[pred_history["equipment_id"] == equipment_id]
                plot_rows = pred_history.sort_values("timestamp").tail(80)
                prediction_time = pd.to_datetime(prediction.get("prediction_origin_timestamp"), utc=True)
                for metric, label, unit in (("temperature_c", "Temperature", "°C"), ("pressure_kpa", "Pressure", "kPa")):
                    hfig = go.Figure()
                    hfig.add_trace(go.Scatter(x=plot_rows["timestamp"], y=plot_rows[metric], name=f"Observed {label.lower()}", line=dict(color=CY, width=2)))
                    hfig.add_trace(go.Scatter(x=[prediction_time], y=[prediction[f"predicted_{metric}"]], name=f"RF forecast", mode="markers", marker=dict(size=14, color=ORANGE, symbol="diamond", line=dict(width=2, color="white"))))
                    _plotly_layout(hfig, f"History & Forecast · {label} ({unit})")
                    st.plotly_chart(hfig, use_container_width=True)

            for limitation in prediction.get("limitations", []):
                st.warning(limitation)

    # ═══ TAB 3: COUNTERFACTUAL COMPARISON ════════════════════════════════
    with tabs[3]:
        st.markdown('<p class="cx-section">Counterfactual Simulation & Intervention Comparison</p>', unsafe_allow_html=True)

        sim_seed = st.number_input("Simulation seed", min_value=0, value=42, step=1, key="sim_seed_tab3")
        if st.button("Run Simulation", type="primary", key="sim_btn_tab3"):
            try:
                sim_req = dict(DEFAULT_SIMULATION)
                sim_req["seed"] = sim_seed
                with st.spinner("Running seeded Monte Carlo rollouts…"):
                    st.session_state["simulation_report"] = api.simulate(scenario_id, sim_req)
                st.rerun()
            except APIError as error:
                _show_api_error("Simulation failed", error)

        simulation_report = st.session_state.get("simulation_report", {})
        if simulation_report and str(simulation_report.get("scenario_id")) == str(scenario_id):
            st.caption(f"Seed {simulation_report.get('seed')} · {simulation_report.get('number_of_rollouts_per_intervention')} rollouts · These are simulated outcomes, not observations.")

            # Side-by-side comparison table
            st.markdown('<p class="cx-section">Side-by-Side Intervention Comparison</p>', unsafe_allow_html=True)
            comparison_df = _intervention_comparison_table(simulation_report)
            st.dataframe(comparison_df, use_container_width=True)

            # Backend comparison data
            comparison_data = simulation_report.get("comparison_of_interventions", [])
            if comparison_data:
                st.markdown('<p class="cx-section">Relative Performance vs No-Intervention</p>', unsafe_allow_html=True)
                st.dataframe(_records_frame(comparison_data), use_container_width=True)

            # Distribution charts
            st.plotly_chart(_distribution_chart(simulation_report, metric_key="temperature_distribution", title="Simulated Temperature Distributions", y_label="Temperature (°C)"), use_container_width=True)
            st.plotly_chart(_distribution_chart(simulation_report, metric_key="pressure_distribution", title="Simulated Pressure Distributions", y_label="Pressure (kPa)"), use_container_width=True)

            # Sequential action plans
            sequential = simulation_report.get("sequential_action_plans", [])
            if sequential:
                st.markdown('<p class="cx-section">Sequential Decision Plans (Conditional Escalation)</p>', unsafe_allow_html=True)
                seq_df = _records_frame(sequential)
                st.dataframe(seq_df.drop(columns=["rollout_samples"], errors="ignore"), use_container_width=True)

                seq_comparison = simulation_report.get("sequential_plan_comparison", [])
                if seq_comparison:
                    st.markdown('<p class="cx-section">Sequential Plan Comparison</p>', unsafe_allow_html=True)
                    st.dataframe(_records_frame(seq_comparison), use_container_width=True)

            # Intervention details
            intervention_frame = _records_frame(simulation_report.get("interventions", []))
            st.markdown('<p class="cx-section">Full Intervention Data</p>', unsafe_allow_html=True)
            st.dataframe(intervention_frame.drop(columns=["rollout_samples"], errors="ignore"), use_container_width=True)

            st.markdown('<p class="cx-section">Simulation Assumptions</p>', unsafe_allow_html=True)
            for assumption in simulation_report.get("simulation_assumptions", []):
                st.markdown(f"- {assumption}")
        else:
            st.info("Run the simulation to compare intervention outcomes. Use the button above or the Response Console.")

    # ═══ TAB 4: AGENT INTELLIGENCE ═══════════════════════════════════════
    with tabs[4]:
        st.markdown('<p class="cx-section">Five-Agent Intelligence Council</p>', unsafe_allow_html=True)

        if st.button("Run Agent Council", type="primary", key="agent_btn_tab4"):
            try:
                with st.spinner("Running five-agent intelligence council…"):
                    st.session_state["agent_report"] = api.run_agents(scenario_id, DEFAULT_SIMULATION)
                st.rerun()
            except APIError as error:
                _show_api_error("Agent run failed", error)

        agent_report = st.session_state.get("agent_report", {})
        if agent_report and str(agent_report.get("scenario_id")) == str(scenario_id):
            st.caption(agent_report.get("summary", ""))

            # Adversarial intelligence - conflicts
            conflicts = agent_report.get("conflicts", [])
            if conflicts:
                st.markdown(f'<div class="cx-card" style="border-color:{RED};"><div class="cx-card-title" style="color:{RED};">Adversarial Intelligence Alert — Cross-Agent Disagreement</div>'
                    '<span class="cx-alert">INTEGRITY WARNING</span> Automated decisions must abstain when agents disagree.</div>', unsafe_allow_html=True)
                st.dataframe(_records_frame(conflicts), use_container_width=True)

            if agent_report.get("integrity_checks"):
                st.markdown('<p class="cx-section">Agent Integrity Verification</p>', unsafe_allow_html=True)
                st.dataframe(_records_frame(agent_report["integrity_checks"]), use_container_width=True)

            # Individual agent panels
            agent_meta = {
                "investigator": {"icon": "🔍", "role": "Root-cause investigation & hypothesis ranking", "color": BLUE},
                "predictor": {"icon": "🔮", "role": "ML anomaly detection & sensor forecasting", "color": CY},
                "optimizer": {"icon": "⚙️", "role": "Risk-aware intervention ranking (OR-Tools)", "color": GREEN},
                "safety": {"icon": "🛡️", "role": "Simulation threshold verification & safety gates", "color": ORANGE},
                "red_team": {"icon": "🎯", "role": "Evidence integrity & adversarial defense", "color": RED},
            }

            for agent_data in agent_report.get("agent_results", []):
                name = str(agent_data.get("agent_name", "agent"))
                display_name = name.replace("_", " ").title()
                agent_status = agent_data.get("status", "UNKNOWN")
                meta = agent_meta.get(name, {"icon": "🤖", "role": "", "color": TEXT_DIM})

                status_badge_class = "cx-fact" if agent_status == "COMPLETED" else "cx-alert" if agent_status == "FAILED" else "cx-hypothesis"

                with st.expander(f"{meta['icon']} {display_name} — {agent_status}", expanded=True):
                    st.markdown(f'<span class="{status_badge_class}">{agent_status}</span> <span style="color:{TEXT_DIM};font-size:.82rem;margin-left:8px;">{meta["role"]}</span>', unsafe_allow_html=True)
                    st.write(agent_data.get("summary", ""))

                    findings = agent_data.get("findings", [])
                    if findings:
                        st.markdown(f'<span style="color:{TEXT_DIM};font-size:.78rem;">Evidence & Findings:</span>', unsafe_allow_html=True)
                        st.dataframe(_records_frame(findings), use_container_width=True)

                    # Agent-specific rich display
                    outputs = agent_data.get("outputs", {})
                    if name == "investigator" and outputs.get("ranked_hypotheses"):
                        hyps = outputs["ranked_hypotheses"]
                        st.markdown(f"**{len(hyps)} competing hypotheses identified** — top: {hyps[0].get('title', 'N/A')} (P={hyps[0].get('posterior_probability', 'N/A')})")

                    if name == "optimizer" and outputs.get("decision_inputs"):
                        st.markdown("**Optimizer Ranking (decision support only):**")
                        st.dataframe(_records_frame(outputs["decision_inputs"]), use_container_width=True)
                        for lim in outputs.get("optimization_limitations", []):
                            st.caption(f"⚠ {lim}")

                    if name == "safety":
                        overall = outputs.get("overall_result", "UNKNOWN")
                        badge = "cx-fact" if overall == "PASS" else "cx-alert" if overall == "FAIL" else "cx-hypothesis"
                        st.markdown(f'Overall safety gate: <span class="{badge}">{overall}</span>', unsafe_allow_html=True)
                        for lim in outputs.get("limitations", []):
                            st.caption(f"⚠ {lim}")

                    if name == "red_team":
                        indicators = outputs.get("potential_manipulation_indicators", 0)
                        conclusion = outputs.get("conclusion", "")
                        st.markdown(f"**{indicators} integrity indicators found.** {conclusion}")

                    if outputs and name not in ("investigator", "optimizer", "safety", "red_team"):
                        st.json(outputs, expanded=False)

                    if agent_data.get("errors"):
                        st.error("\n".join(agent_data["errors"]))
        else:
            st.info("Run the Agent Council to obtain evidence-grounded agent findings.")

    # ═══ TAB 5: DECISION & SAFETY ════════════════════════════════════════
    with tabs[5]:
        st.markdown('<p class="cx-section">Safety Verification & Decision Certificates</p>', unsafe_allow_html=True)

        st.markdown('<div class="cx-card"><div class="cx-card-title">Safety Advisory</div>'
            'Decision output is a prototype aid, not operational approval. All site and equipment inputs must be confirmed by qualified personnel.</div>', unsafe_allow_html=True)

        form_data = _decision_request()
        if form_data.get("submitted"):
            request_payload: dict[str, object] = {
                "simulation": DEFAULT_SIMULATION,
                "operational_constraints": form_data["operational_constraints"],
                "safety_checks": form_data["safety_checks"],
                "risk_aversion_weight": form_data["risk_aversion_weight"],
            }
            try:
                with st.spinner("Evaluating safety gates and intervention feasibility…"):
                    decision_report = api.decide(scenario_id, request_payload)
                    st.session_state["decision_report"] = decision_report
                    st.session_state.pop("audit", None)
                st.rerun()
            except APIError as error:
                _show_api_error("Decision evaluation failed", error)

        decision_report_state = st.session_state.get("decision_report", {})
        if decision_report_state and str(decision_report_state.get("scenario_id")) == str(scenario_id):
            # ── Structured Final Response ──
            st.markdown('<p class="cx-section">Structured Decision Report</p>', unsafe_allow_html=True)

            dr_status = decision_report_state.get("status", "UNKNOWN")
            selected_intervention = decision_report_state.get("selected_feasible_intervention")

            if dr_status == "SELECTED":
                st.markdown(f'<div class="cx-card" style="border-color:{GREEN};"><div class="cx-card-title" style="color:{GREEN};">Decision: SELECTED</div>'
                    f'Recommended action: <strong style="color:{CY};">{INTERVENTION_LABELS.get(selected_intervention, selected_intervention)}</strong></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="cx-card" style="border-color:{ORANGE};"><div class="cx-card-title" style="color:{ORANGE};">Decision: ABSTAINED</div>'
                    f'No intervention passed all safety gates. <strong>Human review is required.</strong></div>', unsafe_allow_html=True)

            # Current incident assessment from reconstruction
            if reconstruction:
                st.markdown('<p class="cx-section">Current Incident Assessment</p>', unsafe_allow_html=True)
                st.write(reconstruction.get("summary", "N/A"))

            # Root cause from investigation
            if causes and causes.get("ranked_hypotheses"):
                st.markdown('<p class="cx-section">Most Likely Root Cause</p>', unsafe_allow_html=True)
                top_hyp = causes["ranked_hypotheses"][0]
                st.markdown(f'<span class="cx-hypothesis">HYPOTHESIS</span> **{top_hyp.get("title", "N/A")}** — P={top_hyp.get("posterior_probability", "N/A")}', unsafe_allow_html=True)
                st.write(top_hyp.get("interpretation", ""))
                if len(causes["ranked_hypotheses"]) > 1:
                    alt = causes["ranked_hypotheses"][1]
                    st.markdown(f'<span class="cx-uncertain">ALTERNATIVE</span> **{alt.get("title", "N/A")}** — P={alt.get("posterior_probability", "N/A")}', unsafe_allow_html=True)

            # Decision explanation
            st.markdown('<p class="cx-section">Decision Explanation</p>', unsafe_allow_html=True)
            st.write(decision_report_state.get("decision_explanation", ""))

            # Decision metrics
            dc1, dc2, dc3, dc4 = st.columns(4)
            dc1.metric("Decision ID", str(decision_report_state.get("decision_id", "N/A"))[:8] + "…")
            dc2.metric("Risk Aversion", f"{decision_report_state.get('risk_aversion_weight', 'N/A')}")
            dc3.metric("Status", dr_status)
            dc4.metric("Seed", decision_report_state.get("simulation_seed", "N/A"))

            # Ranked feasible decisions
            ranked = decision_report_state.get("ranked_feasible_decisions", [])
            st.markdown('<p class="cx-section">Ranked Feasible Interventions</p>', unsafe_allow_html=True)
            if ranked:
                st.dataframe(_records_frame(ranked), use_container_width=True)
            else:
                st.caption("No candidate passed all safety gates.")

            # Rejected alternatives with reasons
            rejected = decision_report_state.get("rejected_alternatives", [])
            if rejected:
                st.markdown('<p class="cx-section">Rejected Alternatives & Reasons</p>', unsafe_allow_html=True)
                for rej in rejected:
                    with st.expander(f"Rejected: {INTERVENTION_LABELS.get(rej.get('intervention_name',''), rej.get('intervention_name',''))}"):
                        for reason in rej.get("reasons", []):
                            st.markdown(f"- {reason}")

            # Safety verification matrix
            safety_results = decision_report_state.get("safety_verification_results", {})
            if safety_results:
                st.markdown('<p class="cx-section">Safety Verification Matrix</p>', unsafe_allow_html=True)
                sv_rows = [{"Intervention": iv, **check} for iv, checks in safety_results.items() for check in checks]
                st.dataframe(_records_frame(sv_rows), use_container_width=True)

            # Human review requirements
            if decision_report_state.get("human_review_reasons"):
                st.markdown(f'<p class="cx-section" style="border-left-color:{ORANGE};">Human Review Requirements</p>', unsafe_allow_html=True)
                for reason in decision_report_state["human_review_reasons"]:
                    st.markdown(f"- {reason}")

            # Agent statuses
            agent_statuses = decision_report_state.get("agent_statuses", {})
            if agent_statuses:
                st.markdown('<p class="cx-section">Agent Statuses at Decision Time</p>', unsafe_allow_html=True)
                for agent_name, agent_st in agent_statuses.items():
                    badge = "cx-fact" if agent_st == "COMPLETED" else "cx-alert" if agent_st == "FAILED" else "cx-hypothesis"
                    st.markdown(f'<span class="{badge}">{agent_st}</span> {agent_name.replace("_"," ").title()}', unsafe_allow_html=True)

            # Uncertainty and limitations
            st.markdown('<p class="cx-section">Uncertainty & Limitations</p>', unsafe_allow_html=True)
            constraints = decision_report_state.get("optimization_constraints", [])
            for c in constraints:
                st.markdown(f"- {c}")

        else:
            st.info("Configure operational constraints above and submit to evaluate a safety-gated decision.")

    # ═══ TAB 6: SCENARIO TESTING ═════════════════════════════════════════
    with tabs[6]:
        st.markdown('<p class="cx-section">Interactive Scenario Testing Lab</p>', unsafe_allow_html=True)
        st.warning("All tests use the existing backend simulator and event ingestion. No results are fabricated in the frontend.")

        # Temporal reconstruction with replay
        if reconstruction:
            st.markdown('<p class="cx-section">Timeline Reconstruction & Time-Travel Replay</p>', unsafe_allow_html=True)

            raw_timeline = reconstruction.get("reconstructed_timeline", [])
            replay_timeline = reconstruction.get("replay_timeline", raw_timeline)
            replay_mode = st.toggle("Use duplicate-collapsed replay", value=True, help="Raw timeline remains available; duplicates are never deleted.")
            timeline_frame = _timeline_frame(
                replay_timeline if replay_mode else raw_timeline
            )

            if not timeline_frame.empty and "timestamp" in timeline_frame:
                timeline_frame["timestamp"] = pd.to_datetime(
                    timeline_frame["timestamp"],
                    format="ISO8601",
                    utc=True,
                    errors="raise",
                )
                timeline_frame = timeline_frame.sort_values(["timestamp", "evidence_id"]).reset_index(drop=True)
                replay_count = st.slider("Time-Travel Replay Scrubber", min_value=1, max_value=len(timeline_frame), value=len(timeline_frame), key=f"replay_{scenario_id}")
                st.dataframe(timeline_frame.iloc[:replay_count], use_container_width=True)
                st.caption(f"Showing {replay_count} of {len(timeline_frame)} {'replay' if replay_mode else 'original'} records.")

            if reconstruction.get("alternative_histories"):
                with st.expander("Unresolved Alternative Histories"):
                    st.dataframe(_records_frame(reconstruction["alternative_histories"]), use_container_width=True)

        # Pre-built test scenarios
        st.markdown('<p class="cx-section">Inject Test Incidents</p>', unsafe_allow_html=True)
        test_tabs = st.tabs([
            "Custom Event",
            "Conflicting Telemetry",
            "Delayed Failure",
            "Forged Event",
            "Resource-Constrained",
            "Unknown Safety",
        ])

        # Custom event injection
        with test_tabs[0]:
            st.markdown("Inject a custom synthetic event into the backend via `POST /events`.")
            with st.form(f"custom_event_{scenario_id}"):
                ev_source = st.text_input("Source / Equipment ID", value="DASHBOARD-DEMO")
                ev_type = st.text_input("Event type", value="manual_test_alarm")
                ev_desc = st.text_input("Description", value="Synthetic command-center test event")
                ev_delay = st.number_input("Arrival delay (seconds)", min_value=0, max_value=86400, value=0)
                ev_severity = st.selectbox("Severity", ["low", "medium", "high", "critical"])
                ev_submitted = st.form_submit_button("Ingest Event", type="primary")
            if ev_submitted:
                received = datetime.now(timezone.utc)
                payload = {
                    "timestamp": (received - timedelta(seconds=int(ev_delay))).isoformat(),
                    "received_at": received.isoformat(),
                    "source": ev_source, "event_type": ev_type, "description": ev_desc,
                    "severity": ev_severity,
                    "provenance": {"origin": "dashboard_synthetic_test", "source_verified": False},
                    "metadata": {"scenario_id": str(scenario_id), "equipment_id": ev_source},
                }
                try:
                    api.ingest_event(payload)
                    st.success("Synthetic event ingested. Refresh to see updated reconstruction.")
                except APIError as error:
                    _show_api_error("Event ingestion failed", error)

        # Conflicting cooling telemetry
        with test_tabs[1]:
            st.markdown("Inject **two simultaneous events** with conflicting cooling-system status to test conflict detection.")
            if st.button("Inject Conflicting Cooling Events", type="primary", key="conflict_btn"):
                now = datetime.now(timezone.utc)
                event_time = (now - timedelta(seconds=30)).isoformat()
                events_to_inject = [
                    {"timestamp": event_time, "received_at": now.isoformat(), "source": "COOLING-SENSOR-A", "event_type": "cooling_status_report", "description": "Cooling pump status: RUNNING (normal operation)", "severity": "low", "provenance": {"origin": "sensor_telemetry", "source_verified": True}, "metadata": {"scenario_id": str(scenario_id), "equipment_id": "COOLING-SENSOR-A", "cooling_pump_status": 1}},
                    {"timestamp": event_time, "received_at": now.isoformat(), "source": "COOLING-SENSOR-B", "event_type": "cooling_status_report", "description": "Cooling pump status: FAULT (pump failure detected)", "severity": "high", "provenance": {"origin": "sensor_telemetry", "source_verified": True}, "metadata": {"scenario_id": str(scenario_id), "equipment_id": "COOLING-SENSOR-B", "cooling_pump_status": 0}},
                ]
                success_count = 0
                for ev in events_to_inject:
                    try:
                        api.ingest_event(ev)
                        success_count += 1
                    except APIError:
                        pass
                st.success(f"Injected {success_count}/2 conflicting cooling events. Refresh to see conflict detection.") if success_count else st.error("Failed to inject events.")

        # Delayed secondary failure
        with test_tabs[2]:
            st.markdown("Inject an event with a **large arrival delay** to test delayed-event detection.")
            if st.button("Inject Delayed Failure Event", type="primary", key="delayed_btn"):
                now = datetime.now(timezone.utc)
                payload = {
                    "timestamp": (now - timedelta(hours=2)).isoformat(),
                    "received_at": now.isoformat(),
                    "source": "SECONDARY-MONITOR", "event_type": "delayed_secondary_failure",
                    "description": "Secondary pressure buildup detected in isolated section (delayed report)",
                    "severity": "high",
                    "provenance": {"origin": "delayed_monitoring_system", "source_verified": False},
                    "metadata": {"scenario_id": str(scenario_id), "equipment_id": "SECONDARY-MONITOR", "delay_seconds": 7200},
                }
                try:
                    api.ingest_event(payload)
                    st.success("Delayed failure event injected (2-hour delay). Refresh to see delay detection.")
                except APIError as error:
                    _show_api_error("Injection failed", error)

        # Forged historical event
        with test_tabs[3]:
            st.markdown("Inject a **backdated, unverified event** to test adversarial evidence detection.")
            if st.button("Inject Forged Event", type="primary", key="forged_btn"):
                now = datetime.now(timezone.utc)
                payload = {
                    "timestamp": (now - timedelta(hours=6)).isoformat(),
                    "received_at": now.isoformat(),
                    "source": "UNKNOWN-EXTERNAL", "event_type": "historical_maintenance_record",
                    "description": "Maintenance completed on cooling unit (unverified historical claim)",
                    "severity": "medium",
                    "provenance": {"origin": "external_unverified_source", "source_verified": False},
                    "metadata": {"scenario_id": str(scenario_id), "equipment_id": "UNKNOWN-EXTERNAL", "forged_test": True},
                }
                try:
                    api.ingest_event(payload)
                    st.success("Forged historical event injected. Red-Team agent should flag this on next council run.")
                except APIError as error:
                    _show_api_error("Injection failed", error)

        # Resource-constrained decision
        with test_tabs[4]:
            st.markdown("Run a decision with **zero operators** and all equipment unknown — tests resource constraint handling.")
            if st.button("Run Resource-Constrained Decision", type="primary", key="resource_btn"):
                payload = {
                    "simulation": DEFAULT_SIMULATION,
                    "operational_constraints": {
                        "backup_cooling_available": None,
                        "system_isolation_available": None,
                        "controlled_shutdown_authorized": None,
                        "available_operators": 0,
                        "model_outputs_reviewed": False,
                    },
                    "safety_checks": [],
                    "risk_aversion_weight": 1.0,
                }
                try:
                    with st.spinner("Evaluating with resource constraints…"):
                        result = api.decide(scenario_id, payload)
                    st.session_state["decision_report"] = result
                    expected_status = "ABSTAINED"
                    actual_status = result.get("status", "")
                    if actual_status == expected_status:
                        st.success(f"Decision correctly abstained under resource constraints. Status: {actual_status}")
                    else:
                        st.warning(f"Decision status: {actual_status}")
                    st.write(result.get("decision_explanation", ""))
                    if result.get("human_review_reasons"):
                        st.markdown("**Human review required because:**")
                        for r in result["human_review_reasons"]:
                            st.markdown(f"- {r}")
                except APIError as error:
                    _show_api_error("Decision failed", error)

        # Unknown safety conditions
        with test_tabs[5]:
            st.markdown("Run a decision with **all safety checks set to UNKNOWN** — tests unknown-state handling.")
            if st.button("Run Unknown-Safety Decision", type="primary", key="unknown_btn"):
                payload = {
                    "simulation": DEFAULT_SIMULATION,
                    "operational_constraints": {
                        "backup_cooling_available": None,
                        "system_isolation_available": None,
                        "controlled_shutdown_authorized": None,
                        "available_operators": 10,
                        "model_outputs_reviewed": False,
                    },
                    "safety_checks": [{"check_name": "site_safety_review", "result": "UNKNOWN", "details": "Safety status unknown for testing", "evidence_ids": []}],
                    "risk_aversion_weight": 1.0,
                }
                try:
                    with st.spinner("Evaluating with unknown safety…"):
                        result = api.decide(scenario_id, payload)
                    st.session_state["decision_report"] = result
                    st.success(f"Decision status: {result.get('status', 'N/A')} — UNKNOWN checks should prevent selection.")
                    st.write(result.get("decision_explanation", ""))
                    if result.get("human_review_reasons"):
                        st.markdown("**Human review required because:**")
                        for r in result["human_review_reasons"]:
                            st.markdown(f"- {r}")
                except APIError as error:
                    _show_api_error("Decision failed", error)

    # ═══ TAB 7: AUDIT & REPLAY ═══════════════════════════════════════════
    with tabs[7]:
        st.markdown('<p class="cx-section">Audit History & Decision Trail</p>', unsafe_allow_html=True)

        if audit:
            st.markdown('<p class="cx-section">Persisted Decision History</p>', unsafe_allow_html=True)
            st.dataframe(_records_frame(audit), use_container_width=True)

            safety_history = [
                {"decision_id": entry.get("decision_id"), "recorded_at": entry.get("recorded_at"), "intervention": intervention, **check}
                for entry in audit for intervention, checks in entry.get("decision", {}).get("safety_verification_results", {}).items() for check in checks
            ]
            if safety_history:
                st.markdown('<p class="cx-section">Safety-Check History</p>', unsafe_allow_html=True)
                st.dataframe(_records_frame(safety_history), use_container_width=True)

            warning_history = [
                {"recorded_at": entry.get("recorded_at"), **assessment}
                for entry in audit for agent in entry.get("agent_report", {}).get("agent_results", [])
                if agent.get("agent_name") == "predictor" for assessment in agent.get("outputs", {}).get("anomaly_assessments", []) if assessment.get("anomaly")
            ]
            if warning_history:
                st.markdown('<p class="cx-section">Anomaly Warning History</p>', unsafe_allow_html=True)
                st.dataframe(_records_frame(warning_history), use_container_width=True)
        else:
            st.caption("No persisted decisions for this scenario.")

        st.markdown('<p class="cx-section">Current Backend Events</p>', unsafe_allow_html=True)
        if events_for_scenario:
            st.dataframe(_records_frame(events_for_scenario), use_container_width=True)
        else:
            st.caption("No matching event records returned.")

        if agent_report:
            st.markdown('<p class="cx-section">Session Agent Reports</p>', unsafe_allow_html=True)
            st.json(agent_report, expanded=False)

        if st.session_state.get("decision_report"):
            st.markdown('<p class="cx-section">Session Decision Results</p>', unsafe_allow_html=True)
            st.json(st.session_state["decision_report"], expanded=False)

    with tabs[8]:
        st.markdown(
            '<p class="cx-section">Analyze a Judge-Provided Runtime Scenario</p>',
            unsafe_allow_html=True,
        )
        st.caption(
            "Submit structured scenario JSON to the runtime API. This route does not load "
            "the predefined demo datasets. Unknown actions, missing thresholds, unsupported "
            "sensor types, or incomplete safety evidence must remain UNKNOWN or trigger abstention."
        )
        runtime_json = st.text_area(
            "Scenario request JSON",
            value="",
            placeholder=(
                '{"name":"...","events":[],"sensor_readings":[],"causal_hypotheses":[],'
                '"causal_relationships":[],"available_interventions":[]}'
            ),
            height=320,
            key="runtime_scenario_json",
        )
        if st.button(
            "Analyze Submitted Scenario",
            type="primary",
            key="analyze_runtime_scenario",
        ):
            try:
                request_payload = json.loads(runtime_json)
                if not isinstance(request_payload, dict):
                    raise ValueError("The top-level JSON value must be an object.")
                with st.spinner("Reconstructing and evaluating submitted scenario evidence…"):
                    result = api.analyze_runtime_scenario(request_payload)
                st.session_state["runtime_scenario_report"] = result
            except json.JSONDecodeError as error:
                st.error(f"Invalid scenario JSON at line {error.lineno}, column {error.colno}: {error.msg}")
            except ValueError as error:
                st.error(str(error))
            except APIError as error:
                _show_api_error("Runtime scenario analysis failed", error)

        runtime_report = st.session_state.get("runtime_scenario_report")
        if runtime_report:
            decision = runtime_report.get("decision", {})
            st.metric(
                "Runtime Decision Status",
                decision.get("status", "UNKNOWN"),
                delta="Human review required" if decision.get("human_review_required") else None,
            )
            st.write(decision.get("decision_explanation", ""))
            for limitation in runtime_report.get("unsupported_capabilities", []):
                st.warning(limitation)
            st.markdown("**Uncertainty and assumptions**")
            for assumption in runtime_report.get("assumptions_and_uncertainty", []):
                st.markdown(f"- {assumption}")
            st.markdown("**Full module outputs**")
            st.json(runtime_report, expanded=False)

    # ═══ TAB 9: MULTI-SENSOR TELEMETRY ═══════════════════════════════════
    with tabs[9]:
        st.markdown(
            '<p class="cx-section">Coolant Flow & Electrical Telemetry</p>',
            unsafe_allow_html=True,
        )
        st.caption(
            "Charts and indicators use readings returned by the backend. Missing "
            "sensors remain unavailable; indicators are not diagnoses."
        )
        with st.form(f"sensor_ingest_{scenario_id}"):
            equipment_default = (
                str(scenario_rows["equipment_id"].iloc[-1])
                if not scenario_rows.empty
                else ""
            )
            equipment_id = st.text_input("Equipment ID", value=equipment_default)
            sensor_id = st.text_input("Sensor ID")
            metric = st.selectbox(
                "Measured signal",
                [
                    "coolant_flow_lpm",
                    "coolant_inlet_flow_lpm",
                    "coolant_outlet_flow_lpm",
                    "flow_fluctuation_lpm",
                    "flow_failure_status",
                    "voltage_v",
                    "voltage_fluctuation_v",
                    "current_a",
                    "current_fluctuation_a",
                    "power_kw",
                    "current_overload",
                    "electrical_fault_status",
                ],
            )
            unit_by_metric = {
                "coolant_flow_lpm": "L/min",
                "coolant_inlet_flow_lpm": "L/min",
                "coolant_outlet_flow_lpm": "L/min",
                "flow_fluctuation_lpm": "L/min",
                "flow_failure_status": "status",
                "voltage_v": "V",
                "voltage_fluctuation_v": "V",
                "current_a": "A",
                "current_fluctuation_a": "A",
                "power_kw": "kW",
                "current_overload": "status",
                "electrical_fault_status": "status",
            }
            value_text = st.text_input("Measured value")
            event_timestamp = st.text_input(
                "Event timestamp (ISO-8601 with timezone)",
                value=datetime.now(timezone.utc).isoformat(),
            )
            operating_state = st.selectbox(
                "Equipment operating state",
                ["Unknown", "Operating", "Not operating"],
            )
            rated_current_text = st.text_input(
                "Equipment rated current (A), if configured",
            )
            submit_sensor = st.form_submit_button("Ingest measured reading")

        if submit_sensor:
            try:
                measured_value = float(value_text)
                metadata: dict[str, object] = {"scenario_id": str(scenario_id)}
                if operating_state != "Unknown":
                    metadata["operating"] = operating_state == "Operating"
                if rated_current_text.strip():
                    metadata["rated_current_a"] = float(rated_current_text)
                api.ingest_sensor_reading(
                    {
                        "sensor_id": sensor_id,
                        "equipment_id": equipment_id,
                        "metric": metric,
                        "value": measured_value,
                        "unit": unit_by_metric[metric],
                        "timestamp": event_timestamp,
                        "provenance": {
                            "origin": "command_center_manual_entry",
                            "source_verified": False,
                        },
                        "metadata": metadata,
                    }
                )
                _include_equipment_in_sensor_selection(
                    scenario_id,
                    equipment_id,
                )
                st.session_state["sensor_ingest_notice"] = (
                    "Reading accepted and telemetry refreshed from the backend."
                )
                st.rerun()
            except (APIError, TypeError, ValueError) as error:
                _show_api_error("Sensor ingestion failed", error)

        ingest_notice = st.session_state.pop("sensor_ingest_notice", None)
        if ingest_notice:
            st.success(ingest_notice)

        st.markdown(
            '<p class="cx-section">Development Demo Data</p>',
            unsafe_allow_html=True,
        )
        st.caption(
            "This opt-in generator creates reproducible SYNTHETIC telemetry and "
            "persists it through the same validated ingestion pipeline. It is not "
            "measured plant data and cannot validate equipment health."
        )
        with st.form(f"synthetic_sensor_generation_{scenario_id}"):
            synthetic_equipment_id = st.text_input(
                "Synthetic equipment ID",
                value="DEMO-EQ-001",
            )
            synthetic_count = st.number_input(
                "Samples per sensor",
                min_value=3,
                max_value=500,
                value=12,
                step=1,
            )
            synthetic_seed = st.number_input(
                "Reproducible seed",
                min_value=0,
                max_value=2_147_483_647,
                value=42,
                step=1,
            )
            generate_demo = st.form_submit_button(
                "Generate and persist SYNTHETIC telemetry"
            )
        if generate_demo:
            try:
                with st.spinner("Generating and persisting synthetic telemetry…"):
                    result = api.generate_synthetic_telemetry(
                        sample_count=int(synthetic_count),
                        seed=int(synthetic_seed),
                        equipment_id=synthetic_equipment_id,
                        scenario_id=scenario_id,
                    )
                _include_equipment_in_sensor_selection(
                    scenario_id,
                    str(result["equipment_id"]),
                )
                st.session_state["sensor_generation_notice"] = (
                    f"Persisted {result['inserted_readings']} SYNTHETIC readings; "
                    f"{result['already_persisted_readings']} deterministic readings "
                    "were already stored."
                )
                st.rerun()
            except APIError as error:
                _show_api_error("Synthetic telemetry generation failed", error)
        generation_notice = st.session_state.pop(
            "sensor_generation_notice",
            None,
        )
        if generation_notice:
            st.success(generation_notice)

        if sensor_api_error:
            st.error(f"Multi-sensor backend data unavailable: {sensor_api_error}")
            analysis_records: list[dict[str, Any]] = []
        else:
            analysis_records = sensor_analysis.get("records", [])
        if not sensor_api_error and not analysis_records:
            st.info(
                "No sensor measurements are currently stored by the backend. Ingest "
                "measured records or explicitly generate SYNTHETIC development data."
            )
        equipment_ids = sorted(
            {
                str(item["equipment_id"])
                for item in analysis_records
                if item.get("equipment_id")
            }
        )
        equipment_selection_key = f"multi_sensor_equipment_{scenario_id}"
        if equipment_selection_key not in st.session_state:
            st.session_state[equipment_selection_key] = equipment_ids
        selected_equipment = st.multiselect(
            "Equipment streams",
            equipment_ids,
            key=equipment_selection_key,
            disabled=not equipment_ids,
        )
        visible_records = [
            item
            for item in analysis_records
            if item.get("equipment_id") in selected_equipment
        ]

        st.markdown(
            '<p class="cx-section">Equipment Health Overview</p>',
            unsafe_allow_html=True,
        )
        health_overview = [
            item
            for item in sensor_analysis.get("equipment_overview", [])
            if item.get("equipment_id") in selected_equipment
        ]
        if health_overview:
            st.dataframe(pd.DataFrame(health_overview), use_container_width=True)
        elif not sensor_api_error:
            st.info("No equipment health evidence is available for this selection.")

        for metrics, title, y_label in (
            (
                {
                    "coolant_flow_lpm",
                    "coolant_inlet_flow_lpm",
                    "coolant_outlet_flow_lpm",
                },
                "Coolant Flow Rate",
                "Flow (L/min)",
            ),
            ({"voltage_v"}, "Electrical Voltage", "Voltage (V)"),
            ({"current_a"}, "Electrical Current", "Current (A)"),
        ):
            figure = _sensor_trend_figure(
                visible_records,
                metrics=metrics,
                title=f"{title} · Backend readings",
                y_label=y_label,
            )
            if figure is None:
                state = "unavailable while the backend is disconnected" if sensor_api_error else "no measurements available"
                st.info(f"{title}: {state}.")
            else:
                st.plotly_chart(figure, use_container_width=True)

        st.markdown(
            '<p class="cx-section">Combined Sensor Correlation View</p>',
            unsafe_allow_html=True,
        )
        correlations = [
            item
            for item in sensor_analysis.get("cross_sensor_correlations", [])
            if item.get("equipment_id") in selected_equipment
        ]
        if correlations:
            correlation_frame = pd.DataFrame(correlations)
            st.dataframe(correlation_frame, use_container_width=True)
            correlation_figure = go.Figure(
                go.Bar(
                    x=[
                        f"{item['equipment_id']}: {item['left_metric']} / {item['right_metric']}"
                        for item in correlations
                    ],
                    y=[item["pearson_r"] for item in correlations],
                    marker_color=CY,
                )
            )
            correlation_figure.update_layout(yaxis_title="Pearson r")
            _plotly_layout(
                correlation_figure,
                "Backward-Aligned Sensor Associations",
            )
            st.plotly_chart(correlation_figure, use_container_width=True)
        elif not sensor_api_error:
            st.info(
                "Correlation estimates require at least three same-equipment, "
                "backward-aligned pairs; correlation is not causation."
            )

        indicators = [
            item
            for item in sensor_analysis.get("anomaly_indicators", [])
            if item.get("equipment_id") in selected_equipment
        ]
        st.markdown(
            '<p class="cx-section">Sensor Anomaly Indicators</p>',
            unsafe_allow_html=True,
        )
        if indicators:
            st.dataframe(pd.DataFrame(indicators), use_container_width=True)
        elif not sensor_api_error:
            st.info(
                "No configured indicator was raised. This is not a health certification."
            )
        with st.expander("Data quality, alignment and limitations"):
            if sensor_analysis.get("missing_intervals"):
                st.dataframe(
                    pd.DataFrame(sensor_analysis["missing_intervals"]),
                    use_container_width=True,
                )
            if sensor_analysis.get("rejected_records"):
                st.dataframe(
                    pd.DataFrame(sensor_analysis["rejected_records"]),
                    use_container_width=True,
                )
            if sensor_analysis.get("duplicate_records"):
                st.dataframe(
                    pd.DataFrame(sensor_analysis["duplicate_records"]),
                    use_container_width=True,
                )
            if sensor_analysis.get("conflicting_observations"):
                st.dataframe(
                    pd.DataFrame(
                        sensor_analysis["conflicting_observations"]
                    ),
                    use_container_width=True,
                )
            for limitation in sensor_analysis.get("limitations", []):
                st.markdown(f"- {limitation}")


if __name__ == "__main__":
    main()
