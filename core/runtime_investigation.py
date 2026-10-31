"""Scenario-specific, evidence-linked causal review without fixed cause priors."""

from typing import Any
from uuid import UUID

import networkx as nx

from models import RuntimeCausalHypothesis, RuntimeCausalRelationship


def _condition_matches(value: float, operator: str, expected: float) -> bool:
    return {
        "eq": value == expected,
        "ne": value != expected,
        "gt": value > expected,
        "gte": value >= expected,
        "lt": value < expected,
        "lte": value <= expected,
    }[operator]


def investigate_runtime_scenario(
    scenario_id: UUID,
    reconstruction: dict[str, Any],
    hypotheses: list[RuntimeCausalHypothesis],
    relationships: list[RuntimeCausalRelationship],
    infrastructure: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Evaluate only scenario-supplied hypotheses and relationships.

    The evidence balance is a transparent ranking heuristic, not a probability.
    """
    if str(reconstruction.get("scenario_id")) != str(scenario_id):
        raise ValueError("Reconstruction scenario_id does not match investigation scenario_id")

    timeline = reconstruction.get("reconstructed_timeline", [])
    reliability = {
        str(item["evidence_id"]): float(item["score"])
        for item in reconstruction.get("evidence_reliability", [])
    }
    graph = nx.DiGraph()
    for relation in relationships:
        graph.add_edge(
            relation.cause,
            relation.effect,
            mechanism=relation.mechanism,
        )

    events_by_type: dict[str, list[dict[str, Any]]] = {}
    sensors_by_metric: dict[str, list[dict[str, Any]]] = {}
    assets_by_id = {
        str(asset.get("asset_id")): asset
        for asset in (infrastructure or [])
    }
    for item in timeline:
        if item.get("kind") == "event":
            events_by_type.setdefault(str(item.get("event_type", "")), []).append(item)
        elif item.get("kind") == "sensor_reading":
            sensors_by_metric.setdefault(str(item.get("metric", "")), []).append(item)

    ranked: list[dict[str, Any]] = []
    for hypothesis in hypotheses:
        supporting: list[dict[str, Any]] = []
        contradicting: list[dict[str, Any]] = []
        missing: list[str] = []

        for event_type in hypothesis.expected_event_types:
            matches = events_by_type.get(event_type, [])
            if matches:
                supporting.append(
                    {
                        "rule": f"event_type == {event_type}",
                        "evidence": [
                            {
                                "evidence_id": str(item["evidence_id"]),
                                "kind": "event",
                                "event_type": event_type,
                                "source": item.get("source"),
                                "reliability_score": reliability.get(
                                    str(item["evidence_id"]), 0.5
                                ),
                            }
                            for item in matches
                        ],
                    }
                )
            else:
                missing.append(f"Expected event type not observed: {event_type}")

        for condition in hypothesis.sensor_conditions:
            observations = sensors_by_metric.get(condition.metric, [])
            numeric_observations = []
            for item in observations:
                try:
                    value = float(item["value"])
                except (KeyError, TypeError, ValueError):
                    continue
                numeric_observations.append((item, value))
            matching = [
                (item, value)
                for item, value in numeric_observations
                if _condition_matches(value, condition.operator, condition.value)
            ]
            nonmatching = [
                (item, value)
                for item, value in numeric_observations
                if not _condition_matches(value, condition.operator, condition.value)
            ]
            rule = condition.description or (
                f"{condition.metric} {condition.operator} {condition.value}"
            )
            if matching:
                supporting.append(
                    {
                        "rule": rule,
                        "evidence": [
                            {
                                "evidence_id": str(item["evidence_id"]),
                                "kind": "sensor_reading",
                                "metric": condition.metric,
                                "value": value,
                                "source": item.get("source"),
                                "reliability_score": reliability.get(
                                    str(item["evidence_id"]), 0.5
                                ),
                            }
                            for item, value in matching
                        ],
                    }
                )
            if nonmatching:
                contradicting.append(
                    {
                        "rule": rule,
                        "evidence": [
                            {
                                "evidence_id": str(item["evidence_id"]),
                                "kind": "sensor_reading",
                                "metric": condition.metric,
                                "value": value,
                                "source": item.get("source"),
                                "reliability_score": reliability.get(
                                    str(item["evidence_id"]), 0.5
                                ),
                            }
                            for item, value in nonmatching
                        ],
                    }
                )
            if not numeric_observations:
                missing.append(f"No numeric observations for metric: {condition.metric}")

        for condition in hypothesis.asset_conditions:
            asset = assets_by_id.get(condition.asset_id)
            state = asset.get("state", {}) if asset else {}
            if condition.state_key not in state:
                missing.append(
                    f"No submitted state for {condition.asset_id}.{condition.state_key}"
                )
                continue
            observed = state[condition.state_key]
            matches = observed == condition.value
            if condition.operator == "ne":
                matches = not matches
            entry = {
                "rule": condition.description
                or (
                    f"{condition.asset_id}.{condition.state_key} "
                    f"{condition.operator} {condition.value}"
                ),
                "evidence": [
                    {
                        "kind": "submitted_infrastructure_state",
                        "asset_id": condition.asset_id,
                        "state_key": condition.state_key,
                        "observed_value": observed,
                        "source": "unverified_runtime_scenario_input",
                    }
                ],
            }
            (supporting if matches else contradicting).append(entry)

        support_weight = sum(
            float(item.get("reliability_score", 0.5))
            for rule in supporting
            for item in rule["evidence"]
        )
        contradiction_weight = sum(
            float(item.get("reliability_score", 0.5))
            for rule in contradicting
            for item in rule["evidence"]
        )
        ranked.append(
            {
                "hypothesis_id": hypothesis.hypothesis_id,
                "title": hypothesis.title,
                "source": "scenario_supplied",
                "rank": 0,
                "supporting_evidence": supporting,
                "contradicting_evidence": contradicting,
                "missing_evidence": missing,
                "evidence_balance_score": round(
                    support_weight - contradiction_weight, 6
                ),
                "confidence": None,
                "interpretation": (
                    "Evidence-balance ranking only; no calibrated likelihoods or "
                    "causal certainty are available."
                ),
                "causal_relationships": [
                    {
                        "cause": source,
                        "effect": target,
                        "mechanism": attributes["mechanism"],
                    }
                    for source, target, attributes in graph.edges(data=True)
                    if source == hypothesis.hypothesis_id
                    or target == hypothesis.hypothesis_id
                ],
            }
        )

    uncertainty_alternatives = (
        (
            "unmodeled_cause",
            "Unmodeled root cause",
            "The supplied hypothesis set may omit the actual mechanism.",
        ),
        (
            "hidden_confounder",
            "Hidden confounder or omitted interaction",
            "An unobserved variable or interaction may explain the observed associations.",
        ),
    )
    existing_hypothesis_ids = {item["hypothesis_id"] for item in ranked}
    for hypothesis_id, title, explanation in uncertainty_alternatives:
        if len(ranked) >= 2:
            break
        if hypothesis_id in existing_hypothesis_ids:
            continue
        ranked.append(
            {
                "hypothesis_id": hypothesis_id,
                "title": title,
                "source": "uncertainty_alternative",
                "rank": 0,
                "supporting_evidence": [],
                "contradicting_evidence": [],
                "missing_evidence": [
                    "No scenario-supplied evidence rule or structural model evaluates this alternative."
                ],
                "evidence_balance_score": 0.0,
                "confidence": None,
                "interpretation": (
                    f"{explanation} This is an explicit uncertainty alternative, "
                    "not an evidence-supported causal conclusion."
                ),
                "causal_relationships": [],
            }
        )
        existing_hypothesis_ids.add(hypothesis_id)

    ranked.sort(
        key=lambda item: (
            item["source"] == "uncertainty_alternative",
            -item["evidence_balance_score"],
            -len(item["supporting_evidence"]),
            item["hypothesis_id"],
        )
    )
    for rank, item in enumerate(ranked, start=1):
        item["rank"] = rank

    graph_relationships = [
        {
            "cause": source,
            "effect": target,
            "mechanism": attributes["mechanism"],
        }
        for source, target, attributes in sorted(
            graph.edges(data=True), key=lambda edge: (edge[0], edge[1])
        )
    ]
    try:
        cycle = nx.find_cycle(graph)
        cycles = [[node for node, _ in cycle] + [cycle[0][0]]]
    except nx.NetworkXNoCycle:
        cycles = []
    return {
        "scenario_id": scenario_id,
        "possible_root_causes": ranked,
        "ranked_hypotheses": ranked,
        "causal_relationships": graph_relationships,
        "evidence_summary": {
            "timeline_records": len(timeline),
            "configured_hypotheses": len(hypotheses),
            "observed_event_types": sorted(events_by_type),
            "observed_sensor_metrics": sorted(sensors_by_metric),
            "causal_graph_acyclic": not cycles,
            "causal_graph_cycles": cycles,
            "reconstruction_summary": reconstruction.get("summary", ""),
        },
        "assumptions": [
            "Only hypotheses, evidence rules, and causal relationships submitted with this scenario were evaluated.",
            "Evidence-balance scores are reliability-weighted rule matches, not probabilities or calibrated causal estimates.",
            "Unmodeled-cause and hidden-confounder alternatives are included when fewer than two scenario hypotheses are supplied; they are not evidence-supported findings.",
            "Unspecified mechanisms and hidden confounders are not inferred from free-text descriptions.",
            "Causal relationships are scenario-supplied claims; temporal order or graph paths do not prove causation.",
        ],
        "summary": (
            f"Evaluated {len(ranked)} scenario-supplied hypotheses against "
            f"{len(timeline)} reconstructed evidence records; no causal probabilities were asserted."
        ),
    }
