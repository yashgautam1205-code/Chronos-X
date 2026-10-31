"""Investigator agent backed by the Bayesian causal investigation engine."""

from uuid import UUID

from core.causal_investigation import investigate_causes
from models import AgentResult


def run_investigator_agent(
    scenario_id: UUID,
    reconstruction: dict[str, object],
) -> AgentResult:
    report = investigate_causes(scenario_id, reconstruction)
    top_hypothesis = report["ranked_hypotheses"][0]
    evidence_ids = sorted(
        {
            str(evidence["evidence_id"])
            for hypothesis in report["ranked_hypotheses"]
            for category in ("supporting_evidence", "contradicting_evidence")
            for item in hypothesis[category]
            for evidence in item["evidence"]
        }
    )
    return AgentResult(
        agent_name="investigator",
        status="COMPLETED",
        summary=report["summary"],
        findings=[
            {
                "hypothesis_id": top_hypothesis["hypothesis_id"],
                "posterior_probability": top_hypothesis["posterior_probability"],
                "interpretation": top_hypothesis["interpretation"],
            }
        ],
        evidence_ids=evidence_ids,
        outputs=report,
    )
