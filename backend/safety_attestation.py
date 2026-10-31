"""Verification for trusted, reviewer-signed site safety checks."""

import hashlib
import hmac
import json
import os
from uuid import UUID

from models import SafetyCheckInput

ATTESTATION_KEY_ENV = "CHRONOS_SAFETY_ATTESTATION_KEY"


def verify_safety_attestation(check: SafetyCheckInput, scenario_id: UUID) -> bool:
    if check.result != "PASS" or not check.reviewer_id or not check.attestation:
        return False
    secret = os.environ.get(ATTESTATION_KEY_ENV)
    if not secret or len(secret.encode("utf-8")) < 32:
        return False

    payload = json.dumps(
        {
            "check_name": check.check_name,
            "result": check.result,
            "details": check.details,
            "evidence_ids": check.evidence_ids,
            "reviewer_id": check.reviewer_id,
            "scenario_id": str(scenario_id),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    expected = hmac.new(secret.encode("utf-8"), payload, hashlib.sha256).hexdigest()
    return hmac.compare_digest(check.attestation, expected)
