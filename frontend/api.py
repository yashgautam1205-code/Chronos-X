"""HTTP client for the CHRONOS-X FastAPI service."""

from typing import Any
from uuid import UUID

import httpx


class APIError(RuntimeError):
    """Raised when the backend cannot provide a valid response."""


class ChronosAPI:
    def __init__(self, base_url: str, *, timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, object] | None = None,
        json: dict[str, object] | None = None,
    ) -> Any:
        try:
            response = httpx.request(
                method,
                f"{self.base_url}{path}",
                params=params,
                json=json,
                timeout=self.timeout,
            )
        except httpx.HTTPError as error:
            raise APIError(f"Could not connect to CHRONOS-X API: {error}") from error
        if not response.is_success:
            try:
                details = response.json().get("detail", response.text)
            except ValueError:
                details = response.text
            raise APIError(f"CHRONOS-X API returned {response.status_code}: {details}")
        try:
            return response.json()
        except ValueError as error:
            raise APIError("CHRONOS-X API returned an invalid JSON response") from error

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def events(self) -> list[dict[str, Any]]:
        return self._request("GET", "/events", params={"limit": 1000})

    def ingest_event(self, event: dict[str, object]) -> dict[str, Any]:
        return self._request("POST", "/events", json=event)

    def ingest_sensor_reading(
        self,
        reading: dict[str, object],
    ) -> dict[str, Any]:
        return self._request("POST", "/sensors", json=reading)

    def generate_synthetic_telemetry(
        self,
        *,
        sample_count: int,
        seed: int,
        equipment_id: str,
        scenario_id: UUID,
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/sensors/synthetic/generate",
            json={
                "sample_count": sample_count,
                "seed": seed,
                "equipment_id": equipment_id,
                "scenario_id": str(scenario_id),
            },
        )

    def sensor_readings(self, scenario_id: UUID) -> list[dict[str, Any]]:
        return self._request(
            "GET",
            "/sensors",
            params={"scenario_id": str(scenario_id), "limit": 1000},
        )

    def analyze_sensors(
        self,
        readings: list[dict[str, object]],
        *,
        rated_current_a: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/sensors/analyze",
            json={
                "sensor_readings": readings,
                "rated_current_a": rated_current_a or {},
            },
        )

    def scenario_sensor_analysis(self, scenario_id: UUID) -> dict[str, Any]:
        return self._request(
            "GET",
            f"/scenarios/{scenario_id}/sensor-analysis",
        )

    def sensor_analysis(self) -> dict[str, Any]:
        """Fetch all persisted sensor telemetry for the monitoring dashboard."""
        return self._request("GET", "/sensors/analysis")

    def reconstruct(self, scenario_id: UUID) -> dict[str, Any]:
        return self._request("GET", f"/scenarios/{scenario_id}/reconstruct")

    def causes(self, scenario_id: UUID) -> dict[str, Any]:
        return self._request("GET", f"/scenarios/{scenario_id}/causes")

    def simulate(
        self,
        scenario_id: UUID,
        request: dict[str, object],
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            f"/scenarios/{scenario_id}/simulate",
            json=request,
        )

    def run_agents(
        self,
        scenario_id: UUID,
        request: dict[str, object],
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            f"/scenarios/{scenario_id}/agents/run",
            json={"simulation": request},
        )

    def decide(
        self,
        scenario_id: UUID,
        request: dict[str, object],
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            f"/scenarios/{scenario_id}/decide",
            json=request,
        )

    def analyze_runtime_scenario(
        self,
        request: dict[str, object],
    ) -> dict[str, Any]:
        return self._request("POST", "/scenarios/analyze", json=request)

    def audit(self, scenario_id: UUID) -> list[dict[str, Any]]:
        return self._request("GET", f"/scenarios/{scenario_id}/audit")
