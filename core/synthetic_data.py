"""Reproducible synthetic industrial sensor and event data generation."""

import argparse
import csv
import json
import random
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid5

from models import IndustrialEvent

SCENARIOS = (
    "normal",
    "gradual_temperature_increase",
    "sudden_pressure_increase",
    "cooling_pump_failure",
    "power_interruption",
    "equipment_malfunction",
    "multiple_simultaneous_faults",
    "sensor_malfunction",
)

FAULT_START_SCENARIOS = frozenset(SCENARIOS[1:])
DEFAULT_START_TIME = datetime(2026, 1, 1, tzinfo=timezone.utc)
SENSOR_COLUMNS = (
    "temperature_c",
    "pressure_kpa",
    "vibration_mm_s",
    "cooling_pump_status",
    "power_supply_status",
    "smoke_detected",
)
SENSOR_ID_FIELDS = (
    ("temperature_sensor_id", "TEMP"),
    ("pressure_sensor_id", "PRES"),
    ("vibration_sensor_id", "VIB"),
    ("cooling_pump_sensor_id", "PUMP"),
    ("power_supply_sensor_id", "POWER"),
    ("smoke_sensor_id", "SMOKE"),
)


def _fault_start(scenario: str, sequence_length: int) -> int:
    if scenario == "gradual_temperature_increase":
        return 1
    return max(1, sequence_length // 2)


def _record(
    *,
    scenario: str,
    sequence_index: int,
    sequence_length: int,
    timestamp: datetime,
    rng: random.Random,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any], bool]:
    equipment_number = SCENARIOS.index(scenario) + 1
    equipment_id = f"EQ-{equipment_number:03d}"
    fault_start = _fault_start(scenario, sequence_length)
    fault_active = scenario in FAULT_START_SCENARIOS and sequence_index >= fault_start

    temperature = rng.gauss(72.0, 1.5)
    pressure = rng.gauss(100.0, 2.0)
    vibration = max(0.0, rng.gauss(2.0, 0.2))
    pump_status = 1
    power_status = 1
    smoke_detected = 0

    if scenario == "gradual_temperature_increase":
        temperature += sequence_index * 0.08
    elif scenario == "sudden_pressure_increase" and fault_active:
        pressure += 35.0 + rng.gauss(0.0, 1.0)
    elif scenario == "cooling_pump_failure" and fault_active:
        pump_status = 0
        temperature += 12.0
    elif scenario == "power_interruption" and fault_active:
        power_status = 0
        pump_status = 0
    elif scenario == "equipment_malfunction" and fault_active:
        vibration += 7.0 + abs(rng.gauss(0.0, 0.5))
    elif scenario == "multiple_simultaneous_faults" and fault_active:
        temperature += 15.0
        pressure += 30.0
        vibration += 6.0
        pump_status = 0
        smoke_detected = 1
    elif scenario == "sensor_malfunction" and fault_active:
        pressure = 100.0

    reading_id = str(
        uuid5(
            NAMESPACE_URL,
            f"chronos-x:{seed}:reading:{scenario}:{sequence_index}",
        )
    )
    sensor_ids = {
        field: f"{equipment_id}-{suffix}" for field, suffix in SENSOR_ID_FIELDS
    }
    row: dict[str, Any] = {
        "reading_id": reading_id,
        "timestamp": timestamp.isoformat(),
        "received_at": timestamp.isoformat(),
        "provenance": "synthetic_generator",
        "equipment_id": equipment_id,
        **sensor_ids,
        "temperature_c": round(temperature, 3),
        "pressure_kpa": round(pressure, 3),
        "vibration_mm_s": round(vibration, 3),
        "cooling_pump_status": pump_status,
        "power_supply_status": power_status,
        "smoke_detected": smoke_detected,
    }
    label = {
        "reading_id": reading_id,
        "scenario_id": str(
            uuid5(NAMESPACE_URL, f"chronos-x:{seed}:scenario:{scenario}")
        ),
        "scenario_name": scenario,
        "ground_truth_fault": fault_active,
        "fault_type": scenario if fault_active else "none",
    }
    return row, label, fault_active


def generate_synthetic_data(
    record_count: int = 10_000,
    seed: int = 42,
    start_time: datetime = DEFAULT_START_TIME,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[IndustrialEvent]]:
    """Generate sequential readings, separate labels, and fault onset events."""
    if record_count < 1:
        raise ValueError("record_count must be at least 1")
    if start_time.tzinfo is None:
        raise ValueError("start_time must include timezone information")

    rng = random.Random(seed)
    sequence_lengths = {
        scenario: (record_count + len(SCENARIOS) - 1 - index) // len(SCENARIOS)
        for index, scenario in enumerate(SCENARIOS)
    }
    sequence_indices = {scenario: 0 for scenario in SCENARIOS}
    readings: list[dict[str, Any]] = []
    labels: list[dict[str, Any]] = []
    events: list[IndustrialEvent] = []
    fault_events_created: set[str] = set()

    for index in range(record_count):
        scenario = SCENARIOS[index % len(SCENARIOS)]
        sequence_index = sequence_indices[scenario]
        sequence_indices[scenario] += 1
        timestamp = start_time + timedelta(seconds=index)
        row, label, _ = _record(
            scenario=scenario,
            sequence_index=sequence_index,
            sequence_length=sequence_lengths[scenario],
            timestamp=timestamp,
            rng=rng,
            seed=seed,
        )
        readings.append(row)
        labels.append(label)

        if (
            scenario in FAULT_START_SCENARIOS
            and sequence_index == _fault_start(scenario, sequence_lengths[scenario])
            and scenario not in fault_events_created
        ):
            equipment_id = row["equipment_id"]
            events.append(
                IndustrialEvent(
                    id=UUID(
                        str(
                            uuid5(
                                NAMESPACE_URL,
                                f"chronos-x:{seed}:event:{scenario}",
                            )
                        )
                    ),
                    timestamp=timestamp,
                    received_at=timestamp
                    + timedelta(seconds=12 if scenario == "sudden_pressure_increase" else 0),
                    source=equipment_id,
                    event_type=scenario,
                    description=f"Simulated {scenario.replace('_', ' ')}.",
                    severity="high" if scenario == "multiple_simultaneous_faults" else "medium",
                    provenance={
                        "origin": "synthetic_generator",
                        "source_verified": True,
                    },
                    metadata={
                        "equipment_id": equipment_id,
                        "scenario_id": label["scenario_id"],
                        "fault_type": scenario,
                    },
                )
            )
            fault_events_created.add(scenario)

    return readings, labels, events


def generate_synthetic_telemetry(
    *,
    sample_count: int = 12,
    seed: int = 42,
    sample_interval_seconds: float = 10.0,
    start_time: datetime | None = None,
    equipment_id: str = "DEMO-EQ-001",
    scenario_id: UUID | None = None,
) -> list[dict[str, Any]]:
    """Generate an explicitly synthetic, reproducible five-category demo stream."""
    if sample_count < 3:
        raise ValueError("sample_count must be at least 3")
    if sample_interval_seconds <= 0:
        raise ValueError("sample_interval_seconds must be positive")
    if not equipment_id.strip():
        raise ValueError("equipment_id must not be blank")
    equipment_id = equipment_id.strip()
    if start_time is None:
        start_time = datetime.now(timezone.utc) - timedelta(
            seconds=(sample_count - 1) * sample_interval_seconds
        )
    if start_time.tzinfo is None:
        raise ValueError("start_time must include timezone information")

    rng = random.Random(seed)
    demo_scenario_id = scenario_id or uuid5(
        NAMESPACE_URL,
        f"chronos-x:synthetic-demo:{seed}:{equipment_id}",
    )
    sensor_metrics = (
        ("temperature_c", "degC", "TEMP"),
        ("pressure_kpa", "kPa", "PRES"),
        ("vibration_mm_s", "mm/s", "VIB"),
        ("coolant_flow_lpm", "L/min", "FLOW"),
        ("voltage_v", "V", "VOLT"),
        ("current_a", "A", "CURRENT"),
    )
    records: list[dict[str, Any]] = []
    for index in range(sample_count):
        timestamp = start_time + timedelta(
            seconds=index * sample_interval_seconds
        )
        shared_variation = rng.uniform(-1.0, 1.0)
        independent_noise = {
            metric: rng.gauss(0.0, noise)
            for metric, noise in (
                ("temperature_c", 0.12),
                ("pressure_kpa", 0.2),
                ("vibration_mm_s", 0.03),
                ("coolant_flow_lpm", 0.15),
                ("voltage_v", 0.08),
                ("current_a", 0.04),
            )
        }
        values = {
            "temperature_c": 72.0 + 0.35 * shared_variation + independent_noise["temperature_c"],
            "pressure_kpa": 100.0 + 0.7 * shared_variation + independent_noise["pressure_kpa"],
            "vibration_mm_s": max(
                0.0,
                2.0 + 0.08 * shared_variation + independent_noise["vibration_mm_s"],
            ),
            "coolant_flow_lpm": max(
                0.0,
                30.0 - 0.9 * shared_variation + independent_noise["coolant_flow_lpm"],
            ),
            "voltage_v": max(
                0.0,
                230.0 + 0.25 * shared_variation + independent_noise["voltage_v"],
            ),
            "current_a": max(
                0.0,
                5.0 + 0.18 * shared_variation + independent_noise["current_a"],
            ),
        }
        for metric, unit, sensor_suffix in sensor_metrics:
            reading_id = uuid5(
                NAMESPACE_URL,
                (
                    f"chronos-x:synthetic-telemetry:{seed}:{equipment_id}:"
                    f"{demo_scenario_id}:{sample_interval_seconds}:{index}:{metric}"
                ),
            )
            records.append(
                {
                    "id": str(reading_id),
                    "equipment_id": equipment_id,
                    "sensor_id": f"{equipment_id}-{sensor_suffix}",
                    "metric": metric,
                    "value": round(values[metric], 4),
                    "unit": unit,
                    "timestamp": timestamp.isoformat(),
                    "scenario_id": str(demo_scenario_id),
                    "provenance": {
                        "origin": "SYNTHETIC_CHRONOS_DEMO_GENERATOR",
                        "synthetic": True,
                        "source_verified": False,
                        "seed": seed,
                    },
                    "metadata": {
                        "data_classification": "SYNTHETIC",
                        "generator": "core.synthetic_data.generate_synthetic_telemetry",
                        "seed": seed,
                        "sample_index": index,
                        "scenario_id": str(demo_scenario_id),
                        "operating": True,
                    },
                }
            )
    return records


def iter_sensor_records(
    scenario: str = "normal",
    seed: int = 42,
    start_time: datetime = DEFAULT_START_TIME,
) -> Iterator[dict[str, Any]]:
    """Yield an endless sequence of readings for dashboard or stream testing."""
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario}")
    if start_time.tzinfo is None:
        raise ValueError("start_time must include timezone information")

    rng = random.Random(seed)
    sequence_index = 0
    while True:
        row, _, _ = _record(
            scenario=scenario,
            sequence_index=sequence_index,
            sequence_length=max(2, sequence_index + 1),
            timestamp=start_time + timedelta(seconds=sequence_index),
            rng=rng,
            seed=seed,
        )
        yield row
        sequence_index += 1


def save_synthetic_data(
    output_dir: Path,
    record_count: int = 10_000,
    seed: int = 42,
) -> tuple[Path, Path, Path, Path, Path]:
    """Write feature readings and separate ground-truth labels in CSV and JSON."""
    readings, labels, events = generate_synthetic_data(record_count, seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    readings_csv = output_dir / "sensor_readings.csv"
    readings_json = output_dir / "sensor_readings.json"
    labels_csv = output_dir / "ground_truth_labels.csv"
    labels_json = output_dir / "ground_truth_labels.json"
    events_json = output_dir / "industrial_events.json"

    for path, records in ((readings_csv, readings), (labels_csv, labels)):
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)

    readings_json.write_text(
        json.dumps(readings, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    labels_json.write_text(
        json.dumps(labels, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    events_json.write_text(
        json.dumps(
            [event.model_dump(mode="json") for event in events],
            indent=2,
            ensure_ascii=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return readings_csv, readings_json, labels_csv, labels_json, events_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    paths = save_synthetic_data(args.output_dir, args.records, args.seed)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
