"""Data collection adapters plus a realistic demo telemetry generator.

Real farms can export the documented CSV schema from their environmental
controller. The simulator makes the full product reproducible without exposing
private farm records and is intentionally isolated from the modeling code.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .config import RANDOM_SEED, RAW_DATA_PATH


SPECIES_PROFILE = {
    "oyster": {"base_yield": 18.0, "ideal_temp": 18.0, "ideal_humidity": 90.0},
    "lion's_mane": {"base_yield": 11.5, "ideal_temp": 17.0, "ideal_humidity": 88.0},
    "shiitake": {"base_yield": 13.0, "ideal_temp": 16.0, "ideal_humidity": 86.0},
}


def generate_demo_data(rows: int = 4200, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Simulate batch-day observations with seasonal and operational structure."""
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2024-01-01", "2026-06-30", periods=rows)
    species = rng.choice(list(SPECIES_PROFILE), rows, p=[0.50, 0.25, 0.25])
    room_id = rng.choice(["GR-01", "GR-02", "GR-03", "GR-04"], rows)
    substrate = np.where(
        species == "shiitake",
        rng.choice(["hardwood_sawdust", "supplemented_sawdust"], rows, p=[0.45, 0.55]),
        rng.choice(["straw", "supplemented_sawdust"], rows, p=[0.55, 0.45]),
    )
    flush_number = rng.choice([1, 2, 3], rows, p=[0.52, 0.33, 0.15])
    room_age_days = np.clip(
        rng.normal(8 + flush_number * 4 + (species == "shiitake") * 5, 3.2, rows), 3, 31
    ).round(1)

    day_angle = 2 * np.pi * timestamps.dayofyear.to_numpy() / 365.25
    room_temp_bias = pd.Series(room_id).map(
        {"GR-01": -0.3, "GR-02": 0.5, "GR-03": 0.0, "GR-04": 0.8}
    ).to_numpy()
    temperature = 18 + 2.0 * np.sin(day_angle) + room_temp_bias + rng.normal(0, 1.7, rows)
    humidity = 88 - 1.4 * np.sin(day_angle) + rng.normal(0, 4.2, rows)
    co2 = np.clip(rng.lognormal(np.log(850), 0.28, rows), 420, 2400)
    substrate_moisture = np.clip(rng.normal(61, 6, rows), 38, 78)
    fresh_air = np.clip(5.4 - (co2 - 850) / 600 + rng.normal(0, 0.8, rows), 1.0, 9.0)
    light_hours = np.clip(rng.normal(9.5, 1.4, rows), 4, 14)

    base = np.array([SPECIES_PROFILE[s]["base_yield"] for s in species])
    ideal_temp = np.array([SPECIES_PROFILE[s]["ideal_temp"] for s in species])
    ideal_humidity = np.array([SPECIES_PROFILE[s]["ideal_humidity"] for s in species])
    flush_factor = np.choose(flush_number - 1, [1.0, 0.72, 0.48])
    temp_penalty = 0.45 * np.abs(temperature - ideal_temp) ** 1.35
    humidity_penalty = 0.12 * np.abs(humidity - ideal_humidity) ** 1.25
    co2_penalty = np.maximum(co2 - 1050, 0) * 0.004
    moisture_penalty = 0.08 * np.abs(substrate_moisture - 62) ** 1.25
    substrate_bonus = np.where(substrate == "supplemented_sawdust", 1.3, 0.0)
    room_effect = pd.Series(room_id).map(
        {"GR-01": 0.7, "GR-02": -0.2, "GR-03": 0.4, "GR-04": -0.7}
    ).to_numpy()

    expected_yield = np.maximum(
        0.6,
        base * flush_factor
        - temp_penalty
        - humidity_penalty
        - co2_penalty
        - moisture_penalty
        + substrate_bonus
        + room_effect,
    )
    previous_yield = np.maximum(0.2, expected_yield + rng.normal(0, 2.0, rows))
    pin_count = np.clip(45 + expected_yield * 4.2 + rng.normal(0, 13, rows), 5, 160)

    logit = (
        -4.00
        + 0.23 * np.maximum(humidity - 90, 0) ** 1.25
        + 0.0078 * np.maximum(co2 - 1150, 0)
        + 0.70 * np.maximum(3.0 - fresh_air, 0)
        + 0.45 * np.maximum(temperature - 21.5, 0) ** 1.35
        + 1.20 * (substrate_moisture > 69)
        + 0.55 * (room_id == "GR-04")
    )
    contam_probability = 1 / (1 + np.exp(-logit))
    contamination = rng.binomial(1, contam_probability)
    next_yield = np.maximum(
        0,
        expected_yield + 0.22 * (previous_yield - expected_yield) - contamination * 3.2 + rng.normal(0, 1.25, rows),
    ).round(2)

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "batch_id": [f"B-{i:05d}" for i in range(rows)],
            "room_id": room_id,
            "species": species,
            "substrate_type": substrate,
            "flush_number": flush_number.astype(str),
            "room_age_days": room_age_days,
            "temperature_c": temperature.round(2),
            "humidity_pct": humidity.round(2),
            "co2_ppm": co2.round(1),
            "substrate_moisture_pct": substrate_moisture.round(2),
            "fresh_air_exchanges_hour": fresh_air.round(2),
            "light_hours": light_hours.round(2),
            "previous_yield_kg": previous_yield.round(2),
            "pin_count_index": pin_count.round(1),
            "next_day_yield_kg": next_yield,
            "contamination_event": contamination,
        }
    )


def collect_csv(source: Path, destination: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Validate and copy a controller CSV into the raw-data zone."""
    frame = pd.read_csv(source)
    required = {
        "timestamp", "batch_id", "room_id", "species", "substrate_type", "flush_number",
        "room_age_days", "temperature_c", "humidity_pct", "co2_ppm",
        "substrate_moisture_pct", "fresh_air_exchanges_hour", "light_hours",
        "previous_yield_kg", "pin_count_index", "next_day_yield_kg", "contamination_event",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Source CSV is missing required columns: {missing}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect mushroom grow-room training data")
    parser.add_argument("--source", type=Path, help="Optional real controller CSV")
    parser.add_argument("--rows", type=int, default=4200, help="Rows for demo collection")
    parser.add_argument("--output", type=Path, default=RAW_DATA_PATH)
    args = parser.parse_args()
    if args.source:
        frame = collect_csv(args.source, args.output)
    else:
        frame = generate_demo_data(args.rows)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.output, index=False)
    print(f"Collected {len(frame):,} observations -> {args.output}")


if __name__ == "__main__":
    main()
