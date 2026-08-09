"""Synthetic multi-crop telemetry for a reproducible portfolio workflow."""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from .config import RANDOM_SEED, RAW_DATA_PATH

CROP_PROFILES = {
    "oyster_mushroom": dict(system="grow_room", base=0.16, temp=18, humidity=90, moisture=62, light=9, horizon=1),
    "lions_mane": dict(system="grow_room", base=0.11, temp=17, humidity=88, moisture=63, light=9, horizon=1),
    "shiitake": dict(system="grow_room", base=0.12, temp=16, humidity=86, moisture=61, light=8, horizon=1),
    "avocado": dict(system="orchard", base=0.42, temp=22, humidity=60, moisture=55, light=11, horizon=14),
    "tomato": dict(system="greenhouse", base=0.22, temp=23, humidity=67, moisture=64, light=13, horizon=7),
    "strawberry": dict(system="greenhouse", base=0.11, temp=20, humidity=70, moisture=62, light=12, horizon=7),
    "cucumber": dict(system="greenhouse", base=0.25, temp=24, humidity=72, moisture=67, light=13, horizon=7),
    "lettuce": dict(system="hydroponic", base=0.10, temp=19, humidity=65, moisture=74, light=14, horizon=7),
    "bell_pepper": dict(system="greenhouse", base=0.16, temp=24, humidity=67, moisture=63, light=13, horizon=7),
    "basil": dict(system="hydroponic", base=0.07, temp=23, humidity=65, moisture=72, light=14, horizon=7),
}


def generate_demo_data(rows: int = 9000, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2023-01-01", "2026-06-30", periods=rows)
    crops = rng.choice(list(CROP_PROFILES), rows)
    profile = [CROP_PROFILES[c] for c in crops]
    system = np.array([p["system"] for p in profile])
    zone = np.array([f"{s[:2].upper()}-{rng.integers(1, 5):02d}" for s in system])
    stage = rng.choice(["establishment", "vegetative", "flowering", "fruiting", "harvest_ready"], rows,
                       p=[.10, .20, .20, .31, .19])
    area = np.clip(rng.lognormal(np.log(95), .65, rows), 12, 800)
    cycle_age = np.clip(rng.normal(55, 38, rows) + (system == "orchard") * 110, 3, 365)
    day_angle = 2 * np.pi * timestamps.dayofyear.to_numpy() / 365.25
    ideal_temp = np.array([p["temp"] for p in profile])
    ideal_humidity = np.array([p["humidity"] for p in profile])
    ideal_moisture = np.array([p["moisture"] for p in profile])
    ideal_light = np.array([p["light"] for p in profile])
    temperature = ideal_temp + 2.2 * np.sin(day_angle) + rng.normal(0, 2.4, rows)
    humidity = np.clip(ideal_humidity + rng.normal(0, 6, rows), 30, 98)
    moisture = np.clip(ideal_moisture + rng.normal(0, 8, rows), 20, 95)
    co2 = np.where(system == "orchard", rng.normal(420, 25, rows), rng.lognormal(np.log(760), .25, rows))
    co2 = np.clip(co2, 350, 2400)
    ventilation = np.where(system == "orchard", rng.normal(8, 1.5, rows), rng.normal(5.5, 1.4, rows))
    ventilation = np.clip(ventilation, 0.5, 12)
    light = np.clip(ideal_light + rng.normal(0, 2, rows), 2, 18)
    development = np.clip(rng.normal(68, 20, rows) + (stage == "harvest_ready") * 22 - (stage == "establishment") * 35, 1, 100)
    stage_factor = pd.Series(stage).map({"establishment": .12, "vegetative": .35, "flowering": .62, "fruiting": .88, "harvest_ready": 1.0}).to_numpy()
    base = np.array([p["base"] * p["horizon"] for p in profile]) * area
    temp_fit = np.exp(-.035 * np.abs(temperature - ideal_temp) ** 1.55)
    humidity_fit = np.exp(-.018 * np.abs(humidity - ideal_humidity) ** 1.35)
    moisture_fit = np.exp(-.021 * np.abs(moisture - ideal_moisture) ** 1.35)
    light_fit = np.exp(-.035 * np.abs(light - ideal_light) ** 1.45)
    expected = np.maximum(.2, base * stage_factor * temp_fit * humidity_fit * moisture_fit * light_fit)
    previous = np.maximum(0, expected * rng.normal(1, .16, rows))
    stress = (.38 * np.abs(temperature - ideal_temp) + .12 * np.abs(humidity - ideal_humidity) +
              .13 * np.abs(moisture - ideal_moisture) + .45 * np.maximum(2.2 - ventilation, 0) +
              .0014 * np.maximum(co2 - 1200, 0))
    logit = -3.5 + 1.05 * stress + .55 * (stage == "harvest_ready") + rng.normal(0, .15, rows)
    probability = 1 / (1 + np.exp(-logit))
    risk = rng.binomial(1, probability)
    forecast = np.maximum(0, expected * .72 + previous * .28 - risk * expected * .18 + rng.normal(0, np.maximum(.35, expected * .08), rows))
    return pd.DataFrame({
        "timestamp": timestamps, "batch_id": [f"Z-{i:05d}" for i in range(rows)], "zone_id": zone,
        "crop": crops, "growing_system": system, "growth_stage": stage,
        "cycle_age_days": cycle_age.round(1), "temperature_c": temperature.round(2),
        "humidity_pct": humidity.round(2), "co2_ppm": co2.round(1), "moisture_pct": moisture.round(2),
        "ventilation_index": ventilation.round(2), "light_hours": light.round(2),
        "previous_yield_kg": previous.round(2), "development_index": development.round(1),
        "production_area_m2": area.round(1), "stress_index": stress.round(3), "forecast_yield_kg": forecast.round(2),
        "operational_risk_event": risk,
    })


def collect_csv(source: Path, destination: Path = RAW_DATA_PATH) -> pd.DataFrame:
    frame = pd.read_csv(source)
    required = set(generate_demo_data(1).columns)
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Source CSV is missing required columns: {missing}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect multi-crop production telemetry")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--rows", type=int, default=9000)
    parser.add_argument("--output", type=Path, default=RAW_DATA_PATH)
    args = parser.parse_args()
    frame = collect_csv(args.source, args.output) if args.source else generate_demo_data(args.rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(f"Collected {len(frame):,} observations -> {args.output}")


if __name__ == "__main__":
    main()
