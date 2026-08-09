from __future__ import annotations

from functools import lru_cache

import joblib
import numpy as np

from .config import MODEL_PATH
from .features import prediction_frame


def _validate_ranges(payload: dict) -> None:
    ranges = {
        "room_age_days": (1, 60), "temperature_c": (5, 35), "humidity_pct": (30, 100),
        "co2_ppm": (300, 5000), "substrate_moisture_pct": (20, 95),
        "fresh_air_exchanges_hour": (0, 20), "light_hours": (0, 24),
        "previous_yield_kg": (0, 60), "pin_count_index": (0, 300),
    }
    for name, (low, high) in ranges.items():
        value = float(payload[name])
        if not low <= value <= high:
            raise ValueError(f"{name} must be between {low} and {high}")
    if str(payload["species"]) not in {"oyster", "lion's_mane", "shiitake"}:
        raise ValueError("species must be oyster, lion's_mane, or shiitake")


@lru_cache(maxsize=1)
def load_artifact(path: str = str(MODEL_PATH)) -> dict:
    return joblib.load(path)


def _recommendations(payload: dict, yield_kg: float, risk: float) -> list[str]:
    recommendations: list[str] = []
    if risk >= 0.40:
        recommendations.append("Isolate this batch and schedule a contamination inspection before harvest.")
    if float(payload["co2_ppm"]) > 1200:
        recommendations.append("Increase fresh-air exchange; CO₂ is above the production target.")
    if float(payload["humidity_pct"]) > 93:
        recommendations.append("Reduce humidity and check for standing water to limit contamination pressure.")
    if float(payload["substrate_moisture_pct"]) < 53:
        recommendations.append("Inspect substrate hydration; moisture is below the productive range.")
    if yield_kg >= 15:
        recommendations.append(f"Reserve packing capacity for approximately {np.ceil(yield_kg):.0f} kg tomorrow.")
    if not recommendations:
        recommendations.append("Conditions are stable; continue the current grow-room recipe.")
    return recommendations


def predict(payload: dict, artifact_path: str = str(MODEL_PATH)) -> dict:
    frame = prediction_frame(payload)
    _validate_ranges(payload)
    artifact = load_artifact(artifact_path)
    yield_kg = max(0.0, float(artifact["yield_model"].predict(frame)[0]))
    risk = float(artifact["risk_model"].predict_proba(frame)[0, 1])
    risk_band = "high" if risk >= 0.65 else "medium" if risk >= 0.40 else "low"
    staffing_hours = round(max(1.0, yield_kg / 6.0), 1)
    return {
        "predicted_yield_kg": round(yield_kg, 2),
        "contamination_probability": round(risk, 4),
        "risk_band": risk_band,
        "recommended_harvest_labor_hours": staffing_hours,
        "recommended_crates": int(np.ceil(yield_kg / 3.0)),
        "recommendations": _recommendations(payload, yield_kg, risk),
        "model_version": artifact["version"],
    }

