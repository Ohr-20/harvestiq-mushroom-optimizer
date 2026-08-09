"""Dependency-free Vercel Function for HarvestIQ predictions."""

from __future__ import annotations

import json
import math
import mimetypes
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse


MODEL_PATH = Path(__file__).resolve().parents[1] / "model_assets" / "model.json"
MODEL = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
STATIC_ROOT = Path(__file__).resolve().parents[1]

RANGES = {
    "room_age_days": (1, 60),
    "temperature_c": (5, 35),
    "humidity_pct": (30, 100),
    "co2_ppm": (300, 5000),
    "substrate_moisture_pct": (20, 95),
    "fresh_air_exchanges_hour": (0, 20),
    "light_hours": (0, 24),
    "previous_yield_kg": (0, 60),
    "pin_count_index": (0, 300),
}


def _validate(payload: dict) -> None:
    required = MODEL["numeric_features"] + MODEL["categorical_features"]
    missing = [name for name in required if name not in payload]
    if missing:
        raise ValueError(f"Missing prediction fields: {missing}")
    for name, (low, high) in RANGES.items():
        value = float(payload[name])
        if not low <= value <= high:
            raise ValueError(f"{name} must be between {low} and {high}")
    if str(payload["species"]) not in {"oyster", "lion's_mane", "shiitake"}:
        raise ValueError("Unsupported mushroom species")


def _vector(payload: dict) -> list[float]:
    vector = [
        (float(payload[name]) - mean) / scale
        for name, mean, scale in zip(
            MODEL["numeric_features"], MODEL["numeric_mean"], MODEL["numeric_scale"]
        )
    ]
    for name, categories in zip(MODEL["categorical_features"], MODEL["categories"]):
        value = str(payload[name])
        vector.extend(1.0 if value == str(category) else 0.0 for category in categories)
    return vector


def _linear(model: dict, vector: list[float]) -> float:
    return model["intercept"] + sum(
        coefficient * value for coefficient, value in zip(model["coefficients"], vector)
    )


def predict(payload: dict) -> dict:
    _validate(payload)
    vector = _vector(payload)
    yield_kg = max(0.0, _linear(MODEL["yield_model"], vector))
    logit = max(-30.0, min(30.0, _linear(MODEL["risk_model"], vector)))
    risk = 1.0 / (1.0 + math.exp(-logit))
    risk_band = "high" if risk >= 0.65 else "medium" if risk >= 0.40 else "low"

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
        recommendations.append(
            f"Reserve packing capacity for approximately {math.ceil(yield_kg):.0f} kg tomorrow."
        )
    if not recommendations:
        recommendations.append("Conditions are stable; continue the current grow-room recipe.")

    return {
        "predicted_yield_kg": round(yield_kg, 2),
        "contamination_probability": round(risk, 4),
        "risk_band": risk_band,
        "recommended_harvest_labor_hours": round(max(1.0, yield_kg / 6.0), 1),
        "recommended_crates": math.ceil(yield_kg / 3.0),
        "recommendations": recommendations,
        "model_version": MODEL["version"],
    }


class handler(BaseHTTPRequestHandler):
    def _json(self, status: int, body: dict) -> None:
        encoded = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        filename = {"/": "index.html", "/styles.css": "styles.css", "/app.js": "app.js"}.get(path)
        if filename is None:
            self._json(404, {"error": "Not found"})
            return
        body = (STATIC_ROOT / filename).read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(filename)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 100_000:
                raise ValueError("Request body must be between 1 and 100,000 bytes")
            self._json(200, predict(json.loads(self.rfile.read(length))))
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            self._json(400, {"error": str(exc)})

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.end_headers()

