"""Dependency-free multi-crop HarvestIQ Vercel function."""

from __future__ import annotations
import json, math, mimetypes
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse

MODEL_PATH = Path(__file__).resolve().parents[1] / "model_assets" / "model.json"
MODEL = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
STATIC_ROOT = Path(__file__).resolve().parents[1]

CROPS = {
    "oyster_mushroom": {"label":"Oyster mushroom","system":"grow_room","temp":18,"humidity":90,"moisture":62,"light":9,"horizon":"next 24 hours","pack":3},
    "lions_mane": {"label":"Lion's mane","system":"grow_room","temp":17,"humidity":88,"moisture":63,"light":9,"horizon":"next 24 hours","pack":3},
    "shiitake": {"label":"Shiitake","system":"grow_room","temp":16,"humidity":86,"moisture":61,"light":8,"horizon":"next 24 hours","pack":3},
    "avocado": {"label":"Avocado","system":"orchard","temp":22,"humidity":60,"moisture":55,"light":11,"horizon":"next 14 days","pack":12},
    "tomato": {"label":"Tomato","system":"greenhouse","temp":23,"humidity":67,"moisture":64,"light":13,"horizon":"next 7 days","pack":8},
    "strawberry": {"label":"Strawberry","system":"greenhouse","temp":20,"humidity":70,"moisture":62,"light":12,"horizon":"next 7 days","pack":4},
    "cucumber": {"label":"Cucumber","system":"greenhouse","temp":24,"humidity":72,"moisture":67,"light":13,"horizon":"next 7 days","pack":10},
    "lettuce": {"label":"Lettuce","system":"hydroponic","temp":19,"humidity":65,"moisture":74,"light":14,"horizon":"next 7 days","pack":6},
    "bell_pepper": {"label":"Bell pepper","system":"greenhouse","temp":24,"humidity":67,"moisture":63,"light":13,"horizon":"next 7 days","pack":6},
    "basil": {"label":"Basil","system":"hydroponic","temp":23,"humidity":65,"moisture":72,"light":14,"horizon":"next 7 days","pack":2},
}
RANGES = {"cycle_age_days":(1,365),"temperature_c":(3,45),"humidity_pct":(20,100),"co2_ppm":(300,5000),"moisture_pct":(10,100),"ventilation_index":(0,15),"light_hours":(0,24),"previous_yield_kg":(0,2000),"development_index":(0,100),"production_area_m2":(5,5000),"stress_index":(0,100)}


def _validate(payload: dict) -> None:
    required = MODEL["numeric_features"] + MODEL["categorical_features"]
    missing = [name for name in required if name not in payload]
    if missing: raise ValueError(f"Missing prediction fields: {missing}")
    for name,(low,high) in RANGES.items():
        value=float(payload[name])
        if not low<=value<=high: raise ValueError(f"{name} must be between {low} and {high}")
    crop=str(payload["crop"])
    if crop not in CROPS: raise ValueError("Unsupported crop")
    if str(payload["growing_system"]) not in {"grow_room","greenhouse","hydroponic","orchard"}: raise ValueError("Unsupported growing system")


def _vector(payload: dict) -> list[float]:
    vector=[(float(payload[n])-m)/s for n,m,s in zip(MODEL["numeric_features"],MODEL["numeric_mean"],MODEL["numeric_scale"])]
    for name,categories in zip(MODEL["categorical_features"],MODEL["categories"]):
        value=str(payload[name]); vector.extend(1.0 if value==str(c) else 0.0 for c in categories)
    return vector


def _linear(model: dict, vector: list[float]) -> float:
    return model["intercept"]+sum(c*v for c,v in zip(model["coefficients"],vector))


def predict(payload: dict) -> dict:
    payload=dict(payload);crop=str(payload.get("crop",""));profile=CROPS.get(crop)
    if profile:
        payload["stress_index"]=(.38*abs(float(payload["temperature_c"])-profile["temp"])+.12*abs(float(payload["humidity_pct"])-profile["humidity"])+.13*abs(float(payload["moisture_pct"])-profile["moisture"])+.45*max(2.2-float(payload["ventilation_index"]),0)+.0014*max(float(payload["co2_ppm"])-1200,0))
    _validate(payload); vector=_vector(payload)
    yield_kg=max(0.0,_linear(MODEL["yield_model"],vector)); logit=max(-30,min(30,_linear(MODEL["risk_model"],vector)))
    risk=1/(1+math.exp(-logit)); risk_band="high" if risk>=.65 else "medium" if risk>=.40 else "low"
    temp=float(payload["temperature_c"]); humidity=float(payload["humidity_pct"]); moisture=float(payload["moisture_pct"]); light=float(payload["light_hours"])
    rec=[]
    if risk>=.4: rec.append(f"Inspect the {profile['label'].lower()} zone for pest, disease, irrigation, and quality stress before harvest.")
    if abs(temp-profile["temp"])>3: rec.append(f"Temperature is outside this crop's modeled comfort band; move it toward {profile['temp']}°C where operationally appropriate.")
    if abs(moisture-profile["moisture"])>10: rec.append(f"Check root-zone or substrate moisture; the crop profile centers near {profile['moisture']}%.")
    if abs(humidity-profile["humidity"])>12: rec.append("Review humidity and leaf-surface moisture to reduce disease and quality pressure.")
    if light<profile["light"]-3: rec.append("Review shading or supplemental lighting; available light is below the crop profile.")
    if crop=="avocado" and payload["growth_stage"] in {"fruiting","harvest_ready"}: rec.append("Sample fruit dry matter and size across the block before setting the avocado harvest date.")
    if not rec: rec.append("Conditions align with this crop profile; maintain the current production recipe and monitor trends.")
    return {"crop_label":profile["label"],"forecast_horizon":profile["horizon"],"predicted_yield_kg":round(yield_kg,2),"operational_risk_probability":round(risk,4),"contamination_probability":round(risk,4),"risk_band":risk_band,"recommended_harvest_labor_hours":round(max(1,yield_kg/10),1),"recommended_crates":math.ceil(yield_kg/profile["pack"]),"recommendations":rec,"model_version":MODEL["version"]}


class handler(BaseHTTPRequestHandler):
    def _json(self,status:int,body:dict)->None:
        encoded=json.dumps(body).encode();self.send_response(status);self.send_header("Content-Type","application/json");self.send_header("Content-Length",str(len(encoded)));self.send_header("Access-Control-Allow-Origin","*");self.end_headers();self.wfile.write(encoded)
    def do_GET(self)->None:
        filename={"/":"index.html","/styles.css":"styles.css","/app.js":"app.js"}.get(urlparse(self.path).path)
        if filename is None:self._json(404,{"error":"Not found"});return
        body=(STATIC_ROOT/filename).read_bytes();self.send_response(200);self.send_header("Content-Type",mimetypes.guess_type(filename)[0] or "application/octet-stream");self.send_header("Content-Length",str(len(body)));self.end_headers();self.wfile.write(body)
    def do_POST(self)->None:
        try:
            length=int(self.headers.get("Content-Length","0"))
            if length<=0 or length>100_000:raise ValueError("Invalid request size")
            self._json(200,predict(json.loads(self.rfile.read(length))))
        except (ValueError,KeyError,json.JSONDecodeError) as exc:self._json(400,{"error":str(exc)})
    def do_OPTIONS(self)->None:
        self.send_response(204);self.send_header("Access-Control-Allow-Origin","*");self.send_header("Access-Control-Allow-Headers","Content-Type");self.send_header("Access-Control-Allow-Methods","POST, OPTIONS");self.end_headers()
