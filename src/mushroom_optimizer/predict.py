from __future__ import annotations
from functools import lru_cache
import joblib
import numpy as np
from .config import MODEL_PATH
from .data_collection import CROP_PROFILES
from .features import prediction_frame

RANGES={"cycle_age_days":(1,365),"temperature_c":(3,45),"humidity_pct":(20,100),"co2_ppm":(300,5000),"moisture_pct":(10,100),"ventilation_index":(0,15),"light_hours":(0,24),"previous_yield_kg":(0,2000),"development_index":(0,100),"production_area_m2":(5,5000),"stress_index":(0,100)}

def _validate_ranges(payload:dict)->None:
    for name,(low,high) in RANGES.items():
        value=float(payload[name])
        if not low<=value<=high:raise ValueError(f"{name} must be between {low} and {high}")
    if str(payload["crop"]) not in CROP_PROFILES:raise ValueError("Unsupported crop")

@lru_cache(maxsize=1)
def load_artifact(path:str=str(MODEL_PATH))->dict:return joblib.load(path)

def predict(payload:dict,artifact_path:str=str(MODEL_PATH))->dict:
    payload=dict(payload);crop=str(payload.get("crop",""));profile=CROP_PROFILES.get(crop)
    if profile:payload["stress_index"]=(.38*abs(float(payload["temperature_c"])-profile["temp"])+.12*abs(float(payload["humidity_pct"])-profile["humidity"])+.13*abs(float(payload["moisture_pct"])-profile["moisture"])+.45*max(2.2-float(payload["ventilation_index"]),0)+.0014*max(float(payload["co2_ppm"])-1200,0))
    frame=prediction_frame(payload);_validate_ranges(payload);artifact=load_artifact(artifact_path)
    yield_kg=max(0,float(artifact["yield_model"].predict(frame)[0]));risk=float(artifact["risk_model"].predict_proba(frame)[0,1]);band="high" if risk>=.65 else "medium" if risk>=.40 else "low"
    profile=CROP_PROFILES[crop];rec=[]
    if risk>=.4:rec.append("Inspect the zone for pest, disease, irrigation, and quality stress before harvest.")
    if abs(float(payload["temperature_c"])-profile["temp"])>3:rec.append("Bring temperature toward the crop-specific modeled target where operationally appropriate.")
    if abs(float(payload["moisture_pct"])-profile["moisture"])>10:rec.append("Check root-zone or substrate moisture before the next irrigation decision.")
    if crop=="avocado":rec.append("Sample avocado fruit dry matter and size across the block before setting the harvest date.")
    if not rec:rec.append("Conditions align with this crop profile; maintain the current recipe and monitor trends.")
    pack=12 if crop=="avocado" else 3 if "mushroom" in crop or crop in {"lions_mane","shiitake"} else 6
    label={"oyster_mushroom":"Oyster mushroom","lions_mane":"Lion's mane","shiitake":"Shiitake","avocado":"Avocado","tomato":"Tomato","strawberry":"Strawberry","cucumber":"Cucumber","lettuce":"Lettuce","bell_pepper":"Bell pepper","basil":"Basil"}[crop]
    horizon="next 24 hours" if profile["system"]=="grow_room" else "next 14 days" if crop=="avocado" else "next 7 days"
    return {"crop_label":label,"forecast_horizon":horizon,"predicted_yield_kg":round(yield_kg,2),"operational_risk_probability":round(risk,4),"contamination_probability":round(risk,4),"risk_band":band,"recommended_harvest_labor_hours":round(max(1,yield_kg/10),1),"recommended_crates":int(np.ceil(yield_kg/pack)),"recommendations":rec,"model_version":artifact["version"]}
