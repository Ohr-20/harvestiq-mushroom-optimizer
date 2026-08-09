# HarvestIQ — Multi-Crop Intelligence

HarvestIQ is an end-to-end data science product for crop-zone planning. It supports ten crop profiles across grow rooms, greenhouses, hydroponics, and orchards:

- Oyster mushroom, lion's mane, and shiitake
- Avocado
- Tomato, strawberry, cucumber, and bell pepper
- Lettuce and basil

The application estimates crop-specific yield over an appropriate planning horizon, predicts operational stress risk, and translates both signals into scouting, climate, irrigation, labor, and packing actions.

## End-to-end workflow

```text
multi-crop telemetry simulation → validation → chronological holdout
→ yield regression + risk classification → compact model export
→ crop-aware API → responsive Vercel dashboard
```

The deterministic simulator produces 9,000 crop-zone observations. Universal features include production area, growing system, crop stage, cycle age, temperature, humidity, CO₂, root-zone/substrate moisture, ventilation, light, previous yield, development, and a derived crop-relative stress index.

## Run locally

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:PYTHONPATH="src"
python scripts\run_pipeline.py
python scripts\export_vercel_model.py
python -m mushroom_optimizer.server --port 8000
```

Open `http://localhost:8000`.

## API example

`POST /api/predict` with an avocado orchard reading:

```json
{
  "crop": "avocado",
  "growing_system": "orchard",
  "growth_stage": "fruiting",
  "zone_id": "OR-01",
  "cycle_age_days": 180,
  "production_area_m2": 240,
  "temperature_c": 22,
  "humidity_pct": 61,
  "co2_ppm": 420,
  "moisture_pct": 56,
  "ventilation_index": 8,
  "light_hours": 11,
  "previous_yield_kg": 82,
  "development_index": 84
}
```

The response includes the crop name, forecast horizon, kilograms, operational-risk probability, labor, packing units, and crop-aware recommendations.

## Deployment

The `vercel/` directory is the production package. It contains a dependency-free Python inference function, compact linear model artifact, responsive interface, and persistent dark mode. Import the repository into Vercel with Root Directory set to `vercel`.

## Responsible use

The bundled data is synthetic. This project demonstrates architecture and product logic, not validated agronomy. Crop profiles are broad defaults and cannot represent cultivar, rootstock, region, planting density, tree age, disease pressure, or farm practices. Recalibrate with governed farm data and agronomist review before operational use.
