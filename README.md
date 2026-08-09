# HarvestIQ — Specialty Mushroom Farm Intelligence

HarvestIQ is an end-to-end data science product for small specialty mushroom farms. It forecasts **next-day harvest weight** and estimates **contamination risk**, then translates both predictions into staffing, crate, airflow, hydration, and inspection recommendations.

This is a deliberately narrow business product rather than a generic ML demo. A farm can use the forecast during its afternoon production meeting to prepare tomorrow's labor and packing capacity while catching risky grow-room conditions early.

## Business case

Specialty mushroom farms operate perishable, short harvest windows. Underestimating a flush creates rushed labor and inadequate cold-storage/packing capacity; overestimating it wastes labor. Contamination can spread quickly and destroy a batch. HarvestIQ targets both decisions from one sensor-and-operations record.

Illustrative value model for a four-room farm:

- Preventing one 30 kg batch loss per month at $12/kg protects about **$4,320/year**.
- Avoiding 6 unnecessary labor hours per week at $20/hour saves about **$6,240/year**.
- The product should therefore be piloted if setup and annual operation cost materially less than roughly **$10,500/year**. These are scenario assumptions, not claimed outcomes.

## What is included

```text
data collection -> validation -> time-based split -> model training -> evaluation
       -> versioned artifact -> prediction API -> farm operations dashboard
```

- Reproducible grow-room telemetry and operations-data simulator
- Adapter and schema validation for real controller CSV exports
- Yield regression and contamination classification pipelines
- Time-ordered holdout evaluation to avoid future-to-past leakage
- Versioned model artifact, metrics, holdout predictions, and feature importance
- Browser dashboard plus JSON API using Python's standard library
- Tests, Docker packaging, health check, and cloud deployment blueprint

## Quick start

Run from this directory with Python 3.11+:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:PYTHONPATH="src"
python scripts\run_pipeline.py
python -m mushroom_optimizer.server --port 8000
```

Open `http://localhost:8000`. The pipeline creates 4,200 demo observations, trains both models, evaluates on the newest 20%, and writes all artifacts locally.

To bring real farm data, export the schema documented in [DATA_CARD.md](DATA_CARD.md), then run:

```powershell
python -m mushroom_optimizer.data_collection --source path\to\controller_export.csv
python -m mushroom_optimizer.train
```

## API

`GET /api/health` reports service and model readiness. `POST /api/predict` accepts one reading:

```json
{
  "species": "oyster",
  "room_id": "GR-01",
  "substrate_type": "straw",
  "flush_number": "1",
  "room_age_days": 12,
  "temperature_c": 18.2,
  "humidity_pct": 89,
  "co2_ppm": 860,
  "substrate_moisture_pct": 62,
  "fresh_air_exchanges_hour": 5.5,
  "light_hours": 10,
  "previous_yield_kg": 15.2,
  "pin_count_index": 108
}
```

The response includes predicted kilograms, contamination probability and risk band, labor hours, crate count, and contextual actions.

## Testing

The test suite uses only the standard library and installed ML dependencies:

```powershell
$env:PYTHONPATH="src"
python -m unittest discover -s tests -v
```

## Deployment

### Docker

```powershell
docker build -t harvestiq .
docker run --rm -p 8000:8000 harvestiq
```

### Render

Push the directory to GitHub and create a Render Blueprint from `render.yaml`. The build trains a deterministic bundled demo model; for a production pilot, persist approved model artifacts in object storage and promote them after evaluation instead of retraining during deployment.

## Responsible use

HarvestIQ is decision support, not an autonomous environmental controller. Operators should verify unusual sensor values, follow farm sanitation protocols, and treat contamination warnings as inspection priorities—not laboratory diagnoses. See [MODEL_CARD.md](MODEL_CARD.md) for limitations and monitoring guidance.

