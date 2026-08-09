# Data card

## Unit of observation

One row represents one active mushroom batch in one grow room at an afternoon planning cutoff. The labels are the harvested kilograms and whether contamination was observed during the following 24 hours.

## Collection contract

| Field | Type | Unit / allowed values | Collection source |
|---|---|---|---|
| `timestamp` | datetime | ISO-8601 | Farm system clock |
| `batch_id` | string | unique batch reference | Batch log |
| `room_id` | category | `GR-01`… | Controller |
| `species` | category | oyster, lion's mane, shiitake | Batch log |
| `substrate_type` | category | straw, hardwood or supplemented sawdust | Batch log |
| `flush_number` | category | 1, 2, 3 | Batch log |
| `room_age_days` | number | days | Batch log |
| `temperature_c` | number | °C | Environmental sensor |
| `humidity_pct` | number | % RH | Environmental sensor |
| `co2_ppm` | number | ppm | Environmental sensor |
| `substrate_moisture_pct` | number | % | Probe/manual sample |
| `fresh_air_exchanges_hour` | number | exchanges/hour | HVAC setting |
| `light_hours` | number | hours/day | Controller |
| `previous_yield_kg` | number | kg | Harvest log |
| `pin_count_index` | number | 0–300 visual estimate | Operator or vision counter |
| `next_day_yield_kg` | number | kg, target | Next-day harvest log |
| `contamination_event` | binary | 0/1, target | Next-day inspection log |

## Demo data provenance

The included pipeline generates synthetic, deterministic data. It encodes plausible relationships between species, flush, temperature, humidity, CO₂, substrate moisture, ventilation, and yield/risk. It is suitable for software demonstration and workflow validation, **not proof of real-farm accuracy**.

No people, customer records, precise locations, or protected attributes are present.

## Production collection protocol

1. Calibrate temperature/humidity sensors monthly and CO₂ sensors to manufacturer guidance.
2. Capture readings at the same planning time daily; use a documented aggregation if sensors sample continuously.
3. Record targets after the 24-hour outcome window closes. Never backfill a reading using future information.
4. Preserve raw exports as append-only files; corrections should add an audit record.
5. Measure at least one complete seasonal cycle and retain rare contamination examples.
6. Monitor missingness, sensor drift, new species/substrates, and label consistency weekly.

