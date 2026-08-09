from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "grow_room_readings.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "training_dataset.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "harvest_models.joblib"
METRICS_PATH = PROJECT_ROOT / "reports" / "metrics.json"
PREDICTIONS_PATH = PROJECT_ROOT / "reports" / "holdout_predictions.csv"
FEATURE_IMPORTANCE_PATH = PROJECT_ROOT / "reports" / "feature_importance.csv"

RANDOM_SEED = 42
TARGET_YIELD = "next_day_yield_kg"
TARGET_CONTAMINATION = "contamination_event"

NUMERIC_FEATURES = [
    "room_age_days",
    "temperature_c",
    "humidity_pct",
    "co2_ppm",
    "substrate_moisture_pct",
    "fresh_air_exchanges_hour",
    "light_hours",
    "previous_yield_kg",
    "pin_count_index",
]

CATEGORICAL_FEATURES = ["species", "substrate_type", "flush_number", "room_id"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

