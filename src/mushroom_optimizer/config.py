from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "crop_zone_readings.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "training_dataset.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "harvest_models.joblib"
METRICS_PATH = PROJECT_ROOT / "reports" / "metrics.json"
PREDICTIONS_PATH = PROJECT_ROOT / "reports" / "holdout_predictions.csv"
FEATURE_IMPORTANCE_PATH = PROJECT_ROOT / "reports" / "feature_importance.csv"

RANDOM_SEED = 42
TARGET_YIELD = "forecast_yield_kg"
TARGET_CONTAMINATION = "operational_risk_event"

NUMERIC_FEATURES = [
    "cycle_age_days",
    "temperature_c",
    "humidity_pct",
    "co2_ppm",
    "moisture_pct",
    "ventilation_index",
    "light_hours",
    "previous_yield_kg",
    "development_index",
    "production_area_m2",
    "stress_index",
]

CATEGORICAL_FEATURES = ["crop", "growing_system", "growth_stage", "zone_id"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
