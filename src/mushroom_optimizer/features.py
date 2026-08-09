from __future__ import annotations

import pandas as pd

from .config import FEATURES, TARGET_CONTAMINATION, TARGET_YIELD


def validate_training_data(frame: pd.DataFrame) -> pd.DataFrame:
    required = ["timestamp", *FEATURES, TARGET_YIELD, TARGET_CONTAMINATION]
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise ValueError(f"Training data missing columns: {missing}")

    cleaned = frame.copy()
    cleaned["timestamp"] = pd.to_datetime(cleaned["timestamp"], errors="raise")
    cleaned = cleaned.drop_duplicates(subset=["timestamp", "batch_id"]).sort_values("timestamp")
    if cleaned[FEATURES + [TARGET_YIELD, TARGET_CONTAMINATION]].isna().any().any():
        raise ValueError("Null values detected in modeling columns")
    if not cleaned[TARGET_CONTAMINATION].isin([0, 1]).all():
        raise ValueError("contamination_event must contain only 0 or 1")
    return cleaned.reset_index(drop=True)


def prediction_frame(payload: dict) -> pd.DataFrame:
    missing = sorted(set(FEATURES).difference(payload))
    if missing:
        raise ValueError(f"Missing prediction fields: {missing}")
    return pd.DataFrame([{key: payload[key] for key in FEATURES}])

