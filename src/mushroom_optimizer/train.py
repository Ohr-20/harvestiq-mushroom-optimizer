from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import (
    CATEGORICAL_FEATURES,
    FEATURES,
    FEATURE_IMPORTANCE_PATH,
    METRICS_PATH,
    MODEL_PATH,
    NUMERIC_FEATURES,
    PREDICTIONS_PATH,
    PROCESSED_DATA_PATH,
    RANDOM_SEED,
    RAW_DATA_PATH,
    TARGET_CONTAMINATION,
    TARGET_YIELD,
)
from .features import validate_training_data


def _preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ]
    )


def _feature_importance(pipeline: Pipeline, model_name: str) -> pd.DataFrame:
    names = pipeline.named_steps["prepare"].get_feature_names_out()
    values = pipeline.named_steps["model"].feature_importances_
    return pd.DataFrame({"model": model_name, "feature": names, "importance": values})


def train_models(
    frame: pd.DataFrame,
) -> tuple[dict, dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = validate_training_data(frame)
    split_idx = int(len(data) * 0.80)
    train = data.iloc[:split_idx]
    test = data.iloc[split_idx:]

    yield_model = Pipeline(
        [
            ("prepare", _preprocessor()),
            ("model", RandomForestRegressor(
                n_estimators=260, min_samples_leaf=3, max_features=0.8,
                random_state=RANDOM_SEED, n_jobs=-1,
            )),
        ]
    )
    risk_model = Pipeline(
        [
            ("prepare", _preprocessor()),
            ("model", RandomForestClassifier(
                n_estimators=300, min_samples_leaf=3, max_features=0.85,
                class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1,
            )),
        ]
    )

    yield_model.fit(train[FEATURES], train[TARGET_YIELD])
    risk_model.fit(train[FEATURES], train[TARGET_CONTAMINATION])
    yield_pred = yield_model.predict(test[FEATURES])
    risk_prob = risk_model.predict_proba(test[FEATURES])[:, 1]
    risk_pred = (risk_prob >= 0.40).astype(int)

    baseline = np.repeat(train[TARGET_YIELD].median(), len(test))
    metrics = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "split_strategy": "oldest 80% train / newest 20% holdout",
        "train_rows": len(train),
        "holdout_rows": len(test),
        "train_end": train["timestamp"].max().isoformat(),
        "holdout_start": test["timestamp"].min().isoformat(),
        "yield_model": {
            "mae_kg": round(float(mean_absolute_error(test[TARGET_YIELD], yield_pred)), 4),
            "rmse_kg": round(float(mean_squared_error(test[TARGET_YIELD], yield_pred) ** 0.5), 4),
            "r2": round(float(r2_score(test[TARGET_YIELD], yield_pred)), 4),
            "baseline_mae_kg": round(float(mean_absolute_error(test[TARGET_YIELD], baseline)), 4),
        },
        "contamination_model": {
            "threshold": 0.40,
            "roc_auc": round(float(roc_auc_score(test[TARGET_CONTAMINATION], risk_prob)), 4),
            "precision": round(float(precision_score(test[TARGET_CONTAMINATION], risk_pred, zero_division=0)), 4),
            "recall": round(float(recall_score(test[TARGET_CONTAMINATION], risk_pred, zero_division=0)), 4),
            "f1": round(float(f1_score(test[TARGET_CONTAMINATION], risk_pred, zero_division=0)), 4),
            "accuracy": round(float(accuracy_score(test[TARGET_CONTAMINATION], risk_pred)), 4),
            "holdout_prevalence": round(float(test[TARGET_CONTAMINATION].mean()), 4),
        },
    }

    predictions = test[["timestamp", "batch_id", TARGET_YIELD, TARGET_CONTAMINATION]].copy()
    predictions["predicted_yield_kg"] = np.round(yield_pred, 3)
    predictions["contamination_probability"] = np.round(risk_prob, 4)
    predictions["predicted_contamination"] = risk_pred

    artifact = {
        "yield_model": yield_model,
        "risk_model": risk_model,
        "features": FEATURES,
        "risk_threshold": 0.40,
        "metadata": metrics,
        "version": "1.0.0",
    }
    importances = pd.concat(
        [_feature_importance(yield_model, "yield"), _feature_importance(risk_model, "contamination")],
        ignore_index=True,
    ).sort_values(["model", "importance"], ascending=[True, False])
    return artifact, metrics, predictions, importances, data


def run_training(source: Path = RAW_DATA_PATH) -> dict:
    frame = pd.read_csv(source)
    artifact, metrics, predictions, importances, cleaned = train_models(frame)
    for path in [MODEL_PATH, METRICS_PATH, PREDICTIONS_PATH, FEATURE_IMPORTANCE_PATH, PROCESSED_DATA_PATH]:
        path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    predictions.to_csv(PREDICTIONS_PATH, index=False)
    importances.to_csv(FEATURE_IMPORTANCE_PATH, index=False)
    cleaned.to_csv(PROCESSED_DATA_PATH, index=False)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train mushroom farm decision models")
    parser.add_argument("--input", type=Path, default=RAW_DATA_PATH)
    args = parser.parse_args()
    metrics = run_training(args.input)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
