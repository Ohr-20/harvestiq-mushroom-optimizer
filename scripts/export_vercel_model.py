"""Export compact linear models for dependency-free Vercel inference."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from mushroom_optimizer.config import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    PROJECT_ROOT,
    RANDOM_SEED,
    TARGET_CONTAMINATION,
    TARGET_YIELD,
)
from mushroom_optimizer.data_collection import generate_demo_data
from mushroom_optimizer.features import validate_training_data


VERCEL_ROOT = PROJECT_ROOT / "vercel"
MODEL_OUTPUT = VERCEL_ROOT / "model_assets" / "model.json"


def export() -> dict:
    data = validate_training_data(generate_demo_data())
    split = int(len(data) * 0.80)
    train, holdout = data.iloc[:split], data.iloc[split:]

    prepare = ColumnTransformer(
        [
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    x_train = prepare.fit_transform(train[NUMERIC_FEATURES + CATEGORICAL_FEATURES])
    x_holdout = prepare.transform(holdout[NUMERIC_FEATURES + CATEGORICAL_FEATURES])

    yield_model = Ridge(alpha=3.0)
    risk_model = LogisticRegression(
        max_iter=2500,
        class_weight="balanced",
        random_state=RANDOM_SEED,
    )
    yield_model.fit(x_train, train[TARGET_YIELD])
    risk_model.fit(x_train, train[TARGET_CONTAMINATION])

    yield_prediction = yield_model.predict(x_holdout)
    risk_probability = risk_model.predict_proba(x_holdout)[:, 1]
    numeric_transformer = prepare.named_transformers_["numeric"]
    category_transformer = prepare.named_transformers_["categorical"]

    artifact = {
        "version": "1.1.0-vercel",
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "numeric_mean": numeric_transformer.mean_.tolist(),
        "numeric_scale": numeric_transformer.scale_.tolist(),
        "categories": [items.tolist() for items in category_transformer.categories_],
        "yield_model": {
            "intercept": float(yield_model.intercept_),
            "coefficients": yield_model.coef_.tolist(),
        },
        "risk_model": {
            "intercept": float(risk_model.intercept_[0]),
            "coefficients": risk_model.coef_[0].tolist(),
        },
        "metrics": {
            "holdout_rows": len(holdout),
            "yield_mae_kg": round(float(mean_absolute_error(holdout[TARGET_YIELD], yield_prediction)), 4),
            "yield_r2": round(float(r2_score(holdout[TARGET_YIELD], yield_prediction)), 4),
            "contamination_roc_auc": round(
                float(roc_auc_score(holdout[TARGET_CONTAMINATION], risk_probability)), 4
            ),
        },
    }

    MODEL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    MODEL_OUTPUT.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    for filename in ["index.html", "styles.css", "app.js"]:
        shutil.copy2(PROJECT_ROOT / "web" / filename, VERCEL_ROOT / filename)
    return artifact["metrics"]


if __name__ == "__main__":
    print(json.dumps(export(), indent=2))

