# Model card — HarvestIQ v2

HarvestIQ v2 ranks production conditions for ten crops across indoor and outdoor systems. A random-forest regressor estimates crop-specific forecast yield and a class-weighted random-forest classifier estimates operational stress risk. The dependency-free Vercel artifact uses Ridge and logistic regression with the same feature schema.

Rows are ordered by timestamp: the oldest 80% trains the models and the newest 20% is held out. Exact metrics are in `reports/metrics.json` and `vercel/model_assets/model.json`. Yield is evaluated with MAE, RMSE, R², and a median baseline. Risk is evaluated with ROC AUC, precision, recall, F1, and accuracy.

The model is synthetic and not agronomically validated. Yield magnitude depends on the simulated area and horizon. The crop-relative stress index makes the same sensor reading mean different things for avocado, lettuce, tomato, or mushrooms, but it remains a simplified engineering feature—not a causal crop model.

Before operational use, train and test on farm-owned outcomes, report error by crop/system/season, run a shadow pilot, select alert thresholds from local costs, obtain agronomist and operator sign-off, and retain a rollback artifact. Never connect recommendations directly to environmental controls without human review.
