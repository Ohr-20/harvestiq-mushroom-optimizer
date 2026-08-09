# Model card — HarvestIQ v1.0

## Intended use

Daily planning support for specialty mushroom farms growing oyster, lion's mane, and shiitake in controlled rooms. The output supports labor/packing preparation and prioritizes batches for contamination inspection.

## Models

- **Yield:** random forest regressor predicting next-day kilograms.
- **Contamination:** class-weighted random forest classifier producing probability; the operational alert threshold is 0.40 to favor earlier inspection.
- Numeric features are standardized and categorical features one-hot encoded inside each fitted pipeline. Unknown categories are safely ignored at inference.

## Evaluation design

Rows are sorted by time; the oldest 80% trains the model and the newest 20% is held out. This better resembles deployment than a random split. `reports/metrics.json` contains the exact reproducible results and date boundary after a pipeline run.

The yield model is compared with a training-median baseline. Contamination is evaluated with ROC AUC plus precision, recall, F1, accuracy, prevalence, and a declared decision threshold. Recall matters operationally because missed contamination is costly; precision matters because false alarms consume inspection time.

## Limitations

- Demo data is synthetic and cannot establish field performance.
- Predictions do not identify a pathogen or replace microscopy/laboratory testing.
- The training domain covers only the listed species, substrates, and four example rooms.
- Extreme sensor values are rejected, but plausible sensor failures can still produce misleading predictions.
- Feature importance is descriptive, not causal; changing a control does not guarantee the modeled response.

## Production acceptance gates

Before using recommendations on real crops:

1. Train and test on farm-owned data across at least one seasonal cycle.
2. Compare against the farm's current planning error and inspection protocol.
3. Run a four-week shadow pilot with no automated control changes.
4. Choose the risk threshold from the local cost of false negatives versus inspections.
5. Get operator sign-off on every recommendation rule.

## Monitoring

- Weekly: missing values, out-of-range readings, unknown categories, alert volume.
- Monthly: yield MAE by species/room/flush and contamination precision/recall after labels mature.
- Retrain trigger: 20% MAE degradation for two periods, material recall loss, sensor replacement, or new production recipe.
- Rollback: keep the previous approved artifact and version; the dashboard exposes the current version.

