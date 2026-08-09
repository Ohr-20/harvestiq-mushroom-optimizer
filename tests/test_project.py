import tempfile
import unittest
from pathlib import Path

import joblib

from mushroom_optimizer.config import FEATURES
from mushroom_optimizer.data_collection import generate_demo_data
from mushroom_optimizer.features import validate_training_data
from mushroom_optimizer.predict import load_artifact, predict
from mushroom_optimizer.train import train_models


class HarvestIQTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = generate_demo_data(rows=900, seed=7)
        cls.artifact, cls.metrics, _, _, _ = train_models(cls.frame)
        cls.temp = tempfile.TemporaryDirectory()
        cls.model_path = str(Path(cls.temp.name) / "model.joblib")
        joblib.dump(cls.artifact, cls.model_path)

    @classmethod
    def tearDownClass(cls):
        load_artifact.cache_clear()
        cls.temp.cleanup()

    def test_generator_has_complete_schema_and_both_classes(self):
        cleaned = validate_training_data(self.frame)
        self.assertTrue(set(FEATURES).issubset(cleaned.columns))
        self.assertEqual(set(cleaned["operational_risk_event"].unique()), {0, 1})
        self.assertFalse(cleaned.isna().any().any())

    def test_evaluation_is_time_ordered_and_beats_yield_baseline(self):
        self.assertLess(self.metrics["train_end"], self.metrics["holdout_start"])
        self.assertLess(
            self.metrics["yield_model"]["mae_kg"],
            self.metrics["yield_model"]["baseline_mae_kg"],
        )
        self.assertGreater(self.metrics["contamination_model"]["roc_auc"], 0.60)

    def test_prediction_contract(self):
        payload = {"crop":"avocado","growing_system":"orchard","growth_stage":"fruiting","zone_id":"OR-01","cycle_age_days":180,"temperature_c":22,"humidity_pct":61,"co2_ppm":420,"moisture_pct":56,"ventilation_index":8,"light_hours":11,"previous_yield_kg":82,"development_index":84,"production_area_m2":240}
        result = predict(payload, self.model_path)
        self.assertGreaterEqual(result["predicted_yield_kg"], 0)
        self.assertGreaterEqual(result["contamination_probability"], 0)
        self.assertLessEqual(result["contamination_probability"], 1)
        self.assertIn(result["risk_band"], {"low", "medium", "high"})
        self.assertTrue(result["recommendations"])

    def test_out_of_range_input_is_rejected(self):
        payload = {key: 1 for key in FEATURES}
        payload.update({"crop":"tomato","growing_system":"greenhouse","growth_stage":"fruiting","zone_id":"GH-01","temperature_c":99})
        with self.assertRaisesRegex(ValueError, "temperature_c"):
            predict(payload, self.model_path)


if __name__ == "__main__":
    unittest.main()
