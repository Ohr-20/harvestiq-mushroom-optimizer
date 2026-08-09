import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "vercel" / "api" / "predict.py"
SPEC = importlib.util.spec_from_file_location("vercel_predict", MODULE_PATH)
VERCEL_API = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERCEL_API)


class VercelPredictionTests(unittest.TestCase):
    def test_static_dashboard_assets_are_packaged(self):
        for filename in ["index.html", "styles.css", "app.js"]:
            self.assertTrue((VERCEL_API.STATIC_ROOT / filename).is_file())

    def test_stable_conditions_return_complete_contract(self):
        result = VERCEL_API.predict(
            {
                "species": "oyster",
                "substrate_type": "straw",
                "flush_number": "1",
                "room_id": "GR-01",
                "room_age_days": 12,
                "temperature_c": 18.2,
                "humidity_pct": 89,
                "co2_ppm": 860,
                "substrate_moisture_pct": 62,
                "fresh_air_exchanges_hour": 5.5,
                "light_hours": 10,
                "previous_yield_kg": 15.2,
                "pin_count_index": 108,
            }
        )
        self.assertGreater(result["predicted_yield_kg"], 0)
        self.assertIn(result["risk_band"], {"low", "medium", "high"})
        self.assertGreaterEqual(result["recommended_crates"], 1)
        self.assertTrue(result["recommendations"])

    def test_adverse_conditions_trigger_operational_actions(self):
        result = VERCEL_API.predict(
            {
                "species": "lion's_mane",
                "substrate_type": "supplemented_sawdust",
                "flush_number": "2",
                "room_id": "GR-04",
                "room_age_days": 17,
                "temperature_c": 23.8,
                "humidity_pct": 95.1,
                "co2_ppm": 1540,
                "substrate_moisture_pct": 72,
                "fresh_air_exchanges_hour": 2.1,
                "light_hours": 9,
                "previous_yield_kg": 8.4,
                "pin_count_index": 66,
            }
        )
        self.assertIn(result["risk_band"], {"medium", "high"})
        self.assertTrue(any("Isolate" in item for item in result["recommendations"]))
        self.assertTrue(any("fresh-air" in item for item in result["recommendations"]))


if __name__ == "__main__":
    unittest.main()

