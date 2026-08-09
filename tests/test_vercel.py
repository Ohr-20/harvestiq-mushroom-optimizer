import importlib.util, unittest
from pathlib import Path
MODULE_PATH=Path(__file__).resolve().parents[1]/"vercel"/"api"/"predict.py"
SPEC=importlib.util.spec_from_file_location("vercel_predict",MODULE_PATH);VERCEL_API=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(VERCEL_API)

def payload(crop="avocado",**changes):
    data={"crop":crop,"growing_system":"orchard","growth_stage":"fruiting","zone_id":"OR-01","cycle_age_days":180,"temperature_c":22,"humidity_pct":61,"co2_ppm":420,"moisture_pct":56,"ventilation_index":8,"light_hours":11,"previous_yield_kg":82,"development_index":84,"production_area_m2":240};data.update(changes);return data

class VercelPredictionTests(unittest.TestCase):
    def test_static_dashboard_assets_are_packaged(self):
        for filename in ["index.html","styles.css","app.js"]:self.assertTrue((VERCEL_API.STATIC_ROOT/filename).is_file())
    def test_all_ten_crop_profiles_are_deployable(self):
        self.assertEqual(len(VERCEL_API.CROPS),10)
        for crop,profile in VERCEL_API.CROPS.items():
            changes={"growing_system":profile["system"]}
            result=VERCEL_API.predict(payload(crop,**changes))
            self.assertGreaterEqual(result["predicted_yield_kg"],0)
            self.assertIn(result["risk_band"],{"low","medium","high"})
    def test_avocado_contract_and_specialist_action(self):
        result=VERCEL_API.predict(payload())
        self.assertEqual(result["crop_label"],"Avocado");self.assertEqual(result["forecast_horizon"],"next 14 days")
        self.assertTrue(any("dry matter" in x for x in result["recommendations"]))
    def test_adverse_conditions_trigger_actions(self):
        result=VERCEL_API.predict(payload("tomato",growing_system="greenhouse",temperature_c=38,moisture_pct=25,humidity_pct=94))
        self.assertGreaterEqual(len(result["recommendations"]),2)

if __name__=="__main__":unittest.main()
