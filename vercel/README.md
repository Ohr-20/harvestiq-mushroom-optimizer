# HarvestIQ Vercel package

This directory is the Vercel Root Directory. It contains the static dashboard,
a standard-library Python Function at `api/predict.py`, and a compact exported
model artifact. It intentionally has no third-party runtime dependencies.
The `pyproject.toml` file declares `api.predict:handler` as the Vercel Python
entrypoint.

Regenerate the package from the repository root with:

```powershell
$env:PYTHONPATH="src"
python scripts/export_vercel_model.py
```

