"""Train the final app model again on this computer and save it to models/.
Only needed if the app cannot load the saved model. The saved portable model (final_model_xgb.json) gives the
exact predictions in the report; a model trained again on another computer can differ by about 0.01 stars.
Run once with:  python retrain_model.py   (takes about 10 seconds)"""
import json
from pathlib import Path

import joblib

from src.explain import MODELS_DIR, rebuild_model

meta = json.loads((MODELS_DIR / "final_model_meta.json").read_text(encoding="utf-8"))
print("Training the final model on this computer ...")
rebuild_model(meta)
model = joblib.load(MODELS_DIR / "final_model.joblib")
print("Done. Model saved and loaded correctly:", type(model).__name__)
print("Now start the app with:  streamlit run app/app.py")
