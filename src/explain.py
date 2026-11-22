"""Helpers used by the demo app: load the saved model, predict one phone, explain it, find similar phones."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
RESULTS = ROOT / "results"

FRIENDLY = {"price_inr": "Price", "ram_gb": "RAM (GB)", "storage_gb": "Storage (GB)",
            "battery_mah": "Battery (mAh)", "charging_w": "Charging (W)", "display_in": "Screen size (inch)",
            "is_amoled": "AMOLED/OLED screen", "rear_mp": "Main rear camera (MP)", "front_mp": "Front camera (MP)",
            "spec_score": "Spec score (91mobiles)", "antutu": "AnTuTu score", "release_year": "Release year",
            "brand": "Brand", "refresh_hz": "Refresh rate (Hz)", "pixel_ppi": "Pixel density (ppi)",
            "chipset_vendor": "Chipset maker"}


def save_portable(model):
    """Save the model in a form that loads in exactly the same way on Windows, macOS and Linux:
    the XGBoost part in XGBoost's own JSON format and the pre-processing with joblib."""
    model.named_steps["m"].save_model(str(MODELS_DIR / "final_model_xgb.json"))
    joblib.dump(model.named_steps["pre"], MODELS_DIR / "final_model_pre.joblib")


def load_portable():
    from sklearn.pipeline import Pipeline
    from xgboost import XGBRegressor
    pre = joblib.load(MODELS_DIR / "final_model_pre.joblib")
    m = XGBRegressor()
    m.load_model(str(MODELS_DIR / "final_model_xgb.json"))
    return Pipeline([("pre", pre), ("m", m)])


def rebuild_model(meta):
    """Train the final model again on this computer (about 10 seconds) and save it.
    Used only when no saved model can be read."""
    from sklearn.base import clone
    from . import models as M, prep
    work, _, _ = prep.build_all("exact_then_key", verbose=False)
    num, cat = meta["numeric"], meta["categorical"]
    model = clone(M.make_models(num, cat, meta["xgb_params"])[meta["model"]])
    model.fit(work[num + cat], work[M.TARGET])
    joblib.dump(model, MODELS_DIR / "final_model.joblib")
    save_portable(model)
    return model


def load():
    """Load the final model: the portable files first (same predictions on every computer),
    then the joblib file, and only if both fail, train it again."""
    meta = json.loads((MODELS_DIR / "final_model_meta.json").read_text(encoding="utf-8"))
    model = None
    if (MODELS_DIR / "final_model_xgb.json").exists() and (MODELS_DIR / "final_model_pre.joblib").exists():
        try:
            model = load_portable()
        except Exception:
            model = None
    if model is None:
        try:
            model = joblib.load(MODELS_DIR / "final_model.joblib")
        except Exception:
            model = rebuild_model(meta)
    ref = pd.read_csv(MODELS_DIR / "reference_phones.csv")
    return model, meta, ref


def _base(name, features):
    for f in features:
        if name == f or name.startswith(f + "_") or name == f"missingindicator_{f}":
            return f
    return name


def predict_explain(model, meta, row):
    """row: one-row DataFrame with the model's input columns. Returns (prediction, base value, contributions)."""
    import xgboost as xgb
    feats = meta["numeric"] + meta["categorical"]
    X = row[feats]
    pred = float(model.predict(X)[0])
    pre, booster = model.named_steps["pre"], model.named_steps["m"].get_booster()
    Z = pre.transform(X)
    contrib = booster.predict(xgb.DMatrix(Z), pred_contribs=True)[0]
    names = [n.split("__", 1)[1] for n in pre.get_feature_names_out()]
    c = pd.Series(contrib[:-1], index=names)
    grouped = c.groupby([_base(n, feats) for n in c.index]).sum().sort_values(key=abs, ascending=False)
    return pred, float(contrib[-1]), grouped


def similar_phones(ref, meta, row, k=5):
    """Nearest real phones on price, RAM, storage, battery, screen and year (each scaled by its spread)."""
    cols = ["price_inr", "ram_gb", "storage_gb", "battery_mah", "display_in", "release_year"]
    R = ref.dropna(subset=["price_inr"]).copy()
    A = R[cols].copy()
    A["price_inr"] = np.log1p(A["price_inr"])
    q = row[cols].iloc[0].astype(float).copy()
    q["price_inr"] = np.log1p(q["price_inr"])
    med = A.median()
    A = A.fillna(med)
    q = q.fillna(med)
    sd = A.std().replace(0, 1)
    dist = (((A - q) / sd) ** 2).sum(axis=1) ** 0.5
    R["distance"] = dist
    return R.nsmallest(k, "distance")


def level(rating, thresholds):
    lo, hi = thresholds
    return "Low" if rating < lo else ("High" if rating >= hi else "Medium")
