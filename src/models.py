"""Feature sets, preprocessing pipeline and model definitions.

All preprocessing (median imputation + missing-value indicators, log of skewed variables, one-hot brand with
rare brands grouped as 'infrequent', scaling for the linear model) sits inside the sklearn Pipeline, so it is
learned on training folds only.
"""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

RANDOM_STATE = 42
TARGET = "user_rating_5"

BASE_NUMERIC = ["price_inr", "ram_gb", "storage_gb", "battery_mah", "charging_w", "display_in",
                "is_amoled", "rear_mp", "front_mp", "spec_score", "antutu", "release_year"]
BASE_CATEG = ["brand"]
D3_NUMERIC = ["refresh_hz", "pixel_ppi"]             # from Dataset 3 (matched phones only)
D3_CATEG = ["chipset_vendor"]
LOG_COLS_ALL = ["price_inr", "storage_gb", "antutu", "rear_mp"]   # |skew| > 1 in the working sample

FEATURE_SETS = {
    "A: progress-stage features": (BASE_NUMERIC, BASE_CATEG),
    "B: without AnTuTu": ([c for c in BASE_NUMERIC if c != "antutu"], BASE_CATEG),
    "C: + Dataset 3 features": (BASE_NUMERIC + D3_NUMERIC, BASE_CATEG + D3_CATEG),
    "D: + Dataset 3, without AnTuTu": ([c for c in BASE_NUMERIC if c != "antutu"] + D3_NUMERIC,
                                       BASE_CATEG + D3_CATEG),
}

XGB_FIXED = dict(subsample=0.8, colsample_bytree=0.8, reg_lambda=5, min_child_weight=5, tree_method="hist",
                 random_state=RANDOM_STATE, n_jobs=-1)
XGB_GRID = {"m__max_depth": [3, 4, 6], "m__learning_rate": [0.03, 0.1], "m__n_estimators": [300, 600]}


def preprocessor(numeric, categ, scale):
    log_cols = [c for c in LOG_COLS_ALL if c in numeric]
    plain_cols = [c for c in numeric if c not in log_cols]
    log_num = Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)),
                        ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one"))])
    plain = SimpleImputer(strategy="median", add_indicator=True)
    cat = Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="unknown")),
                    ("oh", OneHotEncoder(handle_unknown="ignore", min_frequency=10, sparse_output=False))])
    ct = ColumnTransformer([("log", log_num, log_cols), ("num", plain, plain_cols), ("cat", cat, categ)],
                           sparse_threshold=0)
    return Pipeline([("ct", ct), ("scale", StandardScaler())]) if scale else ct


def make_models(numeric, categ, xgb_params):
    return {
        "Dummy (mean)": DummyRegressor(strategy="mean"),
        "Linear Regression": Pipeline([("pre", preprocessor(numeric, categ, True)), ("m", LinearRegression())]),
        "Random Forest": Pipeline([("pre", preprocessor(numeric, categ, False)),
                                   ("m", RandomForestRegressor(n_estimators=400, random_state=RANDOM_STATE,
                                                               n_jobs=-1))]),
        "XGBoost (tuned)": Pipeline([("pre", preprocessor(numeric, categ, False)),
                                     ("m", XGBRegressor(**xgb_params, **XGB_FIXED))]),
    }


def xgb_pipeline(numeric, categ, **params):
    return Pipeline([("pre", preprocessor(numeric, categ, False)), ("m", XGBRegressor(**params, **XGB_FIXED))])
