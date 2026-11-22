"""Run the whole final-stage pipeline from the command line:   python run_pipeline.py

Produces every table in results/, every figure in figures/ and the saved model in models/.
The notebook Final_Pipeline.ipynb runs exactly the same steps and shows the outputs.
"""
import json
import platform
import time

import numpy as np
import pandas as pd

from src import models as M
from src import pipeline as P

t_start = time.time()
np.random.seed(M.RANDOM_STATE)
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)


def step(title):
    print(f"\n{'=' * 90}\n{title}\n{'=' * 90}", flush=True)


step("1. Data and matching")
D = P.stage_data()
work = D["work"]
print(D["match_rates"].to_string())
print(D["steps"].to_string(index=False))
if D["recovery"] is not None:
    print(D["recovery"].to_string())
print(D["count_fix"].to_string())

step("2. Exploratory data analysis")
EDA = P.stage_eda(work)
print(pd.Series(EDA["eda"]).to_string())

step("3-4. Feature-set choice and XGBoost tuning (phones up to 2024 only)")
F = P.stage_features(work)
print(F["feature_sets"].round(4).to_string(index=False))
num, cat = M.FEATURE_SETS[F["chosen"]]
xgb_params = F["xgb_params"]
print("Chosen:", F["chosen"], "| XGBoost:", xgb_params)

step("5. Main evaluation (4 models x 4 protocols, 95% bootstrap intervals)")
EV = P.stage_evaluate(work, num, cat, xgb_params)
print(EV["table"].round(3).to_string(index=False))

step("5b. Feature-set ablation (XGBoost) and paired model comparison")
print(P.stage_ablation(work, F["feature_sets"], EV["splits"]).to_string())
print(P.stage_paired_models(work, EV["preds"], EV["splits"]).to_string(index=False))

step("6. Source agreement and independence test (new matching)")
AG = P.stage_agreement(work)
print(pd.Series(AG["agreement"]).to_string())
print(AG["independence"].to_string())

step("7. Reliability-aware training")
RA = P.stage_reliability(work, num, cat, xgb_params, EV["splits"])
print("Chosen m / k:", RA["chosen"])
print(RA["summary"].to_string())

step("8. Error by review volume")
print(P.stage_error_by_volume(work, EV["preds"]).to_string())

step("9. Ordinal comparison (3 levels)")
OR = P.stage_ordinal(work, num, cat, EV["preds"], EV["splits"])
print(OR["table"].round(3).to_string(index=False))

step("10. Interpretability")
IN = P.stage_interpret(work, num, cat, xgb_params)
print(IN["shap_importance"].round(4).head(12).to_string())

step("11. Complexity")
print(P.stage_complexity(work, num, cat, xgb_params).to_string(index=False))

step("12. Final model")
_, meta = P.stage_final_model(work, num, cat, xgb_params)
print(json.dumps({k: meta[k] for k in ["model", "numeric", "categorical", "xgb_params", "trained_on_phones"]}))

import sklearn, xgboost, scipy, matplotlib  # noqa: E402
versions = {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__,
            "scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__, "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__, "platform": platform.platform()}
(P.RESULTS / "environment.json").write_text(json.dumps(versions, indent=2))
print(f"\nFinished in {time.time() - t_start:.0f} s")
