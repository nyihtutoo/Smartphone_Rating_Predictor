"""Secondary task: three ordered rating levels (Low / Medium / High).

Thresholds come from the observed rating distribution (not equal-frequency cuts):
  Low    < 4.0  - the long left tail (about the lowest 10% of phones)
  Medium 4.0 to 4.4 - the main body of ratings
  High   >= 4.5 - clearly above the typical rating (about the top 18%)
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, cohen_kappa_score
from sklearn.pipeline import Pipeline

from .models import RANDOM_STATE, preprocessor

THRESHOLDS = (4.0, 4.5)
LEVELS = ["Low (<4.0)", "Medium (4.0–4.4)", "High (≥4.5)"]


def to_level(rating):
    r = np.asarray(rating, float)
    return np.digitize(r, THRESHOLDS)          # 0, 1, 2


class OrdinalLogit(BaseEstimator, ClassifierMixin):
    """Proportional-odds (ordinal logistic) regression on preprocessed features (statsmodels OrderedModel)."""

    def __init__(self, numeric=None, categ=None):
        self.numeric, self.categ = numeric, categ

    def fit(self, X, y):
        from statsmodels.miscmodels.ordinal_model import OrderedModel
        self.pre_ = preprocessor(self.numeric, self.categ, True).fit(X)
        Z = self.pre_.transform(X)
        self.keep_ = Z.std(axis=0) > 1e-8                     # drop constant columns
        Z = Z[:, self.keep_]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.res_ = OrderedModel(np.asarray(y, int), Z, distr="logit").fit(method="bfgs", maxiter=2000,
                                                                                 disp=False)
        return self

    def predict_proba(self, X):
        Z = self.pre_.transform(X)[:, self.keep_]
        return np.asarray(self.res_.model.predict(self.res_.params, exog=Z))

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)


def rf_classifier(numeric, categ):
    return Pipeline([("pre", preprocessor(numeric, categ, False)),
                     ("m", RandomForestClassifier(n_estimators=400, random_state=RANDOM_STATE, n_jobs=-1))])


def level_scores(true_lv, pred_lv):
    return {"QWK": cohen_kappa_score(true_lv, pred_lv, weights="quadratic"),
            "MAE (levels)": np.mean(np.abs(true_lv - pred_lv)),
            "Accuracy": accuracy_score(true_lv, pred_lv),
            "Two levels off (%)": np.mean(np.abs(true_lv - pred_lv) == 2) * 100}


def oof_levels(model, X, levels, splits):
    pred = np.full(len(levels), -1)
    for tr, te in splits:
        m = clone(model).fit(X.iloc[tr], levels[tr])
        pred[te] = m.predict(X.iloc[te])
    return pred
