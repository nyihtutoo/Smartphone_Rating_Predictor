"""Evaluation protocols, metrics and bootstrap intervals."""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, LeaveOneGroupOut
from sklearn.pipeline import Pipeline

from .models import RANDOM_STATE

PROTOCOLS = ["Random 5-fold (Kaggle style)", "Grouped 5-fold", "Leave-one-brand-out",
             "Future-year (≤2024 → 2025–26)"]


def brand_groups(work, min_phones=10):
    """Leave-one-brand-out groups: brands with fewer than 10 phones are held out together as 'Other brands'."""
    counts = work["brand"].value_counts()
    return work["brand"].where(work["brand"].map(counts) >= min_phones, "Other brands")


def make_splits(work, include_random=True):
    """Train/test index pairs for every protocol."""
    n = len(work)
    X0 = np.zeros((n, 1))
    year = work["release_year"].to_numpy()
    splits = {}
    if include_random:
        splits["Random 5-fold (Kaggle style)"] = list(KFold(5, shuffle=True, random_state=RANDOM_STATE).split(X0))
    splits["Grouped 5-fold"] = list(GroupKFold(n_splits=5).split(X0, groups=work["family"]))
    splits["Leave-one-brand-out"] = list(LeaveOneGroupOut().split(X0, groups=brand_groups(work)))
    splits["Future-year (≤2024 → 2025–26)"] = [(np.where(year <= 2024)[0], np.where(year >= 2025)[0])]
    return splits


def metrics(y_true, pred):
    return {"MAE": mean_absolute_error(y_true, pred),
            "RMSE": mean_squared_error(y_true, pred) ** 0.5,
            "R2": r2_score(y_true, pred)}


def fit_with(model, X_tr, y_tr, sample_weight=None):
    m = clone(model)
    if sample_weight is None:
        return m.fit(X_tr, y_tr)
    key = "m__sample_weight" if isinstance(m, Pipeline) else "sample_weight"
    return m.fit(X_tr, y_tr, **{key: sample_weight})


def oof_predict(model, X, y, splits, weights=None, labels=None):
    """Out-of-split predictions (NaN for phones never tested).

    weights: array of training weights (or None); labels: replacement training labels (or None)."""
    pred = np.full(len(y), np.nan)
    y_train_all = y if labels is None else labels
    for tr, te in splits:
        w = None if weights is None else np.asarray(weights)[tr]
        m = fit_with(model, X.iloc[tr], y_train_all.iloc[tr], w)
        pred[te] = m.predict(X.iloc[te])
    return pred


def bootstrap_ci(y_true, pred, clusters, sims=2000, seed=RANDOM_STATE):
    """95% interval of MAE and R², resampling phone families (clusters) with replacement."""
    y_true, pred = np.asarray(y_true, float), np.asarray(pred, float)
    codes, uniq = pd.factorize(pd.Series(clusters))
    k = len(uniq)
    # per-cluster sufficient statistics
    df = pd.DataFrame({"g": codes, "ae": np.abs(y_true - pred), "se": (y_true - pred) ** 2,
                       "y": y_true, "y2": y_true ** 2})
    agg = df.groupby("g").agg(n=("ae", "size"), ae=("ae", "sum"), se=("se", "sum"), y=("y", "sum"),
                              y2=("y2", "sum")).to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, k, (sims, k))
    tot = agg[idx].sum(axis=1)                    # sims x 5
    n, ae, se, sy, sy2 = tot.T
    mae = ae / n
    sst = sy2 - sy ** 2 / n
    r2 = 1 - se / sst
    return {"MAE low": np.percentile(mae, 2.5), "MAE high": np.percentile(mae, 97.5),
            "R2 low": np.percentile(r2, 2.5), "R2 high": np.percentile(r2, 97.5)}


def boot_mae_diff(y_true, p_new, p_base, clusters, sims=2000, seed=RANDOM_STATE):
    """MAE(new) - MAE(base) on the same phones with a 95% interval; negative = new is better."""
    d = pd.DataFrame({"g": clusters, "diff": np.abs(y_true - p_new) - np.abs(y_true - p_base)})
    agg = d.groupby("g")["diff"].agg(["sum", "size"])
    s, c = agg["sum"].to_numpy(), agg["size"].to_numpy()
    idx = np.random.default_rng(seed).integers(0, len(s), (sims, len(s)))
    boots = s[idx].sum(1) / c[idx].sum(1)
    return s.sum() / c.sum(), np.percentile(boots, 2.5), np.percentile(boots, 97.5)
