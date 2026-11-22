"""Final-stage pipeline: every table, figure and the saved model come from the functions in this file.

Order (same as the notebook Final_Pipeline.ipynb):
  1 data + matching      2 EDA                 3 feature-set choice     4 XGBoost tuning
  5 main evaluation      6 source agreement    7 reliability-aware      8 error by review volume
  9 ordinal comparison  10 interpretability   11 complexity            12 final model
"""
import json
import pickle
import time
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm, skew
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV, GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from . import evaluation as E
from . import models as M
from . import ordinal as O
from . import prep

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
RESULTS, FIGURES, MODELS_DIR = ROOT / "results", ROOT / "figures", ROOT / "models"
for d in (RESULTS, FIGURES, MODELS_DIR):
    d.mkdir(exist_ok=True)
CACHE = RESULTS / "_cache.pkl"
plt.rcParams.update({"figure.dpi": 110, "font.size": 10})
BLUE, ORANGE, GREEN, RED, GREY = "#2b5c9e", "#e07b39", "#3c9a5f", "#c0392b", "#7f7f7f"
RELIABLE_N = 500
GRID = [30, 100, 300]


def save(df, name, index=False):
    df.to_csv(RESULTS / name, index=index)
    return df


# ----------------------------------------------------------------------------- 1. data + matching
def stage_data():
    work_old, d_old, info_old = prep.build_all("key", verbose=False)
    work, d, info = prep.build_all("exact_then_key", verbose=False)
    (r1, c1), (r2, c2), (r3, c3) = info["raw_sizes"]
    save(pd.DataFrame({"Dataset": ["1 Mobile Phones Specs & Prices (2008–2026)", "2 Mobile Price and Specs (2025)",
                                   "3 Global Smartphone Database 2025"],
                       "Rows": [r1, r2, r3], "Columns": [c1, c2, c3]}), "table_dataset_sizes.csv")

    def rates(dd, col):
        out = {"Overall": dd[col].mean() * 100}
        out.update({int(y): g[col].mean() * 100 for y, g in dd.groupby("release_year")})
        return out
    mr = pd.DataFrame({"D1 → D2, progress rule (variant key)": rates(d_old, "d2_matched"),
                       "D1 → D2, final rule (exact name, then key)": rates(d, "d2_matched"),
                       "D1 → D3, progress rule (variant key)": rates(d_old, "d3_matched"),
                       "D1 → D3, final rule (exact name, then key)": rates(d, "d3_matched")}).T.round(1)
    save(mr, "table_match_rates.csv", index=True)
    steps = pd.DataFrame([{"Join": f"Dataset 1 → Dataset {k[-1]}", "Step": s, **v}
                          for k in ("d2", "d3") for s, v in info[k].items()])
    save(steps, "table_match_steps.csv")

    # Did the new rule recover the phones the manual audit said were missed?
    audit = ROOT.parent / "results" / "unmatched_audit_checked.csv"
    rec = None
    if audit.exists():
        u = pd.read_csv(audit).merge(d[["name", "d2_matched"]], on="name", how="left")
        rec = pd.crosstab(u["should_match"].map({1: "should match (audit)", 0: "no partner (audit)"}),
                          u["d2_matched"].map({True: "matched now", False: "unmatched now"}))
        save(rec, "table_missed_match_recovery.csv", index=True)
    # The manual check of 100 random Dataset 3 matches is saved in results/table_d3_match_audit.csv
    # Effect of the review-count fix ('1.5L' = 150,000 reviews, not 1.5)
    raw = prep.load_raw()[1].set_index("mobile_name")["total_ratings"]
    raw = raw[~raw.index.duplicated()]
    has = work[work["n_reviews"].notna()]
    old_n = has["mobile_name"].map(raw).apply(prep.parse_count_progress)
    edges, labs = [0, 50, 500, 5000, np.inf], ["Fewer than 50", "50 to 500", "500 to 5,000", "More than 5,000"]
    fix = pd.DataFrame({"Progress parser": pd.cut(old_n, edges, right=False, labels=labs).value_counts(),
                        "Fixed parser": pd.cut(has["n_reviews"], edges, right=False, labels=labs).value_counts()
                        }).reindex(labs)
    fix.loc["Phones whose count uses 'L' (lakh)"] = [int(has["mobile_name"].map(raw).astype(str)
                                                         .str.contains(r"\dL\b", regex=True).sum())] * 2
    save(fix, "table_review_count_fix.csv", index=True)
    new_d2 = d[(d["d2_matched"] == True) & (d_old["d2_matched"].reindex(d.index) != True)]
    save(new_d2[["name", "mobile_name", "release_year", "d2_spec_agree"]], "new_d2_matches_list.csv")
    return {"work": work, "d": d, "match_rates": mr, "steps": steps, "recovery": rec, "n_new_d2": len(new_d2),
            "count_fix": fix}


# ------------------------------------------------------------------------------------- 2. EDA
def stage_eda(work):
    y = work[M.TARGET]
    q1, q3 = y.quantile([.25, .75])
    iqr = q3 - q1
    eda = {"Phones": len(work), "Phone families": work["family"].nunique(), "Brands": work["brand"].nunique(),
           "Rating mean": round(y.mean(), 2), "Rating median": round(y.median(), 2),
           "Rating SD": round(y.std(), 3), "Rating skewness": round(skew(y), 2),
           "IQR outliers in rating": int(((y < q1 - 1.5 * iqr) | (y > q3 + 1.5 * iqr)).sum())}
    save(pd.Series(eda).to_frame("value"), "eda_summary.csv", index=True)

    feats = M.BASE_NUMERIC + M.D3_NUMERIC
    miss = (work[feats + ["brand", "chipset_vendor"]].isna().mean() * 100).round(1).sort_values(ascending=False)
    save(miss.to_frame("missing_pct"), "eda_missing_values.csv", index=True)

    # Figure 1: rating distribution
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.hist(y, bins=40, color=BLUE)
    ax.axvline(y.mean(), color=RED, ls="--", label=f"Mean = {y.mean():.2f}")
    ax.axvline(y.median(), color=GREEN, ls=":", lw=2, label=f"Median = {y.median():.2f}")
    ax.set(xlabel="User rating (out of 5)", ylabel="Number of phones",
           title=f"Distribution of user ratings (n = {len(y):,}, skewness = {skew(y):.2f})")
    ax.legend(); fig.tight_layout(); fig.savefig(FIGURES / "fig01_rating_distribution.png", dpi=200); plt.close(fig)

    # Figure 2: correlation vs RF importance (description only, fitted on all phones)
    corr = work[feats].corrwith(y)
    rf = Pipeline([("imp", SimpleImputer(strategy="median")),
                   ("rf", RandomForestRegressor(n_estimators=400, random_state=M.RANDOM_STATE, n_jobs=-1))])
    rf.fit(work[feats], y)
    imp = pd.Series(rf.named_steps["rf"].feature_importances_, index=feats)
    ci = pd.DataFrame({"Pearson r": corr.round(3), "RF importance": imp.round(3)}).sort_values(
        "Pearson r", key=abs, ascending=False)
    save(ci, "eda_corr_vs_importance.csv", index=True)
    ax = ci.iloc[::-1].plot(kind="barh", figsize=(7, 5), color=[BLUE, ORANGE])
    ax.set(xlabel="Value", title="Correlation with rating vs random-forest importance")
    plt.tight_layout(); plt.savefig(FIGURES / "fig02_corr_vs_importance.png", dpi=200); plt.close()

    # Figure 3: rating by brand (10+ phones) and by release year
    top = work["brand"].value_counts()
    top = top[top >= 30].index
    by_brand = work[work["brand"].isin(top)].groupby("brand")[M.TARGET].agg(["mean", "size"]).sort_values("mean")
    by_year = work.groupby("release_year")[M.TARGET].agg(["mean", "median", "size"])
    save(by_brand.round(3), "eda_rating_by_brand.csv", index=True)
    save(by_year.round(3), "eda_rating_by_year.csv", index=True)
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4.4), gridspec_kw={"width_ratios": [1.25, 1]})
    order = work[work["brand"].isin(top)].groupby("brand")[M.TARGET].median().sort_values().index
    a.boxplot([work.loc[work["brand"] == br, M.TARGET] for br in order], vert=False, widths=0.6,
              showmeans=True, patch_artist=True,
              boxprops=dict(facecolor="#cfe0f1", edgecolor=BLUE), medianprops=dict(color=BLUE, linewidth=1.6),
              meanprops=dict(marker="D", markerfacecolor=ORANGE, markeredgecolor=ORANGE, markersize=4),
              flierprops=dict(marker="o", markersize=3, markerfacecolor="none", markeredgecolor="grey"))
    a.set_yticks(range(1, len(order) + 1))
    a.set_yticklabels([f"{br} (n={by_brand.loc[br, 'size']})" for br in order], fontsize=8)
    a.set_xlim(0.9, 5.05); a.grid(axis="x", alpha=0.3)
    a.set(xlabel="User rating (out of 5)", title="Rating by brand (30+ phones): box plot")
    a.text(0.02, 0.02, "box = middle 50%, line = median,\n\u25c6 = mean, o = outlier (1.5 \u00d7 IQR)",
           transform=a.transAxes, fontsize=7, va="bottom")
    b.bar(by_year.index.astype(int).astype(str), by_year["mean"], color=ORANGE)
    b.set_ylim(3.8, 4.5); b.set(ylabel="Mean rating", title="Mean rating by release year")
    for i, (m_, n_) in enumerate(zip(by_year["mean"], by_year["size"])):
        b.text(i, m_ + 0.01, f"{m_:.2f}\nn={n_}", ha="center", fontsize=8)
    fig.tight_layout(); fig.savefig(FIGURES / "fig03_rating_by_brand_year.png", dpi=200); plt.close(fig)
    return {"eda": eda, "missing": miss, "corr_imp": ci}


# ----------------------------------------------------------------------- 3-4. features + tuning
def stage_features(work):
    """Choose the feature set on phones released up to 2024 only (grouped 5-fold CV)."""
    y, fam, tr = work[M.TARGET], work["family"], work["release_year"] <= 2024
    rows = []
    for name, (num, cat) in M.FEATURE_SETS.items():
        X = work[num + cat]
        s = GridSearchCV(M.xgb_pipeline(num, cat), M.XGB_GRID, cv=GroupKFold(5),
                         scoring="neg_mean_absolute_error", n_jobs=1).fit(X[tr], y[tr], groups=fam[tr])
        best = {k.replace("m__", ""): v for k, v in s.best_params_.items()}
        ms = M.make_models(num, cat, best)
        r = {"Feature set": name, "Predictors": len(num) + len(cat), "XGBoost": -s.best_score_}
        for mn in ["Linear Regression", "Random Forest"]:
            p = cross_val_predict(ms[mn], X[tr], y[tr], groups=fam[tr], cv=GroupKFold(5))
            r[mn] = mean_absolute_error(y[tr], p)
        r["Average of 3 models"] = np.mean([r["XGBoost"], r["Linear Regression"], r["Random Forest"]])
        r["XGBoost settings"] = json.dumps(best)
        rows.append(r)
    fs = pd.DataFrame(rows)
    chosen = fs.loc[fs["Average of 3 models"].idxmin(), "Feature set"]
    fs["Chosen"] = np.where(fs["Feature set"] == chosen, "yes", "")
    save(fs.round(4), "table_feature_set_selection.csv")
    best = json.loads(fs.loc[fs["Feature set"] == chosen, "XGBoost settings"].iloc[0])
    return {"feature_sets": fs, "chosen": chosen, "xgb_params": best}


# ------------------------------------------------------------------------------- 5. evaluation
def stage_evaluate(work, num, cat, xgb_params):
    X, y = work[num + cat], work[M.TARGET]
    fam = work["family"].to_numpy()
    splits = E.make_splits(work)
    models = M.make_models(num, cat, xgb_params)
    rows, preds, times = [], {}, {}
    for mname, model in models.items():
        t0 = time.time()
        for proto, sp in splits.items():
            p = E.oof_predict(model, X, y, sp)
            ok = ~np.isnan(p)
            r = {"Protocol": proto, "Model": mname, "Test phones": int(ok.sum()), **E.metrics(y[ok], p[ok]),
                 **E.bootstrap_ci(y[ok], p[ok], fam[ok])}
            rows.append(r)
            preds[(proto, mname)] = p
        times[mname] = time.time() - t0
    tab = pd.DataFrame(rows)
    save(tab.round(3), "table_model_performance.csv")

    # Kaggle-style random split vs honest protocols: how much is the random-split score inflated?
    piv = tab.pivot(index="Model", columns="Protocol", values="R2")[E.PROTOCOLS]
    save(piv.round(3), "table_r2_by_protocol.csv", index=True)
    pm = tab.pivot(index="Model", columns="Protocol", values="MAE")[E.PROTOCOLS]
    save(pm.round(3), "table_mae_by_protocol.csv", index=True)

    # Figure 4: R² by protocol, with 95% intervals
    fig, ax = plt.subplots(figsize=(9, 4))
    mlist = ["Linear Regression", "Random Forest", "XGBoost (tuned)"]
    width = 0.26
    xs = np.arange(len(E.PROTOCOLS))
    for i, (mn, col) in enumerate(zip(mlist, [GREY, ORANGE, BLUE])):
        sub = tab[tab["Model"] == mn].set_index("Protocol").loc[E.PROTOCOLS]
        err = [sub["R2"] - sub["R2 low"], sub["R2 high"] - sub["R2"]]
        ax.bar(xs + (i - 1) * width, sub["R2"], width, yerr=err, capsize=3, color=col, label=mn)
        for x_, v in zip(xs + (i - 1) * width, sub["R2"]):
            ax.text(x_ + 0.02, 0.012 if v >= 0 else -0.03, f"{v:.2f}", ha="center", fontsize=7, color="white"
                    if v > 0.05 else "black", fontweight="bold")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(xs); ax.set_xticklabels(["Random 5-fold\n(Kaggle style)", "Grouped 5-fold", "Leave-one-\nbrand-out",
                                           "Future-year\n(≤2024 → 2025–26)"])
    ax.set(ylabel="R² (higher is better)", title="R² under four evaluation protocols (95% bootstrap interval)")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(FIGURES / "fig04_r2_by_protocol.png", dpi=200)
    plt.close(fig)

    # Figure 4b: MAE by protocol (lower is better) - shows how a random split hides the drop on new phones
    fig, ax = plt.subplots(figsize=(9, 4))
    for i, (mn, col) in enumerate(zip(["Dummy (mean)"] + mlist, ["#c9c9c9", GREY, ORANGE, BLUE])):
        sub = tab[tab["Model"] == mn].set_index("Protocol").loc[E.PROTOCOLS]
        err = [sub["MAE"] - sub["MAE low"], sub["MAE high"] - sub["MAE"]]
        ax.bar(xs + (i - 1.5) * 0.2, sub["MAE"], 0.2, yerr=err, capsize=2, color=col,
               label="Baseline (mean)" if mn.startswith("Dummy") else mn)
        for x_, v in zip(xs + (i - 1.5) * 0.2, sub["MAE"]):
            ax.text(x_, 0.006, f"{v:.3f}", ha="center", va="bottom", fontsize=6.5, rotation=90,
                    color="black" if mn.startswith("Dummy") else "white")
    ax.set_xticks(xs); ax.set_xticklabels(["Random 5-fold\n(Kaggle style)", "Grouped 5-fold", "Leave-one-\nbrand-out",
                                           "Future-year\n(≤2024 → 2025–26)"])
    ax.set(ylabel="MAE in rating points (lower is better)",
           title="Average error under four evaluation protocols (95% bootstrap interval)")
    ax.legend(fontsize=8, ncol=2); fig.tight_layout(); fig.savefig(FIGURES / "fig04b_mae_by_protocol.png", dpi=200)
    plt.close(fig)
    return {"table": tab, "preds": preds, "splits": splits, "fit_times": times}


# ------------------------------------------------------------------------- 6. source agreement
def stage_agreement(work):
    pairs = work[(work["d2_matched"] == True) & work["d2_rating"].notna()].copy()
    pairs["abs_diff"] = (pairs[M.TARGET] - pairs["d2_rating"]).abs()
    pairs["identical"] = pairs["abs_diff"] < 1e-9
    bands = pd.cut(pairs["d2_n_ratings"], [0, 50, 500, 5000, np.inf], right=False,
                   labels=["Fewer than 50", "50 to 500", "500 to 5,000", "More than 5,000"])
    pairs["band"] = bands.astype(str)
    rng = np.random.default_rng(M.RANDOM_STATE)

    def expected_identical(r, n, sigma, sims=2000):
        se = sigma / np.sqrt(np.maximum(n, 1))[:, None]
        true = r[:, None] + rng.uniform(-0.05, 0.05, (len(r), sims))
        a = np.clip(true + rng.normal(0, 1, (len(r), sims)) * se, 1, 5).round(1)
        b = np.clip(true + rng.normal(0, 1, (len(r), sims)) * se, 1, 5).round(1)
        return (a == b).mean(axis=1)

    sim = pairs.dropna(subset=["d2_n_ratings"]).copy()
    for s in (0.7, 1.0, 1.3):
        sim[f"exp_{s}"] = expected_identical(sim[M.TARGET].to_numpy(), sim["d2_n_ratings"].to_numpy(), s)
    order = ["Fewer than 50", "50 to 500", "500 to 5,000", "More than 5,000"]
    ind = sim.groupby("band").agg(Phones=("identical", "size"), Observed=("identical", "mean"),
                                  Expected=("exp_1.0", "mean"), Expected_sigma_1_3=("exp_1.3", "mean"),
                                  Expected_sigma_0_7=("exp_0.7", "mean")).reindex(order)
    ind.loc["All phones"] = [len(sim), sim["identical"].mean(), sim["exp_1.0"].mean(), sim["exp_1.3"].mean(),
                             sim["exp_0.7"].mean()]
    for c in ind.columns[1:]:
        ind[c] = (ind[c] * 100).round(1)
    ind["Phones"] = ind["Phones"].astype(int)
    # z test per review band: observed identical ratings vs expected under independence (sigma = 1.0)
    zs, ps = [], []
    for b in order + ["All phones"]:
        g = sim if b == "All phones" else sim[sim["band"] == b]
        pe = g["exp_1.0"].to_numpy()
        zb = (g["identical"].sum() - pe.sum()) / np.sqrt((pe * (1 - pe)).sum())
        zs.append(round(zb, 1)); ps.append(f"{2 * norm.sf(abs(zb)):.1e}")
    ind["z"], ind["p (two-sided)"] = zs, ps
    save(ind, "table_independence_test.csv", index=True)
    p = sim["exp_1.0"].to_numpy()
    z = (sim["identical"].sum() - p.sum()) / np.sqrt((p * (1 - p)).sum())
    agree = {"Phones rated in both sources": len(pairs),
             "Pearson r": round(pairs[M.TARGET].corr(pairs["d2_rating"]), 3),
             "Mean absolute difference": round(pairs["abs_diff"].mean(), 3),
             "Identical ratings (%)": round(pairs["identical"].mean() * 100, 1),
             "Independence test z": round(z, 1), "Independence test p": f"{2 * norm.sf(abs(z)):.1e}"}
    strict = pairs[pairs["d2_spec_agree"] == True]
    agree["Strict pairs (RAM/storage agree)"] = len(strict)
    agree["Strict pairs Pearson r"] = round(strict[M.TARGET].corr(strict["d2_rating"]), 3)
    save(pd.Series(agree).to_frame("value"), "table_source_agreement.csv", index=True)
    return {"independence": ind, "agreement": agree}


# ---------------------------------------------------------------------- 7. reliability-aware
def stage_reliability(work, num, cat, xgb_params, splits_all):
    X, y = work[num + cat], work[M.TARGET]
    fam = work["family"].to_numpy()
    n_rev = work["n_reviews"].to_numpy(float)
    n_med = np.nanmedian(n_rev)
    reliable = n_rev >= RELIABLE_N
    year = work["release_year"]

    def rel(n, m):
        n = np.where(np.isnan(n), n_med, n)
        return n / (n + m)

    def shrink(idx, k):
        y_tr = y.iloc[idx]
        stats = y_tr.groupby(work["brand"].iloc[idx]).agg(["mean", "size"])
        prior_b = stats["mean"].where(stats["size"] >= 10, y_tr.mean())
        prior = work["brand"].iloc[idx].map(prior_b).to_numpy()
        lam = rel(n_rev[idx], k)
        return lam * y_tr.to_numpy() + (1 - lam) * prior

    models = M.make_models(num, cat, xgb_params)
    ra_models = ["Linear Regression", "Random Forest", "XGBoost (tuned)"]

    def run(mname, method, v, splits):
        pred = np.full(len(y), np.nan)
        for tr, te in splits:
            if method == "weight":
                m = E.fit_with(models[mname], X.iloc[tr], y.iloc[tr], rel(n_rev[tr], v))
            elif method == "shrink":
                m = E.fit_with(models[mname], X.iloc[tr], pd.Series(shrink(tr, v), index=y.index[tr]))
            else:
                m = E.fit_with(models[mname], X.iloc[tr], y.iloc[tr])
            pred[te] = m.predict(X.iloc[te])
        return pred

    # choose m and k on <=2024 phones (grouped 5-fold), by MAE on reliable phones
    tune = np.where(year <= 2024)[0]
    inner = [(tune[a], tune[b]) for a, b in GroupKFold(5).split(tune, groups=work["family"].iloc[tune])]
    rel_t = tune[reliable[tune]]
    sel = []
    for mn in ra_models:
        for meth in ["weight", "shrink"]:
            for v in GRID:
                p = run(mn, meth, v, inner)
                sel.append({"Model": mn, "Method": meth, "Value": v,
                            "MAE reliable (≤2024 CV)": mean_absolute_error(y.iloc[rel_t], p[rel_t])})
    sel = pd.DataFrame(sel)
    chosen = sel.loc[sel.groupby(["Model", "Method"])["MAE reliable (≤2024 CV)"].idxmin()].set_index(
        ["Model", "Method"])["Value"].to_dict()
    save(sel.round(4), "table_reliability_parameter_selection.csv")

    protos = {k: v for k, v in splits_all.items() if not k.startswith("Random")}
    rows, ra_pred = [], {}
    for mn in ra_models:
        for proto, sp in protos.items():
            base = run(mn, "none", None, sp)
            ra_pred[(proto, mn, "Ordinary")] = base
            for label, meth in [("Weighted", "weight"), ("Shrinkage", "shrink")]:
                new = run(mn, meth, chosen[(mn, meth)], sp)
                ra_pred[(proto, mn, label)] = new
                for subset, mask in [("All test phones", ~np.isnan(base)),
                                     ("Reliable test phones", ~np.isnan(base) & reliable)]:
                    if mask.sum() < 20:
                        continue
                    est, lo, hi = E.boot_mae_diff(y.to_numpy()[mask], new[mask], base[mask], fam[mask])
                    bm = mean_absolute_error(y.to_numpy()[mask], base[mask])
                    rows.append({"Model": mn, "Protocol": proto, "Variant": label, "m or k": chosen[(mn, meth)],
                                 "Test phones": subset, "Phones": int(mask.sum()), "Ordinary MAE": bm,
                                 "New MAE": bm + est, "Change (%)": est / bm * 100, "95% CI low": lo,
                                 "95% CI high": hi, "Significant": "yes" if (hi < 0 or lo > 0) else "no"})
    boot = pd.DataFrame(rows)
    save(boot.round(4), "table_reliability_aware_bootstrap.csv")
    t = boot[boot["Test phones"] == "All test phones"].copy()
    t["cell"] = t["Change (%)"].map(lambda v: f"{v:+.1f}") + np.where(t["Significant"] == "yes", "*", "")
    summ = t.pivot_table(index=["Protocol", "Variant"], columns="Model", values="cell", aggfunc="first")
    summ = summ.reindex(index=pd.MultiIndex.from_product([list(protos), ["Weighted", "Shrinkage"]]),
                        columns=ra_models)
    save(summ, "table_reliability_aware_summary.csv", index=True)
    cov = pd.DataFrame({"Phones": work.groupby("release_year").size(),
                        "With review count": work.groupby("release_year")["n_reviews"].apply(
                            lambda s: s.notna().sum()),
                        "Reliable (500+)": pd.Series(reliable, index=work.index).groupby(year).sum()})
    cov.loc["All"] = cov.sum()
    save(cov.astype(int), "table_review_count_coverage.csv", index=True)

    # Figure 5: change in MAE with 95% interval
    fig, ax = plt.subplots(figsize=(8, 6.5))
    ticks, labels, row = [], [], 0
    short = {"Grouped 5-fold": "Grouped", "Leave-one-brand-out": "Unseen brand",
             "Future-year (≤2024 → 2025–26)": "Future"}
    for proto in protos:
        for mn in ra_models:
            for off, (lab, col, mk) in zip([-0.17, 0.17], [("Weighted", BLUE, "o"), ("Shrinkage", RED, "s")]):
                r = t[(t["Protocol"] == proto) & (t["Model"] == mn) & (t["Variant"] == lab)].iloc[0]
                pct = lambda v: v / r["Ordinary MAE"] * 100
                est = r["New MAE"] - r["Ordinary MAE"]
                ax.errorbar(pct(est), row + off, xerr=[[pct(est - r["95% CI low"])], [pct(r["95% CI high"] - est)]],
                            fmt=mk, color=col, ms=5, capsize=3, label=lab if row == 0 else None)
            ticks.append(row); labels.append(f"{short[proto]}: {mn.replace(' (tuned)', '')}")
            row += 1
        row += 0.6
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(ticks); ax.set_yticklabels(labels); ax.invert_yaxis()
    ax.set(xlabel="Change in MAE vs ordinary training (%)   ← better | worse →",
           title="Reliability-aware training (95% bootstrap interval)")
    ax.legend(loc="lower left"); fig.tight_layout(); fig.savefig(FIGURES / "fig05_reliability_aware.png", dpi=200)
    plt.close(fig)
    return {"selection": sel, "chosen": chosen, "boot": boot, "summary": summ, "coverage": cov,
            "ra_pred": ra_pred, "reliable": reliable, "n_reviews": n_rev}


# ------------------------------------------------------------------- 8. error by review volume
def stage_error_by_volume(work, preds):
    n_rev = work["n_reviews"]
    band = pd.cut(n_rev, [0, 50, 500, 5000, np.inf], right=False,
                  labels=["Fewer than 50", "50 to 500", "500 to 5,000", "More than 5,000"])
    band = band.cat.add_categories("No review count").fillna("No review count")
    y = work[M.TARGET].to_numpy()
    out = pd.DataFrame({"Phones": band.value_counts().reindex(band.cat.categories)})
    out["Rating SD"] = work.groupby(band, observed=False)[M.TARGET].std().round(3)
    for mn in ["Linear Regression", "Random Forest", "XGBoost (tuned)"]:
        err = np.abs(y - preds[("Grouped 5-fold", mn)])
        out[f"MAE {mn}"] = pd.Series(err).groupby(band.values, observed=False).mean().reindex(out.index).round(3)
    save(out, "table_error_by_review_volume.csv", index=True)
    eb = out.loc[["Fewer than 50", "50 to 500", "500 to 5,000", "More than 5,000"]]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    xs = np.arange(4)
    for i, (mn, col) in enumerate(zip(["Linear Regression", "Random Forest", "XGBoost (tuned)"],
                                      [GREY, ORANGE, BLUE])):
        ax.bar(xs + (i - 1) * 0.26, eb[f"MAE {mn}"], 0.26, color=col, label=mn)
    ax.set_xticks(xs); ax.set_xticklabels(["<50", "50–500", "500–5,000", ">5,000"])
    ax.set(xlabel="Number of reviews behind the rating", ylabel="MAE (rating points)",
           title="Prediction error by review volume (grouped 5-fold CV)")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(FIGURES / "fig06_error_by_review_volume.png", dpi=200)
    plt.close(fig)
    return out


# ------------------------------------------------------------------------- 9. ordinal comparison
def stage_ordinal(work, num, cat, preds, splits_all):
    X = work[num + cat]
    lv = O.to_level(work[M.TARGET])
    counts = pd.Series(lv).value_counts().sort_index()
    rows = []
    protos = [p for p in splits_all if not p.startswith("Random")]
    cands = {"Random Forest classifier (baseline, ignores order)": O.rf_classifier(num, cat),
             "Ordinal logistic regression": O.OrdinalLogit(num, cat)}
    for proto in protos:
        sp = splits_all[proto]
        tested = np.zeros(len(lv), bool)
        for _, te in sp:
            tested[te] = True
        for name, mdl in cands.items():
            p = O.oof_levels(mdl, X, lv, sp)
            rows.append({"Protocol": proto, "Model": name, **O.level_scores(lv[tested], p[tested])})
        reg = O.to_level(preds[(proto, "XGBoost (tuned)")][tested])
        rows.append({"Protocol": proto, "Model": "XGBoost regression, then cut into levels",
                     **O.level_scores(lv[tested], reg)})
        maj = np.full(tested.sum(), 1)
        rows.append({"Protocol": proto, "Model": "Always 'Medium' (majority level)",
                     **O.level_scores(lv[tested], maj)})
    tab = pd.DataFrame(rows)
    save(tab.round(3), "table_ordinal_comparison.csv")
    save(pd.DataFrame({"Level": O.LEVELS, "Phones": counts.values,
                       "Share (%)": (counts.values / counts.sum() * 100).round(1)}), "table_ordinal_levels.csv")
    return {"table": tab, "levels": counts}


# ---------------------------------------------------------------------- 10. interpretability
def feature_names(fitted_pipe):
    pre = fitted_pipe.named_steps["pre"]
    names = pre.get_feature_names_out()
    return [n.split("__", 1)[1] for n in names]


def stage_interpret(work, num, cat, xgb_params):
    import xgboost as xgb
    X, y = work[num + cat], work[M.TARGET]
    models = M.make_models(num, cat, xgb_params)
    xg = clone(models["XGBoost (tuned)"]).fit(X, y)
    Z = xg.named_steps["pre"].transform(X)
    contrib = xg.named_steps["m"].get_booster().predict(xgb.DMatrix(Z), pred_contribs=True)
    names = feature_names(xg) + ["bias"]
    sh = pd.DataFrame(contrib, columns=names).drop(columns="bias")
    # group one-hot / indicator columns back to their original feature
    def base(n):
        for f in num + cat:
            if n == f or n.startswith(f + "_") or n == f"missingindicator_{f}":
                return f
        return n
    grouped = sh.T.groupby([base(c) for c in sh.columns]).sum().T
    imp = grouped.abs().mean().sort_values(ascending=False)
    save(imp.round(4).to_frame("mean |SHAP| (rating points)"), "table_shap_importance.csv", index=True)
    # direction: correlation of feature value with its SHAP value
    # all features; the three Dataset 3 features in orange
    d3 = {"refresh_hz", "pixel_ppi", "chipset_vendor"}
    nice = {"spec_score": "Spec score", "brand": "Brand", "price_inr": "Price", "battery_mah": "Battery",
            "antutu": "AnTuTu score", "charging_w": "Charging power", "display_in": "Screen size", "ram_gb": "RAM",
            "refresh_hz": "Refresh rate (D3)", "front_mp": "Front camera", "release_year": "Release year",
            "storage_gb": "Storage", "pixel_ppi": "Pixel density (D3)", "chipset_vendor": "Chipset maker (D3)",
            "rear_mp": "Rear camera", "is_amoled": "AMOLED screen"}
    fig, ax = plt.subplots(figsize=(7, 5.6))
    top = imp[::-1]
    ax.barh([nice.get(f, f) for f in top.index], top.values,
            color=[ORANGE if f in d3 else BLUE for f in top.index])
    for i, v in enumerate(top.values):
        ax.text(v + 0.0005, i, f"{v:.3f}", va="center", fontsize=8)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=BLUE, label="Dataset 1 features"),
                       Patch(color=ORANGE, label="Dataset 3 features (D3)")], loc="lower right", fontsize=8)
    ax.set(xlabel="Mean |SHAP value| (rating points)", title="XGBoost: average effect of each feature on the rating")
    fig.tight_layout(); fig.savefig(FIGURES / "fig07_shap_importance.png", dpi=200); plt.close(fig)

    lr = clone(models["Linear Regression"]).fit(X, y)
    coef = pd.Series(lr.named_steps["m"].coef_, index=feature_names(lr)).sort_values(key=abs, ascending=False)
    save(coef.round(4).to_frame("standardised coefficient"), "table_linear_coefficients.csv", index=True)
    return {"shap_importance": imp, "lr_coef": coef}


# ------------------------------------------------------------------------------ 11. complexity
def stage_complexity(work, num, cat, xgb_params):
    X, y = work[num + cat], work[M.TARGET]
    rows = []
    for mn, mdl in M.make_models(num, cat, xgb_params).items():
        t0 = time.perf_counter(); fitted = clone(mdl).fit(X, y); fit_s = time.perf_counter() - t0
        big = pd.concat([X] * 7, ignore_index=True).iloc[:10000]
        t0 = time.perf_counter(); fitted.predict(big); pred_ms = (time.perf_counter() - t0) / len(big) * 1000 * 1000
        path = MODELS_DIR / "_tmp.joblib"; joblib.dump(fitted, path); size_kb = path.stat().st_size / 1024
        path.unlink()
        if mn.startswith("Dummy"):
            params = "1 (the mean)"
        elif mn.startswith("Linear"):
            params = f"{fitted.named_steps['m'].coef_.size + 1} coefficients"
        elif mn.startswith("Random"):
            params = f"400 trees, {sum(t.tree_.node_count for t in fitted.named_steps['m'].estimators_):,} nodes"
        else:
            b = fitted.named_steps["m"].get_booster()
            params = f"{b.num_boosted_rounds()} trees, depth ≤ {xgb_params['max_depth']}"
        rows.append({"Model": mn, "Training time on all phones (s)": round(fit_s, 2),
                     "Prediction time per 1,000 phones (ms)": round(pred_ms, 2),
                     "Saved model size (KB)": round(size_kb, 1), "Size of the model": params})
    return save(pd.DataFrame(rows), "table_complexity.csv")


# ----------------------------------------------------------------------------- 12. final model
def stage_final_model(work, num, cat, xgb_params, final_name="XGBoost (tuned)"):
    X, y = work[num + cat], work[M.TARGET]
    model = clone(M.make_models(num, cat, xgb_params)[final_name]).fit(X, y)
    ranges = {c: {"min": float(work[c].min()), "max": float(work[c].max()), "median": float(work[c].median())}
              for c in num}
    meta = {"model": final_name, "numeric": num, "categorical": cat, "xgb_params": xgb_params,
            "trained_on_phones": int(len(work)), "target": "user rating out of 5",
            "ordinal_thresholds": list(O.THRESHOLDS), "feature_ranges": ranges,
            "brands": sorted(work["brand"].dropna().unique().tolist()),
            "chipset_vendors": sorted(work["chipset_vendor"].dropna().unique().tolist())}
    joblib.dump(model, MODELS_DIR / "final_model.joblib")
    from .explain import save_portable
    save_portable(model)                       # same model in a format that loads the same on every computer
    (MODELS_DIR / "final_model_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    cols = ["name", "brand", "release_year", M.TARGET, "n_reviews", "family"] + [c for c in num if c != "release_year"] + \
        [c for c in cat if c != "brand"] + ["url"]
    work[cols].to_csv(MODELS_DIR / "reference_phones.csv", index=False)
    return model, meta


# ------------------------------------------------------------- 5b. ablation and paired comparison
def stage_ablation(work, feature_sets_table, splits_all):
    """XGBoost with each feature set under every protocol (each set uses its own <=2024-tuned settings)."""
    y = work[M.TARGET]
    fam = work["family"].to_numpy()
    rows = []
    for _, r in feature_sets_table.iterrows():
        num, cat = M.FEATURE_SETS[r["Feature set"]]
        params = json.loads(r["XGBoost settings"])
        X = work[num + cat]
        mdl = M.xgb_pipeline(num, cat, **params)
        for proto, sp in splits_all.items():
            p = E.oof_predict(mdl, X, y, sp)
            ok = ~np.isnan(p)
            rows.append({"Feature set": r["Feature set"], "Protocol": proto, **E.metrics(y[ok], p[ok])})
    tab = pd.DataFrame(rows)
    out = tab.pivot(index="Feature set", columns="Protocol", values="MAE")[E.PROTOCOLS]
    save(tab.round(4), "table_feature_ablation_long.csv")
    save(out.round(4), "table_feature_ablation_mae.csv", index=True)
    return out


def stage_paired_models(work, preds, splits_all):
    """Is XGBoost really better than the baseline, Linear Regression and Random Forest? Paired bootstrap of MAE."""
    y = work[M.TARGET].to_numpy()
    fam = work["family"].to_numpy()
    rows = []
    for proto in splits_all:
        base = preds[(proto, "XGBoost (tuned)")]
        ok = ~np.isnan(base)
        for other in ["Dummy (mean)", "Linear Regression", "Random Forest"]:
            p = preds[(proto, other)]
            est, lo, hi = E.boot_mae_diff(y[ok], base[ok], p[ok], fam[ok])
            om = mean_absolute_error(y[ok], p[ok])
            rows.append({"Protocol": proto, "XGBoost vs": other, "Other MAE": om, "XGBoost MAE": om + est,
                         "Difference": est, "Change (%)": est / om * 100, "95% CI low": lo, "95% CI high": hi,
                         "XGBoost significantly better": "yes" if hi < 0 else ("worse" if lo > 0 else "no")})
    return save(pd.DataFrame(rows).round(4), "table_paired_model_comparison.csv")
