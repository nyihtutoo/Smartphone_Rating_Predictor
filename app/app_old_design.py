"""Smartphone Rating Predictor - demo app for the Neat 2 final project.

Run from the Final Project folder:
    streamlit run app/app.py
"""
import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.explain import FRIENDLY, RESULTS, level, load, predict_explain, similar_phones  # noqa: E402

st.set_page_config(page_title="Smartphone Rating Predictor", page_icon="📱", layout="wide")


@st.cache_resource
def get_model():
    return load()


@st.cache_data
def get_results():
    perf = pd.read_csv(RESULTS / "table_model_performance.csv")
    return perf


model, meta, ref = get_model()
perf = get_results()
FEATS = meta["numeric"] + meta["categorical"]
R = meta["feature_ranges"]


INR_PER_USD = 94.0   # approximate rate, November 2026 (Wise mid-market rate). Used for display only.


def mae_for(protocol_start):
    row = perf[perf["Protocol"].str.startswith(protocol_start) & (perf["Model"] == meta["model"])]
    return float(row["MAE"].iloc[0]) if len(row) else np.nan


st.title("📱 Smartphone Rating Predictor")
st.caption("Predicts the average user rating (out of 5) of a smartphone from its specifications and price. "
           f"Model: {meta['model']}, trained on {meta['trained_on_phones']:,} rated phones released 2022–2026 "
           "(91mobiles, MySmartPrice and the Global Smartphone Database). Neat 2 final project, AIT.")

tab_pred, tab_model, tab_about = st.tabs(["Predict a rating", "How good is the model?", "About the data"])

# ------------------------------------------------------------------------------------------ inputs
with st.sidebar:
    st.header("Phone specifications")
    names = ["— enter specs by hand —"] + sorted(ref["name"].dropna().unique().tolist())
    start = st.selectbox("Start from a real phone (optional)", names, index=0,
                         help="Fills the form with a phone from the dataset. You can then change any value.")
    base = ref[ref["name"] == start].iloc[0] if start != names[0] else None

    def val(col, default):
        if base is not None and col in base and pd.notna(base[col]):
            return base[col]
        return default

    brands = meta["brands"]
    b_default = val("brand", "Samsung")
    brand = st.selectbox("Brand", brands + ["Other (new brand)"],
                         index=brands.index(b_default) if b_default in brands else 0)
    base_inr = float(val("price_inr", 20000))
    usd_default = int(round(base_inr / INR_PER_USD))
    price_usd = st.number_input("Price (US$)", min_value=10, max_value=3200, step=5, value=usd_default,
                                help="Prices in the data are Indian rupees. The model uses rupees; "
                                     f"US$ is converted at about ₹{INR_PER_USD:.0f} per US$.")
    # keep the exact rupee price of a real phone if the user did not change it
    price = base_inr if (base is not None and price_usd == usd_default) else price_usd * INR_PER_USD
    st.caption(f"≈ ₹{price:,.0f} (Indian price used by the model)")
    c1, c2 = st.columns(2)
    ram = c1.selectbox("RAM (GB)", [1, 2, 3, 4, 6, 8, 12, 16, 24],
                       index=[1, 2, 3, 4, 6, 8, 12, 16, 24].index(int(val("ram_gb", 8)))
                       if int(val("ram_gb", 8)) in [1, 2, 3, 4, 6, 8, 12, 16, 24] else 5)
    storage = c2.selectbox("Storage (GB)", [16, 32, 64, 128, 256, 512, 1024],
                           index=[16, 32, 64, 128, 256, 512, 1024].index(int(val("storage_gb", 128)))
                           if int(val("storage_gb", 128)) in [16, 32, 64, 128, 256, 512, 1024] else 3)
    battery = st.slider("Battery (mAh)", 2000, 8000, int(val("battery_mah", 5000)), step=50)
    charging = st.slider("Charging power (W)", 5, 150, int(min(val("charging_w", 33), 150)))
    display = st.slider("Screen size (inch)", 4.0, 8.2, float(val("display_in", 6.6)), step=0.01)
    amoled = st.checkbox("AMOLED / OLED screen", value=bool(val("is_amoled", 1.0) == 1))
    c3, c4 = st.columns(2)
    rear = c3.number_input("Main rear camera (MP)", 2, 200, int(val("rear_mp", 50)))
    front = c4.number_input("Front camera (MP)", 2, 60, int(val("front_mp", 16)))
    year = st.selectbox("Release year", [2022, 2023, 2024, 2025, 2026],
                        index=[2022, 2023, 2024, 2025, 2026].index(int(val("release_year", 2024))))
    st.markdown("**Optional** (leave unticked if you don't know — the model fills in a typical value)")
    spec = antutu = refresh = ppi = np.nan
    chip = np.nan
    if "spec_score" in FEATS and st.checkbox("Spec score known", value=base is not None and pd.notna(val("spec_score", np.nan))):
        spec = st.slider("Spec score (91mobiles, 0–100)", 40, 100, int(val("spec_score", 80)))
    if "antutu" in FEATS and st.checkbox("AnTuTu score known", value=base is not None and pd.notna(val("antutu", np.nan))):
        antutu = st.number_input("AnTuTu score", 50000, 3500000, int(val("antutu", 600000)), step=10000)
    if "refresh_hz" in FEATS and st.checkbox("Refresh rate known", value=base is not None and pd.notna(val("refresh_hz", np.nan))):
        refresh = st.selectbox("Refresh rate (Hz)", [60, 90, 120, 144, 165],
                               index=[60, 90, 120, 144, 165].index(int(val("refresh_hz", 120)))
                               if int(val("refresh_hz", 120)) in [60, 90, 120, 144, 165] else 2)
    if "pixel_ppi" in FEATS and st.checkbox("Pixel density known", value=base is not None and pd.notna(val("pixel_ppi", np.nan))):
        ppi = st.slider("Pixel density (ppi)", 200, 600, int(val("pixel_ppi", 400)))
    if "chipset_vendor" in FEATS:
        vendors = ["Unknown"] + meta["chipset_vendors"]
        v0 = val("chipset_vendor", "Unknown")
        chip = st.selectbox("Chipset maker", vendors, index=vendors.index(v0) if v0 in vendors else 0)
        chip = np.nan if chip == "Unknown" else chip

row = pd.DataFrame([{"price_inr": float(price), "ram_gb": float(ram), "storage_gb": float(storage),
                     "battery_mah": float(battery), "charging_w": float(charging), "display_in": float(display),
                     "is_amoled": 1.0 if amoled else 0.0, "rear_mp": float(rear), "front_mp": float(front),
                     "spec_score": spec, "antutu": antutu, "release_year": float(year),
                     "brand": brand if brand in meta["brands"] else "unknown_brand",
                     "refresh_hz": refresh, "pixel_ppi": ppi, "chipset_vendor": chip}])
for c in FEATS:
    if c not in row:
        row[c] = np.nan

# ------------------------------------------------------------------------------------- prediction
with tab_pred:
    pred, base_value, contrib = predict_explain(model, meta, row)
    pred_c = float(np.clip(pred, 1, 5))
    new_brand = brand not in meta["brands"]
    if year >= 2025:
        typical, situation = mae_for("Future"), "a phone released after the training years (2025–26)"
    elif new_brand:
        typical, situation = mae_for("Leave"), "a brand the model has not seen"
    else:
        typical, situation = mae_for("Grouped"), "a new phone of a known brand from the training period"
    lv = level(pred_c, meta["ordinal_thresholds"])

    k1, k2, k3 = st.columns(3)
    k1.metric("Predicted rating", f"{pred_c:.2f} ★")
    k2.metric("Rating level", lv, help="Low < 4.0, Medium 4.0–4.4, High ≥ 4.5")
    k3.metric("Typical error", f"± {typical:.2f}", help=f"Average error measured for {situation}.")
    if base is not None and pd.notna(base.get("user_rating_5", np.nan)):
        st.info(f"Real rating of **{start}**: **{base['user_rating_5']:.2f} ★**"
                + (f" from {int(base['n_reviews']):,} reviews." if pd.notna(base.get("n_reviews")) else ".")
                + " (This phone was in the training data, so this is not a fair test.)")
    if year >= 2025 or new_brand:
        st.warning(f"Be careful: this is {situation}. Our tests show predictions are less reliable here.")

    left, right = st.columns([1.1, 1])
    with left:
        st.subheader("Why this rating?")
        st.caption(f"Each bar shows how much a feature moves the prediction away from the average rating "
                   f"({base_value:.2f}). Green raises the rating, red lowers it.")
        cdf = contrib.head(10).rename(index=FRIENDLY).reset_index()
        cdf.columns = ["Feature", "Effect"]
        cdf["Direction"] = np.where(cdf["Effect"] >= 0, "raises", "lowers")
        chart = alt.Chart(cdf).mark_bar().encode(
            x=alt.X("Effect:Q", title="Effect on rating (stars)"),
            y=alt.Y("Feature:N", sort=None, title=None),
            color=alt.Color("Direction:N", scale=alt.Scale(domain=["raises", "lowers"],
                                                           range=["#3c9a5f", "#c0392b"]), legend=None),
            tooltip=["Feature", alt.Tooltip("Effect:Q", format="+.3f")]).properties(height=320)
        st.altair_chart(chart, width="stretch")
    with right:
        st.subheader("Similar real phones")
        st.caption("Closest phones in the dataset by price, RAM, storage, battery, screen and year.")
        sim = similar_phones(ref, meta, row, k=6)
        if base is not None:
            sim = sim[sim["name"] != start].head(5)
        sim = sim.assign(price_usd=sim["price_inr"] / INR_PER_USD)
        show = sim[["name", "release_year", "price_usd", "price_inr", "user_rating_5", "n_reviews"]].rename(columns={
            "name": "Phone", "release_year": "Year", "price_usd": "Price (US$)", "price_inr": "Price (₹)",
            "user_rating_5": "Real rating", "n_reviews": "Reviews"})
        show["Year"] = show["Year"].astype(int)
        st.dataframe(show, hide_index=True, width="stretch",
                     column_config={"Price (US$)": st.column_config.NumberColumn(format="$%d"),
                                    "Price (₹)": st.column_config.NumberColumn(format="₹%d"),
                                    "Real rating": st.column_config.NumberColumn(format="%.2f ★"),
                                    "Reviews": st.column_config.NumberColumn(format="%d")})
        st.caption("Reviews = number of ratings on MySmartPrice (empty if not matched). "
                   "Ratings from fewer than 50 reviews are less trustworthy.")

# ------------------------------------------------------------------------------------- model page
with tab_model:
    st.subheader("Average error under four tests (lower is better)")
    st.caption("We test the model in harder, more realistic ways than a random split. "
               "MAE = average distance from the real rating, in stars.")
    order = ["Random 5-fold (Kaggle style)", "Grouped 5-fold", "Leave-one-brand-out", "Future-year (≤2024 → 2025–26)"]
    t = perf.pivot(index="Model", columns="Protocol", values="MAE")[order]
    t.columns = ["Random split (Kaggle style)", "New phone, known brand", "Unseen brand", "Future phones (2025–26)"]
    st.dataframe(t.style.format("{:.3f}").highlight_min(axis=0, color="#d8efdf"), width="stretch")
    t2 = perf.pivot(index="Model", columns="Protocol", values="R2")[order]
    t2.columns = t.columns
    st.subheader("R² (share of rating differences explained, higher is better)")
    st.dataframe(t2.style.format("{:.3f}"), width="stretch")
    st.markdown(
        "- A **random split** lets variants of the same phone appear in training and test, so it looks better "
        "than reality.\n"
        "- Our **grouped, unseen-brand and future** tests show how the model really behaves on new phones.\n"
        "- Ratings resting on **few reviews are noisier**, so we also tested review-volume weighting.")
    fig = RESULTS.parent / "figures" / "fig04_r2_by_protocol.png"
    if fig.exists():
        st.image(str(fig), caption="R² under four evaluation protocols")

with tab_about:
    st.subheader("Data sources")
    st.markdown(
        "1. **Mobile Phones Specs & Prices Dataset (2008–2026)** — Khadka, S. (2026), Kaggle, CC0. "
        "Scraped from 91mobiles. Main source of user ratings and specifications.\n"
        "2. **Mobile Price and Specs [2025]** — Vashisth, R. (2025), Kaggle, CC0. Scraped from MySmartPrice. "
        "Second rating and the number of reviews.\n"
        "3. **Global Smartphone Database 2025** — Dab, R. (2025), Kaggle, Apache 2.0. Extra specifications "
        "(refresh rate, pixel density, chipset).")
    st.subheader("Limitations")
    st.markdown(f"- Indian retail sites and prices (₹) only. US$ prices are approximate, converted at ₹{INR_PER_USD:.0f} "
                "per US$ (November 2026), and are not real US prices.\n"
                "- Specifications cannot capture software quality, after-sales service or brand image.\n"
                "- Predictions for new brands and 2025–26 phones are less accurate.")
