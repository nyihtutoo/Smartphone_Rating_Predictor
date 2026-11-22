"""Smartphone Rating Predictor - demo app for the Neat 2 final project.

Run from the Final Project folder:
    streamlit run app/app.py
"""
import random
import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.explain import FRIENDLY, RESULTS, level, load, predict_explain, similar_phones  # noqa: E402
from src.report_pdf import make_summary_pdf  # noqa: E402

st.set_page_config(page_title="Smartphone Rating Predictor", page_icon="📱", layout="wide")

INR_PER_USD = 94.0   # approximate rate, November 2026 (Wise mid-market rate). Used for display only.

# colours (same as .streamlit/config.toml)
INK, MUTED, AMBER, TEAL, CORAL, NAVY = "#1B2540", "#66708A", "#E8962A", "#16977F", "#DE5753", "#1E2747"

# ------------------------------------------------------------------------------------------ styling
st.markdown(f"""
<style>
  .block-container {{ padding-top: 2.2rem; max-width: 1280px; }}
  /* title */
  .app-title {{ font-size: 2.1rem; font-weight: 800; letter-spacing: -0.02em; color: {INK}; margin: 0; }}
  .app-sub {{ color: {MUTED}; font-size: 0.98rem; margin: 0.2rem 0 1.1rem; max-width: 75ch; }}
  /* tabs as pills */
  [role="tablist"] {{ gap: 0.45rem; border-bottom: none !important; box-shadow: none !important; }}
  [data-testid="stTab"] {{ background: #E8ECF3; border-radius: 999px; padding: 0.42rem 1.15rem; cursor: pointer; }}
  [data-testid="stTab"] p {{ font-weight: 600; font-size: 0.95rem; color: {MUTED} !important; }}
  [data-testid="stTab"]:hover {{ background: #DCE2EE; }}
  [data-testid="stTab"][aria-selected="true"] {{ background: {NAVY}; }}
  [data-testid="stTab"][aria-selected="true"] p {{ color: #fff !important; }}
  [data-testid="stTab"]:focus-visible {{ outline: 2px solid {AMBER}; outline-offset: 2px; }}
  [data-testid="stTab"] .react-aria-SelectionIndicator {{ display: none; }}
  /* buttons */
  [data-testid="stBaseButton-primary"] {{ background: linear-gradient(180deg, #F6B54A, {AMBER}); color: {NAVY};
      border: none; font-weight: 700; box-shadow: 0 2px 0 #B86F12; }}
  [data-testid="stBaseButton-primary"]:hover {{ background: #F6B54A; color: {NAVY}; }}
  [data-testid="stBaseButton-primary"]:active {{ transform: translateY(1px); box-shadow: 0 1px 0 #B86F12; }}
  [data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] {{ background: transparent; color: #E9EDF7;
      border: 1px solid #4A5788; font-weight: 600; }}
  [data-testid="stSidebar"] [data-testid="stBaseButton-secondary"]:hover {{ border-color: #F2A93B; color: #F2A93B; }}
  /* sidebar */
  [data-testid="stSidebar"] h2 {{ font-size: 1.25rem; font-weight: 800; }}
  .side-note {{ color: #AEB8D4; font-size: 0.85rem; margin: -0.3rem 0 0.6rem; }}
  [data-testid="stSidebar"] [data-testid="stExpander"] details {{ border-color: #3A4675; background: #232D52; }}
  /* result card */
  .result {{ background: #fff; border-radius: 1.1rem; padding: 1.4rem 1.6rem 1.1rem;
      box-shadow: 0 1px 2px rgba(27,37,64,.06), 0 8px 24px rgba(27,37,64,.07); margin-bottom: 1rem; }}
  .result-top {{ display: grid; grid-template-columns: auto 1fr 1fr; align-items: end; gap: 1.2rem 2.4rem; }}
  .stat {{ padding-bottom: 1.6rem; }}
  .big {{ font-size: 4.2rem; font-weight: 800; line-height: 0.95; letter-spacing: -0.03em; color: {INK};
      font-variant-numeric: tabular-nums; }}
  .big small {{ font-size: 1.3rem; font-weight: 600; color: {MUTED}; letter-spacing: 0; }}
  .stars {{ position: relative; display: inline-block; font-size: 1.55rem; letter-spacing: 0.12rem; line-height: 1; }}
  .stars .bg {{ color: #DDE2EC; }}
  .stars .fg {{ position: absolute; left: 0; top: 0; overflow: hidden; white-space: nowrap; color: #F2A93B; }}
  .cap {{ color: {MUTED}; font-size: 0.86rem; margin-top: 0.35rem; }}
  .stat .v {{ font-size: 1.5rem; font-weight: 700; color: {INK}; }}
  .pill {{ display: inline-block; padding: 0.15rem 0.8rem; border-radius: 999px; font-weight: 700; font-size: 1.15rem; }}
  .lv-Low {{ background: #FBE3E2; color: #A93431; }}
  .lv-Medium {{ background: #FCEFD8; color: #8A5608; }}
  .lv-High {{ background: #D9F1EB; color: #0E6B59; }}
  /* rating ruler */
  .ruler {{ position: relative; height: 46px; margin: 1.5rem 0 0.2rem; }}
  .track {{ position: absolute; left: 0; right: 0; top: 14px; height: 12px; border-radius: 999px; overflow: hidden;
      display: flex; }}
  .zone {{ height: 100%; }}
  .band {{ position: absolute; top: 10px; height: 20px; border-radius: 999px; background: rgba(30,39,71,.18);
      border: 2px solid rgba(30,39,71,.45); }}
  .dot {{ position: absolute; top: 6px; width: 28px; height: 28px; margin-left: -14px; border-radius: 50%;
      background: {NAVY}; border: 4px solid #fff; box-shadow: 0 2px 6px rgba(27,37,64,.35); }}
  .real {{ position: absolute; top: 9px; width: 18px; height: 18px; margin-left: -9px; transform: rotate(45deg);
      background: #F2A93B; border: 3px solid #fff; box-shadow: 0 1px 4px rgba(27,37,64,.35); }}
  .ticks {{ position: relative; height: 18px; font-size: 0.78rem; color: {MUTED}; }}
  .ticks span {{ position: absolute; transform: translateX(-50%); }}
  .legend {{ display: flex; flex-wrap: wrap; gap: 0.4rem 1.3rem; font-size: 0.82rem; color: {MUTED}; margin-top: 0.4rem; }}
  .legend i {{ display: inline-block; vertical-align: middle; margin-right: 0.35rem; }}
  /* section headings and notes */
  .sec {{ font-size: 1.15rem; font-weight: 700; color: {INK}; margin: 0.6rem 0 0.1rem; }}
  .note {{ border-radius: 0.8rem; padding: 0.65rem 1rem; margin: 0 0 0.6rem; font-size: 0.93rem; }}
  .note.info {{ background: #E5ECFB; color: #23407F; }}
  .note.warn {{ background: #FCEFD8; color: #7A4B06; }}
  .cap b {{ color: {INK}; }}
  .summary {{ font-size: 1.02rem; color: {INK}; margin: 0.9rem 0 0; padding-top: 0.9rem; border-top: 1px solid #E3E7EF; }}
  .summary b {{ font-weight: 800; }}
  @media (max-width: 760px) {{ .big {{ font-size: 3.2rem; }} .result-top {{ grid-template-columns: 1fr; }}
      .stat {{ padding-bottom: 0; }} }}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_model():
    return load()


@st.cache_data
def get_results():
    return pd.read_csv(RESULTS / "table_model_performance.csv")


model, meta, ref = get_model()
perf = get_results()
FEATS = meta["numeric"] + meta["categorical"]
NAMES = ["— enter specs by hand —"] + sorted(ref["name"].dropna().unique().tolist())


def mae_for(protocol_start):
    row = perf[perf["Protocol"].str.startswith(protocol_start) & (perf["Model"] == meta["model"])]
    return float(row["MAE"].iloc[0]) if len(row) else np.nan


def pick_random():
    st.session_state["start"] = random.choice(NAMES[1:])


def clear_phone():
    st.session_state["start"] = NAMES[0]


if "start" not in st.session_state:
    st.session_state["start"] = NAMES[0]

st.markdown('<div class="app-title">📱 Smartphone Rating Predictor</div>'
            '<div class="app-sub">Estimate the average user rating (out of 5) of a smartphone from its specifications '
            f'and price. {meta["model"]} model trained on {meta["trained_on_phones"]:,} rated phones released '
            '2022–2026. Neat 2 final project, AIT.</div>', unsafe_allow_html=True)

tab_pred, tab_model, tab_about = st.tabs(["Predict a rating", "How good is the model?", "About the data"])

# ------------------------------------------------------------------------------------------ inputs
with st.sidebar:
    st.header("Your phone")
    st.markdown('<p class="side-note">Pick a real phone to fill the form, or enter the specs yourself. '
                'The rating updates as you type.</p>', unsafe_allow_html=True)
    start = st.selectbox("Start from a real phone (optional)", NAMES, key="start",
                         help="Fills the form with a phone from the dataset. You can then change any value.")
    b1, b2 = st.columns(2)
    b1.button("🎲 Random", type="primary", on_click=pick_random, width="stretch", help="Load a random phone from the dataset")
    b2.button("Clear", on_click=clear_phone, width="stretch")
    base = ref[ref["name"] == start].iloc[0] if start != NAMES[0] else None

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
    RAMS, STORES = [1, 2, 3, 4, 6, 8, 12, 16, 18, 24], [16, 32, 64, 128, 256, 512, 1024, 2048]
    ram = c1.selectbox("RAM (GB)", RAMS, index=RAMS.index(int(val("ram_gb", 8))) if int(val("ram_gb", 8)) in RAMS else 5)
    storage = c2.selectbox("Storage (GB)", STORES, index=STORES.index(int(val("storage_gb", 128)))
                           if int(val("storage_gb", 128)) in STORES else 3)
    battery = st.slider("Battery (mAh)", 1000, 11000, int(val("battery_mah", 5000)), step=50)
    charging = st.slider("Charging power (W)", 5, 150, int(min(val("charging_w", 33), 150)))
    display = st.slider("Screen size (inch)", 4.0, 8.2, float(val("display_in", 6.6)), step=0.01)
    amoled = st.toggle("AMOLED / OLED screen", value=bool(val("is_amoled", 1.0) == 1))
    c3, c4 = st.columns(2)
    rear = c3.number_input("Rear camera (MP)", 0.0, 200.0, float(val("rear_mp", 50)), step=1.0, format="%.1f")
    front = c4.number_input("Front camera (MP)", 0.0, 60.0, float(val("front_mp", 16)), step=1.0, format="%.1f")
    YEARS = [2022, 2023, 2024, 2025, 2026]
    year = st.segmented_control("Release year", YEARS, default=int(val("release_year", 2024)))
    year = int(year) if year is not None else 2024

    spec = antutu = refresh = ppi = np.nan
    chip = np.nan
    with st.expander("More specs (optional)", expanded=base is not None):
        st.caption("Turn on only the values you know. The model fills in a typical value for the rest.")
        if "spec_score" in FEATS and st.toggle("Spec score known", value=base is not None and pd.notna(val("spec_score", np.nan))):
            spec = st.slider("Spec score (91mobiles, 0–100)", 40, 100, int(val("spec_score", 80)))
        if "antutu" in FEATS and st.toggle("AnTuTu score known", value=base is not None and pd.notna(val("antutu", np.nan))):
            antutu = st.number_input("AnTuTu score", 10000, 5000000, int(val("antutu", 600000)), step=10000)
        if "refresh_hz" in FEATS and st.toggle("Refresh rate known", value=base is not None and pd.notna(val("refresh_hz", np.nan))):
            HZ = [60, 90, 120, 144, 165]
            refresh = st.selectbox("Refresh rate (Hz)", HZ, index=HZ.index(int(val("refresh_hz", 120)))
                                   if int(val("refresh_hz", 120)) in HZ else 2)
        if "pixel_ppi" in FEATS and st.toggle("Pixel density known", value=base is not None and pd.notna(val("pixel_ppi", np.nan))):
            ppi = st.slider("Pixel density (ppi)", 100, 700, int(val("pixel_ppi", 400)))
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


def ruler_html(pred, typical, real=None, lo=3.0, hi=5.0):
    pos = lambda v: float(np.clip((v - lo) / (hi - lo) * 100, 0, 100))
    t1, t2 = meta["ordinal_thresholds"]
    zones = (f'<div class="zone" style="width:{pos(t1)}%;background:#F6CFCD"></div>'
             f'<div class="zone" style="width:{pos(t2) - pos(t1)}%;background:#F8E2B8"></div>'
             f'<div class="zone" style="width:{100 - pos(t2)}%;background:#BFE6DC"></div>')
    band_l, band_r = pos(pred - typical), pos(pred + typical)
    real_html = f'<div class="real" style="left:{pos(real)}%" title="Real rating {real:.2f}"></div>' if real is not None else ""
    ticks = "".join(f'<span style="left:{pos(v)}%">{v:.1f}</span>' for v in np.arange(lo, hi + 0.01, 0.5))
    legend = ('<div class="legend">'
              f'<span><i style="width:14px;height:14px;border-radius:50%;background:{NAVY}"></i>Prediction</span>'
              '<span><i style="width:22px;height:10px;border-radius:999px;background:rgba(30,39,71,.18);'
              'border:2px solid rgba(30,39,71,.45)"></i>Typical error range</span>'
              + ('<span><i style="width:11px;height:11px;transform:rotate(45deg);background:#F2A93B"></i>'
                 'Real rating</span>' if real is not None else "")
              + '<span><i style="width:22px;height:8px;border-radius:999px;background:linear-gradient(90deg,#F6CFCD,'
                '#F8E2B8,#BFE6DC)"></i>Low, Medium and High zones</span></div>')
    return (f'<div class="ruler"><div class="track">{zones}</div>'
            f'<div class="band" style="left:{band_l}%;width:{band_r - band_l}%"></div>{real_html}'
            f'<div class="dot" style="left:{pos(pred)}%"></div></div><div class="ticks">{ticks}</div>{legend}')


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
    real = float(base["user_rating_5"]) if base is not None and pd.notna(base.get("user_rating_5", np.nan)) else None
    n_rev = float(base["n_reviews"]) if base is not None and pd.notna(base.get("n_reviews", np.nan)) else None
    lo_r, hi_r = max(1.0, pred_c - typical), min(5.0, pred_c + typical)
    summary = (f"The model predicts <b>{pred_c:.2f} out of 5</b>, which is a <b>{lv}</b> rating. "
               f"The real rating is usually between <b>{lo_r:.2f} and {hi_r:.2f}</b>.")
    if real is not None:
        diff = real - pred_c
        summary += (f" The real rating is <b>{real:.2f}</b>, so the prediction is "
                    f"<b>{abs(diff):.2f} {'below' if diff > 0 else 'above'}</b> it.")

    st.markdown(f"""
    <div class="result">
      <div class="result-top">
        <div>
          <div class="big">{pred_c:.2f}<small> / 5</small></div>
          <div class="stars"><span class="bg">★★★★★</span><span class="fg" style="width:{pred_c / 5 * 100:.1f}%">★★★★★</span></div>
          <div class="cap">Predicted user rating</div>
        </div>
        <div class="stat"><span class="pill lv-{lv}">{lv}</span>
          <div class="cap">Rating level<br>Low below 4.0, Medium 4.0–4.4, High 4.5+</div></div>
        <div class="stat"><div class="v">± {typical:.2f} ★</div>
          <div class="cap">Typical error<br>for <b>{situation}</b></div></div>
      </div>
      {ruler_html(pred_c, typical, real)}
      <div class="summary">{summary}</div>
    </div>""", unsafe_allow_html=True)

    if real is not None:
        reviews = f" from {int(base['n_reviews']):,} reviews" if pd.notna(base.get("n_reviews")) else ""
        st.markdown(f'<div class="note info">Real rating of <b>{start}</b>: <b>{real:.2f} ★</b>{reviews}. '
                    'This phone was in the training data, so <b>this is not a fair test</b>.</div>', unsafe_allow_html=True)
    if year >= 2025 or new_brand:
        st.markdown(f'<div class="note warn"><b>Be careful:</b> this is <b>{situation}</b>. Our tests show '
                    'predictions are <b>less reliable</b> here, so read the result together with the typical '
                    'error.</div>', unsafe_allow_html=True)

    # ---- PDF export
    inputs = [("Phone", start if base is not None else "Custom phone"), ("Brand", brand),
              ("Price", f"${price / INR_PER_USD:,.0f}  (₹{price:,.0f})"), ("RAM / storage", f"{ram} GB / {storage} GB"),
              ("Battery", f"{battery:,} mAh"), ("Charging", f"{charging} W"), ("Screen", f"{display:.2f} inch"
              + (", AMOLED" if amoled else "")), ("Cameras", f"{rear:g} MP rear, {front:g} MP front"),
              ("Release year", str(year)),
              ("Spec score", "not given" if pd.isna(spec) else f"{spec:.0f}"),
              ("AnTuTu score", "not given" if pd.isna(antutu) else f"{antutu:,.0f}"),
              ("Refresh rate", "not given" if pd.isna(refresh) else f"{refresh:.0f} Hz"),
              ("Pixel density", "not given" if pd.isna(ppi) else f"{ppi:.0f} ppi"),
              ("Chipset maker", "not given" if pd.isna(chip) else str(chip))]
    pdf_contrib = contrib.head(8).rename(index=FRIENDLY).reset_index()
    pdf_contrib.columns = ["Feature", "Effect"]
    sim_pdf = similar_phones(ref, meta, row, k=6)
    if base is not None:
        sim_pdf = sim_pdf[sim_pdf["name"] != start]
    sim_pdf = sim_pdf.head(5).assign(price_usd=lambda d: d["price_inr"] / INR_PER_USD).rename(columns={
        "name": "Phone", "release_year": "Year", "price_usd": "Price (US$)", "user_rating_5": "Real rating",
        "n_reviews": "Reviews"})
    pdf_args = dict(phone_name=start if base is not None else f"Custom {brand} phone", pred=pred_c, lv=lv,
                    typical=typical, situation=situation, real=real, n_reviews=n_rev, inputs=inputs,
                    contrib=pdf_contrib, similar=sim_pdf, thresholds=meta["ordinal_thresholds"],
                    model_name=meta["model"], base_value=base_value)
    fname = ("rating_summary_" + (start if base is not None else "custom_phone")).replace(" ", "_")[:60] + ".pdf"
    st.download_button("📄 Download summary (PDF)", data=lambda: make_summary_pdf(**pdf_args), file_name=fname,
                       mime="application/pdf", type="primary", on_click="ignore",
                       help="Saves a one-page PDF with the prediction, the typical error, the reasons and similar phones.")

    left, right = st.columns([1.1, 1], gap="large")
    with left:
        st.markdown('<p class="sec">Why this rating?</p>', unsafe_allow_html=True)
        st.caption(f"Each bar shows how much a feature moves the prediction away from the **average rating "
                   f"({base_value:.2f})**. **Green raises** the rating, **red lowers** it.")
        cdf = contrib.head(10).rename(index=FRIENDLY).reset_index()
        cdf.columns = ["Feature", "Effect"]
        cdf["Direction"] = np.where(cdf["Effect"] >= 0, "raises", "lowers")
        chart = alt.Chart(cdf).mark_bar(cornerRadius=4, height=18).encode(
            x=alt.X("Effect:Q", title="Effect on rating (stars)", axis=alt.Axis(grid=True, gridColor="#E3E7EF")),
            y=alt.Y("Feature:N", sort=None, title=None, axis=alt.Axis(labelFontSize=12, labelColor=INK, labelLimit=220)),
            color=alt.Color("Direction:N", scale=alt.Scale(domain=["raises", "lowers"], range=[TEAL, CORAL]),
                            legend=None),
            tooltip=["Feature", alt.Tooltip("Effect:Q", format="+.3f")]
        ).properties(height=330).configure_view(stroke=None)
        st.altair_chart(chart, width="stretch")
    with right:
        st.markdown('<p class="sec">Similar real phones</p>', unsafe_allow_html=True)
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
                   "Ratings from **fewer than 50 reviews are less trustworthy**.")

# ------------------------------------------------------------------------------------- model page
with tab_model:
    st.markdown('<p class="sec">Average error under four tests (lower is better)</p>', unsafe_allow_html=True)
    st.caption("We test the model in harder, more realistic ways than a random split. "
               "MAE = average distance from the real rating, in stars.")
    order = ["Random 5-fold (Kaggle style)", "Grouped 5-fold", "Leave-one-brand-out", "Future-year (≤2024 → 2025–26)"]
    t = perf.pivot(index="Model", columns="Protocol", values="MAE")[order]
    t.columns = ["Random split (Kaggle style)", "New phone, known brand", "Unseen brand", "Future phones (2025–26)"]
    st.dataframe(t.style.format("{:.3f}").highlight_min(axis=0, color="#D9F1EB"), width="stretch")
    t2 = perf.pivot(index="Model", columns="Protocol", values="R2")[order]
    t2.columns = t.columns
    st.markdown('<p class="sec">R² (share of rating differences explained, higher is better)</p>',
                unsafe_allow_html=True)
    st.dataframe(t2.style.format("{:.3f}").highlight_max(axis=0, color="#D9F1EB"), width="stretch")
    st.markdown(
        "- A **random split** lets variants of the same phone appear in training and test, so it looks better "
        "than reality.\n"
        "- Our **grouped, unseen-brand and future** tests show how the model really behaves on new phones.\n"
        "- Ratings resting on **few reviews are noisier**, so we also tested review-volume weighting.")
    fig = RESULTS.parent / "figures" / "fig04_r2_by_protocol.png"
    if fig.exists():
        st.image(str(fig), caption="R² under four evaluation protocols")

with tab_about:
    st.markdown('<p class="sec">Data sources</p>', unsafe_allow_html=True)
    st.markdown(
        "1. **Mobile Phones Specs & Prices Dataset (2008–2026)** — Khadka, S. (2026), Kaggle, CC0. "
        "Scraped from 91mobiles. Main source of user ratings and specifications.\n"
        "2. **Mobile Price and Specs [2025]** — Vashisth, R. (2025), Kaggle, CC0. Scraped from MySmartPrice. "
        "Second rating and the number of reviews.\n"
        "3. **Global Smartphone Database 2025** — Dab, R. (2025), Kaggle, Apache 2.0. Extra specifications "
        "(refresh rate, pixel density, chipset).")
    st.markdown('<p class="sec">Limitations</p>', unsafe_allow_html=True)
    st.markdown(f"- Indian retail sites and prices (₹) only. US$ prices are approximate, converted at ₹{INR_PER_USD:.0f} "
                "per US$ (November 2026), and are not real US prices.\n"
                "- Specifications cannot capture software quality, after-sales service or brand image.\n"
                "- Predictions for new brands and 2025–26 phones are less accurate.")
