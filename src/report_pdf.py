"""One-page PDF summary of a prediction, made with matplotlib only (no extra library needed)."""
import io
from datetime import datetime

import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch, Rectangle

INK, MUTED, AMBER, TEAL, CORAL, NAVY = "#1B2540", "#66708A", "#E8962A", "#16977F", "#DE5753", "#1E2747"
ZONE = {"Low": "#F6CFCD", "Medium": "#F8E2B8", "High": "#BFE6DC"}
LEVEL_TEXT = {"Low": "#A93431", "Medium": "#8A5608", "High": "#0E6B59"}


def make_summary_pdf(phone_name, pred, lv, typical, situation, real, n_reviews, inputs, contrib, similar,
                     thresholds, model_name, base_value):
    """Return the PDF as bytes.

    inputs: list of (label, value text); contrib: DataFrame with columns Feature, Effect (top features);
    similar: DataFrame with Phone, Year, Price (US$), Real rating, Reviews.
    """
    fig = Figure(figsize=(8.27, 11.69))          # A4 portrait
    W, H = 8.27, 11.69

    def text(x, y, s, **kw):                     # x, y in inches from the top-left corner
        kw.setdefault("color", INK)
        kw.setdefault("va", "top")
        kw.setdefault("parse_math", False)
        fig.text(x / W, 1 - y / H, s, **kw)

    # ---- header
    fig.patches.append(Rectangle((0, 1 - 1.05 / H), 1, 1.05 / H, transform=fig.transFigure, color=NAVY, zorder=0))
    text(0.6, 0.30, "Smartphone Rating Predictor", fontsize=19, fontweight="bold", color="white")
    text(0.6, 0.72, f"Prediction summary  |  {datetime.now():%d %B %Y, %H:%M}", fontsize=9.5, color="#C9D1E6")

    # ---- phone and main result
    text(0.6, 1.35, phone_name, fontsize=14, fontweight="bold")
    text(0.6, 1.80, f"{pred:.2f}", fontsize=46, fontweight="bold")
    text(2.30, 2.12, "/ 5", fontsize=14, color=MUTED)
    n_full = int(round(pred))
    text(0.62, 2.62, "★" * n_full + "☆" * (5 - n_full), fontsize=17, color="#F2A93B")
    text(0.6, 3.0, "Predicted user rating", fontsize=9, color=MUTED)

    # level badge
    fig.patches.append(FancyBboxPatch((3.25 / W, 1 - 2.22 / H), 1.25 / W, 0.42 / H, transform=fig.transFigure,
                                      boxstyle="round,pad=0.002,rounding_size=0.012", color=ZONE[lv]))
    text(3.875, 1.86, lv, fontsize=14, fontweight="bold", color=LEVEL_TEXT[lv], ha="center")
    text(3.25, 2.40, "Rating level", fontsize=9, color=MUTED)
    text(3.25, 2.60, "Low below 4.0, Medium 4.0–4.4, High 4.5+", fontsize=8, color=MUTED)

    text(5.75, 1.86, f"± {typical:.2f}", fontsize=17, fontweight="bold")
    text(5.75, 2.40, "Typical error", fontsize=9, color=MUTED)
    text(5.75, 2.60, _wrap(f"for {situation}", 38), fontsize=8, color=MUTED, linespacing=1.4)

    # ---- rating ruler (3.0 to 5.0)
    ax = fig.add_axes([0.6 / W, 1 - 3.95 / H, (W - 1.2) / W, 0.55 / H])
    lo, hi = 3.0, 5.0
    t1, t2 = thresholds
    for a, b, c in [(lo, t1, ZONE["Low"]), (t1, t2, ZONE["Medium"]), (t2, hi, ZONE["High"])]:
        ax.add_patch(Rectangle((a, 0.4), b - a, 0.2, color=c, lw=0))
    p = float(np.clip(pred, lo, hi))
    ax.add_patch(FancyBboxPatch((max(lo, p - typical), 0.3), min(hi, p + typical) - max(lo, p - typical), 0.4,
                                boxstyle="round,pad=0,rounding_size=0.08", fc=(0.12, 0.15, 0.28, 0.18),
                                ec=(0.12, 0.15, 0.28, 0.5), lw=1.4))
    ax.plot([p], [0.5], "o", ms=13, color=NAVY, mec="white", mew=2.5, zorder=5)
    if real is not None:
        ax.plot([float(np.clip(real, lo, hi))], [0.5], "D", ms=9, color="#F2A93B", mec="white", mew=1.8, zorder=6)
    ax.set_xlim(lo - 0.02, hi + 0.02); ax.set_ylim(0, 1)
    ax.set_yticks([]); ax.set_xticks(np.arange(lo, hi + 0.01, 0.5))
    ax.tick_params(axis="x", length=0, labelsize=8, labelcolor=MUTED)
    for s in ax.spines.values():
        s.set_visible(False)
    legend = "●  Prediction     ▭  Typical error range" + ("     ◆  Real rating" if real is not None else "")
    text(0.6, 4.12, legend, fontsize=8, color=MUTED)

    # ---- key message
    msg = (f"The model predicts {pred:.2f} out of 5 ({lv} level). For {situation}, the real rating is usually "
           f"within ± {typical:.2f} of the prediction.")
    if real is not None:
        rv = f" from {int(n_reviews):,} reviews" if n_reviews is not None and not np.isnan(n_reviews) else ""
        msg += f" The real rating of this phone is {real:.2f}{rv} (this phone was in the training data)."
    fig.patches.append(FancyBboxPatch((0.6 / W, 1 - 5.12 / H), (W - 1.2) / W, 0.72 / H, transform=fig.transFigure,
                                      boxstyle="round,pad=0.002,rounding_size=0.01", color="#E5ECFB"))
    text(0.75, 4.52, _wrap(msg, 96), fontsize=9.2, color="#23407F", fontweight="bold", linespacing=1.5)

    # ---- inputs (left) and reasons (right)
    text(0.6, 5.40, "Phone specifications used", fontsize=11.5, fontweight="bold")
    y = 5.75
    for lab, v in inputs:
        text(0.6, y, lab, fontsize=8.6, color=MUTED)
        text(2.35, y, v, fontsize=8.6, fontweight="bold")
        y += 0.22

    text(4.35, 5.40, "Why this rating? (top factors)", fontsize=11.5, fontweight="bold")
    text(4.35, 5.68, f"Effect in stars, from the average rating {base_value:.2f}", fontsize=8, color=MUTED)
    c = contrib.head(8).iloc[::-1]
    ax2 = fig.add_axes([5.75 / W, 1 - 8.85 / H, 1.95 / W, 2.95 / H])
    ax2.barh(range(len(c)), c["Effect"], color=[TEAL if e >= 0 else CORAL for e in c["Effect"]], height=0.62)
    ax2.set_yticks(range(len(c)))
    ax2.set_yticklabels([s[:24] for s in c["Feature"]], fontsize=8, color=INK)
    ax2.axvline(0, color="#B8C0D0", lw=0.8)
    ax2.tick_params(axis="x", labelsize=7, labelcolor=MUTED, length=0)
    ax2.tick_params(axis="y", length=0)
    for s in ax2.spines.values():
        s.set_visible(False)
    for i, e in enumerate(c["Effect"]):
        ax2.text(e, i, f" {e:+.3f} " if e >= 0 else f" {e:+.3f} ", va="center", fontsize=7,
                 ha="left" if e >= 0 else "right", color=INK)
    lim = max(abs(c["Effect"]).max() * 1.6, 0.02)
    ax2.set_xlim(-lim, lim)
    text(4.35, 9.0, "Green = raises the rating, red = lowers it.", fontsize=8, color=MUTED)

    # ---- similar phones
    text(0.6, 9.35, "Similar real phones", fontsize=11.5, fontweight="bold")
    cols = [("Phone", 0.6), ("Year", 3.75), ("Price (US$)", 4.45), ("Real rating", 5.55), ("Reviews", 6.65)]
    for name, x in cols:
        text(x, 9.66, name, fontsize=8.4, color=MUTED, fontweight="bold")
    y = 9.90
    for _, r in similar.head(5).iterrows():
        rev = r.get("Reviews")
        vals = [str(r["Phone"])[:34], f"{int(r['Year'])}", f"${r['Price (US$)']:,.0f}",
                f"{r['Real rating']:.2f}", "–" if rev is None or (isinstance(rev, float) and np.isnan(rev)) else f"{int(rev):,}"]
        for (name, x), v in zip(cols, vals):
            text(x, y, v, fontsize=8.6, fontweight="bold" if name == "Real rating" else "normal")
        y += 0.22

    # ---- footer
    text(0.6, 11.05, _wrap(f"Model: {model_name}, trained on 1,597 rated phones (2022–2026) from 91mobiles, MySmartPrice "
                           "and the Global Smartphone Database. Prices are Indian prices; US$ values are approximate "
                           "(₹94 per US$). Specifications cannot capture software, service or brand image. "
                           "Neat 2 final project, AIT.", 140), fontsize=7, color=MUTED, linespacing=1.4)

    buf = io.BytesIO()
    fig.savefig(buf, format="pdf")
    return buf.getvalue()


def _wrap(s, width):
    words, lines, cur = s.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    lines.append(cur)
    return "\n".join(lines)
