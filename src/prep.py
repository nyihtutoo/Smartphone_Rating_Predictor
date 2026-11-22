"""Data loading, cleaning, phone grouping and dataset matching.

Final-stage version of the progress notebook (Neat2_Reproducible.ipynb).
The parsing rules are the same as at the progress stage; the matching adds an
exact-name step before the variant-key rule (progress-stage weakness: 4G/5G
labels were dropped from the key, so e.g. "Galaxy A14" and "Galaxy A14 5G"
became ambiguous and were left unmatched).
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"

D1_FILE = "mobiles_data.csv"                       # Dataset 1 - 91mobiles (Khadka, 2026)
D2_FILE = "mysmartprice_mobile_dataset.csv"        # Dataset 2 - MySmartPrice (Vashisth, 2025)
D3_FILE = "phones_data_20250729_181901.csv"        # Dataset 3 - Global Smartphone Database 2025 (Dab, 2025)


def load_raw(data_dir=DATA):
    d1 = pd.read_csv(Path(data_dir) / D1_FILE)
    d2 = pd.read_csv(Path(data_dir) / D2_FILE)
    d3 = pd.read_csv(Path(data_dir) / D3_FILE, low_memory=False)
    return d1, d2, d3


# --------------------------------------------------------------------------- parsing
def first_float(text):
    if pd.isna(text):
        return np.nan
    m = re.search(r"\d+(?:,\d{3})*(?:\.\d+)?", str(text))
    return float(m.group().replace(",", "")) if m else np.nan


def parse_price_inr(text):
    if pd.isna(text):
        return np.nan
    digits = re.sub(r"[^\d]", "", str(text).split(".")[0])
    return float(digits) if digits else np.nan


def rating_scale(text):
    m = re.search(r"/\s*(\d+)\s*$", str(text))
    return int(m.group(1)) if m else np.nan


def parse_rating_to_5(text):
    """'4.6/5' -> 4.6 ; '8.4/10' -> 4.2"""
    if pd.isna(text):
        return np.nan
    m = re.match(r"\s*([\d.]+)\s*/\s*(\d+)", str(text))
    if not m:
        return np.nan
    val, scale = float(m.group(1)), float(m.group(2))
    return round(val / scale * 5, 3) if scale else np.nan


COUNT_UNITS = {"": 1, "K": 1_000, "L": 100_000, "M": 1_000_000, "CR": 10_000_000}


def parse_count(text):
    """'16.9K Ratings' -> 16900 ; '8,220 Ratings' -> 8220 ; '1.5L Ratings' -> 150000

    FIX (final stage): MySmartPrice uses the Indian unit 'L' (lakh = 100,000) for its most-reviewed phones.
    The progress-stage parser ignored 'L', so '1.5L' was read as 1.5 reviews instead of 150,000.
    """
    if pd.isna(text):
        return np.nan
    m = re.search(r"([\d.]+)\s*(CR|[KLM])?\b", str(text).replace(",", "").upper())
    if not m:
        return np.nan
    return float(m.group(1)) * COUNT_UNITS[m.group(2) or ""]


def parse_count_progress(text):
    """The progress-stage parser, kept only to show the effect of the fix."""
    if pd.isna(text):
        return np.nan
    m = re.search(r"([\d.]+)\s*([KkMm]?)", str(text).replace(",", ""))
    if not m:
        return np.nan
    v, suf = float(m.group(1)), m.group(2).upper()
    return v * 1_000 if suf == "K" else v * 1_000_000 if suf == "M" else v


def parse_ram_storage(text):
    if pd.isna(text):
        return (np.nan, np.nan)
    s = str(text)
    ram = re.search(r"([\d.]+)\s*GB\s*RAM", s, re.I)
    sto = re.search(r"([\d.]+)\s*(GB|TB)\s*Storage", s, re.I)
    ram_v = float(ram.group(1)) if ram else np.nan
    sto_v = float(sto.group(1)) * (1024 if sto.group(2).upper() == "TB" else 1) if sto else np.nan
    return (ram_v, sto_v)


def parse_gb(text):
    """'12GB' -> 12 ; '1TB' -> 1024 (Dataset 3 style)"""
    if pd.isna(text):
        return np.nan
    m = re.search(r"([\d.]+)\s*(GB|TB)", str(text), re.I)
    if not m:
        return np.nan
    return float(m.group(1)) * (1024 if m.group(2).upper() == "TB" else 1)


def parse_battery(text):
    if pd.isna(text):
        return (np.nan, np.nan)
    s = str(text)
    mah = re.search(r"([\d,]+)\s*mAh", s, re.I)
    w = re.search(r"([\d.]+)\s*W\b", s)
    return (float(mah.group(1).replace(",", "")) if mah else np.nan,
            float(w.group(1)) if w else np.nan)


def parse_display(text):
    if pd.isna(text):
        return (np.nan, np.nan)
    s = str(text)
    inch = re.search(r"([\d.]+)\s*(?:inches|inch|″|'')", s, re.I)
    panel = s.split("|")[-1].strip() if "|" in s else np.nan
    return (float(inch.group(1)) if inch else np.nan, panel)


def parse_main_mp(text):
    if pd.isna(text):
        return np.nan
    mps = re.findall(r"([\d.]+)\s*MP", str(text), re.I)
    return float(mps[0]) if mps else np.nan


def parse_year(text):
    if pd.isna(text):
        return np.nan
    m = re.search(r"(?:19|20)\d{2}", str(text))
    return int(m.group()) if m else np.nan


def parse_hz(text):
    """'120 Hz' -> 120 (Dataset 3 refresh rate)"""
    if pd.isna(text):
        return np.nan
    m = re.search(r"(\d+)\s*Hz", str(text), re.I)
    return float(m.group(1)) if m else np.nan


def parse_ppi(text):
    """'453 ppi' -> 453 (Dataset 3 pixel density)"""
    if pd.isna(text):
        return np.nan
    m = re.search(r"(\d+)\s*ppi", str(text), re.I)
    return float(m.group(1)) if m else np.nan


def chipset_vendor(text):
    """Chipset maker from the Dataset 3 chipset text (Qualcomm, MediaTek, Apple, Samsung, Unisoc, Google)."""
    if pd.isna(text):
        return np.nan
    s = str(text).lower()
    for key, label in [("qualcomm", "Qualcomm"), ("snapdragon", "Qualcomm"), ("mediatek", "MediaTek"),
                       ("dimensity", "MediaTek"), ("helio", "MediaTek"), ("apple", "Apple"),
                       ("exynos", "Samsung"), ("unisoc", "Unisoc"), ("tensor", "Google"), ("kirin", "Huawei")]:
        if key in s:
            return label
    return "Other"


# ------------------------------------------------------------------ brands and keys
KNOWN_BRANDS = ["Samsung", "Xiaomi", "Redmi", "Poco", "Apple", "OnePlus", "Oppo", "Vivo", "iQOO", "Realme",
                "Motorola", "Moto", "Nokia", "Google", "Honor", "Huawei", "Asus", "Lenovo", "Nothing", "Infinix",
                "Tecno", "Lava", "Micromax", "Sony", "LG", "HTC", "ZTE", "Nubia", "Meizu", "Gionee", "Coolpad",
                "BlackBerry", "Panasonic", "TCL", "Sharp", "Fairphone", "CAT", "Doogee", "Ulefone", "Blackview",
                "itel", "Jio", "Karbonn", "Sansui", "Alcatel", "Cmf"]
BRAND_ALIASES = {"redmi": "Xiaomi", "poco": "Xiaomi", "moto": "Motorola"}
EDITION_WORDS = {"pro", "plus", "ultra", "lite", "max", "mini", "fe", "neo", "prime", "turbo"}
NETWORK_WORDS = {"5g", "4g", "lte", "nfc"}


def standardize_brand(name):
    if pd.isna(name):
        return np.nan
    low = str(name).strip().lower()
    for b in sorted(KNOWN_BRANDS, key=len, reverse=True):
        if low.startswith(b.lower()):
            return BRAND_ALIASES.get(b.lower(), b)
    return str(name).split()[0].title()


def tokens(name):
    s = str(name).lower()
    s = re.sub(r"\(.*?\)", " ", s)                 # bracketed colour / storage / market text
    s = s.replace("+", " plus ")
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\b\d+\s*(gb|tb|mb)\b", " ", s)    # 8gb, 256 gb, 1tb
    return [t for t in s.split() if t not in NETWORK_WORDS]


def model_key(name):
    """Variant key used at the progress stage: keeps the edition, drops RAM/storage and network words."""
    return "_".join(tokens(name)) if not pd.isna(name) else np.nan


def family_key(name):
    """Phone family = brand + product line + generation (all editions and variants share one family)."""
    return "_".join(t for t in tokens(name) if t not in EDITION_WORDS) if not pd.isna(name) else np.nan


def exact_key(name):
    """NEW (final stage): the full name, only normalised.

    Lower case, '+' -> 'plus', brackets and punctuation removed, spaces collapsed. Unlike model_key it KEEPS
    network words (4G/5G) and RAM/storage words, so 'Galaxy A14' and 'Galaxy A14 5G' stay different phones.
    """
    if pd.isna(name):
        return np.nan
    s = str(name).lower().replace("+", " plus ")
    s = re.sub(r"[()\[\]]", " ", s)
    s = re.sub(r"(\d+)\s*(gb|tb)\b", r"\1\2", s)    # '8 GB' -> '8gb'
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return "_".join(s.split())


# --------------------------------------------------------------------- Dataset 1
def build_d1(d1_raw):
    d1 = d1_raw.copy()
    d1["brand"] = d1["name"].apply(standardize_brand)
    d1["family"] = d1["name"].apply(family_key)
    d1["model_key"] = d1["name"].apply(model_key)
    d1["exact_key"] = d1["name"].apply(exact_key)
    d1["release_year"] = d1["release_date"].apply(parse_year)
    d1["rating_scale"] = d1["user_rating"].apply(rating_scale)
    d1["user_rating_5"] = d1["user_rating"].apply(parse_rating_to_5)          # TARGET (0-5)
    d1["price_inr"] = d1["price"].apply(parse_price_inr)
    d1["spec_score"] = d1["spec_score"].apply(first_float)
    d1["antutu"] = d1["antutu_score"].apply(first_float).replace(0, np.nan)   # 0 is a placeholder
    d1[["ram_gb", "storage_gb"]] = d1["ram_storage"].apply(lambda t: pd.Series(parse_ram_storage(t)))
    d1[["battery_mah", "charging_w"]] = d1["battery"].apply(lambda t: pd.Series(parse_battery(t)))
    d1[["display_in", "panel"]] = d1["display"].apply(lambda t: pd.Series(parse_display(t)))
    d1["is_amoled"] = d1["panel"].astype(str).str.contains("OLED", case=False).astype(float)
    d1.loc[d1["panel"].isna(), "is_amoled"] = np.nan
    d1["rear_mp"] = d1["rear_camera"].apply(parse_main_mp)
    d1["front_mp"] = d1["front_camera"].apply(parse_main_mp)
    return d1


def build_d2(d2_raw):
    d2 = d2_raw.copy()
    d2["model_key"] = d2["mobile_name"].apply(model_key)
    d2["exact_key"] = d2["mobile_name"].apply(exact_key)
    d2["d2_rating"] = pd.to_numeric(d2["avg_rating"], errors="coerce")       # already out of 5
    d2["d2_n_ratings"] = d2["total_ratings"].apply(parse_count)
    d2[["ram_gb", "storage_gb"]] = d2["ram_and_storage"].apply(lambda t: pd.Series(parse_ram_storage(t)))
    return d2


def build_d3(d3_raw):
    d3 = d3_raw.copy()
    status = d3["status"].astype(str).str.lower() if "status" in d3 else pd.Series("", index=d3.index)
    d3 = d3[~status.str.contains("rumor|rumour|not announced|coming soon", regex=True)].copy()
    full = d3["brand"].astype(str) + " " + d3["model"].astype(str)
    d3["model_key"] = full.apply(model_key)
    d3["exact_key"] = full.apply(exact_key)
    d3["storage_gb"] = d3["internal_storage"].apply(parse_gb) if "internal_storage" in d3 else np.nan
    d3["ram_gb"] = d3["ram"].apply(parse_gb) if "ram" in d3 else np.nan
    d3["storage_gb"] = d3["storage_gb"].fillna(
        d3["model"].astype(str).str.extract(r"\((\d+\s*(?:GB|TB))\)", flags=re.I)[0].apply(parse_gb))
    d3["refresh_hz"] = d3["refresh_rate"].apply(parse_hz)
    d3["pixel_ppi"] = d3["pixel_density"].apply(parse_ppi)
    d3["chipset_vendor"] = d3["chipset"].apply(chipset_vendor)
    d3["d3_name"] = full
    return d3


def working_sample(d1):
    """Rated smartphones (screen >= 4 in) released 2022-2026."""
    in_years = d1["release_year"].between(2022, 2026)
    ok = in_years & (d1["display_in"] >= 4.0) & d1["user_rating_5"].notna()
    return d1[ok].copy().reset_index(drop=True)


# ------------------------------------------------------------------------ matching
def _agree(a, b):
    """1 if both known and equal, NaN if either unknown, 0 if different."""
    return pd.Series(np.where(a.notna() & b.notna(), np.isclose(a, b), np.nan), index=a.index)


def _match_on(left, right, key, right_cols):
    """One left row -> at most one right row on `key`, using RAM/storage to break ties (progress-stage rule)."""
    L = left[[key, "ram_gb", "storage_gb"]].reset_index().rename(columns={"index": "lid"})
    R = right[[key, "ram_gb", "storage_gb"] + right_cols].reset_index().rename(columns={"index": "rid"})
    L, R = L[L[key].notna()], R[R[key].notna()]
    cand = L.merge(R, on=key, suffixes=("", "_r"))
    cand["ram_ok"] = _agree(cand["ram_gb"], cand["ram_gb_r"])
    cand["sto_ok"] = _agree(cand["storage_gb"], cand["storage_gb_r"])
    cand["n_cand"] = cand.groupby("lid")["rid"].transform("size")
    single = cand[cand["n_cand"] == 1]
    multi = cand[(cand["n_cand"] > 1) & (cand["ram_ok"] != 0) & (cand["sto_ok"] == 1)]
    multi = multi[multi.groupby("lid")["rid"].transform("size") == 1]           # still ambiguous -> drop
    chosen = pd.concat([single, multi]).set_index("lid")
    chosen["spec_agree"] = (chosen["ram_ok"] != 0) & (chosen["sto_ok"] != 0)    # no known disagreement
    n_amb = cand.loc[cand["n_cand"] > 1, "lid"].nunique() - multi["lid"].nunique() if len(multi) else \
        cand.loc[cand["n_cand"] > 1, "lid"].nunique()
    return chosen, n_amb


def match_left(left, right, right_cols, tag, method="exact_then_key"):
    """Match every Dataset 1 row (left) to the right-hand dataset. Dataset 1 is never changed.

    method = "key"            : progress-stage rule (variant key only)
             "exact_then_key" : NEW - exact normalised name first; phones still unmatched use the variant key
    """
    res = pd.DataFrame(index=left.index)
    res[f"{tag}_matched"] = False
    res[f"{tag}_match_step"] = pd.Series(pd.NA, index=left.index, dtype="object")
    res[f"{tag}_spec_agree"] = np.nan
    for c in right_cols:
        res[c] = np.nan if c not in ("mobile_name", "d3_name", "chipset_vendor") else pd.Series(
            pd.NA, index=left.index, dtype="object")
    steps = [("exact", "exact_key"), ("key", "model_key")] if method == "exact_then_key" else [("key", "model_key")]
    info = {}
    todo = left.index
    for step, key in steps:
        chosen, n_amb = _match_on(left.loc[todo], right, key, right_cols)
        idx = chosen.index
        res.loc[idx, f"{tag}_matched"] = True
        res.loc[idx, f"{tag}_match_step"] = step
        res.loc[idx, f"{tag}_spec_agree"] = chosen["spec_agree"].astype(float)
        for c in right_cols:
            res.loc[idx, c] = chosen[c].values
        info[step] = {"matched": len(idx), "ambiguous": int(n_amb)}
        todo = todo.difference(idx)
    res[f"{tag}_spec_agree"] = res[f"{tag}_spec_agree"].map({1.0: True, 0.0: False})
    return res, info


D2_COLS = ["d2_rating", "d2_n_ratings", "mobile_name"]
D3_COLS = ["refresh_hz", "pixel_ppi", "chipset_vendor", "d3_name"]


def build_all(method="exact_then_key", data_dir=DATA, verbose=True):
    """Returns (work, d1_22_26, info): the working sample with Dataset 2 / Dataset 3 columns attached."""
    d1_raw, d2_raw, d3_raw = load_raw(data_dir)
    d1, d2, d3 = build_d1(d1_raw), build_d2(d2_raw), build_d3(d3_raw)
    in_years = d1["release_year"].between(2022, 2026)
    d1_22_26 = d1[in_years].copy()                       # all 2022-2026 rows, rated or not (match-rate denominator)
    m2, info2 = match_left(d1_22_26, d2, D2_COLS, "d2", method)
    m3, info3 = match_left(d1_22_26, d3, D3_COLS, "d3", method)
    d1_22_26 = d1_22_26.join(m2).join(m3)
    if verbose:
        print(f"[{method}] Dataset 2 matches by step: {info2} | Dataset 3: {info3}")
    work = working_sample(d1)
    extra = d1_22_26.drop_duplicates("name").set_index("name")
    keep = ["d2_matched", "d2_match_step", "d2_spec_agree"] + D2_COLS + ["d3_matched", "d3_match_step"] + D3_COLS
    work = work.join(extra[keep], on="name")
    for c in ["d2_rating", "d2_n_ratings", "refresh_hz", "pixel_ppi"]:
        work[c] = pd.to_numeric(work[c], errors="coerce").astype(float)
    for c in ["mobile_name", "d3_name", "chipset_vendor", "d2_match_step", "d3_match_step"]:
        work[c] = work[c].astype(object).where(work[c].notna(), np.nan)
    # review count only for phones matched to Dataset 2
    work["n_reviews"] = work["d2_n_ratings"].where(work["d2_matched"] == True)
    return work, d1_22_26, {"d2": info2, "d3": info3, "raw_sizes": (d1_raw.shape, d2_raw.shape, d3_raw.shape)}
