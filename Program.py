import os
import re
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import yfinance as yf

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, accuracy_score, roc_curve

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# -----------------------
# Config
# -----------------------
EXCEL_PATH = "NYT Article Data.xlsx"

SPY_START = "2016-01-01"
SPY_END_EXCLUSIVE = "2026-01-01"   # includes through 2025-12-31

TRAIN_END_DATE = "2024-12-31"
TEST_START_DATE = "2025-01-01"
TEST_END_DATE = "2025-12-31"

RET_THRESHOLD = 0.0010  # dead-zone: drop neutral moves

MARKET_TZ = "US/Eastern"
CUTOFF_HOUR = 16  # 4pm ET cutoff for bucketing headlines -> market day

OUTPUT_DIR = "ds207_simple_charts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FINBERT_MODEL = "ProsusAI/finbert"
FINBERT_CACHE_PATH = os.path.join(OUTPUT_DIR, "finbert_cache.parquet")

BATCH_SIZE = 128  # increase for GPU if you have VRAM
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN = 64

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

BUSINESS_SECTION_ALLOWLIST = {"business", "business day", "markets", "economy", "dealbook", "your money"}

MARKET_KEYWORDS = [
    "fed","federal reserve","inflation","cpi","jobs","payroll","unemployment","gdp",
    "rates","interest rate","treasury","bond","yields",
    "earnings","guidance","revenue","profit","margin",
    "stock","stocks","equity","equities","market","markets","s&p","sp500","spy",
    "nasdaq","dow","index","indices",
    "oil","crude","wti","brent","gas","energy",
    "dollar","usd","currency","fx",
    "recession","bank","banking","credit","defaults",
]

# -----------------------
# Helpers
# -----------------------
def log(msg: str):
    print(msg, flush=True)

def clean_fname(s: str) -> str:
    s = s.replace("/", "-")
    s = re.sub(r"\s+", "_", s.strip())
    s = re.sub(r"[^A-Za-z0-9_\-\.]+", "", s)
    return s

def assign_market_day(timestamp_utc: pd.Series) -> pd.Series:
    ts_et = timestamp_utc.dt.tz_convert(MARKET_TZ)
    minutes = ts_et.dt.hour * 60 + ts_et.dt.minute
    shift_next_day = (minutes >= CUTOFF_HOUR * 60).astype(int)
    market_day_dt = ts_et.dt.normalize() + pd.to_timedelta(shift_next_day, unit="D")
    return market_day_dt.dt.date

def time_window_label(timestamp_utc: pd.Series) -> pd.Series:
    ts_et = timestamp_utc.dt.tz_convert(MARKET_TZ)
    mins = ts_et.dt.hour * 60 + ts_et.dt.minute
    pre_start = 4 * 60
    reg_start = 9 * 60 + 30
    reg_end = 16 * 60
    aft_end = 20 * 60
    out = np.where(
        (mins >= pre_start) & (mins < reg_start), "premarket",
        np.where(
            (mins >= reg_start) & (mins < reg_end), "regular",
            np.where((mins >= reg_end) & (mins < aft_end), "after", "overnight")
        )
    )
    return pd.Series(out, index=timestamp_utc.index)

# -----------------------
# Load data
# -----------------------
def load_nyt_excel(excel_path: str, sheet_name=0) -> pd.DataFrame:
    df = pd.read_excel(excel_path, sheet_name=sheet_name, engine="openpyxl")
    df.columns = [c.strip().lower() for c in df.columns]
    if "date" not in df.columns or "headline" not in df.columns:
        raise ValueError(f"Excel must contain 'date' and 'headline'. Found: {df.columns.tolist()}")

    df["timestamp_utc"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    df = df.dropna(subset=["timestamp_utc", "headline"]).copy()
    df["headline"] = df["headline"].astype(str)

    if "section" in df.columns:
        df["section"] = df["section"].astype(str).str.strip().str.lower()

    df["market_day"] = assign_market_day(df["timestamp_utc"])
    df["time_window"] = time_window_label(df["timestamp_utc"])

    cols = ["market_day", "headline", "timestamp_utc", "time_window"] + (["section"] if "section" in df.columns else [])
    return df[cols].copy()

def load_spy_prices(start: str, end_exclusive: str) -> pd.DataFrame:
    spy = yf.download("SPY", start=start, end=end_exclusive, interval="1d", progress=False, auto_adjust=False)
    if spy is None or spy.empty:
        raise RuntimeError("yfinance returned no data")

    spy = spy.reset_index()
    spy.columns = [str(c).lower().replace(" ", "_") for c in spy.columns]
    if "date" in spy.columns:
        spy.rename(columns={"date": "market_day"}, inplace=True)
    spy["market_day"] = pd.to_datetime(spy["market_day"]).dt.date
    return spy[["market_day", "open", "high", "low", "close", "volume"]].copy()

def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ret_1d"] = out["close"].pct_change()
    out["vol_chg_1d"] = out["volume"].pct_change()
    out["ma_5"] = out["close"].rolling(5).mean()
    out["ma_10"] = out["close"].rolling(10).mean()
    out["mom_5"] = out["close"] - out["close"].shift(5)
    out["mom_10"] = out["close"] - out["close"].shift(10)
    return out

def add_target_next_day(df: pd.DataFrame, ret_threshold: float) -> pd.DataFrame:
    out = df.copy()
    out["close_next"] = out["close"].shift(-1)
    out["ret_next"] = (out["close_next"] / out["close"]) - 1.0
    out["target_up"] = np.where(
        out["ret_next"] > ret_threshold, 1,
        np.where(out["ret_next"] < -ret_threshold, 0, np.nan)
    )
    out["market_day_dt"] = pd.to_datetime(out["market_day"])
    return out

def split_train_test(df: pd.DataFrame):
    train = df[df["market_day_dt"] <= pd.to_datetime(TRAIN_END_DATE)].copy()
    test = df[(df["market_day_dt"] >= pd.to_datetime(TEST_START_DATE)) &
              (df["market_day_dt"] <= pd.to_datetime(TEST_END_DATE))].copy()
    return train, test

# -----------------------
# Filters
# -----------------------
def filter_all(df: pd.DataFrame) -> pd.DataFrame:
    return df.copy()

def filter_business_section(df: pd.DataFrame) -> pd.DataFrame:
    if "section" not in df.columns:
        return df.iloc[0:0].copy()
    return df[df["section"].isin(BUSINESS_SECTION_ALLOWLIST)].copy()

def filter_market_keywords(df: pd.DataFrame) -> pd.DataFrame:
    kws = sorted(set([k.lower().strip() for k in MARKET_KEYWORDS]), key=len, reverse=True)
    parts = []
    for kw in kws:
        if " " in kw:
            escaped = re.escape(kw).replace("\\ ", r"\s+")
            parts.append(rf"({escaped})")
        else:
            parts.append(rf"(\b{re.escape(kw)}\b)")
    pat = "|".join(parts)
    mask = df["headline"].str.lower().str.contains(pat, regex=True, na=False)
    return df[mask].copy()

# -----------------------
# FinBERT scoring (GPU-batched + cached)
# -----------------------
def load_finbert(device: str):
    tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL)
    model.to(device)
    model.eval()
    id2label = {int(k): v.lower() for k, v in model.config.id2label.items()}
    return tokenizer, model, id2label

@torch.no_grad()
def finbert_predict_proba(texts, tokenizer, model, device: str, batch_size: int, max_len: int):
    probs_out = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tokenizer(
            batch, padding=True, truncation=True, max_length=max_len, return_tensors="pt"
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
        probs_out.append(probs)
    return np.vstack(probs_out)

def finbert_score_headlines(df: pd.DataFrame, cache_path: str) -> pd.DataFrame:
    headlines = df["headline"].astype(str).tolist()
    uniq = pd.Series(headlines).drop_duplicates()
    uniq_list = uniq.tolist()

    if os.path.exists(cache_path):
        cache = pd.read_parquet(cache_path)
        cache = cache.drop_duplicates(subset=["headline"]).set_index("headline")
    else:
        cache = pd.DataFrame(columns=["p_pos","p_neu","p_neg","sent_score"]).set_index(pd.Index([], name="headline"))

    missing = [h for h in uniq_list if h not in cache.index]
    log(f"FinBERT scoring: unique headlines={len(uniq_list):,}, missing_from_cache={len(missing):,} (device={DEVICE})")

    if missing:
        tokenizer, model, id2label = load_finbert(DEVICE)
        probs = finbert_predict_proba(missing, tokenizer, model, DEVICE, BATCH_SIZE, MAX_LEN)

        labels = [id2label[i] for i in range(probs.shape[1])]
        label_to_col = {}
        for j, lab in enumerate(labels):
            if "positive" in lab:
                label_to_col["p_pos"] = j
            elif "neutral" in lab:
                label_to_col["p_neu"] = j
            elif "negative" in lab:
                label_to_col["p_neg"] = j

        p_pos = probs[:, label_to_col["p_pos"]]
        p_neu = probs[:, label_to_col["p_neu"]]
        p_neg = probs[:, label_to_col["p_neg"]]
        sent_score = p_pos - p_neg

        new = pd.DataFrame({
            "headline": missing,
            "p_pos": p_pos,
            "p_neu": p_neu,
            "p_neg": p_neg,
            "sent_score": sent_score
        }).set_index("headline")

        cache = pd.concat([cache, new], axis=0)
        cache.reset_index().to_parquet(cache_path, index=False)

    scored = df.join(cache, on="headline", how="left")
    for c in ["p_pos","p_neu","p_neg","sent_score"]:
        scored[c] = scored[c].astype(float).fillna(0.0)
    return scored

def aggregate_daily_finbert(scored_df: pd.DataFrame) -> pd.DataFrame:
    base = (
        scored_df.groupby("market_day")
        .agg(
            n_headlines=("sent_score", "count"),
            mean_sent=("sent_score", "mean"),
            mean_p_pos=("p_pos", "mean"),
            mean_p_neg=("p_neg", "mean"),
            mean_p_neu=("p_neu", "mean"),
        )
        .reset_index()
    )

    tw = (
        scored_df.groupby(["market_day", "time_window"])
        .agg(
            tw_mean_sent=("sent_score", "mean"),
            tw_n=("sent_score", "count")
        )
        .reset_index()
        .pivot(index="market_day", columns="time_window")
    )
    tw.columns = [f"{a}_{b}" for a, b in tw.columns]
    tw = tw.reset_index()

    daily = base.merge(tw, on="market_day", how="left")
    for c in daily.columns:
        if c != "market_day" and pd.api.types.is_numeric_dtype(daily[c]):
            daily[c] = daily[c].fillna(0.0)

    daily = daily.sort_values("market_day").copy()
    for col in ["mean_sent", "mean_p_pos", "mean_p_neg", "n_headlines"]:
        daily[f"{col}_lag1"] = daily[col].shift(1)
    lag_cols = [c for c in daily.columns if c.endswith("_lag1")]
    daily[lag_cols] = daily[lag_cols].fillna(0.0)

    return daily

# -----------------------
# Modeling
# -----------------------
@dataclass
class Metrics:
    auc: float
    balacc: float
    acc: float
    n: int

def prepare_xy(df: pd.DataFrame, feature_cols):
    d = df.dropna(subset=feature_cols + ["target_up"]).copy()
    X = d[feature_cols].values
    y = d["target_up"].astype(int).values
    return X, y, d

def time_series_cv_auc(model, X, y, splits=5):
    tscv = TimeSeriesSplit(n_splits=splits)
    aucs = []
    for tr, va in tscv.split(X):
        model.fit(X[tr], y[tr])
        p = model.predict_proba(X[va])[:, 1]
        aucs.append(roc_auc_score(y[va], p))
    return float(np.mean(aucs))

def eval_model(model, Xtr, ytr, Xte, yte):
    model.fit(Xtr, ytr)
    p = model.predict_proba(Xte)[:, 1]
    yhat = (p >= 0.5).astype(int)
    return p, Metrics(
        auc=float(roc_auc_score(yte, p)),
        balacc=float(balanced_accuracy_score(yte, yhat)),
        acc=float(accuracy_score(yte, yhat)),
        n=int(len(yte)),
    )

def model_bundle():
    logreg = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=5000, class_weight="balanced", solver="lbfgs"))
    ])
    hgb = HistGradientBoostingClassifier(max_depth=6, learning_rate=0.05, max_iter=400, random_state=RANDOM_SEED)
    return {"LogReg": logreg, "GradBoost(HGB)": hgb}

# -----------------------
# Simple charts
# -----------------------
def plot_sent_vs_ret(df_merged: pd.DataFrame, outpath: str):
    d = df_merged.sort_values("market_day_dt").copy()
    d["mean_sent_roll20"] = d["mean_sent"].rolling(20).mean()
    d["ret_next_roll20"] = d["ret_next"].rolling(20).mean()

    plt.figure(figsize=(9, 5))
    plt.plot(d["market_day_dt"], d["mean_sent_roll20"], label="FinBERT mean_sent (20d MA)")
    plt.plot(d["market_day_dt"], d["ret_next_roll20"], label="SPY next-day return (20d MA)")
    plt.xlabel("Date")
    plt.ylabel("Value (rolling)")
    plt.title("Rolling FinBERT Sentiment vs Next-day SPY Return")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def plot_model_auc_bar(results_df: pd.DataFrame, outpath: str):
    top = results_df.sort_values("Test_AUC", ascending=False).head(12).copy()
    labels = (top["Experiment"] + " | " + top["FeatureSet"] + " | " + top["Model"]).tolist()
    vals = top["Test_AUC"].tolist()

    plt.figure(figsize=(10, 6))
    plt.barh(labels[::-1], vals[::-1])
    plt.xlabel("Test AUC (2025)")
    plt.title("Top Model Configs by Test AUC")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def plot_best_roc(y_true, y_prob, title, outpath):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

# -----------------------
# Run one experiment
# -----------------------
def run_experiment(headlines: pd.DataFrame, spy_feat: pd.DataFrame, exp_name: str):
    scored = finbert_score_headlines(headlines, FINBERT_CACHE_PATH)
    daily_sent = aggregate_daily_finbert(scored)

    df = spy_feat.merge(daily_sent, on="market_day", how="left")
    for c in df.columns:
        if c != "market_day" and pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].fillna(0.0)

    df = add_target_next_day(df, RET_THRESHOLD)
    train, test = split_train_test(df)

    price_cols = ["ret_1d", "vol_chg_1d", "ma_5", "ma_10", "mom_5", "mom_10"]
    sent_cols = [c for c in daily_sent.columns if c != "market_day"]
    sent_cols = [c for c in sent_cols if c not in ["market_day_dt"]]

    feature_sets = {
        "Price-only": price_cols,
        "Sentiment-only": sent_cols,
        "Price+Sentiment": price_cols + sent_cols,
    }

    models = model_bundle()
    rows = []

    for fs_name, cols in feature_sets.items():
        Xtr, ytr, tr_clean = prepare_xy(train, cols)
        Xte, yte, te_clean = prepare_xy(test, cols)

        for mname, model in models.items():
            cv_auc = time_series_cv_auc(model, Xtr, ytr, splits=5)
            probs, met = eval_model(model, Xtr, ytr, Xte, yte)
            rows.append({
                "Experiment": exp_name,
                "FeatureSet": fs_name,
                "Model": mname,
                "CV_AUC": cv_auc,
                "Test_AUC": met.auc,
                "Test_BalAcc": met.balacc,
                "Test_Acc": met.acc,
                "Test_N": met.n,
            })

    return pd.DataFrame(rows), df

# -----------------------
# Main
# -----------------------
if __name__ == "__main__":
    log("START: FinBERT (GPU-batched) + simple charts pipeline")

    log("Loading NYT headlines from Excel...")
    headlines_all = load_nyt_excel(EXCEL_PATH, sheet_name=0)
    log(f"Loaded {len(headlines_all):,} headlines")

    log("Downloading SPY prices from Yahoo Finance...")
    spy = load_spy_prices(SPY_START, SPY_END_EXCLUSIVE)
    log(f"Downloaded {len(spy):,} SPY trading days")
    spy_feat = add_price_features(spy)

    experiments = [
        ("ALL NYT", filter_all),
        ("SECTION BUSINESS-ONLY", filter_business_section),
        ("KEYWORD MARKET-RELEVANT", filter_market_keywords),
    ]

    all_rows = []
    merged_for_chart = None

    for name, fn in experiments:
        log(f"\nApplying filter: {name}")
        h = fn(headlines_all)
        log(f"Filter {name}: {len(h):,} headlines")
        if len(h) == 0:
            continue

        res, merged = run_experiment(h, spy_feat, name)
        all_rows.append(res)

        if name == "ALL NYT":
            merged_for_chart = merged

    final = pd.concat(all_rows, ignore_index=True)
    final_sorted = final.sort_values("Test_AUC", ascending=False).reset_index(drop=True)

    log("\nFINAL TABLE (sorted by Test AUC)")
    print(final_sorted.to_string(index=False))

    if merged_for_chart is not None:
        plot_sent_vs_ret(
            merged_for_chart,
            os.path.join(OUTPUT_DIR, "chart_1_sent_vs_ret.png")
        )

    plot_model_auc_bar(
        final_sorted,
        os.path.join(OUTPUT_DIR, "chart_2_model_auc.png")
    )

    best = final_sorted.iloc[0]
    best_exp = best["Experiment"]

    if best_exp == "ALL NYT":
        exp_filter = filter_all
    elif best_exp == "SECTION BUSINESS-ONLY":
        exp_filter = filter_business_section
    else:
        exp_filter = filter_market_keywords

    h_best = exp_filter(headlines_all)
    res_best, merged_best = run_experiment(h_best, spy_feat, best_exp)

    price_cols = ["ret_1d", "vol_chg_1d", "ma_5", "ma_10", "mom_5", "mom_10"]
    scored_tmp = finbert_score_headlines(h_best, FINBERT_CACHE_PATH)
    daily_tmp = aggregate_daily_finbert(scored_tmp)
    sent_cols = [c for c in daily_tmp.columns if c != "market_day"]
    sent_cols = [c for c in sent_cols if c not in ["market_day_dt"]]

    fs = best["FeatureSet"]
    if fs == "Price-only":
        cols = price_cols
    elif fs == "Sentiment-only":
        cols = sent_cols
    else:
        cols = price_cols + sent_cols

    df_best = spy_feat.merge(daily_tmp, on="market_day", how="left")
    for c in df_best.columns:
        if c != "market_day" and pd.api.types.is_numeric_dtype(df_best[c]):
            df_best[c] = df_best[c].fillna(0.0)
    df_best = add_target_next_day(df_best, RET_THRESHOLD)
    train_best, test_best = split_train_test(df_best)

    Xtr, ytr, _ = prepare_xy(train_best, cols)
    Xte, yte, _ = prepare_xy(test_best, cols)

    model = model_bundle()[best["Model"]]
    model.fit(Xtr, ytr)
    p = model.predict_proba(Xte)[:, 1]

    plot_best_roc(
        yte, p,
        title=f"Best ROC (2025): {best_exp} | {fs} | {best['Model']}",
        outpath=os.path.join(OUTPUT_DIR, "chart_3_best_roc.png")
    )

    log(f"\nCharts saved to: {os.path.abspath(OUTPUT_DIR)}")