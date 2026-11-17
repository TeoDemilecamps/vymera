"""
Vymera - "Market Weather" volatility early-warning model
=========================================================

PURPOSE
    Predict the probability that the market enters a high-volatility regime
    (VIX closing at or above 25) at least once in the next 10 trading days.
    Vymera combines this market-level probability with each user's own
    exposure (beta, concentration, currency) to decide WHEN an explained
    alert is worth sending, so users are not spammed on every small dip.

INPUTS (features, all computed from information available on day t)
    vix            VIX close on day t
    log_vix        natural log of VIX
    chg_1d         1-day % change in VIX
    chg_5d         5-day % change in VIX
    chg_20d        20-day % change in VIX
    ratio_ma20     VIX / 20-day moving average
    ratio_ma60     VIX / 60-day moving average
    volvol_20d     20-day std. dev. of daily log changes ("volatility of volatility")
    range_pct      (high - low) / close on day t (intraday stress)
    pct_rank_252   percentile of today's VIX within the last 252 days

OUTPUT
    p_storm  in [0, 1]: probability of a high-volatility regime within 10 days.
    The app maps it to "Market Weather":
        < 0.30  Calm      0.30-0.60  Choppy      >= 0.60  Stormy

MODELS COMPARED
    1. Naive baseline: "storm ahead" if VIX >= 25 today
    2. Logistic regression (chosen for the app: transparent, exportable to JS)
    3. Gradient boosting (benchmark: more flexible, harder to explain)

VALIDATION
    Strict time split (no shuffling, no look-ahead):
        train 1990-2015 | validation 2016-2018 | test 2019-2026
    The test period contains COVID-19 (2020), the 2022 rate shock and the
    April 2025 tariff sell-off.

Run:  python vymera_volatility_model.py
Outputs go to ./outputs (metrics, model weights for the app, charts).
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (brier_score_loss, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "vix-daily.csv")
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)

STORM_LEVEL = 25.0      # VIX level treated as "high-volatility regime"
HORIZON = 10            # trading days ahead
THRESHOLD = 0.5         # decision threshold for alert metrics

NAVY, GREEN, LIGHT, GREY, RED = "#1E2263", "#6CC201", "#D2E8F2", "#8A8FA8", "#E4572E"

FEATURES = ["vix", "log_vix", "chg_1d", "chg_5d", "chg_20d", "ratio_ma20",
            "ratio_ma60", "volvol_20d", "range_pct", "pct_rank_252"]


# ---------------------------------------------------------------- 1. data
VIX_URL = "https://raw.githubusercontent.com/datasets/finance-vix/main/data/vix-daily.csv"


def load_data(path=DATA):
    if not os.path.exists(path):          # first run: download the public CBOE VIX history
        os.makedirs(os.path.dirname(path), exist_ok=True)
        pd.read_csv(VIX_URL).to_csv(path, index=False)
    df = pd.read_csv(path, parse_dates=["DATE"]).rename(columns=str.lower)
    df = df.rename(columns={"date": "date"}).set_index("date").sort_index()
    df = df[df["close"] > 0]
    return df


# ---------------------------------------------------------- 2. features
def build_features(df):
    f = pd.DataFrame(index=df.index)
    c = df["close"]
    f["vix"] = c
    f["log_vix"] = np.log(c)
    f["chg_1d"] = c.pct_change(1)
    f["chg_5d"] = c.pct_change(5)
    f["chg_20d"] = c.pct_change(20)
    f["ratio_ma20"] = c / c.rolling(20).mean()
    f["ratio_ma60"] = c / c.rolling(60).mean()
    f["volvol_20d"] = np.log(c).diff().rolling(20).std()
    f["range_pct"] = (df["high"] - df["low"]) / c
    f["pct_rank_252"] = c.rolling(252).apply(lambda w: (w <= w[-1]).mean(), raw=True)
    return f


def build_target(df, level=STORM_LEVEL, horizon=HORIZON):
    c = df["close"]
    # max VIX close over the NEXT `horizon` days (t+1 ... t+horizon)
    future_max = c[::-1].rolling(horizon, min_periods=horizon).max()[::-1].shift(-1)
    y = (future_max >= level).astype(float)
    y[future_max.isna()] = np.nan
    return y


# ------------------------------------------------------------- 3. train
def main():
    raw = load_data()
    X_all = build_features(raw)
    y_all = build_target(raw)
    data = X_all.assign(y=y_all).dropna()

    train = data.loc[:"2015-12-31"]
    valid = data.loc["2016-01-01":"2018-12-31"]
    test = data.loc["2019-01-01":]

    Xtr, ytr = train[FEATURES], train["y"]
    Xva, yva = valid[FEATURES], valid["y"]
    Xte, yte = test[FEATURES], test["y"]

    logit = make_pipeline(StandardScaler(),
                          LogisticRegression(C=1.0, max_iter=2000))
    logit.fit(Xtr, ytr)

    gbm = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05,
                                         max_iter=300, random_state=42)
    gbm.fit(Xtr, ytr)

    preds = {
        "Naive baseline (VIX >= 25 today)": (Xte["vix"] >= STORM_LEVEL).astype(float).values,
        "Logistic regression": logit.predict_proba(Xte)[:, 1],
        "Gradient boosting": gbm.predict_proba(Xte)[:, 1],
    }

    # ------------------------------------------------------- 4. evaluate
    metrics = {}
    for name, p in preds.items():
        yhat = (p >= THRESHOLD).astype(int)
        tn, fp, fn, tp = confusion_matrix(yte, yhat).ravel()
        metrics[name] = {
            "roc_auc": round(float(roc_auc_score(yte, p)), 3),
            "precision": round(float(precision_score(yte, yhat)), 3),
            "recall": round(float(recall_score(yte, yhat)), 3),
            "f1": round(float(f1_score(yte, yhat)), 3),
            "brier": round(float(brier_score_loss(yte, p)), 3),
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        }

    # Early-warning test: of the days that START a storm (VIX < 25 today but
    # >= 25 within 10 days), how many did each model flag in advance?
    calm_today = Xte["vix"] < STORM_LEVEL
    onset = calm_today & (yte == 1)
    for name, p in preds.items():
        flagged = (p[onset.values] >= THRESHOLD).mean() if onset.sum() else 0
        metrics[name]["early_warning_recall"] = round(float(flagged), 3)
    # Early-warning false alarm rate on calm days that stay calm
    stays_calm = calm_today & (yte == 0)
    for name, p in preds.items():
        fa = (p[stays_calm.values] >= THRESHOLD).mean()
        metrics[name]["false_alarm_rate_calm_days"] = round(float(fa), 3)

    summary = {
        "target": f"VIX close >= {STORM_LEVEL} at least once in next {HORIZON} trading days",
        "train_period": f"{train.index.min().date()} to {train.index.max().date()} ({len(train)} days)",
        "valid_period": f"{valid.index.min().date()} to {valid.index.max().date()} ({len(valid)} days)",
        "test_period": f"{test.index.min().date()} to {test.index.max().date()} ({len(test)} days)",
        "test_base_rate": round(float(yte.mean()), 3),
        "storm_onset_days_in_test": int(onset.sum()),
        "decision_threshold": THRESHOLD,
        "models": metrics,
    }
    with open(os.path.join(OUT, "metrics.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    # ------------------------------------- 5. export model for the web app
    scaler = logit.named_steps["standardscaler"]
    lr = logit.named_steps["logisticregression"]
    weights = {
        "model": "logistic_regression",
        "target": summary["target"],
        "features": FEATURES,
        "mean": [round(float(v), 6) for v in scaler.mean_],
        "scale": [round(float(v), 6) for v in scaler.scale_],
        "coef": [round(float(v), 6) for v in lr.coef_[0]],
        "intercept": round(float(lr.intercept_[0]), 6),
        "bands": {"calm": [0, 0.3], "choppy": [0.3, 0.6], "stormy": [0.6, 1.0]},
        "trained_on": summary["train_period"],
        "test_auc": metrics["Logistic regression"]["roc_auc"],
    }
    with open(os.path.join(OUT, "model_weights.json"), "w") as fh:
        json.dump(weights, fh, indent=2)

    # recent VIX history so the app can compute features in the browser
    recent = raw.iloc[-300:][["high", "low", "close"]].round(2)
    recent_out = {"dates": [d.strftime("%Y-%m-%d") for d in recent.index],
                  "high": recent["high"].tolist(), "low": recent["low"].tolist(),
                  "close": recent["close"].tolist()}
    # Real crisis windows (with ~1 year of warm-up history so the browser can
    # compute 252-day features) for the demo replay and rule backtests
    def window(start, end):
        w = raw.loc[start:end][["high", "low", "close"]].round(2)
        p = pd.Series(np.nan, index=w.index)
        feats = X_all.loc[w.index, FEATURES].dropna()
        p.loc[feats.index] = logit.predict_proba(feats)[:, 1]
        return {"dates": [d.strftime("%Y-%m-%d") for d in w.index],
                "high": w["high"].tolist(), "low": w["low"].tolist(),
                "close": w["close"].tolist(),
                "p_python": [None if np.isnan(v) else round(float(v), 4) for v in p]}
    with open(os.path.join(OUT, "vix_recent.json"), "w") as fh:
        json.dump({"recent": recent_out,
                   "tariff_2025": window("2024-01-02", "2025-06-30"),
                   "covid_2020": window("2019-01-02", "2020-06-30")}, fh)

    # ---------------------------------------------------------- 6. charts
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.edgecolor": GREY, "axes.labelcolor": NAVY,
                         "xtick.color": NAVY, "ytick.color": NAVY,
                         "axes.spines.top": False, "axes.spines.right": False})

    # 6a. ROC curves
    fig, ax = plt.subplots(figsize=(6.4, 5))
    colors = [GREY, GREEN, NAVY]
    for (name, p), col in zip(preds.items(), colors):
        fpr, tpr, _ = roc_curve(yte, p)
        ax.plot(fpr, tpr, color=col, lw=2.5 if col == GREEN else 1.8,
                label=f"{name}  (AUC {metrics[name]['roc_auc']:.2f})")
    ax.plot([0, 1], [0, 1], ls=":", color=GREY, lw=1)
    ax.set_xlabel("False alarm rate")
    ax.set_ylabel("Storms caught (recall)")
    ax.set_title("Test set 2019–2026: ROC curves", color=NAVY, loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "roc_curves.png"), dpi=200)
    plt.close(fig)

    # 6b. Probability vs VIX during 2020 and 2025
    p_test = pd.Series(preds["Logistic regression"], index=Xte.index)
    for label, start, end in [("covid_2020", "2020-01-01", "2020-06-30"),
                              ("tariff_2025", "2025-01-01", "2025-06-30")]:
        seg_p = p_test.loc[start:end]
        seg_v = Xte["vix"].loc[start:end]
        fig, ax1 = plt.subplots(figsize=(8, 3.8))
        ax1.fill_between(seg_p.index, 0, seg_p.values, color=GREEN, alpha=0.25, lw=0)
        ax1.plot(seg_p.index, seg_p.values, color=GREEN, lw=2, label="Vymera storm probability")
        ax1.set_ylim(0, 1.02)
        ax1.set_ylabel("Storm probability", color=NAVY)
        ax2 = ax1.twinx()
        ax2.plot(seg_v.index, seg_v.values, color=NAVY, lw=1.6, label="VIX")
        ax2.axhline(STORM_LEVEL, color=RED, ls="--", lw=1)
        ax2.text(seg_v.index[2], STORM_LEVEL + 1.5, "Storm level (VIX 25)", color=RED, fontsize=9)
        ax2.set_ylabel("VIX", color=NAVY)
        ax2.spines["right"].set_visible(True)
        title = "COVID-19 crash, 2020" if label == "covid_2020" else "Tariff sell-off, 2025"
        ax1.set_title(f"{title}: model probability vs VIX", color=NAVY, loc="left", fontweight="bold")
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, frameon=False, fontsize=9, loc="upper left")
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, f"timeline_{label}.png"), dpi=200)
        plt.close(fig)

    # 6c. Logistic regression coefficients (standardised = comparable)
    coefs = pd.Series(lr.coef_[0], index=FEATURES).sort_values()
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.barh(coefs.index, coefs.values,
            color=[GREEN if v > 0 else NAVY for v in coefs.values])
    ax.axvline(0, color=GREY, lw=1)
    ax.set_xlabel("Standardised coefficient (→ raises storm probability)")
    ax.set_title("What drives the forecast", color=NAVY, loc="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "feature_weights.png"), dpi=200)
    plt.close(fig)

    print(json.dumps(summary, indent=2))
    print("\nLatest data point:", raw.index[-1].date(), "VIX", raw["close"].iloc[-1])
    latest = X_all.dropna().iloc[[-1]][FEATURES]
    print("Current storm probability:", round(float(logit.predict_proba(latest)[0, 1]), 3))


if __name__ == "__main__":
    main()
