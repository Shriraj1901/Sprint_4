import os
import time
import pickle
from datetime import datetime, time as dtime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
import pandas_ta as ta
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dateutil import tz
import sys

os.environ["PYTHONWARNINGS"] = "ignore"

# Optional boosters — install with: pip install xgboost lightgbm
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
    print("XGBoost available")
except ImportError:
    HAS_XGB = False
    print("XGBoost not found — run: pip install xgboost")

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
    print("LightGBM available")
except ImportError:
    HAS_LGBM = False
    print("LightGBM not found — run: pip install lightgbm")

st.set_page_config(layout="wide", page_title="NSE AI Trader")

# ── Load stock list ───────────────────────────────────────
if not os.path.exists("nse_stocks.pkl"):
    st.error("Run python stocks_list.py first! nse_stocks.pkl missing")
    st.stop()

with open("nse_stocks.pkl", "rb") as f:
    STOCK_MAP = pickle.load(f)

STOCKS = sorted([v.get("yf_symbol") for v in STOCK_MAP.values() if v.get("yf_symbol")])

# results from training
ALL_RESULTS_PATH = os.path.join("models", "all_results.pkl")

# Helper: load API key pattern (reads .env if present)
def load_groq_key():
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass
    return os.environ.get("GROQ_API_KEY") or os.environ.get("OPENROUTER_API_KEY")

# ── Features ──────────────────────────────────────────────
FEATURES = [
    "RSI_7","RSI_14","RSI_21","RSI_diff",
    "MACD","MACD_sig","MACD_hist","MACD_cross",
    "SMA_10","SMA_20","SMA_50","SMA_200",
    "EMA_9","EMA_21","EMA_50",
    "SMA_cross_10_50","SMA_cross_20_200",
    "Price_SMA20","Price_SMA50","Price_SMA200",
    "BB_upper","BB_lower","BB_width","BB_pos","BB_squeeze",
    "ATR","ATR_pct","ATR_ratio",
    "Vol_ratio","Vol_SMA_ratio","OBV","OBV_slope","VWAP_dist",
    "ROC_5","ROC_10","ROC_20",
    "Return_1d","Return_2d","Return_3d","Return_5d","Return_10d","Return_20d",
    "Pos_10","Pos_20","Pos_52w",
    "Body","Shadow_up","Shadow_dn","Body_ratio",
    "Higher_high","Lower_low",
    "STOCH_k","STOCH_d",
    "WILLR","CCI",
    "ADX","DI_plus","DI_minus","ADX_trend",
    "TENKAN","KIJUN","SENKOU_A","SENKOU_B","ICH_cloud_pos",
    "Trend_strength","Volatility_20","Vol_regime",
    "Breakout_20","Breakdown_20","Gap_pct",
    "Day_of_week","Month","Quarter","Week_of_year",
    "Is_month_start","Is_month_end","Is_quarter_end",
]

def flatten_columns(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

def get_col(df, prefix):
    match = [c for c in df.columns if c.startswith(prefix)]
    if not match:
        raise KeyError(f"No column: '{prefix}'")
    return match[0]

def add_features(df):
    close = df["Close"]; high = df["High"]
    low   = df["Low"];   vol  = df["Volume"]; open_ = df["Open"]

    # RSI
    df["RSI_7"]    = ta.rsi(close, length=7)
    df["RSI_14"]   = ta.rsi(close, length=14)
    df["RSI_21"]   = ta.rsi(close, length=21)
    df["RSI_diff"] = df["RSI_14"] - df["RSI_14"].shift(1)

    # MACD
    macd = ta.macd(close)
    if macd is None: raise ValueError("MACD None")
    df["MACD"]       = macd[get_col(macd, "MACD_")]
    df["MACD_sig"]   = macd[get_col(macd, "MACDs_")]
    df["MACD_hist"]  = macd[get_col(macd, "MACDh_")]
    df["MACD_cross"] = (df["MACD"] > df["MACD_sig"]).astype(int)

    # Moving averages
    df["SMA_10"]  = ta.sma(close, length=10)
    df["SMA_20"]  = ta.sma(close, length=20)
    df["SMA_50"]  = ta.sma(close, length=50)
    df["SMA_200"] = ta.sma(close, length=200)
    df["EMA_9"]   = ta.ema(close, length=9)
    df["EMA_21"]  = ta.ema(close, length=21)
    df["EMA_50"]  = ta.ema(close, length=50)
    df["SMA_cross_10_50"]  = (df["SMA_10"] > df["SMA_50"]).astype(int)
    df["SMA_cross_20_200"] = (df["SMA_20"] > df["SMA_200"]).astype(int)
    df["Price_SMA20"]  = (close > df["SMA_20"]).astype(int)
    df["Price_SMA50"]  = (close > df["SMA_50"]).astype(int)
    df["Price_SMA200"] = (close > df["SMA_200"]).astype(int)

    # Bollinger Bands
    bb = ta.bbands(close, length=20)
    if bb is None: raise ValueError("BB None")
    df["BB_upper"]   = bb[get_col(bb, "BBU_")]
    df["BB_lower"]   = bb[get_col(bb, "BBL_")]
    df["BB_mid"]     = bb[get_col(bb, "BBM_")]
    df["BB_width"]   = (df["BB_upper"] - df["BB_lower"]) / df["BB_mid"]
    df["BB_pos"]     = (close - df["BB_lower"]) / (df["BB_upper"] - df["BB_lower"] + 1e-9)
    df["BB_squeeze"] = (df["BB_width"] < df["BB_width"].rolling(20).mean()).astype(int)

    # Volatility
    df["ATR"]       = ta.atr(high, low, close, length=14)
    df["ATR_pct"]   = df["ATR"] / close
    df["ATR_ratio"] = df["ATR"] / (df["ATR"].rolling(20).mean() + 1e-9)

    # Volume
    df["Vol_SMA20"]     = ta.sma(vol, length=20)
    df["Vol_ratio"]     = vol / (df["Vol_SMA20"] + 1e-9)
    df["Vol_SMA_ratio"] = df["Vol_SMA20"] / (df["Vol_SMA20"].rolling(5).mean() + 1e-9)
    df["OBV"]           = ta.obv(close, vol)
    df["OBV_slope"]     = df["OBV"].diff(5)
    tp                  = (high + low + close) / 3
    df["VWAP_dist"]     = (close - (tp * vol).cumsum() / (vol.cumsum() + 1e-9)) / (close + 1e-9)

    # Momentum
    df["ROC_5"]  = ta.roc(close, length=5)
    df["ROC_10"] = ta.roc(close, length=10)
    df["ROC_20"] = ta.roc(close, length=20)
    for n in [1,2,3,5,10,20]:
        df[f"Return_{n}d"] = close.pct_change(n)

    # Price position
    df["High_10"]  = high.rolling(10).max()
    df["Low_10"]   = low.rolling(10).min()
    df["High_20"]  = high.rolling(20).max()
    df["Low_20"]   = low.rolling(20).min()
    df["High_52w"] = high.rolling(252).max()
    df["Low_52w"]  = low.rolling(252).min()
    df["Pos_10"]   = (close - df["Low_10"])  / (df["High_10"]  - df["Low_10"]  + 1e-9)
    df["Pos_20"]   = (close - df["Low_20"])  / (df["High_20"]  - df["Low_20"]  + 1e-9)
    df["Pos_52w"]  = (close - df["Low_52w"]) / (df["High_52w"] - df["Low_52w"] + 1e-9)

    # Candlestick
    mx = pd.concat([close, open_], axis=1).max(axis=1)
    mn = pd.concat([close, open_], axis=1).min(axis=1)
    df["Body"]       = (close - open_) / (open_ + 1e-9)
    df["Shadow_up"]  = (high - mx) / (open_ + 1e-9)
    df["Shadow_dn"]  = (mn - low)  / (open_ + 1e-9)
    df["Body_ratio"] = abs(df["Body"]) / (df["Shadow_up"] + df["Shadow_dn"] + abs(df["Body"]) + 1e-9)
    df["Higher_high"]= (high > high.shift(1)).astype(int)
    df["Lower_low"]  = (low  < low.shift(1)).astype(int)

    # Stochastic
    stoch = ta.stoch(high, low, close)
    if stoch is not None:
        kc = [c for c in stoch.columns if c.startswith("STOCHk")]
        dc = [c for c in stoch.columns if c.startswith("STOCHd")]
        df["STOCH_k"] = stoch[kc[0]] if kc else 50.0
        df["STOCH_d"] = stoch[dc[0]] if dc else 50.0
    else:
        df["STOCH_k"] = df["STOCH_d"] = 50.0

    # Williams %R
    willr = ta.willr(high, low, close)
    df["WILLR"] = willr if willr is not None else -50.0

    # CCI
    cci = ta.cci(high, low, close)
    df["CCI"] = cci if cci is not None else 0.0

    # ADX
    adx_df = ta.adx(high, low, close)
    if adx_df is not None:
        ac = [c for c in adx_df.columns if c.startswith("ADX_")]
        pc = [c for c in adx_df.columns if c.startswith("DMP_")]
        nc = [c for c in adx_df.columns if c.startswith("DMN_")]
        df["ADX"]      = adx_df[ac[0]] if ac else 25.0
        df["DI_plus"]  = adx_df[pc[0]] if pc else 25.0
        df["DI_minus"] = adx_df[nc[0]] if nc else 25.0
    else:
        df["ADX"] = df["DI_plus"] = df["DI_minus"] = 25.0
    df["ADX_trend"] = (df["ADX"] > 25).astype(int)

    # Ichimoku
    ich = ta.ichimoku(high, low, close)
    if ich is not None and isinstance(ich, tuple) and ich[0] is not None and not ich[0].empty:
        ic = ich[0]; cols = ic.columns.tolist()
        def pick(p):
            m = [c for c in cols if c.startswith(p)]
            return ic[m[0]] if m else pd.Series(close.values, index=close.index)
        df["TENKAN"]   = pick("ITS_")
        df["KIJUN"]    = pick("IKS_")
        df["SENKOU_A"] = pick("ISA_")
        df["SENKOU_B"] = pick("ISB_")
    else:
        for c in ["TENKAN","KIJUN","SENKOU_A","SENKOU_B"]:
            df[c] = close
    df["ICH_cloud_pos"] = (close > df[["SENKOU_A","SENKOU_B"]].max(axis=1)).astype(int)

    # Regime / structure
    df["Trend_strength"] = (df["EMA_21"] - df["EMA_50"]) / (close + 1e-9)
    df["Volatility_20"]  = close.pct_change().rolling(20).std()
    df["Vol_regime"]     = df["Volatility_20"] / (df["Volatility_20"].rolling(100).mean() + 1e-9)
    df["Breakout_20"]    = (close > high.shift(1).rolling(20).max()).astype(int)
    df["Breakdown_20"]   = (close < low.shift(1).rolling(20).min()).astype(int)
    df["Gap_pct"]        = (open_ - close.shift(1)) / (close.shift(1) + 1e-9)

    # Time
    df["Day_of_week"]    = df.index.dayofweek
    df["Month"]          = df.index.month
    df["Quarter"]        = df.index.quarter
    df["Week_of_year"]   = df.index.isocalendar().week.astype(int)
    df["Is_month_start"] = df.index.is_month_start.astype(int)
    df["Is_month_end"]   = df.index.is_month_end.astype(int)
    df["Is_quarter_end"] = df.index.is_quarter_end.astype(int)

    return df


# ── CACHED LOADERS & UTILITIES ──────────────────────────
@st.cache_data(ttl=60 * 60)
def load_all_results():
    if os.path.exists(ALL_RESULTS_PATH):
        with open(ALL_RESULTS_PATH, "rb") as f:
            return pickle.load(f)
    # fallback: scan models folder for per-symbol PKLs (fast lookup)
    res = {}
    try:
        for fn in os.listdir("models"):
            if fn.endswith(".pkl") and fn != "all_results.pkl":
                path = os.path.join("models", fn)
                try:
                    with open(path, "rb") as f:
                        data = pickle.load(f)
                except Exception:
                    data = None
                # some pickles store top-level result dict
                if isinstance(data, dict) and "models" in data and "features" in data:
                    ticker = fn.replace("_NS.pkl", ".NS").replace(".pkl", "")
                    res[ticker] = {"confidence": data.get("weights", [0])[0] if data.get("weights") else 0}
    except Exception:
        pass
    return res


@st.cache_data(ttl=60 * 60)
def load_model_result(symbol):
    # try all_results first
    if os.path.exists(ALL_RESULTS_PATH):
        try:
            with open(ALL_RESULTS_PATH, "rb") as f:
                allr = pickle.load(f)
            if symbol in allr:
                return allr[symbol]
        except Exception:
            pass
    # load per-symbol model file
    fn = os.path.join("models", symbol.replace('.', '_') + ".pkl")
    if os.path.exists(fn):
        try:
            with open(fn, "rb") as f:
                pk = pickle.load(f)
            # some pickles include only models/scaler; return minimal
            return pk
        except Exception:
            pass
    return None


def market_is_open(now_utc=None):
    # NSE hours: Mon-Fri 09:15 to 15:30 IST (UTC+5:30)
    if now_utc is None:
        now_utc = datetime.utcnow().replace(tzinfo=tz.tzutc())
    ist = now_utc.astimezone(tz.gettz('Asia/Kolkata'))
    if ist.weekday() >= 5:
        return False, ist
    open_t = dtime(9, 15)
    close_t = dtime(15, 30)
    return (open_t <= ist.time() <= close_t), ist


@st.cache_data(ttl=60)
def fetch_live_price(symbol):
    # use yfinance as fallback for live quote
    try:
        t = yf.Ticker(symbol)
        hist = t.history(period="2d", interval="1m")
        if hist is not None and not hist.empty:
            last = hist['Close'].iloc[-1]
            prev = hist['Close'].iloc[-2] if len(hist) > 1 else last
            return round(float(last),2), round(float(last - prev),2)
    except Exception:
        pass
    # final fallback: last close from history day
    try:
        h = yf.download(symbol, period='5d', progress=False, auto_adjust=True)
        if not h.empty:
            last = h['Close'].iloc[-1]
            prev = h['Close'].iloc[-2] if len(h) > 1 else last
            return round(float(last),2), round(float(last - prev),2)
    except Exception:
        pass
    return None, None


def semicircle_gauge(conf):
    # conf: 0-100
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=conf,
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={'shape': "angular",
               'axis': {'range': [0,100], 'tickwidth': 1, 'tickcolor': "darkgray"},
               'bar': {'color': "rgba(0,0,0,0)"},
               'steps': [
                   {'range': [0,55], 'color':'#ff4d4d'},
                   {'range': [55,65], 'color':'#f5d94f'},
                   {'range': [65,100], 'color':'#4cd964'},
               ],
               'threshold': {'line': {'color': "black", 'width': 4}, 'value': conf}
              }
    ))
    fig.update_layout(margin=dict(l=0,r=0,t=0,b=0), height=220)
    return fig


# Preload top 10 on startup (non-blocking cache)
_ = load_all_results()


def train_stock(symbol):
    t0 = time.time()

    end_date = (pd.Timestamp.utcnow() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    df = yf.download(symbol, start="2014-01-01", end=end_date,
                     progress=False, auto_adjust=True)
    if df.empty or len(df) < 400:
        return None, "not enough data"

    df = flatten_columns(df)
    needed = [c for c in ["Open","High","Low","Close","Volume"] if c in df.columns]
    if len(needed) < 5:
        return None, "missing OHLCV"
    df = df[needed].copy()
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    try:
        df = add_features(df)
    except Exception as e:
        return None, f"feature error: {e}"

    df = df.replace([np.inf, -np.inf], np.nan)

    # Target — predict 2-day direction (less noise than 1-day)
    future = df["Close"].shift(-2)
    df["Target"] = np.where(future.notna(), (future > df["Close"]).astype(int), np.nan)
    df.dropna(subset=["Target"], inplace=True)
    df["Target"] = df["Target"].astype(int)

    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        return None, f"missing features: {missing[:3]}"

    X = df[FEATURES].copy().fillna(df[FEATURES].median()).replace([np.inf,-np.inf], 0)
    y = df["Target"].copy()

    if len(X) < 300:
        return None, "too little data"

    # ── Time series CV ────────────────────────────────────
    tscv   = TimeSeriesSplit(n_splits=5)
    scores = []
    for tr, te in tscv.split(X):
        m = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        m.fit(X.iloc[tr], y.iloc[tr])
        scores.append(accuracy_score(y.iloc[te], m.predict(X.iloc[te])))
    cv_acc = np.mean(scores) * 100

    # ── Train / Val / Test split ──────────────────────────
    n      = len(X)
    tr_end = int(n * 0.70)
    va_end = int(n * 0.85)

    X_tr = X.iloc[:tr_end]; y_tr = y.iloc[:tr_end]
    X_va = X.iloc[tr_end:va_end]; y_va = y.iloc[tr_end:va_end]
    X_te = X.iloc[va_end:]; y_te = y.iloc[va_end:]

    if len(X_tr) < 200 or len(X_va) < 40 or len(X_te) < 40:
        return None, "not enough samples"

    # ── Scale for LR ─────────────────────────────────────
    scaler  = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_va_sc = scaler.transform(X_va)
    X_te_sc = scaler.transform(X_te)

    # ── Train all models ──────────────────────────────────
    models     = {}
    val_probs  = []
    test_probs = []
    lat_probs  = []

    latest    = X.iloc[-1:].fillna(X.median())
    latest_sc = scaler.transform(latest)

    # 1 — Random Forest
    rf = RandomForestClassifier(
        n_estimators=500, max_depth=15, min_samples_split=15,
        min_samples_leaf=5, max_features="sqrt",
        class_weight="balanced_subsample", random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    models["rf"] = rf
    val_probs.append(rf.predict_proba(X_va)[:,1])
    test_probs.append(rf.predict_proba(X_te)[:,1])
    lat_probs.append(rf.predict_proba(latest)[0][1])

    # 2 — Extra Trees
    et = ExtraTreesClassifier(
        n_estimators=400, max_depth=12, min_samples_split=20,
        class_weight="balanced", random_state=42, n_jobs=-1)
    et.fit(X_tr, y_tr)
    models["et"] = et
    val_probs.append(et.predict_proba(X_va)[:,1])
    test_probs.append(et.predict_proba(X_te)[:,1])
    lat_probs.append(et.predict_proba(latest)[0][1])

    # 3 — Gradient Boosting
    gb = GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, random_state=42)
    gb.fit(X_tr, y_tr)
    models["gb"] = gb
    val_probs.append(gb.predict_proba(X_te)[:,1])
    test_probs.append(gb.predict_proba(X_te)[:,1])
    lat_probs.append(gb.predict_proba(latest)[0][1])

    # 4 — Logistic Regression
    lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42, n_jobs=-1)
    lr.fit(X_tr_sc, y_tr)
    models["lr"] = lr
    val_probs.append(lr.predict_proba(X_va_sc)[:,1])
    test_probs.append(lr.predict_proba(X_te_sc)[:,1])
    lat_probs.append(lr.predict_proba(latest_sc)[0][1])

    # 5 — XGBoost (biggest upgrade)
    if HAS_XGB:
        try:
            xgb = XGBClassifier(
                n_estimators=300, max_depth=5, learning_rate=0.05,
                subsample=0.85, colsample_bytree=0.85,
                reg_lambda=1.5, reg_alpha=0.1,
                eval_metric="logloss", random_state=42, n_jobs=-1,
                verbosity=0
            )
            xgb.fit(X_tr, y_tr,
                    eval_set=[(X_va, y_va)],
                    verbose=False)
            models["xgb"] = xgb
            val_probs.append(xgb.predict_proba(X_va)[:,1])
            test_probs.append(xgb.predict_proba(X_te)[:,1])
            lat_probs.append(xgb.predict_proba(latest)[0][1])
            print("xgb✓", end=" ", flush=True)
        except Exception as e:
            print(f"xgb_err({e})", end=" ", flush=True)

    # 6 — LightGBM (fastest, finds leaf patterns)
    if HAS_LGBM:
        try:
            lgbm = LGBMClassifier(
                n_estimators=300, learning_rate=0.05, num_leaves=31,
                subsample=0.85, colsample_bytree=0.85,
                class_weight="balanced", random_state=42, verbose=-1
            )
            lgbm.fit(X_tr, y_tr,
                     eval_set=[(X_va, y_va)],
                     callbacks=[])
            models["lgbm"] = lgbm
            val_probs.append(lgbm.predict_proba(X_va)[:,1])
            test_probs.append(lgbm.predict_proba(X_te)[:,1])
            lat_probs.append(lgbm.predict_proba(latest)[0][1])
            print("lgbm✓", end=" ", flush=True)
        except Exception as e:
            print(f"lgbm_err({e})", end=" ", flush=True)

    # ── Find best weights using validation set ────────────
    # Grid search over weights — finds the combination that
    # maximizes validation accuracy
    n_models = len(val_probs)
    best_w   = [1.0/n_models] * n_models
    best_acc = -1.0

    rng = np.random.default_rng(42)
    candidates = [np.ones(n_models)/n_models]
    for i in range(n_models):
        w = np.zeros(n_models); w[i] = 1.0
        candidates.append(w)
    for _ in range(200):
        candidates.append(rng.dirichlet(np.ones(n_models)))

    for w in candidates:
        ens  = sum(w[i]*val_probs[i] for i in range(n_models))
        pred = (ens > 0.5).astype(int)
        acc  = accuracy_score(y_va, pred)
        if acc > best_acc:
            best_acc = acc
            best_w   = w.tolist()

    # ── Final test accuracy ───────────────────────────────
    ens_test  = sum(best_w[i]*test_probs[i] for i in range(n_models))
    final_pred = (ens_test > 0.5).astype(int)
    final_acc  = accuracy_score(y_te, final_pred) * 100

    # ── Predict latest ────────────────────────────────────
    prob_up = float(sum(best_w[i]*lat_probs[i] for i in range(n_models)))
    pred    = "UP" if prob_up > 0.5 else "DOWN"
    conf    = round(max(prob_up, 1-prob_up) * 100, 2)

    # ── Feature importance from RF ────────────────────────
    fi = dict(zip(FEATURES, rf.feature_importances_))

    # ── Chart data (last 2 years only to save memory) ─────
    chart_cols = ["Open","High","Low","Close","Volume","SMA_20","SMA_50",
                  "SMA_200","BB_upper","BB_lower","RSI_14","STOCH_k","STOCH_d",
                  "MACD","MACD_sig","MACD_hist","Vol_SMA20","ADX","DI_plus","DI_minus"]
    avail      = [c for c in chart_cols if c in df.columns]
    chart_df   = df[avail].tail(520).copy()
    for c in chart_df.select_dtypes("float64").columns:
        chart_df[c] = chart_df[c].astype("float32")

    result = {
        "symbol"         : symbol,
        "accuracy"       : round(final_acc, 2),
        "cv_accuracy"    : round(cv_acc, 2),
        "val_accuracy"   : round(best_acc*100, 2),
        "prediction"     : pred,
        "confidence"     : conf,
        "prob_up"        : round(prob_up*100, 2),
        "ensemble_models": list(models.keys()),
        "ensemble_weights": [round(float(w),3) for w in best_w],
        "current_price"  : round(float(df["Close"].iloc[-1]), 2),
        "rsi"            : round(float(df["RSI_14"].iloc[-1]), 2),
        "macd"           : round(float(df["MACD"].iloc[-1]), 4),
        "sma_20"         : round(float(df["SMA_20"].iloc[-1]), 2),
        "sma_50"         : round(float(df["SMA_50"].iloc[-1]), 2),
        "sma_200"        : round(float(df["SMA_200"].iloc[-1]), 2),
        "bb_pos"         : round(float(df["BB_pos"].iloc[-1]), 2),
        "bb_squeeze"     : int(df["BB_squeeze"].iloc[-1]),
        "vol_ratio"      : round(float(df["Vol_ratio"].iloc[-1]), 2),
        "atr_pct"        : round(float(df["ATR_pct"].iloc[-1]), 4),
        "adx"            : round(float(df["ADX"].iloc[-1]), 2),
        "stoch_k"        : round(float(df["STOCH_k"].iloc[-1]), 2),
        "willr"          : round(float(df["WILLR"].iloc[-1]), 2),
        "cci"            : round(float(df["CCI"].iloc[-1]), 2),
        "ich_cloud_pos"  : int(df["ICH_cloud_pos"].iloc[-1]),
        "obv_slope"      : round(float(df["OBV_slope"].iloc[-1]), 0),
        "df"             : chart_df,
        "feature_importance": fi,
        "trained_at"     : time.strftime("%Y-%m-%d %H:%M"),
        "elapsed_sec"    : round(time.time()-t0, 1),
    }

    return (result, models, scaler), None


# Training / batch processing is handled via the `--train` flag or the separate `train.py` script.
# To run full batch training: `python app.py --train` or `python train.py`.


# ── TRAINING MAIN (only run when script invoked with --train) ─────────────────
if '--train' in sys.argv:
    os.makedirs("models", exist_ok=True)
    results_path = ALL_RESULTS_PATH

    # training-only imports
    try:
        from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.metrics import accuracy_score
    except Exception as e:
        print("Missing sklearn dependencies for training:", e)
        sys.exit(1)

    if os.path.exists(results_path):
        with open(results_path, "rb") as f:
            all_results = pickle.load(f)
        print(f"Resuming — {len(all_results)} already done")
    else:
        all_results = {}
        print("Starting fresh")

    remaining = [s for s in STOCKS if s not in all_results]
    total     = len(STOCKS)
    done      = len(all_results)
    est_hrs   = len(remaining) * 4.5 / 60

    print(f"Total   : {total}")
    print(f"Done    : {done}")
    print(f"Left    : {len(remaining)}")
    print(f"Est.    : {est_hrs:.1f} hours")
    print(f"Start   : {time.strftime('%H:%M:%S')}")
    print(f"Finish  : {time.strftime('%H:%M', time.localtime(time.time()+est_hrs*3600))}")
    print("="*60 + "\n")

    if not remaining:
        print("All done! Run: python -m streamlit run app.py")
        sys.exit(0)

    t_start = time.time()
    failed  = []
    saved_n = 0

    for i, stock in enumerate(remaining, 1):
        idx = done + i
        print(f"[{idx}/{total}] {stock}", end=" ", flush=True)

        out, err = train_stock(stock)

        if err:
            print(f"SKIP — {err}")
            failed.append((stock, err))
            continue

        result, models, scaler = out
        all_results[stock] = result

        with open(f"models/{stock.replace('.','_')}.pkl", "wb") as f:
            pickle.dump({
                "models"   : models,
                "scaler"   : scaler,
                "features" : FEATURES,
                "weights"  : result["ensemble_weights"],
                "model_names": result["ensemble_models"],
            }, f)

        saved_n += 1
        if saved_n % 10 == 0:
            with open(results_path, "wb") as f:
                pickle.dump(all_results, f)
            print(f"\n  [Saved: {len(all_results)}/{total}]\n")

        elapsed = (time.time()-t_start)/60
        eta     = (len(remaining)-i) * (elapsed/i)
        print(f"| Acc:{result['accuracy']:.1f}% CV:{result['cv_accuracy']:.1f}% "
              f"Val:{result['val_accuracy']:.1f}% "
              f"Models:{'+'.join(result['ensemble_models'])} "
              f"{result['prediction']}({result['confidence']:.0f}%) "
              f"ETA:{eta:.0f}m")

    with open(results_path, "wb") as f:
        pickle.dump(all_results, f)

    mins = (time.time()-t_start)/60
    print(f"\nDONE! Trained {len(all_results)} stocks in {mins:.0f} mins")
    print(f"Failed: {len(failed)}")
    for s,e in failed[:10]:
        print(f"  {s}: {e}")
    print("\nRun: python -m streamlit run app.py")


# ── STREAMLIT UI ───────────────────────────────────────
def main():
    st.title("NSE AI Trader")

    # Header: market status
    open_flag, ist = market_is_open()
    col1, col2 = st.columns([3,1])
    with col1:
        st.write(f"**Time (IST):** {ist.strftime('%Y-%m-%d %H:%M:%S')}")
    with col2:
        color = "#2ecc71" if open_flag else "#ff4d4d"
        label = "Market Open" if open_flag else "Market Closed"
        st.markdown(f"<div style='display:flex;align-items:center'>"
                    f"<div style='width:10px;height:10px;background:{color};border-radius:50%;margin-right:8px'></div>"
                    f"<b>{label}</b></div>", unsafe_allow_html=True)

    # Sidebar: selector + top picks
    st.sidebar.header("Search / Select")
    selected = st.sidebar.selectbox("Select stock", STOCKS, index=0, key="sel_stock")

    # Top picks
    allr = load_all_results()
    top5 = []
    try:
        if isinstance(allr, dict):
            items = [(k, v.get('confidence', 0) if isinstance(v, dict) else 0) for k,v in allr.items()]
            items = sorted(items, key=lambda x: x[1], reverse=True)[:10]
            top5 = items[:5]
    except Exception:
        top5 = []

    st.sidebar.markdown("### Top Picks")
    for sym, conf in top5:
        if st.sidebar.button(f"{sym} — {conf:.1f}%"):
            st.session_state.sel_stock = sym

    # Preload spinner while loading symbol data
    with st.spinner("Loading model and data..."):
        result = load_model_result(selected)
        # if model result is structured as train result, it contains df and keys
        chart_df = None
        if isinstance(result, dict) and 'df' in result:
            chart_df = result['df']
        elif os.path.exists(ALL_RESULTS_PATH):
            try:
                with open(ALL_RESULTS_PATH, 'rb') as f:
                    allr = pickle.load(f)
                if selected in allr:
                    chart_df = allr[selected].get('df')
            except Exception:
                chart_df = None

    # Market closed warning
    if not open_flag:
        st.info("Market is closed. Predictions are based on last trading session data.")

    # Top-level metrics
    live_price, delta = fetch_live_price(selected)
    col1, col2, col3 = st.columns([2,2,2])
    with col1:
        if live_price is not None:
            delta_color = 'green' if delta >= 0 else 'red'
            st.metric(label="Live Price", value=f"₹{live_price}", delta=f"{delta:.2f}", delta_color=delta_color)
        else:
            st.metric(label="Live Price", value="n/a")
    with col2:
        conf = result.get('confidence', 0) if isinstance(result, dict) else 0
        figg = semicircle_gauge(conf)
        st.plotly_chart(figg, use_container_width=True)
    with col3:
        pred = result.get('prediction', 'N/A') if isinstance(result, dict) else 'N/A'
        st.write("**Prediction**")
        st.write(f"{pred}")

    # Tabs: Charts / Compare / Portfolio
    tabs = st.tabs(["Charts","Compare","My Portfolio"])

    # Charts tab
    with tabs[0]:
        if chart_df is None:
            st.warning("Chart data not available for this symbol")
        else:
            dfc = chart_df.copy()
            dfc.index = pd.to_datetime(dfc.index)
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                row_heights=[0.7,0.3], vertical_spacing=0.02,
                                specs=[[{"type":"candlestick"}], [{"type":"bar"}]])
            fig.add_trace(go.Candlestick(x=dfc.index, open=dfc['Open'], high=dfc['High'], low=dfc['Low'], close=dfc['Close'], name='Price'), row=1, col=1)
            if 'SMA_20' in dfc.columns:
                fig.add_trace(go.Scatter(x=dfc.index, y=dfc['SMA_20'], mode='lines', name='SMA20'), row=1, col=1)
            if 'SMA_50' in dfc.columns:
                fig.add_trace(go.Scatter(x=dfc.index, y=dfc['SMA_50'], mode='lines', name='SMA50'), row=1, col=1)
            fig.add_trace(go.Bar(x=dfc.index, y=dfc['Volume'], name='Volume', marker_color='lightgray'), row=2, col=1)
            fig.update(layout_xaxis_rangeslider_visible=False)
            fig.update_layout(height=700, margin=dict(l=10,r=10,t=30,b=10))
            st.plotly_chart(fig, use_container_width=True)

            # RSI + Stoch
            cols = st.columns(2)
            with cols[0]:
                if 'RSI_14' in dfc.columns:
                    fig2 = go.Figure()
                    fig2.add_trace(go.Scatter(x=dfc.index, y=dfc['RSI_14'], name='RSI'))
                    fig2.add_hline(y=70, line_dash='dash', line_color='gray')
                    fig2.add_hline(y=30, line_dash='dash', line_color='gray')
                    st.plotly_chart(fig2, use_container_width=True)
            with cols[1]:
                if 'STOCH_k' in dfc.columns and 'STOCH_d' in dfc.columns:
                    f3 = go.Figure()
                    f3.add_trace(go.Scatter(x=dfc.index, y=dfc['STOCH_k'], name='STOCH_k'))
                    f3.add_trace(go.Scatter(x=dfc.index, y=dfc['STOCH_d'], name='STOCH_d'))
                    st.plotly_chart(f3, use_container_width=True)

            # MACD
            if 'MACD' in dfc.columns and 'MACD_sig' in dfc.columns:
                fm = go.Figure()
                fm.add_trace(go.Scatter(x=dfc.index, y=dfc['MACD'], name='MACD'))
                fm.add_trace(go.Scatter(x=dfc.index, y=dfc['MACD_sig'], name='MACD_sig'))
                st.plotly_chart(fm, use_container_width=True)

            # ADX
            if 'ADX' in dfc.columns:
                fa = go.Figure()
                fa.add_trace(go.Scatter(x=dfc.index, y=dfc['ADX'], name='ADX'))
                st.plotly_chart(fa, use_container_width=True)

    # Compare tab
    with tabs[1]:
        c1, c2 = st.columns([1,1])
        with c1:
            s1 = st.selectbox("Stock A", STOCKS, index=STOCKS.index(selected) if selected in STOCKS else 0, key='cmp_a')
        with c2:
            s2 = st.selectbox("Stock B", STOCKS, index=1, key='cmp_b')
        if st.button("Compare"):
            r1 = load_model_result(s1) or {}
            r2 = load_model_result(s2) or {}
            # side-by-side key metrics
            dfc = pd.DataFrame([
                {"Metric":"Price","A": fetch_live_price(s1)[0], "B": fetch_live_price(s2)[0]},
                {"Metric":"Prediction","A": r1.get('prediction'), "B": r2.get('prediction')},
                {"Metric":"Confidence","A": r1.get('confidence'), "B": r2.get('confidence')},
                {"Metric":"RSI","A": r1.get('rsi'), "B": r2.get('rsi')},
                {"Metric":"ADX","A": r1.get('adx'), "B": r2.get('adx')},
            ])
            st.table(dfc.set_index('Metric'))

    # Portfolio tab
    with tabs[2]:
        st.header("My Portfolio")
        if 'portfolio' not in st.session_state:
            st.session_state.portfolio = []

        with st.form("add_pos"):
            s = st.selectbox("Symbol", STOCKS)
            qty = st.number_input("Quantity", min_value=1, value=1)
            price = st.number_input("Buy Price (₹)", min_value=0.0, format="%.2f")
            add = st.form_submit_button("Add")
            if add:
                st.session_state.portfolio.append({"symbol": s, "qty": qty, "buy_price": price})

        if st.session_state.portfolio:
            rows = []
            for h in st.session_state.portfolio:
                lp, _ = fetch_live_price(h['symbol'])
                curr_val = (lp or 0) * h['qty']
                invested = h['buy_price'] * h['qty']
                pnl = curr_val - invested
                pnl_pct = (pnl / invested * 100) if invested else 0
                res = load_model_result(h['symbol']) or {}
                reco = 'Hold' if res.get('prediction') == 'UP' and res.get('confidence',0) > 60 else 'Sell'
                rows.append({"Symbol": h['symbol'], "Qty": h['qty'], "Buy": h['buy_price'], "Current": lp, "Value": curr_val, "P&L": round(pnl,2), "%": round(pnl_pct,2), "AI": reco})
            st.dataframe(pd.DataFrame(rows))


if __name__ == '__main__':
    main()