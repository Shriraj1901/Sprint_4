import os
import warnings
import time
import pickle
import numpy as np
import pandas as pd
import pandas_ta as ta
import yfinance as yf

os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None
try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None

# ── Load stock list ───────────────────────────────────────
if not os.path.exists("nse_stocks.pkl"):
    print("Run python stocks_list.py first!")
    exit()

with open("nse_stocks.pkl", "rb") as f:
    STOCK_MAP = pickle.load(f)

STOCKS = [v["yf_symbol"] for v in STOCK_MAP.values()]
print(f"Loaded {len(STOCKS)} stocks\n")

# ── Feature list ──────────────────────────────────────────
FEATURES = [
    # RSI family
    "RSI_7","RSI_14","RSI_21","RSI_diff",
    # MACD family
    "MACD","MACD_sig","MACD_hist","MACD_cross",
    # Moving averages
    "SMA_10","SMA_20","SMA_50","SMA_200",
    "EMA_9","EMA_21","EMA_50",
    "SMA_cross_10_50","SMA_cross_20_200","Price_SMA20","Price_SMA50","Price_SMA200",
    # Bollinger Bands
    "BB_upper","BB_lower","BB_width","BB_pos","BB_squeeze",
    # Volatility
    "ATR","ATR_pct","ATR_ratio",
    # Volume
    "Vol_ratio","Vol_SMA_ratio","OBV","OBV_slope","VWAP_dist",
    # Momentum
    "ROC_5","ROC_10","ROC_20",
    "Return_1d","Return_2d","Return_3d","Return_5d","Return_10d","Return_20d",
    # Price patterns
    "Pos_10","Pos_20","Pos_52w",
    "Body","Shadow_up","Shadow_dn","Body_ratio",
    "Higher_high","Lower_low",
    # Stochastic
    "STOCH_k","STOCH_d",
    # Williams %R
    "WILLR",
    # CCI
    "CCI",
    # ADX (trend strength)
    "ADX","DI_plus","DI_minus","ADX_trend",
    # Ichimoku
    "TENKAN","KIJUN","SENKOU_A","SENKOU_B",
    "ICH_cloud_pos",
    # Extra analyzers (regime / structure)
    "Trend_strength","Volatility_20","Vol_regime","Breakout_20","Breakdown_20","Gap_pct",
    # Market-context analyzers (NIFTY + relative strength)
    "MKT_ret_1d","MKT_ret_3d","MKT_ret_5d","MKT_rsi_14","MKT_trend_20_50","MKT_vol_10",
    "Rel_5","Rel_20","Beta_60","Corr_20",
    # Time features
    "Day_of_week","Month","Quarter","Week_of_year",
    "Is_month_start","Is_month_end","Is_quarter_end",
]

PRED_HORIZON_DAYS = 1
FLAT_MOVE_ATR_MULT = 0.7
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
USE_XGB = True
USE_LGBM = True
MARKET_SYMBOL = "^NSEI"
_MARKET_CACHE = None

def flatten_columns(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

def get_col(df, prefix):
    match = [c for c in df.columns if c.startswith(prefix)]
    if not match:
        raise KeyError(f"No column with prefix '{prefix}'")
    return match[0]

def load_market_data(end_date):
    global _MARKET_CACHE
    if _MARKET_CACHE is not None and not _MARKET_CACHE.empty:
        return _MARKET_CACHE.copy()

    mkt = yf.download(
        MARKET_SYMBOL, start="2014-01-01", end=end_date,
        progress=False, auto_adjust=True
    )
    if mkt is None or mkt.empty:
        _MARKET_CACHE = pd.DataFrame()
        return _MARKET_CACHE.copy()

    mkt = flatten_columns(mkt)
    mkt = mkt[[c for c in ["Open", "High", "Low", "Close", "Volume"] if c in mkt.columns]].copy()
    mkt = mkt.replace([np.inf, -np.inf], np.nan).dropna()
    _MARKET_CACHE = mkt
    return _MARKET_CACHE.copy()

def add_market_features(df, market_df):
    if market_df is None or market_df.empty:
        for col in ["MKT_ret_1d","MKT_ret_3d","MKT_ret_5d","MKT_rsi_14","MKT_trend_20_50","MKT_vol_10","Rel_5","Rel_20","Beta_60","Corr_20"]:
            df[col] = 0.0
        return df

    m = market_df.reindex(df.index).copy()
    m["MKT_ret_1d"] = m["Close"].pct_change(1)
    m["MKT_ret_3d"] = m["Close"].pct_change(3)
    m["MKT_ret_5d"] = m["Close"].pct_change(5)
    m["MKT_rsi_14"] = ta.rsi(m["Close"], length=14)
    m["MKT_sma20"] = ta.sma(m["Close"], length=20)
    m["MKT_sma50"] = ta.sma(m["Close"], length=50)
    m["MKT_trend_20_50"] = ((m["MKT_sma20"] > m["MKT_sma50"]).astype(float))
    m["MKT_vol_10"] = m["MKT_ret_1d"].rolling(10).std()

    sret1 = df["Close"].pct_change(1)
    mret1 = m["MKT_ret_1d"]
    mret5 = m["Close"].pct_change(5)
    mret20 = m["Close"].pct_change(20)

    df["MKT_ret_1d"] = mret1
    df["MKT_ret_3d"] = m["MKT_ret_3d"]
    df["MKT_ret_5d"] = m["MKT_ret_5d"]
    df["MKT_rsi_14"] = m["MKT_rsi_14"]
    df["MKT_trend_20_50"] = m["MKT_trend_20_50"]
    df["MKT_vol_10"] = m["MKT_vol_10"]
    df["Rel_5"] = df["Close"].pct_change(5) - mret5
    df["Rel_20"] = df["Close"].pct_change(20) - mret20
    # Rolling beta and correlation against NIFTY
    cov = sret1.rolling(60).cov(mret1)
    var = mret1.rolling(60).var()
    df["Beta_60"] = cov / (var + 1e-9)
    df["Corr_20"] = sret1.rolling(20).corr(mret1)
    return df

def tune_weighted_ensemble(val_probs, y_val, step=0.1):
    n_models = len(val_probs)
    if n_models == 1:
        return [1.0], 0.5, accuracy_score(y_val, (val_probs[0] > 0.5).astype(int))

    rng = np.random.default_rng(42)
    best_w = [1.0 / n_models] * n_models
    best_thr = 0.5
    best_acc = -1.0
    ties = None

    candidates = [np.array(best_w, dtype=float)]
    for i in range(n_models):
        one = np.zeros(n_models, dtype=float)
        one[i] = 1.0
        candidates.append(one)
    # random convex combinations keep search fast per stock
    for _ in range(120):
        candidates.append(rng.dirichlet(np.ones(n_models)))

    for w in candidates:
        ens = np.zeros_like(val_probs[0], dtype=float)
        for i in range(n_models):
            ens += w[i] * val_probs[i]
        # Threshold search
        for thr in np.arange(0.35, 0.66, 0.01):
            pred = (ens > thr).astype(int)
            acc = accuracy_score(y_val, pred)
            tie_break = (abs(thr - 0.5), np.count_nonzero(w), -w.max())
            if (acc > best_acc) or (abs(acc - best_acc) < 1e-12 and (ties is None or tie_break < ties)):
                best_acc = acc
                best_w = w.tolist()
                best_thr = float(thr)
                ties = tie_break

    return best_w, best_thr, best_acc

def add_features(df):
    close  = df["Close"]
    high   = df["High"]
    low    = df["Low"]
    volume = df["Volume"]
    open_  = df["Open"]

    # ── RSI family ────────────────────────────────────────
    df["RSI_7"]    = ta.rsi(close, length=7)
    df["RSI_14"]   = ta.rsi(close, length=14)
    df["RSI_21"]   = ta.rsi(close, length=21)
    df["RSI_diff"] = df["RSI_14"] - df["RSI_14"].shift(1)

    # ── MACD ─────────────────────────────────────────────
    macd = ta.macd(close)
    if macd is None:
        raise ValueError("MACD None")
    df["MACD"]      = macd[get_col(macd, "MACD_")]
    df["MACD_sig"]  = macd[get_col(macd, "MACDs_")]
    df["MACD_hist"] = macd[get_col(macd, "MACDh_")]
    df["MACD_cross"]= (df["MACD"] > df["MACD_sig"]).astype(int)

    # ── Moving averages ───────────────────────────────────
    df["SMA_10"]  = ta.sma(close, length=10)
    df["SMA_20"]  = ta.sma(close, length=20)
    df["SMA_50"]  = ta.sma(close, length=50)
    df["SMA_200"] = ta.sma(close, length=200)
    df["EMA_9"]   = ta.ema(close, length=9)
    df["EMA_21"]  = ta.ema(close, length=21)
    df["EMA_50"]  = ta.ema(close, length=50)

    df["SMA_cross_10_50"]  = (df["SMA_10"]  > df["SMA_50"]).astype(int)
    df["SMA_cross_20_200"] = (df["SMA_20"]  > df["SMA_200"]).astype(int)
    df["Price_SMA20"]      = (close > df["SMA_20"]).astype(int)
    df["Price_SMA50"]      = (close > df["SMA_50"]).astype(int)
    df["Price_SMA200"]     = (close > df["SMA_200"]).astype(int)

    # ── Bollinger Bands ───────────────────────────────────
    bb = ta.bbands(close, length=20)
    if bb is None:
        raise ValueError("BB None")
    df["BB_upper"]   = bb[get_col(bb, "BBU_")]
    df["BB_lower"]   = bb[get_col(bb, "BBL_")]
    df["BB_mid"]     = bb[get_col(bb, "BBM_")]
    df["BB_width"]   = (df["BB_upper"] - df["BB_lower"]) / df["BB_mid"]
    df["BB_pos"]     = (close - df["BB_lower"]) / (df["BB_upper"] - df["BB_lower"] + 1e-9)
    df["BB_squeeze"] = (df["BB_width"] < df["BB_width"].rolling(20).mean()).astype(int)

    # ── Volatility ────────────────────────────────────────
    df["ATR"]      = ta.atr(high, low, close, length=14)
    df["ATR_pct"]  = df["ATR"] / close
    df["ATR_ratio"]= df["ATR"] / df["ATR"].rolling(20).mean()

    # ── Volume ────────────────────────────────────────────
    df["Vol_SMA20"]    = ta.sma(volume, length=20)
    df["Vol_ratio"]    = volume / (df["Vol_SMA20"] + 1e-9)
    df["Vol_SMA_ratio"]= df["Vol_SMA20"] / (df["Vol_SMA20"].rolling(5).mean() + 1e-9)
    df["OBV"]          = ta.obv(close, volume)
    df["OBV_slope"]    = df["OBV"].diff(5)
    # VWAP approximation
    typical_price      = (high + low + close) / 3
    df["VWAP_dist"]    = (close - (typical_price * volume).cumsum() / volume.cumsum()) / close

    # ── Momentum / ROC ────────────────────────────────────
    df["ROC_5"]  = ta.roc(close, length=5)
    df["ROC_10"] = ta.roc(close, length=10)
    df["ROC_20"] = ta.roc(close, length=20)

    for n in [1, 2, 3, 5, 10, 20]:
        df[f"Return_{n}d"] = close.pct_change(n)

    # ── Price position ────────────────────────────────────
    df["High_10"]  = high.rolling(10).max()
    df["Low_10"]   = low.rolling(10).min()
    df["High_20"]  = high.rolling(20).max()
    df["Low_20"]   = low.rolling(20).min()
    df["High_52w"] = high.rolling(252).max()
    df["Low_52w"]  = low.rolling(252).min()
    df["Pos_10"]   = (close - df["Low_10"])  / (df["High_10"]  - df["Low_10"]  + 1e-9)
    df["Pos_20"]   = (close - df["Low_20"])  / (df["High_20"]  - df["Low_20"]  + 1e-9)
    df["Pos_52w"]  = (close - df["Low_52w"]) / (df["High_52w"] - df["Low_52w"] + 1e-9)

    # ── Candlestick ───────────────────────────────────────
    df["Body"]      = (close - open_) / (open_ + 1e-9)
    df["Shadow_up"] = (high - pd.concat([close, open_], axis=1).max(axis=1)) / (open_ + 1e-9)
    df["Shadow_dn"] = (pd.concat([close, open_], axis=1).min(axis=1) - low) / (open_ + 1e-9)
    df["Body_ratio"]= abs(df["Body"]) / (df["Shadow_up"] + df["Shadow_dn"] + abs(df["Body"]) + 1e-9)
    df["Higher_high"]= (high > high.shift(1)).astype(int)
    df["Lower_low"]  = (low  < low.shift(1)).astype(int)

    # ── Stochastic ────────────────────────────────────────
    stoch = ta.stoch(high, low, close)
    if stoch is not None:
        k_col = [c for c in stoch.columns if c.startswith("STOCHk")]
        d_col = [c for c in stoch.columns if c.startswith("STOCHd")]
        df["STOCH_k"] = stoch[k_col[0]] if k_col else 50.0
        df["STOCH_d"] = stoch[d_col[0]] if d_col else 50.0
    else:
        df["STOCH_k"] = 50.0
        df["STOCH_d"] = 50.0

    # ── Williams %R ───────────────────────────────────────
    willr = ta.willr(high, low, close)
    df["WILLR"] = willr if willr is not None else -50.0

    # ── CCI ───────────────────────────────────────────────
    cci = ta.cci(high, low, close)
    df["CCI"] = cci if cci is not None else 0.0

    # ── ADX ───────────────────────────────────────────────
    adx = ta.adx(high, low, close)
    if adx is not None:
        adx_col   = [c for c in adx.columns if c.startswith("ADX_")]
        dmp_col   = [c for c in adx.columns if c.startswith("DMP_")]
        dmn_col   = [c for c in adx.columns if c.startswith("DMN_")]
        df["ADX"]      = adx[adx_col[0]] if adx_col else 25.0
        df["DI_plus"]  = adx[dmp_col[0]] if dmp_col else 25.0
        df["DI_minus"] = adx[dmn_col[0]] if dmn_col else 25.0
    else:
        df["ADX"] = 25.0; df["DI_plus"] = 25.0; df["DI_minus"] = 25.0
    df["ADX_trend"] = (df["ADX"] > 25).astype(int)

    # ── Ichimoku ─────────────────────────────────────────
    ich = ta.ichimoku(high, low, close)
    if ich is not None and isinstance(ich, tuple):
        ich_df = ich[0]
        if ich_df is not None and not ich_df.empty:
            cols = ich_df.columns.tolist()
            def pick(prefix):
                m = [c for c in cols if c.startswith(prefix)]
                return ich_df[m[0]] if m else pd.Series(close.values, index=close.index)
            df["TENKAN"]   = pick("ITS_")
            df["KIJUN"]    = pick("IKS_")
            df["SENKOU_A"] = pick("ISA_")
            df["SENKOU_B"] = pick("ISB_")
        else:
            for c in ["TENKAN","KIJUN","SENKOU_A","SENKOU_B"]:
                df[c] = close
    else:
        for c in ["TENKAN","KIJUN","SENKOU_A","SENKOU_B"]:
            df[c] = close

    df["ICH_cloud_pos"] = (close > df[["SENKOU_A","SENKOU_B"]].max(axis=1)).astype(int)

    # Extra analyzers (regime / structure)
    df["Trend_strength"] = (df["EMA_21"] - df["EMA_50"]) / (close + 1e-9)
    df["Volatility_20"] = close.pct_change().rolling(20).std()
    df["Vol_regime"] = df["Volatility_20"] / (df["Volatility_20"].rolling(100).mean() + 1e-9)
    df["Breakout_20"] = (close > high.shift(1).rolling(20).max()).astype(int)
    df["Breakdown_20"] = (close < low.shift(1).rolling(20).min()).astype(int)
    df["Gap_pct"] = (open_ - close.shift(1)) / (close.shift(1) + 1e-9)

    # ── Time features ─────────────────────────────────────
    df["Day_of_week"]   = df.index.dayofweek
    df["Month"]         = df.index.month
    df["Quarter"]       = df.index.quarter
    df["Week_of_year"]  = df.index.isocalendar().week.astype(int)
    df["Is_month_start"]= df.index.is_month_start.astype(int)
    df["Is_month_end"]  = df.index.is_month_end.astype(int)
    df["Is_quarter_end"]= df.index.is_quarter_end.astype(int)

    return df


def train_stock(symbol):
    t0 = time.time()

    # Download 10 years
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
    df = df.replace([np.inf, -np.inf], np.nan)
    df.dropna(inplace=True)

    try:
        df = add_features(df)
    except Exception as e:
        return None, f"feature error: {e}"

    # Add broader market context analyzers (NIFTY regime + relative strength)
    market_df = load_market_data(end_date)
    df = add_market_features(df, market_df)

    # Replace inf values
    df = df.replace([np.inf, -np.inf], np.nan)

    # TARGET — predict 1-day direction with flat-move filtering
    future_close = df["Close"].shift(-PRED_HORIZON_DAYS)
    future_ret = (future_close - df["Close"]) / (df["Close"] + 1e-9)
    flat_cutoff = (df["ATR_pct"] * FLAT_MOVE_ATR_MULT).clip(lower=0.001)
    valid_move = future_ret.abs() > flat_cutoff
    df["Target"] = np.where(
        future_close.notna() & valid_move,
        (future_close > df["Close"]).astype(int),
        np.nan
    )
    df.dropna(subset=["Target"], inplace=True)
    df["Target"] = df["Target"].astype(int)

    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        return None, f"missing: {missing[:3]}"

    X = df[FEATURES].copy()
    y = df["Target"].copy()

    # Fill remaining NaN with column medians
    X = X.fillna(X.median())
    X = X.replace([np.inf, -np.inf], 0)

    if len(X) < 300:
        return None, "too little data after inf cleanup"

    # ── Time series CV for honest accuracy ───────────────
    tscv   = TimeSeriesSplit(n_splits=5)
    scores = []
    for tr_idx, te_idx in tscv.split(X):
        rf_cv = RandomForestClassifier(
            n_estimators=120, random_state=42, n_jobs=-1, class_weight="balanced_subsample"
        )
        rf_cv.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        scores.append(accuracy_score(y.iloc[te_idx], rf_cv.predict(X.iloc[te_idx])))
    cv_acc = np.mean(scores) * 100

    # ── Final ensemble — 4 models ─────────────────────────
    n = len(X)
    tr_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))
    if tr_end < 200 or (val_end - tr_end) < 40 or (n - val_end) < 40:
        return None, "insufficient samples after target filtering"

    X_train = X.iloc[:tr_end]
    y_train = y.iloc[:tr_end]
    X_val = X.iloc[tr_end:val_end]
    y_val = y.iloc[tr_end:val_end]
    X_test = X.iloc[val_end:]
    y_test = y.iloc[val_end:]

    # Model 1 — Random Forest (best for this task)
    rf = RandomForestClassifier(
        n_estimators=600, max_depth=16, min_samples_split=12,
        min_samples_leaf=5, max_features="sqrt",
        class_weight="balanced_subsample", random_state=42, n_jobs=-1
    )

    # Model 2 — Extra Trees (faster, different bias)
    et = ExtraTreesClassifier(
        n_estimators=400, max_depth=14, min_samples_split=14,
        class_weight="balanced", random_state=42, n_jobs=-1
    )

    # Model 3 — Gradient Boosting (sequential, catches different patterns)
    gb = GradientBoostingClassifier(
        n_estimators=250, max_depth=3, learning_rate=0.04,
        subsample=0.8, random_state=42
    )

    # Model 4 — Logistic Regression on scaled features (linear patterns)
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_val_sc = scaler.transform(X_val)
    X_te_sc = scaler.transform(X_test)
    lr = LogisticRegression(max_iter=1200, class_weight="balanced", random_state=42, n_jobs=-1)

    rf.fit(X_train, y_train)
    et.fit(X_train, y_train)
    gb.fit(X_train, y_train)
    lr.fit(X_tr_sc, y_train)

    # Advanced ensemble: adaptive weights + optional boosters
    xgb = None
    lgbm = None
    val_probs = []
    test_probs = []
    latest_probs = []
    model_names = []

    rf_prob_val = rf.predict_proba(X_val)[:, 1]
    et_prob_val = et.predict_proba(X_val)[:, 1]
    gb_prob_val = gb.predict_proba(X_val)[:, 1]
    lr_prob_val = lr.predict_proba(X_val_sc)[:, 1]
    rf_prob = rf.predict_proba(X_test)[:, 1]
    et_prob = et.predict_proba(X_test)[:, 1]
    gb_prob = gb.predict_proba(X_test)[:, 1]
    lr_prob = lr.predict_proba(X_te_sc)[:, 1]

    latest_x = X.iloc[-1:].fillna(X.median())
    latest_sc = scaler.transform(latest_x)

    val_probs.extend([rf_prob_val, et_prob_val, gb_prob_val, lr_prob_val])
    test_probs.extend([rf_prob, et_prob, gb_prob, lr_prob])
    latest_probs.extend([
        rf.predict_proba(latest_x)[0][1],
        et.predict_proba(latest_x)[0][1],
        gb.predict_proba(latest_x)[0][1],
        lr.predict_proba(latest_sc)[0][1],
    ])
    model_names.extend(["rf", "et", "gb", "lr"])

    if USE_XGB and XGBClassifier is not None:
        try:
            xgb = XGBClassifier(
                n_estimators=260, max_depth=5, learning_rate=0.04,
                subsample=0.85, colsample_bytree=0.85, reg_lambda=1.0,
                objective="binary:logistic", random_state=42, n_jobs=-1, eval_metric="logloss"
            )
            xgb.fit(X_train, y_train)
            val_probs.append(xgb.predict_proba(X_val)[:, 1])
            test_probs.append(xgb.predict_proba(X_test)[:, 1])
            latest_probs.append(xgb.predict_proba(latest_x)[0][1])
            model_names.append("xgb")
        except Exception:
            xgb = None

    if USE_LGBM and LGBMClassifier is not None:
        try:
            lgbm = LGBMClassifier(
                n_estimators=280, learning_rate=0.04, num_leaves=31,
                subsample=0.85, colsample_bytree=0.85, random_state=42
            )
            lgbm.fit(X_train, y_train)
            val_probs.append(lgbm.predict_proba(X_val)[:, 1])
            test_probs.append(lgbm.predict_proba(X_test)[:, 1])
            latest_probs.append(lgbm.predict_proba(latest_x)[0][1])
            model_names.append("lgbm")
        except Exception:
            lgbm = None

    best_weights, best_thr, best_val_acc = tune_weighted_ensemble(val_probs, y_val, step=0.1)
    ensemble = np.zeros_like(test_probs[0], dtype=float)
    for w, p in zip(best_weights, test_probs):
        ensemble += w * p
    final_pred  = (ensemble > best_thr).astype(int)
    final_acc   = accuracy_score(y_test, final_pred) * 100

    # Predict latest
    prob_up = float(sum(w * p for w, p in zip(best_weights, latest_probs)))

    pred = "UP" if prob_up > best_thr else "DOWN"
    if prob_up > best_thr:
        conf = round(((prob_up - best_thr) / (1 - best_thr + 1e-9)) * 100, 2)
    else:
        conf = round(((best_thr - prob_up) / (best_thr + 1e-9)) * 100, 2)

    # ── Feature importance (from RF) ─────────────────────
    fi = dict(zip(FEATURES, rf.feature_importances_))
    chart_cols = [
        "Open", "High", "Low", "Close", "Volume",
        "SMA_20", "SMA_50", "SMA_200",
        "BB_upper", "BB_lower",
        "RSI_14",
        "STOCH_k", "STOCH_d",
        "MACD", "MACD_sig", "MACD_hist",
        "Vol_SMA20",
        "ADX", "DI_plus", "DI_minus",
    ]
    available_chart_cols = [c for c in chart_cols if c in df.columns]
    chart_df = df[available_chart_cols].tail(520).copy()
    for col in chart_df.select_dtypes(include=["float64"]).columns:
        chart_df[col] = chart_df[col].astype("float32")
    for col in chart_df.select_dtypes(include=["int64"]).columns:
        chart_df[col] = chart_df[col].astype("int32")

    result = {
        "symbol"        : symbol,
        "accuracy"      : round(final_acc, 2),
        "cv_accuracy"   : round(cv_acc, 2),
        "val_accuracy"  : round(best_val_acc * 100, 2),
        "threshold"     : round(best_thr, 2),
        "ensemble_models": model_names,
        "ensemble_weights": [round(float(w), 2) for w in best_weights],
        "prediction"    : pred,
        "confidence"    : conf,
        "prob_up"       : round(prob_up * 100, 2),
        "current_price" : round(float(df["Close"].iloc[-1]), 2),
        "rsi"           : round(float(df["RSI_14"].iloc[-1]), 2),
        "macd"          : round(float(df["MACD"].iloc[-1]), 4),
        "sma_20"        : round(float(df["SMA_20"].iloc[-1]), 2),
        "sma_50"        : round(float(df["SMA_50"].iloc[-1]), 2),
        "sma_200"       : round(float(df["SMA_200"].iloc[-1]), 2),
        "bb_pos"        : round(float(df["BB_pos"].iloc[-1]), 2),
        "bb_squeeze"    : int(df["BB_squeeze"].iloc[-1]),
        "vol_ratio"     : round(float(df["Vol_ratio"].iloc[-1]), 2),
        "atr_pct"       : round(float(df["ATR_pct"].iloc[-1]), 4),
        "adx"           : round(float(df["ADX"].iloc[-1]), 2),
        "stoch_k"       : round(float(df["STOCH_k"].iloc[-1]), 2),
        "willr"         : round(float(df["WILLR"].iloc[-1]), 2),
        "cci"           : round(float(df["CCI"].iloc[-1]), 2),
        "ich_cloud_pos" : int(df["ICH_cloud_pos"].iloc[-1]),
        "obv_slope"     : round(float(df["OBV_slope"].iloc[-1]), 0),
        "df"            : chart_df,
        "feature_importance": fi,
        "trained_at"    : time.strftime("%Y-%m-%d %H:%M"),
        "elapsed_sec"   : round(time.time() - t0, 1),
        "samples_train" : int(len(X_train)),
        "samples_val"   : int(len(X_val)),
        "samples_test"  : int(len(X_test)),
    }

    return (result, rf, et, gb, lr, xgb, lgbm, scaler, best_thr, best_weights, model_names), None


# ════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════

os.makedirs("models", exist_ok=True)
results_path = "models/all_results.pkl"
FORCE_RETRAIN = os.environ.get("FORCE_RETRAIN", "0") == "1"

if os.path.exists(results_path) and not FORCE_RETRAIN:
    with open(results_path, "rb") as f:
        all_results = pickle.load(f)
    print(f"Resuming — {len(all_results)} already done")
else:
    all_results = {}
    if FORCE_RETRAIN:
        print("Starting fresh (FORCE_RETRAIN=1)")
    else:
        print("Starting fresh")

remaining = [s for s in STOCKS if s not in all_results]
total     = len(STOCKS)
done      = len(all_results)
est_hrs   = len(remaining) * 3.5 / 60

print(f"\nTotal   : {total}")
print(f"Done    : {done}")
print(f"Left    : {len(remaining)}")
print(f"Est.    : {est_hrs:.1f} hours")
print(f"Start   : {time.strftime('%H:%M:%S')}")
print(f"Finish  : {time.strftime('%H:%M', time.localtime(time.time() + est_hrs*3600))}")
print("="*60 + "\n")

if not remaining:
    print("All done! Run: python -m streamlit run app.py")
    exit()

t_start  = time.time()
failed   = []
saved_n  = 0

for i, stock in enumerate(remaining, 1):
    idx = done + i
    print(f"[{idx}/{total}] {stock}", end=" ", flush=True)

    out, err = train_stock(stock)

    if err:
        print(f"SKIP — {err}")
        failed.append((stock, err))
        continue

    result, rf, et, gb, lr, xgb, lgbm, scaler, best_thr, best_weights, model_names = out
    all_results[stock] = result

    with open(f"models/{stock.replace('.','_')}.pkl", "wb") as f:
        pickle.dump({
            "rf": rf, "et": et, "gb": gb, "lr": lr, "xgb": xgb, "lgbm": lgbm,
            "scaler": scaler, "features": FEATURES, "threshold": best_thr,
            "ensemble_weights": best_weights, "ensemble_models": model_names
        }, f)

    saved_n += 1
    if saved_n % 10 == 0:
        with open(results_path, "wb") as f:
            pickle.dump(all_results, f)
        print(f"\n  [Saved: {len(all_results)}/{total}]\n")

    elapsed  = (time.time() - t_start) / 60
    avg      = elapsed / i
    eta      = (len(remaining) - i) * avg
    print(f"Acc:{result['accuracy']:.1f}% CV:{result['cv_accuracy']:.1f}% "
          f"{result['prediction']}({result['confidence']:.0f}%) ETA:{eta:.0f}m")

with open(results_path, "wb") as f:
    pickle.dump(all_results, f)

mins = (time.time() - t_start) / 60
print(f"\nDONE! Trained {len(all_results)} in {mins:.0f} mins | Failed: {len(failed)}")
print("Run: python -m streamlit run app.py")



