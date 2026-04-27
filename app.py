import os, re, time, pickle, warnings, html
import numpy as np
import pandas as pd
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf
import streamlit as st
from datetime import datetime

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

st.set_page_config(page_title="NSE Stock AI", page_icon="📈", layout="wide")

# ── GROQ API KEY — paste your new key here ────────────────
GROQ_API_KEY = load_groq_key()

st.markdown("""
<style>
.sig-green {
    background:#dcfce7 !important; border-left:4px solid #16a34a;
    padding:10px 14px; border-radius:6px; margin:5px 0;
    font-size:13px; color:#166534 !important; font-weight:500;
}
.sig-red {
    background:#fee2e2 !important; border-left:4px solid #dc2626;
    padding:10px 14px; border-radius:6px; margin:5px 0;
    font-size:13px; color:#991b1b !important; font-weight:500;
}
.sig-yellow {
    background:#fef3c7 !important; border-left:4px solid #d97706;
    padding:10px 14px; border-radius:6px; margin:5px 0;
    font-size:13px; color:#92400e !important; font-weight:500;
}
.sig-gray {
    background:#f3f4f6 !important; border-left:4px solid #9ca3af;
    padding:10px 14px; border-radius:6px; margin:5px 0;
    font-size:13px; color:#374151 !important; font-weight:500;
}
.verdict-green {
    background:#dcfce7 !important; color:#166534 !important;
    padding:18px; border-radius:12px; font-size:24px; font-weight:700;
    text-align:center; margin:10px 0; border:2px solid #16a34a;
}
.verdict-orange {
    background:#fef3c7 !important; color:#92400e !important;
    padding:18px; border-radius:12px; font-size:24px; font-weight:700;
    text-align:center; margin:10px 0; border:2px solid #d97706;
}
.verdict-red {
    background:#fee2e2 !important; color:#991b1b !important;
    padding:18px; border-radius:12px; font-size:24px; font-weight:700;
    text-align:center; margin:10px 0; border:2px solid #dc2626;
}
.ind-card {
    background:#f8fafc !important; border:1.5px solid #e2e8f0;
    border-radius:10px; padding:12px 8px; text-align:center; margin:4px;
}
.ind-val { font-size:18px; font-weight:700; color:#1e293b !important; }
.ind-lbl { font-size:11px; color:#64748b !important; margin-top:2px; }
.chat-user {
    background:#dbeafe !important; color:#1e3a5f !important;
    padding:12px 16px; border-radius:12px 12px 4px 12px;
    margin:8px 0; font-size:14px; font-weight:500;
    display:block; width:fit-content; margin-left:auto;
    max-width:85%;
}
.chat-ai {
    background:#f0fdf4 !important; color:#14532d !important;
    padding:12px 16px; border-radius:12px 12px 12px 4px;
    margin:8px 0; font-size:14px; font-weight:500;
    display:block; max-width:90%;
    border-left:3px solid #16a34a;
}
.chat-box {
    max-height:450px; overflow-y:auto; padding:12px;
    background:#ffffff !important; border-radius:12px;
    border:1.5px solid #e5e7eb; margin-bottom:12px;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────
def sg(r, key, default=0):
    val = r.get(key, default)
    if val is None:
        return default
    try:
        if isinstance(val, float) and np.isnan(val):
            return default
    except:
        pass
    return val


# ── Load data ─────────────────────────────────────────────
@st.cache_resource
def load_results():
    p = "models/all_results.pkl"
    if not os.path.exists(p):
        return None
    with open(p, "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_stock_map():
    if os.path.exists("nse_stocks.pkl"):
        with open("nse_stocks.pkl", "rb") as f:
            return pickle.load(f)
    return {}

@st.cache_data(ttl=30)
def get_price(symbol):
    sym = symbol.replace(".NS", "")
    # Try NSE first
    try:
        s = requests.Session()
        h = {"User-Agent": "Mozilla/5.0", "Accept": "application/json",
             "Referer": "https://www.nseindia.com/"}
        s.get("https://www.nseindia.com", headers=h, timeout=6)
        time.sleep(0.3)
        r  = s.get(f"https://www.nseindia.com/api/quote-equity?symbol={sym}",
                   headers=h, timeout=6)
        pi = r.json()["priceInfo"]
        return {"price": pi["lastPrice"], "change": round(pi["change"], 2),
                "pct": round(pi["pChange"], 2), "source": "NSE Live"}
    except:
        pass
    # Fallback Yahoo
    try:
        t    = yf.Ticker(symbol)
        hist = t.history(period="5d", interval="1d")
        if not hist.empty:
            p    = float(hist["Close"].iloc[-1])
            prev = float(hist["Close"].iloc[-2]) if len(hist) > 1 else p
            chg  = p - prev
            return {"price": round(p,2), "change": round(chg,2),
                    "pct": round((chg/prev)*100, 2), "source": "Yahoo"}
    except:
        pass
    return None


# ── Signals ───────────────────────────────────────────────
def get_signals(r, price):
    s      = []
    rsi    = sg(r, "rsi",           50)
    macd   = sg(r, "macd",          0)
    sma20  = sg(r, "sma_20",        price)
    sma50  = sg(r, "sma_50",        price)
    sma200 = sg(r, "sma_200",       price * 0.9)
    adx    = sg(r, "adx",           20)
    vol    = sg(r, "vol_ratio",     1)
    bb     = sg(r, "bb_pos",        0.5)
    sk     = sg(r, "stoch_k",       50)
    ich    = sg(r, "ich_cloud_pos", 0)

    # RSI
    if rsi < 25:
        s.append(("green",  f"RSI {rsi:.0f} — Heavily OVERSOLD (strong buy zone)"))
    elif rsi < 40:
        s.append(("green",  f"RSI {rsi:.0f} — Oversold (bullish bias)"))
    elif rsi > 75:
        s.append(("red",    f"RSI {rsi:.0f} — Heavily OVERBOUGHT (sell zone)"))
    elif rsi > 60:
        s.append(("yellow", f"RSI {rsi:.0f} — Overbought (watch for reversal)"))
    else:
        s.append(("gray",   f"RSI {rsi:.0f} — Neutral zone"))

    s.append(("green" if macd > 0 else "red",
              f"MACD {'Positive — bullish momentum' if macd > 0 else 'Negative — bearish momentum'}"))

    s.append(("green" if price > sma20 else "red",
              f"Price {'above' if price > sma20 else 'below'} SMA20 Rs.{sma20:,.2f}"))

    s.append(("green" if price > sma50 else "red",
              f"Price {'above' if price > sma50 else 'below'} SMA50 Rs.{sma50:,.2f}"))

    s.append(("green" if price > sma200 else "red",
              f"{'Bull market' if price > sma200 else 'Bear market'} — vs SMA200 Rs.{sma200:,.2f}"))

    if adx > 40:
        s.append(("green",  f"ADX {adx:.0f} — Very strong trend"))
    elif adx > 25:
        s.append(("green",  f"ADX {adx:.0f} — Trending market"))
    else:
        s.append(("yellow", f"ADX {adx:.0f} — Weak/ranging market"))

    if vol > 2.0:
        s.append(("green",  f"Volume {vol:.1f}x — Very high interest"))
    elif vol > 1.3:
        s.append(("green",  f"Volume {vol:.1f}x — Above average"))
    elif vol < 0.6:
        s.append(("yellow", f"Volume {vol:.1f}x — Low interest"))
    else:
        s.append(("gray",   f"Volume {vol:.1f}x — Normal"))

    if bb < 0.1:
        s.append(("green",  "Near lower Bollinger Band — bounce zone"))
    elif bb > 0.9:
        s.append(("red",    "Near upper Bollinger Band — resistance zone"))
    else:
        s.append(("gray",   f"Bollinger position: {bb:.2f}"))

    if sk < 20:
        s.append(("green",  f"Stochastic {sk:.0f} — Oversold"))
    elif sk > 80:
        s.append(("red",    f"Stochastic {sk:.0f} — Overbought"))
    else:
        s.append(("gray",   f"Stochastic {sk:.0f} — Neutral"))

    s.append(("green" if ich else "red",
              f"Price {'above' if ich else 'below'} Ichimoku cloud — {'bullish' if ich else 'bearish'}"))

    return s


def get_verdict(r, signals):
    pred  = r.get("prediction", "DOWN")
    score = 3 if pred == "UP" else 0
    score += 1 if sg(r, "confidence", 50) > 63 else 0
    score += sum(1 for x in signals if x[0] == "green")
    score -= sum(1 for x in signals if x[0] == "red")
    score += 1 if sg(r, "adx", 20) > 25 else 0
    score += 1 if sg(r, "ich_cloud_pos", 0) else -1

    if score >= 8:   return "STRONG BUY",      "green"
    elif score >= 5: return "CONSIDER BUYING", "green"
    elif score >= 2: return "NEUTRAL — WAIT",  "orange"
    elif score >= 0: return "WEAK — AVOID",    "orange"
    else:            return "SELL / SHORT",    "red"


# ── Build context for AI ──────────────────────────────────
def build_context(r, price, company, signals, advice):
    pred   = r.get("prediction", "N/A")
    conf   = sg(r, "confidence",    50)
    acc    = sg(r, "accuracy",      60)
    rsi    = sg(r, "rsi",           50)
    macd   = sg(r, "macd",          0)
    adx    = sg(r, "adx",           20)
    sk     = sg(r, "stoch_k",       50)
    cci    = sg(r, "cci",           0)
    wr     = sg(r, "willr",         -50)
    bb     = sg(r, "bb_pos",        0.5)
    vol    = sg(r, "vol_ratio",     1)
    ich    = sg(r, "ich_cloud_pos", 0)
    sma20  = sg(r, "sma_20",        price)
    sma50  = sg(r, "sma_50",        price)
    sma200 = sg(r, "sma_200",       price)
    atr    = sg(r, "atr_pct",       0.02) * price
    t1     = round(price + atr * 1.5, 2)
    t2     = round(price + atr * 3.0, 2)
    sl     = round(price - atr * 1.5, 2)
    sig_text = "\n".join([f"- {msg}" for _, msg in signals])

    return f"""You are an expert Indian NSE stock market analyst AI.
You have LIVE analysis data for {company}. Always use this data in your answers.

STOCK: {company}
Live Price: Rs.{price:,.2f}
AI Prediction (3-day): {pred}
Confidence: {conf}%
Model Accuracy: {acc}%
Verdict: {advice}

TECHNICAL INDICATORS:
RSI(14): {rsi:.1f} | MACD: {macd:.4f} | ADX: {adx:.1f}
Stochastic %K: {sk:.1f} | CCI: {cci:.1f} | Williams %R: {wr:.1f}
Bollinger Position: {bb:.2f} | Volume Ratio: {vol:.2f}x
Ichimoku: {'Above cloud (bullish)' if ich else 'Below cloud (bearish)'}

MOVING AVERAGES:
SMA20: Rs.{sma20:,.2f} ({'above' if price > sma20 else 'below'})
SMA50: Rs.{sma50:,.2f} ({'above' if price > sma50 else 'below'})
SMA200: Rs.{sma200:,.2f} ({'bull market' if price > sma200 else 'bear market'})

PRICE TARGETS (ATR-based):
Target 1: Rs.{t1:,.2f} | Target 2: Rs.{t2:,.2f} | Stop Loss: Rs.{sl:,.2f}

SIGNALS:
{sig_text}

RULES:
- Always give specific numbers from the data above
- Be conversational and concise (3-5 sentences)
- Always mention stop loss when discussing buying
- Be honest if signals are mixed
- Do NOT say you lack real-time data — you have it above
- Use Rs. for prices"""


# ── Groq chat function ────────────────────────────────────
def ask_groq(messages, context):
    if GROQ_API_KEY == "paste_your_new_groq_key_here" or not GROQ_API_KEY:
        return ("No API key set. Open app.py and paste your Groq API key "
                "at the top where it says GROQ_API_KEY = '...'")
    try:
        payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": [{"role": "system", "content": context}] + messages,
            "max_tokens": 800,
            "temperature": 0.6
        }
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json=payload,
            timeout=30
        )
        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"]
        elif resp.status_code == 401:
            return "Invalid API key. Go to console.groq.com and get a new free key."
        elif resp.status_code == 429:
            return "Rate limit hit. Wait 30 seconds and try again (free tier limit)."
        else:
            return f"Groq API error {resp.status_code}: {resp.text[:200]}"
    except requests.exceptions.Timeout:
        return "Request timed out. Check your internet and try again."
    except Exception as e:
        return f"Error: {str(e)}"


# ════════════════════════════════════════════════════════
# MAIN APP
# ════════════════════════════════════════════════════════

all_results = load_results()
stock_map   = load_stock_map()

if all_results is None:
    st.error("No trained models found.")
    st.code("python train.py", language="bash")
    st.stop()

trained = sorted(all_results.keys())

def display(sym):
    c  = sym.replace(".NS", "")
    sm = stock_map.get(c)
    n  = sm.get("name", "") if isinstance(sm, dict) else ""
    return f"{n} ({c})" if n else c

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.header("📈 NSE Stock AI")
    search = st.text_input("Search", placeholder="e.g. Reliance or TCS")

    if search:
        sl       = search.lower()
        filtered = [s for s in trained
                    if sl in s.lower() or
                    sl in (stock_map.get(s.replace(".NS",""), {}) or {}).get("name","").lower()]
    else:
        filtered = trained

    if not filtered:
        st.warning("No stock found")
        st.stop()

    selected = st.selectbox("Stock", filtered, format_func=display)
    st.markdown("---")
    st.metric("Stocks trained", len(all_results))
    st.markdown("**Model:** RF + ET + GBM + LR")
    st.markdown("**Features:** 65 indicators")
    st.markdown("**Chat:** Groq LLaMA 70B (Free)")

    st.markdown("---")
    if st.button("🔄 Refresh Price", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.api_messages = []
        st.rerun()

    st.markdown("---")
    st.caption("Educational use only. Not financial advice.")

# ── Load result ───────────────────────────────────────────
r       = all_results[selected]
clean   = selected.replace(".NS", "")
sm_val  = stock_map.get(clean)
company = (sm_val.get("name", "") if isinstance(sm_val, dict) else "") or clean

pdata  = get_price(selected)
price  = pdata["price"]  if pdata else sg(r, "current_price", 0)
change = pdata["change"] if pdata else 0
pct    = pdata["pct"]    if pdata else 0
src    = pdata["source"] if pdata else "Cached"

# Reset chat on stock change
if st.session_state.get("last_stock") != selected:
    st.session_state.chat_history = []
    st.session_state.api_messages = []
    st.session_state.last_stock   = selected

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "api_messages" not in st.session_state:
    st.session_state.api_messages = []

signals    = get_signals(r, price)
advice, vc = get_verdict(r, signals)

# ── Header ────────────────────────────────────────────────
h1, h2 = st.columns([3, 1])
with h1:
    st.markdown(f"## {company}")
    badge = "🟢" if change >= 0 else "🔴"
    st.markdown(f"**NSE: {clean}** | {badge} {src} | *{datetime.now().strftime('%d %b %Y %H:%M:%S')}*")
with h2:
    st.caption("Press Refresh for latest price")

st.markdown("---")

# ── Metrics ───────────────────────────────────────────────
pred = r.get("prediction", "N/A")
conf = sg(r, "confidence",  50)
acc  = sg(r, "accuracy",    60)
cv   = sg(r, "cv_accuracy", 60)
adx  = sg(r, "adx",         20)
prob = sg(r, "prob_up",     50)

m1,m2,m3,m4,m5,m6 = st.columns(6)
arrow = "▲" if change >= 0 else "▼"
m1.metric("Live Price",    f"Rs.{price:,.2f}", f"{arrow} {abs(pct):.2f}%")
m2.metric("AI Prediction", pred,               f"Prob UP: {prob}%")
m3.metric("Confidence",    f"{conf}%")
m4.metric("Test Accuracy", f"{acc}%")
m5.metric("CV Accuracy",   f"{cv}%")
m6.metric("ADX (Trend)",   f"{adx:.0f}",       "Strong" if adx > 25 else "Weak")

st.markdown("---")

# ── Verdict ───────────────────────────────────────────────
st.markdown(f'<div class="verdict-{vc}">{advice}</div>', unsafe_allow_html=True)

# ── Indicators ────────────────────────────────────────────
st.markdown("### 📊 Key Indicators")
i1,i2,i3,i4,i5,i6,i7,i8 = st.columns(8)

def ind(col, val, lbl):
    col.markdown(
        f'<div class="ind-card"><div class="ind-val">{val}</div>'
        f'<div class="ind-lbl">{lbl}</div></div>',
        unsafe_allow_html=True
    )

rsi = sg(r,"rsi",50); sk=sg(r,"stoch_k",50); cci=sg(r,"cci",0)
wr  = sg(r,"willr",-50); bb=sg(r,"bb_pos",0.5)
vol = sg(r,"vol_ratio",1); ich=sg(r,"ich_cloud_pos",0)

ind(i1, f"{rsi:.0f}",                  "RSI (14)")
ind(i2, f"{sk:.0f}",                   "Stoch %K")
ind(i3, f"{adx:.0f}",                  "ADX")
ind(i4, f"{cci:.0f}",                  "CCI")
ind(i5, f"{wr:.0f}",                   "Williams %R")
ind(i6, f"{bb:.2f}",                   "BB Position")
ind(i7, f"{vol:.1f}x",                 "Volume")
ind(i8, "Above ☁" if ich else "Below ☁","Ichimoku")

st.markdown("---")

# ── Signals ───────────────────────────────────────────────
st.markdown("### 📡 Signal Breakdown")
c1, c2 = st.columns(2)
for i, (color, msg) in enumerate(signals):
    tgt = c1 if i % 2 == 0 else c2
    safe = html.escape(msg)
    tgt.markdown(f'<div class="sig-{color}">{safe}</div>', unsafe_allow_html=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════
# AI CHAT — shown BEFORE charts so you see it without scroll
# ══════════════════════════════════════════════════════════
st.markdown("### 🤖 Chat with Stock AI")
st.caption(f"Powered by **Groq LLaMA 70B** (free) | Knows all data for **{company}**")

# API key warning
if GROQ_API_KEY == "paste_your_new_groq_key_here":
    st.warning("⚠️ Groq API key not set. Open app.py and paste your key at the top.")

context = build_context(r, price, company, signals, advice)

# Chat history display
if st.session_state.chat_history:
    chat_html = '<div class="chat-box">'
    for msg in st.session_state.chat_history:
        safe_text = html.escape(str(msg.get("content", "")))
        if msg["role"] == "user":
            chat_html += f'<div class="chat-user">You: {safe_text}</div>'
        else:
            content = safe_text.replace("&#10;", "<br>").replace("\n", "<br>")
            chat_html += f'<div class="chat-ai">AI: {content}</div>'
    chat_html += "</div>"
    st.markdown(chat_html, unsafe_allow_html=True)

# Quick questions
st.markdown("**Quick questions:**")
qcols = st.columns(4)
quick = [
    "Should I buy this stock now?",
    "What is the price target?",
    "What is the risk level?",
    "Explain all signals",
]
clicked_q = None
for i, qq in enumerate(quick):
    if qcols[i].button(qq, use_container_width=True, key=f"qq_{i}"):
        clicked_q = qq

# Text input
with st.form("chat_form", clear_on_submit=True):
    col_inp, col_btn = st.columns([5, 1])
    with col_inp:
        user_input = st.text_input(
            "msg", placeholder="Ask anything about this stock...",
            label_visibility="collapsed"
        )
    with col_btn:
        send = st.form_submit_button("Send 📨", use_container_width=True)

# Process message
question = None
if send and user_input.strip():
    question = user_input.strip()
elif clicked_q:
    question = clicked_q

if question:
    st.session_state.chat_history.append({"role": "user", "content": question})
    st.session_state.api_messages.append({"role": "user", "content": question})

    with st.spinner("AI thinking..."):
        reply = ask_groq(st.session_state.api_messages, context)

    st.session_state.chat_history.append({"role": "assistant", "content": reply})
    st.session_state.api_messages.append({"role": "assistant", "content": reply})
    st.rerun()

st.markdown("---")

# ── Charts ────────────────────────────────────────────────
st.markdown("### 📈 Charts")
df_c = r.get("df")

if df_c is not None and not df_c.empty:
    tab1,tab2,tab3,tab4,tab5 = st.tabs(["Price + MAs","RSI + Stoch","MACD","Volume","ADX"])

    with tab1:
        fig, ax = plt.subplots(figsize=(13, 4))
        ax.plot(df_c.index, df_c["Close"], label="Close", lw=1.2, color="#2563eb")
        for col,lbl,clr in [("SMA_20","SMA 20","#f59e0b"),
                              ("SMA_50","SMA 50","#10b981"),
                              ("SMA_200","SMA 200","#ef4444")]:
            if col in df_c.columns:
                ax.plot(df_c.index, df_c[col], label=lbl, lw=0.8, color=clr, ls="--")
        if "BB_upper" in df_c.columns and "BB_lower" in df_c.columns:
            ax.fill_between(df_c.index, df_c["BB_upper"], df_c["BB_lower"],
                            alpha=0.07, color="#6366f1", label="Bollinger Bands")
        ax.set_title(f"{company} — Price + MAs + Bollinger Bands", fontsize=12)
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.2)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        fig.tight_layout(); st.pyplot(fig); plt.close()

    with tab2:
        fig,(ax1,ax2) = plt.subplots(2,1,figsize=(13,5),sharex=True)
        if "RSI_14" in df_c.columns:
            ax1.plot(df_c.index, df_c["RSI_14"], color="#dc2626", lw=1)
            ax1.axhline(70, color="red",   ls="--", lw=0.8, label="Overbought 70")
            ax1.axhline(30, color="green", ls="--", lw=0.8, label="Oversold 30")
            ax1.fill_between(df_c.index, df_c["RSI_14"], 70,
                             where=df_c["RSI_14"]>=70, alpha=0.2, color="red")
            ax1.fill_between(df_c.index, df_c["RSI_14"], 30,
                             where=df_c["RSI_14"]<=30, alpha=0.2, color="green")
            ax1.legend(fontsize=8)
        ax1.set_title("RSI (14)"); ax1.set_ylim(0,100); ax1.grid(True,alpha=0.2)

        sk_col = next((c for c in ["STOCH_k","STOCHk_14_3_3"] if c in df_c.columns), None)
        sd_col = next((c for c in ["STOCH_d","STOCHd_14_3_3"] if c in df_c.columns), None)
        if sk_col: ax2.plot(df_c.index, df_c[sk_col], color="#7c3aed", lw=1, label="Stoch %K")
        if sd_col: ax2.plot(df_c.index, df_c[sd_col], color="#db2777", lw=0.8, ls="--", label="Stoch %D")
        ax2.axhline(80, color="red",   ls="--", lw=0.8)
        ax2.axhline(20, color="green", ls="--", lw=0.8)
        if sk_col or sd_col: ax2.legend(fontsize=8)
        ax2.set_title("Stochastic"); ax2.set_ylim(0,100); ax2.grid(True,alpha=0.2)
        fig.tight_layout(); st.pyplot(fig); plt.close()

    with tab3:
        fig, ax = plt.subplots(figsize=(13,3))
        if "MACD" in df_c.columns:
            ax.plot(df_c.index, df_c["MACD"], label="MACD", color="#2563eb", lw=1)
        if "MACD_sig" in df_c.columns:
            ax.plot(df_c.index, df_c["MACD_sig"], label="Signal", color="#f59e0b", lw=1)
        if "MACD_hist" in df_c.columns:
            clrs = df_c["MACD_hist"].apply(lambda x: "#16a34a" if x>0 else "#dc2626")
            ax.bar(df_c.index, df_c["MACD_hist"], color=clrs, alpha=0.5, width=1)
        ax.axhline(0, color="gray", lw=0.5)
        ax.set_title("MACD"); ax.legend(fontsize=8); ax.grid(True,alpha=0.2)
        fig.tight_layout(); st.pyplot(fig); plt.close()

    with tab4:
        fig, ax = plt.subplots(figsize=(13,3))
        if "Volume" in df_c.columns and "Open" in df_c.columns:
            vc_list = ["#16a34a" if c>=o else "#dc2626"
                       for c,o in zip(df_c["Close"],df_c["Open"])]
            ax.bar(df_c.index, df_c["Volume"], color=vc_list, alpha=0.6, width=1)
        if "Vol_SMA20" in df_c.columns:
            ax.plot(df_c.index, df_c["Vol_SMA20"], color="#2563eb", lw=1, label="Vol SMA20")
            ax.legend(fontsize=8)
        ax.set_title("Volume"); ax.grid(True,alpha=0.2)
        fig.tight_layout(); st.pyplot(fig); plt.close()

    with tab5:
        fig, ax = plt.subplots(figsize=(13,3))
        if "ADX"      in df_c.columns: ax.plot(df_c.index, df_c["ADX"],      label="ADX", color="#7c3aed", lw=1.2)
        if "DI_plus"  in df_c.columns: ax.plot(df_c.index, df_c["DI_plus"],  label="+DI", color="#16a34a", lw=0.8)
        if "DI_minus" in df_c.columns: ax.plot(df_c.index, df_c["DI_minus"], label="-DI", color="#dc2626", lw=0.8)
        ax.axhline(25, color="gray", ls="--", lw=0.8, label="Trend threshold")
        ax.set_title("ADX — Trend Strength"); ax.legend(fontsize=8); ax.grid(True,alpha=0.2)
        fig.tight_layout(); st.pyplot(fig); plt.close()

fi = r.get("feature_importance", {})
if fi:
    with st.expander("🔬 Feature Importance"):
        top = sorted(fi.items(), key=lambda x: x[1], reverse=True)[:15]
        fig, ax = plt.subplots(figsize=(10,5))
        ax.barh([x[0] for x in top[::-1]], [x[1] for x in top[::-1]],
                color="#6366f1", alpha=0.8)
        ax.set_title("Top 15 Features by Importance")
        ax.grid(True, alpha=0.3, axis="x")
        fig.tight_layout(); st.pyplot(fig); plt.close()

st.markdown("---")
st.caption(f"Trained: {r.get('trained_at','N/A')} | "
           f"Time: {sg(r,'elapsed_sec',0):.0f}s | "
           f"Features: {len(fi) if fi else 'N/A'} | "
           f"Chat: Groq LLaMA 70B (Free)")