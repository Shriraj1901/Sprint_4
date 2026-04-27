import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ── 1. Pull data ──────────────────────────────────────────
STOCK = "RELIANCE.NS"

print(f"Downloading {STOCK} data...")
df = yf.download(STOCK, start="2018-01-01", end="2024-01-01", auto_adjust=True)

# ✅ FIX: Flatten MultiIndex columns from yfinance
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
df.dropna(inplace=True)
print(f"Got {len(df)} days of data")

# ── 2. Add technical indicators ───────────────────────────
df['RSI'] = ta.rsi(df['Close'], length=14)

# MACD
macd       = ta.macd(df['Close'])
macd_cols  = macd.columns.tolist()
macd_col   = [c for c in macd_cols if c.startswith('MACD_')][0]
df['MACD'] = macd[macd_col]

# Moving Averages
df['SMA_20'] = ta.sma(df['Close'], length=20)
df['SMA_50'] = ta.sma(df['Close'], length=50)
df['EMA_20'] = ta.ema(df['Close'], length=20)

# Bollinger Bands - auto-detect column names
bb           = ta.bbands(df['Close'], length=20)
bb_cols      = bb.columns.tolist()
upper_col    = [c for c in bb_cols if c.startswith('BBU')][0]
lower_col    = [c for c in bb_cols if c.startswith('BBL')][0]
df['BB_upper'] = bb[upper_col]
df['BB_lower'] = bb[lower_col]

# Time features
df['Day_of_week'] = df.index.dayofweek
df['Month']       = df.index.month

# ── 3. Create target ──────────────────────────────────────
# 1 = price went UP next day, 0 = price went DOWN
df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

# ── 4. Clean data ─────────────────────────────────────────
df.dropna(inplace=True)

features = ['RSI', 'MACD', 'SMA_20', 'SMA_50', 'EMA_20',
            'BB_upper', 'BB_lower', 'Volume',
            'Day_of_week', 'Month']

X = df[features]
y = df['Target']

# ── 5. Train model ────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# ── 6. Results ────────────────────────────────────────────
predictions = model.predict(X_test)
accuracy    = accuracy_score(y_test, predictions)
print(f"\nModel Accuracy: {accuracy*100:.2f}%")

# ── 7. Feature Importance ─────────────────────────────────
importances = pd.Series(model.feature_importances_, index=features)
print("\nFeature Importances:")
print(importances.sort_values(ascending=False).to_string())

# ── 8. Predict tomorrow ───────────────────────────────────
latest     = X.iloc[-1:]
prediction = model.predict(latest)[0]
confidence = model.predict_proba(latest)[0].max() * 100

print(f"\nTomorrow's prediction for {STOCK}:")
print(f"Direction : {'UP ⬆' if prediction == 1 else 'DOWN ⬇'}")
print(f"Confidence: {confidence:.1f}%")

# ── 9. Plot ───────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(14, 12))

# Price + Moving Averages
axes[0].plot(df['Close'],  label='Close Price', color='blue',   linewidth=1)
axes[0].plot(df['SMA_20'], label='SMA 20',      color='orange', linewidth=1)
axes[0].plot(df['SMA_50'], label='SMA 50',      color='green',  linewidth=1)
axes[0].fill_between(df.index, df['BB_upper'], df['BB_lower'],
                     alpha=0.1, color='purple', label='Bollinger Bands')
axes[0].set_title(f'{STOCK} — Price + Moving Averages + Bollinger Bands')
axes[0].legend()

# RSI
axes[1].plot(df['RSI'], label='RSI', color='red', linewidth=1)
axes[1].axhline(70, color='gray', linestyle='--', linewidth=0.8, label='Overbought (70)')
axes[1].axhline(30, color='gray', linestyle=':',  linewidth=0.8, label='Oversold (30)')
axes[1].set_title('RSI')
axes[1].legend()

# MACD
axes[2].plot(df['MACD'], label='MACD', color='darkblue', linewidth=1)
axes[2].axhline(0, color='gray', linestyle='--', linewidth=0.8)
axes[2].set_title('MACD')
axes[2].legend()

plt.tight_layout()
plt.savefig('chart.png')
plt.show()
print("\nChart saved as chart.png")