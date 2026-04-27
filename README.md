# NSE Stock AI 📈

AI-powered Indian stock market prediction and analysis tool.

## Features
- Trained on 1794 NSE stocks (10 years of data)
- 65 technical indicators
- 4-model ensemble (RF + ExtraTrees + GBM + LR)
- Live price from NSE India
- AI chat powered by Groq LLaMA 70B (free)
- Interactive charts with RSI, MACD, ADX, Stochastic

## Setup

### 1. Install dependencies
pip install -r requirements.txt

### 2. Get stock list
python stocks_list.py

### 3. Train models (runs overnight)
python train.py

### 4. Run dashboard
python -m streamlit run app.py

## Tech Stack
- Python, Scikit-learn, Pandas, yfinance
- Streamlit dashboard
- Groq API (free LLaMA 70B) for AI chat
- NSE India for live prices

## Note
Models are not included in this repo (too large).
Run train.py to generate them locally.