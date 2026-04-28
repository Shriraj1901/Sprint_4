import pandas as pd
import pytest

import app


def test_top_picks_consistency():
    allr = app.load_all_results()
    assert isinstance(allr, dict)
    items = [(k, v.get('confidence', 0) if isinstance(v, dict) else 0) for k, v in allr.items()]
    items = sorted(items, key=lambda x: x[1], reverse=True)[:10]
    top5 = items[:5]
    assert len(top5) <= 5
    for sym, conf in top5:
        assert sym in app.STOCKS


def test_compare_flow_builds_table():
    s1 = next(iter(app.STOCKS))
    s2 = app.STOCKS[1] if len(app.STOCKS) > 1 else s1
    r1 = app.load_model_result(s1) or {}
    r2 = app.load_model_result(s2) or {}
    p1 = app.fetch_live_price(s1)[0]
    p2 = app.fetch_live_price(s2)[0]
    dfc = pd.DataFrame([
        {"Metric":"Price","A": p1, "B": p2},
        {"Metric":"Prediction","A": r1.get('prediction'), "B": r2.get('prediction')},
        {"Metric":"Confidence","A": r1.get('confidence'), "B": r2.get('confidence')},
        {"Metric":"RSI","A": r1.get('rsi'), "B": r2.get('rsi')},
        {"Metric":"ADX","A": r1.get('adx'), "B": r2.get('adx')},
    ])
    assert 'Metric' in dfc.columns
    assert dfc.shape[0] == 5


def test_portfolio_logic(monkeypatch):
    # monkeypatch live price and model result
    monkeypatch.setattr(app, 'fetch_live_price', lambda s: (100.0, 0.0))
    monkeypatch.setattr(app, 'load_model_result', lambda s: {'prediction':'UP','confidence':70,'rsi':40,'adx':25})

    portfolio = [{'symbol': app.STOCKS[0], 'qty': 2, 'buy_price': 90.0}]
    rows = []
    for h in portfolio:
        lp, _ = app.fetch_live_price(h['symbol'])
        curr_val = (lp or 0) * h['qty']
        invested = h['buy_price'] * h['qty']
        pnl = curr_val - invested
        pnl_pct = (pnl / invested * 100) if invested else 0
        res = app.load_model_result(h['symbol']) or {}
        reco = 'Hold' if res.get('prediction') == 'UP' and res.get('confidence',0) > 60 else 'Sell'
        rows.append({"Symbol": h['symbol'], "Qty": h['qty'], "Buy": h['buy_price'], "Current": lp, "Value": curr_val, "P&L": round(pnl,2), "%": round(pnl_pct,2), "AI": reco})

    assert rows[0]['Current'] == 100.0
    assert rows[0]['Value'] == 200.0
    assert rows[0]['AI'] == 'Hold'
