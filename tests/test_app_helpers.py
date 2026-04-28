import pandas as pd
import pytest

import app


def test_load_all_results():
    allr = app.load_all_results()
    assert isinstance(allr, dict)
    assert len(allr) > 0


def test_load_model_result():
    # pick a known stock from STOCKS
    symbol = next(iter(app.STOCKS))
    res = app.load_model_result(symbol)
    assert res is not None


def test_semicircle_gauge():
    fig = app.semicircle_gauge(60)
    assert hasattr(fig, 'data')
    assert fig.data[0]['type'] == 'indicator'
    assert fig.data[0]['value'] == 60


def test_fetch_live_price_monkeypatch(monkeypatch):
    # create a tiny history DataFrame with two Close values
    df = pd.DataFrame({'Close': [100.0, 101.5]}, index=pd.date_range('2026-01-01', periods=2, freq='min'))

    class Ticker:
        def history(self, period="2d", interval="1m"):
            return df

    class YF:
        def Ticker(self, symbol):
            return Ticker()

        def download(self, symbol, period='5d', progress=False, auto_adjust=True):
            return df

    # monkeypatch app.yf to avoid network
    monkeypatch.setattr(app, 'yf', YF())

    price, delta = app.fetch_live_price('RELIANCE.NS')
    assert price == 101.5
    assert round(delta, 2) == round(1.5, 2)
