import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

# 10 yrs
START, END = "2016-11-01", "2026-11-01"
DIV_START = "2015-10-01"   # extra year so trailing 12-month dividends are complete
FRED_URL = ("https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS1MO&cosd=2016-10-01&coed=2026-10-31")

FILES = ["spx_vix_daily.csv", "spy_dividend_proxy.csv", "treasury_1m_daily.csv"]
if "--refresh" not in sys.argv and all((RAW / f).exists() for f in FILES):
    sys.exit("Raw data already exists. Use --refresh to download again.")


def yahoo_history(ticker, start, end):
    h = yf.Ticker(ticker).history(
        start=start, end=end, auto_adjust=False, actions=True)
    if h.empty:
        raise RuntimeError(f"No data returned for {ticker}")
    h.index = h.index.tz_localize(None).normalize()
    h.index.name = "date"
    return h


# SPX & VIX
spx = yahoo_history("^GSPC", START, END)["Close"].rename("spx_close")
vix = yahoo_history("^VIX", START, END)["Close"].rename("vix_close")
market = pd.concat([spx, vix], axis=1, sort=True)
market.to_csv(RAW / "spx_vix_daily.csv")

# SPY dividend yield proxy
spy = yahoo_history("SPY", DIV_START, END)
ttm = spy["Dividends"].rolling("365D").sum()
proxy = ttm / spy["Close"]
proxy[spy.index < spy.index[0] + pd.Timedelta(days=365)] = np.nan  # partial window
pd.DataFrame({
    "spy_close": spy["Close"],
    "spy_dividends": spy["Dividends"],
    "dividend_yield_proxy": proxy,
}).to_csv(RAW / "spy_dividend_proxy.csv")

# FRED 1-month treasury yield from FRED 
rates = pd.read_csv(FRED_URL, na_values=".")
rates.columns = ["date", "treasury_1m_pct"]      
rates["date"] = pd.to_datetime(rates["date"])
rates["treasury_1m_pct"] = pd.to_numeric(rates["treasury_1m_pct"], errors="coerce")
rates.to_csv(RAW / "treasury_1m_daily.csv", index=False)

for f in FILES:
    df = pd.read_csv(RAW / f)
    print(f"{f}: {df.shape[0]} rows, {df.iloc[0, 0]} to {df.iloc[-1, 0]}")