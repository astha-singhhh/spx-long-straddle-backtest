from pathlib import Path

import pandas as pd
from src.data import check_inputs

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

market = (pd.read_csv(RAW / "spx_vix_daily.csv", parse_dates=["date"]).dropna(subset=["spx_close"]).sort_values("date"))
rates = (pd.read_csv(RAW / "treasury_1m_daily.csv", parse_dates=["date"]).dropna().sort_values("date"))
divs = (pd.read_csv(RAW / "spy_dividend_proxy.csv", parse_dates=["date"])[["date", "dividend_yield_proxy"]].dropna().sort_values("date"))

def attach_lagged(left, right, tolerance_days=10):
    return pd.merge_asof(
        left, right, on="date", direction="backward",
        allow_exact_matches=False,
        tolerance=pd.Timedelta(days=tolerance_days),
    )

data = attach_lagged(attach_lagged(market, rates), divs)

out = pd.DataFrame({
    "date": data["date"],
    "S": data["spx_close"],
    "iv": data["vix_close"] / 100,  # VIX points -> decimal
    "r": data["treasury_1m_pct"] / 100,  # % -> decimal
    "q": data["dividend_yield_proxy"],  # alr a decimal
})

check_inputs(out)       
out.to_csv(PROCESSED / "inputs.csv", index=False)

print(out.describe().T)
print(f"\nSaved {len(out)} rows, {out.date.min().date()} to {out.date.max().date()}")

