import pandas as pd

'''
date: SPX/VIX file
S: SPX close
iv: VIX close 
r: 1-month treasury yield 
q: SPY dividend yield proxy
'''
COLUMNS = ["date", "S", "iv", "r", "q"]

def check_inputs(df):
    problems = []
    if list(df.columns) != COLUMNS:
        problems.append(f"columns must be {COLUMNS}")
    if df.isna().any().any():
        problems.append("missing values")
    if not df["date"].is_monotonic_increasing:
        problems.append("dates not sorted")
    if df["date"].duplicated().any():
        problems.append("duplicate dates")
    if not df["S"].gt(0).all():
        problems.append("non-positive price")
    if not df["iv"].between(0.05, 1.5).all():
        problems.append("iv outside 5%-150% (percent instead of decimal?)")
    if not df["r"].between(-0.01, 0.10).all():
        problems.append("r outside -1%..10%")
    if not df["q"].between(0.0, 0.05).all():
        problems.append("q outside 0%..5%")
    if (df["date"].diff().dt.days > 7).any():
        problems.append("gap of more than 7 calendar days")
    if problems:
        raise ValueError("; ".join(problems))

def load_inputs(path):
    df = pd.read_csv(path, parse_dates=["date"]).sort_values("date")
    df = df.reset_index(drop=True)
    check_inputs(df)
    return df