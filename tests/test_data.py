import pandas as pd
import pytest

from src.data import check_inputs

def good():
    return pd.DataFrame({
        "date": pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"]),
        "S": [3200.0, 3210.0, 3225.0],
        "iv": [0.14, 0.13, 0.15],
        "r": [0.015] * 3,
        "q": [0.018] * 3,
    })

def test_good_data_passes():
    check_inputs(good())

def test_percent_iv_is_caught():
    df = good()
    df["iv"] = df["iv"] * 100
    with pytest.raises(ValueError, match="iv outside"):
        check_inputs(df)

def test_missing_values_are_caught():
    df = good()
    df.loc[1, "r"] = None
    with pytest.raises(ValueError, match="missing"):
        check_inputs(df)