import math
import numpy as np
import pytest

from src.pricing import black_scholes, straddle_price

def test_known_prices():
    # standard black scholes ex w/ no dividends
    call, put = black_scholes(S=100, K=100, T=1, r=0.05, q=0, sigma=0.20)
    assert call == pytest.approx(10.4506, abs=0.0001)
    assert put == pytest.approx(5.5735, abs=0.0001)

def test_put_call_parity():
    S, K, T = 100, 105, 0.5
    r, q, sigma = 0.04, 0.02, 0.25

    call, put = black_scholes(S, K, T, r, q, sigma)

    expected = S * math.exp(-q * T) - K * math.exp(-r * T)
    assert call - put == pytest.approx(expected)

def test_higher_volatility_increases_straddle_cost():
    low_call, low_put = black_scholes(100, 100, 0.1, 0.04, 0.02, 0.15)
    high_call, high_put = black_scholes(100, 100, 0.1, 0.04, 0.02, 0.30)

    assert high_call + high_put > low_call + low_put


def test_bad_inputs_raise():
    with pytest.raises(ValueError):
        black_scholes(100, 100, 0, 0.04, 0.02, 0.2) # T = 0
    with pytest.raises(ValueError):
        black_scholes(100, 100, 0.1, 0.04, 0.02, -0.2)   # sigma < 0

def test_works_on_arrays():
    S = np.array([90.0, 100.0, 110.0])
    call, put = black_scholes(S, 100, 0.25, 0.03, 0.01, 0.2)
    assert call.shape == (3,) and put.shape == (3,)
    assert straddle_price(S, 100, 0.25, 0.03, 0.01, 0.2).shape == (3,)

def test_risk_neutral_simulation():
    # if simulating prices under risk neutral measure --> discounted avg payoff = forumla price
    S, K, T, r, q, sig = 100, 100, 1 / 12, 0.03, 0.015, 0.20
    z = np.random.default_rng(1).standard_normal(400_000)
    ST = S * np.exp((r - q - 0.5 * sig**2) * T + sig * math.sqrt(T) * z)
    pay = math.exp(-r * T) * np.abs(ST - K)
    se = pay.std() / math.sqrt(len(pay))
    assert abs(pay.mean() - straddle_price(S, K, T, r, q, sig)) < 4 * se