import numpy as np
from scipy.stats import norm


'''
return euro call & put prices per index point
S: index level
K: strike
T: time to maturity
r: continuously compounded interest rate
q: dividend yield
sigma: volatility
'''
def black_scholes(S, K, T, r, q, sigma):
    S, K, T, r, q, sigma = (np.asarray(x, dtype=float) for x in (S, K, T, r, q, sigma))
    if np.any(S <= 0) or np.any(K <= 0):
        raise ValueError("index level and strike must be pos")
    if np.any(T <= 0) or np.any(sigma <= 0):
        raise ValueError("time and volatility must be pos")

    sd = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / sd
    d2 = d1 - sd

    discounted_spot = S * np.exp(-q * T)
    discounted_strike = K * np.exp(-r * T)

    call = discounted_spot * norm.cdf(d1) - discounted_strike * norm.cdf(d2)
    put = discounted_strike * norm.cdf(-d2) - discounted_spot * norm.cdf(-d1)

    if call.ndim == 0:                 
        return float(call), float(put)
    return call, put


def straddle_price(S, K, T, r, q, sigma):
    call, put = black_scholes(S, K, T, r, q, sigma)
    return call + put


if __name__ == "__main__":
    call, put = black_scholes(
        S=100, K=100, T=30 / 365,
        r=0.04, q=0.02, sigma=0.20,
    )
    print(f"Call: {call:.4f}")
    print(f"Put: {put:.4f}")
    print(f"Straddle: {call + put:.4f}")