"""Black-Scholes pricing, implied volatility by bisection, and delta.

Matches the reference IC methodology: IV and delta are both computed from the
ENTRY-time option price, entry-time spot, and entry-time time-to-expiry -- never
from a later intraday snapshot.
"""
import math

import numpy as np

RF = 0.065          # risk-free rate used throughout
YEAR = 365.0


def _nd(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_price(S, K, T, sigma, cp, r=RF):
    if T <= 0 or sigma <= 0:
        return max(0.0, S - K) if cp == "CE" else max(0.0, K - S)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if cp == "CE":
        return S * _nd(d1) - K * math.exp(-r * T) * _nd(d2)
    return K * math.exp(-r * T) * _nd(-d2) - S * _nd(-d1)


def implied_vol(price, S, K, T, cp, r=RF, lo=0.005, hi=4.0, tol=1e-5, iters=80):
    """Bisection. Returns None when the price is outside the no-arbitrage band."""
    if T <= 0 or price <= 0:
        return None
    intrinsic = max(0.0, S - K) if cp == "CE" else max(0.0, K - S)
    if price < intrinsic - 1e-6:
        return None
    if bs_price(S, K, T, hi, cp, r) < price:
        return None
    if bs_price(S, K, T, lo, cp, r) > price:
        return None
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if bs_price(S, K, T, mid, cp, r) > price:
            hi = mid
        else:
            lo = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def delta(S, K, T, sigma, cp, r=RF):
    if T <= 0 or sigma is None or sigma <= 0:
        return (1.0 if S > K else 0.0) if cp == "CE" else (-1.0 if S < K else 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    return _nd(d1) if cp == "CE" else _nd(d1) - 1.0


def years_to_expiry(entry_ts, expiry_yyyymmdd):
    """Expiry is settled at 15:30 IST on the expiry date."""
    import pandas as pd
    exp = pd.Timestamp(expiry_yyyymmdd) + pd.Timedelta(hours=15, minutes=30)
    return max((exp - pd.Timestamp(entry_ts)).total_seconds() / 86400.0, 1e-6) / YEAR
