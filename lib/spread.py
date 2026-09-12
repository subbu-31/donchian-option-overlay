"""Effective-spread estimators for options that quote no bid/ask in the archive.

Both estimators recover a PROPORTIONAL spread from high/low/close data alone:

  Corwin & Schultz (2012, JF)  -- two-day high-low ratio; the insight is that
    the high is nearly always buyer-initiated and the low seller-initiated, so
    a single period's H/L range carries both variance and spread, while two
    periods together carry twice the variance and the same spread.

  Abdi & Ranaldo (2017, RFS)  -- close-high-low; uses the gap between the
    close and the mid-range proxy on consecutive periods.  Less biased than
    Corwin-Schultz when the price trends within the window.

Both are floored at the instrument's tick size: a quoted spread cannot be
narrower than one tick, and NIFTY options tick at Rs 0.05.
"""
import numpy as np

TICK = 0.05
_K = 3.0 - 2.0 * np.sqrt(2.0)


def corwin_schultz(h, l):
    """Proportional spread from consecutive high/low pairs. Returns the median
    of the per-pair estimates, negatives set to zero as the authors specify."""
    h = np.asarray(h, dtype=float)
    l = np.asarray(l, dtype=float)
    ok = np.isfinite(h) & np.isfinite(l) & (h > 0) & (l > 0) & (h >= l)
    h, l = h[ok], l[ok]
    if len(h) < 3:
        return np.nan
    lh, ll = np.log(h), np.log(l)
    b = (lh - ll) ** 2
    beta = b[:-1] + b[1:]
    h2 = np.maximum(h[:-1], h[1:])
    l2 = np.minimum(l[:-1], l[1:])
    gamma = (np.log(h2) - np.log(l2)) ** 2
    alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / _K - np.sqrt(gamma / _K)
    s = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))
    s = s[np.isfinite(s)]
    if not len(s):
        return np.nan
    return float(np.median(np.maximum(s, 0.0)))


def abdi_ranaldo(h, l, c):
    """Proportional spread from close vs mid-range on consecutive periods."""
    h = np.asarray(h, dtype=float)
    l = np.asarray(l, dtype=float)
    c = np.asarray(c, dtype=float)
    ok = np.isfinite(h) & np.isfinite(l) & np.isfinite(c) & (h > 0) & (l > 0) & (c > 0)
    h, l, c = h[ok], l[ok], c[ok]
    if len(h) < 3:
        return np.nan
    eta = (np.log(h) + np.log(l)) / 2.0
    lc = np.log(c)
    v = 4.0 * (lc[:-1] - eta[:-1]) * (lc[:-1] - eta[1:])
    v = v[np.isfinite(v)]
    if not len(v):
        return np.nan
    return float(np.sqrt(max(np.mean(v), 0.0)))


def half_spread_rupees(h, l, c, price, tick=TICK, method="both"):
    """Effective HALF-spread in rupees -- what one crossing actually costs.

    The two estimators are averaged where both are available; the result is
    floored at half a tick, since no crossing can be cheaper than that.
    """
    ests = []
    if method in ("cs", "both"):
        s = corwin_schultz(h, l)
        if np.isfinite(s):
            ests.append(s)
    if method in ("ar", "both"):
        s = abdi_ranaldo(h, l, c)
        if np.isfinite(s):
            ests.append(s)
    if not ests:
        return np.nan
    prop = float(np.mean(ests))
    return max(prop * float(price) / 2.0, tick / 2.0)


def zero_change_fraction(c):
    """Share of consecutive minutes with no price change -- a liquidity read
    that is independent of the two estimators above."""
    c = np.asarray(c, dtype=float)
    c = c[np.isfinite(c)]
    if len(c) < 3:
        return np.nan
    d = np.diff(c)
    return float(np.mean(np.abs(d) < 1e-9))
