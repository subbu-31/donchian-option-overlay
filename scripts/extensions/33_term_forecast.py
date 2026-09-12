#!/usr/bin/env python3
"""
33_term_forecast.py -- Does the IV term-structure slope forecast anything?

Q1  slope -> the near expiry's variance risk premium, controlling for the level
    of near IV. The control matters: slope and level are mechanically related,
    and without it any result is just the level effect in disguise.
Q2  slope -> its own next-session change (mean reversion). A calendar spread is
    a bet on the slope, so this is the precondition for that trade.
Q3  a placebo: the same regression with the slope permuted across sessions.

One observation per session, HC1 errors, and the honest sample is the ~330
sessions -- not the 1,219 curve points.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.core import STORE


def _ncdf(x):
    return 0.5 * (1 + np.vectorize(lambda v: math.erf(v / math.sqrt(2)))(x))


def ols(y, X, names):
    y = np.asarray(y, float); X = np.asarray(X, float)
    n, k = X.shape
    A = np.linalg.pinv(X.T @ X)
    b = A @ (X.T @ y)
    u = y - X @ b
    meat = (X * (u ** 2)[:, None]).T @ X * (n / (n - k))
    V = A @ meat @ A
    se = np.sqrt(np.diag(V))
    t = b / se
    r2 = 1 - float(u @ u) / float(((y - y.mean()) ** 2).sum())
    return pd.DataFrame({"term": names, "coef": b, "se": se, "t": t,
                         "p": 2 * (1 - _ncdf(np.abs(t)))}), r2, n


def main():
    p = pd.read_csv(STORE / "term_structure_panel.csv")
    cur = []
    for d, g in p.groupby("date"):
        g = g.sort_values("dte")
        if len(g) < 2 or g.dte.iloc[-1] / g.dte.iloc[0] < 1.5:
            continue
        near, far = g.iloc[0], g.iloc[-1]
        cur.append({"date": d, "iv_near": near.iv, "iv_far": far.iv,
                    "dte_near": near.dte, "dte_far": far.dte, "rv": near.rv,
                    "slope": (far.iv - near.iv) / math.log(far.dte / near.dte),
                    "n_pts": len(g)})
    c = pd.DataFrame(cur).sort_values("date").reset_index(drop=True)
    c["vrp"] = c.iv_near - c.rv
    c["year"] = pd.to_datetime(c["date"]).dt.year
    c.to_csv(STORE / "term_curve.csv", index=False)

    print(f"{len(c)} sessions with a usable curve, {c.date.min()} .. {c.date.max()}")
    print(c[["iv_near", "iv_far", "slope", "rv", "vrp", "dte_near", "dte_far"]]
          .describe().round(2).loc[["mean", "std", "25%", "50%", "75%"]].to_string())

    print("\n" + "=" * 92)
    print("Q1  near-expiry VRP ~ slope, controlling for the level of near IV")
    print("=" * 92)
    X = np.column_stack([np.ones(len(c)), c.iv_near, c.slope])
    r, r2, n = ols(c.vrp, X, ["const", "iv_near (level)", "slope"])
    print(r.to_string(index=False, float_format=lambda v: f"{v:9.4f}") + f"\n  R2 = {r2:.3f}  n = {n}")
    rng = np.random.default_rng(21)
    rp, _, _ = ols(c.vrp, np.column_stack([np.ones(len(c)), c.iv_near,
                                           rng.permutation(c.slope.values)]),
                   ["const", "iv_near", "slope (permuted)"])
    print(f"  placebo, slope permuted across sessions: coef "
          f"{rp.iloc[2]['coef']:+.4f}  t {rp.iloc[2]['t']:+.2f}")
    for yr in (2025, 2026):
        s = c[c.year == yr]
        if len(s) < 40:
            continue
        rs, _, _ = ols(s.vrp, np.column_stack([np.ones(len(s)), s.iv_near, s.slope]),
                       ["const", "lvl", "slope"])
        print(f"  {yr}: slope coef {rs.iloc[2]['coef']:+.4f}  t {rs.iloc[2]['t']:+.2f}  (n={len(s)})")

    print("\n" + "=" * 92)
    print("Q2  does the slope mean-revert?   next change ~ current level")
    print("=" * 92)
    c["d_slope"] = c.slope.shift(-1) - c.slope
    m = c.dropna(subset=["d_slope"])
    r2_, rr2, n2 = ols(m.d_slope, np.column_stack([np.ones(len(m)), m.slope]),
                       ["const", "slope"])
    print(r2_.to_string(index=False, float_format=lambda v: f"{v:9.4f}") +
          f"\n  R2 = {rr2:.3f}  n = {n2}")
    print(f"  autocorrelation of slope, lag 1: {c.slope.autocorr(1):.3f}")

    print("\n" + "=" * 92)
    print("Q3  VRP by slope quintile  (backwardated = near IV above far IV)")
    print("=" * 92)
    c["q"] = pd.qcut(c.slope, 5, labels=False)
    tab = c.groupby("q").agg(n=("vrp", "size"), slope=("slope", "mean"),
                             iv_near=("iv_near", "mean"), rv=("rv", "mean"),
                             vrp=("vrp", "mean"), ratio=("rv", "mean"))
    tab["rv_over_iv"] = c.groupby("q").apply(lambda g: (g.rv / g.iv_near).mean(),
                                             include_groups=False)
    print(tab.drop(columns=["ratio"]).round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
