#!/usr/bin/env python3
"""
34_vrp_state.py -- Reconciling two results that look contradictory, and testing
the state variable they jointly imply.

The width work found that when REALISED volatility has been high (wide channel)
the realised move is a LARGER fraction of what implied priced -- worse for a
seller. The term-structure work finds that when IMPLIED volatility is high the
realised move is a SMALLER fraction -- better for a seller.

Those are not in conflict: one conditions on backward-looking realised vol, the
other on forward-looking implied. Together they say the state that matters is
implied RELATIVE to recent realised, which is the variance premium itself.

Tested here: does log(iv_near / trailing realised) forecast the near expiry's
variance risk premium better than either level alone, and does it hold in both
subperiods and against a permutation placebo?

Caveat carried forward: this is roughly the sixth signal tested on this sample.
A single t-statistic here should be read against that, not on its own.
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
    b = A @ (X.T @ y); u = y - X @ b
    V = A @ ((X * (u ** 2)[:, None]).T @ X * (n / (n - k))) @ A
    se = np.sqrt(np.diag(V)); t = b / se
    r2 = 1 - float(u @ u) / float(((y - y.mean()) ** 2).sum())
    return pd.DataFrame({"term": names, "coef": b, "se": se, "t": t,
                         "p": 2 * (1 - _ncdf(np.abs(t)))}), r2, n


def main():
    c = pd.read_csv(STORE / "term_curve.csv")
    f = pd.read_csv(STORE / "regime_features.csv")[["date", "rv", "rv_cc"]]
    f = f.rename(columns={"rv": "rv_sess"})
    f = f.sort_values("date")
    # trailing realised, causal: mean of the previous 20 sessions' intraday rv
    f["rv_trail"] = f["rv_sess"].shift(1).rolling(20).mean()
    d = c.merge(f[["date", "rv_trail"]], on="date", how="left").dropna(subset=["rv_trail"])
    d["state"] = np.log(d.iv_near / d.rv_trail)
    d["year"] = pd.to_datetime(d["date"]).dt.year
    print(f"{len(d)} sessions   state = log(near IV / trailing 20-session realised)")
    print(d[["iv_near", "rv_trail", "state", "vrp"]].describe().round(3)
          .loc[["mean", "std", "25%", "50%", "75%"]].to_string())

    specs = [
        ("level only        ", ["iv_near"]),
        ("trailing only     ", ["rv_trail"]),
        ("both levels       ", ["iv_near", "rv_trail"]),
        ("state (ratio)     ", ["state"]),
        ("state + level     ", ["state", "iv_near"]),
    ]
    print("\n" + "=" * 92)
    print("Forecasting the near-expiry variance risk premium")
    print("=" * 92)
    print(f"  {'specification':<20}{'R2':>8}{'key coef':>12}{'t':>9}")
    for lab, cols in specs:
        X = np.column_stack([np.ones(len(d))] + [d[x] for x in cols])
        r, r2, n = ols(d.vrp, X, ["const"] + cols)
        k = r.iloc[1]
        print(f"  {lab:<20}{r2:>8.3f}{k['coef']:>12.4f}{k['t']:>9.2f}   ({k['term']})")

    print("\n  full specification, state + level:")
    X = np.column_stack([np.ones(len(d)), d.state, d.iv_near])
    r, r2, n = ols(d.vrp, X, ["const", "state", "iv_near"])
    print("  " + r.to_string(index=False, float_format=lambda v: f"{v:9.4f}").replace("\n", "\n  "))
    print(f"    R2 = {r2:.3f}   n = {n}")

    rng = np.random.default_rng(33)
    rp, _, _ = ols(d.vrp, np.column_stack([np.ones(len(d)),
                                           rng.permutation(d.state.values), d.iv_near]),
                   ["const", "state (permuted)", "iv_near"])
    print(f"    placebo, state permuted: coef {rp.iloc[1]['coef']:+.4f}  t {rp.iloc[1]['t']:+.2f}")
    for yr in (2025, 2026):
        s = d[d.year == yr]
        if len(s) < 40:
            continue
        rs, _, _ = ols(s.vrp, np.column_stack([np.ones(len(s)), s.state, s.iv_near]),
                       ["const", "state", "lvl"])
        print(f"    {yr}: state coef {rs.iloc[1]['coef']:+.4f}  t {rs.iloc[1]['t']:+.2f}  (n={len(s)})")

    print("\n" + "=" * 92)
    print("VRP and realised/implied by state quintile")
    print("=" * 92)
    d["q"] = pd.qcut(d.state, 5, labels=False)
    t = d.groupby("q").agg(n=("vrp", "size"), state=("state", "mean"),
                           iv_near=("iv_near", "mean"), rv_trail=("rv_trail", "mean"),
                           rv_sess=("rv", "mean"), vrp=("vrp", "mean"))
    t["realised_over_implied"] = d.groupby("q").apply(
        lambda g: (g.rv / g.iv_near).mean(), include_groups=False)
    print(t.round(3).to_string())
    d.to_csv(STORE / "vrp_state.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
