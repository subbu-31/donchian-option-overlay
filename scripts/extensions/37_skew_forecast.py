#!/usr/bin/env python3
"""
37_skew_forecast.py -- Does the 09:45 implied skew forecast anything?

Predictors (all known at 09:45):
  rr25   = IV(-0.25 put) - IV(0.25 call)        skew level
  bf25   = mean(wing IV) - IV_atm               curvature
  rrz    = rr25 standardised by its own trailing 20-session mean and sd
           (causal -- prior sessions only; strips the 2024->2026 regime drift)
  bfz    = same for bf25
  rrn    = rr25 / iv_atm            vol-scaled skew

Outcomes (09:45 -> 15:00 same session):
  ret  down  absr  rv  rskw  and log(rv / iv_atm)   [the VRP state]

Inference: HAC (Newey-West, 5 lags) -- RR is persistent, so plain OLS SEs
understate.  Every reported t is HAC.  A permutation placebo re-runs each
spec with the predictor shuffled across sessions.
"""
import sys
from math import erf, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.core import STORE

LAGS = 5
NPERM = 5000
RNG = np.random.default_rng(20260904)


def ncdf(x):
    return 0.5 * (1 + np.vectorize(lambda v: erf(v / sqrt(2)))(x))


def ols_hac(y, X, lags=LAGS):
    y = np.asarray(y, float); X = np.asarray(X, float)
    n, k = X.shape
    XtXi = np.linalg.pinv(X.T @ X)
    b = XtXi @ (X.T @ y)
    u = y - X @ b
    S = (X * u[:, None]).T @ (X * u[:, None])
    for L in range(1, lags + 1):
        w = 1.0 - L / (lags + 1.0)
        A = (X[L:] * u[L:, None]).T @ (X[:-L] * u[:-L, None])
        S += w * (A + A.T)
    V = XtXi @ S @ XtXi * (n / (n - k))
    se = np.sqrt(np.diag(V))
    t = b / se
    r2 = 1 - float(u @ u) / float(((y - y.mean()) ** 2).sum())
    return b, se, t, 2 * (1 - ncdf(np.abs(t))), r2


def run(d, yname, xnames, label, perm_on=None):
    m = d[[yname] + xnames].dropna()
    if len(m) < 40:
        return None
    y = m[yname].to_numpy(float)
    X = np.column_stack([np.ones(len(m))] + [m[c].to_numpy(float) for c in xnames])
    b, se, t, p, r2 = ols_hac(y, X)
    names = ["const"] + xnames
    print(f"  {label:<34} n={len(m):>4}  R2={r2:>6.3f}")
    for i, nm in enumerate(names):
        if nm == "const":
            continue
        star = "  <--" if p[i] < 0.05 else ""
        print(f"      {nm:<10} b={b[i]:>11.4f}  se={se[i]:>9.4f}  "
              f"t={t[i]:>6.2f}  p={p[i]:>6.3f}{star}")
    if perm_on and perm_on in xnames:
        j = names.index(perm_on)
        obs = abs(t[j])
        hits = 0
        Xp = X.copy()
        col = X[:, j].copy()
        for _ in range(NPERM):
            Xp[:, j] = RNG.permutation(col)
            _, _, tt, _, _ = ols_hac(y, Xp, lags=0)
            if abs(tt[j]) >= obs:
                hits += 1
        print(f"      placebo({perm_on}): |t|>={obs:.2f} in "
              f"{100*hits/NPERM:.1f}% of {NPERM} shuffles")
    return t


def main():
    d = pd.read_csv(STORE / "skew_panel.csv")
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values("date").reset_index(drop=True)

    for k in ("rr25", "bf25", "rr10", "bf10"):
        r = d[k].rolling(20, min_periods=15)
        d[k + "z"] = (d[k] - r.mean().shift(1)) / r.std().shift(1)
    d["rrn"] = d["rr25"] / d["iv_atm"]
    d["vrp"] = np.log(d["rv"] / d["iv_atm"])
    d["year"] = d["date"].dt.year

    print(f"panel {len(d)} sessions  {d.date.min().date()} .. {d.date.max().date()}")
    print(f"rr25  mean {d.rr25.mean():.4f}  sd {d.rr25.std():.4f}  "
          f"AR(1) {d.rr25.autocorr(1):.2f}")
    print(f"rrz   sd {d.rr25z.std():.2f}  AR(1) {d.rr25z.autocorr(1):.2f}\n")

    for label, g in [("FULL", d), ("2025", d[d.year == 2025]), ("2026", d[d.year == 2026])]:
        print(f"===== {label}  n={len(g)} =====")
        run(g, "ret",  ["rr25z"], "A  direction:  ret ~ rrz", perm_on="rr25z" if label == "FULL" else None)
        run(g, "down", ["rr25z"], "B  downside:   min(ret,0) ~ rrz")
        run(g, "absr", ["iv_atm", "rr25z"], "C  size:       |ret| ~ ivatm + rrz")
        run(g, "rv",   ["iv_atm", "rr25z"], "D  vol:        rv ~ ivatm + rrz")
        run(g, "rskw", ["rr25z"], "E  realised skew: rskw ~ rrz", perm_on="rr25z" if label == "FULL" else None)
        run(g, "vrp",  ["rr25z", "bf25z"], "F  VRP state:  log(rv/iv) ~ rrz + bfz",
            perm_on="rr25z" if label == "FULL" else None)
        run(g, "vrp",  ["rrn"], "G  VRP state:  log(rv/iv) ~ rr/iv")
        print()

    print("===== quintile read on rrz (FULL) =====")
    m = d.dropna(subset=["rr25z", "vrp", "ret", "absr"]).copy()
    m["q"] = pd.qcut(m["rr25z"], 5, labels=[1, 2, 3, 4, 5])
    print(m.groupby("q", observed=True)[["rr25", "iv_atm", "rv", "vrp", "ret", "absr", "rskw"]]
          .mean().round(4).to_string())
    print("\n  n per bucket:", m.groupby("q", observed=True).size().to_list())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
