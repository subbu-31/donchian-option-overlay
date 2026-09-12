#!/usr/bin/env python3
"""
28_forecast_regression.py -- Does the channel-width state add anything to
implied volatility as a forecast of the next move?

The P&L test said "priced" on 322 daily observations.  This asks the same
question on ~50,000 five-minute observations, which is the same hypothesis with
roughly an order of magnitude more power.

Specifications, run separately per horizon:

  (1)  rm = a + b*im + e                        implied move alone
  (2)  rm = a + b*im + c*(im * wr_c) + e        does width scale the implied move
  (3)  log rm = a + b*log im + c*log wr + e     elasticity form; c = 0 is the
                                                efficient-market null
  (4)  ratio = rm/im on width quintiles         non-parametric read

wr_c is width_rel centred at 1, so c is the extra move per unit of implied move
per unit of excess width.  Under the null that implied volatility already
impounds the channel state, c = 0 in (2) and (3).

Standard errors are clustered by session throughout.  Overlapping horizons
within a day make the raw OLS errors meaningless; the clustered ones are the
only defensible version, and the effective sample is the ~320 sessions, not the
50,000 rows.  That distinction is the point of the paper this belongs to.
"""
import sys
from pathlib import Path

import math

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.core import STORE


def ols_cluster(y, X, groups, names):
    """OLS with CR1 cluster-robust covariance."""
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    n, k = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    u = y - X @ beta
    meat = np.zeros((k, k))
    for g in np.unique(groups):
        m = groups == g
        Xg, ug = X[m], u[m]
        s = Xg.T @ ug
        meat += np.outer(s, s)
    G = len(np.unique(groups))
    c = (G / (G - 1)) * ((n - 1) / (n - k))
    V = XtX_inv @ meat @ XtX_inv * c
    se = np.sqrt(np.diag(V))
    t = beta / se
    ss_res = float(u @ u)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return pd.DataFrame({"term": names, "coef": beta, "se": se, "t": t,
                         "p": 2 * (1 - _norm_cdf(np.abs(t)))}), 1 - ss_res / ss_tot, G


def _norm_cdf(x):
    return 0.5 * (1 + np.vectorize(lambda v: math.erf(v / math.sqrt(2)))(x))


def main():
    d = pd.read_parquet(STORE / "forecast_panel.parquet")
    d = d[(d.im > 0) & (d.rm >= 0) & d.width_rel.notna() & (d.width_rel > 0)]
    print(f"panel: {len(d):,} observations, {d.date.nunique()} sessions, "
          f"{d.date.min()} .. {d.date.max()}\n")

    for h in sorted(d.h.unique()):
        g = d[d.h == h].copy()
        gid = pd.factorize(g["date"])[0]
        g["wr_c"] = g["width_rel"] - 1.0
        print("=" * 96)
        print(f"HORIZON {h} MINUTES   n = {len(g):,}   sessions = {g.date.nunique()}")
        print("=" * 96)

        X1 = np.column_stack([np.ones(len(g)), g.im])
        r1, r2_1, _ = ols_cluster(g.rm, X1, gid, ["const", "implied move"])
        print("\n  (1) realised move ~ implied move")
        print(r1.to_string(index=False, float_format=lambda v: f"{v:9.4f}") +
              f"\n      R2 = {r2_1:.4f}")

        X2 = np.column_stack([np.ones(len(g)), g.im, g.im * g.wr_c])
        r2, r2_2, G = ols_cluster(g.rm, X2, gid,
                                  ["const", "implied move", "implied move x (width-1)"])
        print("\n  (2) + width interaction   [null: interaction = 0]")
        print(r2.to_string(index=False, float_format=lambda v: f"{v:9.4f}") +
              f"\n      R2 = {r2_2:.4f}   clusters = {G}")

        m = g[(g.rm > 0)]
        X3 = np.column_stack([np.ones(len(m)), np.log(m.im), np.log(m.width_rel)])
        r3, r2_3, _ = ols_cluster(np.log(m.rm), X3, pd.factorize(m["date"])[0],
                                  ["const", "log implied move", "log width_rel"])
        print("\n  (3) log realised ~ log implied + log width   [null: log width = 0]")
        print(r3.to_string(index=False, float_format=lambda v: f"{v:9.4f}") +
              f"\n      R2 = {r2_3:.4f}   n = {len(m):,}")

        print("\n  (4) realised / implied by width quintile")
        g["q"] = pd.qcut(g.width_rel, 5, labels=False, duplicates="drop")
        tab = g.groupby("q").agg(n=("ratio", "size"), width=("width_rel", "mean"),
                                 ratio=("ratio", "mean"))
        print("      " + tab.to_string().replace("\n", "\n      "))

        for yr in (2025, 2026):
            s = g[g.year == yr]
            if len(s) < 500:
                continue
            Xs = np.column_stack([np.ones(len(s)), s.im, s.im * s.wr_c])
            rs, _, _ = ols_cluster(s.rm, Xs, pd.factorize(s["date"])[0],
                                   ["const", "im", "im x (width-1)"])
            row = rs.iloc[2]
            print(f"      {yr}: interaction coef {row['coef']:+.4f}  "
                  f"t = {row['t']:+.2f}  (n={len(s):,}, {s.date.nunique()} sessions)")
        print()
    return 0


if __name__ == "__main__" and "--robust" not in sys.argv:
    raise SystemExit(main())


def robustness():
    """The obvious alternative explanation: the nearest-weekly ATM implied vol is
    an average over days and cannot express the intraday U-shape, so width_rel
    might simply be proxying for time of day.  Absorb bar-of-day with fixed
    effects and add days-to-expiry; if the interaction survives, that story is
    not what is driving it."""
    d = pd.read_parquet(STORE / "forecast_panel.parquet")
    d = d[(d.im > 0) & (d.rm >= 0) & d.width_rel.notna() & (d.width_rel > 0)]
    print("\n\n" + "#" * 96)
    print("#  ROBUSTNESS: bar-of-day fixed effects + days-to-expiry")
    print("#" * 96)
    for h in sorted(d.h.unique()):
        g = d[d.h == h].copy()
        g["wr_c"] = g["width_rel"] - 1.0
        bars = sorted(g.bar_no.unique())[1:]          # drop one for the intercept
        D = np.column_stack([(g.bar_no == b).astype(float) for b in bars])
        X = np.column_stack([np.ones(len(g)), g.im, g.im * g.wr_c, g.dte, D])
        names = ["const", "implied move", "implied move x (width-1)", "dte"] + \
                [f"bar_{b}" for b in bars]
        r, r2, G = ols_cluster(g.rm, X, pd.factorize(g["date"])[0], names)
        keep = r[r.term.isin(["implied move", "implied move x (width-1)", "dte"])]
        print(f"\n  horizon {h} min   n={len(g):,}  clusters={G}  "
              f"bar-of-day dummies={len(bars)}  R2={r2:.4f}")
        print(keep.to_string(index=False, float_format=lambda v: f"{v:9.4f}"))

        # placebo: shuffle width_rel within each session
        rng = np.random.default_rng(11)
        sh = g.groupby("date")["wr_c"].transform(lambda s: rng.permutation(s.values))
        Xp = np.column_stack([np.ones(len(g)), g.im, g.im * sh, g.dte, D])
        rp, _, _ = ols_cluster(g.rm, Xp, pd.factorize(g["date"])[0], names)
        row = rp[rp.term == "implied move x (width-1)"].iloc[0]
        print(f"      placebo (width shuffled within session): coef {row['coef']:+.4f}  "
              f"t = {row['t']:+.2f}")


if __name__ == "__main__" and "--robust" in sys.argv:
    robustness()
