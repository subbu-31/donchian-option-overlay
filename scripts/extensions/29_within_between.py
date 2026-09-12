#!/usr/bin/env python3
"""
29_within_between.py -- Where does the width effect actually live?

Shuffling width_rel WITHIN a session reproduced the interaction coefficient
almost exactly (+0.105, t=6.8 against the real +0.094, t=6.9).  A within-session
shuffle preserves each session's MEAN width, so what that placebo shows is that
the effect is not about which bar of the day has a wide channel -- it is a
session-level relationship.  That matters enormously for the effective sample
size: 54,681 rows collapse to ~341 independent sessions, and the "order of
magnitude more power" claim was wrong.

Three tests that separate the two:

  A  session fixed effects -- identifies only from WITHIN-session variation.
     If the interaction survives, intraday timing carries information.
  B  between-session placebo -- permute whole sessions' width levels across
     sessions, destroying the session-level relationship while keeping
     everything else. This is the correct null for the effect that is there.
  C  the honest specification -- collapse to one observation per session and
     regress. n = 341, which is the real sample, and the standard errors are
     then not doing any work they cannot support.
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


def ols(y, X, groups=None, names=None):
    y = np.asarray(y, float); X = np.asarray(X, float)
    n, k = X.shape
    A = np.linalg.pinv(X.T @ X)
    b = A @ (X.T @ y)
    u = y - X @ b
    if groups is None:                                   # HC1
        meat = (X * (u ** 2)[:, None]).T @ X * (n / (n - k))
    else:
        meat = np.zeros((k, k))
        for g in np.unique(groups):
            m = groups == g
            s = X[m].T @ u[m]
            meat += np.outer(s, s)
        G = len(np.unique(groups))
        meat *= (G / (G - 1)) * ((n - 1) / (n - k))
    V = A @ meat @ A
    se = np.sqrt(np.diag(V))
    t = b / se
    return pd.DataFrame({"term": names, "coef": b, "se": se, "t": t,
                         "p": 2 * (1 - _ncdf(np.abs(t)))})


def demean(df, cols, by):
    out = df.copy()
    for c in cols:
        out[c] = df[c] - df.groupby(by)[c].transform("mean")
    return out


def main():
    d = pd.read_parquet(STORE / "forecast_panel.parquet")
    d = d[(d.im > 0) & (d.rm >= 0) & d.width_rel.notna() & (d.width_rel > 0)].copy()
    d["wr_c"] = d["width_rel"] - 1.0
    d["ix"] = d["im"] * d["wr_c"]
    rng = np.random.default_rng(5)

    for h in sorted(d.h.unique()):
        g = d[d.h == h].copy()
        gid = pd.factorize(g["date"])[0]
        print("=" * 92)
        print(f"HORIZON {h} MIN   n = {len(g):,}   sessions = {g.date.nunique()}")
        print("=" * 92)

        # A -- within-session identification
        w = demean(g, ["rm", "im", "ix"], "date")
        rA = ols(w.rm, np.column_stack([w.im, w.ix]), gid,
                 ["implied move", "implied move x (width-1)"])
        print("\n  A  session fixed effects (within-session variation only)")
        print(rA.to_string(index=False, float_format=lambda v: f"{v:9.4f}"))

        # B -- between-session placebo
        lvl = g.groupby("date")["wr_c"].mean()
        perm = pd.Series(rng.permutation(lvl.values), index=lvl.index)
        g["wr_fake"] = g["date"].map(perm) + (g["wr_c"] - g["date"].map(lvl))
        rB = ols(g.rm, np.column_stack([np.ones(len(g)), g.im, g.im * g.wr_fake]), gid,
                 ["const", "implied move", "implied move x (width-1)"])
        rR = ols(g.rm, np.column_stack([np.ones(len(g)), g.im, g.ix]), gid,
                 ["const", "implied move", "implied move x (width-1)"])
        print("\n  B  real vs between-session placebo (session width levels permuted)")
        print(f"      real     coef {rR.iloc[2]['coef']:+.4f}  t {rR.iloc[2]['t']:+.2f}")
        print(f"      placebo  coef {rB.iloc[2]['coef']:+.4f}  t {rB.iloc[2]['t']:+.2f}")

        # C -- honest session-level specification
        s = g.groupby("date").agg(rm=("rm", "mean"), im=("im", "mean"),
                                  wr=("width_rel", "mean"), iv=("iv", "mean"),
                                  dte=("dte", "mean")).reset_index()
        s["year"] = pd.to_datetime(s["date"]).dt.year
        rC = ols(s.rm, np.column_stack([np.ones(len(s)), s.im, s.im * (s.wr - 1)]), None,
                 ["const", "implied move", "implied move x (width-1)"])
        print(f"\n  C  one observation per session   n = {len(s)}   (HC1 errors)")
        print(rC.to_string(index=False, float_format=lambda v: f"{v:9.4f}"))
        for yr in (2025, 2026):
            ss = s[s.year == yr]
            if len(ss) < 40:
                continue
            r = ols(ss.rm, np.column_stack([np.ones(len(ss)), ss.im, ss.im * (ss.wr - 1)]),
                    None, ["const", "im", "ix"])
            print(f"      {yr}: interaction {r.iloc[2]['coef']:+.4f}  t {r.iloc[2]['t']:+.2f}"
                  f"   (n={len(ss)})")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
