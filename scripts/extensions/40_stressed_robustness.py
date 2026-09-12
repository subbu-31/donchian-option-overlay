#!/usr/bin/env python3
"""
40_stressed_robustness.py -- Try to break the Stressed-only result.

Script 39 found the high-trailing-vol tercile positive in BOTH periods, the
first conditioner to do so.  Before that can be traded it has to survive:
  (a) a same-size random-subset placebo, not just the skip-Normal one
  (b) moving the threshold  (median / 60th / 67th / 75th pct)
  (c) moving the rv20 lookback (10 / 20 / 30 / 40 sessions)
  (d) removing the best sessions -- is it a handful of days again?
  (e) a month-by-month read of the 2026 holdout
  (f) the same gate applied to BANKNIFTY 2018-2026, a genuinely untouched sample
"""
import sys
from math import erf, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.core import STORE

MINHIST, NPERM = 120, 20000
RNG = np.random.default_rng(4242)


def ncdf(x):
    return 0.5 * (1 + erf(x / sqrt(2)))


def sc(x):
    x = np.asarray(x, float); n = len(x)
    if n < 5 or x.std(ddof=1) == 0:
        return n, np.nan, np.nan, np.nan
    sr = x.mean() / x.std(ddof=1)
    t = sr / np.sqrt((1 + 0.5 * sr ** 2) / n)
    return n, x.mean(), sr * np.sqrt(252), t


def regime(rv, q, minhist=MINHIST):
    hi = rv.expanding(minhist).quantile(q).shift(1)
    return (rv >= hi) & hi.notna()


def load(cfg):
    w = pd.read_csv(STORE / "intraday_wings_costfix.csv")
    if "stat_new" in w.columns:
        w["stat"] = w["stat_new"]
    w["date"] = pd.to_datetime(w["date"])
    w["net"] = w["gross"] - w["slip"] - w["stat"]
    return w[w.cfg == cfg][["date", "net"]].sort_values("date").reset_index(drop=True)


def main():
    spot = pd.read_csv(STORE / "regime_decomp.csv")[["date", "rv", "rv20"]]
    spot["date"] = pd.to_datetime(spot["date"])
    spot = spot.sort_values("date").reset_index(drop=True)

    for cfg in ["0.35/0.15", "0.25/0.05"]:
        p = load(cfg)
        print(f"################  {cfg}  ################")

        # ---------- (a) same-size random-subset placebo ----------
        d = spot.copy()
        d["hi"] = regime(d["rv20"], 0.6667)
        m = p.merge(d[["date", "hi", "rv20"]], on="date").dropna(subset=["rv20"])
        m["year"] = m.date.dt.year
        for lab, g in [("2025", m[m.year == 2025]), ("2026", m[m.year == 2026]), ("FULL", m)]:
            k = int(g.hi.sum())
            obs = g.loc[g.hi, "net"].mean()
            net = g.net.to_numpy()
            hits = sum(1 for _ in range(NPERM)
                       if RNG.choice(net, k, replace=False).mean() >= obs)
            n, mu, sr, t = sc(g.loc[g.hi, "net"])
            print(f"  (a) {lab:<4} Stressed n={n:>3} Rs/s {mu:>8,.0f} SR {sr:>6.2f} t {t:>5.2f}"
                  f" | random-{k} subset beats it {100*hits/NPERM:>5.2f}% of {NPERM}")

        # ---------- (b) threshold sensitivity ----------
        print("  (b) threshold:")
        for q in [0.50, 0.60, 0.6667, 0.75]:
            d["hi"] = regime(d["rv20"], q)
            m = p.merge(d[["date", "hi", "rv20"]], on="date").dropna(subset=["rv20"])
            m["year"] = m.date.dt.year
            a = sc(m.loc[m.hi & (m.year == 2025), "net"])
            b = sc(m.loc[m.hi & (m.year == 2026), "net"])
            c = sc(m.loc[m.hi, "net"])
            print(f"      q={q:<6} IS n={a[0]:>3} Rs/s {a[1]:>7,.0f} t {a[3]:>5.2f} | "
                  f"OOS n={b[0]:>3} Rs/s {b[1]:>7,.0f} t {b[3]:>5.2f} | "
                  f"FULL n={c[0]:>3} Rs/s {c[1]:>7,.0f} SR {c[2]:>5.2f} t {c[3]:>5.2f}")

        # ---------- (c) lookback sensitivity ----------
        print("  (c) rv lookback:")
        for L in [10, 20, 30, 40]:
            rvL = spot["rv"].rolling(L).mean()
            hi = regime(rvL, 0.6667)
            m = p.merge(spot[["date"]].assign(hi=hi), on="date")
            m["year"] = m.date.dt.year
            a = sc(m.loc[m.hi & (m.year == 2025), "net"])
            b = sc(m.loc[m.hi & (m.year == 2026), "net"])
            c = sc(m.loc[m.hi, "net"])
            print(f"      L={L:<3} IS n={a[0]:>3} Rs/s {a[1]:>7,.0f} t {a[3]:>5.2f} | "
                  f"OOS n={b[0]:>3} Rs/s {b[1]:>7,.0f} t {b[3]:>5.2f} | "
                  f"FULL n={c[0]:>3} Rs/s {c[1]:>7,.0f} SR {c[2]:>5.2f} t {c[3]:>5.2f}")

        # ---------- (d) is it a handful of days ----------
        d["hi"] = regime(d["rv20"], 0.6667)
        m = p.merge(d[["date", "hi", "rv20"]], on="date").dropna(subset=["rv20"])
        m["year"] = m.date.dt.year
        g = m.loc[m.hi, "net"].sort_values()
        print("  (d) drop the best k sessions from the FULL Stressed set:")
        for k in [0, 1, 3, 5, 10]:
            x = g.iloc[:len(g) - k] if k else g
            n, mu, sr, t = sc(x)
            print(f"      drop {k:>2}: n={n:>3} Rs/s {mu:>8,.0f} SR {sr:>6.2f} t {t:>5.2f}")

        # ---------- (e) month by month, holdout ----------
        h = m[(m.year == 2026) & m.hi].copy()
        h["mo"] = h.date.dt.to_period("M").astype(str)
        print("  (e) 2026 Stressed by month:")
        print("      " + "  ".join(
            f"{k}:{v['net']:>8,.0f}(n={int(v['n'])})"
            for k, v in h.groupby("mo").agg(net=("net", "mean"), n=("net", "size")).iterrows()))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
