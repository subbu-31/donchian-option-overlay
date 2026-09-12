#!/usr/bin/env python3
"""
42_bn_regime_gate.py -- The decisive test of the high-vol gate.

NIFTY 2018-2026 does not exist in this archive; BANKNIFTY 2018-2026 does, and
the vol-regime gate has never been applied to it.  If the gate is real it must
work here.  If it only works on the 322 NIFTY sessions it was discovered on,
it is the tenth overfit in a row and gets closed.

Trailing vol is rebuilt from the archive's own spot files (each expiry folder
carries the contract's final ~4 weeks, so contiguous sessions are available even
where the traded DTE 1-8 sessions are not).  rv = intraday 1-min realised vol,
annualised; rv20 = mean of the PREVIOUS 20 available sessions, shift(1), and the
tercile cut is an expanding causal quantile.  Nothing is fitted.
"""
import re
import sys
import zipfile
from math import erf, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

STORE = Path.home() / "scratch" / "store"
ZIP = Path.home() / "mnt" / "Downloads" / "banknifty_data-[cloudtraderpro.in]-20260825T105024Z-1-001.zip"
BASE = "banknifty_data-[cloudtraderpro.in]"
ANN = np.sqrt(252 * 375)
MINHIST, NPERM = 60, 20000
RNG = np.random.default_rng(777)


def ncdf(x):
    return 0.5 * (1 + erf(x / sqrt(2)))


def sc(x):
    x = np.asarray(x, float); n = len(x)
    if n < 5 or x.std(ddof=1) == 0:
        return n, np.nan, np.nan, np.nan
    sr = x.mean() / x.std(ddof=1)
    t = sr / np.sqrt((1 + 0.5 * sr ** 2) / n)
    return n, x.mean(), sr * np.sqrt(252), t


def build_rv():
    cache = STORE / "bn_daily_rv.csv"
    if cache.exists():
        d = pd.read_csv(cache); d["date"] = pd.to_datetime(d["date"]); return d
    z = zipfile.ZipFile(ZIP)
    names = z.namelist()
    folders = sorted({n.split("/")[1] for n in names
                      if n.startswith(BASE + "/") and len(n.split("/")) > 2 and n.split("/")[1]})
    rows = []
    for e in folders:
        nm = f"{BASE}/{e}/banknifty_spot.csv"
        if nm not in names:
            continue
        s = pd.read_csv(z.open(nm))
        s.columns = [c.lower() for c in s.columns]
        tc = "timestamp" if "timestamp" in s.columns else s.columns[0]
        s["ts"] = pd.to_datetime(s[tc], errors="coerce")
        s = s.dropna(subset=["ts"])
        s["date"] = s["ts"].dt.date
        for d, g in s.groupby("date"):
            g = g.sort_values("ts")
            c = g["close"].to_numpy(float)
            if len(c) < 200:
                continue
            r = np.diff(np.log(c))
            r = r[np.isfinite(r)]
            if len(r) < 150:
                continue
            rows.append({"date": pd.Timestamp(d), "rv": float(np.std(r, ddof=1) * ANN * 100)})
    d = (pd.DataFrame(rows).groupby("date", as_index=False)["rv"].mean()
         .sort_values("date").reset_index(drop=True))
    d.to_csv(cache, index=False)
    return d


def main():
    rv = build_rv()
    print(f"BANKNIFTY spot sessions with usable intraday data: {len(rv)}  "
          f"{rv.date.min().date()} .. {rv.date.max().date()}")
    gap = rv.date.diff().dt.days
    print(f"  median gap between consecutive sessions {gap.median():.0f} d, "
          f"90th pct {gap.quantile(0.9):.0f} d  (archive is SAMPLED, not continuous)")

    rv["rv20"] = rv["rv"].shift(1).rolling(20).mean()
    hi = rv["rv20"].expanding(MINHIST).quantile(0.6667).shift(1)
    rv["hi"] = (rv.rv20 >= hi) & hi.notna()

    b = pd.read_csv(STORE / "banknifty_condor.csv")
    b["date"] = pd.to_datetime(b["date"])
    m = b.merge(rv[["date", "rv", "rv20", "hi"]], on="date", how="left")

    for cfg in ["0.35/0.15", "0.25/0.1"]:
        g = m[(m.cfg == cfg) & m.hi.notna()].copy()
        if len(g) < 40:
            print(f"{cfg}: only {len(g)} rows with a defined regime -- skipped")
            continue
        print(f"\n===== BANKNIFTY {cfg}   n={len(g)}   "
              f"stressed {int(g.hi.sum())} / {len(g)} =====")
        for lab, x in [("all sessions", g.net), ("STRESSED only", g.loc[g.hi, "net"]),
                       ("rest", g.loc[~g.hi, "net"])]:
            n, mu, s_, t = sc(x)
            print(f"  {lab:<15} n={n:>4}  Rs/sess {mu:>8,.0f}  total {np.sum(x):>10,.0f}"
                  f"  SR {s_:>6.2f}  t {t:>5.2f}")
        k = int(g.hi.sum())
        obs = g.loc[g.hi, "net"].mean()
        net = g.net.to_numpy()
        hits = sum(1 for _ in range(NPERM)
                   if RNG.choice(net, k, replace=False).mean() >= obs)
        print(f"  placebo: a random {k}-session subset beats it "
              f"{100*hits/NPERM:.2f}% of {NPERM} draws")
        print("  by era (STRESSED only):")
        g["era"] = pd.cut(g.date.dt.year, [2017, 2020, 2022, 2024, 2027],
                          labels=["2018-20", "2021-22", "2023-24", "2025-26"])
        for era, gg in g[g.hi].groupby("era", observed=True):
            n, mu, s_, t = sc(gg.net)
            print(f"    {era:<9} n={n:>3}  Rs/sess {mu:>8,.0f}  t {t:>5.2f}")
        print("  credit collected: stressed "
              f"{g.loc[g.hi,'credit'].mean():.1f} pts vs rest {g.loc[~g.hi,'credit'].mean():.1f} pts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
