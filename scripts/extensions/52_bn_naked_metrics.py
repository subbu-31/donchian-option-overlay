#!/usr/bin/env python3
"""52_bn_naked_metrics.py -- Score the BANKNIFTY naked structures against NIFTY.
Everything in POINTS PER UNIT; rupees at QTY=200 are indicative only.
Clustered by EXPIRY as well as reported per session -- sessions inside one
contract share an IV level, so 453 sessions is not 453 observations."""
import sys
from math import erf, sqrt
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.core import STORE
QTY = 200
ncdf = lambda x: 0.5 * (1 + erf(x / sqrt(2)))
RNG = np.random.default_rng(20260905)
SPECS = ["strangle_25", "strangle_30", "strangle_35", "straddle_atm"]


def st(x):
    x = np.asarray(pd.Series(x).dropna(), float); n = len(x); sd = x.std(ddof=1)
    if n < 5 or sd == 0: return dict(n=n, mu=np.nan, sra=np.nan, t=np.nan)
    sr = x.mean() / sd
    return dict(n=n, mu=x.mean(), sra=sr*np.sqrt(252), t=sr/np.sqrt((1+0.5*sr**2)/n))


def clus_t(df, col="net_pts", g="expiry"):
    """Mean vs zero with errors clustered by contract."""
    m = df.groupby(g)[col].mean()
    n = len(m)
    if n < 5: return np.nan, n
    return float(m.mean() / (m.std(ddof=1) / np.sqrt(n))), n


def boot(x, nb=8000, blk=10):
    x = np.asarray(pd.Series(x).dropna(), float); n = len(x); out = np.empty(nb)
    for i in range(nb):
        s, j = [], RNG.integers(n)
        while len(s) < n:
            s.append(x[j % n]); j = RNG.integers(n) if RNG.random() < 1/blk else j + 1
        out[i] = np.mean(s)
    return np.percentile(out, 2.5), np.percentile(out, 97.5), float((out > 0).mean())


t = pd.read_csv(STORE / "bn_naked.csv")
t["date"] = pd.to_datetime(t["date"])
print(f"BANKNIFTY naked: {t.date.nunique()} sessions, {len(t)} trades, "
      f"{t.expiry.nunique()} contracts, {t.date.min().date()} .. {t.date.max().date()}")
print(f"  contract kind: {t.groupby('kind').expiry.nunique().to_dict()}  "
      f"sessions {t.groupby('kind').date.nunique().to_dict()}")
print(f"  DTE: median {t.dte.median():.1f}  p10 {t.dte.quantile(.1):.1f}  p90 {t.dte.quantile(.9):.1f}\n")

print("FULL WINDOW, all sessions  (points per unit)")
print(f"  {'structure':<15}{'n':>5}{'credit':>9}{'gross':>8}{'slip':>7}{'stat':>7}"
      f"{'net':>8}{'Sharpe':>8}{'t':>6}{'t(clus)':>9}{'win%':>7}")
for s in SPECS:
    x = t[t.spec == s]
    if len(x) < 20: continue
    a = st(x.net_pts); ct, ne = clus_t(x)
    print(f"  {s:<15}{a['n']:>5}{x.credit.mean():>9.1f}{x.gross.mean():>8.2f}"
          f"{x.slip.mean():>7.2f}{x.stat_pts.mean():>7.2f}{a['mu']:>8.2f}"
          f"{a['sra']:>8.2f}{a['t']:>6.2f}{ct:>9.2f}{100*(x.net_pts>0).mean():>7.1f}")
print(f"  (t(clus) clusters by contract -- {t.expiry.nunique()} contracts, not {t.date.nunique()} sessions)")

print("\nDTE 1-8.5 SLICE  -- directly comparable with the NIFTY run and with script 26")
print(f"  {'structure':<15}{'n':>5}{'net pts':>9}{'Sharpe':>8}{'t':>6}{'t(clus)':>9}"
      f"{'  NIFTY net pts':>16}")
NIF = {"strangle_25": 2.663, "strangle_30": 2.917, "strangle_35": 3.323, "straddle_atm": 3.353}
sl = t[(t.dte >= 1.0) & (t.dte <= 8.5)]
for s in SPECS:
    x = sl[sl.spec == s]
    if len(x) < 20: continue
    a = st(x.net_pts); ct, _ = clus_t(x)
    print(f"  {s:<15}{a['n']:>5}{a['mu']:>9.2f}{a['sra']:>8.2f}{a['t']:>6.2f}{ct:>9.2f}"
          f"{NIF[s]:>16.2f}")

print("\nBY DTE BUCKET  (net points; cost rises with DTE if far-dated strikes are thinner)")
t["db"] = pd.cut(t.dte, [0, 2, 5, 8.5, 14, 21, 40], labels=["0-2", "2-5", "5-8.5", "8.5-14", "14-21", "21+"])
for s in SPECS:
    r = t[t.spec == s].groupby("db", observed=True).agg(
        n=("net_pts", "size"), net=("net_pts", "mean"), gross=("gross", "mean"),
        slip=("slip", "mean"), credit=("credit", "mean"))
    print(f"  {s}")
    for i, x in r.iterrows():
        print(f"      dte {i:<7} n={int(x['n']):>4}  credit {x['credit']:>6.1f}"
              f"  gross {x['gross']:>7.2f}  slip {x['slip']:>5.2f}  net {x['net']:>7.2f}")

print("\nBY CONTRACT KIND  (monthly folders are front-month; weekly folders are not)")
for s in SPECS:
    row = f"  {s:<15}"
    for k in ["MONTHLY", "weekly"]:
        x = t[(t.spec == s) & (t.kind == k)]
        a = st(x.net_pts)
        row += f"{k}: n={a['n']:>4} net {a['mu']:>6.2f} t {a['t']:>5.2f}   "
    print(row)

print("\nBY ERA  (net points, strangle_25 and strangle_35)")
t["era"] = pd.cut(t.date.dt.year, [2017, 2020, 2022, 2024, 2027],
                  labels=["2018-20", "2021-22", "2023-24", "2025-26"])
for s in ["strangle_25", "strangle_35"]:
    row = f"  {s:<15}"
    for e_, g in t[t.spec == s].groupby("era", observed=True):
        a = st(g.net_pts); row += f"{e_}: n={a['n']:>3} {a['mu']:>6.2f} t {a['t']:>5.2f}  "
    print(row)

print("\nBOOTSTRAP AND TAIL  (full window)")
for s in SPECS:
    x = t[t.spec == s]
    lo, hi, pg = boot(x.net_pts)
    print(f"  {s:<15} 95% CI [{lo:>7.2f}, {hi:>7.2f}] pts  P>0 {pg:.3f}"
          f"   worst {x.net_pts.min():>8.1f}  MAE p5 {np.nanpercentile(x.mae,5):>8.1f}")

print("\nSTOP-LOSS OVERLAY at 0.50x credit  (the level that was ~free on NIFTY)")
for s in SPECS:
    g = t[t.spec == s].dropna(subset=["mae"])
    lvl = -0.5 * g.credit; hit = g.mae <= lvl
    y = np.where(hit, lvl - g.slip - g.stat_pts - 0.25, g.net_pts)
    a0, a1 = st(g.net_pts), st(y)
    print(f"  {s:<15} no stop net {a0['mu']:>6.2f} Sharpe {a0['sra']:>5.2f} worst {g.net_pts.min():>8.1f}"
          f"  |  stop: hit {100*hit.mean():>5.1f}%  net {a1['mu']:>6.2f}"
          f" Sharpe {a1['sra']:>5.2f} t {a1['t']:>5.2f} worst {np.min(y):>8.1f}")
