#!/usr/bin/env python3
"""
48_liquidity_and_persistence.py -- Two checks, pre-registered.

TEST 1 (descriptive, no testing-budget cost).  Has NIFTY weekly option
liquidity grown across the window?  Sinclair's tell for a dying inefficiency
is rising volume in the product, not a change in the effect's sign.  Reported:
contract volume, premium turnover, strikes traded, ATM-region volume, and --
the direct mechanism check -- whether the modelled half-spread narrowed.
Lot size is 75 throughout Dec-2024 to Apr-2026, so no lot adjustment is needed;
turnover is normalised by spot to strip the index-level rise.

TEST 2 (the gated one).  Does intraday PATH CHARACTER condition the condor,
independently of the volatility LEVEL?  acf1 (lag-1 autocorrelation of 1-min
returns) and trend_eff (trend efficiency) already exist in regime_features.csv
from script 23 and have never been used as conditioners.  acf1 > 0 is the
trending / Hurst-above-half state; acf1 < 0 is mean-reverting.

  *** BOTH ARE COMPUTED FROM A SESSION'S OWN BARS, SO THEY ARE LAGGED ONE
      SESSION.  Using them same-session would be look-ahead. ***

PRE-REGISTERED PASS CRITERION, fixed before running:
  (a) |t| >= 2.5 on the full sample, AND
  (b) same sign in 2025 and 2026, AND
  (c) circular-rotation placebo p < 0.05.
Miss any one and it is a null, and VWKS does not get built.
"""
import sys
from math import erf, sqrt
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.core import STORE

OPT = STORE / "opt2"
QTY, LOT = 750, 75
ncdf = np.vectorize(lambda v: 0.5 * (1 + erf(v / sqrt(2))))


def sc(x):
    x = np.asarray(pd.Series(x).dropna(), float); n = len(x)
    if n < 5 or x.std(ddof=1) == 0: return n, np.nan, np.nan, np.nan
    sr = x.mean() / x.std(ddof=1); t = sr / np.sqrt((1 + 0.5 * sr ** 2) / n)
    return n, x.mean(), sr * np.sqrt(252), t


def ols_hc1(y, X):
    y = np.asarray(y, float); X = np.asarray(X, float)
    n, k = X.shape
    XtXi = np.linalg.pinv(X.T @ X); b = XtXi @ (X.T @ y); u = y - X @ b
    V = XtXi @ ((X * u[:, None]).T @ (X * u[:, None])) @ XtXi * (n / (n - k))
    se = np.sqrt(np.diag(V)); t = b / se
    return b, se, t, 2 * (1 - ncdf(np.abs(t))), 1 - float(u @ u) / float(((y - y.mean()) ** 2).sum())


# ================= TEST 1 =================
print("=" * 78)
print("TEST 1  LIQUIDITY TREND  (descriptive -- does not spend testing budget)")
print("=" * 78)
cache = STORE / "chain_liquidity.csv"
if cache.exists():
    L = pd.read_csv(cache)
else:
    spot = pd.read_parquet(STORE / "spot_1min.parquet")
    sp = spot.groupby("date")["c"].mean()
    rows = []
    for f in sorted(OPT.glob("*.parquet")):
        d = pd.read_parquet(f, columns=["ts", "strike", "cp", "c", "v"])
        d["date"] = pd.to_datetime(d["ts"]).dt.date.astype(str)
        d = d[d.v > 0]
        if d.empty: continue
        d["S"] = d["date"].map(sp)
        d = d.dropna(subset=["S"])
        d["prem"] = d.v * d.c
        d["atm"] = (d.strike - d.S).abs() / d.S <= 0.01
        g = d.groupby("date").agg(vol=("v", "sum"), prem=("prem", "sum"),
                                  strikes=("strike", "nunique"),
                                  ticks=("v", "size"), S=("S", "first"))
        a = d[d.atm].groupby("date")["v"].sum().rename("vol_atm")
        rows.append(g.join(a))
        print(f"  {f.stem}", flush=True)
    L = pd.concat(rows).groupby(level=0).sum(min_count=1).reset_index()
    L["S"] = L["S"] / L["S"].clip(lower=1) * L["S"]
    L.to_csv(cache, index=False)

L["date"] = pd.to_datetime(L["date"])
L = L.sort_values("date")
L["ym"] = L.date.dt.to_period("M").astype(str)
L["turnover_cr"] = L.prem * LOT / 1e7          # premium turnover, Rs crore
L["vol_norm"] = L.vol / L.S * 1000             # contracts per 1000 index points

m = L.groupby("ym").agg(sessions=("vol", "size"), vol=("vol", "mean"),
                        turnover_cr=("turnover_cr", "mean"),
                        vol_atm=("vol_atm", "mean"), strikes=("strikes", "mean"))
print(m.round(1).to_string())

h1 = L[L.date < "2025-09-01"]; h2 = L[L.date >= "2025-09-01"]
print(f"\n  first half (to Aug-2025, n={len(h1)}) vs second (n={len(h2)}):")
for col, nm in [("vol", "contract volume"), ("turnover_cr", "premium turnover Rs cr"),
                ("vol_atm", "ATM-region volume"), ("strikes", "strikes traded"),
                ("vol_norm", "volume per 1000 index pts")]:
    a, b = h1[col].mean(), h2[col].mean()
    print(f"    {nm:<28} {a:>12,.1f} -> {b:>12,.1f}   {100*(b/a-1):>+7.1f}%")
x = np.arange(len(L)); y = L.turnover_cr.to_numpy(float)
bb, se, tt, pp, r2 = ols_hc1(y, np.column_stack([np.ones(len(x)), x]))
print(f"    trend in premium turnover: {bb[1]:+.3f} Rs cr/session, t = {tt[1]:.2f}, p = {pp[1]:.3f}")

w = pd.read_csv(STORE / "intraday_wings_costfix.csv")
w["date"] = pd.to_datetime(w["date"])
g = w[w.cfg == "0.25/0.1"].sort_values("date")
g["hs"] = g.slip / (8 * QTY)
x = np.arange(len(g)); bb, se, tt, pp, r2 = ols_hc1(g.hs.to_numpy(float),
                                                    np.column_stack([np.ones(len(x)), x]))
print(f"\n  MECHANISM CHECK -- modelled half-spread over time (0.25/0.10):")
print(f"    {g.hs.head(60).mean():.4f} pts/leg/side (first 60 sessions) -> "
      f"{g.hs.tail(60).mean():.4f} (last 60)")
print(f"    trend {bb[1]*1000:+.4f} pts per 1000 sessions, t = {tt[1]:.2f}, p = {pp[1]:.3f}")
print("    (liquidity growth should NARROW spreads; if it did not, the decay story is weaker)")

# ================= TEST 2 =================
print("\n" + "=" * 78)
print("TEST 2  PATH CHARACTER AS A CONDITIONER  (pre-registered, gates VWKS)")
print("=" * 78)
f = pd.read_csv(STORE / "regime_features.csv")[["date", "rv", "acf1", "trend_eff"]]
f["date"] = pd.to_datetime(f["date"]); f = f.sort_values("date").reset_index(drop=True)
for c in ("acf1", "trend_eff"):
    f[c + "_lag"] = f[c].shift(1)                      # LAGGED -- no look-ahead
f["rv20"] = f["rv"].shift(1).rolling(20).mean()
hi = f["rv20"].expanding(120).quantile(0.6667).shift(1)
f["gate"] = (f.rv20 >= hi) & hi.notna()

for cfg in ["0.25/0.1", "0.35/0.15"]:
    d = w[w.cfg == cfg].merge(f, on="date", how="left")
    d["net"] = d.gross - d.slip - d.stat_new if "stat_new" in d.columns else d.gross - d.slip - d.stat
    d["year"] = d.date.dt.year
    d = d.dropna(subset=["acf1_lag", "trend_eff_lag"])
    print(f"\n--- {cfg}   n={len(d)} ---")
    print(f"  acf1(lag): mean {d.acf1_lag.mean():+.4f} sd {d.acf1_lag.std():.4f}"
          f"   trend_eff(lag): mean {d.trend_eff_lag.mean():.4f} sd {d.trend_eff_lag.std():.4f}")
    for v in ["acf1_lag", "trend_eff_lag"]:
        y = d.net.to_numpy(float)
        X = np.column_stack([np.ones(len(d)), d[v].to_numpy(float)])
        b, se, t, p, r2 = ols_hc1(y, X)
        sub = {}
        for yr in (2025, 2026):
            s = d[d.year == yr]
            if len(s) < 30: continue
            bs, _, ts, _, _ = ols_hc1(s.net.to_numpy(float),
                                      np.column_stack([np.ones(len(s)), s[v].to_numpy(float)]))
            sub[yr] = (bs[1], ts[1])
        signs = [np.sign(v_[0]) for v_ in sub.values()]
        same = len(set(signs)) == 1 and len(signs) == 2
        # rotation placebo on the conditioner
        col = d[v].to_numpy(float); n = len(d)
        obs = abs(t[1])
        hits = 0
        for shf in range(1, n):
            Xp = np.column_stack([np.ones(n), np.roll(col, shf)])
            _, _, tp, _, _ = ols_hc1(y, Xp)
            hits += abs(tp[1]) >= obs
        pl = hits / (n - 1)
        ok = (abs(t[1]) >= 2.5) and same and (pl < 0.05)
        print(f"  net ~ {v:<14} b={b[1]:>12,.0f} t={t[1]:>6.2f} p={p[1]:>6.3f} R2={r2:>6.4f}"
              f"  | 2025 t={sub.get(2025,(0,np.nan))[1]:>5.2f} 2026 t={sub.get(2026,(0,np.nan))[1]:>5.2f}"
              f"  same-sign={same}  rot p={pl:.3f}  -> {'PASS' if ok else 'fail'}")
        # tercile read
        dd = d.copy(); dd["b"] = pd.qcut(dd[v], 3, labels=["lo", "mid", "hi"])
        r = dd.groupby("b", observed=True)["net"].agg(["mean", "size"])
        print("      terciles  " + "  ".join(
            f"{i}:{x['mean']:>8,.0f}(n={int(x['size'])})" for i, x in r.iterrows()))
    # orthogonality to the vol gate
    dd = d.dropna(subset=["gate"])
    if len(dd) > 50:
        print(f"  orthogonality: corr(acf1_lag, rv20) = {dd.acf1_lag.corr(dd.rv20):+.3f}"
              f"   corr(trend_eff_lag, rv20) = {dd.trend_eff_lag.corr(dd.rv20):+.3f}")
        ct = dd.groupby([dd.gate, pd.qcut(dd.acf1_lag, 2, labels=["MR", "trend"])],
                        observed=True)["net"].agg(["mean", "size"])
        print("  gate x path-character cells (Rs/session):")
        for i, x in ct.iterrows():
            print(f"      vol-gate={str(i[0]):<5} path={i[1]:<6} {x['mean']:>9,.0f}  n={int(x['size'])}")
