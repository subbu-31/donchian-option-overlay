#!/usr/bin/env python3
"""
50_naked_battery.py -- The adversarial battery on the naked structures.

The rotation placebo from script 43 tested a GATE; an ungated structure has no
label to rotate, so the honest analogues here are: block bootstrap, deflated
Sharpe against the trial count, cost stress, drop-the-best-k, sub-period
stability, PBO across the structure grid, and a quarterly walk-forward that
picks the structure from PAST data only.

Two additions that matter more than any of those:
  * STOP-LOSS OVERLAY. His standing rule makes a GTT stop compulsory on shorts,
    so the no-stop backtest is not the tradeable object. Using the stored MAE,
    a stop at -X closes the trade at -X whenever MAE <= -X, and leaves it alone
    otherwise. That is exact, not an approximation, apart from fill quality --
    so an extra crossing is charged on every stopped trade.
  * THE VOL GATE crossed with the naked structures, which DOES admit a rotation
    placebo since it reintroduces a label.

Everything is in POINTS PER UNIT so the naked run (QTY 650) and the condor
studies (QTY 750) are directly comparable.
"""
import sys
from math import erf, sqrt, exp, log
from pathlib import Path
from itertools import combinations
from statistics import NormalDist
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.core import STORE

NQ, CQ = 650, 750
EULER = 0.5772156649
RNG = np.random.default_rng(20260905)
ncdf = lambda x: 0.5 * (1 + erf(x / sqrt(2)))
nppf = NormalDist().inv_cdf
STOP_X = [0.5, 0.75, 1.0, 1.5, 2.0]
EXTRA_CROSS = 0.25          # points charged on a stopped exit, both legs


def st(x):
    x = np.asarray(pd.Series(x).dropna(), float); n = len(x); sd = x.std(ddof=1)
    if n < 5 or sd == 0: return dict(n=n, mu=np.nan, sr=np.nan, sra=np.nan, t=np.nan, g3=np.nan, g4=np.nan)
    sr = x.mean() / sd; z = (x - x.mean()) / sd
    return dict(n=n, mu=x.mean(), sr=sr, sra=sr * np.sqrt(252),
                t=sr / np.sqrt((1 + 0.5 * sr ** 2) / n),
                g3=float((z ** 3).mean()), g4=float((z ** 4).mean()))


def boot(x, nb=10000, blk=10):
    x = np.asarray(pd.Series(x).dropna(), float); n = len(x)
    out = np.empty(nb)
    for i in range(nb):
        s, j = [], RNG.integers(n)
        while len(s) < n:
            s.append(x[j % n]); j = RNG.integers(n) if RNG.random() < 1 / blk else j + 1
        out[i] = np.mean(s)
    return np.percentile(out, 2.5), np.percentile(out, 97.5), float((out > 0).mean())


def dsr(x, srs, N):
    s = st(x)
    v = np.nanstd(np.asarray(srs, float), ddof=1)
    sr0 = 0.0 if (not np.isfinite(v) or v == 0 or N < 2) else \
        v * ((1 - EULER) * nppf(1 - 1 / N) + EULER * nppf(1 - 1 / (N * exp(1))))
    den = sqrt(max(1e-12, 1 - s["g3"] * s["sr"] + (s["g4"] - 1) / 4 * s["sr"] ** 2))
    return ncdf((s["sr"] - sr0) * sqrt(s["n"] - 1) / den), sr0


# ---------------- assemble one points-per-unit panel ----------------
nk = pd.read_csv(STORE / "naked_premium.csv"); nk["date"] = pd.to_datetime(nk["date"])
nk["net_pts"] = nk.gross - nk.slip - nk.stat_pts
w = pd.read_csv(STORE / "intraday_wings_costfix.csv"); w["date"] = pd.to_datetime(w["date"])
if "stat_new" in w.columns: w["stat"] = w["stat_new"]
w["net_pts"] = (w.gross - w.slip - w.stat) / CQ
w["spec"] = "condor_" + w.cfg
P = pd.concat([nk[["date", "spec", "net_pts", "credit", "gross", "slip", "stat_pts", "mae"]],
               w[["date", "spec", "net_pts"]]], ignore_index=True)
P["year"] = P.date.dt.year
SPECS = ["strangle_25", "strangle_30", "strangle_35", "straddle_atm",
         "condor_0.35/0.15", "condor_0.25/0.1", "condor_0.25/0.05"]
NAKED = SPECS[:4]

print("=" * 88)
print("BASELINE  (points per unit, net of modelled slippage and statutory)")
print("=" * 88)
srs = []
for s in SPECS:
    x = P.loc[P.spec == s, "net_pts"]
    if len(x) < 20: continue
    srs.append(st(x)["sr"])
print(f"  {'structure':<18}{'n':>5}{'net pts':>9}{'Sharpe':>8}{'t':>6}{'p':>7}"
      f"{'boot 95% CI (pts)':>24}{'P>0':>7}")
for s in SPECS:
    x = P.loc[P.spec == s, "net_pts"]
    if len(x) < 20: continue
    a = st(x); lo, hi, pg = boot(x)
    print(f"  {s:<18}{a['n']:>5}{a['mu']:>9.3f}{a['sra']:>8.2f}{a['t']:>6.2f}"
          f"{2*(1-ncdf(abs(a['t']))):>7.3f}   [{lo:>7.3f}, {hi:>7.3f}]{pg:>7.3f}")

print("\n" + "=" * 88)
print("DEFLATED SHARPE  (skew/kurtosis corrected, against the trial count)")
print("=" * 88)
for s in NAKED:
    x = P.loc[P.spec == s, "net_pts"]
    a = st(x)
    row = f"  {s:<18} skew {a['g3']:>6.2f}  kurt {a['g4']:>5.2f}  "
    for N, lab in [(7, "this grid N=7"), (20, "programme N=20")]:
        d, sr0 = dsr(x, srs, N)
        row += f"| {lab}: SR0 {sr0*np.sqrt(252):>4.2f} DSR {d:>5.3f}{' PASS' if d>0.95 else ' fail'} "
    print(row)

print("\n" + "=" * 88)
print("COST STRESS AND DROP-THE-BEST")
print("=" * 88)
for s in NAKED:
    g = nk[nk.spec == s]
    print(f"  {s}")
    for m in (1.0, 2.0, 3.0):
        x = g.gross - m * g.slip - g.stat_pts
        a = st(x); print(f"      {m:.0f}x slip      net {a['mu']:>7.3f} pts  Sharpe {a['sra']:>5.2f}  t {a['t']:>5.2f}")
    x = np.sort(g.net_pts.to_numpy(float))
    for k in (3, 5, 10):
        a = st(x[:len(x) - k]); print(f"      drop best {k:<3} net {a['mu']:>7.3f} pts  Sharpe {a['sra']:>5.2f}  t {a['t']:>5.2f}")

print("\n" + "=" * 88)
print("STOP-LOSS OVERLAY  (stop at X x credit; charged one extra crossing when hit)")
print("=" * 88)
for s in NAKED:
    g = nk[nk.spec == s].dropna(subset=["mae"])
    print(f"  {s}   no stop: net {g.net_pts.mean():>6.3f} pts  Sharpe {st(g.net_pts)['sra']:>5.2f}"
          f"  worst {g.net_pts.min():>7.2f}  MAE p5 {np.percentile(g.mae,5):>7.1f}")
    for X in STOP_X:
        lvl = -X * g.credit
        hit = g.mae <= lvl
        y = np.where(hit, lvl - g.slip - g.stat_pts - EXTRA_CROSS, g.net_pts)
        a = st(y)
        print(f"      stop {X:>4.2f}x credit  hit {100*hit.mean():>5.1f}%  net {a['mu']:>7.3f} pts"
              f"  Sharpe {a['sra']:>5.2f}  t {a['t']:>5.2f}  worst {np.min(y):>7.2f}")

print("\n" + "=" * 88)
print("PBO via CSCV  (S=10, 252 balanced splits, over the 7-structure grid)")
print("=" * 88)
piv = P.pivot_table(index="date", columns="spec", values="net_pts").dropna()
M = piv[SPECS].to_numpy(float); n = len(M); S = 10
idx = np.array_split(np.arange(n), S); logits = []
for combo in combinations(range(S), S // 2):
    tr = np.concatenate([idx[i] for i in combo])
    te = np.concatenate([idx[i] for i in range(S) if i not in combo])
    with np.errstate(invalid="ignore"):
        sa = np.nan_to_num(M[tr].mean(0) / M[tr].std(0, ddof=1), nan=-9)
        sb = np.nan_to_num(M[te].mean(0) / M[te].std(0, ddof=1), nan=-9)
    b = int(np.argmax(sa)); r = min(max((sb < sb[b]).sum() / len(sb), 1e-3), 1 - 1e-3)
    logits.append(log(r / (1 - r)))
logits = np.array(logits)
print(f"  sessions in the common panel: {n}   PBO = {(logits<=0).mean():.3f}"
      f"   median logit {np.median(logits):+.2f}")
print(f"  in-sample-best structure by split: "
      f"{pd.Series([SPECS[int(np.argmax(np.nan_to_num(M[np.concatenate([idx[i] for i in c])].mean(0)/M[np.concatenate([idx[i] for i in c])].std(0,ddof=1),nan=-9)))] for c in combinations(range(S),S//2)]).value_counts().to_dict()}")

print("\n" + "=" * 88)
print("QUARTERLY WALK-FORWARD  (structure chosen from PAST data only)")
print("=" * 88)
piv = piv.sort_index(); q = piv.index.to_period("Q")
rows = []
for p in sorted(set(q)):
    te = (q == p); tr = (q < p)
    if tr.sum() < 150 or te.sum() < 5: continue
    A = piv[SPECS].to_numpy(float)
    sa = np.nan_to_num(A[tr].mean(0) / A[tr].std(0, ddof=1), nan=-9)
    b = SPECS[int(np.argmax(sa))]
    rows.append(dict(q=str(p), n=int(te.sum()), pick=b,
                     wf=float(piv.loc[te, b].sum()),
                     s25=float(piv.loc[te, "strangle_25"].sum()),
                     cond=float(piv.loc[te, "condor_0.25/0.1"].sum())))
print(f"  {'quarter':<9}{'n':>4}  {'picked (from past)':<20}{'WF pts':>9}{'always s25':>12}{'condor':>9}")
for r in rows:
    print(f"  {r['q']:<9}{r['n']:>4}  {r['pick']:<20}{r['wf']:>9.2f}{r['s25']:>12.2f}{r['cond']:>9.2f}")
if rows:
    tot = pd.DataFrame(rows)
    print(f"  OOS quarters {len(rows)}   WF total {tot.wf.sum():.2f} pts"
          f"   always-strangle25 {tot.s25.sum():.2f}   condor {tot.cond.sum():.2f}")

print("\n" + "=" * 88)
print("VOL GATE x NAKED  (rotation placebo applies here -- a label exists again)")
print("=" * 88)
f = pd.read_csv(STORE / "regime_decomp.csv")[["date", "rv20"]]
f["date"] = pd.to_datetime(f["date"]); f = f.dropna().sort_values("date").reset_index(drop=True)
hi = f.rv20.expanding(120).quantile(0.6667).shift(1)
f["gate"] = (f.rv20 >= hi) & hi.notna()
for s in NAKED:
    g = nk[nk.spec == s].merge(f[["date", "gate"]], on="date", how="left").dropna(subset=["gate"])
    g = g.sort_values("date")
    x = g.net_pts.to_numpy(float); gt = g.gate.to_numpy(bool); n = len(x)
    a_all, a_g = st(x), st(x[gt])
    obs = x[gt].mean()
    mus = np.array([x[np.roll(gt, sh)].mean() for sh in range(1, n)])
    lo, hi_, pg = boot(x[gt])
    print(f"  {s:<18} ungated n={a_all['n']:>3} {a_all['mu']:>6.3f} pts t {a_all['t']:>5.2f}"
          f"  |  GATED n={a_g['n']:>3} {a_g['mu']:>6.3f} pts Sharpe {a_g['sra']:>5.2f}"
          f" t {a_g['t']:>5.2f}  rot p={float((mus>=obs).mean()):.3f}  CI [{lo:.2f}, {hi_:.2f}]")
    for yr in (2025, 2026):
        m = (g.date.dt.year == yr).to_numpy() & gt
        if m.sum() > 5:
            b = st(x[m]); print(f"      {yr}: n={b['n']:>3} {b['mu']:>6.3f} pts t {b['t']:>5.2f}")
