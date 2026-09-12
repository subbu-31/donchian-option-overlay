#!/usr/bin/env python3
"""
44_walkforward.py -- Quarterly rolling walk-forward. Nothing known early.

PROTOCOL
  At the first session of each calendar quarter Q, using ONLY sessions that
  closed strictly before Q begins:
      * score every candidate on the training history
      * freeze the winner
      * trade quarter Q with it, unchanged, whatever happens
  Then roll forward.  A quarter's P&L is never used to choose that quarter's
  parameters, and no parameter is ever re-touched inside a quarter.

CANDIDATE SPACE (the selector may also choose NOT to gate)
  wing cfg  x  vol lookback {10,20,30,40}  x  quantile {.50,.60,.6667,.75}
            x  threshold window {expanding, rolling-250}      + "trade everything"

The gate series itself is already causal by construction: rv20 is
rv.shift(1).rolling(L).mean() and the threshold is an expanding or rolling
quantile with .shift(1), so a gate value on day d uses only days < d.  The only
place look-ahead could enter is SELECTION, which this protocol removes.

ARMS REPORTED
  WF-SELECT   full live selection, the honest test
  FIXED       one config fixed a priori (0.25/0.10, L=20, q=.6667, expanding)
              and simply walked forward -- separates "does the gate work" from
              "can the parameters be picked live"
  ALL         trade every session, no gate
  ORACLE      best single config chosen with hindsight over the whole OOS span,
              reported only to size the selection gap
"""
import sys
from math import erf, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.core import STORE

RNG = np.random.default_rng(20260904)
LOOKBACKS, QUANTS = [10, 20, 30, 40], [0.50, 0.60, 0.6667, 0.75]
WINDOWS = [("exp", None), ("r250", 250)]
MIN_TRAIN_GATED = 20


def ncdf(x):
    return 0.5 * (1 + erf(x / sqrt(2)))


def stats(x):
    x = np.asarray(x, float)
    n = len(x)
    if n < 5 or x.std(ddof=1) == 0:
        return dict(n=n, tot=np.nansum(x), mu=np.nan, sr=np.nan, t=np.nan, p=np.nan)
    sr = x.mean() / x.std(ddof=1)
    t = sr / np.sqrt((1 + 0.5 * sr ** 2) / n)
    return dict(n=n, tot=x.sum(), mu=x.mean(), sr=sr * np.sqrt(252), t=t,
                p=2 * (1 - ncdf(abs(t))))


def boot_ci(x, nb=10000, blk=10):
    x = np.asarray(x, float)
    n = len(x)
    if n < 10:
        return np.nan, np.nan, np.nan
    out = np.empty(nb)
    for i in range(nb):
        s, j = [], RNG.integers(n)
        while len(s) < n:
            s.append(x[j % n])
            j = RNG.integers(n) if RNG.random() < 1.0 / blk else j + 1
        out[i] = np.mean(s)
    return np.percentile(out, 2.5), np.percentile(out, 97.5), float((out > 0).mean())


def build(daily, keep):
    """All gate boolean series, aligned to traded sessions. Causal by construction."""
    g = {}
    for L in LOOKBACKS:
        v = daily["rv"].shift(1).rolling(L).mean()
        for q in QUANTS:
            for wn, wv in WINDOWS:
                thr = (v.expanding(120).quantile(q).shift(1) if wv is None
                       else v.rolling(wv, min_periods=120).quantile(q).shift(1))
                hit = ((v >= thr) & thr.notna()).fillna(False)
                m = pd.DataFrame({"date": daily["date"], "hit": hit})
                g[(L, q, wn)] = keep.merge(m, on="date", how="left")["hit"].fillna(False).to_numpy(bool)
    return g


def walkforward(name, daily, trades, cfgs, fixed, crit="sharpe", freq="Q", min_train=150):
    # configs do not always cover identical dates (a leg can be missing), so the
    # walk-forward runs on the INTERSECTION -- otherwise the boolean gate and the
    # P&L vectors silently misalign by a few sessions
    common = None
    for c in cfgs:
        s_ = set(trades.loc[trades.cfg == c, "date"])
        common = s_ if common is None else (common & s_)
    trades = trades[trades["date"].isin(common)]
    keep = (trades[trades.cfg == cfgs[0]][["date"]].drop_duplicates()
            .sort_values("date").reset_index(drop=True))
    gates = build(daily, keep)
    nets = {c: trades[trades.cfg == c].sort_values("date")["net"].to_numpy(float) for c in cfgs}
    for c in cfgs:
        assert len(nets[c]) == len(keep), f"{c}: {len(nets[c])} vs {len(keep)}"
    dates = pd.DatetimeIndex(keep["date"])
    n = len(dates)
    period = pd.Series(dates.to_period(freq))
    periods = sorted(period.unique())

    cands = [("ALL", c, None) for c in cfgs]
    for c in cfgs:
        for k in gates:
            cands.append(("GATE", c, k))

    def sel_score(mask, cfg, key):
        x = nets[cfg][mask] if key is None else nets[cfg][mask & gates[key]]
        if len(x) < (5 if key is None else MIN_TRAIN_GATED):
            return -1e18
        sd = x.std(ddof=1)
        if sd == 0:
            return -1e18
        return x.mean() / sd if crit == "sharpe" else x.mean()

    rows = []
    for p in periods:
        te = (period == p).to_numpy()
        tr = (period < p).to_numpy()
        if tr.sum() < min_train or te.sum() < 5:
            continue
        best, bs = None, -1e18
        for kind, cfg, key in cands:
            s = sel_score(tr, cfg, key)
            if s > bs:
                bs, best = s, (kind, cfg, key)
        kind, cfg, key = best
        oos_mask = te if key is None else (te & gates[key])
        wf = nets[cfg][oos_mask]
        fx = nets[fixed[0]][te & gates[(fixed[1], fixed[2], fixed[3])]]
        al = nets[fixed[0]][te]
        rows.append(dict(
            period=str(p), n_sess=int(te.sum()),
            pick=f"{cfg} {'ALL' if key is None else f'L{key[0]} q{key[1]:.2f} {key[2]}'}",
            wf_n=len(wf), wf=float(np.sum(wf)),
            fx_n=len(fx), fx=float(np.sum(fx)), al=float(np.sum(al)),
            wf_ser=wf, fx_ser=fx, al_ser=al))

    print(f"\n{'='*86}\n{name}  --  {freq}-ly walk-forward, selection criterion = {crit}\n{'='*86}")
    print(f"  {'period':<9}{'sess':>5}{'  selected config (from PAST data only)':<44}"
          f"{'WF n':>6}{'WF Rs':>11}{'FIXED Rs':>11}{'ALL Rs':>11}")
    for r in rows:
        print(f"  {r['period']:<9}{r['n_sess']:>5}  {r['pick']:<42}"
              f"{r['wf_n']:>6}{r['wf']:>11,.0f}{r['fx']:>11,.0f}{r['al']:>11,.0f}")
    if not rows:
        print("  no period had enough training history")
        return None

    print(f"\n  OOS quarters: {len(rows)}   <-- read every number below against this")
    agg = {}
    for k, lab in [("wf_ser", "WF-SELECT"), ("fx_ser", "FIXED a priori"), ("al_ser", "ALL (no gate)")]:
        x = np.concatenate([r[k] for r in rows]) if rows else np.array([])
        s = stats(x)
        lo, hi, pp = boot_ci(x)
        agg[lab] = s
        print(f"  {lab:<15} n={s['n']:>4}  total {s['tot']:>10,.0f}  Rs/sess {s['mu']:>8,.0f}"
              f"  SR {s['sr']:>6.2f}  t {s['t']:>5.2f}  p {s['p']:>5.3f}"
              f"  boot CI [{lo:,.0f}, {hi:,.0f}] P>0 {pp:.3f}")
    # quarter-level sign test on the gated vs ungated arms
    d = np.array([r["fx"] - r["al"] for r in rows])
    print(f"  FIXED beats ALL in {int((d > 0).sum())}/{len(d)} quarters "
          f"(mean quarterly difference Rs {d.mean():,.0f})")
    w = np.array([r["wf"] - r["al"] for r in rows])
    print(f"  WF-SELECT beats ALL in {int((w > 0).sum())}/{len(w)} quarters "
          f"(mean quarterly difference Rs {w.mean():,.0f})")

    # oracle gap
    span = np.zeros(n, bool)
    for r in rows:
        span |= (period.astype(str) == r["period"]).to_numpy()
    bo, bn = -1e18, None
    for kind, cfg, key in cands:
        x = nets[cfg][span] if key is None else nets[cfg][span & gates[key]]
        if len(x) < 20 or x.std(ddof=1) == 0:
            continue
        v = x.mean() / x.std(ddof=1)
        if v > bo:
            bo, bn = v, (cfg, key, x)
    if bn:
        s = stats(bn[2])
        lab = 'ALL' if bn[1] is None else f'L{bn[1][0]} q{bn[1][1]:.2f} {bn[1][2]}'
        print(f"  ORACLE (hindsight)  {bn[0]} {lab}"
              f"  n={s['n']}  Rs/sess {s['mu']:,.0f}  SR {s['sr']:.2f}"
              f"   <-- selection gap vs WF-SELECT: Rs {s['mu']-agg['WF-SELECT']['mu']:,.0f}/session")
    return rows


def main():
    # ---------------- NIFTY ----------------
    f = pd.read_csv(STORE / "regime_decomp.csv")[["date", "rv"]]
    f["date"] = pd.to_datetime(f["date"])
    f = f.dropna().sort_values("date").reset_index(drop=True)
    w = pd.read_csv(STORE / "intraday_wings_costfix.csv")
    if "stat_new" in w.columns:
        w["stat"] = w["stat_new"]
    w["date"] = pd.to_datetime(w["date"])
    w["net"] = w["gross"] - w["slip"] - w["stat"]
    NCFG = ["0.35/0.15", "0.25/0.1", "0.25/0.05"]
    FIXED_N = ("0.25/0.1", 20, 0.6667, "exp")
    for freq in ["Q", "M"]:
        for crit in (["sharpe", "mean"] if freq == "Q" else ["sharpe"]):
            walkforward("NIFTY", f, w, NCFG, FIXED_N, crit=crit, freq=freq)

    # ---------------- BANKNIFTY ----------------
    rv = pd.read_csv(STORE / "bn_daily_rv.csv")
    rv["date"] = pd.to_datetime(rv["date"])
    rv = rv.sort_values("date").reset_index(drop=True)
    b = pd.read_csv(STORE / "banknifty_condor.csv")
    b["date"] = pd.to_datetime(b["date"])
    BCFG = ["0.35/0.15", "0.25/0.1"]
    walkforward("BANKNIFTY", rv, b, BCFG, ("0.25/0.1", 20, 0.6667, "r250"),
                crit="sharpe", freq="Q", min_train=60)
    walkforward("BANKNIFTY", rv, b, BCFG, ("0.25/0.1", 20, 0.6667, "r250"),
                crit="sharpe", freq="Y", min_train=60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
