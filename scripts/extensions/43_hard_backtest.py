#!/usr/bin/env python3
"""
43_hard_backtest.py -- Try to destroy the high-trailing-vol gate.

Five tests, each chosen because it is harsher than what the gate has already
passed.  Nothing here is designed to confirm anything.

T1  BLOCK-PRESERVING PLACEBO.  The earlier placebo drew a random iid subset of
    sessions.  That is far too easy a null: rv20 is highly persistent, so gated
    sessions arrive in long clusters, and any clustered subset of a P&L series
    with serial structure will look unusual against an iid null.  The honest
    null rotates the gate label series circularly against the P&L series --
    every rotation preserves the gate's clustering EXACTLY and destroys only
    the alignment.  This is an exact test over all n-1 rotations.

T2  DEFLATED SHARPE RATIO (Bailey & Lopez de Prado 2014).  Corrects the observed
    Sharpe for the number of trials, the skew and the kurtosis of the return
    stream.  Condor P&L is violently left-skewed, which the plain Lo(2002)
    t-stat ignores and which inflates it.

T3  PBO via CSCV (Bailey, Borwein, Lopez de Prado, Zhu 2017).  Take the whole
    grid of gate variants, split the sample into S blocks, and over every
    balanced in-sample/out-of-sample partition ask where the IS-best variant
    ranks OOS.  PBO is the fraction of partitions where it lands below median.

T4  LIVE LOOKBACK SELECTION.  The L=20 lookback was chosen with hindsight and
    is the known fragility.  Here the lookback is re-chosen at every step from
    past data only, then traded forward.  No parameter is ever known early.

T5  COST AND EXECUTION STRESS + stationary block bootstrap CI.
"""
import sys
from math import erf, sqrt, log, exp
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.core import STORE

QTY = 750
EULER = 0.5772156649
RNG = np.random.default_rng(20260904)

LOOKBACKS = [10, 20, 30, 40]
QUANTS = [0.50, 0.60, 0.6667, 0.75]
WINDOWS = [("expanding", None), ("roll250", 250)]
CFGS = ["0.35/0.15", "0.25/0.1", "0.25/0.05"]


def ncdf(x):
    return 0.5 * (1 + erf(x / sqrt(2)))


def nppf(p):
    # Acklam inverse normal, plenty accurate here
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    dd = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
          3.754408661907416e+00]
    pl, ph = 0.02425, 1 - 0.02425
    if p < pl:
        q = sqrt(-2 * log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((dd[0]*q+dd[1])*q+dd[2])*q+dd[3])*q+1)
    if p > ph:
        q = sqrt(-2 * log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((dd[0]*q+dd[1])*q+dd[2])*q+dd[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def sr_stats(x):
    x = np.asarray(x, float)
    n = len(x)
    sd = x.std(ddof=1)
    if n < 5 or sd == 0:
        return dict(n=n, sr=np.nan, t=np.nan, skew=np.nan, kurt=np.nan)
    sr = x.mean() / sd
    z = (x - x.mean()) / sd
    g3 = float((z ** 3).mean())
    g4 = float((z ** 4).mean())
    t = sr / np.sqrt((1 + 0.5 * sr ** 2) / n)
    return dict(n=n, sr=sr, sr_a=sr * np.sqrt(252), t=t, skew=g3, kurt=g4, mu=x.mean())


def deflated_sharpe(x, sr_trials, n_trials):
    """DSR: P(true SR > 0) given the observed SR, its higher moments, and the
    expected maximum SR attainable by chance over n_trials independent tries."""
    s = sr_stats(x)
    if not np.isfinite(s["sr"]):
        return np.nan, np.nan
    v = np.nanstd(np.asarray(sr_trials, float), ddof=1)
    if not np.isfinite(v) or v == 0 or n_trials < 2:
        sr0 = 0.0
    else:
        sr0 = v * ((1 - EULER) * nppf(1 - 1.0 / n_trials)
                   + EULER * nppf(1 - 1.0 / (n_trials * exp(1))))
    n, sr, g3, g4 = s["n"], s["sr"], s["skew"], s["kurt"]
    den = sqrt(max(1e-12, 1 - g3 * sr + (g4 - 1) / 4.0 * sr ** 2))
    z = (sr - sr0) * sqrt(n - 1) / den
    return ncdf(z), sr0


def make_gate(daily, L, q, win, keep):
    """Gate is built on the FULL daily vol series (every session in the spot
    history), then aligned to the traded sessions.  Building it on the traded
    subset only would make rolling(L) span L TRADED sessions, a different and
    much longer window, and would burn the whole warm-up on the first year."""
    v = daily["rv"].shift(1).rolling(L).mean()
    if win is None:
        thr = v.expanding(120).quantile(q).shift(1)
    else:
        thr = v.rolling(win, min_periods=120).quantile(q).shift(1)
    hit = (v >= thr) & thr.notna()
    m = pd.DataFrame({"date": daily["date"], "hit": hit.fillna(False)})
    return keep.merge(m, on="date", how="left")["hit"].fillna(False).to_numpy(bool)


def main():
    f = pd.read_csv(STORE / "regime_decomp.csv")[["date", "rv"]]
    f["date"] = pd.to_datetime(f["date"])
    f = f.dropna().sort_values("date").reset_index(drop=True)

    w = pd.read_csv(STORE / "intraday_wings_costfix.csv")
    if "stat_new" in w.columns:
        w["stat"] = w["stat_new"]
    w["date"] = pd.to_datetime(w["date"])
    w["net"] = w["gross"] - w["slip"] - w["stat"]

    # ---------- build the full variant grid on a common session index ----------
    keep = w[w.cfg == CFGS[0]][["date"]].sort_values("date").reset_index(drop=True)
    variants, labels = {}, []
    gates = {}
    for L in LOOKBACKS:
        for q in QUANTS:
            for wn, wv in WINDOWS:
                gates[(L, q, wn)] = make_gate(f, L, q, wv, keep)
    for cfg in CFGS:
        net = (w[w.cfg == cfg].sort_values("date")["net"].to_numpy(float))
        for key, g in gates.items():
            if g.sum() < 25:
                continue
            variants[(cfg,) + key] = np.where(g, net, 0.0)
            labels.append((cfg,) + key)
    n_sess = len(keep)
    print(f"grid: {len(labels)} gate variants over {n_sess} sessions "
          f"({len(LOOKBACKS)} lookbacks x {len(QUANTS)} quantiles x "
          f"{len(WINDOWS)} windows x {len(CFGS)} wing configs)\n")

    HEAD = [("0.35/0.15", 20, 0.6667, "expanding"),
            ("0.25/0.05", 20, 0.6667, "expanding"),
            ("0.35/0.15", 20, 0.6667, "roll250"),
            ("0.25/0.05", 20, 0.6667, "roll250")]

    # ================= T1 : block-preserving rotation placebo =================
    print("=" * 78)
    print("T1  CIRCULAR-ROTATION PLACEBO  (preserves the gate's clustering exactly)")
    print("=" * 78)
    for cfg, L, q, wn in HEAD:
        net = w[w.cfg == cfg].sort_values("date")["net"].to_numpy(float)
        g = gates[(L, q, wn)]
        k = int(g.sum())
        obs_mu = net[g].mean()
        obs_sr = sr_stats(net[g])["sr"]
        mus, srs = [], []
        for sh in range(1, n_sess):
            gg = np.roll(g, sh)
            x = net[gg]
            mus.append(x.mean())
            srs.append(x.mean() / x.std(ddof=1) if x.std(ddof=1) > 0 else np.nan)
        mus, srs = np.array(mus), np.array(srs)
        p_mu = float((mus >= obs_mu).mean())
        p_sr = float((np.nan_to_num(srs, nan=-9) >= obs_sr).mean())
        # the iid placebo, for direct comparison
        iid = np.array([RNG.choice(net, k, replace=False).mean() for _ in range(20000)])
        print(f"  {cfg:<10} L={L} q={q} {wn:<9} n={k:>3}  Rs/sess {obs_mu:>7,.0f}")
        print(f"      rotation placebo  p(mean) = {p_mu:>6.3f}   p(Sharpe) = {p_sr:>6.3f}"
              f"   [{n_sess-1} rotations]")
        print(f"      iid    placebo  p(mean) = {(iid >= obs_mu).mean():>6.3f}"
              f"   <-- the easy null used earlier")
        print(f"      rotation dist: mean {mus.mean():>7,.0f}  sd {mus.std():>7,.0f}"
              f"  max {mus.max():>7,.0f}")

    # ================= T2 : deflated Sharpe =================
    print("\n" + "=" * 78)
    print("T2  DEFLATED SHARPE RATIO   (trials, skew and kurtosis corrected)")
    print("=" * 78)
    sr_trials = [sr_stats(v)["sr"] for v in variants.values()]
    sr_trials = [s for s in sr_trials if np.isfinite(s)]
    for cfg, L, q, wn in HEAD:
        net = w[w.cfg == cfg].sort_values("date")["net"].to_numpy(float)
        x = net[gates[(L, q, wn)]]
        s = sr_stats(x)
        for nt in (len(sr_trials), 12):
            dsr, sr0 = deflated_sharpe(x, sr_trials, nt)
            tag = f"grid N={nt}" if nt == len(sr_trials) else f"tests-run N={nt}"
            print(f"  {cfg:<10} {wn:<9} SR_ann {s['sr_a']:>5.2f}  skew {s['skew']:>6.2f}"
                  f"  kurt {s['kurt']:>5.2f}  | {tag:<14} SR0 {sr0*np.sqrt(252):>5.2f}"
                  f"  DSR {dsr:>6.3f}{'  PASS' if dsr > 0.95 else '  fail'}")

    # ================= T3 : PBO via CSCV =================
    print("\n" + "=" * 78)
    print("T3  PROBABILITY OF BACKTEST OVERFITTING  (CSCV, S=10 blocks, 252 splits)")
    print("=" * 78)
    M = np.column_stack([variants[k] for k in labels])
    S = 10
    idx = np.array_split(np.arange(n_sess), S)
    logits = []
    for combo in combinations(range(S), S // 2):
        tr = np.concatenate([idx[i] for i in combo])
        te = np.concatenate([idx[i] for i in range(S) if i not in combo])
        A, B = M[tr], M[te]
        with np.errstate(invalid="ignore", divide="ignore"):
            sa = A.mean(0) / A.std(0, ddof=1)
            sb = B.mean(0) / B.std(0, ddof=1)
        sa = np.nan_to_num(sa, nan=-9)
        sb = np.nan_to_num(sb, nan=-9)
        best = int(np.argmax(sa))
        rank = (sb < sb[best]).sum() / len(sb)          # relative rank OOS, in (0,1)
        rank = min(max(rank, 1e-3), 1 - 1e-3)
        logits.append(log(rank / (1 - rank)))
    logits = np.array(logits)
    pbo = float((logits <= 0).mean())
    print(f"  splits {len(logits)}   PBO = {pbo:.3f}   median logit {np.median(logits):+.2f}")
    print("  reading: PBO is the chance the in-sample-best variant lands below the")
    print("           OOS median.  <0.10 is clean, >0.50 means the selection is noise.")

    # ================= T4 : live lookback selection =================
    print("\n" + "=" * 78)
    print("T4  WALK-FORWARD WITH THE LOOKBACK CHOSEN LIVE  (no parameter known early)")
    print("=" * 78)
    WARM = 150
    for cfg in ["0.35/0.15", "0.25/0.05"]:
        net = w[w.cfg == cfg].sort_values("date")["net"].to_numpy(float)
        for wn, wv in WINDOWS:
            traded, chosen = np.zeros(n_sess, bool), []
            for i in range(WARM, n_sess):
                best, bl = -1e18, None
                for L in LOOKBACKS:
                    g = gates[(L, 0.6667, wn)]
                    hist = net[:i][g[:i]]
                    if len(hist) < 20:
                        continue
                    sd = hist.std(ddof=1)
                    s = hist.mean() / sd if sd > 0 else -1e18
                    if s > best:
                        best, bl = s, L
                if bl is None:
                    continue
                traded[i] = gates[(bl, 0.6667, wn)][i]
                chosen.append(bl)
            x = net[traded]
            s = sr_stats(x) if len(x) > 5 else dict(n=len(x), mu=np.nan, sr_a=np.nan, t=np.nan)
            allx = sr_stats(net[WARM:])
            pick = pd.Series(chosen).value_counts().to_dict()
            print(f"  {cfg:<10} {wn:<9} traded {int(traded.sum())}/{n_sess-WARM}  "
                  f"Rs/sess {s['mu']:>7,.0f}  SR {s['sr_a']:>5.2f}  t {s['t']:>5.2f}"
                  f"  || ungated same span Rs/sess {allx['mu']:>7,.0f} t {allx['t']:>5.2f}")
            print(f"      lookback picked live: {pick}")

    # ================= T5 : cost stress + block bootstrap =================
    print("\n" + "=" * 78)
    print("T5  COST / EXECUTION STRESS  and  STATIONARY BLOCK BOOTSTRAP CI")
    print("=" * 78)
    for cfg, L, q, wn in HEAD[:2] + HEAD[2:]:
        d = w[w.cfg == cfg].sort_values("date")
        g = gates[(L, q, wn)]
        print(f"  {cfg:<10} {wn:<9} n={int(g.sum())}")
        for mult, lab in [(1.0, "modelled slip"), (2.0, "2x slip"), (3.0, "3x slip")]:
            x = (d.gross - mult * d.slip - d.stat).to_numpy(float)[g]
            s = sr_stats(x)
            print(f"      {lab:<15} Rs/sess {s['mu']:>7,.0f}  SR {s['sr_a']:>5.2f}  t {s['t']:>5.2f}")
        # miss the best entries: assume the k best gated sessions were unfillable
        x = np.sort((d.gross - d.slip - d.stat).to_numpy(float)[g])
        for k in (0, 3, 5):
            y = x[:len(x) - k] if k else x
            s = sr_stats(y)
            print(f"      miss best {k:<5} Rs/sess {s['mu']:>7,.0f}  SR {s['sr_a']:>5.2f}  t {s['t']:>5.2f}")
        # stationary block bootstrap, expected block 10
        base = (d.gross - d.slip - d.stat).to_numpy(float)[g]
        nb, p = len(base), 1 / 10.0
        boot = np.empty(10000)
        for bi in range(10000):
            out, j = [], RNG.integers(nb)
            while len(out) < nb:
                out.append(base[j % nb])
                j = RNG.integers(nb) if RNG.random() < p else j + 1
            boot[bi] = np.mean(out)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        print(f"      block bootstrap 95% CI on Rs/sess  [{lo:,.0f}, {hi:,.0f}]"
              f"   P(mean>0) = {(boot > 0).mean():.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
