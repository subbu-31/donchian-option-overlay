#!/usr/bin/env python3
"""
39_vol_regime_gate.py -- The one conditioner with both-period sign consistency.

The regime decomposition found Normal-vol sessions were the condor's WORST in
BOTH 2025 (-Rs 8,057/day) and 2026 (-Rs 7,149/day), with Stressed the best in
both.  Every other conditioner tested flipped sign between periods.  That makes
this the only candidate worth a deployment test.

Gate is made STRICTLY CAUSAL and parameter-free: today's regime is the tercile
its trailing-20-session realised vol falls in, measured against the EXPANDING
history of rv20 up to yesterday.  No cut points are fitted on any sample.

Reported: trade-everything vs skip-Normal, split IS 2025 / OOS 2026, with a
label-permutation placebo and the tail budget a live account would need.
"""
import sys
from math import erf, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.core import STORE

QTY = 750
MINHIST = 120
NPERM = 5000
RNG = np.random.default_rng(20260904)


def ncdf(x):
    return 0.5 * (1 + erf(x / sqrt(2)))


def score(x):
    x = np.asarray(x, float)
    n = len(x)
    if n < 5 or x.std(ddof=1) == 0:
        return dict(n=n, mu=np.nan, sr=np.nan, t=np.nan, p=np.nan, tot=np.nan)
    sr = x.mean() / x.std(ddof=1)
    t = sr / np.sqrt((1 + 0.5 * sr ** 2) / n)
    return dict(n=n, mu=x.mean(), sr=sr * np.sqrt(252), t=t,
                p=2 * (1 - ncdf(abs(t))), tot=x.sum())


def line(tag, s):
    print(f"    {tag:<22} n={s['n']:>4}  Rs/sess {s['mu']:>8,.0f}  "
          f"total {s['tot']:>10,.0f}  SR {s['sr']:>6.2f}  t {s['t']:>5.2f}  p {s['p']:>5.3f}")


def main():
    r = pd.read_csv(STORE / "regime_decomp.csv")[["date", "rv20"]]
    r["date"] = pd.to_datetime(r["date"])
    r = r.dropna().sort_values("date").reset_index(drop=True)

    # expanding, causal terciles: where does today's rv20 sit in history to date?
    lo = r["rv20"].expanding(MINHIST).quantile(0.3333).shift(1)
    hi = r["rv20"].expanding(MINHIST).quantile(0.6667).shift(1)
    r["reg"] = np.where(r.rv20 <= lo, "Quiet",
                np.where(r.rv20 >= hi, "Stressed", "Normal"))
    r.loc[lo.isna(), "reg"] = None

    w = pd.read_csv(STORE / "intraday_wings_costfix.csv")
    if "stat_new" in w.columns:
        w["stat"] = w["stat_new"]
    w["date"] = pd.to_datetime(w["date"])
    w["net"] = w["gross"] - w["slip"] - w["stat"]

    for cfg in ["0.35/0.15", "0.25/0.05"]:
        m = w[w.cfg == cfg].merge(r, on="date", how="inner").dropna(subset=["reg"])
        m["year"] = m.date.dt.year
        print(f"===== {cfg}   merged n={len(m)}   "
              f"regime mix {dict(m.reg.value_counts())} =====")

        for label, g in [("2025 IS", m[m.year == 2025]), ("2026 OOS", m[m.year == 2026]),
                         ("FULL", m)]:
            if len(g) < 30:
                continue
            print(f"  -- {label} --")
            line("trade everything", score(g.net))
            for reg in ["Quiet", "Normal", "Stressed"]:
                line(f"  {reg} only", score(g.loc[g.reg == reg, "net"]))
            gate = g.loc[g.reg != "Normal", "net"]
            line("SKIP Normal", score(gate))
            print(f"    sessions traded {len(gate)}/{len(g)} = {100*len(gate)/len(g):.0f}%")

        # placebo: does skipping a RANDOM 1/3 of sessions do as well?
        g = m[m.year == 2026]
        if len(g) > 40:
            obs = g.loc[g.reg != "Normal", "net"].mean()
            k = int((g.reg != "Normal").sum())
            net = g.net.to_numpy()
            hits = sum(1 for _ in range(NPERM)
                       if RNG.choice(net, k, replace=False).mean() >= obs)
            print(f"  placebo 2026: skipping a random {len(g)-k} sessions beats "
                  f"Rs {obs:,.0f}/session in {100*hits/NPERM:.1f}% of {NPERM} draws")

        # tail budget a live account has to survive
        print(f"  tail budget, FULL sample {cfg}:")
        s = m.net.sort_values()
        print(f"    worst 5 sessions {', '.join(f'{v:,.0f}' for v in s.head(5))}")
        print(f"    5th pct {np.percentile(m.net,5):,.0f}   1st pct {np.percentile(m.net,1):,.0f}"
              f"   mean {m.net.mean():,.0f}   modelled max loss {m.maxloss.mean():,.0f}")
        eq = m.sort_values("date").net.cumsum()
        print(f"    worst peak-to-trough drawdown Rs {(eq - eq.cummax()).min():,.0f}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
