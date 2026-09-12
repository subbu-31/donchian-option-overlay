#!/usr/bin/env python3
"""
25_is_oos_a_regime.py -- Is the out-of-sample failure a regime break, or is
77 sessions simply too short to measure a tail-driven strategy?

2026's entire annual loss is five sessions; drop them and the year is positive
at +Rs 920/session, with a median session (+1,857) indistinguishable from 2025's
(+1,885) and a win rate within 1.3 points.  That is the signature of a sampling
problem, not a regime break -- but it has to be tested, not asserted.

The test: take the 2025 P&L series, which is the "good" regime by construction,
and ask how often a 77-session slice of it is as bad as 2026.

  iid draw      77 sessions sampled without replacement -- ignores clustering
  block draw    every contiguous 77-session window in 2025 -- preserves the
                clustering of volatile days, which is what actually matters

If a large share of 2025's own 77-session windows are negative, then 2026's
negative result is not evidence of anything having changed.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.core import STORE

N_OOS = 77
RNG = np.random.default_rng(2026)


def report(name, pnl25, target, n_oos=N_OOS):
    print(f"\n  {name}")
    print("  " + "-" * 92)
    iid = np.array([RNG.choice(pnl25, n_oos, replace=False).sum() for _ in range(20000)])
    blocks = np.array([pnl25[i:i + n_oos].sum() for i in range(len(pnl25) - n_oos + 1)])
    for label, arr in (("iid 77-session draws from 2025", iid),
                       ("contiguous 77-session windows in 2025", blocks)):
        neg = 100 * (arr < 0).mean()
        worse = 100 * (arr <= target).mean()
        print(f"    {label:<40} n={len(arr):>6}  median Rs {np.median(arr):>10,.0f}"
              f"  P(window < 0) = {neg:>5.1f}%   P(window <= 2026's result) = {worse:>5.1f}%")
    print(f"    2026 actual: Rs {target:,.0f} over {n_oos} sessions")
    return blocks


def main():
    out = {}
    for cfg_name, path, col in (("condor 0.35/0.15", "intraday_wings.csv", "net"),):
        w = pd.read_csv(STORE / path)
        w["year"] = pd.to_datetime(w["date"]).dt.year
        print("=" * 96)
        print("HOW OFTEN IS A 77-SESSION WINDOW OF THE *GOOD* YEAR AS BAD AS THE HOLDOUT?")
        print("=" * 96)
        for cfg in ("0.35/0.15", "0.25/0.1", "0.25/0.05"):
            g = w[w.cfg == cfg]
            p25 = g[g.year == 2025].sort_values("date")["net"].to_numpy()
            p26 = g[g.year == 2026]["net"].sum()
            if len(p25) < 150:
                continue
            out[cfg] = report(f"condor {cfg}", p25, p26)

    print("\n" + "=" * 96)
    print("READING")
    print("=" * 96)
    print("  If 'P(window < 0)' is large, then a 77-session slice of the year the strategy")
    print("  WORKED is routinely negative, and a single negative 77-session holdout carries")
    print("  almost no information about whether the edge is gone.")
    print("\n  The honest conclusion this supports: the Jan-Apr 2026 holdout is too short to")
    print("  reject the strategy. It is long enough to reject a CLAIM of reliable alpha, which")
    print("  is why the write-ups still lead with the negative result -- but it is not evidence")
    print("  that the 2025 edge disappeared.")

    # how many sessions would actually be needed
    print("\n" + "=" * 96)
    print("HOW LONG A HOLDOUT WOULD ACTUALLY BE NEEDED")
    print("=" * 96)
    w = pd.read_csv(STORE / "intraday_wings.csv")
    w["year"] = pd.to_datetime(w["date"]).dt.year
    for cfg in ("0.35/0.15", "0.25/0.05"):
        g = w[(w.cfg == cfg) & (w.year == 2025)]["net"].to_numpy()
        mu, sd = g.mean(), g.std(ddof=1)
        n80 = (2.8 * sd / mu) ** 2 if mu > 0 else np.nan          # 80% power, 5% two-sided
        print(f"  {cfg:<11} 2025 mean Rs {mu:>7,.0f}/session, sd Rs {sd:>8,.0f}  "
              f"-> ~{n80:,.0f} sessions to detect that mean is > 0 at 80% power")
    print("\n  A strategy whose per-session edge is this small relative to its own dispersion")
    print("  cannot be validated on one quarter of data, in either direction.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
