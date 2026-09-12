#!/usr/bin/env python3
"""
07_analyse_straddle.py -- Score the straddle observations by channel-width
state.  Quintile cut points are fitted on 2025 ONLY and applied verbatim to
2026, so the holdout never informs its own bucketing.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.core import STORE
from lib.costs import LOT


def dcci(df, col, n_boot=2000, seed=17):
    rng = np.random.default_rng(seed)
    gs = [g[col].to_numpy() for _, g in df.groupby("date")]
    k = len(gs)
    if k < 8:
        return (np.nan, np.nan)
    m = np.empty(n_boot)
    for i in range(n_boot):
        m[i] = np.concatenate([gs[j] for j in rng.integers(0, k, k)]).mean()
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def block(tr, label, h):
    d = tr[tr["h"] == h]
    if d.empty:
        return
    print(f"\n  {label}   h = {h} min      ({d['date'].nunique()} sessions, {len(d)} straddles)")
    print("   wq   n     premium  spot_move  move/prem   gross(Rs)   net(Rs)   net 95% CI        hit")
    for q in sorted(d["wq"].dropna().unique()):
        g = d[d["wq"] == q]
        lo, hi = dcci(g, "net")
        print(f"   q{int(q)} {len(g):>5}  {g['premium'].mean():>8.1f} {g['spot_move'].mean():>9.1f}"
              f"  {(g['spot_move']/g['premium']).mean():>9.3f}"
              f"  {g['gross'].mean():>9.1f}  {g['net'].mean():>8.1f}"
              f"  [{lo:>7.1f},{hi:>7.1f}]  {(g['net']>0).mean():>5.3f}")


def main():
    tr = pd.read_parquet(STORE / "straddle_trades.parquet").dropna(subset=["width_rel"])
    is_ = tr[tr["year"] == 2025]
    oos = tr[tr["year"] == 2026]
    cuts = np.quantile(is_["width_rel"], [0.2, 0.4, 0.6, 0.8])
    print(f"width_rel quintile cut points fitted on 2025: "
          f"{np.round(cuts, 3).tolist()}   (1.0 = a normal-width channel for that time of day)")
    for df in (tr,):
        df["wq"] = np.digitize(df["width_rel"], cuts)

    print("\n" + "=" * 96)
    print("LONG ATM WEEKLY STRADDLE, sorted by Donchian channel width at entry "
          "(q0 = narrowest, q4 = widest)")
    print("net = after full Zerodha round trip + 0.5 pt/side/leg slippage, 1 lot = 75")
    print("=" * 96)
    for h in [15, 30, 60]:
        block(tr[tr["year"] == 2025], "2025 IN-SAMPLE ", h)
    for h in [15, 30, 60]:
        block(tr[tr["year"] == 2026], "2026 HOLDOUT   ", h)

    print("\n" + "=" * 96)
    print("Same cut, restricted to bars that were an actual DONCHIAN BREAKOUT")
    print("=" * 96)
    for yr, lab in [(2025, "2025 IN-SAMPLE "), (2026, "2026 HOLDOUT   ")]:
        b = tr[(tr["year"] == yr) & (tr["is_breakout"] == 1)]
        block(b, lab + "breakout", 30)

    print("\n" + "=" * 96)
    print("The other side of the same coin: SHORT the straddle in the narrow state")
    print("=" * 96)
    for yr, lab in [(2025, "2025"), (2026, "2026")]:
        d = tr[(tr["year"] == yr) & (tr["h"] == 30)]
        for q in [0, 1]:
            g = d[d["wq"] == q].copy()
            if g.empty:
                continue
            g["short_net"] = -g["gross"] - g["cost"]
            lo, hi = dcci(g, "short_net")
            print(f"   {lab} q{q}  n={len(g):>5}  short net = Rs {g['short_net'].mean():>7.1f}"
                  f"  [{lo:>7.1f},{hi:>7.1f}]   hit {(g['short_net']>0).mean():.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
