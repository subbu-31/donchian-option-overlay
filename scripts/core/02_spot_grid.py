#!/usr/bin/env python3
"""
02_spot_grid.py -- Does the day-scoped Donchian breakout have ANY directional
edge in NIFTY spot points, before any option is bought?

Design sample is calendar 2024 only.  2025 (option in-sample) and 2026
(holdout) are not touched here, so nothing chosen from this grid can have
been chosen with knowledge of them.

Every cell is scored against a matched random-entry control: same day, same
number of trades, same direction, same exit rule, entry minute drawn
uniformly from the same window.  That control absorbs index drift and the
day's realised volatility, so what is left is the value of the breakout
TIMING itself.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.core import (STORE, load_spot, to_5m, donchian_signals, dedupe_signals,
                      simulate, prep_days, summarise, block_bootstrap_ci)

DESIGN_YEARS = [2024]
NS = [6, 9, 12, 18, 24, 36]
HOLDS = [15, 30, 60, 120, 10 ** 6]     # minutes; last = hold to the 15:15 flat
MAX_PER_DAY = 3
N_CONTROL = 25
RNG = np.random.default_rng(20240101)


def control_entries(entries, rng, window=("09:45", "15:00")):
    """Same trades, same days, same directions -- random entry minutes."""
    e = entries.copy()
    lo = pd.Timestamp(window[0]).time()
    hi = pd.Timestamp(window[1]).time()
    day = pd.to_datetime(e["date"])
    lo_m = lo.hour * 60 + lo.minute
    hi_m = hi.hour * 60 + hi.minute
    draw = rng.integers(lo_m, hi_m + 1, len(e))
    e["bar_close_ts"] = day + pd.to_timedelta(draw - 1, unit="m")
    return e.sort_values("bar_close_ts").reset_index(drop=True)


def main():
    spot = load_spot()
    spot["year"] = pd.to_datetime(spot["date"]).dt.year
    design = spot[spot["year"].isin(DESIGN_YEARS)].reset_index(drop=True)
    bars5 = to_5m(design)
    days = prep_days(design)
    print(f"design sample: {design['date'].nunique()} sessions "
          f"{design['date'].min()}..{design['date'].max()}", flush=True)

    rows = []
    for n in NS:
        sig = donchian_signals(bars5, n)
        ent = dedupe_signals(sig, max_per_day=MAX_PER_DAY)
        if ent.empty:
            continue
        for hold in HOLDS:
            tr = simulate(ent, design, max_minutes=hold, days=days)
            if tr.empty:
                continue
            s = summarise(tr, label=f"n={n} hold={hold}")

            ctrl_means, ctrl_hits = [], []
            for r in range(N_CONTROL):
                ce = control_entries(ent, RNG)
                ct = simulate(ce, design, max_minutes=hold, days=days)
                if not ct.empty:
                    ctrl_means.append(ct["pnl"].mean())
                    ctrl_hits.append((ct["pnl"] > 0).mean())
            cm = float(np.mean(ctrl_means))
            cs = float(np.std(ctrl_means))
            s.update({
                "n_param": n, "hold": hold,
                "ctrl_mean": round(cm, 3),
                "ctrl_sd": round(cs, 3),
                "edge_vs_ctrl": round(s["mean"] - cm, 3),
                "z_vs_ctrl": round((s["mean"] - cm) / cs, 2) if cs > 0 else np.nan,
                "ctrl_hit": round(float(np.mean(ctrl_hits)), 3),
            })
            rows.append(s)
            print(f"  n={n:>2} hold={hold:>7}  n_tr={s['n']:>4}  mean={s['mean']:>7.2f}  "
                  f"ctrl={cm:>7.2f}  edge={s['edge_vs_ctrl']:>7.2f}  z={s['z_vs_ctrl']:>6}  "
                  f"hit={s['hit']:.3f} (ctrl {s['ctrl_hit']:.3f})", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(STORE / "spot_grid_2024.csv", index=False)
    print("\nwrote", STORE / "spot_grid_2024.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
