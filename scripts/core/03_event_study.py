#!/usr/bin/env python3
"""
03_event_study.py -- Conditional vs unconditional forward move around a
day-scoped Donchian breakout.

No simulator, no overlapping-trade bookkeeping, no exit rule: just the
question "given the index has just broken its own N-bar intraday channel,
what does it do next, relative to what it does at that time of day
anyway?"

For a breakout bar closing at t with direction s (+1 up, -1 down):

    raw(h)    = s * (close[t+h] - close[t])
    bench(h)  = s * mean over ALL bars in the same 30-minute time-of-day
                bucket of (close[t+h] - close[t])
    excess(h) = raw(h) - bench(h)

`bench` absorbs index drift and intraday time-of-day effects, so a positive
`excess` is the breakout adding information and a negative one is the
breakout being a worse-than-average moment to take that side.

Inference is day-clustered (whole sessions resampled), because breakouts on
the same session are not independent draws.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.core import STORE, load_spot, to_5m, donchian_signals

NS = [6, 9, 12, 18, 24, 36]
HORIZONS = [5, 15, 30, 60, 120]
PERIODS = {"2024 design": (2024, 2024), "2025 in-sample": (2025, 2025), "2026 holdout": (2026, 2026)}


def day_cluster_ci(df, col, n_boot=2000, seed=11):
    rng = np.random.default_rng(seed)
    groups = [g[col].to_numpy() for _, g in df.groupby("date")]
    k = len(groups)
    if k < 5:
        return (np.nan, np.nan)
    m = np.empty(n_boot)
    for i in range(n_boot):
        m[i] = np.concatenate([groups[j] for j in rng.integers(0, k, k)]).mean()
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    spot = load_spot()
    spot["year"] = pd.to_datetime(spot["date"]).dt.year
    # minute-indexed close lookup for forward moves
    px = spot.set_index("ts")["c"]

    rows = []
    for pname, (y0, y1) in PERIODS.items():
        sub = spot[(spot["year"] >= y0) & (spot["year"] <= y1)].reset_index(drop=True)
        bars5 = to_5m(sub)
        bars5["tod"] = bars5["bar_close_ts"].dt.floor("30min").dt.time.astype(str)

        # forward moves from every bar close, for the benchmark
        base = bars5[["date", "bar_close_ts", "tod", "c"]].copy()
        for h in HORIZONS:
            fwd_ts = base["bar_close_ts"] + pd.Timedelta(minutes=h)
            base[f"m{h}"] = fwd_ts.map(px).to_numpy() - base["c"].to_numpy()
        bench = base.groupby("tod")[[f"m{h}" for h in HORIZONS]].mean()

        for n in NS:
            sig = donchian_signals(bars5, n)
            ev = sig[sig["sig"] != 0].copy()
            if ev.empty:
                continue
            ev["tod"] = ev["bar_close_ts"].dt.floor("30min").dt.time.astype(str)
            for h in HORIZONS:
                fwd_ts = ev["bar_close_ts"] + pd.Timedelta(minutes=h)
                move = fwd_ts.map(px).to_numpy() - ev["c"].to_numpy()
                s = ev["sig"].to_numpy()
                b = ev["tod"].map(bench[f"m{h}"]).to_numpy()
                d = ev[["date"]].copy()
                d["raw"] = s * move
                d["excess"] = s * (move - b)
                d = d.dropna()
                if len(d) < 30:
                    continue
                lo, hi = day_cluster_ci(d, "excess")
                rows.append({
                    "period": pname, "n": n, "h": h, "events": int(len(d)),
                    "raw_mean": round(float(d["raw"].mean()), 2),
                    "excess_mean": round(float(d["excess"].mean()), 2),
                    "ci_lo": round(lo, 2), "ci_hi": round(hi, 2),
                    "sig": "yes" if (lo > 0 or hi < 0) else "no",
                    "hit_raw": round(float((d["raw"] > 0).mean()), 3),
                })
        print(f"-- {pname}: {sub['date'].nunique()} sessions", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(STORE / "event_study.csv", index=False)
    for pname in PERIODS:
        p = out[out["period"] == pname]
        if p.empty:
            continue
        print(f"\n=== {pname} : excess move (pts) after breakout, vs same time-of-day baseline ===")
        piv = p.pivot(index="n", columns="h", values="excess_mean")
        print(piv.to_string())
        print("  significant cells (day-clustered 95% CI excludes 0):")
        s = p[p["sig"] == "yes"]
        if s.empty:
            print("    none")
        else:
            for _, r in s.iterrows():
                print(f"    n={r['n']:>2} h={r['h']:>3}m  excess={r['excess_mean']:>7.2f} "
                      f"[{r['ci_lo']:>7.2f},{r['ci_hi']:>7.2f}]  events={r['events']}")
    print("\nwrote", STORE / "event_study.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
