#!/usr/bin/env python3
"""
14_hedge_placebo.py -- Look-ahead placebo for the Donchian buy-stop hedge.

Reruns the hedge with the channel built from the NEXT 15 one-minute bars
instead of the previous 15. That version knows the future, so it should score
far better than the real one. If the two scored alike, the real pipeline would
be leaking. Also reports a random-entry control at matched trigger frequency,
which isolates how much of the hedge's P&L is the Donchian timing rather than
simply being long an option intraday.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.core import STORE, load_spot

OPT = STORE / "opt2"
N, TRAIL, TICK = 15, 10.0, 1.0
ENTRY_T, EXIT_T = pd.Timestamp("09:45").time(), pd.Timestamp("15:00").time()
RNG = np.random.default_rng(4242)


def walk(o, h, l, c, i, k):
    ent = max(0.0, float(o[i])) if np.isfinite(o[i]) else np.nan
    if not np.isfinite(ent) or ent <= 0:
        return None
    run = ent
    for m in range(i + 1, k + 1):
        trail = run - TRAIL
        if np.isfinite(l[m]) and l[m] <= trail:
            fill = min(trail, float(o[m])) if np.isfinite(o[m]) else trail
            return fill - ent
        if np.isfinite(h[m]):
            run = max(run, h[m])
    return float(c[k]) - ent if np.isfinite(c[k]) else None


def run_hedge(o, h, l, c, j, k, mode, ep):
    for i in range(j + N, k - N):
        win = h[i - N:i] if mode == "clean" else h[i + 1:i + 1 + N]
        if len(win) < N or not np.isfinite(win).all():
            continue
        stop = np.nanmax(win) + TICK
        if stop <= ep or not (np.isfinite(h[i]) and h[i] >= stop):
            continue
        ent = max(stop, float(o[i])) if np.isfinite(o[i]) else stop
        run = ent
        for m in range(i + 1, k + 1):
            trail = run - TRAIL
            if np.isfinite(l[m]) and l[m] <= trail:
                fill = min(trail, float(o[m])) if np.isfinite(o[m]) else trail
                return fill - ent
            if np.isfinite(h[m]):
                run = max(run, h[m])
        return float(c[k]) - ent
    return None


def main():
    daily = pd.read_csv(STORE / "ic_donchian_daily.csv")
    daily = daily[daily.cfg == "0.35/0.15"]
    spot = load_spot()
    res = {"clean": [], "leak": [], "random": []}

    for e, gexp in daily.groupby("expiry"):
        f = OPT / f"{int(e)}.parquet"
        if not f.exists():
            continue
        opt = pd.read_parquet(f)
        opt["ts"] = pd.to_datetime(opt["ts"])
        opt["date"] = opt["ts"].dt.date.astype(str)
        for _, r in gexp.iterrows():
            d = r["date"]
            ds_ = spot[spot["date"] == d]
            day_opt = opt[opt["date"] == d]
            if ds_.empty or day_opt.empty:
                continue
            grid = pd.DatetimeIndex(ds_["ts"])
            times = np.array([t.time() for t in grid])
            ji = np.where(times == ENTRY_T)[0]
            ki = np.where(times <= EXIT_T)[0]
            if not len(ji) or not len(ki):
                continue
            j, k = int(ji[0]), int(ki[-1])
            for legk, cp, epcol in ((int(r.sc_k), "CE", "entry_sc"),
                                    (int(r.sp_k), "PE", "entry_sp")):
                g = day_opt[(day_opt.strike == legk) & (day_opt.cp == cp)]
                if g.empty:
                    continue
                g = g.drop_duplicates("ts").set_index("ts").reindex(grid)
                o = g["o"].ffill(limit=3).to_numpy(float)
                h = g["h"].ffill(limit=3).to_numpy(float)
                l = g["l"].ffill(limit=3).to_numpy(float)
                c = g["c"].ffill(limit=3).to_numpy(float)
                ep = float(r[epcol])
                for mode in ("clean", "leak"):
                    v = run_hedge(o, h, l, c, j, k, mode, ep)
                    if v is not None:
                        res[mode].append(v)
                if k - N - (j + N) > 5:
                    i = int(RNG.integers(j + N, k - N))
                    v = walk(o, h, l, c, i, k)
                    if v is not None:
                        res["random"].append(v)

    print("=" * 84)
    print("HEDGE PLACEBO  --  option-Donchian buy-stop, bleed-gated, 0.35/0.15 short legs")
    print("=" * 84)
    print(f"  {'variant':<34}{'triggers':>10}{'mean pts':>12}{'win %':>10}{'total pts':>12}")
    for name, label in (("clean", "clean (channel = prior 15 bars)"),
                        ("leak", "LEAKED (channel = next 15 bars)"),
                        ("random", "random entry, same exit rule")):
        a = np.array(res[name], dtype=float)
        a = a[np.isfinite(a)]
        if not len(a):
            continue
        print(f"  {label:<34}{len(a):>10}{a.mean():>12.2f}{100*(a>0).mean():>10.1f}{a.sum():>12.0f}")
    cl = np.array(res["clean"]); rd = np.array(res["random"])
    print(f"\n  Donchian timing vs random entry: {cl.mean()-rd.mean():+.2f} pts per hedge trade")
    print("  A leaked channel scoring far above clean is the evidence that clean is not leaking.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
