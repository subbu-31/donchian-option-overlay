#!/usr/bin/env python3
"""
16_spread_model.py -- Replace the guessed slippage constant with one estimated
from the data.

For every leg actually traded in the IC backtest, and for every hedge leg, the
effective half-spread is estimated from that leg's own 1-minute high/low/close
path on that session, then summarised by premium level, moneyness and expiry
proximity so the cost can be attributed rather than assumed.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.core import STORE, load_spot
from lib.spread import half_spread_rupees, corwin_schultz, abdi_ranaldo, zero_change_fraction, TICK

OPT = STORE / "opt2"
ENTRY_T, EXIT_T = pd.Timestamp("09:45").time(), pd.Timestamp("15:00").time()


def main():
    daily = pd.read_csv(STORE / "ic_donchian_daily.csv")
    daily = daily[daily.cfg == "0.35/0.15"].copy()
    spot = load_spot()
    rows = []

    for e, gexp in daily.groupby("expiry"):
        f = OPT / f"{int(e)}.parquet"
        if not f.exists():
            continue
        opt = pd.read_parquet(f)
        opt["ts"] = pd.to_datetime(opt["ts"])
        opt["date"] = opt["ts"].dt.date.astype(str)
        for _, r in gexp.iterrows():
            d = r["date"]
            day = opt[opt["date"] == d]
            ds_ = spot[spot["date"] == d]
            if day.empty or ds_.empty:
                continue
            day = day[(day["ts"].dt.time >= ENTRY_T) & (day["ts"].dt.time <= EXIT_T)]
            for role, k, cp, prem in (("short_call", r.sc_k, "CE", r.entry_sc),
                                      ("short_put", r.sp_k, "PE", r.entry_sp),
                                      ("long_call", r.lc_k, "CE", r.entry_lc),
                                      ("long_put", r.lp_k, "PE", r.entry_lp)):
                g = day[(day.strike == int(k)) & (day.cp == cp)]
                if len(g) < 20:
                    continue
                hs = half_spread_rupees(g["h"], g["l"], g["c"], prem)
                if not np.isfinite(hs):
                    continue
                rows.append({
                    "date": d, "expiry": int(e), "role": role, "strike": int(k), "cp": cp,
                    "premium": float(prem), "dte": float(r.dte),
                    "moneyness": abs(float(k) - float(r.spot)) / float(r.spot) * 100,
                    "half_spread": hs,
                    "cs": corwin_schultz(g["h"], g["l"]),
                    "ar": abdi_ranaldo(g["h"], g["l"], g["c"]),
                    "zero_min": zero_change_fraction(g["c"]),
                    "med_vol": float(g["v"].median()),
                })
        print(f"  {int(e)}  legs={len(rows)}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(STORE / "spread_estimates.csv", index=False)

    print("\n" + "=" * 96)
    print("ESTIMATED EFFECTIVE HALF-SPREAD PER LEG  (rupees per unit, one crossing)")
    print("=" * 96)
    print(df.groupby("role")["half_spread"].describe(
        percentiles=[.1, .25, .5, .75, .9]).round(3).to_string())
    print("\n  by premium bucket")
    df["pb"] = pd.cut(df["premium"], [0, 25, 50, 100, 200, 1e9],
                      labels=["<25", "25-50", "50-100", "100-200", ">200"])
    print(df.groupby("pb", observed=True).agg(
        n=("half_spread", "size"), half_spread=("half_spread", "median"),
        as_pct_of_prem=("half_spread", lambda s: np.nan),
        zero_min=("zero_min", "median"), med_vol=("med_vol", "median")).round(3).to_string())
    tmp = df.copy()
    tmp["pct"] = tmp["half_spread"] / tmp["premium"] * 100
    print("\n  half-spread as % of premium, by bucket:")
    print(tmp.groupby("pb", observed=True)["pct"].median().round(2).to_string())
    print("\n  by days to expiry")
    df["db"] = pd.cut(df["dte"], [0, 1.5, 3, 5, 9], labels=["<1.5", "1.5-3", "3-5", "5-9"])
    print(df.groupby("db", observed=True).agg(
        n=("half_spread", "size"), half_spread=("half_spread", "median"),
        premium=("premium", "median")).round(3).to_string())
    print("\n  by year")
    df["year"] = pd.to_datetime(df["date"]).dt.year
    print(df.groupby(["year", "role"])["half_spread"].median().round(3).to_string())

    per_day = df.groupby("date")["half_spread"].sum() * 2      # 4 legs, 2 crossings
    print("\n" + "=" * 96)
    print("MODELLED ROUND-TRIP CROSSING COST, per unit, per session (4 legs x 2 sides)")
    print("=" * 96)
    print(per_day.describe(percentiles=[.1, .25, .5, .75, .9]).round(3).to_string())
    print(f"\n  median modelled equivalent flat slippage = "
          f"{per_day.median()/8:.4f} pts per leg per side")
    print(f"  mean   modelled equivalent flat slippage = "
          f"{per_day.mean()/8:.4f} pts per leg per side")
    print(f"  (the tearsheet's swept ladder used 0.00 / 0.10 / 0.25 / 0.50)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
