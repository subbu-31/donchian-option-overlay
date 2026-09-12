#!/usr/bin/env python3
"""
06_straddle_by_width.py -- The decisive economic test.

Channel width predicts the SIZE of the next move (script 04, Q2, replicated
in all three periods).  Size is exactly what a bought option is long.  So:
buy the at-the-money weekly straddle at each decision bar, hold h minutes,
and sort the result by the channel-width quintile known at entry.

If the width signal is already in the option's price, every quintile loses
the same amount (theta) and there is nothing here.  If it is not, the wide
quintile loses less -- or wins.

Prices are the real 1-minute option bars: entry at the next minute's OPEN,
exit at the exit minute's CLOSE, staleness capped at 3 minutes, entry minute
required to have traded.  Costs are the full Zerodha round trip.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.core import STORE, load_spot, to_5m, donchian_signals
from lib.features import add_features
from lib.costs import leg_cost, LOT

N = 12
HORIZONS = [15, 30, 60]
STRIKE_STEP = 50
OPT = STORE / "opt2"


def series_map(day_opt, grid):
    """{(strike, cp): (open_arr, close_arr)} aligned to the day's minute grid."""
    out = {}
    for (s, cp), g in day_opt.groupby(["strike", "cp"], observed=True):
        g = g.drop_duplicates("ts").set_index("ts").reindex(grid)
        o = g["o"].ffill(limit=3).to_numpy()
        c = g["c"].ffill(limit=3).to_numpy()
        v = g["v"].to_numpy()
        out[(int(s), str(cp))] = (o, c, v)
    return out


def main():
    spot = load_spot()
    spot["year"] = pd.to_datetime(spot["date"]).dt.year
    emap = pd.read_csv(STORE / "expiry_map.csv", index_col=0)["expiry"].astype(str)
    emap.index = emap.index.astype(str)

    bars5 = donchian_signals(to_5m(spot), N)
    bars5 = add_features(bars5, N)
    bars5 = bars5[bars5["bar_no"] >= N].copy()

    rows = []
    by_exp = {}
    for d, e in emap.items():
        by_exp.setdefault(e, []).append(d)

    done = 0
    for e in sorted(by_exp):
        f = OPT / f"{e}.parquet"
        if not f.exists():
            continue
        opt = pd.read_parquet(f)
        opt["ts"] = pd.to_datetime(opt["ts"])
        opt["date"] = opt["ts"].dt.date.astype(str)
        for d in by_exp[e]:
            day_opt = opt[opt["date"] == d]
            if day_opt.empty:
                continue
            day_spot = spot[spot["date"] == d]
            grid = pd.DatetimeIndex(day_spot["ts"])
            pos = {t: i for i, t in enumerate(grid)}
            sm = series_map(day_opt, grid)
            db = bars5[bars5["date"] == d]
            for _, b in db.iterrows():
                ent_ts = b["bar_close_ts"] + pd.Timedelta(minutes=1)
                if ent_ts not in pos:
                    continue
                j = pos[ent_ts]
                atm = int(round(b["c"] / STRIKE_STEP) * STRIKE_STEP)
                ce, pe = sm.get((atm, "CE")), sm.get((atm, "PE"))
                if ce is None or pe is None:
                    continue
                ce_o, ce_c, ce_v = ce
                pe_o, pe_c, pe_v = pe
                e_ce, e_pe = ce_o[j], pe_o[j]
                if not (np.isfinite(e_ce) and np.isfinite(e_pe)) or e_ce <= 1 or e_pe <= 1:
                    continue
                if not (ce_v[j] > 0 and pe_v[j] > 0):      # must have actually traded
                    continue
                for h in HORIZONS:
                    k = j + h
                    if k >= len(grid) or grid[k].time() > pd.Timestamp("15:20").time():
                        continue
                    x_ce, x_pe = ce_c[k], pe_c[k]
                    if not (np.isfinite(x_ce) and np.isfinite(x_pe)):
                        continue
                    prem = float(e_ce + e_pe)
                    gross = float((x_ce - e_ce) + (x_pe - e_pe)) * LOT
                    cost = leg_cost(float(e_ce), float(x_ce)) + leg_cost(float(e_pe), float(x_pe))
                    rows.append({
                        "date": d, "expiry": e, "h": h,
                        "width_rel": float(b["width_rel"]) if pd.notna(b["width_rel"]) else np.nan,
                        "is_breakout": int(b["sig"] != 0),
                        "bar_no": int(b["bar_no"]),
                        "premium": prem,
                        "spot_move": float(abs(day_spot["c"].to_numpy()[k] - b["c"])),
                        "gross": gross, "cost": cost, "net": gross - cost,
                    })
            done += 1
        print(f"  expiry {e}: sessions done {done}", flush=True)

    tr = pd.DataFrame(rows)
    tr["year"] = pd.to_datetime(tr["date"]).dt.year
    tr.to_parquet(STORE / "straddle_trades.parquet", index=False)
    print(f"\n{len(tr)} straddle observations, {tr['date'].nunique()} sessions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
