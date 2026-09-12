#!/usr/bin/env python3
"""
08_daily_arms.py -- One decision per session, so the flat costs amortise
over a whole day's move instead of a 30-minute wiggle.

Decision is taken at the 09:45 bar close (opening range complete, channel
formed).  The Donchian channel width at that moment is the state variable.

  A  long ATM weekly straddle           09:50 -> 15:15
  B  short ~1% OTM weekly strangle      09:50 -> 15:15
  C  long ATM single leg, in the direction of the first Donchian breakout
     after 09:45, held to 15:15  (the literal "Donchian + option buying"
     overlay that was asked for)

All three on real 1-minute option bars, full Zerodha round trip, 0.5 pt/side
slippage per leg.  Width quintile cut points are fitted on 2025 and applied
verbatim to 2026.
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
STEP = 50
OTM = 0.010
OPT = STORE / "opt2"
DECIDE_BAR = 12         # 10:14 bar close -> entry 10:15 (channel needs N prior bars)
EXIT_T = pd.Timestamp("15:15").time()


def dcci(df, col, n_boot=3000, seed=23):
    rng = np.random.default_rng(seed)
    v = df[col].to_numpy()
    k = len(v)
    if k < 8:
        return (np.nan, np.nan)
    m = np.array([v[rng.integers(0, k, k)].mean() for _ in range(n_boot)])
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    spot = load_spot()
    emap = pd.read_csv(STORE / "expiry_map.csv", index_col=0)["expiry"].astype(str)
    emap.index = emap.index.astype(str)
    b5 = add_features(donchian_signals(to_5m(spot), N), N)

    by_exp = {}
    for d, e in emap.items():
        by_exp.setdefault(e, []).append(d)

    rows = []
    for e in sorted(by_exp):
        f = OPT / f"{e}.parquet"
        if not f.exists():
            continue
        opt = pd.read_parquet(f)
        opt["ts"] = pd.to_datetime(opt["ts"])
        opt["date"] = opt["ts"].dt.date.astype(str)
        for d in by_exp[e]:
            day_opt = opt[opt["date"] == d]
            db = b5[b5["date"] == d]
            if day_opt.empty or db.empty:
                continue
            dec = db[db["bar_no"] == DECIDE_BAR]
            if dec.empty or pd.isna(dec["width_rel"].iloc[0]):
                continue
            dec = dec.iloc[0]
            ds = spot[spot["date"] == d]
            grid = pd.DatetimeIndex(ds["ts"])
            pos = {t: i for i, t in enumerate(grid)}
            ent_ts = dec["bar_close_ts"] + pd.Timedelta(minutes=1)
            if ent_ts not in pos:
                continue
            j = pos[ent_ts]
            kx = int(np.searchsorted([t.time() for t in grid], EXIT_T, side="left"))
            kx = min(kx, len(grid) - 1)
            if kx <= j:
                continue

            book = {}
            for (s, cp), g in day_opt.groupby(["strike", "cp"], observed=True):
                g = g.drop_duplicates("ts").set_index("ts").reindex(grid)
                book[(int(s), str(cp))] = (g["o"].ffill(limit=3).to_numpy(),
                                           g["c"].ffill(limit=3).to_numpy(),
                                           g["v"].to_numpy())

            ref = float(dec["c"])
            atm = int(round(ref / STEP) * STEP)
            call_k = int(round(ref * (1 + OTM) / STEP) * STEP)
            put_k = int(round(ref * (1 - OTM) / STEP) * STEP)

            def px(k_strike, cp):
                v = book.get((k_strike, cp))
                if v is None:
                    return None
                o, c, vol = v
                if not (np.isfinite(o[j]) and np.isfinite(c[kx])) or o[j] <= 1 or vol[j] <= 0:
                    return None
                return float(o[j]), float(c[kx])

            rec = {"date": d, "expiry": e, "width_rel": float(dec["width_rel"]),
                   "or_rel": float(dec["or_rel"]) if pd.notna(dec["or_rel"]) else np.nan,
                   "day_range": float(ds["h"].max() - ds["l"].min()),
                   "close_move": float(ds["c"].to_numpy()[kx] - ref)}

            # A -- long ATM straddle
            a1, a2 = px(atm, "CE"), px(atm, "PE")
            if a1 and a2:
                g = ((a1[1] - a1[0]) + (a2[1] - a2[0])) * LOT
                c = leg_cost(a1[0], a1[1]) + leg_cost(a2[0], a2[1])
                rec["A_prem"] = a1[0] + a2[0]
                rec["A_net"] = g - c

            # B -- short OTM strangle
            b1, b2 = px(call_k, "CE"), px(put_k, "PE")
            if b1 and b2:
                g = -((b1[1] - b1[0]) + (b2[1] - b2[0])) * LOT
                c = leg_cost(b1[0], b1[1]) + leg_cost(b2[0], b2[1])
                rec["B_prem"] = b1[0] + b2[0]
                rec["B_net"] = g - c

            # C -- long ATM single leg on the first breakout after 09:45
            fb = db[(db["bar_no"] > DECIDE_BAR) & (db["sig"] != 0)]
            if not fb.empty:
                fb = fb.iloc[0]
                ets = fb["bar_close_ts"] + pd.Timedelta(minutes=1)
                if ets in pos:
                    jj = pos[ets]
                    if jj < kx:
                        cp = "CE" if fb["sig"] > 0 else "PE"
                        k_leg = int(round(float(fb["c"]) / STEP) * STEP)
                        v = book.get((k_leg, cp))
                        if v is not None:
                            o, cc, vol = v
                            if np.isfinite(o[jj]) and np.isfinite(cc[kx]) and o[jj] > 1 and vol[jj] > 0:
                                g = (cc[kx] - o[jj]) * LOT
                                rec["C_prem"] = float(o[jj])
                                rec["C_net"] = g - leg_cost(float(o[jj]), float(cc[kx]))
                                rec["C_dir"] = int(fb["sig"])
            rows.append(rec)
        print(f"  {e}", flush=True)

    df = pd.DataFrame(rows)
    df["year"] = pd.to_datetime(df["date"]).dt.year
    df.to_csv(STORE / "daily_arms.csv", index=False)

    cuts = np.quantile(df[df["year"] == 2025]["width_rel"].dropna(), [0.2, 0.4, 0.6, 0.8])
    df["wq"] = np.digitize(df["width_rel"], cuts)
    print(f"\nwidth cut points from 2025: {np.round(cuts,3).tolist()}")

    for arm, name in [("A", "LONG ATM straddle"), ("B", "SHORT 1% OTM strangle"),
                      ("C", "LONG ATM leg on first Donchian breakout")]:
        col = f"{arm}_net"
        if col not in df:
            continue
        print("\n" + "=" * 88)
        print(f"ARM {arm}: {name}   09:50 -> 15:15, net of all costs, Rs per lot of 75")
        print("=" * 88)
        for yr in (2025, 2026):
            d = df[(df["year"] == yr)].dropna(subset=[col])
            if d.empty:
                continue
            lo, hi = dcci(d, col)
            print(f"  {yr} ALL      n={len(d):>3}  mean=Rs {d[col].mean():>8.1f} "
                  f"[{lo:>8.1f},{hi:>8.1f}]  hit={(d[col]>0).mean():.3f}  "
                  f"total=Rs {d[col].sum():>10.0f}")
            for q in sorted(d["wq"].unique()):
                g = d[d["wq"] == q]
                if len(g) < 10:
                    continue
                lo, hi = dcci(g, col)
                print(f"       wq{int(q)}    n={len(g):>3}  mean=Rs {g[col].mean():>8.1f} "
                      f"[{lo:>8.1f},{hi:>8.1f}]  hit={(g[col]>0).mean():.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
