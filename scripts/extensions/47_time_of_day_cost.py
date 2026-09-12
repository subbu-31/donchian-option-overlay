#!/usr/bin/env python3
"""
47_time_of_day_cost.py -- Where in the session does the edge get eaten?

The overnight study threw off an incidental result worth isolating: the SAME
condor entered 09:20 and exited 15:20 cost 2.39 points a round trip, against
1.51 at 09:45 -> 15:00.  Nearly a point -- more than half the whole net edge.

Two surfaces are reported and they must be read differently:
  COST     mechanical.  Where spreads are widest is a property of the market's
           liquidity clock, not a fitted parameter.  Trust it.
  NET P&L  a 4x3 sweep on 322 sessions is 12 cells on the same data.  Read it
           as a sanity check on the cost surface, never as an entry-time choice.
"""
import sys
from math import erf, sqrt
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.core import STORE, load_spot
from lib.bs import implied_vol, delta, years_to_expiry
from lib.spread import half_spread_rupees, TICK
import lib.ic_costs as IC

OPT = STORE / "opt2"
QTY, STEP, EST = 750, 50, 30
ENTRIES = ["09:20", "09:30", "09:45", "10:00"]
EXITS = ["14:30", "15:00", "15:20"]
DTE_LO, DTE_HI, MIN_PREM = 1.0, 8.5, 1.0
ncdf = lambda x: 0.5 * (1 + erf(x / sqrt(2)))


def sc(x):
    x = np.asarray(pd.Series(x).dropna(), float); n = len(x)
    if n < 5 or x.std(ddof=1) == 0: return n, np.nan, np.nan
    sr = x.mean() / x.std(ddof=1)
    return n, x.mean(), sr / np.sqrt((1 + 0.5 * sr ** 2) / n)


def snap(day, t, back):
    w = day[(day.ts >= t - pd.Timedelta(minutes=back)) & (day.ts <= t) & (day.v > 0)]
    if w.empty: return {}
    return w.sort_values("ts").groupby(["strike", "cp"], observed=True)["c"].last().to_dict()


def hs(day, k, cp, t0, t1, price):
    g = day[(day.strike == k) & (day.cp == cp) & (day.ts >= t0) & (day.ts <= t1)]
    if len(g) < 3: return TICK / 2.0
    v = half_spread_rupees(g.h.to_numpy(float), g.l.to_numpy(float), g.c.to_numpy(float), price)
    return float(v) if np.isfinite(v) else TICK / 2.0


def pick(px, S, T, cp, tgt):
    best, bd = None, 9e9
    for (k, c), p in px.items():
        if c != cp or p <= MIN_PREM: continue
        iv = implied_vol(float(p), S, int(k), T, cp)
        if iv is None: continue
        dl = abs(delta(S, int(k), T, iv, cp))
        if abs(dl - tgt) < bd: bd, best = abs(dl - tgt), (int(k), float(p))
    return best


def main():
    spot = load_spot(); spot["ts"] = pd.to_datetime(spot["ts"])
    g = spot.set_index("ts")
    emap = pd.read_csv(STORE / "expiry_map.csv", index_col=0)["expiry"].astype(str)
    emap.index = emap.index.astype(str)
    by_exp = {}
    for d, e in emap.items(): by_exp.setdefault(e, []).append(d)
    rows = []
    for e in sorted(by_exp):
        f = OPT / f"{e}.parquet"
        if not f.exists(): continue
        opt = pd.read_parquet(f); opt["ts"] = pd.to_datetime(opt["ts"])
        opt["date"] = opt["ts"].dt.date.astype(str)
        for d in sorted(by_exp[e]):
            day = opt[opt.date == d]
            if day.empty: continue
            for en in ENTRIES:
                te = pd.Timestamp(f"{d} {en}:00")
                if te not in g.index: continue
                S = float(g.loc[te, "c"]); T = years_to_expiry(te, e)
                if not (DTE_LO <= T * 365 <= DTE_HI): continue
                pe_ = snap(day, te, 10)
                if not pe_: continue
                sc_, sp_ = pick(pe_, S, T, "CE", 0.25), pick(pe_, S, T, "PE", 0.25)
                lc_, lp_ = pick(pe_, S, T, "CE", 0.10), pick(pe_, S, T, "PE", 0.10)
                if not (sc_ and sp_ and lc_ and lp_ and lc_[0] > sc_[0] and lp_[0] < sp_[0]):
                    continue
                legs = {"sc": (sc_[0], "CE"), "sp": (sp_[0], "PE"),
                        "lc": (lc_[0], "CE"), "lp": (lp_[0], "PE")}
                a = {n: float(pe_[k]) for n, k in legs.items()}
                slip_in = sum(hs(day, k[0], k[1], te - pd.Timedelta(minutes=EST), te, a[n])
                              for n, k in legs.items())
                for ex in EXITS:
                    tx = pd.Timestamp(f"{d} {ex}:00")
                    if tx not in g.index or tx <= te: continue
                    px = snap(day, tx, 10)
                    if not all(k in px for k in legs.values()): continue
                    b = {n: float(px[k]) for n, k in legs.items()}
                    gross = ((a["sc"] - b["sc"]) + (a["sp"] - b["sp"])
                             - (a["lc"] - b["lc"]) - (a["lp"] - b["lp"]))
                    slip = slip_in + sum(
                        hs(day, k[0], k[1], tx, tx + pd.Timedelta(minutes=EST), b[n])
                        for n, k in legs.items())
                    stat = IC._statutory(
                        buy_turn=(a["lc"] + a["lp"] + b["sc"] + b["sp"]) * QTY,
                        sell_turn=(a["sc"] + a["sp"] + b["lc"] + b["lp"]) * QTY,
                        orders=8, on=d)
                    rows.append(dict(date=d, entry=en, exit=ex, gross=gross, slip=slip,
                                     stat_pts=stat / QTY,
                                     net=(gross - slip) * QTY - stat))
        print(f"  {e}", flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(STORE / "tod_cost.csv", index=False)
    t["year"] = pd.to_datetime(t["date"]).dt.year

    print(f"\n{len(t)} trades, {t.date.nunique()} sessions\n")
    print("COST SURFACE -- slippage + statutory, points per round trip  (MECHANICAL, trust it)")
    print(f"  {'entry':<8}" + "".join(f"{x:>10}" for x in EXITS))
    for en in ENTRIES:
        r = [t[(t.entry == en) & (t.exit == ex)] for ex in EXITS]
        print(f"  {en:<8}" + "".join(
            f"{(x.slip+x.stat_pts).mean():>10.2f}" if len(x) else f"{'-':>10}" for x in r))
    print("\n  of which SLIPPAGE only")
    for en in ENTRIES:
        r = [t[(t.entry == en) & (t.exit == ex)] for ex in EXITS]
        print(f"  {en:<8}" + "".join(
            f"{x.slip.mean():>10.2f}" if len(x) else f"{'-':>10}" for x in r))

    print("\nGROSS SURFACE -- points per round trip, before any cost")
    print(f"  {'entry':<8}" + "".join(f"{x:>10}" for x in EXITS))
    for en in ENTRIES:
        r = [t[(t.entry == en) & (t.exit == ex)] for ex in EXITS]
        print(f"  {en:<8}" + "".join(
            f"{x.gross.mean():>10.2f}" if len(x) else f"{'-':>10}" for x in r))

    print("\nNET SURFACE -- Rs/session  (12 cells on one sample: a SANITY CHECK, not a choice)")
    print(f"  {'entry':<8}" + "".join(f"{x:>18}" for x in EXITS))
    for en in ENTRIES:
        out = f"  {en:<8}"
        for ex in EXITS:
            x = t[(t.entry == en) & (t.exit == ex)]
            if len(x) < 20: out += f"{'-':>18}"; continue
            n, mu, tt = sc(x.net)
            out += f"{mu:>11,.0f} (t{tt:>4.1f})"
        print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
