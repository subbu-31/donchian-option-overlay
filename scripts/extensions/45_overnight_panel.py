#!/usr/bin/env python3
"""
45_overnight_panel.py -- The overnight variance premium.

Every result in this programme lives inside 09:45 -> 15:00.  The overnight
window has never been in any sample.  This builds it.

For each consecutive session pair (d, d+1) sharing one expiry:
  ENTRY 15:20 on d   EXIT 09:20 on d+1   -- the OVERNIGHT leg
  ENTRY 09:20 on d+1 EXIT 15:20 on d+1   -- the DAY leg, like-for-like
Structures: short ATM straddle, short 0.25d strangle, short 0.25/0.10 condor,
each priced on its own traded 1-minute bars.

THE CONVENTION TRAP.  Implied vol is quoted against CALENDAR time, so a
"calendar-time implied overnight move" compared against a realised gap makes
overnight look enormously overpriced BY CONSTRUCTION -- ~18 of every 24 hours
carry no trading variance and the quoted IV already averages over that.  The
primary test here is convention-free: what the structure ACTUALLY pays
overnight against what it ACTUALLY pays in the day session, per unit of clock
time and per unit of calendar time.  Implied/realised is a secondary read.

Spreads are estimated in the CLOSE and OPEN windows specifically -- 15:20 and
09:20 are the two worst liquidity moments of the session and the 09:45/15:00
estimates cannot be reused.  DTE at entry >= 1.0, so nothing is ever held into
an expiry session.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.core import STORE, load_spot
from lib.bs import implied_vol, delta, years_to_expiry
from lib.spread import half_spread_rupees, TICK
import lib.ic_costs as IC

OPT, CK = STORE / "opt2", STORE / "on_parts"
QTY, STEP = 750, 50
CLOSE_T, OPEN_T = "15:20", "09:20"
SNAP_MIN = 10          # price snapshot tolerance
EST_MIN = 30           # wider window for the spread estimators (need >=3 bars)
DTE_LO, MIN_PREM = 1.0, 1.0


def snap(day, t0, t1):
    win = day[(day.ts >= t0) & (day.ts <= t1) & (day.v > 0)]
    if win.empty:
        return {}
    return win.sort_values("ts").groupby(["strike", "cp"], observed=True)["c"].last().to_dict()


def hs(day, key, t0, t1, price):
    """Half-spread in rupees for one leg, estimated in [t0, t1]."""
    k, cp = key
    g = day[(day.strike == k) & (day.cp == cp) & (day.ts >= t0) & (day.ts <= t1)]
    if len(g) < 3:
        return TICK / 2.0
    v = half_spread_rupees(g.h.to_numpy(float), g.l.to_numpy(float),
                           g.c.to_numpy(float), price)
    return float(v) if np.isfinite(v) else TICK / 2.0


def pick(px, S, T, cp, target):
    best, bd = None, 9e9
    for (k, c), p in px.items():
        if c != cp or p <= MIN_PREM:
            continue
        iv = implied_vol(float(p), S, int(k), T, cp)
        if iv is None:
            continue
        dl = abs(delta(S, int(k), T, iv, cp))
        if abs(dl - target) < bd:
            bd, best = abs(dl - target), (int(k), float(p), dl)
    return best


def structure(px_a, px_b, day_a, day_b, ta, tb, S, T, tag, rec):
    """Price short straddle / strangle / condor from snapshot a to snapshot b."""
    atm = int(round(S / STEP) * STEP)
    if all((atm, cp) in px_a and (atm, cp) in px_b for cp in ("CE", "PE")):
        ea = float(px_a[(atm, "CE")]) + float(px_a[(atm, "PE")])
        eb = float(px_b[(atm, "CE")]) + float(px_b[(atm, "PE")])
        rec[f"strad_ent_{tag}"] = ea
        rec[f"strad_{tag}"] = ea - eb
        rec[f"strad_slip_{tag}"] = sum(
            hs(day_a, (atm, cp), ta - pd.Timedelta(minutes=EST_MIN), ta, px_a[(atm, cp)])
            + hs(day_b, (atm, cp), tb, tb + pd.Timedelta(minutes=EST_MIN), px_b[(atm, cp)])
            for cp in ("CE", "PE"))
    sc, sp = pick(px_a, S, T, "CE", 0.25), pick(px_a, S, T, "PE", 0.25)
    lc, lp = pick(px_a, S, T, "CE", 0.10), pick(px_a, S, T, "PE", 0.10)
    if not (sc and sp and lc and lp and lc[0] > sc[0] and lp[0] < sp[0]):
        return
    legs = {"sc": (sc[0], "CE"), "sp": (sp[0], "PE"),
            "lc": (lc[0], "CE"), "lp": (lp[0], "PE")}
    if not all(k in px_a and k in px_b for k in legs.values()):
        return
    a = {n: float(px_a[k]) for n, k in legs.items()}
    b = {n: float(px_b[k]) for n, k in legs.items()}
    rec[f"strangle_{tag}"] = (a["sc"] - b["sc"]) + (a["sp"] - b["sp"])
    rec[f"condor_{tag}"] = (rec[f"strangle_{tag}"]
                            - (a["lc"] - b["lc"]) - (a["lp"] - b["lp"]))
    rec[f"credit_{tag}"] = a["sc"] + a["sp"] - a["lc"] - a["lp"]
    rec[f"condor_slip_{tag}"] = sum(
        hs(day_a, k, ta - pd.Timedelta(minutes=EST_MIN), ta, a[n])
        + hs(day_b, k, tb, tb + pd.Timedelta(minutes=EST_MIN), b[n])
        for n, k in legs.items())
    rec[f"buy_{tag}"] = (a["lc"] + a["lp"] + b["sc"] + b["sp"]) * QTY
    rec[f"sell_{tag}"] = (a["sc"] + a["sp"] + b["lc"] + b["lp"]) * QTY
    rec[f"sc_d_{tag}"], rec[f"sp_d_{tag}"] = sc[2], sp[2]


def main():
    spot = load_spot()
    spot["ts"] = pd.to_datetime(spot["ts"])
    sess = sorted(spot["date"].unique())
    nxt = {d: sess[i + 1] for i, d in enumerate(sess[:-1])}
    emap = pd.read_csv(STORE / "expiry_map.csv", index_col=0)["expiry"].astype(str)
    emap.index = emap.index.astype(str)
    by_exp = {}
    for d, e in emap.items():
        by_exp.setdefault(e, []).append(d)

    CK.mkdir(exist_ok=True)
    for e in sorted(by_exp):
        f, part = OPT / f"{e}.parquet", CK / f"{e}.parquet"
        if not f.exists() or part.exists():
            continue
        opt = pd.read_parquet(f)
        opt["ts"] = pd.to_datetime(opt["ts"])
        opt["date"] = opt["ts"].dt.date.astype(str)
        rows = []
        for d in sorted(by_exp[e]):
            d1 = nxt.get(d)
            if d1 is None or d1 not in by_exp[e]:
                continue
            tc = pd.Timestamp(f"{d} {CLOSE_T}:00")
            to = pd.Timestamp(f"{d1} {OPEN_T}:00")
            te = pd.Timestamp(f"{d1} {CLOSE_T}:00")
            g = spot.set_index("ts")
            if not all(t in g.index for t in (tc, to, te)):
                continue
            S_c, S_o, S_e = (float(g.loc[t, "c"]) for t in (tc, to, te))
            T_c, T_o = years_to_expiry(tc, e), years_to_expiry(to, e)
            if T_c * 365 < DTE_LO:
                continue
            dc, d1_ = opt[opt.date == d], opt[opt.date == d1]
            px_c = snap(dc, tc - pd.Timedelta(minutes=SNAP_MIN), tc)
            px_o = snap(d1_, to, to + pd.Timedelta(minutes=SNAP_MIN))
            px_e = snap(d1_, te - pd.Timedelta(minutes=SNAP_MIN), te)
            if not (px_c and px_o and px_e):
                continue
            atm = int(round(S_c / STEP) * STEP)
            if not all((atm, cp) in px_c for cp in ("CE", "PE")):
                continue
            ivc = implied_vol(float(px_c[(atm, "CE")]), S_c, atm, T_c, "CE")
            ivp = implied_vol(float(px_c[(atm, "PE")]), S_c, atm, T_c, "PE")
            if ivc is None or ivp is None:
                continue
            iv = (ivc + ivp) / 2.0
            dtau = (to - tc).total_seconds() / 86400.0 / 365.0
            rec = {"date": d, "next": d1, "expiry": e, "S_c": S_c, "S_o": S_o, "S_e": S_e,
                   "dte_c": T_c * 365, "dte_o": T_o * 365, "iv_close": iv,
                   "gap_pct": 100.0 * (S_o / S_c - 1.0),
                   "day_pct": 100.0 * (S_e / S_o - 1.0),
                   "h_on": (to - tc).total_seconds() / 3600.0,
                   "h_day": (te - to).total_seconds() / 3600.0,
                   "imp_move_on": S_c * iv * np.sqrt(max(dtau, 1e-9)),
                   "real_move_on": abs(S_o - S_c),
                   "real_move_day": abs(S_e - S_o)}
            structure(px_c, px_o, dc, d1_, tc, to, S_c, T_c, "on", rec)
            structure(px_o, px_e, d1_, d1_, to, te, S_o, T_o, "day", rec)
            rows.append(rec)
        pd.DataFrame(rows).to_parquet(part, index=False)
        print(f"  {e}  {len(rows)}", flush=True)

    parts = [pd.read_parquet(x) for x in sorted(CK.glob("*.parquet"))]
    parts = [x for x in parts if len(x)]
    if not parts:
        print("nothing built")
        return 1
    t = pd.concat(parts, ignore_index=True).drop_duplicates("date").sort_values("date")
    t["year"] = pd.to_datetime(t["date"]).dt.year
    for tag in ("on", "day"):
        if f"buy_{tag}" in t.columns:
            t[f"stat_{tag}"] = [
                IC._statutory(buy_turn=b, sell_turn=s, orders=8, on=dd)
                if np.isfinite(b) else np.nan
                for b, s, dd in zip(t[f"buy_{tag}"], t[f"sell_{tag}"], t["date"])]
            t[f"condor_net_{tag}"] = (t[f"condor_{tag}"] * QTY
                                      - t[f"condor_slip_{tag}"] * QTY - t[f"stat_{tag}"])
    t.to_csv(STORE / "overnight_panel.csv", index=False)
    print(f"\n{len(t)} overnight windows  {t.date.min()} .. {t.date.max()}")
    print(f"mean hours  overnight {t.h_on.mean():.1f}   day {t.h_day.mean():.1f}")
    cols = [c for c in ["gap_pct", "day_pct", "iv_close", "dte_c", "strad_ent_on",
                        "strad_on", "strad_day", "condor_on", "condor_day",
                        "credit_on", "credit_day", "real_move_on", "real_move_day",
                        "imp_move_on", "condor_net_on", "condor_net_day"]
            if c in t.columns]
    print(t[cols].describe().T[["count", "mean", "std", "25%", "50%", "75%"]].round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
