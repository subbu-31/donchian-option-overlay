#!/usr/bin/env python3
"""
20_weekly_hold.py -- The iron condor held across the whole week instead of
intraday: enter 09:45 at 6 DTE, exit 14:30 on expiry day (0 DTE).

NOTE: this deliberately violates the standing no-0-DTE rule. Holding a short
condor to 14:30 on expiry day is the maximum-gamma state of the position, and
the intra-trade drawdown column below is the number that matters most.

One trade per weekly expiry, so the sample is ~50 in-sample and ~17 held out.
Every figure here is a small-sample figure and is reported with a bootstrap
interval rather than as a point estimate.

  strikes    delta-selected at the 09:45 entry tick by BS implied-vol
             bisection, primary config 0.25 delta short / 0.10 delta wings
  path       every 1-minute bar from entry to exit, across all sessions in
             between, so overnight gaps and intraweek excursions are in the
             drawdown
  expiry     a leg with no quotes at the 14:30 exit is marked to intrinsic
             against spot, not dropped
  hedge      option-Donchian buy-stop on each short leg, channel reset each
             session, bleed-gated, one trigger per leg per session
  costs      each leg's own estimated half-spread (Corwin-Schultz /
             Abdi-Ranaldo) plus the full Zerodha statutory stack
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.core import STORE, load_spot
from lib.bs import implied_vol, delta, years_to_expiry
from lib.spread import half_spread_rupees
import lib.ic_costs as IC

OPT = STORE / "opt2"
QTY = 750
ENTRY_T = pd.Timestamp("09:45").time()
EXIT_T = pd.Timestamp("14:30").time()
TARGET_DTE, DTE_LO, DTE_HI = 6.24, 5.0, 7.6
CFGS = [(0.25, 0.05), (0.25, 0.10), (0.35, 0.15)]
DON_N, DON_TRAIL, DON_TICK = 15, 10.0, 1.0
MIN_PREM = 0.5


def pick(chain, ds, dl):
    ce, pe = chain[chain.cp == "CE"], chain[chain.cp == "PE"]
    if ce.empty or pe.empty:
        return None
    sc = ce.iloc[(ce.d - ds).abs().argsort().iloc[0]]
    sp = pe.iloc[(pe.d + ds).abs().argsort().iloc[0]]
    cf, pf = ce[ce.k > sc.k], pe[pe.k < sp.k]
    if cf.empty or pf.empty:
        return None
    lc = cf.iloc[(cf.d - dl).abs().argsort().iloc[0]]
    lp = pf.iloc[(pf.d + dl).abs().argsort().iloc[0]]
    return {"sc": sc, "sp": sp, "lc": lc, "lp": lp}


def hedge_leg(o, h, l, c, day_bounds, entry_prem, j0, jN):
    """Donchian buy-stop, channel reset each session, one trigger per session,
    only armed while the short leg is bleeding."""
    out = []
    for (a, b) in day_bounds:
        a2, b2 = max(a, j0), min(b, jN)
        if b2 - a2 < DON_N + 2:
            continue
        for i in range(a2 + DON_N, b2 + 1):
            win = h[i - DON_N:i]
            if not np.isfinite(win).all():
                continue
            stop = np.nanmax(win) + DON_TICK
            if stop <= entry_prem or not (np.isfinite(h[i]) and h[i] >= stop):
                continue
            ent = max(stop, float(o[i])) if np.isfinite(o[i]) else stop
            run, done = ent, None
            for m in range(i + 1, b2 + 1):
                trail = run - DON_TRAIL
                if np.isfinite(l[m]) and l[m] <= trail:
                    fill = min(trail, float(o[m])) if np.isfinite(o[m]) else trail
                    done = (fill - ent, ent, fill)
                    break
                if np.isfinite(h[m]):
                    run = max(run, h[m])
            if done is None and np.isfinite(c[b2]):
                done = (float(c[b2]) - ent, ent, float(c[b2]))
            if done:
                out.append(done)
            break
    return out


def main():
    spot = load_spot()
    spot_by_date = {d: g for d, g in spot.groupby("date")}
    rows = []

    for f in sorted(OPT.glob("*.parquet")):
        e = f.stem
        opt = pd.read_parquet(f)
        if opt.empty:
            continue
        opt["ts"] = pd.to_datetime(opt["ts"])
        opt["date"] = opt["ts"].dt.date.astype(str)
        exp_day = pd.Timestamp(e).strftime("%Y-%m-%d")
        dates = sorted(opt["date"].unique())
        if exp_day not in dates:
            continue

        # entry session: the one whose 09:45 sits closest to 6 DTE
        cand = []
        for d in dates:
            if d >= exp_day or d not in spot_by_date:
                continue
            ds_ = spot_by_date[d]
            t = ds_[ds_["ts"].dt.time == ENTRY_T]
            if t.empty:
                continue
            dte = years_to_expiry(t["ts"].iloc[0], e) * 365
            if DTE_LO <= dte <= DTE_HI:
                cand.append((abs(dte - TARGET_DTE), d, dte, float(t["c"].iloc[0]), t["ts"].iloc[0]))
        if not cand:
            continue
        _, d0, dte0, S0, ts0 = min(cand)

        # sessions spanned, and the concatenated minute grid
        span = [d for d in dates if d0 <= d <= exp_day and d in spot_by_date]
        grid_parts, bounds, n = [], [], 0
        for d in span:
            g = spot_by_date[d]
            grid_parts.append(g)
            bounds.append((n, n + len(g) - 1))
            n += len(g)
        sgrid = pd.concat(grid_parts, ignore_index=True)
        idx = pd.DatetimeIndex(sgrid["ts"])
        pos = {t: i for i, t in enumerate(idx)}
        j0 = pos[ts0]
        exit_rows = sgrid[(sgrid["date"] == exp_day) & (sgrid["ts"].dt.time <= EXIT_T)]
        if exit_rows.empty:
            continue
        jN = pos[exit_rows["ts"].iloc[-1]]
        S_exit = float(exit_rows["c"].iloc[-1])
        if jN <= j0:
            continue

        book = {}
        for (k, cp), g in opt.groupby(["strike", "cp"], observed=True):
            g = g.drop_duplicates("ts").set_index("ts").reindex(idx)
            book[(int(k), str(cp))] = (g["o"].ffill(limit=30).to_numpy(float),
                                       g["h"].ffill(limit=30).to_numpy(float),
                                       g["l"].ffill(limit=30).to_numpy(float),
                                       g["c"].ffill(limit=30).to_numpy(float),
                                       g["v"].to_numpy(float))

        T0 = years_to_expiry(ts0, e)
        ch = []
        for (k, cp), (o, h, l, c, v) in book.items():
            p = c[j0]
            if not np.isfinite(p) or p < MIN_PREM or not (v[j0] > 0):
                continue
            iv = implied_vol(float(p), S0, float(k), T0, cp)
            if iv is None:
                continue
            ch.append((k, cp, float(p), delta(S0, float(k), T0, iv, cp)))
        if not ch:
            continue
        chain = pd.DataFrame(ch, columns=["k", "cp", "px", "d"])

        for ds, dl in CFGS:
            legs = pick(chain, ds, dl)
            if legs is None:
                continue
            arr, ex, miss = {}, {}, 0
            for nm, sign in (("sc", +1), ("sp", +1), ("lc", -1), ("lp", -1)):
                cp = "CE" if nm in ("sc", "lc") else "PE"
                k = int(legs[nm]["k"])
                arr[nm] = book[(k, cp)]
                p = arr[nm][3][jN]
                if not np.isfinite(p):           # no quotes at exit -> intrinsic
                    p = max(0.0, S_exit - k) if cp == "CE" else max(0.0, k - S_exit)
                    miss += 1
                ex[nm] = float(p)
            credit = float(legs["sc"].px + legs["sp"].px - legs["lc"].px - legs["lp"].px)
            cc = (arr["sc"][3] + arr["sp"][3] - arr["lc"][3] - arr["lp"][3])[j0:jN + 1]
            path = credit - cc
            good = np.isfinite(path)
            exit_pnl = credit - (ex["sc"] + ex["sp"] - ex["lc"] - ex["lp"])

            slip = 0.0
            for nm in arr:
                o, h, l, c, v = arr[nm]
                hs = half_spread_rupees(h[j0:jN + 1], l[j0:jN + 1], c[j0:jN + 1], legs[nm]["px"])
                slip += (hs if np.isfinite(hs) else 0.10) * 2 * QTY
            stat = IC._statutory(
                buy_turn=(legs["lc"].px + legs["lp"].px + ex["sc"] + ex["sp"]) * QTY,
                sell_turn=(legs["sc"].px + legs["sp"].px + ex["lc"] + ex["lp"]) * QTY,
                orders=8, on=exp_day)

            rec = {"expiry": e, "cfg": f"{ds}/{dl}", "entry_date": d0, "exit_date": exp_day,
                   "dte": round(dte0, 2), "sessions_held": len(span), "spot_in": S0,
                   "spot_out": S_exit, "legs_marked_intrinsic": miss,
                   "sc_k": int(legs["sc"].k), "sp_k": int(legs["sp"].k),
                   "lc_k": int(legs["lc"].k), "lp_k": int(legs["lp"].k),
                   "sc_d": round(float(legs["sc"].d), 3), "sp_d": round(float(legs["sp"].d), 3),
                   "lc_d": round(float(legs["lc"].d), 3), "lp_d": round(float(legs["lp"].d), 3),
                   "lc_is_outermost": int(int(legs["lc"].k) == int(chain[chain.cp == "CE"].k.max())),
                   "lp_is_outermost": int(int(legs["lp"].k) == int(chain[chain.cp == "PE"].k.min())),
                   "credit": credit, "exit_pnl_pts": float(exit_pnl),
                   "max_profit_pts": float(np.nanmax(path[good])),
                   "max_dd_pts": float(np.nanmin(path[good])),
                   "call_width": int(legs["lc"].k - legs["sc"].k),
                   "put_width": int(legs["sp"].k - legs["lp"].k),
                   "gross": exit_pnl * QTY, "slip": slip, "stat": stat,
                   "net": exit_pnl * QTY - slip - stat}
            rec["maxloss"] = (max(rec["call_width"], rec["put_width"]) - credit) * QTY

            for nm in ("sc", "sp"):
                o, h, l, c, v = arr[nm]
                trg = hedge_leg(o, h, l, c, bounds, float(legs[nm]["px"]), j0, jN)
                rec[f"h_{nm}_n"] = len(trg)
                rec[f"h_{nm}_pnl"] = sum(t[0] for t in trg)
                rec[f"h_{nm}_prem"] = sum(t[1] for t in trg)
                hs = half_spread_rupees(h[j0:jN + 1], l[j0:jN + 1], c[j0:jN + 1], legs[nm]["px"])
                hs = hs if np.isfinite(hs) else 0.10
                rec[f"h_{nm}_cost"] = sum(
                    IC._statutory(t[1] * QTY, t[2] * QTY, 2, on=exp_day) + 2 * hs * QTY for t in trg)
            rec["h_pnl"] = rec["h_sc_pnl"] + rec["h_sp_pnl"]
            rec["h_prem"] = rec["h_sc_prem"] + rec["h_sp_prem"]
            rec["h_cost"] = rec["h_sc_cost"] + rec["h_sp_cost"]
            rec["h_n"] = rec["h_sc_n"] + rec["h_sp_n"]
            rows.append(rec)
        print(f"  {e}  entry {d0} dte {dte0:.2f} span {len(span)}", flush=True)

    t = pd.DataFrame(rows)
    t["year"] = pd.to_datetime(t["entry_date"]).dt.year
    t.to_csv(STORE / "weekly_hold.csv", index=False)
    print(f"\n{len(t)} trades, {t.expiry.nunique()} expiries, "
          f"{t.entry_date.min()} .. {t.exit_date.max()}")
    print(t.groupby(["cfg", "year"]).size().to_string())
    print("\nrealised entry deltas and strike-band diagnostics:")
    print(t.groupby("cfg")[["sc_d", "sp_d", "lc_d", "lp_d", "dte", "sessions_held",
                            "credit", "call_width", "put_width",
                            "lc_is_outermost", "lp_is_outermost"]].mean().round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
