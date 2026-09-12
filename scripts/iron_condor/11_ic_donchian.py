#!/usr/bin/env python3
"""
11_ic_donchian.py -- Blind iron condor + Donchian buy-stop hedge, jointly,
on real 1-minute option price paths.

The two mechanisms that survived the Aug-2026 audit of the IC framework have
never been backtested together.  This does that.  Neither the Fib/R-squared
day-type classifier nor the OR-move defense appears anywhere in this file:
both were found to be look-ahead contaminated with no live form.

IRON CONDOR  (reference methodology, reproduced exactly)
  entry 09:45, exit 15:00 (last tick at or before)
  short legs at the target short delta, wings at the target wing delta
  implied vol by bisection from each strike's ENTRY-time price, at the
    entry-time spot and entry-time time-to-expiry; delta from that same basis
  net_credit    = sc + sp - lc - lp                       (per unit, at entry)
  closing_cost(t) = sc(t) + sp(t) - lc(t) - lp(t)
  pnl(t)        = net_credit - closing_cost(t)
  nearest weekly expiry STRICTLY after the session date (no 0-DTE)

DONCHIAN BUY-STOP HEDGE
  Two constructions, run side by side on each short leg independently:
   OPT   channel on the short leg's OWN 1-minute price, 15-period lookback,
         buy-stop at channel high + Rs 1, exit on a Rs 10 trailing stop
   SPOT  channel on 5-minute SPOT, 15-period lookback (75 min), buy-stop 1
         point beyond the channel, hedge instrument is the short leg's own
         contract, exit on a 40-point trailing stop in spot
  One trigger per leg per session.  Hedge is a bought option: premium only,
  no additional margin.

Output: one row per session per delta config, with hedge P&L broken out so
sizing multiples can be applied downstream without re-running.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.core import STORE, load_spot
from lib.bs import implied_vol, delta, years_to_expiry

OPT = STORE / "opt2"
ENTRY_T = pd.Timestamp("09:45").time()
EXIT_T = pd.Timestamp("15:00").time()
DELTA_CFGS = [(0.35, 0.15), (0.25, 0.10), (0.25, 0.05)]
DON_OPT_N, DON_OPT_TRAIL, DON_OPT_TICK = 15, 10.0, 1.0
DON_SPOT_N, DON_SPOT_TRAIL, DON_SPOT_TICK = 15, 40.0, 1.0
MIN_PREM = 0.5


def align(day_opt, grid):
    out = {}
    for (s, cp), g in day_opt.groupby(["strike", "cp"], observed=True):
        g = g.drop_duplicates("ts").set_index("ts").reindex(grid)
        out[(int(s), str(cp))] = (g["o"].ffill(limit=3).to_numpy(dtype=float),
                                  g["h"].ffill(limit=3).to_numpy(dtype=float),
                                  g["l"].ffill(limit=3).to_numpy(dtype=float),
                                  g["c"].ffill(limit=3).to_numpy(dtype=float),
                                  g["v"].to_numpy(dtype=float))
    return out


def pick_strikes(book, j, S, T, ds, dl):
    """Delta-selected legs from the entry-time chain."""
    rows = []
    for (k, cp), (o, h, l, c, v) in book.items():
        p = c[j]
        if not np.isfinite(p) or p < MIN_PREM or not (v[j] > 0):
            continue
        iv = implied_vol(float(p), S, float(k), T, cp)
        if iv is None:
            continue
        rows.append((k, cp, float(p), delta(S, float(k), T, iv, cp), iv))
    if not rows:
        return None
    ch = pd.DataFrame(rows, columns=["k", "cp", "px", "d", "iv"])
    ce = ch[ch.cp == "CE"]
    pe = ch[ch.cp == "PE"]
    if ce.empty or pe.empty:
        return None
    sc = ce.iloc[(ce["d"] - ds).abs().argsort().iloc[0]]
    sp = pe.iloc[(pe["d"] + ds).abs().argsort().iloc[0]]
    ce_far = ce[ce["k"] > sc["k"]]
    pe_far = pe[pe["k"] < sp["k"]]
    if ce_far.empty or pe_far.empty:
        return None
    lc = ce_far.iloc[(ce_far["d"] - dl).abs().argsort().iloc[0]]
    lp = pe_far.iloc[(pe_far["d"] + dl).abs().argsort().iloc[0]]
    return {"sc": sc, "sp": sp, "lc": lc, "lp": lp}


def hedge_option(series, j, k, entry_prem=None):
    """Donchian buy-stop on the option's own 1-minute price.

    The trailing stop is armed from the bar AFTER entry -- a stop order and its
    own trail cannot both execute on the bar that filled it.

    When `entry_prem` is given, the buy-stop is only live while that short leg
    is actually bleeding (its premium is above where it was sold).  That is the
    "compensate the pressured leg" rule; without it the hedge fires on both
    legs almost every session and simply pays two premiums a day.
    """
    o, h, l, c, v = series
    for i in range(j + DON_OPT_N, k + 1):
        win = h[i - DON_OPT_N:i]
        if not np.isfinite(win).all():
            continue
        stop = np.nanmax(win) + DON_OPT_TICK
        if entry_prem is not None and stop <= entry_prem:
            continue
        if not (np.isfinite(h[i]) and h[i] >= stop):
            continue
        # a buy-stop that gaps through fills at the open, not at the stop level
        ent = max(stop, float(o[i])) if np.isfinite(o[i]) else stop
        run = ent
        for m in range(i + 1, k + 1):
            trail = run - DON_OPT_TRAIL
            if np.isfinite(l[m]) and l[m] <= trail:
                fill = min(trail, float(o[m])) if np.isfinite(o[m]) else trail
                return {"pnl": fill - ent, "entry": ent, "exit": fill,
                        "at": i, "out": m, "why": "trail"}
            if np.isfinite(h[m]):
                run = max(run, h[m])
        return {"pnl": float(c[k]) - ent, "entry": ent, "exit": float(c[k]),
                "at": i, "out": k, "why": "eod"}
    return None


def hedge_spot(series, spot_h, spot_l, spot_c, bar5, j, k, side, entry_prem=None):
    """Donchian buy-stop on 5-minute SPOT; hedge instrument is the short leg.

    Same two corrections as the option-based version: the spot trailing stop is
    armed from the bar after the fill, and the hedge only arms while the short
    leg it protects is bleeding.
    """
    o, h, l, c, v = series
    for b in range(DON_SPOT_N, len(bar5)):
        close_idx = bar5[b]["close_idx"]
        if close_idx < j or close_idx > k:
            continue
        hi = max(bar5[x]["h"] for x in range(b - DON_SPOT_N, b))
        lo = min(bar5[x]["l"] for x in range(b - DON_SPOT_N, b))
        trig = (bar5[b]["h"] >= hi + DON_SPOT_TICK) if side == "CE" else (bar5[b]["l"] <= lo - DON_SPOT_TICK)
        if not trig:
            continue
        i = close_idx + 1
        if i > k or not np.isfinite(o[i]) or o[i] <= 0:
            return None
        ent = float(o[i])
        if entry_prem is not None and ent <= entry_prem:
            continue
        run = spot_h[i] if side == "CE" else spot_l[i]
        for m in range(i + 1, k + 1):
            hit = (spot_l[m] <= run - DON_SPOT_TRAIL) if side == "CE" else (spot_h[m] >= run + DON_SPOT_TRAIL)
            if hit and np.isfinite(c[m]):
                return {"pnl": float(c[m]) - ent, "entry": ent, "exit": float(c[m]),
                        "at": i, "out": m, "why": "trail"}
            run = max(run, spot_h[m]) if side == "CE" else min(run, spot_l[m])
        return {"pnl": float(c[k]) - ent, "entry": ent, "exit": float(c[k]),
                "at": i, "out": k, "why": "eod"}
    return None


def main():
    spot = load_spot()
    emap = pd.read_csv(STORE / "expiry_map.csv", index_col=0)["expiry"].astype(str)
    emap.index = emap.index.astype(str)
    by_exp = {}
    for d, e in emap.items():
        by_exp.setdefault(e, []).append(d)

    rows, skipped = [], {}
    for e in sorted(by_exp):
        f = OPT / f"{e}.parquet"
        if not f.exists():
            continue
        opt = pd.read_parquet(f)
        opt["ts"] = pd.to_datetime(opt["ts"])
        opt["date"] = opt["ts"].dt.date.astype(str)
        for d in sorted(by_exp[e]):
            day_opt = opt[opt["date"] == d]
            ds_ = spot[spot["date"] == d]
            if day_opt.empty or ds_.empty:
                skipped[d] = "no data"
                continue
            grid = pd.DatetimeIndex(ds_["ts"])
            times = np.array([t.time() for t in grid])
            ji = np.where(times == ENTRY_T)[0]
            ki = np.where(times <= EXIT_T)[0]
            if len(ji) == 0 or len(ki) == 0:
                skipped[d] = "no entry/exit tick"
                continue
            j, k = int(ji[0]), int(ki[-1])
            if k - j < 60:
                skipped[d] = "short session"
                continue
            book = align(day_opt, grid)
            S = float(ds_["c"].to_numpy()[j])
            T = years_to_expiry(grid[j], e)
            sh = ds_["h"].to_numpy(dtype=float)
            sl = ds_["l"].to_numpy(dtype=float)
            sc_ = ds_["c"].to_numpy(dtype=float)

            # 5-minute spot bars for the SPOT hedge construction
            bar5 = []
            for b0 in range(0, len(grid), 5):
                b1 = min(b0 + 5, len(grid))
                if b1 - b0 < 5:
                    break
                bar5.append({"h": float(np.nanmax(sh[b0:b1])), "l": float(np.nanmin(sl[b0:b1])),
                             "close_idx": b1 - 1})

            for (dsh, dwg) in DELTA_CFGS:
                legs = pick_strikes(book, j, S, T, dsh, dwg)
                if legs is None:
                    skipped[d] = "no delta match"
                    continue
                arr = {n: book[(int(legs[n]["k"]), "CE" if n in ("sc", "lc") else "PE")]
                       for n in ("sc", "sp", "lc", "lp")}
                if any(not np.isfinite(arr[n][3][k]) for n in arr):
                    skipped[d] = "leg missing at exit"
                    continue
                credit = float(legs["sc"]["px"] + legs["sp"]["px"] - legs["lc"]["px"] - legs["lp"]["px"])
                cc = (arr["sc"][3] + arr["sp"][3] - arr["lc"][3] - arr["lp"][3])[j:k + 1]
                pnl_path = credit - cc
                good = np.isfinite(pnl_path)
                if good.sum() < 60:
                    skipped[d] = "sparse path"
                    continue
                rec = {
                    "date": d, "expiry": e, "cfg": f"{dsh}/{dwg}",
                    "spot": S, "dte": round(T * 365, 2),
                    "sc_k": int(legs["sc"]["k"]), "sp_k": int(legs["sp"]["k"]),
                    "lc_k": int(legs["lc"]["k"]), "lp_k": int(legs["lp"]["k"]),
                    "sc_d": round(float(legs["sc"]["d"]), 3), "sp_d": round(float(legs["sp"]["d"]), 3),
                    "lc_d": round(float(legs["lc"]["d"]), 3), "lp_d": round(float(legs["lp"]["d"]), 3),
                    "credit": credit,
                    "ic_pnl": float(pnl_path[good][-1]),
                    "ic_maxprofit": float(np.nanmax(pnl_path)),
                    "ic_maxdd": float(np.nanmin(pnl_path)),
                    "call_width": int(legs["lc"]["k"] - legs["sc"]["k"]),
                    "put_width": int(legs["sp"]["k"] - legs["lp"]["k"]),
                    "entry_sc": float(legs["sc"]["px"]), "entry_sp": float(legs["sp"]["px"]),
                    "entry_lc": float(legs["lc"]["px"]), "entry_lp": float(legs["lp"]["px"]),
                    "exit_sc": float(arr["sc"][3][k]), "exit_sp": float(arr["sp"][3][k]),
                    "exit_lc": float(arr["lc"][3][k]), "exit_lp": float(arr["lp"][3][k]),
                }
                for tag in ("opt", "spot", "optg", "spotg"):
                    base = "opt" if tag.startswith("opt") else "spot"
                    gated = tag.endswith("g")
                    tot, prem, n_trig = 0.0, 0.0, 0
                    for legname, side in (("sc", "CE"), ("sp", "PE")):
                        ep = float(legs[legname]["px"]) if gated else None
                        if base == "opt":
                            hres = hedge_option(arr[legname], j, k, entry_prem=ep)
                        else:
                            hres = hedge_spot(arr[legname], sh, sl, sc_, bar5, j, k, side,
                                              entry_prem=ep)
                        if hres:
                            n_trig += 1
                            tot += hres["pnl"]
                            prem += hres["entry"]
                            rec[f"h{tag}_{legname}_pnl"] = round(hres["pnl"], 2)
                            rec[f"h{tag}_{legname}_ent"] = round(hres["entry"], 2)
                            rec[f"h{tag}_{legname}_ext"] = round(hres["exit"], 2)
                    rec[f"h{tag}_pnl"] = tot
                    rec[f"h{tag}_prem"] = prem
                    rec[f"h{tag}_n"] = n_trig
                rows.append(rec)
        print(f"  {e}  rows={len(rows)}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(STORE / "ic_donchian_daily.csv", index=False)
    print(f"\n{len(df)} rows, {df['date'].nunique()} sessions, "
          f"{df['date'].min()} .. {df['date'].max()}")
    print("skipped:", len(skipped))
    for cfg, g in df.groupby("cfg"):
        print(f"  cfg {cfg}: n={len(g)}  mean delta sc={g['sc_d'].mean():.3f} "
              f"sp={g['sp_d'].mean():.3f} lc={g['lc_d'].mean():.3f} lp={g['lp_d'].mean():.3f}  "
              f"mean credit={g['credit'].mean():.1f}")
        for t in ("opt", "spot", "optg", "spotg"):
            print(f"      hedge {t:<6} triggers/day={g[f'h{t}_n'].mean():.2f}  "
                  f"gross pts/day={g[f'h{t}_pnl'].mean():+.2f}  "
                  f"win%={100*(g[f'h{t}_pnl'][g[f'h{t}_n']>0]>0).mean():.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
