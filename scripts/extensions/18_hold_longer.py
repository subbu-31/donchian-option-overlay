#!/usr/bin/env python3
"""
18_hold_longer.py -- The lever the cost model actually points at.

Crossing plus statutory cost is ~Rs 1,134 per round trip and barely moves with
holding time.  The condor's edge is theta, which accrues with holding time.  So
the ratio that matters is cost per unit of time held, and the single-session
trade pays the full round trip for 5.25 hours of decay.

This holds the same delta-selected condor for one, two or three sessions,
exiting at 15:00 on the last one, never carrying into expiry day (the standing
no-0-DTE rule).  Entries are taken non-overlapping so the P&L series stays
independent: a 2-session hold trades every other eligible session.

Costs are the same eight crossings in every arm, charged at each leg's own
estimated half-spread.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.core import STORE, load_spot
import lib.ic_costs as IC

OPT = STORE / "opt2"
QTY = 750
RF = 0.065
ENTRY_T, EXIT_T = pd.Timestamp("09:45").time(), pd.Timestamp("15:00").time()
HOLDS = [1, 2, 3]


def main():
    df = pd.read_csv(STORE / "ic_modelled_daily.csv")
    df["year"] = pd.to_datetime(df["date"]).dt.year
    spot = load_spot()
    sessions = sorted(spot["date"].unique())
    sidx = {d: i for i, d in enumerate(sessions)}

    rows = []
    for e, gexp in df.groupby("expiry"):
        f = OPT / f"{int(e)}.parquet"
        if not f.exists():
            continue
        opt = pd.read_parquet(f)
        opt["ts"] = pd.to_datetime(opt["ts"])
        opt["date"] = opt["ts"].dt.date.astype(str)
        exp_s = pd.Timestamp(str(int(e))).strftime("%Y-%m-%d")
        for _, r in gexp.iterrows():
            d = r["date"]
            if d not in sidx:
                continue
            legs = [(int(r.sc_k), "CE", +1, r.entry_sc), (int(r.sp_k), "PE", +1, r.entry_sp),
                    (int(r.lc_k), "CE", -1, r.entry_lc), (int(r.lp_k), "PE", -1, r.entry_lp)]
            rec = {"date": d, "expiry": int(e), "credit": r.credit,
                   "maxloss": r.maxloss, "slip": r.slip_ic, "width_rel": r.width_rel}
            for hold in HOLDS:
                j = sidx[d] + hold - 1
                if j >= len(sessions):
                    continue
                xd = sessions[j]
                # never hold into expiry day itself
                if xd >= exp_s:
                    continue
                day = opt[opt["date"] == xd]
                if day.empty:
                    continue
                cc, ok = 0.0, True
                ex = {}
                for k, cp, sign, _ in legs:
                    g = day[(day.strike == k) & (day.cp == cp)]
                    g = g[g["ts"].dt.time <= EXIT_T]
                    if g.empty:
                        ok = False
                        break
                    px = float(g.sort_values("ts")["c"].iloc[-1])
                    ex[(k, cp)] = px
                    cc += sign * px
                if not ok:
                    continue
                gross = (r.credit - cc) * QTY
                stat = IC._statutory(
                    buy_turn=(r.entry_lc + r.entry_lp + ex[(int(r.sc_k), "CE")] + ex[(int(r.sp_k), "PE")]) * QTY,
                    sell_turn=(r.entry_sc + r.entry_sp + ex[(int(r.lc_k), "CE")] + ex[(int(r.lp_k), "PE")]) * QTY,
                    orders=8, on=xd)
                rec[f"gross_h{hold}"] = gross
                rec[f"net_h{hold}"] = gross - r.slip_ic - stat
                rec[f"exit_h{hold}"] = xd
            rows.append(rec)
        print(f"  {int(e)}", flush=True)

    t = pd.DataFrame(rows)
    t["year"] = pd.to_datetime(t["date"]).dt.year
    t.to_csv(STORE / "hold_longer.csv", index=False)

    print("\n" + "=" * 100)
    print("HOLDING LONGER  --  same 8 crossings, more theta per round trip")
    print("=" * 100)
    print(f"  {'period':<16}{'hold':>6}{'trades':>8}{'gross/trade':>13}{'cost/trade':>12}"
          f"{'net/trade':>11}{'net/day':>10}{'win%':>7}{'sharpe':>8}{'ret/DD':>8}")
    for pname, yr in (("2025 in-sample", 2025), ("2026 holdout", 2026)):
        for hold in HOLDS:
            col = f"net_h{hold}"
            if col not in t:
                continue
            g = t[(t["year"] == yr) & t[col].notna()].sort_values("date").reset_index(drop=True)
            if len(g) < 12:
                continue
            # non-overlapping: take every `hold`-th eligible entry
            g = g.iloc[::hold].reset_index(drop=True)
            pnl = g[col].to_numpy()
            gr = g[f"gross_h{hold}"].to_numpy()
            base = float(np.percentile(g["maxloss"], 95))
            trades_per_yr = 252.0 / hold
            r = pnl / base
            sd = r.std(ddof=1)
            ann = r.mean() * trades_per_yr
            sh = (ann - RF) / (sd * np.sqrt(trades_per_yr)) if sd > 0 else np.nan
            eq = np.cumsum(pnl)
            dd = float(np.max(np.maximum.accumulate(np.concatenate([[0.], eq])) - np.concatenate([[0.], eq])))
            print(f"  {pname:<16}{hold:>6}{len(g):>8}{gr.mean():>13,.0f}"
                  f"{(gr-pnl).mean():>12,.0f}{pnl.mean():>11,.0f}{pnl.mean()/hold:>10,.0f}"
                  f"{100*(pnl>0).mean():>7.1f}{sh:>8.2f}"
                  f"{(pnl.sum()/dd if dd>0 else np.nan):>8.2f}")
    print("\n  net/day is the honest comparison: a 2-session hold ties up capital for two days.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
