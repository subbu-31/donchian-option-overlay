#!/usr/bin/env python3
"""
22_intraday_wings.py -- Wing width on the intraday horizon (09:45 -> 15:00),
scored on the same inline modelled-spread basis as the weekly-hold study so the
two horizons are directly comparable.

Each leg is charged its own effective half-spread, estimated from that leg's own
1-minute path on that session (Corwin-Schultz / Abdi-Ranaldo, floored at half a
tick), twice.  Configs: 0.25/0.05, 0.25/0.10, 0.35/0.15.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.core import STORE, load_spot
from lib.spread import half_spread_rupees
import lib.ic_costs as IC

OPT = STORE / "opt2"
QTY, RF = 750, 0.065
ENTRY_T, EXIT_T = pd.Timestamp("09:45").time(), pd.Timestamp("15:00").time()


def main():
    d = pd.read_csv(STORE / "ic_donchian_daily.csv")
    d["year"] = pd.to_datetime(d["date"]).dt.year
    d = d[d.year.isin([2025, 2026])]
    spot = load_spot()

    # per-leg half spread on the holding window, per row
    slips = []
    for e, gexp in d.groupby("expiry"):
        f = OPT / f"{int(e)}.parquet"
        if not f.exists():
            slips += [np.nan] * len(gexp)
            continue
        opt = pd.read_parquet(f)
        opt["ts"] = pd.to_datetime(opt["ts"])
        opt["date"] = opt["ts"].dt.date.astype(str)
        for _, r in gexp.iterrows():
            day = opt[(opt["date"] == r["date"]) &
                      (opt["ts"].dt.time >= ENTRY_T) & (opt["ts"].dt.time <= EXIT_T)]
            tot = 0.0
            for k, cp, prem in ((r.sc_k, "CE", r.entry_sc), (r.sp_k, "PE", r.entry_sp),
                                (r.lc_k, "CE", r.entry_lc), (r.lp_k, "PE", r.entry_lp)):
                g = day[(day.strike == int(k)) & (day.cp == cp)]
                hs = half_spread_rupees(g["h"], g["l"], g["c"], prem) if len(g) > 20 else np.nan
                tot += (hs if np.isfinite(hs) else 0.10) * 2 * QTY
            slips.append(tot)
        print(f"  {int(e)}", flush=True)
    d = d.assign(slip=slips)
    d["stat"] = [IC._statutory(
        buy_turn=(r.entry_lc + r.entry_lp + r.exit_sc + r.exit_sp) * QTY,
        sell_turn=(r.entry_sc + r.entry_sp + r.exit_lc + r.exit_lp) * QTY,
        orders=8, on=r.date) for r in d.itertuples()]
    d["gross"] = d.ic_pnl * QTY
    d["net"] = d.gross - d.slip - d.stat
    d["maxloss"] = (np.maximum(d.call_width, d.put_width) - d.credit).clip(lower=0) * QTY
    d.to_csv(STORE / "intraday_wings.csv", index=False)

    def score(pnl, cap):
        base = float(np.percentile(cap, 95))
        eq = np.cumsum(pnl)
        dd = float(np.max(np.maximum.accumulate(np.concatenate([[0.], eq])) - np.concatenate([[0.], eq])))
        r = pnl / base
        sd = r.std(ddof=1)
        ann = r.mean() * 252
        return (round(float(pnl.sum())), round(base), round(float(pnl.sum()) / base * 100, 1),
                round((ann - RF) / (sd * np.sqrt(252)), 2) if sd > 0 else np.nan,
                round(dd), round(dd / base * 100, 1),
                round(float((pnl > 0).mean()) * 100, 1))

    print("\n" + "=" * 112)
    print("  INTRADAY 09:45 -> 15:00, wing width compared   |   10 lots x 75   |   modelled per-leg slippage")
    print("=" * 112)
    print(f"  {'cfg':<11}{'period':<8}{'n':>5}{'credit':>8}{'maxloss':>9}{'gross':>8}"
          f"{'cost':>7}{'net pts':>9}{'net Rs':>10}{'capital':>10}{'ret %':>8}{'sharpe':>8}{'DD %':>7}{'win %':>7}")
    for cfg in ("0.25/0.05", "0.25/0.1", "0.35/0.15"):
        for pn, yr in (("2025", 2025), ("2026", 2026)):
            g = d[(d.cfg == cfg) & (d.year == yr)]
            if len(g) < 10:
                continue
            net, cap, ret, sh, dd, ddp, win = score(g.net.to_numpy(), g.maxloss.to_numpy())
            print(f"  {cfg:<11}{pn:<8}{len(g):>5}{g.credit.mean():>8.1f}"
                  f"{(g.maxloss/QTY).mean():>9.1f}{(g.gross/QTY).mean():>8.2f}"
                  f"{((g.slip+g.stat)/QTY).mean():>7.2f}{(g.net/QTY).mean():>9.2f}"
                  f"{net:>10,}{cap:>10,}{ret:>8.1f}{sh:>8.2f}{ddp:>7.1f}{win:>7.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
