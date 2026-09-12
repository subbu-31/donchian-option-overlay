#!/usr/bin/env python3
"""
31_cost_audit.py -- Re-price every stored NIFTY trade under the corrected,
date-aware statutory schedule and report what changed.

Two errors were found in the original constants:
  STT on the options sell side was held flat at 0.10%. It was 0.0625% before
  1 Oct 2024 and rose to 0.15% on 1 Apr 2026.
  The NSE exchange transaction charge was held at 0.03503%. It was 0.0495%
  before 1 Oct 2024 and is 0.03553% on the current published schedule.

The NIFTY sample sits almost entirely inside the middle regime, so the effect
should be confined to April 2026. This verifies that rather than assuming it.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.core import STORE
import lib.ic_costs as IC

QTY = 750
OLD_STT, OLD_EXCH = 0.0010, 0.00035503


def old_statutory(buy, sell, orders):
    brok = orders * IC.BROKERAGE
    exch = (buy + sell) * OLD_EXCH
    stt = sell * OLD_STT
    sebi = (buy + sell) * IC.SEBI
    stamp = buy * IC.STAMP_BUY
    return brok + exch + stt + sebi + stamp + IC.GST * (brok + exch + sebi)


def main():
    d = pd.read_csv(STORE / "intraday_wings.csv")
    d["year"] = pd.to_datetime(d["date"]).dt.year
    sell = (d.entry_sc + d.entry_sp + d.exit_lc + d.exit_lp) * QTY
    buy = (d.entry_lc + d.entry_lp + d.exit_sc + d.exit_sp) * QTY
    d["stat_old"] = [old_statutory(b, s, 8) for b, s in zip(buy, sell)]
    d["stat_new"] = [IC._statutory(b, s, 8, on=dt)
                     for b, s, dt in zip(buy, sell, d["date"])]
    d["net_old"] = d.gross - d.slip - d.stat_old
    d["net_new"] = d.gross - d.slip - d.stat_new
    d["delta"] = d.net_new - d.net_old

    print("=" * 100)
    print("NIFTY -- effect of the corrected date-aware statutory schedule")
    print("=" * 100)
    print(f"  {'cfg':<11}{'period':<9}{'n':>5}{'stat old':>11}{'stat new':>11}"
          f"{'net old':>12}{'net new':>12}{'change':>11}")
    for cfg in sorted(d.cfg.unique()):
        for lab, m in (("2025", d.year == 2025), ("2026", d.year == 2026)):
            g = d[(d.cfg == cfg) & m]
            if g.empty:
                continue
            print(f"  {cfg:<11}{lab:<9}{len(g):>5}{g.stat_old.mean():>11,.0f}"
                  f"{g.stat_new.mean():>11,.0f}{g.net_old.sum():>12,.0f}"
                  f"{g.net_new.sum():>12,.0f}{g.delta.sum():>11,.0f}")
    print("\n  by month, 0.35/0.15, mean statutory per session")
    g = d[d.cfg == "0.35/0.15"].copy()
    g["m"] = pd.to_datetime(g["date"]).dt.to_period("M").astype(str)
    mm = g.groupby("m").agg(n=("stat_old", "size"), old=("stat_old", "mean"),
                            new=("stat_new", "mean"))
    mm["change"] = mm.new - mm.old
    print("  " + mm.round(0).tail(6).to_string().replace("\n", "\n  "))

    # headline Sharpe before and after
    print("\n  headline impact, 0.35/0.15")
    for lab, m in (("2025 in-sample", d.year == 2025), ("2026 holdout", d.year == 2026)):
        g = d[(d.cfg == "0.35/0.15") & m]
        base = float(np.percentile(g.maxloss, 95))
        out = []
        for col in ("net_old", "net_new"):
            r = g[col].to_numpy() / base
            sd = r.std(ddof=1)
            out.append((r.mean() * 252 - 0.065) / (sd * np.sqrt(252)))
        print(f"    {lab:<16} Sharpe {out[0]:+.3f} -> {out[1]:+.3f}   "
              f"net Rs {g.net_old.sum():,.0f} -> {g.net_new.sum():,.0f}")
    d.to_csv(STORE / "intraday_wings_costfix.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
