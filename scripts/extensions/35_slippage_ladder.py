#!/usr/bin/env python3
"""
35_slippage_ladder.py -- How much does execution quality actually buy?

Re-prices the stored intraday condor trades (09:45 -> 15:00, 10 lots x 75)
under progressively better fills, holding everything else fixed.  Statutory
charges are NOT scaled -- they are a tax, not a spread.

Sharpe is reported with Lo (2002) standard errors so the t convention is
explicit and identical across every row and every sample.
"""
import sys
from math import erfc
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.core import STORE

QTY = 750
LEG_SIDES = 8          # 4 condor legs x (entry + exit)
FACTORS = [1.00, 0.75, 0.50, 0.25, 0.00]


def lo_sharpe(x, freq=252):
    x = np.asarray(x, float)
    n = len(x)
    sd = x.std(ddof=1)
    sr_p = x.mean() / sd
    se_p = np.sqrt((1.0 + 0.5 * sr_p ** 2) / n)
    t = sr_p / se_p
    p = erfc(abs(t) / np.sqrt(2.0))
    return n, x.mean(), sd, sr_p * np.sqrt(freq), se_p * np.sqrt(freq), t, p


def main():
    src = STORE / "intraday_wings_costfix.csv"
    d = pd.read_csv(src)
    if "stat_new" in d.columns:
        d["stat"] = d["stat_new"]
    d["date"] = pd.to_datetime(d["date"])
    d["year"] = d["date"].dt.year

    hs = (d["slip"] / (LEG_SIDES * QTY)).mean()
    print(f"source            : {src.name}  ({len(d)} rows, {d.date.nunique()} sessions)")
    print(f"modelled half-spr : {hs:.3f} pts/leg/side  (tick floor 0.025)")
    print(f"statutory         : Rs {d.stat.mean():,.0f}/session = {(d.stat/QTY).mean():.2f} pts  -- not scaled\n")

    for cfg in ["0.35/0.15", "0.25/0.1", "0.25/0.05"]:
        for label, sel in [("2025 IS", d.year == 2025),
                           ("2026 OOS", d.year == 2026),
                           ("full", d.date.notna())]:
            g = d[(d.cfg == cfg) & sel]
            if len(g) < 20:
                continue
            print(f"  {cfg}   {label}   n={len(g)}")
            print(f"    {'xslip':>6}{'half-spr':>10}{'net Rs':>12}{'Rs/sess':>10}"
                  f"{'sd':>9}{'Sharpe':>8}{'SE':>7}{'t':>7}{'p':>8}")
            for f in FACTORS:
                net = g.gross - f * g.slip - g.stat
                n, mu, sd, sr, se, t, p = lo_sharpe(net)
                print(f"    {f:>6.2f}{f*hs:>10.3f}{net.sum():>12,.0f}{mu:>10,.0f}"
                      f"{sd:>9,.0f}{sr:>8.2f}{se:>7.2f}{t:>7.2f}{p:>8.3f}")
            sdf = (g.gross - g.slip - g.stat).std(ddof=1)
            sdz = (g.gross - g.stat).std(ddof=1)
            print(f"    sd modelled {sdf:,.0f} -> sd perfect {sdz:,.0f}  "
                  f"({100*(sdz/sdf-1):+.2f}%)\n")

    g = d[(d.cfg == "0.35/0.15") & (d.year == 2025)]
    print("cost decomposition, 0.35/0.15, 2025:")
    for nm, s in [("slippage", g.slip), ("statutory", g.stat), ("gross P&L", g.gross)]:
        print(f"  {nm:<10} mean Rs {s.mean():>9,.0f}   sd Rs {s.std():>9,.0f}"
              f"   cv {abs(s.std()/s.mean()):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
