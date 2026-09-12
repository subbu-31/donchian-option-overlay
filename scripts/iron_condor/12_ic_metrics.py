#!/usr/bin/env python3
"""
12_ic_metrics.py -- Risk-adjusted scoring of the blind IC and the IC + Donchian
buy-stop hedge.

Slippage is the one assumption that decides this strategy, so it is a swept
parameter rather than a buried constant.  A four-leg condor crosses the spread
eight times a session; the hedge adds two crossings per triggered leg.  At 10
lots that is 750 units per crossing, so every 0.1 point of assumed slippage
costs Rs 600 a day before the strategy has done anything.

Capital base is the condor's own defined risk: max(call width, put width) minus
net credit, per unit, times quantity, plus hedge premium outlaid.  The 95th
percentile of that daily series is used, so the base covers essentially every
session without being set by a single outlier.

Sharpe and Sortino are on daily returns against that base, annualised at 252,
risk-free 6.5%.  The Sharpe RATIO itself is invariant to the base; only the
risk-free subtraction is not, which is why the base is stated on every line.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.core import STORE
import lib.ic_costs as IC

LOTS, LOT_SIZE = 10, 75
QTY = LOTS * LOT_SIZE
RF = 0.065
SLIP_LADDER = [0.00, 0.10, 0.25, 0.50]
PERIODS = {"2025 in-sample": 2025, "2026 holdout": 2026}


def arm_pnl(df, hedge, mult, slip):
    """Rupee P&L per session and the capital that session required."""
    ic_g = df["ic_pnl"].to_numpy() * QTY
    maxloss = (np.maximum(df["call_width"], df["put_width"]) - df["credit"]).clip(lower=0).to_numpy() * QTY

    ic_c = np.array([IC.ic_cost(
        {"sc": r.entry_sc, "sp": r.entry_sp, "lc": r.entry_lc, "lp": r.entry_lp},
        {"sc": r.exit_sc, "sp": r.exit_sp, "lc": r.exit_lc, "lp": r.exit_lp},
        QTY, slip_pts=slip, on=r.date) for r in df.itertuples()])
    if hedge is None:
        return ic_g - ic_c, maxloss, ic_g, ic_c

    hq = QTY * mult
    hp = df[f"h{hedge}_pnl"].to_numpy() * hq
    hprem = df[f"h{hedge}_prem"].to_numpy() * hq
    hc = np.zeros(len(df))
    for i, r in enumerate(df.itertuples()):
        for leg in ("sc", "sp"):
            e = getattr(r, f"h{hedge}_{leg}_ent", np.nan)
            x = getattr(r, f"h{hedge}_{leg}_ext", np.nan)
            if np.isfinite(e) and np.isfinite(x):
                hc[i] += IC.hedge_cost(float(e), float(x), hq, slip_pts=slip, on=r.date)
    return ic_g + hp - ic_c - hc, maxloss + hprem, ic_g + hp, ic_c + hc


def score(pnl, cap, gross, cost, label):
    base = float(np.percentile(cap, 95))
    eq = np.cumsum(pnl)
    dd = float(np.max(np.maximum.accumulate(np.concatenate([[0.0], eq])) -
                      np.concatenate([[0.0], eq])))
    r = pnl / base
    mu, sd = r.mean(), r.std(ddof=1)
    dn = r[r < 0]
    dsd = dn.std(ddof=1) if len(dn) > 1 else np.nan
    ann = mu * 252
    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
    # slippage points per leg-side at which this arm breaks even
    return {
        "arm": label, "n": len(pnl),
        "gross": round(float(gross.sum())), "cost": round(float(cost.sum())),
        "net": round(float(pnl.sum())), "capital": round(base),
        "ret_%": round(float(pnl.sum()) / base * 100, 1),
        "ann_%": round(ann * 100, 1),
        "sharpe": round((ann - RF) / (sd * np.sqrt(252)), 2) if sd > 0 else np.nan,
        "sortino": round((ann - RF) / (dsd * np.sqrt(252)), 2) if dsd and dsd > 0 else np.nan,
        "maxDD": round(dd), "DD_%": round(dd / base * 100, 1),
        "ret/DD": round(float(pnl.sum()) / dd, 2) if dd > 0 else np.nan,
        "win_%": round(float((pnl > 0).mean()) * 100, 1),
        "rr": round(abs(wins.mean() / losses.mean()), 2) if len(wins) and len(losses) else np.nan,
        "pf": round(float(wins.sum() / -losses.sum()), 2) if len(losses) and losses.sum() < 0 else np.nan,
        "worst": round(float(pnl.min())),
        "day": round(float(pnl.mean())),
    }


def breakeven_slip(df, hedge, mult):
    lo, hi = 0.0, 3.0
    f = lambda s: arm_pnl(df, hedge, mult, s)[0].sum()
    if f(lo) <= 0:
        return 0.0
    if f(hi) > 0:
        return hi
    for _ in range(40):
        mid = (lo + hi) / 2
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid
    return round((lo + hi) / 2, 3)


def main():
    df = pd.read_csv(STORE / "ic_donchian_daily.csv")
    df["year"] = pd.to_datetime(df["date"]).dt.year
    df = df[df["year"].isin([2025, 2026])].sort_values("date").reset_index(drop=True)

    arms = [("IC blind", None, 1)]
    for h, nm in (("optg", "optDon-gated"), ("opt", "optDon-always"),
                  ("spotg", "spotDon-gated")):
        for m in (1, 2, 3):
            arms.append((f"IC + {nm} {m}x", h, m))

    cols = ["arm", "n", "gross", "cost", "net", "capital", "ret_%", "ann_%",
            "sharpe", "sortino", "maxDD", "DD_%", "ret/DD", "win_%", "rr", "pf", "worst", "day"]
    out = []
    for cfg in sorted(df["cfg"].unique()):
        gc = df[df.cfg == cfg]
        print("\n" + "#" * 122)
        print(f"#  DELTA CONFIG {cfg}    {LOTS} lots x {LOT_SIZE} = {QTY} units    "
              f"entry 09:45  exit 15:00  no classifier  no OR-move defense")
        print("#" * 122)

        print("\n  BREAKEVEN SLIPPAGE (points per leg, per side, at which the arm's net P&L reaches zero)")
        print("  " + "-" * 96)
        hdr = f"  {'arm':<18}"
        for pn in PERIODS:
            hdr += f"{pn:>20}"
        print(hdr)
        for label, h, m in arms:
            line = f"  {label:<18}"
            for pn, yr in PERIODS.items():
                g = gc[gc.year == yr]
                line += f"{breakeven_slip(g, h, m):>20.3f}"
            print(line)

        for slip in SLIP_LADDER:
            for pn, yr in PERIODS.items():
                g = gc[gc.year == yr]
                if len(g) < 20:
                    continue
                rows = []
                for label, h, m in arms:
                    pnl, cap, gr, co = arm_pnl(g, h, m, slip)
                    r = score(pnl, cap, gr, co, label)
                    r.update({"cfg": cfg, "period": pn, "slip": slip})
                    out.append(r)
                    rows.append(r)
                print(f"\n  --- slippage {slip:.2f} pts/leg/side   |   {pn}   |   {len(g)} sessions ---")
                print(pd.DataFrame(rows)[cols].to_string(index=False))

    res = pd.DataFrame(out)
    res.to_csv(STORE / "ic_donchian_metrics.csv", index=False)
    print("\n\nwrote ic_donchian_metrics.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
