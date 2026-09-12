#!/usr/bin/env python3
"""
32_term_structure.py -- The implied-volatility term structure, and whether its
slope forecasts anything.

Delta hedging is unavailable (no historical NIFTY futures), so the next
structure that harvests variance premium differently is the calendar: sell the
near expiry, buy the far one. Before building any of that machinery, apply the
lesson from the power work and test the FORECAST, not the payoff.

The archive turns out to carry a real curve: 6.2 simultaneously live expiries
per session on average, spanning days to months. For each session at 09:45 the
ATM implied volatility of every live expiry is backed out, and the curve is
summarised by

  iv_near     ATM IV of the nearest expiry strictly after the session
  iv_far      ATM IV of the furthest expiry with a usable ATM chain
  slope       (iv_far - iv_near) / log(dte_far / dte_near)

Two questions, both forecast tests rather than P&L tests:

  Q1  does the slope predict the NEAR expiry's variance risk premium --
      iv_near minus the volatility actually realised over the rest of the
      session? If it does, it is a sizing signal for the short-premium book
      that already exists, orthogonal to the channel-width state.

  Q2  does the slope mean-revert? A calendar spread is long the slope, so a
      predictable slope is the precondition for that trade having anything
      behind it.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.core import STORE, load_spot
from lib.bs import implied_vol, years_to_expiry

OPT = STORE / "opt2"
ENTRY_T, EXIT_T = pd.Timestamp("09:45").time(), pd.Timestamp("15:00").time()
STEP = 50
ANN = np.sqrt(252 * 375)


def main():
    spot = load_spot()
    spot["tm"] = spot["ts"].dt.time
    sess = {}
    for d, g in spot.groupby("date"):
        w = g[(g.tm >= ENTRY_T) & (g.tm <= EXIT_T)].sort_values("ts")
        if len(w) < 200:
            continue
        c = w["c"].to_numpy(float)
        sess[d] = {"ts945": w["ts"].iloc[0], "S": float(c[0]),
                   "rv": float(np.std(np.diff(np.log(c)), ddof=1) * ANN * 100)}

    rows = []
    for f in sorted(OPT.glob("*.parquet")):
        e = f.stem
        d = pd.read_parquet(f)
        if d.empty:
            continue
        d["ts"] = pd.to_datetime(d["ts"])
        d = d[d["ts"].dt.time == ENTRY_T].copy()
        if d.empty:
            continue
        d["date"] = d["ts"].dt.date.astype(str)
        book = {(r.date, int(r.strike), str(r.cp)): (float(r.c), float(r.v))
                for r in d.itertuples()}
        for dt in sorted(set(d["date"])):
            if dt not in sess or dt >= pd.Timestamp(e).strftime("%Y-%m-%d"):
                continue
            S = sess[dt]["S"]
            T = years_to_expiry(sess[dt]["ts945"], e)
            atm = int(round(S / STEP) * STEP)
            ce, pe = book.get((dt, atm, "CE")), book.get((dt, atm, "PE"))
            if not ce or not pe or ce[0] <= 1 or pe[0] <= 1 or ce[1] <= 0 or pe[1] <= 0:
                continue
            a = implied_vol(ce[0], S, atm, T, "CE")
            b = implied_vol(pe[0], S, atm, T, "PE")
            if a is None or b is None:
                continue
            rows.append({"date": dt, "expiry": e, "dte": T * 365,
                         "iv": (a + b) / 2 * 100, "rv": sess[dt]["rv"], "S": S})
        print(f"  {e}", flush=True)

    p = pd.DataFrame(rows)
    p.to_csv(STORE / "term_structure_panel.csv", index=False)
    print(f"\n{len(p)} (session, expiry) points, {p.date.nunique()} sessions")
    print("\npoints per session:")
    print(p.groupby("date").size().value_counts().sort_index().to_string())
    print("\nATM IV by DTE bucket:")
    p["b"] = pd.cut(p.dte, [0, 2, 5, 10, 20, 40, 400],
                    labels=["<2", "2-5", "5-10", "10-20", "20-40", ">40"])
    print(p.groupby("b", observed=True).agg(n=("iv", "size"), iv=("iv", "mean"),
                                            dte=("dte", "mean")).round(2).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
