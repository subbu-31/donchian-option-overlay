#!/usr/bin/env python3
"""
23_regime_features.py -- Why is every out-of-sample test negative?

Nine structural variants have now come back positive in-sample and negative on
the Jan-Apr 2026 holdout.  The blind condor among them has NO fitted parameters
at all -- strike selection is mechanical -- so overfitting cannot explain its
sign flip.  That points at the regime, and this builds the evidence.

Per session, from 1-minute spot (566 sessions, Jan 2024 - Apr 2026) and the
option archive (Jan 2025 - Apr 2026):

  rv          annualised realised volatility of 1-minute log returns, 09:45-15:00
  rv_cc       annualised close-to-close volatility, 20-session trailing
  range_pct   session high-low as a share of the open
  gap_pct     absolute overnight gap
  trend_eff   |close - open| / total absolute 1-minute travel (Kaufman efficiency):
              1.0 is a clean trend, near 0 is chop
  acf1        lag-1 autocorrelation of 5-minute returns
  atm_iv      implied vol backed out of the 09:45 ATM straddle, nearest weekly
              expiry strictly after the session (the same expiry the condor trades)
  vrp         atm_iv - rv, both annualised: what a short-premium book is paid
              minus what it actually has to hedge.  This is the single number
              that decides whether selling the condor works.
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
ANN = np.sqrt(252 * 375)          # 1-minute bars per year


def main():
    spot = load_spot()
    spot["tm"] = spot["ts"].dt.time
    rows = []
    for d, g in spot.groupby("date", sort=True):
        g = g.sort_values("ts")
        win = g[(g.tm >= ENTRY_T) & (g.tm <= EXIT_T)]
        if len(win) < 200:
            continue
        c = win["c"].to_numpy(float)
        r = np.diff(np.log(c))
        m5 = c[::5]
        r5 = np.diff(np.log(m5))
        rows.append({
            "date": d,
            "open": float(g["o"].iloc[0]), "close": float(g["c"].iloc[-1]),
            "high": float(g["h"].max()), "low": float(g["l"].min()),
            "rv": float(np.std(r, ddof=1) * ANN),
            "range_pct": float((g["h"].max() - g["l"].min()) / g["o"].iloc[0] * 100),
            "trend_eff": float(abs(c[-1] - c[0]) / np.abs(np.diff(c)).sum())
                          if np.abs(np.diff(c)).sum() > 0 else np.nan,
            "acf1": float(pd.Series(r5).autocorr(1)) if len(r5) > 20 else np.nan,
            "spot_945": float(win["c"].iloc[0]),
        })
    f = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    f["ret"] = f["close"].pct_change() * 100
    f["gap_pct"] = (f["open"] / f["close"].shift(1) - 1).abs() * 100
    f["rv_cc"] = f["close"].pct_change().rolling(20).std() * np.sqrt(252) * 100
    f["rv"] *= 100

    # ---- ATM implied vol from the archive ----
    emap = pd.read_csv(STORE / "expiry_map.csv", index_col=0)["expiry"].astype(str)
    emap.index = emap.index.astype(str)
    by_exp = {}
    for d, e in emap.items():
        by_exp.setdefault(e, []).append(d)
    iv_rows = []
    for e in sorted(by_exp):
        p = OPT / f"{e}.parquet"
        if not p.exists():
            continue
        opt = pd.read_parquet(p)
        opt["ts"] = pd.to_datetime(opt["ts"])
        tm = opt["ts"].dt.time
        opt = opt[tm == ENTRY_T].copy()
        if opt.empty:
            continue
        opt["date"] = opt["ts"].dt.date.astype(str)
        book = {(r.date, int(r.strike), str(r.cp)): (float(r.c), float(r.v))
                for r in opt.itertuples()}
        for d in by_exp[e]:
            row = f[f.date == d]
            if row.empty:
                continue
            S = float(row["spot_945"].iloc[0])
            ts = pd.Timestamp(d) + pd.Timedelta(hours=9, minutes=45)
            T = years_to_expiry(ts, e)
            atm = int(round(S / STEP) * STEP)
            ce, pe = book.get((d, atm, "CE")), book.get((d, atm, "PE"))
            if not ce or not pe or ce[0] <= 1 or pe[0] <= 1:
                continue
            ivc = implied_vol(ce[0], S, atm, T, "CE")
            ivp = implied_vol(pe[0], S, atm, T, "PE")
            if ivc is None or ivp is None:
                continue
            iv_rows.append({"date": d, "expiry": e, "dte": T * 365,
                            "atm_iv": (ivc + ivp) / 2 * 100,
                            "straddle_pct": (ce[0] + pe[0]) / S * 100})
    iv = pd.DataFrame(iv_rows)
    f = f.merge(iv, on="date", how="left")
    f["vrp"] = f["atm_iv"] - f["rv"]
    f["year"] = pd.to_datetime(f["date"]).dt.year
    f["m"] = pd.to_datetime(f["date"]).dt.to_period("M").astype(str)
    f.to_csv(STORE / "regime_features.csv", index=False)

    print(f"{len(f)} sessions, {f.date.min()} .. {f.date.max()};  "
          f"{f.atm_iv.notna().sum()} with option data\n")
    print("=" * 104)
    print("ANNUAL PROFILE")
    print("=" * 104)
    print(f.groupby("year")[["rv", "rv_cc", "range_pct", "gap_pct", "trend_eff",
                             "acf1", "atm_iv", "vrp", "straddle_pct"]]
          .mean().round(3).to_string())
    print("\n" + "=" * 104)
    print("MONTHLY  (rv = realised vol, atm_iv = implied, vrp = implied - realised, all annualised %)")
    print("=" * 104)
    mm = f.groupby("m").agg(n=("rv", "size"), rv=("rv", "mean"), atm_iv=("atm_iv", "mean"),
                            vrp=("vrp", "mean"), range_pct=("range_pct", "mean"),
                            gap=("gap_pct", "mean"), trend_eff=("trend_eff", "mean"))
    print(mm.round(2).to_string())
    print("\n  vrp > 0 means options were dearer than the move that followed -- the state a")
    print("  short-premium book needs. vrp < 0 means it was paid less than the risk it carried.")
    print("\n  share of sessions with vrp > 0:")
    for y in (2025, 2026):
        g = f[(f.year == y) & f.vrp.notna()]
        if len(g):
            print(f"    {y}: {100*(g.vrp>0).mean():.1f}%  (n={len(g)}, mean vrp {g.vrp.mean():+.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
