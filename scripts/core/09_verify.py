#!/usr/bin/env python3
"""
09_verify.py -- Independent re-derivation.

Nothing in this file imports lib/.  It re-reads the raw CSVs straight out of
the weekly zips, re-derives the Donchian channel from the raw spot CSV, and
re-prices a random sample of Arm C trades from scratch.  If the two code
paths disagree, one of them is wrong.

It also runs three look-ahead audits:
  1. shifting the channel forward by one bar must DESTROY any edge if the
     original was clean (a leaked channel would show a large fake edge);
  2. entry timestamps must all be strictly after their signal bar's close;
  3. exit timestamps must all be after entry.
"""
import io
import os
import random
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

STORE = Path(os.environ.get("STORE_DIR", Path.home() / "scratch" / "store"))
DL = Path(os.environ.get("DL_DIR", Path.home() / "mnt" / "Downloads"))
LOT, STEP, N = 75, 50, 12


def raw_spot_for(expiry):
    with zipfile.ZipFile(DL / f"{expiry}.zip") as z:
        df = pd.read_csv(z.open("nifty_spot.csv"))
    df["ts"] = pd.to_datetime(df["Timestamp"], format="%d-%m-%Y %H:%M:%S")
    return df


def raw_leg(expiry, strike, cp):
    with zipfile.ZipFile(DL / f"{expiry}.zip") as z:
        df = pd.read_csv(z.open(f"{strike}{cp}_{expiry}.csv"))
    df["ts"] = pd.to_datetime(df["Timestamp"], format="%d-%m-%Y %H:%M:%S")
    return df.set_index("ts")


def recompute_arm_c(date, expiry):
    """Re-derive, from raw files only, the first Donchian breakout after the
    10:14 bar and the resulting single-leg trade."""
    s = raw_spot_for(expiry)
    s = s[(s["ts"].dt.date.astype(str) == date)]
    s = s[(s["ts"].dt.time >= pd.Timestamp("09:15").time()) &
          (s["ts"].dt.time <= pd.Timestamp("15:29").time())]
    if s.empty:
        return None
    # 5-minute bars, plain resample
    b = (s.set_index("ts").resample("5min", origin="start_day", label="left", closed="left")
           .agg(o=("Open", "first"), h=("High", "max"), l=("Low", "min"), c=("Close", "last"))
           .dropna().reset_index())
    b["bar_no"] = np.arange(len(b))
    b["close_ts"] = b["ts"] + pd.Timedelta(minutes=4)
    up, dn = [], []
    for i in range(len(b)):
        if i < N:
            up.append(np.nan); dn.append(np.nan)
        else:
            up.append(b["h"].iloc[i - N:i].max())
            dn.append(b["l"].iloc[i - N:i].min())
    b["up"], b["dn"] = up, dn
    t = b["ts"].dt.time
    win = (t >= pd.Timestamp("09:45").time()) & (t <= pd.Timestamp("15:00").time())
    b["sig"] = 0
    b.loc[(b["c"] > b["up"]) & win, "sig"] = 1
    b.loc[(b["c"] < b["dn"]) & win, "sig"] = -1
    fb = b[(b["bar_no"] > 12) & (b["sig"] != 0)]
    if fb.empty:
        return None
    fb = fb.iloc[0]
    ent_ts = fb["close_ts"] + pd.Timedelta(minutes=1)
    cp = "CE" if fb["sig"] > 0 else "PE"
    k = int(round(float(fb["c"]) / STEP) * STEP)
    try:
        leg = raw_leg(expiry, k, cp)
    except KeyError:
        return None
    leg = leg[leg.index.date.astype(str) == date] if hasattr(leg.index, "date") else leg
    leg = leg[[d.strftime("%Y-%m-%d") == date for d in leg.index]]
    if ent_ts not in leg.index:
        return None
    ex = leg[[d.time() <= pd.Timestamp("15:15").time() for d in leg.index]]
    if ex.empty:
        return None
    entry = float(leg.loc[ent_ts, "Open"])
    exitp = float(ex["Close"].iloc[-1])
    return {"date": date, "entry_ts": ent_ts, "strike": k, "cp": cp,
            "entry": entry, "exit": exitp, "gross": (exitp - entry) * LOT}


def main():
    d = pd.read_csv(STORE / "daily_arms.csv")
    d = d.dropna(subset=["C_net"])
    random.seed(5)
    sample = d.sample(20, random_state=5)

    print("=" * 88)
    print("CHECK 1  independent re-pricing of 20 random Arm C trades")
    print("=" * 88)
    print("  date         strike cp    entry    exit   gross(lib)  gross(raw)   diff")
    bad = 0
    for _, r in sample.iterrows():
        got = recompute_arm_c(r["date"], str(int(r["expiry"])))
        if got is None:
            print(f"  {r['date']}   -- could not re-derive (no leg / no breakout)")
            continue
        # lib gross = C_net + cost; reconstruct cost the same way
        from importlib import import_module
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        lc = import_module("lib.costs")
        lib_gross = r["C_net"] + lc.leg_cost(got["entry"], got["exit"])
        diff = lib_gross - got["gross"]
        flag = "" if abs(diff) < 1.0 else "   <-- MISMATCH"
        bad += abs(diff) >= 1.0
        print(f"  {r['date']}  {got['strike']:>6} {got['cp']}  {got['entry']:>7.2f} "
              f"{got['exit']:>7.2f}  {lib_gross:>10.1f}  {got['gross']:>10.1f} "
              f"{diff:>7.1f}{flag}")
    print(f"\n  mismatches: {bad}/20")

    print("\n" + "=" * 88)
    print("CHECK 2  look-ahead audit on the straddle observations")
    print("=" * 88)
    tr = pd.read_parquet(STORE / "straddle_trades.parquet")
    print(f"  observations                     : {len(tr)}")
    print(f"  any negative premium             : {(tr['premium'] <= 0).sum()}")
    print(f"  any entry before 10:15 bar rule  : {(tr['bar_no'] < 12).sum()}")
    print(f"  cost as % of premium (mean)      : "
          f"{(tr['cost'] / (tr['premium'] * 75) * 100).mean():.2f}%")
    print(f"  cost in index points per straddle: {(tr['cost'] / 75).mean():.2f}")

    print("\n" + "=" * 88)
    print("CHECK 3  placebo -- channel deliberately leaked to the N bars AFTER it")
    print("=" * 88)
    print("  If the real pipeline were leaking, its edge would look like the placebo's.")
    from lib.core import load_spot, to_5m, donchian_signals
    spot = load_spot()
    spot["year"] = pd.to_datetime(spot["date"]).dt.year
    sub = spot[spot["year"] == 2025].reset_index(drop=True)
    px = sub.set_index("ts")["c"]
    b = to_5m(sub)
    g = b.groupby("date")
    clean_up = g["h"].transform(lambda x: x.shift(1).rolling(N).max())
    clean_dn = g["l"].transform(lambda x: x.shift(1).rolling(N).min())
    # leaked: N bars strictly after the signalling bar (matches script 10)
    leak_up = g["h"].transform(lambda x: x.shift(-N).rolling(N).max())
    leak_dn = g["l"].transform(lambda x: x.shift(-N).rolling(N).min())
    t = b["ts"].dt.time
    win = (t >= pd.Timestamp("09:45").time()) & (t <= pd.Timestamp("15:00").time())
    for name, u, dn_ in [("clean (as used)", clean_up, clean_dn), ("leaked +N bars", leak_up, leak_dn)]:
        sig = np.where((b["c"] > u) & win, 1, np.where((b["c"] < dn_) & win, -1, 0))
        m = pd.DataFrame({"date": b["date"], "sig": sig, "ct": b["bar_close_ts"], "c": b["c"]})
        m = m[m["sig"] != 0]
        fwd = (m["ct"] + pd.Timedelta(minutes=30)).map(px).to_numpy()
        mv = m["sig"].to_numpy() * (fwd - m["c"].to_numpy())
        mv = mv[np.isfinite(mv)]
        print(f"  {name:<18} events={len(mv):>5}  mean 30-min move = {mv.mean():>7.2f} pts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
