#!/usr/bin/env python3
"""
26_banknifty_replication.py -- Cross-instrument, cross-era replication.

The NIFTY study cannot settle anything on 322 sessions: the per-session edge is
tiny relative to its dispersion, and 39% of 77-session windows inside the GOOD
year are negative.  The fix is more independent data, not more variants.

BANKNIFTY monthly-expiry archive: 57 expiries sampled across 2018-2026, each
folder holding the contract's final ~4 weeks of 1-minute option and spot bars.
Taking only the sessions whose DTE falls in the same 1-8 day band the NIFTY
study used gives a comparable per-session sample spread over eight years and
several regimes, COVID included.

Identical rules to the NIFTY intraday test: enter 09:45, exit 15:00, strikes
delta-selected by BS implied-vol bisection off the entry tick, four legs marked
to their own traded 1-minute path, each leg charged its own estimated
half-spread plus the Zerodha statutory stack.  Nothing is refitted.

BANKNIFTY lot size has changed repeatedly over this span, so points per unit is
the primary figure and rupees are quoted at a stated lot of 30.
"""
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.bs import implied_vol, delta, years_to_expiry
from lib.spread import half_spread_rupees
import lib.ic_costs as IC

STORE = Path.home() / "scratch" / "store"
ZIP = Path.home() / "mnt" / "Downloads" / "banknifty_data-[cloudtraderpro.in]-20260825T105024Z-1-001.zip"
BASE = "banknifty_data-[cloudtraderpro.in]"
QTY = 10 * 30                      # 10 lots at a stated lot size of 30
ENTRY_T, EXIT_T = pd.Timestamp("09:45").time(), pd.Timestamp("15:00").time()
DTE_LO, DTE_HI = 1.0, 8.5
BAND = 0.06
CFGS = [(0.35, 0.15), (0.25, 0.10)]
MIN_PREM = 0.5
LEG_RE = re.compile(r"^(\d+)(CE|PE)_(\d{8})\.csv$")


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


def main():
    z = zipfile.ZipFile(ZIP)
    names = z.namelist()
    folders = sorted({n.split("/")[1] for n in names
                      if n.startswith(BASE + "/") and len(n.split("/")) > 2 and n.split("/")[1]})
    # per-expiry checkpoints: the device link can drop mid-run, so each expiry
    # is written as it completes and finished ones are skipped on restart
    CK = STORE / "bn_parts"
    CK.mkdir(parents=True, exist_ok=True)
    for e in folders:
        sp_name = f"{BASE}/{e}/banknifty_spot.csv"
        part = CK / f"{e}.parquet"
        if sp_name not in names or part.exists():
            continue
        rows = []
        try:
            sp = pd.read_csv(z.open(sp_name))
        except Exception:
            continue
        sp["ts"] = pd.to_datetime(sp["timestamp"], errors="coerce")
        sp = sp.dropna(subset=["ts"]).sort_values("ts")
        sp["date"] = sp["ts"].dt.date.astype(str)
        sp["tm"] = sp["ts"].dt.time

        # which sessions land in the NIFTY study's DTE band
        ent = sp[sp.tm == ENTRY_T]
        keep = []
        for _, r in ent.iterrows():
            d = years_to_expiry(r["ts"], e) * 365
            if DTE_LO <= d <= DTE_HI:
                keep.append((r["date"], d, float(r["close"])))
        if not keep:
            continue
        lo = min(k[2] for k in keep) * (1 - BAND)
        hi = max(k[2] for k in keep) * (1 + BAND)

        legs = {}
        for n in names:
            if not n.startswith(f"{BASE}/{e}/"):
                continue
            m = LEG_RE.match(n.split("/")[-1])
            if not m:
                continue
            k = int(m.group(1))
            if not (lo <= k <= hi):
                continue
            try:
                df = pd.read_csv(z.open(n), usecols=["timestamp", "open", "high", "low",
                                                     "close", "volume"])
            except Exception:
                continue
            df["ts"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df.dropna(subset=["ts"])
            df["date"] = df["ts"].dt.date.astype(str)
            legs[(k, m.group(2))] = df

        if len(legs) < 8:
            pd.DataFrame().to_parquet(part, index=False)
            print(f"  {e}: only {len(legs)} legs in band, skipped", flush=True)
            continue

        for d0, dte, S0 in keep:
            ds_ = sp[sp.date == d0]
            win = ds_[(ds_.tm >= ENTRY_T) & (ds_.tm <= EXIT_T)]
            if len(win) < 120:
                continue
            grid = pd.DatetimeIndex(win["ts"])
            book = {}
            for (k, cp), df in legs.items():
                g = df[df.date == d0]
                if g.empty:
                    continue
                g = g.drop_duplicates("ts").set_index("ts").reindex(grid)
                book[(k, cp)] = (g["open"].ffill(limit=3).to_numpy(float),
                                 g["high"].ffill(limit=3).to_numpy(float),
                                 g["low"].ffill(limit=3).to_numpy(float),
                                 g["close"].ffill(limit=3).to_numpy(float),
                                 g["volume"].to_numpy(float))
            if len(book) < 8:
                continue
            T0 = years_to_expiry(grid[0], e)
            ch = []
            for (k, cp), (o, h, l, c, v) in book.items():
                p = c[0]
                if not np.isfinite(p) or p < MIN_PREM or not (v[0] > 0):
                    continue
                iv = implied_vol(float(p), S0, float(k), T0, cp)
                if iv is None:
                    continue
                ch.append((k, cp, float(p), delta(S0, float(k), T0, iv, cp)))
            if len(ch) < 8:
                continue
            chain = pd.DataFrame(ch, columns=["k", "cp", "px", "d"])

            for dsh, dwg in CFGS:
                sel = pick(chain, dsh, dwg)
                if sel is None:
                    continue
                arr = {n_: book[(int(sel[n_].k), "CE" if n_ in ("sc", "lc") else "PE")]
                       for n_ in ("sc", "sp", "lc", "lp")}
                if any(not np.isfinite(arr[n_][3][-1]) for n_ in arr):
                    continue
                credit = float(sel["sc"].px + sel["sp"].px - sel["lc"].px - sel["lp"].px)
                cc = arr["sc"][3] + arr["sp"][3] - arr["lc"][3] - arr["lp"][3]
                path = credit - cc
                good = np.isfinite(path)
                if good.sum() < 100:
                    continue
                slip = 0.0
                for n_ in arr:
                    o, h, l, c, v = arr[n_]
                    hs = half_spread_rupees(h, l, c, sel[n_].px)
                    slip += (hs if np.isfinite(hs) else 0.15) * 2 * QTY
                ex = {n_: float(arr[n_][3][-1]) for n_ in arr}
                buy_turn = (sel["lc"].px + sel["lp"].px + ex["sc"] + ex["sp"]) * QTY
                sell_turn = (sel["sc"].px + sel["sp"].px + ex["lc"] + ex["lp"]) * QTY
                # statutory rates changed on 2024-10-01 and 2026-04-01; this
                # sample spans all three regimes, so the trade date is required
                stat = IC._statutory(buy_turn, sell_turn, 8, on=d0)
                gross = float(path[good][-1]) * QTY
                rows.append({
                    "expiry": e, "date": d0, "cfg": f"{dsh}/{dwg}", "dte": round(dte, 2),
                    "spot": S0, "credit": credit,
                    "sc_d": round(float(sel["sc"].d), 3), "sp_d": round(float(sel["sp"].d), 3),
                    "lc_d": round(float(sel["lc"].d), 3), "lp_d": round(float(sel["lp"].d), 3),
                    "call_width": int(sel["lc"].k - sel["sc"].k),
                    "put_width": int(sel["sp"].k - sel["lp"].k),
                    "pnl_pts": float(path[good][-1]),
                    "max_dd_pts": float(np.nanmin(path[good])),
                    "gross": gross, "slip": slip, "stat": stat,
                    "buy_turn": buy_turn, "sell_turn": sell_turn,
                    "net": gross - slip - stat})
        pd.DataFrame(rows).to_parquet(part, index=False)
        print(f"  {e}: {len(keep)} candidate sessions, {len(rows)} rows", flush=True)

    parts = [pd.read_parquet(x) for x in sorted(CK.glob("*.parquet"))]
    parts = [x for x in parts if len(x)]
    if not parts:
        print("no completed expiries yet")
        return 1
    t = pd.concat(parts, ignore_index=True)
    t["maxloss"] = (np.maximum(t.call_width, t.put_width) - t.credit).clip(lower=0) * QTY
    t["year"] = pd.to_datetime(t["date"]).dt.year
    t.to_csv(STORE / "banknifty_condor.csv", index=False)
    print(f"\n{len(t)} rows, {t.date.nunique()} sessions, {t.expiry.nunique()} expiries, "
          f"{t.date.min()} .. {t.date.max()}")
    print(t.groupby("cfg")[["sc_d", "sp_d", "lc_d", "lp_d", "dte", "credit"]].mean().round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
