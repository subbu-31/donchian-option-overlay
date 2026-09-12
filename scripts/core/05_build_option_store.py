#!/usr/bin/env python3
"""
05_build_option_store.py -- Extract the near-the-money legs we will actually
need out of the 73 weekly LZMA archives into one parquet per expiry.

Expiry mapping honours the standing no-0-DTE rule: a session trades the
nearest weekly expiry STRICTLY after that session's date.
"""
import json
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.core import STORE, load_spot, load_option_index

OPT = STORE / "opt2"
OPT.mkdir(parents=True, exist_ok=True)
BAND = 0.040        # wide enough for 0.15-delta wings, not just the ATM straddle


def expiry_map(session_dates, expiries):
    """session date -> nearest expiry strictly after it (no 0-DTE)."""
    exp = sorted(pd.Timestamp(e) for e in expiries)
    out = {}
    for d in session_dates:
        dt = pd.Timestamp(d)
        nxt = [e for e in exp if e > dt]
        if nxt:
            out[d] = nxt[0].strftime("%Y%m%d")
    return out


def main():
    spot = load_spot()
    idx = load_option_index()
    sess = sorted(spot["date"].unique())
    emap = expiry_map(sess, idx.keys())
    pd.Series(emap).rename("expiry").to_frame().to_csv(STORE / "expiry_map.csv")

    rng = spot.groupby("date")["c"].agg(["min", "max"])
    need = {}
    for d, e in emap.items():
        if d not in rng.index:
            continue
        lo, hi = rng.loc[d, "min"], rng.loc[d, "max"]
        need.setdefault(e, [1e9, -1e9])
        need[e][0] = min(need[e][0], lo * (1 - BAND))
        need[e][1] = max(need[e][1], hi * (1 + BAND))

    todo = [e for e in sorted(need) if not (OPT / f"{e}.parquet").exists()]
    print(f"{len(need)} expiries needed, {len(todo)} still to extract", flush=True)

    for i, e in enumerate(todo, 1):
        lo, hi = need[e]
        info = idx.get(e)
        if not info or not info["legs"]:
            continue
        legs = [(s, cp) for s, cp in info["legs"] if lo <= s <= hi]
        parts = []
        with zipfile.ZipFile(info["zip"]) as z:
            # some archives nest CSVs under a subdirectory instead of the zip
            # root (e.g. "20260505/21600PE_20260505.csv") -- match by basename.
            by_basename = {Path(n).name: n for n in z.namelist()}
            for s, cp in legs:
                name = f"{s}{cp}_{e}.csv"
                real_name = by_basename.get(name)
                if real_name is None:
                    continue
                try:
                    df = pd.read_csv(z.open(real_name),
                                     usecols=["Timestamp", "Open", "High", "Low", "Close", "Volume"])
                except Exception:
                    continue
                df["ts"] = pd.to_datetime(df["Timestamp"], format="%d-%m-%Y %H:%M:%S", errors="coerce")
                df = df.dropna(subset=["ts"])
                if df.empty:
                    continue
                out = pd.DataFrame({
                    "ts": df["ts"].to_numpy(),
                    "strike": np.int32(s),
                    "cp": cp,
                    "o": df["Open"].to_numpy(dtype="float32"),
                    "h": df["High"].to_numpy(dtype="float32"),
                    "l": df["Low"].to_numpy(dtype="float32"),
                    "c": df["Close"].to_numpy(dtype="float32"),
                    "v": df["Volume"].to_numpy(dtype="float32"),
                })
                parts.append(out)
        if parts:
            allp = pd.concat(parts, ignore_index=True)
            allp["cp"] = allp["cp"].astype("category")
            allp.to_parquet(OPT / f"{e}.parquet", index=False)
        print(f"  [{i}/{len(todo)}] {e}: {len(legs)} legs, {sum(len(p) for p in parts)} rows",
              flush=True)
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
