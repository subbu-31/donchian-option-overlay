#!/usr/bin/env python3
"""
01_build_store.py -- Build the unified 1-minute data store.

Inputs : weekly expiry archives  <DOWNLOADS>/YYYYMMDD.zip
         each containing  nifty_spot.csv  and  {strike}{CE|PE}_{expiry}.csv
Outputs: <STORE>/spot_1min.parquet      deduped 1-min NIFTY 50 spot bars
         <STORE>/option_index.json      {expiry: {"zip": path, "legs": [[strike, cp], ...]}}
         <STORE>/build_report.json      coverage / sanity numbers
"""
import json
import os
import re
import sys
import zipfile
from pathlib import Path

import pandas as pd

DOWNLOADS = Path(os.environ.get("DL_DIR", Path.home() / "mnt" / "Downloads"))
STORE = Path(os.environ.get("STORE_DIR", Path.home() / "scratch" / "store"))
STORE.mkdir(parents=True, exist_ok=True)

ZIP_RE = re.compile(r"^(\d{8})\.zip$")
LEG_RE = re.compile(r"^(\d+)(CE|PE)_(\d{8})\.csv$")


def read_spot(z):
    # some archives nest the CSV under a subdirectory (e.g. "20260505/nifty_spot.csv")
    # instead of storing it at the zip root -- match by basename, not full path.
    names = [n for n in z.namelist() if Path(n).name == "nifty_spot.csv"]
    if not names:
        return None
    df = pd.read_csv(z.open(names[0]))
    keep = ["Timestamp", "Open", "High", "Low", "Close"]
    if not set(keep).issubset(df.columns):
        return None
    df = df[keep].copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%d-%m-%Y %H:%M:%S", errors="coerce")
    return df.dropna(subset=["Timestamp"])


def main():
    zips = sorted(p for p in DOWNLOADS.iterdir() if ZIP_RE.match(p.name))
    if not zips:
        print("no expiry zips under %s" % DOWNLOADS, file=sys.stderr)
        return 1
    print("%d expiry archives found" % len(zips), flush=True)

    spot_parts = []
    index = {}

    for i, zp in enumerate(zips, 1):
        expiry = ZIP_RE.match(zp.name).group(1)
        try:
            with zipfile.ZipFile(zp) as z:
                s = read_spot(z)
                if s is not None and len(s):
                    spot_parts.append(s)
                legs = []
                for n in z.namelist():
                    m = LEG_RE.match(Path(n).name)
                    if m and m.group(3) == expiry:
                        legs.append([int(m.group(1)), m.group(2)])
                legs.sort()
                index[expiry] = {"zip": str(zp), "legs": legs}
        except Exception as e:
            print("  !! %s: %s" % (zp.name, e), flush=True)
            index[expiry] = {"zip": str(zp), "legs": [], "error": str(e)}
        if i % 10 == 0 or i == len(zips):
            print("  %d/%d" % (i, len(zips)), flush=True)

    spot = (pd.concat(spot_parts, ignore_index=True)
              .drop_duplicates(subset="Timestamp", keep="first")
              .sort_values("Timestamp").reset_index(drop=True))
    spot["date"] = spot["Timestamp"].dt.date.astype("string")
    spot = spot.rename(columns={"Timestamp": "ts", "Open": "o", "High": "h", "Low": "l", "Close": "c"})
    spot = spot[["ts", "date", "o", "h", "l", "c"]]

    tod = spot["ts"].dt.time
    session = (tod >= pd.Timestamp("09:15").time()) & (tod <= pd.Timestamp("15:29").time())
    dropped = int((~session).sum())
    spot = spot[session].reset_index(drop=True)

    spot.to_parquet(STORE / "spot_1min.parquet", index=False)
    (STORE / "option_index.json").write_text(json.dumps(index))

    per_day = spot.groupby("date").size()
    report = {
        "archives": len(zips),
        "spot_rows": int(len(spot)),
        "spot_first": str(spot["ts"].iloc[0]),
        "spot_last": str(spot["ts"].iloc[-1]),
        "sessions": int(per_day.size),
        "off_session_rows_dropped": dropped,
        "median_minutes_per_session": float(per_day.median()),
        "short_sessions_lt_300min": int((per_day < 300).sum()),
        "expiries_with_legs": int(sum(1 for v in index.values() if v["legs"])),
        "total_legs": int(sum(len(v["legs"]) for v in index.values())),
    }
    (STORE / "build_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
