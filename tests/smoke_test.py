#!/usr/bin/env python3
"""
smoke_test.py -- Pipeline smoke test.

Real Zerodha archives aren't available in CI, so this fabricates weekly
expiry zips with the same schema (nifty_spot.csv + {strike}{CE|PE}_{expiry}.csv)
and runs scripts 01-09 against them end to end. It does not assert on any
trading result -- the synthetic prices are random walks with no signal in
them -- it only asserts that the documented pipeline runs to completion
without raising. That's enough to catch dependency-version breaks (e.g. a
pandas resample API change) and path mismatches between scripts (e.g. one
script writing to STORE/opt while a later one reads STORE/opt2) before they
reach a real run.
"""
import shutil
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PIPELINE = [
    "01_build_store.py",
    "02_spot_grid.py",
    "03_event_study.py",
    "04_conditioning.py",
    "05_build_option_store.py",
    "06_straddle_by_width.py",
    "07_analyse_straddle.py",
    "08_daily_arms.py",
]


def gen_minute_series(rng, start_ts, n_minutes, start_price):
    ts = [start_ts + timedelta(minutes=i) for i in range(n_minutes)]
    rets = rng.normal(0, 0.0006, n_minutes)
    price = start_price * np.cumprod(1 + rets)
    return pd.DataFrame({
        "Timestamp": [t.strftime("%d-%m-%Y %H:%M:%S") for t in ts],
        "Open": price,
        "High": price * (1 + np.abs(rng.normal(0, 0.0003, n_minutes))),
        "Low": price * (1 - np.abs(rng.normal(0, 0.0003, n_minutes))),
        "Close": price * (1 + rng.normal(0, 0.0002, n_minutes)),
        "Volume": rng.integers(1, 500, n_minutes),
        "OI": rng.integers(100, 10000, n_minutes),
    }), price[-1]


def build_fake_archives(dl_dir, n_weeks=6, strikes_each_side=6, step=50):
    rng = np.random.default_rng(0)
    base = datetime(2025, 1, 2)
    spot = 21500.0
    for week in range(n_weeks):
        expiry = base + timedelta(weeks=week)
        expiry_str = expiry.strftime("%Y%m%d")
        with zipfile.ZipFile(dl_dir / f"{expiry_str}.zip", "w", zipfile.ZIP_DEFLATED) as z:
            days = [expiry - timedelta(weeks=1) + timedelta(days=d) for d in range(5)]
            days = [d for d in days if d.weekday() < 5]
            spot_parts = []
            for d in days:
                start = d.replace(hour=9, minute=15, second=0, microsecond=0)
                df, spot = gen_minute_series(rng, start, 375, spot)
                spot_parts.append(df)
            z.writestr("nifty_spot.csv", pd.concat(spot_parts, ignore_index=True).to_csv(index=False))

            atm = round(spot / step) * step
            for off in range(-strikes_each_side, strikes_each_side + 1):
                strike = int(atm + off * step)
                for cp, base_px in [("CE", max(atm - strike, 0) + 100), ("PE", max(strike - atm, 0) + 100)]:
                    parts = []
                    for d in days:
                        start = d.replace(hour=9, minute=15, second=0, microsecond=0)
                        odf, _ = gen_minute_series(rng, start, 375, max(base_px, 5))
                        parts.append(odf)
                    leg = pd.concat(parts, ignore_index=True)
                    z.writestr(f"{strike}{cp}_{expiry_str}.csv", leg.to_csv(index=False))


def main():
    tmp = Path(tempfile.mkdtemp(prefix="donchian_smoke_"))
    dl_dir, store_dir = tmp / "downloads", tmp / "store"
    dl_dir.mkdir(parents=True)
    store_dir.mkdir(parents=True)
    try:
        build_fake_archives(dl_dir)
        env = {"DL_DIR": str(dl_dir), "STORE_DIR": str(store_dir), "PATH": "/usr/bin:/bin"}
        import os
        env["PATH"] = os.environ.get("PATH", env["PATH"])
        for script in PIPELINE:
            print(f"--- {script} ---", flush=True)
            r = subprocess.run([sys.executable, str(ROOT / "scripts" / "core" / script)],
                                cwd=ROOT, env=env, capture_output=True, text=True)
            if r.returncode != 0:
                print(r.stdout)
                print(r.stderr, file=sys.stderr)
                print(f"FAILED: {script} exited {r.returncode}", file=sys.stderr)
                return 1
        print("\nsmoke test passed: scripts 01-08 ran end to end without error")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
