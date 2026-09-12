#!/usr/bin/env python3
"""
27_forecast_panel.py -- Build the forecast panel.

The P&L test asked "does the width state make money" on 322 daily observations.
This asks the underlying question directly -- "does the width state predict the
size of the next move, over and above what implied volatility already prices"
-- on every 5-minute decision bar instead, which is ~50,000 observations.

For each bar:
  S     spot at the bar close
  iv    implied vol backed out of the ATM straddle at the next minute, nearest
        weekly expiry strictly after the session (bisection, CE and PE averaged)
  IM    implied move over the next h minutes = S * iv * sqrt(h / (252*375))
  RM    realised |move| over the same h minutes
  width_rel  the 12-bar Donchian channel width relative to its own 20-session
        average for that bar-of-day -- causal, known at the bar close

The test is whether width_rel explains RM once IM is accounted for.  If implied
vol already impounds it, it cannot.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.core import STORE, load_spot, to_5m, donchian_signals
from lib.features import add_features
from lib.bs import implied_vol, years_to_expiry

OPT = STORE / "opt2"
N, STEP = 12, 50
HORIZONS = [15, 30, 60]
TRADING_MIN_YEAR = 252 * 375


def main():
    spot = load_spot()
    emap = pd.read_csv(STORE / "expiry_map.csv", index_col=0)["expiry"].astype(str)
    emap.index = emap.index.astype(str)
    b5 = add_features(donchian_signals(to_5m(spot), N), N)
    b5 = b5[b5["bar_no"] >= N]

    by_exp = {}
    for d, e in emap.items():
        by_exp.setdefault(e, []).append(d)

    # per-expiry checkpoints so the build survives interruption and resumes
    CK = STORE / "fp_parts"
    CK.mkdir(exist_ok=True)
    for e in sorted(by_exp):
        f = OPT / f"{e}.parquet"
        part = CK / f"{e}.parquet"
        if not f.exists() or part.exists():
            continue
        rows = []
        opt = pd.read_parquet(f)
        opt["ts"] = pd.to_datetime(opt["ts"])
        opt["date"] = opt["ts"].dt.date.astype(str)
        for d in by_exp[e]:
            day = opt[opt["date"] == d]
            ds_ = spot[spot["date"] == d]
            db = b5[b5["date"] == d]
            if day.empty or ds_.empty or db.empty:
                continue
            grid = pd.DatetimeIndex(ds_["ts"])
            pos = {t: i for i, t in enumerate(grid)}
            close = ds_["c"].to_numpy(float)
            book = {(int(k), str(cp)): g.drop_duplicates("ts").set_index("ts")
                    .reindex(grid)[["c", "v"]].to_numpy(float)
                    for (k, cp), g in day.groupby(["strike", "cp"], observed=True)}
            for _, b in db.iterrows():
                if pd.isna(b["width_rel"]):
                    continue
                ent = b["bar_close_ts"] + pd.Timedelta(minutes=1)
                if ent not in pos:
                    continue
                j = pos[ent]
                S = float(b["c"])
                atm = int(round(S / STEP) * STEP)
                ce, pe = book.get((atm, "CE")), book.get((atm, "PE"))
                if ce is None or pe is None:
                    continue
                pc, pp = ce[j, 0], pe[j, 0]
                if not (np.isfinite(pc) and np.isfinite(pp)) or pc <= 1 or pp <= 1:
                    continue
                if not (ce[j, 1] > 0 and pe[j, 1] > 0):
                    continue
                T = years_to_expiry(grid[j], e)
                ivc = implied_vol(float(pc), S, atm, T, "CE")
                ivp = implied_vol(float(pp), S, atm, T, "PE")
                if ivc is None or ivp is None:
                    continue
                iv = (ivc + ivp) / 2.0
                for h in HORIZONS:
                    k = j + h
                    if k >= len(grid) or grid[k].time() > pd.Timestamp("15:25").time():
                        continue
                    im = S * iv * np.sqrt(h / TRADING_MIN_YEAR)
                    if not np.isfinite(im) or im <= 0:
                        continue
                    rows.append({
                        "date": d, "expiry": e, "h": h, "bar_no": int(b["bar_no"]),
                        "S": S, "iv": iv, "dte": T * 365,
                        "im": float(im), "rm": float(abs(close[k] - S)),
                        "width_rel": float(b["width_rel"]),
                        "straddle": float(pc + pp)})
        pd.DataFrame(rows).to_parquet(part, index=False)
        print(f"  {e}  rows={len(rows)}", flush=True)

    parts = [pd.read_parquet(x) for x in sorted(CK.glob("*.parquet"))]
    parts = [x for x in parts if len(x)]
    if not parts:
        print("no parts yet")
        return 1
    t = pd.concat(parts, ignore_index=True)
    t["year"] = pd.to_datetime(t["date"]).dt.year
    t["ratio"] = t["rm"] / t["im"]
    t.to_parquet(STORE / "forecast_panel.parquet", index=False)
    print(f"\n{len(t)} observations, {t.date.nunique()} sessions, "
          f"{t.date.min()} .. {t.date.max()}")
    print(t.groupby("h")[["im", "rm", "ratio", "iv", "width_rel"]].mean().round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
