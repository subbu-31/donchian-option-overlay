#!/usr/bin/env python3
"""10_placebo_and_export.py -- genuine forward-leak placebo, then export a
compact JSON of every headline number for the write-up."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.core import STORE, load_spot, to_5m, donchian_signals

N = 12

def main():
    spot = load_spot()
    spot["year"] = pd.to_datetime(spot["date"]).dt.year
    sub = spot[spot["year"] == 2025].reset_index(drop=True)
    px = sub.set_index("ts")["c"]
    b = to_5m(sub)
    g = b.groupby("date")
    t = b["ts"].dt.time
    win = (t >= pd.Timestamp("09:45").time()) & (t <= pd.Timestamp("15:00").time())

    # clean: channel from the N bars BEFORE the signalling bar
    up_c = g["h"].transform(lambda x: x.shift(1).rolling(N).max())
    dn_c = g["l"].transform(lambda x: x.shift(1).rolling(N).min())
    # leaked: channel from the N bars AFTER it -- pure future information
    up_l = g["h"].transform(lambda x: x.shift(-N).rolling(N).max())
    dn_l = g["l"].transform(lambda x: x.shift(-N).rolling(N).min())

    placebo = {}
    for name, u, d_ in [("clean", up_c, dn_c), ("forward_leak", up_l, dn_l)]:
        sig = np.where((b["c"] > u) & win, 1, np.where((b["c"] < d_) & win, -1, 0))
        m = pd.DataFrame({"sig": sig, "ct": b["bar_close_ts"], "c": b["c"]})
        m = m[m["sig"] != 0]
        fwd = (m["ct"] + pd.Timedelta(minutes=30)).map(px).to_numpy()
        mv = m["sig"].to_numpy() * (fwd - m["c"].to_numpy())
        mv = mv[np.isfinite(mv)]
        placebo[name] = {"events": int(len(mv)), "mean_30min_pts": round(float(mv.mean()), 2)}
        print(name, placebo[name], flush=True)

    ev = pd.read_csv(STORE / "event_study.csv")
    q2 = pd.read_csv(STORE / "q2_width_vol.csv")
    q1 = pd.read_csv(STORE / "q1_conditioning.csv")
    st = pd.read_parquet(STORE / "straddle_trades.parquet")
    da = pd.read_csv(STORE / "daily_arms.csv")
    da["year"] = pd.to_datetime(da["date"]).dt.year
    cuts = np.quantile(da[da.year == 2025]["width_rel"].dropna(), [.2, .4, .6, .8])
    st["year"] = pd.to_datetime(st["date"]).dt.year
    st["wq"] = np.digitize(st["width_rel"], cuts)

    out = {
        "coverage": {
            "spot_sessions": int(spot["date"].nunique()),
            "spot_from": str(spot["date"].min()), "spot_to": str(spot["date"].max()),
            "option_sessions": int(da["date"].nunique()),
            "expiries": 73, "option_legs_extracted": int(len(st)),
        },
        "placebo": placebo,
        "event_study": ev.to_dict("records"),
        "q1_summary": {
            "cells_tested_2024": int((q1["period"] == "2024 design").sum()),
            "significant_2024": int(((q1["period"] == "2024 design") & q1["sig"]).sum()),
            "expected_by_chance": round(0.05 * int((q1["period"] == "2024 design").sum()), 1),
        },
        "width_vol": q2.to_dict("records"),
        "straddle_by_width": [],
        "daily_arms": [],
        "cost": {
            "pts_per_straddle_roundtrip": round(float((st["cost"] / 75).mean()), 2),
            "pct_of_premium": round(float((st["cost"] / (st["premium"] * 75) * 100).mean()), 2),
        },
        "width_vs_range_corr": {
            str(y): round(float(da[da.year == y]["width_rel"].corr(da[da.year == y]["day_range"])), 3)
            for y in (2025, 2026)},
    }
    for y in (2025, 2026):
        for h in (15, 30, 60):
            d = st[(st.year == y) & (st.h == h)]
            for q in range(5):
                gq = d[d.wq == q]
                if len(gq) < 20:
                    continue
                out["straddle_by_width"].append({
                    "year": int(y), "h": int(h), "wq": int(q), "n": int(len(gq)),
                    "premium": round(float(gq["premium"].mean()), 1),
                    "spot_move": round(float(gq["spot_move"].mean()), 1),
                    "gross": round(float(gq["gross"].mean()), 1),
                    "net": round(float(gq["net"].mean()), 1)})
    for arm in "ABC":
        for y in (2025, 2026):
            s = da[da.year == y][f"{arm}_net"].dropna()
            if s.empty:
                continue
            se = float(s.std() / np.sqrt(len(s)))
            out["daily_arms"].append({
                "arm": arm, "year": int(y), "n": int(len(s)),
                "mean": round(float(s.mean()), 1), "se": round(se, 1),
                "t": round(float(s.mean() / se), 2),
                "hit": round(float((s > 0).mean()), 3),
                "total": round(float(s.sum()), 0)})

    (STORE / "findings.json").write_text(json.dumps(out, indent=1))
    print("\nwrote findings.json", len((STORE / 'findings.json').read_text()), "bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
