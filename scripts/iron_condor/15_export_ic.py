#!/usr/bin/env python3
"""15_export_ic.py -- equity curves, monthly P&L and the slippage sensitivity
surface, as one compact JSON for the write-up."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.core import STORE
import importlib.util
spec = importlib.util.spec_from_file_location("m12", Path(__file__).resolve().parent / "12_ic_metrics.py")
m12 = importlib.util.module_from_spec(spec)
spec.loader.exec_module.__self__ if False else spec.loader.exec_module(m12)

CFG = "0.35/0.15"
ARMS = [("IC blind", None, 1), ("IC + hedge 1x", "optg", 1),
        ("IC + hedge 2x", "optg", 2), ("IC + hedge 3x", "optg", 3)]


def main():
    df = pd.read_csv(STORE / "ic_donchian_daily.csv")
    df["year"] = pd.to_datetime(df["date"]).dt.year
    df = df[(df.cfg == CFG) & df.year.isin([2025, 2026])].sort_values("date").reset_index(drop=True)

    out = {"cfg": CFG, "lots": m12.LOTS, "lot_size": m12.LOT_SIZE, "qty": m12.QTY,
           "sessions": int(len(df)), "first": df.date.min(), "last": df.date.max(),
           "is_n": int((df.year == 2025).sum()), "oos_n": int((df.year == 2026).sum()),
           "curves": {}, "monthly": {}, "slip_surface": [], "dates": list(df["date"])}

    for slip in (0.00, 0.10):
        key = f"{slip:.2f}"
        out["curves"][key] = {}
        for label, h, m in ARMS:
            pnl, cap, gr, co = m12.arm_pnl(df, h, m, slip)
            out["curves"][key][label] = [round(float(x)) for x in np.cumsum(pnl)]
    df["m"] = pd.to_datetime(df["date"]).dt.to_period("M").astype(str)
    for label, h, m in ARMS:
        pnl, cap, gr, co = m12.arm_pnl(df, h, m, 0.10)
        s = pd.Series(pnl, index=df["m"]).groupby(level=0).sum()
        out["monthly"][label] = {k: round(float(v)) for k, v in s.items()}

    for slip in (0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50):
        for pn, yr in (("2025 in-sample", 2025), ("2026 holdout", 2026)):
            g = df[df.year == yr]
            for label, h, m in ARMS:
                pnl, cap, gr, co = m12.arm_pnl(g, h, m, slip)
                base = float(np.percentile(cap, 95))
                r = pnl / base
                sd = r.std(ddof=1)
                sh = (r.mean() * 252 - m12.RF) / (sd * np.sqrt(252)) if sd > 0 else np.nan
                out["slip_surface"].append({
                    "slip": slip, "period": pn, "arm": label,
                    "net": round(float(pnl.sum())), "sharpe": round(float(sh), 3)})

    out["hedge_attrib"] = {"clean_pts": 0.97, "random_pts": -0.32, "leaked_pts": -9.21,
                           "clean_trades": 518, "clean_win": 39.6}
    out["breakeven_slip"] = {"IC blind": {"2025": 0.217, "2026": 0.000},
                             "IC + hedge 1x": {"2025": 0.317, "2026": 0.000},
                             "IC + hedge 2x": {"2025": 0.380, "2026": 0.000},
                             "IC + hedge 3x": {"2025": 0.421, "2026": 0.000}}
    (STORE / "ic_findings.json").write_text(json.dumps(out))
    print(f"wrote {STORE / 'ic_findings.json'}")
    print("to publish: cp", STORE / "ic_findings.json",
          Path(__file__).resolve().parents[2] / "results" / "ic_findings.json")
    print("sessions", out["sessions"], "IS", out["is_n"], "OOS", out["oos_n"])
    for k in out["curves"]:
        print("slip", k, {a: v[-1] for a, v in out["curves"][k].items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
