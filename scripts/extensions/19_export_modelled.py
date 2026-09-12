#!/usr/bin/env python3
"""19_export_modelled.py -- payload for the revised tearsheet, everything on
modelled per-leg slippage rather than a flat assumption."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.core import STORE
import lib.ic_costs as IC

QTY, RF = 750, 0.065


def main():
    d = pd.read_csv(STORE / "ic_modelled_daily.csv")
    d["year"] = pd.to_datetime(d["date"]).dt.year
    d = d[d.year.isin([2025, 2026])].sort_values("date").reset_index(drop=True)
    sp = pd.read_csv(STORE / "spread_estimates.csv")
    sp["year"] = pd.to_datetime(sp["date"]).dt.year

    def hedge(mult):
        hp = d["hoptg_pnl"].fillna(0) * QTY * mult
        prem = d["hoptg_prem"].fillna(0) * QTY * mult
        slip = np.zeros(len(d)); stat = np.zeros(len(d))
        for i, r in enumerate(d.itertuples()):
            for leg, hs in (("sc", r.hs_sc), ("sp", r.hs_sp)):
                e = getattr(r, f"hoptg_{leg}_ent", np.nan)
                x = getattr(r, f"hoptg_{leg}_ext", np.nan)
                if np.isfinite(e) and np.isfinite(x):
                    q = QTY * mult
                    slip[i] += 2 * q * hs
                    stat[i] += IC._statutory(float(e) * q, float(x) * q, 2, on=r.date)
        return hp - slip - stat, prem

    curves = {"IC blind": [round(float(x)) for x in np.cumsum(d.ic_net)]}
    for m in (1, 2, 3):
        h, _ = hedge(m)
        curves[f"IC + hedge {m}x"] = [round(float(x)) for x in np.cumsum(d.ic_net + h)]

    d["m"] = pd.to_datetime(d["date"]).dt.to_period("M").astype(str)
    h2, _ = hedge(2)
    monthly = {"IC blind": {k: round(float(v)) for k, v in d.groupby("m")["ic_net"].sum().items()},
               "IC + hedge 2x": {k: round(float(v)) for k, v in
                                 pd.Series((d.ic_net + h2).to_numpy(), index=d["m"]).groupby(level=0).sum().items()}}

    piv = sp.pivot_table(index="role", columns="year", values="half_spread", aggfunc="median")
    out = {
        "sessions": int(len(d)), "dates": list(d["date"]),
        "curves": curves, "monthly": monthly,
        "spread_by_role": {r: [round(float(piv.loc[r, 2025]), 4), round(float(piv.loc[r, 2026]), 4)]
                           for r in piv.index},
        "spread_pct_by_prem": {"<25": 0.26, "25-50": 0.18, "50-100": 0.17, "100-200": 0.13, ">200": 0.08},
        "measured_slip": {"2025": 0.1089, "2026": 0.1390, "all_mean": 0.1153, "all_median": 0.0941},
        "breakeven": {"IC blind": 0.217, "IC + hedge 1x": 0.317,
                      "IC + hedge 2x": 0.380, "IC + hedge 3x": 0.421},
        "attribution": {
            "2025": {"gross": 1786, "slip": -653, "stat": -481, "net": 651,
                     "hedge_gross": 3167, "hedge_cost": -1155, "total": 2663, "n": 245},
            "2026": {"gross": -113, "slip": -834, "stat": -611, "net": -1558,
                     "hedge_gross": 548, "hedge_cost": -1661, "total": -2670, "n": 77}},
        "hold": {"2025": [[1, 245, 1786, 1134, 651, 651, 57.1, 0.66],
                          [2, 96, 397, 1158, -761, -380, 54.2, -0.34],
                          [3, 47, -4953, 1157, -6111, -2037, 48.9, -1.29]],
                 "2026": [[1, 77, -113, 1445, -1558, -1558, 55.8, -1.86],
                          [2, 30, -813, 1405, -2217, -1109, 50.0, -0.89],
                          [3, 15, 14242, 1295, 12947, 4316, 80.0, 1.63]]},
    }
    # headline metrics on modelled cost
    def score(pnl, cap):
        base = float(np.percentile(cap, 95)); eq = np.cumsum(pnl)
        dd = float(np.max(np.maximum.accumulate(np.concatenate([[0.], eq])) - np.concatenate([[0.], eq])))
        r = pnl / base; sd = r.std(ddof=1); dn = r[r < 0]
        dsd = dn.std(ddof=1) if len(dn) > 1 else np.nan
        ann = r.mean() * 252
        return [round(float(pnl.sum())), round(base), round(float(pnl.sum()) / base * 100, 1),
                round((ann - RF) / (sd * np.sqrt(252)), 2),
                round((ann - RF) / (dsd * np.sqrt(252)), 2) if dsd and dsd > 0 else None,
                round(dd), round(dd / base * 100, 1),
                round(float(pnl.sum()) / dd, 2) if dd > 0 else None,
                round(float((pnl > 0).mean()) * 100, 1)]
    head = {}
    for pname, yr in (("2025 in-sample", 2025), ("2026 holdout", 2026)):
        mask = (d.year == yr).to_numpy()
        head[pname] = {"IC blind": score(d.ic_net.to_numpy()[mask], d.maxloss.to_numpy()[mask])}
        for m in (1, 2, 3):
            h, prem = hedge(m)
            head[pname][f"IC + hedge {m}x"] = score((d.ic_net + h).to_numpy()[mask],
                                                    (d.maxloss + prem).to_numpy()[mask])
    out["headline"] = head
    gate = pd.read_csv(STORE / "cost_modelled_results.csv")
    out["gate"] = gate.to_dict("records")
    (STORE / "modelled_payload.json").write_text(json.dumps(out))
    Path(Path(__file__).resolve().parents[2] / "results" / "modelled_payload.json").write_text(json.dumps(out))
    print("sessions", out["sessions"])
    print(json.dumps(out["headline"], indent=1)[:700])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
