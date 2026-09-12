"""Renders the three README charts from committed results/ data. Not part
of the pipeline -- run from the repo root after results/ changes, commit
the regenerated PNGs: python3 docs/make_charts.py"""
import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.edgecolor": "#444444",
    "axes.labelcolor": "#222222",
    "text.color": "#222222",
    "xtick.color": "#444444",
    "ytick.color": "#444444",
    "axes.grid": True,
    "grid.color": "#e6e6e6",
    "grid.linewidth": 0.8,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})

INK = "#1b2130"
RED = "#c0392b"
GOLD = "#b3803a"
BLUE = "#39738a"
GREY = "#8992a3"

ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------- chart 1
# Iron condor equity curves, all 4 hedge variants, IS/OOS split marked.
ic = json.load(open(f"{ROOT}/results/ic_findings.json"))
dates = [datetime.strptime(d, "%Y-%m-%d") for d in ic["dates"]]
curves = ic["curves"]["0.10"]

fig, ax = plt.subplots(figsize=(9, 5), dpi=160)
colors = {"IC blind": INK, "IC + hedge 1x": BLUE, "IC + hedge 2x": GOLD, "IC + hedge 3x": RED}
for label in ["IC blind", "IC + hedge 1x", "IC + hedge 2x", "IC + hedge 3x"]:
    ax.plot(dates, curves[label], label=label, color=colors[label], linewidth=1.6)

split = datetime(2026, 1, 1)
ax.axvline(split, color="#999999", linestyle="--", linewidth=1)
ymax = max(max(v) for v in curves.values())
ax.text(split, ymax * 1.02, "  2026 holdout →", color="#666666", fontsize=9, va="bottom")

ax.set_title("Iron condor: strong in-sample, negative in its own holdout", fontsize=13, color=INK, pad=12)
ax.set_ylabel("Cumulative P&L (₹, 10 lots, 0.10 pt slippage)")
ax.yaxis.set_major_formatter(lambda x, _: f"{x/1000:,.0f}k")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.legend(loc="upper left", frameon=False, fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(f"{ROOT}/docs/img/ic_equity_curves.png")
plt.close(fig)

# ---------------------------------------------------------------- chart 2
# Long-premium daily arms: total P&L by arm and year, all six bars negative.
findings = json.load(open(f"{ROOT}/results/findings.json"))
arms = findings["daily_arms"]
labels = {"A": "Arm A\nlong straddle", "B": "Arm B\nshort strangle", "C": "Arm C\nlong breakout leg"}

fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
arm_names = ["A", "B", "C"]
x = np.arange(len(arm_names))
width = 0.32
for i, yr in enumerate((2025, 2026)):
    vals = [next(a["total"] for a in arms if a["arm"] == an and a["year"] == yr) for an in arm_names]
    offset = (i - 0.5) * width
    bars = ax.bar(x + offset, vals, width, label=f"{yr}{' (holdout)' if yr == 2026 else ' (in-sample)'}",
                   color=BLUE if yr == 2025 else RED)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v - (2500 if v < 0 else -800),
                f"₹{v:,.0f}", ha="center", va="top" if v < 0 else "bottom", fontsize=8.5, color="#333333")

ax.axhline(0, color="#444444", linewidth=0.8)
ax.set_ylim(top=18000)
ax.set_xticks(x)
ax.set_xticklabels([labels[a] for a in arm_names])
ax.set_ylabel("Total P&L (₹, net of costs)")
ax.set_title("Every long-premium arm loses money, both periods", fontsize=13, color=INK, pad=12)
ax.legend(loc="upper right", frameon=True, framealpha=0.95, edgecolor="none", fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(f"{ROOT}/docs/img/daily_arms.png")
plt.close(fig)

# ---------------------------------------------------------------- chart 3
# Width quintile vs excess absolute move, 2025 & 2026, with CI whiskers.
wv = findings["width_vol"]
fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
width = 0.32
for i, (period, color) in enumerate((("2025 in-sample", BLUE), ("2026 holdout", RED))):
    rows = sorted([r for r in wv if r["period"] == period and r["h"] == 30], key=lambda r: r["q"])
    qs = [r["q"] for r in rows]
    vals = [r["exabs"] for r in rows]
    err = [[r["exabs"] - r["lo"] for r in rows], [r["hi"] - r["exabs"] for r in rows]]
    offset = (i - 0.5) * width
    ax.bar(np.array(qs) + offset, vals, width, yerr=err, capsize=3, color=color,
           label=period, error_kw={"linewidth": 1, "ecolor": "#555555"})

ax.axhline(0, color="#444444", linewidth=0.8)
ax.set_xticks(range(5))
ax.set_xticklabels(["q0\nnarrowest", "q1", "q2", "q3", "q4\nwidest"])
ax.set_ylabel("Excess |30-min move| vs. same time-of-day baseline (pts)")
ax.set_title("Channel width forecasts range — holds out of sample", fontsize=13, color=INK, pad=12)
ax.legend(loc="upper left", frameon=False, fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(f"{ROOT}/docs/img/width_signal.png")
plt.close(fig)

# ---------------------------------------------------------------- chart 4
# IC blind monthly P&L, colored by sign, IS/OOS split marked.
monthly = ic["monthly"]["IC blind"]
months = sorted(monthly.keys())
vals = [monthly[m] for m in months]
fig, ax = plt.subplots(figsize=(9, 4.5), dpi=160)
colors = [BLUE if v >= 0 else RED for v in vals]
ax.bar(range(len(months)), vals, color=colors, width=0.7)
split_idx = months.index("2026-01")
ax.axvline(split_idx - 0.5, color="#999999", linestyle="--", linewidth=1)
ax.text(split_idx - 0.5, max(vals) * 1.02, "  2026 holdout →", color="#666666", fontsize=9, va="bottom")
ax.axhline(0, color="#444444", linewidth=0.8)
ax.set_xticks(range(len(months)))
ax.set_xticklabels([m[2:] for m in months], rotation=45, ha="right", fontsize=8.5)
ax.set_ylabel("Monthly P&L (₹, IC blind, 10 lots)")
ax.yaxis.set_major_formatter(lambda x, _: f"{x/1000:,.0f}k")
ax.set_title("Blind condor month by month: mixed 2025, red most months since", fontsize=13, color=INK, pad=12)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(f"{ROOT}/docs/img/ic_monthly.png")
plt.close(fig)

# ---------------------------------------------------------------- chart 5
# Q1 conditioning grid: significant cells found vs. expected by chance.
q1 = findings["q1_summary"]
fig, ax = plt.subplots(figsize=(5.5, 4.5), dpi=160)
xs = [1, 2]
bars = ax.bar(xs, [q1["significant_2024"], q1["expected_by_chance"]], width=0.5,
              color=[GOLD, GREY])
ax.set_xticks(xs)
ax.set_xticklabels(["Found\nsignificant", "Expected\nby chance\n(5% level)"])
for b, v in zip(bars, [q1["significant_2024"], q1["expected_by_chance"]]):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.08, f"{v:g}", ha="center", va="bottom",
            fontsize=11, color="#222222")
ax.set_ylim(0, max(q1["significant_2024"], q1["expected_by_chance"]) * 1.4)
ax.set_ylabel("Cells")
ax.set_title(f"Q1 direction grid: {q1['cells_tested_2024']} cells tested,\n"
             "result indistinguishable from chance", fontsize=12.5, color=INK, pad=12)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(f"{ROOT}/docs/img/q1_significance.png")
plt.close(fig)

print("wrote 5 charts to docs/img/")
