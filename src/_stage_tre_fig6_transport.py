"""FIGURE 6. The disruption documented at the transport boundary, and the attenuation it produces.

The accounting in Figures 3 to 5 is measured entirely on the destination side of the corridor.
By itself it cannot say that a corridor event occurred, how severe it was at the point of
interruption, or whether the inventory release it records answered a physical shortage. Two
public transport and price series settle those questions, and the third panel puts the corridor
measurement on the same retention scale as the two destination-side measurements. The reader
compares the three boundaries by eye, not reading a comparison asserted in text.

a  Monthly Tanker-category transits of the Strait of Hormuz, January to May 2026.
b  Dated Brent spot and the spot-to-12th-contract proxy, November 2025 to May 2026.
c  The same acute-window-over-baseline retention ratio at the corridor, at the destination
   customs boundary, and at the refinery gate, on one 0 to 100 per cent scale.

TD3C tanker freight was a fourth panel and is not one now. That route prices a lane into Asia, not any lane serving the predominantly European destinations measured in this paper. It could
establish that freight repriced but not that these systems paid it. The baseline, the acute mean
and the ratio stay in the manuscript's transport paragraph, and the weekly assessments stay in
the source-data bundle this script still writes.

March 2026 is the transition month. Hostilities began on 28 February 2026, which places the
whole baseline before the disruption and makes March the month the disruption started in, so
March is excluded from both windows. Every panel marks it, on the axis and as a hatched ground. Therefore no reader can mistake it for part of either window.

Nothing this is computed from the manuscript's estimates. Every plotted value is read from the
reviewer data bundle, from sourcedata/transport_context.csv, or from the primary accounting
panel, and the script writes back everything it draws to sourcedata/fig6_transport_source_data.csv
and sourcedata/fig6_attenuation_source_data.csv. Both file names are new: sourcedata has an
unrelated fig6_source_data.csv belonging to a supplementary panel, which is left alone.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

import os as _os, sys as _sys
_CODE = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_CODE] + [_os.path.join(_CODE, _d) for _d in sorted(_os.listdir(_CODE))
                           if _os.path.isdir(_os.path.join(_CODE, _d))]

from figstyle import (HEAD, INK, MUTED, ROLE, RULE, SMALL, TINY, W2, apply_style,
                       figure_legend, light_grid, save, SD,
                       AX, TERM, TICK, LABEL, TITLE, axis_label, panel_title, country_labels, thin_leader, SD_WRITE)

apply_style()

BUNDLE = Path(__file__).resolve().parents[2] / "reviewer_data_csv_bundle_20260921"

MINUS = chr(8722)                      # the true minus sign, never a hyphen
BASELINE = ["2026-01", "2026-02"]      # window definitions, identical to tre_transport_context
TRANSITION = "2026-03"
ACUTE = ["2026-04", "2026-05"]

EXCLUDED_FACE = "#F3F0EA"              # the ground of a measured-but-excluded month
EXCLUDED_EDGE = "#B9B2A4"
WINDOW_FACE = ROLE["outside"]          # the ground of a month inside an analysis window

C_TRANSIT = ROLE["chokepoint"]         # the strait itself
C_FREIGHT = ROLE["infrastructure"]     # the cost of moving crude through it
C_SPOT = ROLE["availability"]          # the price of the crude
C_SPREAD = ROLE["inventory"]           # the premium on prompt barrels
C_IMPORT = ROLE["import_shock"]        # crude arriving at the destination
TRACK = ROLE["nodata"]                 # the unretained remainder of a 100 per cent track

THOUSANDS = FuncFormatter(lambda v, _: f"{v:,.0f}".replace("-", MINUS))

ONSET_LS = (0, (5.0, 1.6, 1.2, 1.6))
MEAN_LS = (0, (4, 2))        # acute window mean
BASE_LS = (0, ())            # baseline window mean, same color, solid

# ---------------------------------------------------------------------------
TR = pd.read_csv(BUNDLE / "hormuz_tanker_transits_monthly.csv").set_index("date")
FR = pd.read_csv(BUNDLE / "td3c_monthly.csv").set_index("date")
BR = pd.read_csv(BUNDLE / "brent_price_curve.csv").set_index("date")
CTX = pd.read_csv(SD / "transport_context.csv", header=None, index_col=0).squeeze("columns")

MON5 = list(TR.index)                                   # 2026-01 . 2026-05
MON7 = list(BR.index)                                   # 2025-11 . 2026-05
LAB = {"2025-11": "Nov", "2025-12": "Dec", "2026-01": "Jan", "2026-02": "Feb",
       "2026-03": "Mar", "2026-04": "Apr", "2026-05": "May"}

transits = TR.tanker_transits.astype(float)
t_base, t_acute = transits[BASELINE].mean(), transits[ACUTE].mean()
t_mar_pct = float(TR.monthly_pct_change[TRANSITION])

ws = FR.mean_worldscale.astype(float)
ws_lo, ws_hi = FR.min_worldscale.astype(float), FR.max_worldscale.astype(float)
f_base, f_acute = ws[BASELINE].mean(), ws[ACUTE].mean()

spot = BR.eia_monthly_average_brent_spot_usd_bbl.astype(float)
spot_me = BR.eia_month_end_brent_spot_usd_bbl.astype(float)
spread = BR.spot_to_12th_contract_proxy_usd_bbl.astype(float)
s_base = spread[BASELINE].mean()

P = pd.read_csv(SD / "primary_panel.csv")
CON = P[(P.population == "primary") & (P.import_retention < 0.95)]
r_transit = float(CTX["transits_retention_acute"])
r_import = float(CTX["import_retention_for_contrast"])
r_avail = float(CON.availability_retention.mean())
assert abs(r_import - CON.import_retention.mean()) < 5e-5, "contrast set does not match context"
assert set(CON.window) == {"AprMay"}, "contrast set is not the acute window"

STAGES = [
    ("Strait of Hormuz corridor", "Tanker transits through the strait", r_transit, C_TRANSIT),
    ("Destination customs boundary", f"Crude imports, {len(CON)} contracting systems", r_import, C_IMPORT),
    ("Refinery gate", f"Domestic crude availability, the same {len(CON)} systems", r_avail, C_SPOT),
]

# ---------------------------------------------------------------------------
def windows(ax, months, ground=None):
    """Analysis-window ground, the excluded transition month, and the onset rule.

    The transition month carries a hatched ground, not a warning in words, since a
    reader reads position before prose, and a month drawn on the same ground as its neighbours
    is read as one of them. ``ground`` names the axes the bands are painted on, which is not
    always the axes that carries the ticks: in a twinned panel the bands have to sit under the
    lower of the two, or they bury whatever that axes draws.
    """
    i = {m: k for k, m in enumerate(months)}
    lo, hi = -0.6, len(months) - 0.4
    g = ground if ground is not None else ax
    for grp in (BASELINE, ACUTE):
        g.axvspan(i[grp[0]] - 0.5, i[grp[-1]] + 0.5, facecolor=WINDOW_FACE, lw=0.0, zorder=0)
    g.axvspan(i[TRANSITION] - 0.5, i[TRANSITION] + 0.5, facecolor=EXCLUDED_FACE,
              edgecolor=EXCLUDED_EDGE, hatch="////", lw=0.0, zorder=0.5)
    ax.axvline(i[TRANSITION] - 0.5, color=INK, lw=0.9, ls=ONSET_LS, zorder=5)
    ax.set_xlim(lo, hi)
    ax.set_xticks(range(len(months)))
    ax.set_xticklabels([LAB[m] for m in months],
                       fontsize=TINY, linespacing=1.1)
    for k, t in enumerate(ax.get_xticklabels()):
        if months[k] == TRANSITION:
            t.set_color(MUTED)
            t.set_style("italic")
    ax.tick_params(axis="x", length=0, pad=2.5)
    return i

def mean_rule(ax, x0, x1, y, text, color=INK, tx=None, ty=None, ha="center",
              kw_ls=MEAN_LS):
    """A window mean drawn where it applies, labelled in the empty part of the panel.

    The label is carried out to clear air on a hairline leader, not set against the
    bars, since a value label that lands on a filled bar stops being a value label.
    """
    ax.plot([x0, x1], [y, y], color=color, lw=0.95, ls=kw_ls, zorder=6,
            solid_capstyle="butt")
    if text:
        ax.annotate(text, xy=((x0 + x1) / 2 if tx is None else tx, y),
                    xytext=((x0 + x1) / 2 if tx is None else tx, ty), ha=ha, va="bottom",
                    fontsize=TINY, color=color, zorder=8,
                    arrowprops=dict(arrowstyle="-", lw=0.5, color=color, shrinkA=1.0,
                                    shrinkB=0.5),
                    path_effects=[pe.withStroke(linewidth=2.2, foreground="white")])

def panel_note(ax, lines):
    """The provider's own scope caveat, under the panel it qualifies."""
    ax.text(0.0, -0.175, lines, transform=ax.transAxes, ha="left", va="top",
            fontsize=TINY, color=MUTED, linespacing=1.28)

def head(ax, tag, title, right=None):
    ax.set_title(f"{tag}   {title}", fontsize=HEAD, loc="left", pad=4.0)
    if right:
        ax.set_title(right, fontsize=TINY, loc="right", pad=4.5, color=MUTED)

fig = plt.figure(figsize=(W2, 5.55))
gst = fig.add_gridspec(1, 2, left=0.070, right=0.934, top=0.935, bottom=0.452, wspace=0.34)
gsb = fig.add_gridspec(1, 1, left=0.268, right=0.934, top=0.300, bottom=0.185)

# --- a. corridor transits ---------------------------------------------------
ax = fig.add_subplot(gst[0])
ix = windows(ax, MON5)
ax.set_ylim(0, 1520)
for m, v in transits.items():
    x = ix[m]
    if m == TRANSITION:
        ax.bar(x, v, width=0.58, facecolor="white", edgecolor=C_TRANSIT, lw=0.8,
               hatch="///", zorder=3)
    else:
        ax.bar(x, v, width=0.58, color=C_TRANSIT, edgecolor="white", lw=0.5, zorder=3)
    dy = 7.5 if m in ACUTE else 3
    ax.annotate(f"{v:,.0f}", xy=(x, v), xytext=(0, dy), textcoords="offset points",
                ha="center", va="bottom", fontsize=TINY, color=INK, zorder=8,
                path_effects=[pe.withStroke(linewidth=2.0, foreground="white")])
mean_rule(ax, ix["2026-01"] - 0.45, ix["2026-02"] + 0.45, t_base, None,
          kw_ls=BASE_LS)
mean_rule(ax, ix["2026-04"] - 0.45, ix["2026-05"] + 0.45, t_acute, None)
ax.set_ylabel(AX["transits"], fontsize=LABEL)
ax.yaxis.set_major_formatter(THOUSANDS)
light_grid(ax)
panel_title(ax, "a", "Tanker transits")

# --- b. crude price and the shape of the curve ------------------------------
ax = fig.add_subplot(gst[1])
ax2 = ax.twinx()
ax2.set_zorder(ax.get_zorder() - 1)
ax.patch.set_visible(False)
ix = windows(ax, MON7, ground=ax2)
L0, L1, R0, R1 = 40.0, 138.0, 0.0, 62.0
ax.set_ylim(L0, L1)
ax2.set_ylim(R0, R1)
ax2.spines["right"].set_visible(True)
ax2.spines["top"].set_visible(False)
ax2.spines["left"].set_visible(False)

def r2l(v):
    """A spread value in the coordinates of the price axis.

    Everything drawn on the lower axes sits under everything drawn on the upper one, whatever
    the local z order says. The bars' own value labels have to be placed on the upper axes
    or the price line and the grid are free to run straight through them.
    """
    return L0 + (v - R0) / (R1 - R0) * (L1 - L0)

for m, v in spread.items():
    if m == TRANSITION:
        ax2.bar(ix[m], v, width=0.52, facecolor="white", edgecolor=C_SPREAD, lw=0.8,
                hatch="///", zorder=3)
    else:
        ax2.bar(ix[m], v, width=0.52, color=C_SPREAD, edgecolor="white", lw=0.4, zorder=3)
    # a tall solid bar carries its label inside, since above it is where the price line runs
    inside = v >= 20 and m != TRANSITION
    ax.annotate(f"{v:,.1f}", xy=(ix[m], r2l(0.58 * v if inside else v)),
                xytext=(0, 0 if inside else 2.5), textcoords="offset points",
                ha="center", va="center" if inside else "bottom", fontsize=TINY,
                color="white" if inside else C_SPREAD, zorder=9,
                fontweight="bold" if inside else "normal",
                path_effects=None if inside
                else [pe.withStroke(linewidth=2.0, foreground="white")])
ax2.plot([-0.6, 6.6], [s_base, s_base], color=C_SPREAD, lw=0.9, ls=MEAN_LS, zorder=6)

ax.plot(range(len(MON7)), spot.values, color=C_SPOT, lw=1.2, marker="o", ms=3.0,
        mfc=C_SPOT, mec="white", mew=0.6, zorder=7)
ax.plot(range(len(MON7)), spot_me.values, ls="none", marker="o", ms=4.2, mfc="none",
        mec=C_SPOT, mew=0.8, zorder=7)

ax.legend(handles=[
    Line2D([], [], color=C_SPOT, lw=1.2, marker="o", ms=3.0, mfc=C_SPOT, mec="white",
           mew=0.6, label="Monthly average price"),
    Line2D([], [], color="none", marker="o", ms=4.2, mfc="none", mec=C_SPOT, mew=0.8,
           label="Month-end price"),
    Patch(facecolor=C_SPREAD, edgecolor="white", lw=0.4, label="Price difference"),
    Line2D([], [], color=C_SPREAD, lw=0.9, ls=MEAN_LS,
           label=f"Pre-disruption difference ({s_base:,.2f})")],
    loc="upper left", fontsize=TICK, frameon=False, labelspacing=0.30,
    handletextpad=0.45, handlelength=1.5, borderaxespad=0.15, borderpad=0.0)

ax.set_ylabel(axis_label("Brent, immediate delivery", "USD/bbl"), fontsize=LABEL, color=C_SPOT)
ax2.set_ylabel(axis_label("price difference, immediate minus twelve months", "USD/bbl"), fontsize=LABEL,
               color=C_SPREAD)
ax2.tick_params(axis="y", colors=C_SPREAD, labelsize=7.5)
ax2.spines["right"].set_color(C_SPREAD)
ax.set_axisbelow(True)
for t in ax.get_yticks():
    if L0 < t < L1:
        ax2.axhline((t - L0) / (L1 - L0) * (R1 - R0) + R0, color=RULE, lw=0.4, alpha=0.7,
                    zorder=0.8)
panel_title(ax, "b", "Price and price difference")

axd = fig.add_subplot(gsb[0])
y = np.arange(len(STAGES))[::-1]
for yi, (name, basis, r, col) in zip(y, STAGES):
    axd.barh(yi, 100.0, height=0.52, color=TRACK, edgecolor="white", lw=0.5, zorder=2)
    axd.barh(yi, 100 * r, height=0.52, color=col, edgecolor="white", lw=0.5, zorder=3)
    axd.annotate(f"{100 * r:.1f}%", xy=(100 * r, yi), xytext=(4, 0),
                 textcoords="offset points", ha="left", va="center", fontsize=SMALL,
                 fontweight="bold", color=col, zorder=8,
                 path_effects=[pe.withStroke(linewidth=2.2, foreground="white")])
axd.axhline(y[0] - 0.5, color=RULE, lw=0.7, zorder=1)
axd.annotate("different population\nfrom the two below", xy=(109.5, y[0]),
             ha="left", va="center", fontsize=TINY, color=MUTED, zorder=9)
for k in range(1, len(STAGES) - 1):
    a, b = 100 * STAGES[k][2], 100 * STAGES[k + 1][2]
    ya, yb = y[k], y[k + 1]
    ym = (ya + yb) / 2
    for x, y0 in ((a, ya), (b, yb)):
        axd.plot([x, x], [y0 + (0.27 if y0 < ym else -0.27), ym], ls=(0, (1, 1.7)),
                 color=MUTED, lw=0.6, zorder=6, solid_capstyle="butt")
    axd.annotate("", xy=(b, ym), xytext=(a, ym),
                 arrowprops=dict(arrowstyle="<|-|>,head_width=0.10,head_length=0.28",
                                 lw=0.6, color=MUTED, shrinkA=0.0, shrinkB=0.0))
    wide = (b - a) > 25
    axd.text((a + b) / 2 if wide else b + 11.0, ym, f"+{b - a:.1f} pp",
             ha="center" if wide else "left", va="center", fontsize=TINY, color=MUTED,
             zorder=9, bbox=dict(facecolor="white", edgecolor="none", pad=1.4))
axd.set_yticks(y)
ROW_LABEL = ["Tanker transits", f"Crude imports, {len(CON)} countries",
             f"Refinery supply, {len(CON)} countries"]
axd.set_yticklabels(ROW_LABEL, fontsize=TICK)
axd.tick_params(axis="y", length=0, pad=3.0)
for t, (_, _, _, col) in zip(axd.get_yticklabels(), STAGES):
    t.set_color(INK)
axd.set_xlim(0, 137)
axd.set_xticks([0, 20, 40, 60, 80, 100])
axd.set_ylim(-0.62, len(STAGES) - 0.38)
axd.set_xlabel(AX["retention_pct"], fontsize=LABEL)
light_grid(axd, axis="x")
axd.spines["left"].set_visible(False)
panel_title(axd, "c", "Ratios at three boundaries")

figure_legend(fig, [
    Patch(facecolor=EXCLUDED_FACE, edgecolor=EXCLUDED_EDGE, hatch="////",
          label=TERM["transition"]),
    Line2D([], [], color=INK, lw=0.9, ls=ONSET_LS,
           label="Onset of hostilities (28 Feb 2026)"),
    Line2D([], [], color=INK, lw=0.95, ls=BASE_LS,
           label="Pre-disruption mean (Jan-Feb 2026)"),
    Line2D([], [], color=INK, lw=0.95, ls=MEAN_LS,
           label="Disruption mean (Apr-May 2026)"),
], ncol=3, y=0.006, handlelength=1.5, columnspacing=1.5)

save(fig, "Fig6_transport_boundary")

# --- the source data, exactly what is drawn ---------------------------------
rows = []
for m in MON7:
    rows.append({
        "month": m, "panel_a_tanker_transits": transits.get(m, np.nan),
        "td3c_mean_worldscale": ws.get(m, np.nan),
        "td3c_min_worldscale": ws_lo.get(m, np.nan),
        "td3c_max_worldscale": ws_hi.get(m, np.nan),
        "panel_b_brent_monthly_avg_spot_usd_bbl": spot.get(m, np.nan),
        "panel_b_brent_month_end_spot_usd_bbl": spot_me.get(m, np.nan),
        "panel_b_spot_to_12th_proxy_usd_bbl": spread.get(m, np.nan),
        "window": ("baseline" if m in BASELINE else
                   "transition, excluded" if m == TRANSITION else
                   "acute" if m in ACUTE else "outside the five-month window"),
    })
S = pd.DataFrame(rows)
S.to_csv(SD_WRITE / "fig6_transport_source_data.csv", index=False)

D = pd.DataFrame([{"panel_c_stage": n, "basis": b, "retention": round(r, 4),
                   "retention_pct": round(100 * r, 1)} for n, b, r, _ in STAGES])
D.to_csv(SD_WRITE / "fig6_attenuation_source_data.csv", index=False)

print(f"  a  transits baseline {t_base:,.1f}  March {transits[TRANSITION]:,.0f} "
      f"({t_mar_pct:.3f}%)  acute {t_acute:,.1f}")
print(f"  -  TD3C baseline WS{f_base:.1f}  acute WS{f_acute:.1f}  (deposited, not plotted)  "
      f"x{f_acute / f_base:.2f}")
print(f"  b  spot baseline {spot[BASELINE].mean():.2f}  acute {spot[ACUTE].mean():.2f}; "
      f"spread baseline {s_base:.2f}  March {spread[TRANSITION]:.2f}  "
      f"acute {spread[ACUTE].mean():.2f}")
print(f"  d  corridor {100 * r_transit:.1f}%  imports {100 * r_import:.1f}%  "
      f"availability {100 * r_avail:.1f}%  (n = {len(CON)}: "
      f"{', '.join(CON.country_iso2)})")
print(f"  wrote {SD / 'fig6_transport_source_data.csv'} and fig6_attenuation_source_data.csv")
