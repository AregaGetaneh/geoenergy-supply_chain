"""FIGURES 4 and 5, both on the structurally defined primary population.

FIGURE 4  Crude-import retention against domestic availability, and the historical comparison
          that addresses selection on falling imports as the explanation.
FIGURE 5  Reported inventory position and temporal buffering: the reported pre-event stock
          level, the inventory adjustment, and the baseline-equivalent and prospective horizons
          over which the observed draw rate could have continued.

It also writes the matched corridor-dependence panel that the supplement's FigS5 reads.
The corridor plate itself is drawn only by tre_si.py; this file used to draw a second copy
of it as Fig6, which no document referenced, and that plate is now in figures/_withdrawn/.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.patheffects as pe
import matplotlib.transforms as mtransforms
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats

import os as _os, sys as _sys
_CODE = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_CODE] + [_os.path.join(_CODE, _d) for _d in sorted(_os.listdir(_CODE))
                           if _os.path.isdir(_os.path.join(_CODE, _d))]

from figstyle import (DEST_MARKER, FAINT_MARKER, HEAD, HUB_MARKER, INK, MUTED, NAME,
                       ROLE, RULE, SMALL, TINY, W2, apply_style,
                       light_grid, off_scale_band, save, SD, ROOT,
                       AX, TERM, TICK, LABEL, TITLE, axis_label, panel_title, country_labels, thin_leader, plain_log_ticks, SD_WRITE)

apply_style()
P = pd.read_csv(SD / "primary_panel.csv")
# ADMISSIBLE_ONLY: physically valid destinations plus crude-exporting importers
P = P[P.population.isin(["primary", "crude-exporting importer"])].copy()
PL = pd.read_csv(SD / "placebo_windows.csv")
COV = pd.read_csv(SD / "baseline_days_of_cover.csv").rename(
    columns={"REF_AREA": "country_iso2"})
DEP = pd.read_csv(ROOT / "paper_nature_communications_gate" / "event_identification" /
                  "country_event_anomaly.csv")[["country_iso2", "dependence",
                                                "dep_lower", "dep_upper"]]
PRIM = set(P.country_iso2[P.population == "primary"])
HUB = set(P.country_iso2[P.population == "crude-exporting importer"])
P["contracts"] = P.import_retention < 0.95
CP = P[P.population == "primary"]

DEST_C, HUB_C, FAINT = DEST_MARKER, HUB_MARKER, FAINT_MARKER

fig = plt.figure(figsize=(W2, 3.80))
gs = fig.add_gridspec(1, 3, width_ratios=[1.16, 1.02, 0.86], wspace=0.40,
                      left=0.070, right=0.986, top=0.912, bottom=0.175)

ax = fig.add_subplot(gs[0, 0])
XL, YL = (0.60, 1.40), (0.80, 1.40)
ax.fill_between(XL, XL, [YL[1]] * 2, color=DEST_C, alpha=0.07, zorder=1)
ax.plot([YL[0], XL[1]], [YL[0], XL[1]], color=INK, lw=1.0, ls=(0, (3, 2)), zorder=3)

ALL = P[P.import_retention.notna()]
W = ALL[ALL.import_retention <= XL[1]]
OFFSCALE = ALL[ALL.import_retention > XL[1]]
MS = 26
for msk, fc, ec in [(W.country_iso2.isin(PRIM) & W.contracts, DEST_C, "white"),
                    (W.country_iso2.isin(PRIM) & ~W.contracts, "white", MUTED),
                    (W.country_iso2.isin(HUB), HUB_C, INK)]:
    ax.scatter(W.import_retention[msk], W.availability_retention[msk], s=MS,
               facecolor=fc, edgecolor=ec, lw=0.8, zorder=4, alpha=0.92)

LBL = {"CZ": (7, 2, "left"), "ID": (7, 2, "left"), "PL": (0, -9, "center"),
       "IE": (-7, -1, "right"), "GR": (-6, 5, "right"), "US": (7, 2, "left"),
       "CH": (-6, 0, "right"), "DE": (31, 5, "left")}
for _, r in W.iterrows():
    if not r.contracts:
        continue
    dx, dy, ha = LBL.get(r.country_iso2, (6, 4, "left"))
    lead = dict(arrowstyle="-", lw=0.5, color=MUTED, shrinkA=1.0,
                shrinkB=3.5) if r.country_iso2 == "DE" else None
    ax.annotate(r.country_iso2, xy=(r.import_retention, r.availability_retention),
                xytext=(dx, dy), textcoords="offset points", ha=ha, va="center",
                fontsize=TINY, color=INK, fontweight="bold", arrowprops=lead)

OFF_LBL = {"HR": (-7, 0, "right", "center"), "SK": (-6, 5, "right", "bottom"),
           "HU": (-6, -5, "right", "top"), "BN": (-6, -5, "right", "top")}
for _, r in OFFSCALE.iterrows():
    up = r.availability_retention > YL[1]
    yv = min(max(r.availability_retention, YL[0] + 0.03), YL[1] - 0.02)
    ax.plot([XL[1]], [yv], marker=">", ms=5.0, color="white", mec=INK, mew=0.7,
            zorder=6, clip_on=False)
    txt = r.country_iso2
    dx, dy, ha, va = OFF_LBL.get(r.country_iso2, (-6, -5, "right", "top"))
    ax.annotate(txt, xy=(XL[1], yv), xytext=(dx, dy), textcoords="offset points",
                ha=ha, va=va, fontsize=TINY, color=INK, fontweight="bold", zorder=7,
                linespacing=1.35,
                path_effects=[pe.withStroke(linewidth=2.4, foreground="white")])
ax.set_xlim(*XL)
ax.set_ylim(*YL)
ax.set_xlabel(AX["import_ret"], fontsize=LABEL)
ax.set_ylabel(AX["avail_ret"], fontsize=LABEL)
light_grid(ax, axis="both")
pos = ax.get_position()
ax.set_position([pos.x0, pos.y0 + 0.148, pos.width, pos.height - 0.148])
ax.legend(handles=[
    Line2D([], [], marker="o", ms=4.6, ls="none", mfc=DEST_C, mec="white", mew=0.8,
           label=TERM["imports_fell"]),
    Line2D([], [], marker="o", ms=4.6, ls="none", mfc="white", mec=MUTED, mew=0.8,
           label=TERM["imports_stable"]),
    Line2D([], [], marker="o", ms=4.6, ls="none", mfc=HUB_C, mec=INK, mew=0.8,
           label="Crude-exporting importer"),
    # the diagonal was drawn and never declared
    Line2D([], [], color=MUTED, lw=0.9, ls=(0, (3, 2)), label="Equal ratios"),
    Line2D([], [], marker=">", ms=5.0, ls="none", mfc="white", mec=INK, mew=0.7,
           label="Off scale, see source data"),
    # the shaded half was drawn and never declared
    Patch(facecolor=DEST_C, alpha=0.07, edgecolor="none",
          label="Refinery supply fell less\nthan imports")],
    loc="upper left", bbox_to_anchor=(0.0, -0.195), ncol=2, columnspacing=0.9,
    fontsize=TICK, frameon=False, labelspacing=0.35,
    handletextpad=0.40, handlelength=1.0, borderaxespad=0.25, borderpad=0.0)
panel_title(ax, "a", "Crude imports against refinery supply")

# --- b. contracting destination systems -------------------------------------
ax = fig.add_subplot(gs[0, 1])
C = CP[CP.contracts].sort_values("import_retention")
y = np.arange(len(C))[::-1]
for yi, (_, r) in zip(y, C.iterrows()):
    up = r.availability_retention > r.import_retention
    ax.plot([r.import_retention, r.availability_retention], [yi, yi],
            color=DEST_C if up else ROLE["import_shock"], lw=2.0, alpha=0.55, zorder=2,
            solid_capstyle="round")
    ax.plot(r.import_retention, yi, "o", ms=5.6, mfc=ROLE["import_shock"], mec="white",
            mew=0.8, zorder=4)
    ax.plot(r.availability_retention, yi, "o", ms=5.6, mfc=ROLE["availability"],
            mec="white", mew=0.8, zorder=4)
ax.axvline(1.0, color=INK, lw=0.9, ls=(0, (3, 2)), zorder=1)
ax.set_yticks(y)
ax.set_yticklabels([NAME.get(a, a) for a in C.country_iso2], fontsize=SMALL)
ax.set_xlabel(AX["retention"], fontsize=LABEL)
ax.set_xlim(0.58, 1.07)
ax.set_ylim(-0.80, len(C) - 0.20)
light_grid(ax, axis="x")
k = int((C.availability_retention > C.import_retention).sum())
pv = stats.binomtest(k, len(C), 0.5, alternative="greater").pvalue
ax.legend(handles=[
    Line2D([], [], marker="o", ms=5.2, mfc=ROLE["import_shock"], mec="white", mew=0.8,
           ls="none", label="Crude imports"),
    Line2D([], [], marker="o", ms=5.2, mfc=ROLE["availability"], mec="white", mew=0.8,
           ls="none", label="Refinery supply")],
    loc="lower left", fontsize=TINY, frameon=False, labelspacing=0.35,
    handletextpad=0.40, handlelength=1.0, borderaxespad=0.25, borderpad=0.0)
panel_title(ax, "b", "Countries with an import reduction")

# --- c. historical comparison -------------------------------------------------------------
ax = fig.add_subplot(gs[0, 2])
Q = PL[PL.country_iso2.isin(PRIM) & (PL.import_retention < 0.95)]
pre, ev = Q[Q.year < 2026], Q[Q.year == 2026]
rng = np.random.default_rng(7)
ax.axhline(0, color=INK, lw=0.8, zorder=2)
for i, (g, col) in enumerate([(pre, FAINT), (ev, DEST_C)]):
    ax.scatter(i + rng.uniform(-0.17, 0.17, len(g)), g.wedge, s=17, facecolor=col,
               edgecolor="white", lw=0.4, alpha=0.88, zorder=4)
    ax.plot([i - 0.30, i + 0.30], [g.wedge.mean()] * 2, color=INK, lw=1.8, zorder=5)
    ax.text(i, -0.335, f"{int((g.wedge > 0).sum())} of {len(g)}", ha="center", va="top",
            fontsize=TINY, color=col if i else MUTED,
            fontweight="bold" if i else "normal")
mw = stats.mannwhitneyu(ev.wedge, pre.wedge, alternative="greater").pvalue
ax.set_xticks([0, 1])
ax.set_xticklabels(["Historical\n2020–2025", "Event\n2026"], fontsize=TICK, linespacing=1.2)
ax.set_xlim(-1.02, 2.35)
ax.set_ylim(-0.38, 0.35)
ax.set_ylabel(AX["gap_diff"], fontsize=LABEL)
light_grid(ax)
pm, em = pre.wedge.mean(), ev.wedge.mean()
dm = em - pm
ax.annotate(f"{pm:+.3f}", xy=(-0.31, pm), xytext=(-2, 0), textcoords="offset points",
            ha="right", va="center", fontsize=TINY, color=INK, fontweight="bold")
ax.annotate(f"{em:+.3f}", xy=(1.30, em), xytext=(0, 3), textcoords="offset points",
            ha="right", va="bottom", fontsize=TINY, color=INK, fontweight="bold")
GX = 1.58                                   # the comparison lives right of both swarms
ax.plot([0.32, GX], [pm] * 2, color=RULE, lw=0.6, ls=(0, (2, 2)), zorder=1)
ax.plot([1.32, GX], [em] * 2, color=RULE, lw=0.6, ls=(0, (2, 2)), zorder=1)
ax.annotate("", xy=(GX, em), xytext=(GX, pm),
            arrowprops=dict(arrowstyle="<->", lw=1.0, color=INK, shrinkA=0, shrinkB=0))
# One short label on the dimension, not a value plus a two-line gloss under it.
ax.text(GX - 0.10, (em + pm) / 2, f"{dm:+.3f}", fontsize=TICK,
        fontweight="bold", color=INK, ha="right", va="center")
ax.set_xlabel("Comparison group", fontsize=LABEL)
panel_title(ax, "c", "Event against historical years")
save(fig, "Fig4_arrivals_vs_availability")
CP[["country_iso2", "population", "baseline_imports_kbd",
    "baseline_availability_kbd", "import_retention", "availability_retention",
    "gap", "contracts"]].to_csv(SD_WRITE / "fig4_source_data.csv", index=False)
PL[["year", "country_iso2", "import_retention", "availability_retention",
    "wedge"]].to_csv(SD_WRITE / "fig4c_source_data.csv", index=False)
print(f"  Fig4: {k}/{len(C)} above line, p={pv:.4f}; placebo MW p={mw:.4f}")

M = P.merge(COV[["country_iso2", "CLOSTLV", "REFINOBS", "days_cover"]],
            on="country_iso2", how="left")
M["cover_ok"] = M.CLOSTLV > 0
M["s_stock"] = M.d_stockdraw_kbd / (-M.d_imports_kbd)
DC = pd.read_csv(SD / "depletion_corrected.csv")[["country_iso2", "acute_draw_kbd"]]
M = M.merge(DC, on="country_iso2", how="left")
M["months"] = np.where(M.acute_draw_kbd > 0,
                       M.CLOSTLV / (M.acute_draw_kbd * 30.5), np.nan)
M["draw_share"] = M.acute_draw_kbd / M.REFINOBS
M["h_stable"] = M.draw_share >= 0.01
M["display_name"] = [NAME.get(a, a) for a in M.country_iso2]
Md = M[M.country_iso2.isin(PRIM)].sort_values("display_name", ascending=False)
DEPP = pd.read_csv(SD / "depletion_prospective.csv")
DEPP["display_name"] = [NAME.get(a, a) for a in DEPP.country_iso2]

CAUTION_FACE = "#EDE7DB"
CAUTION_EDGE = "#B0A48C"
BASE_FACE = "#FFFFFF"        # the baseline-equivalent horizon, drawn open
FIG_H = 3.52
fig = plt.figure(figsize=(W2, FIG_H))

def fr(inches):
    """Figure fraction from a height in inches, which is how the strip below is placed."""
    return inches / FIG_H

gs = fig.add_gridspec(1, 3, width_ratios=[1.18, 0.98, 0.98], wspace=0.44,
                      left=0.128, right=0.985, top=0.930, bottom=fr(0.560))

ax = fig.add_subplot(gs[0, 0])
COVER = Md[Md.cover_ok]
ZERO = Md[~Md.cover_ok]
y = np.arange(len(COVER))
for yi, (_, r) in zip(y, COVER.iterrows()):
    col = DEST_C if r.contracts else FAINT
    ax.plot([0.6, r.days_cover], [yi, yi], color=col, lw=1.0, alpha=0.5, zorder=3)
    ax.plot(r.days_cover, yi, "o", ms=5.0, mfc=col, mec="white", mew=0.7, zorder=4)
ax.set_xscale("log")
ax.set_xlim(0.55, 300)
plain_log_ticks(ax, "x", [1, 10, 100])
ax.set_ylim(-0.95, len(COVER) + 2.1)
ax.set_yticks(list(y))
ax.set_yticklabels(list(COVER.display_name), fontsize=TINY)
ax.set_xlabel(AX["stock"], fontsize=LABEL)
light_grid(ax, axis="x")
ax.legend(handles=[
    Line2D([], [], marker="o", ms=5, mfc=DEST_C, mec="white", mew=0.7, ls="none",
           label=TERM["imports_fell"]),
    Line2D([], [], marker="o", ms=5, mfc=FAINT, mec="white", mew=0.7, ls="none",
           label=TERM["imports_stable"])],
    loc="upper left", fontsize=TINY, frameon=False, labelspacing=0.35,
    handletextpad=0.45, borderaxespad=0.2, borderpad=0.0)
panel_title(ax, "a", "Reported pre-disruption inventory")

ax = fig.add_subplot(gs[0, 1])
K = Md[Md.contracts & Md.cover_ok & Md.s_stock.notna()]
ax.scatter(K.days_cover, K.s_stock, s=42, facecolor=DEST_C, edgecolor="white", lw=0.7,
           zorder=4)
ax.axhline(1.0, color=INK, lw=0.9, ls=(0, (3, 2)), zorder=2)
BLBL = {"CH": (9, 0, "left", "center"), "IE": (-9, -1, "right", "center"),
        "GR": (8, 3, "left", "bottom"), "CZ": (-9, 0, "right", "center"),
        "DE": (-9, 3, "right", "bottom"), "PL": (-9, -3, "right", "top")}
for _, r in K.iterrows():
    dx, dy, ha, va = BLBL[r.country_iso2]
    ax.annotate(NAME.get(r.country_iso2, r.country_iso2), xy=(r.days_cover, r.s_stock),
                xytext=(dx, dy), textcoords="offset points", ha=ha, va=va,
                fontsize=TINY, color=INK)
ax.set_xscale("log")
ax.set_xlim(1.4, 220)
plain_log_ticks(ax, "x", [1, 10, 100])
ax.set_ylim(0.0, 1.45)
ax.set_xlabel(AX["stock"], fontsize=LABEL)
ax.set_ylabel(AX["inv_adj"], fontsize=LABEL)
light_grid(ax, axis="both")
# The reference line is declared in the panel key, not written along the line.
ax.legend(handles=[Line2D([], [], color=INK, lw=0.9, ls=(0, (3, 2)),
                          label="Inventory equals the import shortage")],
          loc="upper left", fontsize=TICK, frameon=False, handlelength=1.6,
          handletextpad=0.45, borderaxespad=0.2, borderpad=0.0)
rho = K.days_cover.corr(K.s_stock, method="spearman")
ax.text(0.025, 0.022,
        f"$n$ = {len(K)}, Spearman $\\rho$ = {rho:+.2f}".replace("-", chr(8722)),
        transform=ax.transAxes, va="bottom", fontsize=SMALL, color=MUTED)
panel_title(ax, "b", "Inventory and inventory use")

ax = fig.add_subplot(gs[0, 2])
D2 = DEPP[DEPP.country_iso2.isin(PRIM)].sort_values("display_name", ascending=False)
y = np.arange(len(D2))
BH, GAPB = 0.30, 0.025          # sub-bar height and the gap between the pair
for yi, (_, r) in zip(y, D2.iterrows()):
    ok = bool(r.stable)
    # baseline above, prospective below. The pair reads from the old figure to the new
    ax.barh(yi + BH / 2 + GAPB, r.horizon_baseline, height=BH, zorder=3, lw=0.7,
            color=BASE_FACE, edgecolor=DEST_C, hatch=None if ok else "///")
    ax.barh(yi - BH / 2 - GAPB, r.horizon_prospective, height=BH, zorder=3, lw=0.7,
            color=DEST_C if ok else BASE_FACE, edgecolor="white" if ok else DEST_C,
            hatch=None if ok else "///")
    mark = "" if ok else " " + chr(8225)
    ax.annotate(f"{r.horizon_baseline:,.1f}{mark}",
                xy=(r.horizon_baseline, yi + BH / 2 + GAPB), xytext=(3, 1.6),
                textcoords="offset points", va="center", fontsize=TINY, color=MUTED)
    ax.annotate(f"{r.horizon_prospective:,.1f}{mark}",
                xy=(r.horizon_prospective, yi - BH / 2 - GAPB), xytext=(3, -1.6),
                textcoords="offset points", va="center", fontsize=TINY,
                color=INK if ok else MUTED)
ax.axvline(2.0, color=ROLE["import_shock"], lw=1.0, ls=(0, (3, 2)), zorder=4)
ax.set_ylim(-0.78, len(D2) + 0.95)
ax.set_yticks(y)
ax.set_yticklabels(list(D2.display_name), fontsize=TINY)
ax.set_xlabel(AX["horizon"], fontsize=LABEL)
ax.set_xlim(0, 78)
light_grid(ax, axis="x")

pos = ax.get_position()
ax.set_position([pos.x0, pos.y0 + fr(0.600), pos.width, pos.height - fr(0.600)])
ax.legend(handles=[
    Patch(facecolor=BASE_FACE, edgecolor=DEST_C, lw=0.7,
          label="From the pre-disruption level"),
    Patch(facecolor=DEST_C, edgecolor="white", lw=0.7,
          label="From the end-of-period level"),
    Line2D([], [], color=ROLE["import_shock"], lw=1.0, ls=(0, (3, 2)),
           label="Length of the disruption period"),
    Patch(facecolor=BASE_FACE, edgecolor=DEST_C, lw=0.7, hatch="///",
          label="\u2021 Withdrawal below 1%\nof refinery intake")],
    loc="upper left", bbox_to_anchor=(0.0, -0.20), fontsize=TICK,
    frameon=False, labelspacing=0.32,
          handletextpad=0.45, handlelength=1.6, borderaxespad=0.0, borderpad=0.0)
panel_title(ax, "c", "Time to depletion")

save(fig, "Fig5_inventory_buffering")
M.to_csv(SD_WRITE / "fig5_source_data.csv", index=False)
print(f"  Fig5: cover rho={rho:+.4f} n={len(K)}; depletion n={len(D2)}; "
      f"baseline {D2.horizon_baseline.min():.1f}-{D2.horizon_baseline.max():.1f} mo, "
      f"prospective {D2.horizon_prospective.min():.1f}-{D2.horizon_prospective.max():.1f} mo; "
      f"unstable {sorted(D2.country_iso2[~D2.stable])}")

G = P.merge(DEP, on="country_iso2", how="inner")
G["d_avail_norm"] = G.d_availability_kbd / G.baseline_availability_kbd
G["stock_norm"] = G.d_stockdraw_kbd / G.baseline_availability_kbd
G.to_csv(SD_WRITE / "fig6_source_data.csv", index=False)
print(f"  corridor panel: n={len(G)} matched to dependence, written for FigS5")
