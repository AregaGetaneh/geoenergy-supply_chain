"""Supplementary figures and tables, generated from the same source data as the main text."""

from __future__ import annotations

import os

from pathlib import Path

import io
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.transforms import Bbox
from scipy import stats

import os as _os, sys as _sys
_CODE = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_CODE] + [_os.path.join(_CODE, _d) for _d in sorted(_os.listdir(_CODE))
                           if _os.path.isdir(_os.path.join(_CODE, _d))]

from tre_style import (HEAD, HUB_MARKER, INK, MUTED, NAME, ROLE, RULE, SMALL,
                       STATUS_COLOR, TINY, W2, apply_style, figure_legend, light_grid,
                       off_scale_panel, save, SD)
from figstyle import (LABEL, TICK, TITLE, axis_label, panel_title,
                      plain_log_ticks)

apply_style()
ROOT = Path(__file__).resolve().parents[3]
SIF = ROOT / "paper_TRE" / "si" / "figures"
SIT = ROOT / "paper_TRE" / "si" / "tables"
FIGURES_ONLY = bool(os.environ.get("TRE_FIG_OUT"))
if not FIGURES_ONLY:
    SIF.mkdir(parents=True, exist_ok=True)
    SIT.mkdir(parents=True, exist_ok=True)

D = pd.read_csv(SD / "primary_panel.csv")
PRIM = set(D.country_iso2[D.population == "primary"])
HUBS = set(D.country_iso2[D.population == "high-throughput"])
ST = pd.read_csv(SD / "country_analytical_status.csv")
FC = pd.read_csv(SD / "final_classification.csv")
FCS = FC.set_index("country_iso2")
PANEL = sorted(FC.country_iso2[FC.status == "accounting panel"])
DESTS = sorted(FC.country_iso2[FC["class"] == "destination system"])
EXPIMP = sorted(FC.country_iso2[FC["class"] == "crude-exporting importer"])
PL = pd.read_csv(SD / "placebo_windows.csv")
DEP = pd.read_csv(SD / "depletion_prospective.csv")
DEP = DEP.rename(columns={"acute_draw_kbd": "draw_kbd",
                          "draw_share_of_intake": "draw_share"})
DEP["display_name"] = [NAME.get(a, a) for a in DEP.country_iso2]
DEP = DEP.sort_values("display_name")
COV = pd.read_csv(SD / "baseline_days_of_cover.csv").rename(
    columns={"REF_AREA": "country_iso2"})
NET_EXPORTERS = {"SA", "KW", "NG", "DZ", "NO", "VE"}
SAMPLE = set(D.country_iso2)

def num(v, places=3):
    """A signed number with a typographic minus, never a hyphen."""
    return f"{v:+.{places}f}".replace("-", chr(8722))

def wt(name, body):
    if FIGURES_ONLY:
        print(f"  table {name} skipped, TRE_FIG_OUT is set")
        return
    (SIT / name).write_text(body, encoding="utf-8")
    print(f"  table {name}")

# ---------------------------------------------------------------------------

RING_RADII = (2.0, 4.0, 7.0, 11.0, 16.0, 22.0, 29.0, 37.0)
RING_DIRS = [(1.0, 0.0), (-1.0, 0.0), (0.86, 0.51), (-0.86, 0.51), (0.86, -0.51),
             (-0.86, -0.51), (0.0, 1.0), (0.0, -1.0), (0.51, 0.86), (-0.51, 0.86),
             (0.51, -0.86), (-0.51, -0.86)]

def _overlap(a, b):
    """Overlap area of two display-space boxes, zero when they are disjoint."""
    w = min(a.x1, b.x1) - max(a.x0, b.x0)
    h = min(a.y1, b.y1) - max(a.y0, b.y0)
    return w * h if w > 0 and h > 0 else 0.0

def keepout(artist, fig):
    """Display box of an already-drawn annotation, for the label placer to avoid."""
    fig.canvas.draw()
    return artist.get_window_extent(fig.canvas.get_renderer())

def direct_labels(ax, xs, ys, labels, fontsize=TINY, color=INK, weight="normal",
                  obstacles=(), marker_pt=3.6, leader_at=6.0):
    """Place one short label per point so that nothing collides.

    Candidates run from the tightest position outwards, and a leader line is added
    only where a label had to be pushed clear of its own marker.
    """
    fig = ax.figure
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    px = fig.dpi / 72.0
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    pts = ax.transData.transform(np.column_stack([xs, ys]))

    probe = ax.text(0, 0, "", fontsize=fontsize, fontweight=weight)
    dims = []
    for s in labels:
        probe.set_text(s)
        bb = probe.get_window_extent(rend)
        dims.append((bb.width, bb.height))
    probe.remove()

    clear = marker_pt * px               # marker radius plus clearance, in device units
    # labels that merely abut read as one word. Each box carries visible slack
    gx, gy = 3.0 * px, 1.7 * px
    blocked = [Bbox.from_bounds(q[0] - clear, q[1] - clear, 2 * clear, 2 * clear)
               for q in pts]
    blocked += list(obstacles)
    frame = ax.get_window_extent()

    # crowded neighbourhoods first. Isolated points take what is left over
    order = np.argsort([-int(np.sum(np.hypot(*(pts - q).T) < 40 * px)) for q in pts])
    for i in order:
        x0, y0 = pts[i]
        w, h = dims[i]
        best, best_pen, best_r = None, None, 0.0
        for r in RING_RADII:
            for ux, uy in RING_DIRS:
                # the box is pushed out until its near edge clears the marker by r
                cx = x0 + ux * (clear + r * px) + ux * (abs(ux) * w + abs(uy) * h) / 2
                cy = y0 + uy * (clear + r * px) + uy * (abs(ux) * w + abs(uy) * h) / 2
                box = Bbox.from_bounds(cx - w / 2 - gx, cy - h / 2 - gy,
                                       w + 2 * gx, h + 2 * gy)
                # a label may sit against its own marker; that is what makes it readable
                pen = sum(_overlap(box, o) for k, o in enumerate(blocked) if k != i)
                pen += 1e5 * (box.x0 < frame.x0 or box.x1 > frame.x1
                              or box.y0 < frame.y0 or box.y1 > frame.y1)
                if best_pen is None or pen < best_pen:
                    best, best_pen, best_r = (cx - w / 2 - x0, cy - h / 2 - y0), pen, r
                if pen == 0:
                    break
            if best_pen == 0:
                break
        dx, dy = best
        ax.annotate(labels[i], xy=(xs[i], ys[i]), xytext=(dx / px, dy / px),
                    textcoords="offset points", fontsize=fontsize, color=color,
                    fontweight=weight, ha="left", va="bottom", zorder=8,
                    annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", lw=0.5, color=MUTED, alpha=0.8,
                                    shrinkA=0.0, shrinkB=1.2)
                    if best_r >= leader_at else None)
        blocked.append(Bbox.from_bounds(x0 + dx - gx, y0 + dy - gy,
                                        w + 2 * gx, h + 2 * gy))

# ---------------------------------------------------------------------------
EVD = FC[FC.status != "residual not evaluable"].copy()
EVD["pct"] = 100 * EVD.c_i
EVD = EVD.sort_values("pct", na_position="last").reset_index(drop=True)
STAT_COL = {
    "accounting panel":           (STATUS_COLOR["accounting panel"], "white",
                                   "Included"),
    "net exporter":               (STATUS_COLOR["net exporter"], "white",
                                   "Net crude exporters"),
    "closure above tolerance":    (STATUS_COLOR["closure above tolerance"], "white",
                                   "Residual above 5%"),
    "physically invalid balance": (STATUS_COLOR["physically invalid balance"], "white",
                                   "Negative balance"),
    "partial reporting":          (STATUS_COLOR["partial reporting"],
                                   STATUS_COLOR["partial reporting"],
                                   "Fewer than five months"),
}
MFC = {k: ("white" if k == "partial reporting" else v[0]) for k, v in STAT_COL.items()}
MEW = {k: (1.0 if k == "partial reporting" else 0.6) for k in STAT_COL}

fig = plt.figure(figsize=(W2, 3.75))
gsS1 = fig.add_gridspec(2, 1, height_ratios=[15.0, 1.25], hspace=0.055)
ax = fig.add_subplot(gsS1[0])
axn = fig.add_subplot(gsS1[1], sharex=ax)
FLOOR = 1e-4
xs = np.arange(len(EVD))
nodef = EVD.c_i.isna().values
defined = ~nodef
ys = EVD.pct.values
cols = [STAT_COL[t][0] for t in EVD.status]
ax.vlines(xs[defined], FLOOR, ys[defined],
          color=[c for c, d in zip(cols, defined) if d], lw=0.9, zorder=3)
for st, (col, edge, _) in STAT_COL.items():
    m = defined & (EVD.status == st).values
    if m.any():
        ax.scatter(xs[m], ys[m], s=16, facecolor=MFC[st], edgecolor=edge, lw=MEW[st],
                   zorder=4)
ax.axhline(5.0, color=INK, lw=0.8, ls=(0, (3, 2)), zorder=6)
ax.set_yscale("log")
ax.set_ylim(FLOOR * 0.55, 300)
plain_log_ticks(ax, "y", [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100])
# the quantity and its normalization on the axis, the choice of scale stated separately
ax.set_ylabel(axis_label("mean absolute residual", "% of supply"),
              fontsize=LABEL)
ax.set_xlim(-1.0, len(EVD) - 0.2)
ax.tick_params(axis="x", length=0, labelbottom=False)
light_grid(ax, axis="y")
ax.legend(handles=[Line2D([], [], marker="o", ms=4.6, mfc=MFC[k], ls="none", mec=e,
                          mew=MEW[k], label=l) for k, (c, e, l) in STAT_COL.items()]
          + [Line2D([], [], color=INK, lw=0.8, ls=(0, (3, 2)),
                    label="Accepted discrepancy, $\\tau$ = 5%"),
             Line2D([], [], marker="x", ms=5.0, ls="none", mew=1.1,
                    color=STATUS_COLOR["physically invalid balance"],
                    label="No positive supply-side balance")],
          loc="lower left", bbox_to_anchor=(-0.012, 1.004), fontsize=TICK,
          labelspacing=0.32, handlelength=1.2, handletextpad=0.5, columnspacing=1.4,
          ncol=4, frameon=False)

# the off-scale strip: no axis, no position, only the fact that the ratio does not exist
off_scale_panel(axn, "", hatch="//", edge="#CFC7B6")
axn.set_ylim(-1.0, 1.0)
axn.set_yticks([])
axn.scatter(xs[nodef], np.zeros(int(nodef.sum())), s=26, marker="x",
            c=STATUS_COLOR["physically invalid balance"], lw=1.1, zorder=5)
axn.set_xticks(xs)
axn.set_xticklabels(EVD.country_iso2, fontsize=TICK, rotation=90)
axn.set_xlabel("Country", fontsize=LABEL, labelpad=2.0)
axn.tick_params(axis="x", pad=1.5)
fig.subplots_adjust(left=0.092, right=0.995, top=0.858, bottom=0.138)
save(fig, "FigS1_closure_residuals", SIF)

# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(W2, 2.9))
yrs = sorted(PL.year.unique())
rng = np.random.default_rng(3)
for i, yr in enumerate(yrs):
    lo = PL[(PL.year == yr) & (PL.import_retention < 0.95)
            & PL.country_iso2.isin(DESTS)]
    if not len(lo):
        continue
    c = ROLE["inventory"] if yr == 2026 else "#98A4AC"
    ax.scatter(i + rng.uniform(-0.16, 0.16, len(lo)), lo.wedge, s=16, facecolor=c,
               edgecolor="white", lw=0.4, alpha=0.9, zorder=4)
    ax.plot([i - 0.30, i + 0.30], [lo.wedge.mean()] * 2, color=INK, lw=1.6, zorder=5)
    # counts sit below the axis, which lets the y-range close in on the observations
    ax.annotate(f"{int((lo.wedge > 0).sum())}/{len(lo)}", xy=(i, -0.125),
                xycoords=("data", "axes fraction"), ha="center", va="top",
                fontsize=TINY, color=c if yr == 2026 else MUTED,
                fontweight="bold" if yr == 2026 else "normal", annotation_clip=False)
ax.axhline(0, color=INK, lw=0.7, zorder=2)
ax.set_xticks(range(len(yrs)))
ax.set_xticklabels(yrs, fontsize=SMALL)
ax.set_xlim(-0.6, len(yrs) - 0.4)
ax.set_ylim(-0.27, 0.31)
ax.set_ylabel("Availability retention minus" + chr(10) + "import retention",
              fontsize=SMALL)
ax.set_xlabel("Year (January–February baseline against April–May)",
              fontsize=SMALL, labelpad=16)
light_grid(ax)
ax.text(len(yrs) - 1, 0.262, "disruption year", ha="center", va="bottom",
        fontsize=TINY, color=ROLE["inventory"], fontweight="bold")
ax.text(0.0, 1.02, "Destination systems whose crude imports fell more than 5%. Bars are "
        "annual means; counts below the axis give the number with a positive gap.",
        transform=ax.transAxes, fontsize=TINY, color=MUTED, va="bottom")
fig.subplots_adjust(left=0.093, right=0.995, top=0.912, bottom=0.235)
save(fig, "FigS2_placebo_by_year", SIF)

# ---------------------------------------------------------------------------
A = D[D.import_retention.notna() & D.country_iso2.isin(DESTS)].copy()
A["wedge"] = A.availability_retention - A.import_retention
A = A.sort_values("import_retention")
LGX = np.log10(A.import_retention.values)
LGY = np.log10(A.availability_retention.values)
sl, ic, rr, pp, se = stats.linregress(LGX, LGY)
# n = 18 leaves 16 degrees of freedom. Therefore the interval is Student t, not normal.
TCRIT = stats.t.ppf(0.975, len(A) - 2)
CI_LO, CI_HI = sl - TCRIT * se, sl + TCRIT * se
FW, FH = 5.512, 3.1
fig, ax = plt.subplots(figsize=(FW, FH))
L, R, B, T = 0.100, 0.994, 0.140, 0.968
fig.subplots_adjust(left=L, right=R, bottom=B, top=T)
xlim = (0.55, 4.75)
dec_y = np.log10(xlim[1] / xlim[0]) * ((T - B) * FH) / ((R - L) * FW)
mid_y = np.sqrt(A.availability_retention.min() * A.availability_retention.max())
ylim = (mid_y / 10 ** (dec_y / 2), mid_y * 10 ** (dec_y / 2))
ax.plot(xlim, xlim, color=INK, lw=0.9, ls=(0, (3, 2)), zorder=2)
# straight in logarithms, which is what the eye then reads off the logarithmic axes
grid = np.logspace(LGX.min(), LGX.max(), 60)
ax.plot(grid, 10 ** (ic + sl * np.log10(grid)), color=ROLE["inventory"], lw=1.3, zorder=3)
ax.scatter(A.import_retention, A.availability_retention, s=20,
           facecolor=ROLE["inventory"], edgecolor="white", lw=0.5, zorder=4)
ax.axvline(0.95, color=ROLE["import_shock"], lw=0.8, ls=(0, (2, 2)), zorder=2)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(*xlim); ax.set_ylim(*ylim)
ax.set_xticks([0.6, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0])
ax.set_yticks([0.8, 1.0, 1.2, 1.5, 2.0])
for axis in (ax.get_xaxis(), ax.get_yaxis()):
    axis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    axis.set_minor_formatter(mpl.ticker.NullFormatter())
ax.set_xlabel("Crude-import retention", fontsize=SMALL)
ax.set_ylabel("Domestic-availability retention", fontsize=SMALL)
ax.set_aspect("equal", adjustable="box", anchor="C")
light_grid(ax, axis="both")
fixed = [
    ax.annotate("equal retention", xy=(1.75, 1.75), xytext=(-3, 4),
                textcoords="offset points", rotation=45, rotation_mode="anchor",
                fontsize=TINY, color=INK, ha="right", va="bottom"),
    ax.annotate("contraction rule, 0.95", xy=(0.95, ylim[1]), xytext=(-4, -4),
                textcoords="offset points", ha="right", va="top", fontsize=TINY,
                color=ROLE["import_shock"]),
    ax.text(0.985, 0.035,
            "slope in logarithms " + num(sl, 3) + f" (s.e. {se:.3f})" + chr(10) +
            "95% CI " + num(CI_LO, 3) + " to " + num(CI_HI, 3) +
            f", $t$, {len(A) - 2} d.f." + chr(10) +
            "$r$ = " + num(rr, 3) + f", n = {len(A)}",
            transform=ax.transAxes, fontsize=TINY,
            color=MUTED, va="bottom", ha="right", linespacing=1.45)]
direct_labels(ax, A.import_retention.values, A.availability_retention.values,
              list(A.country_iso2), fontsize=TINY, color=MUTED,
              obstacles=[keepout(a, fig) for a in fixed])
save(fig, "FigS3_reversion_diagnostic", SIF)

# ---------------------------------------------------------------------------
C = D[D.country_iso2.isin(PANEL) & (D.import_retention < 0.95)]\
    .sort_values("import_retention")
MARGINS = [("d_stockdraw_kbd", ROLE["inventory"], "inventory adjustment", 1),
           ("d_production_kbd", ROLE["production"], "domestic production", 1),
           ("d_exports_kbd", ROLE["export"], "crude exports retained", -1)]
fig, axes = plt.subplots(2, 4, figsize=(W2, 3.52))
for ax, (_, r) in zip(axes.ravel(), C.iterrows()):
    vals = [sgn * r[col] + 0.0 for col, _, _, sgn in MARGINS]   # +0.0 clears -0.0
    ax.bar([0, 1, 2], vals, color=[c for _, c, _, _ in MARGINS], edgecolor="white",
           lw=0.4, width=0.66, zorder=3)
    ax.axhline(0, color=INK, lw=0.7, zorder=2)
    ax.axhline(-r.d_imports_kbd, color=ROLE["import_shock"], lw=1.0, ls=(0, (3, 2)),
               zorder=4)
    hi = max(max(vals), -r.d_imports_kbd, 0.0)
    lo = min(min(vals), 0.0)
    span = hi - lo
    ax.set_ylim(lo - (0.22 if lo < 0 else 0.08) * span, hi + 0.24 * span)
    for xi, v in zip([0, 1, 2], vals):
        # a near-zero margin draws no visible bar. Therefore every value is written out
        lab = (f"{v:,.1f}" if abs(v) < 100 else f"{v:,.0f}").replace("-", chr(8722))
        ax.annotate(lab, xy=(xi, v),
                    xytext=(0, 2.5 if v >= 0 else -2.5), textcoords="offset points",
                    ha="center", va="bottom" if v >= 0 else "top", fontsize=TINY,
                    color=INK, zorder=5)
    ax.set_title(NAME.get(r.country_iso2, r.country_iso2), fontsize=SMALL, pad=10.5)
    ax.annotate(f"import retention {r.import_retention:.2f}", xy=(0.5, 1.015),
                xycoords="axes fraction", ha="center", va="bottom", fontsize=TINY,
                color=MUTED)
    ax.set_xticks([])
    ax.set_xlim(-0.80, 2.80)
    ax.tick_params(axis="y", labelsize=TINY)
    light_grid(ax)
fig.text(0.010, 0.520, "Change from baseline (kb/d)", rotation=90, va="center",
         ha="left", fontsize=SMALL)
fig.text(0.5, 0.988, "Note the varying vertical scales: every panel is scaled to its "
         "own economy.", ha="center", va="top", fontsize=SMALL, color=INK)
figure_legend(fig, [Patch(facecolor=c, edgecolor="white", lw=0.4, label=l)
                    for _, c, l, _ in MARGINS]
                   + [Line2D([], [], color=ROLE["import_shock"], lw=1.0, ls=(0, (3, 2)),
                             label="crude-import shortfall to offset")],
              ncol=4, y=-0.006, handlelength=1.6, handletextpad=0.6, columnspacing=1.9)
fig.subplots_adjust(left=0.076, right=0.988, top=0.862, bottom=0.108, hspace=0.66,
                    wspace=0.42)
save(fig, "FigS4_margins_by_economy", SIF)

# ---------------------------------------------------------------------------
G = pd.read_csv(SD / "fig6_source_data.csv")
DEST_C = ROLE["inventory"]
HUB_C = HUB_MARKER         # crude-exporting importers, as in the main figures
fig = plt.figure(figsize=(W2, 2.92))
gs = fig.add_gridspec(1, 3, wspace=0.30, left=0.078, right=0.992, top=0.878,
                      bottom=0.235)
XLAB = ("Dependence on the Strait of Hormuz" + chr(10) + "(share of crude imports)")
SPEC = [("import_retention", axis_label("import ratio", "disruption / pre-disruption"),
         "a", "Import ratio", 1.0, (0.66, 1.36)),
        ("d_avail_norm", axis_label("change in refinery supply", "share of pre-disruption level"),
         "b", "Change in refinery supply", 0.0, (-0.12, 0.38)),
        ("stock_norm", axis_label("inventory adjustment", "share of pre-disruption level"),
         "c", "Inventory adjustment", 0.0, (-0.17, 0.25))]
for i, (col, ylab, tag, title, ref, ylim) in enumerate(SPEC):
    ax = fig.add_subplot(gs[0, i])
    g = G[G[col].notna()]
    hub = (g.population == "crude-exporting importer").values
    ax.hlines(g[col], g.dep_lower, g.dep_upper, color=MUTED, lw=0.9, alpha=0.7, zorder=2)
    for e in ("dep_lower", "dep_upper"):
        ax.plot(g[e], g[col], marker="|", ms=4.2, mew=0.9, color=MUTED, alpha=0.7,
                ls="none", zorder=2)
    ax.axhline(ref, color=INK, lw=0.8, ls=(0, (3, 2)), zorder=1)
    for msk, fc, ec in [(~hub, DEST_C, "white"), (hub, HUB_C, INK)]:
        ax.scatter(g.dependence[msk], g[col][msk], s=32, facecolor=fc, edgecolor=ec,
                   lw=0.7, zorder=4)
    ax.set_xlabel(XLAB, fontsize=LABEL, linespacing=1.25)
    ax.set_ylabel(ylab, fontsize=LABEL)
    ax.set_xlim(-0.055, g.dependence.max() * 1.22)
    ax.set_ylim(*ylim)
    ax.tick_params(labelsize=TICK)
    light_grid(ax, axis="both")
    r = stats.spearmanr(g.dependence, g[col])
    stat = ax.text(0.975, 0.96, (f"$\\rho$ = {r.statistic:+.2f}" + chr(10) +
                                 f"$p$ = {r.pvalue:.2f}, $n$ = {len(g)}")
                   .replace("-", chr(8722)), transform=ax.transAxes,
                   ha="right", va="top", fontsize=TINY, color=INK, linespacing=1.4)
    panel_title(ax, tag, title)

    inside = ((g[col] >= ylim[0]) & (g[col] <= ylim[1])).values
    fr = ax.get_window_extent()
    y_ref = ax.transData.transform((0, ref))[1]
    blocks = [keepout(stat, fig),
              Bbox.from_bounds(fr.x0, y_ref - 2.0, fr.width, 4.0)]
    for _, rr in g[~inside].iterrows():          # values outside the display range
        edge = ylim[1] if rr[col] > ylim[1] else ylim[0]
        ax.plot(rr.dependence, edge, marker="^" if rr[col] > ylim[1] else "v", ms=6,
                mfc="white", mec=INK, mew=0.9, clip_on=False, zorder=6)
        flag = ax.annotate(f"{rr.country_iso2}",
                           xy=(rr.dependence, edge),
                           xytext=(7, -9 if rr[col] > ylim[1] else 4),
                           textcoords="offset points", fontsize=TINY, color=INK,
                           fontweight="bold", zorder=6)
        blocks.append(keepout(flag, fig))
    direct_labels(ax, g.dependence[inside].values, g[col][inside].values,
                  list(g.country_iso2[inside]), fontsize=TICK, color=INK,
                  weight="bold", obstacles=blocks, marker_pt=5.0, leader_at=0.0)
figure_legend(fig, [
    Line2D([], [], marker="o", ms=5, mfc=DEST_C, mec="white", ls="none",
           label="Importing country"),
    Line2D([], [], marker="o", ms=5, mfc=HUB_C, mec=INK, mew=0.7, ls="none",
           label="Crude-exporting importer"),
    Line2D([], [], color=MUTED, lw=0.9, alpha=0.7, marker="|", ms=4.2, mew=0.9,
           label="Bounds of dependence"),
    Line2D([], [], marker="^", ms=5.5, mfc="white", mec=INK, mew=0.9, ls="none",
           label="Off scale")], ncol=4, y=0.006, fontsize=TICK)
save(fig, "FigS5_corridor_dependence", SIF)

LAB1 = {"accounting panel": "accounting panel", "net exporter": "net exporter",
        "closure above tolerance": "closure above tolerance",
        "physically invalid balance": "physically invalid balance",
        "partial reporting": "partial reporting"}
def resid_cell(a):
    """Normalised residual, or n.a. where no month has a positive supply-side balance."""
    v = FCS.loc[a, 'c_i']
    return f'{100 * v:.2f}' if np.isfinite(v) else '\\emph{n.a.}'

wt("tabS1_closure_all.tex", "\n".join([
    "\\begin{tabular}{@{}lL{0.14\\linewidth}L{0.22\\linewidth}l@{}}", "\\toprule",
    "Economy & Physically valid months & Mean absolute residual "
    "(percent of supply-side balance) & Analytical status \\\\", "\\midrule",
    *[f"{a} & {int(FCS.loc[a, 'n_valid'])} & {resid_cell(a)} & "
      f"{LAB1.get(FCS.loc[a, 'status'], FCS.loc[a, 'status'])} \\\\"
      for a in EVD.country_iso2],
    "\\bottomrule", "\\end{tabular}"]))

CLs = pd.read_csv(SD / "identity_closure_test.csv")
CLs["valid"] = CLs.A_supply_kbd > 0
CLs["ratio"] = (CLs.residual_kbd.abs() / CLs.A_supply_kbd).where(CLs["valid"])
Gs = CLs.groupby("country_iso2").agg(n=("valid", "sum"),
                                     c=("ratio", lambda s: s.mean(skipna=True)))
TXs = pd.read_csv(SD / "pre_event_taxonomy.csv").set_index("country_iso2")
rows2 = []
for thr in (0.03, 0.04, 0.05, 0.06, 0.07, 0.10, 0.15, 0.20):
    adm = set(Gs.index[(Gs.n == 5) & (Gs.c <= thr)])
    panel = {a for a in adm if a in TXs.index and TXs.loc[a, "net_imports"] > 0}
    dest = {a for a in panel
            if not (TXs.loc[a, "chi_gross_export"] >= 0.10)}
    d = A[A.country_iso2.isin(dest)]
    c = d[d.import_retention.notna() & (d.import_retention < 0.95)]
    k = int((c.wedge > 0).sum()) if len(c) else 0
    rows2.append(f"{100 * thr:.0f} & {len(adm)} & {len(panel)} & {len(dest)} & {len(c)} & "
                 f"{k} & {c.import_retention.mean():.4f} & {c.availability_retention.mean():.4f} "
                 + "\\\\")
wt("tabS2_threshold.tex", "\n".join([
    "\\begin{tabular}{@{}L{0.11\\linewidth}*{5}{L{0.095\\linewidth}}*{2}{L{0.115\\linewidth}}@{}}", "\\toprule",
    "Closure tolerance (percent) & Admitted economies & Accounting panel & "
    "Destination systems & Contracting destination systems & Gap positive & "
    "Mean crude-import retention & Mean availability retention \\\\", "\\midrule", *rows2,
    "\\bottomrule", "\\end{tabular}"]))

by = []
for yr in sorted(PL.year.unique()):
    g2 = PL[(PL.year == yr) & PL.country_iso2.isin(DESTS)]
    lo = g2[g2.import_retention < 0.95]
    sl2, _, r2, _, se2 = stats.linregress(g2.import_retention, g2.availability_retention)
    by.append(f"{yr} & {len(g2)} & {len(lo)} & {int((lo.wedge > 0).sum())} & "
              f"{lo.wedge.mean():+.4f} " + "\\\\")
wt("tabS3_placebo.tex", "\n".join([
    "\\begin{tabular}{@{}rL{0.19\\linewidth}L{0.21\\linewidth}L{0.12\\linewidth}L{0.16\\linewidth}@{}}", "\\toprule",
    "Year & Destination systems reporting & Contracting destination systems & "
    "Gap positive & Mean availability gap \\\\",
    "\\midrule", *by,
    "\\bottomrule", "\\end{tabular}"]))

def flag(ok):
    """Horizon usability label for the depletion table."""
    return "stable" if ok else "unstable"

wt("tabS4_depletion.tex", "\n".join([
    "\\begin{tabular}{@{}lL{0.15\\linewidth}L{0.12\\linewidth}L{0.12\\linewidth}"
    "L{0.12\\linewidth}L{0.12\\linewidth}l@{}}", "\\toprule",
    "Economy & Reported baseline closing stock (thousand barrels) & "
    "Acute-window draw rate (kb/d) & Draw as share of refinery intake & "
    "Baseline-equivalent horizon (months) & Prospective horizon (months) & "
    "Horizon interpretation \\\\", "\\midrule",
    *[f"{r.display_name} & {r.CLOSTLV:,.0f} & "
      f"{r.draw_kbd:,.1f} & {r.draw_share:.3f} & "
      f"{r.horizon_baseline:,.1f} & {r.horizon_prospective:,.1f} & "
      f"{flag(r.stable)} \\\\"
      for _, r in DEP.iterrows()],
    "\\bottomrule", "\\end{tabular}"]))

ORDER5 = ["accounting panel", "net exporter", "closure above tolerance",
          "physically invalid balance", "partial reporting", "residual not evaluable"]
rows5 = []
for k in ORDER5:
    m = sorted(FC.country_iso2[FC.status == k])
    cell = ", ".join(m) if len(m) <= 26 else f"{len(m)} economies, not individually listed"
    rows5.append(f"{k} ({len(m)}) & \\footnotesize {cell} \\\\")
rows5.append("\\midrule")
for k in ("destination system", "crude-exporting importer"):
    m = sorted(FC.country_iso2[FC["class"] == k])
    rows5.append(f"\\quad {k} ({len(m)}) & \\footnotesize {', '.join(m)} \\\\")
wt("tabS5_status_full.tex", "\n".join([
    "\\begin{tabular}{@{}lL{0.62\\linewidth}@{}}", "\\toprule",
    "Status & Economies \\\\", "\\midrule", *rows5,
    "\\bottomrule", "\\end{tabular}"]))

wt("tabS6_cover.tex", "\n".join([
    "\\begin{tabular}{@{}lL{0.22\\linewidth}L{0.20\\linewidth}L{0.16\\linewidth}@{}}", "\\toprule",
    "Economy & Reported baseline closing stock (thousand barrels) & "
    "Baseline refinery intake (kb/d) & Reported stock, days of intake \\\\",
    "\\midrule",
    *[f"{NAME.get(r.country_iso2, r.country_iso2)} & {r.CLOSTLV:,.0f} & "
      f"{r.REFINOBS:,.1f} & "
      f"{'n.r.' if not r.CLOSTLV > 0 else f'{r.days_cover:,.1f}'} \\\\"
      # ordered by economy, not by level: see the note on Table S4
      for _, r in COV[COV.country_iso2.isin(PANEL)].assign(
          display_name=lambda t: [NAME.get(a, a) for a in t.country_iso2]
      ).sort_values("display_name").iterrows()],
    "\\bottomrule", "\\end{tabular}"]))

A2 = D[D.import_retention.notna() & D.country_iso2.isin(PANEL)].copy()
A2["wedge"] = A2.availability_retention - A2.import_retention
wt("tabS7_wedge_all.tex", "\n".join([
    "\\begin{tabular}{@{}lL{0.16\\linewidth}L{0.16\\linewidth}L{0.13\\linewidth}L{0.18\\linewidth}@{}}", "\\toprule",
    "Economy & Crude-import retention & Availability retention & "
    "Availability gap & Baseline availability (kb/d) \\\\",
    "\\midrule",
    *[f"{NAME.get(r.country_iso2, r.country_iso2)} & {r.import_retention:.4f} & "
      f"{r.availability_retention:.4f} & {r.wedge:+.4f} & "
      f"{r.baseline_availability_kbd:,.0f} \\\\"
      for _, r in A2.sort_values("wedge").iterrows()],
    "\\bottomrule", "\\end{tabular}"]))

print("\nSI complete: 5 figures, 7 tables")
