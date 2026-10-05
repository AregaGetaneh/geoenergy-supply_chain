"""FIGURE 3. Shock-absorption accounting, with the concentration of the response made visible.

a  Per-economy adjustment margins, as multiples of each economy's own crude-import shortfall.
   The United States occupies a separate strip on its own scale: its margins run from -3.3 to
   +5.6 shortfalls. Sharing one axis with the destination systems would squeeze all seven
   of them into a band a few percent of the axis wide.
b  Aggregate bridge for the primary population of destination systems.
c  Aggregate bridge for contracting net importers, which adds the one crude-exporting importer
   system that meets the contraction rule, the United States.
   Placing b and c side by side is the influence analysis: the sign of the aggregate depends
   on whether structurally distinct trading systems are pooled with destinations.

The inventory bar of each bridge is divided into its per-economy contributions. That is the
concentration of the inventory response, drawn on the quantity it describes: the destination
bar breaks into seven comparable slices, the pooled bar into one slice and a remainder. An
earlier second figure carried the same claim as a cumulative-share curve and a pair of bars
repeating two totals already drawn in this figure, and has been withdrawn.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MultipleLocator

import os as _os, sys as _sys
_CODE = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_CODE] + [_os.path.join(_CODE, _d) for _d in sorted(_os.listdir(_CODE))
                           if _os.path.isdir(_os.path.join(_CODE, _d))]

from figstyle import (HEAD, INK, MUTED, NAME, ROLE, RULE, SMALL, TICK, TINY, W2, apply_style,
                      panel_title,
                       figure_legend, light_grid, off_scale_panel, save, SD,
                       AX, TERM, TICK, LABEL, TITLE, axis_label, panel_title, country_labels, thin_leader, SD_WRITE)

apply_style()

MARGINS = (("s_stock", "inventory"), ("s_prod", "production"), ("s_exp", "export"))
ROLE_LABEL = {"import_shock": "Import shortage $L$",
              "inventory": "Inventory adjustment $b^{S}$",
              "production": "Domestic production $b^{P}$",
              "export": "Export adjustment $b^{X}$",
              "availability": "Change in refinery supply $\\Delta A$"}
# the same names broken over two lines, for the category axis where width is tight
AXIS_LABEL = {"import_shock": "Import\nshortage\n$L$",
              "inventory": "Inventory\nadjustment\n$b^{S}$",
              "production": "Domestic\nproduction\n$b^{P}$",
              "export": "Export\nadjustment\n$b^{X}$",
              "availability": "Refinery\nsupply\n$\\Delta A$"}
# The table this figure replaces, for the assertion below. Nothing here is plotted.
TABLE = {7: {"import_shock": (386, 1.000), "inventory": (232, 0.602),
             "production": (7, 0.019), "export": (25, 0.065),
             "availability": (-121, -0.314)},
         8: {"import_shock": (780, 1.000), "inventory": (2062, 2.645),
             "production": (397, 0.509), "export": (-1290, -1.654),
             "availability": (390, 0.500)}}
# the true minus sign, not a hyphen, wherever a signed number is set
THOUSANDS = FuncFormatter(lambda v, _: f"{v:,.0f}".replace("-", chr(8722)))

P = pd.read_csv(SD / "primary_panel.csv")
# ADMISSIBLE_ONLY: physically valid destinations plus crude-exporting importers
P = P[P.population.isin(["primary", "crude-exporting importer"])].copy()
P["loss"] = -P.d_imports_kbd
C = P[P.import_retention.notna() & (P.import_retention < 0.95)].copy()
for col, m in (("s_stock", "d_stockdraw_kbd"), ("s_prod", "d_production_kbd")):
    C[col] = C[m] / C.loss
C["s_exp"] = -C.d_exports_kbd / C.loss
C = C.sort_values(["population", "import_retention"], ascending=[True, True])
DEST = C[C.population == "primary"]
XPORT = C[C.population == "crude-exporting importer"]

def margin_limits(sub, pad=0.11):
    """Limits taken from the data. No bar can ever be silently clipped.

    The upper bound is held at 1.0 or above so the fully-offset reference line stays on
    the axis even for a group whose margins never reach its own shortfall.
    """
    cols = [c for c, _ in MARGINS]
    lo = min(sub[cols].clip(upper=0).sum(axis=1).min(), 0.0)
    hi = max(sub[cols].clip(lower=0).sum(axis=1).max(), 1.0)
    return lo - pad * (hi - lo), hi + pad * (hi - lo)

def margin_rows(ax, sub, bar_h, tick_step):
    """One row per economy: the signed margins stacked out from zero, then the net offset.

    Positives stack rightward and negatives leftward, both at full saturation. Position
    relative to zero carries the sign, not a lighter tint.
    """
    y = np.arange(len(sub))[::-1]
    for yi, (_, r) in zip(y, sub.iterrows()):
        right = left = 0.0
        for col, role in MARGINS:
            v = float(getattr(r, col))
            if not np.isfinite(v) or abs(v) < 1e-9:
                continue
            base = right if v > 0 else left
            ax.barh(yi, v, left=base, height=bar_h, color=ROLE[role], edgecolor="white",
                    lw=0.5, zorder=3)
            right, left = (right + v, left) if v > 0 else (right, left + v)
        ax.plot([right + left], [yi], marker="D", ms=4.4, mfc="white", mec=INK, mew=0.8,
                zorder=6)
    ax.axvline(0, color=INK, lw=0.9, zorder=4)
    ax.axvline(1, color=INK, lw=1.0, ls=(0, (3, 2)), zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels([NAME.get(a, a) for a in sub.country_iso2], fontsize=SMALL)
    ax.set_xlim(*margin_limits(sub))
    ax.xaxis.set_major_locator(MultipleLocator(tick_step))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    light_grid(ax, axis="x")
    return y

def step_label(ax, x, base, v, color, span, size=TINY):
    """Value labels go inside the bar when it is deep enough to hold one, otherwise just
    beyond its outer end.

    Labelling every step outside puts the largest bars' labels against the zero line, where
    dark text lands on a dark fill and stops being readable at print size.
    """
    if abs(v) >= 0.16 * span:
        ax.annotate(f"{v:+,.0f}".replace("-", chr(8722)), xy=(x, base + v / 2), ha="center", va="center",
                    fontsize=size, color="white", fontweight="bold", zorder=6)
    else:
        ax.annotate(f"{v:+,.0f}".replace("-", chr(8722)), xy=(x, base + v),
                    xytext=(0, 5 if v > 0 else -5),
                    textcoords="offset points", ha="center",
                    va="bottom" if v > 0 else "top", fontsize=size, color=color,
                    fontweight="bold", zorder=6,
                    path_effects=[pe.withStroke(linewidth=2.0, foreground="white")])

def slice_inventory(ax, x, base, sub, width=0.62):
    """Draw the inventory step as its per-economy contributions instead of one block.

    This is the concentration of the response, drawn on the quantity it describes, not as a separate cumulative-share curve. Slices run largest first from the base of
    the step. Therefore the top contributor's share is read directly against the whole bar, and
    the count of visible slices is itself the distribution.
    """
    d = sub[sub.d_stockdraw_kbd > 0].sort_values("d_stockdraw_kbd", ascending=False)
    total = float(d.d_stockdraw_kbd.sum())
    run = base
    for _, r in d.iterrows():
        ax.bar(x, r.d_stockdraw_kbd, bottom=run, width=width, color=ROLE["inventory"],
               edgecolor="white", lw=0.55, zorder=3)
        run += r.d_stockdraw_kbd
    top = d.iloc[0]
    share = 100 * top.d_stockdraw_kbd / total
    ymid = base + top.d_stockdraw_kbd / 2
    ax.annotate(f"{NAME.get(top.country_iso2, top.country_iso2)}, {share:.0f}%",
                xy=(x + width / 2, ymid),
                xytext=(x + width / 2 + 0.20, base + 0.94 * total),
                ha="left", va="center", fontsize=TINY, color=INK, zorder=8,
                arrowprops=dict(arrowstyle="-", lw=0.5, color=MUTED, shrinkA=0.5,
                                shrinkB=2.8),
                path_effects=[pe.withStroke(linewidth=2.4, foreground="white")])
    return len(d)

def components(sub):
    """The five quantities of the table, read from the panel and not hard-coded.

    L is the shortfall as a positive loss. Each margin is positive when it reduces that loss.
    Delta A is a signed change. That is the sign convention of the table.
    """
    t = sub[["d_imports_kbd", "d_stockdraw_kbd", "d_production_kbd", "d_exports_kbd",
             "d_availability_kbd"]].sum()
    return [("import_shock", -float(t.d_imports_kbd)),
            ("inventory", float(t.d_stockdraw_kbd)),
            ("production", float(t.d_production_kbd)),
            ("export", -float(t.d_exports_kbd)),
            ("availability", float(t.d_availability_kbd))]

def assert_against_table(sub, comps):
    """Stop before drawing if the data has moved off the table this figure replaces."""
    want = TABLE[len(sub)]
    L = comps[0][1]
    for role, v in comps:
        kb, mult = want[role]
        if round(v) != kb:
            raise SystemExit("Figure 3 halted. %s is %+.4f kb/d, which rounds to %+d against the "
                             "table's %+d" % (role, v, round(v), kb))
        if abs(round(v / L, 3) - mult) > 5e-4:
            raise SystemExit("Figure 3 halted. %s is %.6f of L, which rounds to %.3f against the "
                             "table's %.3f" % (role, v / L, round(v / L, 3), mult))
    return L

def value_label(ax, x, v, L, role, span, is_shortfall):
    """Both of the table's values for one bar, on two short lines, clear of the bar end.

    Every label sits outside its bar. Putting large ones inside and small ones outside, which is
    what this figure did while it was a waterfall, gives two reading conventions in one panel and
    collides on the small bars of panel b at +7 and +25.
    """
    kb = f"{v:+,.0f} kb/d".replace("-", chr(8722))
    mult = ("1.000 L" if is_shortfall
            else f"{v / L:+.3f} L".replace("-", chr(8722)))
    ax.annotate(kb + "\n" + mult, xy=(x, v), xytext=(0, 4 if v > 0 else -4),
                textcoords="offset points", ha="center",
                va="bottom" if v > 0 else "top", fontsize=TINY, color=INK,
                linespacing=1.25, zorder=7,
                path_effects=[pe.withStroke(linewidth=2.2, foreground="white")])

def bridge(ax, sub, title, tag):
    """Five component bars, a kb/d axis on the left and a multiple-of-L axis on the right."""
    comps = components(sub)
    L = assert_against_table(sub, comps)

    lo = min(0.0, min(v for _, v in comps))
    hi = max(0.0, max(v for _, v in comps))
    pad = 0.30 * (hi - lo)
    ylim = (lo - pad, hi + pad)
    span = ylim[1] - ylim[0]

    for i, (role, v) in enumerate(comps):
        if role == "inventory":
            slice_inventory(ax, i, 0.0, sub)
        else:
            ax.bar(i, v, width=0.62, color=ROLE[role], edgecolor="white", lw=0.5, zorder=3)
        value_label(ax, i, v, L, role, span, is_shortfall=(role == "import_shock"))

    # where the table has its midrule, between the components and the quantity they produce
    ax.axvline(len(comps) - 1.5, color=RULE, lw=0.7, ls=(0, (2, 2)), zorder=1)
    ax.axhline(0, color=INK, lw=0.8, zorder=4)
    ax.set_xticks(range(len(comps)))
    ax.set_xticklabels([AXIS_LABEL[r] for r, _ in comps], fontsize=TINY, linespacing=1.15)
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.62, len(comps) - 0.38)
    ax.set_ylim(*ylim)
    ax.yaxis.set_major_formatter(THOUSANDS)
    ax.set_ylabel(AX["volume"], fontsize=LABEL)
    light_grid(ax)

    # the same bars on the scale the table's second column uses
    sec = ax.secondary_yaxis("right", functions=(lambda y: y / L, lambda m: m * L))
    sec.set_ylabel(AX["mult_L"], fontsize=LABEL)
    sec.tick_params(labelsize=TINY)

    panel_title(ax, tag, title)
    return comps[-1][1]

fig = plt.figure(figsize=(W2, 5.35))
gs = fig.add_gridspec(2, 2, width_ratios=[1.00, 1.06], left=0.098, right=0.932,
                      top=0.928, bottom=0.148, wspace=0.30, hspace=0.46)

gsa = gs[:, 0].subgridspec(2, 1, height_ratios=[7.0, 1.9], hspace=0.30)

ax = fig.add_subplot(gsa[0])
margin_rows(ax, DEST, bar_h=0.62, tick_step=0.25)
ax.set_ylim(-0.62, len(DEST) + 0.05)

panel_title(ax, "a", "Sources of adjustment")

ax = fig.add_subplot(gsa[1])
margin_rows(ax, XPORT, bar_h=0.46, tick_step=2.0)
ax.set_ylim(-0.72, 0.72)
ax.set_xlabel(AX["mult_L"], fontsize=LABEL)
tk = [t for t in ax.get_xticks() if ax.get_xlim()[0] <= t <= ax.get_xlim()[1]]
off_scale_panel(ax, f"Separate scale, {tk[0]:+.0f} to {tk[-1]:+.0f}".replace("-", "−"),
                hatch="//", edge="#CFC7B6", label_loc="right")
ax.set_title("Crude-exporting importer", fontsize=TICK,
             loc="left", pad=3.0, color=MUTED)

# --- b, c. the two aggregates -----------------------------------------------
axb = fig.add_subplot(gs[0, 1])
rb = bridge(axb, DEST, f"Countries with an import reduction ($n={len(DEST)}$)", "b")

rc = bridge(fig.add_subplot(gs[1, 1]), C,
            f"Net importers with an import reduction ($n={len(C)}$)", "c")

# one legend for the whole figure: the same five role colors run through all three panels
figure_legend(fig, [Patch(facecolor=ROLE[r], label=ROLE_LABEL[r])
                    for r in ("import_shock", "inventory", "production", "export",
                              "availability")]
                   + [Line2D([], [], marker="D", ms=4.4, mfc="white", mec=INK, ls="none",
                             label="Net offset"),
                      Line2D([], [], color=INK, lw=1.0, ls=(0, (3, 2)),
                             label="Import shortage fully covered")],
              ncol=4, y=0.004, handlelength=1.4, columnspacing=1.4)

save(fig, "Fig3_absorption")

C.to_csv(SD_WRITE / "fig3_source_data.csv", index=False)

pd.DataFrame(
    [{"panel": tag, "population": pop, "n": len(sub), "component": ROLE_LABEL[role],
      "kbd": round(v, 4), "multiple_of_L": round(v / comps[0][1], 6)}
     for tag, pop, sub in (("b", "destination systems", DEST),
                           ("c", "contracting net importers", C))
     for comps in [components(sub)]
     for role, v in comps]
).to_csv(SD_WRITE / "fig3_components.csv", index=False)
print("  wrote sourcedata/fig3_components.csv")
print(f"  destination bridge availability {rb:+,.1f} kb/d;  contracting net importers {rc:+,.1f}")
