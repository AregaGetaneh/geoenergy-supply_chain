"""FIGURE 2. Coverage of the accounting framework and observed destination outcomes.

Composition rationale. Seventeen of nineteen destination systems are European. Therefore four full
world maps repeat mostly empty ocean. One global map carries analytical coverage, where the
geography genuinely spans the world, and one large Europe map carries the outcome, where the
sample actually lies. A compact ranking panel gives the same quantity for every economy,
including the two outside Europe. Nothing is displaced into a decorative tile.

Cartography. Natural Earth 1:50m boundaries and coastline, Robinson for the world and Albers
Equal Area for Europe. Countries outside the accounting sample never take a value color.
Circle area encodes baseline crude-system scale so that country area cannot be read as
economic importance.

Two grammars for the absence of a value. A country that reported crude oil in the event
window and still received no outcome is a FINDING: the accounting ladder says which rung it
failed on, and it takes a ladder color. A country that never appears in the extract for the
event window is a LIMIT OF THE DATA, not a finding about that country, and it takes a hatched
neutral surface. Conflating the two lets a reader take the emptiness of Africa or South
America for an evaluated result. The reporting universe is not asserted: it is read from
final_classification.csv, which is built on country_analytical_status.csv, and it is checked
here against the redistributed JODI crude extract. The 96 economies carrying a ladder rung are
exactly the economies with at least one crude oil observation in 2026-01 to 2026-05. Every
other Natural Earth territory is hatched, including the 22 economies that appear elsewhere in
the 2020 to 2026 extract but fall silent inside the window, and the few Natural Earth records
that carry no usable ISO code, which earlier fell through the fill loop and rendered as water.

One binned scale for panels b and c. The availability gap is cut into ten bins of 0.05
between -0.25 and 0.25, with zero as a bin limit so the sign of the gap is never inside a
bin, and with the two economies beyond each end declared by an arrow, not absorbed
into the end bin. The bin limits are printed under the key. A country color on the map
decodes to an interval without the reader having to estimate a position on a continuous ramp.
The bars in panel c take the bin colors and the bin limits are ruled across them. Therefore the map
and the ranking are read with one scale. Bin colors are sampled from RETENTION_CMAP away from
its near-white centre: the exact centre of a red-blue diverging ramp passes through the tone
of the ocean fill, which would have put a country with a small positive gap at 2 CIE76 units
from the water it sits in.

Cartographic documentation
  boundaries   Natural Earth 1:50m admin_0_countries
  coastline    Natural Earth 1:50m coastline, ocean 1:50m
  projection   Robinson, central meridian 10 E; Albers Equal Area, 14 E / 52 N, SP 40/65 N
  disputed     Natural Earth default sovereignty; no adjudication is implied
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.patheffects as pe
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon, Rectangle

import os as _os, sys as _sys
_CODE = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_CODE] + [_os.path.join(_CODE, _d) for _d in sorted(_os.listdir(_CODE))
                           if _os.path.isdir(_os.path.join(_CODE, _d))]

from figstyle import (HEAD, INK, NA_HATCH, RETENTION_CMAP, RULE, SMALL, TINY, W2,
                       apply_style, map_layers, map_linework, save, SD, NAME,
                       STATUS_COLOR, STATUS_LABEL,
                       MAP_SEA, MAP_LAND, MAP_LAND_EDGE, MAP_COAST, MAP_BORDER, MAP_GRAT,
                       AX, TERM, TICK, LABEL, TITLE, axis_label, panel_title, country_labels, thin_leader, SD_WRITE)

apply_style()
mpl.rcParams["hatch.linewidth"] = 0.32

# ---------------------------------------------------------------------------
FC = pd.read_csv(SD / "final_classification.csv")
P = pd.read_csv(SD / "primary_panel.csv")
# ADMISSIBLE_ONLY: physically valid destinations plus crude-exporting importers
P = P[P.population.isin(["primary", "crude-exporting importer"])].copy()
STATUS = dict(zip(FC.country_iso2, FC.status))
UNIVERSE = set(FC.country_iso2)          # economies that reported crude in the event window
SAMPLE = set(FC.country_iso2[FC.status == "accounting panel"])
HUBS = set(P.country_iso2[P.population == "crude-exporting importer"])
GAP = dict(zip(P.country_iso2, P.gap))
SCALE = dict(zip(P.country_iso2, P.baseline_availability_kbd))

WINDOW = ["2026-01", "2026-02", "2026-03", "2026-04", "2026-05"]
EXTRACT = SD.parent / "reproduction" / "data" / "jodi_crude_2020_2026.csv.gz"
if EXTRACT.exists():
    _x = pd.read_csv(EXTRACT, dtype=str, usecols=["REF_AREA", "TIME_PERIOD"])
    _in_window = set(_x.REF_AREA[_x.TIME_PERIOD.isin(WINDOW)])
    _ever = set(_x.REF_AREA)
    assert _in_window == UNIVERSE, sorted(_in_window ^ UNIVERSE)
    print(f"  reporting universe {len(UNIVERSE)} economies, verified against the extract; "
          f"{len(_ever - UNIVERSE)} more report crude outside the window only")
else:
    print(f"  reporting universe {len(UNIVERSE)} economies, from final_classification.csv")

STATUS_STYLE = {k: (STATUS_COLOR[k], STATUS_LABEL[k]) for k in
                ("accounting panel", "net exporter", "closure above tolerance",
                 "physically invalid balance", "partial reporting",
                 "residual not evaluable")}

BUBBLE = "#F7D97C"
SAMPLE_EDGE = "#4A565F"      # outline of a country carrying a value in panel b

NODATA_FILL = "#D8D2C7"
NODATA_LINE = "#8A8172"
NODATA_LABEL = "No crude oil reported"

PC = ccrs.PlateCarree()

# ---------------------------------------------------------------------------
CAP = 0.25
STEP = 0.05
BOUNDS = np.round(np.arange(-CAP, CAP + STEP / 2, STEP), 2) + 0.0   # 11 limits, 10 bins
NBIN = len(BOUNDS) - 1

_INNER, _OUTER = 0.12, 0.45
_POS = sorted(0.5 + s * (_INNER + (_OUTER - _INNER) * j / (NBIN / 2 - 1))
              for s in (-1, 1) for j in range(NBIN // 2))
BIN_COLOR = [RETENTION_CMAP(p) for p in _POS]
UNDER_COLOR = RETENTION_CMAP(0.0)
OVER_COLOR = RETENTION_CMAP(1.0)

def gap_color(v):
    """The bin a gap falls in. Limits are upper-closed. Therefore zero is a negative-side limit."""
    if v <= BOUNDS[0]:
        return UNDER_COLOR
    if v > BOUNDS[-1]:
        return OVER_COLOR
    return BIN_COLOR[int(np.searchsorted(BOUNDS, v, side="left")) - 1]

def fmt_limit(v):
    return f"{v:+.2f}".replace("-", "−").replace("+", "") if v else "0"

WORLD = ccrs.Robinson(central_longitude=10)
WORLD_LAT = (-55.0, 82.0)            # below and above this tthis is no sample and no ocean traffic
EUROPE_EXT = (-13.0, 49.0, 34.0, 70.0)
EUROPE = ccrs.AlbersEqualArea(central_longitude=17, central_latitude=52,
                              standard_parallels=(40, 65))

# ---------------------------------------------------------------------------
NE_COUNTRIES = shpreader.natural_earth(resolution="50m", category="cultural",
                                       name="admin_0_countries")
NE_COAST = shpreader.natural_earth(resolution="50m", category="physical",
                                   name="coastline")
NE_BORDERS = shpreader.natural_earth(resolution="50m", category="cultural",
                                     name="admin_0_boundary_lines_land")

def iso2(a):
    for k in ("ISO_A2_EH", "ISO_A2", "WB_A2"):
        v = a.get(k)
        if v and str(v) not in ("-99", "", "NA", "None"):
            return str(v)
    return None

GEOM, ANCHOR, UNCODED = {}, {}, []
for r in shpreader.Reader(NE_COUNTRIES).records():
    i = iso2(r.attributes)
    if not i:
        UNCODED.append(r.geometry)
        continue
    GEOM.setdefault(i, []).append(r.geometry)
    if i not in ANCHOR:
        g = r.geometry
        # anchor on the largest part. An overseas fragment cannot carry the marker
        main = max(g.geoms, key=lambda q: q.area) if g.geom_type == "MultiPolygon" else g
        p = main.representative_point()
        ANCHOR[i] = (p.x, p.y)

def crop(geoms, extent, pad=6.0):
    """Only geometry that can be seen. Clipped paths are still written to the PDF."""
    w, e, s, n = extent
    out = []
    for g in geoms:
        x0, y0, x1, y1 = g.bounds
        if x1 >= w - pad and x0 <= e + pad and y1 >= s - pad and y0 <= n + pad:
            out.append(g)
    return out

def generalise(geoms, tol):
    return [g.simplify(tol, preserve_topology=True) for g in geoms]

def draw_nodata(ax, geoms, tol=None, edge_lw=0.15):
    """The hatched surface, in two passes.

    Matplotlib takes the hatch color from the edge color, and the edge color on this map is
    the white hairline every country carries. One collection therefore cannot carry both. Therefore the hatch is drawn with a zero-width edge and the hairline is laid over it.
    """
    if not geoms:
        return
    g = generalise(geoms, tol) if tol else list(geoms)
    ax.add_geometries(g, PC, zorder=2, facecolor=NODATA_FILL, hatch=NA_HATCH,
                      edgecolor=NODATA_LINE, linewidth=0.0)
    ax.add_geometries(g, PC, zorder=2.05, facecolor="none", edgecolor=MAP_LAND_EDGE,
                      linewidth=edge_lw)

# ---------------------------------------------------------------------------
def projected_aspect(proj, extent, k=180):
    """Width over height of an extent once projected, with the edges densified."""
    w, e, s, n = extent
    lon, lat = np.linspace(w, e, k), np.linspace(s, n, k)
    ring = np.vstack([np.column_stack([lon, np.full(k, s)]),
                      np.column_stack([np.full(k, e), lat]),
                      np.column_stack([lon[::-1], np.full(k, n)]),
                      np.column_stack([np.full(k, w), lat[::-1]])])
    xy = proj.transform_points(PC, ring[:, 0], ring[:, 1])[:, :2]
    return xy, np.ptp(xy[:, 0]) / np.ptp(xy[:, 1])

EPS = 1e-6
WORLD_EXT = (10 - 180 + EPS, 10 + 180 - EPS) + WORLD_LAT
WORLD_RING, WORLD_ASPECT = projected_aspect(WORLD, WORLD_EXT)
_, EUROPE_ASPECT = projected_aspect(EUROPE, EUROPE_EXT)

PAD_L, PAD_R = 0.10, 0.10
W_A = 5.42                       # world map; the remaining strip carries its key
W_B = 3.42                       # Europe map
H_A, H_B = W_A / WORLD_ASPECT, W_B / EUROPE_ASPECT
TITLE_H, ROW_GAP, TOP_PAD, BOT_PAD = 0.22, 0.43, 0.06, 0.08
KEY_DROP = ROW_GAP - 0.10

# the shared key below panels b and c, measured from the foot of the figure upward
MARK_H = 0.17                    # marker row: hub outline, beyond-scale arrow, no-data
MARK_GAP = 0.07
TICKLAB_H = 0.14                 # the bin limits
STRIP_H = 0.105                  # the swatches themselves
KEYLAB_H = 0.16                  # what the scale measures
C_AXIS_H = 0.31                  # panel c keeps its own tick labels and axis title
KEY_H = MARK_H + MARK_GAP + TICKLAB_H + STRIP_H + KEYLAB_H + C_AXIS_H

FIG_H = TOP_PAD + TITLE_H + H_A + ROW_GAP + TITLE_H + H_B + KEY_H + BOT_PAD
fig = plt.figure(figsize=(W2, FIG_H))

def rect(x, y, w, h):
    """Inches from the bottom-left, which is how the layout above is reasoned about."""
    return [x / W2, y / FIG_H, w / W2, h / FIG_H]

Y_A = FIG_H - TOP_PAD - TITLE_H - H_A
Y_B = BOT_PAD + KEY_H

# ---------------------------------------------------------------------------
axa = fig.add_axes(rect(PAD_L, Y_A, W_A, H_A), projection=WORLD)
axa.set_boundary(mpath.Path(WORLD_RING), transform=axa.transData)
axa.set_xlim(WORLD_RING[:, 0].min(), WORLD_RING[:, 0].max())
axa.set_ylim(WORLD_RING[:, 1].min(), WORLD_RING[:, 1].max())
axa.patch.set_facecolor(MAP_SEA)
axa.spines["geo"].set_edgecolor(MAP_COAST)
axa.spines["geo"].set_linewidth(0.6)

WORLD_TOL = 0.10                 # ~11 km, roughly a third of a printed pixel at this width
absent = list(crop(UNCODED, (-180, 180) + WORLD_LAT, pad=2.0))
for a, gs in GEOM.items():
    vis = crop(gs, (-180, 180) + WORLD_LAT, pad=2.0)
    if not vis:
        continue
    if a in STATUS:
        axa.add_geometries(generalise(vis, WORLD_TOL), PC, zorder=2,
                           facecolor=STATUS_STYLE[STATUS[a]][0],
                           edgecolor=MAP_LAND_EDGE, linewidth=0.15)
    else:
        absent.extend(vis)
draw_nodata(axa, absent, tol=WORLD_TOL)

# under the country fills. The graticule reads on the water and never rules across land
axa.gridlines(crs=PC, draw_labels=False, color=MAP_GRAT, linewidth=0.3, alpha=0.9,
              xlocs=np.arange(-180, 181, 30), ylocs=np.arange(-60, 90, 30), zorder=1.5)
axa.add_geometries(generalise(shpreader.Reader(NE_BORDERS).geometries(), WORLD_TOL), PC,
                   facecolor="none", edgecolor=MAP_BORDER, linewidth=0.28, zorder=6)
axa.add_geometries(generalise(shpreader.Reader(NE_COAST).geometries(), WORLD_TOL), PC,
                   facecolor="none", edgecolor=MAP_COAST, linewidth=0.42, zorder=7)

SMAX = max(SCALE.values())

def bubble(v):
    return 6.0 + 190.0 * (v / SMAX)

for a, v in sorted(SCALE.items(), key=lambda kv: -kv[1]):
    if a in ANCHOR:
        axa.scatter(*ANCHOR[a], s=bubble(v), transform=PC, facecolor=BUBBLE,
                    edgecolor=INK, lw=0.55, alpha=0.94, zorder=9)

panel_title(axa, "a", "Analytical coverage and system scale")

axk = fig.add_axes(rect(PAD_L + W_A + 0.16, Y_A - KEY_DROP,
                        W2 - PAD_L - W_A - 0.16 - PAD_R, H_A + KEY_DROP))
axk.axis("off")
axk.set_xlim(0, 1)
axk.set_ylim(0, 1)

BLOCK_GAP = 0.024

def stack(rule=True, **kw):
    """Add a key block under the one above it, measured, not guessed.

    Three blocks of type whose depth depends on how the labels wrap cannot be placed from a
    table of constants: the first attempt put the hatched entry through the middle of the
    rung above it. Each block is drawn, measured, and the next one is hung below it. Therefore the
    column stays correct if a label is ever reworded.
    """
    opts = dict(loc="upper left", bbox_to_anchor=(0.0, stack.y), fontsize=TINY,
                title_fontsize=TINY, alignment="left", handlelength=1.15,
                handleheight=0.95, handletextpad=0.5, borderpad=0.0, frameon=False)
    opts.update(kw)
    leg = axk.legend(**opts)
    axk.legend_ = None
    axk.add_artist(leg)
    fig.canvas.draw()
    bb = leg.get_window_extent(fig.canvas.get_renderer())
    stack.y = bb.transformed(axk.transAxes.inverted()).y0 - BLOCK_GAP
    if rule:
        axk.plot([0.0, 0.62], [stack.y, stack.y], color=RULE, lw=0.5, clip_on=False,
                 zorder=1)
        stack.y -= BLOCK_GAP
    return leg

stack.y = 1.0
stack(handles=[Patch(facecolor=c, edgecolor=MAP_LAND_EDGE, lw=0.4, label=lab)
               for _, (c, lab) in STATUS_STYLE.items()],
      title="Analytical status", labelspacing=0.40)
stack(handles=[Patch(facecolor=NODATA_FILL, edgecolor=NODATA_LINE, lw=0.4, hatch=NA_HATCH,
                     label=NODATA_LABEL)],
      title="Data coverage", handlelength=1.7, handleheight=1.15)
SIZE_KEYS = [250, 2500, 15000]
stack(handles=[Line2D([], [], marker="o", ls="none", mfc=BUBBLE, mec=INK, mew=0.55,
                      ms=np.sqrt(bubble(v)), label=f"{v:,}")
               for v in SIZE_KEYS],
      title="Pre-disruption crude supply (kb/d)", labelspacing=0.70,
      handletextpad=0.6, rule=False)

# ---------------------------------------------------------------------------
axb = fig.add_axes(rect(PAD_L, Y_B, W_B, H_B), projection=EUROPE)
map_layers(axb, EUROPE_EXT, PC)
absent_b = list(crop(UNCODED, EUROPE_EXT))
for a, gs in GEOM.items():
    vis = crop(gs, EUROPE_EXT)
    if not vis:
        continue
    if a not in UNIVERSE:
        absent_b.extend(vis)
        continue
    known = a in SAMPLE and np.isfinite(GAP.get(a, np.nan))
    axb.add_geometries(vis, PC, zorder=3 if known else 2,
                       edgecolor=SAMPLE_EDGE if known else MAP_LAND_EDGE,
                       linewidth=0.5 if known else 0.3,
                       facecolor=gap_color(GAP[a]) if known else MAP_LAND)
draw_nodata(axb, absent_b, edge_lw=0.3)
map_linework(axb, coast_lw=0.55, border_lw=0.35)
for a in HUBS & set(GEOM):
    axb.add_geometries(crop(GEOM[a], EUROPE_EXT), PC, facecolor="none", edgecolor=INK,
                       linewidth=1.0, zorder=8)

LEADER = "#5A646C"
LEADER_MIN = 15.0                # beyond this the label needs a pointer to stay unambiguous
LABEL_NUDGE = {
    "IE": (-17, 1), "GB": (2, -4), "PT": (-11, -2), "ES": (0, -2), "DK": (-14, 7),
    "NL": (0, 16), "BE": (-16, -9), "CH": (-20, -10), "AT": (-6, -16), "HR": (-5, -17),
    "SK": (21, 4), "HU": (1, -4), "CZ": (0, 3), "DE": (-1, 2), "PL": (1, 0),
    "SE": (-2, -4), "BG": (4, -1), "GR": (-2, -9), "TR": (0, -2),
}
for a in sorted(SAMPLE & set(ANCHOR)):
    lon, lat = ANCHOR[a]
    w, e, s, n = EUROPE_EXT
    if not (w < lon < e and s < lat < n):
        continue
    dx, dy = LABEL_NUDGE.get(a, (0, 0))
    at_country = PC._as_mpl_transform(axb)
    if np.hypot(dx, dy) > LEADER_MIN:
        axb.annotate("", xy=(lon, lat), xycoords=at_country, textcoords="offset points",
                     xytext=(dx * 0.66, dy * 0.66), zorder=9,
                     arrowprops=dict(arrowstyle="-", lw=0.4, color=LEADER, shrinkA=0.0,
                                     shrinkB=0.0))
    axb.annotate(a, xy=(lon, lat), xycoords=at_country, xytext=(dx, dy),
                 textcoords="offset points", ha="center", va="center", fontsize=TINY,
                 color=INK, zorder=10,
                 # the darkest two bins swallow a thin halo, which is where HU, AT and CZ sit
                 path_effects=[pe.withStroke(linewidth=2.1, foreground="white")])

panel_title(axb, "b", "Supply-import gap, Europe", pad=3.0)

# ---------------------------------------------------------------------------
X_C = PAD_L + W_B + 0.96         # leaves room for the country names
axc = fig.add_axes(rect(X_C, Y_B, W2 - X_C - PAD_R - 0.06, H_B))
D = P[P.gap.notna()].sort_values("gap")
y = np.arange(len(D))
LIM = 0.345
axc.barh(y, D.gap.clip(-CAP, CAP), height=0.66, zorder=3, edgecolor=INK, lw=0.35,
         color=[gap_color(g) for g in D.gap])

# The gutter is where a value larger than the color scale is declared, not hidden.
axc.axvspan(-LIM, -CAP, color="#F7F5F1", zorder=0)
axc.axvspan(CAP, LIM, color="#F7F5F1", zorder=0)
X_HUB = -LIM - 0.030         # in the margin between the names and the plot area
for yi, (_, r) in zip(y, D.iterrows()):
    if r.country_iso2 in HUBS:
        axc.plot(X_HUB, yi, marker="s", ms=3.4, mfc="none", mec=INK, mew=0.9, zorder=5,
                 clip_on=False)
    if abs(r.gap) > CAP:
        side = -1.0 if r.gap < 0 else 1.0
        axc.plot(side * (CAP + 0.010), yi, marker="<" if side < 0 else ">", ms=4.4,
                 mfc="white", mec=INK, mew=0.7, zorder=6)
        axc.annotate(f"{r.gap:+.2f}".replace("-", "−"),
                     xy=(side * (CAP + 0.020), yi), xytext=(-2.0 * side, 0),
                     textcoords="offset points", ha="right" if side < 0 else "left",
                     va="center", fontsize=TINY, color=INK, fontweight="bold", zorder=7)
# the bin limits, ruled across the bars. A bar end and a map color resolve to one interval
for b in BOUNDS[1:-1]:
    axc.axvline(b, color=RULE, lw=0.35, zorder=1)
axc.axvline(0, color=INK, lw=0.8, zorder=4)
for x in (-CAP, CAP):
    axc.axvline(x, color=RULE, lw=0.5, zorder=1)
axc.set_yticks(y)
axc.set_yticklabels([NAME.get(a, a) for a in D.country_iso2], fontsize=TINY)
axc.set_xlabel(AX["gap"], fontsize=LABEL, labelpad=1.5)
axc.set_xticks([-0.25, 0.0, 0.25])
axc.set_xlim(-LIM, LIM)
axc.set_ylim(-0.75, len(D) - 0.25)
axc.tick_params(axis="y", length=0, pad=13.0)
axc.set_axisbelow(True)
for side in ("top", "right", "left"):
    axc.spines[side].set_visible(False)
panel_title(axc, "c", "Supply-import gap, accounting sample", pad=3.0)

# ---------------------------------------------------------------------------
STRIP_W = 5.20
X_KEY = (W2 - STRIP_W) / 2.0
Y_STRIP = BOT_PAD + MARK_H + MARK_GAP + TICKLAB_H
axs = fig.add_axes(rect(X_KEY, Y_STRIP, STRIP_W, STRIP_H))
axs.set_xlim(0, NBIN)
axs.set_ylim(0, 1)
axs.axis("off")
for i, c in enumerate(BIN_COLOR):
    axs.add_patch(Rectangle((i, 0), 1, 1, facecolor=c, edgecolor=INK, lw=0.35,
                            clip_on=False, zorder=3))
# beyond the scale, in the same grammar as the arrowheads in panel c
TIP = 0.22
for xy, col in (([(0, 0), (0, 1), (-TIP, 0.5)], UNDER_COLOR),
                ([(NBIN, 0), (NBIN, 1), (NBIN + TIP, 0.5)], OVER_COLOR)):
    axs.add_patch(Polygon(xy, closed=True, facecolor=col, edgecolor=INK, lw=0.35,
                          clip_on=False, zorder=3))
for i, b in enumerate(BOUNDS):
    axs.plot([i, i], [0, -0.30], color=INK, lw=0.4, clip_on=False, zorder=4)
    axs.annotate(fmt_limit(b), xy=(i, -0.42), ha="center", va="top", fontsize=TINY,
                 color=INK, annotation_clip=False)
axs.annotate(AX["gap"], xy=(NBIN / 2.0, 1.40), ha="center", va="bottom",
             fontsize=LABEL, color=INK, annotation_clip=False)

axm = fig.add_axes(rect(PAD_L, BOT_PAD, W2 - PAD_L - PAD_R, MARK_H))
axm.axis("off")
axm.legend(handles=[
    Patch(facecolor=MAP_LAND, edgecolor=MAP_LAND_EDGE, lw=0.4,
          label="Not included"),
    Patch(facecolor=NODATA_FILL, edgecolor=NODATA_LINE, lw=0.4, hatch=NA_HATCH,
          label="No crude oil reported"),
    Patch(facecolor="none", edgecolor=INK, lw=1.0,
          label="Crude-exporting importer"),
    Line2D([], [], marker="<", ls="none", ms=4.4, mfc="white", mec=INK, mew=0.7,
           label="Beyond the color scale")],
    loc="center", bbox_to_anchor=(0.5, 0.5), ncol=4, fontsize=TINY, handlelength=1.15,
    handleheight=0.95, handletextpad=0.45, columnspacing=1.5, borderpad=0.0, frameon=False)

save(fig, "Fig2_world_outcomes")
P.to_csv(SD_WRITE / "fig2_source_data.csv", index=False)
print(f"  gap range {D.gap.min():+.3f} to {D.gap.max():+.3f}; "
      f"{int((D.gap.abs() > CAP).sum())} beyond the color scale")
print("  bin limits " + ", ".join(f"{b:+.2f}" for b in BOUNDS))
