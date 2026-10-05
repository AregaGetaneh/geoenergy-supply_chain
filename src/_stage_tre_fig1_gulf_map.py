"""FIGURE 1. Geographic anatomy of the Hormuz crude supply system, and the observed
upstream export signal.

Real GIS cartography. Natural Earth 1:10m cultural and physical vectors, Lambert Conformal
Conic centred on the Gulf. No hand-drawn polygons and no stylised country shapes.

Panel a  Gulf region: coastlines, national boundaries, the Strait of Hormuz, verified crude
         terminals, and the two overland bypass corridors drawn as labelled schematic
         alignments between verified endpoints. Reporting status of each Gulf economy is
         encoded, since most of the corridor's export response is not observable.
Panel b  World locator, showing the corridor in global context.
Panel c  Reported crude exports of the three reporting Gulf economies through the window.

Cartographic documentation
  boundaries   Natural Earth 1:10m admin_0_countries
  coastline    Natural Earth 1:10m coastline, land and lakes; the sea is the ground of the
               map frame, not a drawn polygon, and every vector is clipped to a box
               larger than the visible frame
  projection   Lambert Conformal Conic, central meridian 52 E, standard parallels 22/30 N
  disputed     Natural Earth default sovereignty; no adjudication is implied
  pipelines    schematic geographic alignments between verified endpoints, NOT survey
               polylines; no capacity is asserted anywhere in this figure
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import matplotlib.patheffects as pe
import shapely.geometry as sgeom
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import os as _os, sys as _sys
_CODE = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_CODE] + [_os.path.join(_CODE, _d) for _d in sorted(_os.listdir(_CODE))
                           if _os.path.isdir(_os.path.join(_CODE, _d))]

from figstyle import (BODY, HEAD, INK, MUTED, SMALL, TINY, W2, apply_style,
                       off_scale_panel,
                       light_grid, save, SD, ROLE,
                       MAP_SEA, MAP_LAND, MAP_COAST, MAP_BORDER, MAP_GRAT,
                       AX, TERM, TICK, LABEL, TITLE, axis_label, panel_title, country_labels, thin_leader, SD_WRITE)

apply_style()

# Map lettering follows the plate, which keeps one family across all eight figures.
SANS = {}

# ---------------------------------------------------------------------------
E = pd.read_csv(SD / "gulf_crude_exports_kbd.csv", index_col=0).apply(
    pd.to_numeric, errors="coerce")
MONTHS = list(E.columns)
REPORTS = [a for a in E.index if E.loc[a].notna().any()]
GULF = list(E.index)

NAMES = {"SA": "Saudi Arabia", "KW": "Kuwait", "BH": "Bahrain", "IQ": "Iraq",
         "IR": "Iran", "AE": "United Arab\nEmirates", "QA": "Qatar", "OM": "Oman",
         "YE": "Yemen"}

# Verified geographic locations (place coordinates, not statistics).
TERMINALS = {
    "Ras Tanura":     (50.16, 26.64, "right", (-8, 9)),
    "Mina al-Ahmadi": (48.15, 29.07, "right", (-7, -7)),
    "Basra":          (48.81, 29.68, "left",  (6, 2)),
    "Kharg Island":   (50.32, 29.25, "left",  (7, -2)),
    "Ras Laffan":     (51.55, 25.90, "right", (-9, -13)),
    "Fujairah":       (56.33, 25.12, "left",  (8, -7)),
    "Yanbu":          (38.06, 24.09, "left",  (7, 3)),
}
WAYPOINTS = {"Abqaiq": (49.68, 25.93, "right", (-6, 6)),
             "Habshan": (53.60, 23.75, "right", (-8, 10))}
HORMUZ = (56.25, 26.57)

CORRIDORS = [
    ("Abqaiq–Yanbu",
     [(49.68, 25.93), (46.5, 25.5), (43.0, 25.1), (40.0, 24.4), (38.06, 24.09)]),
    ("Habshan–Fujairah",
     [(53.60, 23.75), (54.9, 24.3), (56.0, 24.8), (56.33, 25.12)]),
]

SEAS = [("Persian Gulf", 52.30, 27.45, -28),
        ("Gulf of Oman", 58.55, 24.35, -28),
        ("Red Sea", 38.35, 20.30, -52),
        ("Arabian Sea", 58.20, 18.30, 0)]

# ---------------------------------------------------------------------------
FNC = shpreader.natural_earth(resolution="10m", category="cultural",
                              name="admin_0_countries")

def iso2(a):
    for k in ("ISO_A2_EH", "ISO_A2", "WB_A2"):
        v = a.get(k)
        if v and str(v) not in ("-99", "", "NA", "None"):
            return str(v)
    return None

GEOM: dict[str, list] = {}
for r in shpreader.Reader(FNC).records():
    i = iso2(r.attributes)
    if i:
        GEOM.setdefault(i, []).append(r.geometry)

PC = ccrs.PlateCarree()
GULF_PROJ = ccrs.LambertConformal(central_longitude=52.0, central_latitude=26.0,
                                  standard_parallels=(22.0, 30.0))

SEA_DEEP = MAP_SEA
LAND_BG = MAP_LAND
SHELF = "#E4EFF6"
REPORT_FILL = "#EBDFC6"      # reporting exporters: one warm tint, no pattern
REPORT_EDGE = "#BCA475"
PIPE = "#7A4E1D"
PIPE_TEXT = "#6F5427"
HYDRO_TEXT = "#4E7286"

# area names sit slightly above the house minimum so they hold rank over point features
T_AREA = 7.6

def frame_of(ax, pad=1.5):
    """Longitude/latitude box that safely contains what the axes actually shows.

    Cartopy widens the requested extent to honour the projection aspect. The drawn frame
    is larger than the extent asked for. Sampling the projected axes border and converting
    back gives a box that is guaranteed to enclose it.
    """
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    t = np.linspace(0.0, 1.0, 60)
    xs = np.concatenate([x0 + t * (x1 - x0), x0 + t * (x1 - x0),
                         np.full_like(t, x0), np.full_like(t, x1)])
    ys = np.concatenate([np.full_like(t, y0), np.full_like(t, y1),
                         y0 + t * (y1 - y0), y0 + t * (y1 - y0)])
    ll = PC.transform_points(GULF_PROJ, xs, ys)
    lon, lat = ll[:, 0], ll[:, 1]
    return sgeom.box(lon.min() - pad, lat.min() - pad, lon.max() + pad, lat.max() + pad)

def clip(geoms, frame):
    """Cut vectors down to the frame.

    Cartopy keeps whole geometries whose bounding box touches the view. Therefore a single 1:10m
    Eurasia polygon carries the entire continent into the PDF. Clipping first keeps the
    coastline detail the Gulf needs while holding the file to a size a journal will accept.
    """
    out = []
    for g in geoms:
        if g.intersects(frame):
            c = g.intersection(frame)
            if not c.is_empty:
                out.append(c)
    return out

# ===========================================================================
fig = plt.figure(figsize=(W2, 4.95))

# --- a. the Gulf ------------------------------------------------------------
axa = fig.add_axes([0.052, 0.115, 0.615, 0.815], projection=GULF_PROJ)
axa.set_extent((36.0, 60.5, 16.5, 32.0), crs=PC)
FRAME = frame_of(axa)

axa.set_facecolor(SEA_DEEP)

COAST = clip(cfeature.COASTLINE.with_scale("10m").geometries(), FRAME)
LAND = clip(cfeature.LAND.with_scale("10m").geometries(), FRAME)

# shelf rim first, then land over its inland half. The pale band survives only in water
axa.add_geometries(COAST, PC, facecolor="none", edgecolor=SHELF, linewidth=1.6,
                   zorder=1)
axa.add_geometries(LAND, PC, facecolor=LAND_BG, edgecolor="none", zorder=2)

for a in GULF:
    if a in REPORTS and a in GEOM:
        axa.add_geometries(clip(GEOM[a], FRAME), PC, zorder=3, facecolor=REPORT_FILL,
                           edgecolor=REPORT_EDGE, linewidth=0.45)

axa.add_geometries(clip(cfeature.LAKES.with_scale("10m").geometries(), FRAME), PC,
                   facecolor=SEA_DEEP, edgecolor="none", zorder=4)
axa.add_geometries(clip(cfeature.BORDERS.with_scale("10m").geometries(), FRAME), PC,
                   facecolor="none", edgecolor=MAP_BORDER, linewidth=0.4, zorder=5)
axa.add_geometries(COAST, PC, facecolor="none", edgecolor=MAP_COAST, linewidth=0.55,
                   zorder=6)

gl = axa.gridlines(draw_labels=True, linewidth=0.3, color=MAP_GRAT, alpha=0.9,
                   linestyle=(0, (4, 3)), zorder=4.5, rotate_labels=False,
                   x_inline=False, y_inline=False,
                   xlocs=range(38, 62, 4), ylocs=range(18, 34, 4))
gl.top_labels = gl.right_labels = False
gl.xlabel_style = gl.ylabel_style = {"size": TINY, "color": MUTED, **SANS}

for nm, lon, lat, rot in SEAS:
    axa.text(lon, lat, nm, transform=PC, fontsize=TINY, style="italic",
             color=HYDRO_TEXT, ha="center", va="center", rotation=rot,
             rotation_mode="anchor", zorder=8, **SANS)

# bypass corridors
for _, pts in CORRIDORS:
    lon, lat = zip(*pts)
    axa.plot(lon, lat, transform=PC, color="white", lw=3.4, zorder=6.5,
             solid_capstyle="round")
    axa.plot(lon, lat, transform=PC, color=PIPE, lw=2.0, zorder=7,
             solid_capstyle="round")

axa.annotate("Abqaiq–Yanbu (schematic)",
             xy=(44.5, 25.35), xycoords=PC._as_mpl_transform(axa), xytext=(0, -17),
             textcoords="offset points", ha="center", va="top", fontsize=TINY,
             color=PIPE_TEXT, zorder=11, **SANS,
             path_effects=[pe.withStroke(linewidth=2.0, foreground=LAND_BG)],
             arrowprops=dict(arrowstyle="-", color=PIPE_TEXT, lw=0.5, shrinkB=2))
axa.annotate("Habshan–Fujairah\n(schematic)",
             xy=(55.4, 24.50), xycoords=PC._as_mpl_transform(axa), xytext=(3, -31),
             textcoords="offset points", ha="left", va="top", fontsize=TINY,
             color=PIPE_TEXT, linespacing=1.25, zorder=11, **SANS,
             path_effects=[pe.withStroke(linewidth=2.0, foreground=LAND_BG)],
             arrowprops=dict(arrowstyle="-", color=PIPE_TEXT, lw=0.5, shrinkB=2))

# terminals and the two inland pipeline heads
for nm, (lon, lat, ha, off) in TERMINALS.items():
    axa.plot(lon, lat, marker="o", ms=4.0, mfc="white", mec=INK, mew=0.9,
             transform=PC, zorder=10)
    axa.annotate(nm, xy=(lon, lat), xycoords=PC._as_mpl_transform(axa), xytext=off,
                 textcoords="offset points", fontsize=TINY, color=INK, ha=ha,
                 va="center", zorder=11, **SANS,
                 path_effects=[pe.withStroke(linewidth=2.0, foreground="white")])
for nm, (lon, lat, ha, off) in WAYPOINTS.items():
    axa.plot(lon, lat, marker="s", ms=3.4, mfc="#F3E4C7", mec=PIPE_TEXT, mew=0.8,
             transform=PC, zorder=10)
    axa.annotate(nm, xy=(lon, lat), xycoords=PC._as_mpl_transform(axa), xytext=off,
                 textcoords="offset points", fontsize=TINY, color=PIPE_TEXT, ha=ha,
                 va="center", zorder=11, **SANS,
                 path_effects=[pe.withStroke(linewidth=2.0, foreground=LAND_BG)])

axa.plot(*HORMUZ, marker="o", ms=12, mfc="none", mec=ROLE["chokepoint"], mew=1.4,
         transform=PC, zorder=12)
axa.annotate("Strait of Hormuz", xy=HORMUZ, xycoords=PC._as_mpl_transform(axa),
             xytext=(22, 26), textcoords="offset points", fontsize=SMALL,
             fontweight="bold", color=INK, ha="center", va="bottom", zorder=13, **SANS,
             path_effects=[pe.withStroke(linewidth=2.2, foreground="white")],
             arrowprops=dict(arrowstyle="-", color=ROLE["chokepoint"], lw=0.9,
                             shrinkA=1, shrinkB=8))

# country labels, manual anchors over land
CLAB = {"SA": (44.4, 20.6), "IR": (55.2, 30.7), "IQ": (43.6, 31.1),
        "KW": (47.35, 30.0), "AE": (54.8, 22.9), "OM": (57.2, 20.0),
        "QA": (51.25, 25.35), "YE": (45.8, 17.3), "BH": (50.62, 26.05)}
for a, (lon, lat) in CLAB.items():
    if a in ("QA", "BH"):        # too small to hold their own name
        axa.annotate(NAMES[a].replace("\n", " "), xy=(lon, lat),
                     xycoords=PC._as_mpl_transform(axa), xytext=(13, 6),
                     textcoords="offset points", fontsize=TINY, color="#2E3A44",
                     fontweight="bold", zorder=11, **SANS,
                     path_effects=[pe.withStroke(linewidth=2.0, foreground="white")],
                     arrowprops=dict(arrowstyle="-", color="#2E3A44", lw=0.5, shrinkB=1))
        continue
    axa.text(lon, lat, NAMES[a], transform=PC, fontsize=T_AREA, ha="center",
             va="center", zorder=11, linespacing=1.2, fontweight="bold",
             color="#2E3A44", **SANS,
             path_effects=[pe.withStroke(linewidth=2.4, foreground=LAND_BG)])

# scale bar, computed in projected metres
x0, x1 = axa.get_xlim()
y0, y1 = axa.get_ylim()
bar_m = 400_000.0
bx, by = x0 + 0.06 * (x1 - x0), y0 + 0.075 * (y1 - y0)
halo = [pe.withStroke(linewidth=2.6, foreground="white")]
axa.plot([bx, bx + bar_m], [by, by], color=INK, lw=2.0, solid_capstyle="butt", zorder=14,
         path_effects=halo)
for xt in (bx, bx + bar_m):
    axa.plot([xt, xt], [by, by + 0.012 * (y1 - y0)], color=INK, lw=0.9, zorder=14,
             path_effects=halo)
axa.text(bx + bar_m / 2, by + 0.022 * (y1 - y0), "400 km", ha="center", va="bottom",
         fontsize=TINY, color=INK, zorder=14, path_effects=halo, **SANS)

axa.legend(handles=[
    Patch(facecolor=REPORT_FILL, edgecolor=REPORT_EDGE, linewidth=0.45,
          label=TERM["reports_exports"]),
    Patch(facecolor="#EBE7DF", edgecolor="none", label=TERM["baseline"]),
    Patch(facecolor="white", edgecolor="#B0A896", hatch="////", lw=0.6,
          label="Not reported"),
    Line2D([], [], color=PIPE, lw=2.0, label="Bypass pipeline"),
    Line2D([], [], marker="o", ms=4.0, mfc="white", mec=INK, ls="none",
           label="Crude terminal")],
    loc="upper left", bbox_to_anchor=(-0.015, -0.055), fontsize=SMALL, ncol=3,
    labelspacing=0.40, columnspacing=1.6, handlelength=1.5, frameon=False,
    borderpad=0.0).set_zorder(15)
panel_title(axa, "a", "The Gulf export route")

axb = fig.add_axes([0.752, 0.772, 0.213, 0.188],
                   projection=ccrs.Robinson(central_longitude=40))
axb.set_global()
axb.patch.set_facecolor(SEA_DEEP)
axb.add_feature(cfeature.LAND.with_scale("110m"), facecolor=LAND_BG, zorder=1)
axb.add_feature(cfeature.COASTLINE.with_scale("110m"), edgecolor=MAP_COAST,
                linewidth=0.25, zorder=2)
for a in GULF:
    if a in GEOM:
        axb.add_geometries(GEOM[a], PC, facecolor=REPORT_FILL, edgecolor="none", zorder=3)
axb.plot(*HORMUZ, marker="o", ms=8, mfc="none", mec=ROLE["chokepoint"], mew=1.2,
         transform=PC, zorder=5)
axb.spines["geo"].set_edgecolor(MAP_COAST)
axb.spines["geo"].set_linewidth(0.5)
panel_title(axb, "b", "Locator", pad=3.0)

xs = np.arange(len(MONTHS))
MLAB = ["Jan", "Feb", "Mar", "Apr", "May"]
XLIM = (-0.45, len(MONTHS) - 0.05)
CX, CW = 0.752, 0.213
axc = fig.add_axes([CX, 0.300, CW, 0.415])
axcb = fig.add_axes([CX, 0.128, CW, 0.112])

SERIES = {"SA": dict(color="#B2182B", ls="-", marker="o", lw=2.0, ms=4.2),
          "KW": dict(color=ROLE["export"], ls=(0, (4, 2)), marker="s", lw=1.9, ms=4.0),
          "BH": dict(color=ROLE["availability"], ls=(0, (1.6, 1.4)), marker="^", lw=1.7,
                     ms=4.6)}

for ax in (axc, axcb):
    ax.axvspan(-0.4, 1.4, color="#EBE7DF", zorder=1, lw=0)
    ax.set_xlim(*XLIM)
    ax.set_xticks(xs)
    ax.tick_params(labelsize=TINY)

axc.set_xticklabels([])
axcb.set_xticklabels(MLAB, fontsize=TINY)

END = {"SA": (3.93, -7, "top"), "KW": (3.93, 17, "bottom")}
for a in ("SA", "KW"):
    st = SERIES[a]
    v = E.loc[a].values.astype(float)
    axc.plot(xs, v, color=st["color"], ls=st["ls"], lw=st["lw"], marker=st["marker"],
             ms=st["ms"], mfc="white", mew=1.1, zorder=4)
    ok = np.isfinite(v)
    xa, dy, va = END[a]
    axc.annotate(f"{NAMES[a]} {v[ok][-1]:,.0f}, {MLAB[int(np.flatnonzero(ok)[-1])]}",
                 xy=(xa, v[ok][-1]), xytext=(0, dy),
                 textcoords="offset points", fontsize=TINY, color=st["color"],
                 ha="right", va=va, fontweight="bold", zorder=5,
                 path_effects=[pe.withStroke(linewidth=2.0, foreground="white")])

axc.set_ylim(-260, 7600)
axc.set_yticks(range(0, 7001, 1000))
axc.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter("{x:,.0f}"))
light_grid(axc)
panel_title(axc, "c", "Reported Gulf crude exports")

# Bahrain, on its own scale
bh = E.loc["BH"].values.astype(float)
seen = np.flatnonzero(np.isfinite(bh))
st = SERIES["BH"]
axcb.plot(xs[seen], bh[seen], color=st["color"], ls=st["ls"], lw=st["lw"],
          marker=st["marker"], ms=st["ms"], mfc="white", mew=1.1, zorder=4)
axcb.set_ylim(-28, 215)
axcb.set_yticks([0, 100, 200])
light_grid(axcb)
for i in range(len(MLAB)):
    if not np.isfinite(bh[i]):
        axcb.axvspan(i - 0.44, i + 0.44, facecolor="none", edgecolor="#B0A896",
                     hatch="////", lw=0.0, zorder=2)

axcb.annotate(f"{bh[seen[-1]]:,.0f}", xy=(xs[seen[-1]], bh[seen[-1]]),
              xytext=(5, 3), textcoords="offset points", fontsize=TINY,
              color=st["color"], ha="left", va="bottom", fontweight="bold", zorder=5,
              path_effects=[pe.withStroke(linewidth=2.0, foreground="white")])
off_scale_panel(axcb, "", hatch=None, label_loc="right")
axcb.set_ylabel(AX["exports"], fontsize=TICK)
axcb.set_xlabel(AX["month26"], fontsize=LABEL)
axcb.set_title(NAMES["BH"], fontsize=TICK, loc="left", pad=3.0,
               color=ROLE["availability"], fontweight="bold")
fig.text(CX - 0.060, (0.128 + 0.715) / 2, AX["exports"], rotation=90,
         va="center", ha="center", fontsize=LABEL, color=INK)

# --- provenance ------------------------------------------------------------

save(fig, "Fig1_gulf_anatomy")

b = E[MONTHS[:2]].mean(axis=1)
a2 = E[MONTHS[3:]].mean(axis=1)
print("  reported export retention, April–May vs January–February:")
for k in REPORTS:
    if np.isfinite(b.get(k, np.nan)) and b[k] > 0:
        print(f"    {NAMES[k].replace(chr(10),' '):22s} {b[k]:8,.0f} -> {a2[k]:8,.0f} "
              f"kb/d   retention {a2[k]/b[k]:.4f}")

E.to_csv(SD_WRITE / "fig1_source_data.csv")
print(f"  Fig1: panel c series written for {len(E)} Gulf economies")
