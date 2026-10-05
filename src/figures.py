"""The ten published figures.

Each function draws one plate and writes it to outputs/ as PDF and PNG. Inputs are
the tables in data/panel and data/market.
"""

from __future__ import annotations

from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.patches import Patch, Polygon, Rectangle
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import FuncFormatter, MultipleLocator
from matplotlib.transforms import Bbox
from pathlib import Path
from scipy import stats
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import io
import matplotlib as mpl
import matplotlib.path as mpath
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import os
import pandas as pd
import shapely.geometry as sgeom

from style import *  # noqa: F401,F403
from style import MARKET, OUTPUTS, TABLES, apply_style, save

apply_style()
TABLE_DIR = OUTPUTS / "tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)


def wt(name, body):
    """Write one supplementary table as a LaTeX fragment."""
    (TABLE_DIR / name).write_text(body, encoding="utf-8")


def figure_1_gulf_anatomy():
    """Draw figure 1 gulf anatomy."""
    # Map lettering follows the plate, which keeps one family across all eight figures.
    SANS = {}

    # ---------------------------------------------------------------------------
    E = pd.read_csv(TABLES / "gulf_crude_exports_kbd.csv", index_col=0).apply(
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

    E.to_csv(OUTPUTS / "fig1_source_data.csv")
    print(f"  Fig1: panel c series written for {len(E)} Gulf economies")


def figure_2_transport_boundary():
    """Draw figure 2 transport boundary."""
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
    TR = pd.read_csv(MARKET / "hormuz_tanker_transits_monthly.csv").set_index("date")
    FR = pd.read_csv(MARKET / "td3c_monthly.csv").set_index("date")
    BR = pd.read_csv(MARKET / "brent_price_curve.csv").set_index("date")
    CTX = pd.read_csv(TABLES / "transport_context.csv", header=None, index_col=0).squeeze("columns")

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

    P = pd.read_csv(TABLES / "primary_panel.csv")
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
    S.to_csv(OUTPUTS / "fig6_transport_source_data.csv", index=False)

    D = pd.DataFrame([{"panel_c_stage": n, "basis": b, "retention": round(r, 4),
                       "retention_pct": round(100 * r, 1)} for n, b, r, _ in STAGES])
    D.to_csv(OUTPUTS / "fig6_attenuation_source_data.csv", index=False)

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
    print(f"  wrote {TABLES / 'fig6_transport_source_data.csv'} and fig6_attenuation_source_data.csv")


def figure_3_world_outcomes():
    """Draw figure 3 world outcomes."""
    mpl.rcParams["hatch.linewidth"] = 0.32

    # ---------------------------------------------------------------------------
    FC = pd.read_csv(TABLES / "final_classification.csv")
    P = pd.read_csv(TABLES / "primary_panel.csv")
    # ADMISSIBLE_ONLY: physically valid destinations plus crude-exporting importers
    P = P[P.population.isin(["primary", "crude-exporting importer"])].copy()
    STATUS = dict(zip(FC.country_iso2, FC.status))
    UNIVERSE = set(FC.country_iso2)          # economies that reported crude in the event window
    SAMPLE = set(FC.country_iso2[FC.status == "accounting panel"])
    HUBS = set(P.country_iso2[P.population == "crude-exporting importer"])
    GAP = dict(zip(P.country_iso2, P.gap))
    SCALE = dict(zip(P.country_iso2, P.baseline_availability_kbd))

    WINDOW = ["2026-01", "2026-02", "2026-03", "2026-04", "2026-05"]
    EXTRACT = TABLES.parent / "reproduction" / "data" / "jodi_crude_2020_2026.csv.gz"
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
    P.to_csv(OUTPUTS / "fig2_source_data.csv", index=False)
    print(f"  gap range {D.gap.min():+.3f} to {D.gap.max():+.3f}; "
          f"{int((D.gap.abs() > CAP).sum())} beyond the color scale")
    print("  bin limits " + ", ".join(f"{b:+.2f}" for b in BOUNDS))


def figure_4_absorption():
    """Draw figure 4 absorption."""
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

    P = pd.read_csv(TABLES / "primary_panel.csv")
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

    C.to_csv(OUTPUTS / "fig3_source_data.csv", index=False)

    pd.DataFrame(
        [{"panel": tag, "population": pop, "n": len(sub), "component": ROLE_LABEL[role],
          "kbd": round(v, 4), "multiple_of_L": round(v / comps[0][1], 6)}
         for tag, pop, sub in (("b", "destination systems", DEST),
                               ("c", "contracting net importers", C))
         for comps in [components(sub)]
         for role, v in comps]
    ).to_csv(OUTPUTS / "fig3_components.csv", index=False)
    print("  wrote sourcedata/fig3_components.csv")
    print(f"  destination bridge availability {rb:+,.1f} kb/d;  contracting net importers {rc:+,.1f}")


def figures_5_and_8():
    """Draw figures 5 and 8."""
    P = pd.read_csv(TABLES / "primary_panel.csv")
    # ADMISSIBLE_ONLY: physically valid destinations plus crude-exporting importers
    P = P[P.population.isin(["primary", "crude-exporting importer"])].copy()
    PL = pd.read_csv(TABLES / "placebo_windows.csv")
    COV = pd.read_csv(TABLES / "baseline_days_of_cover.csv").rename(
        columns={"REF_AREA": "country_iso2"})
    DEP = pd.read_csv(TABLES / "country_event_anomaly.csv")[["country_iso2", "dependence",
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
        "gap", "contracts"]].to_csv(OUTPUTS / "fig4_source_data.csv", index=False)
    PL[["year", "country_iso2", "import_retention", "availability_retention",
        "wedge"]].to_csv(OUTPUTS / "fig4c_source_data.csv", index=False)
    print(f"  Fig4: {k}/{len(C)} above line, p={pv:.4f}; placebo MW p={mw:.4f}")

    M = P.merge(COV[["country_iso2", "CLOSTLV", "REFINOBS", "days_cover"]],
                on="country_iso2", how="left")
    M["cover_ok"] = M.CLOSTLV > 0
    M["s_stock"] = M.d_stockdraw_kbd / (-M.d_imports_kbd)
    DC = pd.read_csv(TABLES / "depletion_corrected.csv")[["country_iso2", "acute_draw_kbd"]]
    M = M.merge(DC, on="country_iso2", how="left")
    M["months"] = np.where(M.acute_draw_kbd > 0,
                           M.CLOSTLV / (M.acute_draw_kbd * 30.5), np.nan)
    M["draw_share"] = M.acute_draw_kbd / M.REFINOBS
    M["h_stable"] = M.draw_share >= 0.01
    M["display_name"] = [NAME.get(a, a) for a in M.country_iso2]
    Md = M[M.country_iso2.isin(PRIM)].sort_values("display_name", ascending=False)
    DEPP = pd.read_csv(TABLES / "depletion_prospective.csv")
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
    M.to_csv(OUTPUTS / "fig5_source_data.csv", index=False)
    print(f"  Fig5: cover rho={rho:+.4f} n={len(K)}; depletion n={len(D2)}; "
          f"baseline {D2.horizon_baseline.min():.1f}-{D2.horizon_baseline.max():.1f} mo, "
          f"prospective {D2.horizon_prospective.min():.1f}-{D2.horizon_prospective.max():.1f} mo; "
          f"unstable {sorted(D2.country_iso2[~D2.stable])}")

    G = P.merge(DEP, on="country_iso2", how="inner")
    G["d_avail_norm"] = G.d_availability_kbd / G.baseline_availability_kbd
    G["stock_norm"] = G.d_stockdraw_kbd / G.baseline_availability_kbd
    G.to_csv(OUTPUTS / "fig6_source_data.csv", index=False)
    print(f"  corridor panel: n={len(G)} matched to dependence, written for FigS5")


def figure_6_boundaries():
    """Draw figure 6 boundaries."""
    EPISODES = ["Hormuz 2026", "Russia 2022-23", "Red Sea 2023-24"]
    SHORT = {"Hormuz 2026": "Hormuz 2026", "Russia 2022-23": "Russia 2022–23",
             "Red Sea 2023-24": "Red Sea 2023–24"}
    MECH = {"Hormuz 2026": "chokepoint", "Russia 2022-23": "origin switch",
            "Red Sea 2023-24": "routing"}
    # Short tag for the pass-through block, where the full episode name does not fit.
    TAG = {"Hormuz 2026": "Hormuz", "Russia 2022-23": "Russia", "Red Sea 2023-24": "Red Sea"}
    # One line style per episode. Therefore the three are separable without color.
    DASH = {"Hormuz 2026": (0, ()), "Russia 2022-23": (0, (4, 1.6)),
            "Red Sea 2023-24": (0, (1.1, 1.4))}
    NAME = {"CZ": "Czechia", "DE": "Germany", "GR": "Greece", "IE": "Ireland", "PL": "Poland"}

    def load():
        ep = pd.read_csv(TABLES / "multi_episode.csv").set_index("episode")
        corr = pd.read_csv(TABLES / "corridor_episodes.csv")
        ctrl = pd.read_csv(TABLES / "corridor_placebo.csv")
        mode = pd.read_csv(TABLES / "comext_mode_episode.csv")
        mode = mode[(mode.episode == "Hormuz 2026") & (mode.in_comext == True)]  # noqa: E712
        return ep, corr, ctrl, mode

    def main() -> None:
        apply_style()
        ep, corr, ctrl, mode = load()

        primary = corr[corr.primary == True].set_index("episode")  # noqa: E712
        rows = []
        for name in EPISODES:
            e = ep.loc[name]
            bS, bP, bX = (float(e[k]) for k in
                          ("margin_inventory", "margin_production", "margin_export"))
            rows.append({
                "episode": name,
                "corridor": primary.loc[name, "corridor"],
                "corridor_retention": float(primary.loc[name, "transit_retention"]),
                "import_retention": float(e["mean_import_retention"]),
                "availability_retention": float(e["mean_availability_retention"]),
                "pass_through": round(1.0 - bS - bP - bX, 3),
            })
        ladder = pd.DataFrame(rows)

        # The claim the figure has to carry: lowest at the corridor, highest at the gate.
        worst_corridor = ladder.loc[ladder.corridor_retention.idxmin(), "episode"]
        best_gate = ladder.loc[ladder.availability_retention.idxmax(), "episode"]
        assert worst_corridor == best_gate == "Hormuz 2026", (worst_corridor, best_gate)

        fig = plt.figure(figsize=(W2, 3.15))
        gs = fig.add_gridspec(1, 3, width_ratios=[1.06, 1.0, 0.94], wspace=0.42,
                              left=0.062, right=0.987, top=0.90, bottom=0.325)
        axa, axb, axc = (fig.add_subplot(gs[0, i]) for i in range(3))

        # ---------------------------------------------------------------- a, retention ladder
        xs = [0, 1, 2]
        shapes = ["o", "s", "D"]
        axa.axvline(0.5, color=RULE, lw=0.7, zorder=1)
        for _, r in ladder.iterrows():
            ys = [r.corridor_retention, r.import_retention, r.availability_retention]
            axa.plot(xs[1:], ys[1:], color=MUTED, lw=1.0,
                     dashes=DASH[r.episode][1] or (None, None), zorder=2, solid_capstyle="round")
            for x, yv, mk in zip(xs, ys, shapes):
                axa.plot([x], [yv], marker=mk, ms=4.2, mfc=INK, mec="white", mew=0.7, zorder=3)
            axa.annotate(SHORT[r.episode], xy=(0, r.corridor_retention), xytext=(-4, 0),
                         textcoords="offset points", va="center", ha="right",
                         fontsize=TINY, color=INK)
        axa.annotate("different\npopulation", xy=(0, 0.68), ha="center", va="center",
                     fontsize=TICK, color=MUTED, zorder=4, linespacing=1.3)
        axa.set_xticks(xs)
        axa.set_xticklabels(["Chokepoint", "Imports", "Refinery\nsupply"], fontsize=TICK)
        axa.set_xlabel("Measurement boundary", fontsize=LABEL)
        axa.set_xlim(-1.60, 2.18)
        axa.set_ylim(0, 1.12)
        axa.set_ylabel(AX["retention"], fontsize=LABEL)
        axa.axhline(1.0, color=MUTED, lw=0.55, ls=(0, (3, 2)), zorder=1)
        light_grid(axa)
        panel_title(axa, "a", "Ratios at three boundaries", pad=5)

        # ---------------------------------------------------------------- b, corridor vs control
        order = corr.sort_values("transit_retention").reset_index(drop=True)
        y = range(len(order))
        for i, r in order.iterrows():
            c = ctrl[(ctrl.corridor == r.corridor) & (ctrl.episode == r.episode)].iloc[0]
            lo, hi = sorted([r.transit_retention, c.placebo_retention])
            axb.plot([lo, hi], [i, i], color=MUTED, lw=0.8, zorder=2)
            axb.plot([c.placebo_retention], [i], marker="|", ms=7, color=MUTED, mew=1.2, zorder=3)
            face = ROLE["production"] if r.role == "reroute" else ROLE["chokepoint"]
            axb.plot([r.transit_retention], [i], marker="o", ms=4.8, mfc=face,
                     mec="white", mew=0.7, zorder=4)
        axb.set_yticks(list(y))
        axb.set_yticklabels(["%s" % s.replace(" Strait", "").replace("Strait of ", "")
                             for s in order.corridor], fontsize=TINY)
        axb.axvline(1.0, color=MUTED, lw=0.55, ls=(0, (3, 2)), zorder=1)
        axb.set_xlim(-0.04, 2.02)
        axb.set_xlabel(AX["transit_ret"], fontsize=LABEL)
        axb.invert_yaxis()
        light_grid(axb, axis="x")
        panel_title(axb, "b", "Transit ratio by chokepoint", pad=5)

        # ---------------------------------------------------------------- c, transport mode
        m = mode.copy()
        m["sea"] = m.baseline_sea_share.astype(float)
        m = m.sort_values("sea", ascending=False).reset_index(drop=True)
        for i, r in m.iterrows():
            axc.barh(i, r.sea, height=0.62, color=ROLE["chokepoint"], zorder=3)
            axc.barh(i, 1 - r.sea, left=r.sea, height=0.62,
                     color=ROLE["infrastructure"], zorder=3)
        axc.set_yticks(range(len(m)))
        axc.set_yticklabels(country_labels(m.reporter, "name"), fontsize=TICK)
        axc.set_xlim(0, 1)
        axc.set_xticks([0, 0.25, 0.5, 0.75, 1])
        axc.set_xticklabels(["0", "25", "50", "75", "100"], fontsize=TICK)
        axc.set_xlabel(AX["sea_share"], fontsize=LABEL)
        axc.invert_yaxis()
        panel_title(axc, "c", "Arrival mode of crude imports", pad=5)

        figure_legend(fig, [
            Line2D([], [], marker="o", ms=4.2, mfc=INK, mec="white", ls="none",
                   label="Chokepoint (a)"),
            Line2D([], [], marker="s", ms=4.2, mfc=INK, mec="white", ls="none",
                   label="Imports (a)"),
            Line2D([], [], marker="D", ms=4.2, mfc=INK, mec="white", ls="none",
                   label="Refinery supply (a)"),
            Line2D([], [], marker="o", ms=4.8, mfc=ROLE["chokepoint"], mec="white", ls="none",
                   label="Episode, affected chokepoint (b)"),
            Line2D([], [], marker="o", ms=4.8, mfc=ROLE["production"], mec="white", ls="none",
                   label="Episode, diversion leg (b)"),
            Line2D([], [], marker="|", ms=7, color=MUTED, mew=1.2, ls="none",
                   label="Same periods, prior year (b)"),
            Patch(facecolor=ROLE["chokepoint"], label="By sea (c)"),
            Patch(facecolor=ROLE["infrastructure"], label="Overland (c)")]
            + [Line2D([], [], color=INK, lw=1.15, dashes=DASH[e][1] or (None, None),
                      label=f"{SHORT[e]} (a)") for e in SHORT],
            ncol=4, y=0.012, handlelength=1.7, columnspacing=1.3, handletextpad=0.45)

        save(fig, "Fig8_boundaries")

        out = ladder.merge(
            corr[["episode", "corridor", "transit_retention"]]
            .rename(columns={"transit_retention": "corridor_all_counting_points"}),
            on=["episode", "corridor"], how="left")
        out.to_csv(OUTPUTS / "fig8_source_data.csv", index=False)
        print("  wrote sourcedata/fig8_source_data.csv")
        print(out.to_string(index=False))

    if True:
        main()


def figure_7_episode_maps():
    """Draw figure 7 episode maps."""
    MINUS = chr(8722)                       # the true minus sign, never a hyphen
    CONTRACTION_RULE = 0.95                 # the screen used in tre_multi_episode

    # ---------------------------------------------------------------------------
    P = pd.read_csv(TABLES / "multi_episode_panel.csv")
    E = pd.read_csv(TABLES / "multi_episode.csv")

    EPISODES = list(E.episode)                       # drawing order is the episode order on file
    NOTE = dict(zip(E.episode, E.note))
    MECHANISM = {"Hormuz 2026": "chokepoint closure", "Russia 2022-23": "origin switch",
                 "Red Sea 2023-24": "rerouting"}
    SAMPLE = {ep: sorted(P.economy[P.episode == ep]) for ep in EPISODES}
    GAP = {ep: dict(zip(P.economy[P.episode == ep], P.gap[P.episode == ep])) for ep in EPISODES}
    IMPRET = {ep: dict(zip(P.economy[P.episode == ep], P.import_retention[P.episode == ep]))
              for ep in EPISODES}
    UNION = sorted(set(P.economy))
    CONTRACTING = {ep: {a for a, v in IMPRET[ep].items() if v < CONTRACTION_RULE}
                   for ep in EPISODES}

    for _, r in E.iterrows():
        assert len(CONTRACTING[r.episode]) == int(r.contracting), r.episode
        assert len(SAMPLE[r.episode]) == int(r.destinations), r.episode

    # ---------------------------------------------------------------------------
    BOUNDS = [-0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20]
    BIN_POS = [0.00, 0.10, 0.24, 0.36, 0.64, 0.76, 0.90, 1.00]      # sampling points on the ramp
    BIN_COLORS = [mpl.colors.to_hex(RETENTION_CMAP(p)) for p in BIN_POS]
    CMAP = mpl.colors.ListedColormap(BIN_COLORS[1:-1])
    CMAP.set_under(BIN_COLORS[0])
    CMAP.set_over(BIN_COLORS[-1])
    NORM = mpl.colors.BoundaryNorm(BOUNDS, ncolors=CMAP.N)

    NOT_ADMITTED_FACE = "#C9C2B2"    # stone, clearly darker than the neutral land tone
    NOT_ADMITTED_LINE = "#8C8474"    # its hatch, and nothing else in this figure
    SAMPLE_EDGE = "#4A565F"          # outline of any country that belongs to the union sample

    PC = ccrs.PlateCarree()
    MAIN_EXT = (-15.0, 55.0, 33.0, 71.0)     # holds every European member with Arctic room to spare
    MAIN = ccrs.AlbersEqualArea(central_longitude=20, central_latitude=52,
                                standard_parallels=(40, 65))
    AP_EXT = (94.0, 147.0, -12.0, 47.0)      # Indonesia, Thailand, Korea, Japan
    AP = ccrs.AlbersEqualArea(central_longitude=120, central_latitude=18,
                              standard_parallels=(0, 36))
    INSET_BOX = (0.675, 0.580, 0.305, 0.400)

    # ---------------------------------------------------------------------------
    NE_COUNTRIES = shpreader.natural_earth(resolution="50m", category="cultural",
                                           name="admin_0_countries")

    def iso2(a):
        for k in ("ISO_A2_EH", "ISO_A2", "WB_A2"):
            v = a.get(k)
            if v and str(v) not in ("-99", "", "NA", "None"):
                return str(v)
        return None

    GEOM, ANCHOR = {}, {}
    for r in shpreader.Reader(NE_COUNTRIES).records():
        i = iso2(r.attributes)
        if not i:
            continue
        GEOM.setdefault(i, []).append(r.geometry)
        if i not in ANCHOR:
            g = r.geometry
            main = max(g.geoms, key=lambda q: q.area) if g.geom_type == "MultiPolygon" else g
            p = main.representative_point()
            ANCHOR[i] = (p.x, p.y)

    missing = [a for a in UNION if a not in GEOM]
    assert not missing, f"no Natural Earth geometry for {missing}"

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
        """Drop vertices that cannot survive the printed scale.

        Six maps of the same continent carry six copies of 1:50m country outlines, and at these
        widths most of those vertices land inside one printed dot. The tolerances below are set
        from the scale of each panel, not from the file: on the main maps 0.02 degrees is about
        2 km, which is a third of a 600 dpi pixel on the page, and on the inset 0.10 degrees is
        the same fraction of a pixel again. The drawn outline is unchanged; the file is not.
        """
        return [g.simplify(tol, preserve_topology=True) for g in geoms]

    MAIN_TOL, AP_TOL = 0.02, 0.10

    def projected_aspect(proj, extent, k=180):
        """Width over height of an extent once projected, with the edges densified."""
        w, e, s, n = extent
        lon, lat = np.linspace(w, e, k), np.linspace(s, n, k)
        ring = np.vstack([np.column_stack([lon, np.full(k, s)]),
                          np.column_stack([np.full(k, e), lat]),
                          np.column_stack([lon[::-1], np.full(k, n)]),
                          np.column_stack([np.full(k, w), lat[::-1]])])
        xy = proj.transform_points(PC, ring[:, 0], ring[:, 1])[:, :2]
        return np.ptp(xy[:, 0]) / np.ptp(xy[:, 1])

    MAIN_ASPECT = projected_aspect(MAIN, MAIN_EXT)
    AP_ASPECT = projected_aspect(AP, AP_EXT)

    # ---------------------------------------------------------------------------
    PAD_L, PAD_R, GAP_M = 0.10, 0.10, 0.15
    W_M = (W2 - PAD_L - PAD_R - 2 * GAP_M) / 3.0
    H_M = W_M / MAIN_ASPECT

    TOP_PAD = 0.05
    TITLE_H = 0.44          # panel letter, mechanism line, count line
    GAP_AB = 0.13
    KEY_H = 0.64
    GAP_BD = 0.10
    TITLE_D = 0.30
    H_D = 1.00
    GAP_DE = 0.62
    TITLE_E = 0.26
    H_E = 0.60
    FOOT = 0.42             # economy codes and the panel-e key

    FIG_H = (TOP_PAD + TITLE_H + H_M + GAP_AB + KEY_H + GAP_BD + TITLE_D + H_D
             + GAP_DE + TITLE_E + H_E + FOOT)

    fig = plt.figure(figsize=(W2, FIG_H))

    def rect(x, y, w, h):
        """Inches from the bottom-left, which is how the layout above is reasoned about."""
        return [x / W2, y / FIG_H, w / W2, h / FIG_H]

    Y_MAPS = FIG_H - TOP_PAD - TITLE_H - H_M
    Y_KEY = Y_MAPS - GAP_AB - KEY_H
    Y_D = Y_KEY - GAP_BD - TITLE_D - H_D
    Y_E = Y_D - GAP_DE - TITLE_E - H_E

    # ---------------------------------------------------------------------------
    def draw_fills(ax, extent, episode, pad=6.0, lw_value=0.45, lw_edge=0.38, tol=MAIN_TOL):
        """The three country states, in the order that keeps the darkest line work on top."""
        gap = GAP[episode]
        for a, gs in GEOM.items():
            vis = crop(gs, extent, pad=pad)
            if not vis:
                continue
            vis = generalise(vis, tol)
            if a in gap:                                   # carries a value in this episode
                ax.add_geometries(vis, PC, zorder=3, facecolor=CMAP(NORM(gap[a])),
                                  edgecolor=SAMPLE_EDGE, linewidth=lw_value)
            elif a in UNION:                               # in the sample, not admitted here
                ax.add_geometries(vis, PC, zorder=2.6, facecolor=NOT_ADMITTED_FACE,
                                  edgecolor=NOT_ADMITTED_LINE, linewidth=0.0, hatch="///")
                ax.add_geometries(vis, PC, zorder=3, facecolor="none", edgecolor=SAMPLE_EDGE,
                                  linewidth=lw_value)
            else:                                          # outside the sample entirely
                ax.add_geometries(vis, PC, zorder=2, facecolor=MAP_LAND,
                                  edgecolor=MAP_LAND_EDGE, linewidth=lw_edge)

    AP_LABEL_OFFSET = {"JP": (-13, -6), "KR": (-15, -3), "TH": (4, 9), "ID": (5, 8)}
    axes_maps = []
    for j, ep in enumerate(EPISODES):
        x = PAD_L + j * (W_M + GAP_M)
        ax = fig.add_axes(rect(x, Y_MAPS, W_M, H_M), projection=MAIN)
        map_layers(ax, MAIN_EXT, PC, scale="110m")
        draw_fills(ax, MAIN_EXT, ep)
        map_linework(ax, coast_lw=0.45, border_lw=0.0, scale="110m")   # borders are carried by the country edges
        axes_maps.append(ax)

        bx, by, bw, bh = INSET_BOX
        ip = [(x + bx * W_M) / W2, (Y_MAPS + by * H_M) / FIG_H, bw * W_M / W2, bh * H_M / FIG_H]
        axi = fig.add_axes(ip, projection=AP)
        map_layers(axi, AP_EXT, PC, scale="110m")
        draw_fills(axi, AP_EXT, ep, pad=3.0, lw_value=0.35, lw_edge=0.2, tol=AP_TOL)
        map_linework(axi, coast_lw=0.3, border_lw=0.0, scale="110m")
        axi.spines["geo"].set_edgecolor(INK)
        axi.spines["geo"].set_linewidth(0.7)
        for a in ("JP", "KR", "TH", "ID"):
            lon, lat = ANCHOR[a]
            dx, dy = AP_LABEL_OFFSET[a]
            axi.annotate(a, xy=(lon, lat), xycoords=PC._as_mpl_transform(axi),
                         xytext=(dx, dy), textcoords="offset points", ha="center", va="center",
                         fontsize=TINY, color=INK, zorder=10,
                         path_effects=[pe.withStroke(linewidth=1.5, foreground="white")])

        # three-line panel head: letter and episode, transport mechanism, population counts
        row = E[E.episode == ep].iloc[0]
        ax.text(0.0, 1.0 + (TITLE_H - 0.22) / H_M,
                f"$\\bf{{({'abc'[j]})}}$ {ep}, {MECHANISM[ep]}", transform=ax.transAxes,
                fontsize=TITLE, va="baseline", ha="left", color=INK)

    inv = axes_maps[0].transAxes.inverted()
    for a in UNION:
        if a in ("ID", "JP", "KR", "TH"):
            continue
        px, py = MAIN.transform_point(*ANCHOR[a], PC)
        fx, fy = inv.transform(axes_maps[0].transData.transform((px, py)))
        bx, by, bw, bh = INSET_BOX
        assert not (bx - 0.02 < fx < bx + bw and by - 0.02 < fy < by + bh), \
            f"{a} lies under the Asia-Pacific inset at ({fx:.2f}, {fy:.2f})"

    # ---------------------------------------------------------------------------
    CB_W, CB_X = 3.05, PAD_L + 0.22
    cax = fig.add_axes(rect(CB_X, Y_KEY + 0.455, CB_W, 0.105))
    cb = mpl.colorbar.ColorbarBase(cax, cmap=CMAP, norm=NORM, extend="both",
                                   extendfrac=0.055, orientation="horizontal",
                                   spacing="uniform", ticks=BOUNDS, drawedges=True)
    cb.ax.tick_params(labelsize=TINY, width=0.5, length=2.0, pad=1.6)
    cb.ax.set_xticklabels([("0" if b == 0 else f"{b:+.2f}").replace("-", MINUS)
                           for b in BOUNDS])
    cb.outline.set_linewidth(0.5)
    cb.outline.set_edgecolor(INK)
    cb.dividers.set_linewidth(0.5)
    cb.dividers.set_color("white")
    fig.text(CB_X / W2, (Y_KEY + 0.645) / FIG_H, AX["gap"],
             fontsize=LABEL, ha="left", va="baseline", color=INK)

    # the country-state grammar, stated in words beside the scale it does not belong to
    KX = PAD_L + CB_W + 0.62
    axk = fig.add_axes(rect(KX, Y_KEY + 0.14, W2 - KX - PAD_R, KEY_H - 0.14))
    axk.axis("off")
    axk.legend(handles=[
        Patch(facecolor=BIN_COLORS[5], edgecolor=SAMPLE_EDGE, lw=0.6,
              label=TERM["admitted"]),
        Patch(facecolor=NOT_ADMITTED_FACE, edgecolor=SAMPLE_EDGE, lw=0.6, hatch="///",
              label=TERM["not_admitted"]),
        Patch(facecolor=MAP_LAND, edgecolor=MAP_COAST, lw=0.4,
              label="Outside the sample")],
        loc="upper left", bbox_to_anchor=(0.0, 1.0), fontsize=TINY, frameon=False,
        labelspacing=0.60, handlelength=1.3, handleheight=0.95, handletextpad=0.5,
        borderpad=0.0, alignment="left")

    # ---------------------------------------------------------------------------
    MARGINS = (("margin_inventory", "inventory", "Inventory adjustment"),
               ("margin_production", "production", "Domestic production"),
               ("margin_export", "export", "Export adjustment"))
    RESIDUAL_LABEL = "Reached refineries"

    axd = fig.add_axes(rect(PAD_L + 1.02, Y_D, W2 - PAD_L - 1.02 - PAD_R - 0.86, H_D))
    D_LIM = (-0.10, 1.10)
    comp_rows = []
    for i, ep in enumerate(EPISODES):
        r = E[E.episode == ep].iloc[0]
        y = len(EPISODES) - 1 - i
        net = sum(float(r[c]) for c, _, _ in MARGINS)
        axd.barh(y, 1.0 - net, left=net, height=0.56, color=ROLE["availability"], alpha=0.22,
                 edgecolor=ROLE["availability"], lw=0.5, zorder=2)
        right = left = 0.0
        for col, role, _ in MARGINS:
            v = float(r[col])
            if abs(v) < 1e-9:
                continue
            base = right if v > 0 else left
            axd.barh(y, v, left=base, height=0.56, color=ROLE[role], edgecolor="white", lw=0.6,
                     zorder=3)
            right, left = (right + v, left) if v > 0 else (right, left + v)
            comp_rows.append({"episode": ep, "margin": role, "share_of_shortfall": v})
        comp_rows.append({"episode": ep, "margin": "not offset", "share_of_shortfall": 1.0 - net})

        inv_v = float(r.margin_inventory)
        if inv_v > 0.18:
            axd.annotate(f"{inv_v:.0%}", xy=(inv_v / 2, y), ha="center", va="center",
                         fontsize=TINY, fontweight="bold", color="white", zorder=6)
        else:
            axd.annotate(f"{inv_v:.0%}", xy=(right, y), xytext=(3, 9),
                         textcoords="offset points", ha="left", va="center", fontsize=TICK,
                         fontweight="bold", color=ROLE["inventory"], zorder=7,
                         arrowprops=dict(arrowstyle="-", lw=0.5, color=ROLE["inventory"],
                                         shrinkA=0.0, shrinkB=1.0))
        axd.annotate(f"{1.0 - net:.0%}", xy=(net + (1.0 - net) / 2, y),
                     ha="center", va="center", fontsize=TICK, color=INK, zorder=6,
                     path_effects=[pe.withStroke(linewidth=2.0, foreground="white")])
        # the denominator, written where it cannot be mistaken for a share
        axd.annotate(f"{r.shortfall_kbd:,.1f} kb/d", xy=(1.0, y), xytext=(52, 0),
                     textcoords="offset points", ha="right", va="center", fontsize=TINY,
                     color=MUTED, annotation_clip=False)

    axd.axvline(0, color=INK, lw=0.9, zorder=4)
    axd.axvline(1.0, color=INK, lw=1.0, ls=(0, (3, 2)), zorder=4)
    axd.set_xlim(*D_LIM)
    axd.set_ylim(-0.62, len(EPISODES) - 0.38)
    axd.set_yticks(range(len(EPISODES)))
    axd.set_yticklabels(EPISODES[::-1], fontsize=SMALL)
    axd.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    axd.set_xticklabels(["0", "25", "50", "75", "100"], fontsize=TICK)
    axd.set_xlabel(axis_label("share of import shortage", "%"), fontsize=LABEL, labelpad=1.8,
                   color=INK)
    axd.tick_params(axis="y", length=0, pad=3.0)
    axd.grid(axis="x", color=RULE, lw=0.4, alpha=0.7)
    axd.set_axisbelow(True)
    for side in ("top", "right", "left"):
        axd.spines[side].set_visible(False)

    fig.text(PAD_L / W2, (Y_D + H_D + 0.08) / FIG_H,
             "$\\bf{(d)}$ Coverage of the import shortage by episode",
             fontsize=TITLE, ha="left", va="baseline", color=INK)

    figure_legend(fig, [Patch(facecolor=ROLE[role], edgecolor="white", lw=0.6, label=lab)
                        for _, role, lab in MARGINS]
                  + [Patch(facecolor=ROLE["availability"], alpha=0.22,
                           edgecolor=ROLE["availability"], lw=0.5, label=RESIDUAL_LABEL),
                     # the dashed rule lost its words when the x label was shortened
                     Line2D([], [], color=INK, lw=1.0, ls=(0, (3, 2)),
                            label="Import shortage fully covered")],
                  ncol=5, y=(Y_D - 0.54) / FIG_H, handlelength=1.25, handleheight=0.9,
                  handletextpad=0.45, columnspacing=1.1, borderpad=0.0)

    # ---------------------------------------------------------------------------
    axe = fig.add_axes(rect(PAD_L + 0.72, Y_E, W2 - PAD_L - 0.72 - PAD_R - 0.04, H_E))
    n = len(UNION)
    for j, ep in enumerate(EPISODES):
        y = len(EPISODES) - 1 - j
        for i, a in enumerate(UNION):
            if a in GAP[ep]:
                hit = a in CONTRACTING[ep]
                axe.add_patch(Rectangle((i + 0.10, y - 0.36), 0.80, 0.72,
                                        facecolor=CMAP(NORM(GAP[ep][a])),
                                        edgecolor=INK if hit else SAMPLE_EDGE,
                                        lw=1.2 if hit else 0.3, zorder=4 if hit else 3))
            else:
                axe.add_patch(Rectangle((i + 0.10, y - 0.36), 0.80, 0.72,
                                        facecolor=NOT_ADMITTED_FACE, edgecolor=NOT_ADMITTED_LINE,
                                        lw=0.35, hatch="///", zorder=3))
    axe.set_xlim(0, n)
    axe.set_ylim(-0.5, len(EPISODES) - 0.5)
    axe.set_xticks(np.arange(n) + 0.5)
    axe.set_xticklabels(UNION, fontsize=TINY)
    axe.set_yticks(range(len(EPISODES)))
    axe.set_yticklabels(EPISODES[::-1], fontsize=TINY)
    axe.tick_params(axis="both", length=0, pad=2.4)
    for side in ("top", "right", "left", "bottom"):
        axe.spines[side].set_visible(False)

    fig.text(PAD_L / W2, (Y_E + H_E + 0.10) / FIG_H,
             "$\\bf{(e)}$ Supply-import gap by country and episode", fontsize=TITLE,
             ha="left", va="baseline", color=INK)
    cax_e = fig.add_axes(rect(W2 - PAD_R - 1.84, Y_E - 0.315, 1.80, 0.052))
    cb_e = mpl.colorbar.ColorbarBase(cax_e, cmap=CMAP, norm=NORM, extend="both",
                                     extendfrac=0.055, orientation="horizontal",
                                     spacing="uniform",
                                     ticks=[BOUNDS[0], -0.10, 0, 0.10, BOUNDS[-1]])
    cb_e.ax.tick_params(labelsize=TICK, width=0.5, length=1.8, pad=1.0)
    cb_e.ax.set_xticklabels([("0" if b == 0 else f"{b:+.2f}").replace("-", MINUS)
                             for b in (BOUNDS[0], -0.10, 0, 0.10, BOUNDS[-1])])
    cb_e.outline.set_linewidth(0.5)
    cb_e.outline.set_edgecolor(INK)
    fig.text((W2 - PAD_R - 1.84) / W2, (Y_E - 0.225) / FIG_H, AX["gap"], fontsize=TICK,
             ha="left", va="baseline", color=INK)

    figure_legend(fig, [Patch(facecolor=BIN_COLORS[5], edgecolor=INK, lw=1.2,
                              label="With an import reduction"),
                        Patch(facecolor=BIN_COLORS[5], edgecolor=SAMPLE_EDGE, lw=0.3,
                              label=TERM["admitted"]),
                        Patch(facecolor=NOT_ADMITTED_FACE, edgecolor=NOT_ADMITTED_LINE, lw=0.3,
                              hatch="///", label=TERM["not_admitted"])],
                  ncol=3, y=(Y_E - 0.34) / FIG_H, handlelength=1.25, handleheight=0.9,
                  handletextpad=0.45, columnspacing=1.6, borderpad=0.0)

    # ---------------------------------------------------------------------------
    save(fig, "Fig7_episode_maps")

    out = P[["economy", "episode", "import_retention", "availability_retention", "gap"]].copy()
    out["contracting"] = [a in CONTRACTING[e] for a, e in zip(out.economy, out.episode)]
    edge = np.searchsorted(BOUNDS, out.gap.to_numpy(), side="right")
    out["bin_lower"] = [BOUNDS[i - 1] if i > 0 else np.nan for i in edge]
    out["bin_upper"] = [BOUNDS[i] if i < len(BOUNDS) else np.nan for i in edge]
    for g, i in zip(out.gap, edge):
        painted = CMAP(NORM(g))
        expect = (CMAP.get_under() if i == 0 else
                  CMAP.get_over() if i == len(BOUNDS) else CMAP(i - 1))
        assert np.allclose(painted, expect), (g, painted, expect)
        assert i == 0 or BOUNDS[i - 1] <= g, g
        assert i == len(BOUNDS) or g < BOUNDS[i], g
    out.to_csv(OUTPUTS / "fig7_source_data.csv", index=False)
    pd.DataFrame(comp_rows).to_csv(OUTPUTS / "fig7_composition_source_data.csv", index=False)

    print(f"  union sample {len(UNION)} economies: {' '.join(UNION)}")
    for ep in EPISODES:
        absent = [a for a in UNION if a not in GAP[ep]]
        print(f"  {ep:16s} {len(SAMPLE[ep])} admitted, {len(CONTRACTING[ep])} contracting, "
              f"not admitted: {' '.join(absent) if absent else 'none'}")
    print(f"  gap range {P.gap.min():+.3f} to {P.gap.max():+.3f}; bins {BOUNDS}")


def figures_s1_and_s5():
    """Draw figures s1 and s5."""
    D = pd.read_csv(TABLES / "primary_panel.csv")
    PRIM = set(D.country_iso2[D.population == "primary"])
    HUBS = set(D.country_iso2[D.population == "high-throughput"])
    ST = pd.read_csv(TABLES / "country_analytical_status.csv")
    FC = pd.read_csv(TABLES / "final_classification.csv")
    FCS = FC.set_index("country_iso2")
    PANEL = sorted(FC.country_iso2[FC.status == "accounting panel"])
    DESTS = sorted(FC.country_iso2[FC["class"] == "destination system"])
    EXPIMP = sorted(FC.country_iso2[FC["class"] == "crude-exporting importer"])
    PL = pd.read_csv(TABLES / "placebo_windows.csv")
    DEP = pd.read_csv(TABLES / "depletion_prospective.csv")
    DEP = DEP.rename(columns={"acute_draw_kbd": "draw_kbd",
                              "draw_share_of_intake": "draw_share"})
    DEP["display_name"] = [NAME.get(a, a) for a in DEP.country_iso2]
    DEP = DEP.sort_values("display_name")
    COV = pd.read_csv(TABLES / "baseline_days_of_cover.csv").rename(
        columns={"REF_AREA": "country_iso2"})
    NET_EXPORTERS = {"SA", "KW", "NG", "DZ", "NO", "VE"}
    SAMPLE = set(D.country_iso2)

    def num(v, places=3):
        """A signed number with a typographic minus, never a hyphen."""
        return f"{v:+.{places}f}".replace("-", chr(8722))

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
    save(fig, "FigS1_closure_residuals", OUTPUTS)

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
    save(fig, "FigS2_placebo_by_year", OUTPUTS)

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
    save(fig, "FigS3_reversion_diagnostic", OUTPUTS)

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
    save(fig, "FigS4_margins_by_economy", OUTPUTS)

    # ---------------------------------------------------------------------------
    G = pd.read_csv(TABLES / "fig6_source_data.csv")
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
    save(fig, "FigS5_corridor_dependence", OUTPUTS)

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

    CLs = pd.read_csv(TABLES / "identity_closure_test.csv")
    CLs["valid"] = CLs.A_supply_kbd > 0
    CLs["ratio"] = (CLs.residual_kbd.abs() / CLs.A_supply_kbd).where(CLs["valid"])
    Gs = CLs.groupby("country_iso2").agg(n=("valid", "sum"),
                                         c=("ratio", lambda s: s.mean(skipna=True)))
    TXs = pd.read_csv(TABLES / "pre_event_taxonomy.csv").set_index("country_iso2")
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


PUBLISHED = [figure_1_gulf_anatomy, figure_2_transport_boundary, figure_3_world_outcomes, figure_4_absorption, figures_5_and_8, figure_6_boundaries, figure_7_episode_maps, figures_s1_and_s5]


def draw_all():
    """Draw every published figure in order."""
    for fn in PUBLISHED:
        fn()
