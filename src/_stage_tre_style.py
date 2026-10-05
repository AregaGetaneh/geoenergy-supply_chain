"""One semantic visual system for the whole TR-E manuscript.

Color encodes ROLE IN THE SUPPLY-CHAIN ACCOUNTING, never decoration. The palette and
typography carry forward the standard developed during the earlier visual benchmarking:
native publication dimensions, direct labels over legends, explicit no-data grammar,
restrained annotation, and one scientific story per figure.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
SD = ROOT / "paper_TRE" / "sourcedata"
FIGS = ROOT / "paper_TRE" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)
_FIG_OUT = os.environ.get("TRE_FIG_OUT")
SD_WRITE = (Path(_FIG_OUT) / "source_data") if _FIG_OUT else SD
if _FIG_OUT:
    SD_WRITE.mkdir(parents=True, exist_ok=True)

# Elsevier / TR-E column widths
W1 = 3.543      # 90 mm single column
W15 = 5.512     # 140 mm
W2 = 7.480      # 190 mm double column

# ---------------------------------------------------------------------------
ROLE = {
    "import_shock":   "#B03A2E",   # loss of reported crude imports
    "inventory":      "#1F6F8B",   # inventory draw or build
    "production":     "#3F9E72",   # domestic crude production
    "export":         "#C08A1E",   # export adjustment
    "availability":   "#22303C",   # resulting domestic crude availability
    "infrastructure": "#7A6A55",   # pipelines, terminals, physical assets
    "chokepoint":     "#C0392B",   # the strait itself
    "nodata":         "#E6E2DA",   # missing or non-admissible
    "outside":        "#F4F2EE",   # outside the analytical sample
}

# ---------------------------------------------------------------------------
STATUS_COLOR = {
    "accounting panel":           "#1F3A4D",   # L* 23
    "net exporter":               ROLE["export"],
    "closure above tolerance":    "#6E8494",   # L* 54
    "physically invalid balance": ROLE["import_shock"],
    "partial reporting":          "#B3A186",   # L* 67
    "residual not evaluable":     "#EDE8DF",   # L* 92, the unfilled-land value exactly
}
# ---------------------------------------------------------------------------
DEST_MARKER = ROLE["inventory"]       # destination system
HUB_MARKER = ROLE["export"]           # crude-exporting importer
FAINT_MARKER = "#98A4AC"              # admitted, but carrying no contraction

STATUS_LABEL = {
    "accounting panel":           "Included",
    "net exporter":               "Net crude exporters",
    "closure above tolerance":    "Residual above 5%",
    "physically invalid balance": "Negative balance",
    "partial reporting":          "Fewer than five months",
    "residual not evaluable":     "Residual not computable",
}

def on_fill(color):
    """Ink or white, whichever survives on top of a filled bar.

    The role palette now spans a wide lightness range. Therefore a fixed white value label is
    unreadable on the two light fills and a fixed dark one is unreadable on the two
    darkest. The threshold is on relative luminance, not on the raw hex.
    """
    r, g, b = mpl.colors.to_rgb(color)

    def lin(c):
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    y = 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)
    return INK if y > 0.34 else "white"
INK = "#1C2126"
MUTED = "#6B747C"
RULE = "#C9CDD2"
SEA = "#DCE6EC"
LAND = "#EFEBE3"

# diverging scale for retention-type quantities, centred on no change
RETENTION_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "tre_ret", ["#67001F", "#B2182B", "#D6604D", "#F4A582", "#F7F3EE",
                "#92C5DE", "#4393C3", "#2166AC", "#053061"])

# minimum effective type sizes. Figures survive journal reduction
TINY, SMALL, BODY, HEAD = 7.0, 7.5, 8.0, 9.0

NA_HATCH = "///"

def apply_style():
    mpl.rcParams.update({
        "figure.dpi": 200,
        "savefig.dpi": 400,
        # serif type, matching the Times body of the manuscript
        "font.family": "serif",
        "font.serif": ["Nimbus Roman", "Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "axes.linewidth": 0.6,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.2,
        "ytick.major.size": 2.2,
        "lines.linewidth": 1.1,
        "legend.frameon": False,
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

def panel_tag(ax, letter, dx=-0.085, dy=1.03, size=9):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=size,
            fontweight="bold", va="baseline", ha="left", color=INK)

def off_scale_panel(ax, label, hatch="////", face="#F3F0EA", edge="#B9B2A4",
                    label_loc="right"):
    """Mark an axes that does not share the scale of the panel above it.

    A warning in words is not enough when two stacked strips look like one chart: the
    reader's eye reads position, not the note. The hatched ground makes the strip a
    visibly different surface, and the break marks on the shared edge say where the
    scale is cut.
    """
    ax.set_facecolor(face)
    ax.patch.set_hatch(hatch)
    ax.patch.set_edgecolor(edge)
    ax.patch.set_linewidth(0.0)
    ax.patch.set_zorder(0)
    ax.set_title(label, fontsize=TINY, loc=label_loc, pad=3.0, color=INK)
    # break marks: two short parallel strokes across the top-left and top-right corners
    for x in (0.0, 1.0):
        for dy in (-0.012, 0.012):
            ax.plot([x - 0.013, x + 0.013], [1.0 + dy - 0.022, 1.0 + dy + 0.022],
                    transform=ax.transAxes, color=INK, lw=0.8, clip_on=False, zorder=12,
                    solid_capstyle="butt")
    return ax

def off_scale_band(ax, lo, hi, axis="y", face="#F3F0EA", edge="#B9B2A4", hatch="////"):
    """A strip inside an axes that carries no position on that axis.

    An observation of zero has no position on a logarithmic axis. Parking it at the
    foot of the scale makes it read as the smallest value, not as the absence
    of one. It goes here instead: a hatched ground, outside the plotted range, with
    break marks on the boundary it shares with the data. Set the limits first, since
    the break marks are placed from them.
    """
    if axis == "y":
        ax.axhspan(lo, hi, facecolor=face, edgecolor=edge, hatch=hatch, lw=0.0, zorder=1.2)
        v0, v1 = ax.get_ylim()
    else:
        ax.axvspan(lo, hi, facecolor=face, edgecolor=edge, hatch=hatch, lw=0.0, zorder=1.2)
        v0, v1 = ax.get_xlim()
    f = (hi - v0) / (v1 - v0)
    for a in (0.02, 0.98):
        for d in (-0.011, 0.011):
            xs, ys = [a + d - 0.011, a + d + 0.011], [f - 0.012, f + 0.012]
            if axis != "y":
                xs, ys = [f - 0.012, f + 0.012], [a + d - 0.011, a + d + 0.011]
            ax.plot(xs, ys, transform=ax.transAxes, color=INK, lw=0.7, clip_on=False,
                    zorder=12, solid_capstyle="butt")
    return ax

LEGEND_Y = 0.008

def figure_legend(fig, handles, ncol, y=LEGEND_Y, fontsize=None, **kw):
    """The one legend placement this paper uses.

    Every multi-panel figure that is not a map carries a single horizontal key below
    the panel row, outside every axes and without a frame. Keys parked in different
    corners of different panels make the reader hunt for the grammar of each panel
    separately, and a key inside the data area competes with the observations.
    """
    opts = dict(labelspacing=0.35, handletextpad=0.45, columnspacing=1.7)
    opts.update(kw)
    return fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, y),
                      ncol=ncol, fontsize=fontsize or TINY, frameon=False, **opts)

def light_grid(ax, axis="y"):
    ax.grid(axis=axis, color=RULE, lw=0.4, alpha=0.7)
    ax.set_axisbelow(True)

def save(fig, stem, outdir=None):
    """Native-size save: no tight bbox. Therefore the declared width is the real width.

    TRE_FIG_OUT redirects every figure to one directory, whatever the caller asked for.
    It exists so that a relabelling pass can be rendered to a candidate folder while the
    published figures stay locked and untouched.
    """
    env = os.environ.get("TRE_FIG_OUT")
    d = Path(env) if env else (outdir or FIGS)
    d.mkdir(parents=True, exist_ok=True)
    with mpl.rc_context({"savefig.bbox": "standard"}):
        fig.savefig(d / f"{stem}.pdf")
        fig.savefig(d / f"{stem}.png", dpi=400)
    w, h = fig.get_size_inches()
    plt.close(fig)
    print(f"  {stem}: {w:.3f} x {h:.3f} in = {25.4*w:.0f} x {25.4*h:.0f} mm")

NAME = {
    "US": "United States", "DE": "Germany", "NL": "Netherlands", "ES": "Spain",
    "GB": "United Kingdom", "ID": "Indonesia", "TR": "Türkiye", "BE": "Belgium",
    "PL": "Poland", "GR": "Greece", "SE": "Sweden", "FI": "Finland",
    "PT": "Portugal", "AT": "Austria", "DK": "Denmark", "CZ": "Czechia",
    "BG": "Bulgaria", "SK": "Slovakia", "HU": "Hungary", "CH": "Switzerland",
    "IE": "Ireland", "HR": "Croatia", "JP": "Japan", "KR": "Korea", "CN": "China",
    "IN": "India", "SA": "Saudi Arabia", "KW": "Kuwait", "NG": "Nigeria",
    "BN": "Brunei", "NO": "Norway", "IT": "Italy", "FR": "France",
}
ISO3 = {
    "US": "USA", "DE": "DEU", "NL": "NLD", "ES": "ESP", "GB": "GBR", "ID": "IDN",
    "TR": "TUR", "BE": "BEL", "PL": "POL", "GR": "GRC", "SE": "SWE", "FI": "FIN",
    "PT": "PRT", "AT": "AUT", "DK": "DNK", "CZ": "CZE", "BG": "BGR", "SK": "SVK",
    "HU": "HUN", "CH": "CHE", "IE": "IRL", "HR": "HRV",
}

# ---------------------------------------------------------------------------
MAP_SEA = "#CBDCE6"          # water, clearly cooler and darker than any land
MAP_LAND = "#EDE8DF"         # land with no analytical role
MAP_LAND_EDGE = "#FFFFFF"    # hairline between neighbours on filled land
MAP_COAST = "#5C6B75"        # land against water, the strongest line on the map
MAP_BORDER = "#94A2AC"       # political boundary, subordinate to the coast
MAP_GRAT = "#B9C6CF"         # graticule

def map_layers(ax, extent=None, crs=None, graticule=None, coast_lw=0.5,
               border_lw=0.32, scale="50m"):
    """Water, land and the line work every map in this paper shares.

    Drawn in three passes so the ordering is explicit: water and land as the
    ground at the bottom, thematic country fills added by the caller in
    between, and the coast and borders last so no fill can bury them.
    """
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    crs = crs or ccrs.PlateCarree()
    if extent is not None:
        ax.set_extent(extent, crs=crs)
    ax.add_feature(cfeature.OCEAN.with_scale(scale), facecolor=MAP_SEA, zorder=0)
    ax.add_feature(cfeature.LAND.with_scale(scale), facecolor=MAP_LAND, zorder=1)
    if graticule:
        gl = ax.gridlines(crs=crs, draw_labels=False, xlocs=graticule[0],
                          ylocs=graticule[1], color=MAP_GRAT, linewidth=0.3,
                          alpha=0.9, zorder=4)
        gl.top_labels = gl.right_labels = False
    return ax

def map_linework(ax, coast_lw=0.5, border_lw=0.32, scale="50m"):
    """Coast and political boundaries, added after the thematic fills."""
    import cartopy.feature as cfeature

    ax.add_feature(cfeature.BORDERS.with_scale(scale), edgecolor=MAP_BORDER,
                   linewidth=border_lw, zorder=6)
    ax.add_feature(cfeature.COASTLINE.with_scale(scale), edgecolor=MAP_COAST,
                   linewidth=coast_lw, zorder=7)
    ax.spines["geo"].set_edgecolor(MAP_COAST)
    ax.spines["geo"].set_linewidth(0.6)
