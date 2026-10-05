"""Shared plotting style: palette, type sizes, axis labels, panel titles, save helper.

Colour encodes the role a quantity plays in the crude balance, never decoration. Figures
are written at their final print size, so nothing is rescaled afterwards.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "data" / "panel"
MARKET = ROOT / "data" / "market"
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)

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
    """Write the figure at its native size as PDF and PNG."""
    d = Path(outdir) if outdir else OUTPUTS
    d.mkdir(parents=True, exist_ok=True)
    with mpl.rc_context({"savefig.bbox": "standard"}):
        fig.savefig(d / (stem + ".pdf"))
        fig.savefig(d / (stem + ".png"), dpi=400)
    plt.close(fig)
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


SERIF = True

# ---------------------------------------------------------------------------- type sizes
TICK, LABEL, TITLE = 7.0, 8.0, 9.0
FLOOR = 7.0                        # nothing below this at final print width

SERIF_STACK = ["Nimbus Roman", "Times New Roman", "Times", "DejaVu Serif"]
SANS_STACK = ["Arial", "Helvetica", "Nimbus Sans", "DejaVu Sans"]

def apply_style() -> None:
    """House style for every figure. Call once, at the top of a script."""
    mpl.rcParams.update({
        "figure.dpi": 200,
        "savefig.dpi": 600,
        "font.family": "serif" if SERIF else "sans-serif",
        "font.sans-serif": SANS_STACK,
        "font.serif": SERIF_STACK,
        "mathtext.fontset": "stix" if SERIF else "dejavusans",
        "font.size": LABEL,
        "axes.labelsize": LABEL,
        "axes.titlesize": TITLE,
        "xtick.labelsize": TICK,
        "ytick.labelsize": TICK,
        "legend.fontsize": LABEL,
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

# ---------------------------------------------------------------------------- axis labels
def axis_label(quantity: str, unit: str | None = None, log: bool = False) -> str:
    """"Quantity (unit)", first word capitalised and the rest lower case.

    A log axis says so inside the unit, which is where a reader looks for it.
    """
    q = quantity[0].upper() + quantity[1:]
    if unit is None:
        return q
    u = unit + (", log scale" if log else "")
    return f"{q} ({u})"

# The labels the brief fixes, so no script invents its own wording for a shared quantity.
AX = {
    "transits": axis_label("tanker transits", "count per month"),
    "exports": axis_label("crude exports", "kb/d"),
    "import_ret": axis_label("import ratio", "disruption / pre-disruption"),
    "avail_ret": axis_label("supply ratio", "disruption / pre-disruption"),
    "gap": axis_label("supply-import gap", "ratio points"),
    "gap_diff": axis_label("supply ratio minus import ratio", "ratio points"),
    "stock": axis_label("closing inventory", "days of refinery intake"),
    "horizon": axis_label("time to depletion", "months"),
    "volume": axis_label("volume", "kb/d"),
    "inv_adj": axis_label("inventory adjustment", "multiple of $L$"),
    "mult_L": axis_label("multiple of import shortage $L$", "ratio"),
    "sea_share": axis_label("share of extra-EU crude imports", "%"),
    "retention": axis_label("ratio to pre-disruption"),
    "retention_pct": axis_label("ratio to pre-disruption", "%"),
    "transit_ret": axis_label("transit ratio", "disruption / pre-disruption"),
    "month26": axis_label("month", "2026"),
}

# ---------------------------------------------------------------------------- legend wording
TERM = {
    "admitted": "Included",
    "not_admitted": "Not included",
    "residual_above": "Residual above 5%",
    "negative_balance": "Negative balance",
    "residual_na": "Residual not computable",
    "baseline": "Pre-disruption (Jan-Feb 2026)",
    "acute": "Disruption period (Apr-May 2026)",
    "transition": "Transition month (excluded)",
    "shortfall": "Import shortage $L$",
    "inventory": "Inventory adjustment",
    "export": "Export adjustment",
    "production": "Production adjustment",
    "availability": "Change in refinery supply",
    "imports_fell": "Imports fell by more than 5%",
    "imports_stable": "Imports stable or up",
    "reports_exports": "Reports crude exports",
}

COMPONENT_COLOR = {
    "shortfall": ROLE["import_shock"],
    "inventory": ROLE["inventory"],
    "production": ROLE["production"],
    "export": ROLE["export"],
    "availability": ROLE["availability"],
}

# ---------------------------------------------------------------------------- panel titles
def panel_title(ax, letter: str, title: str, pad: float = 4.0) -> None:
    """"(a) Short noun phrase", top left, bold letter, at most six words.

    The title carries no statistic and no sentence. Those belong in the caption.
    """
    phrase = re.sub(r"\s*\(\$?n\s*=\s*[^)]*\)\s*$", "", title)
    words = phrase.split()
    if len(words) > 6:
        raise ValueError("panel title over six words: %r" % phrase)
    ax.set_title(f"$\\bf{{({letter})}}$ {title}", fontsize=TITLE, loc="left", pad=pad,
                 color=INK)

def country_labels(iso2s, style: str):
    """Country identifiers for one panel, in one system.

    style="name"  full names, for a bar panel with room on the axis
    style="iso2"  two-letter codes, for a map or a crowded scatter

    The style is a panel-level decision and the call sites pass it explicitly, which is what stops
    a single panel carrying both forms.
    """
    if style not in ("name", "iso2"):
        raise ValueError("country label style must be 'name' or 'iso2', got %r" % style)
    if style == "iso2":
        return [str(a) for a in iso2s]
    return [NAME.get(a, a) for a in iso2s]

def thin_leader(ax, xy, xytext, **kw):
    """A short leader line, for a label that has to sit off its mark."""
    kw.setdefault("arrowprops", dict(arrowstyle="-", lw=0.5, color=MUTED,
                                     shrinkA=0.0, shrinkB=1.5))
    kw.setdefault("fontsize", TICK)
    kw.setdefault("color", INK)
    return ax.annotate(xy=xy, xytext=xytext, textcoords="offset points", **kw)

def plain_log_ticks(ax, axis="x", ticks=None):
    """Decade ticks written as plain numbers on a log axis.

    A reader who can see 1, 10, 100 does not need the words "log scale" in the label, so
    the scale is carried by the ticks and the label names only the quantity and its unit.
    """
    import matplotlib.ticker as mticker

    def fmt(v, _pos):
        if v >= 1:
            return f"{v:,.0f}"
        s = ("%.10f" % v).rstrip("0")
        return s

    a = ax.xaxis if axis == "x" else ax.yaxis
    get = ax.get_xlim if axis == "x" else ax.get_ylim
    put = ax.set_xlim if axis == "x" else ax.set_ylim
    lim = get()
    lo, hi = min(lim), max(lim)
    if ticks is not None:
        inside = [t for t in ticks if lo <= t <= hi]
        (ax.set_xticks if axis == "x" else ax.set_yticks)(inside)
    a.set_major_formatter(mticker.FuncFormatter(fmt))
    a.set_minor_formatter(mticker.NullFormatter())
    put(lim)
