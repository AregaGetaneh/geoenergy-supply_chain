"""Shared styling for the eight manuscript figures. One module, imported by every figure script.

This sits on top of tre_style, which already holds the semantic palette, the save helper and the
country-name map. Everything tre_style exports is re-exported here, so a script needs only this
import. What this module adds is the house rules the figures are being brought into line with.

    sizes        axis labels and legends 8 pt, ticks 7 pt, panel titles 9 pt, floor 7 pt
    type         one family across all eight
    labels       axis_label() builds "Quantity (unit)" in one place
    titles       panel_title() puts "(a) Short noun phrase" top left, bold letter
    terms        TERM holds the manuscript's wording for every legend entry
    output       600 dpi PNG beside the vector PDF

Type is serif, matching the Times body of the manuscript. The rule the figures are held to is one
family across all eight, not a particular family, and a figure that matches the text around it is
the better reading of it.

Country identifiers are chosen per panel, not globally. A bar panel with room spells the country
out, a map or a crowded scatter uses the ISO2 code, and no single panel mixes the two. That is
what country_labels() enforces.
"""

from __future__ import annotations

import re

import matplotlib as mpl

from tre_style import *            # noqa: F401,F403  palette, save, helpers, NAME
from tre_style import INK, MUTED, NAME, ROLE

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
