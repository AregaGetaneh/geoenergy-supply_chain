"""FIGURE 7. The same measurement applied to three disruption episodes.

Composition rationale. A framework run once on one event shows that it can be run, not that
it discriminates. Three episodes with different transport signatures are the test, and a
small multiple is the only arrangement in which the reader performs that test by eye. The
three maps therefore share one binned diverging scale, one extent, one projection and one
country grammar. Therefore the only thing that changes between panels is the measurement itself.

a  Hormuz 2026, a chokepoint transit disruption at the loading end.
b  Russia 2022-23, a sanctions-driven origin switch in which the European sample is the
   directly affected population, not a bystander.
c  Red Sea 2023-24, a routing disruption that lengthened voyages without removing barrels.
d  Where the crude-import shortfall went, as shares of that episode's own shortfall.
e  Every economy in the union of the three episode samples, on the scale of the maps.

Admissibility is re-evaluated inside each episode's own five months. An economy admitted
in one episode need not be admitted in another. That is a property of the screen, not a hole
in the data, and the maps give it its own fill, not letting it collapse into the
neutral land tone. Three country states, three visual states:

    admitted in this episode             binned diverging fill, dark outline
    admitted in another episode only     hatched stone fill, dark outline
    outside the sample in all three      neutral land, hairline only

Panels a to c cover every destination system admitted in that episode. Panel d covers only the contracting
economies of each episode, since a share of a shortfall is undefined for an economy whose
imports did not contract; the subtitle says so, and panel e marks which cells those are.

Every plotted number is read from sourcedata/multi_episode_panel.csv and
sourcedata/multi_episode.csv. The script writes back everything it draws to
sourcedata/fig7_source_data.csv and sourcedata/fig7_composition_source_data.csv, and it
refuses to build if the counts, the contraction screen or the painted class of any value
disagree with those two inputs.

Cartographic documentation
  boundaries   Natural Earth 1:50m admin_0_countries
  coastline    Natural Earth 1:50m coastline, ocean 1:50m
  projection   Albers Equal Area, 20 E / 52 N, SP 40/65 N (main);
               Albers Equal Area, 120 E / 18 N, SP 0/36 N (Asia-Pacific inset)
  disputed     Natural Earth default sovereignty; no adjudication is implied
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

import os as _os, sys as _sys
_CODE = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_CODE] + [_os.path.join(_CODE, _d) for _d in sorted(_os.listdir(_CODE))
                           if _os.path.isdir(_os.path.join(_CODE, _d))]

from figstyle import (HEAD, INK, MUTED, RETENTION_CMAP, ROLE, RULE, SMALL, TINY, W2,
                       apply_style, figure_legend, map_layers, map_linework, save, SD,
                       MAP_LAND, MAP_LAND_EDGE, MAP_COAST,
                       AX, TERM, TICK, LABEL, TITLE, axis_label, panel_title, country_labels, thin_leader, SD_WRITE)

apply_style()

MINUS = chr(8722)                       # the true minus sign, never a hyphen
CONTRACTION_RULE = 0.95                 # the screen used in tre_multi_episode

# ---------------------------------------------------------------------------
P = pd.read_csv(SD / "multi_episode_panel.csv")
E = pd.read_csv(SD / "multi_episode.csv")

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
out.to_csv(SD_WRITE / "fig7_source_data.csv", index=False)
pd.DataFrame(comp_rows).to_csv(SD_WRITE / "fig7_composition_source_data.csv", index=False)

print(f"  union sample {len(UNION)} economies: {' '.join(UNION)}")
for ep in EPISODES:
    absent = [a for a in UNION if a not in GAP[ep]]
    print(f"  {ep:16s} {len(SAMPLE[ep])} admitted, {len(CONTRACTING[ep])} contracting, "
          f"not admitted: {' '.join(absent) if absent else 'none'}")
print(f"  gap range {P.gap.min():+.3f} to {P.gap.max():+.3f}; bins {BOUNDS}")
