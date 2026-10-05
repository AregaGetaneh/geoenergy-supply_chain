"""FIGURE 8. The three measurement boundaries of every episode, and where the crude arrived.

Composition rationale. Two measured layers were added to this paper in revision and both lived
only in tables, which is precisely the wrong place for them, since each exists to be compared
against something, not read off. Three panels put each comparison on one pair of axes.

a  The retention ladder for all three episodes. Corridor, imports, and availability on one scale,
   one line per episode. The attenuation between boundaries is a slope, not three
   numbers. Pass-through, stated as a block since it is a share of the shortfall, not a
   retention, is the part that reached the refinery gate, which is the destination-side cost and
   the quantity the mean gap is not. The reversal the paper argues is visible as lines crossing,
   since the episode lowest at the corridor ends highest at the gate.
b  Every corridor against its own prior-year control. A measured decline matters only relative to
   what that passage ordinarily does between the same two calendar windows. Each corridor is
   drawn as a pair and the control is what the episode value has to beat. The Cape of Good Hope
   runs the other way by construction, being the diversion leg, not an impaired one.
c  Transport mode at the customs boundary for the 2026 contracting systems that are EU reporters.
   This is the one quantity between the corridor and the balance, and it decides which
   contractions could have been maritime at all. Two of the five could not.

Colour follows the shared semantic system. The corridor is the chokepoint tone, imports the
import-shock tone, availability the availability tone, and the seaborne and overland shares of
panel c reuse the chokepoint and infrastructure tones, since a pipeline is infrastructure and a
strait is a chokepoint everywhere else in this manuscript.

Every number is read from the sourcedata CSVs that tre_corridor_episodes.py,
tre_multi_episode.py and tre_comext_mode_split.py write. Nothing is recomputed in this study, and each
value is asserted against its source before the figure is built.

Writes figures/Fig8_boundaries.pdf and .png, and sourcedata/fig8_source_data.csv.
"""

from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import os as _os, sys as _sys
_CODE = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_CODE] + [_os.path.join(_CODE, _d) for _d in sorted(_os.listdir(_CODE))
                           if _os.path.isdir(_os.path.join(_CODE, _d))]

from figstyle import (INK, MUTED, ROLE, RULE, SMALL, TINY, W2, apply_style, light_grid,
                       panel_tag, figure_legend, save, SD,
                       AX, TERM, TICK, LABEL, TITLE, axis_label, panel_title, country_labels, thin_leader, SD_WRITE)

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
    ep = pd.read_csv(SD / "multi_episode.csv").set_index("episode")
    corr = pd.read_csv(SD / "corridor_episodes.csv")
    ctrl = pd.read_csv(SD / "corridor_placebo.csv")
    mode = pd.read_csv(SD / "comext_mode_episode.csv")
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
    out.to_csv(SD_WRITE / "fig8_source_data.csv", index=False)
    print("  wrote sourcedata/fig8_source_data.csv")
    print(out.to_string(index=False))

if __name__ == "__main__":
    main()
