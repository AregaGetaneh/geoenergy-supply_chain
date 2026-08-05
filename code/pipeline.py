"""End-to-end reproduction pipeline for GeoEnergy-SCRO.

Runs the full study. It builds the network, solves the stochastic MIP under every
regime and scenario class, and writes all headline tables and figures into
outputs/. This is the script form of the original analysis notebook; run it from
the repository root:

    python code/pipeline.py

A full run takes roughly an hour. Figures are written as vector PDFs (headless).
"""
import os
import sys

# Force UTF-8 stdout so table prints with symbols (checkmarks, x, dashes) do not
# crash on a Windows cp1252 console. No effect on Linux, which is UTF-8 already.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
os.chdir(_ROOT)
sys.path.insert(0, _HERE)
os.environ.setdefault("GEOENERGY_SOLVER", "appsi_gurobi")

import matplotlib
matplotlib.use("Agg")  # headless: savefig works, no GUI needed


# ---------------------------------------------------------------------------
# cell 0
# ---------------------------------------------------------------------------
import sys, os, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import pyomo.environ as pyo
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from IPython.display import display, Markdown, Image
warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 120)

# Locate the code/ folder (which holds model.py and analysis.py) and the
# repository root (which holds data/ and outputs/, one level above code/).
CODE_DIR = Path.cwd().resolve()
if not (CODE_DIR / "model.py").exists() and (CODE_DIR / "code" / "model.py").exists():
    CODE_DIR = CODE_DIR / "code"          # launched from the repository root
PROJECT_ROOT = CODE_DIR.parent if (CODE_DIR.parent / "outputs").exists() else CODE_DIR
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))     # so `import model` / `import analysis` work
os.chdir(PROJECT_ROOT)                     # so "outputs"/"data" resolve at the root

# Optimization stack (config, data, model, solve) and post-solve analysis
# (results, figures, validation, robustness) as two flat modules.
from model import (
    SECTORS, NUM_STAGES, STAGE_DAYS, POLICIES, BETA, THRESHOLDS,
    PAL, REGION_COLORS, REGIME_COLORS, SECTOR_COLORS, SEVERITY_COLORS,
    CASCADE_COLORS, SERIES2, COUNTRY_COLORS,
    build_all, EnergyNetwork, build_model, build_scenario_set,
    compute_cascade_losses, clear_cache)
import analysis as res
import analysis as rob
from analysis import (
    compute_escvi, compute_system_losses, model_variant_comparison,
    regime_comparison, attribution_table, timing_delay_mip,
    duration_sensitivity_mip, compute_diagnostics,
    generate_chokepoint_map, export_country_losses, make_country_loss_map,
    export_optimized_flows, make_flow_sankey)

# Plot style (EJOR). Font sizes >= 13 so figures stay legible when scaled in LaTeX.
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 14, "axes.labelsize": 15, "axes.titlesize": 15,
    "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 13,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.grid": True, "grid.alpha": 0.20, "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
})

print(f"Project root: {PROJECT_ROOT}")
_n = clear_cache()
print(f"Cleared {_n} cached solves; every result recomputes from the solver.")

# ---------------------------------------------------------------------------
# cell 1
# ---------------------------------------------------------------------------
# Build all network components
countries, sources, chokepoints, processing, arcs = build_all()
network = EnergyNetwork(sources, chokepoints, processing, countries, arcs)

print(network.summary())
print(f"Policy instruments: {len(POLICIES)}")
print(f"Sectors per country: {len(SECTORS)}")
print(f"Stage horizon: {NUM_STAGES} stages × {STAGE_DAYS} days = {NUM_STAGES * STAGE_DAYS} days")

# ---------------------------------------------------------------------------
# cell 2
# ---------------------------------------------------------------------------
# Display country calibration data
country_df = pd.DataFrame([{
    'Code': c.code, 'Name': c.name, 'Region': c.region,
    'Demand (mbd)': c.daily_oil_mbd,
    'Hormuz dep.': c.hormuz_dep, 'Malacca dep.': c.malacca_dep,
    'Suez dep.': c.suez_dep, 'SPR (days)': c.spr_days,
    'GDP ($B)': c.gdp_B, 'Pop (M)': c.pop_M, 'Renew (%)': c.renew_pct,
} for c in countries.values()])

display(Markdown("### Table: Country calibration parameters (53 countries)"))
display(country_df)
country_df.to_csv('outputs/data_countries.csv', index=False)
print(f"\nTotal global demand: {country_df['Demand (mbd)'].sum():.1f} mbd")

# ---------------------------------------------------------------------------
# cell 3
# ---------------------------------------------------------------------------
# Display network arc summary
arc_df = pd.DataFrame([{
    'ID': a.aid, 'Origin': a.origin, 'Destination': a.destination,
    'Capacity (mbd)': round(a.capacity_mbd, 2), 'Unit cost': round(a.unit_cost, 2),
    'Chokepoint': a.chokepoint if a.chokepoint else '-',
    'Bypass': '✓' if a.is_bypass else '',
} for a in arcs])

display(Markdown("### Table: Network arc summary"))
print(f"Normal arcs: {sum(1 for a in arcs if not a.is_bypass)}")
print(f"Bypass arcs: {sum(1 for a in arcs if a.is_bypass)}")
print(f"Total arcs: {len(arcs)}")
display(arc_df.head(20))
arc_df.to_csv('outputs/data_arcs.csv', index=False)

# ---------------------------------------------------------------------------
# cell 4
# ---------------------------------------------------------------------------
# Display chokepoint, source and policy data
print("Chokepoints:")
for cp in chokepoints.values():
    print(f"  {cp.name}: {cp.throughput_mbd} mbd throughput, "
          f"{cp.bypass_capacity_mbd} mbd bypass capacity, "
          f"{cp.bypass_cost_premium}x cost premium")

print("\nSource regions:")
for src in sources.values():
    print(f"  {src.name}: {src.capacity_mbd} mbd, "
          f"Hormuz-routed: {src.hormuz_routed_frac*100:.0f}%")

print("\nPolicy instruments:")
for p in POLICIES:
    irr = " [IRREVERSIBLE]" if p.irreversible else ""
    print(f"  {p.pid} - {p.name}: activation cost ${p.cost}B/stage{irr}")

# ---------------------------------------------------------------------------
# cell 5
# ---------------------------------------------------------------------------
# Geospatial overview: EIA-style world oil transit chokepoint map
# (Plotly basemap, gold throughput bubbles, dark-red sea-lane arrows, no Cartopy).
_p = generate_chokepoint_map('outputs')
print('Saved', _p)
display(Image(_p.replace('.pdf', '.png')))

# ---------------------------------------------------------------------------
# cell 6
# ---------------------------------------------------------------------------
# Generate the full decision-dependent scenario tree for each disruption class
disruption_classes = [
    ("partial_restriction", "Partial Restriction"),
    ("full_closure",        "Full Closure"),
    ("multi_chokepoint",    "Multi-Chokepoint"),
    ("attrition_mild",      "Attrition (Mild)"),
    ("attrition_moderate",  "Attrition (Moderate)"),
    ("attrition_severe",    "Attrition (Severe)"),
]

scenario_sets = {}
print(f"{'Class':<25} {'Tree paths':>10} {'Mean avail t=0':>15}")
print("-" * 55)

for dc_key, dc_name in disruption_classes:
    scen = build_scenario_set(dc_key)
    scenario_sets[dc_key] = scen
    mean_avail = sum(s.availability[0] * s.prob for s in scen)
    print(f"{dc_name:<25} {len(scen):>10} {mean_avail:>15.3f}")
    sc_df = pd.DataFrame([{
        'omega': s.omega, 'prob': round(s.prob, 6),
        **{f'state_t{t}': s.states[t] for t in range(NUM_STAGES)},
        **{f'avail_t{t}': s.availability[t] for t in range(NUM_STAGES)},
    } for s in scen])
    sc_df.to_csv(f'outputs/scenarios_{dc_key}.csv', index=False)

# ---------------------------------------------------------------------------
# cell 7
# ---------------------------------------------------------------------------
# Build the Pyomo model to report its size, then free it. The model is large
# (hundreds of MB), keeping it resident would starve later cells for memory.
import gc

scenarios_fc = scenario_sets["full_closure"]
model = build_model(network, scenarios_fc, "CP_HORMUZ", enable_cascade=True)

n_vars = sum(1 for _ in model.component_data_objects(pyo.Var))
n_cons = sum(1 for _ in model.component_data_objects(pyo.Constraint, active=True))
print("Model statistics (full decision-dependent tree):")
print(f"  Total variables:   {n_vars:,}")
print(f"  Constraints:       {n_cons:,}")
print(f"  Scenarios:         {len(scenarios_fc)}")
print(f"  Stages:            {NUM_STAGES}")
print(f"  Countries:         {len(network.country_nodes)}")
print(f"  Arcs:              {len(network.arcs)}")
print(f"  Sectors:           {len(SECTORS)}")
print()
print("Variable classes:")
for v in model.component_objects(pyo.Var):
    print(f"  {v.name}: {len(v)} elements")

# Release the model before the figure cells that follow.
del model
gc.collect()

# ---------------------------------------------------------------------------
# cell 8
# ---------------------------------------------------------------------------
# Compute ESCVI for all 53 countries (Hormuz scenario)
escvi_df = compute_escvi(countries, "hormuz")
escvi_df.to_csv('outputs/table_escvi.csv', index=False)

display(Markdown("### Table 3: Top 12 nations ranked by ESCVI (100-day full closure)"))
display(escvi_df.head(12))

# ---------------------------------------------------------------------------
# cell 9
# ---------------------------------------------------------------------------
# Figure 4: Vulnerability heatmap
top20 = escvi_df.head(20)
dimensions = ["Import Dep.", "Reserve Vuln.", "Alt. Gap", "Cascade Sev."]
data = top20[dimensions].values

fig, ax = plt.subplots(figsize=(10, 7))
im = ax.imshow(data, cmap='YlOrRd', aspect='auto', vmin=0, vmax=100)
ax.set_xticks(range(len(dimensions)))
ax.set_xticklabels(dimensions, fontsize=13)
ax.set_yticks(range(len(top20)))
ax.set_yticklabels(top20["Country"].values, fontsize=13)

for i in range(len(top20)):
    for j in range(len(dimensions)):
        val = data[i, j]
        color = 'white' if val > 60 else 'black'
        ax.text(j, i, f'{val:.0f}', ha='center', va='center',
                fontsize=13, color=color)

plt.colorbar(im, ax=ax, label='Vulnerability score', shrink=0.8)
plt.tight_layout()
plt.savefig('outputs/fig_vulnerability_heatmap.pdf')
plt.show()

# ---------------------------------------------------------------------------
# cell 10
# ---------------------------------------------------------------------------
# Figure 4: Hormuz exposure vs SPR coverage (publication version, reads the
# calibration inputs in outputs/data_countries.csv, no solver needed).
res.make_exposure_figure()
plt.show()

# ---------------------------------------------------------------------------
# cell 11
# ---------------------------------------------------------------------------
# Cross-sector cascade decomposition for Pakistan and India
pak = countries["PAK"]
ind = countries["IND"]

cascade_res_pak = compute_cascade_losses(pak, pak.hormuz_dep, 100)
cascade_res_ind = compute_cascade_losses(ind, ind.hormuz_dep, 100)

# Pakistan decomposition (thin-buffer importer)
cascade_rows = []
for s in SECTORS:
    r = cascade_res_pak[s]
    cascade_rows.append({
        'Sector': s.capitalize(),
        'Direct (%)': r['direct'], 'Cascade (%)': r['cascade'],
        'Total (%)': r['total'],
    })
cascade_df = pd.DataFrame(cascade_rows)
cascade_df.to_csv('outputs/table_cascade_decomposition.csv', index=False)

display(Markdown("### Sectoral output loss decomposition - Pakistan (68% Hormuz dep., 12-day SPR)"))
display(cascade_df)

# India decomposition (deep-buffer importer, for comparison)
cascade_rows_ind = []
for s in SECTORS:
    r = cascade_res_ind[s]
    cascade_rows_ind.append({
        'Sector': s.capitalize(),
        'Direct (%)': r['direct'], 'Cascade (%)': r['cascade'],
        'Total (%)': r['total'],
    })
cascade_df_ind = pd.DataFrame(cascade_rows_ind)
cascade_df_ind.to_csv('outputs/table_cascade_decomposition_india.csv', index=False)

display(Markdown("### Comparison: India (64% Hormuz dep., 74-day SPR)"))
display(cascade_df_ind)

# ---------------------------------------------------------------------------
# cell 12
# ---------------------------------------------------------------------------
# Figure: cascade decomposition - two country cases (Pakistan vs India)
# Bottleneck amplification is zero for both (calibrated stress stays below tau_s),
# so only the direct and cascade components are shown.
panels = [(cascade_res_pak, "(a) Pakistan"), (cascade_res_ind, "(b) India")]
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
sectors = [s.capitalize() for s in SECTORS]
x = np.arange(len(sectors)); w = 0.55
for idx, (panel_res, title) in enumerate(panels):
    ax = axes[idx]
    direct = np.array([panel_res[s]["direct"] for s in SECTORS])
    cascade = np.array([panel_res[s]["cascade"] for s in SECTORS])
    total = [panel_res[s]["total"] for s in SECTORS]
    ax.bar(x, direct, w, color=CASCADE_COLORS["direct"], label="Direct oil shortfall",
           edgecolor="white", linewidth=0.6)
    ax.bar(x, cascade, w, bottom=direct, color=CASCADE_COLORS["cascade"],
           label="Inter-sector cascade", edgecolor="white", linewidth=0.6)
    for i, t in enumerate(total):
        ax.text(i, t + 0.5, f"{t:.1f}%", ha="center", fontsize=13, color=PAL["navy"])
    ax.set_xticks(x); ax.set_xticklabels(sectors, fontsize=13, rotation=25, ha="right")
    ax.set_title(title, loc="left", fontsize=13)
    if idx == 0:
        ax.set_ylabel("Output loss (%)")
        ax.legend(loc="upper right", fontsize=13, frameon=False)
plt.tight_layout()
plt.savefig("outputs/fig_cascade_stack.pdf", dpi=300)
plt.show()

# ---------------------------------------------------------------------------
# cell 13
# ---------------------------------------------------------------------------
# Show cascade dependency matrix (β coefficients)
display(Markdown("### Inter-sector cascade dependency matrix (β)"))
beta_matrix = pd.DataFrame(0.0, index=SECTORS, columns=SECTORS)
for s in SECTORS:
    for sp, val in BETA.get(s, {}).items():
        beta_matrix.loc[s, sp] = val

try:
    display(beta_matrix.style.background_gradient(cmap='YlOrRd', vmin=0, vmax=0.35).format("{:.2f}"))
except (AttributeError, ImportError):
    # jinja2 not installed, fall back to plain display
    display(beta_matrix)

# Show thresholds
display(Markdown("### Sector bottleneck thresholds (θ, μ)"))
for s in SECTORS:
    theta, mu = THRESHOLDS[s]
    print(f"  {s:>15}: θ = {theta:.2f}, μ = {mu:.1f}")

# ---------------------------------------------------------------------------
# cell 14
# ---------------------------------------------------------------------------
# Regional loss decomposition (MIP)
_f = 'outputs/table_regional_losses_mip.csv'
reg_df = res.regional_losses()
reg_df.to_csv(_f, index=False)
display(Markdown("### Regional expected damage (USD billion, 100-day full Hormuz closure) -- MIP"))
display(reg_df)

# ---------------------------------------------------------------------------
# cell 15
# ---------------------------------------------------------------------------
# Table 4b: Top 10 countries by absolute loss (MIP)
_f = 'outputs/table_top_losses_mip.csv'
top_df = res.top_absolute_losses(top_n=10)
top_df.to_csv(_f, index=False)
display(Markdown("### Table 4b: Top 10 countries by absolute expected damage (100-day full Hormuz closure) -- MIP"))
display(top_df[['Country', 'Region', 'Inaction loss ($B)', 'Early-stage loss ($B)', 'Saving ($B)', 'Global share (%)']])

# ---------------------------------------------------------------------------
# cell 16
# ---------------------------------------------------------------------------
# Figure 9 data: efficiency-equity frontiers by scenario severity (MIP), each
# normalized to its own efficiency-only optimum so the trade-off shapes compare.
_classes = ['partial_restriction', 'full_closure', 'multi_chokepoint']
_frames = [res.equity_frontier(scenario_class=_cls).sort_values('Equity weight')
           for _cls in _classes]
pd.concat(_frames, ignore_index=True).to_csv('outputs/data_pareto_frontier_mip.csv', index=False)
display(pd.concat(_frames, ignore_index=True))
res.make_pareto_figure()      # publication version, reads the CSV just written
plt.show()

# ---------------------------------------------------------------------------
# cell 17
# ---------------------------------------------------------------------------
# Table 9: War-of-attrition analysis (MIP)
_f = 'outputs/table_attrition_mip.csv'
attr_df = regime_comparison(
    scenario_classes=("attrition_mild", "attrition_moderate", "attrition_severe"), verbose=False)
attr_df.to_csv(_f, index=False)
display(Markdown("### Table 9: War-of-attrition scenario analysis (100-day horizon, USD billion) -- MIP"))
display(attr_df)

# ---------------------------------------------------------------------------
# cell 18
# ---------------------------------------------------------------------------
# Table 10: Transferability across chokepoints (MIP)
_f = 'outputs/table_transferability_mip.csv'
trans_df = res.transferability()
trans_df.to_csv(_f, index=False)
display(Markdown("### Table 10: Transferability across primary global chokepoints (100-day full closure) -- MIP"))
display(trans_df)

# ---------------------------------------------------------------------------
# cell 19
# ---------------------------------------------------------------------------
# Table 11: Model variant comparison
mv_df = model_variant_comparison(countries)
mv_df.to_csv('outputs/table_model_variants.csv', index=False)

display(Markdown("### Table 11: Estimated losses under simplistic vs. advanced model variants (USD billion)"))
display(mv_df)

# ---------------------------------------------------------------------------
# cell 20
# ---------------------------------------------------------------------------
# Configuration for the MIP-based analyses below. The full decision-dependent
# disruption-path tree is always solved, target_count only sets cache-key
# granularity (6 for a quick check, 15 default, 30 for the final run).
MIP_TARGET_COUNT = 15
MIP_SCENARIO_CLASSES = ["partial_restriction", "full_closure", "multi_chokepoint"]

# ---------------------------------------------------------------------------
# cell 21
# ---------------------------------------------------------------------------
_f = 'outputs/table_scenario_comparison_mip.csv'
df_regime_mip = regime_comparison(scenario_classes=MIP_SCENARIO_CLASSES,
                                  target_count=MIP_TARGET_COUNT, verbose=True)
df_regime_mip.to_csv(_f, index=False)
display(Markdown('### Table 4 (MIP-based): regime comparison across scenario classes'))
display(df_regime_mip)

# ---------------------------------------------------------------------------
# cell 22
# ---------------------------------------------------------------------------
# Figure 8: per-scenario expected loss under no intervention, the reactive
# bundle, and the early-stage optimum (publication version).
res.make_scenario_bars_figure(df_regime_mip)
plt.show()

# ---------------------------------------------------------------------------
# cell 23
# ---------------------------------------------------------------------------
# Figure: loss composition (model output)
#   (a) total loss by sector across representative importers
#   (b) per-stage loss by country over the 100-day horizon
REPS = [('CHN', 'China'), ('IND', 'India'), ('JPN', 'Japan'),
        ('KOR', 'S. Korea'), ('PAK', 'Pakistan'), ('DEU', 'Germany')]
sector_color = SECTOR_COLORS
country_color = COUNTRY_COLORS
m0 = res.solve(())  # no intervention, full closure, 100-day benchmark
has = hasattr(m0, 'Delta')
pw = {w: pyo.value(m0.prob[w]) for w in m0.OMEGA}
stages = list(m0.T)
repcodes = {c for c, _ in REPS}

totals = {code: {s: res.sector_damage(m0, code)[s]['total'] for s in SECTORS}
          for code, _ in REPS}

def stage_loss_country(code):
    out = np.zeros(len(stages))
    for s in SECTORS:
        pi = pyo.value(m0.pi_ns[code, s]); eta = pyo.value(m0.eta_ns[code, s]) if has else 0.0
        for i, t in enumerate(stages):
            for w in m0.OMEGA:
                out[i] += pw[w] * pi * pyo.value(m0.h[code, s, t, w]) / 1000.0
                if has: out[i] += pw[w] * eta * pyo.value(m0.Delta[code, s, t, w]) / 1000.0
    return out

band = {name: stage_loss_country(code) for code, name in REPS}
rest = np.zeros(len(stages))
for code in m0.C:
    if code not in repcodes:
        rest += stage_loss_country(code)
band['Rest of world'] = rest
order = ['China', 'India', 'Japan', 'S. Korea', 'Pakistan', 'Germany', 'Rest of world']

fig, (axa, axb) = plt.subplots(1, 2, figsize=(13, 5.4))
# (a) cross-country, stacked by sector
xc = np.arange(len(REPS)); bottom = np.zeros(len(REPS))
for s in SECTORS:
    vals = np.array([totals[code][s] for code, _ in REPS])
    axa.bar(xc, vals, bottom=bottom, width=0.62, label=s.title(),
            color=sector_color[s], edgecolor='black', linewidth=0.4)
    bottom += vals
for i, _ in enumerate(REPS):
    axa.annotate(f'{bottom[i]:.0f}', xy=(xc[i], bottom[i]), xytext=(0, 3),
                 textcoords='offset points', ha='center', fontsize=13)
axa.set_xticks(xc); axa.set_xticklabels([n for _, n in REPS], rotation=20, ha='right')
axa.set_ylabel('Expected output loss ($B)'); axa.set_title('(a)', loc='left', fontsize=13)
axa.legend(loc='upper right', frameon=False, ncol=2, fontsize=13)
# (b) per-stage, stacked by country
edges = [0, 25, 50, 75, 100]; bottom = np.zeros(len(stages))
for n in order:
    y = band[n]; bs = np.append(bottom, bottom[-1]); ys = np.append(y, y[-1])
    axb.fill_between(edges, bs, bs + ys, step='post', color=country_color[n],
                     alpha=0.92, linewidth=0.4, edgecolor='white', label=n)
    bottom += y
axb.set_xlim(0, 100); axb.set_ylim(bottom=0); axb.set_xticks(edges)
axb.set_xlabel('Disruption day'); axb.set_ylabel('Expected loss in stage ($B / 25 days)')
axb.set_title('(b)', loc='left', fontsize=13)
axb.legend(loc='upper right', frameon=False, ncol=2, fontsize=13)
plt.tight_layout()
plt.savefig('outputs/fig_sector_country_stack.pdf')
plt.show()

import gc; del m0; gc.collect()

# ---------------------------------------------------------------------------
# cell 24
# ---------------------------------------------------------------------------
_f = 'outputs/table_policy_attribution_mip.csv'
df_attribution = attribution_table(scenario_class='full_closure',
                                   target_count=MIP_TARGET_COUNT, verbose=True)
df_attribution.to_csv(_f, index=False)
display(Markdown('### Consolidated policy attribution (MIP, 64 subset solves)'))
display(df_attribution)

# ---------------------------------------------------------------------------
# cell 25
# ---------------------------------------------------------------------------
# Figure: Side-by-side Shapley and LOO bars (excludes Total row)
df_plot = df_attribution.iloc[:-1].copy()
df_plot = df_plot.sort_values('Shapley ($B)', ascending=True)

fig, ax = plt.subplots(figsize=(11, 5.5))
y = np.arange(len(df_plot))
h = 0.38
b1 = ax.barh(y - h/2, df_plot['Shapley ($B)'], height=h,
             label='Shapley value', color=SERIES2[0],
             edgecolor='black', linewidth=0.4)
b2 = ax.barh(y + h/2, df_plot['LOO ($B)'], height=h,
             label='Leave-one-out', color=SERIES2[1],
             edgecolor='black', linewidth=0.4)
ax.set_yticks(y)
ax.set_yticklabels(df_plot['Instrument'])
ax.set_xlabel('Contribution to expected-loss reduction ($B)')
ax.axvline(0, color='black', linewidth=0.6)
ax.legend(loc='lower right', frameon=False)
for bar in list(b1) + list(b2):
    val = bar.get_width()
    if val > 0.5:
        ax.annotate(f'{val:.1f}',
                    xy=(val, bar.get_y() + bar.get_height()/2),
                    xytext=(4, 0), textcoords='offset points',
                    va='center', fontsize=13)
plt.tight_layout()
plt.savefig('outputs/fig_policy_attribution_mip.pdf')
plt.show()

# ---------------------------------------------------------------------------
# cell 26
# ---------------------------------------------------------------------------
_f = 'outputs/table_timing_delay_mip.csv'
df_timing_mip = timing_delay_mip(scenario_class='full_closure',
                                 target_count=MIP_TARGET_COUNT, verbose=True)
df_timing_mip.to_csv(_f, index=False)
display(Markdown('### Cost of delaying activation by stage (MIP)'))
display(df_timing_mip)

# ---------------------------------------------------------------------------
# cell 27
# ---------------------------------------------------------------------------
# Figure: timing-delay curve
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(df_timing_mip['Delay (days)'],
        df_timing_mip['Expected loss ($B)'],
        marker='o', markersize=9, color=PAL['navy'], linewidth=2.0)
for _, row in df_timing_mip.iterrows():
    label = row['Penalty vs immediate']
    if label != '---':
        ax.annotate(label,
                    xy=(row['Delay (days)'], row['Expected loss ($B)']),
                    xytext=(5, 8), textcoords='offset points',
                    fontsize=13, color=PAL['navy'])
ax.set_xlabel('Delay before policy activation (days)')
ax.set_ylabel('Expected loss ($B)')
plt.tight_layout()
plt.savefig('outputs/fig_timing_delay_mip.pdf')
plt.show()

# ---------------------------------------------------------------------------
# cell 28
# ---------------------------------------------------------------------------
_f = 'outputs/table_duration_sensitivity_mip.csv'
df_duration_mip = duration_sensitivity_mip(durations=[40, 80, 100, 200, 300, 365],
                                           scenario_class='full_closure',
                                           target_count=MIP_TARGET_COUNT, verbose=True)
df_duration_mip.to_csv(_f, index=False)
display(Markdown('### Duration sensitivity from the MIP'))
display(df_duration_mip)

# ---------------------------------------------------------------------------
# cell 29
# ---------------------------------------------------------------------------
# Figure: duration-sensitivity curves
fig, ax = plt.subplots(figsize=(9.5, 5.5))
ax.plot(df_duration_mip['Duration (days)'], df_duration_mip['No intervention'],
        marker='s', color=REGIME_COLORS['No intervention'], linewidth=2, label='No intervention')
ax.plot(df_duration_mip['Duration (days)'], df_duration_mip['Reactive bundle'],
        marker='o', color=REGIME_COLORS['Reactive bundle'], linewidth=2, label='Reactive bundle')
ax.plot(df_duration_mip['Duration (days)'], df_duration_mip['Full proactive'],
        marker='D', color=REGIME_COLORS['Full proactive'], linewidth=2, label='Full proactive')
ax.set_xlabel('Disruption duration (days)')
ax.set_ylabel('Expected loss ($B)')
ax.legend(loc='upper left', frameon=False)
plt.tight_layout()
plt.savefig('outputs/fig_duration_sensitivity_mip.pdf')
plt.show()

# ---------------------------------------------------------------------------
# cell 30
# ---------------------------------------------------------------------------
# Cascade-coefficient robustness: beta scaled +/-20% around the calibration.
beta_robust = rob.cascade_beta_robustness(scales=[0.8, 1.0, 1.2])
beta_robust.to_csv('outputs/table_cascade_robustness.csv', index=False)
display(Markdown('### Cascade coefficient robustness (beta +/- 20%)'))
display(beta_robust)

# ---------------------------------------------------------------------------
# cell 31
# ---------------------------------------------------------------------------
# Policy-cost robustness: every activation cost scaled by a common factor.
policy_robust = rob.policy_cost_robustness(factors=[0.5, 0.75, 1.0, 1.5, 2.0])
policy_robust.to_csv('outputs/table_policy_cost_robustness.csv', index=False)
display(Markdown('### Policy-cost robustness (activation costs 0.5-2.0x)'))
display(policy_robust)

# ---------------------------------------------------------------------------
# cell 32
# ---------------------------------------------------------------------------
# Transition-row perturbation: make every row of the disruption
# transition matrix uniformly more or less adverse, sweeping the closed-to-closed
# persistence around its baseline, and re-solve. Tests whether the saving and the
# active bundle survive misspecification of the least data-informed block.
transition_robust = rob.transition_perturbation_robustness(
    deltas=(-0.10, 0.0, 0.10), target_count=MIP_TARGET_COUNT)
transition_robust.to_csv('outputs/table_transition_robustness.csv', index=False)
display(Markdown('### Transition-row perturbation (full-closure Hormuz)'))
display(transition_robust)

# ---------------------------------------------------------------------------
# cell 33
# ---------------------------------------------------------------------------
# 27-path loss distribution and CVaR: solve the full-closure Hormuz
# case under no intervention and under the optimized portfolio, then report the
# spread of total cost across the enumerated paths (mean, std, min, max, and CVaR
# at 90% and 95%) rather than the expectation alone. Feeds Table 8.
loss_dist = rob.loss_distribution_cvar(target_count=MIP_TARGET_COUNT)
loss_dist.to_csv('outputs/table_loss_distribution.csv', index=False)
display(Markdown('### 27-path loss distribution and tail risk (CVaR)'))
display(loss_dist)

# ---------------------------------------------------------------------------
# cell 34
# ---------------------------------------------------------------------------
# Instrument ranking under attrition: solve the optimized response
# under the mild, moderate, and severe war-of-attrition regimes and report the
# active bundle, confirming the ranking is stable under attrition dynamics.
attrition_rank = rob.attrition_ranking_stability(target_count=MIP_TARGET_COUNT)
attrition_rank.to_csv('outputs/table_attrition_ranking.csv', index=False)
display(Markdown('### Instrument ranking under attrition dynamics'))
display(attrition_rank)

# ---------------------------------------------------------------------------
# cell 35
# ---------------------------------------------------------------------------
# Cascade weights, first-order vs full Leontief inverse (item N5): quantify how
# far the single-pass stress weights sit below the total-requirements weights of
# (I - beta)^{-1}. Confirms the single pass is a conservative lower bound while
# the sector ranking is preserved. No solve.
cascade_gap = rob.cascade_weight_gap()
cascade_gap.to_csv('outputs/table_cascade_weight_gap.csv', index=False)
display(Markdown(f"### Cascade weights: first-order vs Leontief inverse "
                 f"(spectral radius {cascade_gap.attrs['spectral_radius']}, "
                 f"max relative row-sum gap {cascade_gap.attrs['max_relative_gap_pct']}%)"))
display(cascade_gap)

# ---------------------------------------------------------------------------
# cell 36
# ---------------------------------------------------------------------------
# ESCVI ranking stability under weight perturbation: perturb all four
# ESCVI dimension weights by +/-25% over a 625-point grid and report the rank
# correlation to the baseline and the top-10 / top-12 retention. No solve.
escvi_stab = rob.escvi_weight_stability()
escvi_stab.to_csv('outputs/table_escvi_stability.csv', index=False)
display(Markdown('### ESCVI ranking stability under +/-25% weight perturbation'))
display(escvi_stab)

# ---------------------------------------------------------------------------
# cell 37
# ---------------------------------------------------------------------------
_f = 'outputs/table_stochastic_diagnostics.csv'
df_diag = compute_diagnostics(scenario_classes=MIP_SCENARIO_CLASSES,
                              target_count=MIP_TARGET_COUNT, verbose=True)
df_diag.to_csv(_f, index=False)
display(Markdown('### Stochastic-programming diagnostics (MIP)'))
display(df_diag)

# ---------------------------------------------------------------------------
# cell 38
# ---------------------------------------------------------------------------
# Headline figure 1: national economic-loss choropleth (model-derived)
df_country_loss = export_country_losses('outputs')
_t = df_country_loss.iloc[0]
print(f"National losses exported for {len(df_country_loss)} importers; "
      f"largest: {_t['Country']} ${_t['Inaction ($B)']:.0f}B inaction")
display(df_country_loss.head(12))
_p = make_country_loss_map(df_country_loss, 'outputs')
print('Saved', _p)
display(Image(_p.replace('.pdf', '.png')))

# ---------------------------------------------------------------------------
# cell 39
# ---------------------------------------------------------------------------
# Headline figure 2: optimized crude oil flow Sankey under the proactive response
df_flows = export_optimized_flows('outputs')
print(f'{len(df_flows)} flow links; total importer inflow '
      f"{df_flows[df_flows.dst_layer=='importer'].flow_mbd.sum():.1f} mbd")
display(df_flows.head(15))
_p = make_flow_sankey(df_flows, 'outputs')
print('Saved', _p)
display(Image(_p.replace('.pdf', '.png')))

# ---------------------------------------------------------------------------
# cell 40
# ---------------------------------------------------------------------------
# Cost decomposition: every objective channel, term by term (including the
# maritime freight / war-risk premium and priced demand curtailment). The
# proactive response barely touches the oil-price term, which is set by the
# physical supply loss and remains the dominant residual cost. Coordination
# instead closes the direct shortage.
rows = []
for lbl, sub in [('No intervention', ()), ('Proactive', ('P1','P2','P3','P4','P5','P6'))]:
    c = res.cost_components(res.solve(sub, cache=False)); c['Regime'] = lbl; rows.append(c)
df_decomp = pd.DataFrame(rows).set_index('Regime')[['Transport','Policy','Direct','Cascade','Oil price','Curtailment','Reserve','Substitute','Equity floor','Total']].round(1)
df_decomp.to_csv('outputs/table_cost_decomposition.csv')
display(df_decomp)

# ---------------------------------------------------------------------------
# cell 41
# ---------------------------------------------------------------------------
# No-disruption benchmark: with the chokepoint open the disruption cost is ~0,
# so every reported figure is a clean delta from undisrupted operations.
m_normal  = res.solve((), scenario_class='no_disruption', cache=False)
m_closure = res.solve((), scenario_class='full_closure',  cache=False)
print(f'Normal operations (no disruption): ${m_normal._objective_value/1e3:,.1f}B')
print(f'Full Hormuz closure, no action:    ${m_closure._objective_value/1e3:,.1f}B')
print(f'Disruption-attributable cost:      ${(m_closure._objective_value-m_normal._objective_value)/1e3:,.1f}B')

import gc; del m_normal, m_closure; gc.collect()

# ---------------------------------------------------------------------------
# cell 42
# ---------------------------------------------------------------------------
# List all generated outputs
import glob

files = sorted(glob.glob('outputs/*'))
tables = [f for f in files if 'table_' in f]
data = [f for f in files if 'data_' in f or 'scenarios_' in f]
figures = [f for f in files if 'fig_' in f]

print(f"{'='*60}")
print(f"  GENERATED OUTPUTS: {len(files)} total files")
print(f"{'='*60}")
print(f"\n  Tables ({len(tables)}):")
for f in tables:
    print(f"    {os.path.basename(f)}")
print(f"\n  Data ({len(data)}):")
for f in data:
    print(f"    {os.path.basename(f)}")
print(f"\n  Figures ({len(figures)}):")
for f in figures:
    print(f"    {os.path.basename(f)}")
print(f"\n{'='*60}")
print(f"  Pipeline complete. All results are reproducible.")
print(f"{'='*60}")

