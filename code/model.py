"""Optimization stack for the maritime chokepoint model.

Config, supply-chain network data, the decision-dependent stochastic MILP, and
the cached MIP solve wrapper in one module. Results, figures, and validation
live in analysis.py.
"""
from __future__ import annotations


# ======================================================================
# CONFIG: constants and calibration
# ======================================================================
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Iterable
from functools import lru_cache
import numpy as np
from scipy.linalg import logm, expm

# ---------------------------------------------------------------------------
# Time horizon
# ---------------------------------------------------------------------------
NUM_STAGES = 4                       # T = {0,1,2,3}
STAGE_DAYS = 25                      # each stage = 25 days
HORIZON_DAYS = NUM_STAGES * STAGE_DAYS  # 100 days

# ---------------------------------------------------------------------------
# Sectors
# ---------------------------------------------------------------------------
SECTORS: List[str] = [
    "electricity", "industry", "agriculture",
    "transport", "residential", "healthcare",
]

# Inter-sector cascade coefficients beta_{ss'}: dependence of sector s on energy
# supplier s'. From the WIOD 2016 tables (Timmer et al., 2015), supplier side
# restricted to energy-carrying industries, averaged over 15 major economies and
# rescaled to the calibrated overall intensity. See experiments.py build-cascade.
BETA: Dict[str, Dict[str, float]] = {
    "electricity":  {"industry": 0.27, "transport": 0.17},
    "industry":     {"electricity": 0.13, "transport": 0.21},
    "agriculture":  {"electricity": 0.11, "industry": 0.16, "transport": 0.15},
    "transport":    {"electricity": 0.09, "industry": 0.61},
    "residential":  {"electricity": 0.10, "industry": 0.02, "transport": 0.06},
    "healthcare":   {"electricity": 0.16, "industry": 0.06, "transport": 0.27},
}

# Damage-function (tau, mu) per sector for g_s(u) = mu*u^2 + mu*max(0, u-tau)^2.
# tau is the stress level where damage accelerates, lowest for tight critical
# sectors (healthcare 0.25, electricity 0.30), highest for substitutable ones
# (residential 0.50). mu is the curvature.
THRESHOLDS: Dict[str, Tuple[float, float]] = {
    "electricity":  (0.30, 3.5),
    "industry":     (0.40, 2.8),
    "agriculture":  (0.45, 4.2),
    "transport":    (0.35, 3.0),
    "residential":  (0.50, 2.5),
    "healthcare":   (0.25, 5.0),
}

# PWL breakpoints for epigraph reformulation. The tangent points are concentrated
# near u=0 because the optimized allocation drives sectoral stress into the small-u
# regime, where a uniform grid would leave the nearest tangent (at u_mid=0.1) far
# below g(u) and let the epigraph under-count amplification. The
# concentrated grid keeps the convex under-approximation tight there. See
# PWL_BREAKPOINTS and the certified/realized approximation error in
# analysis.amplification_pwl_error.
PWL_BREAKPOINTS = [0.0, 0.015, 0.035, 0.06, 0.1, 0.16, 0.25, 0.37, 0.52, 0.7, 0.85, 1.0]
NUM_PWL_SEGMENTS = len(PWL_BREAKPOINTS) - 1


def _damage_norm(sector: str) -> float:
    """Normalization constant g_s(1) = mu + mu*(1-tau)^2, the raw damage at full
    inoperability. Dividing g_s by it makes the inoperability index
    Delta = g_s(u)/g_s(1) lie in [0,1]. The same constant is
    used as the sector damage intensity eta_s = g_s(1) in the objective, so the
    amplification loss eta_s * VA * Delta = VA * g_s(u) is unchanged. eta_s exceeds
    one for critical sectors, whose failure costs more than their own value added."""
    tau, mu = THRESHOLDS[sector]
    return mu + mu * (1.0 - tau) ** 2


def damage_value(sector: str, u: float) -> float:
    """Normalized convex sector damage g_s(u) = [mu*u^2 + mu*max(0, u-tau)^2]/g_s(1).

    A second quadratic steepens the curve above tau (bottleneck regime). Convex
    and nondecreasing on u>=0, so its epigraph is the exact max of tangent lines.
    Normalized so g_s(0)=0 and g_s(1)=1, preserving the sector-specific shape.
    """
    tau, mu = THRESHOLDS[sector]
    over = max(0.0, u - tau)
    return (mu * u * u + mu * over * over) / _damage_norm(sector)


def damage_slope(sector: str, u: float) -> float:
    """Derivative g_s'(u) of the normalized damage, for the tangent-line segments."""
    tau, mu = THRESHOLDS[sector]
    return (2.0 * mu * u + 2.0 * mu * max(0.0, u - tau)) / _damage_norm(sector)


def build_pwl_segments(sector: str):
    """Tangent lines (slope alpha, intercept b) to g_s at the midpoints of the
    PWL_BREAKPOINTS intervals on [0, 1]. Their max is the convex epigraph used in
    the MIP. The grid is concentrated near u=0 so the epigraph stays tight in the
    small-stress regime the optimum occupies."""
    segments = []
    breakpoints = PWL_BREAKPOINTS
    for i in range(len(breakpoints) - 1):
        u_mid = 0.5 * (breakpoints[i] + breakpoints[i + 1])
        alpha = damage_slope(sector, u_mid)
        b = damage_value(sector, u_mid) - alpha * u_mid
        segments.append((alpha, b))
    return segments


# ---------------------------------------------------------------------------
# Policy instruments
# ---------------------------------------------------------------------------
@dataclass
class PolicyInstrument:
    """A single policy instrument.

    Effects are modelled structurally through the constraints each policy gates,
    not through any reduced-form multiplier, so only the activation ``cost`` and
    the ``irreversible`` flag are carried here.
    """

    pid: str
    name: str
    cost: float                    # activation cost ($B per stage)
    irreversible: bool             # once activated, stays on?


POLICIES: List[PolicyInstrument] = [
    PolicyInstrument("P1", "Coordinated reserve release",  5.0,  False),
    PolicyInstrument("P2", "Rerouting support",            8.0,  False),
    PolicyInstrument("P3", "Targeted demand management",   2.0,  False),
    PolicyInstrument("P4", "Diversification investment",  15.0,  True),
    PolicyInstrument("P5", "Bilateral sharing agreements",  3.0,  False),
    PolicyInstrument("P6", "Emergency substitute supply",   1.0,  False),
]

POLICY_DICT = {p.pid: p for p in POLICIES}

# Stage-wise policy-activation budget caps B_t ($B): ceiling on total activation
# cost sum_p gamma_p z_pt per stage. Rise across the horizon (20 -> 40) and start
# below the full-bundle cost (5+8+2+15+3+1 = 34 $B) so the planner phases
# instruments in. Costs are far below the damages averted, so the optimal bundle
# is insensitive to these caps. MODELLING ASSUMPTION, not a data calibration.
STAGE_BUDGETS = [20.0, 30.0, 35.0, 40.0]

# ---------------------------------------------------------------------------
# Sectoral economic calibration. Values match the paper appendix tables.
# ---------------------------------------------------------------------------
# Direct-shortage penalty pi_ns ($/bbl unmet): first-round end-use value of a
# delivered barrel, ~150 $/bbl (EIA refined-product value, net of tax) times a
# substitutability factor in [0.67, 2.0]. Excludes the oil-price and cascade
# effects (carried by the dP and eta terms), so channels do not double-count
# (Rose 2004).
SHORTAGE_PENALTY = {
    "electricity": 250.0, "industry": 150.0, "agriculture": 200.0,
    "transport": 120.0, "residential": 100.0, "healthcare": 300.0,
}

# Cascade penalty eta_ns ($/unit amplified damage Delta). Same ordering as pi_ns
# but below it, so a directly delivered barrel always beats second-round relief.
# (Legacy weight, retained for reference, the amplification loss is now built on an
# explicit dollar base, see AMP_MULTIPLIER below.)
CASCADE_WEIGHT = {
    "electricity": 200.0, "industry": 110.0, "agriculture": 160.0,
    "transport": 90.0, "residential": 75.0, "healthcare": 250.0,
}

# Within-country amplification base = sector VALUE-ADDED. The
# amplification loss of a country-sector is L_nst = phi * VA_ns * g_s(u_nst), where
# VA_ns = GDP_n * VA_SHARE_s is the sector's value added (an observable dollar base,
# decoupled from its oil demand so that a small-oil, high-criticality sector such as
# healthcare still carries real weight) and phi is the fraction of that value added
# lost at full oil stress (g_s=1). g_s already carries the sector-specific curvature
# and threshold, so a single phi suffices. This replaces the earlier oil-expenditure
# base, which under-weighted critical low-oil sectors once the demand shares were made
# transport-dominant.
VA_SHARE = {   # sector share of GDP (value-added), national-accounts orders of magnitude
    "electricity": 0.025, "industry": 0.18, "agriculture": 0.05,
    "transport": 0.06, "residential": 0.10, "healthcare": 0.09,
}
VA_LOSS_FRAC = 1.0    # eta: at full inoperability (g_s=1) a sector loses its stage
                      # value added, so the amplification loss is VA_nst * g_s(u).

# Import dependence: share of a country's crude oil demand met by imports rather than
# domestic production. National chokepoint exposure is the product
# of this import dependence and the share of imports routed through the disrupted
# passage, so a producer such as China or Brazil is not treated as fully exposed on
# its whole demand. Pure importers default to 1.0, only economies with material
# domestic output are listed. Values are approximate net-import shares of consumption
# from EIA/IEA country balances and are calibration inputs, not estimated quantities.
IMPORT_SHARE: Dict[str, float] = {
    "CHN": 0.72, "IND": 0.85, "IDN": 0.65, "MYS": 0.60, "VNM": 0.60,
    "EGY": 0.55, "THA": 0.85, "BRA": 0.25, "COL": 0.20, "ARG": 0.30,
    "NGA": 0.10, "AUS": 0.75, "GBR": 0.75, "COD": 0.90,
}

# Max demand curtailment by targeted demand management (P3), fraction of baseline:
# low for essential uses (healthcare, electricity), higher for discretionary ones.
MAX_DEMAND_RED = {
    "electricity": 0.05, "industry": 0.15, "agriculture": 0.08,
    "transport": 0.12, "residential": 0.20, "healthcare": 0.02,
}

# Split of managed demand reduction into genuine efficiency/fuel-switching and
# forced rationing. The first EFFICIENCY_FRAC of the reduction is
# low-cost efficiency that avoids downstream harm, the remainder is forced rationing
# that propagates to dependent sectors (it enters the cross-sector stress exactly
# like an involuntary shortfall) and is therefore not a free way to erase demand.
# The share is a scenario parameter to be swept, so the optimizer can no longer
# relabel welfare-reducing rationing as costless demand management.
EFFICIENCY_FRAC = 0.40
# Cost of the efficiency portion as a fraction of the crude price (cheap abatement /
# fuel switching that preserves most output). Forced rationing, by contrast, is an
# unmet need and is priced at the sector's direct-shortage penalty pi_ns (C6).
EFFICIENCY_COST_FRAC = 0.30

# Equity service floor: minimum service ratio guaranteed to protected priority
# sectors of vulnerable countries while oil is physically available. MODELLING CHOICE.
SERVICE_FLOOR = 0.60

# Preemptive (lexicographic) weight ($/bbl) on any priority-floor breach (MODELLING
# WEIGHT, not data). Set above the sum of the maximum competing per-barrel marginals
# (direct penalty <=300, amplification relief, and price channel, together bounded
# below ~1000/bbl by the calibration), so meeting the floor always dominates any
# economic saving from breaching it. This is the Archimedean realization of
# lexicographic goal programming: the solution equals the two-phase optimum that
# first minimizes floor breach and then minimizes economic cost. In every solved
# scenario the breach is exactly zero, so the term contributes nothing to reported
# welfare and the floor acts as a hard guarantee whenever it is physically feasible.
FLOOR_PENALTY = 2000.0

# Emergency substitute supply (P6) cap, fraction of daily demand coverable per
# stage via refinery-yield reallocation, product-stock release, or fuel switching.
SUBSTITUTE_SUPPLY_FRAC = 0.05

# Per-barrel resource/opportunity cost of emergency substitute supply ($/bbl).
# Beyond the fixed activation charge gamma_{P6}, each substitute
# barrel consumes a real resource: product stocks sold from inventory, refinery
# reconfiguration, or a switch to a pricier alternative fuel. We price it above the
# reserve-replacement cost because these are second-best, higher-marginal-cost
# sources, which stops P6 from being an almost-free bulk lever and tests whether its
# leading rank in the attribution survives a transparent variable cost.
SUBSTITUTE_SUPPLY_COST = 95.0
# P6 disaggregated into three physically distinct sub-instruments,
# each with its own per-barrel cost, its own share of the substitute capacity, and its
# own dynamics. Product-stock release draws a FINITE depleting inventory (cheapest,
# existing barrels), refinery-yield reallocation shifts processing toward the short
# product at a moderate cost, inter-fuel switching is the priciest second-best source
# and is an oil-demand reduction in eligible sectors. The share-weighted mean cost is
# close to the former single SUBSTITUTE_SUPPLY_COST, so the aggregate calibration is
# preserved while the channels are no longer fungible.
SUBSTITUTE_COST_PS = 70.0     # product-stock release ($/bbl)
SUBSTITUTE_COST_RF = 95.0     # refinery-yield reallocation ($/bbl)
SUBSTITUTE_COST_FS = 120.0    # inter-fuel switching ($/bbl)
SUBSTITUTE_CHANNEL_SHARE = {"ps": 0.40, "rf": 0.35, "fs": 0.25}  # per-stage cap split
# Product-stock inventory as a FIXED number of days of daily release capacity, so the
# physical stock does not scale with the stage length. At the default
# STAGE_DAYS=25 this equals 2 stages of per-stage capacity, so headline results are
# unchanged, but a longer- or shorter-stage sensitivity holds the barrels fixed.
PRODUCT_STOCK_DAYS = 50.0     # = 2 stages x 25 days at the default stage length

# Reserve-cover threshold (days) below which a country's healthcare and electricity
# are priority pairs for the floor. Well under the IEA 90-day obligation.
PRIORITY_SPR_THRESHOLD = 30.0

# Replacement (opportunity) cost of a drawn-down reserve barrel ($/bbl). A released
# barrel is consumed during the crisis but must be repurchased afterward at roughly
# the market price, and holding it forgoes its option value against a worse later
# state. Charging the pre-crisis price is a conservative, documented value: it stops
# reserve release from being a free lever and tests whether the reserve-first ranking
# survives a realistic drawdown cost.
RESERVE_REPLACEMENT_COST = 72.0

# ---------------------------------------------------------------------------
# Oil-price-shock cost. The world supply loss raises the world price by
# dP = P0 * (loss / Qbar) / |eta_D| (short-run inverse demand) and every barrel
# still consumed pays dP. See paper Table tab_priceparams. Literature values.
# ---------------------------------------------------------------------------
BASELINE_OIL_PRICE = 72.0       # P0, pre-disruption Brent ($/bbl), EIA/IEA
OIL_DEMAND_ELASTICITY = -0.20   # eta_D short-run, elastic end of range (Caldara et al. 2019)
WORLD_OIL_SUPPLY_MBD = 102.0    # Qbar, global liquids supply (EIA/IEA)
# Sector-specific short-run price responsiveness. The price-
# induced contraction is NOT uniform across sectors: switchable and discretionary
# uses (oil-fired power, some industry) cut more per unit price rise than essential,
# inelastic uses (residential heating, healthcare). These are RELATIVE weights on the
# common contraction fraction, build_model rescales them by one factor so the demand-
# weighted mean is 1, which preserves aggregate market clearing (the coalition-wide
# contraction is unchanged, only its sectoral incidence differs). Orders of magnitude
# follow short-run sectoral oil-demand elasticities (transport the numeraire).
PRICE_CONTRACTION_WEIGHT: Dict[str, float] = {
    "electricity": 1.5, "industry": 1.2, "transport": 1.0,
    "agriculture": 0.8, "residential": 0.7, "healthcare": 0.3,
}

# Regional differentiation of the cross-sector cascade. Emerging
# importers carry more concentrated and less substitutable energy dependence, so
# their inter-sector amplification is scaled up relative to the global WIOD anchor of
# BETA and advanced importers down. build_model renormalizes the scales so the demand-
# weighted mean is 1, which holds the overall calibrated intensity fixed and only
# REDISTRIBUTES it across regions, turning the single global matrix into regional
# matrices (the global matrix is recovered when both scales are 1). Countries not
# listed as advanced are treated as emerging.
ADVANCED_ECONOMIES = {"JPN", "KOR", "TWN", "HKG", "SGP", "DEU", "ITA", "ESP", "FRA",
                      "GBR", "NLD", "BEL", "AUT", "CHE", "SWE", "AUS", "NZL", "ISR",
                      "CAN", "USA", "NOR", "DNK", "FIN", "IRL"}
REGIONAL_BETA_SCALE: Dict[str, float] = {"advanced": 0.90, "emerging": 1.15}


def cascade_region_scale(code: str) -> float:
    """Raw regional cascade-intensity multiplier for a country (renormalized in
    build_model so the demand-weighted mean is 1)."""
    return REGIONAL_BETA_SCALE["advanced" if code in ADVANCED_ECONOMIES else "emerging"]


@lru_cache(maxsize=1)
def leontief_cascade_matrix() -> Tuple[Tuple[str, Dict[str, float]], ...]:
    """Total-requirements cascade weights M = (I - beta)^{-1} - I, the full
    Leontief-inverse benchmark for the single-pass BETA.
    Returned as an immutable tuple of (sector, {supplier: weight}) so it can be
    cached; M >= BETA elementwise because it sums every higher-order indirect path."""
    secs = list(SECTORS)
    idx = {s: i for i, s in enumerate(secs)}
    B = np.zeros((len(secs), len(secs)))
    for s, row in BETA.items():
        for sp, val in row.items():
            B[idx[s], idx[sp]] = val
    M = np.linalg.inv(np.eye(len(secs)) - B) - np.eye(len(secs))
    return tuple((s, {sp: float(M[idx[s], idx[sp]]) for sp in secs
                      if abs(M[idx[s], idx[sp]]) > 1e-12}) for s in secs)


def cascade_matrix_for(amp_spec: str) -> Dict[str, Dict[str, float]]:
    """Cross-sector cascade weights for an amplification specification: the
    single-pass BETA for 'headline'/'capped', the total-requirements Leontief
    inverse for 'leontief'."""
    if amp_spec == "leontief":
        return {s: dict(row) for s, row in leontief_cascade_matrix()}
    return BETA
# Spare capacity (mbd) reaching the market WITHOUT crossing the chokepoint. World
# surplus is ~4 mbd (EIA 2024-25) but mostly behind Hormuz, so only ~2 mbd is
# off-strait.
GLOBAL_SPARE_CAPACITY_MBD = 2.0

# Freight/war-risk premium on sea legs: effective cost = c_a*(1 + FREIGHT_PREMIUM*
# (1 - availability)). At full closure sea freight roughly triples, matching
# war-risk insurance and VLCC charter-rate spikes in Hormuz/Red Sea crises.
# Pipelines and inland delivery are unaffected.
FREIGHT_PREMIUM = 2.0

# Diversification (P4): irreversible build-out adding DIVERSIFICATION_FACTOR of each
# bypass arc's capacity, only from DIVERSIFICATION_STAGE onward and while P4 is on.
# Half-capacity is a partial medium-term build-out, not a full network duplicate.
DIVERSIFICATION_FACTOR = 0.5
DIVERSIFICATION_STAGE = 2

# Bilateral sharing (P5): each exposed importer may receive up to SHARING_FRAC of
# its daily demand from Western hubs. The implied ~1 mbd total is on the scale of
# IEA collective stock-release actions.
SHARING_FRAC = 0.03

# ---------------------------------------------------------------------------
# Disruption states
# ---------------------------------------------------------------------------
DISRUPTION_STATES = ["open", "restricted", "closed"]

# Chokepoint availability fraction per state
STATE_AVAILABILITY = {
    "open": 1.0,
    "restricted": 0.5,
    "closed": 0.0,
}

# ---------------------------------------------------------------------------
# Loss scaling
# ---------------------------------------------------------------------------
# Multiplier on GDP-converted losses in the reduced-form screen (analysis.py).
# Set to 1.0 for no rescaling. Reported losses come from the solved MIP.
LOSS_SCALING = 1.0

# ---------------------------------------------------------------------------
# Plotting palette
# ---------------------------------------------------------------------------
# Colorblind-aware, print-safe palette. Figures draw from these hues through
# the semantic maps below, so each concept keeps a consistent color.
PAL = {
    'navy':   '#2F4B7C',   # deep blue
    'blue':   '#4C72B0',   # blue
    'teal':   '#4E9CA6',   # teal
    'forest': '#4F9D69',   # green
    'amber':  '#E0A458',   # warm amber
    'orange': '#DD8452',   # orange
    'coral':  '#C0524F',   # red
    'red':    '#A83C3C',   # deep red
    'purple': '#8172B3',   # purple
    'slate':  '#7F8C99',   # slate grey
    'light':  '#E8F0F8',   # pale fill
}

# Policy regime, ordered worst -> best (red -> amber -> green).
REGIME_COLORS = {
    'No intervention': PAL['coral'],
    'Reactive bundle': PAL['amber'],
    'Full proactive':  PAL['forest'],
}

# Six end-use sectors (categorical, stable across all sector figures).
SECTOR_COLORS = {
    'electricity': PAL['amber'],
    'industry':    PAL['navy'],
    'agriculture': PAL['forest'],
    'transport':   PAL['teal'],
    'residential': PAL['slate'],
    'healthcare':  PAL['coral'],
}

# Scenario classes, ordered mild -> severe (for the efficiency-equity frontiers).
SEVERITY_COLORS = {
    'partial_restriction': PAL['teal'],
    'full_closure':        PAL['blue'],
    'multi_chokepoint':    PAL['coral'],
}

# Cascade decomposition components.
CASCADE_COLORS = {
    'direct':     PAL['navy'],
    'cascade':    PAL['amber'],
    'bottleneck': PAL['coral'],
}

# Two-series comparison (e.g. Shapley vs leave-one-out): primary / secondary.
SERIES2 = (PAL['navy'], PAL['teal'])

# Representative importers + aggregate (per-stage by-country panel).
COUNTRY_COLORS = {
    'China':         PAL['navy'],
    'India':         PAL['coral'],
    'Japan':         PAL['forest'],
    'S. Korea':      PAL['blue'],
    'Pakistan':      PAL['amber'],
    'Germany':       PAL['purple'],
    'Rest of world': PAL['slate'],
}

REGION_COLORS = {
    "Asia-Pac.": PAL['coral'],
    "Europe":    PAL['navy'],
    "MENA":      PAL['amber'],
    "Africa":    PAL['forest'],
}


# ======================================================================
# DATA: countries, regions, chokepoints, network
# ======================================================================
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Dict, List, Set, Tuple
import numpy as np



# ===================================================================
# Country
# ===================================================================
@dataclass
class Country:
    code: str
    name: str
    region: str
    daily_oil_mbd: float      # million barrels/day demand
    hormuz_dep: float          # fraction of supply via Hormuz
    malacca_dep: float         # fraction via Malacca
    suez_dep: float            # fraction via Suez
    spr_days: float            # strategic reserve coverage (days)
    spr_max_draw_rate: float   # max daily drawdown (mbd)
    gdp_B: float               # GDP (USD billion)
    pop_M: float               # population (millions)
    renew_pct: float           # renewable share (%)
    sector_weights: Dict[str, float] = field(default_factory=dict)

# Share of each country's oil demand by end-use sector, as regional averages from
# IEA oil (petroleum-product) final-consumption balances. Petroleum end use is
# transport-dominated, direct oil-fired electricity is small outside a few
# economies, industry captures petrochemical feedstock and process heat. The
# "healthcare" row is not a standalone IEA category but a documented proxy for
# critical-services oil use (hospitals, water, cold chain, emergency logistics),
# carved from the residential/commercial and transport rows. Each region sums to 1.0.
SECTOR_PROFILES = {
    "Asia-Pac.": {"electricity": 0.08, "industry": 0.24, "agriculture": 0.07,
                  "transport": 0.48, "residential": 0.08, "healthcare": 0.05},
    "Europe":    {"electricity": 0.06, "industry": 0.20, "agriculture": 0.04,
                  "transport": 0.52, "residential": 0.12, "healthcare": 0.06},
    "MENA":      {"electricity": 0.12, "industry": 0.22, "agriculture": 0.08,
                  "transport": 0.45, "residential": 0.08, "healthcare": 0.05},
    "Africa":    {"electricity": 0.10, "industry": 0.15, "agriculture": 0.13,
                  "transport": 0.50, "residential": 0.07, "healthcare": 0.05},
    "Americas":  {"electricity": 0.05, "industry": 0.22, "agriculture": 0.05,
                  "transport": 0.55, "residential": 0.08, "healthcare": 0.05},
    "Oceania":   {"electricity": 0.05, "industry": 0.24, "agriculture": 0.06,
                  "transport": 0.52, "residential": 0.08, "healthcare": 0.05},
}


def _build_country(code, name, region, dm, hd, md, sd, spr, gdp, pop, ren):
    max_draw = dm * 0.10  # cap draw rate at 10% of daily demand
    sw = dict(SECTOR_PROFILES[region])
    return Country(code, name, region, dm, hd, md, sd, spr, max_draw,
                   gdp, pop, ren, sw)


def build_countries() -> Dict[str, Country]:
    """Build the full 53-country dataset.

    Oil demand from EIA/IEA, chokepoint-dependence fractions from the EIA World
    Oil Transit Chokepoints assessment, SPR coverage from the IEA 90-day mandate
    (3-15 day buffers for non-members), GDP and population from the World Bank,
    renewable share from IEA/IRENA.
    """
    # (code, name, region, demand_mbd, hormuz_dep, malacca_dep, suez_dep,
    #  spr_days, gdp_B, pop_M, renew_pct)
    raw = [
        # --- Asia-Pacific and Oceania (18 + 2 = 20 countries) ---
        ("JPN", "Japan",         "Asia-Pac.",  3.20, 0.87, 0.82, 0.10, 230, 4231, 125.7, 22),
        ("KOR", "South Korea",   "Asia-Pac.",  2.60, 0.82, 0.80, 0.08,  93, 1665,  51.7,  8),
        ("IND", "India",         "Asia-Pac.",  5.30, 0.64, 0.25, 0.20,  74, 3385, 1408.0, 12),
        ("CHN", "China",         "Asia-Pac.", 15.70, 0.45, 0.70, 0.12,  80, 17963, 1412.0, 16),
        ("SGP", "Singapore",     "Asia-Pac.",  1.50, 0.78, 0.90, 0.05,  65,  397,   5.9,  4),
        ("THA", "Thailand",      "Asia-Pac.",  1.20, 0.70, 0.85, 0.05,  56,  514,  71.8, 15),
        ("TWN", "Taiwan",        "Asia-Pac.",  1.00, 0.75, 0.78, 0.08,  90,  790,  23.9,  7),
        ("PHL", "Philippines",   "Asia-Pac.",  0.50, 0.60, 0.75, 0.05,  15,  404, 113.9, 22),
        ("PAK", "Pakistan",      "Asia-Pac.",  0.60, 0.68, 0.15, 0.10,  12,  340, 231.4,  7),
        ("BGD", "Bangladesh",    "Asia-Pac.",  0.15, 0.55, 0.20, 0.08,   8,  460, 169.4,  4),
        ("LKA", "Sri Lanka",     "Asia-Pac.",  0.09, 0.62, 0.30, 0.10,  10,   75,  22.2, 14),
        ("VNM", "Vietnam",       "Asia-Pac.",  0.55, 0.40, 0.60, 0.05,  20,  409,  99.5, 12),
        ("MYS", "Malaysia",      "Asia-Pac.",  0.70, 0.50, 0.65, 0.05,  35,  407,  33.6, 10),
        ("IDN", "Indonesia",     "Asia-Pac.",  1.60, 0.35, 0.55, 0.05,  25, 1319, 277.5,  8),
        ("MMR", "Myanmar",       "Asia-Pac.",  0.12, 0.45, 0.50, 0.05,   5,   65,  54.4, 10),
        ("KHM", "Cambodia",      "Asia-Pac.",  0.06, 0.40, 0.55, 0.05,   3,   30,  16.9,  6),
        ("NPL", "Nepal",         "Asia-Pac.",  0.04, 0.55, 0.10, 0.05,   5,   40,  30.5, 25),
        ("HKG", "Hong Kong",     "Asia-Pac.",  0.35, 0.55, 0.75, 0.05,  40,  360,   7.5,  2),
        ("NZL", "New Zealand",   "Oceania",    0.18, 0.20, 0.30, 0.05,  30,  252,   5.1, 40),
        ("AUS", "Australia",     "Oceania",    1.05, 0.15, 0.25, 0.05,  28, 1700,  26.4, 32),
        # --- Europe (14 countries) ---
        ("DEU", "Germany",       "Europe",     2.00, 0.35, 0.10, 0.25, 120, 4082,  84.4, 42),
        ("ITA", "Italy",         "Europe",     1.20, 0.38, 0.08, 0.30, 105, 2010,  59.0, 35),
        ("ESP", "Spain",         "Europe",     1.20, 0.30, 0.05, 0.20,  92, 1398,  47.4, 40),
        ("FRA", "France",        "Europe",     1.50, 0.22, 0.05, 0.18,  98, 2780,  67.8, 25),
        ("GRC", "Greece",        "Europe",     0.30, 0.45, 0.05, 0.35,  90,  220,  10.4, 30),
        ("NLD", "Netherlands",   "Europe",     0.85, 0.32, 0.08, 0.22, 100, 1010,  17.7, 33),
        ("BEL", "Belgium",       "Europe",     0.55, 0.30, 0.07, 0.20,  95,  583,  11.6, 25),
        ("POL", "Poland",        "Europe",     0.60, 0.28, 0.05, 0.15,  90,  689,  37.7, 20),
        ("PRT", "Portugal",      "Europe",     0.22, 0.25, 0.05, 0.18,  85,  252,  10.3, 45),
        ("SWE", "Sweden",        "Europe",     0.32, 0.18, 0.05, 0.12, 110,  586,  10.4, 55),
        ("GBR", "United Kingdom","Europe",     1.30, 0.20, 0.08, 0.15, 100, 3070,  67.7, 38),
        ("AUT", "Austria",       "Europe",     0.25, 0.22, 0.05, 0.15,  85,  471,   9.1, 35),
        ("CZE", "Czech Republic","Europe",     0.20, 0.25, 0.05, 0.12,  90,  282,  10.8, 18),
        ("ROU", "Romania",       "Europe",     0.20, 0.30, 0.05, 0.18,  70,  301,  19.0, 24),
        # --- MENA (6 countries) ---
        ("TUR", "Turkey",        "MENA",       1.00, 0.48, 0.08, 0.25,  60,  906,  85.3, 18),
        ("EGY", "Egypt",         "MENA",       0.70, 0.30, 0.05, 0.55,  30,  398, 109.3, 12),
        ("ISR", "Israel",        "MENA",       0.25, 0.40, 0.05, 0.30,  60,  525,   9.8, 15),
        ("JOR", "Jordan",        "MENA",       0.12, 0.55, 0.05, 0.30,  20,   47,  11.3,  8),
        ("LBN", "Lebanon",       "MENA",       0.15, 0.50, 0.05, 0.25,  10,   22,   5.5,  5),
        ("MAR", "Morocco",       "MENA",       0.30, 0.25, 0.05, 0.35,  30,  140,  37.5, 20),
        # --- Africa (9 countries) ---
        ("ETH", "Ethiopia",      "Africa",     0.08, 0.85, 0.05, 0.70,   3,  156, 126.5, 12),
        ("KEN", "Kenya",         "Africa",     0.12, 0.75, 0.10, 0.55,   5,  114,  54.0, 45),
        ("UGA", "Uganda",        "Africa",     0.04, 0.80, 0.05, 0.55,   2,   50,  47.2, 10),
        ("DJI", "Djibouti",      "Africa",     0.02, 0.90, 0.05, 0.80,   2,    4,   1.1,  5),
        ("RWA", "Rwanda",        "Africa",     0.02, 0.70, 0.05, 0.45,   2,   13,  14.1, 15),
        ("ZAF", "South Africa",  "Africa",     0.60, 0.35, 0.10, 0.25,  25,  399,  60.4, 11),
        ("NGA", "Nigeria",       "Africa",     0.45, 0.10, 0.05, 0.15,  15,  477, 218.5,  5),
        ("TZA", "Tanzania",      "Africa",     0.08, 0.70, 0.10, 0.50,   5,   75,  65.5, 15),
        ("MOZ", "Mozambique",    "Africa",     0.04, 0.50, 0.10, 0.35,   3,   18,  33.0,  8),
        # --- Americas (4 countries) ---
        ("BRA", "Brazil",        "Americas",   2.20, 0.08, 0.05, 0.05,  30, 2126, 215.3, 45),
        ("CHL", "Chile",         "Americas",   0.35, 0.10, 0.08, 0.05,  20,  301,  19.5, 35),
        ("ARG", "Argentina",     "Americas",   0.55, 0.05, 0.05, 0.03,  25,  641,  46.3, 15),
        ("COL", "Colombia",      "Americas",   0.35, 0.08, 0.05, 0.05,  20,  343,  52.1, 18),
    ]
    countries = {}
    for row in raw:
        c = _build_country(*row)
        countries[c.code] = c
    return countries


# ===================================================================
# Source regions
# ===================================================================
@dataclass
class SourceRegion:
    sid: str
    name: str
    capacity_mbd: float        # max export capacity
    hormuz_routed_frac: float  # fraction routed via Hormuz
    malacca_routed_frac: float
    suez_routed_frac: float

def build_sources() -> Dict[str, SourceRegion]:
    # Export capacities (mbd) from EIA/OPEC crude production statistics,
    # chokepoint-routed fractions from the EIA World Oil Transit Chokepoints
    # assessment. Tuple: (id, name, capacity, Hormuz, Malacca, Suez).
    raw = [
        ("SRC_GULF",   "Persian Gulf",        20.5, 0.95, 0.40, 0.15),
        ("SRC_RUSIA",  "Russia / CIS",         5.0, 0.00, 0.05, 0.10),
        ("SRC_NAFR",   "North Africa",         3.2, 0.00, 0.00, 0.60),
        ("SRC_WAFR",   "West Africa",          4.8, 0.00, 0.00, 0.20),
        ("SRC_NSEA",   "North Sea",            3.0, 0.00, 0.00, 0.05),
        ("SRC_NAFTA",  "North America",        5.5, 0.00, 0.00, 0.02),
        ("SRC_LATAM",  "Latin America",        4.0, 0.00, 0.00, 0.05),
        ("SRC_SEASI",  "Southeast Asia",       2.5, 0.10, 0.60, 0.05),
        ("SRC_CASP",   "Caspian",              1.8, 0.00, 0.05, 0.20),
        ("SRC_AUSNZ",  "Australia / NZ",       0.8, 0.00, 0.15, 0.02),
        ("SRC_EAFR",   "East Africa",          0.5, 0.00, 0.10, 0.30),
        ("SRC_OTHER",  "Other / Spot",         2.0, 0.10, 0.15, 0.10),
    ]
    sources = {}
    for sid, name, cap, h, m, s in raw:
        sources[sid] = SourceRegion(sid, name, cap, h, m, s)
    return sources


# ===================================================================
# Chokepoints
# ===================================================================
@dataclass
class Chokepoint:
    cid: str
    name: str
    throughput_mbd: float       # normal throughput
    bypass_capacity_mbd: float  # emergency bypass capacity
    bypass_cost_premium: float  # cost multiplier for bypass

def build_chokepoints() -> Dict[str, Chokepoint]:
    # Normal crude throughput (mbd) from the EIA World Oil Transit Chokepoints
    # assessment. Bypass capacity and cost premium reflect available overland
    # pipelines and longer sea routes (UAE Fujairah and Saudi East-West/Yanbu
    # around Hormuz, Cape of Good Hope around Suez, Lombok/Sunda around Malacca).
    # Tuple: (id, name, throughput, bypass capacity, bypass cost premium).
    raw = [
        ("CP_HORMUZ",  "Strait of Hormuz",    20.5, 3.5, 2.5),
        ("CP_MALACCA", "Strait of Malacca",   16.0, 4.0, 1.8),
        ("CP_SUEZ",    "Suez Canal",           5.5, 5.5, 2.0),   # Cape route
        ("CP_BABEL",   "Bab el-Mandeb",        6.2, 4.0, 2.2),
        ("CP_PANAMA",  "Panama Canal",         1.0, 0.8, 1.5),
    ]
    return {cid: Chokepoint(cid, name, tp, bp, bcp)
            for cid, name, tp, bp, bcp in raw}


# ===================================================================
# Processing / transshipment nodes
# ===================================================================
@dataclass
class ProcessingNode:
    nid: str
    name: str
    capacity_mbd: float

def build_processing_nodes() -> Dict[str, ProcessingNode]:
    # 18 refineries and oil hubs serving the importing regions. Capacities (mbd)
    # anchored to published refinery and port-terminal capacities.
    # Tuple: (id, name, capacity).
    raw = [
        ("PROC_FUJAIRAH",  "Fujairah Hub",      5.0),
        ("PROC_JEBEL",     "Jebel Ali",         3.5),
        ("PROC_JURONG",    "Jurong (Singapore)", 4.5),
        ("PROC_ROTTER",    "Rotterdam",          5.0),
        ("PROC_TRIESTE",   "Trieste / Med",      2.5),
        ("PROC_YOKOHAM",   "Yokohama",           3.0),
        ("PROC_ULSAN",     "Ulsan",              2.8),
        ("PROC_JAMNAGAR",  "Jamnagar",           2.5),
        ("PROC_SIDI",      "Sidi Kerir (Egypt)", 1.5),
        ("PROC_DURBAN",    "Durban",             0.8),
        ("PROC_HOUSTON",   "Houston",            4.5),
        ("PROC_SANTOS",    "Santos (Brazil)",    1.2),
        ("PROC_QINGDAO",   "Qingdao",           4.0),
        ("PROC_MUMBAI",    "Mumbai",             2.0),
        ("PROC_TANJUNG",   "Tanjung Pelepas",    2.0),
        ("PROC_DJIBOUTI",  "Djibouti Port",      0.5),
        ("PROC_MOMBASA",   "Mombasa",            0.6),
        ("PROC_YANBU",     "Yanbu (Red Sea)",     2.0),
    ]
    return {nid: ProcessingNode(nid, name, cap)
            for nid, name, cap in raw}


# ===================================================================
# Network arcs
# ===================================================================
@dataclass
class Arc:
    aid: str
    origin: str
    destination: str
    capacity_mbd: float
    unit_cost: float            # $/barrel transport cost
    chokepoint: str = ""        # empty if no chokepoint traversed
    is_bypass: bool = False     # emergency rerouting/bypass arc (gated by P2)
    is_sharing: bool = False    # cross-border bilateral-sharing arc (gated by P5)
    is_pipeline: bool = False   # land pipeline leg (no maritime freight/war-risk premium)
    relieves: str = ""          # chokepoint this arc physically bypasses, set only on the
                                # bypass route so its flow counts once toward restored supply

def build_arcs(sources, chokepoints, processing, countries) -> List[Arc]:
    """Build the 299 arcs connecting sources -> chokepoints -> processing -> countries."""
    arcs = []
    aid_counter = [0]

    def _aid():
        aid_counter[0] += 1
        return f"A{aid_counter[0]:04d}"

    # --- Source → Chokepoint arcs ---
    # Gulf → Hormuz
    arcs.append(Arc(_aid(), "SRC_GULF", "CP_HORMUZ", 20.5, 1.0, "CP_HORMUZ"))
    # Gulf → Fujairah bypass (UAE Habshan-Fujairah land pipeline, ~1.5 mbd, avoids Hormuz)
    arcs.append(Arc(_aid(), "SRC_GULF", "PROC_FUJAIRAH", 1.5, 2.5, "", True,
                    is_pipeline=True, relieves="CP_HORMUZ"))
    # Gulf → Yanbu bypass (Saudi East-West land pipeline, ~2.0 mbd, avoids Hormuz)
    arcs.append(Arc(_aid(), "SRC_GULF", "PROC_YANBU", 2.0, 2.8, "", True,
                    is_pipeline=True, relieves="CP_HORMUZ"))
    # Southeast Asia → Malacca
    arcs.append(Arc(_aid(), "SRC_SEASI", "CP_MALACCA", 2.5, 0.8, "CP_MALACCA"))
    # North Africa → Suez
    arcs.append(Arc(_aid(), "SRC_NAFR", "CP_SUEZ", 2.0, 0.9, "CP_SUEZ"))
    # West Africa → Bab el-Mandeb route
    arcs.append(Arc(_aid(), "SRC_WAFR", "CP_BABEL", 2.0, 1.2, "CP_BABEL"))
    # Direct routes (no chokepoint)
    for src in ["SRC_RUSIA", "SRC_NSEA", "SRC_NAFTA", "SRC_LATAM",
                "SRC_CASP", "SRC_AUSNZ"]:
        cap = sources[src].capacity_mbd
        arcs.append(Arc(_aid(), src, "PROC_ROTTER", min(cap, 3.0), 1.5))
        arcs.append(Arc(_aid(), src, "PROC_HOUSTON", min(cap, 2.0), 1.3))

    # --- Chokepoint → Processing arcs ---
    # Hormuz -> India direct: Indian refineries are on the Arabian Sea, reached
    # from the Gulf without a second chokepoint.
    for proc, cap, cost in [("PROC_JAMNAGAR", 4.0, 1.2), ("PROC_MUMBAI", 2.5, 1.0)]:
        arcs.append(Arc(_aid(), "CP_HORMUZ", proc, cap, cost))
    # Gulf crude for East Asia goes Hormuz -> Malacca -> refineries (Malacca is
    # the only sea route to China/Japan/Korea short of the Lombok/Sunda bypass).
    # Tagged to Malacca so its closure severs this corridor.
    arcs.append(Arc(_aid(), "CP_HORMUZ", "CP_MALACCA", 12.0, 1.0, "CP_MALACCA"))

    # Malacca → East Asian refineries
    for proc, cap, cost in [("PROC_JURONG", 4.5, 0.5), ("PROC_YOKOHAM", 4.0, 1.5),
                             ("PROC_ULSAN", 3.0, 1.4), ("PROC_QINGDAO", 5.0, 1.3),
                             ("PROC_TANJUNG", 2.0, 0.3)]:
        arcs.append(Arc(_aid(), "CP_MALACCA", proc, cap, cost))

    # Malacca bypass (Lombok/Sunda). Originates post-Hormuz, where the crude sits
    # before it would reach Malacca, so the detour goes around the Malacca node
    # rather than through it. It is therefore available when Malacca is closed
    # (Hormuz open) and is naturally zero when Hormuz itself is closed.
    for proc, cap, cost in [("PROC_JURONG", 2.5, 1.5), ("PROC_YOKOHAM", 2.0, 2.5),
                             ("PROC_QINGDAO", 2.5, 2.3)]:
        arcs.append(Arc(_aid(), "CP_HORMUZ", proc, cap, cost * 1.3, "", True,
                        relieves="CP_MALACCA"))

    # Suez -> Europe/Med, tagged to Suez so a closure severs the Gulf/Red Sea
    # -> Europe corridor (Cape of Good Hope bypass remains via rerouting).
    for proc, cap, cost in [("PROC_ROTTER", 3.0, 1.8), ("PROC_TRIESTE", 2.5, 1.2),
                             ("PROC_SIDI", 1.5, 0.3)]:
        arcs.append(Arc(_aid(), "CP_SUEZ", proc, cap, cost, "CP_SUEZ"))

    # Suez bypass (Cape of Good Hope). Originates post-Hormuz, so the route runs
    # around Africa rather than through the Suez node. Available when Suez is
    # closed (Hormuz open) and naturally zero when Hormuz itself is closed.
    for proc, cap, cost in [("PROC_ROTTER", 3.5, 3.5), ("PROC_TRIESTE", 2.0, 3.2),
                             ("PROC_DURBAN", 0.8, 1.5)]:
        arcs.append(Arc(_aid(), "CP_HORMUZ", proc, cap, cost, "", True,
                        relieves="CP_SUEZ"))

    # Bab el-Mandeb → Suez link
    arcs.append(Arc(_aid(), "CP_BABEL", "CP_SUEZ", 5.0, 0.5, "CP_BABEL"))
    arcs.append(Arc(_aid(), "CP_BABEL", "PROC_SIDI", 1.5, 0.8))

    # Bab el-Mandeb → East African ports (critical for Ethiopia, Kenya, etc.)
    arcs.append(Arc(_aid(), "CP_BABEL", "PROC_DJIBOUTI", 0.5, 0.3, "CP_BABEL"))
    arcs.append(Arc(_aid(), "CP_BABEL", "PROC_MOMBASA", 0.6, 0.8, "CP_BABEL"))

    # Yanbu → Suez (East-West pipeline output reaching Europe via Suez)
    arcs.append(Arc(_aid(), "PROC_YANBU", "CP_SUEZ", 2.0, 1.2, "", True))
    # Yanbu → Bab el-Mandeb (for East African/Indian Ocean markets)
    arcs.append(Arc(_aid(), "PROC_YANBU", "CP_BABEL", 1.5, 1.0, "", True))
    # Yanbu → Sidi Kerir direct (Mediterranean via Red Sea pipeline)
    arcs.append(Arc(_aid(), "PROC_YANBU", "PROC_SIDI", 1.0, 1.5, "", True))

    # Hormuz -> Bab el-Mandeb: Gulf crude for Europe and East Africa transits
    # Bab (then Suez). Tagged to Bab so its closure severs the Red Sea corridor.
    arcs.append(Arc(_aid(), "CP_HORMUZ", "CP_BABEL", 4.0, 1.0, "CP_BABEL"))

    # Djibouti → landlocked East African nations
    for c_code in ["ETH", "DJI"]:
        if c_code in countries:
            arcs.append(Arc(_aid(), "PROC_DJIBOUTI", c_code, 0.3, 0.5))

    # Mombasa → East African nations (Kenya, Uganda, Rwanda, Tanzania)
    for c_code in ["KEN", "UGA", "RWA", "TZA", "MOZ"]:
        if c_code in countries:
            arcs.append(Arc(_aid(), "PROC_MOMBASA", c_code, 0.3, 0.6))

    # Fujairah bypass → Asian refineries
    for proc, cap, cost in [("PROC_JURONG", 1.5, 2.8), ("PROC_JAMNAGAR", 2.0, 2.0),
                             ("PROC_MUMBAI", 1.5, 1.8)]:
        arcs.append(Arc(_aid(), "PROC_FUJAIRAH", proc, cap, cost, "", True))

    # --- Processing → Country arcs (by regional proximity) ---
    asia_countries = [c for c in countries.values() if c.region in ("Asia-Pac.", "Oceania")]
    europe_countries = [c for c in countries.values() if c.region == "Europe"]
    mena_countries = [c for c in countries.values() if c.region == "MENA"]
    africa_countries = [c for c in countries.values() if c.region == "Africa"]
    americas_countries = [c for c in countries.values() if c.region == "Americas"]

    asia_procs = ["PROC_JURONG", "PROC_YOKOHAM", "PROC_ULSAN", "PROC_JAMNAGAR",
                  "PROC_QINGDAO", "PROC_MUMBAI", "PROC_TANJUNG"]
    europe_procs = ["PROC_ROTTER", "PROC_TRIESTE"]
    mena_procs = ["PROC_SIDI", "PROC_JEBEL", "PROC_FUJAIRAH", "PROC_YANBU"]
    africa_procs = ["PROC_DURBAN", "PROC_SIDI", "PROC_DJIBOUTI", "PROC_MOMBASA"]
    americas_procs = ["PROC_HOUSTON", "PROC_SANTOS"]

    def _add_country_arcs(proc_list, country_list, base_cost=0.5):
        # Last-mile cost = regional base handling charge plus a volume-dependent
        # margin that falls with volume (economies of scale), spanning 0.10 $/bbl
        # (largest in region) to 0.50 $/bbl (smallest). Cf. Rodrigue, The
        # Geography of Transport Systems.
        if not country_list:
            return
        vol_max = max(c.daily_oil_mbd for c in country_list)
        for j, proc in enumerate(proc_list):
            # Hubs are listed by descending regional connectivity, so more
            # peripheral hubs carry a small handling premium, making each
            # (hub, country) arc cost distinct.
            hub_term = 0.03 * j
            for c in country_list:
                cap = max(0.1, c.daily_oil_mbd * 0.6)
                scale = (c.daily_oil_mbd / vol_max) if vol_max > 0 else 1.0
                margin = 0.10 + 0.40 * (1.0 - scale)
                arcs.append(Arc(_aid(), proc, c.code, cap,
                               round(base_cost + margin + hub_term, 3)))

    _add_country_arcs(asia_procs, asia_countries, 0.5)
    _add_country_arcs(europe_procs, europe_countries, 0.4)
    _add_country_arcs(mena_procs, mena_countries, 0.3)
    _add_country_arcs(africa_procs, africa_countries, 0.6)
    _add_country_arcs(americas_procs, americas_countries, 0.4)

    # Cross-region emergency sharing arcs: Western hubs supply the most exposed
    # major importers. Gated by sharing policy P5 (is_sharing=True), not P2.
    # Each importer receives up to SHARING_FRAC of its daily demand, split
    # across the two contributing hubs.
    share_hubs = ["PROC_ROTTER", "PROC_HOUSTON"]
    for proc in share_hubs:
        for c in asia_countries[:5]:  # five largest Asian importers
            cap = c.daily_oil_mbd * SHARING_FRAC / len(share_hubs)
            arcs.append(Arc(_aid(), proc, c.code, round(cap, 4), 3.5,
                            chokepoint="", is_bypass=False, is_sharing=True))

    return arcs


# ===================================================================
# Convenience: build everything
# ===================================================================
def build_all():
    """Return (countries, sources, chokepoints, processing, arcs)."""
    countries = build_countries()
    sources = build_sources()
    chokepoints = build_chokepoints()
    processing = build_processing_nodes()
    arcs = build_arcs(sources, chokepoints, processing, countries)
    return countries, sources, chokepoints, processing, arcs


class EnergyNetwork:
    """Directed multi-echelon supply graph over sources, chokepoints, processing
    nodes, and importing countries, with the arcs connecting them."""

    def __init__(
        self,
        sources: Dict[str, SourceRegion],
        chokepoints: Dict[str, Chokepoint],
        processing: Dict[str, ProcessingNode],
        countries: Dict[str, Country],
        arcs: List[Arc],
    ) -> None:
        self.sources = sources
        self.chokepoints = chokepoints
        self.processing = processing
        self.countries = countries
        self.arcs = arcs

        # Node sets
        self.source_nodes: Set[str] = set(sources.keys())
        self.country_nodes: Set[str] = set(countries.keys())

        transshipment = set()
        transshipment.update(chokepoints.keys())
        transshipment.update(processing.keys())
        self.transshipment_nodes: Set[str] = transshipment

        all_nodes = self.source_nodes | self.country_nodes | self.transshipment_nodes
        self.all_nodes = all_nodes

        # Adjacency lists
        self.out_arcs: Dict[str, List[Arc]] = defaultdict(list)
        self.in_arcs: Dict[str, List[Arc]] = defaultdict(list)
        for a in arcs:
            self.out_arcs[a.origin].append(a)
            self.in_arcs[a.destination].append(a)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    def summary(self) -> str:
        """Return a short text summary of the network."""
        n_normal = sum(1 for a in self.arcs if not a.is_bypass)
        n_bypass = sum(1 for a in self.arcs if a.is_bypass)
        lines = [
            f"EnergyNetwork: {len(self.all_nodes)} nodes, "
            f"{len(self.arcs)} arcs",
            f"  Sources:        {len(self.source_nodes)}",
            f"  Transshipment:  {len(self.transshipment_nodes)} "
            f"(chokepoints: {len(self.chokepoints)}, "
            f"hubs: {len(self.processing)})",
            f"  Countries:      {len(self.country_nodes)}",
            f"  Normal arcs:    {n_normal}",
            f"  Bypass arcs:    {n_bypass}",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (f"EnergyNetwork(sources={len(self.sources)}, "
                f"countries={len(self.countries)}, "
                f"arcs={len(self.arcs)})")


# ======================================================================
# MODEL: scenarios, cascade, stochastic MILP
# ======================================================================
import gc
import pyomo.environ as pyo
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple



@dataclass
class Scenario:
    """A single disruption scenario path through the stage horizon."""

    omega: int                     # scenario index
    prob: float                    # probability weight
    states: List[str]              # state at each stage
    availability: List[float]      # chokepoint availability at each stage


# Transition matrices by regime
_TRANSITIONS: Dict[str, Dict[str, Dict[str, float]]] = {
    "standard": {
        "open":       {"open": 0.80, "restricted": 0.15, "closed": 0.05},
        "restricted": {"open": 0.20, "restricted": 0.50, "closed": 0.30},
        "closed":     {"open": 0.05, "restricted": 0.25, "closed": 0.70},
    },
    "escalation": {
        "open":       {"open": 0.60, "restricted": 0.25, "closed": 0.15},
        "restricted": {"open": 0.10, "restricted": 0.40, "closed": 0.50},
        "closed":     {"open": 0.02, "restricted": 0.13, "closed": 0.85},
    },
    "attrition": {
        "open":       {"open": 0.70, "restricted": 0.25, "closed": 0.05},
        "restricted": {"open": 0.25, "restricted": 0.55, "closed": 0.20},
        "closed":     {"open": 0.10, "restricted": 0.35, "closed": 0.55},
    },
}

_STATE_ORDER = {"open": 0, "restricted": 1, "closed": 2}
_STATE_SEQ = ("open", "restricted", "closed")

# ---------------------------------------------------------------------------
# Continuous-time formulation of the disruption dynamics.
#
# The matrices above are calibrated at a reference stage length of 25 days. To
# keep the disruption-duration distribution consistent across stage lengths, we
# treat each as the 25-day transition of a continuous-time Markov chain with
# generator Q, recovered once by the matrix logarithm Q = log(P_ref)/SD_ref, and
# use the stage-t transition expm(Q * SD). At SD = 25 this reproduces the
# reference matrices exactly (to machine precision), so the headline results are
# unchanged, while a different stage length now scales through the same generator
# rather than through an ad hoc rescaling. All three reference matrices are
# embeddable: their generators have non-negative off-diagonal rates and zero row
# sums, so expm(Q * SD) is a valid stochastic matrix for every SD > 0.
STAGE_DAYS_REF = 25.0


def _generator_from_matrix(rows: Dict[str, Dict[str, float]]) -> np.ndarray:
    """Continuous-time generator Q = log(P_ref)/SD_ref for a reference matrix."""
    P = np.array([[rows[i][j] for j in _STATE_SEQ] for i in _STATE_SEQ])
    Q = logm(P).real / STAGE_DAYS_REF
    # Numerical clean-up: clip any tiny negative off-diagonal rate to zero and
    # restore exact zero row sums, so Q is a bona fide generator.
    for i in range(3):
        for j in range(3):
            if i != j and Q[i, j] < 0.0:
                Q[i, j] = 0.0
        Q[i, i] = -sum(Q[i, j] for j in range(3) if j != i)
    return Q


_GENERATORS: Dict[str, np.ndarray] = {
    name: _generator_from_matrix(rows) for name, rows in _TRANSITIONS.items()
}


@lru_cache(maxsize=None)
def _stage_matrix(regime: str, stage_days: float) -> Tuple[Tuple[float, ...], ...]:
    """Stage transition expm(Q * SD) for a regime, as an immutable nested tuple."""
    Q = _GENERATORS.get(regime, _GENERATORS["standard"])
    P = expm(Q * float(stage_days))
    P = np.clip(P, 0.0, None)
    P = P / P.sum(axis=1, keepdims=True)
    return tuple(tuple(float(P[i, j]) for j in range(3)) for i in range(3))


# Decision-dependent uncertainty. When all three instruments in the coordinated
# bundle are active at a stage, that stage's transition is drawn from the
# responsive (de-escalated) matrix, making disruption probabilities a function of
# the policy decisions.
DEESCALATION_BUNDLE: Tuple[str, ...] = ("P1", "P2", "P3")

# De-escalation effectiveness zeta: a coordinated response cuts the adverse
# transition probability by this factor and reallocates the freed mass to
# recovery. The HEADLINE specification is exogenous transitions (zeta = 0): policy
# does not bend the disruption dynamics, since coordinated action has no estimated
# causal effect on whether a chokepoint physically reopens. The policy-dependent
# channel is treated as an exploratory upper bound, reported by re-solving with
# zeta = 0.40 as a sensitivity, not as the main result.
DEESCALATION_FACTOR: float = 0.0


def transition_matrix(regime: str,
                      stage_days: float = STAGE_DAYS) -> Dict[str, Dict[str, float]]:
    """The unresponsive (baseline) transition matrix for a regime.

    Built from the regime's continuous-time generator as expm(Q * stage_days), so
    it is consistent for any stage length. At stage_days = 25 it reproduces the
    calibrated reference matrix to machine precision.
    """
    P = _stage_matrix(regime, float(stage_days))
    return {_STATE_SEQ[i]: {_STATE_SEQ[j]: P[i][j] for j in range(3)}
            for i in range(3)}


def responsive_matrix(unresponsive: Dict[str, Dict[str, float]],
                      rho: Optional[float] = None
                      ) -> Dict[str, Dict[str, float]]:
    """The de-escalated matrix induced by an active coordinated response.

    Adverse transition mass is scaled by (1 - rho) and reallocated to recovery.
    Rows stay valid distributions, and rho = 0 recovers the unresponsive matrix.
    rho defaults to DEESCALATION_FACTOR, read at call time for sensitivity sweeps.
    """
    if rho is None:
        rho = DEESCALATION_FACTOR
    worst = max(_STATE_ORDER.values())
    resp: Dict[str, Dict[str, float]] = {}
    for i, row in unresponsive.items():
        ii = _STATE_ORDER[i]
        adverse = [i] if ii == worst else [j for j in row if _STATE_ORDER[j] > ii]
        favorable = [j for j in row if _STATE_ORDER[j] < ii]
        freed = rho * sum(row[j] for j in adverse)
        new_row = dict(row)
        for j in adverse:
            new_row[j] = row[j] * (1.0 - rho)
        if favorable:
            base = sum(row[j] for j in favorable)
            for j in favorable:
                share = (row[j] / base) if base > 1e-12 else 1.0 / len(favorable)
                new_row[j] += freed * share
        else:
            new_row[i] += freed  # already open, response keeps it open
        resp[i] = new_row
    return resp


# Disruption class -> (initial state, unresponsive regime). The unresponsive
# regime is each class's calibrated baseline. The responsive matrix is derived
# from it, so only coordinated early action bends the transitions.
_CLASS_CONFIG: Dict[str, Tuple[str, str]] = {
    "partial_restriction": ("restricted", "standard"),
    "full_closure":        ("closed",     "standard"),
    "multi_chokepoint":    ("closed",     "escalation"),
    "attrition_mild":      ("open",       "attrition"),
    "attrition_moderate":  ("restricted", "attrition"),
    "attrition_severe":    ("closed",     "attrition"),
}


def class_regime(disruption_class: str) -> Tuple[str, str]:
    """(initial state, unresponsive regime name) for a disruption class."""
    return _CLASS_CONFIG.get(disruption_class, ("closed", "standard"))


def ddu_edge_coefficients(scenario: "Scenario",
                          unresponsive: Dict[str, Dict[str, float]],
                          responsive: Dict[str, Dict[str, float]]
                          ) -> List[Tuple[float, float]]:
    """Per-edge (a, b) so the stage-t transition probability is a + b * d, with d
    the de-escalation indicator. b is the responsive minus unresponsive entry."""
    coeffs = []
    for t in range(len(scenario.states) - 1):
        i, j = scenario.states[t], scenario.states[t + 1]
        a = unresponsive[i][j]
        b = responsive[i][j] - unresponsive[i][j]
        coeffs.append((a, b))
    return coeffs


def decision_dependent_probs(scenarios: List["Scenario"],
                             deescalation: Dict[int, List[int]],
                             unresponsive_regime: str
                             ) -> Dict[int, float]:
    """Realized scenario probabilities given a de-escalation pattern.

    deescalation[w] is the per-stage binary indicators along scenario w's path.
    Returns {w: p_w}, the product of per-edge probabilities under those
    indicators. Used for reporting expectations at the optimal policy.
    """
    unresp = transition_matrix(unresponsive_regime)
    resp = responsive_matrix(unresp)
    probs: Dict[int, float] = {}
    for w, sc in enumerate(scenarios):
        coeffs = ddu_edge_coefficients(sc, unresp, resp)
        d = deescalation.get(w, [0] * len(coeffs))
        p = 1.0
        for (a, b), dt in zip(coeffs, d):
            p *= a + b * dt
        probs[w] = p
    return probs


# Full scenario tree generation
def generate_full_tree(
    initial_state: str = "closed",
    regime: str = "standard",
    num_stages: int = NUM_STAGES,
    stage_days: float = STAGE_DAYS,
) -> List[Scenario]:
    """Enumerate all paths through the Markov disruption tree.

    initial_state is the stage-0 state, regime selects the transition matrix, and
    stage_days sets the per-stage transition through the continuous-time generator.
    Returns every path with its probability, states, and availability.
    """
    trans = transition_matrix(regime, stage_days)

    paths: List[Tuple[List[str], float]] = [([initial_state], 1.0)]

    for _ in range(1, num_stages):
        new_paths = []
        for states, prob in paths:
            current = states[-1]
            for next_state, tp in trans[current].items():
                if tp > 0:
                    new_paths.append((states + [next_state], prob * tp))
        paths = new_paths

    scenarios = []
    for idx, (states, prob) in enumerate(paths):
        avail = [STATE_AVAILABILITY[s] for s in states]
        scenarios.append(Scenario(omega=idx, prob=prob,
                                  states=states, availability=avail))

    return scenarios


# Convenience wrapper
def build_scenario_set(
    disruption_class: str = "full_closure",
    target_count: int = 30,
    stage_days: float = STAGE_DAYS,
) -> List[Scenario]:
    """Generate the full scenario tree for a named disruption class.

    Returns the complete support of disruption paths, each carrying its nominal
    probability under the class's unresponsive regime. Under decision-dependent
    uncertainty the model overrides these with policy-dependent probabilities
    from the same regime. The full tree is needed so the non-anticipativity links
    and per-node de-escalation indicators are well defined. target_count is
    retained for interface compatibility."""
    # Normal-operations benchmark: chokepoint open every stage, so no shortfall,
    # cascade, or price increase. Single deterministic scenario anchoring every
    # reported loss as a delta from undisrupted operations.
    if disruption_class == "no_disruption":
        n = NUM_STAGES
        return [Scenario(omega=0, prob=1.0, states=["open"] * n,
                         availability=[STATE_AVAILABILITY["open"]] * n)]

    initial, regime = class_regime(disruption_class)
    return generate_full_tree(initial_state=initial, regime=regime,
                              stage_days=stage_days)


# Chokepoint set per disruption class
# The multi-chokepoint class closes the three coupled passages at once (Hormuz,
# Bab el-Mandeb, Suez), severing the Gulf-to-Europe corridor and leaving only
# the policy-activated bypasses (Fujairah, East-West pipeline to Yanbu, Cape of
# Good Hope) as outlets for Gulf crude oil.
MULTI_CHOKEPOINT_CLOSURE = ["CP_HORMUZ", "CP_BABEL", "CP_SUEZ"]


def disrupted_chokepoints_for_class(disruption_class: str,
                                    default_chokepoint: str = "CP_HORMUZ"):
    """Chokepoint(s) disrupted in a class. Single chokepoint for all classes
    except multi_chokepoint, which closes MULTI_CHOKEPOINT_CLOSURE at once."""
    if disruption_class == "multi_chokepoint":
        return list(MULTI_CHOKEPOINT_CLOSURE)
    return default_chokepoint


def cascade_single_pass(
    fulfillment: Dict[str, float],
) -> Dict[str, float]:
    """Post-cascade fulfillment ratios from a single beta-matrix pass.

    Matches the first-order inoperability input-output term in the optimization.
    Input and output are sector ratios in [0, 1].
    """
    phi = {s: fulfillment.get(s, 1.0) for s in SECTORS}

    phi_adj = {}
    for s in SECTORS:
        stress = 0.0
        for sp, beta_val in BETA.get(s, {}).items():
            stress += beta_val * (1.0 - phi.get(sp, 1.0))
        # Stress cuts fulfillment proportionally
        phi_adj[s] = max(0.0, min(1.0, phi[s] * (1.0 - stress)))

    return phi_adj


def compute_stress(
    fulfillment: Dict[str, float],
) -> Dict[str, float]:
    """Single-pass stress index per sector from fulfillment ratios in [0, 1]."""
    stress = {}
    for s in SECTORS:
        u = 0.0
        for sp, beta_val in BETA.get(s, {}).items():
            u += beta_val * (1.0 - fulfillment.get(sp, 1.0))
        stress[s] = u
    return stress


def compute_cascade_losses(
    country,
    chokepoint_dep: float,
    duration_days: int,
) -> Dict:
    """Decompose one country's output losses into direct, cascade, and
    bottleneck components.

    chokepoint_dep is the supply fraction via the disrupted chokepoint. Returns a
    per-sector dict (direct/cascade/bottleneck/total) plus _summary.
    """
    energy_red = chokepoint_dep  # oil reduction fraction
    spr_offset = min(energy_red,
                     (country.spr_days / max(duration_days, 1)) * energy_red)
    net_red = max(0.0, energy_red - spr_offset)

    # Pre-cascade fulfillment
    direct_fulfill = {}
    for s in SECTORS:
        w = country.sector_weights.get(s, 0.0)
        direct_fulfill[s] = max(0.0, 1.0 - net_red * w)

    stress = compute_stress(direct_fulfill)

    cascade_fulfill = {}
    for s in SECTORS:
        cascade_fulfill[s] = max(0.0, min(1.0,
            direct_fulfill[s] * (1.0 - stress[s])))

    result = {}
    weighted_total = 0.0
    for s in SECTORS:
        w = country.sector_weights.get(s, 0.0)
        direct_loss = (1.0 - direct_fulfill[s]) * 100.0
        cascade_adj_loss = (1.0 - cascade_fulfill[s]) * 100.0
        cascade_add = max(0.0, cascade_adj_loss - direct_loss)

        # Bottleneck: extra PWL damage once stress exceeds sector threshold tau
        tau, mu = THRESHOLDS.get(s, (0.5, 2.0))
        u = stress[s]
        bottleneck = mu * max(0.0, u - tau) ** 2 * 100.0

        total = direct_loss + cascade_add + bottleneck

        result[s] = {
            "direct": round(direct_loss, 2),
            "cascade": round(cascade_add, 2),
            "bottleneck": round(bottleneck, 2),
            "total": round(total, 2),
        }
        weighted_total += (1.0 - cascade_fulfill[s]) * w

    result["_summary"] = {
        "weighted_output_loss": weighted_total,
        "net_oil_reduction": net_red,
    }
    return result


def build_damage_segments() -> Dict[str, List[Tuple[float, float]]]:
    """PWL damage segments per sector for the MIP epigraph: sector -> [(slope, intercept)]."""
    return {s: build_pwl_segments(s) for s in SECTORS}


def realized_scenario_probs(m) -> Dict[int, float]:
    """Scenario probabilities implied by a solved model.

    Under decision-dependent uncertainty these follow from the optimal
    de-escalation pattern, so post-solve reporting reweights by them rather than
    the nominal probabilities. Falls back to nominal when uncertainty is exogenous.
    """
    scen = m._scenarios_ref
    if getattr(m, "ddu_active", False):
        n_edges = len(scen[0].states) - 1
        deesc = {w: [int(round(pyo.value(m.dd[t, w]))) for t in range(n_edges)]
                 for w in range(len(scen))}
        return decision_dependent_probs(scen, deesc, m._unresponsive_regime)
    return {w: float(pyo.value(m.prob[w])) for w in range(len(scen))}


def _cost_upper_bound(m, network, enable_cascade, SD, equity_weight):
    """Tight upper bound on a scenario's total cost C_w, the big-M for the
    McCormick products in the decision-dependent objective. Each component is a
    worst case, and the equity weight scales only the priority pairs it multiplies
    in the objective, keeping the bound tight at high weights."""
    eq = max(1.0, float(equity_weight))
    prio = set(m.K_PRIO)

    def _eq(n, s):
        return eq if (n, s) in prio else 1.0

    # Every barrel short costs at most its penalty (with the equity weight on
    # priority pairs), and h <= demand bounds it.
    short = sum(pyo.value(m.demand[n, s, t]) * pyo.value(m.pi_ns[n, s]) * _eq(n, s)
                for n in m.C for s in m.S for t in m.T)
    # price_loss[t,w] <= max(0, gross - spare) (realized rerouting only lowers it),
    # so its worst case per stage bounds the price transfer.
    price = sum(
        pyo.value(m.price_coef[t]) * max(
            (max(0.0, pyo.value(m.gross_disruption[t, w]) - float(m.spare_stage))
             for w in m.OMEGA), default=0.0)
        for t in m.T)
    casc = 0.0
    if enable_cascade:
        DELTA_MAX = 1.0  # normalized Delta = g_s(u)/g_s(1) is bounded by 1
        casc = sum(pyo.value(m.eta_ns[n, s]) * pyo.value(m.casc_scale[n, s, t]) * _eq(n, s)
                   for n in m.C for s in m.S for t in m.T) * DELTA_MAX
    floor = FLOOR_PENALTY * sum(
        pyo.value(m.phi_floor[n, s]) * pyo.value(m.demand[n, s, t])
        for (n, s) in m.K_PRIO for t in m.T)
    pol = sum(pyo.value(m.gamma[p]) for p in m.P for _ in m.T)
    max_cost = max((pyo.value(m.arc_cost[a]) for a in m.A), default=0.0)
    total_flow = sum(pyo.value(m.demand[n, s, t])
                     for n in m.C for s in m.S for t in m.T)
    transport = total_flow * max_cost * (1.0 + FREIGHT_PREMIUM) * 1.5
    # Managed-curtailment and reserve-replacement costs are each bounded by their
    # unit cost times total baseline volume (r, rho*d <= demand).
    curtail = BASELINE_OIL_PRICE * total_flow
    reserve = RESERVE_REPLACEMENT_COST * total_flow
    substitute = SUBSTITUTE_SUPPLY_COST * total_flow
    return 1.15 * (short + price + casc + floor + pol + transport
                   + curtail + reserve + substitute)


def build_model(
    network: EnergyNetwork,
    scenarios: List[Scenario],
    disrupted_chokepoint: str = "CP_HORMUZ",
    duration_days: int = 100,
    enable_cascade: bool = True,
    endogenous_uncertainty: bool = True,
    unresponsive_regime: str = "standard",
    policy_subset: Optional[List[str]] = None,
    stage_days: Optional[int] = None,
    service_floor: float = 0.60,
    equity_weight: float = 1.0,
    enable_price: bool = True,
    demand_scale: float = 1.0,
    coordinated_price: bool = True,
    coordinating_countries: Optional[Iterable[str]] = None,
    amp_spec: str = "headline",
    restrict_substitute: bool = False,
    displacement: Optional[dict] = None,
    price_curve: str = "linear",
) -> pyo.ConcreteModel:
    """Build the complete stochastic MIP.

    coordinated_price / coordinating_countries control the decentralization analysis
    (the world-price public good). With coordinated_price=True and
    coordinating_countries=None (default, the planner), the aggregate managed demand
    reduction (conservation) of ALL importers moderates the world price. With
    coordinated_price=False (the atomistic price-taking lower-bound benchmark), no
    importer internalizes its price impact, so the price responds only to the
    structural shortfall net of spare and rerouting. With coordinating_countries set to
    a subset, only that coalition's conservation moderates the price (partial
    coordination), used to build the cooperative-game value.

    enable_cascade=False omits cascade amplification and policy_subset restricts
    available policies (ablations). stage_days sets the per-stage length in days
    (default STAGE_DAYS=25). Demand and capacities scale with it, so longer
    durations (180, 270, 365) enlarge stage length, not stage count.
    """
    # Stage length in days. Demand and capacities scale linearly with SD, so a
    # fractional value represents a horizon as NUM_STAGES equal stages.
    SD = STAGE_DAYS if stage_days is None else float(stage_days)

    m = pyo.ConcreteModel("GeoEnergy_SCRO")
    # Kept for post-solve reporting: realized scenario probabilities depend on the
    # de-escalation decisions, so reporting must reweight by them, not the nominal.
    m._scenarios_ref = scenarios
    m._unresponsive_regime = unresponsive_regime

    # ------------------------------------------------------------------
    # Sets
    # ------------------------------------------------------------------
    stages = list(range(NUM_STAGES))
    omega_set = list(range(len(scenarios)))

    m.T = pyo.Set(initialize=stages)
    m.OMEGA = pyo.Set(initialize=omega_set)
    m.S = pyo.Set(initialize=SECTORS)
    m.A = pyo.Set(initialize=[a.aid for a in network.arcs])
    m.C = pyo.Set(initialize=list(network.country_nodes))
    m.E = pyo.Set(initialize=list(network.source_nodes))
    m.J = pyo.Set(initialize=list(network.transshipment_nodes))

    # Available policies
    if policy_subset is None:
        policy_subset = [p.pid for p in POLICIES]
    m.P = pyo.Set(initialize=policy_subset)

    # Mobilizable arcs: rerouting/bypass (gated by P2/P4) and bilateral-sharing
    # (gated by P5). Both carry flow only up to a policy-activated capacity y.
    alt_aids = [a.aid for a in network.arcs if a.is_bypass or a.is_sharing]
    m.A_ALT = pyo.Set(initialize=alt_aids)

    # Irreversible policies
    irr_pols = [p.pid for p in POLICIES if p.irreversible and p.pid in policy_subset]
    m.P_IRR = pyo.Set(initialize=irr_pols)

    # PWL segments
    damage_segs = build_damage_segments()
    seg_indices = list(range(max(len(v) for v in damage_segs.values())))
    m.L = pyo.Set(initialize=seg_indices)

    # Priority country-sector pairs (service floor)
    priority_pairs = []
    for c in network.countries.values():
        if c.spr_days < PRIORITY_SPR_THRESHOLD:  # low-buffer, vulnerable countries
            priority_pairs.append((c.code, "healthcare"))
            priority_pairs.append((c.code, "electricity"))
    m.K_PRIO = pyo.Set(initialize=priority_pairs, dimen=2)

    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------
    arc_dict = {a.aid: a for a in network.arcs}

    # Per-stage arc capacity (Mbbl), scenario-dependent via chokepoint
    # availability. The disrupted chokepoint may be one id or a set (the
    # multi-chokepoint case closes Hormuz, Bab el-Mandeb, Suez together), so
    # normalise to a set for membership.
    _disrupted = ({disrupted_chokepoint} if isinstance(disrupted_chokepoint, str)
                  else set(disrupted_chokepoint))

    # Maritime sea legs carry the freight / war-risk premium under disruption.
    # Pipelines and last-mile inland delivery (arcs into a country node) are
    # excluded, being unexposed to tanker freight spikes or hull war-risk insurance.
    _country_ids = set(network.country_nodes)
    _maritime = {a.aid for a in network.arcs
                 if not a.is_pipeline and a.destination not in _country_ids}
    m._maritime_ids = _maritime  # used by the cost-component decomposition

    # Escape routes for the disrupted chokepoint(s): bypass arcs tagged with the
    # chokepoint they physically relieve. Only their realized flow counts as
    # restored world supply, capped per chokepoint by the physical bypass
    # capacity. Arcs relieving a non-disrupted chokepoint do not count.
    _escape_by_cp = {}
    for a in network.arcs:
        cp = getattr(a, "relieves", "")
        if cp and cp in _disrupted:
            _escape_by_cp.setdefault(cp, []).append(a.aid)
    _escape_aids = [aid for aids in _escape_by_cp.values() for aid in aids]

    # Maritime freight / war-risk multiplier per stage and scenario.
    def mar_mult_rule(m, t, w):
        return 1.0 + FREIGHT_PREMIUM * (1.0 - scenarios[w].availability[t])
    m.mar_mult = pyo.Param(m.T, m.OMEGA, initialize=mar_mult_rule)

    def arc_cap_rule(m, a, t, w):
        arc = arc_dict[a]
        avail = scenarios[w].availability[t]
        if arc.chokepoint in _disrupted:
            if arc.is_bypass:
                return arc.capacity_mbd * SD
            return arc.capacity_mbd * avail * SD
        if arc.is_bypass:
            return 0.0
        return arc.capacity_mbd * SD
    m.arc_cap = pyo.Param(m.A, m.T, m.OMEGA, initialize=arc_cap_rule, default=0.0)

    # Arc unit cost
    m.arc_cost = pyo.Param(m.A, initialize={a.aid: a.unit_cost for a in network.arcs})

    # Source capacities (per stage, in Mbbl)
    def source_cap_rule(m, e, t):
        src = network.sources.get(e)
        return src.capacity_mbd * SD if src else 0.0
    m.source_cap = pyo.Param(m.E, m.T, initialize=source_cap_rule)

    # Country demand by sector (Mbbl per stage)
    def demand_rule(m, n, s, t):
        c = network.countries.get(n)
        if c is None:
            return 0.0
        # demand_scale feeds a price-induced consumption contraction back into
        # the physical demand balance (market-clearing convergence check).
        return c.daily_oil_mbd * c.sector_weights.get(s, 0.0) * SD * demand_scale
    m.demand = pyo.Param(m.C, m.S, m.T, initialize=demand_rule)

    # Baseline non-chokepoint supply (Mbbl per stage). The share of a country's
    # crude oil not routed through the disrupted chokepoint(s) is always
    # available, so only the dependent share is at risk and reported losses are
    # clean deltas from undisrupted operations.
    _DEP_ATTR = {"CP_HORMUZ": "hormuz_dep", "CP_MALACCA": "malacca_dep",
                 "CP_SUEZ": "suez_dep"}

    def base_supply_rule(m, n, t):
        c = network.countries.get(n)
        if c is None:
            return 0.0
        # Route share of imports through the disrupted passage(s).
        route_share = 0.0
        for cp in _disrupted:
            attr = _DEP_ATTR.get(cp)
            if attr:
                route_share += getattr(c, attr, 0.0)
        route_share = min(1.0, route_share)
        # Exposure applies only to the imported fraction of demand: the
        # domestic-production and non-imported share is never at risk. exposure =
        # import_dependence * route_share.
        exposure = min(1.0, IMPORT_SHARE.get(n, 1.0) * route_share)
        return (1.0 - exposure) * c.daily_oil_mbd * SD
    m.base_cap = pyo.Param(m.C, m.T, initialize=base_supply_rule)

    # Initial reserves
    def reserve_init_rule(m, n):
        c = network.countries.get(n)
        if c is None:
            return 0.0
        return c.daily_oil_mbd * c.spr_days
    m.R_init = pyo.Param(m.C, initialize=reserve_init_rule)

    # Max drawdown per stage (Mbbl)
    def max_draw_rule(m, n):
        c = network.countries.get(n)
        if c is None:
            return 0.0
        return c.spr_max_draw_rate * SD
    m.r_max = pyo.Param(m.C, initialize=max_draw_rule)

    # Per-stage policy activation cost, rescaled $B to $M to match objective
    # units (shortage penalty in $/bbl times demand in Mbbl).
    COST_SCALE_USD_M_PER_USD_B = 1000.0
    # Activation costs and budgets are quoted per 25-day reference stage and scaled
    # by the actual stage length SD/25, so keeping a policy active
    # costs in proportion to calendar time. At the 25-day headline the factor is one,
    # and a duration experiment that lengthens the stage now charges activation per
    # day rather than a flat amount per stage.
    _stage_cost_factor = float(SD) / 25.0
    m.gamma = pyo.Param(
        m.P,
        initialize={p.pid: p.cost * COST_SCALE_USD_M_PER_USD_B * _stage_cost_factor
                    for p in POLICIES if p.pid in policy_subset},
    )

    # Budget per stage ($M, rescaled from config's $B, scaled to stage length)
    m.budget = pyo.Param(
        m.T,
        initialize={t: STAGE_BUDGETS[t] * COST_SCALE_USD_M_PER_USD_B * _stage_cost_factor
                    for t in stages},
    )

    # Scenario probabilities
    m.prob = pyo.Param(m.OMEGA, initialize={w: scenarios[w].prob for w in omega_set})

    # Shortage penalty pi_ns ($/barrel) and cascade penalty eta_ns, from the
    # CONFIG section (SHORTAGE_PENALTY / CASCADE_WEIGHT) so the extensive form and
    # the decomposition share one source of truth.
    m.pi_ns = pyo.Param(m.C, m.S,
                         initialize={(n, s): SHORTAGE_PENALTY[s]
                                     for n in network.country_nodes
                                     for s in SECTORS})

    # Amplification damage intensity eta_s (dimensionless multiplier). The damage
    # function is normalized so its inoperability index Delta = g_s(u)/g_s(1) lies in
    # [0,1], and eta_s = VA_LOSS_FRAC * g_s(1) restores the
    # sector-specific damage intensity: larger for critical services (healthcare,
    # electricity) whose failure inflicts economic and human costs exceeding their own
    # value added (the cost of unserved critical load). The objective term
    # eta_s * VA_nst * Delta = VA_nst * g_s(u) is therefore identical to before.
    # Amplification specification. The capped variant bounds
    # the loss at own value added (eta_s = min(1, g_s(1))), the Leontief variant keeps
    # eta_s = g_s(1) but uses the total-requirements cascade matrix in the stress
    # constraint below. 'headline' is the reported normalized first-pass model.
    cascade_mat = cascade_matrix_for(amp_spec)
    def _eta_of(s):
        g1 = _damage_norm(s)
        return VA_LOSS_FRAC * (min(1.0, g1) if amp_spec == "capped" else g1)
    m.eta_ns = pyo.Param(m.C, m.S,
                          initialize={(n, s): _eta_of(s)
                                      for n in network.country_nodes
                                      for s in SECTORS})

    # Value-added base for the amplification term: the STAGE value added of the
    # country-sector, VA_nst = GDP_n * VA_SHARE_s * (SD/365), in the model's
    # dollar-million units (GDP is in $B, so times 1000). Scaling annual value added
    # to the stage length keeps the loss time-consistent, so a shortfall over a
    # 25-day stage can destroy at most that stage's value added, not a full year's
    #. The amplification loss is then eta * casc_scale * Delta =
    # eta * VA_nst * g_s(u_nst): with eta = 1 the loss is the sector's stage value
    # added times its inoperability g_s(u), the standard IIM output loss.
    def _va_base(n, s):
        c = network.countries.get(n)
        gdp_b = getattr(c, "gdp_B", 0.0) if c is not None else 0.0
        return float(gdp_b) * VA_SHARE[s] * 1000.0 * (SD / 365.0)
    m.casc_scale = pyo.Param(
        m.C, m.S, m.T,
        initialize={(n, s, t): _va_base(n, s)
                    for n in m.C for s in m.S for t in m.T},
        default=0.0)

    # Max demand reduction fractions (MAX_DEMAND_RED)
    m.rho_max = pyo.Param(m.C, m.S,
                           initialize={(n, s): MAX_DEMAND_RED[s]
                                       for n in network.country_nodes
                                       for s in SECTORS})

    # Service floor for priority pairs (tunable for the equity-frontier sweep)
    m.phi_floor = pyo.Param(m.K_PRIO, initialize=float(service_floor),
                            default=float(service_floor))

    # Equity weight on the objective penalty of vulnerable countries' priority
    # sectors (K_PRIO pairs). Raising it above 1 makes the planner prioritise
    # vulnerable economies, tracing the efficiency-equity frontier. Reported
    # tables always evaluate damage at the true penalties (pi_ns / eta_ns).
    prio_set = set(priority_pairs)
    m.equity_mult = pyo.Param(
        m.C, m.S,
        initialize={(n, s): (float(equity_weight) if (n, s) in prio_set else 1.0)
                    for n in network.country_nodes for s in SECTORS},
        default=1.0,
    )

    # ------------------------------------------------------------------
    # Variables
    # ------------------------------------------------------------------
    m.x = pyo.Var(m.A, m.T, m.OMEGA, within=pyo.NonNegativeReals)   # arc flow
    m.q = pyo.Var(m.C, m.S, m.T, m.OMEGA, within=pyo.NonNegativeReals)  # served
    m.h = pyo.Var(m.C, m.S, m.T, m.OMEGA, within=pyo.NonNegativeReals)  # shortage
    m.r = pyo.Var(m.C, m.T, m.OMEGA, within=pyo.NonNegativeReals)   # reserve draw
    m.R = pyo.Var(m.C, m.T, m.OMEGA, within=pyo.NonNegativeReals)   # reserve level
    m.g = pyo.Var(m.C, m.T, m.OMEGA, within=pyo.NonNegativeReals)   # substitute supply (total)
    # P6 sub-instruments: product-stock release, refinery-yield
    # reallocation, inter-fuel switching, plus the depleting product-stock level.
    m.g_ps = pyo.Var(m.C, m.T, m.OMEGA, within=pyo.NonNegativeReals)
    m.g_rf = pyo.Var(m.C, m.T, m.OMEGA, within=pyo.NonNegativeReals)
    m.g_fs = pyo.Var(m.C, m.T, m.OMEGA, within=pyo.NonNegativeReals)
    m.PS = pyo.Var(m.C, m.T, m.OMEGA, within=pyo.NonNegativeReals)   # product-stock level
    m.base = pyo.Var(m.C, m.T, m.OMEGA, within=pyo.NonNegativeReals)  # baseline supply
    m.y = pyo.Var(m.A_ALT, m.T, m.OMEGA, within=pyo.NonNegativeReals)  # bypass flow
    m.rho = pyo.Var(m.C, m.S, m.T, m.OMEGA, within=pyo.NonNegativeReals)  # demand red (total)
    # The coalition moderates the world price purely from the demand side, through
    # coordinated demand restraint (conservation). Reserves and substitute supply are
    # domestic coping that meet a country's own shortfall and do not add crude to the
    # world market, so they carry no price effect. Conservation, by contrast, is a
    # conserved demand withdrawal: the barrels the coalition does not consume are ones
    # it does not draw from the tight world market, so they stay available to the rest
    # of the world and ease the price, with every barrel accounted for once.
    m.z = pyo.Var(m.P, m.T, m.OMEGA, within=pyo.Binary)             # policy on/off
    m.fslack = pyo.Var(m.K_PRIO, m.T, m.OMEGA, within=pyo.NonNegativeReals)  # priority-floor breach

    if enable_cascade:
        m.Delta = pyo.Var(m.C, m.S, m.T, m.OMEGA, within=pyo.NonNegativeReals)
        m.u_stress = pyo.Var(m.C, m.S, m.T, m.OMEGA, within=pyo.NonNegativeReals)

    # The oil-price premium dP_{t,w} is defined below, once gross_disruption and
    # global spare capacity are in scope. It is levied on served oil q, so it
    # needs the exogenous world-supply loss, not the endogenous shortfall.
    world_supply_stage = WORLD_OIL_SUPPLY_MBD * SD

    # Sector-specific price-contraction weights kappa_s, rescaled by
    # one factor so the demand-weighted mean is 1. This keeps the coalition-wide
    # contraction (and hence the deadweight and market clearing) exactly as in the
    # uniform case while redistributing incidence toward elastic sectors. The realized
    # contraction of sector (n,s) is kappa_s * price_loss / world_supply_stage.
    _D_s = {s: sum(pyo.value(m.demand[n, s, t]) for n in m.C for t in stages)
            for s in SECTORS}
    _tot_D = sum(_D_s.values())
    _wmean = sum(PRICE_CONTRACTION_WEIGHT[s] * _D_s[s] for s in SECTORS) / max(_tot_D, 1e-9)
    m.kappa_contract = pyo.Param(
        m.S, initialize={s: PRICE_CONTRACTION_WEIGHT[s] / max(_wmean, 1e-9)
                         for s in SECTORS}, default=1.0)

    # Regional cascade-intensity scale, renormalized to a demand-weighted
    # mean of 1 so the aggregate calibrated intensity is unchanged and only its
    # regional incidence differs. Applied to beta in the stress constraint below.
    _Dn = {n: sum(pyo.value(m.demand[n, s, t]) for s in m.S for t in stages) for n in m.C}
    _rraw = {n: cascade_region_scale(n) for n in m.C}
    _rmean = sum(_rraw[n] * _Dn[n] for n in m.C) / max(sum(_Dn.values()), 1e-9)
    m.region_cascade_scale = pyo.Param(
        m.C, initialize={n: _rraw[n] / max(_rmean, 1e-9) for n in m.C}, default=1.0)

    # Gross UNIQUE supply removed from the world market per stage/scenario (Mbbl).
    # When several chokepoints on the same corridor close together
    # (Hormuz, Bab el-Mandeb, Suez), a Gulf-to-Europe barrel traverses them
    # SEQUENTIALLY, so summing the raw passage throughputs counts that barrel up to
    # three times. We instead count each baseline barrel exactly once as the CUT
    # INFLOW into the disrupted set: crude that enters the set from a non-disrupted
    # origin (a source or an open chokepoint), summed over the normal (non-bypass,
    # non-sharing) arcs crossing the cut. A barrel already inside the set that relays
    # from one closed passage to another has a disrupted origin and is therefore not
    # recounted. The cut inflow is capped at the set's total calibrated throughput as
    # a consistency bound (an entry arc's nameplate capacity can exceed the receiving
    # passage's EIA throughput). For a single closure this reduces to that passage's
    # throughput, for {Hormuz, Bab el-Mandeb, Suez} it is 24.5 mbd, not the naive
    # 32.2 mbd. _assert_unique_barrel (preflight) checks no barrel is counted twice.
    _throughput_sum = sum(network.chokepoints[cp].throughput_mbd
                          for cp in _disrupted if cp in network.chokepoints)
    _cut_inflow = sum(
        a.capacity_mbd for a in network.arcs
        if (not a.is_bypass) and (not getattr(a, "is_sharing", False))
        and (a.destination in _disrupted) and (a.origin not in _disrupted))
    _unique_disruption_rate = min(_cut_inflow, _throughput_sum)
    # Exposed for the automated unique-barrel accounting test and for reporting.
    m._unique_disruption_rate = float(_unique_disruption_rate)
    m._naive_throughput_sum = float(_throughput_sum)

    def gross_disruption_rule(m, t, w):
        avail = scenarios[w].availability[t]
        return _unique_disruption_rate * (1.0 - avail) * SD
    m.gross_disruption = pyo.Param(m.T, m.OMEGA, initialize=gross_disruption_rule,
                                   default=0.0)
    m.spare_stage = float(GLOBAL_SPARE_CAPACITY_MBD * SD)

    # Oil-price welfare cost (deadweight). The world price rises with the residual
    # GLOBAL market imbalance: the gross supply removed at the disrupted passages,
    # net of every action that returns crude to the world market or withdraws demand
    # from it. Three channels moderate the price, each counted once: (i) global spare
    # production, (ii) crude coordinated rerouting ACTUALLY returns on the escape arcs
    # (realized flow, not the installed ceiling, each rerouted barrel crosses exactly
    # one escape arc, so the sum over escape arcs is a cut-set), and (iii) the
    # coalition's managed demand reduction (conservation), a conserved demand
    # withdrawal whose un-consumed barrels stay available to the rest of the world.
    # Reserves and substitute supply are domestic coping that meet a country's own
    # shortfall and add no crude to the world market, so they carry no price effect
    # and no barrel is both consumed at home and credited to the market. This is the
    # market-clearing reduced form: net imbalance = disrupted unique barrels - spare
    # - rerouted - conservation. We close the world market with an isoelastic inverse
    # demand
    # P(Q)=P0 (Q/Qbar)^(-1/|eta|): a net loss dQ raises the price by
    # dP/P0 = (1/|eta|) dQ/Qbar and contracts consumption, so the world clears at the
    # reduced quantity. The REAL resource cost of the price channel is then the
    # deadweight triangle DWL = 1/2 dP dQ_coalition, NOT the full terms-of-trade
    # transfer (which is a redistribution to producers, reported separately in
    # analysis). The objective charges only this deadweight. It is convex quadratic
    # in the realized net loss, linearized exactly from below by its tangents at a
    # grid of breakpoints, so the program stays an exact MILP.
    _Q0_stage = {t: sum(pyo.value(m.demand[n, s, t]) for n in m.C for s in m.S)
                 for t in stages}
    # Terms-of-trade transfer coefficient dP*Q0 (retained for post-solve reporting
    # of importer expenditure and the producer transfer, not in the objective).
    m.price_coef = pyo.Param(
        m.T,
        initialize={t: (BASELINE_OIL_PRICE * _Q0_stage[t]
                        / (world_supply_stage * abs(OIL_DEMAND_ELASTICITY)))
                    for t in stages},
        default=0.0)
    # Coalition-share deadweight coefficient c_t, with DWL_t = c_t (dQ)^2, obtained
    # from 1/2 dP dQ_coalition, dP = P0 (1/|eta|) dQ/Qbar, dQ_coalition = (Q0/Qbar) dQ.
    m.dwl_coef = pyo.Param(
        m.T,
        initialize={t: (BASELINE_OIL_PRICE * _Q0_stage[t]
                        / (2.0 * abs(OIL_DEMAND_ELASTICITY) * world_supply_stage ** 2))
                    for t in stages},
        default=0.0)

    # Realized net world-supply loss (Mbbl/stage). A free non-negative variable
    # bounded below by gross disruption net of spare capacity and realized
    # rerouting, at the optimum it equals max(0, gross - spare - rerouted).
    m.price_loss = pyo.Var(m.T, m.OMEGA, within=pyo.NonNegativeReals)

    # Material-balance closure. Where the gross disruption net of
    # spare capacity exceeds the total escape (bypass) capacity, the residual world
    # imbalance is unabsorbable by rerouting alone, so the price rations it exactly and
    # price_loss is PINNED by equality (below), with the induced demand contraction
    # netted out of the country balances (see demand_balance_con). This closes the
    # market in one balance and removes any incentive to inflate the price: a barrel
    # cannot count as both an involuntary shortage and an uncounted price-induced
    # contraction. Rerouting and coordinated conservation
    # all lower the residual and hence the contraction. Where rerouting alone can
    # absorb the loss we keep the one-sided bound (the deadweight pins it to max(0,.)).
    _escape_cap_total = float(sum(
        network.chokepoints[cp].bypass_capacity_mbd for cp in _disrupted
        if cp in network.chokepoints) * SD)
    _net_ok = {(t, w): (pyo.value(m.gross_disruption[t, w]) - float(m.spare_stage)
                        > _escape_cap_total)
               for t in stages for w in m.OMEGA}
    m._net_ok = _net_ok

    # Decentralization control (Management Science coordination analysis). The price-
    # moderating demand-side action (coordinated conservation) is a PUBLIC GOOD: it
    # lowers the world price for every importer. Only the coordinating set internalizes
    # this. The planner is the full set, the atomistic price-taking benchmark is the
    # empty set (structural price only), a strict subset is partial coordination, used
    # for the cooperative-game value.
    if not coordinated_price:
        _coord = set()
    elif coordinating_countries is None:
        _coord = set(m.C)
    else:
        _coord = set(coordinating_countries) & set(m.C)
    m._coord_set = _coord

    # Channel-specific market-displacement of domestic coping. By
    # default reserve draws and substitute supply meet a country's own cut shortfall and
    # displace no world-market purchase (phi=0, the headline). Setting a positive
    # coefficient lets a share of each channel's volume also reduce world demand, so it
    # moderates the price like conservation: it enters the incentive residual for the
    # coordinating set and the physical residual for all importers. Cases phi in
    # {0, 0.5, 1} span zero, partial, and one-for-one displacement.
    _disp = displacement or {}
    _phi = {k: float(_disp.get(k, 0.0))
            for k in ("reserve", "product_stock", "refinery", "fuel_switch")}
    _disp_on = any(v != 0.0 for v in _phi.values())
    m._displacement = dict(_phi)

    def _displaced(nset, t, w):
        if not _disp_on:
            return 0.0
        return sum(_phi["reserve"] * m.r[n, t, w]
                   + _phi["product_stock"] * m.g_ps[n, t, w]
                   + _phi["refinery"] * m.g_rf[n, t, w]
                   + _phi["fuel_switch"] * m.g_fs[n, t, w]
                   for n in nset)

    def price_loss_def(m, t, w):
        if not enable_price:
            return m.price_loss[t, w] <= 0.0
        # Net world market imbalance = gross - spare - realized rerouting
        #   - reserves released to market - substitute released to market
        #   - managed demand reduction (conservation, in physical barrels),
        # where only the coordinating set's demand-side actions moderate the price.
        conservation = sum(m.rho[n, s, t, w] * m.demand[n, s, t]
                           for n in _coord for s in m.S)
        rhs = (m.gross_disruption[t, w] - m.spare_stage
               - sum(m.x[a, t, w] for a in _escape_aids)
               - conservation - _displaced(_coord, t, w))
        # Market-clearing PIN. Where the loss is structurally unabsorbable, the price
        # rations EXACTLY the residual imbalance, so price_loss == rhs (equality). This
        # is essential: a one-sided bound would let the planner inflate price_loss for
        # free and reclassify costly hard shortage as cheap price-rationing. Because
        # price_loss >= 0 (domain), the equality also caps the market-facing response
        # at the physical loss (no barrel creates a below-baseline price surplus).
        # Where rerouting alone can absorb the loss, we keep the one-sided bound (the
        # deadweight pins price_loss to max(0, rhs)) and net no contraction.
        if _net_ok[(t, w)]:
            return m.price_loss[t, w] == rhs
        return m.price_loss[t, w] >= rhs
    m.con_price_loss = pyo.Constraint(m.T, m.OMEGA, rule=price_loss_def)

    # Physical market residual. price_loss above keeps only the
    # coordinating set's conservation and enters the OBJECTIVE deadweight: it is the
    # price moderation a price-taking importer does not internalize when choosing its own
    # restraint. But a realized managed reduction physically lowers world demand
    # regardless of who chose it, so the PHYSICAL residual nets out EVERY importer's
    # conservation. It drives the consumption contraction (demand balance and priority
    # floor below) and the reported deadweight and terms-of-trade transfer. For the
    # planner (coordinating set = all importers) the two residuals coincide, so the
    # headline is unchanged, the atomistic and partial-coalition benchmarks now clear
    # physically on the reductions they actually select, while only the coordinating set
    # internalizes the price effect of its own reductions.
    m.price_loss_phys = pyo.Var(m.T, m.OMEGA, within=pyo.NonNegativeReals)

    def price_loss_phys_def(m, t, w):
        if not enable_price:
            return m.price_loss_phys[t, w] <= 0.0
        conservation_all = sum(m.rho[n, s, t, w] * m.demand[n, s, t]
                               for n in m.C for s in m.S)
        rhs = (m.gross_disruption[t, w] - m.spare_stage
               - sum(m.x[a, t, w] for a in _escape_aids)
               - conservation_all - _displaced(m.C, t, w))
        if _net_ok[(t, w)]:
            return m.price_loss_phys[t, w] == rhs
        return m.price_loss_phys[t, w] >= rhs
    m.con_price_loss_phys = pyo.Constraint(m.T, m.OMEGA, rule=price_loss_phys_def)

    # Deadweight welfare loss, PWL-linearized from below by the tangents of the
    # convex quadratic c_t (dQ)^2 at NBP+1 breakpoints spanning [0, dQ_max].
    m.dwl = pyo.Var(m.T, m.OMEGA, within=pyo.NonNegativeReals)
    _dq_max = max([pyo.value(m.gross_disruption[t, w])
                   for t in stages for w in m.OMEGA] + [1.0])
    # Breakpoint count for the deadweight tangent grid. Configurable via the
    # environment so a grid-refinement test can double it without
    # touching the default 12-breakpoint headline configuration.
    _NBP = int(os.environ.get("GEOENERGY_DWL_NBP", "12"))
    _bp = [_dq_max * k / _NBP for k in range(_NBP + 1)]
    m.BP = pyo.RangeSet(0, _NBP)
    m._dwl_bp = list(_bp)  # breakpoints for post-solve PWL deadweight on the physical residual

    # Isoelastic (constant-elasticity) price benchmark. The
    # market clears at the same reduced quantity in either curve, so only the price, and
    # hence the deadweight area and the transfer, differ. With P(q)=P0 (q/Qbar)^{-eps},
    # eps=1/|eta|, the coalition deadweight for a fractional loss f=dQ/Qbar is
    # P0 Q0 [ (1-(1-f)^{1-eps})/(1-eps) - f ], convex and increasing, tangent-linearized
    # from below exactly like the linear-demand quadratic. eps>1 (|eta|<1) makes it
    # steeper than the linear form, so the linear channel understates the price move.
    m._price_curve = str(price_curve)
    _eps = 1.0 / abs(OIL_DEMAND_ELASTICITY)
    _P0 = BASELINE_OIL_PRICE
    _Qbar = float(world_supply_stage)

    def dwl_tangent(m, t, w, k):
        if not enable_price:
            return m.dwl[t, w] <= 0.0
        q_k = _bp[k]
        if price_curve == "isoelastic":
            Q0 = _Q0_stage[t]
            f = min(q_k / _Qbar, 0.95)
            dwl_k = _P0 * Q0 * ((1.0 - (1.0 - f) ** (1.0 - _eps)) / (1.0 - _eps) - f)
            slope = (_P0 * Q0 / _Qbar) * ((1.0 - f) ** (-_eps) - 1.0)
            return m.dwl[t, w] >= dwl_k + slope * (m.price_loss[t, w] - q_k)
        c = pyo.value(m.dwl_coef[t])
        return m.dwl[t, w] >= 2.0 * c * q_k * m.price_loss[t, w] - c * q_k ** 2
    m.con_dwl = pyo.Constraint(m.T, m.OMEGA, m.BP, rule=dwl_tangent)

    # ------------------------------------------------------------------
    # Objective
    # ------------------------------------------------------------------
    # Per-scenario total cost C_w (all stages), unweighted by probability.
    def scenario_cost_expr(w):
        expr = 0.0
        for t in m.T:
            # Sea-freight / war-risk premium scales with severity (1 -
            # availability) on maritime legs. Pipelines and last-mile delivery
            # are billed at base cost.
            expr += sum((m.mar_mult[t, w] if a in _maritime else 1.0)
                        * m.arc_cost[a] * m.x[a, t, w] for a in m.A)
            expr += sum(m.gamma[p] * m.z[p, t, w] for p in m.P)
            expr += sum(m.equity_mult[n, s] * m.pi_ns[n, s] * m.h[n, s, t, w]
                        for n in m.C for s in m.S)
            if enable_price:
                expr += m.dwl[t, w]
            if enable_cascade:
                expr += sum(m.equity_mult[n, s] * m.eta_ns[n, s]
                            * m.casc_scale[n, s, t] * m.Delta[n, s, t, w]
                            for n in m.C for s in m.S)
            expr += FLOOR_PENALTY * sum(m.fslack[n, s, t, w]
                                        for (n, s) in m.K_PRIO)
            # Managed demand (C6): the efficiency share (EFFICIENCY_FRAC of the
            # reduction) is cheap abatement priced at a fraction of the crude price;
            # the rationing share (the remainder) is an unmet need priced like a
            # direct shortage (pi_ns) and propagated to dependent sectors through the
            # stress term above.
            expr += sum(
                (EFFICIENCY_COST_FRAC * BASELINE_OIL_PRICE * EFFICIENCY_FRAC
                 + m.equity_mult[n, s] * m.pi_ns[n, s] * (1.0 - EFFICIENCY_FRAC))
                * m.rho[n, s, t, w] * m.demand[n, s, t]
                for n in m.C for s in m.S)
            # Replacement/opportunity cost of reserve draws: a
            # released barrel must be repurchased and forgoes its option value, so
            # reserve release is no longer free.
            expr += RESERVE_REPLACEMENT_COST * sum(m.r[n, t, w] for n in m.C)
            # Variable resource cost of emergency substitute supply:
            # each substitute barrel draws down product stocks, reconfigures refinery
            # yield, or switches fuel at a real marginal cost, so P6 is no longer an
            # almost-free bulk source beyond its activation charge.
            expr += sum(SUBSTITUTE_COST_PS * m.g_ps[n, t, w]
                        + SUBSTITUTE_COST_RF * m.g_rf[n, t, w]
                        + SUBSTITUTE_COST_FS * m.g_fs[n, t, w] for n in m.C)
        return expr

    # Inert per-scenario cost expression, exposed for post-solve loss-distribution
    # and CVaR reporting. Not referenced by the objective, so results are unchanged.
    m.scenario_cost = pyo.Expression(m.OMEGA,
                                     rule=lambda mm, w: scenario_cost_expr(w))

    # Decision-dependent uncertainty is active only when the full coordinated
    # bundle can be fielded. Otherwise probabilities stay at the unresponsive
    # baseline (m.prob) and the objective is the ordinary expectation.
    bundle = list(DEESCALATION_BUNDLE)
    ddu_active = endogenous_uncertainty and all(p in policy_subset for p in bundle)
    m.ddu_active = ddu_active

    if not ddu_active:
        def obj_rule(m):
            return sum(m.prob[w] * scenario_cost_expr(w) for w in m.OMEGA)
        m.OBJ = pyo.Objective(rule=obj_rule, sense=pyo.minimize)
    else:
        # Endogenous probabilities: p_w(d) = prod_t (a_t + b_t d_{t,w}), with
        # d_{t,w} = 1 iff the coordinated bundle is active at stage t on w's
        # path. Expanding gives a multilinear form in the binary d's, and each
        # monomial times the bounded cost C_w is linearized exactly.
        unresp = transition_matrix(unresponsive_regime)
        resp = responsive_matrix(unresp)
        n_edges = NUM_STAGES - 1
        dstages = list(range(n_edges))

        # De-escalation indicator d_{t,w}, the logical AND of the bundle's
        # activation at stage t. Non-anticipativity on z makes indicators that
        # share history equal automatically.
        m.DSTG = pyo.Set(initialize=dstages)
        m.BIDX = pyo.Set(initialize=list(range(len(bundle))))
        m.dd = pyo.Var(m.DSTG, m.OMEGA, within=pyo.Binary)
        m.con_dd_le = pyo.Constraint(
            m.DSTG, m.OMEGA, m.BIDX,
            rule=lambda m, t, w, idx: m.dd[t, w] <= m.z[bundle[idx], t, w])
        m.con_dd_ge = pyo.Constraint(
            m.DSTG, m.OMEGA,
            rule=lambda m, t, w: m.dd[t, w]
            >= sum(m.z[p, t, w] for p in bundle) - (len(bundle) - 1))

        # Upper bound on C_w for the exact McCormick products.
        Cmax = _cost_upper_bound(m, network, enable_cascade, SD, equity_weight)
        m.ddu_Cmax = float(Cmax)

        # Per-scenario cost variable bounded in [0, Cmax].
        m.Cw = pyo.Var(m.OMEGA, within=pyo.NonNegativeReals, bounds=(0.0, Cmax))
        m.con_Cw = pyo.Constraint(
            m.OMEGA, rule=lambda m, w: m.Cw[w] == scenario_cost_expr(w))

        # Build, per scenario, the monomials (subsets of its edges with |S|>=2)
        # and the McCormick products v_{w,S} = (prod_{t in S} d_{t,w}) * C_w.
        from itertools import combinations
        mono_keys = []      # (w, frozenset(S)) for |S|>=2  (AND-linearized)
        prod_keys = []      # (w, frozenset(S)) for |S|>=1  (McCormick product)
        coef = {}           # (w, frozenset(S)) -> coefficient in p_w * C_w
        scen_coeffs = {}
        for w in omega_set:
            ab = ddu_edge_coefficients(scenarios[w], unresp, resp)
            scen_coeffs[w] = ab
            for r in range(0, n_edges + 1):
                for S in combinations(dstages, r):
                    Sf = frozenset(S)
                    c = 1.0
                    for t in dstages:
                        c *= (ab[t][1] if t in Sf else ab[t][0])
                    coef[(w, Sf)] = c
                    if r >= 1:
                        prod_keys.append((w, Sf))
                    if r >= 2:
                        mono_keys.append((w, Sf))

        m.MONO = pyo.Set(initialize=mono_keys, dimen=2)
        m.PRODK = pyo.Set(initialize=prod_keys, dimen=2)
        m.mono = pyo.Var(m.MONO, within=pyo.Binary)      # AND of d's over S
        m.vprod = pyo.Var(m.PRODK, within=pyo.NonNegativeReals, bounds=(0.0, Cmax))

        # AND linearization of mono_{w,S} = prod_{t in S} d_{t,w}.
        def mono_le_rule(m, w, Sf, t):
            if t not in Sf:
                return pyo.Constraint.Skip
            return m.mono[w, Sf] <= m.dd[t, w]
        m.con_mono_le = pyo.Constraint(m.MONO, m.DSTG, rule=mono_le_rule)
        m.con_mono_ge = pyo.Constraint(
            m.MONO, rule=lambda m, w, Sf:
            m.mono[w, Sf] >= sum(m.dd[t, w] for t in Sf) - (len(Sf) - 1))

        # indicator expression for a product key: single d, or its mono var.
        def _ind(m, w, Sf):
            if len(Sf) == 1:
                (t,) = tuple(Sf)
                return m.dd[t, w]
            return m.mono[w, Sf]

        # McCormick: v = ind * C_w, with ind binary and 0 <= C_w <= Cmax.
        m.con_v1 = pyo.Constraint(
            m.PRODK, rule=lambda m, w, Sf: m.vprod[w, Sf] <= Cmax * _ind(m, w, Sf))
        m.con_v2 = pyo.Constraint(
            m.PRODK, rule=lambda m, w, Sf: m.vprod[w, Sf] <= m.Cw[w])
        m.con_v3 = pyo.Constraint(
            m.PRODK, rule=lambda m, w, Sf:
            m.vprod[w, Sf] >= m.Cw[w] - Cmax * (1.0 - _ind(m, w, Sf)))

        def obj_rule(m):
            total = 0.0
            for w in m.OMEGA:
                empty = frozenset()
                total += coef[(w, empty)] * m.Cw[w]
                for (ww, Sf) in prod_keys:
                    if ww == w:
                        total += coef[(w, Sf)] * m.vprod[w, Sf]
            return total
        m.OBJ = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    # ------------------------------------------------------------------
    # Constraints
    # ------------------------------------------------------------------
    # Source capacity
    def source_cap_con(m, e, t, w):
        out_arcs = [a.aid for a in network.out_arcs.get(e, []) if a.aid in set(m.A)]
        if not out_arcs:
            return pyo.Constraint.Skip
        return sum(m.x[a, t, w] for a in out_arcs) <= m.source_cap[e, t]
    m.con_source = pyo.Constraint(m.E, m.T, m.OMEGA, rule=source_cap_con)

    # Transshipment balance
    def transship_con(m, j, t, w):
        in_a = [a.aid for a in network.in_arcs.get(j, []) if a.aid in set(m.A)]
        out_a = [a.aid for a in network.out_arcs.get(j, []) if a.aid in set(m.A)]
        if not in_a and not out_a:
            return pyo.Constraint.Skip
        return (sum(m.x[a, t, w] for a in in_a) ==
                sum(m.x[a, t, w] for a in out_a))
    m.con_transship = pyo.Constraint(m.J, m.T, m.OMEGA, rule=transship_con)

    # Country balance. Only the DOMESTIC part of each reserve/substitute draw
    # (total draw net of the part released to the world market) stays in the country
    # to meet its own sectors. Reserves and substitute supply are consumed at home.
    def country_balance_con(m, n, t, w):
        in_a = [a.aid for a in network.in_arcs.get(n, []) if a.aid in set(m.A)]
        out_a = [a.aid for a in network.out_arcs.get(n, []) if a.aid in set(m.A)]
        return (sum(m.x[a, t, w] for a in in_a)
                - sum(m.x[a, t, w] for a in out_a)
                + m.r[n, t, w] + m.g[n, t, w] + m.base[n, t, w]
                == sum(m.q[n, s, t, w] for s in m.S))
    m.con_country = pyo.Constraint(m.C, m.T, m.OMEGA, rule=country_balance_con)

    # Baseline non-chokepoint supply is bounded by the always-available share.
    def base_cap_con(m, n, t, w):
        return m.base[n, t, w] <= m.base_cap[n, t]
    m.con_base = pyo.Constraint(m.C, m.T, m.OMEGA, rule=base_cap_con)

    # Sector demand balance
    def demand_balance_con(m, n, s, t, w):
        d = m.demand[n, s, t]
        if d <= 1e-10:
            return pyo.Constraint.Skip
        # Price-induced demand contraction (C2), netted out of the physical
        # requirement where price_loss is pinned by equality. The common contraction
        # fraction is |eta| dP/P0 = price_loss/Qbar, scaled by the sector weight
        # kappa_s so elastic sectors cut more and essential sectors less;
        # the demand-weighted mean of kappa_s is 1, so the coalition contraction and
        # the deadweight are unchanged. Its forgone consumer surplus is the deadweight
        # already in the objective, so the two are reconciled and no barrel is double-
        # counted. Linear because d and kappa_s are parameters.
        contraction = (m.kappa_contract[s] * m.price_loss_phys[t, w] / world_supply_stage) \
            if (enable_price and m._net_ok[(t, w)]) else 0.0
        return (m.q[n, s, t, w] + m.h[n, s, t, w]
                == d * (1 - m.rho[n, s, t, w] - contraction))
    m.con_demand = pyo.Constraint(m.C, m.S, m.T, m.OMEGA, rule=demand_balance_con)

    # Arc capacity. Nominal arcs are bounded by surviving capacity, bypass and
    # sharing arcs carry flow only up to the policy-mobilized capacity y.
    def arc_cap_con(m, a, t, w):
        arc = arc_dict[a]
        if arc.is_bypass or arc.is_sharing:
            if a in m.A_ALT:
                return m.x[a, t, w] <= m.y[a, t, w]
            return m.x[a, t, w] <= 0
        else:
            cap = m.arc_cap[a, t, w]
            return m.x[a, t, w] <= cap
    m.con_arc_cap = pyo.Constraint(m.A, m.T, m.OMEGA, rule=arc_cap_con)

    # Mobilized capacity on the alternative arcs (Mbbl per stage): sharing arcs
    # unlocked by P5, rerouting/bypass arcs unlocked by P2, plus the irreversible
    # diversification investment P4 which adds DIVERSIFICATION_FACTOR of the arc
    # capacity from stage DIVERSIFICATION_STAGE onward.
    def alt_cap_con(m, a, t, w):
        arc = arc_dict[a]
        base = arc.capacity_mbd * SD
        if arc.is_sharing:
            if "P5" in policy_subset:
                return m.y[a, t, w] <= base * m.z["P5", t, w]
            return m.y[a, t, w] <= 0
        terms = []
        if "P2" in policy_subset:
            terms.append(base * m.z["P2", t, w])
        if "P4" in policy_subset and t >= DIVERSIFICATION_STAGE:
            terms.append(base * DIVERSIFICATION_FACTOR * m.z["P4", t, w])
        if not terms:
            return m.y[a, t, w] <= 0
        return m.y[a, t, w] <= sum(terms)
    m.con_bypass = pyo.Constraint(m.A_ALT, m.T, m.OMEGA, rule=alt_cap_con)

    # Emergency substitute supply linked to policy P6, disaggregated into three
    # sub-instruments. Total substitute is their sum.
    def substitute_total_con(m, n, t, w):
        return m.g[n, t, w] == m.g_ps[n, t, w] + m.g_rf[n, t, w] + m.g_fs[n, t, w]
    m.con_substitute_total = pyo.Constraint(m.C, m.T, m.OMEGA, rule=substitute_total_con)

    def _ss_cap(n, chan):
        c = network.countries.get(n)
        base = c.daily_oil_mbd * SUBSTITUTE_SUPPLY_FRAC * SD if c else 0.0
        return base * SUBSTITUTE_CHANNEL_SHARE[chan]

    _P6 = "P6" in policy_subset
    # Restricted-P6 bound: only the universally available
    # commercial product-stock channel is enabled, refinery-yield reallocation and
    # fuel switching, which need country-specific refining and dual-fired capacity,
    # are switched off.
    _P6_rf_fs = _P6 and not restrict_substitute

    # Refinery-yield reallocation and fuel switching: per-stage capacity, gated by P6.
    def g_rf_cap(m, n, t, w):
        if _P6_rf_fs:
            return m.g_rf[n, t, w] <= _ss_cap(n, "rf") * m.z["P6", t, w]
        return m.g_rf[n, t, w] <= 0
    m.con_g_rf = pyo.Constraint(m.C, m.T, m.OMEGA, rule=g_rf_cap)

    def g_fs_cap(m, n, t, w):
        if _P6_rf_fs:
            return m.g_fs[n, t, w] <= _ss_cap(n, "fs") * m.z["P6", t, w]
        return m.g_fs[n, t, w] <= 0
    m.con_g_fs = pyo.Constraint(m.C, m.T, m.OMEGA, rule=g_fs_cap)

    # Product-stock release: per-stage rate gated by P6 AND bounded by a FINITE
    # inventory that depletes as it is drawn (like the strategic reserve).
    def g_ps_rate(m, n, t, w):
        if _P6:
            return m.g_ps[n, t, w] <= _ss_cap(n, "ps") * m.z["P6", t, w]
        return m.g_ps[n, t, w] <= 0
    m.con_g_ps_rate = pyo.Constraint(m.C, m.T, m.OMEGA, rule=g_ps_rate)

    def ps_init(m, n, w):
        # Fixed inventory in barrels (PRODUCT_STOCK_DAYS of daily release capacity),
        # independent of the stage length. _ss_cap is the per-stage flow (daily x SD),
        # so dividing by SD recovers the daily rate before scaling by the fixed days.
        return m.PS[n, 0, w] == _ss_cap(n, "ps") * (PRODUCT_STOCK_DAYS / SD)
    m.con_ps_init = pyo.Constraint(m.C, m.OMEGA, rule=ps_init)

    def ps_dyn(m, n, t, w):
        if t == 0:
            return pyo.Constraint.Skip
        return m.PS[n, t, w] == m.PS[n, t - 1, w] - m.g_ps[n, t - 1, w]
    m.con_ps_dyn = pyo.Constraint(m.C, m.T, m.OMEGA, rule=ps_dyn)

    def ps_avail(m, n, t, w):
        return m.g_ps[n, t, w] <= m.PS[n, t, w]
    m.con_ps_avail = pyo.Constraint(m.C, m.T, m.OMEGA, rule=ps_avail)

    # Reserve dynamics
    def reserve_init_con(m, n, w):
        return m.R[n, 0, w] == m.R_init[n]
    m.con_reserve_init = pyo.Constraint(m.C, m.OMEGA, rule=reserve_init_con)

    def reserve_trans_con(m, n, t, w):
        if t == 0:
            return pyo.Constraint.Skip
        return m.R[n, t, w] == m.R[n, t - 1, w] - m.r[n, t - 1, w]
    m.con_reserve_trans = pyo.Constraint(m.C, m.T, m.OMEGA, rule=reserve_trans_con)

    def reserve_draw_max_con(m, n, t, w):
        if "P1" in policy_subset:
            return m.r[n, t, w] <= m.r_max[n] * m.z["P1", t, w]
        return m.r[n, t, w] <= 0
    m.con_reserve_draw = pyo.Constraint(m.C, m.T, m.OMEGA, rule=reserve_draw_max_con)

    def reserve_nonneg_con(m, n, t, w):
        return m.R[n, t, w] >= 0
    m.con_reserve_nn = pyo.Constraint(m.C, m.T, m.OMEGA, rule=reserve_nonneg_con)

    # Terminal stock: the final-stage draw cannot exceed the reserve left after
    # it. There is no stage-T level variable, so without this the last draw would
    # be bounded only by the per-stage rate and could deplete reserves that do
    # not exist.
    def reserve_terminal_con(m, n, w):
        return m.R[n, NUM_STAGES - 1, w] - m.r[n, NUM_STAGES - 1, w] >= 0
    m.con_reserve_terminal = pyo.Constraint(m.C, m.OMEGA, rule=reserve_terminal_con)

    # Demand reduction linked to policy P3
    def demand_red_con(m, n, s, t, w):
        if "P3" in policy_subset:
            return m.rho[n, s, t, w] <= m.rho_max[n, s] * m.z["P3", t, w]
        return m.rho[n, s, t, w] <= 0
    m.con_demand_red = pyo.Constraint(m.C, m.S, m.T, m.OMEGA, rule=demand_red_con)


    # Physical escape cap: the realized rerouted flow per disrupted chokepoint
    # cannot exceed that chokepoint's bypass capacity (Mbbl per stage). This binds
    # the restored supply to what the alternative routes can actually carry.
    dcp_with_escape = [cp for cp in _escape_by_cp]
    m.DCP = pyo.Set(initialize=dcp_with_escape)

    def escape_cap_con(m, cp, t, w):
        cap = network.chokepoints[cp].bypass_capacity_mbd * SD
        return sum(m.x[a, t, w] for a in _escape_by_cp[cp]) <= cap
    m.con_escape_cap = pyo.Constraint(m.DCP, m.T, m.OMEGA, rule=escape_cap_con)

    # Stage budget: total activation cost per stage is capped. Skipped when no
    # policy is available (empty sum).
    def budget_con(m, t, w):
        if len(list(m.P)) == 0:
            return pyo.Constraint.Skip
        return sum(m.gamma[p] * m.z[p, t, w] for p in m.P) <= m.budget[t]
    m.con_budget = pyo.Constraint(m.T, m.OMEGA, rule=budget_con)

    # Irreversibility
    def irrev_con(m, p, t, w):
        if t >= NUM_STAGES - 1:
            return pyo.Constraint.Skip
        return m.z[p, t, w] <= m.z[p, t + 1, w]
    m.con_irrev = pyo.Constraint(m.P_IRR, m.T, m.OMEGA, rule=irrev_con)

    # Service floor: priority country-sector pairs (low-reserve countries'
    # healthcare and electricity) must be served at least phi_floor of managed
    # demand. Soft floor: breach slack fslack is heavily penalised (FLOOR_PENALTY),
    # so the floor holds whenever oil can physically reach the country and relaxes
    # only when the disruption makes it unreachable (e.g. simultaneous closures
    # cutting the country off). Keeps the model feasible without weakening it.
    def floor_con(m, n, s, t, w):
        d = m.demand[n, s, t]
        if d <= 1e-10:
            return pyo.Constraint.Skip
        # The floor is measured on the demand the sector actually faces AFTER both the
        # managed reduction and the involuntary price contraction, so it
        # uses the same base as the demand balance. The floor still guarantees a served
        # minimum: it protects the essential core of a priority sector even where the
        # sector's discretionary oil use is otherwise price-elastic.
        contraction = (m.kappa_contract[s] * m.price_loss_phys[t, w] / world_supply_stage) \
            if (enable_price and m._net_ok[(t, w)]) else 0.0
        return (m.q[n, s, t, w] + m.fslack[n, s, t, w]
                >= m.phi_floor[n, s] * d * (1 - m.rho[n, s, t, w] - contraction))
    m.con_floor = pyo.Constraint(m.K_PRIO, m.T, m.OMEGA, rule=floor_con)

    # Nonanticipativity
    def _scenarios_same_at_t(w1, w2, t):
        """True if scenarios w1, w2 share history through stage t."""
        s1 = scenarios[w1].states[:t + 1]
        s2 = scenarios[w2].states[:t + 1]
        return s1 == s2

    nonant_pairs = []
    for t in stages:
        for w1 in omega_set:
            for w2 in omega_set:
                if w1 < w2 and _scenarios_same_at_t(w1, w2, t):
                    nonant_pairs.append((t, w1, w2))

    m.NONANT = pyo.Set(initialize=nonant_pairs, dimen=3)

    nonant_indexed = [(p, t, w1, w2) for (t, w1, w2) in nonant_pairs
                      for p in policy_subset]
    m.NONANT_FULL = pyo.Set(initialize=nonant_indexed, dimen=4)
    m.con_nonant = pyo.Constraint(m.NONANT_FULL, rule=lambda m, p, t, w1, w2:
                                   m.z[p, t, w1] == m.z[p, t, w2])

    # ------------------------------------------------------------------
    # Cascade amplification constraints
    # ------------------------------------------------------------------
    if enable_cascade:
        # Cross-sector stress (Inoperability Input-Output form): stress on sector
        # s is the dependence-weighted sum, over the sectors s' it draws inputs
        # from, of the fraction of s' left unserved. This is the involuntary
        # shortfall h_{s'}/d_{s'} PLUS the forced-rationing share of managed demand
        # reduction, (1 - EFFICIENCY_FRAC) rho_{s'} (C6): the efficiency share is
        # genuine conservation that preserves output, but the rationing share
        # starves downstream users exactly as an involuntary shortfall does. Using a
        # fixed share avoids an extra variable per country-sector-stage-scenario.
        def stress_con(m, n, s, t, w):
            rscale = m.region_cascade_scale[n]
            expr = 0.0
            for sp, beta_val in cascade_mat.get(s, {}).items():
                d = m.demand[n, sp, t]
                if d > 1e-10:
                    expr += rscale * beta_val * (m.h[n, sp, t, w] / d
                                        + (1.0 - EFFICIENCY_FRAC) * m.rho[n, sp, t, w])
            return m.u_stress[n, s, t, w] >= expr
        m.con_stress = pyo.Constraint(m.C, m.S, m.T, m.OMEGA, rule=stress_con)

        def epigraph_con(m, n, s, t, w, l):
            if l >= len(damage_segs.get(s, [])):
                return pyo.Constraint.Skip
            alpha, b = damage_segs[s][l]
            return m.Delta[n, s, t, w] >= alpha * m.u_stress[n, s, t, w] + b
        m.con_epigraph = pyo.Constraint(m.C, m.S, m.T, m.OMEGA, m.L,
                                         rule=epigraph_con)

    return m


def solve_model(
    model: pyo.ConcreteModel,
    solver_name: str = "glpk",
    time_limit: int = 3600,
    mip_gap: float = 1e-3,
) -> dict:
    """Solve the model and return a summary dict (status, objective, solver,
    result). time_limit is in seconds, mip_gap is the relative gap."""
    solver = pyo.SolverFactory(solver_name)
    if solver_name in ("gurobi", "gurobi_direct", "appsi_gurobi"):
        solver.options['TimeLimit'] = time_limit
        solver.options['MIPGap'] = mip_gap
        # Thread count drives peak memory (Gurobi replicates working data per
        # thread), so two threads keep the peak small on the largest
        # extensive-form solves. On the HPC (ample RAM) set GUROBI_THREADS higher
        # for faster solves. Seed fixes the tie-breaks.
        solver.options['Threads'] = int(os.environ.get("GUROBI_THREADS", "2"))
        solver.options['Seed'] = 0
        # Memory backstops, tuned for a memory-constrained host (see GUROBI_MEM_GB).
        # Method=1 (dual simplex) avoids the barrier root factorization, which is the
        # largest single memory consumer for this 235k-variable model. NodefileStart
        # spills the branch-and-bound tree to disk aggressively, and SoftMemLimit (GB)
        # makes Gurobi stop opening new nodes and return its best incumbent near the
        # limit, degrading gracefully to a larger gap instead of crashing out of
        # memory. Raise GUROBI_MEM_GB when more RAM is free for tighter, faster solves;
        # on the HPC set GUROBI_METHOD=-1 (automatic, lets Gurobi pick barrier) since the
        # factorization memory is no longer a constraint there.
        _mem_gb = float(os.environ.get("GUROBI_MEM_GB", "6.0"))
        solver.options['Method'] = int(os.environ.get("GUROBI_METHOD", "1"))
        solver.options['NodefileStart'] = 0.5
        solver.options['SoftMemLimit'] = _mem_gb
    elif solver_name == "cplex":
        solver.options['timelimit'] = time_limit
        solver.options['mipgap'] = mip_gap
    elif solver_name == "glpk":
        solver.options['tmlim'] = time_limit
        solver.options['mipgap'] = mip_gap
    elif "highs" in solver_name:
        # HiGHS ('highs' / 'appsi_highs'): relative gap, time limit, and a node
        # ceiling as a memory backstop.
        solver.options['time_limit'] = float(time_limit)
        solver.options['mip_rel_gap'] = mip_gap
        solver.options['mip_max_nodes'] = 2000000

    result = solver.solve(model, tee=False)

    status = str(result.solver.termination_condition)
    # Accept the incumbent even when the solver stopped at a resource limit. HiGHS
    # reports ``maxTimeLimit`` while still holding a feasible solution, which at
    # the 0.1% gap used here is a valid result. ``exception=False`` returns None
    # when no solution was loaded (a genuine no-incumbent or infeasible solve).
    _accept = ("optimal", "feasible", "locallyOptimal", "globallyOptimal",
               "maxTimeLimit", "maxIterations", "maxEvaluations", "other")
    obj_val = pyo.value(model.OBJ, exception=False) if status in _accept else None

    # Best (dual) bound and the achieved relative optimality gap, retained so a
    # downstream game solve can certify how far each objective may sit from its
    # optimum. Read from the Pyomo results first, then fall back to
    # the live Gurobi model. None when the solver does not expose a bound.
    best_bound = None
    try:
        pb = getattr(result, "problem", None)
        lb = getattr(pb, "lower_bound", None) if pb is not None else None
        if lb is not None and lb not in (float("inf"), float("-inf")):
            best_bound = float(lb)
    except Exception:
        best_bound = None
    if best_bound is None:
        try:
            sm = getattr(solver, "_solver_model", None)
            if sm is not None and hasattr(sm, "ObjBound"):
                best_bound = float(sm.ObjBound)
        except Exception:
            best_bound = None
    gap = None
    if obj_val is not None and best_bound is not None:
        gap = abs(obj_val - best_bound) / max(abs(obj_val), 1e-9)

    out = {
        "status": status,
        "objective": obj_val,
        "best_bound": best_bound,
        "gap": gap,
        "solver": solver_name,
        "result": None,  # heavy Pyomo results object intentionally not retained
    }

    # Bound memory across long solve loops. Persistent solvers
    # (appsi_gurobi/appsi_highs) keep the loaded model alive, so we dispose the
    # solver model and drop the solver. We deliberately do NOT release the Gurobi
    # licence/env, since re-acquiring a WLS licence every solve is slow.
    try:
        sm = getattr(solver, "_solver_model", None)
        if sm is not None and hasattr(sm, "dispose"):
            sm.dispose()  # gurobipy Model.dispose() frees the C-side model
    except Exception:
        pass
    del solver, result
    gc.collect()

    return out


# ======================================================================
# SOLVE: MIP wrapper and result cache
# ======================================================================
import gc
import hashlib
import json
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pyomo.environ as pyo



# All six policy identifiers in canonical order
ALL_POLICIES: Tuple[str, ...] = ("P1", "P2", "P3", "P4", "P5", "P6")

DEFAULT_CACHE_DIR = "outputs/mip_cache"


# Process-local caches, rebuilt on import.
_network_cache: Optional[EnergyNetwork] = None
_scenario_cache: Dict[Tuple[str, int], list] = {}


def get_network() -> EnergyNetwork:
    """Return the cached network instance, building it on first call."""
    global _network_cache
    if _network_cache is None:
        countries, sources, chokepoints, processing, arcs = build_all()
        _network_cache = EnergyNetwork(sources, chokepoints, processing,
                                       countries, arcs)
    return _network_cache


def get_scenarios(scenario_class: str, target_count: int,
                  stage_days: Optional[int] = None) -> list:
    """Return reduced scenarios for the given class, cached in-process."""
    sd = STAGE_DAYS if stage_days is None else float(stage_days)
    key = (scenario_class, target_count, sd)
    if key not in _scenario_cache:
        _scenario_cache[key] = build_scenario_set(scenario_class, target_count,
                                                  stage_days=sd)
    return _scenario_cache[key]


@dataclass
class MIPResult:
    """Compact summary of a single MIP solve."""
    objective: Optional[float]
    status: str
    runtime_sec: float
    policy_subset: Tuple[str, ...]
    scenario_class: str
    target_count: int
    duration_days: int
    chokepoint: str
    enable_cascade: bool
    delay_stages: int
    stage_days: Optional[int] = None
    z_star: Dict[str, int] = field(default_factory=dict)
    solver: str = "appsi_highs"
    cached: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def _cache_key(policy_subset: Tuple[str, ...],
               scenario_class: str,
               target_count: int,
               duration_days: int,
               chokepoint: str,
               enable_cascade: bool,
               delay_stages: int,
               stage_days: Optional[int]) -> str:
    """Deterministic SHA-1 hash of all solve arguments."""
    payload = json.dumps({
        "subset": sorted(policy_subset),
        "scenario_class": scenario_class,
        "target_count": target_count,
        "duration_days": duration_days,
        "chokepoint": chokepoint,
        "enable_cascade": enable_cascade,
        "delay_stages": delay_stages,
        "stage_days": stage_days,
    }, sort_keys=True)
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


def _load_cached(cache_dir: str, key: str) -> Optional[MIPResult]:
    path = Path(cache_dir) / f"{key}.json"
    if not path.exists():
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        data["cached"] = True
        data["policy_subset"] = tuple(data["policy_subset"])
        return MIPResult(**data)
    except (json.JSONDecodeError, TypeError, KeyError):
        return None


def _save_cached(cache_dir: str, key: str, result: MIPResult) -> None:
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    path = Path(cache_dir) / f"{key}.json"
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(result.to_dict(), f, indent=2)
    os.replace(tmp, path)  # atomic on POSIX


class NoSolverAvailableError(RuntimeError):
    """Raised when no usable MIP solver is found on this machine."""


def _verify_solver(name: str) -> bool:
    """Smoke-test a solver by solving a trivial LP. ``available()`` is unreliable
    across Pyomo versions, especially APPSI interfaces."""
    try:
        m = pyo.ConcreteModel()
        m.x = pyo.Var(domain=pyo.NonNegativeReals, bounds=(0.0, 1.0))
        m.OBJ = pyo.Objective(expr=m.x, sense=pyo.minimize)
        s = pyo.SolverFactory(name)
        # Cheap pre-filter when it exists
        if hasattr(s, "available"):
            try:
                if not s.available(exception_flag=False):
                    return False
            except Exception:
                return False
        s.solve(m, tee=False)
        return True
    except Exception:
        return False


# Cache the detected solver so the smoke test runs only once per process.
_detected_solver: Optional[str] = None


def _detect_solver(preferred: Optional[str] = None) -> str:
    """Return a verified-working solver, preferring Gurobi then HiGHS.

    Raises NoSolverAvailableError with install instructions when nothing works,
    instead of a generic Pyomo error deep in a notebook cell.
    """
    global _detected_solver
    if preferred is not None:
        # Honour explicit choice even if the smoke test fails. solve_mip
        # surfaces the underlying error.
        return preferred
    if _detected_solver is not None:
        return _detected_solver

    # appsi_gurobi uses fast bulk model loading. gurobi_direct loads
    # variable-by-variable and is slow for large models, so it ranks later.
    # Override via the ``preferred`` arg or GEOENERGY_SOLVER env var.
    env_choice = os.environ.get("GEOENERGY_SOLVER")
    if env_choice:
        _detected_solver = env_choice
        return env_choice
    candidates = ["appsi_gurobi", "appsi_highs", "highs", "glpk", "cbc",
                  "gurobi_direct", "gurobi", "cplex"]
    for name in candidates:
        if _verify_solver(name):
            _detected_solver = name
            return name

    raise NoSolverAvailableError(
        "No usable MIP solver was found on this machine.\n"
        "\n"
        "The fastest fix on any platform is to install HiGHS via pip:\n"
        "    pip install highspy\n"
        "\n"
        "After installation, restart the Jupyter kernel (so Pyomo re-scans\n"
        "available solvers) and re-run this cell.\n"
        "\n"
        "Alternative solvers (any one is sufficient):\n"
        "  - GLPK   : Windows  -> https://winglpk.sourceforge.net/  (add bin/ to PATH)\n"
        "             macOS    -> brew install glpk\n"
        "             Linux    -> sudo apt install glpk-utils\n"
        "  - Gurobi : commercial; pip install gurobipy (academic licenses are free)\n"
        "  - CBC    : conda install -c conda-forge coincbc\n"
    )


def solve_mip(policy_subset: Tuple[str, ...] = ALL_POLICIES,
              scenario_class: str = "full_closure",
              target_count: int = 15,
              duration_days: int = 100,
              chokepoint: str = "CP_HORMUZ",
              enable_cascade: bool = True,
              delay_stages: int = 0,
              stage_days: Optional[int] = None,
              solver: Optional[str] = None,
              time_limit: int = 900,
              mip_gap: float = 1e-3,
              cache_dir: str = DEFAULT_CACHE_DIR,
              use_cache: bool = True,
              amp_spec: str = "headline",
              verbose: bool = False) -> MIPResult:
    """Build and solve the multi-stage stochastic MIP under a given policy subset.

    Parameters
    ----------
    policy_subset : tuple of str
        Policies allowed to activate. Others are fixed to ``z = 0``. Default all six.
    scenario_class : str
        One of ``"partial_restriction"``, ``"full_closure"``,
        ``"multi_chokepoint"``, ``"attrition_*"``.
    target_count : int
        Number of reduced scenarios.
    duration_days : int
        Disruption duration. Pass a matching ``stage_days`` to stretch the
        horizon. Feeds the cache key so duration sweeps get distinct entries.
    stage_days : int or None
        Stage length in days. ``None`` uses ``STAGE_DAYS`` (25).
    chokepoint : str
        Disrupted chokepoint node.
    enable_cascade : bool
        If False, omit cascade PWL constraints (ablation).
    delay_stages : int
        Fix ``z = 0`` for stages 0..delay_stages-1. ``0`` = immediate (default).
    solver : str or None
        Solver name, or ``None`` to auto-detect.
    time_limit : int
        Solver wall-clock limit in seconds.
    mip_gap : float
        Relative MIP gap tolerance.
    use_cache : bool
        If False, re-solve even when a cache entry exists.
    """
    policy_subset = tuple(sorted(policy_subset))
    disrupted = disrupted_chokepoints_for_class(scenario_class, chokepoint)
    choke_label = "+".join(disrupted) if isinstance(disrupted, list) else disrupted
    # Keep the amplification specification out of the default cache namespace so
    # 'capped'/'leontief' never collide with the headline entry.
    choke_label_key = choke_label if amp_spec == "headline" else f"{choke_label}__{amp_spec}"
    key = _cache_key(policy_subset, scenario_class, target_count,
                     duration_days, choke_label_key, enable_cascade,
                     delay_stages, stage_days)

    if use_cache:
        hit = _load_cached(cache_dir, key)
        if hit is not None:
            if verbose:
                print(f"  [cache hit] subset={policy_subset} "
                      f"obj={hit.objective:.1f}")
            return hit

    network = get_network()
    scenarios = get_scenarios(scenario_class, target_count, stage_days)

    # Policies outside the subset are not added to ``m.P``, which is how
    # build_model supports ablation.
    model = build_model(
        network, scenarios,
        disrupted_chokepoint=disrupted,
        duration_days=duration_days,
        enable_cascade=enable_cascade,
        unresponsive_regime=class_regime(scenario_class)[1],
        policy_subset=list(policy_subset),
        stage_days=stage_days,
        amp_spec=amp_spec,
    )

    # Timing-delay constraint: fix z=0 for stages 0..delay_stages-1.
    if delay_stages > 0:
        for p in policy_subset:
            for t in range(delay_stages):
                for w in model.OMEGA:
                    model.z[p, t, w].fix(0)

    solver_name = _detect_solver(solver)
    t0 = time.time()
    info = solve_model(model, solver_name=solver_name,
                       time_limit=time_limit, mip_gap=mip_gap)
    elapsed = time.time() - t0

    # Extract z*. Nonanticipativity fixes stage 0 across scenarios, but later
    # stages may differ, so record the modal value per (policy, stage).
    z_star: Dict[str, int] = {}
    if info["objective"] is not None:
        for p in policy_subset:
            for t in model.T:
                vals = [int(round(pyo.value(model.z[p, t, w])))
                        for w in model.OMEGA]
                z_star[f"{p}_t{t}"] = max(set(vals), key=vals.count)

    result = MIPResult(
        objective=info["objective"],
        status=info["status"],
        runtime_sec=elapsed,
        policy_subset=policy_subset,
        scenario_class=scenario_class,
        target_count=target_count,
        duration_days=duration_days,
        chokepoint=chokepoint,
        enable_cascade=enable_cascade,
        delay_stages=delay_stages,
        stage_days=stage_days,
        z_star=z_star,
        solver=solver_name,
        cached=False,
    )

    if use_cache and info["objective"] is not None:
        _save_cached(cache_dir, key, result)

    if verbose:
        obj = "infeasible" if result.objective is None else f"{result.objective:.1f}"
        print(f"  [{solver_name}] subset={policy_subset} "
              f"delay={delay_stages} time={elapsed:.1f}s obj={obj}")

    # Solved models are large and hold reference cycles, so free them to bound
    # RAM across long solve loops.
    del model, info
    gc.collect()

    return result


def clear_cache(cache_dir: str = DEFAULT_CACHE_DIR) -> int:
    """Delete all cached MIP results. Returns count deleted."""
    p = Path(cache_dir)
    if not p.exists():
        return 0
    files = list(p.glob("*.json"))
    for f in files:
        f.unlink()
    return len(files)
